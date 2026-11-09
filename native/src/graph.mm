#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tokspur/carrier_phase.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/q8_many.hpp"
#include "tokspur/sampling.hpp"
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
#include "tokspur/json.hpp"
#endif
#if defined(TOKSPUR_PRIVATE_PIPELINE_LOOKUP) && TOKSPUR_PRIVATE_PIPELINE_LOOKUP
#include "tokspur/pipeline_key.hpp"
#include <unordered_map>
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#include "tokspur/raw_decode.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#include "tokspur/q8_three_rows.hpp"
#endif
#include "tokspur/q8_split.hpp"
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
#include "tokspur/grouped_mm.hpp"
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
#include "tokspur/moe_mm_many.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
#include "tokspur/grouped_mm_f32.hpp"
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_CHUNK
#include "tokspur/gdn_chunk.hpp"
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
#include "tokspur/gdn_legacy_batch.hpp"
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
#include "tokspur/qsa_prefill_ops.hpp"
#include "tokspur/qsa_prefill_plan.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_Q8_MPP
#include "tokspur/q8_mpp.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
#include "tokspur/qsa_stage_v8.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
#include "tokspur/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstdio>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <mach/mach_time.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#ifndef TOKSPUR_PRIVATE_QSA_STAGE_V8
#error "batch phase clocks are a separate private Stage-V8 diagnostic only"
#endif
#endif

namespace tokspur {
namespace {
std::atomic<uint64_t> next_owner{1};
using Clock = std::chrono::steady_clock;
constexpr uint32_t counter_chunk_dispatches = 2048; // Two timestamps/operation fit 32 KiB.

#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
bool phase_stderr_writable() noexcept {
    struct stat status{};
    const auto flags = fcntl(STDERR_FILENO, F_GETFL);
    return flags >= 0 && (flags & O_ACCMODE) != O_RDONLY &&
        fstat(STDERR_FILENO, &status) == 0 && S_ISREG(status.st_mode);
}
int64_t phase_nanoseconds(Clock::time_point point) noexcept {
    static_assert(Clock::period::num == 1 && Clock::period::den == 1'000'000'000);
    static_assert(std::numeric_limits<Clock::rep>::is_signed && std::numeric_limits<Clock::rep>::digits <= 63);
    const auto value = point.time_since_epoch().count();
    return value < 0 ? int64_t(-1) : int64_t(value);
}
bool emit_phase_record(const BatchPhaseClockSample& sample) noexcept {
    std::array<char, 2048> storage{};
    const auto size = format_batch_phase_clock(sample, storage);
    // Root owns immutable regular stderr for this diagnostic's lifetime. This
    // recheck detects sequential replacement; it is not atomic against another
    // thread concurrently replacing fd2 before write (outside that contract).
    if (!size || !phase_stderr_writable()) return false;
    size_t written = 0;
    for (uint32_t attempt = 0; attempt < 8 && written < size; ++attempt) {
        const auto count = write(STDERR_FILENO, storage.data() + written, size - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        written += size_t(count);
    }
    return written == size;
}
#endif

void require(bool condition, const char* message) { if (!condition) throw FormatError(message); }
#ifdef TOKSPUR_PRIVATE_HC_EXACT
const char* hc_exact_kernel(size_t bucket) {
    static constexpr std::array names{"graph_hc_q8_exact_r1_u16", "graph_hc_exact_many_1",
        "graph_hc_exact_many_2", "graph_hc_exact_many_4", "graph_hc_exact_many_8"};
    require(bucket < names.size(), "invalid private exact-HC route"); return names[bucket];
}
bool hc_exact_pipeline(const char* name) {
    for (size_t i = 0; i < 5; ++i) if (std::strcmp(name, hc_exact_kernel(i)) == 0) return true;
    return false;
}
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
const char* wide_q8_exact_kernel(size_t shape) {
    require(shape < 2, "invalid private wide-Q8 shape");
    return shape ? "q8_wide_c6144_r2560_u16" : "q8_wide_c2560_r10240_u16";
}
int wide_q8_exact_shape(uint32_t columns, uint32_t rows) {
    return columns == 2560 && rows == 10240 ? 0 : columns == 6144 && rows == 2560 ? 1 : -1;
}
#endif

#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
const char* wide_q8_many_kernel(size_t bucket) {
    static constexpr std::array names{"q8_wide_many_c2560_r10240_n2_u4", "q8_wide_many_c2560_r10240_n4_u4",
        "q8_wide_many_c2560_r10240_n8_u4", "q8_wide_many_c6144_r2560_n2_u4",
        "q8_wide_many_c6144_r2560_n4_u4", "q8_wide_many_c6144_r2560_n8_u4"};
    require(bucket < names.size(), "invalid private wide-Q8-many route"); return names[bucket];
}
#endif

bool overlaps(GpuView a, GpuView b) {
    return a.owner == b.owner && a.slot == b.slot && a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
}
bool same(GpuView a, GpuView b) {
    return a.owner == b.owner && a.slot == b.slot && a.offset == b.offset && a.bytes == b.bytes;
}
void separate(GpuView output, std::initializer_list<GpuView> inputs, bool exact_in_place = false) {
    for (const auto input : inputs) require(!overlaps(output, input) || (exact_in_place && same(output, input)), "unsupported GPU buffer alias");
}
const char* q8_many_kernel(uint32_t width) {
    require(width == 1 || width == 2 || width == 4 || width == 8,"unsupported Q8-many pipeline width");
    return width == 1 ? "graph_q8_many_1" : width == 2 ? "graph_q8_many_2" : width == 4 ? "graph_q8_many_4" : "graph_q8_many_8";
}
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
const char* lane_many_kernel(uint32_t width) {
    require(width == 1 || width == 2 || width == 4 || width == 8, "unsupported lane Q8-many width");
    return width == 1 ? "graph_lanes_q8_many_1" : width == 2 ? "graph_lanes_q8_many_2" : width == 4 ? "graph_lanes_q8_many_4" : "graph_lanes_q8_many_8";
}
bool lane_kernel(const char* name) {
    return name && (std::strcmp(name, "dense_lanes_q8") == 0 || std::strcmp(name, "dense_lanes_q8_rg4") == 0 ||
        std::strcmp(name, "route_lanes_q51") == 0 || std::strcmp(name, "moe_three_down") == 0 ||
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        std::strcmp(name, "graph_grouped_mm") == 0 || std::strcmp(name, "graph_dense_mm_q8") == 0 ||
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        std::strcmp(name, "graph_lanes_q8_three_rows") == 0 || std::strcmp(name, "graph_lanes_q8_three_rows_msg") == 0 ||
#endif
        std::strcmp(name, "graph_dense_q8_staged_f32") == 0 || std::strcmp(name, "graph_dense_q8_staged_f32_m64") == 0 ||
        std::strcmp(name, "graph_dense_q8_multiseg2_f32") == 0 || std::strcmp(name, "graph_dense_q8_multiseg2_f32_m64") == 0 ||
        std::strcmp(name, "graph_dense_q8_multiseg3_f32") == 0 || std::strcmp(name, "graph_dense_q8_multiseg3_f32_m64") == 0 ||
        std::strcmp(name, "nax_prefill_dense_q8_m32") == 0 ||
        std::strcmp(name, "nax_prefill_dense_q8_m8") == 0 ||
        std::strcmp(name, "nax_prefill_dense_q8_swiglu_m32") == 0 ||
        std::strcmp(name, "nax_prefill_dense_q8_swiglu_m8") == 0 ||
        std::strcmp(name, "nax_verify_dense_q8_m4") == 0 ||
        std::strcmp(name, "nax_verify_dense_q8_m8") == 0 ||
        std::strcmp(name, "nax_verify_dense_q8_swiglu_m4") == 0 ||
        std::strcmp(name, "nax_verify_dense_q8_swiglu_m8") == 0 ||
        std::strcmp(name, "nax_m8_dense_q8") == 0 ||
        std::strcmp(name, "nax_m8_qmv_q8") == 0 ||
        std::strcmp(name, "nax_m8_dense_q8_swiglu") == 0 ||
        std::strcmp(name, "nax_grouped_moe") == 0 ||
        std::strcmp(name, "nax_grouped_moe_gate_up_swiglu") == 0 ||
        std::strcmp(name, "nax_verify_moe_routed") == 0 ||
        std::strcmp(name, "nax_verify_moe_gate_up_swiglu") == 0 ||
        std::strcmp(name, "nax_verify_moe_down") == 0 ||
        std::strcmp(name, "nax_m8_moe_gate_up_swiglu") == 0 ||
        std::strcmp(name, "nax_m8_moe_down") == 0 ||
        std::strcmp(name, "graph_grouped_mm_f32") == 0 ||
        std::strcmp(name, "graph_hc_down_q8_window") == 0 ||
        std::strncmp(name, "verify_fused_hc_q8_", 19) == 0 ||
        std::strcmp(name, "graph_lanes_q8_many_1") == 0 || std::strcmp(name, "graph_lanes_q8_many_2") == 0 ||
        std::strcmp(name, "graph_lanes_q8_many_4") == 0 || std::strcmp(name, "graph_lanes_q8_many_8") == 0 ||
        std::strcmp(name, "dflash2_dense_q8_m8") == 0 || std::strcmp(name, "dflash2_dense_q8_r4_m8") == 0 ||
        std::strcmp(name, "dflash2_gate_up_swiglu_q8_m8") == 0);
}
NSUInteger lane_weight_binding_limit(const char* name) {
    if (!name) return 0;
    if (std::strcmp(name, "graph_dense_q8_multiseg3_f32") == 0 || std::strcmp(name, "graph_dense_q8_multiseg3_f32_m64") == 0) return 3;
    if (std::strcmp(name, "graph_dense_q8_multiseg2_f32") == 0 || std::strcmp(name, "graph_dense_q8_multiseg2_f32_m64") == 0 ||
        std::strcmp(name, "nax_prefill_dense_q8_swiglu_m32") == 0 || std::strcmp(name, "nax_prefill_dense_q8_swiglu_m8") == 0 ||
        std::strcmp(name, "nax_verify_dense_q8_swiglu_m4") == 0 || std::strcmp(name, "nax_verify_dense_q8_swiglu_m8") == 0 ||
        std::strcmp(name, "nax_m8_dense_q8_swiglu") == 0 ||
        std::strcmp(name, "nax_grouped_moe_gate_up_swiglu") == 0 || std::strcmp(name, "nax_verify_moe_gate_up_swiglu") == 0 ||
        std::strcmp(name, "nax_m8_moe_gate_up_swiglu") == 0 || std::strcmp(name, "dflash2_gate_up_swiglu_q8_m8") == 0) return 2;
    return lane_kernel(name) ? 1 : 0;
}
#endif
bool planar_v3_kernel(const char* name) {
    return name && (std::strcmp(name, "planar_moe_gate_up_swiglu_m1") == 0 ||
                    std::strcmp(name, "planar_moe_gate_up_swiglu_m4") == 0 ||
                    std::strcmp(name, "planar_moe_gate_up_swiglu_m8") == 0 ||
                    std::strcmp(name, "planar_moe_down_m1") == 0 ||
                    std::strcmp(name, "planar_moe_down_m4") == 0 ||
                    std::strcmp(name, "planar_moe_down_m8") == 0 ||
                    std::strcmp(name, "nax_grouped_moe_gate_up_swiglu_v3") == 0 ||
                    std::strcmp(name, "nax_grouped_moe_down_v3") == 0);
}
NSUInteger planar_v3_weight_binding_limit(const char* name) {
    if (!name) return 0;
    if (std::strcmp(name, "planar_moe_gate_up_swiglu_m1") == 0 ||
        std::strcmp(name, "planar_moe_gate_up_swiglu_m4") == 0 ||
        std::strcmp(name, "planar_moe_gate_up_swiglu_m8") == 0 ||
        std::strcmp(name, "nax_grouped_moe_gate_up_swiglu_v3") == 0) return 2;
    if (std::strcmp(name, "planar_moe_down_m1") == 0 ||
        std::strcmp(name, "planar_moe_down_m4") == 0 ||
        std::strcmp(name, "planar_moe_down_m8") == 0 ||
        std::strcmp(name, "nax_grouped_moe_down_v3") == 0) return 1;
    return 0;
}
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
// Kernels that load the error flag and return early once it is set. Every kernel ORs into the flag, so a reader is ordered against every other dispatch. The current-carrier source test checks this list against the MSL.
constexpr std::array flag_reading_kernels{"qsa_score_accumulate", "qsa_chronological_factors", "qsa_chronological_channels", "qsa_chronological_serial_factors", "qsa_verify_rows_factors", "qsa_verify_rows_channels"
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    , "graph_capture_routes", "carrier_readback_bytes"
#endif
};
bool reads_error_flag(const char* name) {
    return std::any_of(flag_reading_kernels.begin(), flag_reading_kernels.end(), [&](const char* reader) { return std::strcmp(reader, name) == 0; });
}
#endif
std::runtime_error metal_error(const char* operation, NSError* error) {
    return std::runtime_error(std::string(operation) + (error ? ": " + std::string(error.localizedDescription.UTF8String) : " failed"));
}
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
constexpr uint32_t kVerifyNaxSplitSlots = 16u;
constexpr uint64_t kVerifyNaxSplitSlotPartialsBytes = 3'145'728ull;
constexpr uint64_t kVerifyNaxSplitSlotCountersBytes = 4096ull;

uint32_t pick_nax_m8_q8_splits(uint32_t columns, uint32_t out_cols, uint32_t cols_per_tile, bool dual_out) {
    const uint32_t tiles_x = (out_cols + cols_per_tile - 1u) / cols_per_tile;
    if ((columns % 512u) != 0u || columns < 1024u || tiles_x == 0u || tiles_x > 320u || tiles_x > 1024u) {
        return 1u;
    }
    const uint32_t runs = columns / 512u;
    const uint64_t p_stride = uint64_t((out_cols + 15u) & ~15u);
    const uint64_t per_split_bytes = (dual_out ? 2ull : 1ull) * 8ull * p_stride * sizeof(float);
    for (const uint32_t s : {20u, 16u, 12u, 10u, 8u, 6u, 5u, 4u, 3u, 2u}) {
        if (s <= runs && (runs % s) == 0u &&
            uint64_t(tiles_x) * s <= 960ull &&
            uint64_t(s) * per_split_bytes <= kVerifyNaxSplitSlotPartialsBytes) {
            return s;
        }
    }
    return 1u;
}

uint32_t pick_nax_m8_q8_t64_splits(uint32_t columns, uint32_t out_cols, uint32_t cols_per_tile, bool dual_out) {
    const uint32_t tiles_x = (out_cols + cols_per_tile - 1u) / cols_per_tile;
    const uint32_t blocks = columns / 32u;
    if ((columns % 32u) != 0u || blocks < 8u || tiles_x == 0u || tiles_x >= 160u || tiles_x > 1024u) {
        return 1u;
    }
    const uint32_t min_blocks_per_split = (tiles_x <= 32u) ? 2u : 4u;
    const uint64_t p_stride = dual_out ? uint64_t((out_cols + 31u) & ~31u) : uint64_t((out_cols + 63u) & ~63u);
    const uint64_t per_split_bytes = (dual_out ? 2ull : 1ull) * 8ull * p_stride * sizeof(float);
    for (const uint32_t s : {16u, 12u, 10u, 8u, 6u, 5u, 4u, 3u, 2u}) {
        if (s <= blocks && (blocks % s) == 0u && (blocks / s) >= min_blocks_per_split &&
            uint64_t(tiles_x) * s <= 320ull &&
            uint64_t(s) * per_split_bytes <= kVerifyNaxSplitSlotPartialsBytes) {
            return s;
        }
    }
    return 1u;
}

uint32_t pick_nax_m8_bf16_splits(uint32_t columns, uint32_t out_cols) {
    const uint32_t tiles_x = (out_cols + 31u) / 32u;
    if ((columns % 128u) != 0u || columns < 256u || tiles_x == 0u || tiles_x >= 128u || tiles_x > 1024u) {
        return 1u;
    }
    const uint32_t units = columns / 128u;
    const uint64_t per_split_bytes = 8ull * uint64_t(tiles_x) * 32ull * sizeof(float);
    for (const uint32_t s : {10u, 8u, 5u, 4u, 2u}) {
        if (s <= units && (units % s) == 0u &&
            uint64_t(tiles_x) * s <= 320ull &&
            uint64_t(s) * per_split_bytes <= kVerifyNaxSplitSlotPartialsBytes) {
            return s;
        }
    }
    return 1u;
}
#endif
} // namespace

struct MetalGraph::Impl {
    struct BindingAlias {
        id<MTLBuffer> buffer;
        uint64_t span, charged;
    };
    struct Allocation {
        // Destruction order: release Metal buffer before its mapped source.
        std::shared_ptr<const ModelFiles> source;
        id<MTLBuffer> buffer;
        uint64_t requested;
        uint64_t charged;
        bool immutable;
        // Declared last so aliases die before the buffer and its mapped source.
        std::map<std::pair<uint64_t, uint64_t>, BindingAlias> capture_aliases;
    };
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLBuffer> error_flag;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
#if defined(TOKSPUR_PRIVATE_PIPELINE_LOOKUP) && TOKSPUR_PRIVATE_PIPELINE_LOOKUP
    std::unordered_map<PipelineKey, id<MTLComputePipelineState>, PipelineKeyHash, PipelineKeyEqual> pipelines;
#else
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    std::set<std::pair<uint32_t, uint32_t>> grouped_mm_prepared;
    std::set<uint32_t> dense_mm_prepared;
    bool prefill_dense_f32 = false;
    bool prefill_dense_nax = false;
    bool mtp_priming_scope = false;
    bool verify_nax = false;
    bool verify_nax_configured = false;
    bool verify_nax_m8 = false;
    bool verify_nax_m8_configured = false;
    bool target_verification_role = false;
    NaxSplitOverrides target_m8_splits;
    bool verify_nax_prepared = false;
    bool verify_nax_scope = false;
    std::set<std::pair<uint32_t, uint32_t>> verify_nax_moe_prepared;
    GpuView nax_bf16_scratch{};
    GpuView nax_indirect_args{};
    GpuView verify_nax_moe_meta{};
    GpuView verify_nax_split_partials{};
    GpuView verify_nax_split_counters{};
    uint32_t verify_nax_split_cursor = 0;
    bool nax_gathered_broadcast_valid = false;
    size_t nax_gathered_input_slot = 0;
    uint64_t nax_gathered_input_offset = 0;
    uint32_t nax_gathered_columns = 0;
    uint64_t nax_gathered_stride = 0;
    bool nax_gathered_handoff_valid = false;
    size_t nax_gathered_handoff_slot = 0;
    uint64_t nax_gathered_handoff_offset = 0;
    uint32_t nax_gathered_handoff_rows = 0;
    uint32_t nax_gathered_handoff_columns = 0;
    uint64_t nax_gathered_handoff_stride = 0;
    uint64_t nax_gathered_handoff_scratch_offset = 0;
    uint64_t nax_gathered_handoff_scratch_bytes = 0;
    bool nax_dense_bf16_valid = false;
    size_t nax_dense_input_slot = 0;
    uint64_t nax_dense_input_offset = 0;
    uint64_t nax_dense_input_end = 0;
    uint32_t nax_dense_rows = 0;
    uint32_t nax_dense_padded_rows = 0;
    uint32_t nax_dense_columns = 0;
    uint64_t nax_dense_stride = 0;
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    bool prefill_hc_down_f32 = false;
    uint64_t prefill_hc_down_f32_encoded = 0;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    bool moe_router_f32_batch = false, moe_router_f32_prepared = false, moe_router_f32_locked = false;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    bool router_window_enabled = false, router_window_fast = false, router_window_prepared = false, router_window_configured = false, router_window_locked = false;
    std::array<uint64_t, 4> router_window_encoded{};
    GpuView router_margins_scratch;
    bool router_margin_audit_enabled = false;
    float router_margin_epsilon = 1.0e-4f;
    RouterMarginTelemetry router_margin_telemetry{};
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    bool hc_down_window_enabled = false, hc_down_window_prepared = false, hc_down_window_configured = false, hc_down_window_locked = false;
    uint64_t hc_down_window_encoded = 0;
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    bool prefill_grouped_f32 = false;
    bool prefill_grouped_auto = false;
    bool prefill_grouped_nax = false;
    // An explicit active layer lease, never a route lookup cache. Generation never rewinds on abort or new batches.
    bool grouped_routes_active = false;
    uint64_t grouped_routes_generation = 0;
    uint32_t grouped_routes_experts = 0, grouped_routes_selected = 0, grouped_routes_tokens = 0;
    std::array<GpuView, 4> grouped_routes_views{};
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    bool hc_rows_enabled = false, hc_rows_configured = false, hc_rows_locked = false;
    GpuView hc_rows_scratch;
    std::array<uint64_t, 3> hc_rows_encoded{};
#endif
#ifdef TOKSPUR_PRIVATE_GDN_CHUNK
    bool gdn_chunk_prepared = false;
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    bool gdn_legacy_batch_prepared = false;
    uint64_t gdn_legacy_batch_encoded = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
    bool qsa_prefill_prepared = false;
#endif
    std::vector<Allocation> allocations;
    GpuView draft_output_slice_{};
    GpuView draft_embedding_slice_{};
    bool dflash2_prepared = false;
    bool dflash2_scope = false;
    bool logit_topk_prepared = false;
    bool verify_hc_gdn_fused = false;
    bool verify_hc_gdn_fused_prepared = false;
    bool verify_hc_gdn_fused_locked = false;
    GpuView verify_hc_partials_scratch{};
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::optional<LaneMatrixShape> draft_output_shape_{};
#endif
    const uint64_t owner = next_owner.fetch_add(1);
    const std::thread::id thread = std::this_thread::get_id();
    GraphMemory ledger;
    uint64_t dispatches = 0;
    Clock::time_point started;
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    BatchPhaseClockState phase_clock;
#endif
    struct Mark {
        const char* kernel; uint32_t encoding; std::array<uint32_t, 4> shape;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        uint32_t carrier_layer = UINT32_MAX, carrier_selected = 0;
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        Task4ProfileTag task4_tag{};
        std::array<uint64_t, 3> task4_grid{}, task4_group{};
        bool task4_aligned = true;
#endif
    };
    std::vector<id<MTLCounterSampleBuffer>> counters;
    ProfileMode profile_mode = ProfileMode::disabled;
    uint32_t profile_capacity = 0;
    Mark selected{};
    std::vector<Mark> marks;
    GraphProfile profile;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    CarrierDiagnosticWindow carrier_diagnostic;
    std::unique_ptr<FILE, decltype(&std::fclose)> carrier_capture{nullptr, &std::fclose};
    uint64_t carrier_capture_bytes = 0;
    GpuView carrier_route_scratch{};
    struct CarrierRouteRecord { uint32_t layer, rows, selected, experts, offset, row_begin; };
    std::vector<CarrierRouteRecord> carrier_routes;
    uint32_t carrier_route_bytes = 0, carrier_row_offset = 0;
    std::vector<Mark> carrier_records;
    bool carrier_diagnostic_preparing = false;
    uint32_t carrier_layer = UINT32_MAX;
    uint64_t carrier_flag_barriers = 0;
    GpuView carrier_readback_staging{};
    bool carrier_gpu_readback = false;
    uint32_t carrier_readback_commands = 0;
    id<MTLComputePipelineState> carrier_readback_byte_pipeline;
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    Task4ProfileTag task4_tag{};
    Task4ProfileSink task4_sink = nullptr;
    void* task4_context = nullptr;
    uint32_t task4_batches = 0, task4_max_batches = 0;
    bool task4_preparing = false;
    Task4ProfileBoundary task4_boundary = Task4ProfileBoundary::dispatch;
    bool task4_active = false;
    uint32_t task4_encoders = 0;
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    MtpProfileWindow mtp_profile;
    MtpProfileSink mtp_profile_sink = nullptr;
    void* mtp_profile_context = nullptr;
    bool mtp_profile_preparing = false;
    MtpProfileBoundary mtp_profile_boundary = MtpProfileBoundary::dispatch;
#endif
    bool profiling_active() const { return profile_mode != ProfileMode::disabled; }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    uint32_t target_split(NaxSplitFamily family, uint32_t columns, uint32_t outputs, uint32_t tokens,
                          uint32_t tile, bool dual, uint32_t automatic) const {
        return target_m8_splits.choose(target_verification_role && !dflash2_scope && !mtp_priming_scope,
            {family, columns, outputs, tokens, tile, dual}, automatic,
            verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes ? kVerifyNaxSplitSlotPartialsBytes : 0);
    }
#endif
    bool fixed_route_profile_allowed() const {
        return profile_mode == ProfileMode::disabled || std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE")
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            || carrier_diagnostic.active()
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            || (task4_sink && task4_active && profile_mode == (task4_boundary == Task4ProfileBoundary::dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary))
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
            || (mtp_profile.active && mtp_profile_sink &&
                profile_mode == (mtp_profile_boundary == MtpProfileBoundary::dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary))
#endif
            ;
    }
    MTLTimestamp cpu_start = 0, gpu_start = 0;
    Clock::time_point calibration_start;
    uint32_t small_gemv_splits = 0;
    GpuView small_gemv_scratch{};
    uint32_t block_gemv_formats = 0;
#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
    bool native_plain_loads = false;
#endif
    std::set<uint32_t> q8_many_prepared;
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    bool raw_decode = false, raw_decode_locked = false;
    RawDecodeRoutes raw_decode_pending{}, raw_decode_completed{};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    bool q8_three_rows = false, q8_three_locked = false;
    Q8ThreeRoutes q8_three_pending{}, q8_three_completed{};
#endif
    bool swiglu_fused = false, swiglu_fused_locked = false;
    uint64_t swiglu_fused_pending = 0, swiglu_fused_completed = 0;
    Q8GemvRg4Mode q8_gemv_rg4 = Q8GemvRg4Mode::off;
    bool q8_gemv_rg4_locked = false;
    Q8GemvRg4Routes q8_gemv_rg4_pending{}, q8_gemv_rg4_completed{};
    Q8ThreeMsgMode q8_three_msg = Q8ThreeMsgMode::off;
    bool q8_three_msg_locked = false;
    Q8ThreeMsgRoutes q8_three_msg_pending{}, q8_three_msg_completed{};
    bool moe_three_union = false, moe_three_union_locked = false;
    GpuView moe_three_union_meta{};
    uint64_t moe_three_union_pending = 0, moe_three_union_completed = 0;
    PrefillM64Mode prefill_m64 = PrefillM64Mode::off;
    bool prefill_m64_locked = false;
    uint64_t prefill_m64_pending = 0, prefill_m64_completed = 0;
    bool prefill_multiseg = false, prefill_multiseg_locked = false;
    uint64_t prefill_multiseg_pending = 0, prefill_multiseg_completed = 0;
    bool prefill_swizzle = false, prefill_swizzle_locked = false;
    bool qsa_prefill_fused = false, qsa_prefill_fused_locked = false;
    bool hc_consumer_fused = false, hc_consumer_fused_locked = false;
    uint64_t hc_consumer_fused_pending = 0, hc_consumer_fused_completed = 0;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::map<std::tuple<size_t, uint64_t, uint64_t>, LaneMatrixShape> lane_matrices;
    std::set<size_t> lane_slots;
    bool lane_locked = false;
#endif
    std::map<std::tuple<size_t, uint64_t, uint64_t>, PlanarMatrixShape> planar_matrices;
    std::set<size_t> planar_slots;
    bool planar_locked = false;
    GpuView planar_union_meta{};
    // Private post-submit diagnostic only. Capacity follows the registered gate inventory, never a presumed layer count.
    static constexpr uint64_t planar_union_profile_stride = 9280;
    static constexpr uint32_t planar_union_profile_limit = 256;
    struct PlanarUnionProfileRecord {
        uint32_t matrix = 0, tokens = 0, experts = 0, selected = 0;
        uint32_t gu_encoding = 0, down_encoding = 0, channels = 0, hidden = 0;
        uint64_t dispatch = 0;
        bool complete = false;
        std::array<uint32_t, 8> histogram{};
        double union_ms = 0, gu_ms = 0, down_ms = 0;
    };
    bool planar_union_profile_enabled = false, planar_union_profile_binding = false;
    uint32_t planar_union_profile_capacity = 0;
    uint64_t planar_union_profile_batch = 0;
    GpuView planar_union_profile_bank{};
    std::vector<PlanarUnionProfileRecord> planar_union_profile_records;
    std::vector<char> planar_union_profile_output;
    std::set<uint32_t> planar_gate_up_prepared;
    std::set<uint32_t> planar_down_prepared;
    std::set<uint32_t> planar_nax_gate_up_prepared;
    std::set<uint32_t> planar_nax_down_prepared;
    struct Q8MatrixMeta {
        uint32_t columns = 0;
        uint32_t rows = 0;
        uint32_t src_layout = 0;
    };
    struct Q8Tile64Entry {
        GpuView alloc{};
        GpuView plane0{};
        GpuView meta{};
        uint32_t columns = 0;
        uint32_t rows = 0;
        size_t source_slot = 0;
        bool is_alias = false;
    };
    std::map<std::tuple<size_t, uint64_t, uint64_t>, Q8MatrixMeta> q8_matrices;
    std::map<std::tuple<size_t, uint64_t, uint64_t>, Q8Tile64Entry> q8_tile64_entries;
    std::optional<std::tuple<GpuView, uint32_t, uint32_t>> draft_output_parent_;
    bool q8_tile64_ready = false;
#ifdef TOKSPUR_PRIVATE_Q8_MPP
    std::set<Q8MppLayout> q8_mpp_prepared;
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    std::array<bool, 2> dense_q8_staged_prepared{};
    bool prefill_dense_staged_f32 = false, dense_staged_locked = false;
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MPP
    std::set<std::tuple<uint32_t,uint32_t,uint32_t>> grouped_mpp_prepared;
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
    RouteAttributionSink route_sink = nullptr;
    void* route_context = nullptr;
    uint32_t route_phase = 0;
    GpuView route_logits{}, route_ids{}, route_weights{};
#endif
    OperatorCheckpointSink operator_sink = nullptr;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    QsaRouteAttributionSink qsa_route_sink = nullptr;
    void* qsa_route_context = nullptr;
    uint32_t qsa_route_phase = 0;
    std::array<GpuView, 3> qsa_route_backup{};
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    GdnRouteAttributionSink gdn_route_sink = nullptr;
    void* gdn_route_context = nullptr;
    uint32_t gdn_route_phase = 0;
    std::optional<uint32_t> gdn_route_prefill_prefix{};
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    RouteLockSink route_lock_sink = nullptr;
    void* route_lock_context = nullptr;
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    KernelSearchCaptureSink kernel_search_sink = nullptr;
    void* kernel_search_context = nullptr;
    uint32_t kernel_search_mask = 0;
#endif
    void* operator_context = nullptr;
    bool operator_locked = false, operator_cutting = false;
    uint64_t operator_owners = 0, operator_segments = 0, operator_limit = 0, operator_barriers = 0;
    BatchResult operator_rollup{};
    OperatorCheckpointRecord operator_tag{};
    struct OperatorRoutes {
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        HcExactRoutes hc{};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        WideQ8ExactRoutes wide{};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        WideQ8ManyRoutes many{};
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        QsaVerifyRowsRoutes qsa_rows{};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        Q8ThreeRoutes n3{};
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        RawDecodeRoutes decode{};
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        uint64_t stage = 0;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        uint64_t channel = 0;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        uint64_t qsa_sparse_decode = 0, qsa_sparse_verify = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        GdnPortedRoutes gdn_ported{};
#endif
        uint64_t swiglu_fused = 0;
        Q8GemvRg4Routes q8_gemv_rg4{};
        Q8ThreeMsgRoutes q8_three_msg{};
        uint64_t moe_three_union = 0;
        uint64_t prefill_m64 = 0, prefill_multiseg = 0, hc_consumer_fused = 0;
    } operator_routes;
    void operator_collect_routes() {
        const auto add = [](uint64_t& to, uint64_t from) { require(from <= std::numeric_limits<uint64_t>::max() - to, "operator route rollup overflow"); to += from; };
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        for (size_t i = 0; i < hc_exact_pending.size(); ++i) add(operator_routes.hc[i], hc_exact_pending[i]);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        for (size_t i = 0; i < wide_q8_exact_pending.size(); ++i) add(operator_routes.wide[i], wide_q8_exact_pending[i]);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        for (size_t i = 0; i < wide_q8_many_pending.size(); ++i) add(operator_routes.many[i], wide_q8_many_pending[i]);
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        for (size_t i = 0; i < qsa_rows_pending.size(); ++i) add(operator_routes.qsa_rows[i], qsa_rows_pending[i]);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        for (size_t i = 0; i < q8_three_pending.size(); ++i) add(operator_routes.n3[i], q8_three_pending[i]);
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        for (size_t i = 0; i < raw_decode_pending.size(); ++i) add(operator_routes.decode[i], raw_decode_pending[i]);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        add(operator_routes.stage, qsa_stage_v8_pending);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        add(operator_routes.channel, qsa_channel_pending);
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        add(operator_routes.qsa_sparse_decode, qsa_sparse_decode_pending);
        add(operator_routes.qsa_sparse_verify, qsa_sparse_verify_pending);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        for (size_t i = 0; i < gdn_ported_pending.size(); ++i) add(operator_routes.gdn_ported[i], gdn_ported_pending[i]);
#endif
        add(operator_routes.swiglu_fused, swiglu_fused_pending);
        for (size_t i = 0; i < q8_gemv_rg4_pending.size(); ++i) add(operator_routes.q8_gemv_rg4[i], q8_gemv_rg4_pending[i]);
        for (size_t i = 0; i < q8_three_msg_pending.size(); ++i) add(operator_routes.q8_three_msg[i], q8_three_msg_pending[i]);
        add(operator_routes.moe_three_union, moe_three_union_pending);
        add(operator_routes.prefill_m64, prefill_m64_pending);
        add(operator_routes.prefill_multiseg, prefill_multiseg_pending);
        add(operator_routes.hc_consumer_fused, hc_consumer_fused_pending);
    }
    void operator_publish_routes(const OperatorRoutes& routes) {
        const auto add = [](uint64_t& to, uint64_t from) { require(from <= std::numeric_limits<uint64_t>::max() - to, "operator route publication overflow"); to += from; };
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        for (size_t i = 0; i < hc_exact_completed.size(); ++i) add(hc_exact_completed[i], routes.hc[i]);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        for (size_t i = 0; i < wide_q8_exact_completed.size(); ++i) add(wide_q8_exact_completed[i], routes.wide[i]);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        for (size_t i = 0; i < wide_q8_many_completed.size(); ++i) add(wide_q8_many_completed[i], routes.many[i]);
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        for (size_t i = 0; i < qsa_rows_completed.size(); ++i) add(qsa_rows_completed[i], routes.qsa_rows[i]);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        for (size_t i = 0; i < q8_three_completed.size(); ++i) add(q8_three_completed[i], routes.n3[i]);
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        for (size_t i = 0; i < raw_decode_completed.size(); ++i) add(raw_decode_completed[i], routes.decode[i]);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        add(qsa_stage_v8_completed, routes.stage);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        add(qsa_channel_completed, routes.channel);
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        add(qsa_sparse_decode_completed, routes.qsa_sparse_decode);
        add(qsa_sparse_verify_completed, routes.qsa_sparse_verify);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        for (size_t i = 0; i < gdn_ported_completed.size(); ++i) add(gdn_ported_completed[i], routes.gdn_ported[i]);
#endif
        add(swiglu_fused_completed, routes.swiglu_fused);
        for (size_t i = 0; i < q8_gemv_rg4_completed.size(); ++i) add(q8_gemv_rg4_completed[i], routes.q8_gemv_rg4[i]);
        for (size_t i = 0; i < q8_three_msg_completed.size(); ++i) add(q8_three_msg_completed[i], routes.q8_three_msg[i]);
        add(moe_three_union_completed, routes.moe_three_union);
        add(prefill_m64_completed, routes.prefill_m64);
        add(prefill_multiseg_completed, routes.prefill_multiseg);
        add(hc_consumer_fused_completed, routes.hc_consumer_fused);
    }

    void operator_add(const BatchResult& physical) {
        require(physical.dispatches <= std::numeric_limits<uint64_t>::max() - operator_rollup.dispatches &&
            physical.capture_executes <= std::numeric_limits<uint64_t>::max() - operator_rollup.capture_executes, "operator checkpoint rollup overflow");
        operator_rollup.dispatches += physical.dispatches; operator_rollup.capture_executes += physical.capture_executes;
        operator_rollup.wall_ms += physical.wall_ms; operator_rollup.gpu_ms += physical.gpu_ms;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        require(completed_concurrent_barriers <= std::numeric_limits<uint64_t>::max() - operator_barriers, "operator checkpoint barrier overflow");
        operator_barriers += completed_concurrent_barriers;
#endif
    }
    void operator_emit(const char* category, const char* tag, std::array<uint32_t, 4> dimensions, bool final, const BatchResult* physical, const BatchResult& total) {
        require(operator_segments < operator_limit, "operator checkpoint record bound exhausted");
        auto record = operator_tag; record.category = category; record.tag = tag; record.dimensions = dimensions;
        record.owner_batch = operator_owners; record.segment = operator_segments++; record.owner_final = final;
        require(operator_sink(operator_context, record, physical, total), "operator checkpoint sink refused; diagnostic invalidated");
    }
#endif
    bool q8_split_prepared = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    bool mtp_head_prepared = false, mtp_head_tiled_prepared = false;
#endif
    HcQ8Policy hc_q8{};
    GpuView hc_q8_scratch{};
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    HcExactMode hc_exact = HcExactMode::off;
    bool hc_exact_locked = false;
    HcExactRoutes hc_exact_pending{}, hc_exact_completed{};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    bool wide_q8_exact = false, wide_q8_exact_locked = false;
    WideQ8ExactRoutes wide_q8_exact_pending{}, wide_q8_exact_completed{};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    bool wide_q8_many = false, wide_q8_many_locked = false;
    WideQ8ManyRoutes wide_q8_many_pending{}, wide_q8_many_completed{};
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    bool qsa_stage_v8_locked = false, qsa_score_stage_prepared = false;
    QsaStageV8Policy qsa_stage_v8{};
    GpuView qsa_stage_v8_scratch{};
    uint64_t qsa_stage_v8_pending = 0, qsa_stage_v8_completed = 0;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    bool qsa_channel_locked = false;
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    bool kernel_qsa_f32 = false, kernel_qsa_selected = false;
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    bool kernel_grouped_fp16 = false, kernel_grouped_selected = false;
    uint64_t kernel_grouped_encoded = 0;
    const char* grouped_step2_kernel(const char* original) const {
        return kernel_grouped_fp16 && std::strcmp(original, grouped_mm_f32_kernel) == 0 ? "graph_grouped_search_fp16" : original;
    }
#endif
#endif
    QsaChannelPolicy qsa_channel{};
    GpuView qsa_channel_factors{};
    uint64_t qsa_channel_pending = 0, qsa_channel_completed = 0;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
#ifndef TOKSPUR_QSA_SPARSE_DECODE_BN
#define TOKSPUR_QSA_SPARSE_DECODE_BN 32
#endif
#ifndef TOKSPUR_QSA_SPARSE_VERIFY_BN
#define TOKSPUR_QSA_SPARSE_VERIFY_BN 32
#endif
    static constexpr uint32_t qsa_sparse_decode_bn = TOKSPUR_QSA_SPARSE_DECODE_BN;
    static constexpr uint32_t qsa_sparse_verify_bn = TOKSPUR_QSA_SPARSE_VERIFY_BN;
    static constexpr uint32_t qsa_sparse_decode_threads = 32u * qsa_sparse_decode_bn;
    static constexpr uint32_t qsa_sparse_verify_threads = 32u * qsa_sparse_verify_bn;
    bool qsa_sparse_configured = false, qsa_sparse_enabled = false, qsa_sparse_locked = false, qsa_sparse_variants_prepared = false;
    uint64_t qsa_sparse_decode_pending = 0, qsa_sparse_decode_completed = 0;
    uint64_t qsa_sparse_verify_pending = 0, qsa_sparse_verify_completed = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    bool qsa_rows_prepared = false, qsa_rows_locked = false;
    QsaVerifyRowsRoutes qsa_rows_pending{}, qsa_rows_completed{};
#endif
    uint32_t gdn_tile_threads = 0;
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
#ifndef TOKSPUR_GDN_VERIFY_DB
#define TOKSPUR_GDN_VERIFY_DB 8
#endif
    static constexpr uint32_t gdn_verify_db = TOKSPUR_GDN_VERIFY_DB;
    bool gdn_ported_configured = false, gdn_ported_enabled = false, gdn_ported_locked = false, gdn_ported_variants_prepared = false;
    bool gdn_prefill_ported_configured = false, gdn_prefill_ported_enabled = false, gdn_prefill_ported_locked = false;
    GdnPortedRoutes gdn_ported_pending{}, gdn_ported_completed{};
    bool gdn_route_shadow_active() const {
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        return gdn_route_phase == 2;
#else
        return false;
#endif
    }
    bool gdn_use_ported_active() const {
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        if (gdn_route_phase == 3) return false;
        if (gdn_route_phase == 2) return true;
#endif
        return gdn_ported_enabled;
    }
#endif
    struct Capture {
        id<MTLBuffer> parameters;
        id<MTLIndirectCommandBuffer> commands;
        std::map<size_t, bool> leases; // Allocation slot -> writable anywhere.
        std::map<std::tuple<size_t, uint64_t, uint64_t>, id<MTLBuffer>> aliases;
        std::vector<id<MTLResource>> reads, writes;
        std::array<uint32_t, 3> policy{};
        uint32_t capacity = 0, count = 0;
        uint64_t charged = 0;
    };
    std::vector<Capture> captures;
    Capture draft;
    bool recording = false, pending_capture_dispatch = false, compatible_pipelines = false;
    uint64_t capture_executes = 0;
    static constexpr uint32_t capture_bindings = 16, capture_constant_stride = 256;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    // Arm B: byte ranges read and written since the last barrier, and the current dispatch's bindings.
    struct Span { size_t slot; uint64_t begin, end; };
    bool concurrent = false, concurrent_locked = false, pending_flag_reader = false, flag_written = false, flag_read = false;
    std::vector<Span> span_reads, span_writes;
    std::vector<std::pair<Span, bool>> span_pending;
    uint64_t concurrent_barrier_count = 0, completed_concurrent_barriers = 0;
    uint64_t lifetime_dispatch_count = 0, lifetime_command_count = 0, lifetime_barrier_count = 0;
    static bool touches(const std::vector<Span>& spans, const Span& span) {
        return std::any_of(spans.begin(), spans.end(), [&](const Span& other) { return other.slot == span.slot && other.begin < span.end && span.begin < other.end; });
    }
    void reset_spans() { span_reads.clear(); span_writes.clear(); span_pending.clear(); flag_written = flag_read = pending_flag_reader = false; }
    // Called before each concurrent dispatch is encoded: RAW and WAW against ranges written since the barrier, WAR against ranges read since it, and the error flag (every kernel ORs it, readers also load it).
    void concurrent_fence();
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    // Dispatch-graph tooling: records only, never changes encoding or bytes.
    std::FILE* trace = nullptr;
    uint64_t trace_batch = 0;
    bool trace_pending = false;
    Json::Object trace_dispatch;
    Json::Array trace_bindings, trace_constants;
    std::set<std::string> trace_pipelines;
    void trace_record(Json::Object record) const {
        if (!trace) return;
        const auto line = Json(std::move(record)).dump() + "\n";
        require(std::fwrite(line.data(), 1, line.size(), trace) == line.size() && std::fflush(trace) == 0,
            "dispatch trace write failed");
    }
    static std::string trace_digest(const void* data, uint64_t bytes) {
        CC_SHA256_CTX context; CC_SHA256_Init(&context);
        const auto* at = static_cast<const unsigned char*>(data);
        for (uint64_t done = 0; done < bytes;) {
            const auto part = static_cast<CC_LONG>(std::min<uint64_t>(bytes - done, uint64_t(1) << 30));
            CC_SHA256_Update(&context, at + done, part); done += part;
        }
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256_Final(digest.data(), &context);
        static constexpr char digits[] = "0123456789abcdef";
        std::string hex; for (const auto byte : digest) { hex += digits[byte >> 4]; hex += digits[byte & 15]; }
        return hex;
    }
    static Json trace_range(GpuView view) {
        return Json(Json::Object{{"slot", Json::integer(view.slot)}, {"offset", Json::integer(view.offset)}, {"bytes", Json::integer(view.bytes)}});
    }
    void trace_view(const char* event, GpuView view, const void* contents) const {
        if (!trace) return;
        trace_record({{"event", Json(event)}, {"slot", Json::integer(view.slot)}, {"offset", Json::integer(view.offset)},
            {"bytes", Json::integer(view.bytes)}, {"sha256", Json(trace_digest(contents, view.bytes))}});
    }
    void trace_bytes(const char* event, GpuView storage, uint64_t offset, uint64_t bytes, const void* contents) const {
        storage.offset = checked_add(storage.offset, offset);
        storage.bytes = bytes;
        trace_view(event, storage, contents);
    }
    void trace_allocation(size_t slot, const std::string& label, const char* kind, bool digest) const {
        if (!trace) return;
        const auto& item = allocations[slot];
        trace_record({{"event", Json("allocation")}, {"slot", Json::integer(slot)}, {"label", Json(label)}, {"kind", Json(kind)},
            {"bytes", Json::integer(item.requested)}, {"immutable", Json(item.immutable)},
            {"sha256", digest ? Json(trace_digest(item.buffer.contents, item.requested)) : Json()}});
    }
    void trace_pipeline(const std::string& key, const char* name, uint32_t type, uint32_t minimum_threads, id<MTLFunction> function) {
        if (!trace || !trace_pipelines.insert(key).second) return;
        // Reflection only: a separate compile whose state is discarded.
        NSError* error = nil; MTLAutoreleasedComputePipelineReflection reflection = nil;
        const auto reflected = [device newComputePipelineStateWithFunction:function options:MTLPipelineOptionBindingInfo
            reflection:&reflection error:&error];
        if (!reflected || !reflection) throw metal_error("reflecting owned graph pipeline", error);
        Json::Array bindings;
        for (id<MTLBinding> binding in reflection.bindings) {
            if (binding.type != MTLBindingTypeBuffer) continue;
            const char* access = binding.access == MTLBindingAccessReadOnly ? "read" :
                binding.access == MTLBindingAccessWriteOnly ? "write" : "read_write";
            bindings.push_back(Json(Json::Object{{"index", Json::integer(binding.index)}, {"name", Json(binding.name.UTF8String)},
                {"access", Json(access)}, {"used", Json(bool(binding.used))}}));
        }
        trace_record({{"event", Json("pipeline")}, {"key", Json(key)}, {"function", Json(name)},
            {"constant", type == uint32_t(-1) ? Json() : Json::integer(type)}, {"threads", Json::integer(minimum_threads)},
            {"bindings", Json(std::move(bindings))}});
    }
#endif

    void thread_check() const { require(thread == std::this_thread::get_id(), "Metal graph used from another host thread"); }
    void idle() const { thread_check(); require(!command && !recording, "operation is forbidden during a Metal batch/capture"); }
    void active() const { thread_check(); require(command != nil || recording, "no Metal batch/capture is active"); }
    id<MTLIndirectComputeCommand> captured_command() {
        require(recording && pending_capture_dispatch && draft.count < draft.capacity, "no pending captured command");
        return [draft.commands indirectComputeCommandAtIndex:draft.count];
    }
    void clear_draft() {
        draft.commands = nil; draft.parameters = nil;
        ledger.live_bytes -= draft.charged; ledger.capture_resource_bytes -= draft.charged;
        draft = {}; recording = false; pending_capture_dispatch = false;
    }
    const Allocation& allocation(GpuView view, bool writable = false) const {
        TOKSPUR_CARRIER_PHASE(structural);
        thread_check();
        require(view.owner == owner && view.slot < allocations.size(), "foreign or invalid GPU view");
        const auto& result = allocations[view.slot];
        require(result.buffer && view.offset <= result.requested && view.bytes && view.bytes <= result.requested - view.offset, "released or out-of-bounds GPU view");
        require(!writable || !result.immutable, "immutable GPU allocation cannot be an output");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (writable && grouped_routes_active) for (const auto& route : grouped_routes_views)
            require(view.slot != route.slot || view.offset >= route.offset + route.bytes || route.offset >= view.offset + view.bytes,
                "active grouped route IDs and metadata are read-only until end");
#endif
        require(view.offset % 4 == 0, "GPU view offset must be four-byte aligned");
        return result;
    }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    const LaneMatrixShape* lane_shape(GpuView view) const {
        if (!lane_slots.contains(view.slot)) return nullptr;
        const auto found = lane_matrices.find({view.slot, view.offset, view.bytes});
        if (found != lane_matrices.end()) return &found->second;
        if (draft_output_shape_.has_value() && draft_output_slice_.bytes &&
            view.slot == draft_output_slice_.slot &&
            view.offset == draft_output_slice_.offset &&
            view.bytes == draft_output_slice_.bytes) {
            return &(*draft_output_shape_);
        }
            for (const auto& [key, shape] : lane_matrices) {
            const auto [slot, offset, bytes] = key;
            if (slot == view.slot && overlaps(view, {owner, slot, offset, bytes}))
                throw FormatError("unregistered subview of packed weight storage");
        }
        return nullptr;
    }
#endif
    const PlanarMatrixShape* planar_shape(GpuView view) const {
        if (!planar_slots.contains(view.slot)) return nullptr;
        const auto found = planar_matrices.find({view.slot, view.offset, view.bytes});
        if (found != planar_matrices.end()) return &found->second;
        for (const auto& [key, shape] : planar_matrices) {
            static_cast<void>(shape);
            const auto [slot, offset, bytes] = key;
            if (slot == view.slot && overlaps(view, {owner, slot, offset, bytes}))
                throw FormatError("unregistered subview of planar v3 weight storage");
        }
        return nullptr;
    }
    const Q8Tile64Entry* q8_tile64_entry(GpuView view, uint32_t columns, uint32_t rows) const {
        const auto found = q8_tile64_entries.find({view.slot, view.offset, view.bytes});
        if (found == q8_tile64_entries.end()) return nullptr;
        if (found->second.columns != columns || found->second.rows != rows) return nullptr;
        return &found->second;
    }
    void refresh_draft_output_tile64_alias() {
        if (!draft_output_slice_.bytes || !draft_output_parent_.has_value()) return;
        const auto [parent_view, entries, channels] = *draft_output_parent_;
        if (!entries || (entries % 64u) != 0u || !channels || (channels % 32u) != 0u) return;
        const auto sub_key = std::tuple{draft_output_slice_.slot, draft_output_slice_.offset, draft_output_slice_.bytes};
        if (q8_tile64_entries.contains(sub_key)) return;
        const auto parent_it = q8_tile64_entries.find({parent_view.slot, parent_view.offset, parent_view.bytes});
        if (parent_it == q8_tile64_entries.end()) return;
        const uint64_t g_blocks = channels / 32u;
        const uint64_t sub_nt = entries / 64u;
        const uint64_t p0_bytes = sub_nt * g_blocks * 2048ull;
        const uint64_t m_bytes = sub_nt * g_blocks * 128ull;
        if (p0_bytes > parent_it->second.plane0.bytes || m_bytes > parent_it->second.meta.bytes) return;
        const GpuView sub_p0{owner, parent_it->second.plane0.slot, parent_it->second.plane0.offset, p0_bytes};
        const GpuView sub_m{owner, parent_it->second.meta.slot, parent_it->second.meta.offset, m_bytes};
        q8_tile64_entries.emplace(sub_key, Q8Tile64Entry{{}, sub_p0, sub_m, channels, entries, parent_view.slot, true});
    }
    void ensure_q8_tile64_all() {
        struct RepackQ8Tile64Args {
            uint32_t columns, rows, row_bytes, src_layout;
        };
        static_assert(sizeof(RepackQ8Tile64Args) == 16, "RepackQ8Tile64Args ABI");
        idle();
        if (ledger.live_captures || ![device supportsFamily:MTLGPUFamilyMetal4]) return;
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (trace) return;
#endif
        @autoreleasepool {
            id<MTLComputePipelineState> pipe = nil;
            id<MTLCommandBuffer> cmd = nil;
            id<MTLComputeCommandEncoder> enc = nil;
            decltype(q8_tile64_entries) pending;
            bool committed = false;
            const uint64_t page = static_cast<uint64_t>(getpagesize());
            const uint64_t reserve = std::min<uint64_t>(128ull << 20, ledger.budget / 2ull);
            try {
                for (const auto& [key, info] : q8_matrices) {
                    if (q8_tile64_entries.contains(key)) continue;
                    const auto [slot, offset, bytes] = key;
                    static_cast<void>(bytes);
                    if (slot >= allocations.size() || !allocations[slot].buffer) continue;
                    const uint64_t g_blocks = info.columns / 32u;
                    const uint64_t nt = (info.rows + 63u) / 64u;
                    const uint64_t p0_bytes = checked_mul(checked_mul(nt, g_blocks), 2048ull);
                    const uint64_t m_bytes = checked_mul(checked_mul(nt, g_blocks), 128ull);
                    const uint64_t total_bytes = checked_add(p0_bytes, m_bytes);
                    if (!total_bytes || total_bytes > ledger.max_buffer_bytes || allocations.size() >= 65535) continue;
                    const uint64_t padded = checked_add(total_bytes, page - 1) / page * page;
                    if (ledger.live_bytes + reserve > ledger.budget ||
                        padded > ledger.budget - ledger.live_bytes - reserve) {
                        continue;
                    }
                    if (!pipe) pipe = pipeline("nax_repack_q8_tile64", uint32_t(-1), 64);
                    // Allocate the owning map node first so any subsequent failure can release the unpublished buffer.
                    auto& entry = pending.try_emplace(key).first->second;
                    const GpuView alloc = allocate(total_bytes, "q8_tile64", true, nullptr);
                    const GpuView p0{owner, alloc.slot, 0, p0_bytes};
                    const GpuView m{owner, alloc.slot, p0_bytes, m_bytes};
                    entry = Q8Tile64Entry{alloc, p0, m, info.columns, info.rows, slot, false};
                    if (!cmd) {
                        std::memset(error_flag.contents, 0, 4);
                        cmd = [queue commandBuffer];
                        require(cmd != nil, "cannot create Q8_tile64 repack command buffer");
                        enc = [cmd computeCommandEncoder];
                        require(enc != nil, "cannot create Q8_tile64 repack encoder");
                        [enc setComputePipelineState:pipe];
                        [enc setBuffer:error_flag offset:0 atIndex:7];
                    }
                    const RepackQ8Tile64Args args{info.columns, info.rows, (info.columns / 32u) * 34u, info.src_layout};
                    [enc setBuffer:allocations[slot].buffer offset:offset atIndex:0];
                    [enc setBuffer:allocations[alloc.slot].buffer offset:0 atIndex:1];
                    [enc setBuffer:allocations[alloc.slot].buffer offset:p0_bytes atIndex:2];
                    [enc setBytes:&args length:sizeof(args) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(g_blocks, nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }
                if (enc) {
                    [enc endEncoding];
                    enc = nil;
                    [cmd commit];
                    committed = true;
                    [cmd waitUntilCompleted];
                    require(cmd.status == MTLCommandBufferStatusCompleted, "Q8_tile64 repack GPU command failed");
                    uint32_t err = 0;
                    std::memcpy(&err, error_flag.contents, 4);
                    require(err == 0, "Q8_tile64 repack kernel reported error");
                }
                // Node transfer cannot allocate. Only completed, validated planes become available to consumers.
                q8_tile64_entries.merge(pending);
            } catch (...) {
                if (enc) [enc endEncoding];
                if (committed) [cmd waitUntilCompleted];
                enc = nil;
                cmd = nil;
                // These buffers were never published, imported, traced or leased by captures.
                for (const auto& [key, entry] : pending) {
                    static_cast<void>(key);
                    if (!entry.alloc.owner) continue;
                    auto& item = allocations[entry.alloc.slot];
                    ledger.live_bytes -= item.charged;
                    item.buffer = nil;
                }
                throw;
            }
        }
        refresh_draft_output_tile64_alias();
    }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    void prefill_weight(GpuView view, PrefillWeightLayout layout, uint32_t type,
                        uint32_t columns, uint32_t rows, uint32_t experts) const {
        static_cast<void>(allocation(view));
        require(!planar_shape(view), "planar v3 weight view cannot be consumed by legacy prefill layout");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        const auto* shape = lane_shape(view);
        if (layout == PrefillWeightLayout::gguf) require(!shape, "row-major prefill cannot consume packed weights");
        else require(shape && shape->encoding == type && shape->columns == columns &&
            shape->rows == rows && shape->experts == experts &&
            ((layout == PrefillWeightLayout::q8_lanes_v1 && type == 8) ||
             (layout == PrefillWeightLayout::q51_lanes_v1 && type == 7)),
            "prefill layout does not match the registered weight identity");
#else
        static_cast<void>(type); static_cast<void>(columns); static_cast<void>(rows); static_cast<void>(experts);
        require(layout == PrefillWeightLayout::gguf, "packed prefill requires the lane-layout build");
#endif
    }
#endif
    void bind(GpuView view, NSUInteger index, bool output = false) {
        TOKSPUR_CARRIER_PHASE(encoding);
        const auto& item = allocation(view, output);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        if (output && nax_dense_bf16_valid) {
            const uint64_t dense_bf16_end = uint64_t(nax_dense_padded_rows) * uint64_t(nax_dense_columns) * 2ull;
            if ((nax_bf16_scratch.bytes && view.slot == nax_bf16_scratch.slot && view.offset < dense_bf16_end) ||
                (view.slot == nax_dense_input_slot && view.offset < nax_dense_input_end &&
                 nax_dense_input_offset < checked_add(view.offset, view.bytes))) {
                nax_dense_bf16_valid = false;
            }
        }
        if (output && nax_gathered_handoff_valid && nax_bf16_scratch.bytes && view.slot == nax_bf16_scratch.slot &&
            view.offset < checked_add(nax_gathered_handoff_scratch_offset, nax_gathered_handoff_scratch_bytes) &&
            nax_gathered_handoff_scratch_offset < checked_add(view.offset, view.bytes)) {
            nax_gathered_handoff_valid = false;
        }
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (lane_shape(view)) require(index < lane_weight_binding_limit(selected.kernel) && !output && !recording && lane_kernel(selected.kernel),
            "packed weight view requires a direct layout-aware matrix consumer");
#endif
        if (planar_shape(view)) require(index < planar_v3_weight_binding_limit(selected.kernel) && !output && !recording && planar_v3_kernel(selected.kernel),
            "planar v3 weight view requires a direct planar-v3 matrix consumer");
        if (recording) {
            require(index < capture_bindings, "captured buffer binding limit");
            draft.leases[view.slot] = draft.leases[view.slot] || output;
            id<MTLBuffer> buffer = item.buffer;
            auto offset = view.offset;
            if (offset > UINT32_MAX) {
                // The local M5 API probe observes low-32-bit offset wrapping in
                // compute ICB bindings. Ordinary encoding is unaffected. Rebase
                // immutable mapped input bindings onto the same existing pages;
                // neither bytes nor arithmetic change. Other large bindings
                // refuse rather than silently truncate.
                require(!output && item.immutable && item.source,
                    "large captured offsets require immutable mapped weights");
                const auto page = uint64_t(getpagesize());
                const auto base = offset / page * page;
                const auto end = checked_add(checked_add(offset, view.bytes), page - 1) / page * page;
                const auto length = end - base;
                const auto key = std::pair{base, length};
                auto& aliases = allocations[view.slot].capture_aliases;
                auto found = aliases.find(key);
                if (found == aliases.end()) {
                    @autoreleasepool {
                    require(ledger.capture_binding_aliases < 4096, "captured binding alias count limit");
                    const auto address = reinterpret_cast<uintptr_t>(item.buffer.contents);
                    require(base % page == 0 && address % page == 0 && length % page == 0 &&
                        base <= item.requested && length && length <= item.requested - base &&
                        length <= ledger.max_buffer_bytes,
                        "unaligned or oversized captured binding alias");
                    require(length <= ledger.budget - ledger.live_bytes,
                        "captured binding alias resource exceeds graph budget");
                    id<MTLBuffer> alias = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(address + base)
                        length:length options:MTLResourceStorageModeShared deallocator:nil];
                    require(alias != nil, "Metal refused captured binding alias");
                    // Shared CPU pages do not imply free GPU resource residency.
                    // Charge the complete alias resource, not allocatedSize-length.
                    const auto charged = std::max(length, uint64_t(alias.allocatedSize));
                    require(charged <= ledger.budget - ledger.live_bytes,
                        "actual captured binding alias resource exceeds graph budget");
                    const auto span = checked_add(ledger.capture_binding_alias_span_bytes, length);
                    const auto extra = checked_add(ledger.capture_binding_alias_resource_bytes, charged);
                    alias.label = @"owned immutable no-copy capture binding alias";
                    found = aliases.emplace(key, BindingAlias{alias, length, charged}).first;
                    ledger.live_bytes += charged;
                    ledger.peak_bytes = std::max(ledger.peak_bytes, ledger.live_bytes);
                    ++ledger.allocations; ++ledger.capture_binding_aliases;
                    ledger.capture_binding_alias_span_bytes = span;
                    ledger.capture_binding_alias_resource_bytes = extra;
                    }
                }
                buffer = found->second.buffer;
                offset -= base;
                require(offset < page && offset <= found->second.span && view.bytes <= found->second.span - offset,
                    "captured binding exceeds rebased source span");
                draft.aliases.emplace(std::tuple{view.slot, base, length}, buffer);
            }
            require(offset <= UINT32_MAX, "unsafe captured buffer offset");
            [captured_command() setKernelBuffer:buffer offset:offset atIndex:index];
        } else [encoder setBuffer:item.buffer offset:view.offset atIndex:index];
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (concurrent && !recording) span_pending.push_back({{view.slot, view.offset, view.offset + view.bytes}, output});
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (trace && !recording) {
            require(trace_pending, "traced binding outside a dispatch");
            trace_bindings.push_back(Json(Json::Object{{"index", Json::integer(index)}, {"slot", Json::integer(view.slot)},
                {"offset", Json::integer(view.offset)}, {"bytes", Json::integer(view.bytes)}, {"access", Json(output ? "write" : "read")}}));
        }
#endif
    }
    uint32_t floats(GpuView view, bool output = false) const {
        static_cast<void>(allocation(view, output));
        require(view.bytes % 4 == 0 && view.bytes / 4 <= 16'777'216, "unsupported GPU float-vector shape");
        return static_cast<uint32_t>(view.bytes / 4);
    }
    uint64_t floats64(GpuView view, bool output = false) const {
        static_cast<void>(allocation(view, output));
        require(view.bytes % 4 == 0 && view.bytes / 4 <= 67'108'864ull, "unsupported GPU float-matrix shape");
        return view.bytes / 4;
    }
    id<MTLComputePipelineState> pipeline(const char* name, uint32_t type = uint32_t(-1), uint32_t minimum_threads = 128, bool indirect = false, uint32_t layout = uint32_t(-1)) {
        require(minimum_threads == 128 || (std::strcmp(name, "graph_gdn_delta_tile") == 0 &&
            (minimum_threads == 32 || minimum_threads == 64))
#ifdef TOKSPUR_PRIVATE_HC_EXACT
            || (minimum_threads == 32 && hc_exact_pipeline(name))
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
            || (minimum_threads == qsa_channel_threads &&
                (std::strcmp(name, "qsa_chronological_factors") == 0 || std::strcmp(name, "qsa_chronological_channels") == 0
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
                || std::strcmp(name, "qsa_channels_search_f32") == 0 || std::strcmp(name, "qsa_rows_channels_search_f32") == 0
#endif
                ))
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            || (minimum_threads == 256 && (std::strcmp(name, "qsa_verify_rows_factors") == 0 ||
                std::strcmp(name, "qsa_verify_rows_channels") == 0))
#endif
#ifdef TOKSPUR_PRIVATE_Q8_MPP
            || (minimum_threads == 32 && (std::strcmp(name, "graph_q8_mpp") == 0 || std::strcmp(name, "graph_q8_mpp_wide") == 0))
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
            || (minimum_threads == 64 && (std::strcmp(name, "raw_decode_q4_k") == 0 || std::strcmp(name, "raw_decode_q5_1") == 0))
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MPP
            || (minimum_threads == 32 && (std::strcmp(name, "graph_grouped_mpp16") == 0 || std::strcmp(name, "graph_grouped_mpp32") == 0))
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
            || (minimum_threads == 32 && (std::strcmp(name, "qsa_prefill_attention_partial128") == 0 || std::strcmp(name, "qsa_prefill_attention_reduce") == 0))
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
            || (minimum_threads == 512 && std::strcmp(name, "graph_hc_down_q8_window") == 0)
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
            || ((minimum_threads == 256 || minimum_threads == 512 || minimum_threads == 1024) && std::strncmp(name, "qsa_sparse_attention", 20) == 0)
#endif
            || ((minimum_threads == 128 || minimum_threads == 256) && std::strncmp(name, "dflash2_", 8) == 0)
            || ((minimum_threads == 128 || minimum_threads == 256) &&
                (std::strncmp(name, "logit_topk_", 11) == 0 || std::strncmp(name, "verify_fused_", 13) == 0))
            || (minimum_threads == 256 && (std::strcmp(name, "graph_swiglu") == 0 ||
                std::strcmp(name, "graph_q8_gemv_rg4") == 0 || std::strcmp(name, "dense_lanes_q8_rg4") == 0 ||
                std::strcmp(name, "graph_q8_three_rows_msg") == 0 || std::strcmp(name, "graph_lanes_q8_three_rows_msg") == 0 ||
                std::strncmp(name, "nax_verify_dense_q8_", 20) == 0 ||
                std::strncmp(name, "nax_verify_dense_bf16_", 22) == 0 ||
                std::strcmp(name, "nax_verify_bf16_m8") == 0 ||
                std::strcmp(name, "nax_grouped_routes") == 0 ||
                std::strcmp(name, "nax_qsa_pack_queries_bf16") == 0 ||
                std::strcmp(name, "nax_qsa_build_tile_lists") == 0 ||
                std::strcmp(name, "nax_qsa_attention_g12") == 0 ||
                std::strcmp(name, "nax_qsa_attention_g12_reduce") == 0 ||
                std::strncmp(name, "nax_qsa_", 8) == 0))
            || (minimum_threads == 64 && (std::strcmp(name, "nax_verify_moe_routed") == 0 ||
                std::strcmp(name, "nax_m8_dense_q8") == 0 ||
                std::strcmp(name, "nax_m8_dense_q8_t64") == 0 ||
                std::strcmp(name, "nax_m8_dense_q8_swiglu_t64") == 0 ||
                std::strcmp(name, "nax_repack_q8_tile64") == 0 ||
                std::strcmp(name, "nax_m8_moe_gate_up_swiglu") == 0 ||
                std::strcmp(name, "nax_m8_moe_down") == 0 ||
                std::strcmp(name, "nax_grouped_moe_gate_up_swiglu_v3") == 0 ||
                std::strcmp(name, "nax_grouped_moe_down_v3") == 0))
            || (minimum_threads == 32 && (std::strcmp(name, "moe_three_build_union") == 0 ||
                std::strcmp(name, "nax_verify_moe_build_union") == 0 ||
                std::strcmp(name, "planar_moe_build_union_m8") == 0 ||
                std::strcmp(name, "verify_fused_hc_write_rms_rows") == 0))
            , "unsupported threadgroup capability for the selected kernel");
#if !defined(TOKSPUR_PRIVATE_PIPELINE_LOOKUP) || !TOKSPUR_PRIVATE_PIPELINE_LOOKUP || defined(TOKSPUR_PRIVATE_DISPATCH_GRAPH)
        const auto text_key = [&] {
            std::string key = std::string(name) + ":" + std::to_string(type);
            if (layout != uint32_t(-1)) key += ":layout=" + std::to_string(layout);
            if (minimum_threads != 128) key += ":" + std::to_string(minimum_threads);
            if (indirect) key += ":indirect";
            return key;
        };
#endif
#if defined(TOKSPUR_PRIVATE_PIPELINE_LOOKUP) && TOKSPUR_PRIVATE_PIPELINE_LOOKUP
        const PipelineKeyView key{name, type, minimum_threads, layout, indirect};
#else
        const auto key = text_key();
#endif
        const auto found = pipelines.find(key);
        if (found != pipelines.end()) return found->second;
        @autoreleasepool {
        NSError* error = nil;
        NSString* function_name = [NSString stringWithUTF8String:name];
        id<MTLFunction> function;
        if (type != uint32_t(-1)) {
            MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
            [constants setConstantValue:&type type:MTLDataTypeUInt atIndex:0];
            if (layout != uint32_t(-1)) [constants setConstantValue:&layout type:MTLDataTypeUInt atIndex:1];
            function = [library newFunctionWithName:function_name constantValues:constants error:&error];
        } else function = [library newFunctionWithName:function_name];
        if (!function) {
            const auto operation = std::string("loading owned graph function ") + name;
            throw metal_error(operation.c_str(), error);
        }
        id<MTLComputePipelineState> result;
        if (indirect) {
            MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function; descriptor.supportIndirectCommandBuffers = YES;
            result = [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
        } else result = [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) throw metal_error("building owned graph pipeline", error);
        require(!indirect || result.supportIndirectCommandBuffers, "pipeline refused indirect compute support");
        require(result.threadExecutionWidth == 32 && result.maxTotalThreadsPerThreadgroup >= minimum_threads,
            "unsupported Metal graph execution geometry");
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        require(minimum_threads != qsa_channel_threads || result.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
            "chronological QSA threadgroup memory exceeds device capability");
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        require(std::strncmp(name, "qsa_sparse_attention", 20) != 0 || result.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
            "sparse QSA threadgroup memory exceeds device capability");
#endif
#endif
#if defined(TOKSPUR_PRIVATE_PIPELINE_LOOKUP) && TOKSPUR_PRIVATE_PIPELINE_LOOKUP
        pipelines.emplace(PipelineKey{std::string(name), type, minimum_threads, layout, indirect}, result);
#else
        pipelines.emplace(key, result);
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (!indirect) trace_pipeline(text_key(), name, type, minimum_threads, function);
#endif
        return result;
        }
    }
    void select(const char* name, uint32_t type = uint32_t(-1), std::array<uint32_t, 4> shape = {}, uint32_t minimum_threads = 128, uint32_t layout = uint32_t(-1)) {
        TOKSPUR_CARRIER_PHASE(encoding);
        @autoreleasepool {
        active();
        if (recording) {
            require(!pending_capture_dispatch && draft.count < draft.capacity, "captured command capacity or incomplete dispatch");
            const auto state = pipeline(name, type, minimum_threads, true, layout);
            pending_capture_dispatch = true;
            const auto c = captured_command();
            [c reset]; [c setBarrier]; [c setComputePipelineState:state];
            [c setKernelBuffer:error_flag offset:0 atIndex:7];
            return;
        }
        if (profiling_active()) require(dispatches < profile_capacity, "GPU profiling dispatch capacity exceeded");
        if (!encoder) {
            require(profile_mode == ProfileMode::encoder_boundary, "missing serial graph encoder");
            MTLComputePassDescriptor* descriptor = [MTLComputePassDescriptor computePassDescriptor];
            descriptor.dispatchType = MTLDispatchTypeSerial;
            auto attachment = descriptor.sampleBufferAttachments[0];
            attachment.sampleBuffer = counters.at(dispatches / counter_chunk_dispatches);
            attachment.startOfEncoderSampleIndex = (dispatches % counter_chunk_dispatches) * 2;
            attachment.endOfEncoderSampleIndex = (dispatches % counter_chunk_dispatches) * 2 + 1;
            encoder = [command computeCommandEncoderWithDescriptor:descriptor];
            require(encoder != nil, "cannot create counter-sampled compute pass");
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            ++task4_encoders;
#endif
        }
        selected = {name, type, shape};
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        selected.carrier_layer = carrier_layer;
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        selected.task4_tag = task4_tag;
#endif
        [encoder setComputePipelineState:pipeline(name, type, minimum_threads, compatible_pipelines, layout)];
        [encoder setBuffer:error_flag offset:0 atIndex:7];
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (concurrent) { span_pending.clear(); pending_flag_reader = reads_error_flag(name); }
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (trace) {
            require(!trace_pending, "traced dispatch left incomplete");
            std::string key = std::string(name) + ":" + std::to_string(type);
            if (layout != uint32_t(-1)) key += ":layout=" + std::to_string(layout);
            if (minimum_threads != 128) key += ":" + std::to_string(minimum_threads);
            trace_pending = true; trace_bindings.clear(); trace_constants.clear();
            trace_dispatch = {{"event", Json("dispatch")}, {"pipeline", Json(key)}};
            trace_bindings.push_back(Json(Json::Object{{"index", Json::integer(7)}, {"buffer", Json("error_flag")}}));
        }
#endif
        }
    }
    template <class T> void parameter(const T& value, NSUInteger index) {
        TOKSPUR_CARRIER_PHASE(encoding);
#if defined(TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC) && defined(TOKSPUR_PRIVATE_GROUPED_MM)
        if constexpr (std::is_same_v<T, GroupedMmArgs>) selected.carrier_selected = value.selected;
#endif
        if (recording) {
            require(index < capture_bindings && sizeof(value) <= capture_constant_stride, "captured constant binding limit");
            const auto offset = (uint64_t(draft.count) * capture_bindings + index) * capture_constant_stride;
            std::memcpy(static_cast<std::byte*>(draft.parameters.contents) + offset, &value, sizeof(value));
            [captured_command() setKernelBuffer:draft.parameters offset:offset atIndex:index];
        } else [encoder setBytes:&value length:sizeof(value) atIndex:index];
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (trace && !recording) {
            require(trace_pending, "traced constant outside a dispatch");
            trace_constants.push_back(Json(Json::Object{{"index", Json::integer(index)}, {"bytes", Json::integer(sizeof(value))},
                {"sha256", Json(trace_digest(&value, sizeof(value)))}}));
        }
#endif
    }
    void before_dispatch() {
        if (recording) { static_cast<void>(captured_command()); return; }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        if (carrier_diagnostic.active() && carrier_diagnostic.records_dispatches()) {
            require(carrier_records.size() < CarrierDiagnosticWindow::maximum_dispatches, "carrier dispatch record budget exhausted");
            carrier_records.push_back(selected);
        }
#endif
        if (!profiling_active()) return;
        require(dispatches < profile_capacity && marks.size() == dispatches, "GPU profiler dispatch bookkeeping failed");
        marks.push_back(selected); // Reserved before any command batch.
        if (profile_mode == ProfileMode::dispatch_boundary) [encoder sampleCountersInBuffer:counters.at(dispatches / counter_chunk_dispatches) atSampleIndex:(dispatches % counter_chunk_dispatches) * 2 withBarrier:YES];
    }
    void after_dispatch() {
        if (recording) { ++draft.count; pending_capture_dispatch = false; ++dispatches; return; }
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (!concurrent) {
#endif
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        ++concurrent_barrier_count;
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        trace_record({{"event", Json("barrier")}, {"scope", Json("buffers")}});
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        }
#endif
        if (profile_mode == ProfileMode::dispatch_boundary) [encoder sampleCountersInBuffer:counters.at(dispatches / counter_chunk_dispatches) atSampleIndex:(dispatches % counter_chunk_dispatches) * 2 + 1 withBarrier:YES];
        else if (profile_mode == ProfileMode::encoder_boundary) {
            [encoder endEncoding]; encoder = nil;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
            if (concurrent) reset_spans();
#endif
        }
        ++dispatches;
    }
    void dispatch(bool aligned, MTLSize grid, MTLSize group) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        selected.task4_grid = {grid.width, grid.height, grid.depth};
        selected.task4_group = {group.width, group.height, group.depth};
        selected.task4_aligned = aligned;
#endif
        before_dispatch();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (concurrent && !recording) concurrent_fence();
#endif
        if (recording) {
            if (aligned) [captured_command() concurrentDispatchThreadgroups:grid threadsPerThreadgroup:group];
            else [captured_command() concurrentDispatchThreads:grid threadsPerThreadgroup:group];
        } else if (aligned) [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        else [encoder dispatchThreads:grid threadsPerThreadgroup:group];
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (trace && !recording) {
            require(trace_pending, "traced dispatch without a pipeline");
            const auto size = [](MTLSize s) { return Json(Json::Array{Json::integer(s.width), Json::integer(s.height), Json::integer(s.depth)}); };
            auto record = std::move(trace_dispatch);
            record.emplace_back("aligned", Json(aligned)); record.emplace_back("grid", size(grid)); record.emplace_back("group", size(group));
            record.emplace_back("bindings", Json(std::move(trace_bindings))); record.emplace_back("constants", Json(std::move(trace_constants)));
            trace_pending = false; trace_bindings = {}; trace_constants = {}; trace_dispatch = {};
            trace_record(std::move(record));
        }
#endif
        after_dispatch();
    }
    void dispatch_indirect(GpuView indirect_buffer, uint64_t offset, MTLSize group, bool aligned = true) {
        (void)aligned;
        require(!recording && !compatible_pipelines, "indirect dispatch requires direct batches");
        const auto& item = allocation(indirect_buffer, false);
        require(offset + 12 <= indirect_buffer.bytes, "indirect dispatch offset exceeds buffer bounds");
        require((indirect_buffer.offset + offset) % 4 == 0, "indirect buffer offset must be four-byte aligned");
        before_dispatch();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (concurrent && !recording) {
            span_pending.push_back({{indirect_buffer.slot, indirect_buffer.offset + offset,
                                     indirect_buffer.offset + offset + 12}, false});
            concurrent_fence();
        }
#endif
        [encoder dispatchThreadgroupsWithIndirectBuffer:item.buffer
                                  indirectBufferOffset:indirect_buffer.offset + offset
                                 threadsPerThreadgroup:group];
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        if (trace && !recording) {
            require(trace_pending, "traced indirect dispatch without a pipeline");
            const auto size = [](MTLSize s) { return Json(Json::Array{Json::integer(s.width), Json::integer(s.height), Json::integer(s.depth)}); };
            auto record = std::move(trace_dispatch);
            record.emplace_back("aligned", Json(true)); record.emplace_back("indirect", Json(true)); record.emplace_back("indirect_offset", Json::integer(indirect_buffer.offset + offset)); record.emplace_back("group", size(group));
            record.emplace_back("bindings", Json(std::move(trace_bindings))); record.emplace_back("constants", Json(std::move(trace_constants)));
            trace_pending = false; trace_bindings = {}; trace_constants = {}; trace_dispatch = {};
            trace_record(std::move(record));
        }
#endif
        after_dispatch();
    }
    void groups(MTLSize grid) { dispatch(true, grid, MTLSizeMake(128, 1, 1)); }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    void carrier_route(GpuView ids, GpuView weights, uint32_t rows, uint32_t selected_count, uint32_t experts) {
        if (!carrier_diagnostic.active() || !carrier_capture || carrier_layer == UINT32_MAX) return;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_priming_scope || dflash2_scope) return;
#endif
        require(profile_mode == ProfileMode::disabled && !recording && encoder && carrier_routes.size() < 512 && rows && rows <= 8 && selected_count && selected_count <= 64,
            "route capture requires a bounded direct correctness batch");
        require(carrier_row_offset <= 8 && rows <= 8 - carrier_row_offset, "diagnostic router rows exceed the fixed target width");
        const uint32_t bytes = rows * selected_count * 4;
        require(carrier_route_scratch.bytes && carrier_route_bytes + 2 * bytes <= carrier_route_scratch.bytes && ids.bytes >= bytes && weights.bytes >= bytes,
            "diagnostic route capture byte budget exhausted");
        auto target = carrier_route_scratch;
        target.offset += carrier_route_bytes; target.bytes = 2 * bytes;
        // Snapshot in the same compute encoder instead of ending and recreating
        // it at every router. Copy represented IDs and weights without FP work.
        select("graph_capture_routes", uint32_t(-1), {rows, selected_count, experts, 0});
        bind(ids, 0); bind(weights, 1); bind(target, 2, true);
        parameter(rows * selected_count, 3);
        dispatch(false, MTLSizeMake(rows * selected_count, 1, 1), MTLSizeMake(128, 1, 1));
        carrier_routes.push_back({carrier_layer, rows, selected_count, experts, carrier_route_bytes, carrier_row_offset});
        carrier_route_bytes += 2 * bytes;
        carrier_row_offset += rows;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    void exact_hc(GpuView matrix, GpuView input, GpuView output, size_t bucket) {
        require(!recording && !compatible_pipelines && fixed_route_profile_allowed(),
            "private exact-HC requires direct unprofiled execution");
        require(hc_exact != HcExactMode::off && (bucket == 0 || hc_exact == HcExactMode::combined),
            "private exact-HC pipeline was not prepared");
        require(hc_exact_pending[bucket] < UINT64_MAX - hc_exact_completed[bucket], "exact-HC route counter overflow");
        select(hc_exact_kernel(bucket), uint32_t(-1), {10240, 320, 1, 16}, 32);
        bind(matrix, 0); bind(input, 1); bind(output, 2, true);
        dispatch(true, MTLSizeMake(320, 1, 1), MTLSizeMake(32, 1, 1));
        ++hc_exact_pending[bucket]; // Draft only, never a completed route.
    }
#endif
    void threads(uint32_t count) { dispatch(false, MTLSizeMake(count, 1, 1), MTLSizeMake(128, 1, 1)); }
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    void exact_wide_q8(GpuView matrix, GpuView input, GpuView output, size_t shape, bool many1) {
        require(wide_q8_exact && !recording && !compatible_pipelines && profile_mode == ProfileMode::disabled,
            "private wide-Q8 requires prepared direct unprofiled execution");
        const size_t bucket = shape * 2 + size_t(many1);
        require(shape < 2 && wide_q8_exact_pending[bucket] < UINT64_MAX - wide_q8_exact_completed[bucket], "wide-Q8 route counter overflow");
        const uint32_t columns = shape ? 6144 : 2560, rows = shape ? 2560 : 10240;
        select(wide_q8_exact_kernel(shape), uint32_t(-1), {columns, rows, 4, 16});
        bind(matrix, 0); bind(input, 1); bind(output, 2, true);
        groups(MTLSizeMake(rows / 4, 1, 1));
        ++wide_q8_exact_pending[bucket];
    }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    void exact_wide_q8_many(GpuView matrix, GpuView input, GpuView output, size_t shape, uint32_t width) {
        require(wide_q8_many && !recording && !compatible_pipelines && profile_mode == ProfileMode::disabled,
            "private wide-Q8-many requires prepared direct unprofiled execution");
        require(shape < 2 && (width == 2 || width == 4 || width == 8), "invalid private wide-Q8-many geometry");
        const size_t bucket = shape * 3 + (width == 2 ? 0 : width == 4 ? 1 : 2);
        require(wide_q8_many_pending[bucket] < UINT64_MAX - wide_q8_many_completed[bucket], "wide-Q8-many route counter overflow");
        const uint32_t columns = shape ? 6144 : 2560, rows = shape ? 2560 : 10240;
        select(wide_q8_many_kernel(bucket), uint32_t(-1), {columns, rows, width, 4});
        bind(matrix, 0); bind(input, 1); bind(output, 2, true);
        groups(MTLSizeMake(rows / 4, 1, 1));
        ++wide_q8_many_pending[bucket]; // Not visible until successful GPU completion.
    }
#endif
    bool planar_union_profile_active() const {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        return planar_union_profile_enabled && verify_nax && verify_nax_prepared && verify_nax_scope;
#else
        return false;
#endif
    }
    void prepare_planar_union_profile() {
        if (!planar_union_profile_enabled || planar_union_profile_capacity) return;
        idle();
        uint32_t count = 0;
        for (const auto& [key, shape] : planar_matrices) {
            static_cast<void>(key);
            if (planar_v3::is_coloc_gate_kind(static_cast<planar_v3::PlanarKind>(shape.kind))) ++count;
        }
        require(count && count <= planar_union_profile_limit, "planar union diagnostic gate inventory exceeds bounded capacity");
        planar_union_profile_records.reserve(count);
        planar_union_profile_output.resize(4096ull + 1024ull * count);
        planar_union_profile_bank = allocate(checked_mul(count, planar_union_profile_stride), "private planar union diagnostic bank", false, nullptr);
        planar_union_profile_capacity = count;
    }
    void require_planar_union_profile_scope() const {
        require(planar_union_profile_active() && planar_union_profile_capacity && !recording && !compatible_pipelines &&
            !ledger.live_captures && (profile_mode == ProfileMode::dispatch_boundary || profile_mode == ProfileMode::encoder_boundary),
            "planar union diagnostic requires prepared NAX verification and direct dispatch profiling");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        require(!concurrent, "planar union diagnostic requires serial dispatch");
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        require(!operator_sink && !operator_cutting, "planar union diagnostic refuses per-operator submission");
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        require(!task4_active, "planar union diagnostic refuses an overlapping Task4 profile");
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        require(!mtp_profile.active, "planar union diagnostic refuses an overlapping MTP profile");
#endif
    }
    void dump_planar_union_profile() {
        if (!planar_union_profile_enabled || planar_union_profile_records.empty()) return;
        require(!planar_union_profile_binding && profile.mode == profile_mode &&
            (profile.mode == ProfileMode::dispatch_boundary || profile.mode == ProfileMode::encoder_boundary) && profile.timings_valid &&
            profile.kernels.size() == dispatches && !capture_executes && planar_union_profile_batch != UINT64_MAX,
            "planar union diagnostic refuses an incomplete or uncalibrated batch");
        const auto* bank = static_cast<const std::byte*>(allocation(planar_union_profile_bank).buffer.contents);
        std::array<uint64_t, 8> histogram{};
        std::array<double, 8> gu_cost_proxy{}, down_cost_proxy{};
        double total_union_ms = 0, total_gu_ms = 0, total_down_ms = 0;
        const auto timing = [&](uint64_t at, const char* name, uint32_t encoding, std::array<uint32_t, 4> shape) {
            require(at < profile.kernels.size(), "planar union diagnostic dispatch index is out of bounds");
            const auto& k = profile.kernels[at];
            require(k.kernel == name && k.encoding == encoding && k.shape == shape && k.end > k.start &&
                std::isfinite(k.milliseconds) && k.milliseconds > 0, "planar union diagnostic dispatch attribution mismatch");
            return k.milliseconds;
        };
        for (size_t i = 0; i < planar_union_profile_records.size(); ++i) {
            auto& r = planar_union_profile_records[i];
            require(r.complete && r.tokens >= 2 && r.tokens <= 8 && r.selected >= 1 && r.selected <= 16 &&
                r.experts >= r.selected && r.experts <= 512 && r.dispatch <= UINT64_MAX - 2,
                "planar union diagnostic record is incomplete");
            const auto* meta = reinterpret_cast<const uint32_t*>(bank + i * planar_union_profile_stride);
            const uint32_t total = r.tokens * r.selected, count = meta[6], m_cap = r.tokens <= 4 ? 4 : 8;
#if defined(TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS) && TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS
            constexpr uint32_t gate_tile = 4, down_tile = 8;
#else
            constexpr uint32_t gate_tile = 8, down_tile = 16;
#endif
            require(count >= 1 && count <= total && count <= 128 && meta[0] == (r.hidden + gate_tile - 1) / gate_tile &&
                meta[1] == count && meta[2] == 1 && meta[3] == (r.channels + down_tile - 1) / down_tile && meta[4] == count && meta[5] == 1 && meta[7] == 0,
                "planar union diagnostic indirect metadata mismatch");
            std::array<bool, 128> seen{};
            uint32_t routes = 0;
            r.histogram = {};
            for (uint32_t u = 0; u < count; ++u) {
                const uint32_t n = meta[136u + u];
                require(meta[8u + u] < r.experts && n >= 1 && n <= m_cap, "planar union diagnostic occupancy is invalid");
                ++r.histogram[n - 1];
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t token = meta[264u + 8u * u + k], flat = meta[1288u + 8u * u + k];
                    require(token < r.tokens && flat < total && token == flat / r.selected && !seen[flat],
                        "planar union diagnostic route coverage is invalid");
                    seen[flat] = true;
                    ++routes;
                }
            }
            require(routes == total, "planar union diagnostic omitted routed rows");
            r.union_ms = timing(r.dispatch, "planar_moe_build_union_m8", uint32_t(-1), {r.experts, r.selected, r.hidden, r.channels});
            r.gu_ms = timing(r.dispatch + 1, r.tokens <= 4 ? "planar_moe_gate_up_swiglu_m4" : "planar_moe_gate_up_swiglu_m8", r.gu_encoding, {r.channels, r.hidden, r.experts, r.tokens});
            r.down_ms = timing(r.dispatch + 2, r.tokens <= 4 ? "planar_moe_down_m4" : "planar_moe_down_m8", r.down_encoding, {r.hidden, r.channels, r.experts, r.tokens});
            total_union_ms += r.union_ms;
            total_gu_ms += r.gu_ms;
            total_down_ms += r.down_ms;
            for (size_t n = 0; n < 8; ++n) {
                histogram[n] += r.histogram[n];
                gu_cost_proxy[n] += r.gu_ms * double(r.histogram[n]) / count;
                down_cost_proxy[n] += r.down_ms * double(r.histogram[n]) / count;
            }
        }
        // Each proxy allocates a call's measured time uniformly across its union records. It is not a measured per-occupancy kernel time.
        size_t used = 0;
        const auto append = [&](const char* format, auto... values) {
            const auto remaining = planar_union_profile_output.size() - used;
            if constexpr (sizeof...(values) == 0) {
                const size_t n = std::strlen(format);
                require(n < remaining, "planar union diagnostic output capacity exceeded");
                std::memcpy(planar_union_profile_output.data() + used, format, n);
                used += n;
            } else {
                const int n = std::snprintf(planar_union_profile_output.data() + used, remaining, format, values...);
                require(n >= 0 && static_cast<size_t>(n) < remaining, "planar union diagnostic output capacity exceeded");
                used += static_cast<size_t>(n);
            }
        };
        const auto host_bytes = planar_union_profile_records.capacity() * sizeof(PlanarUnionProfileRecord) + planar_union_profile_output.capacity();
        append("PLANAR_UNION_PROFILE {\"schema\":1,\"diagnostic_only\":1,\"counter_boundary\":\"%s\",\"batch\":%llu,\"registered_gate_matrices\":%u,\"observed_calls\":%zu,\"bank_bytes\":%llu,\"bank_charged_bytes\":%llu,\"host_payload_bytes\":%zu,\"dispatches\":%llu,\"calls\":[",
            profile.mode == ProfileMode::dispatch_boundary ? "dispatch" : "encoder", static_cast<unsigned long long>(planar_union_profile_batch), planar_union_profile_capacity, planar_union_profile_records.size(),
            static_cast<unsigned long long>(planar_union_profile_bank.bytes), static_cast<unsigned long long>(allocation(planar_union_profile_bank).charged), host_bytes, static_cast<unsigned long long>(dispatches));
        for (size_t i = 0; i < planar_union_profile_records.size(); ++i) {
            const auto& r = planar_union_profile_records[i];
            append("%s{\"ordinal\":%zu,\"gate_matrix_ordinal\":%u,\"tokens\":%u,\"experts\":%u,\"selected\":%u,\"gu_encoding\":%u,\"down_encoding\":%u,\"channels\":%u,\"hidden\":%u,\"dispatch_indices\":[%llu,%llu,%llu],\"histogram_n1_to_n8\":[",
                i ? "," : "", i, r.matrix, r.tokens, r.experts, r.selected, r.gu_encoding, r.down_encoding, r.channels, r.hidden,
                static_cast<unsigned long long>(r.dispatch), static_cast<unsigned long long>(r.dispatch + 1), static_cast<unsigned long long>(r.dispatch + 2));
            for (size_t n = 0; n < 8; ++n) append("%s%u", n ? "," : "", r.histogram[n]);
            append("],\"union_ms\":%.9g,\"gu_ms\":%.9g,\"down_ms\":%.9g}", r.union_ms, r.gu_ms, r.down_ms);
        }
        append("],\"histogram_n1_to_n8\":[");
        for (size_t n = 0; n < 8; ++n) append("%s%llu", n ? "," : "", static_cast<unsigned long long>(histogram[n]));
        append("],\"uniform_record_cost_proxy_gu_ms_n1_to_n8\":[");
        for (size_t n = 0; n < 8; ++n) append("%s%.9g", n ? "," : "", gu_cost_proxy[n]);
        append("],\"uniform_record_cost_proxy_down_ms_n1_to_n8\":[");
        for (size_t n = 0; n < 8; ++n) append("%s%.9g", n ? "," : "", down_cost_proxy[n]);
        append("],\"union_ms\":%.9g,\"gu_ms\":%.9g,\"down_ms\":%.9g,\"complete\":1}\n", total_union_ms, total_gu_ms, total_down_ms);
        require(std::fwrite(planar_union_profile_output.data(), 1, used, stderr) == used && std::fflush(stderr) == 0,
            "planar union diagnostic output failed");
        ++planar_union_profile_batch;
        planar_union_profile_records.clear();
    }
    void collect_profile() {
        if (!profiling_active()) return;
        MTLTimestamp cpu_end = 0, gpu_end = 0;
        const auto before = Clock::now(); [device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end]; const auto after = Clock::now();
        const auto finish = before + (after - before) / 2;
        require(marks.size() == dispatches, "GPU counter mark/dispatch bookkeeping differs");
        const bool retain_invalid =
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            task4_sink && task4_active;
#else
            false;
#endif
        const bool positive_clocks = cpu_end > cpu_start && gpu_end > gpu_start;
        require(positive_clocks || retain_invalid, "invalid GPU counter clock span");
        const double elapsed_ns = std::chrono::duration<double, std::nano>(finish - calibration_start).count();
        mach_timebase_info_data_t base{};
        const bool units_available = mach_timebase_info(&base) == KERN_SUCCESS && base.denom;
        require(units_available || retain_invalid, "cannot inspect Mach timestamp units");
        require(elapsed_ns > 0 || retain_invalid, "GPU counter CPU-clock span is not positive");
        const double unavailable = std::numeric_limits<double>::quiet_NaN();
        const bool safe_span =
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            task4_calibration_span_valid(cpu_start, cpu_end, gpu_start, gpu_end, elapsed_ns, units_available && base.numer);
#else
            positive_clocks && units_available && elapsed_ns > 0;
#endif
        const double cpu_span = safe_span ? double(cpu_end - cpu_start) : unavailable;
        const double mach_factor = safe_span ? double(base.numer) / base.denom : unavailable;
        const bool is_mach = safe_span && std::abs(cpu_span * mach_factor - elapsed_ns) < std::abs(cpu_span - elapsed_ns);
        const double calibrated_cpu_ns = safe_span ? cpu_span * (is_mach ? mach_factor : 1.0) : unavailable;
        // Two sampleTimestamps: calls bracket the span, so a short batch can exceed a
        // relative tolerance through call latency alone. A completed batch is never
        // failed on this calibration. Reject its calibrated timings instead.
        const double mismatch = safe_span ? std::abs(calibrated_cpu_ns - elapsed_ns) / elapsed_ns : unavailable;
        const double tick_ns = safe_span ? calibrated_cpu_ns / double(gpu_end - gpu_start) : unavailable;
        GraphProfile result; result.mode = profile_mode; result.cpu_clock_is_mach = is_mach;
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        result.task4_calibration_gpu_start = gpu_start; result.task4_calibration_gpu_end = gpu_end;
        result.task4_calibration_cpu_start = cpu_start; result.task4_calibration_cpu_end = cpu_end;
        result.task4_calibration_steady_span_ns = elapsed_ns;
#endif
        result.timings_valid = profile_calibration_valid(tick_ns, mismatch);
        result.gpu_tick_nanoseconds = result.timings_valid ? tick_ns : unavailable;
        result.sampled_milliseconds = result.timings_valid ? 0 : unavailable;
        result.cpu_clock_mismatch_fraction = mismatch; result.kernels.reserve(dispatches);
        for (size_t first = 0; first < marks.size(); first += counter_chunk_dispatches) {
            const auto count = std::min(size_t(counter_chunk_dispatches), marks.size() - first);
            NSData* resolved = [counters.at(first / counter_chunk_dispatches) resolveCounterRange:NSMakeRange(0, count * 2)];
            require(resolved && resolved.length == count * 2 * sizeof(MTLCounterResultTimestamp), "GPU counter resolution failed");
            const auto* values = static_cast<const MTLCounterResultTimestamp*>(resolved.bytes);
            for (size_t i = 0; i < count; ++i) {
                const auto start = values[2 * i].timestamp, end = values[2 * i + 1].timestamp;
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                const bool sample_valid = task4_counter_ticks_valid(start, end, gpu_start, gpu_end);
                require(sample_valid || task4_sink, "invalid/unbounded GPU timestamp sample");
                result.task4_counter_ticks_valid = result.task4_counter_ticks_valid && sample_valid;
                const double ms = result.timings_valid && sample_valid ? double(end - start) * tick_ns / 1e6 : unavailable;
#else
                require(start && end && start != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= start && start >= gpu_start && end <= gpu_end,
                        "invalid/unbounded GPU timestamp sample");
                const double ms = result.timings_valid ? double(end - start) * tick_ns / 1e6 : unavailable;
#endif
                const auto& mark = marks[first + i];
                result.kernels.push_back({mark.kernel, mark.encoding, mark.shape, start, end, ms}); result.sampled_milliseconds += ms;
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                auto& timing = result.kernels.back();
                timing.task4_tag = mark.task4_tag;
                timing.task4_dispatch_index = uint32_t(first + i);
                timing.task4_grid = mark.task4_grid;
                timing.task4_threads_per_threadgroup = mark.task4_group;
                timing.task4_grid_is_threadgroups = mark.task4_aligned;
#endif
            }
        }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        if (!result.task4_counter_ticks_valid) {
            result.timings_valid = false; result.gpu_tick_nanoseconds = unavailable; result.sampled_milliseconds = unavailable;
        }
#endif
        profile = std::move(result);
    }
    GpuView allocate(uint64_t bytes, std::string label, bool immutable, const void* data) {
        @autoreleasepool {
        idle();
        require(bytes && bytes <= ledger.max_buffer_bytes && allocations.size() < 65536, "GPU allocation size/count limit");
        const auto page = static_cast<uint64_t>(getpagesize());
        const auto padded = checked_add(bytes, page - 1) / page * page;
        require(padded <= ledger.budget - ledger.live_bytes, "GPU memory budget exceeded");
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) throw FormatError("Metal graph allocation refused");
        const auto charged = std::max(bytes, static_cast<uint64_t>(buffer.allocatedSize));
        require(charged <= ledger.budget - ledger.live_bytes, "actual GPU allocation exceeds budget");
        buffer.label = [NSString stringWithUTF8String:label.c_str()];
        if (data) std::memcpy(buffer.contents, data, static_cast<size_t>(bytes));
        else std::memset(buffer.contents, 0, static_cast<size_t>(bytes));
        const auto slot = allocations.size();
        allocations.push_back({{}, buffer, bytes, charged, immutable, {}});
        ledger.live_bytes += charged;
        ledger.peak_bytes = std::max(ledger.peak_bytes, ledger.live_bytes);
        ++ledger.allocations;
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        trace_allocation(slot, label, data ? "upload" : "zeros", true);
#endif
        return {owner, slot, 0, bytes};
        }
    }
};

MetalGraph::MetalGraph(const std::filesystem::path& owned_library, uint64_t byte_budget) : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        auto& p = *impl_;
        if (const char* value = std::getenv("TOKSPUR_DUMP_PLANAR_UNION_PROFILE")) {
            require(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0, "planar union diagnostic selector must be 0 or 1");
            p.planar_union_profile_enabled = std::strcmp(value, "1") == 0;
#ifndef TOKSPUR_PRIVATE_GROUPED_MM
            require(!p.planar_union_profile_enabled, "planar union diagnostic requires the NAX carrier build");
#endif
        }
        p.device = MTLCreateSystemDefaultDevice();
        if (!p.device) throw FormatError("no Metal device for owned graph");
        p.ledger.budget = byte_budget;
        p.ledger.max_buffer_bytes = p.device.maxBufferLength;
        p.ledger.recommended_working_set_bytes = p.device.recommendedMaxWorkingSetSize;
        require(byte_budget >= 16384 && byte_budget <= p.ledger.recommended_working_set_bytes, "unsafe Metal graph memory budget");
        p.queue = [p.device newCommandQueue];
        if (!p.queue) throw FormatError("cannot create owned graph queue");
        NSError* error = nil;
        p.library = [p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:owned_library.c_str()]] error:&error];
        require(!p.library || [p.library newFunctionWithName:@"probe_unchecked"] == nil, "probe-only metallib is refused by serving graphs");
#ifndef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        require(!p.library || [p.library newFunctionWithName:@"graph_capture_routes"] == nil,
            "diagnostic metallib is refused by serving graphs");
#endif
        if (!p.library) throw metal_error("loading owned graph library", error);
        const uint64_t flag_bytes = std::min(uint64_t(65536), byte_budget);
        p.error_flag = [p.device newBufferWithLength:flag_bytes options:MTLResourceStorageModeShared];
        if (!p.error_flag) throw FormatError("cannot allocate graph error flag");
        p.ledger.live_bytes = std::max(uint64_t(4), static_cast<uint64_t>(p.error_flag.allocatedSize));
        require(p.ledger.live_bytes <= byte_budget, "graph control allocation exceeds budget");
        p.ledger.peak_bytes = p.ledger.live_bytes;
        p.ledger.allocations = 1;
    }
}

MetalGraph::~MetalGraph() {
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    if (impl_->trace) { std::fclose(impl_->trace); impl_->trace = nullptr; }
#endif
    if (impl_->encoder) [impl_->encoder endEncoding];
    // submit is synchronous; an outstanding command here is uncommitted.
    impl_->encoder = nil;
    impl_->command = nil;
    // Imported buffers have nil mmap deallocators: drop every indirect command
    // and binding reference before allocation/source owners can be destroyed.
    impl_->draft.commands = nil; impl_->draft.parameters = nil;
    impl_->captures.clear();
    impl_->draft.aliases.clear();
    for (auto& allocation : impl_->allocations) allocation.capture_aliases.clear();
}

GpuView MetalGraph::zeros(uint64_t bytes, std::string label) { return impl_->allocate(bytes, std::move(label), false, nullptr); }
GpuView MetalGraph::upload(std::span<const std::byte> data, std::string label, bool immutable) {
    return impl_->allocate(data.size(), std::move(label), immutable, data.data());
}
ImportedWeights MetalGraph::import_weights(std::shared_ptr<const ModelFiles> source, std::span<const TensorSlice> slices, uint64_t maximum_region_bytes
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                                          , bool additional_source
#endif
                                          ) {
    auto& p = *impl_; p.idle();
    require(source && !slices.empty() && slices.size() <= 8192, "invalid mapped-weight import source/count");
    for (const auto& item : p.allocations) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (additional_source) {
            require(!p.ledger.live_captures, "auxiliary weight import refuses live captures");
            require(!item.buffer || !item.source || item.source != source,
                "batch all imports from the same auxiliary source together");
        } else
#endif
        require(!item.buffer || !item.source, "batch all mapped-weight imports together; existing import remains live");
    }
    source->assert_unchanged();
    const uint64_t page = getpagesize();
    require(!maximum_region_bytes || (maximum_region_bytes >= page && maximum_region_bytes % page == 0 &&
        maximum_region_bytes <= p.ledger.max_buffer_bytes), "invalid mapped-weight region cap");
    struct Region { size_t shard; uint64_t begin, end; GpuView view{}; bool dense_q8 = false; };
    struct Part { size_t shard; uint64_t begin, bytes; bool dense_q8 = false; };
    std::vector<Region> ranges; std::vector<Part> parts; ranges.reserve(slices.size()); parts.reserve(slices.size());
    for (const auto& slice : slices) {
        const auto location = source->tensors().find(slice.name); require(location != source->tensors().end(), "unknown imported tensor");
        const auto& tensor = source->tensor(slice.name); const auto& file = source->shards()[location->second.shard];
        require(slice.offset < tensor.bytes && slice.offset % 4 == 0, "invalid/misaligned imported tensor offset");
        const auto bytes = slice.bytes ? slice.bytes : tensor.bytes - slice.offset;
        require(bytes && bytes <= tensor.bytes - slice.offset, "imported tensor slice exceeds source");
        const auto begin = checked_add(tensor.offset, slice.offset), end = checked_add(begin, bytes);
        const auto first = begin / page * page, last = checked_add(end, page - 1) / page * page;
        require(last <= checked_add(file.file_size(), page - 1) / page * page, "page-aligned import exceeds mapped source");
        require(!maximum_region_bytes || last - first <= maximum_region_bytes,
            "complete tensor slice exceeds mapped-weight region cap");
        bool dense_q8 = false;
#if defined(TOKSPUR_PRIVATE_Q8_IMPORT_ISOLATION) && TOKSPUR_PRIVATE_Q8_IMPORT_ISOLATION
        // Keep dense Q8 preparation from making adjacent expert weights resident through a shared Metal buffer.
        dense_q8 = tensor.type == 8 && tensor.dimensions.size() == 2;
#endif
        parts.push_back({location->second.shard, begin, bytes, dense_q8});
        ranges.push_back({location->second.shard, first, last, {}, dense_q8});
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.shard == b.shard ? a.begin < b.begin : a.shard < b.shard; });
    std::vector<Region> merged; merged.reserve(ranges.size());
    for (auto r : ranges) {
        if (!merged.empty() && r.shard == merged.back().shard && r.dense_q8 == merged.back().dense_q8 && r.begin <= merged.back().end &&
            (!maximum_region_bytes || std::max(r.end, merged.back().end) - merged.back().begin <= maximum_region_bytes))
            merged.back().end = std::max(merged.back().end, r.end);
        else merged.push_back(r);
    }
    uint64_t total = 0;
    for (const auto& r : merged) { require(r.end - r.begin <= p.ledger.max_buffer_bytes, "mapped-weight region exceeds Metal buffer limit"); total = checked_add(total, r.end - r.begin); }
    require(total <= p.ledger.budget - p.ledger.live_bytes && p.allocations.size() + merged.size() <= 65536, "mapped-weight import exceeds budget/count");
    ImportedWeights result; result.tensors.reserve(parts.size()); result.regions.reserve(merged.size());
    p.allocations.reserve(p.allocations.size() + merged.size());
    try {
        for (auto& r : merged) {
            @autoreleasepool {
            const auto& file = source->shards()[r.shard]; const auto& tensor = file.tensors().front();
            // ModelFiles is file-backed; each Gguf owns one regular read-only mmap.
            // Its VM allocation includes the rounded final page. Kernels see only
            // checked tensor slices, never file-header/tail padding as matrices.
            const auto tensor_data = file.tensor_bytes(tensor);
            const auto base = reinterpret_cast<uintptr_t>(tensor_data.data()) - tensor.offset;
            require(base % page == 0, "source mapping base is not page aligned");
            const auto length = r.end - r.begin;
            id<MTLBuffer> buffer = [p.device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base + r.begin) length:length options:MTLResourceStorageModeShared deallocator:nil];
            require(buffer != nil, "Metal refused read-only mapped-weight region");
            const auto charge = std::max(length, uint64_t(buffer.allocatedSize));
            require(charge <= p.ledger.budget - p.ledger.live_bytes, "actual mapped-weight resource exceeds budget");
            buffer.label = @"owned immutable mapped weights";
            const auto slot = p.allocations.size(); p.allocations.push_back({source, buffer, length, charge, true, {}});
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
            p.trace_allocation(slot, "owned immutable mapped weights", "import", false);
#endif
            p.ledger.live_bytes += charge; p.ledger.imported_backing_bytes += length;
            p.ledger.peak_bytes = std::max(p.ledger.peak_bytes, p.ledger.live_bytes); ++p.ledger.allocations;
            r.view = {p.owner, slot, 0, length}; result.regions.push_back(r.view);
            }
        }
        for (auto part : parts) {
            const auto found = std::find_if(merged.begin(), merged.end(), [&](const auto& r) { return r.shard == part.shard && r.dense_q8 == part.dense_q8 && r.begin <= part.begin && part.begin + part.bytes <= r.end; });
            require(found != merged.end(), "internal mapped-weight plan mismatch");
            result.tensors.push_back(slice(found->view, part.begin - found->begin, part.bytes));
        }
        source->assert_unchanged();
    } catch (...) { for (auto v : result.regions) release(v); throw; }
    return result;
}
GpuView MetalGraph::slice(GpuView buffer, uint64_t offset, uint64_t bytes) const {
    static_cast<void>(impl_->allocation(buffer));
    require(offset <= buffer.bytes && bytes && bytes <= buffer.bytes - offset, "GPU slice exceeds parent");
    GpuView result{buffer.owner, buffer.slot, checked_add(buffer.offset, offset), bytes};
    static_cast<void>(impl_->allocation(result));
    return result;
}
GpuView MetalGraph::suballocate_draft_output_slice(GpuView output_weight_view, uint32_t entries, uint32_t channels) {
    const uint64_t bytes = uint64_t(entries) * (channels / 32) * 34;
    if (impl_->draft_output_slice_.bytes) {
        const auto old_key = std::tuple{impl_->draft_output_slice_.slot, impl_->draft_output_slice_.offset, impl_->draft_output_slice_.bytes};
        const auto found = impl_->q8_tile64_entries.find(old_key);
        if (found != impl_->q8_tile64_entries.end() && found->second.is_alias) impl_->q8_tile64_entries.erase(found);
    }
    impl_->draft_output_slice_ = slice(output_weight_view, 0, bytes);
    impl_->draft_output_parent_ = std::tuple{output_weight_view, entries, channels};
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (impl_->lane_slots.contains(output_weight_view.slot)) {
        impl_->draft_output_shape_ = LaneMatrixShape{8, channels, entries, 1};
    } else {
        impl_->draft_output_shape_.reset();
    }
#endif
    if (impl_->q8_tile64_ready) impl_->refresh_draft_output_tile64_alias();
    return impl_->draft_output_slice_;
}
GpuView MetalGraph::allocate_draft_embedding_slice(uint32_t entries, uint32_t channels) {
    const uint64_t bytes = uint64_t(entries) * (channels / 32) * 34;
    impl_->draft_embedding_slice_ = zeros(bytes, "MTP draft embedding slice (32K Q8_0)");
    return impl_->draft_embedding_slice_;
}
GpuView MetalGraph::draft_output_slice() const noexcept {
    return impl_->draft_output_slice_;
}
GpuView MetalGraph::draft_embedding_slice() const noexcept {
    return impl_->draft_embedding_slice_;
}
bool MetalGraph::has_draft_slices() const noexcept {
    return impl_->draft_output_slice_.bytes != 0 && impl_->draft_embedding_slice_.bytes != 0;
}
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
void MetalGraph::register_lane_matrix(GpuView matrix, LaneMatrixShape shape) {
    auto& p = *impl_; p.idle();
    const auto& storage = p.allocation(matrix);
    require(!p.lane_locked && !p.ledger.live_captures && storage.immutable && matrix.offset % 16 == 0,
        "lane matrices require immutable aligned startup registration");
    require(!p.hc_q8.splits && p.lane_matrices.size() < 2048, "lane matrix policy/count refusal");
    require((shape.encoding == 8 && shape.experts == 1 && shape.columns && shape.columns <= 65536 &&
        shape.columns % 32 == 0 && (shape.columns % 256 == 0 || shape.columns < 512) && shape.rows && shape.rows <= 1'000'000) ||
        (shape.encoding == 7 && shape.columns == 640 && shape.rows && shape.rows <= 65536 && shape.experts > 1 && shape.experts <= 4096),
        "unsupported lane matrix type or safe geometry");
    const auto row_bytes = checked_mul(shape.columns / 32, shape.encoding == 8 ? 34 : 24);
    require(matrix.bytes == checked_mul(checked_mul(row_bytes, shape.rows), shape.experts), "lane matrix byte geometry mismatch");
    for (const auto& [key, previous] : p.lane_matrices) {
        static_cast<void>(previous);
        const auto [slot, offset, bytes] = key;
        require(!overlaps(matrix, {p.owner, slot, offset, bytes}), "lane matrix registration overlaps an existing matrix");
    }
    // Resolve every consumer before publishing any layout identity. No CPU/GPU payload copy or scratch is introduced here.
    static_cast<void>(p.pipeline("dense_lanes_q8")); static_cast<void>(p.pipeline("route_lanes_q51"));
    for (const auto width : {1u, 2u, 4u, 8u}) static_cast<void>(p.pipeline(lane_many_kernel(width)));
    const bool new_slot = p.lane_slots.insert(matrix.slot).second;
    try { p.lane_matrices.emplace(std::tuple{matrix.slot, matrix.offset, matrix.bytes}, shape); }
    catch (...) { if (new_slot) p.lane_slots.erase(matrix.slot); throw; }
    if (shape.encoding == 8u && shape.experts == 1u) {
        p.q8_matrices[std::tuple{matrix.slot, matrix.offset, matrix.bytes}] = Impl::Q8MatrixMeta{shape.columns, shape.rows, 1u};
        if (p.q8_tile64_ready) p.ensure_q8_tile64_all();
    }
}
uint64_t MetalGraph::lane_matrix_count() const { impl_->thread_check(); return impl_->lane_matrices.size(); }
#endif
void MetalGraph::register_q8_matrix(GpuView matrix, uint32_t columns, uint32_t rows) {
    auto& p = *impl_; p.idle();
    const auto& storage = p.allocation(matrix);
    if (!storage.immutable || p.ledger.live_captures || matrix.offset % 4 != 0 ||
        columns < 32 || columns > 65536 || (columns % 32) != 0 || rows == 0 || rows > 1'000'000) {
        return;
    }
    const uint64_t expected_bytes = uint64_t(columns / 32u) * 34ull * rows;
    if (matrix.bytes != expected_bytes) return;
    try {
        if (p.planar_shape(matrix)) return;
    } catch (...) {
        return;
    }
    const auto key = std::tuple{matrix.slot, matrix.offset, matrix.bytes};
    if (p.q8_matrices.contains(key)) return;
    uint32_t src_layout = 0u;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    try {
        if (const auto* ls = p.lane_shape(matrix)) {
            if (ls->encoding != 8u || ls->experts != 1u || ls->columns != columns || ls->rows != rows) return;
            src_layout = 1u;
        }
    } catch (...) {
        return;
    }
#endif
    p.q8_matrices.emplace(key, Impl::Q8MatrixMeta{columns, rows, src_layout});
    if (p.q8_tile64_ready) p.ensure_q8_tile64_all();
}
void MetalGraph::write(GpuView destination, std::span<const std::byte> data) {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    impl_->nax_dense_bf16_valid = false;
    impl_->nax_gathered_broadcast_valid = false;
    impl_->nax_gathered_handoff_valid = false;
#endif
    const auto& item = impl_->allocation(destination, true);
    require(data.size() == destination.bytes, "GPU upload span mismatch");
    std::memcpy(static_cast<std::byte*>(item.buffer.contents) + destination.offset, data.data(), data.size());
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    impl_->trace_view("host_write", destination, data.data());
#endif
}
void MetalGraph::write_bytes(GpuView storage, uint64_t offset, std::span<const std::byte> data) {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    impl_->nax_dense_bf16_valid = false;
    impl_->nax_gathered_broadcast_valid = false;
    impl_->nax_gathered_handoff_valid = false;
#endif
    const auto& item = impl_->allocation(storage, true);
    require(offset <= storage.bytes && data.size() <= storage.bytes - offset, "GPU byte upload exceeds storage");
    require(item.buffer.storageMode == MTLStorageModeShared && item.buffer.contents, "GPU byte upload requires shared storage");
    if (data.empty()) return;
    std::memmove(static_cast<std::byte*>(item.buffer.contents) + storage.offset + offset, data.data(), data.size());
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    impl_->trace_bytes("host_write", storage, offset, data.size(), data.data());
#endif
}
void MetalGraph::read_bytes(GpuView storage, uint64_t offset, std::span<std::byte> destination) const {
    TOKSPUR_CARRIER_PHASE(readback);
    impl_->idle(); const auto& item = impl_->allocation(storage);
    require(offset <= storage.bytes && destination.size() <= storage.bytes - offset, "GPU byte readback exceeds storage");
    require(item.buffer.storageMode == MTLStorageModeShared && item.buffer.contents, "GPU byte readback requires shared storage");
    if (destination.empty()) return;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (impl_->carrier_gpu_readback) {
        for (uint64_t at = 0; at < destination.size();) {
            const auto bytes = std::min<uint64_t>(destination.size() - at, 16ull << 20);
            const CarrierStateRead read{storage, offset + at, bytes, 0};
            read_carrier_state_segments(std::span(&read, 1), destination.subspan(at, bytes)); at += bytes;
        }
    } else
#endif
    {
        std::memmove(destination.data(), static_cast<const std::byte*>(item.buffer.contents) + storage.offset + offset, destination.size());
    }
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    impl_->trace_bytes("host_read", storage, offset, destination.size(), destination.data());
#endif
}
std::vector<std::byte> MetalGraph::read(GpuView source) const {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (impl_->carrier_gpu_readback) {
        static_cast<void>(impl_->allocation(source));
        std::vector<std::byte> result(source.bytes);
        read(source, result); return result;
    }
#endif
    const auto& item = impl_->allocation(source);
    const auto* first = static_cast<const std::byte*>(item.buffer.contents) + source.offset;
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    impl_->trace_view("host_read", source, first);
#endif
    return {first, first + source.bytes};
}
void MetalGraph::read(GpuView source, std::span<std::byte> destination) const {
        TOKSPUR_CARRIER_PHASE(readback);
    impl_->idle();
    const auto& item = impl_->allocation(source);
    require(destination.size() == source.bytes, "GPU readback span mismatch");
    require(item.buffer.storageMode == MTLStorageModeShared && item.buffer.contents,
        "GPU readback requires CPU-accessible shared storage");
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (impl_->carrier_gpu_readback) {
        auto& p = *impl_;
        require(!p.carrier_diagnostic.active(), "diagnostic GPU readback requires idle state");
        const auto first = reinterpret_cast<uintptr_t>(item.buffer.contents) + source.offset;
        const auto output = reinterpret_cast<uintptr_t>(destination.data());
        require(source.bytes <= UINTPTR_MAX - first && destination.size() <= UINTPTR_MAX - output &&
            (output + destination.size() <= first || first + source.bytes <= output),
            "diagnostic GPU readback refuses an aliased host destination");
        const auto& staging = p.allocation(p.carrier_readback_staging, true);
        require(!overlaps(source, p.carrier_readback_staging), "diagnostic readback source aliases staging");
        for (uint64_t at = 0; at < source.bytes;) {
            const auto bytes = std::min(source.bytes - at, p.carrier_readback_staging.bytes);
            const bool byte_copy = (source.offset + at) % 4 != 0 || bytes % 4 != 0;
            require(p.carrier_readback_commands < 4096, "diagnostic GPU state-readback command bound exhausted");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
            const auto next_command_count = checked_add(p.lifetime_command_count, 1);
            const auto next_dispatch_count = byte_copy ? checked_add(p.lifetime_dispatch_count, 1) : p.lifetime_dispatch_count;
#endif
            @autoreleasepool {
                id<MTLCommandBuffer> command = [p.queue commandBuffer];
                require(command != nil, "cannot create diagnostic state-readback command");
                command.label = @"owned diagnostic state readback";
                if (!byte_copy) {
                    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
                    require(encoder != nil, "cannot encode diagnostic state readback");
                    [encoder copyFromBuffer:item.buffer sourceOffset:source.offset + at
                        toBuffer:staging.buffer destinationOffset:0 size:bytes];
                    [encoder endEncoding];
                } else {
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    require(encoder != nil, "cannot encode diagnostic byte readback");
                    const uint64_t offset = source.offset + at, source_bytes = source.offset + source.bytes;
                    const uint64_t output_offset = 0, source_begin = source.offset;
                    const uint32_t count = uint32_t(bytes);
                    [encoder setComputePipelineState:p.carrier_readback_byte_pipeline];
                    [encoder setBuffer:item.buffer offset:0 atIndex:0];
                    [encoder setBuffer:staging.buffer offset:0 atIndex:1];
                    [encoder setBytes:&offset length:sizeof(offset) atIndex:2];
                    [encoder setBytes:&count length:sizeof(count) atIndex:3];
                    [encoder setBytes:&source_bytes length:sizeof(source_bytes) atIndex:4];
                    [encoder setBytes:&output_offset length:sizeof(output_offset) atIndex:5];
                    [encoder setBytes:&source_begin length:sizeof(source_begin) atIndex:6];
                    [encoder setBuffer:p.error_flag offset:0 atIndex:7];
                    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];
                }
                [command commit];
                ++p.carrier_readback_commands;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                p.lifetime_command_count = next_command_count;
                p.lifetime_dispatch_count = next_dispatch_count;
#endif
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted, "diagnostic GPU state readback failed");
                require(*static_cast<const uint32_t*>(p.error_flag.contents) == 0, "diagnostic GPU byte readback rejected its range");
                std::memcpy(destination.data() + at, staging.buffer.contents, size_t(bytes));
            }
            at += bytes;
        }
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        impl_->trace_view("host_read", source, destination.data());
#endif
        return;
    }
#endif
    std::memmove(destination.data(), static_cast<const std::byte*>(item.buffer.contents) + source.offset, destination.size());
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    impl_->trace_view("host_read", source, destination.data());
#endif
}
const float* MetalGraph::host_floats(GpuView source) const {
    impl_->idle();
    const auto& item = impl_->allocation(source);
    require(item.buffer.storageMode == MTLStorageModeShared && item.buffer.contents,
        "GPU host_floats requires CPU-accessible shared storage");
    require(source.offset % sizeof(float) == 0, "GPU host_floats unaligned offset");
    return reinterpret_cast<const float*>(static_cast<const std::byte*>(item.buffer.contents) + source.offset);
}
void MetalGraph::copy(std::span<const GraphCopy> copies) {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    impl_->nax_dense_bf16_valid = false;
    impl_->nax_gathered_broadcast_valid = false;
    impl_->nax_gathered_handoff_valid = false;
#endif
    require(!copies.empty() && copies.size() <= 1024, "idle GPU copy count limit");
    std::array<const std::byte*, 1024> sources{};
    std::array<std::byte*, 1024> destinations{};
    std::array<size_t, 1024> lengths{};
    for (size_t i = 0; i < copies.size(); ++i) {
        const auto& pair = copies[i];
        const auto& source = impl_->allocation(pair.source);
        const auto& destination = impl_->allocation(pair.destination, true);
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        require(!impl_->lane_slots.contains(pair.source.slot) && !impl_->lane_slots.contains(pair.destination.slot),
            "GPU copy refuses packed storage whose layout identity would be lost");
#endif
        require(!impl_->planar_slots.contains(pair.source.slot) && !impl_->planar_slots.contains(pair.destination.slot),
            "GPU copy refuses planar v3 storage whose layout identity would be lost");
        require(pair.source.bytes == pair.destination.bytes, "idle GPU copy span mismatch");
        require(source.buffer.storageMode == MTLStorageModeShared && destination.buffer.storageMode == MTLStorageModeShared &&
            source.buffer.contents && destination.buffer.contents, "idle GPU copy requires CPU-accessible shared storage");
        for (size_t j = 0; j < copies.size(); ++j) {
            require(!overlaps(pair.destination, copies[j].source), "idle GPU copy source/destination overlap");
            if (j < i) require(!overlaps(pair.destination, copies[j].destination), "idle GPU copy destinations overlap");
        }
        sources[i] = static_cast<const std::byte*>(source.buffer.contents) + pair.source.offset;
        destinations[i] = static_cast<std::byte*>(destination.buffer.contents) + pair.destination.offset;
        lengths[i] = static_cast<size_t>(pair.source.bytes);
    }
    // No validation, allocation or other fallible work follows the first copy.
    for (size_t i = 0; i < copies.size(); ++i)
        std::memcpy(destinations[i], sources[i], lengths[i]);
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    if (impl_->trace) {
        Json::Array pairs;
        for (const auto& pair : copies)
            pairs.push_back(Json(Json::Object{{"source", Impl::trace_range(pair.source)}, {"destination", Impl::trace_range(pair.destination)},
                {"sha256", Json(Impl::trace_digest(destinations[&pair - copies.data()], pair.destination.bytes))}}));
        impl_->trace_record({{"event", Json("host_copy")}, {"copies", Json(std::move(pairs))}});
    }
#endif
}
void MetalGraph::release(GpuView full_buffer) {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    impl_->nax_dense_bf16_valid = false;
    impl_->nax_gathered_broadcast_valid = false;
    impl_->nax_gathered_handoff_valid = false;
#endif
    const auto& item = impl_->allocation(full_buffer);
    require(!full_buffer.offset && full_buffer.bytes == item.requested, "release requires the complete allocation");
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    impl_->trace_record({{"event", Json("release")}, {"slot", Json::integer(full_buffer.slot)}});
#endif
    for (const auto& c : impl_->captures)
        require(!c.commands || !c.leases.contains(full_buffer.slot), "allocation is leased by an immutable capture");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::erase_if(impl_->lane_matrices, [&](const auto& entry) { return std::get<0>(entry.first) == full_buffer.slot; });
    impl_->lane_slots.erase(full_buffer.slot);
#endif
    std::erase_if(impl_->planar_matrices, [&](const auto& entry) { return std::get<0>(entry.first) == full_buffer.slot; });
    impl_->planar_slots.erase(full_buffer.slot);
    if (impl_->draft_output_parent_.has_value() && std::get<0>(*impl_->draft_output_parent_).slot == full_buffer.slot) {
        impl_->draft_output_parent_.reset();
    }
    std::erase_if(impl_->q8_matrices, [&](const auto& entry) { return std::get<0>(entry.first) == full_buffer.slot; });
    std::erase_if(impl_->q8_tile64_entries, [&](const auto& entry) {
        return entry.second.is_alias && (std::get<0>(entry.first) == full_buffer.slot ||
            entry.second.source_slot == full_buffer.slot || entry.second.plane0.slot == full_buffer.slot);
    });
    std::vector<GpuView> owned_t64_to_release;
    for (auto it = impl_->q8_tile64_entries.begin(); it != impl_->q8_tile64_entries.end(); ) {
        if (!it->second.is_alias && it->second.source_slot == full_buffer.slot && it->second.alloc.slot != full_buffer.slot) {
            owned_t64_to_release.push_back(it->second.alloc);
            it = impl_->q8_tile64_entries.erase(it);
        } else if (!it->second.is_alias && it->second.alloc.slot == full_buffer.slot) {
            it = impl_->q8_tile64_entries.erase(it);
        } else {
            ++it;
        }
    }
    for (const auto& [base, alias] : item.capture_aliases) {
        static_cast<void>(base);
        impl_->ledger.live_bytes -= alias.charged;
        impl_->ledger.capture_binding_alias_resource_bytes -= alias.charged;
        impl_->ledger.capture_binding_alias_span_bytes -= alias.span;
        --impl_->ledger.capture_binding_aliases;
    }
    impl_->allocations[full_buffer.slot].capture_aliases.clear();
    impl_->ledger.live_bytes -= item.charged;
    if (item.source) impl_->ledger.imported_backing_bytes -= item.requested;
    impl_->allocations[full_buffer.slot].buffer = nil;
    impl_->allocations[full_buffer.slot].source.reset();
    for (const auto& owned_alloc : owned_t64_to_release) release(owned_alloc);
}
GraphMemory MetalGraph::memory() const { impl_->thread_check(); return impl_->ledger; }
std::string MetalGraph::device_name() const { impl_->thread_check(); return impl_->device.name.UTF8String; }
#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
void MetalGraph::configure_native_plain_loads(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.recording && !p.ledger.live_captures, "native plain loads refuse captures");
    if (enabled) for (uint32_t type : {0u, 30u}) static_cast<void>(p.pipeline("graph_native_plain_gemv", type));
    p.native_plain_loads = enabled;
}
bool MetalGraph::native_plain_loads() const { impl_->thread_check(); return impl_->native_plain_loads; }
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
void MetalGraph::configure_dispatch_trace(const std::filesystem::path& path) {
    auto& p = *impl_; p.idle();
    require(!p.trace && p.allocations.empty() && p.profile_mode == ProfileMode::disabled,
        "dispatch trace must start before the first allocation");
    p.trace = std::fopen(path.c_str(), "wx");
    require(p.trace != nullptr, "cannot create dispatch trace");
    p.trace_record({{"event", Json("trace")}, {"schema", Json::integer(1)}, {"producer", Json("tokspur-cpp-metal-graph")},
        {"device", Json(std::string(p.device.name.UTF8String))}});
}
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
void MetalGraph::Impl::concurrent_fence() {
    bool hazard = flag_read || (pending_flag_reader && flag_written);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (hazard) ++carrier_flag_barriers;
#endif
    for (const auto& [span, output] : span_pending) hazard = hazard || touches(span_writes, span) || (output && touches(span_reads, span));
    if (hazard) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        span_reads.clear(); span_writes.clear(); flag_written = flag_read = false; ++concurrent_barrier_count;
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
        trace_record({{"event", Json("barrier")}, {"scope", Json("buffers")}});
#endif
    }
    for (const auto& [span, output] : span_pending) (output ? span_writes : span_reads).push_back(span);
    flag_written = true; flag_read = flag_read || pending_flag_reader;
    span_pending.clear(); pending_flag_reader = false;
}
void MetalGraph::configure_concurrent_dispatch(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.concurrent_locked, "concurrent dispatch is fixed before the first batch");
    require(!enabled || p.profile_mode == ProfileMode::disabled, "concurrent dispatch requires unprofiled execution");
    p.concurrent = enabled;
}
bool MetalGraph::concurrent_dispatch() const { impl_->thread_check(); return impl_->concurrent; }
uint64_t MetalGraph::lifetime_dispatches() const { impl_->thread_check(); return impl_->lifetime_dispatch_count; }
uint64_t MetalGraph::lifetime_command_buffers() const { impl_->thread_check(); return impl_->lifetime_command_count; }
uint64_t MetalGraph::lifetime_barriers() const { impl_->thread_check(); return impl_->lifetime_barrier_count; }
uint64_t MetalGraph::concurrent_barriers() const { impl_->thread_check(); return impl_->completed_concurrent_barriers; }
#endif
void MetalGraph::dispatch_indirect(GpuView indirect_buffer, uint64_t offset, uint32_t group_x, uint32_t group_y, uint32_t group_z, bool aligned) {
    impl_->thread_check();
    impl_->dispatch_indirect(indirect_buffer, offset, MTLSizeMake(group_x, group_y, group_z), aligned);
}
void MetalGraph::dispatch_indirect(GpuView indirect_buffer, uint64_t offset, MTLSize group, bool aligned) {
    impl_->thread_check();
    impl_->dispatch_indirect(indirect_buffer, offset, group, aligned);
}
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
void MetalGraph::begin_carrier_gpu_readback() {
    auto& p = *impl_; p.idle();
    require(p.carrier_diagnostic.valid() && p.carrier_diagnostic.mode() != CarrierDiagnosticMode::off && !p.carrier_diagnostic.active() && !p.carrier_gpu_readback,
        "GPU state readback requires idle bounded diagnostic preparation");
    if (!p.carrier_readback_staging.bytes)
        p.carrier_readback_staging = zeros(16ull << 20, "bounded diagnostic state readback");
    p.carrier_readback_byte_pipeline = p.pipeline("carrier_readback_bytes", uint32_t(-1), 128, false, uint32_t(-1));
    require(*static_cast<const uint32_t*>(p.error_flag.contents) == 0, "GPU graph is invalid before state inspection");
    p.carrier_readback_commands = 0; p.carrier_gpu_readback = true;
}
void MetalGraph::end_carrier_gpu_readback() noexcept { impl_->carrier_gpu_readback = false; }
uint32_t MetalGraph::carrier_gpu_readback_commands() const { impl_->thread_check(); return impl_->carrier_readback_commands; }
bool MetalGraph::carrier_gpu_readback_enabled() const { impl_->thread_check(); return impl_->carrier_gpu_readback; }
uint32_t MetalGraph::prime_carrier_mapped_weights() const {
    auto& p = *impl_; p.idle();
    require(p.carrier_gpu_readback && !p.carrier_diagnostic.active(), "mapped-weight preparation requires idle diagnostic inspection");
    std::vector<CarrierStateRead> reads;
    for (size_t slot = 0; slot < p.allocations.size(); ++slot) {
        const auto& a = p.allocations[slot];
        if (!a.buffer || !a.source || !a.immutable) continue;
        require(reads.size() < 8192 && a.requested >= 4, "mapped-weight probe region bound");
        reads.push_back({{p.owner, slot, 0, a.requested}, 0, 4, uint64_t(reads.size()) * 4});
    }
    require(!reads.empty(), "mapped-weight preparation found no immutable imports");
    std::vector<std::byte> words(reads.size() * 4);
    read_carrier_state_segments(reads, words);
    return uint32_t(reads.size());
}
void MetalGraph::read_carrier_state_segments(std::span<const CarrierStateRead> reads, std::span<std::byte> destination) const {
    auto& p = *impl_; p.idle();
    require(p.carrier_gpu_readback && !p.carrier_diagnostic.active() && !reads.empty() && reads.size() <= 8192 &&
        !destination.empty() && destination.size() <= p.carrier_readback_staging.bytes, "invalid diagnostic state-readback batch");
    const auto first = reinterpret_cast<uintptr_t>(destination.data());
    require(destination.size() <= UINTPTR_MAX - first, "state-readback host span overflow");
    for (const auto& a : p.allocations) if (a.buffer && a.buffer.contents) {
        const auto start = reinterpret_cast<uintptr_t>(a.buffer.contents);
        require(a.requested <= UINTPTR_MAX - start && (first + destination.size() <= start || start + a.requested <= first),
            "state readback requires external host storage");
    }
    uint64_t cursor = 0;
    for (const auto& read : reads) {
        static_cast<void>(p.allocation(read.storage));
        require(!overlaps(read.storage, p.carrier_readback_staging) && read.bytes && read.source_offset <= read.storage.bytes &&
            read.bytes <= read.storage.bytes - read.source_offset && read.output_offset == cursor &&
            read.bytes <= destination.size() - cursor, "invalid or incomplete state-readback range");
        cursor += read.bytes;
    }
    require(cursor == destination.size() && p.carrier_readback_commands < 4096, "state-readback coverage or command bound exhausted");
    const auto& staging = p.allocation(p.carrier_readback_staging, true);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    const auto commands = checked_add(p.lifetime_command_count, 1);
    const auto dispatches = checked_add(p.lifetime_dispatch_count, reads.size());
#endif
    @autoreleasepool {
        id<MTLCommandBuffer> command = [p.queue commandBuffer]; require(command != nil, "cannot create state-readback batch");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "cannot encode state-readback batch");
        [encoder setComputePipelineState:p.carrier_readback_byte_pipeline];
        [encoder setBuffer:staging.buffer offset:0 atIndex:1]; [encoder setBuffer:p.error_flag offset:0 atIndex:7];
        for (const auto& read : reads) {
            const auto& source = p.allocation(read.storage);
            const uint64_t offset = read.storage.offset + read.source_offset, end = read.storage.offset + read.storage.bytes;
            const uint64_t begin = read.storage.offset, output = read.output_offset; const uint32_t count = uint32_t(read.bytes);
            [encoder setBuffer:source.buffer offset:0 atIndex:0];
            [encoder setBytes:&offset length:sizeof(offset) atIndex:2]; [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder setBytes:&end length:sizeof(end) atIndex:4]; [encoder setBytes:&output length:sizeof(output) atIndex:5];
            [encoder setBytes:&begin length:sizeof(begin) atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding]; [command commit]; ++p.carrier_readback_commands;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        p.lifetime_command_count = commands; p.lifetime_dispatch_count = dispatches;
#endif
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted && *static_cast<const uint32_t*>(p.error_flag.contents) == 0,
            "GPU state-readback batch failed");
        std::memcpy(destination.data(), staging.buffer.contents, destination.size());
    }
}
void MetalGraph::prepare_carrier_diagnostic(CarrierDiagnosticMode mode, uint32_t actions, bool record_dispatches) {
    auto& p = *impl_; p.idle();
    require(mode != CarrierDiagnosticMode::dispatch || !p.concurrent,
        "carrier dispatch diagnostic requires explicitly serial execution");
    if (mode == CarrierDiagnosticMode::dispatch) {
        require(!p.profile_capacity && !p.ledger.profile_counter_payload_bytes,
            "carrier diagnostic requires exclusive unallocated counters");
        require([p.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary],
            "carrier dispatch diagnostic requires encoder-boundary counters");
    }
    p.carrier_diagnostic.prepare(mode, actions, record_dispatches);
    if (record_dispatches) p.carrier_records.reserve(CarrierDiagnosticWindow::maximum_dispatches);
    configure_batch_phase_clock(true);
}
bool MetalGraph::begin_carrier_diagnostic(bool generation_v2, bool component) {
    auto& p = *impl_; p.idle();
    const bool active = p.carrier_diagnostic.begin(generation_v2, component);
    if (active && p.carrier_diagnostic.mode() == CarrierDiagnosticMode::dispatch) {
        // Stack and deferred prompt preparation keep their existing unprofiled
        // admission. Counter storage belongs to the bounded action window.
        p.carrier_diagnostic_preparing = true;
        try {
            if (!p.profile_capacity) configure_profiling(CarrierDiagnosticWindow::maximum_dispatches);
            require(p.profile_capacity == CarrierDiagnosticWindow::maximum_dispatches,
                "carrier diagnostic counter capacity changed");
            p.profile_mode = ProfileMode::encoder_boundary;
        } catch (...) {
            p.carrier_diagnostic_preparing = false; p.carrier_diagnostic.invalidate(); throw;
        }
        p.carrier_diagnostic_preparing = false;
    }
    if (active) carrier_cpu_begin();
    return active;
}
void MetalGraph::end_carrier_diagnostic() {
    auto& p = *impl_; p.idle(); p.carrier_diagnostic.end();
    const auto phases = carrier_cpu_finish();
    const auto record = Json(Json::Object{{"event", Json("carrier_diagnostic_action")}, {"action", Json::integer(p.carrier_diagnostic.actions())},
        {"component", Json(p.carrier_diagnostic.component())}, {"correctness_capture", Json(bool(p.carrier_capture))}, {"phases", phases}, {"intrusive", Json(true)}, {"complete", Json(true)}}).dump() + "\n";
    require(std::fwrite(record.data(), 1, record.size(), stderr) == record.size() && std::fflush(stderr) == 0, "carrier host ledger write failed");
    p.profile_mode = ProfileMode::disabled; p.carrier_layer = UINT32_MAX;
    if (p.carrier_diagnostic.mode() == CarrierDiagnosticMode::dispatch) configure_profiling(0);
}
uint32_t MetalGraph::carrier_diagnostic_action() const { impl_->thread_check(); return impl_->carrier_diagnostic.actions(); }
void MetalGraph::finish_carrier_diagnostic() {
    auto& p = *impl_; p.idle();
    require(p.carrier_diagnostic.valid() && !p.carrier_diagnostic.active() && p.carrier_diagnostic.actions(), "carrier diagnostic invocation invalid or empty");
}
bool MetalGraph::carrier_diagnostic_active() const {
    impl_->thread_check(); return impl_->carrier_diagnostic.active();
}
bool MetalGraph::carrier_capture_enabled() const { impl_->thread_check(); return bool(impl_->carrier_capture); }
uint32_t MetalGraph::carrier_route_snapshot_dispatches(uint32_t layers, uint32_t rows, uint32_t first_layer) const {
    const auto& p = *impl_; p.thread_check();
    if (!p.carrier_diagnostic.active() || !p.carrier_capture) { require(p.carrier_routes.empty(), "unscoped route snapshots"); return 0; }
    require(layers && layers <= 512 && first_layer <= 512 - layers && rows && rows <= 8, "invalid fixed route snapshot geometry");
    std::array<uint16_t, 512> coverage{};
    for (const auto& record : p.carrier_routes) {
        require(record.layer >= first_layer && record.layer - first_layer < layers && record.rows && record.row_begin < rows && record.rows <= rows - record.row_begin, "route snapshot outside fixed target geometry");
        const auto mask = uint16_t(((1u << record.rows) - 1u) << record.row_begin);
        const auto layer = record.layer - first_layer;
        require(!(coverage[layer] & mask), "duplicate target route row snapshot");
        coverage[layer] |= mask;
    }
    for (uint32_t layer = 0; layer < layers; ++layer) require(coverage[layer] == (1u << rows) - 1u, "incomplete target route row snapshots");
    // Count only these explicit bit-copy dispatches. Expected model dispatches
    // stay independent of the observed batch delta and of router row grouping.
    return uint32_t(p.carrier_routes.size());
}
void MetalGraph::configure_carrier_capture(const std::string& path) {
    auto& p = *impl_; p.idle();
    static_cast<void>(p.pipeline("graph_capture_routes"));
    require(!p.carrier_capture && !path.empty(), "diagnostic capture already configured or empty");
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    require(fd >= 0, "cannot exclusively create private diagnostic capture");
    FILE* file = ::fdopen(fd, "wb");
    if (!file) { ::close(fd); throw FormatError("cannot open diagnostic capture stream"); }
    p.carrier_capture.reset(file);
    p.carrier_route_scratch = zeros(256ull << 10, "private diagnostic route snapshots");
    p.carrier_routes.reserve(512);
}
void MetalGraph::capture_carrier_logits(std::span<const float> logits, uint32_t row, uint32_t rows,
                                      const GpuLogitTopKRow* topk) {
    auto& p = *impl_; p.idle();
    if (!p.carrier_diagnostic.active() || !p.carrier_capture) return;
    require(rows && rows <= 8 && row < rows && !logits.empty() && logits.size() <= maximum_sampling_vocabulary,
        "invalid complete diagnostic logits");
    const std::array<uint32_t, 7> header{0x33474f4c, p.carrier_diagnostic.actions(), rows, row, uint32_t(logits.size()), topk ? 1u : 0u, 4};
    const auto write = [&](const void* data, size_t bytes) {
        require(bytes <= (256ull << 20) - p.carrier_capture_bytes, "diagnostic complete-logit capture byte budget exhausted");
        require(std::fwrite(data, 1, bytes, p.carrier_capture.get()) == bytes, "diagnostic complete-logit capture failed");
        p.carrier_capture_bytes += bytes;
    };
    write(header.data(), sizeof(header)); write(logits.data(), logits.size_bytes());
    if (topk) write(topk, sizeof(*topk));
    // Compare and retain the actual CPU full-logit and eligible GPU-summary distributions for all registered policies. No RNG is consumed.
    uint32_t mode = 0;
    for (const auto policy : {SamplingPolicy{}, SamplingPolicy{1, 0, 1, 0}, SamplingPolicy{1, 20, 1, 0}, SamplingPolicy{1, 20, .95, 0}}) {
        const auto full = sampling_distribution(logits, policy);
        const auto actual = topk && sampling_topk_eligible(policy, logits.size()) ? sampling_distribution_from_topk(*topk, policy) : full;
        require(full.support().size() == actual.support().size() &&
            std::equal(full.support().begin(), full.support().end(), actual.support().begin()) &&
            std::equal(full.support_probabilities().begin(), full.support_probabilities().end(), actual.support_probabilities().begin()),
            "diagnostic full-logit and top-k sampling distributions differ");
        const std::array<uint32_t, 2> distribution_header{mode++, uint32_t(actual.support().size())};
        write(distribution_header.data(), sizeof(distribution_header));
        write(actual.support().data(), actual.support().size_bytes());
        write(actual.support_probabilities().data(), actual.support_probabilities().size_bytes());
    }
    require(std::fflush(p.carrier_capture.get()) == 0, "diagnostic capture flush failed");
}
void MetalGraph::set_carrier_diagnostic_layer(uint32_t layer) {
    impl_->thread_check(); impl_->carrier_layer = layer; impl_->carrier_row_offset = 0;
}
#endif
ProfileMode MetalGraph::configure_profiling(uint32_t capacity) {
    @autoreleasepool {
    auto& p = *impl_; p.idle(); require(capacity <= 16384, "GPU profiling capacity limit");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    require(!capacity || !p.concurrent, "concurrent dispatch requires unprofiled execution");
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    require(!p.task4_sink, "reserved Task4 counters are immutable");
#endif
    [[maybe_unused]] const bool diagnostic =
        (std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE") != nullptr)
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        || p.carrier_diagnostic_preparing
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        || p.mtp_profile_preparing
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        || p.task4_preparing
#endif
        ;
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!p.mtp_profile.requested, "reserved MTP counters are immutable");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    require(!capacity || diagnostic || !p.qsa_stage_v8.prepared, "prepared staged QSA arms require unprofiled execution");
#endif
    require(!capacity || !p.hc_q8.splits, "HC Q8 split policy requires unprofiled execution");
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(!capacity || diagnostic || p.hc_exact == HcExactMode::off, "private exact-HC refuses profiling");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!capacity || !p.wide_q8_exact, "private wide-Q8 refuses profiling");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!capacity || !p.wide_q8_many, "private wide-Q8-many refuses profiling");
#endif
    if (!capacity) {
        p.counters.clear(); p.ledger.live_bytes -= p.ledger.profile_counter_payload_bytes; p.ledger.profile_counter_payload_bytes = 0;
        p.profile_mode = ProfileMode::disabled; p.profile_capacity = 0; p.marks.clear(); p.profile = {}; return p.profile_mode;
    }
    const bool dispatch = [p.device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    const bool stage = [p.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    require(dispatch || stage, "device does not support scoped GPU counter sampling");
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!p.mtp_profile_preparing || mtp_profile_boundary_supported(p.mtp_profile_boundary, dispatch, stage),
        "explicit MTP profile boundary is unsupported; automatic fallback refused");
#endif
    id<MTLCounterSet> timestamp_set = nil;
    for (id<MTLCounterSet> candidate in p.device.counterSets) if ([candidate.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamp_set = candidate;
    require(timestamp_set != nil, "device has no timestamp counter set");
    const auto page = uint64_t(getpagesize());
    uint64_t payload = 0;
    for (uint32_t first = 0; first < capacity; first += counter_chunk_dispatches)
        payload = checked_add(payload, checked_add(uint64_t(std::min(counter_chunk_dispatches, capacity - first)) * 2 * sizeof(MTLCounterResultTimestamp), page - 1) / page * page);
    require(payload <= p.ledger.budget - p.ledger.live_bytes + p.ledger.profile_counter_payload_bytes, "GPU counter payload exceeds graph budget");
    std::vector<Impl::Mark> marks; marks.reserve(capacity);
    std::vector<id<MTLCounterSampleBuffer>> samples;
    samples.reserve((capacity + counter_chunk_dispatches - 1) / counter_chunk_dispatches);
    for (uint32_t first = 0; first < capacity; first += counter_chunk_dispatches) {
        MTLCounterSampleBufferDescriptor* descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
        descriptor.counterSet = timestamp_set; descriptor.sampleCount = std::min(counter_chunk_dispatches, capacity - first) * 2;
        descriptor.storageMode = MTLStorageModeShared; descriptor.label = @"owned opt-in dispatch profiler chunk";
        NSError* error = nil; id<MTLCounterSampleBuffer> buffer = [p.device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        if (!buffer) throw metal_error("allocating scoped GPU timestamp buffer", error);
        samples.push_back(buffer);
    }
    p.counters = std::move(samples); p.marks = std::move(marks); p.profile_capacity = capacity; p.profile = {};
    p.profile_mode = dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    if (p.mtp_profile_preparing) p.profile_mode = p.mtp_profile_boundary == MtpProfileBoundary::dispatch
        ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    if (p.task4_preparing) p.profile_mode = p.task4_boundary == Task4ProfileBoundary::dispatch
        ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
#endif
    p.ledger.live_bytes = p.ledger.live_bytes - p.ledger.profile_counter_payload_bytes + payload; p.ledger.profile_counter_payload_bytes = payload;
    p.ledger.peak_bytes = std::max(p.ledger.peak_bytes, p.ledger.live_bytes); ++p.ledger.allocations;
    return p.profile_mode;
    }
}
const GraphProfile& MetalGraph::last_profile() const { impl_->idle(); return impl_->profile; }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
void MetalGraph::prepare_task4_profile(uint32_t capacity, uint32_t batches, Task4ProfileSink sink, void* context, Task4ProfileBoundary boundary) {
    auto& p = *impl_; p.idle();
    require(capacity && capacity <= 16384 && batches && batches <= 8192 && sink && context &&
        !p.task4_sink && p.counters.empty() && p.profile_mode == ProfileMode::disabled &&
        !p.recording && !p.compatible_pipelines && p.captures.empty() &&
        p.small_gemv_splits == 16 && p.block_gemv_formats == 3,
        "Task4 requires bounded explicit serial fixed-policy diagnostic setup");
    require(task4_profile_boundary_supported(boundary,
        [p.device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary],
        [p.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),
        "explicit Task4 counter boundary is unsupported; automatic fallback refused");
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!p.mtp_profile.requested && !p.mtp_profile.active, "Task4 and MTP round profiling are exclusive");
#endif
    p.task4_boundary = boundary; p.task4_preparing = true;
    const auto expected = boundary == Task4ProfileBoundary::dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
    try { require(configure_profiling(capacity) == expected, "Task4 selected counter boundary differs"); }
    catch (...) { p.task4_preparing = false; throw; }
    p.task4_preparing = false;
    p.task4_sink = sink; p.task4_context = context; p.task4_max_batches = batches;
    p.task4_active = boundary == Task4ProfileBoundary::dispatch;
    if (!p.task4_active) p.profile_mode = ProfileMode::disabled;
}
void MetalGraph::begin_task4_profile_window(uint32_t prefix) {
    auto& p = *impl_; p.idle();
    require(p.task4_sink && p.task4_boundary == Task4ProfileBoundary::encoder_late8 && !p.task4_active &&
        !p.task4_batches && prefix == 4088 && !p.counters.empty() && p.profile_mode == ProfileMode::disabled,
        "Task4 encoder window must begin once before the final input8 packet");
    p.profile_mode = ProfileMode::encoder_boundary; p.task4_active = true;
}
void MetalGraph::set_task4_profile_tag(Task4ProfileTag tag) {
    auto& p = *impl_; p.thread_check();
    require(tag.layer >= -1 && tag.layer < 48 && tag.prefix <= 4097 && tag.rows <= 8 &&
        tag.rows <= 4097 - tag.prefix && uint32_t(tag.phase) <= uint32_t(Task4ProfilePhase::mtp_output),
        "Task4 diagnostic tag outside the frozen 4K/next-token scope");
    p.task4_tag = tag;
}
bool MetalGraph::task4_profile_reserved() const { impl_->thread_check(); return impl_->task4_sink != nullptr; }
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
void MetalGraph::prepare_mtp_generation_profile(uint32_t rounds, MtpProfileBoundary boundary, MtpProfileSink sink, void* context) {
    auto& p = *impl_; p.idle();
    MtpProfileWindow candidate; candidate.prepare(rounds);
    require(boundary == MtpProfileBoundary::dispatch || boundary == MtpProfileBoundary::encoder,
        "unknown explicit MTP profile boundary");
    require(sink && context && !p.mtp_profile.requested && p.counters.empty() &&
        p.profile_mode == ProfileMode::disabled && p.captures.empty() && !p.recording && !p.compatible_pipelines &&
        p.hc_exact == HcExactMode::combined && p.qsa_stage_v8.prepared && p.qsa_stage_v8.enabled &&
        p.qsa_channel.prepared && p.qsa_channel.enabled && p.qsa_rows_prepared &&
        !p.hc_q8.splits && p.small_gemv_splits == 16 && p.block_gemv_formats == 3 &&
        !p.wide_q8_exact && !p.wide_q8_many && !p.lane_matrices.empty(),
        "MTP profiling requires the fixed prepared lane/channel/QSA-row/combined-HC carrier");
    p.mtp_profile_boundary = boundary;
    p.mtp_profile_preparing = true;
    try { static_cast<void>(configure_profiling(mtp_profile_dispatch_capacity)); }
    catch (...) { p.mtp_profile_preparing = false; throw; }
    p.mtp_profile_preparing = false;
    // The same charged buffers persist through unprofiled prefill and all later
    // rounds. Counter payloads never allocate per round. An explicitly selected
    // encoder boundary uses the existing one-dispatch encoder profiler below.
    p.profile_mode = ProfileMode::disabled;
    p.mtp_profile = candidate; p.mtp_profile_sink = sink; p.mtp_profile_context = context;
}
bool MetalGraph::begin_mtp_profile_round(uint32_t base) {
    auto& p = *impl_; p.idle();
    require(p.profile_mode == ProfileMode::disabled && !p.mtp_profile.active, "MTP profile window overlaps another profiler");
    if (!p.mtp_profile.begin(base)) return false;
    require(p.mtp_profile_sink && !p.counters.empty(), "MTP profile reservation lost");
    p.profile_mode = p.mtp_profile_boundary == MtpProfileBoundary::dispatch
        ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
    return true;
}
void MetalGraph::end_mtp_profile_round() {
    auto& p = *impl_; p.idle(); p.mtp_profile.end(); p.profile_mode = ProfileMode::disabled; p.profile = {};
}
void MetalGraph::set_mtp_profile_phase(MtpProfilePhase phase) {
    auto& p = *impl_; p.idle();
    if (!p.mtp_profile.active) return;
    require(phase >= MtpProfilePhase::draft && phase <= MtpProfilePhase::prime, "invalid MTP profile phase");
    p.mtp_profile.phase = phase;
}
bool MetalGraph::mtp_profile_reserved() const { impl_->thread_check(); return impl_->mtp_profile.requested && impl_->mtp_profile_sink; }
MtpProfileWindow MetalGraph::mtp_profile_window() const { impl_->idle(); return impl_->mtp_profile; }
#endif
void MetalGraph::configure_small_gemv(uint32_t splits) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    require(!p.qsa_stage_v8.prepared || splits == 16, "immutable staged QSA small policy");
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(p.hc_exact == HcExactMode::off || splits == 16, "immutable private exact-HC small policy");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact || splits == 16, "immutable private wide-Q8 small policy");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many || splits == 16, "immutable private wide-Q8-many small policy");
#endif
    require(splits == 0 || splits == 1 || splits == 4 || splits == 8 || splits == 16 || splits == 32, "unsupported small GEMV split count");
    if (splits <= 1) {
        if (p.small_gemv_scratch.owner) release(p.small_gemv_scratch);
        p.small_gemv_scratch = {}; p.small_gemv_splits = splits; return;
    }
    if (!p.small_gemv_scratch.owner) p.small_gemv_scratch = zeros(64 * 32 * sizeof(float), "opt-in small GEMV partials");
    p.small_gemv_splits = splits;
}
uint32_t MetalGraph::gemv_dispatches(uint32_t type, uint32_t columns, uint32_t rows) const {
    const auto& p = *impl_; p.thread_check();
    if (p.hc_q8.splits && type == 8 && columns == 10240 && rows == 320) return 2;
    return p.small_gemv_splits > 1 && type == 0 && rows <= 64 && columns >= 1024 ? 2 : 1;
}
void MetalGraph::configure_block_gemv(uint32_t formats) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    require(!p.qsa_stage_v8.prepared || formats == 3, "immutable staged QSA block policy");
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(p.hc_exact == HcExactMode::off || formats == 3, "immutable private exact-HC block policy");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact || formats == 3, "immutable private wide-Q8 block policy");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many || formats == 3, "immutable private wide-Q8-many block policy");
#endif
    require(formats <= 3, "unsupported block GEMV format mask");
    require(!p.hc_q8.splits || (formats & 1), "HC Q8 split policy requires block-Q8 base policy");
    p.block_gemv_formats = formats;
}
std::array<uint32_t, 2> MetalGraph::kernel_policy() const {
    const auto& p = *impl_; p.thread_check();
    return {p.small_gemv_splits, p.block_gemv_formats};
}
void MetalGraph::configure_hc_q8_splits(uint32_t splits) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    require(!splits || !p.qsa_stage_v8.prepared, "prepared staged QSA arms refuse split-HC");
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(!splits || p.hc_exact == HcExactMode::off, "private exact-HC refuses split-HC");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!splits || !p.wide_q8_exact, "private wide-Q8 refuses split-HC");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!splits || !p.wide_q8_many, "private wide-Q8-many refuses split-HC");
#endif
    require(splits == 0 || splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16 || splits == 32,
        "unsupported HC Q8 split count");
    require(!p.ledger.live_captures, "HC Q8 policy cannot change with live captures");
    if (!splits) {
        if (p.hc_q8_scratch.owner) release(p.hc_q8_scratch);
        p.hc_q8_scratch = {}; p.hc_q8 = {}; return;
    }
    require(p.profile_mode == ProfileMode::disabled && (p.block_gemv_formats & 1),
        "HC Q8 split policy requires unprofiled block-Q8 execution");
    prepare_q8_matvec_split();
    if (!p.hc_q8_scratch.owner) {
        const auto before = p.ledger.live_bytes;
        const auto scratch = zeros(320 * 32 * sizeof(float), "opt-in HC Q8 maximum partials");
        p.hc_q8_scratch = scratch;
        p.hc_q8.scratch_logical_bytes = scratch.bytes;
        p.hc_q8.scratch_charged_bytes = p.ledger.live_bytes - before;
    }
    p.hc_q8.splits = splits; // Publish after both pipelines and scratch exist.
}
HcQ8Policy MetalGraph::hc_q8_policy() const {
    impl_->thread_check(); return impl_->hc_q8;
}
#ifdef TOKSPUR_PRIVATE_HC_EXACT
void MetalGraph::configure_hc_exact(HcExactMode mode) {
    auto& p = *impl_; p.idle();
    require(uint32_t(mode) <= 2 && !p.hc_exact_locked, "private exact-HC mode is startup-only");
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    require(!p.qsa_channel.prepared || mode == HcExactMode::combined, "prepared chronological QSA pins combined exact-HC");
#endif
    if (mode != HcExactMode::off) {
        require(p.profile_mode == ProfileMode::disabled && !p.hc_q8.splits && p.captures.empty() &&
            p.small_gemv_splits == 16 && p.block_gemv_formats == 3, "private exact-HC requires fixed split16/block3, no profiling/captures/split-HC");
        for (size_t i = 0; i < (mode == HcExactMode::combined ? 5u : 1u); ++i)
            static_cast<void>(p.pipeline(hc_exact_kernel(i), uint32_t(-1), 32));
    }
    p.hc_exact = mode; // Publish only after every required PSO/capability succeeds.
}
HcExactMode MetalGraph::hc_exact_mode() const { impl_->thread_check(); return impl_->hc_exact; }
HcExactRoutes MetalGraph::hc_exact_routes() const { impl_->idle(); return impl_->hc_exact_completed; }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
void MetalGraph::configure_wide_q8_exact(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.wide_q8_exact_locked, "private wide-Q8 policy is startup-only");
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    require(!enabled || !p.qsa_channel.prepared, "prepared chronological QSA refuses scalar wide-Q8");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!enabled || !p.wide_q8_many, "private wide-Q8-many refuses scalar wide policy");
#endif
    if (enabled) {
        require(p.profile_mode == ProfileMode::disabled && !p.hc_q8.splits && p.captures.empty() &&
            p.small_gemv_splits == 16 && p.block_gemv_formats == 3,
            "private wide-Q8 requires split16/block3, no profiling/captures/split-HC");
        for (size_t shape = 0; shape < 2; ++shape) static_cast<void>(p.pipeline(wide_q8_exact_kernel(shape)));
    }
    p.wide_q8_exact = enabled; // Only after both selected pipelines succeed; no buffer/scratch allocation.
}
bool MetalGraph::wide_q8_exact_enabled() const { impl_->thread_check(); return impl_->wide_q8_exact; }
WideQ8ExactRoutes MetalGraph::wide_q8_exact_routes() const { impl_->idle(); return impl_->wide_q8_exact_completed; }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
void MetalGraph::configure_wide_q8_many(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.wide_q8_many_locked, "private wide-Q8-many policy is startup-only");
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    require(!enabled || !p.qsa_channel.prepared, "prepared chronological QSA refuses wide-Q8-many");
#endif
    if (enabled) {
        require(p.profile_mode == ProfileMode::disabled && !p.hc_q8.splits && p.captures.empty() &&
            !p.wide_q8_exact && p.small_gemv_splits == 16 && p.block_gemv_formats == 3,
            "private wide-Q8-many requires split16/block3, scalar wide off, no profiling/captures/split-HC");
        for (size_t bucket = 0; bucket < 6; ++bucket) static_cast<void>(p.pipeline(wide_q8_many_kernel(bucket)));
    }
    p.wide_q8_many = enabled; // No buffer/scratch allocation; publish only after every PSO succeeds.
}
bool MetalGraph::wide_q8_many_enabled() const { impl_->thread_check(); return impl_->wide_q8_many; }
WideQ8ManyRoutes MetalGraph::wide_q8_many_routes() const { impl_->idle(); return impl_->wide_q8_many_completed; }
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
void MetalGraph::configure_qsa_stage_v8(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.qsa_stage_v8_locked && (!p.qsa_stage_v8.prepared || enabled == p.qsa_stage_v8.enabled),
        "private staged QSA composition is immutable startup policy");
    if (p.qsa_stage_v8.prepared) return;
    require(!p.hc_q8.splits && p.profile_mode == ProfileMode::disabled && p.captures.empty() &&
        p.small_gemv_splits == 16 && p.block_gemv_formats == 3,
        "private staged QSA composition requires direct unprofiled split16/block3/HC0 without captures");
    prepare_qsa_score_stage();
    const auto before = p.ledger.live_bytes;
    const auto scratch = zeros(maximum_qsa_stage_v8_scores, "private composed QSA maximum staged scores");
    p.qsa_stage_v8_scratch = scratch;
    p.qsa_stage_v8 = {true, enabled, scratch.bytes, p.ledger.live_bytes - before};
}
QsaStageV8Policy MetalGraph::qsa_stage_v8_policy() const { impl_->thread_check(); return impl_->qsa_stage_v8; }
uint64_t MetalGraph::qsa_stage_v8_calls() const { impl_->idle(); return impl_->qsa_stage_v8_completed; }
#endif

#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
void MetalGraph::configure_qsa_channel_chronological(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.qsa_channel_locked && (!p.qsa_channel.prepared || enabled == p.qsa_channel.enabled),
        "private chronological QSA is immutable startup policy");
    if (p.qsa_channel.prepared) return;
    require(p.qsa_stage_v8.prepared && p.qsa_stage_v8.enabled && p.hc_exact == HcExactMode::combined &&
        p.profile_mode == ProfileMode::disabled && !p.hc_q8.splits && p.captures.empty() &&
        p.small_gemv_splits == 16 && p.block_gemv_formats == 3 && !p.wide_q8_exact && !p.wide_q8_many,
        "chronological QSA requires prepared Stage-V8 ON, combined exact-HC and fixed direct unprofiled policy");
    // Validate BOTH new functions/capabilities before any allocation/publication,
    // in both arms. Stage already owns/prepares the unchanged score pass.
    static_cast<void>(p.pipeline("qsa_chronological_factors", uint32_t(-1), qsa_channel_threads));
    static_cast<void>(p.pipeline(
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
            p.kernel_qsa_f32 ? "qsa_channels_search_f32" :
#endif
            "qsa_chronological_channels", uint32_t(-1), qsa_channel_threads));
    const auto before = p.ledger.live_bytes;
    const auto factors = zeros(maximum_qsa_channel_factors, "private chronological QSA maximum factors");
    p.qsa_channel_factors = factors;
    p.qsa_channel = {true, enabled, factors.bytes, p.ledger.live_bytes - before};
}
QsaChannelPolicy MetalGraph::qsa_channel_policy() const { impl_->thread_check(); return impl_->qsa_channel; }
uint64_t MetalGraph::qsa_channel_calls() const { impl_->idle(); return impl_->qsa_channel_completed; }
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
void MetalGraph::configure_kernel_qsa_f32(bool enabled) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    require(!enabled || !p.kernel_grouped_fp16, "step-2 candidates must run separately");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    require(!enabled || !p.qsa_sparse_enabled, "step-2 QSA candidate conflicts with sparse QSA");
#endif
    require(!p.qsa_channel_locked && !p.kernel_qsa_selected && p.qsa_channel.prepared && p.qsa_channel.enabled,
        "QSA search candidate requires immutable prepared chronological startup");
    static_cast<void>(p.pipeline("qsa_channels_search_f32", uint32_t(-1), 256));
    static_cast<void>(p.pipeline("qsa_rows_channels_search_f32", uint32_t(-1), 256));
    p.kernel_qsa_f32 = enabled; p.kernel_qsa_selected = true;
}
bool MetalGraph::kernel_qsa_f32_enabled() const { impl_->thread_check(); return impl_->kernel_qsa_f32; }
bool MetalGraph::kernel_step2_rollback_enabled() const { impl_->thread_check(); return impl_->kernel_qsa_selected; }
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
void MetalGraph::configure_kernel_grouped_fp16(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.kernel_grouped_selected && p.grouped_mm_prepared.empty() && p.prefill_grouped_f32 &&
        p.kernel_qsa_selected && !p.kernel_qsa_f32, "grouped step-2 requires fixed FP32 composition and explicit QSA off before preparation");
    p.kernel_grouped_fp16 = enabled; p.kernel_grouped_selected = true;
}
bool MetalGraph::kernel_grouped_fp16_enabled() const { impl_->thread_check(); return impl_->kernel_grouped_fp16; }
uint64_t MetalGraph::kernel_grouped_fp16_encoded_calls() const { impl_->thread_check(); return impl_->kernel_grouped_encoded; }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
void MetalGraph::configure_qsa_sparse(bool enabled) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    require(!enabled || !p.kernel_qsa_f32, "sparse QSA conflicts with step-2 QSA F32 candidate");
#endif
    require(!p.qsa_sparse_locked && !p.qsa_channel_locked && (!p.qsa_sparse_configured || p.qsa_sparse_enabled == enabled) &&
        p.qsa_channel.prepared && p.qsa_channel.enabled && p.qsa_stage_v8.prepared && p.qsa_stage_v8.enabled &&
        p.hc_exact == HcExactMode::combined && p.profile_mode == ProfileMode::disabled && !p.hc_q8.splits &&
        p.small_gemv_splits == 16 && p.block_gemv_formats == 3 && !p.wide_q8_exact && !p.wide_q8_many && p.captures.empty(),
        "sparse QSA requires immutable prepared fixed chronological startup");
    static_cast<void>(p.pipeline("qsa_sparse_attention", uint32_t(-1), Impl::qsa_sparse_decode_threads));
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    static_cast<void>(p.pipeline("qsa_sparse_attention_rows", uint32_t(-1), Impl::qsa_sparse_verify_threads));
#endif
    p.qsa_sparse_configured = true;
    p.qsa_sparse_enabled = enabled;
}
bool MetalGraph::qsa_sparse_enabled() const { impl_->thread_check(); return impl_->qsa_sparse_enabled; }
bool MetalGraph::qsa_sparse_rollback_enabled() const { impl_->thread_check(); return impl_->qsa_sparse_configured; }
uint64_t MetalGraph::qsa_sparse_decode_calls() const { impl_->idle(); return impl_->qsa_sparse_decode_completed; }
uint64_t MetalGraph::qsa_sparse_verify_calls() const { impl_->idle(); return impl_->qsa_sparse_verify_completed; }
void MetalGraph::prepare_qsa_sparse_variants() {
    auto& p = *impl_; p.idle();
    require(!p.qsa_sparse_locked && p.qsa_channel.prepared && p.qsa_channel.enabled && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "sparse QSA variants require idle prepared chronological startup");
    static_cast<void>(p.pipeline("qsa_sparse_attention_bn8", uint32_t(-1), 256));
    static_cast<void>(p.pipeline("qsa_sparse_attention_bn16", uint32_t(-1), 512));
    static_cast<void>(p.pipeline("qsa_sparse_attention_bn32", uint32_t(-1), 1024));
    static_cast<void>(p.pipeline("qsa_sparse_attention_rows_bn8", uint32_t(-1), 256));
    static_cast<void>(p.pipeline("qsa_sparse_attention_rows_bn16", uint32_t(-1), 512));
    static_cast<void>(p.pipeline("qsa_sparse_attention_rows_bn32", uint32_t(-1), 1024));
    p.qsa_sparse_variants_prepared = true;
}
#endif
#endif

#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
void MetalGraph::configure_batch_phase_clock(bool enabled) {
    auto& p = *impl_; p.idle();
    auto candidate = p.phase_clock;
    require(candidate.configure(enabled), "private batch phase clock is immutable startup policy");
    require(p.qsa_stage_v8.prepared && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "private batch phase clock requires the prepared unprofiled Stage-V8 path");
    require(!enabled || phase_stderr_writable(), "private batch phase clock ON requires regular writable stderr");
    p.phase_clock = candidate;
}
BatchPhaseClockState MetalGraph::batch_phase_clock_state() const { impl_->idle(); return impl_->phase_clock; }
#endif

void MetalGraph::validate_disjoint(std::span<const GpuView> inputs, std::span<const GpuView> outputs) const {
    for (const auto input : inputs) static_cast<void>(impl_->allocation(input));
    for (size_t i = 0; i < outputs.size(); ++i) {
        static_cast<void>(impl_->allocation(outputs[i], true));
        for (const auto input : inputs) require(!overlaps(outputs[i], input), "composite GPU output overlaps a protected input");
        for (size_t j = 0; j < i; ++j) require(!overlaps(outputs[i], outputs[j]), "composite GPU outputs overlap");
    }
}

#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
void MetalGraph::configure_operator_checkpoints(OperatorCheckpointSink sink, void* context, uint32_t limit) {
    auto& p = *impl_; p.idle();
    constexpr uint32_t max_limit = 16384;
    const bool valid_limit = limit && (limit <= max_limit
#if defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED)
        || limit == 131072
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        || limit == 1048576
#endif
    );
    require(!p.operator_locked && !p.ledger.live_captures && p.profile_mode == ProfileMode::disabled &&
        (sink ? context && valid_limit : !context), "operator checkpoints require startup, direct unprofiled batches and a bounded sink");
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    require(!p.phase_clock.enabled, "operator checkpoints refuse overlapping batch-phase diagnostics");
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    require(!p.trace, "operator checkpoints refuse the per-dispatch trace ledger");
#endif
    p.operator_sink = sink; p.operator_context = context; p.operator_limit = sink ? limit : 0;
}
void MetalGraph::operator_checkpoint(const char* category, const char* tag, std::array<uint32_t, 4> dimensions) {
    auto& p = *impl_; p.thread_check(); if (!p.operator_sink) return; p.active();
    require(!p.operator_cutting && !p.recording && !p.compatible_pipelines && p.operator_segments < p.operator_limit,
        "operator checkpoint requires a bounded direct batch");
    if (!p.dispatches) return;
    p.operator_cutting = true;
    try {
        const auto physical = submit(); p.operator_add(physical);
        p.operator_emit(category, tag, dimensions, false, &physical, p.operator_rollup);
        begin(); p.operator_cutting = false;
    } catch (...) {
        p.operator_cutting = false; p.operator_rollup = {}; p.operator_routes = {}; p.operator_barriers = 0;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        // A physical submit may already have made the Graph idle before the sink fails. Revoke the lease even when abort has no active command left to discard.
        p.grouped_routes_active = false; p.grouped_routes_views = {};
#endif
        if (p.command) try { abort(); } catch (const FormatError&) {}
        throw;
    }
}
void MetalGraph::set_operator_checkpoint_context(const char* phase, int32_t layer, uint32_t prefix, uint32_t rows) {
    auto& p = *impl_; p.thread_check(); if (!p.operator_sink) return;
    if (p.command) operator_checkpoint("unclassified", "before_context_change", {});
    p.operator_tag.phase = phase; p.operator_tag.layer = layer; p.operator_tag.prefix = prefix; p.operator_tag.rows = rows;
}
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
void MetalGraph::configure_qsa_route_attribution(QsaRouteAttributionSink sink, void* context) {
    auto& p = *impl_; p.idle();
    require(sink && context && !p.qsa_route_sink && !p.operator_locked && p.operator_sink &&
        p.qsa_sparse_configured && !p.qsa_sparse_enabled
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        && !p.route_sink
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        && !p.route_lock_sink
#endif
        , "QSA route attribution requires a fresh exact sparse-configured operator diagnostic graph");
    p.qsa_route_backup = {
        zeros(65536, "qsa route attribution keys backup"),
        zeros(65536, "qsa route attribution values backup"),
        zeros(65536, "qsa route attribution index backup")
    };
    p.qsa_route_sink = sink; p.qsa_route_context = context;
}
bool MetalGraph::qsa_route_attribution_enabled() const { impl_->thread_check(); return impl_->qsa_route_sink != nullptr; }
std::array<GpuView, 3> MetalGraph::qsa_route_attribution_backup() const {
    auto& p = *impl_; p.thread_check(); require(p.qsa_route_sink != nullptr, "QSA route attribution is not configured");
    return p.qsa_route_backup;
}
void MetalGraph::qsa_route_attribution_phase(uint32_t phase) {
    auto& p = *impl_; p.thread_check();
    require(p.qsa_route_sink && (phase == 0 || phase == 2 || phase == 3), "invalid QSA route attribution phase");
    p.qsa_route_phase = phase;
}
void MetalGraph::qsa_route_attribution_emit(QsaRouteAttributionRecord record, std::span<const GraphCopy> copies) {
    auto& p = *impl_; p.active();
    require(p.qsa_route_sink && p.operator_sink && !p.operator_cutting, "QSA route attribution needs the explicit checkpoint owner");
    p.operator_cutting = true;
    try {
        if (p.dispatches) {
            const auto physical = submit(); p.operator_add(physical);
            p.operator_emit("qsa_route_attribution", "capture", {record.phase, record.rows, record.experts, record.selected}, false, &physical, p.operator_rollup);
        } else {
            const auto total = p.operator_rollup; const auto routes = p.operator_routes; const auto barriers = p.operator_barriers;
            abort(); p.operator_rollup = total; p.operator_routes = routes; p.operator_barriers = barriers; p.operator_cutting = true;
        }
        if (!copies.empty()) copy(copies);
        record.layer = p.operator_tag.layer; record.prefix = p.operator_tag.prefix;
        require(p.qsa_route_sink(p.qsa_route_context, *this, record), "QSA route attribution sink refused");
        begin(); p.operator_cutting = false;
    } catch (...) {
        p.operator_cutting = false; p.qsa_route_phase = 0; p.operator_rollup = {}; p.operator_routes = {}; p.operator_barriers = 0;
        if (p.command) try { abort(); } catch (const FormatError&) {}
        throw;
    }
}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
void MetalGraph::configure_gdn_route_attribution(GdnRouteAttributionSink sink, void* context, std::optional<uint32_t> prefill_prefix) {
    auto& p = *impl_; p.idle();
    require(sink && context && !p.gdn_route_sink && !p.operator_locked && p.operator_sink &&
        p.gdn_ported_configured && !p.gdn_ported_enabled
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        && !p.qsa_route_sink
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        && !p.route_sink
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        && !p.route_lock_sink
#endif
        , "GDN route attribution requires a fresh exact ported-configured operator diagnostic graph");
    p.gdn_route_prefill_prefix = prefill_prefix;
    p.gdn_route_sink = sink; p.gdn_route_context = context;
}
bool MetalGraph::gdn_route_attribution_enabled() const { impl_->thread_check(); return impl_->gdn_route_sink != nullptr; }
bool MetalGraph::gdn_route_attribution_prefill(uint32_t prefix, uint32_t steps) const {
    const auto& p = *impl_; p.thread_check();
    return p.gdn_route_sink != nullptr && steps >= 2 && (!p.gdn_route_prefill_prefix.has_value() || *p.gdn_route_prefill_prefix == prefix);
}
void MetalGraph::gdn_route_attribution_phase(uint32_t phase) {
    auto& p = *impl_; p.thread_check();
    require(p.gdn_route_sink && (phase == 0 || phase == 2 || phase == 3), "invalid GDN route attribution phase");
    p.gdn_route_phase = phase;
}
void MetalGraph::gdn_route_attribution_emit(GdnRouteAttributionRecord record, std::span<const GraphCopy> copies) {
    auto& p = *impl_; p.active();
    require(p.gdn_route_sink && p.operator_sink && !p.operator_cutting, "GDN route attribution needs the explicit checkpoint owner");
    p.operator_cutting = true;
    try {
        if (p.dispatches) {
            const auto physical = submit(); p.operator_add(physical);
            p.operator_emit("gdn_route_attribution", "capture", {record.phase, record.rows, record.experts, record.selected}, false, &physical, p.operator_rollup);
        } else {
            const auto total = p.operator_rollup; const auto routes = p.operator_routes; const auto barriers = p.operator_barriers;
            abort(); p.operator_rollup = total; p.operator_routes = routes; p.operator_barriers = barriers; p.operator_cutting = true;
        }
        if (!copies.empty()) copy(copies);
        record.layer = p.operator_tag.layer; record.prefix = p.operator_tag.prefix;
        require(p.gdn_route_sink(p.gdn_route_context, *this, record), "GDN route attribution sink refused");
        begin(); p.operator_cutting = false;
    } catch (...) {
        p.operator_cutting = false; p.gdn_route_phase = 0; p.operator_rollup = {}; p.operator_routes = {}; p.operator_barriers = 0;
        if (p.command) try { abort(); } catch (const FormatError&) {}
        throw;
    }
}
#endif
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
void MetalGraph::configure_route_attribution(RouteAttributionSink sink, void* context) {
    auto& p = *impl_; p.idle();
    require(sink && context && !p.route_sink && !p.operator_locked && p.operator_sink && p.prefill_dense_staged_f32 &&
        p.prefill_grouped_f32 && !p.prefill_hc_down_f32 && !p.moe_router_f32_batch,
        "route attribution requires a fresh explicit staged/scalar diagnostic graph");
    const auto pipeline = p.pipeline(moe_router_f32_kernel);
    require(pipeline.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength, "route attribution router pipeline capability");
    p.route_logits = zeros(512ull * 512 * 4, "route attribution logits");
    p.route_ids = zeros(512ull * 10 * 4, "route attribution ids");
    p.route_weights = zeros(512ull * 10 * 4, "route attribution weights");
    p.route_sink = sink; p.route_context = context; p.moe_router_f32_prepared = true;
}
bool MetalGraph::route_attribution_enabled() const { impl_->thread_check(); return impl_->route_sink != nullptr; }
void MetalGraph::route_attribution_phase(uint32_t phase) {
    auto& p = *impl_; p.thread_check(); require(p.route_sink && (phase == 0 || phase == 2 || phase == 3), "route attribution phase"); p.route_phase = phase;
}
void MetalGraph::route_attribution_emit(RouteAttributionRecord record, std::span<const GraphCopy> restore) {
    auto& p = *impl_; p.active(); require(p.route_sink && p.operator_sink && !p.operator_cutting, "route attribution needs the explicit checkpoint owner");
    p.operator_cutting = true;
    try {
        if (p.dispatches) {
            const auto physical=submit();p.operator_add(physical);
            p.operator_emit("route_attribution","capture",{record.phase,record.rows,record.experts,record.selected},false,&physical,p.operator_rollup);
        } else {
            // An earlier checkpoint may already have drained this segment. Retain the logical owner while discarding only its empty encoder.
            const auto total=p.operator_rollup;const auto routes=p.operator_routes;const auto barriers=p.operator_barriers;
            abort();p.operator_rollup=total;p.operator_routes=routes;p.operator_barriers=barriers;p.operator_cutting=true;
        }
        if (!restore.empty()) copy(restore);
        record.layer = p.operator_tag.layer;
        require(p.route_sink(p.route_context, *this, record), "route attribution sink refused");
        begin(); p.operator_cutting = false;
    } catch (...) {
        p.operator_cutting=false;p.route_phase=0;p.operator_rollup={};p.operator_routes={};p.operator_barriers=0;
        if(p.command)try{abort();}catch(const FormatError&){}throw;
    }
}
void MetalGraph::route_attribution_common(uint32_t phase, std::array<GpuView, 6> views, std::span<const GraphCopy> restore) {
    require(phase == 1 || phase == 5, "route attribution common-input phase");
    RouteAttributionRecord record; record.phase = phase; record.views = views; route_attribution_emit(record, restore);
}
void MetalGraph::route_attribution_rows(GpuView matrix, GpuRowsF32 input, GpuView logits, GpuView ids, GpuView weights,
    uint32_t selected, RouterMode mode, bool normalise) {
    auto& p = *impl_; if (!p.route_sink || !p.route_phase) return;
    require((p.route_phase == 2 || p.route_phase == 3) && input.rows >= 2 && input.rows <= 512 && input.columns == 2560 &&
        input.stride_bytes == 2560 * 4 && selected == 10 && logits.bytes == uint64_t(input.rows)*512*4,
        "route attribution requires actual F32 top10 model rows");
    const auto x = slice(input.view, 0, uint64_t(input.rows)*input.stride_bytes);
    RouteAttributionRecord record{p.route_phase,input.rows,input.columns,512,selected,-1,{x,logits,ids,weights,{},{}}};
    route_attribution_emit(record);
    if (p.route_phase == 3) {
        const auto y = slice(p.route_logits,0,uint64_t(input.rows)*512*4), ri = slice(p.route_ids,0,uint64_t(input.rows)*selected*4), rw = slice(p.route_weights,0,ri.bytes);
        moe_router_f32_batch(matrix,input,{y,input.rows,512,512*4});
        for (uint32_t t=0;t<input.rows;++t) route(slice(y,uint64_t(t)*512*4,512*4),slice(ri,uint64_t(t)*selected*4,selected*4),slice(rw,uint64_t(t)*selected*4,selected*4),selected,mode,normalise);
        record.phase=4;record.views={x,y,ri,rw,{},{}};route_attribution_emit(record);
    }
}
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
void MetalGraph::configure_route_lock(RouteLockSink sink, void* context) {
    auto& p=*impl_;p.idle();
    require(sink && context && !p.route_lock_sink && !p.operator_locked && p.operator_sink && !p.route_sink,
        "route lock needs a fresh explicit operator diagnostic without common-input replay");
    p.route_lock_sink=sink;p.route_lock_context=context;
}
void MetalGraph::route_lock_rows(GpuView logits,GpuView ids,GpuView weights,uint32_t rows,uint32_t channels,uint32_t experts,uint32_t selected,uint32_t row_offset) {
    auto& p=*impl_;if(!p.route_lock_sink)return;p.active();
    require(rows && rows<=512 && channels==2560 && experts==512 && selected==10 && logits.bytes==uint64_t(rows)*experts*4 &&
        ids.bytes==uint64_t(rows)*selected*4 && weights.bytes==ids.bytes && !p.operator_cutting,"route lock actual geometry or owner");
    const RouteLockRecord record{p.operator_tag,rows,channels,experts,selected,row_offset,logits,ids,weights};
    p.operator_cutting=true;
    try {
        if(p.dispatches){const auto physical=submit();p.operator_add(physical);p.operator_emit("route_lock","candidate_computed",{rows,experts,selected,row_offset},false,&physical,p.operator_rollup);}
        else {const auto total=p.operator_rollup;const auto routes=p.operator_routes;const auto barriers=p.operator_barriers;abort();p.operator_rollup=total;p.operator_routes=routes;p.operator_barriers=barriers;p.operator_cutting=true;}
        require(p.route_lock_sink(p.route_lock_context,*this,record),"route lock sink refused");
        begin();p.operator_cutting=false;
    }catch(...){p.operator_cutting=false;p.operator_rollup={};p.operator_routes={};p.operator_barriers=0;if(p.command)try{abort();}catch(const FormatError&){}throw;}
}
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
void MetalGraph::configure_kernel_search_capture(KernelSearchCaptureSink sink, void* context) {
    auto& p = *impl_; p.idle();
    require(sink && context && !p.kernel_search_sink && !p.operator_locked && p.operator_sink && !p.route_sink && !p.route_lock_sink &&
        p.prefill_dense_staged_f32 && p.prefill_grouped_f32 && p.hc_rows_enabled && !p.prefill_hc_down_f32 && !p.moe_router_f32_batch,
        "kernel slice capture requires a fresh explicit exact carrier");
    p.kernel_search_sink = sink; p.kernel_search_context = context;
}
void MetalGraph::kernel_search_capture(KernelSearchCaptureRecord record, uint32_t bit) {
    auto& p = *impl_; if (!p.kernel_search_sink || (p.kernel_search_mask & bit)) return; p.active();
    require(bit && !(bit & (bit - 1)) && bit < 256 && !p.operator_cutting, "kernel capture record bit or owner");
    record.context = p.operator_tag; p.operator_cutting = true;
    try {
        if (p.dispatches) { const auto physical = submit(); p.operator_add(physical); p.operator_emit("kernel_search", "capture", {record.kind, record.encoding, record.dimensions[0], record.dimensions[1]}, false, &physical, p.operator_rollup); }
        else {
            const auto total = p.operator_rollup; const auto routes = p.operator_routes; const auto barriers = p.operator_barriers;
            const auto grouped_active = p.grouped_routes_active; const auto grouped_views = p.grouped_routes_views;
            abort(); p.operator_rollup = total; p.operator_routes = routes; p.operator_barriers = barriers; p.operator_cutting = true;
            p.grouped_routes_active = grouped_active; p.grouped_routes_views = grouped_views;
        }
        require(p.kernel_search_sink(p.kernel_search_context, *this, record), "kernel slice sink refused");
        p.kernel_search_mask |= bit; begin(); p.operator_cutting = false;
    } catch (...) { p.operator_cutting = false; try { abort(); } catch (const FormatError&) {} throw; }
}
#endif
void MetalGraph::begin() {
    @autoreleasepool {
    auto& p = *impl_;
    p.idle();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (p.carrier_diagnostic.active()) p.carrier_diagnostic.before_batch();
#endif
    if (p.planar_union_profile_enabled) {
        require(!p.planar_union_profile_binding, "planar union diagnostic metadata binding leaked");
        p.planar_union_profile_records.clear();
        if (p.planar_union_profile_active()) p.require_planar_union_profile_scope();
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    p.hc_rows_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    p.dense_staged_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    require(!p.grouped_routes_active
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        || p.operator_cutting
#endif
        , "active grouped route lease permits only an internal operator continuation");
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    p.operator_locked = true;
    if (p.operator_sink && !p.operator_cutting) { require(!p.operator_rollup.dispatches, "owner begin would discard operator rollup"); ++p.operator_owners; p.operator_barriers = 0; }
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    p.raw_decode_locked = true; p.raw_decode_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    p.q8_three_locked = true; p.q8_three_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    require(!p.task4_active || p.task4_batches < p.task4_max_batches, "Task4 lifetime batch bound exhausted");
    p.task4_tag = {}; p.task4_encoders = 0;
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    p.concurrent_locked = true; p.reset_spans(); p.concurrent_barrier_count = 0;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    p.carrier_flag_barriers = 0; p.carrier_records.clear();
    p.carrier_routes.clear(); p.carrier_route_bytes = 0; p.carrier_row_offset = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!p.mtp_profile.active || p.mtp_profile.next_batch_allowed(), "MTP diagnostic phase/batch bound refused before encoding");
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    p.lane_locked = true;
#endif
    p.planar_locked = true;
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    p.phase_clock.lock();
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    p.hc_exact_locked = true; p.hc_exact_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    p.wide_q8_exact_locked = true; p.wide_q8_exact_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    p.wide_q8_many_locked = true; p.wide_q8_many_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    p.qsa_stage_v8_locked = true; p.qsa_stage_v8_pending = 0;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    p.qsa_channel_locked = true; p.qsa_channel_pending = 0;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    p.qsa_sparse_locked = true; p.qsa_sparse_decode_pending = 0; p.qsa_sparse_verify_pending = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    p.qsa_rows_locked = true; p.qsa_rows_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    p.moe_router_f32_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    p.router_window_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    p.hc_down_window_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    p.gdn_ported_locked = true; p.gdn_prefill_ported_locked = true; p.gdn_ported_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    if (
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        !p.operator_cutting
#else
        true
#endif
    ) {
        p.nax_dense_bf16_valid = false;
        p.nax_gathered_broadcast_valid = false;
        p.nax_gathered_handoff_valid = false;
    }
#endif
    p.swiglu_fused_locked = true; p.swiglu_fused_pending = 0;
    p.q8_gemv_rg4_locked = true; p.q8_gemv_rg4_pending = {};
    p.q8_three_msg_locked = true; p.q8_three_msg_pending = {};
    p.moe_three_union_locked = true; p.moe_three_union_pending = 0;
    p.prefill_m64_locked = true; p.prefill_m64_pending = 0;
    p.prefill_multiseg_locked = true; p.prefill_multiseg_pending = 0;
    p.prefill_swizzle_locked = true;
    p.qsa_prefill_fused_locked = true;
    p.hc_consumer_fused_locked = true; p.hc_consumer_fused_pending = 0;
    p.verify_hc_gdn_fused_locked = true;
    *static_cast<uint32_t*>(p.error_flag.contents) = 0;
    p.command = [p.queue commandBuffer];
    if (!p.command) throw FormatError("cannot create owned graph command buffer");
    if (p.profile_mode != ProfileMode::encoder_boundary) {
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        p.encoder = p.concurrent ? [p.command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [p.command computeCommandEncoder];
#else
        p.encoder = [p.command computeCommandEncoder];
#endif
        if (!p.encoder) { p.command = nil; throw FormatError("cannot create owned graph encoder"); }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        ++p.task4_encoders;
#endif
    }
    p.dispatches = 0; p.capture_executes = 0; p.compatible_pipelines = false;
    p.marks.clear(); p.profile = {};
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    if (p.trace) {
        require(p.profile_mode == ProfileMode::disabled && !p.trace_pending, "dispatch trace requires direct unprofiled batches");
        p.trace_record({{"event", Json("begin")}, {"batch", Json::integer(p.trace_batch)},
            {"small_gemv_splits", Json::integer(p.small_gemv_splits)}, {"block_gemv_formats", Json::integer(p.block_gemv_formats)},
            {"hc_q8_splits", Json::integer(p.hc_q8.splits)}});
    }
#endif
    p.started = Clock::now();
    if (p.profiling_active()) {
        const auto before = Clock::now(); [p.device sampleTimestamps:&p.cpu_start gpuTimestamp:&p.gpu_start]; const auto after = Clock::now();
        p.calibration_start = before + (after - before) / 2;
    }
    }
}
void MetalGraph::begin_indirect_compatible() {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    p.moe_router_f32_locked = true; require(!p.moe_router_f32_batch,"router FP32 batch requires direct execution");
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    p.router_window_locked = true; require(!p.router_window_enabled, "router window requires direct execution");
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    p.hc_down_window_locked = true; require(!p.hc_down_window_enabled, "HC down window requires direct execution");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    p.gdn_ported_locked = true; p.gdn_prefill_ported_locked = true;
    require(!p.gdn_ported_configured && !p.gdn_prefill_ported_configured && !p.gdn_ported_variants_prepared, "ported GDN kernels require direct execution");
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    p.raw_decode_locked = true;
    require(!p.raw_decode, "raw decode selection requires direct batches");
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(p.lane_matrices.empty(), "packed weights do not admit indirect execution");
#endif
    require(p.planar_matrices.empty(), "planar v3 weights do not admit indirect execution");
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    p.qsa_channel_locked = true;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    p.qsa_sparse_locked = true;
    require(!p.qsa_sparse_configured, "private sparse QSA preparation refuses indirect comparison");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    p.qsa_rows_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    p.phase_clock.lock();
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    p.qsa_stage_v8_locked = true;
    require(!p.qsa_stage_v8.prepared, "private staged QSA preparation refuses indirect comparison");
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(p.hc_exact == HcExactMode::off, "private exact-HC refuses indirect comparison");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact, "private wide-Q8 refuses indirect comparison");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many, "private wide-Q8-many refuses indirect comparison");
#endif
    require(p.profile_mode == ProfileMode::disabled, "indirect comparator does not support profiling");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    require(!p.concurrent, "concurrent dispatch covers direct batches only");
#endif
    require(!p.hc_q8.splits, "HC Q8 split policy does not support indirect comparison");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    require(!p.concurrent, "concurrent dispatch covers direct batches only");
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    require(!p.trace, "dispatch trace covers direct batches only");
#endif
    p.swiglu_fused_locked = true; p.q8_gemv_rg4_locked = true; p.q8_three_msg_locked = true; p.moe_three_union_locked = true;
    p.prefill_m64_locked = true; p.prefill_multiseg_locked = true; p.prefill_swizzle_locked = true; p.qsa_prefill_fused_locked = true; p.hc_consumer_fused_locked = true; p.verify_hc_gdn_fused_locked = true;
    require(p.q8_gemv_rg4 == Q8GemvRg4Mode::off && p.q8_three_msg == Q8ThreeMsgMode::off && !p.moe_three_union &&
        p.prefill_m64 == PrefillM64Mode::off && !p.prefill_multiseg && !p.prefill_swizzle && !p.qsa_prefill_fused && !p.hc_consumer_fused && !p.verify_hc_gdn_fused,
        "multi-SG Q8 and unioned three-row MoE require direct batches");
    begin(); p.compatible_pipelines = true;
}
void MetalGraph::begin_capture(uint32_t capacity) {
    @autoreleasepool {
    auto& p = *impl_; p.idle();
    p.swiglu_fused_locked = true; p.q8_gemv_rg4_locked = true; p.q8_three_msg_locked = true; p.moe_three_union_locked = true;
    p.prefill_m64_locked = true; p.prefill_multiseg_locked = true; p.prefill_swizzle_locked = true; p.qsa_prefill_fused_locked = true; p.hc_consumer_fused_locked = true; p.verify_hc_gdn_fused_locked = true;
    require(p.q8_gemv_rg4 == Q8GemvRg4Mode::off && p.q8_three_msg == Q8ThreeMsgMode::off && !p.moe_three_union &&
        p.prefill_m64 == PrefillM64Mode::off && !p.prefill_multiseg && !p.prefill_swizzle && !p.qsa_prefill_fused && !p.hc_consumer_fused && !p.verify_hc_gdn_fused,
        "multi-SG Q8 and unioned three-row MoE refuse captured execution");
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    p.moe_router_f32_locked = true; require(!p.moe_router_f32_batch,"router FP32 batch refuses capture");
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    p.raw_decode_locked = true;
    require(!p.raw_decode, "raw decode selection refuses captured execution");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    p.gdn_ported_locked = true; p.gdn_prefill_ported_locked = true;
    require(!p.gdn_ported_configured && !p.gdn_prefill_ported_configured && !p.gdn_ported_variants_prepared, "ported GDN kernels refuse captured execution");
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    require(!p.concurrent, "concurrent dispatch covers direct batches only");
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    require(!p.trace, "dispatch trace covers direct batches only");
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(p.lane_matrices.empty(), "packed weights do not admit captured execution");
    p.lane_locked = true;
#endif
    require(p.planar_matrices.empty(), "planar v3 weights do not admit captured execution");
    p.planar_locked = true;
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    p.qsa_channel_locked = true;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    p.qsa_sparse_locked = true;
    require(!p.qsa_sparse_configured, "private sparse QSA preparation refuses indirect comparison");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    p.qsa_rows_locked = true;
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    p.phase_clock.lock();
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    p.qsa_stage_v8_locked = true;
    require(!p.qsa_stage_v8.prepared, "private staged QSA preparation refuses captures");
#endif
    require(!p.hc_q8.splits, "HC Q8 split policy does not support captures");
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(p.hc_exact == HcExactMode::off, "private exact-HC refuses captures");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact, "private wide-Q8 refuses captures");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many, "private wide-Q8-many refuses captures");
#endif
    require(p.profile_mode == ProfileMode::disabled && capacity && capacity <= 1024 && p.captures.size() < 1024,
        "unsupported capture profile/capacity/count");
    require([p.device supportsFamily:MTLGPUFamilyApple4],
        "selected device lacks admitted compute ICB family");
    const uint64_t constant_bytes = uint64_t(capacity) * Impl::capture_bindings * Impl::capture_constant_stride;
    const auto page = uint64_t(getpagesize());
    require(checked_add(constant_bytes, page - 1) / page * page <= p.ledger.budget - p.ledger.live_bytes,
        "capture constants exceed graph budget");
    Impl::Capture candidate;
    candidate.parameters = [p.device newBufferWithLength:constant_bytes options:MTLResourceStorageModeShared];
    require(candidate.parameters != nil, "cannot allocate persistent capture constants");
    MTLIndirectCommandBufferDescriptor* descriptor = [[MTLIndirectCommandBufferDescriptor alloc] init];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    descriptor.inheritPipelineState = NO; descriptor.inheritBuffers = NO;
    descriptor.maxKernelBufferBindCount = Impl::capture_bindings;
    candidate.commands = [p.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:capacity options:MTLResourceStorageModeShared];
    require(candidate.commands != nil, "device refused compute indirect command buffer");
    candidate.charged = checked_add(std::max(constant_bytes, uint64_t(candidate.parameters.allocatedSize)),
        std::max(uint64_t(1), uint64_t(candidate.commands.allocatedSize)));
    require(candidate.charged <= p.ledger.budget - p.ledger.live_bytes, "actual capture resources exceed graph budget");
    candidate.capacity = capacity; candidate.policy = {p.small_gemv_splits, p.block_gemv_formats, p.gdn_tile_threads};
    candidate.parameters.label = @"owned immutable captured constants";
    candidate.commands.label = @"owned static compute capture";
    p.draft = std::move(candidate);
    p.ledger.live_bytes += p.draft.charged; p.ledger.capture_resource_bytes += p.draft.charged;
    p.ledger.peak_bytes = std::max(p.ledger.peak_bytes, p.ledger.live_bytes); p.ledger.allocations += 2;
    p.recording = true; p.pending_capture_dispatch = false; p.dispatches = 0;
    }
}
GraphCapture MetalGraph::finish_capture() {
    auto& p = *impl_; p.thread_check(); require(p.recording, "no capture recording is active");
    try {
        require(p.draft.count && !p.pending_capture_dispatch, "empty or incomplete capture cannot publish");
        p.draft.reads.reserve(p.draft.leases.size() + p.draft.aliases.size() + 1); p.draft.writes.reserve(p.draft.leases.size() + 1);
        p.draft.reads.push_back(p.draft.parameters); p.draft.writes.push_back(p.error_flag);
        for (const auto& [key, alias] : p.draft.aliases) {
            static_cast<void>(key); p.draft.reads.push_back(alias);
        }
        for (const auto& [slot, writable] : p.draft.leases) {
            const auto& a = p.allocations.at(slot); require(a.buffer != nil, "capture binding was released");
            if (writable) p.draft.writes.push_back(a.buffer); else p.draft.reads.push_back(a.buffer);
        }
        const GraphCapture handle{p.owner, p.captures.size()};
        p.captures.push_back(std::move(p.draft));
        p.draft = {}; p.recording = false; p.pending_capture_dispatch = false; ++p.ledger.live_captures;
        p.dispatches = 0; return handle;
    } catch (...) { p.clear_draft(); p.dispatches = 0; throw; }
}
void MetalGraph::execute_capture(GraphCapture handle) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.profile_mode == ProfileMode::disabled,
        "capture replay requires an ordinary unprofiled serial batch");
    require(handle.owner == p.owner && handle.slot < p.captures.size(), "foreign or invalid capture handle");
    const auto& c = p.captures[handle.slot]; require(c.commands != nil, "released capture handle");
    require(c.policy == std::array{p.small_gemv_splits, p.block_gemv_formats, p.gdn_tile_threads}, "capture kernel policy changed");
    require(p.encoder != nil && p.encoder.dispatchType == MTLDispatchTypeSerial, "capture requires a serial parent encoder");
    [p.encoder useResources:c.reads.data() count:c.reads.size() usage:MTLResourceUsageRead];
    [p.encoder useResources:c.writes.data() count:c.writes.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [p.encoder executeCommandsInBuffer:c.commands withRange:NSMakeRange(0, c.count)];
    p.dispatches += c.count; ++p.capture_executes;
}
void MetalGraph::release_capture(GraphCapture handle) {
    auto& p = *impl_; p.idle();
    require(handle.owner == p.owner && handle.slot < p.captures.size(), "foreign or invalid capture handle");
    auto& c = p.captures[handle.slot]; require(c.commands != nil, "released capture handle");
    c.commands = nil; c.parameters = nil; c.reads.clear(); c.writes.clear(); c.leases.clear(); c.aliases.clear();
    p.ledger.live_bytes -= c.charged; p.ledger.capture_resource_bytes -= c.charged; --p.ledger.live_captures;
    c.charged = 0;
}
void MetalGraph::abort() {
    auto& p = *impl_;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    p.thread_check();
    if (p.carrier_diagnostic.active()) { p.carrier_diagnostic.invalidate(); carrier_cpu_ledger.active = false; }
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    p.thread_check(); p.operator_rollup = {}; p.operator_routes = {}; p.operator_barriers = 0; p.operator_cutting = false;
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    p.thread_check(); p.raw_decode_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    p.thread_check(); p.gdn_ported_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    p.thread_check(); p.q8_three_pending = {};
#endif
    p.thread_check();
    p.swiglu_fused_pending = 0; p.q8_gemv_rg4_pending = {}; p.q8_three_msg_pending = {}; p.moe_three_union_pending = 0;
    p.prefill_m64_pending = 0; p.prefill_multiseg_pending = 0; p.hc_consumer_fused_pending = 0;
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    p.thread_check(); p.qsa_rows_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    p.thread_check(); p.qsa_channel_pending = 0;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    p.qsa_sparse_decode_pending = 0; p.qsa_sparse_verify_pending = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    p.thread_check(); p.qsa_stage_v8_pending = 0;
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    p.thread_check(); p.wide_q8_many_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    p.thread_check(); p.wide_q8_exact_pending = {};
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    p.thread_check(); // Refusal on another thread must not mutate the owner's draft.
    p.hc_exact_pending = {};
#endif
    p.active();
    if (p.planar_union_profile_enabled) {
        require(!p.planar_union_profile_binding, "planar union diagnostic abort inside an operator");
        p.planar_union_profile_records.clear();
    }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    p.grouped_routes_active = false; p.grouped_routes_views = {};
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    p.nax_dense_bf16_valid = false;
    p.nax_gathered_broadcast_valid = false;
    p.nax_gathered_handoff_valid = false;
#endif
    if (p.recording) { p.clear_draft(); p.dispatches = 0; return; }
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    p.reset_spans(); p.concurrent_barrier_count = 0;
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    if (p.trace) { p.trace_pending = false; p.trace_record({{"event", Json("abort")}, {"batch", Json::integer(p.trace_batch++)}}); }
#endif
    if (p.encoder) [p.encoder endEncoding];
    p.encoder = nil;
    p.command = nil;
    p.dispatches = 0;
    p.compatible_pipelines = false; p.capture_executes = 0;
    p.marks.clear(); p.profile = {};
}
BatchResult MetalGraph::submit() {
    @autoreleasepool {
    auto& p = *impl_;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    p.active();
    require(!p.grouped_routes_active
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        || p.operator_cutting
#endif
        , "end grouped route scope before owner submit");
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    p.thread_check();
    struct ClearDecodeRoutes { RawDecodeRoutes& pending; ~ClearDecodeRoutes() { pending = {}; } } clear_decode{p.raw_decode_pending};
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    p.thread_check();
    struct ClearGdnPortedRoutes { GdnPortedRoutes& pending; ~ClearGdnPortedRoutes() { pending = {}; } } clear_gdn_ported{p.gdn_ported_pending};
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    // OFF/absent take no additional clock samples. The ordinary wall start and
    // end remain shared by both builds and preserve their original scope.
    p.thread_check();
    const bool trace_phases = p.phase_clock.enabled
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        && p.carrier_diagnostic.active()
#endif
        ;
    std::array<Clock::time_point, 8> phase_points{};
    if (trace_phases) { phase_points[0] = p.started; phase_points[1] = Clock::now(); }
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    p.thread_check();
    struct ClearStageCalls { uint64_t& pending; ~ClearStageCalls() { pending = 0; } } clear_stage_calls{p.qsa_stage_v8_pending};
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    p.thread_check();
    struct ClearChannelCalls { uint64_t& pending; ~ClearChannelCalls() { pending = 0; } } clear_channel_calls{p.qsa_channel_pending};
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    struct ClearSparseCalls { uint64_t &decode, &verify; ~ClearSparseCalls() { decode = 0; verify = 0; } } clear_sparse_calls{p.qsa_sparse_decode_pending, p.qsa_sparse_verify_pending};
#endif
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    p.thread_check();
    struct ClearQsaRows { QsaVerifyRowsRoutes& pending; ~ClearQsaRows() { pending = {}; } } clear_qsa_rows{p.qsa_rows_pending};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    p.thread_check();
    struct ClearManyRoutes { WideQ8ManyRoutes& pending; ~ClearManyRoutes() { pending = {}; } } clear_many_routes{p.wide_q8_many_pending};
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    p.thread_check();
    struct ClearWideRoutes { WideQ8ExactRoutes& pending; ~ClearWideRoutes() { pending = {}; } } clear_wide_routes{p.wide_q8_exact_pending};
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    p.thread_check(); // Before installing the draft-clearing exception guard.
    // The command may already be idle when wait/error/profile collection throws.
    // Always discard unpublished drafts; only the final no-throw loop publishes.
    struct ClearRoutes { HcExactRoutes& pending; ~ClearRoutes() { pending = {}; } } clear_routes{p.hc_exact_pending};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    p.thread_check();
    struct ClearThreeRoutes { Q8ThreeRoutes& pending; ~ClearThreeRoutes() { pending = {}; } } clear_three{p.q8_three_pending};
#endif
    p.thread_check();
    struct ClearExactCandidates {
        uint64_t& swiglu; Q8GemvRg4Routes& rg4; Q8ThreeMsgRoutes& msg; uint64_t& u3;
        uint64_t& m64; uint64_t& mseg; uint64_t& hcf;
        ~ClearExactCandidates() { swiglu = 0; rg4 = {}; msg = {}; u3 = 0; m64 = 0; mseg = 0; hcf = 0; }
    } clear_exact_candidates{p.swiglu_fused_pending, p.q8_gemv_rg4_pending, p.q8_three_msg_pending, p.moe_three_union_pending,
        p.prefill_m64_pending, p.prefill_multiseg_pending, p.hc_consumer_fused_pending};
    p.active();
    require(!p.recording, "capture preparation cannot submit GPU work");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (p.operator_sink && !p.operator_cutting) {
        require(p.operator_segments < p.operator_limit, "operator checkpoint record bound exhausted before owner submission");
        if (!p.dispatches && p.operator_rollup.dispatches) {
            const auto total = p.operator_rollup; const auto routes = p.operator_routes; const auto barriers = p.operator_barriers; abort();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
            p.completed_concurrent_barriers = barriers;
#else
            static_cast<void>(barriers);
#endif
            p.operator_emit("owner", "empty_final", {}, true, nullptr, total); p.operator_publish_routes(routes); return total;
        }
    }
#endif
    require(p.dispatches > 0, "cannot submit an empty graph batch");
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    if (trace_phases) phase_points[2] = Clock::now();
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    const bool mtp_diagnostic = p.mtp_profile.active;
    const auto encode_end = mtp_diagnostic ? Clock::now() : Clock::time_point{};
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    const bool task4_diagnostic = p.task4_sink && p.task4_active;
    const auto task4_encode_end = task4_diagnostic ? Clock::now() : Clock::time_point{};
#endif
    if (p.encoder) [p.encoder endEncoding];
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    if (
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        !p.operator_cutting
#else
        true
#endif
    ) {
        p.nax_dense_bf16_valid = false;
        p.nax_gathered_broadcast_valid = false;
        p.nax_gathered_handoff_valid = false;
    }
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    if (trace_phases) phase_points[3] = Clock::now();
#endif
    p.encoder = nil;
    id<MTLCommandBuffer> command = p.command;
    p.command = nil;
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    if (trace_phases) phase_points[4] = Clock::now();
#endif
    { TOKSPUR_CARRIER_PHASE(submission); [command commit]; }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    const auto task4_submit_end = task4_diagnostic ? Clock::now() : Clock::time_point{};
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    const auto submit_end = mtp_diagnostic ? Clock::now() : Clock::time_point{};
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    if (trace_phases) phase_points[5] = Clock::now();
#endif
    { TOKSPUR_CARRIER_PHASE(waiting); [command waitUntilCompleted]; }
    require(command.status == MTLCommandBufferStatusCompleted, "GPU graph command failed");
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    const auto task4_wait_end = task4_diagnostic ? Clock::now() : Clock::time_point{};
    const auto task4_ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const double task4_encode_ms = task4_diagnostic ? task4_ms(p.started, task4_encode_end) : 0;
    const double task4_submit_ms = task4_diagnostic ? task4_ms(task4_encode_end, task4_submit_end) : 0;
    const double task4_wait_ms = task4_diagnostic ? task4_ms(task4_submit_end, task4_wait_end) : 0;
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    const auto wait_end = mtp_diagnostic ? Clock::now() : Clock::time_point{};
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    if (trace_phases) phase_points[6] = Clock::now();
#endif
    const auto err_val = *static_cast<const uint32_t*>(p.error_flag.contents);
    require(err_val == 0, "GPU graph produced nonfinite values or invalid gates");
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    const auto wall_end = Clock::now();
    const BatchResult result{p.dispatches, std::chrono::duration<double, std::milli>(wall_end - p.started).count(), (command.GPUEndTime - command.GPUStartTime) * 1000, p.capture_executes};
#else
    const BatchResult result{p.dispatches, std::chrono::duration<double, std::milli>(Clock::now() - p.started).count(), (command.GPUEndTime - command.GPUStartTime) * 1000, p.capture_executes};
#endif
    p.collect_profile();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (p.carrier_diagnostic.active()) {
        require(command.status == MTLCommandBufferStatusCompleted, "carrier diagnostic GPU command failed");
        p.carrier_diagnostic.batch(result.dispatches, p.carrier_records.size());
        require(std::isfinite(result.gpu_ms) && result.gpu_ms > 0 && std::isfinite(result.wall_ms) && result.wall_ms > 0, "invalid carrier batch clock");
        Json::Array kernels;
        if (p.carrier_diagnostic.mode() == CarrierDiagnosticMode::dispatch) {
            require(p.profile.timings_valid && p.profile.kernels.size() == p.marks.size(),
                "carrier dispatch clocks are invalid or incomplete");
            for (size_t i = 0; i < p.profile.kernels.size(); ++i) {
                const auto& k = p.profile.kernels[i];
                Json::Array shape; for (const auto d : p.marks[i].shape) shape.push_back(Json::integer(d));
                kernels.push_back(Json(Json::Object{{"kernel", Json(k.kernel)}, {"gpu_ms", Json::number(k.milliseconds)},
                    {"shape", Json(std::move(shape))}, {"selected_or_splits", Json::integer(p.marks[i].carrier_selected)}, {"encoding", Json::integer(p.marks[i].encoding)}, {"layer", Json::integer(p.marks[i].carrier_layer)}}));
            }
        }
        if (p.carrier_diagnostic.mode() != CarrierDiagnosticMode::dispatch && p.carrier_diagnostic.records_dispatches()) {
            require(p.carrier_records.size() == result.dispatches, "carrier concurrent dispatch ledger differs");
            for (const auto& mark : p.carrier_records) {
                Json::Array shape; for (auto n : mark.shape) shape.push_back(Json::integer(n));
                kernels.push_back(Json(Json::Object{{"kernel", Json(mark.kernel)}, {"gpu_ms", Json()},
                    {"shape", Json(std::move(shape))}, {"encoding", Json::integer(mark.encoding)}, {"selected_or_splits", Json::integer(mark.carrier_selected)}, {"layer", Json::integer(mark.carrier_layer)}}));
            }
        }
        const auto record = Json(Json::Object{{"event", Json("carrier_diagnostic_batch")}, {"schema", Json::integer(1)},
            {"action", Json::integer(p.carrier_diagnostic.actions())}, {"batch", Json::integer(p.carrier_diagnostic.batches())},
            {"component", Json(p.carrier_diagnostic.component())}, {"dispatches", Json::integer(result.dispatches)}, {"wall_ms", Json::number(result.wall_ms)},
            {"gpu_ms", Json::number(result.gpu_ms)}, {"barriers", Json::integer(p.concurrent_barrier_count)},
            {"error_flag_barriers", Json::integer(p.carrier_flag_barriers)},
            {"dispatch_records_enabled", Json(p.carrier_diagnostic.records_dispatches())},
            {"dispatch_records", Json::integer(p.carrier_records.size())},
            {"serial_counter_schedule", Json(p.carrier_diagnostic.mode() == CarrierDiagnosticMode::dispatch)},
            {"kernels", Json(std::move(kernels))}, {"timings_qualified", Json(false)}}).dump() + "\n";
        require(std::fwrite(record.data(), 1, record.size(), stderr) == record.size() && std::fflush(stderr) == 0,
            "carrier diagnostic record write failed");
        if (!p.carrier_routes.empty()) {
            const auto& snapshot = p.allocation(p.carrier_route_scratch);
            for (const auto& route : p.carrier_routes) {
                const auto* values = reinterpret_cast<const uint32_t*>(static_cast<const std::byte*>(snapshot.buffer.contents) + route.offset);
                const uint32_t count = route.rows * route.selected;
                Json::Array ids, weight_bits;
                for (uint32_t i = 0; i < count; ++i) {
                    require(values[i] < route.experts, "diagnostic route index outside expert table");
                    ids.push_back(Json::integer(values[i])); weight_bits.push_back(Json::integer(values[count + i]));
                }
                const auto captured = Json(Json::Object{{"event", Json("carrier_diagnostic_routes")}, {"action", Json::integer(p.carrier_diagnostic.actions())},
                    {"batch", Json::integer(p.carrier_diagnostic.batches())}, {"layer", Json::integer(route.layer)}, {"rows", Json::integer(route.rows)},
                    {"row_begin", Json::integer(route.row_begin)}, {"selected", Json::integer(route.selected)}, {"ids", Json(std::move(ids))}, {"weight_bits", Json(std::move(weight_bits))}}).dump() + "\n";
                require(std::fwrite(captured.data(), 1, captured.size(), stderr) == captured.size() && std::fflush(stderr) == 0, "diagnostic route capture write failed");
            }
        }
    }
#endif
    if (p.planar_union_profile_enabled && !p.planar_union_profile_records.empty()) {
        require(command.status == MTLCommandBufferStatusCompleted, "planar union diagnostic GPU batch did not complete");
        p.dump_planar_union_profile();
    }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    if (task4_diagnostic) {
        const auto expected = p.task4_boundary == Task4ProfileBoundary::dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
        require(p.task4_batches < p.task4_max_batches && p.profile.mode == expected && p.profile.kernels.size() == p.dispatches &&
            p.task4_encoders == (expected == ProfileMode::encoder_boundary ? p.dispatches : 1), "Task4 batch/profile bookkeeping failed");
        const Task4ProfileBatch scope{p.task4_batches++, task4_encode_ms, task4_submit_ms, task4_wait_ms, p.task4_encoders};
        require(p.task4_sink(p.task4_context, scope, p.profile, result), "Task4 sink failed; diagnostic request invalidated");
    }
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    p.completed_concurrent_barriers = p.concurrent_barrier_count;
    p.lifetime_dispatch_count = checked_add(p.lifetime_dispatch_count, result.dispatches);
    p.lifetime_command_count = checked_add(p.lifetime_command_count, 1);
    p.lifetime_barrier_count = checked_add(p.lifetime_barrier_count, p.concurrent_barrier_count);
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    if (p.trace) {
        p.trace_record({{"event", Json("submit")}, {"batch", Json::integer(p.trace_batch++)}, {"dispatches", Json::integer(p.dispatches)}});
        for (size_t slot = 0; slot < p.allocations.size(); ++slot) {
            const auto& item = p.allocations[slot];
            if (item.buffer && !item.immutable)
                p.trace_record({{"event", Json("state")}, {"slot", Json::integer(slot)},
                    {"sha256", Json(Impl::trace_digest(item.buffer.contents, item.requested))}});
        }
    }
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    if (mtp_diagnostic) {
        p.mtp_profile.submitted();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const MtpProfileBatch scope{p.mtp_profile.rounds, p.mtp_profile.base, p.mtp_profile.round_batches,
            p.mtp_profile.phase, ms(p.started, encode_end), ms(encode_end, submit_end), ms(submit_end, wait_end)};
        require(p.mtp_profile_sink(p.mtp_profile_context, scope, p.profile, result), "MTP diagnostic sink failed; request invalidated");
    }
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (p.operator_sink) p.operator_collect_routes();
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < 5; ++i) p.hc_exact_completed[i] += p.hc_exact_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < 4; ++i) p.wide_q8_exact_completed[i] += p.wide_q8_exact_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < 6; ++i) p.wide_q8_many_completed[i] += p.wide_q8_many_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    p.qsa_stage_v8_completed += p.qsa_stage_v8_pending;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    p.qsa_channel_completed += p.qsa_channel_pending;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    {
        p.qsa_sparse_decode_completed += p.qsa_sparse_decode_pending;
        p.qsa_sparse_verify_completed += p.qsa_sparse_verify_pending;
    }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < p.qsa_rows_completed.size(); ++i) p.qsa_rows_completed[i] += p.qsa_rows_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    const auto batch = p.phase_clock.completed();
    if (trace_phases) {
        phase_points[7] = wall_end;
        BatchPhaseClockSample sample; sample.batch = batch; sample.dispatches = result.dispatches;
        sample.capture_executes = result.capture_executes; sample.reported_wall_ms = result.wall_ms; sample.reported_gpu_ms = result.gpu_ms;
        for (size_t i = 0; i < phase_points.size(); ++i) sample.cpu_ns[i] = phase_nanoseconds(phase_points[i]);
        const bool valid = batch_phase_clock_intervals(sample).valid;
        p.phase_clock.emitted(emit_phase_record(sample), valid);
    }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < p.q8_three_completed.size(); ++i) p.q8_three_completed[i] += p.q8_three_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < p.raw_decode_completed.size(); ++i) p.raw_decode_completed[i] += p.raw_decode_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    for (size_t i = 0; i < p.gdn_ported_completed.size(); ++i) p.gdn_ported_completed[i] += p.gdn_ported_pending[i];
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (!p.operator_sink)
#endif
    {
        p.swiglu_fused_completed += p.swiglu_fused_pending;
        for (size_t i = 0; i < p.q8_gemv_rg4_completed.size(); ++i) p.q8_gemv_rg4_completed[i] += p.q8_gemv_rg4_pending[i];
        for (size_t i = 0; i < p.q8_three_msg_completed.size(); ++i) p.q8_three_msg_completed[i] += p.q8_three_msg_pending[i];
        p.moe_three_union_completed += p.moe_three_union_pending;
        p.prefill_m64_completed += p.prefill_m64_pending;
        p.prefill_multiseg_completed += p.prefill_multiseg_pending;
        p.hc_consumer_fused_completed += p.hc_consumer_fused_pending;
    }
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (p.operator_sink && !p.operator_cutting) {
        p.operator_add(result); const auto total = p.operator_rollup; const auto routes = p.operator_routes; const auto barriers = p.operator_barriers; p.operator_rollup = {}; p.operator_routes = {}; p.operator_barriers = 0;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        p.completed_concurrent_barriers = barriers;
#else
        static_cast<void>(barriers);
#endif
        p.operator_emit("owner", "unclassified_tail", {}, true, &result, total); p.operator_publish_routes(routes); return total;
    }
#endif
    return result;
    }
}

void MetalGraph::gemv(uint32_t type, GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows) {
    auto& p = *impl_;
    p.active();
    require(can_decode(type) && columns && columns <= 65536 && rows && rows <= 1'000'000, "unsupported graph GEMV shape/type");
    const auto& layout = encoding(type);
    const auto row_bytes = checked_mul(columns / layout.elements_per_block, layout.bytes_per_block);
    require(columns % layout.elements_per_block == 0 && checked_mul(row_bytes, rows) == matrix.bytes &&
            p.floats(input) == columns && p.floats(output, true) == rows, "graph GEMV buffer shape mismatch");
    static_cast<void>(p.allocation(matrix));
    separate(output, {matrix, input});
    const auto rg4_eligible = [&](bool packed) {
        if (p.q8_gemv_rg4 == Q8GemvRg4Mode::off || p.recording || p.compatible_pipelines) return false;
        const uint32_t min_rows = p.q8_gemv_rg4 == Q8GemvRg4Mode::on ? 16384u : 4096u;
        if (rows < min_rows || columns < 512u || columns % 32u != 0u) return false;
        return !packed || columns % 256u == 0u;
    };
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (const auto* lane = p.lane_shape(matrix)) {
        require(lane->encoding == type && type == 8 && lane->columns == columns && lane->rows == rows && lane->experts == 1,
            "lane dense consumer shape mismatch");
        if (rg4_eligible(true)) {
            require(p.q8_gemv_rg4_pending[1] < UINT64_MAX - p.q8_gemv_rg4_completed[1], "multi-SG lane Q8 GEMV route overflow");
            p.select("dense_lanes_q8_rg4", uint32_t(-1), {columns, rows, 0, 0}, 256);
            p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
            p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), 1}, 3);
            p.dispatch(true, MTLSizeMake((rows + 31) / 32, 1, 1), MTLSizeMake(256, 1, 1));
            ++p.q8_gemv_rg4_pending[1]; return;
        }
        p.select("dense_lanes_q8", uint32_t(-1), {columns, rows, 0, 0});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), 1}, 3);
        p.groups(MTLSizeMake((rows + 3) / 4, 1, 1)); return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    const int wide_shape = wide_q8_exact_shape(columns, rows);
    if (p.wide_q8_exact && type == 8 && wide_shape >= 0) {
        p.exact_wide_q8(matrix, input, output, size_t(wide_shape), false); return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    if (p.hc_exact != HcExactMode::off && type == 8 && columns == 10240 && rows == 320) {
        p.exact_hc(matrix, input, output, 0); return;
    }
#endif
    if (p.hc_q8.splits && type == 8 && columns == 10240 && rows == 320) {
        const auto bytes = q8_split_bytes({columns, rows, p.hc_q8.splits});
        q8_matvec_split(matrix, input, output, slice(p.hc_q8_scratch, 0, bytes.partials), columns, rows, p.hc_q8.splits);
        return;
    }
    if (p.small_gemv_splits == 1 && type == 0 && rows <= 64 && columns >= 1024) {
        p.select("graph_small_gemv_direct", uint32_t(-1), {columns, rows, 1, 0});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, 0, 0}, 3); p.groups(MTLSizeMake((rows + 3) / 4, 1, 1));
        return;
    }
    if (gemv_dispatches(type, columns, rows) == 2) {
        separate(p.small_gemv_scratch, {matrix, input, output});
        p.select("graph_small_gemv_partials", uint32_t(-1), {columns, rows, p.small_gemv_splits, 0});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(p.small_gemv_scratch, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, p.small_gemv_splits, 0}, 3);
        p.groups(MTLSizeMake((rows + 3) / 4, p.small_gemv_splits, 1));
        p.select("graph_small_gemv_reduce", uint32_t(-1), {columns, rows, p.small_gemv_splits, 0});
        p.bind(p.small_gemv_scratch, 0); p.bind(output, 1, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, p.small_gemv_splits, 0}, 2); p.threads(rows);
        return;
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
    if (p.native_plain_loads && (type == 0 || type == 30)) {
        p.select("graph_native_plain_gemv", type, {columns, rows, 0, 0});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), 0}, 3);
        p.groups(MTLSizeMake((rows + 3) / 4, 1, 1)); return;
    }
#endif
    const bool block = (type == 8 && (p.block_gemv_formats & 1)) || (type == 12 && (p.block_gemv_formats & 2));
    if (type == 8 && (p.block_gemv_formats & 1) && rg4_eligible(false)) {
        require(p.q8_gemv_rg4_pending[0] < UINT64_MAX - p.q8_gemv_rg4_completed[0], "multi-SG Q8 GEMV route overflow");
        p.select("graph_q8_gemv_rg4", uint32_t(-1), {columns, rows, 0, 0}, 256);
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), 0}, 3);
        p.dispatch(true, MTLSizeMake((rows + 31) / 32, 1, 1), MTLSizeMake(256, 1, 1));
        ++p.q8_gemv_rg4_pending[0]; return;
    }
    p.select(block ? "graph_block_gemv" : "graph_quantized_gemv", type, {columns, rows, 0, 0});
    p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), 0}, 3);
    p.groups(MTLSizeMake((rows + 3) / 4, 1, 1));
}
void MetalGraph::configure_q8_three_rows(bool enabled) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    require(!p.q8_three_locked && p.q8_many_prepared.empty() && !p.ledger.live_captures,
        "three-row Q8 is fixed before many-width preparation and the first batch");
    p.q8_three_rows = enabled;
#else
    require(!enabled, "three-row Q8 requires the private build");
#endif
}
bool MetalGraph::q8_three_rows_enabled() const {
    impl_->thread_check();
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    return impl_->q8_three_rows;
#else
    return false;
#endif
}
MetalGraph::Q8ThreeRoutes MetalGraph::q8_three_rows_routes() const {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    return impl_->q8_three_completed;
#else
    return {};
#endif
}
uint32_t MetalGraph::q8_many_width(uint32_t remaining) const { return tokspur::q8_many_width(remaining, q8_three_rows_enabled()); }
uint32_t MetalGraph::q8_many_chunks(uint32_t rows) const { return tokspur::q8_many_chunks(rows, q8_three_rows_enabled()); }
void MetalGraph::prepare_q8_matvec_many_all() {
    for (const uint32_t width : {1, 2, 4, 8}) prepare_q8_matvec_many(width);
    if (q8_three_rows_enabled()) prepare_q8_matvec_many(3);
}
void MetalGraph::prepare_q8_matvec_many(uint32_t width) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    if (width == 3) {
        require(p.q8_three_rows, "three-row Q8 selector is off");
        static_cast<void>(p.pipeline(q8_three_rows_kernel(Q8ThreeLayout::gguf)));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        static_cast<void>(p.pipeline(q8_three_rows_kernel(Q8ThreeLayout::q8_lanes_v1)));
#endif
        p.q8_many_prepared.insert(3); return;
    }
#endif
    const auto* name = q8_many_kernel(width);
    static_cast<void>(p.pipeline(name)); p.q8_many_prepared.insert(width);
}
void MetalGraph::q8_matvec_many(GpuView matrix, GpuView input, GpuView output,
                               uint32_t columns, uint32_t rows, uint32_t width) {
    auto& p = *impl_; p.active(); const auto shape = q8_many_bytes({columns, rows, width});
    require(matrix.bytes == shape.matrix && uint64_t(p.floats(input)) * 4 == shape.input &&
        uint64_t(p.floats(output, true)) * 4 == shape.output, "Q8-many GPU buffer shape mismatch");
    static_cast<void>(p.allocation(matrix)); separate(output, {matrix, input});
    require(p.q8_many_prepared.contains(width),"Q8-many pipeline must be prepared while idle before encoding");
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    if (width == 3) {
        require(p.q8_three_rows && !p.recording && !p.compatible_pipelines, "three-row Q8 requires selected direct batches");
        auto layout = Q8ThreeLayout::gguf; size_t bucket = 0;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (const auto* lane = p.lane_shape(matrix)) {
            require(lane->encoding == 8 && lane->columns == columns && lane->rows == rows && lane->experts == 1,
                "three-row Q8 registered lane identity mismatch");
            layout = Q8ThreeLayout::q8_lanes_v1; bucket = 1;
        }
#endif
        if (layout == Q8ThreeLayout::gguf) {
#ifdef TOKSPUR_PRIVATE_HC_EXACT
            if (p.hc_exact == HcExactMode::combined && columns == 10240 && rows == 320) bucket = 2;
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
            if (p.wide_q8_many) {
                const int wide = wide_q8_exact_shape(columns, rows);
                if (wide >= 0) bucket = 3 + size_t(wide);
            }
#endif
        }
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        if (bucket == 2) require(p.fixed_route_profile_allowed(), "three-row raw HC requires its selected profile policy");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        if (bucket >= 3) require(p.profile_mode == ProfileMode::disabled, "three-row raw wide requires unprofiled execution");
#endif
        const auto plan = q8_three_rows_plan(layout, columns, rows, uint64_t(columns) * 4, uint64_t(rows) * 4);
        require(matrix.offset % (layout == Q8ThreeLayout::gguf ? 4 : 16) == 0,
            "three-row Q8 matrix base alignment");
        require(p.q8_three_pending[bucket] < UINT64_MAX - p.q8_three_completed[bucket], "three-row Q8 route counter overflow");
        const uint32_t msg_min_rows = p.q8_three_msg == Q8ThreeMsgMode::on ? 16384u : 4096u;
        const bool use_msg = p.q8_three_msg != Q8ThreeMsgMode::off && bucket <= 1 &&
            columns >= 512u && columns % 32u == 0u &&
            (layout != Q8ThreeLayout::q8_lanes_v1 || columns % 256u == 0u) && rows >= msg_min_rows;
        if (use_msg) {
            require(p.q8_three_msg_pending[bucket] < UINT64_MAX - p.q8_three_msg_completed[bucket], "multi-SG three-row Q8 route overflow");
            p.select(q8_three_rows_msg_kernel(layout), uint32_t(-1), {columns, rows, 3, uint32_t(layout)}, 256);
            p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true); p.parameter(plan.args, 3);
            p.dispatch(true, MTLSizeMake((rows + 31) / 32, 1, 1), MTLSizeMake(256, 1, 1));
            ++p.q8_three_pending[bucket]; ++p.q8_three_msg_pending[bucket]; return;
        }
        p.select(plan.kernel, uint32_t(-1), {columns, rows, 3, uint32_t(layout)});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true); p.parameter(plan.args, 3);
        p.groups(MTLSizeMake(plan.groups, 1, 1)); ++p.q8_three_pending[bucket]; return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (const auto* lane = p.lane_shape(matrix)) {
        require(lane->encoding == 8 && lane->columns == columns && lane->rows == rows && lane->experts == 1,
            "lane many-row consumer shape mismatch");
        p.select(lane_many_kernel(width), uint32_t(-1), {columns, rows, width, 0});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, columns / 32 * 34, width}, 3);
        p.groups(MTLSizeMake((rows + 3) / 4, 1, 1)); return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    const int many_shape = wide_q8_exact_shape(columns, rows);
    if (p.wide_q8_many && width != 1 && many_shape >= 0) {
        p.exact_wide_q8_many(matrix, input, output, size_t(many_shape), width); return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    const int wide_shape = wide_q8_exact_shape(columns, rows);
    if (p.wide_q8_exact && width == 1 && wide_shape >= 0) {
        p.exact_wide_q8(matrix, input, output, size_t(wide_shape), true); return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    if (p.hc_exact == HcExactMode::combined && columns == 10240 && rows == 320) {
        const size_t bucket = width == 1 ? 1 : width == 2 ? 2 : width == 4 ? 3 : 4;
        p.exact_hc(matrix, input, output, bucket); return;
    }
#endif
    const char* name = q8_many_kernel(width);
    p.select(name, uint32_t(-1), {columns, rows, width, 0});
    p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{columns, rows, columns / 32 * 34, width}, 3);
    p.groups(MTLSizeMake((rows + 3) / 4, 1, 1));
}
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
PrefillWeightLayout MetalGraph::prefill_weight_layout(GpuView matrix) const {
    const auto& p = *impl_; static_cast<void>(p.allocation(matrix));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (const auto* shape = p.lane_shape(matrix)) {
        if (shape->encoding == 8) return PrefillWeightLayout::q8_lanes_v1;
        if (shape->encoding == 7) return PrefillWeightLayout::q51_lanes_v1;
        throw FormatError("unsupported registered prefill weight encoding");
    }
#endif
    return PrefillWeightLayout::gguf;
}
void MetalGraph::prepare_grouped_mm(uint32_t type, PrefillWeightLayout layout) {
    auto& p = *impl_; p.idle();
    require([p.device supportsFamily:MTLGPUFamilyMetal4], "grouped prefill requires Metal 4 support");
#ifndef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(layout == PrefillWeightLayout::gguf, "packed prefill requires the lane-layout build");
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (p.prefill_grouped_nax) {
        require(layout == PrefillWeightLayout::gguf ||
            (layout == PrefillWeightLayout::q51_lanes_v1 && type == 7) ||
            (layout == PrefillWeightLayout::q8_lanes_v1 && type == 8),
            "NAX grouped prefill requires GGUF, Q5_1 lanes or Q8_0 lanes layout");
        static_cast<void>(grouped_mm_kernel(type, layout));
        static_cast<void>(p.pipeline("graph_grouped_routes"));
        static_cast<void>(p.pipeline("nax_grouped_routes", uint32_t(-1), 256));
        static_cast<void>(p.pipeline("nax_grouped_tasks_indirect"));
        static_cast<void>(p.pipeline("nax_moe_gather_bf16"));
        const auto pipeline = p.pipeline("nax_grouped_moe", type, 128, false, uint32_t(layout));
        const auto gu_pipe = p.pipeline("nax_grouped_moe_gate_up_swiglu", type, 128, false, uint32_t(layout));
        require(pipeline.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            gu_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "NAX grouped prefill threadgroup memory exceeds device capability");
        if (p.nax_bf16_scratch.bytes < 335'544'320ull) {
            if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
            p.nax_bf16_scratch = zeros(335'544'320ull, "nax_bf16_scratch");
        }
        if (p.nax_indirect_args.bytes < 96ull) {
            if (p.nax_indirect_args.bytes) release(p.nax_indirect_args);
            p.nax_indirect_args = zeros(96, "nax_indirect_args");
        }
        p.grouped_mm_prepared.insert({type, uint32_t(layout)});
        return;
    }
    if (p.prefill_grouped_f32 || p.prefill_grouped_auto) {
        require(layout == PrefillWeightLayout::gguf || (layout == PrefillWeightLayout::q51_lanes_v1 && type == 7),
            "FP32 grouped prefill requires original GGUF or Q5_1 lanes layout");
        static_cast<void>(grouped_mm_kernel(type, layout));
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
        if (p.kernel_grouped_selected) {
            static_cast<void>(p.pipeline(grouped_mm_f32_kernel, type, 128, false, uint32_t(layout)));
            static_cast<void>(p.pipeline("graph_grouped_search_fp16", type, 128, false, uint32_t(layout)));
        }
#endif
        const auto pipeline = p.pipeline(
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
            p.grouped_step2_kernel(grouped_mm_f32_kernel),
#else
            grouped_mm_f32_kernel,
#endif
            type, 128, false, uint32_t(layout));
        require(pipeline.staticThreadgroupMemoryLength + grouped_mm_f32_threadgroup_bytes <= p.device.maxThreadgroupMemoryLength,
            "FP32 grouped prefill threadgroup memory exceeds device capability");
        static_cast<void>(p.pipeline("graph_grouped_routes"));
        static_cast<void>(p.pipeline("graph_grouped_tasks"));
        if (!p.prefill_grouped_auto) {
            p.grouped_mm_prepared.insert({type, uint32_t(layout)}); return;
        }
    }
#endif
    static_cast<void>(p.pipeline("graph_grouped_routes"));
    static_cast<void>(p.pipeline("graph_grouped_tasks"));
    if (layout == PrefillWeightLayout::gguf) static_cast<void>(p.pipeline("graph_grouped_serial", type, 128, false, uint32_t(layout)));
    const auto pipeline = p.pipeline(grouped_mm_kernel(type, layout), type, 128, false, uint32_t(layout));
    require(pipeline.staticThreadgroupMemoryLength + 8192 <= p.device.maxThreadgroupMemoryLength, "grouped prefill threadgroup memory exceeds device capability");
    p.grouped_mm_prepared.insert({type, uint32_t(layout)});
}
void MetalGraph::grouped_mm(uint32_t type, PrefillWeightLayout layout, GpuView matrices, GpuView indices,
    GpuRowsF32 input, GpuRowsF32 output, GpuView token_map, GpuView counts, GpuView tasks,
    uint32_t experts, uint32_t selected, bool broadcast) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.grouped_mm_prepared.contains({type, uint32_t(layout)}), "grouped prefill requires idle preparation and direct batches");
    require(selected && output.rows % selected == 0, "grouped prefill route-row count mismatch");
    const auto tokens = output.rows / selected;
    require(input.rows == (broadcast ? tokens : output.rows), "grouped prefill input-row count mismatch");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (p.prefill_grouped_nax) {
        const auto scope = grouped_mm_routes(indices, token_map, counts, tasks, experts, selected, tokens);
        grouped_mm_project(type, layout, matrices, scope, input, output, broadcast);
        end_grouped_mm_routes(scope);
        return;
    }
#endif
    const auto plan =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        p.prefill_grouped_f32 ? grouped_mm_f32_plan(type, layout, input.columns, output.columns, experts, selected, tokens,
            input.stride_bytes, output.stride_bytes, broadcast) :
#endif
        grouped_mm_plan(type, layout, input.columns, output.columns, experts, selected, tokens,
            input.stride_bytes, output.stride_bytes, broadcast);
    const std::array bindings{matrices, input.view, output.view, indices, token_map, counts, tasks};
    const std::array required{plan.matrix_bytes, plan.input_bytes, plan.output_bytes, plan.indices_bytes,
        plan.token_map_bytes, plan.counts_bytes, plan.tasks_bytes};
    for (size_t i = 0; i < bindings.size(); ++i) {
        static_cast<void>(p.allocation(bindings[i], i == 2 || i >= 4));
        require(bindings[i].bytes >= required[i] && bindings[i].offset % 4 == 0, "grouped prefill buffer span or alignment is invalid");
    }
    p.prefill_weight(matrices, layout, type, input.columns, output.columns, experts);
    validate_disjoint(std::array{matrices, input.view, indices}, std::array{output.view, token_map, counts, tasks});
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("unclassified", "before_grouped_routes", {input.columns, output.columns, experts, tokens});
#endif
    const std::array<uint32_t, 4> grouping{experts, selected, tokens, 32};
    p.select("graph_grouped_routes"); p.bind(indices, 0); p.bind(token_map, 1, true); p.bind(counts, 2, true); p.parameter(grouping, 3);
    p.groups(MTLSizeMake(experts, 1, 1));
    p.select("graph_grouped_tasks"); p.bind(counts, 0); p.bind(tasks, 1, true); p.parameter(grouping, 2);
    p.groups(MTLSizeMake(1, 1, 1));
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("grouped", "route_and_task_preparation", {experts, selected, tokens, type});
#endif
    if (layout == PrefillWeightLayout::gguf && input.stride_bytes == uint64_t(input.columns) * 4 &&
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        (!p.prefill_grouped_f32 && !p.prefill_grouped_auto) &&
#endif
        output.stride_bytes == uint64_t(output.columns) * 4) {
        p.select("graph_grouped_serial", type, {input.columns, output.columns, experts, tokens}, 128, uint32_t(layout));
        p.bind(matrices, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
        p.bind(indices, 4); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
        p.groups(MTLSizeMake((output.columns + 3) / 4, plan.grid_y, 1));
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        operator_checkpoint("grouped", "serial_matmul", {input.columns, output.columns, experts, tokens});
#endif
        return;
    }
    const char* selected_kernel = plan.kernel;
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    selected_kernel = p.grouped_step2_kernel(selected_kernel);
    if (p.kernel_grouped_fp16) { require(p.kernel_grouped_encoded < UINT64_MAX, "grouped candidate encoded counter exhausted"); ++p.kernel_grouped_encoded; }
#endif
    auto args = plan.args;
    if (p.prefill_swizzle && std::strcmp(selected_kernel, "graph_grouped_mm_f32") == 0) args.broadcast |= 2u;
    p.select(selected_kernel, type, {input.columns, output.columns, experts, tokens}, 128, uint32_t(layout));
    p.bind(matrices, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.bind(indices, 4); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
    [p.encoder setThreadgroupMemoryLength:plan.threadgroup_bytes atIndex:0];
    p.groups(MTLSizeMake(plan.grid_x, plan.grid_y, 1));
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("grouped", selected_kernel, {input.columns, output.columns, experts, tokens});
#endif
}
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
GroupedMmRoutes MetalGraph::grouped_mm_routes(GpuView indices, GpuView token_map, GpuView counts, GpuView tasks,
    uint32_t experts, uint32_t selected, uint32_t tokens) {
    auto& p = *impl_; p.active();
    require((p.prefill_grouped_f32 || p.prefill_grouped_auto || p.prefill_grouped_nax) && !p.recording && !p.compatible_pipelines && !p.grouped_mm_prepared.empty() && !p.grouped_routes_active,
        "grouped route scope requires prepared FP32 or NAX direct batches without nesting");
    const uint32_t max_tokens = p.prefill_grouped_nax ? 2048u : 512u;
    require(experts && experts <= 512 && selected && selected <= 16 && selected <= experts && tokens && tokens <= max_tokens,
        "grouped route scope geometry");
    require(p.grouped_routes_generation < UINT64_MAX, "grouped route scope generation exhausted");
    const uint64_t route_count = checked_mul(tokens, selected), maximum_tasks = route_count/32 + experts;
    const std::array views{indices, token_map, counts, tasks};
    const std::array<uint64_t, 4> required{checked_mul(route_count, 4), checked_mul(checked_mul(experts, tokens), 4), checked_mul(experts, 4), checked_mul(checked_add(2, checked_mul(maximum_tasks, 2)), 4)};
    for (size_t i = 0; i < views.size(); ++i) {
        static_cast<void>(p.allocation(views[i], i != 0));
        require(views[i].bytes >= required[i], "grouped route scope metadata span");
    }
    validate_disjoint(std::array{indices}, std::array{token_map, counts, tasks});
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("unclassified", "before_grouped_routes", {experts, selected, tokens, 0});
#endif
    const std::array<uint32_t, 4> grouping{experts, selected, tokens, 32};
    if (p.prefill_grouped_nax) {
        const std::array<uint32_t, 4> nax_grouping{experts, selected, tokens, 32};
        p.select("nax_grouped_routes", uint32_t(-1), {experts, selected, tokens, 0}, 256);
        p.bind(indices, 0); p.bind(token_map, 1, true); p.bind(counts, 2, true); p.parameter(nax_grouping, 3);
        p.dispatch(true, MTLSizeMake((experts + 7u) / 8u, 1, 1), MTLSizeMake(256, 1, 1));
        const std::array<uint32_t, 2> grid_x{8u, 40u};
        p.select("nax_grouped_tasks_indirect");
        p.bind(counts, 0); p.bind(tasks, 1, true); p.parameter(nax_grouping, 2);
        p.bind(p.nax_indirect_args, 3, true); p.parameter(grid_x, 4);
        p.groups(MTLSizeMake(1, 1, 1));
        p.nax_gathered_broadcast_valid = false;
        p.nax_gathered_handoff_valid = false;
    } else {
        p.select("graph_grouped_routes"); p.bind(indices, 0); p.bind(token_map, 1, true); p.bind(counts, 2, true); p.parameter(grouping, 3);
        p.groups(MTLSizeMake(experts, 1, 1));
        p.select("graph_grouped_tasks"); p.bind(counts, 0); p.bind(tasks, 1, true); p.parameter(grouping, 2); p.groups(MTLSizeMake(1, 1, 1));
    }
    p.grouped_routes_views = views; p.grouped_routes_experts = experts; p.grouped_routes_selected = selected; p.grouped_routes_tokens = tokens;
    ++p.grouped_routes_generation; p.grouped_routes_active = true;
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("grouped", "route_and_task_preparation", {experts, selected, tokens, 0});
#endif
    GroupedMmRoutes handle; handle.owner_ = p.owner; handle.generation_ = p.grouped_routes_generation; return handle;
}
void MetalGraph::grouped_mm_project(uint32_t type, PrefillWeightLayout layout, GpuView matrices, const GroupedMmRoutes& routes,
    GpuRowsF32 input, GpuRowsF32 output, bool broadcast) {
    auto& p = *impl_; p.active();
    require((p.prefill_grouped_f32 || p.prefill_grouped_auto || p.prefill_grouped_nax) && p.grouped_routes_active && routes.owner_ == p.owner && routes.generation_ == p.grouped_routes_generation,
        "foreign, ended or stale grouped route scope");
    const auto experts = p.grouped_routes_experts, selected = p.grouped_routes_selected;
    const auto [indices, token_map, counts, tasks] = p.grouped_routes_views;
    require(!p.recording && !p.compatible_pipelines && p.grouped_mm_prepared.contains({type, uint32_t(layout)}), "grouped prefill requires idle preparation and direct batches");
    require(selected && output.rows % selected == 0, "grouped prefill route-row count mismatch");
    const auto tokens = output.rows / selected;
    require(tokens == p.grouped_routes_tokens, "grouped route scope token extent mismatch");
    require(input.rows == (broadcast ? tokens : output.rows), "grouped prefill input-row count mismatch");
    const auto plan =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        (p.prefill_grouped_f32 || p.prefill_grouped_auto || p.prefill_grouped_nax) ? grouped_mm_f32_plan(type, layout, input.columns, output.columns, experts, selected, tokens,
            input.stride_bytes, output.stride_bytes, broadcast, p.prefill_grouped_nax) :
#endif
        grouped_mm_plan(type, layout, input.columns, output.columns, experts, selected, tokens,
            input.stride_bytes, output.stride_bytes, broadcast, false);
    const std::array bindings{matrices, input.view, output.view, indices, token_map, counts, tasks};
    const std::array required{plan.matrix_bytes, plan.input_bytes, plan.output_bytes, plan.indices_bytes,
        plan.token_map_bytes, plan.counts_bytes, plan.tasks_bytes};
    for (size_t i = 0; i < bindings.size(); ++i) {
        static_cast<void>(p.allocation(bindings[i], i == 2));
        require(bindings[i].bytes >= required[i] && bindings[i].offset % 4 == 0, "grouped prefill buffer span or alignment is invalid");
    }
    p.prefill_weight(matrices, layout, type, input.columns, output.columns, experts);
    validate_disjoint(std::array{matrices, input.view, indices, token_map, counts, tasks}, std::array{output.view});
    if (p.prefill_grouped_nax) {
        require(input.columns % 32 == 0, "NAX grouped prefill requires columns divisible by 32");
        constexpr uint64_t kNaxGroupedScratchBase = 10'485'760ull;
        const uint64_t nax_max_tiles = checked_add(checked_mul(tokens, selected) / 32ull, experts);
        const uint64_t gathered_bytes = checked_mul(checked_mul(nax_max_tiles, 32ull), checked_mul(uint64_t(input.columns), 2ull));
        const bool use_handoff = !broadcast && p.nax_gathered_handoff_valid &&
            p.nax_gathered_handoff_slot == input.view.slot &&
            p.nax_gathered_handoff_offset == input.view.offset &&
            p.nax_gathered_handoff_rows == input.rows &&
            p.nax_gathered_handoff_columns == input.columns &&
            p.nax_gathered_handoff_stride == input.stride_bytes &&
            p.nax_gathered_handoff_scratch_bytes >= gathered_bytes;
        GpuView gathered{};
        if (use_handoff) {
            gathered = slice(p.nax_bf16_scratch, p.nax_gathered_handoff_scratch_offset, gathered_bytes);
            p.nax_gathered_handoff_valid = false;
        } else {
            const uint64_t gathered_base = checked_add(kNaxGroupedScratchBase, gathered_bytes) <= p.nax_bf16_scratch.bytes
                ? kNaxGroupedScratchBase : 0ull;
            require(p.nax_bf16_scratch.bytes >= checked_add(gathered_base, gathered_bytes), "NAX grouped prefill BF16 scratch too small");
            gathered = slice(p.nax_bf16_scratch, gathered_base, gathered_bytes);
            const bool reuse_gathered = broadcast && p.nax_gathered_broadcast_valid &&
                p.nax_gathered_input_slot == input.view.slot &&
                p.nax_gathered_input_offset == input.view.offset &&
                p.nax_gathered_columns == input.columns &&
                p.nax_gathered_stride == input.stride_bytes;
            if (!reuse_gathered) {
                p.select("nax_moe_gather_bf16", uint32_t(-1), {input.columns, output.columns, experts, tokens});
                p.bind(input.view, 0); p.bind(gathered, 1, true); p.parameter(plan.args, 2);
                p.bind(indices, 3); p.bind(token_map, 4); p.bind(tasks, 5); p.bind(counts, 6);
                p.dispatch_indirect(p.nax_indirect_args, 32, MTLSizeMake(128, 1, 1));
                if (broadcast) {
                    p.nax_gathered_broadcast_valid = true;
                    p.nax_gathered_input_slot = input.view.slot;
                    p.nax_gathered_input_offset = input.view.offset;
                    p.nax_gathered_columns = input.columns;
                    p.nax_gathered_stride = input.stride_bytes;
                } else {
                    p.nax_gathered_broadcast_valid = false;
                }
            }
        }
        p.select("nax_grouped_moe", type, {input.columns, output.columns, experts, tokens}, 128, uint32_t(layout));
        p.bind(matrices, 0); p.bind(gathered, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
        p.bind(indices, 4); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
        if (plan.grid_x == 8) {
            p.dispatch_indirect(p.nax_indirect_args, 0, MTLSizeMake(128, 1, 1));
        } else if (plan.grid_x == 40) {
            p.dispatch_indirect(p.nax_indirect_args, 16, MTLSizeMake(128, 1, 1));
        } else if (plan.grid_x == 10) {
            p.dispatch_indirect(p.nax_indirect_args, 64, MTLSizeMake(128, 1, 1));
        } else {
            p.groups(MTLSizeMake(plan.grid_x, plan.grid_y, 1));
        }
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        operator_checkpoint("grouped", "nax_grouped_moe", {input.columns, output.columns, experts, tokens});
#endif
        return;
    }
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    if (p.kernel_search_sink && p.operator_tag.prefix == 0 && (p.operator_tag.layer == 0 || p.operator_tag.layer == 1) && tokens == 512 && experts == 512 && selected == 10 && (type == 12 || type == 7)) {
        KernelSearchCaptureRecord record; record.kind = 1; record.encoding = type;
        record.dimensions = {input.columns, output.columns, experts, selected, tokens, uint32_t(broadcast), uint32_t(input.stride_bytes), uint32_t(output.stride_bytes)};
        static_assert(sizeof(plan.args) <= sizeof(record.parameters)); record.parameter_bytes = sizeof(plan.args); std::memcpy(record.parameters.data(), &plan.args, sizeof(plan.args));
        record.views = {slice(matrices, 0, plan.matrix_bytes), slice(input.view, 0, plan.input_bytes), slice(indices, 0, plan.indices_bytes), slice(token_map, 0, plan.token_map_bytes), slice(tasks, 0, plan.tasks_bytes), slice(counts, 0, plan.counts_bytes)};
        kernel_search_capture(record, 1u << (uint32_t(p.operator_tag.layer) * 2 + uint32_t(type == 7)));
    }
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("unclassified", "before_grouped_matmul", {input.columns, output.columns, experts, tokens});
#endif
    const char* selected_kernel = plan.kernel;
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    selected_kernel = p.grouped_step2_kernel(selected_kernel);
    if (p.kernel_grouped_fp16) { require(p.kernel_grouped_encoded < UINT64_MAX, "grouped candidate encoded counter exhausted"); ++p.kernel_grouped_encoded; }
#endif
    auto args = plan.args;
    if (p.prefill_swizzle && std::strcmp(selected_kernel, "graph_grouped_mm_f32") == 0) args.broadcast |= 2u;
    p.select(selected_kernel, type, {input.columns, output.columns, experts, tokens}, 128, uint32_t(layout));
    p.bind(matrices, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.bind(indices, 4); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
    [p.encoder setThreadgroupMemoryLength:plan.threadgroup_bytes atIndex:0];
    p.groups(MTLSizeMake(plan.grid_x, plan.grid_y, 1));
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("grouped", selected_kernel, {input.columns, output.columns, experts, tokens});
#endif
}
bool MetalGraph::can_grouped_mm_handoff_bf16(uint32_t experts, uint32_t selected, uint32_t tokens,
    uint32_t in_columns, uint32_t out_columns) const {
    const auto& p = *impl_; p.thread_check();
    if (!p.prefill_grouped_nax || p.recording || p.compatible_pipelines || !experts || !selected || !tokens ||
        !in_columns || !out_columns || in_columns % 32 != 0 || out_columns % 32 != 0) return false;
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (p.operator_sink) return false;
#endif
    constexpr uint64_t kNaxGroupedScratchBase = 10'485'760ull;
    const uint64_t scratch_cap = p.nax_bf16_scratch.bytes ? p.nax_bf16_scratch.bytes : 335'544'320ull;
    const uint64_t nax_max_tiles = checked_add(checked_mul(tokens, selected) / 32ull, experts);
    const uint64_t gathered_bytes = checked_mul(checked_mul(nax_max_tiles, 32ull), checked_mul(uint64_t(in_columns), 2ull));
    const uint64_t gathered_base = checked_add(kNaxGroupedScratchBase, gathered_bytes) <= scratch_cap
        ? kNaxGroupedScratchBase : 0ull;
    const uint64_t handoff_offset = (checked_add(gathered_base, gathered_bytes) + 255ull) & ~255ull;
    const uint64_t handoff_bytes = checked_mul(checked_mul(nax_max_tiles, 32ull), checked_mul(uint64_t(out_columns), 2ull));
    return checked_add(handoff_offset, handoff_bytes) <= scratch_cap;
}
void MetalGraph::grouped_mm_gate_up_swiglu(uint32_t type, PrefillWeightLayout layout, GpuView gate_matrices,
    GpuView up_matrices, const GroupedMmRoutes& routes, GpuRowsF32 input, GpuRowsF32 output, bool broadcast, bool handoff_bf16) {
    auto& p = *impl_; p.active();
    require(p.prefill_grouped_nax && p.grouped_routes_active && routes.owner_ == p.owner && routes.generation_ == p.grouped_routes_generation,
        "NAX grouped fused gate/up/SwiGLU requires active NAX grouped route scope");
    const auto experts = p.grouped_routes_experts, selected = p.grouped_routes_selected;
    const auto [indices, token_map, counts, tasks] = p.grouped_routes_views;
    require(!p.recording && !p.compatible_pipelines && p.grouped_mm_prepared.contains({type, uint32_t(layout)}),
        "NAX grouped fused gate/up/SwiGLU requires idle preparation and direct batches");
    require(selected && output.rows % selected == 0, "grouped prefill route-row count mismatch");
    const auto tokens = output.rows / selected;
    require(tokens == p.grouped_routes_tokens, "grouped route scope token extent mismatch");
    require(input.rows == (broadcast ? tokens : output.rows), "grouped prefill input-row count mismatch");
    require(input.columns % 32 == 0, "NAX grouped prefill requires columns divisible by 32");
    const auto plan = grouped_mm_f32_plan(type, layout, input.columns, output.columns, experts, selected, tokens,
        input.stride_bytes, output.stride_bytes, broadcast, true);
    const std::array bindings{gate_matrices, up_matrices, input.view, output.view, indices, token_map, counts, tasks};
    const std::array required{plan.matrix_bytes, plan.matrix_bytes, plan.input_bytes, plan.output_bytes,
        plan.indices_bytes, plan.token_map_bytes, plan.counts_bytes, plan.tasks_bytes};
    for (size_t i = 0; i < bindings.size(); ++i) {
        static_cast<void>(p.allocation(bindings[i], i == 3));
        require(bindings[i].bytes >= required[i] && bindings[i].offset % 4 == 0, "grouped prefill buffer span or alignment is invalid");
    }
    p.prefill_weight(gate_matrices, layout, type, input.columns, output.columns, experts);
    p.prefill_weight(up_matrices, layout, type, input.columns, output.columns, experts);
    validate_disjoint(std::array{gate_matrices, up_matrices, input.view, indices, token_map, counts, tasks}, std::array{output.view});
    constexpr uint64_t kNaxGroupedScratchBase = 10'485'760ull;
    const uint64_t nax_max_tiles = checked_add(checked_mul(tokens, selected) / 32ull, experts);
    const uint64_t gathered_bytes = checked_mul(checked_mul(nax_max_tiles, 32ull), checked_mul(uint64_t(input.columns), 2ull));
    const uint64_t gathered_base = checked_add(kNaxGroupedScratchBase, gathered_bytes) <= p.nax_bf16_scratch.bytes
        ? kNaxGroupedScratchBase : 0ull;
    require(p.nax_bf16_scratch.bytes >= checked_add(gathered_base, gathered_bytes), "NAX grouped prefill BF16 scratch too small");
    const auto gathered = slice(p.nax_bf16_scratch, gathered_base, gathered_bytes);
    const bool reuse_gathered = broadcast && p.nax_gathered_broadcast_valid &&
        p.nax_gathered_input_slot == input.view.slot &&
        p.nax_gathered_input_offset == input.view.offset &&
        p.nax_gathered_columns == input.columns &&
        p.nax_gathered_stride == input.stride_bytes;
    if (!reuse_gathered) {
        p.select("nax_moe_gather_bf16", uint32_t(-1), {input.columns, output.columns, experts, tokens});
        p.bind(input.view, 0); p.bind(gathered, 1, true); p.parameter(plan.args, 2);
        p.bind(indices, 3); p.bind(token_map, 4); p.bind(tasks, 5); p.bind(counts, 6);
        p.dispatch_indirect(p.nax_indirect_args, 32, MTLSizeMake(128, 1, 1));
        if (broadcast) {
            p.nax_gathered_broadcast_valid = true;
            p.nax_gathered_input_slot = input.view.slot;
            p.nax_gathered_input_offset = input.view.offset;
            p.nax_gathered_columns = input.columns;
            p.nax_gathered_stride = input.stride_bytes;
        } else {
            p.nax_gathered_broadcast_valid = false;
        }
    }
    auto args = plan.args;
    GpuView handoff_view = slice(output.view, 0, 4);
    uint64_t handoff_offset = 0, handoff_bytes = 0;
    if (handoff_bf16) {
        require(can_grouped_mm_handoff_bf16(experts, selected, tokens, input.columns, output.columns),
            "NAX grouped BF16 handoff geometry or scratch budget unsupported");
        handoff_offset = (checked_add(gathered_base, gathered_bytes) + 255ull) & ~255ull;
        handoff_bytes = checked_mul(checked_mul(nax_max_tiles, 32ull), checked_mul(uint64_t(output.columns), 2ull));
        handoff_view = slice(p.nax_bf16_scratch, handoff_offset, handoff_bytes);
        args.broadcast |= 2u;
    }
    p.select("nax_grouped_moe_gate_up_swiglu", type, {input.columns, output.columns, experts, tokens}, 128, uint32_t(layout));
    p.bind(gate_matrices, 0); p.bind(up_matrices, 1); p.bind(gathered, 2); p.bind(output.view, 3, true);
    p.parameter(args, 4); p.bind(indices, 5); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
    p.bind(handoff_view, 10, true);
    if (handoff_bf16) {
        p.nax_gathered_handoff_valid = true;
        p.nax_gathered_handoff_slot = output.view.slot;
        p.nax_gathered_handoff_offset = output.view.offset;
        p.nax_gathered_handoff_rows = output.rows;
        p.nax_gathered_handoff_columns = output.columns;
        p.nax_gathered_handoff_stride = output.stride_bytes;
        p.nax_gathered_handoff_scratch_offset = handoff_offset;
        p.nax_gathered_handoff_scratch_bytes = handoff_bytes;
    } else {
        p.nax_gathered_handoff_valid = false;
    }
    const uint32_t out_tiles = (output.columns + 31u) / 32u;
    if (out_tiles == 16u) {
        p.dispatch_indirect(p.nax_indirect_args, 48, MTLSizeMake(128, 1, 1));
    } else if (out_tiles == 20u) {
        p.dispatch_indirect(p.nax_indirect_args, 80, MTLSizeMake(128, 1, 1));
    } else {
        p.groups(MTLSizeMake(out_tiles, plan.grid_y, 1));
    }
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    operator_checkpoint("grouped", "nax_grouped_moe_gate_up_swiglu", {input.columns, output.columns, experts, tokens});
#endif
}
void MetalGraph::end_grouped_mm_routes(const GroupedMmRoutes& routes) {
    auto& p = *impl_; p.active();
    require(p.grouped_routes_active && routes.owner_ == p.owner && routes.generation_ == p.grouped_routes_generation,
        "foreign, ended or stale grouped route scope end");
    p.grouped_routes_active = false; p.grouped_routes_views = {};
    p.nax_gathered_broadcast_valid = false;
    p.nax_gathered_handoff_valid = false;
}
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
void MetalGraph::configure_prefill_grouped_f32(bool enabled) {
    auto& p = *impl_; p.idle();
    require(p.grouped_mm_prepared.empty(), "prefill grouped precision is fixed before preparation");
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    require(enabled || !p.prefill_dense_staged_f32, "staged dense requires the fixed FP32 grouped policy");
#endif
    p.prefill_grouped_f32 = enabled;
}
bool MetalGraph::prefill_grouped_f32() const { impl_->thread_check(); return impl_->prefill_grouped_f32; }
void MetalGraph::configure_prefill_grouped_auto(bool enabled) {
    auto& p = *impl_; p.idle();
    require(p.grouped_mm_prepared.empty(), "prefill grouped precision is fixed before preparation");
    p.prefill_grouped_auto = enabled;
}
bool MetalGraph::prefill_grouped_auto() const { impl_->thread_check(); return impl_->prefill_grouped_auto; }
void MetalGraph::configure_prefill_grouped_nax(bool enabled) {
    auto& p = *impl_; p.idle();
    require(p.grouped_mm_prepared.empty(), "prefill grouped precision is fixed before preparation");
    require(!enabled || [p.device supportsFamily:MTLGPUFamilyMetal4], "NAX grouped prefill requires Metal 4 support");
    p.prefill_grouped_nax = enabled;
}
bool MetalGraph::prefill_grouped_nax() const { impl_->thread_check(); return impl_->prefill_grouped_nax; }
#endif
void MetalGraph::configure_router_margin_epsilon(float epsilon) {
    auto& p = *impl_; p.idle();
    require(std::isfinite(epsilon) && epsilon > 0.0f, "router margin epsilon must be positive finite");
    p.router_margin_epsilon = epsilon;
}
float MetalGraph::router_margin_epsilon() const { impl_->thread_check(); return impl_->router_margin_epsilon; }
void MetalGraph::configure_router_margin_audit(bool enabled) {
    auto& p = *impl_; p.idle();
    p.router_margin_audit_enabled = enabled;
}
bool MetalGraph::router_margin_audit() const { impl_->thread_check(); return impl_->router_margin_audit_enabled; }
RouterMarginTelemetry MetalGraph::router_margin_telemetry() const {
    impl_->thread_check(); return impl_->router_margin_telemetry;
}
void MetalGraph::reset_router_margin_telemetry() {
    impl_->thread_check(); impl_->router_margin_telemetry = {};
}
GpuView MetalGraph::router_margins_scratch() const {
    impl_->thread_check(); return impl_->router_margins_scratch;
}
bool MetalGraph::evaluate_router_margin(uint32_t tokens) {
    auto& p = *impl_; p.active();
    if (!p.router_margins_scratch.bytes || !tokens) return false;
    if (p.encoder != nil) {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        p.nax_dense_bf16_valid = false;
        p.nax_gathered_broadcast_valid = false;
        p.nax_gathered_handoff_valid = false;
#endif
        [p.encoder endEncoding];
        p.encoder = nil;
        [p.command commit];
        [p.command waitUntilCompleted];
        const auto err_val = *static_cast<const uint32_t*>(p.error_flag.contents);
        require(err_val == 0, "GPU graph produced nonfinite values or invalid gates");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        // This margin readback closes a physical command buffer inside the logical batch. Dispatches and barriers remain accumulated until the owner's submit.
        p.lifetime_command_count = checked_add(p.lifetime_command_count, 1);
#endif
    }
    const auto& item = p.allocation(p.router_margins_scratch);
    const float* margins = reinterpret_cast<const float*>(
        static_cast<const std::byte*>(item.buffer.contents) + p.router_margins_scratch.offset);
    float min_m = std::numeric_limits<float>::infinity();
    double sum_m = 0.0;
    uint64_t violations = 0;
    const float threshold = 2.0f * p.router_margin_epsilon;
    for (uint32_t i = 0; i < tokens; ++i) {
        const float m = margins[i];
        if (m < min_m) min_m = m;
        sum_m += m;
        if (m <= threshold) ++violations;
    }
    p.router_margin_telemetry.min_margin = std::min(p.router_margin_telemetry.min_margin, min_m);
    p.router_margin_telemetry.sum_margin += sum_m;
    p.router_margin_telemetry.total_tokens += tokens;
    p.router_margin_telemetry.violation_count += violations;
    p.router_margin_telemetry.total_layers += 1;
    const bool fallback = (violations > 0);
    if (fallback) p.router_margin_telemetry.fallback_layers += 1;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    p.reset_spans();
#endif
    p.command = [p.queue commandBuffer];
    if (!p.command) throw FormatError("cannot create owned graph command buffer");
    if (p.profile_mode != ProfileMode::encoder_boundary) {
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        p.encoder = p.concurrent ? [p.command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [p.command computeCommandEncoder];
#else
        p.encoder = [p.command computeCommandEncoder];
#endif
        if (!p.encoder) { p.command = nil; throw FormatError("cannot create owned graph encoder"); }
    }
    return fallback;
}
void MetalGraph::configure_prefill_dense_f32(bool enabled) {
    auto& p = *impl_; p.idle();
    require(p.dense_mm_prepared.empty(), "prefill dense precision is fixed before preparation");
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    require(enabled || !p.prefill_hc_down_f32, "HC down batching requires the FP32 dense policy");
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    require(enabled || !p.prefill_dense_staged_f32, "staged dense requires the fixed FP32 dense policy");
#endif
#endif
#ifndef TOKSPUR_PRIVATE_Q8_MPP
    require(!enabled, "FP32 prefill dense requires the private Q8 TensorOps build");
#endif
    p.prefill_dense_f32 = enabled;
}
bool MetalGraph::prefill_dense_f32() const { impl_->thread_check(); return impl_->prefill_dense_f32; }
void MetalGraph::configure_prefill_dense_nax(bool enabled) {
    auto& p = *impl_; p.idle();
    require(p.dense_mm_prepared.empty(), "prefill dense precision is fixed before preparation");
    require(!enabled || (p.prefill_dense_f32 && [p.device supportsFamily:MTLGPUFamilyMetal4]),
        "NAX prefill dense requires FP32 dense policy and Metal 4 support");
    p.prefill_dense_nax = enabled;
}
bool MetalGraph::prefill_dense_nax() const { impl_->thread_check(); return impl_->prefill_dense_nax; }
uint32_t MetalGraph::dense_mm_q8_dispatches(uint32_t columns) const {
    impl_->thread_check();
    return (impl_->prefill_dense_nax && columns % 64 == 0) ? 2u : 1u;
}
bool MetalGraph::can_reuse_nax_dense_bf16(GpuRowsF32 input) const {
    const auto& p = *impl_; p.thread_check();
    if (!p.prefill_dense_nax || p.verify_nax_scope || !p.nax_dense_bf16_valid || !input.rows || !input.columns || input.columns % 64 != 0) return false;
    const uint32_t needed_padded = ((input.rows + 31u) / 32u) * 32u;
    if (p.nax_dense_padded_rows < needed_padded) return false;
    const uint64_t end = checked_add(input.view.offset, checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4));
    return p.nax_dense_input_slot == input.view.slot &&
        p.nax_dense_input_offset == input.view.offset &&
        p.nax_dense_input_end == end &&
        p.nax_dense_rows == input.rows &&
        p.nax_dense_columns == input.columns &&
        p.nax_dense_stride == input.stride_bytes;
}
void MetalGraph::set_mtp_priming_scope(bool active) {
    impl_->thread_check();
    impl_->mtp_priming_scope = active;
}
void MetalGraph::configure_verify_nax(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.verify_nax_configured && !p.verify_nax_prepared, "verify NAX selection is fixed before preparation");
    require(!enabled || [p.device supportsFamily:MTLGPUFamilyMetal4], "verify NAX requires Metal 4 support");
    p.verify_nax = enabled;
    p.verify_nax_configured = true;
}
bool MetalGraph::verify_nax() const {
    impl_->thread_check();
    return impl_->verify_nax;
}
void MetalGraph::configure_verify_nax_m8(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.verify_nax_m8_configured && !p.verify_nax_prepared, "verify NAX m8 selection is fixed before preparation");
    require(!enabled || [p.device supportsFamily:MTLGPUFamilyMetal4], "verify NAX m8 requires Metal 4 support");
    p.verify_nax_m8 = enabled;
    p.verify_nax_m8_configured = true;
}
void MetalGraph::configure_target_m8_splits(NaxSplitOverrides overrides) {
    auto& p = *impl_; p.idle();
    require(p.verify_nax_m8 && !p.verify_nax_prepared && p.target_m8_splits.digest().empty(),
        "target M8 overrides require immutable M8 selection before preparation");
    p.target_m8_splits = std::move(overrides);
}
const std::string& MetalGraph::target_m8_split_digest() const { impl_->thread_check(); return impl_->target_m8_splits.digest(); }
void MetalGraph::set_target_verification_role(bool active) {
    auto& p = *impl_; p.thread_check();
    require(!active || !p.target_verification_role, "nested target-verification role");
    p.target_verification_role = active;
}
bool MetalGraph::verify_nax_m8() const {
    impl_->thread_check();
    return impl_->verify_nax_m8;
}
bool MetalGraph::verify_nax_active() const {
    const auto& p = *impl_; p.thread_check();
    return p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope;
}
void MetalGraph::set_verify_nax_scope(bool active) {
    auto& p = *impl_; p.thread_check();
    p.verify_nax_scope = active;
    p.nax_dense_bf16_valid = false;
    p.nax_gathered_broadcast_valid = false;
    p.nax_gathered_handoff_valid = false;
}
void MetalGraph::prepare_verify_nax(std::span<const std::pair<uint32_t, PrefillWeightLayout>> moe_layouts) {
    auto& p = *impl_; p.idle();
    require(p.verify_nax && [p.device supportsFamily:MTLGPUFamilyMetal4], "verify NAX preparation requires enabled Metal 4 policy");
    for (const uint32_t layout : {0u, 1u}) {
#ifndef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (layout != 0u) continue;
#endif
        const auto m4 = p.pipeline("nax_verify_dense_q8_m4", 8, 128, false, layout);
        const auto m8 = p.pipeline("nax_verify_dense_q8_m8", 8, 128, false, layout);
        const auto sw4 = p.pipeline("nax_verify_dense_q8_swiglu_m4", 8, 128, false, layout);
        const auto sw8 = p.pipeline("nax_verify_dense_q8_swiglu_m8", 8, 128, false, layout);
        const auto nax_m8 = p.pipeline("nax_m8_dense_q8", 8, 64, false, layout);
        const auto nax_m8_qmv = p.pipeline("nax_m8_qmv_q8", 8, 128, false, layout);
        const auto nax_m8_sw = p.pipeline("nax_m8_dense_q8_swiglu", 8, 128, false, layout);
        require(m4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            sw4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            sw8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_m8_qmv.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_m8_sw.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "verify NAX dense threadgroup memory exceeds device capability");
    }
    for (const char* kname : {"nax_verify_dense_bf16_r1_m4", "nax_verify_dense_bf16_r1_m8",
                              "nax_verify_dense_bf16_m4", "nax_verify_dense_bf16_m8"}) {
        const auto pipe = p.pipeline(kname, uint32_t(-1), 256);
        require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "verify NAX dense BF16 threadgroup memory exceeds device capability");
    }
    const auto bf16_m8 = p.pipeline("nax_verify_bf16_m8", uint32_t(-1), 128);
    const auto nax_bf16_m8 = p.pipeline("nax_m8_dense_bf16", uint32_t(-1), 128);
    require(bf16_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
        nax_bf16_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
        "verify NAX BF16 threadgroup memory exceeds device capability");
    constexpr uint64_t need_partials = uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes;
    constexpr uint64_t need_counters = uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotCountersBytes + 64ull;
    if (p.ledger.budget - p.ledger.live_bytes >= need_partials + need_counters + (4ull << 20)) {
        if (p.verify_nax_split_partials.bytes < need_partials) {
            if (p.verify_nax_split_partials.bytes) release(p.verify_nax_split_partials);
            p.verify_nax_split_partials = zeros(need_partials, "verify_nax_split_partials");
        }
        if (p.verify_nax_split_counters.bytes < need_counters) {
            if (p.verify_nax_split_counters.bytes) release(p.verify_nax_split_counters);
            p.verify_nax_split_counters = zeros(need_counters, "verify_nax_split_counters");
        }
    } else if (p.verify_nax_split_counters.bytes < 64ull) {
        if (p.verify_nax_split_counters.bytes) release(p.verify_nax_split_counters);
        p.verify_nax_split_counters = zeros(64ull, "verify_nax_split_counters_idle");
    }
    const auto gdn_gates = p.pipeline("nax_verify_gdn_gates_f32");
    const auto gated_mix = p.pipeline("nax_verify_gated_mix_rows");
    require(gdn_gates.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
        gated_mix.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
        "verify NAX helper threadgroup memory exceeds device capability");
    const auto union_pipe = p.pipeline("nax_verify_moe_build_union", uint32_t(-1), 32);
    require(union_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
        "verify NAX MoE union builder threadgroup memory exceeds device capability");
    if (p.verify_nax_moe_meta.bytes < 8192ull) {
        if (p.verify_nax_moe_meta.bytes) release(p.verify_nax_moe_meta);
        p.verify_nax_moe_meta = zeros(8192, "verify_nax_moe_meta");
    }
    for (const auto& [type, layout] : moe_layouts) {
#ifndef TOKSPUR_PRIVATE_LANE_LAYOUT
        require(layout == PrefillWeightLayout::gguf, "packed verify NAX requires the lane-layout build");
#endif
        require(type == 7 || type == 8 || type == 12 || type == 13, "verify NAX routed MoE encoding");
        require(layout == PrefillWeightLayout::gguf ||
            (layout == PrefillWeightLayout::q51_lanes_v1 && type == 7) ||
            (layout == PrefillWeightLayout::q8_lanes_v1 && type == 8),
            "verify NAX routed MoE layout/encoding mismatch");
        const auto pipe = p.pipeline("nax_verify_moe_routed", type, 64, false, uint32_t(layout));
        const auto gu_pipe = p.pipeline("nax_verify_moe_gate_up_swiglu", type, 128, false, uint32_t(layout));
        const auto dn_pipe = p.pipeline("nax_verify_moe_down", type, 128, false, uint32_t(layout));
        const auto m8_gu_pipe = p.pipeline("nax_m8_moe_gate_up_swiglu", type, 64, false, uint32_t(layout));
        const auto m8_dn_pipe = p.pipeline("nax_m8_moe_down", type, 64, false, uint32_t(layout));
        require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            gu_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            dn_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            m8_gu_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            m8_dn_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "verify NAX routed MoE threadgroup memory exceeds device capability");
        p.verify_nax_moe_prepared.insert({type, uint32_t(layout)});
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    if (p.small_gemv_splits == 16) {
        prepare_hc_rows();
        if (p.hc_rows_scratch.bytes < 8ull * 4 * 16 * 4) {
            if (p.hc_rows_scratch.bytes) release(p.hc_rows_scratch);
            p.hc_rows_scratch = zeros(8ull * 4 * 16 * 4, "verify NAX HC row injection partials");
        }
    }
#endif
#ifdef TOKSPUR_PRIVATE_GDN_CHUNK
    prepare_gdn_chunk();
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
    prepare_qsa_prefill();
#endif
    const auto t64_m8 = p.pipeline("nax_m8_dense_q8_t64", uint32_t(-1), 64);
    const auto t64_sw = p.pipeline("nax_m8_dense_q8_swiglu_t64", uint32_t(-1), 64);
    require(t64_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
        t64_sw.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
        "verify NAX Q8_tile64 threadgroup memory exceeds device capability");
    p.q8_tile64_ready = true;
    p.ensure_q8_tile64_all();
    p.verify_nax_prepared = true;
}
void MetalGraph::nax_m8_dense_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu, float silu_scale) {
    auto& p = *impl_; p.active();
    require(std::isfinite(silu_scale) && (!swiglu || silu_scale == 0.0f),
        "NAX m8 dense cannot combine swiglu with scaled SiLU epilogue");
    const bool allowed = (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope) || (p.dflash2_prepared && p.dflash2_scope);
    require(allowed && !p.recording && !p.compatible_pipelines,
        "NAX m8 dense requires prepared active verify or DFlash2 scope and direct batches");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "NAX m8 dense geometry mismatch");
    require(input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "NAX m8 dense row stride mismatch");
    const auto layout = prefill_weight_layout(matrix);
    require(layout == PrefillWeightLayout::gguf ||
        (layout == PrefillWeightLayout::q8_lanes_v1 && (input.columns < 512 || input.columns % 256 == 0)),
        "NAX m8 dense requires GGUF or safe q8_lanes_v1 layout");
    const uint32_t row_bytes = (input.columns / 32u) * 34u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes >= matrix_bytes && input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        matrix.offset % 4 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "NAX m8 dense buffer span or alignment is invalid");
    p.prefill_weight(matrix, layout, 8, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    const uint32_t mode = swiglu ? 1u : (silu_scale != 0.0f ? 2u : 0u);
    const uint32_t scale_bits = silu_scale != 0.0f ? std::bit_cast<uint32_t>(silu_scale) : 1u;
    if (p.verify_nax_m8) {
        if (const auto* entry = p.q8_tile64_entry(matrix, input.columns, output.columns)) {
            const uint32_t splits = p.target_split(NaxSplitFamily::q8_t64, input.columns, output.columns, input.rows, 64u, false, (p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
                ? pick_nax_m8_q8_t64_splits(input.columns, output.columns, 64u, false)
                : 1u));
            const GroupedMmArgs args{input.columns, output.columns, row_bytes, scale_bits, splits, input.rows, 1u, mode,
                matrix_bytes, input.stride_bytes, output.stride_bytes};
            p.select("nax_m8_dense_q8_t64", uint32_t(-1), {input.columns, output.columns, input.rows, uint32_t(layout)}, 64);
            p.bind(entry->plane0, 0); p.bind(entry->meta, 1); p.bind(input.view, 2); p.bind(output.view, 3, true); p.parameter(args, 4);
            if (splits > 1u) {
                const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
                const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
                const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
                p.bind(part_view, 5, true);
                p.bind(cnt_view, 6, true);
            } else {
                const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
                p.bind(idle, 5, false);
                p.bind(idle, 6, false);
            }
            p.dispatch(true, MTLSizeMake((output.columns + 63u) / 64u, splits, 1), MTLSizeMake(64, 1, 1));
            return;
        }
    }
    const uint32_t max_nr = input.rows <= 4u ? 4u : 2u;
    const uint32_t nr = output.columns <= 1024u ? 1u : (output.columns <= 4096u ? 2u : max_nr);
    const uint32_t cols_per_tile = 4u * nr;
    const uint32_t splits = p.target_split(NaxSplitFamily::q8, input.columns, output.columns, input.rows, cols_per_tile, false, (p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
        ? pick_nax_m8_q8_splits(input.columns, output.columns, cols_per_tile, false)
        : 1u));
    const GroupedMmArgs args{input.columns, output.columns, row_bytes, scale_bits, splits, input.rows, 1u, mode,
        matrix_bytes, input.stride_bytes, output.stride_bytes};
    p.select("nax_m8_qmv_q8", 8, {input.columns, output.columns, input.rows, uint32_t(layout)}, 128, uint32_t(layout));
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    if (splits > 1u) {
        const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
        const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
        const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
        p.bind(part_view, 4, true);
        p.bind(cnt_view, 5, true);
    } else {
        const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
        p.bind(idle, 4, false);
        p.bind(idle, 5, false);
    }
    p.dispatch(true, MTLSizeMake((output.columns + cols_per_tile - 1u) / cols_per_tile, splits, 1), MTLSizeMake(128, 1, 1));
}
void MetalGraph::verify_nax_dense_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu, float silu_scale) {
    auto& p = *impl_; p.active();
    require(std::isfinite(silu_scale) && (!swiglu || silu_scale == 0.0f),
        "verify NAX dense cannot combine swiglu with scaled SiLU epilogue");
    if (p.verify_nax_m8 && (p.verify_nax_scope || p.dflash2_scope)) {
        nax_m8_dense_q8(matrix, input, output, swiglu, silu_scale);
        return;
    }
    const bool allowed = (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope) || (p.dflash2_prepared && p.dflash2_scope);
    require(allowed && !p.recording && !p.compatible_pipelines,
        "verify NAX dense requires prepared active verify scope and direct batches");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "verify NAX dense geometry mismatch");
    require(input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "verify NAX dense row stride mismatch");
    const auto layout = prefill_weight_layout(matrix);
    require(layout == PrefillWeightLayout::gguf ||
        (layout == PrefillWeightLayout::q8_lanes_v1 && (input.columns < 512 || input.columns % 256 == 0)),
        "verify NAX dense requires GGUF or safe q8_lanes_v1 layout");
    const uint32_t row_bytes = (input.columns / 32u) * 34u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes >= matrix_bytes && input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        matrix.offset % 4 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "verify NAX dense buffer span or alignment is invalid");
    p.prefill_weight(matrix, layout, 8, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    const uint32_t mode = swiglu ? 1u : (silu_scale != 0.0f ? 2u : 0u);
    const uint32_t scale_bits = silu_scale != 0.0f ? std::bit_cast<uint32_t>(silu_scale) : 1u;
    const GroupedMmArgs args{input.columns, output.columns, row_bytes, scale_bits, 1u, input.rows, 1u, mode,
        matrix_bytes, input.stride_bytes, output.stride_bytes};
    const char* kname = input.rows <= 4u ? "nax_verify_dense_q8_m4" : "nax_verify_dense_q8_m8";
    const uint32_t max_nr = input.rows <= 4u ? 4u : 2u;
    const uint32_t nr = output.columns <= 1024u ? 1u : (output.columns <= 4096u ? 2u : max_nr);
    const uint32_t rows_per_tg = 4u * nr;
    p.select(kname, 8, {input.columns, output.columns, input.rows, uint32_t(layout)}, 128, uint32_t(layout));
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.dispatch(true, MTLSizeMake((output.columns + rows_per_tg - 1u) / rows_per_tg, 1, 1), MTLSizeMake(128, 1, 1));
}
void MetalGraph::verify_nax_dense_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output) {
    auto& p = *impl_; p.active();
    const bool allowed = (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope) || (p.dflash2_prepared && p.dflash2_scope);
    require(allowed && !p.recording && !p.compatible_pipelines,
        "verify NAX dense BF16 requires prepared active verify scope and direct batches");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "verify NAX dense BF16 geometry mismatch");
    require(input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "verify NAX dense BF16 row stride mismatch");
    require(prefill_weight_layout(matrix) == PrefillWeightLayout::gguf,
        "verify NAX dense BF16 requires original GGUF layout");
    const uint32_t row_bytes = input.columns * 2u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes >= matrix_bytes && input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        matrix.offset % 2 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "verify NAX dense BF16 buffer span or alignment is invalid");
    p.prefill_weight(matrix, PrefillWeightLayout::gguf, 30, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, 1u, input.rows, 1u, 0u,
        matrix_bytes, input.stride_bytes, output.stride_bytes};
    const bool use_r4 = output.columns > 2560u;
    const char* kname = use_r4
        ? (input.rows <= 4u ? "nax_verify_dense_bf16_m4" : "nax_verify_dense_bf16_m8")
        : (input.rows <= 4u ? "nax_verify_dense_bf16_r1_m4" : "nax_verify_dense_bf16_r1_m8");
    const uint32_t rows_per_tg = use_r4 ? 32u : 8u;
    p.select(kname, uint32_t(-1), {input.columns, output.columns, input.rows, 30u}, 256);
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.dispatch(true, MTLSizeMake((output.columns + rows_per_tg - 1u) / rows_per_tg, 1, 1), MTLSizeMake(256, 1, 1));
}
void MetalGraph::verify_nax_gate_up_swiglu_q8(GpuView gate_matrix, GpuView up_matrix, GpuRowsF32 input, GpuRowsF32 output) {
    verify_nax_dense_q8_swiglu(gate_matrix, up_matrix, input, output);
}
void MetalGraph::verify_nax_dense_q8_swiglu(GpuView gate_matrix, GpuView up_matrix, GpuRowsF32 input, GpuRowsF32 output) {
    auto& p = *impl_; p.active();
    const bool allowed = (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope) || (p.dflash2_prepared && p.dflash2_scope);
    require(allowed && !p.recording && !p.compatible_pipelines,
        "verify NAX dense SwiGLU requires prepared active verify scope and direct batches");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "verify NAX dense SwiGLU geometry mismatch");
    require(input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "verify NAX dense SwiGLU row stride mismatch");
    const auto g_layout = prefill_weight_layout(gate_matrix);
    const auto u_layout = prefill_weight_layout(up_matrix);
    require(g_layout == u_layout && (g_layout == PrefillWeightLayout::gguf ||
        (g_layout == PrefillWeightLayout::q8_lanes_v1 && (input.columns < 512 || input.columns % 256 == 0))),
        "verify NAX dense SwiGLU requires matching GGUF or safe q8_lanes_v1 layouts");
    const uint32_t row_bytes = (input.columns / 32u) * 34u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(gate_matrix)); static_cast<void>(p.allocation(up_matrix));
    static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(gate_matrix.bytes >= matrix_bytes && up_matrix.bytes >= matrix_bytes &&
        input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        gate_matrix.offset % 4 == 0 && up_matrix.offset % 4 == 0 &&
        input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "verify NAX dense SwiGLU buffer span or alignment is invalid");
    p.prefill_weight(gate_matrix, g_layout, 8, input.columns, output.columns, 1);
    p.prefill_weight(up_matrix, u_layout, 8, input.columns, output.columns, 1);
    separate(output.view, {gate_matrix, up_matrix, input.view});
    if (p.verify_nax_m8) {
        const auto* ge = p.q8_tile64_entry(gate_matrix, input.columns, output.columns);
        const auto* ue = p.q8_tile64_entry(up_matrix, input.columns, output.columns);
        if (ge && ue) {
            const uint32_t splits = p.target_split(NaxSplitFamily::q8_t64, input.columns, output.columns, input.rows, 32u, true, (p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
                ? pick_nax_m8_q8_t64_splits(input.columns, output.columns, 32u, true)
                : 1u));
            const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, splits, input.rows, 1u, 0u,
                matrix_bytes, input.stride_bytes, output.stride_bytes};
            p.select("nax_m8_dense_q8_swiglu_t64", uint32_t(-1), {input.columns, output.columns, input.rows, uint32_t(g_layout)}, 64);
            p.bind(ge->plane0, 0); p.bind(ge->meta, 1); p.bind(ue->plane0, 2); p.bind(ue->meta, 3);
            p.bind(input.view, 4); p.bind(output.view, 5, true); p.parameter(args, 6);
            if (splits > 1u) {
                const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
                const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
                const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
                p.bind(part_view, 8, true);
                p.bind(cnt_view, 9, true);
            } else {
                const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
                p.bind(idle, 8, false);
                p.bind(idle, 9, false);
            }
            p.dispatch(true, MTLSizeMake((output.columns + 31u) / 32u, splits, 1), MTLSizeMake(64, 1, 1));
            return;
        }
        const uint32_t splits = p.target_split(NaxSplitFamily::q8, input.columns, output.columns, input.rows, 4u, true, (p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
            ? pick_nax_m8_q8_splits(input.columns, output.columns, 4u, true)
            : 1u));
        const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, splits, input.rows, 1u, 0u,
            matrix_bytes, input.stride_bytes, output.stride_bytes};
        p.select("nax_m8_dense_q8_swiglu", 8, {input.columns, output.columns, input.rows, uint32_t(g_layout)}, 128, uint32_t(g_layout));
        p.bind(gate_matrix, 0); p.bind(up_matrix, 1); p.bind(input.view, 2); p.bind(output.view, 3, true); p.parameter(args, 4);
        if (splits > 1u) {
            const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
            const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
            const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
            p.bind(part_view, 5, true);
            p.bind(cnt_view, 6, true);
        } else {
            const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
            p.bind(idle, 5, false);
            p.bind(idle, 6, false);
        }
        p.dispatch(true, MTLSizeMake((output.columns + 3u) / 4u, splits, 1), MTLSizeMake(128, 1, 1));
        return;
    }
    const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, 1u, input.rows, 1u, 0u,
        matrix_bytes, input.stride_bytes, output.stride_bytes};
    const char* kname = input.rows <= 4u ? "nax_verify_dense_q8_swiglu_m4" : "nax_verify_dense_q8_swiglu_m8";
    p.select(kname, 8, {input.columns, output.columns, input.rows, uint32_t(g_layout)}, 128, uint32_t(g_layout));
    p.bind(gate_matrix, 0); p.bind(up_matrix, 1); p.bind(input.view, 2); p.bind(output.view, 3, true); p.parameter(args, 4);
    p.dispatch(true, MTLSizeMake((output.columns + 3u) / 4u, 1, 1), MTLSizeMake(128, 1, 1));
}
void MetalGraph::verify_nax_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output) {
    auto& p = *impl_; p.active();
    require(p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope && !p.recording && !p.compatible_pipelines,
        "verify NAX BF16 requires prepared active verify scope and direct batches");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "verify NAX BF16 geometry mismatch");
    require(input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "verify NAX BF16 row stride mismatch");
    require(prefill_weight_layout(matrix) == PrefillWeightLayout::gguf,
        "verify NAX BF16 requires original GGUF layout");
    const uint32_t row_bytes = input.columns * 2u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes >= matrix_bytes && input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        matrix.offset % 4 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "verify NAX BF16 buffer span or alignment is invalid");
    p.prefill_weight(matrix, PrefillWeightLayout::gguf, 30, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    if (p.verify_nax_m8) {
        const uint32_t splits = p.target_split(NaxSplitFamily::bf16, input.columns, output.columns, input.rows, 32u, false, (p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
            ? pick_nax_m8_bf16_splits(input.columns, output.columns)
            : 1u));
        const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, splits, input.rows, 1u, 0u,
            matrix_bytes, input.stride_bytes, output.stride_bytes};
        p.select("nax_m8_dense_bf16", uint32_t(-1), {input.columns, output.columns, input.rows, 0}, 128);
        p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
        if (splits > 1u) {
            const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
            const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
            const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
            p.bind(part_view, 4, true);
            p.bind(cnt_view, 5, true);
        } else {
            const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
            p.bind(idle, 4, false);
            p.bind(idle, 5, false);
        }
        p.dispatch(true, MTLSizeMake((output.columns + 31u) / 32u, splits, 1), MTLSizeMake(128, 1, 1));
        return;
    }
    const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, 1u, input.rows, 1u, 0u,
        matrix_bytes, input.stride_bytes, output.stride_bytes};
    const uint32_t threads_per_tg = 128u;
    const uint32_t rows_per_tg = output.columns <= 512u ? 4u : 8u;
    p.select("nax_verify_bf16_m8", uint32_t(-1), {input.columns, output.columns, input.rows, 0}, 128);
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.dispatch(true, MTLSizeMake((output.columns + rows_per_tg - 1u) / rows_per_tg, 1, 1), MTLSizeMake(threads_per_tg, 1, 1));
}
void MetalGraph::verify_nax_gdn_gates_f32(GpuView alpha_w, GpuView beta_w, GpuView input,
    GpuView alpha_out, GpuView beta_out, uint32_t columns, uint32_t heads, uint32_t steps) {
    auto& p = *impl_; p.active();
    require(p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope && !p.recording && !p.compatible_pipelines,
        "verify NAX GDN gates require prepared active verify scope and direct batches");
    require(columns >= 32 && columns <= 65536 && columns % 4 == 0 && heads >= 1 && heads <= 512 && steps >= 1 && steps <= 8,
        "verify NAX GDN gates geometry mismatch");
    require(p.floats64(alpha_w) == uint64_t(columns) * heads && p.floats64(beta_w) == uint64_t(columns) * heads &&
        p.floats64(input) == uint64_t(steps) * columns &&
        p.floats(alpha_out, true) == steps * heads && p.floats(beta_out, true) == steps * heads,
        "verify NAX GDN gates buffer size mismatch");
    validate_disjoint(std::array{alpha_w, beta_w, input}, std::array{alpha_out, beta_out});
    separate(alpha_out, {beta_out});
    p.select("nax_verify_gdn_gates_f32", uint32_t(-1), {columns, heads, steps, 0});
    p.bind(alpha_w, 0); p.bind(beta_w, 1); p.bind(input, 2); p.bind(alpha_out, 3, true); p.bind(beta_out, 4, true);
    p.parameter(std::array<uint32_t, 4>{columns, heads, steps, 0}, 5);
    p.groups(MTLSizeMake(heads * 2u, 1, 1));
}
void MetalGraph::verify_nax_gated_mix_rows(GpuView normalized, GpuView scores, GpuView output,
    uint32_t branches, uint32_t channels, uint32_t steps) {
    auto& p = *impl_; p.active();
    require(p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope && !p.recording && !p.compatible_pipelines,
        "verify NAX gated mix rows require prepared active verify scope and direct batches");
    require(branches >= 1 && branches <= 64 && channels >= 1 && channels <= 65536 && steps >= 1 && steps <= 8,
        "verify NAX gated mix rows geometry mismatch");
    const uint64_t width = uint64_t(branches) * channels;
    require(p.floats64(normalized) == uint64_t(steps) * width && p.floats64(scores) == uint64_t(steps) * width &&
        p.floats64(output, true) == uint64_t(steps) * channels,
        "verify NAX gated mix rows buffer size mismatch");
    separate(output, {normalized, scores});
    p.select("nax_verify_gated_mix_rows", uint32_t(-1), {branches, channels, steps, 0});
    p.bind(normalized, 0); p.bind(scores, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{branches, channels, steps, 0}, 3);
    p.threads(steps * channels);
}
void MetalGraph::verify_nax_moe_routed(uint32_t type, PrefillWeightLayout layout, GpuView matrices, GpuView indices,
    GpuRowsF32 input, GpuRowsF32 output, uint32_t experts, uint32_t selected, bool broadcast, bool swiglu) {
    struct NaxVerifyMoeArgs {
        uint32_t columns, rows, row_bytes, experts;
        uint32_t selected, tokens, broadcast, swiglu;
        uint64_t expert_stride_bytes, input_stride_bytes, output_stride_bytes;
    };
    static_assert(sizeof(NaxVerifyMoeArgs) == 56, "NAX verify MoE ABI");
    auto& p = *impl_; p.active();
    require(p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope && !p.recording && !p.compatible_pipelines &&
        p.verify_nax_moe_prepared.contains({type, uint32_t(layout)}),
        "verify NAX routed MoE requires prepared active verify scope and direct batches");
    require(experts >= 1 && experts <= 512 && selected >= 1 && selected <= 16 && selected <= experts &&
        output.rows >= selected && output.rows % selected == 0,
        "verify NAX routed MoE expert/selection geometry mismatch");
    const uint32_t tokens = output.rows / selected;
    require(tokens >= 1 && tokens <= 8 && input.rows == (broadcast ? tokens : output.rows),
        "verify NAX routed MoE token/input row count mismatch");
    const uint32_t block_cols = (type == 12 || type == 13) ? 256u : 32u;
    const uint32_t block_bytes = type == 12 ? 144u : (type == 13 ? 176u : (type == 8 ? 34u : 24u));
    require(input.columns >= block_cols && input.columns <= 65536 && input.columns % block_cols == 0 &&
        output.columns >= 1 && output.columns <= 65536 &&
        input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "verify NAX routed MoE matrix/stride geometry mismatch");
    require(layout == PrefillWeightLayout::gguf ||
        (layout == PrefillWeightLayout::q51_lanes_v1 && type == 7) ||
        (layout == PrefillWeightLayout::q8_lanes_v1 && type == 8 && (input.columns < 512 || input.columns % 256 == 0)),
        "verify NAX routed MoE layout/encoding mismatch");
    const uint32_t row_bytes = (input.columns / block_cols) * block_bytes;
    const uint64_t expert_stride = uint64_t(row_bytes) * output.columns;
    const uint64_t matrix_bytes = checked_mul(expert_stride, experts);
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    const uint64_t indices_bytes = uint64_t(output.rows) * 4ull;
    const std::array bindings{matrices, input.view, output.view, indices};
    const std::array required{matrix_bytes, input_bytes, output_bytes, indices_bytes};
    for (size_t i = 0; i < bindings.size(); ++i) {
        static_cast<void>(p.allocation(bindings[i], i == 2));
        require(bindings[i].bytes >= required[i] && bindings[i].offset % 4 == 0,
            "verify NAX routed MoE buffer span or alignment is invalid");
    }
    p.prefill_weight(matrices, layout, type, input.columns, output.columns, experts);
    validate_disjoint(std::array{matrices, input.view, indices}, std::array{output.view});
    const NaxVerifyMoeArgs args{input.columns, output.columns, row_bytes, experts, selected, tokens,
        broadcast ? 1u : 0u, swiglu ? 1u : 0u, expert_stride, input.stride_bytes, output.stride_bytes};
    const uint32_t rows_per_tg = (type == 12 || type == 13) ? 4u : 8u;
    p.select("nax_verify_moe_routed", type, {input.columns, output.columns, experts, tokens}, 64, uint32_t(layout));
    p.bind(matrices, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3); p.bind(indices, 4);
    p.dispatch(true, MTLSizeMake((output.columns + rows_per_tg - 1u) / rows_per_tg, output.rows, 1), MTLSizeMake(64, 1, 1));
}
void MetalGraph::verify_nax_moe_union(uint32_t up_type, PrefillWeightLayout up_layout, GpuView gate_matrices, GpuView up_matrices,
    uint32_t down_type, PrefillWeightLayout down_layout, GpuView down_matrices, GpuView indices,
    GpuRowsF32 input, GpuRowsF32 gate_scratch, GpuRowsF32 down_output, uint32_t experts, uint32_t selected) {
    struct NaxVerifyMoeArgs {
        uint32_t columns, rows, row_bytes, experts;
        uint32_t selected, tokens, broadcast, swiglu;
        uint64_t expert_stride_bytes, input_stride_bytes, output_stride_bytes;
    };
    struct NaxVerifyMoeUnionArgs {
        uint32_t experts, selected, tokens, grid_x_gate_up;
        uint32_t grid_x_down, reserved0, reserved1, reserved2;
    };
    static_assert(sizeof(NaxVerifyMoeArgs) == 56, "NAX verify MoE ABI");
    static_assert(sizeof(NaxVerifyMoeUnionArgs) == 32, "NAX verify MoE union args ABI");
    auto& p = *impl_; p.active();
    require(p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope && !p.recording && !p.compatible_pipelines &&
        p.verify_nax_moe_prepared.contains({up_type, uint32_t(up_layout)}) &&
        p.verify_nax_moe_prepared.contains({down_type, uint32_t(down_layout)}),
        "verify NAX unioned MoE requires prepared active verify scope and direct batches");
    require(experts >= 1 && experts <= 512 && selected >= 1 && selected <= 16 && selected <= experts &&
        gate_scratch.rows >= selected && gate_scratch.rows % selected == 0 && gate_scratch.rows == down_output.rows,
        "verify NAX unioned MoE expert/selection geometry mismatch");
    const uint32_t tokens = gate_scratch.rows / selected;
    require(tokens >= 1 && tokens <= 8 && input.rows == tokens && down_output.columns == input.columns,
        "verify NAX unioned MoE token/column geometry mismatch");
    const uint32_t up_block_cols = (up_type == 12 || up_type == 13) ? 256u : 32u;
    const uint32_t up_block_bytes = up_type == 12 ? 144u : (up_type == 13 ? 176u : (up_type == 8 ? 34u : 24u));
    const uint32_t dn_block_cols = (down_type == 12 || down_type == 13) ? 256u : 32u;
    const uint32_t dn_block_bytes = down_type == 12 ? 144u : (down_type == 13 ? 176u : (down_type == 8 ? 34u : 24u));
    require(input.columns >= up_block_cols && input.columns <= 65536 && input.columns % up_block_cols == 0 &&
        gate_scratch.columns >= dn_block_cols && gate_scratch.columns <= 65536 && gate_scratch.columns % dn_block_cols == 0 &&
        input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        gate_scratch.stride_bytes >= uint64_t(gate_scratch.columns) * 4ull && gate_scratch.stride_bytes % 4 == 0 &&
        down_output.stride_bytes >= uint64_t(down_output.columns) * 4ull && down_output.stride_bytes % 4 == 0,
        "verify NAX unioned MoE matrix/stride geometry mismatch");
    require((up_layout == PrefillWeightLayout::gguf ||
            (up_layout == PrefillWeightLayout::q51_lanes_v1 && up_type == 7) ||
            (up_layout == PrefillWeightLayout::q8_lanes_v1 && up_type == 8 && (input.columns < 512 || input.columns % 256 == 0))) &&
        (down_layout == PrefillWeightLayout::gguf ||
            (down_layout == PrefillWeightLayout::q51_lanes_v1 && down_type == 7) ||
            (down_layout == PrefillWeightLayout::q8_lanes_v1 && down_type == 8 && (gate_scratch.columns < 512 || gate_scratch.columns % 256 == 0))),
        "verify NAX unioned MoE layout/encoding mismatch");
    const uint32_t up_row_bytes = (input.columns / up_block_cols) * up_block_bytes;
    const uint64_t up_expert_stride = uint64_t(up_row_bytes) * gate_scratch.columns;
    const uint64_t up_matrix_bytes = checked_mul(up_expert_stride, experts);
    const uint32_t dn_row_bytes = (gate_scratch.columns / dn_block_cols) * dn_block_bytes;
    const uint64_t dn_expert_stride = uint64_t(dn_row_bytes) * down_output.columns;
    const uint64_t dn_matrix_bytes = checked_mul(dn_expert_stride, experts);
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t gate_bytes = checked_add(checked_mul(uint64_t(gate_scratch.rows - 1), gate_scratch.stride_bytes), uint64_t(gate_scratch.columns) * 4ull);
    const uint64_t down_bytes = checked_add(checked_mul(uint64_t(down_output.rows - 1), down_output.stride_bytes), uint64_t(down_output.columns) * 4ull);
    const uint64_t indices_bytes = uint64_t(gate_scratch.rows) * 4ull;
    const std::array bindings{gate_matrices, up_matrices, down_matrices, input.view, gate_scratch.view, down_output.view, indices};
    const std::array required{up_matrix_bytes, up_matrix_bytes, dn_matrix_bytes, input_bytes, gate_bytes, down_bytes, indices_bytes};
    for (size_t i = 0; i < bindings.size(); ++i) {
        static_cast<void>(p.allocation(bindings[i], i == 4 || i == 5));
        require(bindings[i].bytes >= required[i] && bindings[i].offset % 4 == 0,
            "verify NAX unioned MoE buffer span or alignment is invalid");
    }
    p.prefill_weight(gate_matrices, up_layout, up_type, input.columns, gate_scratch.columns, experts);
    p.prefill_weight(up_matrices, up_layout, up_type, input.columns, gate_scratch.columns, experts);
    p.prefill_weight(down_matrices, down_layout, down_type, gate_scratch.columns, down_output.columns, experts);
    validate_disjoint(std::array{gate_matrices, up_matrices, down_matrices, input.view, indices, p.verify_nax_moe_meta},
        std::array{gate_scratch.view, down_output.view});
    separate(down_output.view, {gate_scratch.view});

    const bool use_m8 = p.verify_nax_m8;
    const uint32_t grid_x_gu = use_m8 ? ((gate_scratch.columns + 31u) / 32u) : ((gate_scratch.columns + 7u) / 8u);
    const uint32_t dn_rows_per_tg = use_m8 ? 64u : 16u;
    const uint32_t grid_x_dn = (down_output.columns + dn_rows_per_tg - 1u) / dn_rows_per_tg;
    const uint32_t max_tile_routes = 8u;
    const NaxVerifyMoeUnionArgs uargs{experts, selected, tokens, grid_x_gu, grid_x_dn, max_tile_routes, 0u, 0u};
    p.select("nax_verify_moe_build_union", uint32_t(-1), {experts, selected, tokens, max_tile_routes}, 32);
    p.bind(indices, 0); p.bind(p.verify_nax_moe_meta, 1, true); p.parameter(uargs, 2);
    p.dispatch(true, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));

    const uint32_t moe_threads = use_m8 ? 64u : 128u;
    const char* gu_kernel = use_m8 ? "nax_m8_moe_gate_up_swiglu" : "nax_verify_moe_gate_up_swiglu";
    const char* dn_kernel = use_m8 ? "nax_m8_moe_down" : "nax_verify_moe_down";

    const NaxVerifyMoeArgs gu_args{input.columns, gate_scratch.columns, up_row_bytes, experts, selected, tokens,
        1u, 1u, up_expert_stride, input.stride_bytes, gate_scratch.stride_bytes};
    p.select(gu_kernel, up_type, {input.columns, gate_scratch.columns, experts, tokens}, moe_threads, uint32_t(up_layout));
    p.bind(gate_matrices, 0); p.bind(up_matrices, 1); p.bind(input.view, 2); p.bind(gate_scratch.view, 3, true);
    p.parameter(gu_args, 4); p.bind(p.verify_nax_moe_meta, 5);
    p.dispatch_indirect(p.verify_nax_moe_meta, 0, MTLSizeMake(moe_threads, 1, 1));

    const NaxVerifyMoeArgs dn_args{gate_scratch.columns, down_output.columns, dn_row_bytes, experts, selected, tokens,
        0u, 0u, dn_expert_stride, gate_scratch.stride_bytes, down_output.stride_bytes};
    p.select(dn_kernel, down_type, {gate_scratch.columns, down_output.columns, experts, tokens}, moe_threads, uint32_t(down_layout));
    p.bind(down_matrices, 0); p.bind(gate_scratch.view, 1); p.bind(down_output.view, 2, true);
    p.parameter(dn_args, 3); p.bind(p.verify_nax_moe_meta, 4);
    p.dispatch_indirect(p.verify_nax_moe_meta, 16, MTLSizeMake(moe_threads, 1, 1));
}
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
void MetalGraph::configure_prefill_hc_down_f32(bool enabled) {
    auto& p = *impl_; p.idle();
    require(p.dense_mm_prepared.empty(), "HC down batching is fixed before dense preparation");
#ifndef TOKSPUR_PRIVATE_Q8_MPP
    require(!enabled, "HC down batching requires the private Q8 TensorOps build");
#endif
    require(!enabled || p.prefill_dense_f32, "HC down batching requires the FP32 dense policy");
#if defined(TOKSPUR_PRIVATE_DENSE_Q8_STAGED) && !defined(TOKSPUR_PRIVATE_HC_DOWN_STAGED)
    require(!enabled || !p.prefill_dense_staged_f32, "staged dense refuses the legacy HC down experiment");
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    require(!enabled || !p.hc_down_window_enabled, "HC down batching and exact HC down window are exclusive");
#endif
    p.prefill_hc_down_f32 = enabled;
}
bool MetalGraph::prefill_hc_down_f32() const { impl_->thread_check(); return impl_->prefill_hc_down_f32; }
bool MetalGraph::prefill_hc_down_f32_rows(uint32_t rows) const {
    const auto& p = *impl_; p.thread_check();
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
    if (p.route_sink && p.route_phase == 2) return p.prefill_dense_staged_f32 && rows >= 2 && rows <= 512;
#endif
    if (!p.prefill_hc_down_f32 || !p.prefill_dense_f32) return false;
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    if (p.prefill_dense_staged_f32) return rows >= 2 && rows <= 512;
#endif
    return rows == 512;
}
void MetalGraph::dense_hc_down_f32(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output) {
    auto& p = *impl_; p.active();
    require(prefill_hc_down_f32_rows(input.rows) && output.rows == input.rows && input.columns == 10240 && output.columns == 320,
        "HC down batching requires its explicit selected row range, model geometry and FP32 policy");
    require(p.prefill_hc_down_f32_encoded < UINT64_MAX, "HC down encoded call counter overflow");
    dense_mm_q8(PrefillWeightLayout::gguf, matrix, input, output);
    ++p.prefill_hc_down_f32_encoded;
}
uint64_t MetalGraph::prefill_hc_down_f32_encoded_calls() const { impl_->thread_check(); return impl_->prefill_hc_down_f32_encoded; }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
void MetalGraph::configure_moe_router_f32_batch(bool enabled) {
    auto& p=*impl_; p.idle();
    require(!p.moe_router_f32_prepared && !p.moe_router_f32_locked, "router FP32 batch selection is fixed before preparation or any batch");
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    require(!enabled || !p.router_window_enabled, "router FP32 batch and router window are exclusive");
#endif
    p.moe_router_f32_batch=enabled;
}
bool MetalGraph::moe_router_f32_batch_enabled() const { impl_->thread_check(); return impl_->moe_router_f32_batch; }
const char* MetalGraph::moe_router_f32_projection_kernel() const {
    impl_->thread_check(); if(impl_->moe_router_f32_batch)return moe_router_f32_kernel;
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    if(impl_->router_window_enabled)return moe_router_window_kernel;
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
    if(impl_->native_plain_loads)return "graph_native_plain_gemv";
#endif
    return "graph_quantized_gemv";
}
void MetalGraph::prepare_moe_router_f32_batch() {
    auto& p=*impl_; p.idle();
    if (p.moe_router_f32_batch) {
        require([p.device supportsFamily:MTLGPUFamilyMetal4], "router FP32 batch requires Metal 4 support");
        const auto pipeline=p.pipeline(moe_router_f32_kernel);
        require(pipeline.staticThreadgroupMemoryLength<=p.device.maxThreadgroupMemoryLength, "router FP32 batch threadgroup memory capability");
    }
    p.moe_router_f32_prepared=true;
}
void MetalGraph::moe_router_f32_batch(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output) {
    auto& p=*impl_; p.active();
    require((p.moe_router_f32_batch
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        || (p.route_sink && p.route_phase == 3)
#endif
        ) && p.moe_router_f32_prepared && !p.recording && !p.compatible_pipelines, "router FP32 batch requires explicit idle selection/preparation and direct batches");
    require(input.rows==output.rows, "router FP32 batch row count mismatch");
    const auto plan=moe_router_f32_plan(input.columns,output.columns,input.rows,input.stride_bytes,output.stride_bytes);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view,true));
    require(matrix.bytes==plan.matrix_bytes && input.view.bytes>=plan.input_bytes && output.view.bytes>=plan.output_bytes && matrix.offset%4==0 && input.view.offset%4==0 && output.view.offset%4==0, "router FP32 batch view extent/alignment");
    p.prefill_weight(matrix,PrefillWeightLayout::gguf,0,input.columns,output.columns,1);
    separate(output.view,{matrix,input.view});
    p.select(moe_router_f32_kernel,uint32_t(-1),{input.columns,output.columns,input.rows,0});
    p.bind(matrix,0); p.bind(input.view,1); p.bind(output.view,2,true); p.parameter(plan.args,3);
    p.groups(MTLSizeMake(plan.grid_x,plan.grid_y,1));
}
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
void MetalGraph::configure_prefill_router_window(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!enabled, "this optimisation was removed because its source could not be verified");
    require(!p.router_window_prepared && !p.router_window_locked && !p.router_window_configured, "router window selection is fixed before preparation or any batch");
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    require(!enabled || !p.moe_router_f32_batch, "router window and router FP32 batch are exclusive");
#endif
#ifndef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    require(!enabled || !p.prefill_hc_down_f32, "router window refuses the legacy HC down experiment");
#endif
    p.router_window_enabled = enabled;
    p.router_window_configured = true;
}
bool MetalGraph::prefill_router_window() const { impl_->thread_check(); return impl_->router_window_enabled; }
void MetalGraph::configure_prefill_router_fast(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.router_window_prepared && !p.router_window_locked, "router fast precision is fixed before preparation or any batch");
    p.router_window_fast = enabled;
}
bool MetalGraph::prefill_router_fast() const { impl_->thread_check(); return impl_->router_window_fast; }
const char* MetalGraph::moe_router_window_projection_kernel() const {
    impl_->thread_check();
    if (impl_->router_window_enabled) {
        if (impl_->prefill_dense_nax || impl_->prefill_grouped_nax) {
            return impl_->router_window_fast ? "nax_router_bf16_fast" : "nax_router_f32";
        }
        return moe_router_window_kernel;
    }
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    if (impl_->moe_router_f32_batch) return moe_router_f32_kernel;
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
    if (impl_->native_plain_loads) return "graph_native_plain_gemv";
#endif
    return "graph_quantized_gemv";
}
std::array<uint64_t, 4> MetalGraph::router_window_encoded() const { impl_->thread_check(); return impl_->router_window_encoded; }
void MetalGraph::prepare_moe_router_window() {
    auto& p = *impl_; p.idle();
    if (p.router_window_fast) {
        require(p.router_window_enabled && (p.prefill_dense_nax || p.prefill_grouped_nax),
            "fast BF16 router requires enabled router window and NAX prefill");
    }
    if (p.router_window_enabled) {
        for (const auto* name : {moe_router_window_kernel, moe_route_rows_kernel, moe_shared_router_rows_kernel, moe_mixture_rows_kernel}) {
            const auto pipeline = p.pipeline(name);
            require(pipeline.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength, "router window threadgroup memory capability");
        }
        if (p.prefill_dense_nax || p.prefill_grouped_nax) {
            static_cast<void>(p.pipeline("nax_f32_to_bf16"));
            const auto nax_pipe = p.pipeline(p.router_window_fast ? "nax_router_bf16_fast" : "nax_router_f32");
            require(nax_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength, "NAX router threadgroup memory capability");
            const uint64_t min_bf16 = p.prefill_grouped_nax ? 188'743'680ull : 41'943'040ull;
            if (p.nax_bf16_scratch.bytes < min_bf16) {
                if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
                p.nax_bf16_scratch = zeros(min_bf16, "nax_bf16_scratch");
            }
        }
        if (!p.router_margins_scratch.bytes) {
            p.router_margins_scratch = p.allocate(2048 * sizeof(float), "router_margins_scratch", false, nullptr);
        }
    }
    p.router_window_prepared = true;
}
void MetalGraph::moe_router_f32_window(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output) {
    struct NaxConvertBf16Args {
        uint32_t columns, rows, padded_rows, reserved;
        uint64_t input_stride_bytes;
    };
    static_assert(sizeof(NaxConvertBf16Args) == 24, "NAX convert BF16 ABI");
    auto& p = *impl_; p.active();
    const bool v_nax = p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope;
    require(((p.router_window_enabled && p.router_window_prepared) || v_nax) && !p.recording && !p.compatible_pipelines, "router window requires explicit idle selection/preparation and direct batches");
    require(input.rows == output.rows, "router window row count mismatch");
    const bool use_nax = (p.prefill_dense_nax || p.prefill_grouped_nax) && !v_nax;
    const auto plan = moe_router_window_plan(input.columns, output.columns, input.rows, input.stride_bytes, output.stride_bytes, use_nax);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes == plan.matrix_bytes && input.view.bytes >= plan.input_bytes && output.view.bytes >= plan.output_bytes &&
        matrix.offset % 4 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0, "router window view extent/alignment");
    p.prefill_weight(matrix, PrefillWeightLayout::gguf, 0, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    require(p.router_window_encoded[0] < UINT64_MAX, "router window projection encoded counter exhausted");
    if (use_nax) {
        const uint32_t padded_rows = ((input.rows + 63) / 64) * 64;
        const uint64_t bf16_bytes = uint64_t(padded_rows) * input.columns * 2;
        require(p.nax_bf16_scratch.bytes >= bf16_bytes, "NAX router BF16 scratch too small");
        const auto bf16 = slice(p.nax_bf16_scratch, 0, bf16_bytes);
        const NaxConvertBf16Args conv{input.columns, input.rows, padded_rows, 0, input.stride_bytes};
        p.select("nax_f32_to_bf16", uint32_t(-1), {input.columns, input.rows, padded_rows, 0});
        p.bind(input.view, 0); p.bind(bf16, 1, true); p.parameter(conv, 2);
        p.threads(padded_rows * (input.columns / 4));
        if (p.prefill_dense_nax && !p.verify_nax_scope) {
            p.nax_dense_bf16_valid = true;
            p.nax_dense_input_slot = input.view.slot;
            p.nax_dense_input_offset = input.view.offset;
            p.nax_dense_input_end = checked_add(input.view.offset, plan.input_bytes);
            p.nax_dense_rows = input.rows;
            p.nax_dense_padded_rows = padded_rows;
            p.nax_dense_columns = input.columns;
            p.nax_dense_stride = input.stride_bytes;
        }
        p.select(p.router_window_fast ? "nax_router_bf16_fast" : "nax_router_f32", uint32_t(-1), {input.columns, output.columns, input.rows, 0});
        p.bind(matrix, 0); p.bind(bf16, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
        p.groups(MTLSizeMake(16, (input.rows + 63) / 64, 1));
    } else if (v_nax && input.rows <= 8u) {
        p.select("graph_moe_router_f32_m8", uint32_t(-1), {input.columns, output.columns, input.rows, 0});
        p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
        p.groups(MTLSizeMake((output.columns + 7u) / 8u, 1, 1));
    } else {
        p.select(moe_router_window_kernel, uint32_t(-1), {input.columns, output.columns, input.rows, 0});
        p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
        p.groups(MTLSizeMake(plan.grid_x, plan.grid_y, 1));
    }
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.router_window_encoded[0];
}
void MetalGraph::moe_route_rows(GpuView logits, GpuView indices, GpuView weights, uint32_t experts, uint32_t selected, uint32_t tokens, RouterMode mode, bool normalise) {
    auto& p = *impl_; p.active();
    const bool v_nax = p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope;
    require(((p.router_window_enabled && p.router_window_prepared) || v_nax) && !p.recording && !p.compatible_pipelines, "router row top-k requires explicit idle selection/preparation and direct batches");
    const uint32_t max_tokens = (p.prefill_dense_nax || p.prefill_grouped_nax) ? 2048u : 512u;
    require(experts == 512 && selected && selected <= 64 && selected <= experts && tokens >= 2 && tokens <= max_tokens && uint32_t(mode) <= 1 &&
        p.floats(logits) == uint64_t(tokens) * experts && p.floats(indices, true) == uint64_t(tokens) * selected && p.floats(weights, true) == uint64_t(tokens) * selected &&
        logits.offset % 4 == 0 && indices.offset % 4 == 0 && weights.offset % 4 == 0, "router row top-k geometry/mode mismatch");
    validate_disjoint(std::array{logits}, std::array{indices, weights});
    require(p.router_window_encoded[1] < UINT64_MAX, "router row top-k encoded counter exhausted");
    p.select(moe_route_rows_kernel, uint32_t(-1), {experts, selected, tokens, uint32_t(mode)});
    p.bind(logits, 0); p.bind(indices, 1, true); p.bind(weights, 2, true);
    p.parameter(std::array<uint32_t, 4>{experts, selected, uint32_t(mode), normalise}, 3);
    p.parameter(tokens, 4);
    if (p.router_margins_scratch.bytes) {
        p.bind(p.router_margins_scratch, 5, true);
    }
    p.groups(MTLSizeMake(1, tokens, 1));
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    p.carrier_route(indices, weights, tokens, selected, experts);
#endif
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.router_window_encoded[1];
}
void MetalGraph::moe_shared_router_rows(GpuView weights, GpuRowsF32 input, GpuView output) {
    auto& p = *impl_; p.active();
    const bool v_nax = p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope;
    require(((p.router_window_enabled && p.router_window_prepared) || v_nax) && !p.recording && !p.compatible_pipelines, "shared router rows require explicit idle selection/preparation and direct batches");
    const uint32_t max_tokens = (p.prefill_dense_nax || p.prefill_grouped_nax) ? 2048u : 512u;
    require(input.columns == 2560 && input.rows >= 2 && input.rows <= max_tokens && input.stride_bytes >= 2560 * 4 && input.stride_bytes % 16 == 0, "shared router rows input geometry");
    const auto input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), 2560ull * 4);
    static_cast<void>(p.allocation(weights)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output, true));
    require(p.floats(weights) == 2560 && input.view.bytes >= input_bytes && p.floats(output, true) == input.rows &&
        weights.offset % 4 == 0 && input.view.offset % 4 == 0 && output.offset % 4 == 0, "shared router rows view extent/alignment");
    p.prefill_weight(weights, PrefillWeightLayout::gguf, 0, 2560, 1, 1);
    validate_disjoint(std::array{weights, input.view}, std::array{output});
    require(p.router_window_encoded[2] < UINT64_MAX, "shared router rows encoded counter exhausted");
    const MoeRouterWindowArgs args{2560, 1, input.rows, p.small_gemv_splits, input.stride_bytes, 4};
    p.select(moe_shared_router_rows_kernel, uint32_t(-1), {2560, 1, input.rows, p.small_gemv_splits});
    p.bind(weights, 0); p.bind(input.view, 1); p.bind(output, 2, true); p.parameter(args, 3);
    p.groups(MTLSizeMake(input.rows, 1, 1));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.router_window_encoded[2];
}
void MetalGraph::moe_mixture_rows(GpuView experts, GpuView weights, GpuView shared, GpuView shared_logit, GpuRowsF32 output, uint32_t selected) {
    auto& p = *impl_; p.active();
    const bool v_nax = p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope;
    require(((p.router_window_enabled && p.router_window_prepared) || v_nax) && !p.recording && !p.compatible_pipelines, "mixture rows require explicit idle selection/preparation and direct batches");
    const uint32_t max_tokens = (p.prefill_dense_nax || p.prefill_grouped_nax) ? 2048u : 512u;
    require(output.columns == 2560 && selected && selected <= 64 && output.rows >= 2 && output.rows <= max_tokens && output.stride_bytes >= 2560 * 4 && output.stride_bytes % 16 == 0, "mixture rows output geometry");
    const auto output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), 2560ull * 4);
    static_cast<void>(p.allocation(experts)); static_cast<void>(p.allocation(weights)); static_cast<void>(p.allocation(shared)); static_cast<void>(p.allocation(shared_logit)); static_cast<void>(p.allocation(output.view, true));
    require(experts.bytes == uint64_t(output.rows) * selected * 2560 * 4 && p.floats(weights) == uint64_t(output.rows) * selected &&
        shared.bytes == uint64_t(output.rows) * 2560 * 4 && p.floats(shared_logit) == output.rows && output.view.bytes >= output_bytes &&
        experts.offset % 4 == 0 && weights.offset % 4 == 0 && shared.offset % 4 == 0 && shared_logit.offset % 4 == 0 && output.view.offset % 4 == 0, "mixture rows view extent/alignment");
    validate_disjoint(std::array{experts, weights, shared, shared_logit}, std::array{output.view});
    require(p.router_window_encoded[3] < UINT64_MAX, "mixture rows encoded counter exhausted");
    const MoeRouterWindowArgs args{2560, selected, output.rows, 0, 0, output.stride_bytes};
    p.select(moe_mixture_rows_kernel, uint32_t(-1), {2560, selected, output.rows, 0});
    p.bind(experts, 0); p.bind(weights, 1); p.bind(shared, 2); p.bind(shared_logit, 3); p.bind(output.view, 4, true); p.parameter(args, 5);
    p.threads(output.rows * (2560 / 4));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.router_window_encoded[3];
}
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
void MetalGraph::configure_prefill_hc_down_window(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.hc_down_window_prepared && !p.hc_down_window_locked && !p.hc_down_window_configured, "HC down window selection is fixed before preparation or any batch");
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    require(!enabled || !p.prefill_hc_down_f32, "HC down window and reassociating HC down batching are exclusive");
#endif
    p.hc_down_window_enabled = enabled;
    p.hc_down_window_configured = true;
}
bool MetalGraph::prefill_hc_down_window() const { impl_->thread_check(); return impl_->hc_down_window_enabled; }
uint64_t MetalGraph::hc_down_window_encoded() const { impl_->thread_check(); return impl_->hc_down_window_encoded; }
void MetalGraph::prepare_hc_down_window() {
    auto& p = *impl_; p.idle();
    if (p.hc_down_window_enabled) {
        const auto pipeline = p.pipeline("graph_hc_down_q8_window", uint32_t(-1), 512);
        require(pipeline.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength, "HC down window threadgroup memory capability");
        if (p.prefill_dense_nax) {
            static_cast<void>(p.pipeline("nax_f32_to_bf16"));
            for (const uint32_t layout : {0u, 1u}) {
                const auto m32 = p.pipeline("nax_prefill_dense_q8_m32", 8, 128, false, layout);
                const auto m8 = p.pipeline("nax_prefill_dense_q8_m8", 8, 128, false, layout);
                require(m32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
                    m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
                    "NAX HC down window threadgroup memory capability");
            }
            const uint64_t min_bf16 = p.prefill_grouped_nax ? 188'743'680ull : 41'943'040ull;
            if (p.nax_bf16_scratch.bytes < min_bf16) {
                if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
                p.nax_bf16_scratch = zeros(min_bf16, "nax_bf16_scratch");
            }
            const auto t64_m32 = p.pipeline("nax_prefill_dense_q8_t64_m32", uint32_t(-1), 128);
            const auto t64_m8 = p.pipeline("nax_prefill_dense_q8_t64_m8", uint32_t(-1), 128);
            require(t64_m32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
                t64_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
                "NAX HC down window Q8_tile64 threadgroup memory capability");
            p.q8_tile64_ready = true;
            p.ensure_q8_tile64_all();
        }
    }
    p.hc_down_window_prepared = true;
}
void MetalGraph::hc_down_q8_window(GpuView weights, GpuRowsF32 input, GpuRowsF32 output) {
    struct HcDownWindowArgs {
        uint32_t columns, rows, tokens, reserved;
        uint64_t input_stride_bytes, output_stride_bytes;
    };
    struct NaxConvertBf16Args {
        uint32_t columns, rows, padded_rows, reserved;
        uint64_t input_stride_bytes;
    };
    static_assert(sizeof(HcDownWindowArgs) == 32, "HC down window argument ABI");
    static_assert(sizeof(NaxConvertBf16Args) == 24, "NAX convert BF16 ABI");
    auto& p = *impl_; p.active();
    require(p.hc_down_window_enabled && p.hc_down_window_prepared && !p.recording && !p.compatible_pipelines, "HC down window requires explicit idle selection/preparation and direct batches");
    require(p.hc_exact != HcExactMode::off && !p.hc_q8.splits, "HC down window requires the combined unsplit exact HC policy");
    const uint32_t max_rows = p.prefill_dense_nax ? 2048u : 512u;
    require(input.columns == 10240 && output.columns == 320 && input.rows == output.rows && input.rows >= 1 && input.rows <= max_rows &&
        input.stride_bytes >= 10240ull * 4 && output.stride_bytes >= 320ull * 4 && input.stride_bytes % 4 == 0 && output.stride_bytes % 4 == 0,
        "HC down window geometry/stride mismatch");
    const uint64_t matrix_bytes = uint64_t(10240 / 32) * 34 * 320;
    const auto input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), 10240ull * 4);
    const auto output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), 320ull * 4);
    static_cast<void>(p.allocation(weights)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(weights.bytes == matrix_bytes && input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        weights.offset % 4 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0, "HC down window view extent/alignment");
    const auto layout = prefill_weight_layout(weights);
    require(layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1,
        "HC down window requires gguf or q8_lanes_v1 layout");
    p.prefill_weight(weights, layout, 8, 10240, 320, 1);
    separate(output.view, {weights, input.view});
    require(p.hc_down_window_encoded < UINT64_MAX, "HC down window encoded counter exhausted");
    if (p.prefill_dense_nax) {
        const uint32_t padded_rows = ((input.rows + 31u) / 32u) * 32u;
        const uint64_t bf16_bytes = uint64_t(padded_rows) * 10240ull * 2ull;
        require(p.nax_bf16_scratch.bytes >= bf16_bytes, "NAX HC down BF16 scratch too small");
        const auto bf16 = slice(p.nax_bf16_scratch, 0, bf16_bytes);
        const NaxConvertBf16Args conv{10240u, input.rows, padded_rows, 0, input.stride_bytes};
        p.select("nax_f32_to_bf16", uint32_t(-1), {10240u, input.rows, padded_rows, 0});
        p.bind(input.view, 0); p.bind(bf16, 1, true); p.parameter(conv, 2);
        p.threads(padded_rows * (10240u / 4u));
        p.nax_dense_bf16_valid = true;
        p.nax_dense_input_slot = input.view.slot;
        p.nax_dense_input_offset = input.view.offset;
        p.nax_dense_input_end = checked_add(input.view.offset, input_bytes);
        p.nax_dense_rows = input.rows;
        p.nax_dense_padded_rows = padded_rows;
        p.nax_dense_columns = 10240u;
        p.nax_dense_stride = input.stride_bytes;
        const GroupedMmArgs nax_args{10240u, 320u, 10240u / 32u * 34u, 1u, 1u, input.rows, 1u, (input.rows + 31u) / 32u,
            matrix_bytes, input.stride_bytes, output.stride_bytes};
        const bool use_m32 = input.rows >= 128u;
        const uint32_t gx = (320u + 63u) / 64u;
        const uint32_t gy = use_m32 ? (input.rows + 127u) / 128u : (input.rows + 31u) / 32u;
        if (const auto* entry = p.q8_tile64_entry(weights, 10240u, 320u)) {
            const char* kname = use_m32 ? "nax_prefill_dense_q8_t64_m32" : "nax_prefill_dense_q8_t64_m8";
            p.select(kname, uint32_t(-1), {10240u, 320u, input.rows, uint32_t(layout)}, 128);
            p.bind(entry->plane0, 0); p.bind(entry->meta, 1); p.bind(bf16, 2); p.bind(output.view, 3, true); p.parameter(nax_args, 4);
            p.groups(MTLSizeMake(gx, gy, 1));
            if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.hc_down_window_encoded;
            return;
        }
        const char* kname = use_m32 ? "nax_prefill_dense_q8_m32" : "nax_prefill_dense_q8_m8";
        p.select(kname, 8, {10240u, 320u, input.rows, uint32_t(layout)}, 128, uint32_t(layout));
        p.bind(weights, 0); p.bind(bf16, 1); p.bind(output.view, 2, true); p.parameter(nax_args, 3);
        p.groups(MTLSizeMake(gx, gy, 1));
        if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.hc_down_window_encoded;
        return;
    }
    const uint32_t layout_flag = (layout == PrefillWeightLayout::q8_lanes_v1) ? 1 : 0;
    const HcDownWindowArgs args{input.columns, output.columns, input.rows, layout_flag, input.stride_bytes, output.stride_bytes};
    p.select("graph_hc_down_q8_window", uint32_t(-1), {input.columns, output.columns, input.rows, 0}, 512);
    p.bind(weights, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.dispatch(true, MTLSizeMake(10, (input.rows + 15) / 16, 1), MTLSizeMake(512, 1, 1));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.hc_down_window_encoded;
}
#endif
void MetalGraph::prepare_dense_mm_q8(PrefillWeightLayout layout) {
    auto& p = *impl_; p.idle();
    require([p.device supportsFamily:MTLGPUFamilyMetal4], "dense prefill requires Metal 4 support");
#ifndef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(layout == PrefillWeightLayout::gguf, "packed prefill requires the lane-layout build");
#endif
    if (p.prefill_dense_nax) {
        require(layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1,
            "NAX prefill dense requires GGUF or q8_lanes_v1 layout");
        static_cast<void>(p.pipeline("nax_f32_to_bf16"));
        const auto m32 = p.pipeline("nax_prefill_dense_q8_m32", 8, 128, false, uint32_t(layout));
        const auto m8 = p.pipeline("nax_prefill_dense_q8_m8", 8, 128, false, uint32_t(layout));
        const auto sw32 = p.pipeline("nax_prefill_dense_q8_swiglu_m32", 8, 128, false, uint32_t(layout));
        const auto sw8 = p.pipeline("nax_prefill_dense_q8_swiglu_m8", 8, 128, false, uint32_t(layout));
        const auto bf32 = p.pipeline("nax_prefill_dense_bf16_m32", uint32_t(-1), 128);
        const auto bf8 = p.pipeline("nax_prefill_dense_bf16_m8", uint32_t(-1), 128);
        require(m32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            sw32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            sw8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            bf32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            bf8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "NAX prefill dense threadgroup memory exceeds device capability");
        static_cast<void>(p.pipeline("nax_ple_gate_rows"));
        static_cast<void>(p.pipeline("nax_ple_conv_write_rows"));
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        prepare_dense_q8_staged(false);
#endif
        const uint64_t min_bf16 = p.prefill_grouped_nax ? 188'743'680ull : 41'943'040ull;
        if (p.nax_bf16_scratch.bytes < min_bf16) {
            if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
            p.nax_bf16_scratch = zeros(min_bf16, "nax_bf16_scratch");
        }
        const auto t64_m32 = p.pipeline("nax_prefill_dense_q8_t64_m32", uint32_t(-1), 128);
        const auto t64_m8 = p.pipeline("nax_prefill_dense_q8_t64_m8", uint32_t(-1), 128);
        const auto t64_sw32 = p.pipeline("nax_prefill_dense_q8_swiglu_t64_m32", uint32_t(-1), 128);
        const auto t64_sw8 = p.pipeline("nax_prefill_dense_q8_swiglu_t64_m8", uint32_t(-1), 128);
        require(t64_m32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            t64_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            t64_sw32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            t64_sw8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "NAX prefill Q8_tile64 threadgroup memory exceeds device capability");
        p.q8_tile64_ready = true;
        p.ensure_q8_tile64_all();
        p.dense_mm_prepared.insert(uint32_t(layout));
        return;
    }
    if (p.prefill_dense_f32) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        if (p.prefill_dense_staged_f32) {
            require(layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1,
                "FP32 prefill dense requires GGUF or q8_lanes_v1 layout");
            prepare_dense_q8_staged(false); p.dense_mm_prepared.insert(uint32_t(layout)); return;
        }
#endif
        require(layout == PrefillWeightLayout::gguf, "FP32 prefill dense requires original GGUF layout");
#ifdef TOKSPUR_PRIVATE_Q8_MPP
        prepare_q8_mpp(Q8MppLayout::single_wide);
        p.dense_mm_prepared.insert(uint32_t(layout)); return;
#endif
    }
    if (layout == PrefillWeightLayout::gguf) static_cast<void>(p.pipeline("graph_q8_serial_batch"));
    const auto pipeline = p.pipeline(dense_mm_q8_kernel(layout), 8, 128, false, uint32_t(layout));
    require(pipeline.staticThreadgroupMemoryLength + 8192 <= p.device.maxThreadgroupMemoryLength, "dense prefill threadgroup memory exceeds device capability");
    p.dense_mm_prepared.insert(uint32_t(layout));
}
void MetalGraph::dense_mm_q8(PrefillWeightLayout layout, GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool reuse_bf16) {
    struct NaxConvertBf16Args {
        uint32_t columns, rows, padded_rows, reserved;
        uint64_t input_stride_bytes;
    };
    static_assert(sizeof(NaxConvertBf16Args) == 24, "NAX convert BF16 ABI");
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.dense_mm_prepared.contains(uint32_t(layout)), "dense prefill requires idle preparation and direct batches");
    require(input.rows == output.rows, "dense prefill row count mismatch");
    const auto plan = dense_mm_q8_plan(layout, input.columns, output.columns, input.rows, input.stride_bytes, output.stride_bytes, p.prefill_dense_nax);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes >= plan.matrix_bytes && input.view.bytes >= plan.input_bytes && output.view.bytes >= plan.output_bytes, "dense prefill buffer span is too small");
    p.prefill_weight(matrix, layout, 8, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    if (p.prefill_dense_nax) {
        require(layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1,
            "NAX prefill dense requires GGUF or q8_lanes_v1 layout");
        if (input.columns % 64 == 0) {
            const uint32_t padded_rows = ((input.rows + 31) / 32) * 32;
            const uint64_t bf16_bytes = checked_mul(checked_mul(uint64_t(padded_rows), uint64_t(input.columns)), 2ull);
            require(p.nax_bf16_scratch.bytes >= bf16_bytes, "NAX prefill dense BF16 scratch too small");
            const auto bf16 = slice(p.nax_bf16_scratch, 0, bf16_bytes);
            if (reuse_bf16) {
                require(can_reuse_nax_dense_bf16(input), "NAX prefill dense BF16 reuse requires matching valid input");
            } else {
                const NaxConvertBf16Args conv{input.columns, input.rows, padded_rows, 0, input.stride_bytes};
                p.select("nax_f32_to_bf16", uint32_t(-1), {input.columns, input.rows, padded_rows, 0});
                p.bind(input.view, 0); p.bind(bf16, 1, true); p.parameter(conv, 2);
                p.threads(padded_rows * (input.columns / 4));
                p.nax_dense_bf16_valid = true;
                p.nax_dense_input_slot = input.view.slot;
                p.nax_dense_input_offset = input.view.offset;
                p.nax_dense_input_end = checked_add(input.view.offset, plan.input_bytes);
                p.nax_dense_rows = input.rows;
                p.nax_dense_padded_rows = padded_rows;
                p.nax_dense_columns = input.columns;
                p.nax_dense_stride = input.stride_bytes;
            }
            const bool use_m32 = input.rows >= 128;
            const uint32_t gx = (output.columns + 63) / 64;
            const uint32_t gy = use_m32 ? (input.rows + 127) / 128 : (input.rows + 31) / 32;
            if (const auto* entry = p.q8_tile64_entry(matrix, input.columns, output.columns)) {
                const char* kname = use_m32 ? "nax_prefill_dense_q8_t64_m32" : "nax_prefill_dense_q8_t64_m8";
                p.select(kname, uint32_t(-1), {input.columns, output.columns, input.rows, uint32_t(layout)}, 128);
                p.bind(entry->plane0, 0); p.bind(entry->meta, 1); p.bind(bf16, 2); p.bind(output.view, 3, true); p.parameter(plan.args, 4);
                p.groups(MTLSizeMake(gx, gy, 1));
                return;
            }
            const char* kname = use_m32 ? "nax_prefill_dense_q8_m32" : "nax_prefill_dense_q8_m8";
            p.select(kname, 8, {input.columns, output.columns, input.rows, uint32_t(layout)}, 128, uint32_t(layout));
            p.bind(matrix, 0); p.bind(bf16, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
            p.groups(MTLSizeMake(gx, gy, 1));
            return;
        }
        require(!reuse_bf16, "NAX prefill dense BF16 reuse requires NAX column geometry");
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        if (input.rows <= 512 && input.stride_bytes == uint64_t(input.columns) * 4 &&
            output.stride_bytes == uint64_t(output.columns) * 4) {
            dense_q8_staged(slice(matrix, 0, plan.matrix_bytes), slice(input.view, 0, plan.input_bytes), slice(output.view, 0, plan.output_bytes),
                input.columns, output.columns, input.rows, false, layout);
            return;
        }
#endif
    }
    require(!reuse_bf16, "dense prefill BF16 reuse requires NAX prefill dense");
    if (p.prefill_dense_f32) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        if (p.prefill_dense_staged_f32) {
            require((layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1) &&
                input.stride_bytes == uint64_t(input.columns) * 4 &&
                output.stride_bytes == uint64_t(output.columns) * 4, "FP32 prefill dense requires tight rows");
            dense_q8_staged(slice(matrix, 0, plan.matrix_bytes), slice(input.view, 0, plan.input_bytes), slice(output.view, 0, plan.output_bytes),
                input.columns, output.columns, input.rows, false, layout); return;
        }
#endif
        require(layout == PrefillWeightLayout::gguf && input.stride_bytes == uint64_t(input.columns) * 4 &&
            output.stride_bytes == uint64_t(output.columns) * 4, "FP32 prefill dense requires raw weights and tight rows");
#ifdef TOKSPUR_PRIVATE_Q8_MPP
        q8_mpp(slice(matrix, 0, plan.matrix_bytes), slice(input.view, 0, plan.input_bytes), slice(output.view, 0, plan.output_bytes),
            input.columns, output.columns, input.rows, Q8MppLayout::single_wide); return;
#endif
    }
    if (layout == PrefillWeightLayout::gguf && input.stride_bytes == uint64_t(input.columns) * 4 &&
        output.stride_bytes == uint64_t(output.columns) * 4) {
        require(input.columns % 32 == 0, "serial Q8 prefill columns must be a whole number of blocks");
        p.select("graph_q8_serial_batch", uint32_t(-1), {input.columns, output.columns, input.rows, 0});
        p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true);
        p.parameter(std::array<uint32_t, 4>{input.columns, output.columns, input.columns / 32 * 34, input.rows}, 3);
        p.groups(MTLSizeMake((output.columns + 3) / 4, 1, 1));
        return;
    }
    p.select(plan.kernel, 8, {input.columns, output.columns, input.rows, uint32_t(layout)}, 128, uint32_t(layout));
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(plan.args, 3);
    [p.encoder setThreadgroupMemoryLength:plan.threadgroup_bytes atIndex:0];
    p.groups(MTLSizeMake(plan.grid_x, plan.grid_y, 1));
}
void MetalGraph::nax_dense_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool reuse_bf16) {
    struct NaxConvertBf16Args {
        uint32_t columns, rows, padded_rows, reserved;
        uint64_t input_stride_bytes;
    };
    static_assert(sizeof(NaxConvertBf16Args) == 24, "NAX convert BF16 ABI");
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.prefill_dense_nax && !p.dense_mm_prepared.empty(),
        "NAX prefill dense BF16 requires idle preparation, direct batches and enabled NAX dense policy");
    require(input.rows >= 1 && input.rows <= 2048 && input.rows == output.rows &&
        input.columns >= 64 && input.columns <= 65536 && input.columns % 64 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "NAX prefill dense BF16 geometry mismatch");
    require(input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0,
        "NAX prefill dense BF16 row stride mismatch");
    require(prefill_weight_layout(matrix) == PrefillWeightLayout::gguf,
        "NAX prefill dense BF16 requires original GGUF layout");
    const uint32_t row_bytes = input.columns * 2u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t input_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes >= matrix_bytes && input.view.bytes >= input_bytes && output.view.bytes >= output_bytes &&
        matrix.offset % 2 == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "NAX prefill dense BF16 buffer span or alignment is invalid");
    p.prefill_weight(matrix, PrefillWeightLayout::gguf, 30, input.columns, output.columns, 1);
    separate(output.view, {matrix, input.view});
    const uint32_t padded_rows = ((input.rows + 31u) / 32u) * 32u;
    const uint64_t bf16_bytes = checked_mul(checked_mul(uint64_t(padded_rows), uint64_t(input.columns)), 2ull);
    require(p.nax_bf16_scratch.bytes >= bf16_bytes, "NAX prefill dense BF16 scratch too small");
    const auto bf16 = slice(p.nax_bf16_scratch, 0, bf16_bytes);
    if (reuse_bf16) {
        require(can_reuse_nax_dense_bf16(input), "NAX prefill dense BF16 reuse requires matching valid input");
    } else {
        const NaxConvertBf16Args conv{input.columns, input.rows, padded_rows, 0, input.stride_bytes};
        p.select("nax_f32_to_bf16", uint32_t(-1), {input.columns, input.rows, padded_rows, 0});
        p.bind(input.view, 0); p.bind(bf16, 1, true); p.parameter(conv, 2);
        p.threads(padded_rows * (input.columns / 4u));
        p.nax_dense_bf16_valid = true;
        p.nax_dense_input_slot = input.view.slot;
        p.nax_dense_input_offset = input.view.offset;
        p.nax_dense_input_end = checked_add(input.view.offset, input_bytes);
        p.nax_dense_rows = input.rows;
        p.nax_dense_padded_rows = padded_rows;
        p.nax_dense_columns = input.columns;
        p.nax_dense_stride = input.stride_bytes;
    }
    const GroupedMmArgs args{input.columns, output.columns, row_bytes, 1u, 1u, input.rows, 1u, (input.rows + 31u) / 32u,
        matrix_bytes, input.stride_bytes, output.stride_bytes};
    const bool use_m32 = input.rows >= 64u;
    const char* kname = use_m32 ? "nax_prefill_dense_bf16_m32" : "nax_prefill_dense_bf16_m8";
    const uint32_t gy = use_m32 ? (input.rows + 127u) / 128u : (input.rows + 31u) / 32u;
    p.select(kname, uint32_t(-1), {input.columns, output.columns, input.rows, 30u}, 128);
    p.bind(matrix, 0); p.bind(bf16, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.groups(MTLSizeMake((output.columns + 63u) / 64u, gy, 1));
}
bool MetalGraph::can_nax_dense_q8_swiglu(PrefillWeightLayout l0, PrefillWeightLayout l1, uint32_t columns, uint32_t hidden, uint32_t batches) const {
    impl_->thread_check();
    if (!impl_->prefill_multiseg || !impl_->prefill_dense_nax || l0 != l1 ||
        batches < 1 || batches > 2048 || columns == 0 || columns % 64 != 0 || hidden == 0 || hidden % 32 != 0) {
        return false;
    }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    return l0 == PrefillWeightLayout::gguf || l0 == PrefillWeightLayout::q8_lanes_v1;
#else
    return l0 == PrefillWeightLayout::gguf;
#endif
}
void MetalGraph::nax_dense_q8_swiglu(PrefillWeightLayout l0, GpuView w0, PrefillWeightLayout l1, GpuView w1,
    GpuRowsF32 input, GpuRowsF32 output, bool reuse_bf16) {
    struct NaxConvertBf16Args {
        uint32_t columns, rows, padded_rows, reserved;
        uint64_t input_stride_bytes;
    };
    static_assert(sizeof(NaxConvertBf16Args) == 24, "NAX convert BF16 ABI");
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines &&
        can_nax_dense_q8_swiglu(l0, l1, input.columns, output.columns, input.rows) &&
        p.dense_mm_prepared.contains(uint32_t(l0)),
        "NAX fused dense SwiGLU requires idle preparation, direct batches and enabled multiseg NAX policy");
    require(input.rows == output.rows, "NAX fused dense SwiGLU row count mismatch");
    const auto plan = dense_mm_q8_plan(l0, input.columns, output.columns, input.rows, input.stride_bytes, output.stride_bytes, true);
    static_cast<void>(p.allocation(w0)); static_cast<void>(p.allocation(w1));
    static_cast<void>(p.allocation(input.view)); static_cast<void>(p.allocation(output.view, true));
    require(w0.bytes >= plan.matrix_bytes && w1.bytes >= plan.matrix_bytes &&
        input.view.bytes >= plan.input_bytes && output.view.bytes >= plan.output_bytes,
        "NAX fused dense SwiGLU buffer span is too small");
    p.prefill_weight(w0, l0, 8, input.columns, output.columns, 1);
    p.prefill_weight(w1, l1, 8, input.columns, output.columns, 1);
    separate(output.view, {w0, w1, input.view});
    require(p.prefill_multiseg_pending < UINT64_MAX - p.prefill_multiseg_completed,
        "NAX fused dense SwiGLU route overflow");
    const uint32_t padded_rows = ((input.rows + 31u) / 32u) * 32u;
    const uint64_t bf16_bytes = checked_mul(checked_mul(uint64_t(padded_rows), uint64_t(input.columns)), 2ull);
    require(p.nax_bf16_scratch.bytes >= bf16_bytes, "NAX fused dense SwiGLU BF16 scratch too small");
    const auto bf16 = slice(p.nax_bf16_scratch, 0, bf16_bytes);
    if (reuse_bf16) {
        require(can_reuse_nax_dense_bf16(input), "NAX fused dense SwiGLU BF16 reuse requires matching valid input");
    } else {
        const NaxConvertBf16Args conv{input.columns, input.rows, padded_rows, 0, input.stride_bytes};
        p.select("nax_f32_to_bf16", uint32_t(-1), {input.columns, input.rows, padded_rows, 0});
        p.bind(input.view, 0); p.bind(bf16, 1, true); p.parameter(conv, 2);
        p.threads(padded_rows * (input.columns / 4u));
        p.nax_dense_bf16_valid = true;
        p.nax_dense_input_slot = input.view.slot;
        p.nax_dense_input_offset = input.view.offset;
        p.nax_dense_input_end = checked_add(input.view.offset, plan.input_bytes);
        p.nax_dense_rows = input.rows;
        p.nax_dense_padded_rows = padded_rows;
        p.nax_dense_columns = input.columns;
        p.nax_dense_stride = input.stride_bytes;
    }
    const bool use_m32 = input.rows >= 64u;
    const uint32_t gy = use_m32 ? (input.rows + 127u) / 128u : (input.rows + 31u) / 32u;
    const auto* ge = p.q8_tile64_entry(w0, input.columns, output.columns);
    const auto* ue = p.q8_tile64_entry(w1, input.columns, output.columns);
    if (ge && ue) {
        const char* kname = use_m32 ? "nax_prefill_dense_q8_swiglu_t64_m32" : "nax_prefill_dense_q8_swiglu_t64_m8";
        p.select(kname, uint32_t(-1), {input.columns, output.columns, input.rows, uint32_t(l0)}, 128);
        p.bind(ge->plane0, 0); p.bind(ge->meta, 1); p.bind(ue->plane0, 2); p.bind(ue->meta, 3);
        p.bind(bf16, 4); p.bind(output.view, 5, true); p.parameter(plan.args, 6);
        p.groups(MTLSizeMake((output.columns + 31u) / 32u, gy, 1));
        if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.prefill_multiseg_pending;
        return;
    }
    const char* kname = use_m32 ? "nax_prefill_dense_q8_swiglu_m32" : "nax_prefill_dense_q8_swiglu_m8";
    p.select(kname, 8, {input.columns, output.columns, input.rows, uint32_t(l0)}, 128, uint32_t(l0));
    p.bind(w0, 0); p.bind(w1, 1); p.bind(bf16, 2); p.bind(output.view, 3, true); p.parameter(plan.args, 4);
    p.groups(MTLSizeMake((output.columns + 31u) / 32u, gy, 1));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.prefill_multiseg_pending;
}
bool MetalGraph::can_dense_mm_q8_multiseg(std::initializer_list<PrefillWeightLayout> layouts, uint32_t batches) const {
    impl_->thread_check();
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    if (!impl_->prefill_multiseg || impl_->prefill_dense_nax || !impl_->prefill_dense_f32 || !impl_->prefill_dense_staged_f32 || batches < 1 || batches > 512) return false;
    for (const auto layout : layouts) {
        if (layout != PrefillWeightLayout::gguf && layout != PrefillWeightLayout::q8_lanes_v1) return false;
    }
    return true;
#else
    static_cast<void>(layouts); static_cast<void>(batches);
    return false;
#endif
}
void MetalGraph::dense_mm_q8_multiseg2(PrefillWeightLayout l0, GpuView w0, PrefillWeightLayout l1, GpuView w1,
    GpuRowsF32 input, GpuRowsF32 out0, GpuRowsF32 out1) {
    struct DenseQ8MultisegArgs {
        uint32_t columns, batches, row_bytes, swizzle;
        uint32_t rows[3], pad0;
        uint32_t tile_offsets[4];
        uint32_t layouts[3], seg_count;
    };
    static_assert(sizeof(DenseQ8MultisegArgs) == 64, "dense Q8 multiseg ABI");
    auto& p = *impl_; p.active();
    const uint32_t batches = input.rows;
    require(!p.recording && !p.compatible_pipelines && can_dense_mm_q8_multiseg({l0, l1}, batches) &&
        p.dense_mm_prepared.contains(uint32_t(l0)) && p.dense_mm_prepared.contains(uint32_t(l1)),
        "multiseg2 dense prefill requires idle preparation, direct batches and enabled staged FP32 policy");
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    require(p.dense_q8_staged_prepared[0], "multiseg2 dense prefill requires prepared staged Q8");
#endif
    require(batches == out0.rows && batches == out1.rows && input.columns && input.columns % 32 == 0 && out0.columns && out1.columns &&
        input.stride_bytes == uint64_t(input.columns) * 4 && out0.stride_bytes == uint64_t(out0.columns) * 4 &&
        out1.stride_bytes == uint64_t(out1.columns) * 4, "multiseg2 dense prefill tight row geometry mismatch");
    const auto b0 = q8_mpp_bytes(input.columns, out0.columns, batches);
    const auto b1 = q8_mpp_bytes(input.columns, out1.columns, batches);
    static_cast<void>(p.allocation(w0)); static_cast<void>(p.allocation(w1)); static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(out0.view, true)); static_cast<void>(p.allocation(out1.view, true));
    require(w0.bytes >= b0.matrix && w1.bytes >= b1.matrix && input.view.bytes >= b0.input &&
        out0.view.bytes >= b0.output && out1.view.bytes >= b1.output &&
        w0.offset % 2 == 0 && w1.offset % 2 == 0 && input.view.offset % 4 == 0 &&
        out0.view.offset % 4 == 0 && out1.view.offset % 4 == 0, "multiseg2 dense prefill buffer span or alignment mismatch");
    const auto w0s = slice(w0, 0, b0.matrix), w1s = slice(w1, 0, b1.matrix), ins = slice(input.view, 0, b0.input);
    const auto o0s = slice(out0.view, 0, b0.output), o1s = slice(out1.view, 0, b1.output);
    p.prefill_weight(w0s, l0, 8, input.columns, out0.columns, 1);
    p.prefill_weight(w1s, l1, 8, input.columns, out1.columns, 1);
    validate_disjoint(std::array{w0s, w1s, ins}, std::array{o0s, o1s});
    const bool use_m64 = prefill_m64_active(batches);
    require(p.prefill_multiseg_pending < UINT64_MAX - p.prefill_multiseg_completed &&
        (!use_m64 || p.prefill_m64_pending < UINT64_MAX - p.prefill_m64_completed), "multiseg2 dense prefill route overflow");
    const uint32_t t0 = (out0.columns + 63) / 64, t1 = (out1.columns + 63) / 64;
    const DenseQ8MultisegArgs args{
        input.columns, batches, input.columns / 32 * 34, p.prefill_swizzle ? 8u : 0u,
        {out0.columns, out1.columns, 0u}, 0u,
        {0u, t0, t0 + t1, t0 + t1},
        {l0 == PrefillWeightLayout::q8_lanes_v1 ? 1u : 0u, l1 == PrefillWeightLayout::q8_lanes_v1 ? 1u : 0u, 0u}, 2u
    };
    p.select(use_m64 ? "graph_dense_q8_multiseg2_f32_m64" : "graph_dense_q8_multiseg2_f32",
        uint32_t(-1), {input.columns, out0.columns + out1.columns, batches, 2u});
    p.bind(w0s, 0); p.bind(w1s, 1); p.bind(ins, 2); p.bind(o0s, 3, true); p.bind(o1s, 4, true); p.parameter(args, 5);
    [p.encoder setThreadgroupMemoryLength:(use_m64 ? 16384 : 12288) atIndex:0];
    p.groups(MTLSizeMake(t0 + t1, use_m64 ? (batches + 63) / 64 : (batches + 31) / 32, 1));
    ++p.prefill_multiseg_pending;
    if (use_m64) ++p.prefill_m64_pending;
}
void MetalGraph::dense_mm_q8_multiseg3(PrefillWeightLayout l0, GpuView w0, PrefillWeightLayout l1, GpuView w1,
    PrefillWeightLayout l2, GpuView w2, GpuRowsF32 input, GpuRowsF32 out0, GpuRowsF32 out1, GpuRowsF32 out2) {
    struct DenseQ8MultisegArgs {
        uint32_t columns, batches, row_bytes, swizzle;
        uint32_t rows[3], pad0;
        uint32_t tile_offsets[4];
        uint32_t layouts[3], seg_count;
    };
    static_assert(sizeof(DenseQ8MultisegArgs) == 64, "dense Q8 multiseg ABI");
    auto& p = *impl_; p.active();
    const uint32_t batches = input.rows;
    require(!p.recording && !p.compatible_pipelines && can_dense_mm_q8_multiseg({l0, l1, l2}, batches) &&
        p.dense_mm_prepared.contains(uint32_t(l0)) && p.dense_mm_prepared.contains(uint32_t(l1)) &&
        p.dense_mm_prepared.contains(uint32_t(l2)),
        "multiseg3 dense prefill requires idle preparation, direct batches and enabled staged FP32 policy");
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    require(p.dense_q8_staged_prepared[0], "multiseg3 dense prefill requires prepared staged Q8");
#endif
    require(batches == out0.rows && batches == out1.rows && batches == out2.rows && input.columns && input.columns % 32 == 0 &&
        out0.columns && out1.columns && out2.columns && input.stride_bytes == uint64_t(input.columns) * 4 &&
        out0.stride_bytes == uint64_t(out0.columns) * 4 && out1.stride_bytes == uint64_t(out1.columns) * 4 &&
        out2.stride_bytes == uint64_t(out2.columns) * 4, "multiseg3 dense prefill tight row geometry mismatch");
    const auto b0 = q8_mpp_bytes(input.columns, out0.columns, batches);
    const auto b1 = q8_mpp_bytes(input.columns, out1.columns, batches);
    const auto b2 = q8_mpp_bytes(input.columns, out2.columns, batches);
    static_cast<void>(p.allocation(w0)); static_cast<void>(p.allocation(w1)); static_cast<void>(p.allocation(w2));
    static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(out0.view, true)); static_cast<void>(p.allocation(out1.view, true)); static_cast<void>(p.allocation(out2.view, true));
    require(w0.bytes >= b0.matrix && w1.bytes >= b1.matrix && w2.bytes >= b2.matrix && input.view.bytes >= b0.input &&
        out0.view.bytes >= b0.output && out1.view.bytes >= b1.output && out2.view.bytes >= b2.output &&
        w0.offset % 2 == 0 && w1.offset % 2 == 0 && w2.offset % 2 == 0 && input.view.offset % 4 == 0 &&
        out0.view.offset % 4 == 0 && out1.view.offset % 4 == 0 && out2.view.offset % 4 == 0,
        "multiseg3 dense prefill buffer span or alignment mismatch");
    const auto w0s = slice(w0, 0, b0.matrix), w1s = slice(w1, 0, b1.matrix), w2s = slice(w2, 0, b2.matrix), ins = slice(input.view, 0, b0.input);
    const auto o0s = slice(out0.view, 0, b0.output), o1s = slice(out1.view, 0, b1.output), o2s = slice(out2.view, 0, b2.output);
    p.prefill_weight(w0s, l0, 8, input.columns, out0.columns, 1);
    p.prefill_weight(w1s, l1, 8, input.columns, out1.columns, 1);
    p.prefill_weight(w2s, l2, 8, input.columns, out2.columns, 1);
    validate_disjoint(std::array{w0s, w1s, w2s, ins}, std::array{o0s, o1s, o2s});
    const bool use_m64 = prefill_m64_active(batches);
    require(p.prefill_multiseg_pending < UINT64_MAX - p.prefill_multiseg_completed &&
        (!use_m64 || p.prefill_m64_pending < UINT64_MAX - p.prefill_m64_completed), "multiseg3 dense prefill route overflow");
    const uint32_t t0 = (out0.columns + 63) / 64, t1 = (out1.columns + 63) / 64, t2 = (out2.columns + 63) / 64;
    const DenseQ8MultisegArgs args{
        input.columns, batches, input.columns / 32 * 34, p.prefill_swizzle ? 8u : 0u,
        {out0.columns, out1.columns, out2.columns}, 0u,
        {0u, t0, t0 + t1, t0 + t1 + t2},
        {l0 == PrefillWeightLayout::q8_lanes_v1 ? 1u : 0u, l1 == PrefillWeightLayout::q8_lanes_v1 ? 1u : 0u,
         l2 == PrefillWeightLayout::q8_lanes_v1 ? 1u : 0u}, 3u
    };
    p.select(use_m64 ? "graph_dense_q8_multiseg3_f32_m64" : "graph_dense_q8_multiseg3_f32",
        uint32_t(-1), {input.columns, out0.columns + out1.columns + out2.columns, batches, 3u});
    p.bind(w0s, 0); p.bind(w1s, 1); p.bind(w2s, 2); p.bind(ins, 3);
    p.bind(o0s, 4, true); p.bind(o1s, 5, true); p.bind(o2s, 6, true); p.parameter(args, 8);
    [p.encoder setThreadgroupMemoryLength:(use_m64 ? 16384 : 12288) atIndex:0];
    p.groups(MTLSizeMake(t0 + t1 + t2, use_m64 ? (batches + 63) / 64 : (batches + 31) / 32, 1));
    ++p.prefill_multiseg_pending;
    if (use_m64) ++p.prefill_m64_pending;
}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_CHUNK
void MetalGraph::prepare_gdn_chunk() {
    auto& p = *impl_; p.idle();
    for (const auto* name : gdn_chunk_kernel_names) static_cast<void>(p.pipeline(name));
    if (p.prefill_dense_nax || p.prefill_grouped_nax || p.verify_nax) {
        for (const auto* name : {"gdn_chunk_f32_nax", "gdn_chunk_rms_gate_nax", "gdn_chunk_rms_rows_nax", "gdn_chunk_rms_rows_fast"})
            static_cast<void>(p.pipeline(name));
    }
    if (p.prefill_dense_nax) static_cast<void>(p.pipeline("nax_gdn_proj_gates_f32"));
    p.gdn_chunk_prepared = true;
}
void MetalGraph::gdn_chunk(const GdnChunkDispatch& dispatch) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_chunk_prepared, "GDN prefill requires idle preparation and direct batches");
    const bool allow_2k = p.prefill_dense_nax
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        || (dispatch.op == GdnChunkOp::rms_rows && p.dflash2_prepared && p.dflash2_scope)
#endif
        ;
    validate_gdn_chunk_dispatch(dispatch, allow_2k);
    auto spec = gdn_chunk_spec(dispatch); const auto used = uint16_t(spec.reads | spec.writes);
    const bool use_nax = p.prefill_dense_nax || p.prefill_grouped_nax ||
        (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope);
    if (use_nax) {
        if (dispatch.op == GdnChunkOp::projection_f32) {
            spec.kernel = "gdn_chunk_f32_nax";
            spec.groups[1] = (dispatch.params.rows + 7u) / 8u;
        } else if (dispatch.op == GdnChunkOp::rms_gate) spec.kernel = "gdn_chunk_rms_gate_nax";
        else if (dispatch.op == GdnChunkOp::rms_rows) spec.kernel = "gdn_chunk_rms_rows_nax";
    }
    require(spec.kernel && spec.threads == 128 && !(used & ~uint16_t(0x37f)), "invalid GDN prefill binding specification");
    for (size_t i = 0; i < dispatch.buffers.size(); ++i) if (used & (1u << i))
        static_cast<void>(p.allocation(dispatch.buffers[i], spec.writes & (1u << i)));
    if (dispatch.op == GdnChunkOp::projection_gates_f32 && p.prefill_dense_nax &&
        dispatch.params.rows >= 16u && dispatch.params.rows <= 2048u &&
        dispatch.params.columns >= 32u && (dispatch.params.columns % 32u) == 0u &&
        dispatch.params.heads >= 1u && dispatch.params.heads == dispatch.params.outputs &&
        can_reuse_nax_dense_bf16({dispatch.buffers[2], dispatch.params.rows, dispatch.params.columns, dispatch.params.input_stride})) {
        const uint32_t padded_rows = ((dispatch.params.rows + 31u) / 32u) * 32u;
        const uint64_t bf16_bytes = checked_mul(checked_mul(uint64_t(padded_rows), uint64_t(dispatch.params.columns)), 2ull);
        const auto bf16 = slice(p.nax_bf16_scratch, 0, bf16_bytes);
        p.select("nax_gdn_proj_gates_f32", uint32_t(-1), {dispatch.params.rows, dispatch.params.columns, dispatch.params.heads, dispatch.params.value_width});
        p.bind(dispatch.buffers[0], 0);
        p.bind(dispatch.buffers[1], 1);
        p.bind(bf16, 2);
        p.bind(dispatch.buffers[3], 3);
        p.bind(dispatch.buffers[4], 4);
        p.bind(dispatch.buffers[5], 5, true);
        p.bind(dispatch.buffers[6], 6, true);
        p.parameter(dispatch.params, 14);
        p.groups(MTLSizeMake((dispatch.params.heads + 15u) / 16u, (dispatch.params.rows + 63u) / 64u, 1));
        return;
    }
    if (use_nax && dispatch.op == GdnChunkOp::rms_rows) {
        const uint64_t total_groups = uint64_t(dispatch.params.rows) * (dispatch.params.columns / dispatch.params.value_width);
        const bool aligned128 = (dispatch.params.value_width & 127u) == 0u && dispatch.params.value_width <= 4096u;
        const bool reg_cached_wide = aligned128 && dispatch.params.value_width >= 512u;
        if (total_groups >= 1 && aligned128 && (total_groups <= 384 || reg_cached_wide)) {
            p.select("gdn_chunk_rms_rows_fast", uint32_t(-1), {dispatch.params.rows, dispatch.params.columns, dispatch.params.heads, dispatch.params.value_width});
            for (size_t i = 0; i < dispatch.buffers.size(); ++i) if (used & (1u << i))
                p.bind(dispatch.buffers[i], i, spec.writes & (1u << i));
            p.parameter(dispatch.params, 14);
            p.groups(MTLSizeMake(uint32_t(total_groups), 1, 1));
            return;
        }
    }
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    if (dispatch.op == GdnChunkOp::fused && (p.gdn_use_ported_active() || p.gdn_prefill_ported_enabled)) {
        require((p.gdn_ported_configured || p.gdn_prefill_ported_configured) && p.fixed_route_profile_allowed() && dispatch.params.key_width == 128,
            "ported GDN prefill requires configured direct batches and key_width 128");
        const bool count_ported = p.gdn_ported_configured && p.gdn_use_ported_active() && !p.gdn_route_shadow_active();
        if (count_ported) require(p.gdn_ported_pending[0] < UINT64_MAX - p.gdn_ported_completed[0], "ported GDN prefill route overflow");
        const char* kernel = (dispatch.params.value_width % 16 == 0) ? "gdn_pipelined_tb12" : "gdn_blocked_seq_tb16";
        p.select(kernel, uint32_t(-1), {dispatch.params.rows, dispatch.params.columns, dispatch.params.heads, dispatch.params.value_width});
        for (size_t i = 0; i < dispatch.buffers.size(); ++i) if (used & (1u << i))
            p.bind(dispatch.buffers[i], i, spec.writes & (1u << i));
        p.parameter(dispatch.params, 14);
        p.groups(MTLSizeMake((dispatch.params.value_width + 15) / 16, dispatch.params.heads, 1));
        if (count_ported) ++p.gdn_ported_pending[0];
        return;
    }
#endif
    p.select(spec.kernel, uint32_t(-1), {dispatch.params.rows, dispatch.params.columns, dispatch.params.heads, dispatch.params.value_width});
    for (size_t i = 0; i < dispatch.buffers.size(); ++i) if (used & (1u << i))
        p.bind(dispatch.buffers[i], i, spec.writes & (1u << i));
    p.parameter(dispatch.params, 14);
    p.groups(MTLSizeMake(spec.groups[0], spec.groups[1], spec.groups[2]));
}
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
void MetalGraph::prepare_gdn_legacy_batch() {
    auto& p = *impl_; p.idle(); static_cast<void>(p.pipeline(gdn_legacy_batch_kernel)); p.gdn_legacy_batch_prepared = true;
}
uint64_t MetalGraph::gdn_legacy_batch_encoded_calls() const { impl_->idle(); return impl_->gdn_legacy_batch_encoded; }
void MetalGraph::gdn_legacy_batch(const GdnChunkDispatch& dispatch) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_legacy_batch_prepared, "old-order GDN requires idle preparation and direct batches");
    validate_gdn_legacy_batch_dispatch(dispatch); const auto spec = gdn_legacy_batch_spec(dispatch); const auto used = uint16_t(spec.reads | spec.writes);
    require(spec.threads == 128 && used == 0x17f, "old-order GDN binding specification");
    for (size_t i = 0; i < dispatch.buffers.size(); ++i) if (used & (1u << i)) static_cast<void>(p.allocation(dispatch.buffers[i], spec.writes & (1u << i)));
    p.select(spec.kernel, uint32_t(-1), {dispatch.params.rows, dispatch.params.columns, dispatch.params.heads, dispatch.params.value_width});
    for (size_t i = 0; i < dispatch.buffers.size(); ++i) if (used & (1u << i)) p.bind(dispatch.buffers[i], i, spec.writes & (1u << i));
    p.parameter(dispatch.params, 14); p.groups(MTLSizeMake(spec.groups[0], spec.groups[1], spec.groups[2]));
    p.gdn_legacy_batch_encoded = checked_add(p.gdn_legacy_batch_encoded, 1);
}
#endif
void MetalGraph::rms_rows(GpuRowsF32 input, GpuView gains, GpuRowsF32 output, uint32_t group_width, float epsilon, bool shared_gains) {
    require(input.rows == output.rows && input.columns == output.columns, "row RMS shape mismatch");
    GdnChunkDispatch dispatch; dispatch.op = GdnChunkOp::rms_rows;
    dispatch.buffers[0] = input.view; dispatch.buffers[1] = gains; dispatch.buffers[2] = output.view;
    dispatch.params.rows = input.rows; dispatch.params.columns = input.columns; dispatch.params.value_width = group_width;
    dispatch.params.input_stride = input.stride_bytes; dispatch.params.output_stride = output.stride_bytes;
    dispatch.params.rms_epsilon = epsilon; dispatch.params.mode = shared_gains ? 0 : 1;
    gdn_chunk(dispatch);
}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
namespace {
struct GdnVerifyParams {
    uint32_t heads, key_heads, key_width, value_width;
    uint32_t capacity, steps, mode, l2_mode;
    float query_scale, l2_epsilon, rms_epsilon, reserved;
};
static_assert(sizeof(GdnVerifyParams) == 48);

void validate_gdn_verify_tile(StateTile tile) {
    if (tile.capacity == 1) {
        require(tile.steps == 1, "GDN single-step verify tile requires steps == 1");
    } else {
        validate_state_tile(tile);
    }
}
} // namespace

void MetalGraph::configure_gdn_ported(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!enabled, "this optimisation was removed because its source could not be verified");
    require(!p.gdn_ported_locked && (!p.gdn_ported_configured || p.gdn_ported_enabled == enabled) &&
        !p.recording && !p.compatible_pipelines && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "ported GDN policy is immutable startup policy on direct unprofiled batches");
    if (enabled) for (const char* name : {"gdn_pipelined_tb12", "gdn_blocked_seq_tb16", "gdn_prepare_rows",
                             "gdn_verify_main", "gdn_verify_replay", "gdn_norm_gate"}) {
        const auto state = p.pipeline(name);
        require(state.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "ported GDN threadgroup memory exceeds device capability");
    }
    p.gdn_ported_configured = true;
    p.gdn_ported_enabled = enabled;
}
bool MetalGraph::gdn_ported_enabled() const { impl_->thread_check(); return impl_->gdn_ported_enabled; }
void MetalGraph::configure_gdn_prefill_ported(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!enabled, "this optimisation was removed because its source could not be verified");
    require(!p.gdn_prefill_ported_locked && (!p.gdn_prefill_ported_configured || p.gdn_prefill_ported_enabled == enabled) &&
        !p.recording && !p.compatible_pipelines && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "prefill-only ported GDN policy is immutable startup policy on direct unprofiled batches");
    if (enabled) for (const char* name : {"gdn_pipelined_tb12", "gdn_blocked_seq_tb16"}) {
        const auto state = p.pipeline(name);
        require(state.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "ported GDN threadgroup memory exceeds device capability");
    }
    p.gdn_prefill_ported_configured = true;
    p.gdn_prefill_ported_enabled = enabled;
}
bool MetalGraph::gdn_prefill_ported_enabled() const { impl_->thread_check(); return impl_->gdn_prefill_ported_enabled; }
bool MetalGraph::gdn_ported_rollback_enabled() const { impl_->thread_check(); return impl_->gdn_ported_configured; }
bool MetalGraph::gdn_use_ported() const { impl_->thread_check(); return impl_->gdn_use_ported_active(); }
GdnPortedRoutes MetalGraph::gdn_ported_routes() const { impl_->idle(); return impl_->gdn_ported_completed; }
void MetalGraph::prepare_gdn_ported_variants() {
    auto& p = *impl_; p.idle();
    require(false, "ported GDN variants were removed because their source could not be verified");
    require(!p.gdn_ported_locked && !p.recording && !p.compatible_pipelines && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "ported GDN variants require idle direct unprofiled startup");
    for (const char* name : {"gdn_pipelined_tb12", "gdn_blocked_seq_tb16", "gdn_prepare_rows",
                             "gdn_verify_main", "gdn_verify_main_db8", "gdn_verify_main_db16",
                             "gdn_verify_replay", "gdn_verify_replay_db8", "gdn_verify_replay_db16",
                             "gdn_verify_main_replay", "gdn_norm_gate"}) {
        const auto state = p.pipeline(name);
        require(state.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "ported GDN variant threadgroup memory exceeds device capability");
    }
    p.gdn_ported_variants_prepared = true;
}
void MetalGraph::gdn_prepare_rows(GpuView input, GpuView query, GpuView key, GpuView value,
                                  GdnShape shape, GdnPolicy policy, StateTile tile) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && (p.gdn_ported_configured || p.gdn_ported_variants_prepared) && p.fixed_route_profile_allowed(),
        "ported GDN prepare requires configured direct batches");
    validate_gdn_shape(shape); validate_gdn_policy(policy); validate_gdn_verify_tile(tile);
    require(shape.key_width == 128 && shape.value_heads <= 64 && shape.value_width <= 256, "ported GDN prepare geometry");
    const auto raw = gdn_qkv_count(shape), qk = uint64_t(shape.value_heads) * 128, val = uint64_t(shape.value_heads) * shape.value_width;
    require(p.floats(input) == checked_mul(raw, tile.capacity) && p.floats(query, true) == checked_mul(qk, tile.capacity) &&
        p.floats(key, true) == checked_mul(qk, tile.capacity) && p.floats(value, true) == checked_mul(val, tile.capacity),
        "ported GDN prepare buffer shape mismatch");
    validate_disjoint(std::array{input}, std::array{query, key, value});
    const GdnVerifyParams params{shape.value_heads, shape.key_heads, shape.key_width, shape.value_width,
        tile.capacity, tile.steps, 0, uint32_t(policy.l2_mode), policy.query_scale, policy.l2_epsilon, policy.rms_epsilon, 0};
    p.select("gdn_prepare_rows", uint32_t(-1), {shape.value_heads, shape.value_width, tile.capacity, tile.steps});
    p.bind(input, 0); p.bind(query, 1, true); p.bind(key, 2, true); p.bind(value, 3, true); p.parameter(params, 4);
    p.groups(MTLSizeMake((tile.steps * shape.value_heads + 3) / 4, 1, 1));
}
void MetalGraph::gdn_verify_main(GpuView accepted_state, GpuView query, GpuView key, GpuView value,
                                 GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates, GpuView dt_bias,
                                 GpuView candidate_state, GpuView output, GpuView tape_decay, GpuView tape_beta,
                                 uint32_t heads, uint32_t value_width, StateTile tile, bool write_tape) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_ported_configured && p.fixed_route_profile_allowed(),
        "ported GDN verify main requires configured direct batches");
    validate_gdn_verify_tile(tile);
    require(heads && heads <= 64 && value_width && value_width <= 256, "ported GDN verify main geometry");
    const uint64_t state_n = uint64_t(heads) * 128 * value_width, qk_n = uint64_t(tile.capacity) * heads * 128;
    const uint64_t val_n = uint64_t(tile.capacity) * heads * value_width, gate_n = uint64_t(tile.capacity) * heads;
    require(p.floats(accepted_state) == state_n && p.floats(query) == qk_n && p.floats(key) == qk_n && p.floats(value) == val_n &&
        p.floats(alpha_logits) == gate_n && p.floats(beta_logits) == gate_n && p.floats(negative_rates) == heads && p.floats(dt_bias) == heads &&
        p.floats(candidate_state, true) == state_n && p.floats(output, true) == val_n &&
        p.floats(tape_decay, true) == gate_n && p.floats(tape_beta, true) == gate_n,
        "ported GDN verify main buffer shape mismatch");
    validate_disjoint(std::array{accepted_state, query, key, value, alpha_logits, beta_logits, negative_rates, dt_bias},
        std::array{candidate_state, output, tape_decay, tape_beta});
    const bool shadow = p.gdn_route_shadow_active();
    const size_t bucket = tile.capacity == 1 ? 1 : 2;
    if (!shadow) require(p.gdn_ported_pending[bucket] < UINT64_MAX - p.gdn_ported_completed[bucket], "ported GDN verify route overflow");
    const GdnVerifyParams params{heads, heads, 128, value_width, tile.capacity, tile.steps, write_tape ? 1u : 0u, 0, 0, 0, 0, 0};
    p.select("gdn_verify_main", uint32_t(-1), {heads, value_width, tile.capacity, tile.steps});
    p.bind(accepted_state, 0); p.bind(query, 1); p.bind(key, 2); p.bind(value, 3);
    p.bind(alpha_logits, 4); p.bind(beta_logits, 5); p.bind(negative_rates, 6); p.bind(dt_bias, 8);
    p.bind(candidate_state, 9, true); p.bind(output, 10, true); p.bind(tape_decay, 11, true); p.bind(tape_beta, 12, true);
    p.parameter(params, 13);
    p.groups(MTLSizeMake((value_width + Impl::gdn_verify_db - 1) / Impl::gdn_verify_db, heads, 1));
    if (!shadow) ++p.gdn_ported_pending[bucket];
}
void MetalGraph::gdn_verify_main_db(GpuView accepted_state, GpuView query, GpuView key, GpuView value,
                                    GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates, GpuView dt_bias,
                                    GpuView candidate_state, GpuView output, GpuView tape_decay, GpuView tape_beta,
                                    uint32_t heads, uint32_t value_width, StateTile tile, bool write_tape, uint32_t db) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_ported_variants_prepared && p.fixed_route_profile_allowed() && (db == 8 || db == 16),
        "ported GDN verify main variant requires prepared variants, direct batches and db 8 or 16");
    validate_gdn_verify_tile(tile);
    require(heads && heads <= 64 && value_width && value_width <= 256, "ported GDN verify main variant geometry");
    const uint64_t state_n = uint64_t(heads) * 128 * value_width, qk_n = uint64_t(tile.capacity) * heads * 128;
    const uint64_t val_n = uint64_t(tile.capacity) * heads * value_width, gate_n = uint64_t(tile.capacity) * heads;
    require(p.floats(accepted_state) == state_n && p.floats(query) == qk_n && p.floats(key) == qk_n && p.floats(value) == val_n &&
        p.floats(alpha_logits) == gate_n && p.floats(beta_logits) == gate_n && p.floats(negative_rates) == heads && p.floats(dt_bias) == heads &&
        p.floats(candidate_state, true) == state_n && p.floats(output, true) == val_n &&
        p.floats(tape_decay, true) == gate_n && p.floats(tape_beta, true) == gate_n,
        "ported GDN verify main variant buffer shape mismatch");
    validate_disjoint(std::array{accepted_state, query, key, value, alpha_logits, beta_logits, negative_rates, dt_bias},
        std::array{candidate_state, output, tape_decay, tape_beta});
    const GdnVerifyParams params{heads, heads, 128, value_width, tile.capacity, tile.steps, write_tape ? 1u : 0u, 0, 0, 0, 0, 0};
    const char* kernel = db == 8 ? "gdn_verify_main_db8" : "gdn_verify_main_db16";
    p.select(kernel, uint32_t(-1), {heads, value_width, tile.capacity, tile.steps});
    p.bind(accepted_state, 0); p.bind(query, 1); p.bind(key, 2); p.bind(value, 3);
    p.bind(alpha_logits, 4); p.bind(beta_logits, 5); p.bind(negative_rates, 6); p.bind(dt_bias, 8);
    p.bind(candidate_state, 9, true); p.bind(output, 10, true); p.bind(tape_decay, 11, true); p.bind(tape_beta, 12, true);
    p.parameter(params, 13);
    p.groups(MTLSizeMake((value_width + db - 1) / db, heads, 1));
}
void MetalGraph::gdn_verify_replay(GpuView accepted_state, GpuView key, GpuView value,
                                   GpuView decay, GpuView beta, GpuView replayed_state,
                                   uint32_t heads, uint32_t value_width, StateTile tile) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_ported_configured && p.fixed_route_profile_allowed(),
        "ported GDN verify replay requires configured direct batches");
    validate_gdn_verify_tile(tile);
    require(heads && heads <= 64 && value_width && value_width <= 256, "ported GDN verify replay geometry");
    const uint64_t state_n = uint64_t(heads) * 128 * value_width, qk_n = uint64_t(tile.capacity) * heads * 128;
    const uint64_t val_n = uint64_t(tile.capacity) * heads * value_width, gate_n = uint64_t(tile.capacity) * heads;
    require(p.floats(accepted_state) == state_n && p.floats(key) == qk_n && p.floats(value) == val_n &&
        p.floats(decay) == gate_n && p.floats(beta) == gate_n && p.floats(replayed_state, true) == state_n,
        "ported GDN verify replay buffer shape mismatch");
    validate_disjoint(std::array{accepted_state, key, value, decay, beta}, std::array{replayed_state});
    const bool shadow = p.gdn_route_shadow_active();
    if (!shadow) require(p.gdn_ported_pending[3] < UINT64_MAX - p.gdn_ported_completed[3], "ported GDN replay route overflow");
    const GdnVerifyParams params{heads, heads, 128, value_width, tile.capacity, tile.steps, 0, 0, 0, 0, 0, 0};
    p.select("gdn_verify_replay", uint32_t(-1), {heads, value_width, tile.capacity, tile.steps});
    p.bind(accepted_state, 0); p.bind(key, 1); p.bind(value, 2); p.bind(decay, 3); p.bind(beta, 4);
    p.bind(replayed_state, 5, true); p.parameter(params, 6);
    p.groups(MTLSizeMake((value_width + Impl::gdn_verify_db - 1) / Impl::gdn_verify_db, heads, 1));
    if (!shadow) ++p.gdn_ported_pending[3];
}
void MetalGraph::gdn_verify_replay_db(GpuView accepted_state, GpuView key, GpuView value,
                                      GpuView decay, GpuView beta, GpuView replayed_state,
                                      uint32_t heads, uint32_t value_width, StateTile tile, uint32_t db) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_ported_variants_prepared && p.fixed_route_profile_allowed() && (db == 8 || db == 16),
        "ported GDN verify replay variant requires prepared variants, direct batches and db 8 or 16");
    validate_gdn_verify_tile(tile);
    require(heads && heads <= 64 && value_width && value_width <= 256, "ported GDN verify replay variant geometry");
    const uint64_t state_n = uint64_t(heads) * 128 * value_width, qk_n = uint64_t(tile.capacity) * heads * 128;
    const uint64_t val_n = uint64_t(tile.capacity) * heads * value_width, gate_n = uint64_t(tile.capacity) * heads;
    require(p.floats(accepted_state) == state_n && p.floats(key) == qk_n && p.floats(value) == val_n &&
        p.floats(decay) == gate_n && p.floats(beta) == gate_n && p.floats(replayed_state, true) == state_n,
        "ported GDN verify replay variant buffer shape mismatch");
    validate_disjoint(std::array{accepted_state, key, value, decay, beta}, std::array{replayed_state});
    const GdnVerifyParams params{heads, heads, 128, value_width, tile.capacity, tile.steps, 0, 0, 0, 0, 0, 0};
    const char* kernel = db == 8 ? "gdn_verify_replay_db8" : "gdn_verify_replay_db16";
    p.select(kernel, uint32_t(-1), {heads, value_width, tile.capacity, tile.steps});
    p.bind(accepted_state, 0); p.bind(key, 1); p.bind(value, 2); p.bind(decay, 3); p.bind(beta, 4);
    p.bind(replayed_state, 5, true); p.parameter(params, 6);
    p.groups(MTLSizeMake((value_width + db - 1) / db, heads, 1));
}
void MetalGraph::gdn_verify_main_replay(GpuView base_state, GpuView prefix_key, GpuView prefix_value,
                                        GpuView prefix_decay, GpuView prefix_beta, GpuView query, GpuView key, GpuView value,
                                        GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates, GpuView dt_bias,
                                        GpuView next_state, GpuView output, uint32_t heads, uint32_t value_width,
                                        StateTile prefix_tile, StateTile main_tile) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.gdn_ported_variants_prepared && p.fixed_route_profile_allowed(),
        "ported GDN deferred main+replay requires prepared variants and direct batches");
    validate_gdn_verify_tile(prefix_tile); validate_gdn_verify_tile(main_tile);
    require(prefix_tile.capacity == main_tile.capacity && heads && heads <= 64 && value_width && value_width <= 256,
        "ported GDN main+replay geometry");
    const uint64_t state_n = uint64_t(heads) * 128 * value_width, qk_n = uint64_t(main_tile.capacity) * heads * 128;
    const uint64_t val_n = uint64_t(main_tile.capacity) * heads * value_width, gate_n = uint64_t(main_tile.capacity) * heads;
    require(p.floats(base_state) == state_n && p.floats(prefix_key) == qk_n && p.floats(prefix_value) == val_n &&
        p.floats(prefix_decay) == gate_n && p.floats(prefix_beta) == gate_n &&
        p.floats(query) == qk_n && p.floats(key) == qk_n && p.floats(value) == val_n &&
        p.floats(alpha_logits) == gate_n && p.floats(beta_logits) == gate_n &&
        p.floats(negative_rates) == heads && p.floats(dt_bias) == heads &&
        p.floats(next_state, true) == state_n && p.floats(output, true) == val_n,
        "ported GDN main+replay buffer shape mismatch");
    validate_disjoint(std::array{base_state, prefix_key, prefix_value, prefix_decay, prefix_beta,
        query, key, value, alpha_logits, beta_logits, negative_rates, dt_bias},
        std::array{next_state, output});
    const GdnVerifyParams params{heads, heads, 128, value_width, main_tile.capacity, main_tile.steps, 0, prefix_tile.steps, 0, 0, 0, 0};
    p.select("gdn_verify_main_replay", uint32_t(-1), {heads, value_width, main_tile.capacity, main_tile.steps});
    p.bind(base_state, 0); p.bind(prefix_key, 1); p.bind(prefix_value, 2); p.bind(prefix_decay, 3); p.bind(prefix_beta, 4);
    p.bind(query, 5); p.bind(key, 6); p.bind(value, 8); p.bind(alpha_logits, 9); p.bind(beta_logits, 10);
    p.bind(negative_rates, 11); p.bind(dt_bias, 12); p.bind(next_state, 13, true); p.bind(output, 14, true);
    p.parameter(params, 15);
    p.groups(MTLSizeMake((value_width + Impl::gdn_verify_db - 1) / Impl::gdn_verify_db, heads, 1));
}
void MetalGraph::gdn_norm_gate(GpuView raw, GpuView effective_gains, GpuView gate, GpuView output,
                               uint32_t heads, uint32_t value_width, float epsilon, StateTile tile) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && (p.gdn_ported_configured || p.gdn_ported_variants_prepared) && p.fixed_route_profile_allowed(),
        "ported GDN norm+gate requires configured direct batches");
    validate_gdn_verify_tile(tile);
    require(heads && heads <= 64 && value_width && value_width <= 65536 && std::isfinite(epsilon) && epsilon > 0,
        "ported GDN norm+gate geometry");
    const uint64_t val_n = checked_mul(checked_mul(tile.capacity, heads), value_width);
    require(p.floats(raw) == val_n && p.floats(effective_gains) == value_width &&
        p.floats(gate) == val_n && p.floats(output, true) == val_n,
        "ported GDN norm+gate buffer shape mismatch");
    validate_disjoint(std::array{raw, effective_gains, gate}, std::array{output});
    const bool shadow = p.gdn_route_shadow_active();
    if (!shadow && p.gdn_ported_configured) require(p.gdn_ported_pending[4] < UINT64_MAX - p.gdn_ported_completed[4], "ported GDN norm+gate route overflow");
    const GdnVerifyParams params{heads, heads, 128, value_width, tile.capacity, tile.steps, 0, 0, 0, 0, epsilon, 0};
    p.select("gdn_norm_gate", uint32_t(-1), {heads, value_width, tile.capacity, tile.steps});
    p.bind(raw, 0); p.bind(effective_gains, 1); p.bind(gate, 2); p.bind(output, 3, true); p.parameter(params, 4);
    p.groups(MTLSizeMake((tile.steps * heads + 3) / 4, 1, 1));
    if (!shadow && p.gdn_ported_configured) ++p.gdn_ported_pending[4];
}
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
void MetalGraph::prepare_qsa_prefill() {
    auto& p = *impl_; p.idle();
    for (const auto op : qsa_prefill_operations) {
        const auto threads = op == QsaPrefillOp::attention_partial128 || op == QsaPrefillOp::attention_reduce ? 32u : 128u;
        static_cast<void>(p.pipeline(qsa_prefill_kernel_name(op), uint32_t(-1), threads));
    }
    if (p.prefill_dense_nax || p.verify_nax) {
        const auto nax_idx = p.pipeline("nax_qsa_index_scores", uint32_t(-1), 128);
        const auto pack_q = p.pipeline("nax_qsa_pack_queries_bf16", uint32_t(-1), 256);
        const auto build_lists = p.pipeline("nax_qsa_build_tile_lists", uint32_t(-1), 256);
        const auto g12_attn = p.pipeline("nax_qsa_attention_g12", uint32_t(-1), 256);
        const auto g12_red = p.pipeline("nax_qsa_attention_g12_reduce", uint32_t(-1), 256);
        const auto pack_q_r4 = p.pipeline("nax_qsa_pack_queries_bf16_r4", uint32_t(-1), 256);
        const auto build_lists_r4 = p.pipeline("nax_qsa_build_tile_lists_r4", uint32_t(-1), 256);
        const auto g12_attn_r4 = p.pipeline("nax_qsa_attention_g12_r4", uint32_t(-1), 128);
        const auto g12_red_r4 = p.pipeline("nax_qsa_attention_g12_reduce_r4", uint32_t(-1), 256);
        require(nax_idx.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            pack_q.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            build_lists.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            g12_attn.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            g12_red.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            pack_q_r4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            build_lists_r4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            g12_attn_r4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            g12_red_r4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "NAX QSA prefill threadgroup memory exceeds device capability");
        const uint64_t min_bf16 = p.prefill_dense_nax ? 41'943'040ull : 4'194'304ull;
        if (p.nax_bf16_scratch.bytes < min_bf16) {
            if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
            p.nax_bf16_scratch = zeros(min_bf16, "nax_bf16_scratch");
        }
    }
    if (p.prefill_dense_nax) {
        static_cast<void>(p.pipeline("qsa_prefill_bf16_project_nax", uint32_t(-1), 128));
        static_cast<void>(p.pipeline("nax_f32_to_bf16"));
        const auto bf32 = p.pipeline("nax_prefill_dense_bf16_m32", uint32_t(-1), 128);
        const auto bf8 = p.pipeline("nax_prefill_dense_bf16_m8", uint32_t(-1), 128);
        const auto nax_attn = p.pipeline("nax_qsa_attention", uint32_t(-1), 128);
        require(bf32.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            bf8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_attn.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "NAX QSA prefill threadgroup memory exceeds device capability");
    }
    p.qsa_prefill_prepared = true;
}
void MetalGraph::qsa_prefill_dispatch(QsaPrefillOp op, std::span<const GpuView> buffers, const std::array<uint32_t, 16>& parameters) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.qsa_prefill_prepared, "QSA prefill requires idle preparation and direct batches");
    validate_qsa_prefill_dispatch(op, buffers, parameters, p.prefill_dense_nax);
    auto spec = qsa_prefill_spec(op, parameters, p.prefill_dense_nax);
    if (op == QsaPrefillOp::bf16_project && p.prefill_dense_nax) spec.kernel = "qsa_prefill_bf16_project_nax";
    const bool use_nax_idx = p.prefill_dense_nax || (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope);
    if (use_nax_idx && op == QsaPrefillOp::index_scores &&
        parameters[2] == 4u && parameters[3] == 128u && (parameters[1] + parameters[0]) / 4u > parameters[6]) {
        spec.kernel = "nax_qsa_index_scores";
        spec.groups = {((parameters[1] + parameters[0]) / 4u + 31u) / 32u, (parameters[0] + 7u) / 8u, 1u};
    }
    require(spec.kernel && buffers.size() == spec.buffers && spec.threads[1] == 1 && spec.threads[2] == 1 &&
        (spec.threads[0] == 32 || spec.threads[0] == 128), "invalid QSA prefill binding specification");
    for (size_t i = 0; i < buffers.size(); ++i) static_cast<void>(p.allocation(buffers[i], spec.writable_mask & (1u << i)));
    p.select(spec.kernel, uint32_t(-1), {parameters[0], parameters[1], parameters[2], parameters[3]}, spec.threads[0]);
    for (size_t i = 0; i < buffers.size(); ++i) p.bind(buffers[i], i, spec.writable_mask & (1u << i));
    p.parameter(parameters, 6);
    p.dispatch(true, MTLSizeMake(spec.groups[0], spec.groups[1], spec.groups[2]), MTLSizeMake(spec.threads[0], 1, 1));
}
bool MetalGraph::nax_qsa_attention_supported(QsaShape shape) const {
    const auto& p = *impl_; p.thread_check();
    return p.prefill_dense_nax && shape.head_width == 256 && shape.kv_heads && shape.query_heads &&
        shape.query_heads % shape.kv_heads == 0 && (shape.query_heads / shape.kv_heads) <= 16 &&
        shape.compression == 4 && shape.token_budget && shape.token_budget % 4 == 0 && shape.token_budget / 4 <= 512;
}
bool MetalGraph::verify_nax_qsa_attention_supported(QsaShape shape) const {
    const auto& p = *impl_; p.thread_check();
    const bool no_route_sink =
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        p.qsa_route_sink == nullptr;
#else
        true;
#endif
    return p.verify_nax && no_route_sink && shape.head_width == 256 && shape.kv_heads && shape.kv_heads <= 2 &&
        shape.query_heads == shape.kv_heads * 12 && shape.compression == 4 &&
        shape.token_budget && shape.token_budget % 4 == 0 && shape.token_budget / 4 <= 512;
}
void MetalGraph::nax_qsa_select_blocks(GpuView rotary_index_query, GpuView bf16_index_keys, GpuView selected_ids,
    uint32_t base, uint32_t rows, QsaShape shape, uint32_t block_budget) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.qsa_prefill_prepared && nax_qsa_attention_supported(shape),
        "NAX QSA block selection requires prepared NAX QSA prefill and direct batches");
    require(rows && rows <= 2048 && base < shape.capacity && rows <= shape.capacity - base &&
        block_budget == shape.token_budget / 4 && (base + rows) / 4u > block_budget,
        "NAX QSA block selection row/budget geometry");
    const uint32_t iq = shape.index_heads * shape.index_width;
    const uint32_t chunk_max_blocks = (base + rows) / 4u;
    const uint32_t score_stride = ((chunk_max_blocks + 31u) / 32u) * 32u;
    const uint32_t sel_stride = ((score_stride + 511u) / 512u) * block_budget;
    const uint32_t idx_stripe = qsa_nax_indexer_stripe(base, rows, block_budget);
    const uint64_t scores_bytes = uint64_t(idx_stripe) * score_stride * 4ull;
    const uint64_t sel_bytes = uint64_t(idx_stripe) * sel_stride * 4ull;
    require(p.nax_bf16_scratch.bytes >= scores_bytes + 4ull * sel_bytes, "NAX QSA indexer scratch too small");
    validate_disjoint(std::array{rotary_index_query, bf16_index_keys}, std::array{selected_ids, p.nax_bf16_scratch});
    validate_disjoint(std::array{selected_ids}, std::array{p.nax_bf16_scratch});
    p.nax_dense_bf16_valid = false;
    const auto scores = slice(p.nax_bf16_scratch, 0, scores_bytes);
    const auto score_a = slice(p.nax_bf16_scratch, scores_bytes, sel_bytes);
    const auto id_a = slice(p.nax_bf16_scratch, scores_bytes + sel_bytes, sel_bytes);
    const auto score_b = slice(p.nax_bf16_scratch, scores_bytes + 2ull * sel_bytes, sel_bytes);
    const auto id_b = slice(p.nax_bf16_scratch, scores_bytes + 3ull * sel_bytes, sel_bytes);
    for (uint32_t first = 0; first < rows; first += idx_stripe) {
        const uint32_t sn = std::min(idx_stripe, rows - first), sb = base + first, max_blocks = (sb + sn) / 4u;
        if (max_blocks <= block_budget) continue;
        const auto iq_slice = slice(rotary_index_query, uint64_t(first) * iq * 4ull, uint64_t(sn) * iq * 4ull);
        const auto id_slice = slice(selected_ids, uint64_t(first) * block_budget * 4ull, uint64_t(sn) * block_budget * 4ull);
        const std::array<GpuView, 3> b_scores{iq_slice, bf16_index_keys, scores};
        qsa_prefill_dispatch(QsaPrefillOp::index_scores, b_scores,
            {sn, sb, shape.index_heads, shape.index_width, score_stride, iq, block_budget});
        const std::array<GpuView, 3> b_sort{scores, score_a, id_a};
        qsa_prefill_dispatch(QsaPrefillOp::select_sort, b_sort,
            {sn, sb, score_stride, sel_stride, block_budget});
        GpuView in_scores = score_a, in_ids = id_a, out_scores = score_b, out_ids = id_b;
        for (uint32_t span = 512u; span < max_blocks; span *= 2u) {
            const std::array<GpuView, 4> b_merge{in_scores, in_ids, out_scores, out_ids};
            qsa_prefill_dispatch(QsaPrefillOp::select_merge, b_merge,
                {sn, sb, sel_stride, block_budget, span});
            std::swap(in_scores, out_scores);
            std::swap(in_ids, out_ids);
        }
        const std::array<GpuView, 2> b_ids{in_ids, id_slice};
        qsa_prefill_dispatch(QsaPrefillOp::select_ids, b_ids,
            {sn, sb, sel_stride, block_budget, block_budget});
    }
}
void MetalGraph::nax_qsa_attention(GpuView query, GpuView bf16_keys, GpuView bf16_values, GpuView selected_ids,
    GpuView output, uint32_t base, uint32_t rows, QsaShape shape, uint32_t block_budget) {
    struct NaxQsaAttentionArgs {
        uint32_t rows, base, query_heads, kv_heads;
        uint32_t head_width, block_budget, q_stride, id_stride;
    };
    struct NaxQsaG12Args {
        uint32_t rows, base, query_heads, kv_heads;
        uint32_t head_width, block_budget, q_stride, id_stride;
        uint32_t num_tiles, list_stride, splits, sparse_mode;
    };
    static_assert(sizeof(NaxQsaAttentionArgs) == 32 && sizeof(NaxQsaG12Args) == 48, "NAX QSA attention ABI");
    auto& p = *impl_; p.active();
    const bool allow_prefill = nax_qsa_attention_supported(shape);
    const bool allow_verify = p.verify_nax_prepared && p.verify_nax_scope && verify_nax_qsa_attention_supported(shape);
    require(!p.recording && !p.compatible_pipelines && p.qsa_prefill_prepared && (allow_prefill || allow_verify),
        "NAX QSA attention requires prepared NAX QSA prefill or verify scope and direct batches");
    const uint32_t max_rows = allow_prefill ? 2048u : 8u;
    require(rows && rows <= max_rows && base < shape.capacity && rows <= shape.capacity - base &&
        block_budget == shape.token_budget / 4, "NAX QSA attention row/budget geometry");
    const uint32_t q_width = shape.query_heads * 256u;
    const uint64_t kv_bytes = uint64_t(base + rows) * shape.kv_heads * 256u * 2u;
    const uint64_t q_bytes = uint64_t(rows) * q_width * 4u;
    const uint64_t id_bytes = uint64_t(rows) * block_budget * 4u;
    static_cast<void>(p.allocation(query)); static_cast<void>(p.allocation(bf16_keys));
    static_cast<void>(p.allocation(bf16_values)); static_cast<void>(p.allocation(selected_ids));
    static_cast<void>(p.allocation(output, true));
    require(query.bytes >= q_bytes && output.bytes >= q_bytes && bf16_keys.bytes >= kv_bytes &&
        bf16_values.bytes >= kv_bytes && selected_ids.bytes >= id_bytes &&
        query.offset % 4 == 0 && output.offset % 4 == 0 && bf16_keys.offset % 2 == 0 &&
        bf16_values.offset % 2 == 0 && selected_ids.offset % 4 == 0,
        "NAX QSA attention buffer extent/alignment");
    validate_disjoint(std::array{query, bf16_keys, bf16_values, selected_ids}, std::array{output});
    if (shape.query_heads == shape.kv_heads * 12u && shape.kv_heads <= 2u) {
        validate_disjoint(std::array{query, bf16_keys, bf16_values, selected_ids, output}, std::array{p.nax_bf16_scratch});
        p.nax_dense_bf16_valid = false;
        const bool sparse_mode = (base + rows) / 4u > block_budget;
        const bool use_r4 = false;
        const uint32_t rows_per_tile = use_r4 ? 4u : 8u;
        const uint32_t m_tile = use_r4 ? 48u : 96u;
        const uint32_t num_tiles = (rows + rows_per_tile - 1u) / rows_per_tile;
        const uint32_t list_stride = sparse_mode ? (use_r4 ? 2112u : 4112u) : 0u;
        const uint32_t splits = (num_tiles * shape.kv_heads >= 32u) ? 1u : std::min(16u, std::max(1u, 32u / (num_tiles * shape.kv_heads)));
        const uint64_t q_bf16_bytes = ((uint64_t(shape.kv_heads) * num_tiles * uint64_t(m_tile) * 256ull * 2ull + 255ull) / 256ull) * 256ull;
        const uint64_t lists_bytes = sparse_mode ? (((uint64_t(num_tiles) * list_stride * 4ull) + 255ull) / 256ull) * 256ull : 256ull;
        const uint64_t counts_bytes = sparse_mode ? (((uint64_t(num_tiles) * 4ull) + 255ull) / 256ull) * 256ull : 256ull;
        const uint64_t total_slots = uint64_t(num_tiles) * shape.kv_heads * splits;
        const uint64_t partials_bytes = (splits > 1u) ? (total_slots * uint64_t(m_tile) * 256ull * 4ull) : 256ull;
        const uint64_t stats_bytes = (splits > 1u) ? (total_slots * uint64_t(m_tile) * 2ull * 4ull) : 256ull;
        const uint64_t total_scratch = q_bf16_bytes + lists_bytes + counts_bytes + partials_bytes + stats_bytes;
        require(p.nax_bf16_scratch.bytes >= total_scratch, "NAX QSA G12 scratch too small");
        const auto q_bf16 = slice(p.nax_bf16_scratch, 0, q_bf16_bytes);
        const auto tile_lists = slice(p.nax_bf16_scratch, q_bf16_bytes, lists_bytes);
        const auto tile_counts = slice(p.nax_bf16_scratch, q_bf16_bytes + lists_bytes, counts_bytes);
        const auto partials = slice(p.nax_bf16_scratch, q_bf16_bytes + lists_bytes + counts_bytes, partials_bytes);
        const auto stats = slice(p.nax_bf16_scratch, q_bf16_bytes + lists_bytes + counts_bytes + partials_bytes, stats_bytes);
        const NaxQsaG12Args g12_args{
            rows, base, shape.query_heads, shape.kv_heads, shape.head_width, block_budget, q_width, block_budget,
            num_tiles, list_stride, splits, sparse_mode ? 1u : 0u
        };
        const uint32_t total_vec4 = shape.kv_heads * num_tiles * m_tile * 64u;
        p.select(use_r4 ? "nax_qsa_pack_queries_bf16_r4" : "nax_qsa_pack_queries_bf16",
            uint32_t(-1), {base, rows, shape.query_heads, shape.kv_heads}, 256);
        p.bind(query, 0); p.bind(q_bf16, 1, true); p.parameter(g12_args, 2);
        p.dispatch(true, MTLSizeMake((total_vec4 + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1));
        if (sparse_mode) {
            p.select(use_r4 ? "nax_qsa_build_tile_lists_r4" : "nax_qsa_build_tile_lists",
                uint32_t(-1), {base, rows, block_budget, num_tiles}, 256);
            p.bind(selected_ids, 0); p.bind(tile_lists, 1, true); p.bind(tile_counts, 2, true); p.parameter(g12_args, 3);
            p.dispatch(true, MTLSizeMake(num_tiles, 1, 1), MTLSizeMake(256, 1, 1));
        }
        const uint32_t attn_threads = use_r4 ? 128u : 256u;
        p.select(use_r4 ? "nax_qsa_attention_g12_r4" : "nax_qsa_attention_g12",
            uint32_t(-1), {base, rows, num_tiles, splits}, attn_threads);
        p.bind(q_bf16, 0); p.bind(bf16_keys, 1); p.bind(bf16_values, 2);
        p.bind(tile_lists, 3); p.bind(tile_counts, 4);
        p.bind(output, 5, splits == 1u);
        p.bind(partials, 6, splits > 1u);
        p.bind(stats, 8, splits > 1u);
        p.parameter(g12_args, 9);
        p.dispatch(true, MTLSizeMake(shape.kv_heads, num_tiles, splits), MTLSizeMake(attn_threads, 1, 1));
        if (splits > 1u) {
            p.select(use_r4 ? "nax_qsa_attention_g12_reduce_r4" : "nax_qsa_attention_g12_reduce",
                uint32_t(-1), {base, rows, num_tiles, splits}, 256);
            p.bind(partials, 0); p.bind(stats, 1); p.bind(tile_counts, 2); p.bind(output, 3, true);
            p.parameter(g12_args, 4);
            p.dispatch(true, MTLSizeMake(shape.kv_heads, rows * 12u, 1), MTLSizeMake(256, 1, 1));
        }
        return;
    }
    const NaxQsaAttentionArgs args{rows, base, shape.query_heads, shape.kv_heads, shape.head_width, block_budget, q_width, block_budget};
    p.select("nax_qsa_attention", uint32_t(-1), {base, rows, shape.query_heads, shape.kv_heads}, 128);
    p.bind(query, 0); p.bind(bf16_keys, 1); p.bind(bf16_values, 2); p.bind(selected_ids, 3); p.bind(output, 4, true);
    p.parameter(args, 5);
    p.dispatch(true, MTLSizeMake(shape.kv_heads, rows, 1), MTLSizeMake(128, 1, 1));
}
#endif
#ifdef TOKSPUR_PRIVATE_Q8_MPP
void MetalGraph::prepare_q8_mpp(Q8MppLayout layout) {
    auto& p = *impl_; p.idle();
    require([p.device supportsFamily:MTLGPUFamilyMetal4], "Q8 TensorOps requires Metal 4 GPU support");
    const auto* name = q8_mpp_kernel(layout); const auto groups = q8_mpp_simdgroups(layout);
    static_cast<void>(p.pipeline(name, uint32_t(-1), groups * 32)); p.q8_mpp_prepared.insert(layout);
}
void MetalGraph::q8_mpp(GpuView matrix, GpuView input, GpuView output,
                      uint32_t columns, uint32_t rows, uint32_t batches, Q8MppLayout layout) {
    auto& p = *impl_; p.active(); const auto shape = q8_mpp_bytes(columns, rows, batches);
    const auto* name = q8_mpp_kernel(layout); const auto groups = q8_mpp_simdgroups(layout);
    require(!p.recording && p.q8_mpp_prepared.contains(layout), "Q8 TensorOps requires idle preparation and direct batches");
    // This operator's bound is 134,217,728 bytes (134.218 MB). The generic
    // 16,777,216-float cap is retained for the existing vector primitives.
    require(matrix.bytes == shape.matrix && input.bytes == shape.input &&
        output.bytes == shape.output, "Q8 TensorOps buffer geometry mismatch");
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input));
    static_cast<void>(p.allocation(output, true)); separate(output, {matrix, input});
    p.select(name, uint32_t(-1), {columns, rows, batches, uint32_t(layout)}, groups * 32);
    p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{columns, rows, columns / 32 * 34, batches}, 3);
    const uint32_t side = layout == Q8MppLayout::single ? 16 : 32;
    p.dispatch(true, MTLSizeMake((rows + side - 1) / side, (batches + side - 1) / side, 1), MTLSizeMake(groups * 32, 1, 1));
}
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
void MetalGraph::configure_prefill_dense_staged_f32(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.dense_staged_locked && p.dense_mm_prepared.empty() && !p.dense_q8_staged_prepared[0] && !p.dense_q8_staged_prepared[1],
        "staged dense route is fixed before preparation or the first batch");
    require(!enabled || (p.prefill_dense_f32 && (p.prefill_grouped_f32 || p.prefill_grouped_auto)), "staged dense requires raw FP32/FP32 policy");
#ifndef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    require(!enabled || !p.prefill_hc_down_f32, "staged dense refuses the legacy HC down experiment");
#endif
    p.prefill_dense_staged_f32 = enabled;
}
bool MetalGraph::prefill_dense_staged_f32() const { impl_->thread_check(); return impl_->prefill_dense_staged_f32; }
const char* MetalGraph::prefill_dense_kernel() const {
    impl_->thread_check();
    return impl_->prefill_dense_nax ? "nax_prefill_dense_q8_m32" : (impl_->prefill_dense_staged_f32 ? "graph_dense_q8_staged_f32" : "graph_q8_mpp_wide");
}
void MetalGraph::prepare_dense_q8_staged(bool half_operands) {
    auto& p = *impl_; p.idle(); require([p.device supportsFamily:MTLGPUFamilyMetal4], "staged dense Q8 requires Metal 4");
    const auto state = half_operands ? p.pipeline("graph_dense_mm_q8", 8, 128, false, uint32_t(PrefillWeightLayout::gguf)) : p.pipeline("graph_dense_q8_staged_f32");
    const uint32_t shared = half_operands ? 8192 : 12288;
    require(state.staticThreadgroupMemoryLength + shared <= p.device.maxThreadgroupMemoryLength, "staged dense Q8 threadgroup memory exceeds device capability");
    if (!half_operands) {
        if (p.prefill_m64 != PrefillM64Mode::off) {
            const auto s64 = p.pipeline("graph_dense_q8_staged_f32_m64");
            require(s64.staticThreadgroupMemoryLength + 16384 <= p.device.maxThreadgroupMemoryLength, "staged dense Q8 m64 threadgroup memory exceeds device capability");
        }
        if (p.prefill_multiseg) {
            for (const auto* name : {"graph_dense_q8_multiseg2_f32", "graph_dense_q8_multiseg3_f32"}) {
                const auto ms = p.pipeline(name);
                require(ms.staticThreadgroupMemoryLength + 12288 <= p.device.maxThreadgroupMemoryLength, "multiseg staged dense Q8 threadgroup memory exceeds device capability");
            }
            if (p.prefill_m64 != PrefillM64Mode::off) {
                for (const auto* name : {"graph_dense_q8_multiseg2_f32_m64", "graph_dense_q8_multiseg3_f32_m64"}) {
                    const auto ms64 = p.pipeline(name);
                    require(ms64.staticThreadgroupMemoryLength + 16384 <= p.device.maxThreadgroupMemoryLength, "multiseg staged dense Q8 m64 threadgroup memory exceeds device capability");
                }
            }
        }
    }
    p.dense_q8_staged_prepared[half_operands ? 1 : 0] = true;
}
void MetalGraph::dense_q8_staged(GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows, uint32_t batches, bool half_operands) {
    dense_q8_staged(matrix, input, output, columns, rows, batches, half_operands, PrefillWeightLayout::gguf);
}
void MetalGraph::dense_q8_staged(GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows, uint32_t batches, bool half_operands, PrefillWeightLayout layout) {
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && batches <= 512 && p.dense_q8_staged_prepared[half_operands ? 1 : 0], "staged dense Q8 requires idle preparation, direct batches and at most 512 rows");
    require(layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1, "unsupported staged dense layout");
    const auto bytes = q8_mpp_bytes(columns, rows, batches);
    require(matrix.bytes == bytes.matrix && input.bytes == bytes.input && output.bytes == bytes.output && matrix.offset % 2 == 0 && input.offset % 4 == 0 && output.offset % 4 == 0, "staged dense Q8 geometry or alignment mismatch");
    static_cast<void>(p.allocation(matrix)); static_cast<void>(p.allocation(input)); static_cast<void>(p.allocation(output, true)); separate(output, {matrix, input});
    p.prefill_weight(matrix, layout, 8, columns, rows, 1);
    if (half_operands) {
        const auto plan = dense_mm_q8_plan(PrefillWeightLayout::gguf, columns, rows, batches, uint64_t(columns) * 4, uint64_t(rows) * 4);
        // This explicit comparison dispatch bypasses the ordinary raw-tight serial shortcut and the FP32 selector.
        p.select("graph_dense_mm_q8", 8, {columns, rows, batches, 0}, 128, uint32_t(PrefillWeightLayout::gguf));
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true); p.parameter(plan.args, 3);
        [p.encoder setThreadgroupMemoryLength:8192 atIndex:0];
        p.groups(MTLSizeMake((rows + 63) / 64, (batches + 31) / 32, 1));
    } else {
        const bool use_m64 = prefill_m64_active(batches);
        if (use_m64) require(p.prefill_m64_pending < UINT64_MAX - p.prefill_m64_completed, "prefill m64 route overflow");
        p.select(use_m64 ? "graph_dense_q8_staged_f32_m64" : "graph_dense_q8_staged_f32", uint32_t(-1), {columns, rows, batches, uint32_t(layout)});
        p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
        p.parameter(std::array<uint32_t, 4>{columns, rows, columns / 32 * 34, batches}, 3);
        const uint32_t layout_val = ((layout == PrefillWeightLayout::q8_lanes_v1) ? 1u : 0u) | (p.prefill_swizzle ? 2u : 0u);
        p.parameter(layout_val, 4);
        [p.encoder setThreadgroupMemoryLength:(use_m64 ? 16384 : 12288) atIndex:0];
        p.groups(MTLSizeMake((rows + 63) / 64, use_m64 ? (batches + 63) / 64 : (batches + 31) / 32, 1));
        if (use_m64) ++p.prefill_m64_pending;
    }
}
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MPP
void MetalGraph::prepare_routed_matmul_many(uint32_t type, uint32_t tile, uint32_t staging_k) {
    auto& p = *impl_; p.idle();
    require([p.device supportsFamily:MTLGPUFamilyMetal4], "grouped TensorOps requires Metal 4 GPU support");
    const auto* name = grouped_mpp_kernel(type,tile,staging_k);
    static_cast<void>(p.pipeline("graph_grouped_routes"));
    static_cast<void>(p.pipeline("graph_grouped_tasks"));
    static_cast<void>(p.pipeline(name,type,staging_k ? 128 : 32));
    p.grouped_mpp_prepared.insert({type,tile,staging_k});
}
void MetalGraph::routed_matmul_many(uint32_t type, GpuView matrices, GpuView indices, GpuView input,
    GpuView output, GpuView token_map, GpuView counts, GpuView tasks, GroupedMppShape s, bool broadcast) {
    auto& p = *impl_; p.active(); const auto bytes = grouped_mpp_bytes(type,s,broadcast);
    require(s.staging_k != 1 || s.tile != 1 || p.block_gemv_formats == 3,
        "packed ordinary GEMVs require the unchanged block3 traversal policy");
    require(!p.recording && !p.compatible_pipelines && p.grouped_mpp_prepared.contains({type,s.tile,s.staging_k}),
        "grouped TensorOps requires explicit preparation and direct batches");
    const std::array views{matrices,indices,input,output,token_map,counts,tasks};
    const std::array<uint64_t,7> expected{bytes.matrix,bytes.indices,bytes.input,bytes.output,bytes.token_map,bytes.counts,bytes.tasks};
    for (size_t i = 0; i < views.size(); ++i)
        require(views[i].bytes == expected[i] && views[i].offset % 4 == 0, "grouped TensorOps view geometry/alignment mismatch");
    validate_disjoint(std::array{matrices,indices,input},std::array{output,token_map,counts,tasks});
    const std::array<uint32_t,4> grouping{s.experts,s.selected,s.batches,s.tile};
    p.select("graph_grouped_routes"); p.bind(indices,0); p.bind(token_map,1,true); p.bind(counts,2,true); p.parameter(grouping,3);
    p.groups(MTLSizeMake(s.experts,1,1));
    p.select("graph_grouped_tasks"); p.bind(counts,0); p.bind(tasks,1,true); p.parameter(grouping,2); p.groups(MTLSizeMake(1,1,1));
    p.select(grouped_mpp_kernel(type,s.tile,s.staging_k),type,{s.columns,s.rows,s.experts,s.tile},s.staging_k ? 128 : 32);
    p.bind(matrices,0); p.bind(input,1); p.bind(output,2,true); p.bind(indices,4); p.bind(token_map,6); p.bind(tasks,8); p.bind(counts,9);
    p.parameter(std::array<uint32_t,4>{s.columns,s.rows,bytes.row_bytes,s.experts},3);
    p.parameter(std::array<uint32_t,4>{s.selected,s.batches,s.tile,uint32_t(broadcast)},5);
    const uint32_t output_tile = s.staging_k == 1 ? 4 : s.tile;
    p.dispatch(true,MTLSizeMake((s.rows+output_tile-1)/output_tile,bytes.maximum_tasks,1),MTLSizeMake(s.staging_k ? 128 : 32,1,1));
}
#endif
void MetalGraph::prepare_q8_matvec_split() {
    auto& p = *impl_; p.idle();
    static_cast<void>(p.pipeline("graph_q8_split_partials"));
    static_cast<void>(p.pipeline("graph_q8_split_reduce"));
    p.q8_split_prepared = true; // Publish only after both owned pipelines exist.
}
void MetalGraph::q8_matvec_split(GpuView matrix, GpuView input, GpuView output, GpuView partials,
                                uint32_t columns, uint32_t rows, uint32_t splits) {
    auto& p = *impl_; p.active(); const auto b = q8_split_bytes({columns, rows, splits});
    require(!p.recording && !p.compatible_pipelines, "Q8 split requires an ordinary direct batch");
    require(p.q8_split_prepared, "Q8 split pipelines must be prepared while idle");
    require(matrix.bytes == b.matrix && uint64_t(p.floats(input)) * 4 == b.input &&
        uint64_t(p.floats(output, true)) * 4 == b.output && uint64_t(p.floats(partials, true)) * 4 == b.partials,
        "Q8 split GPU buffer shape mismatch");
    const std::array reads{matrix, input}; const std::array writes{partials, output};
    validate_disjoint(reads, writes); // Check every capability/alias before either pass.
    require(!p.profiling_active() || (p.dispatches <= p.profile_capacity && p.profile_capacity - p.dispatches >= 2),
        "Q8 split requires two available GPU profiling dispatch slots");
    const std::array<uint32_t, 4> shape{columns, rows, splits, 0};
    p.select("graph_q8_split_partials", uint32_t(-1), shape);
    p.bind(matrix, 0); p.bind(input, 1); p.bind(partials, 2, true);
    p.parameter(shape, 3); p.groups(MTLSizeMake((rows + 3) / 4, splits, 1));
    p.select("graph_q8_split_reduce", uint32_t(-1), shape);
    p.bind(partials, 0); p.bind(output, 1, true); p.parameter(shape, 2); p.threads(rows);
}
void MetalGraph::q8_soa_gemv(GpuView scales, GpuView values, GpuView input, GpuView output, uint32_t columns, uint32_t rows) {
    auto& p = *impl_; p.active();
    require(columns && columns <= 65536 && columns % 32 == 0 && rows && rows <= 1'000'000, "unsupported Q8 SoA GEMV shape");
    const auto elements = checked_mul(columns, rows), blocks = elements / 32;
    require(scales.bytes == checked_mul(blocks, 2) && values.bytes == elements &&
        p.floats(input) == columns && p.floats(output, true) == rows, "Q8 SoA GEMV buffer shape mismatch");
    static_cast<void>(p.allocation(scales)); static_cast<void>(p.allocation(values));
    separate(output, {scales, values, input});
    p.select("graph_q8_soa_gemv", uint32_t(-1), {columns, rows, 0, 0});
    p.bind(scales, 0); p.bind(values, 1); p.bind(input, 2); p.bind(output, 3, true);
    p.parameter(std::array<uint32_t, 2>{columns, rows}, 4); p.groups(MTLSizeMake((rows + 3) / 4, 1, 1));
}
void MetalGraph::typed_f32_gemv(GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows) {
    auto& p = *impl_; p.active();
    require(columns && columns <= 65536 && rows && rows <= 1'000'000, "unsupported typed FP32 GEMV shape");
    require(matrix.bytes == checked_mul(checked_mul(columns, rows), 4) && p.floats(input) == columns && p.floats(output, true) == rows,
        "typed FP32 GEMV buffer shape mismatch");
    static_cast<void>(p.allocation(matrix)); separate(output, {matrix, input});
    // The already owned direct kernel is not inherently limited to small rows;
    // the opt-in small-output policy above is its other, separately gated caller.
    p.select("graph_small_gemv_direct", uint32_t(-1), {columns, rows, 0, 0});
    p.bind(matrix, 0); p.bind(input, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{columns, rows, 0, 0}, 3); p.groups(MTLSizeMake((rows + 3) / 4, 1, 1));
}
void MetalGraph::prepare_gdn_tiles(uint32_t threads) {
    auto& p = *impl_; p.idle();
    require(threads == 32 || threads == 64 || threads == 128, "unsupported GDN tile threadgroup");
    static_cast<void>(p.pipeline("graph_gdn_convolution_tile"));
    static_cast<void>(p.pipeline("graph_gdn_delta_tile", uint32_t(-1), threads));
    // No new mode is published if either pipeline preparation refuses.
    p.gdn_tile_threads = threads;
}
void MetalGraph::gdn_convolution_tile(GpuView history, GpuView projected, GpuView weights,
                                      GpuView candidate, GpuView output,
                                      uint32_t channels, StateTile tile) {
    auto& p = *impl_; p.active(); const auto b = gdn_conv_tile_bytes(channels, tile);
    require(p.gdn_tile_threads, "GDN tile pipelines must be prepared while idle");
    require(uint64_t(p.floats(history)) * 4 == b.history && uint64_t(p.floats(projected)) * 4 == b.projected &&
        uint64_t(p.floats(weights)) * 4 == b.weights && uint64_t(p.floats(candidate, true)) * 4 == b.history &&
        uint64_t(p.floats(output, true)) * 4 == b.output, "GDN tile convolution buffer geometry mismatch");
    const std::array reads{history, projected, weights}; const std::array writes{candidate, output};
    validate_disjoint(reads, writes);
    p.select("graph_gdn_convolution_tile", uint32_t(-1), {channels, tile.capacity, tile.steps, 0});
    p.bind(history, 0); p.bind(projected, 1); p.bind(weights, 2); p.bind(candidate, 3, true); p.bind(output, 4, true);
    p.parameter(std::array<uint32_t, 4>{channels, tile.capacity, tile.steps, 0}, 5); p.threads(channels);
}
void MetalGraph::gdn_delta_tile(GpuView state, GpuView query, GpuView key, GpuView value,
                                GpuView decay, GpuView beta, GpuView candidate,
                                GpuView output, uint32_t heads, uint32_t width, StateTile tile) {
    auto& p = *impl_; p.active(); const auto b = gdn_delta_tile_bytes(heads, width, tile);
    require(p.gdn_tile_threads, "GDN tile pipelines must be prepared while idle");
    require(uint64_t(p.floats(state)) * 4 == b.state && uint64_t(p.floats(query)) * 4 == b.query_key &&
        uint64_t(p.floats(key)) * 4 == b.query_key && uint64_t(p.floats(value)) * 4 == b.value_output &&
        uint64_t(p.floats(decay)) * 4 == b.gates && uint64_t(p.floats(beta)) * 4 == b.gates &&
        uint64_t(p.floats(candidate, true)) * 4 == b.state && uint64_t(p.floats(output, true)) * 4 == b.value_output,
        "GDN tile delta buffer geometry mismatch");
    const std::array reads{state, query, key, value, decay, beta}; const std::array writes{candidate, output};
    validate_disjoint(reads, writes);
    p.select("graph_gdn_delta_tile", uint32_t(-1), {heads, width, tile.capacity, tile.steps}, p.gdn_tile_threads);
    p.bind(state, 0); p.bind(query, 1); p.bind(key, 2); p.bind(value, 3); p.bind(decay, 4); p.bind(beta, 5);
    p.bind(candidate, 6, true); p.bind(output, 8, true);
    p.parameter(std::array<uint32_t, 4>{heads, width, tile.capacity, 128}, 9); p.parameter(tile.steps, 10);
    p.dispatch(false, MTLSizeMake(heads * width, 1, 1), MTLSizeMake(p.gdn_tile_threads, 1, 1));
}
void MetalGraph::rms(GpuView input, GpuView gains, GpuView output, uint32_t group_width, float epsilon) {
    auto& p = *impl_;
    p.active();
    const auto count = p.floats(input);
    require(group_width && count % group_width == 0 && group_width <= 65536 && p.floats(output, true) == count &&
            (p.floats(gains) == count || p.floats(gains) == group_width) && std::isfinite(epsilon) && epsilon > 0, "graph RMS shape/epsilon mismatch");
    separate(output, {input, gains});
    p.select("graph_rms"); p.bind(input, 0); p.bind(gains, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{count, group_width, gains.bytes == uint64_t(group_width) * 4, 0}, 3); p.parameter(epsilon, 4);
    p.groups(MTLSizeMake((count / group_width + 3) / 4, 1, 1));
}
void MetalGraph::activate(GpuView input, GpuView output, Activation kind, float scale) {
    auto& p = *impl_;
    p.active();
    const auto count = p.floats(input);
    require(p.floats(output, true) == count && uint32_t(kind) <= 3 && std::isfinite(scale), "graph activation shape/mode mismatch");
    separate(output, {input}, true);
    p.select("graph_activate"); p.bind(input, 0); p.bind(output, 1, true);
    p.parameter(std::array<uint32_t, 2>{count, uint32_t(kind)}, 2); p.parameter(scale, 3); p.threads(count);
}
void MetalGraph::multiply(GpuView input, GpuView gate, GpuView output, bool sigmoid_gate) {
    auto& p = *impl_;
    p.active();
    const auto count = p.floats(input);
    require(p.floats(gate) == count && p.floats(output, true) == count, "graph multiplication shape mismatch");
    separate(output, {input, gate}, true);
    p.select("graph_multiply"); p.bind(input, 0); p.bind(gate, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 2>{count, sigmoid_gate}, 3); p.threads(count);
}
void MetalGraph::configure_swiglu_fused(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.swiglu_fused_locked && !p.ledger.live_captures, "fused SwiGLU is fixed before the first batch");
    if (enabled) static_cast<void>(p.pipeline("graph_swiglu", uint32_t(-1), 256));
    p.swiglu_fused = enabled;
}
bool MetalGraph::swiglu_fused_enabled() const { impl_->thread_check(); return impl_->swiglu_fused; }
uint64_t MetalGraph::swiglu_fused_calls() const { impl_->idle(); return impl_->swiglu_fused_completed; }
void MetalGraph::swiglu(GpuView gate, GpuView up, GpuView output, float scale) {
    if (impl_->swiglu_fused) { swiglu_direct(gate, up, output, scale); return; }
    activate(gate, output, Activation::silu, scale);
    multiply(output, up, output);
}
void MetalGraph::swiglu_direct(GpuView gate, GpuView up, GpuView output, float scale) {
    auto& p = *impl_; p.active();
    const auto count = p.floats(gate);
    require(p.floats(up) == count && p.floats(output, true) == count && std::isfinite(scale), "graph SwiGLU shape/scale mismatch");
    separate(output, {up}); separate(output, {gate}, true);
    if (!p.recording) require(p.swiglu_fused_pending < UINT64_MAX - p.swiglu_fused_completed, "fused SwiGLU route overflow");
    p.select("graph_swiglu", uint32_t(-1), {count, 0, 0, 0}, 256);
    p.bind(gate, 0); p.bind(up, 1); p.bind(output, 2, true);
    p.parameter(count, 3); p.parameter(scale, 4);
    p.dispatch(false, MTLSizeMake(count, 1, 1), MTLSizeMake(256, 1, 1));
    if (!p.recording) ++p.swiglu_fused_pending;
}
void MetalGraph::configure_q8_gemv_rg4(Q8GemvRg4Mode mode) {
    auto& p = *impl_; p.idle();
    require(mode == Q8GemvRg4Mode::off, "this optimisation was removed because its source could not be verified");
    require(!p.q8_gemv_rg4_locked && !p.ledger.live_captures && uint8_t(mode) <= 2, "multi-SG Q8 GEMV is fixed before the first batch");
    if (mode != Q8GemvRg4Mode::off) {
        static_cast<void>(p.pipeline("graph_q8_gemv_rg4", uint32_t(-1), 256));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        static_cast<void>(p.pipeline("dense_lanes_q8_rg4", uint32_t(-1), 256));
#endif
    }
    p.q8_gemv_rg4 = mode;
}
Q8GemvRg4Mode MetalGraph::q8_gemv_rg4_mode() const { impl_->thread_check(); return impl_->q8_gemv_rg4; }
MetalGraph::Q8GemvRg4Routes MetalGraph::q8_gemv_rg4_routes() const { impl_->idle(); return impl_->q8_gemv_rg4_completed; }
void MetalGraph::configure_q8_three_msg(Q8ThreeMsgMode mode) {
    auto& p = *impl_; p.idle();
    require(mode == Q8ThreeMsgMode::off, "this optimisation was removed because its source could not be verified");
    require(!p.q8_three_msg_locked && !p.ledger.live_captures && uint8_t(mode) <= 2, "multi-SG three-row Q8 is fixed before the first batch");
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    if (mode != Q8ThreeMsgMode::off) {
        require(q8_three_rows_enabled(), "multi-SG three-row Q8 requires three-row Q8 enabled");
        static_cast<void>(p.pipeline(q8_three_rows_msg_kernel(Q8ThreeLayout::gguf), uint32_t(-1), 256));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        static_cast<void>(p.pipeline(q8_three_rows_msg_kernel(Q8ThreeLayout::q8_lanes_v1), uint32_t(-1), 256));
#endif
    }
    p.q8_three_msg = mode;
#else
    require(mode == Q8ThreeMsgMode::off, "multi-SG three-row Q8 requires the private build");
#endif
}
Q8ThreeMsgMode MetalGraph::q8_three_msg_mode() const { impl_->thread_check(); return impl_->q8_three_msg; }
MetalGraph::Q8ThreeMsgRoutes MetalGraph::q8_three_msg_routes() const { impl_->idle(); return impl_->q8_three_msg_completed; }
void MetalGraph::configure_moe_three_union(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!enabled, "this optimisation was removed because its source could not be verified");
    require(!p.moe_three_union_locked && !p.ledger.live_captures, "unioned three-row MoE is fixed before the first batch");
    if (enabled) {
        require(q8_three_rows_enabled(), "unioned three-row MoE requires three-row Q8 enabled");
        static_cast<void>(p.pipeline("graph_swiglu", uint32_t(-1), 256));
        static_cast<void>(p.pipeline("moe_three_router"));
        static_cast<void>(p.pipeline("moe_three_route"));
        static_cast<void>(p.pipeline("moe_three_build_union", uint32_t(-1), 32));
        static_cast<void>(p.pipeline("moe_three_shared_router"));
        static_cast<void>(p.pipeline("moe_three_mixture"));
        for (const uint32_t type : {7u, 8u, 12u, 13u}) {
            static_cast<void>(p.pipeline("moe_three_gate_up_swiglu", type, 128, false, 0u));
            static_cast<void>(p.pipeline("moe_three_down", type, 128, false, 0u));
        }
        static_cast<void>(p.pipeline("moe_three_down", 7u, 128, false, 1u));
        if (!p.moe_three_union_meta.owner) p.moe_three_union_meta = zeros(6168, "three-row MoE union metadata");
    }
    p.moe_three_union = enabled;
}
bool MetalGraph::moe_three_union_enabled() const { impl_->thread_check(); return impl_->moe_three_union; }
uint64_t MetalGraph::moe_three_union_calls() const { impl_->idle(); return impl_->moe_three_union_completed; }
bool MetalGraph::moe_three_union_matrix_supported(uint32_t type, GpuView matrix,
    uint32_t columns, uint32_t rows, uint32_t experts, bool allow_packed_q51) const {
    impl_->thread_check();
    if (type != 7 && type != 8 && type != 12 && type != 13) return false;
    if (!columns || columns > 65536 || !rows || rows > 65536 || !experts || experts > 4096) return false;
    if ((type == 12 || type == 13) ? (columns % 256 != 0) : (columns % 32 != 0)) return false;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (const auto* lane = impl_->lane_shape(matrix)) {
        return allow_packed_q51 && type == 7 && lane->encoding == 7 && columns == 640 &&
            lane->columns == columns && lane->rows == rows && lane->experts == experts;
    }
#else
    static_cast<void>(matrix); static_cast<void>(allow_packed_q51);
#endif
    return true;
}
void MetalGraph::moe_three_router(GpuView weights, GpuView input, GpuView logits, uint32_t columns, uint32_t experts) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && !p.recording && !p.compatible_pipelines && columns && columns <= 65536 && experts && experts <= 4096,
        "unioned three-row MoE router state/geometry");
    require(p.floats(weights) == checked_mul(columns, experts) && p.floats(input) == checked_mul(3, columns) &&
        p.floats(logits, true) == checked_mul(3, experts), "unioned three-row MoE router buffer mismatch");
    separate(logits, {weights, input});
    p.select("moe_three_router", uint32_t(-1), {columns, experts, 3, 0});
    p.bind(weights, 0); p.bind(input, 1); p.bind(logits, 2, true);
    p.parameter(std::array<uint32_t, 4>{columns, experts, p.small_gemv_splits, 0}, 3);
    p.groups(MTLSizeMake((experts + 3) / 4, 1, 1));
}
void MetalGraph::moe_three_route(GpuView logits, GpuView indices, GpuView weights,
    uint32_t experts, uint32_t selected, RouterMode mode, bool normalize_selected) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && !p.recording && !p.compatible_pipelines && experts && experts <= 4096 &&
        selected && selected <= 64 && selected <= experts && uint32_t(mode) <= 1, "unioned three-row MoE route geometry");
    require(p.floats(logits) == checked_mul(3, experts) && p.floats(indices, true) == checked_mul(3, selected) &&
        p.floats(weights, true) == checked_mul(3, selected), "unioned three-row MoE route buffer mismatch");
    separate(indices, {logits, weights}); separate(weights, {logits});
    p.select("moe_three_route", uint32_t(-1), {experts, selected, 3, 0});
    p.bind(logits, 0); p.bind(indices, 1, true); p.bind(weights, 2, true);
    p.parameter(std::array<uint32_t, 4>{experts, selected, uint32_t(mode), uint32_t(normalize_selected)}, 3);
    p.groups(MTLSizeMake(3, 1, 1));
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    p.carrier_route(indices, weights, 3, selected, experts);
#endif
}
void MetalGraph::moe_three_build_union(GpuView indices, uint32_t experts, uint32_t selected, uint32_t hidden, uint32_t channels) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && p.moe_three_union_meta.owner && !p.recording && !p.compatible_pipelines &&
        experts && experts <= 4096 && selected && selected <= 64 && selected <= experts && hidden && channels,
        "unioned three-row MoE build_union geometry");
    require(p.floats(indices) == checked_mul(3, selected), "unioned three-row MoE build_union indices mismatch");
    separate(p.moe_three_union_meta, {indices});
    require(p.moe_three_union_pending < UINT64_MAX - p.moe_three_union_completed, "unioned three-row MoE route overflow");
    p.select("moe_three_build_union", uint32_t(-1), {experts, selected, hidden, channels}, 32);
    p.bind(indices, 0); p.bind(p.moe_three_union_meta, 1, true);
    p.parameter(std::array<uint32_t, 4>{experts, selected, hidden, channels}, 2);
    p.dispatch(true, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
    ++p.moe_three_union_pending;
}
void MetalGraph::moe_three_gate_up_swiglu(uint32_t type, GpuView gate_weights, GpuView up_weights,
    GpuView input, GpuView expert_hidden, uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && p.moe_three_union_meta.owner && !p.recording && !p.compatible_pipelines &&
        moe_three_union_matrix_supported(type, gate_weights, columns, rows, experts, false) &&
        moe_three_union_matrix_supported(type, up_weights, columns, rows, experts, false) &&
        selected && selected <= 64 && selected <= experts, "unioned three-row MoE gate+up geometry");
    const auto& fmt = encoding(type);
    const auto row_bytes = checked_mul(columns / fmt.elements_per_block, fmt.bytes_per_block);
    const auto mat_bytes = checked_mul(checked_mul(row_bytes, rows), experts);
    require(gate_weights.bytes == mat_bytes && up_weights.bytes == mat_bytes &&
        p.floats(input) == checked_mul(3, columns) && p.floats(expert_hidden, true) == checked_mul(checked_mul(3, selected), rows),
        "unioned three-row MoE gate+up buffer mismatch");
    static_cast<void>(p.allocation(gate_weights)); static_cast<void>(p.allocation(up_weights));
    separate(expert_hidden, {gate_weights, up_weights, input, p.moe_three_union_meta});
    p.select("moe_three_gate_up_swiglu", type, {columns, rows, experts, selected}, 128, 0u);
    p.bind(gate_weights, 0); p.bind(up_weights, 1); p.bind(input, 2); p.bind(expert_hidden, 3, true); p.bind(p.moe_three_union_meta, 4);
    p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), experts}, 5);
    p.dispatch_indirect(p.moe_three_union_meta, 0, MTLSizeMake(128, 1, 1));
}
void MetalGraph::moe_three_down(uint32_t type, GpuView down_weights, GpuView expert_hidden,
    GpuView expert_down, uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && p.moe_three_union_meta.owner && !p.recording && !p.compatible_pipelines &&
        moe_three_union_matrix_supported(type, down_weights, columns, rows, experts, true) &&
        selected && selected <= 64 && selected <= experts, "unioned three-row MoE down geometry");
    uint32_t layout_id = 0u;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (p.lane_shape(down_weights)) layout_id = 1u;
#endif
    const auto& fmt = encoding(type);
    const auto row_bytes = checked_mul(columns / fmt.elements_per_block, fmt.bytes_per_block);
    const auto mat_bytes = checked_mul(checked_mul(row_bytes, rows), experts);
    require(down_weights.bytes == mat_bytes && p.floats(expert_hidden) == checked_mul(checked_mul(3, selected), columns) &&
        p.floats(expert_down, true) == checked_mul(checked_mul(3, selected), rows), "unioned three-row MoE down buffer mismatch");
    static_cast<void>(p.allocation(down_weights));
    separate(expert_down, {down_weights, expert_hidden, p.moe_three_union_meta});
    p.select("moe_three_down", type, {columns, rows, experts, selected}, 128, layout_id);
    p.bind(down_weights, 0); p.bind(expert_hidden, 1); p.bind(expert_down, 2, true); p.bind(p.moe_three_union_meta, 3);
    p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), experts}, 4);
    p.dispatch_indirect(p.moe_three_union_meta, 12, MTLSizeMake(128, 1, 1));
}
void MetalGraph::moe_three_shared_router(GpuView weights, GpuView input, GpuView output, uint32_t columns) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && !p.recording && !p.compatible_pipelines && columns && columns <= 65536,
        "unioned three-row MoE shared router geometry");
    require(p.floats(weights) == columns && p.floats(input) == checked_mul(3, columns) && p.floats(output, true) == 3,
        "unioned three-row MoE shared router buffer mismatch");
    separate(output, {weights, input});
    p.select("moe_three_shared_router", uint32_t(-1), {columns, 1, 3, 0});
    p.bind(weights, 0); p.bind(input, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 2>{columns, p.small_gemv_splits}, 3);
    p.groups(MTLSizeMake(1, 1, 1));
}
void MetalGraph::moe_three_mixture(GpuView experts, GpuView weights, GpuView shared, GpuView shared_logit,
    GpuView output, uint32_t channels, uint32_t selected) {
    auto& p = *impl_; p.active();
    require(p.moe_three_union && !p.recording && !p.compatible_pipelines && channels && channels <= 65536 && selected && selected <= 64,
        "unioned three-row MoE mixture geometry");
    require(p.floats(experts) == checked_mul(checked_mul(3, selected), channels) && p.floats(weights) == checked_mul(3, selected) &&
        p.floats(shared) == checked_mul(3, channels) && p.floats(shared_logit) == 3 && p.floats(output, true) == checked_mul(3, channels),
        "unioned three-row MoE mixture buffer mismatch");
    separate(output, {experts, weights, shared, shared_logit});
    p.select("moe_three_mixture", uint32_t(-1), {channels, selected, 3, 0});
    p.bind(experts, 0); p.bind(weights, 1); p.bind(shared, 2); p.bind(shared_logit, 3); p.bind(output, 4, true);
    p.parameter(std::array<uint32_t, 2>{channels, selected}, 5);
    p.threads(3 * channels);
}
void MetalGraph::configure_prefill_m64(PrefillM64Mode mode) {
    auto& p = *impl_; p.idle();
    require(!p.prefill_m64_locked && !p.ledger.live_captures && uint8_t(mode) <= 2, "prefill m64 is fixed before the first batch");
    if (mode != PrefillM64Mode::off) {
        require([p.device supportsFamily:MTLGPUFamilyMetal4], "prefill m64 requires Metal 4");
        const auto s64 = p.pipeline("graph_dense_q8_staged_f32_m64");
        require(s64.staticThreadgroupMemoryLength + 16384 <= p.device.maxThreadgroupMemoryLength, "prefill m64 threadgroup memory exceeds device capability");
        if (p.prefill_multiseg) {
            for (const auto* name : {"graph_dense_q8_multiseg2_f32_m64", "graph_dense_q8_multiseg3_f32_m64"}) {
                const auto ms64 = p.pipeline(name);
                require(ms64.staticThreadgroupMemoryLength + 16384 <= p.device.maxThreadgroupMemoryLength, "multiseg prefill m64 threadgroup memory exceeds device capability");
            }
        }
    }
    p.prefill_m64 = mode;
}
PrefillM64Mode MetalGraph::prefill_m64_mode() const { impl_->thread_check(); return impl_->prefill_m64; }
bool MetalGraph::prefill_m64_active(uint32_t batches) const {
    impl_->thread_check();
    if (impl_->prefill_m64 == PrefillM64Mode::on) return batches >= 33;
    if (impl_->prefill_m64 == PrefillM64Mode::short_only) return batches >= 33 && batches <= 64;
    return false;
}
uint64_t MetalGraph::prefill_m64_calls() const { impl_->idle(); return impl_->prefill_m64_completed; }
void MetalGraph::configure_prefill_multiseg(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!enabled, "this optimisation was removed because its source could not be verified");
    require(!p.prefill_multiseg_locked && !p.ledger.live_captures, "prefill multiseg is fixed before the first batch");
    if (enabled) {
        require([p.device supportsFamily:MTLGPUFamilyMetal4], "prefill multiseg requires Metal 4");
        for (const auto* name : {"graph_dense_q8_multiseg2_f32", "graph_dense_q8_multiseg3_f32"}) {
            const auto ms = p.pipeline(name);
            require(ms.staticThreadgroupMemoryLength + 12288 <= p.device.maxThreadgroupMemoryLength, "prefill multiseg threadgroup memory exceeds device capability");
        }
        if (p.prefill_m64 != PrefillM64Mode::off) {
            for (const auto* name : {"graph_dense_q8_multiseg2_f32_m64", "graph_dense_q8_multiseg3_f32_m64"}) {
                const auto ms64 = p.pipeline(name);
                require(ms64.staticThreadgroupMemoryLength + 16384 <= p.device.maxThreadgroupMemoryLength, "prefill multiseg m64 threadgroup memory exceeds device capability");
            }
        }
    }
    p.prefill_multiseg = enabled;
}
bool MetalGraph::prefill_multiseg_enabled() const { impl_->thread_check(); return impl_->prefill_multiseg; }
uint64_t MetalGraph::prefill_multiseg_calls() const { impl_->idle(); return impl_->prefill_multiseg_completed; }
void MetalGraph::configure_prefill_swizzle(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.prefill_swizzle_locked && !p.ledger.live_captures, "prefill swizzle is fixed before the first batch");
    p.prefill_swizzle = enabled;
}
bool MetalGraph::prefill_swizzle_enabled() const { impl_->thread_check(); return impl_->prefill_swizzle; }
void MetalGraph::configure_qsa_prefill_fused(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.qsa_prefill_fused_locked && !p.ledger.live_captures, "fused QSA prefill prep is fixed before the first batch");
    if (enabled) {
        for (const auto* name : {"qsa_prefill_prep_query", "qsa_prefill_prep_kv", "qsa_prefill_prep_index_query", "qsa_prefill_prep_index_pool"})
            static_cast<void>(p.pipeline(name, uint32_t(-1), 128));
    }
    p.qsa_prefill_fused = enabled;
}
bool MetalGraph::qsa_prefill_fused_enabled() const { impl_->thread_check(); return impl_->qsa_prefill_fused; }
void MetalGraph::configure_hc_consumer_fused(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.hc_consumer_fused_locked && !p.ledger.live_captures, "fused HC consumer is fixed before the first batch");
    if (enabled) static_cast<void>(p.pipeline("graph_hc_rows_write_rms", uint32_t(-1), 128));
    p.hc_consumer_fused = enabled;
}
bool MetalGraph::hc_consumer_fused_enabled() const { impl_->thread_check(); return impl_->hc_consumer_fused; }
uint64_t MetalGraph::hc_consumer_fused_calls() const { impl_->idle(); return impl_->hc_consumer_fused_completed; }
void MetalGraph::residual_read(GpuView normalized, GpuView scores, GpuView inject, GpuView output,
                              GpuView scales, uint32_t branches, uint32_t channels) {
    auto& p = *impl_;
    p.active();
    require(branches && branches <= 64 && channels && channels <= 65536 && p.floats(normalized) == checked_mul(branches, channels) &&
            p.floats(scores) == p.floats(normalized) && p.floats(inject) == branches && p.floats(output, true) == channels &&
            p.floats(scales, true) == branches, "graph residual-read shape mismatch");
    separate(output, {normalized, scores, inject, scales}); separate(scales, {normalized, scores, inject});
    p.select("graph_residual_read"); p.bind(normalized, 0); p.bind(scores, 1); p.bind(inject, 2); p.bind(output, 3, true); p.bind(scales, 4, true);
    p.parameter(std::array<uint32_t, 2>{branches, channels}, 5); p.threads(std::max(branches, channels));
}
void MetalGraph::residual_write(GpuView streams, GpuView input, GpuView scales, GpuView destination) {
    auto& p = *impl_;
    p.active();
    const auto channels = p.floats(input), branches = p.floats(scales);
    require(branches <= 64 && channels <= 65536 && p.floats(streams) == checked_mul(branches, channels) && p.floats(destination, true) == p.floats(streams), "graph residual-write shape mismatch");
    separate(destination, {input, scales}); separate(destination, {streams}, true);
    p.select("graph_residual_write"); p.bind(streams, 0); p.bind(input, 1); p.bind(scales, 2); p.bind(destination, 3, true);
    p.parameter(std::array<uint32_t, 2>{branches, channels}, 4); p.threads(branches * channels);
}
#ifdef TOKSPUR_PRIVATE_HC_ROWS
void MetalGraph::prepare_hc_rows() {
    auto& p = *impl_; p.idle(); require(p.small_gemv_splits == 16, "HC rows retain split16 injection");
    for (const auto* name : {"graph_hc_rows_inject_partials", "graph_hc_rows_inject_reduce", "graph_hc_rows_read", "graph_hc_rows_write"}) static_cast<void>(p.pipeline(name));
    static_cast<void>(p.pipeline("graph_hc_rows_write_rms", uint32_t(-1), 128));
    if (p.prefill_dense_nax || p.prefill_grouped_nax || p.verify_nax) {
        static_cast<void>(p.pipeline("graph_hc_rows_inject_partials_nax"));
        static_cast<void>(p.pipeline("graph_hc_rows_write_rms_nax", uint32_t(-1), 128));
    }
}
void MetalGraph::configure_prefill_hc_rows(bool enabled) {
    auto& p = *impl_; p.idle(); require(!p.hc_rows_locked && !p.hc_rows_configured, "prefill HC rows are fixed before the first batch");
    require(!enabled || ((p.prefill_dense_staged_f32 || p.prefill_dense_nax) &&
        (p.prefill_grouped_f32 || p.prefill_grouped_auto || p.prefill_grouped_nax)), "prefill HC rows require staged or NAX prefill policy");
#ifndef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    require(!enabled || !p.prefill_hc_down_f32, "prefill HC rows refuse the legacy HC down experiment");
#endif
    if (enabled) {
        prepare_hc_rows();
        const uint64_t max_hc_tokens = p.prefill_dense_nax ? 2048ull : 512ull;
        p.hc_rows_scratch = zeros(max_hc_tokens * 4 * 16 * 4, "private exact HC row injection partials");
    }
    p.hc_rows_enabled = enabled; p.hc_rows_configured = true;
}
bool MetalGraph::prefill_hc_rows() const { impl_->thread_check(); return impl_->hc_rows_enabled; }
std::array<uint64_t, 3> MetalGraph::hc_rows_encoded() const { impl_->thread_check(); return impl_->hc_rows_encoded; }
void MetalGraph::prefill_hc_inject_rows(GpuView weights, GpuView normalized, GpuView inject, uint32_t tokens) {
    auto& p = *impl_; p.active();
    const bool v_nax = p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope;
    const uint32_t max_tokens = p.prefill_dense_nax ? 2048u : 512u;
    require((p.hc_rows_enabled || v_nax) && tokens >= 2 && tokens <= max_tokens, "prefill HC injection rows are disabled or out of bounds");
    hc_inject_rows(weights, normalized, slice(p.hc_rows_scratch, 0, uint64_t(tokens) * 4 * 16 * 4), inject, tokens);
}
void MetalGraph::hc_inject_rows(GpuView weights, GpuView normalized, GpuView partials, GpuView inject, uint32_t tokens) {
    auto& p = *impl_; p.active();
    const uint32_t max_tokens = p.prefill_dense_nax ? 2048u : 512u;
    require(!p.recording && !p.compatible_pipelines && p.small_gemv_splits == 16 && tokens >= 2 && tokens <= max_tokens, "HC rows require direct split16 batches of2 to512");
    require(p.floats(weights) == 4 * 10240 && p.floats64(normalized) == uint64_t(tokens) * 10240 && p.floats(partials, true) == uint64_t(tokens) * 4 * 16 && p.floats(inject, true) == uint64_t(tokens) * 4, "HC injection row geometry");
    validate_disjoint(std::array{weights, normalized}, std::array{partials, inject});
    require(p.hc_rows_encoded[0] < UINT64_MAX, "HC injection encoded counter exhausted");
    const std::array<uint32_t, 4> shape{10240, 4, 16, tokens};
    const bool use_nax = p.prefill_dense_nax || p.prefill_grouped_nax ||
        (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope);
    p.select(use_nax ? "graph_hc_rows_inject_partials_nax" : "graph_hc_rows_inject_partials"); p.bind(weights, 0); p.bind(normalized, 1); p.bind(partials, 2, true); p.parameter(shape, 3); p.groups(MTLSizeMake(1, 16, (tokens + 15) / 16));
    p.select("graph_hc_rows_inject_reduce"); p.bind(partials, 0); p.bind(inject, 1, true); p.parameter(shape, 2); p.threads(tokens * 4);
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.hc_rows_encoded[0];
}
void MetalGraph::hc_read_rows(GpuView normalized, GpuView scores, GpuView inject, GpuView output, GpuView scales, uint32_t tokens) {
    auto& p = *impl_; p.active();
    const uint32_t max_tokens = p.prefill_dense_nax ? 2048u : 512u;
    require(!p.recording && !p.compatible_pipelines && tokens >= 2 && tokens <= max_tokens, "HC read rows require direct batches of2 to512");
    require(p.floats64(normalized) == uint64_t(tokens) * 10240 && p.floats64(scores) == uint64_t(tokens) * 10240 && p.floats(inject) == uint64_t(tokens) * 4 && p.floats(output, true) == uint64_t(tokens) * 2560 && p.floats(scales, true) == uint64_t(tokens) * 4, "HC read row geometry");
    validate_disjoint(std::array{normalized, scores, inject}, std::array{output, scales});
    require(p.hc_rows_encoded[1] < UINT64_MAX, "HC read encoded counter exhausted");
    p.select("graph_hc_rows_read"); p.bind(normalized, 0); p.bind(scores, 1); p.bind(inject, 2); p.bind(output, 3, true); p.bind(scales, 4, true);
    p.parameter(std::array<uint32_t, 4>{4, 2560, tokens, 0}, 5); p.threads(tokens * (2560 / 4));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.hc_rows_encoded[1];
}
void MetalGraph::hc_write_rows(GpuView streams, GpuView input, GpuView scales, GpuView destination, uint32_t tokens) {
    auto& p = *impl_; p.active();
    const uint32_t max_tokens = p.prefill_dense_nax ? 2048u : 512u;
    require(!p.recording && !p.compatible_pipelines && tokens >= 2 && tokens <= max_tokens, "HC write rows require direct batches of2 to512");
    require(p.floats64(streams) == uint64_t(tokens) * 10240 && p.floats(input) == uint64_t(tokens) * 2560 && p.floats(scales) == uint64_t(tokens) * 4 && p.floats64(destination, true) == uint64_t(tokens) * 10240, "HC write row geometry");
    validate_disjoint(std::array{streams, input, scales}, std::array{destination});
    require(p.hc_rows_encoded[2] < UINT64_MAX, "HC write encoded counter exhausted");
    p.select("graph_hc_rows_write"); p.bind(streams, 0); p.bind(input, 1); p.bind(scales, 2); p.bind(destination, 3, true);
    p.parameter(std::array<uint32_t, 4>{4, 2560, tokens, 0}, 4); p.threads(tokens * (10240 / 4));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) ++p.hc_rows_encoded[2];
}
void MetalGraph::hc_write_rms_rows(GpuView streams, GpuView input, GpuView scales, GpuView gains,
    GpuView destination, GpuView normalized, uint32_t tokens, float epsilon) {
    auto& p = *impl_; p.active();
    const uint32_t max_tokens = p.prefill_dense_nax ? 2048u : 512u;
    require(!p.recording && !p.compatible_pipelines && tokens >= 2 && tokens <= max_tokens && std::isfinite(epsilon) && epsilon > 0,
        "HC write+RMS rows require direct batches of 2 to 512 and positive finite epsilon");
    require(p.floats64(streams) == uint64_t(tokens) * 10240 && p.floats(input) == uint64_t(tokens) * 2560 &&
        p.floats(scales) == uint64_t(tokens) * 4 && p.floats(gains) == 10240 &&
        p.floats64(destination, true) == uint64_t(tokens) * 10240 && p.floats64(normalized, true) == uint64_t(tokens) * 10240,
        "HC write+RMS row geometry");
    validate_disjoint(std::array{input, scales, gains}, std::array{destination, normalized});
    separate(normalized, {streams, destination});
    separate(destination, {streams}, true);
    require(p.hc_rows_encoded[2] < UINT64_MAX && p.hc_consumer_fused_pending < UINT64_MAX - p.hc_consumer_fused_completed,
        "HC write+RMS encoded counter exhausted");
    const bool use_nax = p.prefill_dense_nax || p.prefill_grouped_nax ||
        (p.verify_nax && p.verify_nax_prepared && p.verify_nax_scope);
    p.select(use_nax ? "graph_hc_rows_write_rms_nax" : "graph_hc_rows_write_rms", uint32_t(-1), {4, 2560, tokens, 0}, 128);
    p.bind(streams, 0); p.bind(input, 1); p.bind(scales, 2); p.bind(gains, 3);
    p.bind(destination, 4, true); p.bind(normalized, 5, true);
    p.parameter(std::array<uint32_t, 4>{4, 2560, tokens, std::bit_cast<uint32_t>(epsilon)}, 6);
    p.dispatch(true, MTLSizeMake(use_nax ? tokens * 4u : tokens, 1, 1), MTLSizeMake(128, 1, 1));
    if (!p.mtp_priming_scope && !p.verify_nax_scope) { ++p.hc_rows_encoded[2]; ++p.hc_consumer_fused_pending; }
}
#endif
void MetalGraph::convolution(GpuView history, GpuView input, GpuView weights, GpuView next_history, GpuView output, uint32_t kernel, uint32_t dilation) {
    auto& p = *impl_;
    p.active();
    const auto channels = p.floats(input);
    // Kernel=1 needs no history; the production model uses 4. A dummy nonempty
    // view for an absent history is deliberately not an accepted convention.
    require(kernel >= 2 && kernel <= 64 && dilation && dilation <= 16 && channels <= 65536 && p.floats(history) == checked_mul(checked_mul(channels, kernel - 1), dilation) &&
            p.floats(next_history, true) == p.floats(history) && p.floats(weights) == checked_mul(channels, kernel) && p.floats(output, true) == channels, "graph convolution shape mismatch");
    separate(output, {history, input, weights, next_history}); separate(next_history, {history, input, weights});
    p.select("graph_convolution"); p.bind(history, 0); p.bind(input, 1); p.bind(weights, 2); p.bind(next_history, 3, true); p.bind(output, 4, true);
    p.parameter(std::array<uint32_t, 4>{channels, kernel, dilation, 0}, 5); p.threads(channels);
}

void MetalGraph::ple_gate(GpuView query, GpuView key, GpuView value, GpuView gated, GpuView gates, uint32_t branches, uint32_t channels, PleGateTransform transform) {
    auto& p = *impl_; p.active();
    require(branches && branches <= 64 && channels && channels <= 65536 && p.floats(query) == checked_mul(branches, channels) &&
        p.floats(key) == p.floats(query) && p.floats(value) == channels && p.floats(gated, true) == p.floats(query) && p.floats(gates, true) == branches &&
        uint32_t(transform) <= uint32_t(PleGateTransform::signed_root), "PLE gate shape/mode mismatch");
    validate_disjoint(std::array{query, key, value}, std::array{gated, gates});
    p.select("graph_ple_gate"); p.bind(query, 0); p.bind(key, 1); p.bind(value, 2); p.bind(gated, 3, true); p.bind(gates, 4, true);
    p.parameter(std::array<uint32_t, 4>{branches, channels, uint32_t(transform), 0}, 5);
    p.groups(MTLSizeMake((branches + 3) / 4, 1, 1));
}
void MetalGraph::ple_write(GpuView streams, GpuView gated, GpuView convolved, GpuView output) {
    auto& p = *impl_; p.active(); const auto count = p.floats(streams);
    require(p.floats(gated) == count && p.floats(convolved) == count && p.floats(output, true) == count, "PLE write shape mismatch");
    validate_disjoint(std::array{streams, gated, convolved}, std::array{output});
    p.select("graph_ple_write"); p.bind(streams, 0); p.bind(gated, 1); p.bind(convolved, 2); p.bind(output, 3, true);
    p.parameter(count, 4); p.threads(count);
}
void MetalGraph::ple_gate_rows(GpuView query, GpuView key, GpuView value, GpuView gated,
                               uint32_t branches, uint32_t channels, uint32_t rows, PleGateTransform transform) {
    auto& p = *impl_; p.active();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    const uint32_t max_rows = p.prefill_dense_nax ? 2048u : 512u;
#else
    const uint32_t max_rows = 512u;
#endif
    const uint64_t total_hc = checked_mul(checked_mul(uint64_t(rows), branches), channels);
    const uint64_t total_val = checked_mul(uint64_t(rows), channels);
    require(!p.recording && !p.compatible_pipelines && rows && rows <= max_rows && branches && branches <= 64 && channels && channels <= 65536 &&
        p.floats64(query) == total_hc && p.floats64(key) == total_hc && p.floats64(value) == total_val && p.floats64(gated, true) == total_hc &&
        uint32_t(transform) <= uint32_t(PleGateTransform::signed_root), "PLE gate rows shape/mode mismatch");
    validate_disjoint(std::array{query, key, value}, std::array{gated});
    p.select("nax_ple_gate_rows", uint32_t(-1), {branches, channels, uint32_t(transform), rows});
    p.bind(query, 0); p.bind(key, 1); p.bind(value, 2); p.bind(gated, 3, true);
    p.parameter(std::array<uint32_t, 4>{branches, channels, uint32_t(transform), rows}, 4);
    p.groups(MTLSizeMake((rows * branches + 3) / 4, 1, 1));
}
void MetalGraph::ple_conv_write_rows(GpuView history, GpuView norm_gated, GpuView weights, GpuView streams, GpuView gated,
                                     GpuView next_history, GpuView output, uint32_t width, uint32_t kernel, uint32_t dilation,
                                     uint32_t rows, uint64_t stream_stride_bytes, uint64_t output_stride_bytes) {
    auto& p = *impl_; p.active();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    const uint32_t max_rows = p.prefill_dense_nax ? 2048u : 512u;
#else
    const uint32_t max_rows = 512u;
#endif
    const uint64_t hist_floats = checked_mul(checked_mul(uint64_t(kernel - 1), dilation), width);
    const uint64_t rows_floats = checked_mul(uint64_t(rows), width);
    const uint64_t streams_bytes = checked_add(checked_mul(uint64_t(rows - 1), stream_stride_bytes), checked_mul(uint64_t(width), 4));
    const uint64_t output_bytes = checked_add(checked_mul(uint64_t(rows - 1), output_stride_bytes), checked_mul(uint64_t(width), 4));
    static_cast<void>(p.allocation(streams)); static_cast<void>(p.allocation(output, true));
    require(!p.recording && !p.compatible_pipelines && rows && rows <= max_rows && kernel >= 2 && kernel <= 64 && dilation && dilation <= 16 &&
        width && width <= 65536 && p.floats(history) == hist_floats && p.floats(next_history, true) == hist_floats &&
        p.floats64(norm_gated) == rows_floats && p.floats64(gated) == rows_floats && p.floats(weights) == checked_mul(width, kernel) &&
        streams.bytes >= streams_bytes && output.bytes >= output_bytes && stream_stride_bytes >= uint64_t(width) * 4 &&
        output_stride_bytes >= uint64_t(width) * 4 && stream_stride_bytes % 4 == 0 && output_stride_bytes % 4 == 0 &&
        streams.offset % 4 == 0 && output.offset % 4 == 0, "PLE conv+write rows shape mismatch");
    separate(output, {history, norm_gated, weights, gated, next_history});
    separate(output, {streams}, stream_stride_bytes == output_stride_bytes);
    separate(next_history, {history, norm_gated, weights, streams, gated});
    p.select("nax_ple_conv_write_rows", uint32_t(-1), {width, kernel, dilation, rows});
    p.bind(history, 0); p.bind(norm_gated, 1); p.bind(weights, 2); p.bind(streams, 3); p.bind(gated, 4);
    p.bind(next_history, 5, true); p.bind(output, 6, true);
    p.parameter(std::array<uint32_t, 4>{width, kernel, dilation, rows}, 8);
    p.parameter(std::array<uint64_t, 2>{stream_stride_bytes, output_stride_bytes}, 9);
    p.threads(width);
}
void MetalGraph::gated_mix(GpuView normalized, GpuView scores, GpuView output, uint32_t branches, uint32_t channels) {
    auto& p = *impl_; p.active();
    require(branches && branches <= 64 && channels && channels <= 65536 && p.floats(normalized) == checked_mul(branches, channels) &&
        p.floats(scores) == p.floats(normalized) && p.floats(output, true) == channels, "final gated mix shape mismatch");
    validate_disjoint(std::array{normalized, scores}, std::array{output});
    p.select("graph_gated_mix"); p.bind(normalized, 0); p.bind(scores, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 2>{branches, channels}, 3); p.threads(channels);
}
void MetalGraph::delta(GpuView state, GpuView query, GpuView key, GpuView value, GpuView decay, GpuView beta,
                      GpuView next_state, GpuView output, uint32_t heads, uint32_t key_width, uint32_t value_width) {
    auto& p = *impl_;
    p.active();
    require(heads && heads <= 64 && key_width && key_width <= 256 && value_width && value_width <= 256 &&
            p.floats(state) == checked_mul(checked_mul(heads, key_width), value_width) && p.floats(next_state, true) == p.floats(state) &&
            p.floats(query) == heads * key_width && p.floats(key) == heads * key_width && p.floats(value) == heads * value_width &&
            p.floats(output, true) == heads * value_width && p.floats(decay) == heads && p.floats(beta) == heads, "graph delta shape mismatch");
    separate(output, {state, query, key, value, decay, beta, next_state}); separate(next_state, {state, query, key, value, decay, beta});
    p.select("graph_delta"); p.bind(state, 0); p.bind(query, 1); p.bind(key, 2); p.bind(value, 3); p.bind(decay, 4); p.bind(beta, 5); p.bind(next_state, 6, true); p.bind(output, 8, true);
    p.parameter(std::array<uint32_t, 4>{heads, key_width, value_width, 0}, 9); p.threads(heads * value_width);
}

void MetalGraph::route(GpuView logits, GpuView indices, GpuView weights, uint32_t selected, RouterMode mode, bool normalize_selected) {
    auto& p = *impl_;
    p.active();
    const auto experts = p.floats(logits);
    require(experts <= 4096 && selected && selected <= 64 && selected <= experts && uint32_t(mode) <= 1 &&
            p.floats(indices, true) == selected && p.floats(weights, true) == selected, "graph router shape/mode mismatch");
    separate(indices, {logits, weights}); separate(weights, {logits});
    p.select("graph_route"); p.bind(logits, 0); p.bind(indices, 1, true); p.bind(weights, 2, true);
    p.parameter(std::array<uint32_t, 4>{experts, selected, uint32_t(mode), normalize_selected}, 3);
    p.groups(MTLSizeMake(1, 1, 1));
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    p.carrier_route(indices, weights, 1, selected, uint32_t(experts));
#endif
}

void MetalGraph::configure_raw_decode(bool enabled) {
    auto& p = *impl_; p.idle();
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    require(!p.raw_decode_locked && !p.ledger.live_captures, "raw decode selection is fixed before the first batch");
    if (enabled) {
        static_cast<void>(p.pipeline("raw_decode_q4_k", uint32_t(-1), 64));
        static_cast<void>(p.pipeline("raw_decode_q5_1", uint32_t(-1), 64));
    }
    p.raw_decode = enabled;
#else
    require(!enabled, "raw decode selection requires the private build");
#endif
}
bool MetalGraph::raw_decode_enabled() const {
    impl_->thread_check();
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    return impl_->raw_decode;
#else
    return false;
#endif
}
MetalGraph::RawDecodeRoutes MetalGraph::raw_decode_routes() const {
    impl_->idle();
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    return impl_->raw_decode_completed;
#else
    return {};
#endif
}
void MetalGraph::routed_gemv(uint32_t type, GpuView matrices, GpuView indices, GpuView input, GpuView output,
                            uint32_t columns, uint32_t rows, uint32_t experts, bool broadcast) {
    auto& p = *impl_;
    p.active();
    require(can_decode(type) && columns && columns <= 65536 && rows && rows <= 65536 && experts && experts <= 4096, "unsupported routed GEMV shape/type");
    const auto selected = p.floats(indices);
    const auto& layout = encoding(type);
    const auto row_bytes = checked_mul(columns / layout.elements_per_block, layout.bytes_per_block);
    require(selected <= 64 && selected <= experts && columns % layout.elements_per_block == 0 &&
            matrices.bytes == checked_mul(checked_mul(row_bytes, rows), experts) &&
            p.floats(input) == checked_mul(columns, broadcast ? 1 : selected) && p.floats(output, true) == checked_mul(rows, selected), "routed GEMV buffer shape mismatch");
    static_cast<void>(p.allocation(matrices));
    separate(output, {matrices, indices, input});
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    if (p.raw_decode && (type == 12 || type == 7)) {
        require(!p.recording && !p.compatible_pipelines && p.fixed_route_profile_allowed(), "raw decode requires prepared direct batches");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        require(!p.lane_shape(matrices), "raw decode cannot consume packed weight storage");
#endif
        raw_decode::Plan plan{};
        try {
            plan = raw_decode::make_plan(static_cast<raw_decode::Format>(type), columns, rows, experts, selected, broadcast);
            raw_decode::validate_bindings(plan, matrices.bytes, input.bytes, output.bytes, indices.bytes);
        } catch (const std::exception& error) { throw FormatError(std::string("raw decode plan: ") + error.what()); }
        const size_t bucket = type == 12 ? 0 : 1;
        require(p.raw_decode_pending[bucket] < UINT64_MAX - p.raw_decode_completed[bucket], "raw decode route count overflow");
        p.select(plan.kernel, uint32_t(-1), {columns, rows, experts, selected}, plan.threads);
        p.bind(matrices, 0); p.bind(input, 1); p.bind(output, 2, true); p.bind(indices, 4); p.parameter(plan.args, 3);
        p.dispatch(true, MTLSizeMake(plan.grid_x, plan.grid_y, 1), MTLSizeMake(plan.threads, 1, 1));
        ++p.raw_decode_pending[bucket]; return;
    }
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (const auto* lane = p.lane_shape(matrices)) {
        require(type == 7 && lane->encoding == type && lane->columns == columns && lane->rows == rows && lane->experts == experts,
            "lane routed consumer shape mismatch");
        p.select("route_lanes_q51", uint32_t(-1), {columns, rows, experts, selected});
        p.bind(matrices, 0); p.bind(input, 1); p.bind(output, 2, true); p.bind(indices, 4);
        p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), experts}, 3);
        p.parameter(std::array<uint32_t, 2>{selected, broadcast}, 5);
        p.groups(MTLSizeMake((rows + 3) / 4, selected, 1)); return;
    }
#endif
    const bool block = (type == 8 && (p.block_gemv_formats & 1)) || (type == 12 && (p.block_gemv_formats & 2));
    p.select(block ? "graph_block_routed_gemv" : "graph_routed_gemv", type, {columns, rows, experts, selected}); p.bind(matrices, 0); p.bind(input, 1); p.bind(output, 2, true); p.bind(indices, 4);
    p.parameter(std::array<uint32_t, 4>{columns, rows, static_cast<uint32_t>(row_bytes), experts}, 3);
    p.parameter(std::array<uint32_t, 2>{selected, broadcast}, 5);
    p.groups(MTLSizeMake((rows + 3) / 4, selected, 1));
}

void MetalGraph::mixture(GpuView experts, GpuView weights, GpuView shared, GpuView shared_logit, GpuView output) {
    auto& p = *impl_;
    p.active();
    const auto channels = p.floats(shared), selected = p.floats(weights);
    require(channels <= 65536 && selected <= 64 && p.floats(shared_logit) == 1 &&
            p.floats(experts) == checked_mul(channels, selected) && p.floats(output, true) == channels, "graph mixture shape mismatch");
    separate(output, {experts, weights, shared, shared_logit});
    p.select("graph_mixture"); p.bind(experts, 0); p.bind(weights, 1); p.bind(shared, 2); p.bind(shared_logit, 3); p.bind(output, 4, true);
    p.parameter(std::array<uint32_t, 2>{channels, selected}, 5); p.threads(channels);
}

void MetalGraph::gdn_prepare(GpuView input, GpuView query, GpuView key, GpuView value, GdnShape shape, GdnPolicy policy) {
    auto& p = *impl_;
    p.active();
    validate_gdn_policy(policy);
    require(p.floats(input) == gdn_qkv_count(shape) && p.floats(query, true) == shape.value_heads * shape.key_width &&
            p.floats(key, true) == shape.value_heads * shape.key_width && p.floats(value, true) == shape.value_heads * shape.value_width, "graph GDN preparation shape mismatch");
    validate_disjoint(std::array{input}, std::array{query, key, value});
    p.select("graph_gdn_prepare"); p.bind(input, 0); p.bind(query, 1, true); p.bind(key, 2, true); p.bind(value, 3, true);
    p.parameter(std::array<uint32_t, 4>{shape.key_heads, shape.value_heads, shape.key_width, shape.value_width}, 4);
    p.parameter(std::array<float, 2>{policy.query_scale, policy.l2_epsilon}, 5); p.parameter(uint32_t(policy.l2_mode), 6);
    p.groups(MTLSizeMake((shape.value_heads + 3) / 4, 1, 1));
}

void MetalGraph::gdn_gates(GpuView alpha, GpuView beta_logits, GpuView rates, GpuView bias, GpuView decay, GpuView beta) {
    auto& p = *impl_;
    p.active();
    const auto heads = p.floats(alpha);
    require(heads <= 64 && p.floats(beta_logits) == heads && p.floats(rates) == heads && p.floats(bias) == heads &&
            p.floats(decay, true) == heads && p.floats(beta, true) == heads, "graph GDN gates shape mismatch");
    validate_disjoint(std::array{alpha, beta_logits, rates, bias}, std::array{decay, beta});
    p.select("graph_gdn_gates"); p.bind(alpha, 0); p.bind(beta_logits, 1); p.bind(rates, 2); p.bind(bias, 3);
    p.bind(decay, 4, true); p.bind(beta, 5, true); p.parameter(heads, 6); p.threads(heads);
}

void MetalGraph::split_query_gate(GpuView packed, GpuView query, GpuView gate, uint32_t heads, uint32_t width, QueryPacking packing) {
    auto& p = *impl_; p.active();
    require(heads && heads <= 64 && width && width <= 256 && uint32_t(packing) <= 1 && p.floats(packed) == 2ull * heads * width &&
        p.floats(query, true) == heads * width && p.floats(gate, true) == heads * width, "query/gate GPU shape mismatch");
    validate_disjoint(std::array{packed}, std::array{query, gate});
    p.select("qsa_split"); p.bind(packed, 0); p.bind(query, 1, true); p.bind(gate, 2, true);
    p.parameter(std::array<uint32_t, 4>{heads, width, uint32_t(packing), 0}, 3); p.threads(heads * width);
}
void MetalGraph::rotary(GpuView input, GpuView frequencies, GpuView output, uint32_t heads, uint32_t width, uint32_t position, RotaryPairs pairs) {
    auto& p = *impl_; p.active();
    const auto half = p.floats(frequencies);
    require(heads && heads <= 64 && width && width <= 256 && half * 2 <= width && position < 65536 && uint32_t(pairs) <= 1 &&
        p.floats(input) == heads * width && p.floats(output, true) == heads * width, "rotary GPU shape mismatch");
    validate_disjoint(std::array{input, frequencies}, std::array{output});
    p.select("qsa_rotary"); p.bind(input, 0); p.bind(frequencies, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{heads, width, half, position}, 3); p.parameter(uint32_t(pairs), 4); p.threads(heads * width);
}
void MetalGraph::store_bf16(GpuView input, GpuView row) {
    auto& p = *impl_; p.active();
    const auto count = p.floats(input);
    require(count <= 65536 && row.bytes == uint64_t(count) * 2, "BF16 GPU storage shape mismatch");
    validate_disjoint(std::array{input}, std::array{row});
    p.select("qsa_store_bf16"); p.bind(input, 0); p.bind(row, 1, true); p.parameter(count, 2); p.threads(count);
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
void MetalGraph::prepare_mtp_head() {
    auto& p = *impl_; p.idle();
    static_cast<void>(p.pipeline("mtp_concat"));
    static_cast<void>(p.pipeline("mtp_dense_attention"));
    static_cast<void>(p.pipeline("mtp_embed_from_topk_q8"));
    static_cast<void>(p.pipeline("mtp_sample_and_embed_from_topk_q8"));
    if (p.prefill_dense_nax) {
        static_cast<void>(p.pipeline("nax_mtp_concat_rows"));
        static_cast<void>(p.pipeline("nax_mtp_rope_store_kv_bf16"));
    }
    p.mtp_head_prepared = true;
}
void MetalGraph::prepare_mtp_head_tiled() {
    auto& p = *impl_; p.idle();
    static_cast<void>(p.pipeline("mtp_dense_attention_tile"));
    static_cast<void>(p.pipeline("mtp_dense_attention_merge"));
    p.mtp_head_tiled_prepared = true;
}
void MetalGraph::mtp_concat(GpuView embedding, GpuView hidden, GpuView output, uint32_t branches, uint32_t channels) {
    auto& p = *impl_; p.active();
    require(p.mtp_head_prepared && branches && branches <= 64 && channels && channels <= 65536 &&
        uint64_t(branches) * channels <= 65536 && p.floats(embedding) == channels &&
        p.floats(hidden) == branches * channels && p.floats(output, true) == branches * channels * 2,
        "MTP normalized concat geometry mismatch");
    validate_disjoint(std::array{embedding, hidden}, std::array{output});
    p.select("mtp_concat"); p.bind(embedding, 0); p.bind(hidden, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 2>{branches, channels}, 3); p.threads(branches * channels * 2);
}
void MetalGraph::mtp_embed_from_topk_q8(GpuView token_embd, GpuView topk_row, GpuView output,
                                        uint32_t channels, uint32_t vocabulary) {
    static_assert(offsetof(GpuLogitTopKRow, count) == 4 &&
                  offsetof(GpuLogitTopKRow, first_nonfinite_col) == 8 &&
                  offsetof(GpuLogitTopKRow, ids) == 16, "GpuLogitTopKRow ABI mismatch");
    auto& p = *impl_; p.active();
    require(p.mtp_head_prepared && channels && channels <= 65536 && channels % 32 == 0 &&
        vocabulary && vocabulary <= maximum_sampling_vocabulary &&
        topk_row.bytes == sizeof(GpuLogitTopKRow) && topk_row.offset % 4 == 0 &&
        token_embd.bytes == uint64_t(channels / 32) * 34 * vocabulary &&
        p.floats(output, true) == channels, "MTP top-k Q8_0 embedding geometry mismatch");
    static_cast<void>(p.allocation(token_embd)); static_cast<void>(p.allocation(topk_row));
    validate_disjoint(std::array{token_embd, topk_row}, std::array{output});
    p.select("mtp_embed_from_topk_q8");
    p.bind(token_embd, 0);
    p.bind(topk_row, 1);
    p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 2>{channels, vocabulary}, 3);
    p.threads(channels);
}
void MetalGraph::mtp_sample_and_embed_from_topk_q8(GpuView token_embd, GpuView topk_row, GpuView tickets_buf,
                                                   GpuView law_out, GpuView output, uint32_t channels,
                                                   uint32_t embd_vocabulary, uint32_t max_k, bool embed_output,
                                                   float inv_temp, float min_p, float top_p) {
    struct MtpSampleEmbedArgs {
        uint32_t channels, embd_vocab, max_k, embed_flag;
        float inv_temp, min_p, top_p, pad;
    };
    static_assert(sizeof(MtpSampleEmbedArgs) == 32, "MtpSampleEmbedArgs ABI");
    static_assert(offsetof(GpuLogitTopKRow, vocabulary) == 0 &&
                  offsetof(GpuLogitTopKRow, count) == 4 &&
                  offsetof(GpuLogitTopKRow, first_nonfinite_col) == 8 &&
                  offsetof(GpuLogitTopKRow, ids) == 16 &&
                  offsetof(GpuLogitTopKRow, logits) == 272, "GpuLogitTopKRow ABI mismatch");
    auto& p = *impl_; p.active();
    require(p.mtp_head_prepared && channels && channels <= 65536 && channels % 32 == 0 &&
        embd_vocabulary && embd_vocabulary <= maximum_sampling_vocabulary &&
        max_k >= 1 && max_k <= 20 && std::isfinite(inv_temp) && inv_temp > 0.0f &&
        topk_row.bytes == sizeof(GpuLogitTopKRow) && topk_row.offset % 4 == 0 &&
        tickets_buf.bytes == 32 && tickets_buf.offset % 8 == 0 &&
        law_out.bytes == 256 && law_out.offset % 8 == 0 &&
        token_embd.bytes == uint64_t(channels / 32) * 34 * embd_vocabulary &&
        p.floats(output, embed_output) == channels,
        "MTP sampled top-k Q8_0 embedding geometry mismatch");
    static_cast<void>(p.allocation(token_embd)); static_cast<void>(p.allocation(topk_row));
    static_cast<void>(p.allocation(tickets_buf, true)); static_cast<void>(p.allocation(law_out, true));
    validate_disjoint(std::array{token_embd, topk_row}, std::array{tickets_buf, law_out, output});
    validate_disjoint(std::array{tickets_buf}, std::array{law_out, output});
    validate_disjoint(std::array{law_out}, std::array{output});
    const MtpSampleEmbedArgs args{channels, embd_vocabulary, max_k, embed_output ? 1u : 0u, inv_temp, min_p, top_p, 0.0f};
    p.select("mtp_sample_and_embed_from_topk_q8");
    p.bind(token_embd, 0);
    p.bind(topk_row, 1);
    p.bind(tickets_buf, 2, true);
    p.bind(law_out, 3, true);
    p.bind(output, 4, embed_output);
    p.parameter(args, 5);
    p.groups(MTLSizeMake(1, 1, 1));
}
void MetalGraph::mtp_concat_rows(GpuView embedding, GpuView hidden, GpuView output,
                                 uint32_t branches, uint32_t channels, uint32_t rows) {
    auto& p = *impl_; p.active();
    const uint64_t emb_floats = checked_mul(uint64_t(rows), channels);
    const uint64_t hc_floats = checked_mul(emb_floats, branches);
    const uint64_t out_floats = checked_mul(hc_floats, 2);
    require(!p.recording && !p.compatible_pipelines && p.mtp_head_prepared && p.prefill_dense_nax &&
        rows && rows <= 512 && branches && branches <= 64 && channels && channels <= 65536 &&
        p.floats(embedding) == emb_floats && p.floats(hidden) == hc_floats && p.floats(output, true) == out_floats,
        "MTP normalized concat rows geometry mismatch");
    validate_disjoint(std::array{embedding, hidden}, std::array{output});
    p.select("nax_mtp_concat_rows", uint32_t(-1), {branches, channels, rows, 0});
    p.bind(embedding, 0); p.bind(hidden, 1); p.bind(output, 2, true);
    p.parameter(std::array<uint32_t, 4>{branches, channels, rows, 0}, 3);
    p.threads(uint32_t(out_floats));
}
void MetalGraph::mtp_rope_store_kv_bf16(GpuView norm_key, GpuView raw_value, GpuView frequencies,
                                        GpuView bf16_keys, GpuView bf16_values, uint32_t kv_heads, uint32_t head_width,
                                        uint32_t first_position, RotaryPairs pairs, uint32_t rows) {
    auto& p = *impl_; p.active();
    const auto half = p.floats(frequencies);
    const uint64_t kv_elems = checked_mul(checked_mul(uint64_t(rows), kv_heads), head_width);
    static_cast<void>(p.allocation(bf16_keys, true)); static_cast<void>(p.allocation(bf16_values, true));
    require(!p.recording && !p.compatible_pipelines && p.mtp_head_prepared && p.prefill_dense_nax &&
        rows && rows <= 512 && kv_heads && kv_heads <= 64 && head_width && head_width <= 256 && half * 2 <= head_width &&
        first_position + rows <= 65536 && uint32_t(pairs) <= 1 &&
        p.floats(norm_key) == kv_elems && p.floats(raw_value) == kv_elems &&
        bf16_keys.bytes >= kv_elems * 2 && bf16_values.bytes >= kv_elems * 2 &&
        bf16_keys.offset % 2 == 0 && bf16_values.offset % 2 == 0,
        "MTP rotary+store BF16 rows geometry mismatch");
    validate_disjoint(std::array{norm_key, raw_value, frequencies}, std::array{bf16_keys, bf16_values});
    p.select("nax_mtp_rope_store_kv_bf16", uint32_t(-1), {kv_heads, head_width, first_position, rows});
    p.bind(norm_key, 0); p.bind(raw_value, 1); p.bind(frequencies, 2);
    p.bind(bf16_keys, 3, true); p.bind(bf16_values, 4, true);
    p.parameter(std::array<uint32_t, 4>{kv_heads, head_width, half, first_position}, 5);
    p.parameter(std::array<uint32_t, 2>{uint32_t(pairs), rows}, 6);
    p.threads(uint32_t(kv_elems));
}
void MetalGraph::mtp_dense_attention(GpuView query, GpuView keys, GpuView values, GpuView output,
                                    uint32_t prefix, uint32_t heads, uint32_t kv_heads, uint32_t width) {
    auto& p = *impl_; p.active();
    require(p.mtp_head_prepared && prefix && prefix <= 65536 && heads && heads <= 64 && kv_heads &&
        kv_heads <= heads && heads % kv_heads == 0 && width && width <= 256 && width % 2 == 0 &&
        p.floats(query) == heads * width && p.floats(output, true) == heads * width && keys.bytes == values.bytes &&
        keys.bytes >= uint64_t(prefix) * kv_heads * width * 2, "MTP dense attention geometry mismatch");
    validate_disjoint(std::array{query, keys, values}, std::array{output});
    p.select("mtp_dense_attention"); p.bind(query, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(output, 3, true);
    p.parameter(std::array<uint32_t, 4>{heads, kv_heads, width, prefix}, 4);
    p.groups(MTLSizeMake((heads + 3) / 4, 1, 1));
}
void MetalGraph::mtp_dense_attention_tiled(GpuView query, GpuView keys, GpuView values, GpuView output,
                                          GpuView partials, uint32_t prefix, uint32_t heads,
                                          uint32_t kv_heads, uint32_t width, uint32_t tile_tokens) {
    auto& p = *impl_; p.active();
    require(p.mtp_head_tiled_prepared && prefix && prefix <= 65536 && heads && heads <= 64 && kv_heads &&
        kv_heads <= heads && heads % kv_heads == 0 && width && width <= 256 && width % 2 == 0 &&
        (tile_tokens == 128 || tile_tokens == 256) && p.floats(query) == heads * width &&
        p.floats(output, true) == heads * width && keys.bytes == values.bytes &&
        keys.bytes >= uint64_t(prefix) * kv_heads * width * 2, "MTP tiled draft attention geometry mismatch");
    const auto tiles = (prefix + tile_tokens - 1) / tile_tokens;
    const auto needed = checked_mul(checked_mul(uint64_t(heads), tiles), uint64_t(width) + 2);
    require(p.floats(partials, true) >= needed && partials.bytes <= 64'000'000,
        "MTP tiled draft attention workspace bound");
    validate_disjoint(std::array{query, keys, values}, std::array{output, partials});
    const std::array<uint32_t, 4> shape{heads, kv_heads, width, prefix};
    p.select("mtp_dense_attention_tile");
    p.bind(query, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(partials, 3, true);
    p.parameter(shape, 4); p.parameter(tile_tokens, 5);
    p.groups(MTLSizeMake((heads + 3) / 4, tiles, 1));
    p.select("mtp_dense_attention_merge");
    p.bind(partials, 0); p.bind(output, 1, true); p.parameter(shape, 4); p.parameter(tile_tokens, 5);
    p.groups(MTLSizeMake((heads + 3) / 4, 1, 1));
}
#endif
void MetalGraph::index_accumulate(GpuView sum, GpuView key, GpuView next, GpuView pooled, uint32_t position, uint32_t compression) {
    auto& p = *impl_; p.active();
    const auto width = p.floats(sum);
    require(width <= 256 && p.floats(key) == width && p.floats(next, true) == width && p.floats(pooled, true) == width &&
        position < 65536 && compression && compression <= 16, "index pooling GPU shape mismatch");
    validate_disjoint(std::array{sum, key}, std::array{next, pooled});
    p.select("qsa_index_accumulate"); p.bind(sum, 0); p.bind(key, 1); p.bind(next, 2, true); p.bind(pooled, 3, true);
    p.parameter(std::array<uint32_t, 4>{width, (position + 1) % compression == 0, compression, 0}, 4); p.threads(width);
}
void MetalGraph::index_scores(GpuView query, GpuView keys, GpuView scores, uint32_t heads, uint32_t width, uint32_t blocks) {
    auto& p = *impl_; p.active();
    require(heads && heads <= 8 && width && width <= 256 && blocks && blocks <= 65536 && p.floats(query) == heads * width &&
        keys.bytes >= uint64_t(blocks) * width * 2 && p.floats(scores, true) >= blocks, "index scoring GPU shape mismatch");
    validate_disjoint(std::array{query, keys}, std::array{scores});
    p.select("qsa_index_scores"); p.bind(query, 0); p.bind(keys, 1); p.bind(scores, 2, true);
    p.parameter(std::array<uint32_t, 4>{heads, width, blocks, 0}, 3);
    p.groups(MTLSizeMake((blocks + 3) / 4, 1, 1));
}
void MetalGraph::select_blocks(GpuView scores, GpuView ids, SelectionScratch s, uint32_t blocks, uint32_t budget) {
    auto& p = *impl_; p.active();
    require(blocks && blocks <= 65536 && budget && budget <= 512 && p.floats(scores) >= blocks && p.floats(ids, true) == budget, "sparse selection GPU shape mismatch");
    const auto chunks = (blocks + 511) / 512, slots = chunks * budget;
    for (const auto v : {s.scores_a, s.ids_a, s.scores_b, s.ids_b}) require(p.floats(v, true) >= slots, "sparse selection workspace too short");
    validate_disjoint(std::array{scores}, std::array{ids, s.scores_a, s.ids_a, s.scores_b, s.ids_b});
    p.select("qsa_sort_chunks"); p.bind(scores, 0); p.bind(s.scores_a, 1, true); p.bind(s.ids_a, 2, true);
    p.parameter(std::array<uint32_t, 2>{blocks, budget}, 3);
    p.groups(MTLSizeMake(chunks, 1, 1));
    auto a_scores = s.scores_a, a_ids = s.ids_a, b_scores = s.scores_b, b_ids = s.ids_b;
    for (uint32_t span = 512; span < blocks; span *= 2) {
        p.select("qsa_merge"); p.bind(a_scores, 0); p.bind(a_ids, 1); p.bind(b_scores, 2, true); p.bind(b_ids, 3, true);
        p.parameter(std::array<uint32_t, 4>{blocks, budget, span, 0}, 4);
        p.groups(MTLSizeMake((blocks + span * 2 - 1) / (span * 2), 1, 1));
        std::swap(a_scores, b_scores); std::swap(a_ids, b_ids);
    }
    p.select("qsa_canonical_ids"); p.bind(a_ids, 0); p.bind(ids, 1, true); p.parameter(std::array<uint32_t, 2>{std::min(blocks, budget), blocks}, 2);
    p.groups(MTLSizeMake(1, 1, 1));
}
void MetalGraph::sparse_attention(GpuView query, GpuView keys, GpuView values, GpuView ids, GpuView output, uint32_t prefix, QsaShape s) {
    auto& p = *impl_; p.active(); validate_qsa_shape(s);
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    if (p.qsa_stage_v8.enabled) {
        const auto bytes = qsa_stage_v8_bytes(prefix, s);
        sparse_attention_stage_v8(query, keys, values, ids, output, slice(p.qsa_stage_v8_scratch, 0, bytes.scores), prefix, s); return;
    }
#endif
    require(prefix && prefix <= s.capacity && p.floats(query) == s.query_heads * s.head_width && p.floats(output, true) == p.floats(query) &&
        keys.bytes >= uint64_t(prefix) * s.kv_heads * s.head_width * 2 && values.bytes >= uint64_t(prefix) * s.kv_heads * s.head_width * 2 &&
        p.floats(ids) == qsa_block_budget(s), "sparse attention GPU buffers/prefix mismatch");
    validate_disjoint(std::array{query, keys, values, ids}, std::array{output});
    p.select("qsa_attention"); p.bind(query, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(ids, 3); p.bind(output, 4, true);
    p.parameter(std::array<uint32_t, 4>{s.query_heads, s.kv_heads, s.head_width, s.compression}, 5);
    p.parameter(std::array<uint32_t, 4>{prefix, qsa_block_budget(s), 0, 0}, 6);
    p.groups(MTLSizeMake((s.query_heads + 3) / 4, 1, 1));
}

#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
void MetalGraph::prepare_qsa_score_stage() {
    auto& p = *impl_; p.idle();
    static_cast<void>(p.pipeline("qsa_score_stage"));
    static_cast<void>(p.pipeline("qsa_score_accumulate"));
    p.qsa_score_stage_prepared = true;
}
void MetalGraph::sparse_attention_score_stage(GpuView query, GpuView keys, GpuView values,
    GpuView ids, GpuView output, GpuView scores, uint32_t prefix, QsaShape s) {
    static_cast<void>(qsa_score_stage_bytes(prefix, s)); // Original fixed guard, unchanged.
    sparse_attention_stage_v8(query, keys, values, ids, output, scores, prefix, s);
}
void MetalGraph::sparse_attention_stage_v8(GpuView query, GpuView keys, GpuView values,
    GpuView ids, GpuView output, GpuView scores, uint32_t prefix, QsaShape s) {
    auto& p = *impl_; p.active(); const auto b = qsa_stage_v8_bytes(prefix, s);
    require(!p.recording && !p.compatible_pipelines, "QSA score staging requires an ordinary direct batch");
    require(p.qsa_score_stage_prepared, "both QSA score-stage pipelines must be prepared idle");
    const uint64_t row_bytes = uint64_t(s.kv_heads) * s.head_width * 2;
    require(p.floats(query) == s.query_heads * s.head_width && p.floats(output, true) == p.floats(query) &&
        keys.bytes == values.bytes && keys.bytes >= uint64_t(prefix) * row_bytes &&
        keys.bytes <= uint64_t(s.capacity) * row_bytes && keys.bytes % row_bytes == 0 &&
        p.floats(ids) == qsa_block_budget(s) && uint64_t(p.floats(scores, true)) * 4 == b.scores,
        "QSA score-stage buffer geometry mismatch");
    const std::array reads{query, keys, values, ids}; const std::array writes{scores, output};
    validate_disjoint(reads, writes);
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    GpuView factors{};
    if (p.qsa_channel.enabled) {
        require(p.qsa_channel.prepared && p.qsa_stage_v8.prepared && p.qsa_stage_v8.enabled &&
            p.hc_exact == HcExactMode::combined && p.fixed_route_profile_allowed() &&
            !p.hc_q8.splits && p.small_gemv_splits == 16 && p.block_gemv_formats == 3 &&
            !p.wide_q8_exact && !p.wide_q8_many && p.captures.empty(), "chronological QSA fixed policy changed");
        factors = slice(p.qsa_channel_factors, 0, qsa_channel_factor_bytes(prefix, s));
        validate_disjoint(reads, std::array{scores, factors, output});
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        const bool qsa_route_shadow =
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
            p.qsa_route_phase == 2;
#else
            false;
#endif
        if (p.qsa_sparse_enabled || qsa_route_shadow) {
            require(p.qsa_sparse_configured, "sparse QSA requires startup configuration");
            static_cast<void>(p.pipeline("qsa_sparse_attention", uint32_t(-1), Impl::qsa_sparse_decode_threads));
            if (!qsa_route_shadow) require(p.qsa_sparse_decode_pending < UINT64_MAX - p.qsa_sparse_decode_completed, "sparse QSA decode call counter overflow");
            require(!p.profiling_active() || p.dispatches < p.profile_capacity, "sparse QSA requires one available profiling slot");
            const std::array<uint32_t, 4> shape{s.query_heads, s.kv_heads, s.head_width, s.compression};
            const std::array<uint32_t, 4> plan{prefix, qsa_block_budget(s), 0, 0};
            p.select("qsa_sparse_attention", uint32_t(-1), {prefix, b.attended_tokens, s.query_heads, s.head_width}, Impl::qsa_sparse_decode_threads);
            p.bind(query, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(ids, 3); p.bind(output, 4, true);
            p.parameter(shape, 5); p.parameter(plan, 6);
            p.dispatch(true, MTLSizeMake(s.query_heads, 1, 1), MTLSizeMake(Impl::qsa_sparse_decode_threads, 1, 1));
            if (!qsa_route_shadow) ++p.qsa_sparse_decode_pending;
            return;
        }
#endif
        // Capability/overflow/alias checks all precede the first score dispatch.
        static_cast<void>(p.pipeline("qsa_chronological_factors", uint32_t(-1), qsa_channel_threads));
        static_cast<void>(p.pipeline(
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
            p.kernel_qsa_f32 ? "qsa_channels_search_f32" :
#endif
            "qsa_chronological_channels", uint32_t(-1), qsa_channel_threads));
        require(p.qsa_channel_pending < UINT64_MAX - p.qsa_channel_completed, "chronological QSA call counter overflow");
    }
    require(p.qsa_stage_v8_pending < UINT64_MAX - p.qsa_stage_v8_completed, "staged QSA call counter overflow");
#endif
    require(!p.profiling_active() || (p.dispatches <= p.profile_capacity && p.profile_capacity - p.dispatches >= 2),
        "QSA score staging requires two available profiling slots");
    const std::array<uint32_t, 4> shape{s.query_heads, s.kv_heads, s.head_width, s.compression};
    const std::array<uint32_t, 4> plan{prefix, qsa_block_budget(s), 0, 0};
    p.select("qsa_score_stage", uint32_t(-1), {prefix, b.attended_tokens, s.query_heads, s.head_width});
    p.bind(query, 0); p.bind(keys, 1); p.bind(ids, 2); p.bind(scores, 3, true); p.parameter(shape, 5); p.parameter(plan, 6);
    p.groups(MTLSizeMake((b.attended_tokens + 3) / 4, s.query_heads, 1));
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    if (p.qsa_channel.enabled) {
        p.select("qsa_chronological_factors", uint32_t(-1), {prefix, b.attended_tokens, s.query_heads, s.head_width}, qsa_channel_threads);
        p.bind(scores, 0); p.bind(factors, 1, true); p.parameter(shape, 5); p.parameter(plan, 6);
        p.dispatch(true, MTLSizeMake(s.query_heads, 1, 1), MTLSizeMake(qsa_channel_threads, 1, 1));
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
        if (p.kernel_search_sink && (prefix == 4096 || prefix == 4097) && (p.operator_tag.layer == 3 || p.operator_tag.layer == 7)) {
            KernelSearchCaptureRecord record; record.kind = 2; record.dimensions = {s.query_heads, s.kv_heads, s.head_width, s.compression, prefix, qsa_block_budget(s), b.attended_tokens, 0};
            record.parameter_bytes = 32; std::memcpy(record.parameters.data(), shape.data(), 16); std::memcpy(record.parameters.data() + 4, plan.data(), 16);
            record.views = {factors, slice(values, 0, uint64_t(prefix) * s.kv_heads * s.head_width * 2), slice(ids, 0, uint64_t(qsa_block_budget(s)) * 4), {}, {}, {}};
            kernel_search_capture(record, 1u << (4 + uint32_t(p.operator_tag.layer == 7) * 2 + uint32_t(prefix == 4097)));
        }
#endif
        p.select(
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
            p.kernel_qsa_f32 ? "qsa_channels_search_f32" :
#endif
            "qsa_chronological_channels", uint32_t(-1), {prefix, b.attended_tokens, s.query_heads, s.head_width}, qsa_channel_threads);
        p.bind(factors, 0); p.bind(values, 1); p.bind(ids, 2); p.bind(output, 3, true); p.parameter(shape, 5); p.parameter(plan, 6);
        p.dispatch(true, MTLSizeMake(s.query_heads, 1, 1), MTLSizeMake(qsa_channel_threads, 1, 1));
        ++p.qsa_stage_v8_pending; ++p.qsa_channel_pending; return;
    }
#endif
    p.select("qsa_score_accumulate", uint32_t(-1), {prefix, b.attended_tokens, s.query_heads, s.head_width});
    p.bind(scores, 0); p.bind(values, 1); p.bind(ids, 2); p.bind(output, 3, true); p.parameter(shape, 5); p.parameter(plan, 6);
    p.groups(MTLSizeMake((s.query_heads + 3) / 4, 1, 1)); ++p.qsa_stage_v8_pending;
}
#endif

#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
void MetalGraph::prepare_qsa_verify_rows() {
    auto& p = *impl_; p.idle();
    require(!p.qsa_rows_locked, "QSA row pipelines must be prepared before the first batch attempt");
    require(p.qsa_channel.prepared && p.qsa_channel.enabled && p.qsa_stage_v8.prepared && p.qsa_stage_v8.enabled &&
        p.hc_exact == HcExactMode::combined && p.profile_mode == ProfileMode::disabled && !p.hc_q8.splits &&
        p.small_gemv_splits == 16 && p.block_gemv_formats == 3 && !p.wide_q8_exact && !p.wide_q8_many && p.captures.empty(),
        "QSA row preparation requires the corrected fixed chronological-channel policy");
    static_cast<void>(p.pipeline("qsa_verify_rows_score"));
    static_cast<void>(p.pipeline("qsa_verify_rows_factors", uint32_t(-1), 256));
    static_cast<void>(p.pipeline(
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
            p.kernel_qsa_f32 ? "qsa_rows_channels_search_f32" :
#endif
            "qsa_verify_rows_channels", uint32_t(-1), 256));
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    if (p.qsa_sparse_configured)
        static_cast<void>(p.pipeline("qsa_sparse_attention_rows", uint32_t(-1), Impl::qsa_sparse_verify_threads));
#endif
    p.qsa_rows_prepared = true; // All three PSOs or no publication.
}
QsaVerifyRowsRoutes MetalGraph::qsa_verify_rows_routes() const { impl_->idle(); return impl_->qsa_rows_completed; }
void MetalGraph::sparse_attention_rows_chronological(GpuView queries, GpuView keys, GpuView values,
    GpuView ids, GpuView outputs, GpuView scores, GpuView factors, uint32_t base, uint32_t rows, QsaShape shape) {
    auto& p = *impl_; p.active(); const auto plan = qsa_verify_rows_plan(base, rows, shape);
    require(p.qsa_rows_prepared && p.qsa_channel.prepared && p.qsa_channel.enabled && !p.recording &&
        !p.compatible_pipelines && p.fixed_route_profile_allowed() && p.captures.empty(),
        "QSA row core requires its prepared fixed direct unprofiled policy");
    const uint64_t kv_row = uint64_t(shape.kv_heads) * shape.head_width * 2;
    require(queries.bytes == 8 * plan.vector_row_bytes && outputs.bytes == queries.bytes &&
        ids.bytes == 8 * plan.id_row_bytes && scores.bytes == maximum_qsa_verify_rows_scores &&
        factors.bytes == maximum_qsa_verify_rows_factors && factors.offset % 8 == 0 && keys.bytes == values.bytes &&
        keys.bytes >= uint64_t(base + rows) * kv_row && keys.bytes <= uint64_t(shape.capacity) * kv_row && keys.bytes % kv_row == 0,
        "QSA row core fixed workspace or causal cache geometry mismatch");
    validate_disjoint(std::array{queries, keys, values, ids}, std::array{outputs, scores, factors});
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    const bool qsa_route_shadow =
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        p.qsa_route_phase == 2;
#else
        false;
#endif
    if (p.qsa_sparse_enabled || qsa_route_shadow) {
        require(p.qsa_sparse_configured, "sparse QSA rows require startup configuration");
        const QsaVerifyRowsRoutes delta{1, rows, 0, 0, 0};
        if (!qsa_route_shadow) {
            for (size_t i = 0; i < delta.size(); ++i) require(p.qsa_rows_completed[i] <= UINT64_MAX - delta[i] &&
                p.qsa_rows_pending[i] <= UINT64_MAX - delta[i] - p.qsa_rows_completed[i], "QSA row route counter overflow");
            require(p.qsa_sparse_verify_pending < UINT64_MAX - p.qsa_sparse_verify_completed, "sparse QSA verify call counter overflow");
        }
        require(!p.profiling_active() || p.dispatches < p.profile_capacity, "sparse QSA rows require one available profiling slot");
        static_cast<void>(p.pipeline("qsa_sparse_attention_rows", uint32_t(-1), Impl::qsa_sparse_verify_threads));
        const std::array<uint32_t, 4> geometry{shape.query_heads, shape.kv_heads, shape.head_width, shape.compression};
        const std::array<uint32_t, 4> tile{base, qsa_block_budget(shape), rows, qsa_verify_rows_stride};
        p.select("qsa_sparse_attention_rows", uint32_t(-1), {base, rows, plan.maximum_attended, shape.head_width}, Impl::qsa_sparse_verify_threads);
        p.bind(queries, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(ids, 3); p.bind(outputs, 4, true);
        p.parameter(geometry, 5); p.parameter(tile, 6);
        p.dispatch(true, MTLSizeMake(shape.query_heads, rows, 1), MTLSizeMake(Impl::qsa_sparse_verify_threads, 1, 1));
        if (!qsa_route_shadow) {
            for (size_t i = 0; i < delta.size(); ++i) p.qsa_rows_pending[i] += delta[i];
            ++p.qsa_sparse_verify_pending;
        }
        return;
    }
#endif
    const QsaVerifyRowsRoutes delta{1, rows, 1, 1, 1};
    for (size_t i = 0; i < delta.size(); ++i) require(p.qsa_rows_completed[i] <= UINT64_MAX - delta[i] &&
        p.qsa_rows_pending[i] <= UINT64_MAX - delta[i] - p.qsa_rows_completed[i], "QSA row route counter overflow");
    const std::array<uint32_t, 4> geometry{shape.query_heads, shape.kv_heads, shape.head_width, shape.compression};
    const std::array<uint32_t, 4> tile{base, qsa_block_budget(shape), rows, qsa_verify_rows_stride};
    p.select("qsa_verify_rows_score", uint32_t(-1), {base, rows, plan.maximum_attended, shape.head_width});
    p.bind(queries, 0); p.bind(keys, 1); p.bind(ids, 2); p.bind(scores, 3, true); p.parameter(geometry, 5); p.parameter(tile, 6);
    p.groups(MTLSizeMake((plan.maximum_attended + 3) / 4, shape.query_heads, rows));
    p.select("qsa_verify_rows_factors", uint32_t(-1), {base, rows, plan.maximum_attended, shape.head_width}, 256);
    p.bind(scores, 0); p.bind(factors, 1, true); p.parameter(geometry, 5); p.parameter(tile, 6);
    p.dispatch(true, MTLSizeMake(shape.query_heads, rows, 1), MTLSizeMake(256, 1, 1));
    p.select(
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
            p.kernel_qsa_f32 ? "qsa_rows_channels_search_f32" :
#endif
            "qsa_verify_rows_channels", uint32_t(-1), {base, rows, plan.maximum_attended, shape.head_width}, 256);
    p.bind(factors, 0); p.bind(values, 1); p.bind(ids, 2); p.bind(outputs, 3, true); p.parameter(geometry, 5); p.parameter(tile, 6);
    p.dispatch(true, MTLSizeMake(shape.query_heads, rows, 1), MTLSizeMake(256, 1, 1));
    for (size_t i = 0; i < delta.size(); ++i) p.qsa_rows_pending[i] += delta[i];
    // Deliberately no Stage/channel pending increments: legacy counters are legacy physical calls.
}
#endif

#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
void MetalGraph::sparse_attention_bn(GpuView query, GpuView keys, GpuView values,
    GpuView ids, GpuView output, uint32_t prefix, QsaShape s, uint32_t bn) {
    auto& p = *impl_; p.active();
    const auto b = qsa_stage_v8_bytes(prefix, s);
    require(p.qsa_sparse_variants_prepared && !p.recording && !p.compatible_pipelines && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "sparse QSA BN variant requires prepared direct unprofiled startup");
    const char* kernel = bn == 8 ? "qsa_sparse_attention_bn8" : bn == 16 ? "qsa_sparse_attention_bn16" : bn == 32 ? "qsa_sparse_attention_bn32" : nullptr;
    require(kernel != nullptr, "unsupported sparse QSA BN variant");
    const uint32_t threads = 32u * bn;
    const uint64_t row_bytes = uint64_t(s.kv_heads) * s.head_width * 2;
    require(p.floats(query) == s.query_heads * s.head_width && p.floats(output, true) == p.floats(query) &&
        keys.bytes == values.bytes && keys.bytes >= uint64_t(prefix) * row_bytes &&
        keys.bytes <= uint64_t(s.capacity) * row_bytes && keys.bytes % row_bytes == 0 &&
        p.floats(ids) == qsa_block_budget(s),
        "sparse QSA BN buffer geometry mismatch");
    validate_disjoint(std::array{query, keys, values, ids}, std::array{output});
    const std::array<uint32_t, 4> shape{s.query_heads, s.kv_heads, s.head_width, s.compression};
    const std::array<uint32_t, 4> plan{prefix, qsa_block_budget(s), 0, 0};
    p.select(kernel, uint32_t(-1), {prefix, b.attended_tokens, s.query_heads, s.head_width}, threads);
    p.bind(query, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(ids, 3); p.bind(output, 4, true);
    p.parameter(shape, 5); p.parameter(plan, 6);
    p.dispatch(true, MTLSizeMake(s.query_heads, 1, 1), MTLSizeMake(threads, 1, 1));
}
void MetalGraph::sparse_attention_rows_bn(GpuView queries, GpuView keys, GpuView values,
    GpuView ids, GpuView outputs, uint32_t base, uint32_t rows, QsaShape shape, uint32_t bn) {
    auto& p = *impl_; p.active();
    const auto plan = qsa_verify_rows_plan(base, rows, shape);
    require(p.qsa_sparse_variants_prepared && !p.recording && !p.compatible_pipelines && p.profile_mode == ProfileMode::disabled && p.captures.empty(),
        "sparse QSA row BN variant requires prepared direct unprofiled startup");
    const char* kernel = bn == 8 ? "qsa_sparse_attention_rows_bn8" : bn == 16 ? "qsa_sparse_attention_rows_bn16" : bn == 32 ? "qsa_sparse_attention_rows_bn32" : nullptr;
    require(kernel != nullptr, "unsupported sparse QSA row BN variant");
    const uint32_t threads = 32u * bn;
    const uint64_t kv_row = uint64_t(shape.kv_heads) * shape.head_width * 2;
    require(queries.bytes == 8 * plan.vector_row_bytes && outputs.bytes == queries.bytes &&
        ids.bytes == 8 * plan.id_row_bytes && keys.bytes == values.bytes &&
        keys.bytes >= uint64_t(base + rows) * kv_row && keys.bytes <= uint64_t(shape.capacity) * kv_row && keys.bytes % kv_row == 0,
        "sparse QSA row BN fixed workspace or causal cache geometry mismatch");
    validate_disjoint(std::array{queries, keys, values, ids}, std::array{outputs});
    const std::array<uint32_t, 4> geometry{shape.query_heads, shape.kv_heads, shape.head_width, shape.compression};
    const std::array<uint32_t, 4> tile{base, qsa_block_budget(shape), rows, qsa_verify_rows_stride};
    p.select(kernel, uint32_t(-1), {base, rows, plan.maximum_attended, shape.head_width}, threads);
    p.bind(queries, 0); p.bind(keys, 1); p.bind(values, 2); p.bind(ids, 3); p.bind(outputs, 4, true);
    p.parameter(geometry, 5); p.parameter(tile, 6);
    p.dispatch(true, MTLSizeMake(shape.query_heads, rows, 1), MTLSizeMake(threads, 1, 1));
}
#endif

#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
void MetalGraph::prepare_dflash2() {
    auto& p = *impl_; p.idle();
    require([p.device supportsFamily:MTLGPUFamilyMetal4], "DFlash2 requires Metal 4 support");
    for (const auto& [name, threads] : std::array<std::pair<const char*, uint32_t>, 11>{{
        {"dflash2_stream_mean", 256},
        {"dflash2_context_kv_write", 128},
        {"dflash2_conv_f32", 256},
        {"dflash2_conv_finish_rms_f32", 128},
        {"dflash2_qkv_prepare", 128},
        {"dflash2_attention_split", 256},
        {"dflash2_attention_reduce", 256},
        {"dflash2_top16_sharded", 256},
        {"dflash2_dense_bf16_m8", 256},
        {"dflash2_dense_bf16_r4_m8", 256},
        {"dflash2_gate_up_swiglu_bf16_m8", 256},
    }}) {
        const auto pipe = p.pipeline(name, uint32_t(-1), threads);
        require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "DFlash2 threadgroup memory exceeds device capability");
    }
    for (const uint32_t layout : {0u, 1u}) {
#ifndef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (layout != 0u) continue;
#endif
        for (const char* q8_name : {"dflash2_dense_q8_m8", "dflash2_dense_q8_r4_m8", "dflash2_gate_up_swiglu_q8_m8"}) {
            const auto pipe = p.pipeline(q8_name, 8u, 256, false, layout);
            require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
                "DFlash2 staged Q8 threadgroup memory exceeds device capability");
        }
    }
    const auto edges_pipe = p.pipeline("dflash2_select_edges", uint32_t(-1), 256);
    require(edges_pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
        "DFlash2 selector threadgroup memory exceeds device capability");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    for (const uint32_t layout : {0u, 1u}) {
#ifndef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (layout != 0u) continue;
#endif
        const auto m4 = p.pipeline("nax_verify_dense_q8_m4", 8, 128, false, layout);
        const auto m8 = p.pipeline("nax_verify_dense_q8_m8", 8, 128, false, layout);
        const auto sw4 = p.pipeline("nax_verify_dense_q8_swiglu_m4", 8, 128, false, layout);
        const auto sw8 = p.pipeline("nax_verify_dense_q8_swiglu_m8", 8, 128, false, layout);
        const auto nax_m8 = p.pipeline("nax_m8_dense_q8", 8, 64, false, layout);
        const auto nax_m8_qmv = p.pipeline("nax_m8_qmv_q8", 8, 128, false, layout);
        const auto nax_m8_sw = p.pipeline("nax_m8_dense_q8_swiglu", 8, 128, false, layout);
        require(m4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            sw4.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            sw8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_m8_qmv.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
            nax_m8_sw.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "DFlash2 dense Q8 threadgroup memory exceeds device capability");
    }
    constexpr uint64_t need_partials = uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes;
    constexpr uint64_t need_counters = uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotCountersBytes + 64ull;
    if (p.ledger.budget - p.ledger.live_bytes >= need_partials + need_counters + (4ull << 20)) {
        if (p.verify_nax_split_partials.bytes < need_partials) {
            if (p.verify_nax_split_partials.bytes) release(p.verify_nax_split_partials);
            p.verify_nax_split_partials = zeros(need_partials, "verify_nax_split_partials");
        }
        if (p.verify_nax_split_counters.bytes < need_counters) {
            if (p.verify_nax_split_counters.bytes) release(p.verify_nax_split_counters);
            p.verify_nax_split_counters = zeros(need_counters, "verify_nax_split_counters");
        }
    } else if (p.verify_nax_split_counters.bytes < 64ull) {
        if (p.verify_nax_split_counters.bytes) release(p.verify_nax_split_counters);
        p.verify_nax_split_counters = zeros(64ull, "verify_nax_split_counters_idle");
    }
    for (const char* kname : {"nax_verify_dense_bf16_r1_m4", "nax_verify_dense_bf16_r1_m8",
                              "nax_verify_dense_bf16_m4", "nax_verify_dense_bf16_m8"}) {
        const auto pipe = p.pipeline(kname, uint32_t(-1), 256);
        require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "DFlash2 verify dense BF16 threadgroup memory exceeds device capability");
    }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    prepare_dense_mm_q8(PrefillWeightLayout::q8_lanes_v1);
    if (p.prefill_dense_nax) {
        constexpr uint64_t min_dflash2_bf16 = 52'428'800ull; // 2048 * 12800 * sizeof(bfloat)
        if (p.nax_bf16_scratch.bytes < min_dflash2_bf16) {
            if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
            p.nax_bf16_scratch = zeros(min_dflash2_bf16, "nax_bf16_scratch");
        }
    }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_CHUNK
    prepare_gdn_chunk();
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    const auto dflash_t64_m8 = p.pipeline("nax_m8_dense_q8_t64", uint32_t(-1), 64);
    const auto dflash_t64_sw = p.pipeline("nax_m8_dense_q8_swiglu_t64", uint32_t(-1), 64);
    require(dflash_t64_m8.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength &&
        dflash_t64_sw.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
        "DFlash2 Q8_tile64 threadgroup memory exceeds device capability");
    p.q8_tile64_ready = true;
    p.ensure_q8_tile64_all();
#endif
    p.dflash2_prepared = true;
}

void MetalGraph::set_dflash2_scope(bool active) {
    auto& p = *impl_; p.thread_check();
    p.dflash2_scope = active;
    p.nax_dense_bf16_valid = false;
}

void MetalGraph::dflash2_stream_mean(GpuView streams, GpuView captured, uint32_t rows, uint32_t width,
    uint32_t slot_index, uint32_t total_slots, uint64_t input_stride_bytes, uint64_t output_stride_bytes) {
    struct DFlash2StreamMeanArgs {
        uint32_t rows, width, slot_index, total_slots;
        uint64_t input_stride_bytes, output_stride_bytes;
    };
    static_assert(sizeof(DFlash2StreamMeanArgs) == 32, "DFlash2 stream mean ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 stream mean requires prepared direct startup");
    require(rows >= 1 && rows <= 2048 && width >= 64 && width % 4 == 0 && slot_index < total_slots && total_slots <= 16 &&
        input_stride_bytes >= uint64_t(width) * 16ull && input_stride_bytes % 4 == 0 &&
        output_stride_bytes >= uint64_t(total_slots) * width * 4ull && output_stride_bytes % 4 == 0,
        "DFlash2 stream mean geometry mismatch");
    const uint64_t in_bytes = checked_add(checked_mul(uint64_t(rows - 1), input_stride_bytes), uint64_t(width) * 16ull);
    const uint64_t out_bytes = checked_add(checked_mul(uint64_t(rows - 1), output_stride_bytes), uint64_t(total_slots) * width * 4ull);
    static_cast<void>(p.allocation(streams)); static_cast<void>(p.allocation(captured, true));
    require(streams.bytes >= in_bytes && captured.bytes >= out_bytes && streams.offset % 4 == 0 && captured.offset % 4 == 0,
        "DFlash2 stream mean buffer span or alignment mismatch");
    validate_disjoint(std::array{streams}, std::array{captured});
    const DFlash2StreamMeanArgs args{rows, width, slot_index, total_slots, input_stride_bytes, output_stride_bytes};
    const uint32_t quads = rows * (width / 4u);
    p.select("dflash2_stream_mean", uint32_t(-1), {rows, width, slot_index, total_slots}, 256);
    p.bind(streams, 0); p.bind(captured, 1, true); p.parameter(args, 2);
    p.dispatch(true, MTLSizeMake((quads + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1));
}

void MetalGraph::dflash2_context_kv_write(GpuView context_kv, GpuView qk_norm_freq, GpuView ring_k, GpuView ring_v,
    uint32_t rows, uint32_t start_pos, uint32_t window, float eps) {
    struct DFlash2ContextKvArgs {
        uint32_t rows, start_pos, window;
        float eps;
    };
    static_assert(sizeof(DFlash2ContextKvArgs) == 16, "DFlash2 context KV ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 context KV requires prepared direct startup");
    constexpr uint64_t ring_bytes = 8ull * 2048ull * 128ull * 2ull;
    require(rows >= 1 && rows <= 2048 && window == 2048 && eps > 0.0f &&
        context_kv.bytes >= uint64_t(rows) * 2048ull * 4ull && qk_norm_freq.bytes == 320ull * 4ull &&
        ring_k.bytes == ring_bytes && ring_v.bytes == ring_bytes,
        "DFlash2 context KV geometry mismatch");
    validate_disjoint(std::array{context_kv, qk_norm_freq}, std::array{ring_k, ring_v});
    const DFlash2ContextKvArgs args{rows, start_pos, window, eps};
    p.select("dflash2_context_kv_write", uint32_t(-1), {rows, start_pos, window, 8}, 128);
    p.bind(context_kv, 0); p.bind(qk_norm_freq, 1); p.bind(ring_k, 2, true); p.bind(ring_v, 3, true); p.parameter(args, 4);
    p.dispatch(true, MTLSizeMake(uint64_t(rows) * 8ull, 1, 1), MTLSizeMake(128, 1, 1));
}

void MetalGraph::dflash2_conv_f32(GpuView input, GpuView dynamic, GpuView base, GpuView residual, GpuView output,
    uint32_t hidden, bool finish, uint32_t rows, uint32_t group_size) {
    struct DFlash2ConvArgs {
        uint32_t hidden, finish, rows, group_size;
    };
    static_assert(sizeof(DFlash2ConvArgs) == 16, "DFlash2 conv ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 conv requires prepared direct startup");
    require(rows == 8 && group_size == 16 && hidden >= 64 && hidden % group_size == 0, "DFlash2 conv shape mismatch");
    const uint64_t hidden_bytes = uint64_t(rows) * hidden * 4ull;
    const uint64_t dyn_bytes = uint64_t(rows) * 4ull * (hidden / group_size) * 4ull;
    require(input.bytes >= hidden_bytes && output.bytes >= hidden_bytes && residual.bytes >= hidden_bytes &&
        dynamic.bytes >= dyn_bytes && base.bytes == 4ull * hidden * 4ull,
        "DFlash2 conv buffer span mismatch");
    validate_disjoint(std::array{input, dynamic, base, residual}, std::array{output});
    const DFlash2ConvArgs args{hidden, finish ? 1u : 0u, rows, group_size};
    const uint32_t total = rows * hidden;
    p.select("dflash2_conv_f32", uint32_t(-1), {rows, hidden, finish ? 1u : 0u, group_size}, 256);
    p.bind(input, 0); p.bind(dynamic, 1); p.bind(base, 2); p.bind(residual, 3); p.bind(output, 4, true); p.parameter(args, 5);
    p.dispatch(true, MTLSizeMake((total + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1));
}

void MetalGraph::dflash2_qkv_prepare(GpuView proposal_qkv, GpuView qk_norm_freq, GpuView queries, GpuView query_keys,
    GpuView query_values, uint32_t cache_len, uint32_t rows, float eps) {
    struct DFlash2QkvArgs {
        uint32_t cache_len, rows;
        float eps, reserved;
    };
    static_assert(sizeof(DFlash2QkvArgs) == 16, "DFlash2 QKV ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 QKV prepare requires prepared direct startup");
    require(rows == 8 && eps > 0.0f && proposal_qkv.bytes >= 8ull * 6144ull * 4ull &&
        qk_norm_freq.bytes == 320ull * 4ull && queries.bytes == 32ull * 8ull * 128ull * 2ull &&
        query_keys.bytes == 8ull * 8ull * 128ull * 2ull && query_values.bytes == 8ull * 128ull * 8ull * 2ull,
        "DFlash2 QKV prepare geometry mismatch");
    validate_disjoint(std::array{proposal_qkv, qk_norm_freq}, std::array{queries, query_keys, query_values});
    const DFlash2QkvArgs args{cache_len, rows, eps, 0.0f};
    const uint32_t groups = std::max<uint32_t>((8u * 1024u + 127u) / 128u, 8u * (32u + 8u));
    p.select("dflash2_qkv_prepare", uint32_t(-1), {rows, cache_len, 32, 8}, 128);
    p.bind(proposal_qkv, 0); p.bind(qk_norm_freq, 1); p.bind(queries, 2, true); p.bind(query_keys, 3, true); p.bind(query_values, 4, true);
    p.parameter(args, 5);
    p.dispatch(true, MTLSizeMake(groups, 1, 1), MTLSizeMake(128, 1, 1));
}

void MetalGraph::dflash2_attention(GpuView queries, GpuView ring_k, GpuView ring_v, GpuView query_keys,
    GpuView query_values, GpuView partials, GpuView output, uint32_t cache_len, uint32_t window) {
    struct DFlash2AttnArgs {
        uint32_t cache_len, window, splits, reserved;
    };
    static_assert(sizeof(DFlash2AttnArgs) == 16, "DFlash2 attention ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 attention requires prepared direct startup");
    constexpr uint64_t ring_bytes = 8ull * 2048ull * 128ull * 2ull;
    constexpr uint64_t partial_bytes = 8ull * 16ull * (32ull * 128ull + 64ull) * 4ull;
    require(window == 2048 && queries.bytes == 32ull * 8ull * 128ull * 2ull &&
        ring_k.bytes == ring_bytes && ring_v.bytes == ring_bytes &&
        query_keys.bytes == 8ull * 8ull * 128ull * 2ull && query_values.bytes == 8ull * 128ull * 8ull * 2ull &&
        partials.bytes >= partial_bytes && output.bytes >= 8ull * 4096ull * 4ull,
        "DFlash2 attention geometry mismatch");
    validate_disjoint(std::array{queries, ring_k, ring_v, query_keys, query_values}, std::array{partials, output});
    const DFlash2AttnArgs args{cache_len, window, 16u, 0u};
    p.select("dflash2_attention_split", uint32_t(-1), {cache_len, window, 8, 16}, 256);
    p.bind(queries, 0); p.bind(ring_k, 1); p.bind(ring_v, 2); p.bind(query_keys, 3); p.bind(query_values, 4);
    p.bind(partials, 5, true); p.parameter(args, 6);
    p.dispatch(true, MTLSizeMake(8, 1, 16), MTLSizeMake(256, 1, 1));

    p.select("dflash2_attention_reduce", uint32_t(-1), {cache_len, window, 8, 32}, 256);
    p.bind(partials, 0); p.bind(output, 1, true); p.parameter(args, 2);
    p.dispatch(true, MTLSizeMake(8, 1, 1), MTLSizeMake(256, 1, 1));
}

void MetalGraph::dflash2_select(GpuView logits, GpuView partial_ids, GpuView partial_values, GpuView selector_out,
    GpuView selector_hidden, GpuView predecessor_codebook, GpuView successor_codebook,
    uint32_t positions, uint32_t vocabulary, uint32_t anchor, uint32_t codebook_vocabulary) {
    struct DFlash2SelectArgs {
        uint32_t positions, vocabulary, anchor, reserved;
    };
    static_assert(sizeof(DFlash2SelectArgs) == 16, "DFlash2 selector ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 selector requires prepared direct startup");
    const uint32_t cb_vocab = codebook_vocabulary ? codebook_vocabulary : vocabulary;
    require(positions >= 1 && positions <= 7 && vocabulary >= 16 && cb_vocab >= vocabulary && anchor < cb_vocab,
        "DFlash2 selector bounds mismatch");
    const uint64_t cb_bytes = uint64_t(cb_vocab) * 256ull * 2ull;
    require(logits.bytes >= uint64_t(positions) * vocabulary * 4ull &&
        partial_ids.bytes >= uint64_t(positions) * 8ull * 16ull * 4ull &&
        partial_values.bytes >= uint64_t(positions) * 8ull * 16ull * 4ull &&
        selector_out.bytes >= 7ull * 16ull * 8ull + 7ull * 256ull * 4ull &&
        selector_hidden.bytes >= uint64_t(positions) * 256ull * 4ull &&
        predecessor_codebook.bytes == cb_bytes && successor_codebook.bytes == cb_bytes,
        "DFlash2 selector buffer span mismatch");
    validate_disjoint(std::array{logits}, std::array{partial_ids, partial_values});
    validate_disjoint(std::array{partial_ids, partial_values, selector_hidden, predecessor_codebook, successor_codebook}, std::array{selector_out});
    const DFlash2SelectArgs args{positions, vocabulary, anchor, cb_vocab};
    p.select("dflash2_top16_sharded", uint32_t(-1), {positions, vocabulary, 8, 16}, 256);
    p.bind(logits, 0); p.bind(partial_ids, 1, true); p.bind(partial_values, 2, true); p.parameter(args, 3);
    p.dispatch(true, MTLSizeMake(uint64_t(positions) * 8ull, 1, 1), MTLSizeMake(256, 1, 1));

    p.select("dflash2_select_edges", uint32_t(-1), {positions, vocabulary, anchor, 256}, 256);
    p.bind(partial_ids, 0); p.bind(partial_values, 1); p.bind(selector_out, 2, true);
    p.bind(selector_hidden, 3); p.bind(predecessor_codebook, 4); p.bind(successor_codebook, 5); p.parameter(args, 6);
    p.dispatch(true, MTLSizeMake(positions, 1, 1), MTLSizeMake(256, 1, 1));
}

void MetalGraph::dflash2_dense_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu) {
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && p.dflash2_scope && !p.recording && !p.compatible_pipelines,
        "DFlash2 dense BF16 requires prepared direct DFlash2 scope");
    require(input.rows >= 1 && input.rows <= 2048 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "DFlash2 dense BF16 shape bounds mismatch");
    const uint32_t row_bytes = input.columns * 2u;
    const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
    const uint64_t min_in_stride = uint64_t(input.columns) * sizeof(float);
    const uint64_t min_out_stride = uint64_t(output.columns) * sizeof(float);
    require(matrix.bytes == matrix_bytes && matrix.offset % 2 == 0 &&
        input.stride_bytes >= min_in_stride && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= min_out_stride && output.stride_bytes % 4 == 0 &&
        input.view.offset % 4 == 0 && output.view.offset % 4 == 0 &&
        input.view.bytes >= uint64_t(input.rows - 1) * input.stride_bytes + min_in_stride &&
        output.view.bytes >= uint64_t(output.rows - 1) * output.stride_bytes + min_out_stride,
        "DFlash2 dense BF16 buffer span or alignment mismatch");
    separate(output.view, {matrix, input.view});
    const GroupedMmArgs args{
        input.columns, output.columns, row_bytes, 1u, 1u, input.rows, 1u, swiglu ? 1u : 0u,
        matrix_bytes, input.stride_bytes, output.stride_bytes
    };
    const bool r4 = output.columns > 2560u;
    const uint32_t rows_per_tg = r4 ? 32u : 8u;
    const char* kname = r4 ? "dflash2_dense_bf16_r4_m8" : "dflash2_dense_bf16_m8";
    p.select(kname, uint32_t(-1), {input.columns, output.columns, input.rows, swiglu ? 1u : 0u}, 256);
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.dispatch(true, MTLSizeMake((output.columns + rows_per_tg - 1u) / rows_per_tg, (input.rows + 7u) / 8u, 1), MTLSizeMake(256, 1, 1));
}

void MetalGraph::dflash2_dense_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu) {
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && p.dflash2_scope && !p.recording && !p.compatible_pipelines,
        "DFlash2 dense Q8 requires prepared direct DFlash2 scope");
    nax_m8_dense_q8(matrix, input, output, swiglu, 0.0f);
}

void MetalGraph::dflash2_gate_up_swiglu(GpuView gate_matrix, GpuView up_matrix, GpuRowsF32 input, GpuRowsF32 output, bool bf16) {
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && p.dflash2_scope && !p.recording && !p.compatible_pipelines,
        "DFlash2 fused gate+up+SwiGLU requires prepared direct DFlash2 scope");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 524288,
        "DFlash2 fused gate+up+SwiGLU shape bounds mismatch");
    const uint64_t min_in_stride = uint64_t(input.columns) * sizeof(float);
    const uint64_t min_out_stride = uint64_t(output.columns) * sizeof(float);
    require(input.stride_bytes >= min_in_stride && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= min_out_stride && output.stride_bytes % 4 == 0 &&
        input.view.offset % 4 == 0 && output.view.offset % 4 == 0 &&
        input.view.bytes >= uint64_t(input.rows - 1) * input.stride_bytes + min_in_stride &&
        output.view.bytes >= uint64_t(output.rows - 1) * output.stride_bytes + min_out_stride,
        "DFlash2 fused gate+up+SwiGLU row span or alignment mismatch");
    static_cast<void>(p.allocation(gate_matrix));
    static_cast<void>(p.allocation(up_matrix));
    static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(output.view, true));
    separate(output.view, {gate_matrix, up_matrix, input.view});
    if (bf16) {
        const uint32_t row_bytes = input.columns * 2u;
        const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
        require(gate_matrix.bytes == matrix_bytes && up_matrix.bytes == matrix_bytes &&
            gate_matrix.offset % 2 == 0 && up_matrix.offset % 2 == 0,
            "DFlash2 BF16 gate+up weight span mismatch");
        const GroupedMmArgs args{
            input.columns, output.columns, row_bytes, 1u, 1u, input.rows, 1u, 1u,
            matrix_bytes, input.stride_bytes, output.stride_bytes
        };
        p.select("dflash2_gate_up_swiglu_bf16_m8", uint32_t(-1), {input.columns, output.columns, input.rows, 1u}, 256);
        p.bind(gate_matrix, 0); p.bind(up_matrix, 1); p.bind(input.view, 2); p.bind(output.view, 3, true); p.parameter(args, 4);
        p.dispatch(true, MTLSizeMake((output.columns + 15u) / 16u, 1, 1), MTLSizeMake(256, 1, 1));
    } else {
        const auto g_layout = prefill_weight_layout(gate_matrix);
        const auto u_layout = prefill_weight_layout(up_matrix);
        require(g_layout == u_layout && (g_layout == PrefillWeightLayout::gguf ||
            (g_layout == PrefillWeightLayout::q8_lanes_v1 && (input.columns < 512 || input.columns % 256 == 0))),
            "DFlash2 Q8 gate+up layout mismatch");
        const uint32_t row_bytes = (input.columns / 32u) * 34u;
        const uint64_t matrix_bytes = uint64_t(row_bytes) * output.columns;
        require(gate_matrix.bytes >= matrix_bytes && up_matrix.bytes >= matrix_bytes &&
            gate_matrix.offset % 4 == 0 && up_matrix.offset % 4 == 0,
            "DFlash2 Q8 gate+up weight span mismatch");
        p.prefill_weight(gate_matrix, g_layout, 8, input.columns, output.columns, 1);
        p.prefill_weight(up_matrix, u_layout, 8, input.columns, output.columns, 1);
        if (p.verify_nax_m8) {
            const auto* ge = p.q8_tile64_entry(gate_matrix, input.columns, output.columns);
            const auto* ue = p.q8_tile64_entry(up_matrix, input.columns, output.columns);
            if (ge && ue) {
                const uint32_t splits = p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
                    ? pick_nax_m8_q8_t64_splits(input.columns, output.columns, 32u, true)
                    : 1u;
                const GroupedMmArgs args{
                    input.columns, output.columns, row_bytes, 1u, splits, input.rows, 1u, 0u,
                    matrix_bytes, input.stride_bytes, output.stride_bytes
                };
                p.select("nax_m8_dense_q8_swiglu_t64", uint32_t(-1), {input.columns, output.columns, input.rows, uint32_t(g_layout)}, 64);
                p.bind(ge->plane0, 0); p.bind(ge->meta, 1); p.bind(ue->plane0, 2); p.bind(ue->meta, 3);
                p.bind(input.view, 4); p.bind(output.view, 5, true); p.parameter(args, 6);
                if (splits > 1u) {
                    const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
                    const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
                    const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
                    p.bind(part_view, 8, true);
                    p.bind(cnt_view, 9, true);
                } else {
                    const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
                    p.bind(idle, 8, false);
                    p.bind(idle, 9, false);
                }
                p.dispatch(true, MTLSizeMake((output.columns + 31u) / 32u, splits, 1), MTLSizeMake(64, 1, 1));
                return;
            }
        }
        const uint32_t splits = p.verify_nax_split_partials.bytes >= uint64_t(kVerifyNaxSplitSlots) * kVerifyNaxSplitSlotPartialsBytes
            ? pick_nax_m8_q8_splits(input.columns, output.columns, 4u, true)
            : 1u;
        const GroupedMmArgs args{
            input.columns, output.columns, row_bytes, 1u, splits, input.rows, 1u, 0u,
            matrix_bytes, input.stride_bytes, output.stride_bytes
        };
        p.select("nax_m8_dense_q8_swiglu", 8u, {input.columns, output.columns, input.rows, uint32_t(g_layout)}, 128, uint32_t(g_layout));
        p.bind(gate_matrix, 0); p.bind(up_matrix, 1); p.bind(input.view, 2); p.bind(output.view, 3, true); p.parameter(args, 4);
        if (splits > 1u) {
            const uint32_t slot = (p.verify_nax_split_cursor++) % kVerifyNaxSplitSlots;
            const auto part_view = slice(p.verify_nax_split_partials, uint64_t(slot) * kVerifyNaxSplitSlotPartialsBytes, kVerifyNaxSplitSlotPartialsBytes);
            const auto cnt_view = slice(p.verify_nax_split_counters, uint64_t(slot) * kVerifyNaxSplitSlotCountersBytes, kVerifyNaxSplitSlotCountersBytes);
            p.bind(part_view, 5, true);
            p.bind(cnt_view, 6, true);
        } else {
            const auto idle = slice(p.verify_nax_split_counters, p.verify_nax_split_counters.bytes - 64ull, 64ull);
            p.bind(idle, 5, false);
            p.bind(idle, 6, false);
        }
        p.dispatch(true, MTLSizeMake((output.columns + 3u) / 4u, splits, 1), MTLSizeMake(128, 1, 1));
    }
}

void MetalGraph::dflash2_conv_finish_rms_f32(GpuView input, GpuView dynamic, GpuView base, GpuView residual,
    GpuView gains, GpuView residual_out, GpuView norm_out, uint32_t hidden, float eps, uint32_t rows, uint32_t group_size) {
    struct DFlash2ConvRmsArgs {
        uint32_t hidden, rows, group_size;
        float eps;
    };
    static_assert(sizeof(DFlash2ConvRmsArgs) == 16, "DFlash2 conv+RMS ABI");
    auto& p = *impl_; p.active();
    require(p.dflash2_prepared && !p.recording && !p.compatible_pipelines, "DFlash2 conv+RMS requires prepared direct startup");
    require(rows == 8 && group_size == 16 && hidden == 2560 && std::isfinite(eps) && eps > 0.0f, "DFlash2 conv+RMS shape mismatch");
    const uint64_t hidden_bytes = uint64_t(rows) * hidden * 4ull;
    const uint64_t dyn_bytes = uint64_t(rows) * 4ull * (hidden / group_size) * 4ull;
    require(input.bytes >= hidden_bytes && residual.bytes >= hidden_bytes &&
        residual_out.bytes >= hidden_bytes && norm_out.bytes >= hidden_bytes &&
        dynamic.bytes >= dyn_bytes && base.bytes == 4ull * hidden * 4ull && gains.bytes == uint64_t(hidden) * 4ull,
        "DFlash2 conv+RMS buffer span mismatch");
    validate_disjoint(std::array{input, dynamic, base, residual, gains}, std::array{residual_out, norm_out});
    validate_disjoint(std::array{residual_out}, std::array{norm_out});
    const DFlash2ConvRmsArgs args{hidden, rows, group_size, eps};
    p.select("dflash2_conv_finish_rms_f32", uint32_t(-1), {rows, hidden, 1u, group_size}, 128);
    p.bind(input, 0); p.bind(dynamic, 1); p.bind(base, 2); p.bind(residual, 3); p.bind(gains, 4);
    p.bind(residual_out, 5, true); p.bind(norm_out, 6, true); p.parameter(args, 8);
    p.dispatch(true, MTLSizeMake(rows, 1, 1), MTLSizeMake(128, 1, 1));
}
#endif

void MetalGraph::register_planar_matrix(GpuView matrix, PlanarMatrixShape shape) {
    auto& p = *impl_; p.idle();
    const auto& storage = p.allocation(matrix);
    require(!p.planar_union_profile_capacity, "planar union diagnostic requires all matrix registration before preparation");
    require(!p.planar_locked && !p.ledger.live_captures && storage.immutable && matrix.offset % 16 == 0,
        "planar v3 matrices require immutable 16-byte aligned startup registration");
    require(p.planar_matrices.size() < 2048, "planar v3 matrix count refusal");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(!p.lane_shape(matrix), "cannot register a lane matrix as planar v3");
#endif
    const auto kind = static_cast<planar_v3::PlanarKind>(shape.kind);
    require(planar_v3::is_planar_v3_kind(kind), "register_planar_matrix requires a planar v3 kind");
    require(shape.experts >= 1 && shape.experts <= 512 && shape.columns >= 32 && shape.columns <= 65536 &&
        shape.rows >= 4 && shape.rows <= 65536 && shape.rows % 4 == 0,
        "planar v3 matrix shape bounds mismatch");
    if (planar_v3::is_coloc_gate_kind(kind) || planar_v3::is_coloc_up_kind(kind)) {
        require(shape.experts >= 2 && shape.experts % 2 == 0, "co-located planar v3 requires even experts >= 2");
        if (kind == planar_v3::PlanarKind::q4k_coloc_gate_v3 || kind == planar_v3::PlanarKind::q4k_coloc_up_v3)
            require(shape.source_type == 12, "q4k_coloc kind requires source_type 12");
        else if (kind == planar_v3::PlanarKind::q5k_coloc_gate_v3 || kind == planar_v3::PlanarKind::q5k_coloc_up_v3)
            require(shape.source_type == 13, "q5k_coloc kind requires source_type 13");
        else
            require(shape.source_type == 8, "q8_coloc kind requires source_type 8");
    } else if (kind == planar_v3::PlanarKind::q51_planar_v3) {
        require(shape.source_type == 7, "q51_planar_v3 requires source_type 7");
    } else if (kind == planar_v3::PlanarKind::q8_planar_v3) {
        require(shape.source_type == 8, "q8_planar_v3 requires source_type 8");
    }
    const auto spec = planar_v3::make_expert_spec(shape.source_type, shape.columns, shape.rows);
    require(matrix.bytes == checked_mul(spec.planar_expert_bytes, shape.experts), "planar v3 matrix byte span mismatch");
    for (const auto& [key, previous] : p.planar_matrices) {
        static_cast<void>(previous);
        const auto [slot, offset, bytes] = key;
        require(!overlaps(matrix, {p.owner, slot, offset, bytes}), "planar v3 matrix registration overlaps an existing matrix");
    }
    const bool new_slot = p.planar_slots.insert(matrix.slot).second;
    try { p.planar_matrices.emplace(std::tuple{matrix.slot, matrix.offset, matrix.bytes}, shape); }
    catch (...) { if (new_slot) p.planar_slots.erase(matrix.slot); throw; }
}

uint64_t MetalGraph::planar_matrix_count() const {
    impl_->thread_check();
    return impl_->planar_matrices.size();
}

std::optional<PlanarMatrixShape> MetalGraph::planar_matrix_shape(GpuView matrix) const {
    static_cast<void>(impl_->allocation(matrix));
    if (const auto* s = impl_->planar_shape(matrix)) return *s;
    return std::nullopt;
}

void MetalGraph::prepare_planar_moe_v3(uint32_t gate_up_encoding, uint32_t down_encoding, bool prepare_nax,
    uint32_t max_nax_tokens, uint32_t max_experts, uint32_t max_selected, uint32_t max_channels, uint32_t max_hidden) {
    auto& p = *impl_; p.idle();
    require(!p.recording && !p.ledger.live_captures, "planar v3 preparation refuses captured execution");
    require((gate_up_encoding == 8 || gate_up_encoding == 12 || gate_up_encoding == 13) &&
            (down_encoding == 7 || down_encoding == 8),
            "unsupported planar v3 encoding pair");
    static_cast<void>(p.pipeline("planar_moe_build_union_m8", uint32_t(-1), 32));
    for (const auto* gu_k : {"planar_moe_gate_up_swiglu_m1", "planar_moe_gate_up_swiglu_m4", "planar_moe_gate_up_swiglu_m8"}) {
        const auto pipe = p.pipeline(gu_k, gate_up_encoding, 128);
        require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "planar v3 gate/up threadgroup memory exceeds device capability");
    }
    for (const auto* dn_k : {"planar_moe_down_m1", "planar_moe_down_m4", "planar_moe_down_m8"}) {
        const auto pipe = p.pipeline(dn_k, down_encoding, 128);
        require(pipe.staticThreadgroupMemoryLength <= p.device.maxThreadgroupMemoryLength,
            "planar v3 down threadgroup memory exceeds device capability");
    }
    if (!p.planar_union_meta.owner) {
        p.planar_union_meta = zeros(9280, "planar v3 m8 union metadata");
    }
    p.planar_gate_up_prepared.insert(gate_up_encoding);
    p.planar_down_prepared.insert(down_encoding);
    p.prepare_planar_union_profile();

    if (prepare_nax) {
        require(max_nax_tokens >= 1 && max_nax_tokens <= 2048 && max_experts >= 2 && max_experts <= 512 &&
            max_selected >= 1 && max_selected <= 16 && max_channels >= 32 && max_channels % 32 == 0 &&
            max_hidden >= 32 && max_hidden % 32 == 0,
            "invalid planar v3 NAX scratch bounds");
        static_cast<void>(p.pipeline("graph_grouped_routes"));
        static_cast<void>(p.pipeline("nax_grouped_tasks_indirect"));
        static_cast<void>(p.pipeline("nax_moe_gather_bf16_v3"));
        const auto gu_pipe = p.pipeline("nax_grouped_moe_gate_up_swiglu_v3", gate_up_encoding, 64);
        const auto dn_pipe = p.pipeline("nax_grouped_moe_down_v3", down_encoding, 64);
        require(gu_pipe.staticThreadgroupMemoryLength <= 8192u &&
            dn_pipe.staticThreadgroupMemoryLength <= 8192u,
            "planar v3 NAX grouped MoE threadgroup memory exceeds 8 KiB occupancy budget");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        constexpr uint64_t kNaxGroupedScratchBase = 10'485'760ull;
        constexpr uint64_t kPlanarNaxIndirectBytes = 96ull;
        const uint64_t max_tasks = (uint64_t(max_nax_tokens) * max_selected) / 32ull + max_experts;
        const uint64_t max_padded_rows = ((uint64_t(max_nax_tokens) + 63ull) / 64ull) * 64ull;
        const uint64_t raw_x_cap = ((checked_mul(max_padded_rows, uint64_t(max_channels) * 2ull) + 255ull) / 256ull) * 256ull;
        const uint64_t base_cap = std::max<uint64_t>(kNaxGroupedScratchBase, raw_x_cap);
        const uint64_t gx_bytes = ((checked_mul(max_tasks * 32ull, uint64_t(max_channels) * 2ull) + 255ull) / 256ull) * 256ull;
        const uint64_t h_bytes = ((checked_mul(max_tasks * 32ull, uint64_t(max_hidden) * 2ull) + 255ull) / 256ull) * 256ull;
        const uint64_t total_bf16 = checked_add(checked_add(base_cap, gx_bytes), h_bytes);
        if (p.nax_bf16_scratch.bytes < total_bf16) {
            if (p.nax_bf16_scratch.bytes) release(p.nax_bf16_scratch);
            p.nax_bf16_scratch = zeros(total_bf16, "nax_bf16_scratch");
        }
        if (p.nax_indirect_args.bytes < kPlanarNaxIndirectBytes) {
            if (p.nax_indirect_args.bytes) release(p.nax_indirect_args);
            p.nax_indirect_args = zeros(kPlanarNaxIndirectBytes, "nax_indirect_args");
        }
#endif
        p.planar_nax_gate_up_prepared.insert(gate_up_encoding);
        p.planar_nax_down_prepared.insert(down_encoding);
    }
}

void MetalGraph::planar_moe_gate_up_swiglu_m8(GpuView pair_lo, GpuView pair_hi, GpuView indices,
    GpuRowsF32 input, GpuRowsF32 expert_hidden, uint32_t experts, uint32_t selected) {
    struct PlanarUnionArgs {
        uint32_t experts, selected, tokens, hidden, channels, r0, r1, r2;
    };
    struct PlanarSmallMoeArgs {
        uint32_t columns, rows, experts, selected;
        uint32_t tokens, use_union, r0, r1;
        uint64_t planar_expert_bytes, qs_offset, qh_offset, sm_offset, dm_offset;
        uint64_t input_stride_bytes, output_stride_bytes;
    };
    static_assert(sizeof(PlanarSmallMoeArgs) == 88, "planar small MoE ABI");
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines && p.planar_union_meta.owner,
        "planar v3 small MoE requires prepared direct batches");
    const auto* lo_s = p.planar_shape(pair_lo);
    const auto* hi_s = p.planar_shape(pair_hi);
    require(lo_s && hi_s &&
        planar_v3::is_coloc_gate_kind(static_cast<planar_v3::PlanarKind>(lo_s->kind)) &&
        planar_v3::is_coloc_up_kind(static_cast<planar_v3::PlanarKind>(hi_s->kind)) &&
        lo_s->source_type == hi_s->source_type && lo_s->columns == hi_s->columns &&
        lo_s->rows == hi_s->rows && lo_s->experts == hi_s->experts && lo_s->experts == experts,
        "planar v3 co-located gate/up registration mismatch");
    require(p.planar_gate_up_prepared.contains(lo_s->source_type), "planar v3 gate/up encoding not prepared");

    const uint32_t tokens = input.rows;
    require(tokens >= 1 && tokens <= 8 && selected >= 1 && selected <= 16 && selected <= experts &&
        input.columns == lo_s->columns && expert_hidden.columns == lo_s->rows &&
        expert_hidden.rows == tokens * selected &&
        input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        expert_hidden.stride_bytes >= uint64_t(expert_hidden.columns) * 4ull && expert_hidden.stride_bytes % 4 == 0,
        "planar v3 gate/up row or stride geometry mismatch");
    const uint64_t in_bytes = checked_add(checked_mul(tokens - 1, input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t out_bytes = checked_add(checked_mul(expert_hidden.rows - 1, expert_hidden.stride_bytes), uint64_t(expert_hidden.columns) * 4ull);
    require(input.view.bytes >= in_bytes && expert_hidden.view.bytes >= out_bytes &&
        indices.bytes >= uint64_t(expert_hidden.rows) * 4ull &&
        input.view.offset % 4 == 0 && expert_hidden.view.offset % 4 == 0 && indices.offset % 4 == 0,
        "planar v3 gate/up buffer span or alignment mismatch");
    static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(indices));
    static_cast<void>(p.allocation(expert_hidden.view, true));
    validate_disjoint(std::array{pair_lo, pair_hi, input.view, indices}, std::array{expert_hidden.view, p.planar_union_meta});

    const auto gu_spec = planar_v3::make_expert_spec(lo_s->source_type, lo_s->columns, lo_s->rows);
    const bool use_union = tokens > 1u;
    if (use_union && p.planar_union_profile_active()) require(p.planar_union_profile_binding, "planar union diagnostic requires the complete routed operator");
    if (use_union) {
        const PlanarUnionArgs u_args{experts, selected, tokens, lo_s->rows, lo_s->columns, 0u, 0u, 0u};
        p.select("planar_moe_build_union_m8", uint32_t(-1), {experts, selected, lo_s->rows, lo_s->columns}, 32);
        p.bind(indices, 0); p.bind(p.planar_union_meta, 1, true); p.parameter(u_args, 2);
        p.dispatch(true, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
    }

    const PlanarSmallMoeArgs gu_args{
        lo_s->columns, lo_s->rows, experts, selected, tokens, use_union ? 1u : 0u, 0u, 0u,
        gu_spec.planar_expert_bytes, gu_spec.qs_offset, gu_spec.qh_offset, gu_spec.sm_offset, gu_spec.dm_offset,
        input.stride_bytes, expert_hidden.stride_bytes
    };
    const char* gu_kernel = !use_union
        ? "planar_moe_gate_up_swiglu_m1"
        : (tokens <= 4u ? "planar_moe_gate_up_swiglu_m4" : "planar_moe_gate_up_swiglu_m8");
    p.select(gu_kernel, lo_s->source_type, {lo_s->columns, lo_s->rows, experts, tokens}, 128);
    p.bind(pair_lo, 0); p.bind(pair_hi, 1); p.bind(input.view, 2); p.bind(expert_hidden.view, 3, true);
    p.bind(use_union ? p.planar_union_meta : indices, 4); p.parameter(gu_args, 5);
    if (use_union) {
        p.dispatch_indirect(p.planar_union_meta, 0, MTLSizeMake(128, 1, 1));
    } else {
        p.dispatch(true, MTLSizeMake((lo_s->rows + 7u) / 8u, selected, 1), MTLSizeMake(128, 1, 1));
    }
}

void MetalGraph::planar_moe_routed_m8(GpuView pair_lo, GpuView pair_hi, GpuView down_weights, GpuView indices,
    GpuRowsF32 input, GpuRowsF32 expert_hidden, GpuRowsF32 expert_down,
    uint32_t experts, uint32_t selected) {
    struct PlanarSmallMoeArgs {
        uint32_t columns, rows, experts, selected;
        uint32_t tokens, use_union, r0, r1;
        uint64_t planar_expert_bytes, qs_offset, qh_offset, sm_offset, dm_offset;
        uint64_t input_stride_bytes, output_stride_bytes;
    };
    auto& p = *impl_; p.active();
    const auto* dn_s = p.planar_shape(down_weights);
    require(dn_s &&
        (static_cast<planar_v3::PlanarKind>(dn_s->kind) == planar_v3::PlanarKind::q51_planar_v3 ||
         static_cast<planar_v3::PlanarKind>(dn_s->kind) == planar_v3::PlanarKind::q8_planar_v3) &&
        dn_s->experts == experts && dn_s->columns == expert_hidden.columns && dn_s->rows == input.columns,
        "planar v3 down registration mismatch");
    require(p.planar_down_prepared.contains(dn_s->source_type), "planar v3 down encoding not prepared");

    const uint32_t tokens = input.rows;
    require(expert_down.rows == tokens * selected && expert_down.columns == dn_s->rows &&
        expert_down.stride_bytes >= uint64_t(expert_down.columns) * 4ull && expert_down.stride_bytes % 4 == 0,
        "planar v3 down output geometry mismatch");
    const uint64_t dn_out_bytes = checked_add(checked_mul(expert_down.rows - 1, expert_down.stride_bytes), uint64_t(expert_down.columns) * 4ull);
    require(expert_down.view.bytes >= dn_out_bytes && expert_down.view.offset % 4 == 0,
        "planar v3 down output buffer span mismatch");
    static_cast<void>(p.allocation(expert_down.view, true));
    const bool diagnostic = tokens > 1 && p.planar_union_profile_active();
    struct RestoreUnionMetadata {
        Impl& p;
        GpuView original;
        bool active;
        ~RestoreUnionMetadata() { if (active) { p.planar_union_meta = original; p.planar_union_profile_binding = false; } }
    } restore{p, p.planar_union_meta, false};
    size_t diagnostic_record = 0;
    if (diagnostic) {
        p.require_planar_union_profile_scope();
        require(!p.planar_union_profile_binding && p.planar_union_profile_records.size() < p.planar_union_profile_capacity,
            "planar union diagnostic batch exceeds the registered gate inventory");
        const auto* gu_s = p.planar_shape(pair_lo);
        require(gu_s && planar_v3::is_coloc_gate_kind(static_cast<planar_v3::PlanarKind>(gu_s->kind)), "planar union diagnostic gate registration is missing");
        uint32_t ordinal = 0;
        bool found = false;
        for (const auto& [key, shape] : p.planar_matrices) {
            if (!planar_v3::is_coloc_gate_kind(static_cast<planar_v3::PlanarKind>(shape.kind))) continue;
            if (key == std::tuple{pair_lo.slot, pair_lo.offset, pair_lo.bytes}) { found = true; break; }
            ++ordinal;
        }
        require(found && ordinal < p.planar_union_profile_capacity, "planar union diagnostic gate ordinal is invalid");
        for (const auto& previous : p.planar_union_profile_records)
            require(previous.matrix != ordinal, "planar union diagnostic refuses repeated gate matrices in a batch");
        diagnostic_record = p.planar_union_profile_records.size();
        p.planar_union_profile_records.push_back({ordinal, tokens, experts, selected, gu_s->source_type,
            dn_s->source_type, input.columns, expert_hidden.columns, p.dispatches});
        p.planar_union_meta = slice(p.planar_union_profile_bank, diagnostic_record * Impl::planar_union_profile_stride, Impl::planar_union_profile_stride);
        p.planar_union_profile_binding = true;
        restore.active = true;
    }
    validate_disjoint(std::array{pair_lo, pair_hi, down_weights, input.view, indices},
                      std::array{expert_hidden.view, expert_down.view, p.planar_union_meta});

    planar_moe_gate_up_swiglu_m8(pair_lo, pair_hi, indices, input, expert_hidden, experts, selected);

    const auto dn_spec = planar_v3::make_expert_spec(dn_s->source_type, dn_s->columns, dn_s->rows);
    const bool use_union = tokens > 1u;
    const PlanarSmallMoeArgs dn_args{
        dn_s->columns, dn_s->rows, experts, selected, tokens, use_union ? 1u : 0u, 0u, 0u,
        dn_spec.planar_expert_bytes, dn_spec.qs_offset, dn_spec.qh_offset, dn_spec.sm_offset, dn_spec.dm_offset,
        expert_hidden.stride_bytes, expert_down.stride_bytes
    };
    const char* dn_kernel = !use_union
        ? "planar_moe_down_m1"
        : (tokens <= 4u ? "planar_moe_down_m4" : "planar_moe_down_m8");
    p.select(dn_kernel, dn_s->source_type, {dn_s->columns, dn_s->rows, experts, tokens}, 128);
    p.bind(down_weights, 0); p.bind(expert_hidden.view, 1); p.bind(expert_down.view, 2, true);
    p.bind(use_union ? p.planar_union_meta : indices, 3); p.parameter(dn_args, 4);
    if (use_union) {
        p.dispatch_indirect(p.planar_union_meta, 12, MTLSizeMake(128, 1, 1));
    } else {
        p.dispatch(true, MTLSizeMake((dn_s->rows + 15u) / 16u, selected, 1), MTLSizeMake(128, 1, 1));
    }
    if (diagnostic) {
        auto& record = p.planar_union_profile_records[diagnostic_record];
        require(record.dispatch <= UINT64_MAX - 3 && p.dispatches == record.dispatch + 3,
            "planar union diagnostic routed dispatch count differs");
        record.complete = true;
    }
}

void MetalGraph::planar_nax_grouped_moe(GpuView pair_lo, GpuView pair_hi, GpuView down_weights, GpuView indices,
    GpuRowsF32 input, GpuRowsF32 expert_down, GpuView token_map, GpuView counts, GpuView tasks,
    uint32_t experts, uint32_t selected) {
    struct PlanarNaxMoeArgs {
        uint32_t columns, rows, experts, selected;
        uint32_t tokens, maximum_tasks, write_bf16_gathered, r0;
        uint64_t planar_expert_bytes, qs_offset, qh_offset, sm_offset, dm_offset;
        uint64_t input_stride_bytes, output_stride_bytes;
    };
    static_assert(sizeof(PlanarNaxMoeArgs) == 88, "planar NAX MoE ABI");
    auto& p = *impl_; p.active();
    require(!p.recording && !p.compatible_pipelines, "planar v3 NAX grouped MoE requires direct batches");
#ifndef TOKSPUR_PRIVATE_GROUPED_MM
    static_cast<void>(pair_lo); static_cast<void>(pair_hi); static_cast<void>(down_weights);
    static_cast<void>(indices); static_cast<void>(input); static_cast<void>(expert_down);
    static_cast<void>(token_map); static_cast<void>(counts); static_cast<void>(tasks);
    static_cast<void>(experts); static_cast<void>(selected);
    throw FormatError("planar v3 NAX grouped MoE requires the carrier grouped-mm build");
#else
    const auto* lo_s = p.planar_shape(pair_lo);
    const auto* hi_s = p.planar_shape(pair_hi);
    const auto* dn_s = p.planar_shape(down_weights);
    require(lo_s && hi_s && dn_s &&
        planar_v3::is_coloc_gate_kind(static_cast<planar_v3::PlanarKind>(lo_s->kind)) &&
        planar_v3::is_coloc_up_kind(static_cast<planar_v3::PlanarKind>(hi_s->kind)) &&
        (static_cast<planar_v3::PlanarKind>(dn_s->kind) == planar_v3::PlanarKind::q51_planar_v3 ||
         static_cast<planar_v3::PlanarKind>(dn_s->kind) == planar_v3::PlanarKind::q8_planar_v3) &&
        lo_s->source_type == hi_s->source_type && lo_s->columns == hi_s->columns &&
        lo_s->rows == hi_s->rows && lo_s->experts == hi_s->experts &&
        lo_s->experts == experts && dn_s->experts == experts &&
        lo_s->columns == input.columns && lo_s->rows == dn_s->columns && dn_s->rows == expert_down.columns,
        "planar v3 NAX grouped MoE registration mismatch");
    require(p.planar_nax_gate_up_prepared.contains(lo_s->source_type) &&
            p.planar_nax_down_prepared.contains(dn_s->source_type),
            "planar v3 NAX grouped MoE encodings not prepared");

    const uint32_t tokens = input.rows;
    const uint32_t channels = lo_s->columns;
    const uint32_t hidden = lo_s->rows;
    require(tokens >= 1 && tokens <= 2048 && selected >= 1 && selected <= 16 && selected <= experts &&
        channels % 32 == 0 && hidden % 32 == 0 && expert_down.rows == tokens * selected &&
        input.stride_bytes >= uint64_t(channels) * 4ull && input.stride_bytes % 4 == 0 &&
        expert_down.stride_bytes >= uint64_t(channels) * 4ull && expert_down.stride_bytes % 4 == 0,
        "planar v3 NAX grouped MoE shape or stride mismatch");

    const uint32_t task_rows = 32u;
    const uint64_t route_count = checked_mul(tokens, selected);
    const uint32_t maximum_tasks = static_cast<uint32_t>(route_count / task_rows + experts);
    const uint64_t in_bytes = checked_add(checked_mul(tokens - 1, input.stride_bytes), uint64_t(channels) * 4ull);
    const uint64_t out_bytes = checked_add(checked_mul(expert_down.rows - 1, expert_down.stride_bytes), uint64_t(channels) * 4ull);
    require(input.view.bytes >= in_bytes && expert_down.view.bytes >= out_bytes &&
        indices.bytes >= route_count * 4ull &&
        token_map.bytes >= checked_mul(checked_mul(experts, tokens), 4ull) &&
        counts.bytes >= uint64_t(experts) * 4ull &&
        tasks.bytes >= checked_mul(checked_add(2ull, uint64_t(maximum_tasks) * 2ull), 4ull),
        "planar v3 NAX grouped MoE buffer span mismatch");
    static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(indices));
    static_cast<void>(p.allocation(expert_down.view, true));
    static_cast<void>(p.allocation(token_map, true));
    static_cast<void>(p.allocation(counts, true));
    static_cast<void>(p.allocation(tasks, true));

    constexpr uint64_t kNaxGroupedScratchBase = 10'485'760ull;
    const bool reuse_x_bf16 = can_reuse_nax_dense_bf16(input);
    const uint32_t padded_rows = ((tokens + 31u) / 32u) * 32u;
    const uint64_t raw_x_bytes = checked_mul(checked_mul(uint64_t(padded_rows), uint64_t(channels)), 2ull);
    const uint64_t aligned_raw_x_bytes = ((raw_x_bytes + 255ull) / 256ull) * 256ull;
    const uint64_t gx_base = std::max<uint64_t>(kNaxGroupedScratchBase, aligned_raw_x_bytes);
    const uint64_t gx_bytes = checked_mul(uint64_t(maximum_tasks) * task_rows, uint64_t(channels) * 2ull);
    const uint64_t aligned_gx_bytes = ((gx_bytes + 255ull) / 256ull) * 256ull;
    const uint64_t h_base = checked_add(gx_base, aligned_gx_bytes);
    const uint64_t h_bytes = checked_mul(uint64_t(maximum_tasks) * task_rows, uint64_t(hidden) * 2ull);
    require(p.nax_bf16_scratch.bytes >= checked_add(h_base, h_bytes) &&
            p.nax_indirect_args.bytes >= 48ull,
        "planar v3 NAX BF16 scratch too small for requested batch");
    const auto x_bf16 = slice(p.nax_bf16_scratch, 0, raw_x_bytes);
    const auto gathered_x = slice(p.nax_bf16_scratch, gx_base, gx_bytes);
    const auto gathered_h = slice(p.nax_bf16_scratch, h_base, h_bytes);
    p.nax_gathered_broadcast_valid = false;

    if (reuse_x_bf16) {
        validate_disjoint(std::array{pair_lo, pair_hi, down_weights, input.view, indices, x_bf16},
                          std::array{expert_down.view, token_map, counts, tasks, gathered_x, gathered_h, p.nax_indirect_args});
    } else {
        validate_disjoint(std::array{pair_lo, pair_hi, down_weights, input.view, indices},
                          std::array{expert_down.view, token_map, counts, tasks, gathered_x, gathered_h, p.nax_indirect_args});
    }

    const std::array<uint32_t, 4> grouping{experts, selected, tokens, task_rows};
    p.select("graph_grouped_routes");
    p.bind(indices, 0); p.bind(token_map, 1, true); p.bind(counts, 2, true); p.parameter(grouping, 3);
    p.groups(MTLSizeMake(experts, 1, 1));

    const std::array<uint32_t, 2> grid_x{(hidden + 31u) / 32u, (channels + 63u) / 64u};
    p.select("nax_grouped_tasks_indirect");
    p.bind(counts, 0); p.bind(tasks, 1, true); p.parameter(grouping, 2);
    p.bind(p.nax_indirect_args, 3, true); p.parameter(grid_x, 4);
    p.groups(MTLSizeMake(1, 1, 1));

    const auto gu_spec = planar_v3::make_expert_spec(lo_s->source_type, channels, hidden);
    const PlanarNaxMoeArgs gu_args{
        channels, hidden, experts, selected, tokens, maximum_tasks, 1u, reuse_x_bf16 ? 1u : 0u,
        gu_spec.planar_expert_bytes, gu_spec.qs_offset, gu_spec.qh_offset, gu_spec.sm_offset, gu_spec.dm_offset,
        input.stride_bytes, 0ull
    };
    p.select("nax_moe_gather_bf16_v3", uint32_t(-1), {channels, tokens, maximum_tasks, reuse_x_bf16 ? 1u : 0u}, 128);
    p.bind(reuse_x_bf16 ? x_bf16 : input.view, 0); p.bind(gathered_x, 1, true); p.parameter(gu_args, 2);
    p.bind(indices, 3); p.bind(token_map, 4); p.bind(tasks, 5); p.bind(counts, 6);
    p.dispatch_indirect(p.nax_indirect_args, 32, MTLSizeMake(128, 1, 1));

    p.select("nax_grouped_moe_gate_up_swiglu_v3", lo_s->source_type, {channels, hidden, experts, tokens}, 64);
    p.bind(pair_lo, 0); p.bind(pair_hi, 1); p.bind(gathered_x, 2); p.bind(gathered_h, 3, true);
    p.parameter(gu_args, 4); p.bind(indices, 5); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
    p.dispatch_indirect(p.nax_indirect_args, 0, MTLSizeMake(64, 1, 1));

    const auto dn_spec = planar_v3::make_expert_spec(dn_s->source_type, hidden, channels);
    const PlanarNaxMoeArgs dn_args{
        hidden, channels, experts, selected, tokens, maximum_tasks, 0u, 0u,
        dn_spec.planar_expert_bytes, dn_spec.qs_offset, dn_spec.qh_offset, dn_spec.sm_offset, dn_spec.dm_offset,
        0ull, expert_down.stride_bytes
    };
    p.select("nax_grouped_moe_down_v3", dn_s->source_type, {hidden, channels, experts, tokens}, 64);
    p.bind(down_weights, 0); p.bind(gathered_h, 1); p.bind(expert_down.view, 2, true);
    p.parameter(dn_args, 3); p.bind(indices, 4); p.bind(token_map, 6); p.bind(tasks, 8); p.bind(counts, 9);
    p.dispatch_indirect(p.nax_indirect_args, 16, MTLSizeMake(64, 1, 1));
#endif
}

uint64_t MetalGraph::logit_topk_scratch_bytes(uint32_t max_rows, uint32_t max_vocabulary) {
    require(max_rows >= 1 && max_rows <= 2048 && max_vocabulary >= 1 && max_vocabulary <= maximum_sampling_vocabulary,
        "GPU logit top-k scratch bounds mismatch");
    const uint32_t tiles = std::min<uint32_t>(gpu_logit_topk_max_tiles,
        (max_vocabulary + gpu_logit_topk_tile_span - 1u) / gpu_logit_topk_tile_span);
    return checked_mul(checked_mul(uint64_t(max_rows), tiles), sizeof(GpuLogitTopKRow));
}

uint64_t MetalGraph::logit_topk_output_bytes(uint32_t max_rows) {
    require(max_rows >= 1 && max_rows <= 2048, "GPU logit top-k output row count out of bounds");
    return checked_mul(uint64_t(max_rows), sizeof(GpuLogitTopKRow));
}

void MetalGraph::prepare_logit_topk() {
    auto& p = *impl_; p.idle();
    require(!p.recording && !p.ledger.live_captures, "GPU logit top-k refuses captures");
    static_cast<void>(p.pipeline("logit_topk_stage1", uint32_t(-1), 256));
    static_cast<void>(p.pipeline("logit_topk_stage2", uint32_t(-1), 256));
    p.logit_topk_prepared = true;
}

void MetalGraph::logit_topk(GpuRowsF32 logits, GpuView partial_scratch, GpuView out_rows) {
    struct LogitTopKArgs {
        uint32_t rows, vocabulary, tiles, tile_span;
        uint64_t input_stride_bytes, reserved;
    };
    static_assert(sizeof(LogitTopKArgs) == 32, "LogitTopKArgs ABI");
    auto& p = *impl_; p.active();
    require(p.logit_topk_prepared && !p.recording && !p.compatible_pipelines,
        "GPU logit top-k requires prepared direct batches");
    require(logits.rows >= 1 && logits.rows <= 2048 &&
        logits.columns >= 1 && logits.columns <= maximum_sampling_vocabulary &&
        logits.stride_bytes >= uint64_t(logits.columns) * 4ull && logits.stride_bytes % 4 == 0,
        "GPU logit top-k geometry mismatch");
    const uint32_t tiles = (logits.columns + gpu_logit_topk_tile_span - 1u) / gpu_logit_topk_tile_span;
    require(tiles >= 1 && tiles <= gpu_logit_topk_max_tiles, "GPU logit top-k tile count out of bounds");
    const uint64_t in_bytes = checked_add(checked_mul(uint64_t(logits.rows - 1), logits.stride_bytes), uint64_t(logits.columns) * 4ull);
    const uint64_t scratch_bytes = checked_mul(checked_mul(uint64_t(logits.rows), tiles), sizeof(GpuLogitTopKRow));
    const uint64_t out_bytes = checked_mul(uint64_t(logits.rows), sizeof(GpuLogitTopKRow));
    static_cast<void>(p.allocation(logits.view));
    static_cast<void>(p.allocation(partial_scratch, true));
    static_cast<void>(p.allocation(out_rows, true));
    require(logits.view.bytes >= in_bytes && partial_scratch.bytes >= scratch_bytes && out_rows.bytes >= out_bytes &&
        logits.view.offset % 4 == 0 && partial_scratch.offset % 4 == 0 && out_rows.offset % 4 == 0,
        "GPU logit top-k buffer span or alignment mismatch");
    validate_disjoint(std::array{logits.view}, std::array{partial_scratch, out_rows});
    validate_disjoint(std::array{partial_scratch}, std::array{out_rows});

    const LogitTopKArgs args{logits.rows, logits.columns, tiles, gpu_logit_topk_tile_span, logits.stride_bytes, 0ull};
    p.select("logit_topk_stage1", uint32_t(-1), {logits.rows, logits.columns, tiles, gpu_logit_topk_tile_span}, 256);
    p.bind(logits.view, 0); p.bind(partial_scratch, 1, true); p.parameter(args, 2);
    p.dispatch(true, MTLSizeMake(tiles, logits.rows, 1), MTLSizeMake(256, 1, 1));

    p.select("logit_topk_stage2", uint32_t(-1), {logits.rows, logits.columns, tiles, gpu_logit_topk_capacity}, 256);
    p.bind(partial_scratch, 0); p.bind(out_rows, 1, true); p.parameter(args, 2);
    p.dispatch(true, MTLSizeMake(1, logits.rows, 1), MTLSizeMake(256, 1, 1));
}

namespace {
const char* verify_fused_hc_q8_kernel_name(uint32_t tokens) {
    static constexpr std::array<const char*, 8> names{
        "verify_fused_hc_q8_1", "verify_fused_hc_q8_2", "verify_fused_hc_q8_3", "verify_fused_hc_q8_4",
        "verify_fused_hc_q8_5", "verify_fused_hc_q8_6", "verify_fused_hc_q8_7", "verify_fused_hc_q8_8",
    };
    require(tokens >= 1 && tokens <= names.size(), "fused verify HC Q8 token count out of bounds");
    return names[tokens - 1];
}
} // namespace

void MetalGraph::prepare_verify_hc_gdn_fused() {
    auto& p = *impl_; p.idle();
    require(!p.recording && !p.ledger.live_captures, "fused verify HC/GDN kernels refuse captures");
    for (uint32_t t = 1u; t <= 8u; ++t) {
        const char* kname = verify_fused_hc_q8_kernel_name(t);
        static_cast<void>(p.pipeline(kname, 8u, 128, false, 0u));
        static_cast<void>(p.pipeline(kname, 8u, 128, false, 1u));
    }
    static_cast<void>(p.pipeline("verify_fused_hc_write_rms_rows", uint32_t(-1), 32));
    for (const auto* name : {"verify_fused_hc_inject_partials",
                             "verify_fused_hc_read_rows",
                             "verify_fused_hc_gated_mix_rows",
                             "verify_fused_hc_write_rows",
                             "verify_fused_gdn_alpha_beta_f32"}) {
        static_cast<void>(p.pipeline(name, uint32_t(-1), 128));
    }
    if (!p.verify_hc_partials_scratch.owner) {
        p.verify_hc_partials_scratch = zeros(64ull * 16ull * 32ull * sizeof(float), "verify fused HC inject partials");
    }
    p.verify_hc_gdn_fused_prepared = true;
}

void MetalGraph::configure_verify_hc_gdn_fused(bool enabled) {
    auto& p = *impl_; p.idle();
    require(!p.verify_hc_gdn_fused_locked && !p.recording && !p.ledger.live_captures,
        "fused verify HC/GDN policy is fixed before the first batch");
    if (enabled) prepare_verify_hc_gdn_fused();
    p.verify_hc_gdn_fused = enabled;
}

bool MetalGraph::verify_hc_gdn_fused_enabled() const {
    impl_->thread_check();
    return impl_->verify_hc_gdn_fused;
}

void MetalGraph::verify_fused_hc_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool silu, float scale) {
    struct VerifyFusedHcQ8Args {
        uint32_t columns, rows, row_bytes, tokens;
        uint32_t silu_flag;
        float scale;
        uint32_t reserved0, reserved1;
        uint64_t input_stride_bytes, output_stride_bytes;
    };
    static_assert(sizeof(VerifyFusedHcQ8Args) == 48, "VerifyFusedHcQ8Args ABI");
    auto& p = *impl_; p.active();
    require(p.verify_hc_gdn_fused_prepared && !p.recording && !p.compatible_pipelines,
        "fused verify HC Q8 requires prepared direct batches");
    require(input.rows >= 1 && input.rows <= 8 && input.rows == output.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        output.columns >= 1 && output.columns <= 65536 &&
        input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        output.stride_bytes >= uint64_t(output.columns) * 4ull && output.stride_bytes % 4 == 0 &&
        std::isfinite(scale),
        "fused verify HC Q8 geometry mismatch");
    require(!p.planar_shape(matrix), "fused verify HC Q8 refuses planar MoE weights");
    uint32_t layout = 0u;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (const auto* lane = p.lane_shape(matrix)) {
        require(lane->encoding == 8 && lane->columns == input.columns && lane->rows == output.columns && lane->experts == 1 &&
            (input.columns < 512 || input.columns % 256 == 0),
            "fused verify HC Q8 packed lane shape mismatch");
        layout = 1u;
    }
#endif
    const uint32_t row_bytes = (input.columns / 32u) * 34u;
    const uint64_t mat_bytes = checked_mul(uint64_t(row_bytes), output.columns);
    const uint64_t in_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t out_bytes = checked_add(checked_mul(uint64_t(output.rows - 1), output.stride_bytes), uint64_t(output.columns) * 4ull);
    static_cast<void>(p.allocation(matrix));
    static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(output.view, true));
    require(matrix.bytes == mat_bytes && input.view.bytes >= in_bytes && output.view.bytes >= out_bytes &&
        matrix.offset % (layout ? 16 : 4) == 0 && input.view.offset % 4 == 0 && output.view.offset % 4 == 0,
        "fused verify HC Q8 buffer span or alignment mismatch");
    separate(output.view, {matrix, input.view});

    const VerifyFusedHcQ8Args args{
        input.columns, output.columns, row_bytes, input.rows,
        silu ? 1u : 0u, scale, 0u, 0u,
        input.stride_bytes, output.stride_bytes
    };
    const char* kname = verify_fused_hc_q8_kernel_name(input.rows);
    p.select(kname, 8u, {input.columns, output.columns, input.rows, silu ? 1u : 0u}, 128, layout);
    p.bind(matrix, 0); p.bind(input.view, 1); p.bind(output.view, 2, true); p.parameter(args, 3);
    p.groups(MTLSizeMake((output.columns + 3u) / 4u, 1, 1));
}

void MetalGraph::verify_fused_hc_inject_read_rows(GpuView weights, GpuView normalized, GpuView scores,
    GpuView inject_logits, GpuView output, GpuView scales,
    uint32_t branches, uint32_t channels, uint32_t tokens) {
    auto& p = *impl_; p.active();
    require(p.verify_hc_gdn_fused_prepared && p.verify_hc_partials_scratch.owner &&
        !p.recording && !p.compatible_pipelines,
        "fused verify HC inject+read requires prepared direct batches");
    const uint32_t columns = checked_mul(branches, channels);
    require(branches >= 1 && branches <= 16 && channels >= 32 && channels >= branches &&
        columns <= 65536 && columns % 32 == 0 && tokens >= 1 && tokens <= 64,
        "fused verify HC inject+read geometry mismatch");
    require(!p.planar_shape(weights), "fused verify HC inject+read refuses planar weights");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(!p.lane_shape(weights), "fused verify HC inject+read requires FP32 row-major weights");
#endif
    require(p.floats(weights) == checked_mul(branches, columns) &&
        p.floats64(normalized) == uint64_t(tokens) * columns &&
        p.floats64(scores) == uint64_t(tokens) * columns &&
        p.floats(inject_logits, true) == checked_mul(tokens, branches) &&
        p.floats64(output, true) == uint64_t(tokens) * channels &&
        p.floats(scales, true) == checked_mul(tokens, branches),
        "fused verify HC inject+read buffer span mismatch");

    const uint32_t splits = (columns >= 1024 && branches <= 64) ? p.small_gemv_splits : 1u;
    require(splits >= 1 && splits <= 32, "fused verify HC inject+read invalid split count");
    const GpuView partials = slice(p.verify_hc_partials_scratch, 0,
        uint64_t(tokens) * branches * splits * sizeof(float));
    validate_disjoint(std::array{weights, normalized, scores}, std::array{partials, inject_logits, output, scales});
    validate_disjoint(std::array{partials}, std::array{inject_logits, output, scales});
    validate_disjoint(std::array{inject_logits}, std::array{output, scales});
    validate_disjoint(std::array{output}, std::array{scales});

    const std::array<uint32_t, 4> pshape{columns, branches, splits, tokens};
    p.select("verify_fused_hc_inject_partials", uint32_t(-1), pshape, 128);
    p.bind(weights, 0); p.bind(normalized, 1); p.bind(partials, 2, true); p.parameter(pshape, 3);
    p.dispatch(true, MTLSizeMake(branches == 4u ? 1u : branches, (splits + 3u) / 4u, (tokens + 7u) / 8u), MTLSizeMake(128, 1, 1));

    const std::array<uint32_t, 4> rshape{branches, channels, tokens, splits};
    p.select("verify_fused_hc_read_rows", uint32_t(-1), rshape, 128);
    p.bind(normalized, 0); p.bind(scores, 1); p.bind(partials, 2);
    p.bind(output, 3, true); p.bind(inject_logits, 4, true); p.bind(scales, 5, true);
    p.parameter(rshape, 6);
    p.threads(tokens * ((channels + 3u) / 4u));
}

void MetalGraph::verify_fused_hc_read_rows(GpuView normalized, GpuView scores, GpuView output,
    uint32_t branches, uint32_t channels, uint32_t tokens) {
    auto& p = *impl_; p.active();
    require(p.verify_hc_gdn_fused_prepared && !p.recording && !p.compatible_pipelines,
        "fused verify HC gated_mix requires prepared direct batches");
    const uint32_t width = checked_mul(branches, channels);
    require(branches >= 1 && branches <= 16 && channels >= 1 && channels <= 65536 && tokens >= 1 && tokens <= 64,
        "fused verify HC gated_mix geometry mismatch");
    require(p.floats64(normalized) == uint64_t(tokens) * width &&
        p.floats64(scores) == uint64_t(tokens) * width &&
        p.floats64(output, true) == uint64_t(tokens) * channels,
        "fused verify HC gated_mix buffer span mismatch");
    validate_disjoint(std::array{normalized, scores}, std::array{output});

    const std::array<uint32_t, 4> shape{branches, channels, tokens, 0u};
    p.select("verify_fused_hc_gated_mix_rows", uint32_t(-1), shape, 128);
    p.bind(normalized, 0); p.bind(scores, 1); p.bind(output, 2, true); p.parameter(shape, 3);
    p.threads(tokens * ((channels + 3u) / 4u));
}

void MetalGraph::verify_fused_hc_write_rows(GpuView streams, GpuView input, GpuView scales, GpuView destination,
    uint32_t branches, uint32_t channels, uint32_t tokens) {
    auto& p = *impl_; p.active();
    require(p.verify_hc_gdn_fused_prepared && !p.recording && !p.compatible_pipelines,
        "fused verify HC write requires prepared direct batches");
    const uint32_t width = checked_mul(branches, channels);
    require(branches >= 1 && branches <= 16 && channels >= 1 && channels <= 65536 && tokens >= 1 && tokens <= 64,
        "fused verify HC write geometry mismatch");
    require(p.floats64(streams) == uint64_t(tokens) * width &&
        p.floats64(input) == uint64_t(tokens) * channels &&
        p.floats(scales) == checked_mul(tokens, branches) &&
        p.floats64(destination, true) == uint64_t(tokens) * width,
        "fused verify HC write buffer span mismatch");
    separate(destination, {input, scales});
    separate(destination, {streams}, true);

    const std::array<uint32_t, 4> shape{branches, channels, tokens, 0u};
    p.select("verify_fused_hc_write_rows", uint32_t(-1), shape, 128);
    p.bind(streams, 0); p.bind(input, 1); p.bind(scales, 2); p.bind(destination, 3, true); p.parameter(shape, 4);
    p.threads(tokens * ((width + 3u) / 4u));
}

void MetalGraph::verify_fused_hc_write_rms_rows(GpuView streams, GpuView input, GpuView scales, GpuView gains,
    GpuView destination, GpuView normalized, uint32_t branches, uint32_t channels, uint32_t tokens,
    float epsilon, bool shared_gains) {
    auto& p = *impl_; p.active();
    require(p.verify_hc_gdn_fused_prepared && !p.recording && !p.compatible_pipelines,
        "fused verify HC write+RMS requires prepared direct batches");
    const uint32_t width = checked_mul(branches, channels);
    require(branches >= 1 && branches <= 16 && channels >= 32 && channels <= 65536 && channels % 32 == 0 &&
        tokens >= 1 && tokens <= 64 && std::isfinite(epsilon) && epsilon > 0.0f,
        "fused verify HC write+RMS geometry mismatch");
    const uint64_t expected_gains = shared_gains ? channels : width;
    require(p.floats64(streams) == uint64_t(tokens) * width &&
        p.floats64(input) == uint64_t(tokens) * channels &&
        p.floats(scales) == checked_mul(tokens, branches) &&
        p.floats(gains) == expected_gains &&
        p.floats64(destination, true) == uint64_t(tokens) * width &&
        p.floats64(normalized, true) == uint64_t(tokens) * width,
        "fused verify HC write+RMS buffer span mismatch");
    separate(destination, {input, scales, gains, normalized});
    separate(destination, {streams}, true);
    separate(normalized, {streams, input, scales, gains, destination});

    const uint32_t packed_tokens = tokens | (shared_gains ? (1u << 16u) : 0u);
    const std::array<uint32_t, 4> shape{branches, channels, packed_tokens, std::bit_cast<uint32_t>(epsilon)};
    p.select("verify_fused_hc_write_rms_rows", uint32_t(-1), {branches, channels, tokens, shared_gains ? 1u : 0u}, 32);
    p.bind(streams, 0); p.bind(input, 1); p.bind(scales, 2); p.bind(gains, 3);
    p.bind(destination, 4, true); p.bind(normalized, 5, true); p.parameter(shape, 6);
    p.dispatch(true, MTLSizeMake(branches, tokens, 1), MTLSizeMake(32, 1, 1));
}

void MetalGraph::verify_fused_gdn_alpha_beta_f32(GpuView w_alpha, GpuView w_beta, GpuRowsF32 input,
    GpuRowsF32 out_alpha, GpuRowsF32 out_beta) {
    struct VerifyFusedGdnGatesF32Args {
        uint32_t rows, columns, outputs, half_groups;
        uint64_t input_stride_bytes, alpha_stride_bytes, beta_stride_bytes, reserved_stride;
    };
    static_assert(sizeof(VerifyFusedGdnGatesF32Args) == 48, "VerifyFusedGdnGatesF32Args ABI");
    auto& p = *impl_; p.active();
    require(p.verify_hc_gdn_fused_prepared && !p.recording && !p.compatible_pipelines,
        "fused verify GDN alpha/beta requires prepared direct batches");
    require(input.rows >= 1 && input.rows <= 64 && input.rows == out_alpha.rows && input.rows == out_beta.rows &&
        input.columns >= 32 && input.columns <= 65536 && input.columns % 32 == 0 &&
        out_alpha.columns >= 1 && out_alpha.columns <= 256 && out_alpha.columns == out_beta.columns &&
        input.stride_bytes >= uint64_t(input.columns) * 4ull && input.stride_bytes % 4 == 0 &&
        out_alpha.stride_bytes >= uint64_t(out_alpha.columns) * 4ull && out_alpha.stride_bytes % 4 == 0 &&
        out_beta.stride_bytes >= uint64_t(out_beta.columns) * 4ull && out_beta.stride_bytes % 4 == 0,
        "fused verify GDN alpha/beta geometry mismatch");
    require(!p.planar_shape(w_alpha) && !p.planar_shape(w_beta), "fused verify GDN alpha/beta refuses planar weights");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(!p.lane_shape(w_alpha) && !p.lane_shape(w_beta), "fused verify GDN alpha/beta requires FP32 row-major weights");
#endif
    const uint64_t w_floats = checked_mul(input.columns, out_alpha.columns);
    const uint64_t in_bytes = checked_add(checked_mul(uint64_t(input.rows - 1), input.stride_bytes), uint64_t(input.columns) * 4ull);
    const uint64_t a_bytes = checked_add(checked_mul(uint64_t(out_alpha.rows - 1), out_alpha.stride_bytes), uint64_t(out_alpha.columns) * 4ull);
    const uint64_t b_bytes = checked_add(checked_mul(uint64_t(out_beta.rows - 1), out_beta.stride_bytes), uint64_t(out_beta.columns) * 4ull);
    require(p.floats(w_alpha) == w_floats && p.floats(w_beta) == w_floats, "fused verify GDN alpha/beta weight span mismatch");
    static_cast<void>(p.allocation(input.view));
    static_cast<void>(p.allocation(out_alpha.view, true));
    static_cast<void>(p.allocation(out_beta.view, true));
    require(input.view.bytes >= in_bytes && out_alpha.view.bytes >= a_bytes && out_beta.view.bytes >= b_bytes &&
        input.view.offset % 4 == 0 && out_alpha.view.offset % 4 == 0 && out_beta.view.offset % 4 == 0,
        "fused verify GDN alpha/beta buffer span or alignment mismatch");
    validate_disjoint(std::array{w_alpha, w_beta, input.view}, std::array{out_alpha.view, out_beta.view});
    validate_disjoint(std::array{out_alpha.view}, std::array{out_beta.view});

    const uint32_t half_groups = (out_alpha.columns + 3u) / 4u;
    const VerifyFusedGdnGatesF32Args args{
        input.rows, input.columns, out_alpha.columns, half_groups,
        input.stride_bytes, out_alpha.stride_bytes, out_beta.stride_bytes, 0ull
    };
    p.select("verify_fused_gdn_alpha_beta_f32", uint32_t(-1), {input.rows, input.columns, out_alpha.columns, 2u}, 128);
    p.bind(w_alpha, 0); p.bind(w_beta, 1); p.bind(input.view, 2);
    p.bind(out_alpha.view, 3, true); p.bind(out_beta.view, 4, true); p.parameter(args, 5);
    p.groups(MTLSizeMake(2u * half_groups, (input.rows + 3u) / 4u, 1));
}

} // namespace tokspur
