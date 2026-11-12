#if defined(TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL) && defined(TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK) && !defined(TOKSPUR_PRIVATE_DIAGNOSTIC_BATCH_CLOCK)
#error The closed scalar-channel target must not emit batch-phase-clock metadata.
#endif
#if defined(TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL) && !defined(TOKSPUR_PRIVATE_KNOWN_QSA_ROWS)
#error Sampled input8 requires the existing exact known-row stack composition.
#endif

#if defined(TOKSPUR_PRIVATE_TASK4_PROFILE) && !defined(TOKSPUR_PRIVATE_LOGIT_TRACE)
#error "Task4 private target wiring requires the existing full-vector trace component"
#endif
#include "tokspur/carrier_phase.hpp"
#include "tokspur/carrier_cost_replay.hpp"
#include "tokspur/target.hpp"
#include "tokspur/gdn_tile.hpp"
#include "tokspur/gdn_core_tile.hpp"
#include <exception>
#include "tokspur/worker_identity.hpp"
#include "tokspur/stack.hpp"
#include "tokspur/shared_prefix_restore.hpp"
#include "tokspur/memory_guard.hpp"
#include "tokspur/shared_prefix_eviction.hpp"
#if defined(TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM) && TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM
#include <malloc/malloc.h>
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#include "tokspur/raw_decode_wire.hpp"
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
#include "tokspur/grouped_mm.hpp"
#endif
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
#include "tokspur/mtp_wire.hpp"
#if !defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) || !defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
#error "HC down batching target integration requires the explicit FP32 HTTP input512 build"
#endif
#endif
#if defined(TOKSPUR_PRIVATE_GDN_LEGACY_BATCH) && (!defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) || !defined(TOKSPUR_PRIVATE_GROUPED_MM_F32))
#error "old-order GDN target integration requires the explicit FP32 HTTP input512 build"
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
#include "tokspur/gdn_chunk.hpp"
#endif
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"
#include "tokspur/derived_pack.hpp"
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/sampling.hpp"
#include "tokspur/prepared_sampling.hpp"
#include "tokspur/prompt_lookup.hpp"
#if defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH) && !defined(TOKSPUR_PRIVATE_HC_EXACT)
#error The private concurrent-dispatch target requires the exact-HC diagnostic build.
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
#include "tokspur/lane_sidecar.hpp"
#ifndef TOKSPUR_PRIVATE_NATIVE_MTP
#error Lane target imports require the isolated auxiliary-import capable build.
#endif
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#include "tokspur/mtp.hpp"
#include "tokspur/mtp_head.hpp"
#include "tokspur/mtp_sampling.hpp"
#include "tokspur/mtp_anchor_state.hpp"
#include "tokspur/dflash2_drafter.hpp"
#include "tokspur/dflash2_checkpoint.hpp"
#include "tokspur/checkpoint_drafter.hpp"
#include "tokspur/sampled_path.hpp"
#include "tokspur/speculative_controls.hpp"
#include "tokspur/ssd_checkpoint.hpp"
#include <libproc.h>
#endif
#if defined(TOKSPUR_PRIVATE_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
#include "tokspur/verification_cost_manifest.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
#include "tokspur/sampled_cost_hash.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
#include "tokspur/logit_trace.hpp"
#endif
#if defined(TOKSPUR_PRIVATE_HC_EXACT) || defined(TOKSPUR_PRIVATE_NATIVE_MTP)
#include <CommonCrypto/CommonDigest.h>
#include <bit>
#endif

#include <libproc.h>
#include <mach/mach.h>
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#if defined(TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE) || defined(TOKSPUR_PRIVATE_TASK4_PROFILE) || defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) || defined(TOKSPUR_PRIVATE_NATIVE_MTP)
#include <fcntl.h>
#include <sys/stat.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <tuple>
#include <type_traits>
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#include <iomanip>
#include <sstream>
#if !defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) || !defined(TOKSPUR_PRIVATE_LOGIT_TRACE) || !defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
#error "operator checkpoints require the explicit input512, native MTP2 and private full-vector trace carrier"
#endif
#endif

#ifndef TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN
#define TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN 0
#endif
#if TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN != 0 && TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN != 1
#error "TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN must be 0 or 1"
#endif
#ifndef TOKSPUR_PRIVATE_PLE_CACHE_MIB
#define TOKSPUR_PRIVATE_PLE_CACHE_MIB 0
#endif
#if TOKSPUR_PRIVATE_PLE_CACHE_MIB != 0 && TOKSPUR_PRIVATE_PLE_CACHE_MIB != 256 && TOKSPUR_PRIVATE_PLE_CACHE_MIB != 512
#error "TOKSPUR_PRIVATE_PLE_CACHE_MIB must be 0 (automatic), 256 or 512"
#endif

namespace tokspur {
namespace {
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
[[gnu::used]] const char carrier_diagnostic_build_marker[] = "tokspur-carrier-diagnostic-v1";
#endif
constexpr uint64_t gib = uint64_t(1024) * 1024 * 1024, native_limit = 96 * gib;
constexpr uint64_t cache_budget = TOKSPUR_PRIVATE_PLE_CACHE_MIB
    ? uint64_t(TOKSPUR_PRIVATE_PLE_CACHE_MIB) * 1024 * 1024
    : (TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN ? gib / 4 : gib / 2);
constexpr uint64_t gpu_budget = native_limit - gib / 2 - 2 * gib;
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
Json number(double v) {
    require(std::isfinite(v), "nonfinite target diagnostic"); std::array<char, 64> data{};
    const auto r = std::to_chars(data.data(), data.data() + data.size(), v); require(r.ec == std::errc{}, "target JSON number encoding failed");
    return parse_json({data.data(), size_t(r.ptr - data.data())});
}
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
Json verify_nax_batch_metadata(Json value, const MetalGraph& gpu) {
    const auto* event = value.find("event");
    if (!event || (event->string() != "ready" && event->string() != "done")) return value;
    auto fields = value.object();
    const auto t = gpu.verify_nax_top10_mixture_telemetry();
    fields.emplace_back("native_verify_nax_batch_top10_mixture_enabled", Json(verify_nax_batch_top10_mixture_enabled));
    fields.emplace_back("native_verify_nax_ops_counter_scope", Json("worker-lifetime completed verification operator encodings; includes subsequently aborted submissions; excludes prefill and drafter"));
    fields.emplace_back("native_verify_top10_batched_calls", Json::integer(t.top10_batched_calls));
    fields.emplace_back("native_verify_top10_batched_rows", Json::integer(t.top10_batched_rows));
    fields.emplace_back("native_verify_top10_scalar_calls", Json::integer(t.top10_scalar_calls));
    fields.emplace_back("native_verify_mixture_batched_calls", Json::integer(t.mixture_batched_calls));
    fields.emplace_back("native_verify_mixture_batched_rows", Json::integer(t.mixture_batched_rows));
    fields.emplace_back("native_verify_mixture_scalar_calls", Json::integer(t.mixture_scalar_calls));
    return Json(std::move(fields));
}
#endif
Json cache_memory_metadata(Json value, const MetalGraph& gpu) {
    const auto* event = value.find("event");
    if (!event || (event->string() != "ready" && event->string() != "done")) return value;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    value = verify_nax_batch_metadata(std::move(value), gpu);
#endif
    auto fields = value.object();
    // The selected limit remains stable if pressure reclamation releases the cache.
    fields.emplace_back("host_row_cache_budget_bytes", Json::integer(cache_budget));
    const auto tile64 = gpu.q8_tile64_cache_stats();
    fields.emplace_back("q8_tile64_cache_enabled", Json(tile64.enabled));
    fields.emplace_back("q8_tile64_registered_candidates", Json::integer(tile64.registered_candidates));
    fields.emplace_back("q8_tile64_registered_source_bytes", Json::integer(tile64.registered_source_bytes));
    fields.emplace_back("q8_tile64_candidate_repack_logical_bytes", Json::integer(tile64.candidate_repack_logical_bytes));
    fields.emplace_back("q8_tile64_covered_candidates", Json::integer(tile64.covered_candidates));
    fields.emplace_back("q8_tile64_omitted_candidates", Json::integer(tile64.omitted_candidates));
    fields.emplace_back("q8_tile64_omitted_repack_logical_bytes", Json::integer(tile64.omitted_repack_logical_bytes));
    fields.emplace_back("q8_tile64_owning_allocations", Json::integer(tile64.owning_allocations));
    fields.emplace_back("q8_tile64_owned_logical_bytes", Json::integer(tile64.owned_logical_bytes));
    fields.emplace_back("q8_tile64_owned_charged_bytes", Json::integer(tile64.owned_charged_bytes));
    fields.emplace_back("q8_tile64_aliases", Json::integer(tile64.aliases));
    fields.emplace_back("q8_tile64_candidate_scope", Json("raw registered matrices; sums may overlap; omissions checked per source key; allocation bytes count distinct live owners only"));
    return Json(std::move(fields));
}
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
Json ple_gate_metadata(Json value, std::optional<PleGateTransform> transform) {
    if (!transform) return value;
    const auto* event = value.find("event");
    if (!event || (event->string() != "ready" && event->string() != "done")) return value;
    auto fields = value.object();
    fields.emplace_back("ple_gate_transform", Json(*transform == PleGateTransform::plain ? "plain" : "signed-root"));
    fields.emplace_back("ple_gate_transform_schema", Json::integer(1));
    return Json(std::move(fields));
}
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
Json qsa_channel_metadata(Json value, const MetalGraph& gpu) {
    const auto policy = gpu.qsa_channel_policy();
    if (!policy.prepared) return value; // Absent selector preserves every old event schema.
    const auto* event = value.find("event");
    if (!event || (event->string() != "ready" && event->string() != "done")) return value;
    const auto calls = gpu.qsa_channel_calls();
    auto fields = value.object();
    fields.emplace_back("qsa_channel_chronological_mode", Json(policy.enabled ? "on" : "off"));
    fields.emplace_back("qsa_channel_chronological_schema", Json::integer(1));
    fields.emplace_back("qsa_channel_factor_logical_bytes", Json::integer(policy.factor_logical_bytes));
    fields.emplace_back("qsa_channel_factor_charged_bytes", Json::integer(policy.factor_charged_bytes));
    fields.emplace_back("qsa_channel_calls", Json::integer(calls));
    // Stage's original +1/two-pass accounting remains unchanged. This distinct
    // +1/call counts the third pass; total attention physical =2*Stage+channel.
    fields.emplace_back("qsa_channel_extra_dispatches", Json::integer(calls));
    return Json(std::move(fields));
}
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
[[maybe_unused]] Json batch_phase_metadata(Json value, const BatchPhaseClockState& state) {
    if (!state.selected) return value; // Absent selector preserves the old wire.
    const auto* event = value.find("event");
    if (!event || (event->string() != "ready" && event->string() != "done")) return value;
    auto fields = value.object();
    fields.emplace_back("batch_phase_clock_mode", Json(state.enabled ? "on" : "off"));
    fields.emplace_back("batch_phase_clock_schema", Json::integer(1));
    fields.emplace_back("batch_phase_clock_batches", Json::integer(state.successful_batches));
    fields.emplace_back("batch_phase_clock_records", Json::integer(state.records));
    fields.emplace_back("batch_phase_clock_failures", Json::integer(state.failures));
    return Json(std::move(fields));
}
#endif
const Json& field(const Json& object, std::string_view key) { const auto* p = object.find(key); if (!p) throw FormatError("missing explicit target policy/request field: " + std::string(key)); return *p; }
uint32_t choice(const Json& object, std::string_view key, std::initializer_list<std::string_view> values) {
    const auto value = field(object, key).string(); uint32_t i = 0;
    for (auto option : values) { if (value == option) return i; ++i; }
    throw FormatError("unsupported target policy: " + std::string(key));
}
void keys(const Json& object, const std::set<std::string_view>& allowed) {
    for (const auto& [name, value] : object.object()) { static_cast<void>(value); require(allowed.contains(name), "unknown target policy/request field"); }
}
Json policy_file(const std::filesystem::path& path) {
    MappedFile input(path); require(input.size() && input.size() <= 65536, "target policy file size limit");
    const auto bytes = input.bytes(); auto out = parse_json({reinterpret_cast<const char*>(bytes.data()), bytes.size()}, {65536, 256, 8});
    input.assert_unchanged(); keys(out, {"router", "normalize_selected", "query_scale", "l2_epsilon", "l2_mode", "query_packing", "rotary_pairs", "hash_order", "boundary", "image_as_eos", "dilation"}); return out;
}
StackPolicy policy(const Json& data, float epsilon) {
    const auto text = field(data, "l2_epsilon").number_lexeme(); float l2 = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), l2);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(l2) && l2 > 0 && l2 <= 0.01f, "invalid target L2 epsilon");
    return {{choice(data, "query_scale", {"unit", "inverse-sqrt-key-width"}) ? 1 / std::sqrt(128.0f) : 1, l2,
        L2Mode(choice(data, "l2_mode", {"add", "floor"})), epsilon},
        {QueryPacking(choice(data, "query_packing", {"head", "flat"})), RotaryPairs(choice(data, "rotary_pairs", {"split", "adjacent"})), epsilon},
        RouterMode(choice(data, "router", {"softmax", "sigmoid"})), field(data, "normalize_selected").boolean(),
        {HashOrder(choice(data, "hash_order", {"newest", "oldest"})), NgramBoundary(choice(data, "boundary", {"literal", "before", "after"})), field(data, "image_as_eos").boolean()}};
}
uint32_t dilation(const Json& data) {
    const auto value = field(data, "dilation").unsigned_integer(); require(value == 1 || value == 3, "target dilation must be explicitly 1 or 3"); return uint32_t(value);
}
const char* execution_name(StackExecution execution) {
    static_assert(uint32_t(StackExecution::direct) == 0 && uint32_t(StackExecution::indirect_compatible) == 1 &&
        uint32_t(StackExecution::recurrent_replay) == 2, "CPU request wire identities must match stack execution modes");
    switch (execution) {
        case StackExecution::direct: return "direct";
        case StackExecution::indirect_compatible: return "indirect-compatible";
        case StackExecution::recurrent_replay: return "recurrent-replay";
    }
    throw FormatError("unsupported target stack execution mode");
}
uint64_t sysctl_value(const char* name) {
    uint64_t value = 0; size_t size = sizeof(value);
    require(sysctlbyname(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value), "cannot inspect target host memory"); return value;
}
struct Memory { uint64_t current, peak, free_inactive, free_file, swap; uint32_t pressure; uint64_t resident_size, pageins, compressed; };
Memory memory() {
    rusage_info_v4 info{};
    require(proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&info)) == 0, "cannot inspect own physical footprint");
    vm_statistics64_data_t stats{}; mach_msg_type_number_t count = HOST_VM_INFO64_COUNT; vm_size_t page = 0;
    const auto host = mach_host_self(); const auto page_result = host_page_size(host, &page);
    const auto stats_result = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
    mach_port_deallocate(mach_task_self(), host);
    require(page_result == KERN_SUCCESS && stats_result == KERN_SUCCESS && page, "cannot inspect host VM counters");
    xsw_usage swap{}; size_t swap_size = sizeof(swap); uint32_t pressure = 0; size_t pressure_size = sizeof(pressure);
    require(sysctlbyname("vm.swapusage", &swap, &swap_size, nullptr, 0) == 0 && swap_size == sizeof(swap) &&
        sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &pressure_size, nullptr, 0) == 0 && pressure_size == sizeof(pressure), "cannot inspect host pressure/swap");
    task_vm_info_data_t vm_info{}; mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT; uint64_t compressed = 0;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        compressed = vm_info.compressed;
    }
    // Apple declares speculative pages already part of free_count. Do not add
    // them twice. Inactive pages are a coarse reclaimability hint, not a guarantee.
    return {info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint,
        checked_mul(uint64_t(stats.free_count) + stats.inactive_count, page),
        checked_mul(uint64_t(stats.free_count - std::min(stats.free_count, stats.speculative_count)) + stats.external_page_count, page),
        swap.xsu_used, pressure, info.ri_resident_size, info.ri_pageins, compressed};
}
Json describe(Memory m) { return Json(Json::Object{{"physical_footprint_bytes", Json::integer(m.current)}, {"lifetime_peak_footprint_bytes", Json::integer(m.peak)},
    {"free_plus_inactive_hint_bytes", Json::integer(m.free_inactive)}, {"free_plus_file_backed_hint_bytes", Json::integer(m.free_file)},
    {"swap_used_bytes", Json::integer(m.swap)}, {"pressure_level", Json::integer(m.pressure)},
    {"resident_size_bytes", Json::integer(m.resident_size)}, {"pageins", Json::integer(m.pageins)},
    {"compressed_bytes", Json::integer(m.compressed)}}); }
void own_memory_guard() {
    const auto m = memory();
    // Lifetime peak remains diagnostic because releasing memory cannot lower it.
    if (!memory_work_admitted({m.current, m.pressure}, native_limit))
        throw FormatError("target process/host memory guard refused further work: " + describe(m).dump());
}
void other_workers() {
    // proc_name() yields nothing for processes owned by other users, so a root or
    // other-user worker would be invisible. kinfo_proc names are readable for all.
    int selector[] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL};
    std::vector<kinfo_proc> processes;
    for (int attempt = 0;; ++attempt) {
        size_t bytes = 0;
        require(sysctl(selector, 3, nullptr, &bytes, nullptr, 0) == 0 && bytes >= sizeof(kinfo_proc), "cannot enumerate inference-worker exclusion set");
        processes.assign(bytes / sizeof(kinfo_proc) + 64, kinfo_proc{});
        bytes = processes.size() * sizeof(kinfo_proc);
        if (sysctl(selector, 3, processes.data(), &bytes, nullptr, 0) == 0) { processes.resize(bytes / sizeof(kinfo_proc)); break; }
        require(errno == ENOMEM && attempt < 8, "cannot enumerate inference-worker exclusion set");
    }
    const std::set<std::string_view> known{"splash", "llama-server", "llama-cli", "ollama", "omlx-server", "tokspur-native"
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        , "hc-exact-target", "qsa-stage", "qsa-real-probe", "q8-exact-bench"
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        , "branch8-target"
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        , "stage8-target"
#endif
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        , "wide-q8-target", "q8-wide-bench"
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        , "q8-many-target", "q8-wide-many", "moe-tile-target"
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        , "verify8-target", "q8-many-target", "moe-tile-target"
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , "route8-target"
#endif
    };
    for (const auto& process : processes) {
        const pid_t pid = process.kp_proc.p_pid;
        if (!live_worker_candidate(pid, getpid(), process.kp_proc.p_stat == SZOMB)) continue;
        std::array<char, MAXCOMLEN + 1> name{}; std::memcpy(name.data(), process.kp_proc.p_comm, MAXCOMLEN);
        if (known.contains(name.data()) || scalar_kernel_worker_name(name.data(), MAXCOMLEN))
            throw FormatError("known inference worker is still running: " + std::string(name.data()) + " (" + std::to_string(pid) + ")");
    }
}
void admit(const FlashLayout& layout) {
    other_workers(); require(sysctl_value("hw.memsize") >= 128 * gib, "full target requires at least 128 GiB unified memory");
    const auto m = memory(); const uint64_t estimate = checked_add(layout.resident_matrix_bytes, 5 * gib);
    // Read-only file cache can be active, not just inactive. Do not count
    // anonymous inactive pages as freely disposable or double-count speculative
    // file cache. This remains a hint; the unchanged pressure/footprint guard and
    // external headroom/ownership check apply as well.
    if (m.pressure != 1 || m.free_file < checked_add(estimate, 8 * gib))
        throw FormatError("insufficient host headroom for complete target: " + describe(m).dump());
}
bool read_line(std::istream& input, std::string& out) {
    out.clear(); bool any = false, too_long = false; char c;
    while (input.get(c)) { any = true; if (c == '\n') break; if (out.size() < 1024 * 1024) out.push_back(c); else too_long = true; }
    require(!too_long, "target request exceeds 1-MiB bound"); return any;
}
// BEGIN owned target protocol: this CPU-only region is checked independently.
// The retained value is published only after a complete stack step. A failed
// request may have committed earlier steps; never label old logits as its prefix.
template<class Result> class CommittedResult {
    std::optional<Result> value_;
    uint32_t prefix_ = 0;
    std::optional<uint32_t> stop_;
    bool logits_available_ = false;
public:
    [[nodiscard]] bool matches(uint32_t prefix) const noexcept { return value_ && logits_available_ && prefix && prefix_ == prefix; }
    [[nodiscard]] bool continuable(uint32_t prefix) const noexcept { return matches(prefix) && !stop_; }
    [[nodiscard]] std::optional<uint32_t> stop_id() const noexcept { return stop_; }
    [[nodiscard]] const Result& get(uint32_t prefix) const { require(matches(prefix), "no logits for the committed target prefix"); return *value_; }
    void clear() noexcept { value_.reset(); prefix_ = 0; stop_.reset(); logits_available_ = false; }
    void commit(Result&& value, uint32_t prefix, bool logits_available = true) noexcept {
        static_assert(std::is_nothrow_move_constructible_v<Result> && std::is_nothrow_move_assignable_v<Result>);
        value_ = std::move(value); prefix_ = prefix; stop_.reset(); logits_available_ = logits_available;
    }
    void stopped(uint32_t id) noexcept { stop_ = id; }
    void exchange(std::optional<Result>& prepared, uint32_t prefix, std::optional<uint32_t> stop) noexcept {
        static_assert(std::is_nothrow_swappable_v<Result>);
        value_.swap(prepared); prefix_ = prefix; stop_ = stop; logits_available_ = true;
    }
};
struct TargetSampling { SamplingPolicy policy; uint64_t seed = 0; };
double sampling_number(const Json& value, double minimum, double maximum, bool exclude_minimum = false) {
    const auto text = value.number_lexeme(); double out = 0;
    require(text.size() <= 64, "sampling number lexeme exceeds limit");
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), out);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(out) &&
        (exclude_minimum ? out > minimum : out >= minimum) && out <= maximum, "invalid target sampling number");
    return out;
}
TargetSampling target_sampling(const Json& data, bool reset) {
    TargetSampling out;
    const auto* parameters = data.find("sampling"); const auto* seed = data.find("seed");
    require(reset || (!parameters && !seed), "sampling policy and seed are reset-only; append inherits its exact entropy stream");
    if (parameters) {
        keys(*parameters, {"temperature", "top_k", "top_p", "min_p"});
        out.policy.temperature = sampling_number(field(*parameters, "temperature"), 0, 100);
        const auto k = field(*parameters, "top_k").unsigned_integer();
        require(k <= 248320, "target sampling top-k exceeds vocabulary"); out.policy.top_k = uint32_t(k);
        out.policy.top_p = sampling_number(field(*parameters, "top_p"), 0, 1, true);
        out.policy.min_p = sampling_number(field(*parameters, "min_p"), 0, 1);
    }
    if (seed) out.seed = seed->unsigned_integer();
    require(out.policy.temperature == 0 || seed, "stochastic reset requires an explicit uint64 seed");
    return out;
}

// Entropy belongs to this serial generation session, not to reusable model
// state. New input and zero-work queries retain the exact stream; only a
// validated, successfully committed reset installs a fresh policy/seed.
class TargetSampler {
    TargetSampling options_;
    SamplingRng random_;
    uint32_t prefix_ = 0;
    bool valid_ = true;
public:
    explicit TargetSampler(TargetSampling options = {}) : options_(options), random_(options.seed) {}
    static TargetSampler fresh_branch(TargetSampling options, uint32_t prefix) {
        require(options.policy.temperature == 0 && prefix && prefix <= 65536,
            "branch restore requires greedy policy and nonempty bounded prefix");
        TargetSampler out(options); out.prefix_ = prefix; return out;
    }
    static TargetSampler restore_branch(TargetSampling options, uint32_t prefix, const SamplingRng& rng) {
        require(prefix && prefix <= 65536,
            "branch restore requires nonempty bounded prefix");
        TargetSampler out(options); out.prefix_ = prefix; out.random_ = rng; return out;
    }
    static TargetSampler restore_stochastic_branch(TargetSampling options, uint32_t prefix, const SamplingRng& rng) {
        return restore_branch(options, prefix, rng);
    }
    [[nodiscard]] bool matches(uint32_t prefix) const noexcept { return valid_ && prefix_ == prefix; }
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] uint32_t prefix() const noexcept { return prefix_; }
    [[nodiscard]] const TargetSampling& options() const noexcept { return options_; }
    [[nodiscard]] const SamplingRng& random() const noexcept { return random_; }
    void invalidate() noexcept { valid_ = false; }
    bool committed_input(uint32_t prefix) noexcept {
        if (!valid_ || prefix_ >= 65536 || prefix != prefix_ + 1) { invalidate(); return false; }
        prefix_ = prefix; return true;
    }
    void publish(SamplingRng&& random) noexcept {
        static_assert(std::is_nothrow_move_assignable_v<SamplingRng>);
        random_ = std::move(random);
    }
};

template<class Result> void reconcile_failure(CommittedResult<Result>& retained, TargetSampler& sampler,
                                            uint32_t position, bool fatal) noexcept {
    if (fatal || !sampler.matches(position) || (position && !retained.matches(position))) {
        sampler.invalidate(); retained.clear();
    }
}

template<class Result, class Reset> void reset_session(CommittedResult<Result>& retained, TargetSampler& sampler,
                                                      TargetSampling options, Reset&& reset) {
    TargetSampler replacement(options); // Construct before any model mutation.
    reset(); // Refusal preserves the old policy, engine and retained logits.
    retained.clear();
    static_assert(std::is_nothrow_move_assignable_v<TargetSampler>);
    sampler = std::move(replacement);
}

template<class Commit> uint32_t sampled_step(TargetSampler& sampler, std::span<const float> logits,
                                           uint32_t prefix, Commit&& commit) {
    require(sampler.matches(prefix), "sampling entropy does not match the committed prefix; reset required");
    auto tentative = sampler.random();
    uint32_t id;
    if (sampler.options().policy.temperature == 0) {
        // Preserve the old greedy selection and avoid allocating a dense
        // distribution on the default path. Ties still choose the lowest ID.
        require(!logits.empty() && logits.size() <= maximum_sampling_vocabulary, "target sampling vocabulary limit");
        for (auto value : logits) require(std::isfinite(value), "nonfinite target sampling logit");
        id = uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    } else id = sample(sampling_distribution(logits, sampler.options().policy), tentative);
    try { commit(id); }
    catch (...) {
        // A callback that publishes model/prefix state then throws cannot
        // leave the old entropy stream reusable. An actual stack commit that
        // throws BEFORE prefix publication is caught by the outer mismatch
        // check against stack.position(). No automatic generation retry.
        if (!sampler.matches(prefix)) sampler.invalidate();
        throw;
    }
    if (prefix == std::numeric_limits<uint32_t>::max() || !sampler.matches(prefix + 1)) {
        sampler.invalidate(); throw FormatError("sampled token commit did not publish the expected prefix");
    }
    // The callback returned after stack and retained-result publication. This
    // final entropy assignment is nonthrowing; later output/allocation errors
    // therefore cannot reuse a draw for an already committed token.
    sampler.publish(std::move(tentative));
    return id;
}

// CPU-only coordinator owns every native retain/restore call. Records mirror
// native first-free/oldest-generation eviction and strictly-later invalidation.
// Result's only required payload member is logits; other fields are restored
// from a default value, never reused as historical batch timing/accounting.
template<class Handle, class Result> class BranchSession {
    struct Record {
        Handle handle{};
        Result result{};
        std::optional<uint32_t> stop;
        uint64_t order = 0;
        uint32_t prefix = 0;
        bool valid = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        bool has_rng = false;
        std::array<uint8_t, 32> state_digest{};
        SamplingRng rng{};
        StageAControllerState controller_state{};
        uint32_t mtp_first = 1;
        uint32_t mtp_rows = 0;
        std::optional<SampledPathToken> carried_anchor{};
        uint32_t pending_overflow = 0;
#endif
    };
    std::vector<Record> records_;
    std::vector<uint32_t> history_;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::vector<uint32_t> partitions_;
    std::optional<uint32_t> chunk_mode_;
    std::array<uint32_t, 2> kernel_policy_{};
    std::array<uint8_t, 32> active_controller_policy_digest_{default_controller_policy_digest};
    bool measured_controller_policy_ = false;
    std::string numerical_policy_digest_;
#endif
    std::optional<Result> transfer_;
    uint64_t order_ = 0;
    uint32_t used_ = 0, vocabulary_;
    bool valid_ = true;
    void stage(std::span<const float> logits) {
        require(logits.size() == vocabulary_ && transfer_ && transfer_->logits.size() == vocabulary_, "branch logit storage mismatch");
        for (auto value : logits) require(std::isfinite(value), "nonfinite saved branch logits");
        Result clean{}; clean.logits.swap(transfer_->logits); *transfer_ = std::move(clean);
        std::copy(logits.begin(), logits.end(), transfer_->logits.begin());
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    [[nodiscard]] std::array<uint8_t, 32> compute_state_digest(uint32_t prefix) const {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        CC_SHA256_Update(&ctx, &prefix, sizeof(prefix));
        const uint8_t cm = chunk_mode_.has_value() ? static_cast<uint8_t>(*chunk_mode_) : 0xff;
        CC_SHA256_Update(&ctx, &cm, sizeof(cm));
        uint32_t accumulated = 0;
        for (const auto part : partitions_) {
            if (accumulated >= prefix) break;
            const uint32_t chunk = std::min(part, prefix - accumulated);
            if (chunk > 0) {
                CC_SHA256_Update(&ctx, &chunk, sizeof(chunk));
                accumulated += chunk;
            }
        }
        CC_SHA256_Update(&ctx, kernel_policy_.data(), sizeof(kernel_policy_));
        if (!numerical_policy_digest_.empty()) CC_SHA256_Update(&ctx, numerical_policy_digest_.data(), CC_LONG(numerical_policy_digest_.size()));
        CC_SHA256_Update(&ctx, legacy_checkpoint_identity_tag.data(), sizeof(legacy_checkpoint_identity_tag));
        std::array<uint8_t, 32> digest{};
        CC_SHA256_Final(digest.data(), &ctx);
        return digest;
    }
#endif
public:
    static uint64_t minimum_bytes(uint32_t slots, uint32_t vocabulary) {
        require(slots && slots <= 4 && vocabulary && vocabulary <= maximum_sampling_vocabulary, "branch session shape bound");
        return sizeof(BranchSession) + uint64_t(slots) * sizeof(Record) +
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            2ull * 65536ull * sizeof(uint32_t) +
#else
            65536ull * sizeof(uint32_t) +
#endif
            uint64_t(slots + 1) * vocabulary * sizeof(float);
    }
    BranchSession(uint32_t slots, uint32_t vocabulary) : vocabulary_(vocabulary) {
        static_assert(std::is_nothrow_swappable_v<Result> && std::is_nothrow_copy_assignable_v<Handle> &&
            std::is_nothrow_copy_constructible_v<Handle>);
        static_cast<void>(minimum_bytes(slots, vocabulary));
        records_.resize(slots); history_.resize(65536);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        partitions_.reserve(65536);
#endif
        transfer_.emplace(); transfer_->logits.resize(vocabulary);
        for (auto& record : records_) record.result.logits.resize(vocabulary);
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    void configure_policy_digests(std::array<uint32_t, 2> kp, std::string numerical = {}) {
        kernel_policy_ = kp;
        numerical_policy_digest_ = std::move(numerical);
#if defined(TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS) && TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS
        numerical_policy_digest_ += "verify-narrow-rows-experiment-v1";
#endif
    }
    void configure_controller_policy(const std::array<uint8_t, 32>& active, bool measured) {
        active_controller_policy_digest_ = active; measured_controller_policy_ = measured;
    }
    void record_partition(uint32_t count, std::optional<uint32_t> mode) {
        static_cast<void>(count);
        if (mode.has_value()) chunk_mode_ = mode;
    }
#endif
    [[nodiscard]] uint64_t charged_bytes() const noexcept {
        uint64_t bytes = sizeof(*this) + uint64_t(records_.capacity()) * sizeof(Record) + uint64_t(history_.capacity()) * sizeof(uint32_t) +
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            uint64_t(partitions_.capacity()) * sizeof(uint32_t) +
#endif
            uint64_t(transfer_->logits.capacity()) * sizeof(float);
        for (const auto& record : records_) bytes += uint64_t(record.result.logits.capacity()) * sizeof(float);
        return bytes;
    }
    [[nodiscard]] bool matches(uint32_t prefix) const noexcept { return valid_ && used_ == prefix; }
    [[nodiscard]] uint32_t retained_count() const noexcept {
        uint32_t count = 0; for (const auto& record : records_) count += record.valid; return count;
    }
    [[nodiscard]] size_t common_prefix(std::span<const uint32_t> prompt) const noexcept {
        if (!valid_) return 0;
        size_t common = 0; while (common < prompt.size() && common < used_ && history_[common] == prompt[common]) ++common;
        return common;
    }
    void reset() noexcept {
        used_ = 0; valid_ = true;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        partitions_.clear();
#endif
        for (auto& record : records_) record.valid = false;
    }
    void invalidate() noexcept { reset(); valid_ = false; }
    bool committed(uint32_t prefix, std::span<const uint32_t> tokens) noexcept {
        if (!valid_ || tokens.size() > history_.size() - used_ || prefix != used_ + tokens.size()) { invalidate(); return false; }
        std::copy(tokens.begin(), tokens.end(), history_.begin() + used_);
        used_ = prefix;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        partitions_.push_back(uint32_t(tokens.size()));
#endif
        return true;
    }
    [[nodiscard]] std::optional<size_t> select(std::span<const uint32_t> prompt, uint32_t generate) const {
        if (!valid_) return {};
        const auto common = common_prefix(prompt);
        std::optional<size_t> selected;
        for (size_t i = 0; i < records_.size(); ++i) if (records_[i].valid && records_[i].prefix <= common &&
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            records_[i].state_digest == compute_state_digest(records_[i].prefix) &&
#endif
            (!selected || records_[i].prefix > records_[*selected].prefix ||
             (records_[i].prefix == records_[*selected].prefix && records_[i].order > records_[*selected].order))) selected = i;
        if (selected) require(!generate || records_[*selected].prefix < prompt.size() || !records_[*selected].stop,
            "restored terminal prefix requires new input before generation");
        return selected;
    }
    template<class Retain> void capture(uint32_t prefix, const CommittedResult<Result>& retained, Retain&& retain) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        capture(prefix, retained, nullptr, nullptr, 1, 0, std::nullopt, 0, std::forward<Retain>(retain));
#else
        require(matches(prefix) && prefix && retained.matches(prefix), "branch capture prefix/logits mismatch");
        require(order_ < std::numeric_limits<uint64_t>::max(), "branch capture order exhausted");
        stage(retained.get(prefix).logits);
        size_t index = 0;
        for (size_t i = 0; i < records_.size(); ++i) {
            if (!records_[i].valid) { index = i; break; }
            if (records_[i].order < records_[index].order) index = i;
        }
        const auto handle = retain();
        auto& record = records_[index]; std::swap(record.result, *transfer_);
        record.handle = handle; record.prefix = prefix; record.stop = retained.stop_id(); record.order = ++order_; record.valid = true;
#endif
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    template<class Retain> void capture(uint32_t prefix, const CommittedResult<Result>& retained,
                                         const TargetSampler* sampler,
                                         const StageAControllerState* controller_state,
                                         uint32_t mtp_first, uint32_t mtp_rows,
                                         std::optional<SampledPathToken> carried_anchor,
                                         uint32_t pending_overflow,
                                         Retain&& retain) {
        require(matches(prefix) && prefix && retained.matches(prefix), "branch capture prefix/logits mismatch");
        require(order_ < std::numeric_limits<uint64_t>::max(), "branch capture order exhausted");
        stage(retained.get(prefix).logits);
        size_t index = 0;
        for (size_t i = 0; i < records_.size(); ++i) {
            if (!records_[i].valid) { index = i; break; }
            if (records_[i].order < records_[index].order) index = i;
        }
        const auto handle = retain(); // Refusal preserves all saved records.
        auto& record = records_[index]; std::swap(record.result, *transfer_);
        record.handle = handle; record.prefix = prefix; record.stop = retained.stop_id(); record.order = ++order_; record.valid = true;
        record.state_digest = compute_state_digest(prefix);
        record.has_rng = (sampler != nullptr);
        if (sampler) record.rng = sampler->random();
        if (controller_state) record.controller_state = *controller_state;
        else record.controller_state = StageAControllerState{};
        record.mtp_first = mtp_first;
        record.mtp_rows = mtp_rows;
        record.carried_anchor = carried_anchor;
        record.pending_overflow = pending_overflow;
    }
#endif
    template<class Restore> uint32_t restore(size_t index, CommittedResult<Result>& retained, TargetSampler& sampler,
                                             TargetSampling options, Restore&& restore
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                                             , StageAControllerState* out_controller_state = nullptr,
                                             uint32_t* out_mtp_rows = nullptr,
                                             std::optional<SampledPathToken>* out_carried_anchor = nullptr,
                                             uint32_t* out_pending_overflow = nullptr
#endif
                                             ) {
        require(valid_ && index < records_.size() && records_[index].valid && retained.matches(used_) && sampler.matches(used_),
            "branch restoration requires coherent current/saved session state");
        const auto& record = records_[index]; const auto prefix = record.prefix;
        require(prefix <= used_, "branch restore is ancestor-only");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        const auto expected_digest = compute_state_digest(prefix);
        require(record.state_digest == expected_digest, "checkpoint state digest mismatch: partition history or chunk mode diverged");
#endif
        require(retained.get(used_).logits.size() == vocabulary_ &&
            retained.get(used_).logits.capacity() <= transfer_->logits.capacity(),
            "current logits cannot preserve bounded branch transfer storage");
        for (auto value : retained.get(used_).logits) require(std::isfinite(value), "nonfinite current branch logits");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        auto replacement = (record.has_rng && options.policy.temperature > 0 && options.seed == record.rng.seed())
            ? TargetSampler::restore_branch(options, prefix, record.rng)
            : TargetSampler::fresh_branch(options, prefix);
#else
        auto replacement = TargetSampler::fresh_branch(options, prefix);
#endif
        stage(record.result.logits);
        const auto actual = restore(record.handle);
        if (actual != prefix) {
            invalidate(); retained.clear(); sampler.invalidate(); throw FormatError("native restore published unexpected branch prefix; reset required");
        }
        // Every payload is already prepared. No allocation/fallible work after
        // native publication; transfer receives the previous current logits.
        retained.exchange(transfer_, prefix, record.stop); sampler = std::move(replacement); used_ = prefix;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (out_controller_state) *out_controller_state = restore_controller_policy(record.controller_state, active_controller_policy_digest_, measured_controller_policy_);
        if (out_mtp_rows) *out_mtp_rows = record.mtp_rows;
        if (out_carried_anchor) *out_carried_anchor = record.carried_anchor;
        if (out_pending_overflow) *out_pending_overflow = record.pending_overflow;
        uint32_t sum = 0; size_t keep = 0;
        for (size_t i = 0; i < partitions_.size(); ++i) {
            if (sum + partitions_[i] <= prefix) {
                sum += partitions_[i];
                keep = i + 1;
                if (sum == prefix) break;
            } else {
                partitions_[i] = prefix - sum;
                keep = i + 1;
                break;
            }
        }
        if (prefix == 0) keep = 0;
        partitions_.resize(keep);
#endif
        for (auto& other : records_) if (other.valid && other.prefix > prefix) other.valid = false;
        return prefix;
    }
};
// Fixed-depth proposal caller state. It owns only committed IDs and prepared
// host scratch; speculative IDs never enter history before native publication.
template<class Result> class TargetLookup {
    PromptLookup history_;
    std::optional<Result> spare_;
    bool valid_ = true;
    static LookupConfig config(uint32_t vocabulary, uint64_t budget
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        , uint32_t proposal_tokens
#endif
    ) {
        LookupConfig out; out.match_tokens = 16; out.proposal_tokens = 2;
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        require(proposal_tokens == 2 || proposal_tokens == 7, "private lookup proposal limit must be 2 or 7");
        out.proposal_tokens = proposal_tokens;
#endif
        out.history_tokens = 65536; out.token_limit = vocabulary; out.memory_budget_bytes = budget; return out;
    }
public:
    std::vector<float> rows, embeddings;
    TargetLookup(uint32_t vocabulary, uint32_t channels, uint32_t embedding, uint64_t budget
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        , uint32_t proposal_tokens = 2
#endif
    )
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        : history_(config(vocabulary, budget, proposal_tokens)), rows(uint64_t(proposal_tokens) * channels), embeddings(uint64_t(proposal_tokens) * embedding) {
#else
        : history_(config(vocabulary, budget)), rows(2ull * channels), embeddings(2ull * embedding) {
#endif
        spare_.emplace(); spare_->logits.resize(vocabulary);
        require(charged_bytes() < budget, "lookup/caller host preparation exceeds combined verification budget");
    }
    [[nodiscard]] uint64_t charged_bytes() const {
        return sizeof(*this) - sizeof(PromptLookup) + history_.charged_bytes() +
            (spare_->logits.capacity() + rows.capacity() + embeddings.capacity()) * sizeof(float);
    }
    [[nodiscard]] uint64_t epoch() const { return history_.epoch(); }
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    [[nodiscard]] uint32_t proposal_tokens() const { return history_.config().proposal_tokens; }
#endif
    [[nodiscard]] const LookupLedger& ledger() const { return history_.ledger(); }
    [[nodiscard]] bool matches(uint32_t prefix) const { return valid_ && history_.committed_length() == prefix; }
    void invalidate() noexcept { valid_ = false; }
    void reset() { static_cast<void>(history_.reset()); valid_ = true; }
    void restore(uint32_t prefix) {
        require(valid_ && prefix <= history_.committed_length(), "lookup restore requires matching committed ancestry");
        static_cast<void>(history_.truncate(history_.epoch(), prefix));
    }
    void committed(uint32_t prefix, std::span<const uint32_t> tokens) {
        require(tokens.size() <= prefix && matches(prefix - uint32_t(tokens.size())), "lookup committed prefix mismatch; reset required");
        history_.append(history_.epoch(), prefix - tokens.size(), tokens);
    }
    [[nodiscard]] LookupProposal propose(uint32_t prefix) {
        require(matches(prefix), "lookup history is stale; explicit reset required");
        return history_.propose(history_.epoch(), prefix);
    }
    static uint32_t greedy(std::span<const float> logits) {
        require(!logits.empty(), "lookup requires current retained logits");
        for (auto value : logits) require(std::isfinite(value), "lookup requires finite greedy logits");
        return uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    [[nodiscard]] bool eligible(const LookupProposal& proposal, uint32_t prefix, std::span<const float> logits,
                                uint32_t remaining, std::span<const uint32_t> stops) const {
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        require(matches(prefix) && proposal.epoch == epoch() && proposal.base_prefix == prefix && proposal.count <= proposal_tokens(),
            "stale lookup proposal epoch/base");
        const auto first = greedy(logits);
        return proposal.count >= 2 && remaining >= 2 && prefix <= 65534 && proposal.tokens[0] == first &&
            std::find(stops.begin(), stops.end(), first) == stops.end();
#else
        require(matches(prefix) && proposal.epoch == epoch() && proposal.base_prefix == prefix && proposal.count <= 2,
            "stale lookup proposal epoch/base");
        const auto first = greedy(logits);
        return proposal.count == 2 && remaining >= 2 && prefix <= 65534 && proposal.tokens[0] == first &&
            std::find(stops.begin(), stops.end(), first) == stops.end();
#endif
    }
    void prepare(const CommittedResult<Result>& retained, const TargetSampler& sampler, uint32_t prefix) {
        require(matches(prefix) && sampler.matches(prefix) && sampler.options().policy.temperature == 0 &&
            retained.continuable(prefix), "lookup verification requires exact current greedy session identity");
        const auto& current = retained.get(prefix).logits;
        require(current.size() == spare_->logits.size() && current.capacity() <= spare_->logits.capacity(),
            "retained logits exceed prepared lookup spare");
        Result clean{}; clean.logits.swap(spare_->logits); *spare_ = std::move(clean);
    }
    template<class Track> void publish(const LookupProposal& proposal, uint32_t accepted, uint32_t actual_prefix,
                                      std::span<const float> logits, CommittedResult<Result>& retained,
                                      TargetSampler& sampler, Track&& track) {
        try {
            require(matches(uint32_t(proposal.base_prefix)) && proposal.epoch == epoch() && accepted &&
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                accepted <= proposal_tokens() &&
#else
                accepted <= 2 &&
#endif
                accepted <= proposal.count && actual_prefix == proposal.base_prefix + accepted &&
                logits.size() == spare_->logits.size(), "unexpected verified lookup publication; reset required");
            for (auto value : logits) require(std::isfinite(value), "nonfinite verified retained logits");
            // All capacities were checked before GPU work. A float copy and
            // retained exchange cannot allocate; accounting failure revokes
            // this session rather than invisibly generating again.
            std::copy(logits.begin(), logits.end(), spare_->logits.begin());
            retained.exchange(spare_, actual_prefix, {});
            for (uint32_t i = 1; i <= accepted; ++i)
                require(sampler.committed_input(uint32_t(proposal.base_prefix) + i), "verified lookup entropy prefix mismatch");
            const auto tokens = proposal.proposed().first(accepted);
            committed(actual_prefix, tokens); track(actual_prefix, tokens);
        } catch (...) { invalidate(); retained.clear(); sampler.invalidate(); throw; }
    }
};
struct LookupRoundToken { uint32_t id = 0; float source_logit = 0; bool verified = false; };
struct LookupRound {
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    std::array<LookupRoundToken, 8> tokens{};
    uint32_t evaluated = 0, head_dispatches = 0;
#else
    std::array<LookupRoundToken, 2> tokens{};
#endif
    uint32_t count = 0, accepted = 0, proposed = 0, base = 0, committed = 0;
    bool verified = false;
};
// The same fixed-capacity bookkeeping is extracted into CPU tests. Verify owns
// gather/native verification; ordinary MUST evaluate its greedy correction.
template<class Result, class Position, class Verify, class Ordinary, class Track>
LookupRound lookup_round(TargetLookup<Result>& lookup, CommittedResult<Result>& retained, TargetSampler& sampler,
                         uint32_t remaining, std::span<const uint32_t> stops, Position&& position,
                         Verify&& verify, Ordinary&& ordinary, Track&& track) {
    LookupRound round; round.base = position();
    try {
        require(remaining && remaining <= 4096, "lookup round requires a bounded nonzero output budget");
        lookup.prepare(retained, sampler, round.base);
        const auto proposal = lookup.propose(round.base); round.proposed = proposal.count;
        if (lookup.eligible(proposal, round.base, retained.get(round.base).logits, remaining, stops)) {
            round.verified = true;
            const auto verified = verify(proposal.proposed(), round.base, retained.get(round.base).logits, remaining, stops);
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
            require(verified.evaluated <= proposal.count && verified.accepted <= verified.evaluated &&
                verified.head_dispatches <= 3, "verifier returned invalid active/head work");
            round.evaluated = verified.evaluated; round.head_dispatches = verified.head_dispatches;
            require(verified.position == round.base && verified.accepted <= proposal.count && position() == round.base + verified.accepted,
#else
            require(verified.position == round.base && verified.accepted <= 2 && position() == round.base + verified.accepted,
#endif
                "verifier returned unexpected committed lookup prefix");
            round.accepted = verified.accepted;
            if (round.accepted) {
                for (uint32_t i = 0; i < round.accepted; ++i)
                    require(std::isfinite(verified.source_logits[i]), "nonfinite verified source logit");
                lookup.publish(proposal, round.accepted, position(), verified.logits, retained, sampler, track);
                for (uint32_t i = 0; i < round.accepted; ++i) {
                    round.tokens[round.count++] = {proposal.tokens[i], verified.source_logits[i], true};
                }
            }
        }
        // k=0 uses ordinary decoding; k=1 evaluates a real correction from the
        // first-row retained logits. k=2 needs no extra target step.
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        const bool terminal = round.accepted && std::find(stops.begin(), stops.end(), proposal.tokens[round.accepted - 1]) != stops.end();
        if ((!round.evaluated || round.accepted < round.evaluated) && !terminal) round.tokens[round.count++] = ordinary();
#else
        if (round.accepted < 2) round.tokens[round.count++] = ordinary();
#endif
        round.committed = position();
        require(round.count && round.count <= remaining && round.committed == round.base + round.count &&
            lookup.matches(round.committed) && sampler.matches(round.committed) && retained.matches(round.committed),
            "lookup round bookkeeping disagrees with committed model state");
        for (uint32_t i = 0; i < round.count; ++i)
            if (std::find(stops.begin(), stops.end(), round.tokens[i].id) != stops.end()) {
                require(i + 1 == round.count, "lookup committed beyond first terminal token"); retained.stopped(round.tokens[i].id);
            }
        return round;
    } catch (...) { lookup.invalidate(); retained.clear(); sampler.invalidate(); throw; }
}
[[maybe_unused]] void lookup_preparation(bool enabled, uint64_t budget, uint32_t execution, bool profiling,
                                        uint32_t splits, uint32_t blocks, bool kernels, bool scheduling,
                                        bool prefill, bool broker, bool selectors = false
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                                        , bool bounded_branch_broker = false
#endif
                                        ) {
    require(bool(budget) == enabled && budget <= (128ull << 20), "lookup requires explicit paired verification budget at most128MiB");
    require(!selectors || enabled, "lookup request experiments require startup preparation");
    const bool transport = !broker
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        || bounded_branch_broker
#endif
        ;
    require(!enabled || (!execution && !profiling && splits == 16 && blocks == 3 && !kernels && !scheduling && !prefill && transport),
        "lookup requires fixed direct unprofiled split16/block3 without worker broker or request experiments");
}
[[maybe_unused]] void hc_q8_preparation(uint32_t splits, uint32_t execution, bool profiling, uint32_t blocks,
                                      bool kernels, bool scheduling, bool prefill, bool lookup, bool reuse) {
    require(splits == 0 || splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16 || splits == 32,
        "invalid HC Q8 split count");
    require(!splits || (!execution && !profiling && (blocks & 1) && !kernels && !scheduling && !prefill && !lookup && !reuse),
        "HC Q8 splits require fixed direct unprofiled block-Q8, no request experiments, prompt lookup or residual reuse");
}
[[maybe_unused]] const char* hc_q8_numerical_tier(uint32_t splits) {
    return !splits ? "ordinary" : splits == 1 ? "split1-exact-block-q8" : "reassociated-fp32";
}
// Separate zero-model-step control broker. Metadata is never forwarded to a
// model input loop. The only producer is the serial owned worker transport.
struct BrokerCommand {
    Json body;
    uint64_t command = 0, lease = 0;
    bool begin = false, capture = false, fresh = false;
};
class TargetBroker {
    uint64_t epoch_, command_ = 0, lease_ = 0, reply_lease_ = 0;
    bool revoked_ = true, input_boundary_ = false;
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    const bool batch8_;
#endif
public:
    explicit TargetBroker(uint64_t epoch
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , bool batch8 = false
#endif
        ) : epoch_(epoch)
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , batch8_(batch8)
#endif
        { require(epoch, "nonzero native broker epoch required"); }
    [[nodiscard]] uint64_t epoch() const noexcept { return epoch_; }
    [[nodiscard]] BrokerCommand admit(const Json& wire, uint32_t prefix) const {
        require(field(wire, "broker_epoch").unsigned_integer() == epoch_ &&
            command_ < std::numeric_limits<uint64_t>::max() && field(wire, "command_id").unsigned_integer() == command_ + 1 &&
            field(wire, "expected_prefix").unsigned_integer() == prefix, "stale broker epoch/command/expected prefix");
        Json payload_wire = wire;
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (batch8_) {
            require(field(wire, "broker_protocol").string() == "branch8-v1" &&
                field(wire, "broker_batch_width").unsigned_integer() == 8, "immutable branch8 wire identity changed");
            Json::Object fields;
            for (const auto& item : wire.object()) if (item.first != "broker_protocol" && item.first != "broker_batch_width") fields.push_back(item);
            payload_wire = Json(std::move(fields)); // Remove transport identity, never model-state metadata.
        }
#endif
        const auto& input_wire = payload_wire;
        BrokerCommand out; out.command = command_ + 1; out.lease = field(input_wire, "lease_id").unsigned_integer();
        if (const auto* op = input_wire.find("op")) {
            const auto name = op->string(); out.begin = name == "branch_begin"; out.capture = name == "branch_capture";
            require(out.begin || out.capture, "unknown broker control operation");
            if (out.begin) {
                keys(input_wire, {"broker_epoch", "command_id", "lease_id", "expected_prefix", "op", "prompt_ids", "generation_requested", "allow_restore", "sampling", "seed"});
                const auto& prompt = field(input_wire, "prompt_ids").array(); require(!prompt.empty() && prompt.size() <= 65536, "broker future prompt bound");
                for (const auto& token : prompt) require(token.unsigned_integer() < 248320, "broker future token bound");
                static_cast<void>(field(input_wire, "generation_requested").boolean());
                const auto restore = field(input_wire, "allow_restore").boolean();
                require(!revoked_ || !restore, "revoked broker requires explicit cold reset");
                const auto sampling = target_sampling(input_wire, true);
                require(sampling.policy.temperature == 0, "broker restore requires fresh greedy sampling");
                static_cast<void>(field(input_wire, "seed").unsigned_integer()); out.fresh = true;
            } else {
                keys(input_wire, {"broker_epoch", "command_id", "lease_id", "expected_prefix", "op"});
                require(!revoked_ && input_boundary_ && prefix, "broker capture requires a committed input boundary");
            }
        } else {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (batch8_) keys(input_wire, {"broker_epoch", "command_id", "lease_id", "expected_prefix", "tokens", "generate", "reset", "stop_ids"});
            else
#endif
            keys(input_wire, {"broker_epoch", "command_id", "lease_id", "expected_prefix", "tokens", "generate", "reset", "stop_ids", "sampling", "seed"});
            const auto& input = field(input_wire, "tokens").array(); const auto output = field(input_wire, "generate").unsigned_integer();
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (batch8_) require(input.size() <= 8 && output <= 8 && bool(input.size()) != bool(output), "branch8 broker requires <=8 inputs OR outputs");
            else
#endif
            require(input.size() <= 1 && output <= 1 && input.size() + output <= 1, "broker step is bounded to one model token");
            for (const auto& token : input) require(token.unsigned_integer() < 248320, "broker input token bound");
            out.fresh = field(input_wire, "reset").boolean();
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            require(!batch8_ || !out.fresh, "branch8 model packets cannot reset; use explicit branch_begin control");
#endif
            require(!revoked_ || out.fresh, "revoked broker requires explicit reset");
        }
        require(out.fresh ? out.lease > lease_ : lease_ && out.lease == lease_, "stale broker lease");
        Json::Object payload;
        for (const auto& [name, value] : input_wire.object())
            if (name != "broker_epoch" && name != "command_id" && name != "lease_id" && name != "expected_prefix") payload.emplace_back(name, value);
        out.body = Json(std::move(payload)); return out;
    }
    void started(const BrokerCommand& command) noexcept { command_ = command.command; reply_lease_ = command.lease; }
    void published(const BrokerCommand& command) noexcept { lease_ = command.lease; revoked_ = false; if (command.fresh) input_boundary_ = false; }
    void input() noexcept { input_boundary_ = true; }
    void generated() noexcept { input_boundary_ = false; }
    void captured() noexcept { input_boundary_ = false; }
    void revoke() noexcept { revoked_ = true; input_boundary_ = false; }
    Json decorate(Json value) const {
        auto fields = value.object();
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (batch8_ && field(value, "event").string() == "ready") {
            // Startup already publishes its broker epoch. Admit that exact
            // initial identity instead of emitting a duplicate JSON member.
            require(!command_ && !lease_ && !reply_lease_ && revoked_ && !input_boundary_ &&
                field(value, "checkpoint_broker").boolean() && field(value, "broker_epoch").unsigned_integer() == epoch_,
                "branch8 readiness requires its authoritative initial broker identity");
        } else
#endif
        fields.emplace_back("broker_epoch", Json::integer(epoch_));
        fields.emplace_back("command_id", Json::integer(command_)); fields.emplace_back("lease_id", Json::integer(reply_lease_));
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (batch8_) { fields.emplace_back("broker_protocol", Json("branch8-v1")); fields.emplace_back("broker_batch_width", Json::integer(8)); }
#endif
        return Json(std::move(fields));
    }
};
[[maybe_unused]] void branch_preparation(uint32_t slots, uint64_t budget, uint32_t execution, bool profiling,
                        bool kernels, bool scheduling, bool prefill) {
    require(slots <= 4 && budget <= (512ull << 20) && bool(slots) == bool(budget), "explicit paired branch slot/budget bound");
    require(!slots || (!execution && !profiling && !kernels && !scheduling && !prefill),
        "branch sessions require fixed direct unprofiled execution without request-time experiments");
}
[[maybe_unused]] void final_input_trace_preparation(bool final_input_logits, bool full_logit_trace) {
    require(!final_input_logits || !full_logit_trace, "final-input logits refuse full-logit tracing; each trace row requires computed logits");
}
[[maybe_unused]] void hc_down_f32_preparation(bool selected, bool input512, bool dense_f32, bool grouped_f32, bool fused, bool depth2_carry, bool overrides) {
    require(!selected || (input512 && dense_f32 && grouped_f32 && fused && depth2_carry && !overrides), "HC down batching requires fixed f32/f32 input512 fused MTP2");
}
[[maybe_unused]] void gdn_old_order_packet(bool selected, size_t inputs) {
    require(!selected || inputs <= 512, "old-order GDN packet exceeds fixed input512 bound");
}
struct Request { std::vector<uint32_t> tokens, stops; uint32_t generate, small_splits, block_formats, execution; bool reset, known_prefill, residual_reuse, branch, lookup; bool prefill_final = true, final_logits = false; TargetSampling sampling; };
Request request(const Json& data, uint32_t position, uint32_t current_splits, uint32_t current_block_formats, bool allow_kernel_experiments,
                bool has_logits, bool continuable, bool sampling_matches = true,
                uint32_t current_execution = 0, bool allow_stack_experiments = false, bool replay_prepared = false,
                uint32_t known_capacity = 0, bool allow_prefill_experiments = false, bool current_residual_reuse = false,
                uint32_t checkpoint_slots = 0, bool lookup_prepared = false, bool allow_lookup_experiments = false,
                bool current_lookup = false, bool final_input_logits = false) {
    keys(data, {"tokens", "generate", "reset", "stop_ids", "small_gemv_splits", "block_gemv_formats", "sampling", "seed", "stack_execution", "known_prefill", "known_residual_reuse", "checkpoint_mode", "prompt_lookup"
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , "mtp_generation"
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        , "prefill_final"
#endif
    }); Request out;
    const auto& ids = field(data, "tokens").array(); require(ids.size() <= 65536, "target input token count limit");
    out.tokens.reserve(ids.size());
    for (const auto& id : ids) { const auto token = id.unsigned_integer(); require(token < 248320, "target token outside vocabulary"); out.tokens.push_back(uint32_t(token)); }
    const auto gen = field(data, "generate").unsigned_integer(); require(gen <= 4096, "target generate limit is 4096"); out.generate = uint32_t(gen);
    out.final_logits = final_input_logits;
    const auto* final = data.find("prefill_final");
    require(bool(final) == (final_input_logits && !ids.empty()), "final-input flag presence differs from startup consent");
    if (final) { require(!gen, "final-input flag cannot accompany generation"); out.prefill_final = final->boolean(); }
    out.reset = field(data, "reset").boolean(); const auto prefix = out.reset ? 0 : position;
    out.sampling = target_sampling(data, out.reset);
    require(!current_lookup || lookup_prepared, "active lookup requires prepared capability");
    require(!allow_lookup_experiments || lookup_prepared, "lookup selectors require startup preparation");
    out.lookup = current_lookup;
    if (const auto* selected = data.find("prompt_lookup")) {
        require(lookup_prepared && allow_lookup_experiments && out.reset, "lookup selector requires explicit startup consent and reset");
        out.lookup = selected->boolean();
    }
    require(checkpoint_slots <= 4, "invalid prepared checkpoint capacity");
    out.branch = false;
    if (const auto* mode = data.find("checkpoint_mode")) {
        require(checkpoint_slots && checkpoint_slots <= 4 && out.reset && mode->string() == "branch"
#ifndef TOKSPUR_PRIVATE_NATIVE_MTP
            && out.sampling.policy.temperature == 0
#endif
            , "branch mode requires prepared slots, fresh full input and greedy sampling"); out.branch = true;
    }
    require(!checkpoint_slots || (!current_execution && !replay_prepared && !allow_kernel_experiments &&
        !allow_stack_experiments && !allow_prefill_experiments && !data.find("small_gemv_splits") &&
        !data.find("block_gemv_formats") && !data.find("stack_execution") && !data.find("known_prefill") && !data.find("known_residual_reuse")),
        "prepared branch sessions refuse request-time kernel/scheduling/reuse changes");
    require(out.reset || sampling_matches, "invalidated sampling session requires reset before input or generation");
    require(!ids.empty() || (!out.reset && position && has_logits), "empty target input requires retained logits for a nonempty committed prefix");
    require(!ids.empty() || !out.generate || continuable, "stopped completion requires new input before further generation");
    // Numeric wire identity matches the public StackExecution enum without
    // depending on Metal headers in this independently extracted CPU protocol.
    require(current_execution <= 2, "invalid current stack execution mode"); out.execution = current_execution;
    require((!allow_stack_experiments && current_execution != 2) || replay_prepared,
        "stack execution/consent requires startup replay preparation");
    if (const auto* mode = data.find("stack_execution")) {
        require(allow_stack_experiments && replay_prepared && out.reset,
            "request-time stack experiment requires explicit consent, replay preparation and a fresh prefix");
        const auto value = mode->string();
        if (value == "direct") out.execution = 0;
        else if (value == "indirect-compatible") out.execution = 1;
        else if (value == "recurrent-replay") out.execution = 2;
        else throw FormatError("invalid request stack execution mode");
    }
    out.small_splits = current_splits;
    if (const auto* splits = data.find("small_gemv_splits")) {
        require(allow_kernel_experiments && out.reset, "request-time kernel experiment requires explicit startup consent and a fresh prefix");
        const auto n = splits->unsigned_integer(); require(n == 0 || n == 1 || n == 4 || n == 8 || n == 16 || n == 32, "invalid request small GEMV split count");
        out.small_splits = uint32_t(n);
    }
    out.block_formats = current_block_formats;
    if (const auto* formats = data.find("block_gemv_formats")) {
        require(allow_kernel_experiments && out.reset, "request-time kernel experiment requires explicit startup consent and a fresh prefix");
        const auto n = formats->unsigned_integer(); require(n <= 3, "invalid request block GEMV format mask"); out.block_formats = uint32_t(n);
    }
    require(!replay_prepared || (!data.find("small_gemv_splits") && !data.find("block_gemv_formats")),
        "immutable replay preparation refuses request-time kernel overrides in every scheduling arm");
    require((known_capacity == 0 || known_capacity == 8
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        || known_capacity == 512 || known_capacity == 1024 || known_capacity == 2048
#endif
        ) && (!allow_prefill_experiments || known_capacity == 8),
        "invalid known-token preparation/consent");
    require(!current_residual_reuse || known_capacity == 8
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        || known_capacity == 512 || known_capacity == 1024 || known_capacity == 2048
#endif
        , "residual reuse requires prepared known-token capacity eight");
    require(!known_capacity || (current_execution == 0 && !allow_kernel_experiments && !allow_stack_experiments && !replay_prepared),
        "known-token preparation requires fixed direct kernels without captures");
    require(!final_input_logits || ((known_capacity == 512 || known_capacity == 1024 || known_capacity == 2048) && current_residual_reuse && !allow_prefill_experiments), "final-input logits require fixed prepared input512");
    out.known_prefill = known_capacity != 0;
    if (const auto* mode = data.find("known_prefill")) {
        require(known_capacity == 8 && allow_prefill_experiments && out.reset,
            "prefill mode override requires explicit startup consent and a fresh prefix");
        out.known_prefill = mode->boolean();
    }
    out.residual_reuse = current_residual_reuse;
    if (const auto* reuse = data.find("known_residual_reuse")) {
        require(known_capacity == 8 && allow_prefill_experiments && out.reset,
            "residual reuse override requires explicit startup consent and a fresh prefix");
        out.residual_reuse = reuse->boolean();
    }
    require(uint64_t(prefix) + out.tokens.size() + out.generate <= 65536, "target request exceeds total context reservation");
    if (const auto* stops = data.find("stop_ids")) {
        require(stops->array().size() <= 16, "too many target stop IDs");
        for (const auto& value : stops->array()) { const auto id = value.unsigned_integer(); require(id < 248320, "stop ID outside vocabulary"); out.stops.push_back(uint32_t(id)); }
    }
    return out;
}
// END owned target protocol.
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
Json q8_three_routes_json(const MetalGraph::Q8ThreeRoutes& routes) {
    static constexpr std::array names{"raw", "packed", "hc_raw", "wide_qkv_raw", "wide_output_raw"};
    Json::Object fields;for(size_t i=0;i<names.size();++i)fields.emplace_back(names[i],Json::integer(routes[i]));return Json(std::move(fields));
}
MetalGraph::Q8ThreeRoutes q8_three_routes_delta(const MetalGraph::Q8ThreeRoutes& after,const MetalGraph::Q8ThreeRoutes& before) {
    MetalGraph::Q8ThreeRoutes result{};for(size_t i=0;i<result.size();++i){require(after[i]>=before[i],"three-row Q8 routes decreased");result[i]=after[i]-before[i];}return result;
}
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
const char* hc_exact_name(HcExactMode mode) {
    switch (mode) { case HcExactMode::off: return "off"; case HcExactMode::scalar: return "scalar"; case HcExactMode::combined: return "combined"; }
    throw FormatError("invalid private exact-HC mode");
}
Json hc_routes_json(const HcExactRoutes& routes) {
    return Json(Json::Object{{"scalar", Json::integer(routes[0])}, {"many1", Json::integer(routes[1])},
        {"many2", Json::integer(routes[2])}, {"many4", Json::integer(routes[3])}, {"many8", Json::integer(routes[4])}});
}
HcExactRoutes hc_routes_delta(const HcExactRoutes& after, const HcExactRoutes& before) {
    HcExactRoutes result{};
    for (size_t i = 0; i < 5; ++i) { require(after[i] >= before[i], "exact-HC route lifetime regressed"); result[i] = after[i] - before[i]; }
    return result;
}
std::string hc_exact_logits_hash(std::span<const float> logits) {
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    require(logits.size() == 248320, "private full-logit hash shape differs");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(logits.data(), static_cast<CC_LONG>(logits.size_bytes()), digest.data());
    constexpr char hex[] = "0123456789abcdef"; std::string out; out.reserve(64);
    for (const auto x : digest) { out += hex[x >> 4]; out += hex[x & 15]; } return out;
}
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
Json wide_routes_json(const WideQ8ExactRoutes& routes) {
    return Json(Json::Object{{"qkv_scalar", Json::integer(routes[0])}, {"qkv_many1", Json::integer(routes[1])},
        {"output_scalar", Json::integer(routes[2])}, {"output_many1", Json::integer(routes[3])}});
}
WideQ8ExactRoutes wide_routes_delta(const WideQ8ExactRoutes& after, const WideQ8ExactRoutes& before) {
    WideQ8ExactRoutes result{};
    for (size_t i = 0; i < 4; ++i) { require(after[i] >= before[i], "wide-Q8 route lifetime regressed"); result[i] = after[i] - before[i]; }
    return result;
}
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
Json wide_many_routes_json(const WideQ8ManyRoutes& routes) {
    return Json(Json::Object{{"qkv_many2", Json::integer(routes[0])}, {"qkv_many4", Json::integer(routes[1])},
        {"qkv_many8", Json::integer(routes[2])}, {"output_many2", Json::integer(routes[3])},
        {"output_many4", Json::integer(routes[4])}, {"output_many8", Json::integer(routes[5])}});
}
WideQ8ManyRoutes wide_many_routes_delta(const WideQ8ManyRoutes& after, const WideQ8ManyRoutes& before) {
    WideQ8ManyRoutes result{};
    for (size_t i = 0; i < 6; ++i) { require(after[i] >= before[i], "wide-Q8-many route lifetime regressed"); result[i] = after[i] - before[i]; }
    return result;
}
#endif
Json sampling_json(const TargetSampler& sampler, uint32_t committed_prefix) {
    const auto& options = sampler.options();
    return Json(Json::Object{{"temperature", number(options.policy.temperature)}, {"top_k", Json::integer(options.policy.top_k)},
        {"top_p", number(options.policy.top_p)}, {"min_p", number(options.policy.min_p)}, {"seed", Json::integer(options.seed)},
        {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"valid", Json(sampler.matches(committed_prefix))},
        {"entropy_prefix", Json::integer(sampler.prefix())},
        {"filter_order", Json("temperature,top-k,min-p,top-p,normalize")},
        {"rng", Json("mt19937_64; (word >> 11) * 2^-53; point masses consume no entropy")},
        {"ownership", Json("serial session; append/no-work preserve stream; reset installs policy/seed after stack reset or explicit greedy branch restore commits")}});
}
Json top_logits(std::span<const float> logits) {
    std::vector<uint32_t> ids(logits.size()); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 8, ids.end(), [&](auto a, auto b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; });
    Json::Array out; for (size_t i = 0; i < 8; ++i) out.push_back(Json(Json::Object{{"id", Json::integer(ids[i])}, {"logit", number(logits[ids[i]])}})); return Json(std::move(out));
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
struct TargetMtpSelection {
    uint32_t base, anchor;
    const MtpDraftDraw& draft;
    SamplingPolicy policy;
    std::span<const uint32_t> stops;
    std::optional<SampledPathDecision> decision;
    std::optional<MtpAnchorPublication> anchor_publication;
    static uint32_t choose(void* raw, const StackSampledVerificationRows& rows) {
        auto& self = *static_cast<TargetMtpSelection*>(raw);
        require(rows.position == self.base && rows.count == 2 && rows.residual_width == 10240,
            "native MTP target verification row identity mismatch");
        if (rows.has_topk) {
            self.decision.emplace(decide_mtp_depth1_topk(self.base + 1, self.anchor, self.draft,
                rows.topk[0], rows.topk[1], self.policy, self.stops));
        } else {
            self.decision.emplace(decide_mtp_depth1(self.base + 1, self.anchor, self.draft,
                rows.logits[0], rows.logits[1], self.policy, self.stops));
        }
        require(self.decision->accepted <= 1 && self.decision->count >= 1 && self.decision->count <= 2,
            "native MTP sampled decision shape mismatch");
        self.anchor_publication.emplace(prepare_mtp_anchor_publication(self.base, *self.decision));
        return 1 + self.decision->accepted;
    }
};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
void carrier_state_record(TextStack& stack, std::string_view event, uint32_t action) {
    std::vector<std::byte> scratch(1000000);
    const auto fingerprint = stack.accepted_fingerprint(scratch);
    std::string hash; const char* digits = "0123456789abcdef";
    for (auto byte : fingerprint.sha256) { hash += digits[byte >> 4]; hash += digits[byte & 15]; }
    const auto record = Json(Json::Object{{"event", Json(std::string(event))}, {"action", Json::integer(action)},
        {"prefix", Json::integer(stack.position())}, {"bytes", Json::integer(fingerprint.bytes)}, {"accepted_state_sha256", Json(hash)}}).dump() + "\n";
    require(std::fwrite(record.data(), 1, record.size(), stderr) == record.size() && std::fflush(stderr) == 0, "accepted-state diagnostic write failed");
}
struct CarrierPacketTail {
    const Json::Array& generated; const std::optional<uint32_t>& stop_id;
    uint32_t maximum; bool single, open = false;
    bool final_action() const { return single || generated.size() >= maximum || stop_id.has_value(); }
};
struct CarrierAction {
    MetalGraph& graph; TextStack& stack; bool capture; bool active; CarrierPacketTail* tail;
    double alpha; StageAControllerState& state;
    CarrierAction(MetalGraph& g, TextStack& s, CarrierDiagnosticMode mode, bool v2, bool cap, StageAControllerState& controller, bool component = false, CarrierPacketTail* packet_tail = nullptr) : graph(g), stack(s), capture(cap),
        active(mode != CarrierDiagnosticMode::off && g.begin_carrier_diagnostic(v2, component)), tail(packet_tail), alpha(controller.ewma_alpha1), state(controller) {}
    ~CarrierAction() noexcept(false) {
        if (!active) return;
        if (std::uncaught_exceptions()) { try { graph.end_carrier_diagnostic(); } catch (...) { /* The owning failure already invalidates this measurement. */ } }
        else {
            if (capture) {
                carrier_state_record(stack, "carrier_diagnostic_state", graph.carrier_diagnostic_action());
            }
            state.ewma_alpha1 = alpha;
            if (tail && tail->final_action()) tail->open = true;
            else graph.end_carrier_diagnostic();
        }
    }
};
struct CarrierComponentSelection {
    uint32_t prefix;
    static uint32_t choose(void* raw, const StackSampledVerificationRows& rows) {
        const auto prefix = static_cast<CarrierComponentSelection*>(raw)->prefix;
        require(prefix && prefix <= rows.count, "carrier component selector prefix mismatch");
        return prefix;
    }
};
#endif
AcceptanceRates dflash_measured_probabilities(const DFlash2Drafter& drafter, std::span<const float, 7> confidence) {
    AcceptanceRates out{}; const auto& a = drafter.verify_budget().acceptance();
    for (size_t i = 0; i < out.size(); ++i) {
        const double q = std::clamp(double(confidence[i]), 1e-6, 1 - 1e-6);
        const double z = double(a.slope) * std::log(q / (1 - q)) + double(a.intercepts[i]);
        out[i] = z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z));
    }
    return out;
}
struct MeasuredRoundCompletion {
    MeasuredController* controller; StageAControllerState* state;
    uint32_t context, rows, accepted, drafted_rows; bool stopped;
    Clock::time_point started; AcceptanceRates probabilities;
    void publish() const {
        controller->observe(context, rows, accepted, controller_observed_decode_ms(milliseconds(started)), probabilities, true, drafted_rows, stopped);
        state->measured = controller->observation(); state->ewma_alpha1 = controller->acceptance()[0];
        state->current_mode = controller->speculative_mode() ? 0 : 1;
        state->consecutive_ordinary_steps = uint8_t(controller->actions_since_deep_probe());
    }
};
struct MeasuredRoundObserver {
    MeasuredController* controller; StageAControllerState* state;
    uint32_t context, drafted_rows = 0; bool stopped = false; Clock::time_point started;
    const uint64_t &proposed, &accepted; uint64_t before_proposed, before_accepted;
    AcceptanceRates probabilities{}; int exceptions = std::uncaught_exceptions();
    std::optional<MeasuredRoundCompletion>& tail;
    const Json::Array& generated; const std::optional<uint32_t>& stop_id; uint32_t packet_maximum;
    MeasuredRoundObserver(MeasuredController* c, StageAControllerState* s, uint32_t ctx, Clock::time_point start,
                          const uint64_t& p, const uint64_t& a, std::optional<MeasuredRoundCompletion>& last,
                          const Json::Array& output, const std::optional<uint32_t>& stop, uint32_t maximum) : controller(c), state(s), context(ctx), started(start),
        proposed(p), accepted(a), before_proposed(p), before_accepted(a), tail(last), generated(output), stop_id(stop), packet_maximum(maximum) {
        if (controller) probabilities = controller->acceptance();
    }
    ~MeasuredRoundObserver() noexcept(false) {
        if (!controller || std::uncaught_exceptions() != exceptions) return;
        const auto rows = uint32_t(proposed - before_proposed + 1), labels = uint32_t(accepted - before_accepted);
        const MeasuredRoundCompletion completed{controller, state, context, rows, labels, drafted_rows,
            stopped || stop_id.has_value(), started, probabilities};
        if (generated.size() >= packet_maximum || stop_id) {
            require(!tail, "multiple final measured actions in one packet"); tail = completed;
        } else completed.publish();
    }
};
struct TargetMtpSelection2 {
    uint32_t base, anchor;
    const MtpDraftDraw &first;
    const MtpDraftDraw *second = nullptr;
    const MtpDraftDraw *third = nullptr;
    SamplingPolicy policy; std::span<const uint32_t> stops;
    std::optional<SampledPathDecision> decision;
    std::optional<MtpAnchorPublication> publication;
    static uint32_t choose(void* raw, const StackSampledVerificationRows& rows) {
        auto& self = *static_cast<TargetMtpSelection2*>(raw);
        if (self.third) {
            require(self.second && rows.position == self.base && rows.count == 4 && rows.residual_width == 10240,
                "depth-three MTP target verification row identity mismatch");
            if (rows.has_topk) {
                self.decision.emplace(decide_mtp_depth3_topk(self.base + 1, self.anchor, self.first, *self.second, *self.third,
                    rows.topk[0], rows.topk[1], rows.topk[2], rows.topk[3], self.policy, self.stops));
            } else {
                self.decision.emplace(decide_mtp_depth3(self.base + 1, self.anchor, self.first, *self.second, *self.third,
                    rows.logits[0], rows.logits[1], rows.logits[2], rows.logits[3], self.policy, self.stops));
            }
        } else if (self.second) {
            require(rows.position == self.base && rows.count == 3 && rows.residual_width == 10240,
                "depth-two MTP target verification row identity mismatch");
            if (rows.has_topk) {
                self.decision.emplace(decide_mtp_depth2_topk(self.base + 1, self.anchor, self.first, *self.second,
                    rows.topk[0], rows.topk[1], rows.topk[2], self.policy, self.stops));
            } else {
                self.decision.emplace(decide_mtp_depth2(self.base + 1, self.anchor, self.first, *self.second,
                    rows.logits[0], rows.logits[1], rows.logits[2], self.policy, self.stops));
            }
        } else {
            require(rows.position == self.base && rows.count == 2 && rows.residual_width == 10240,
                "depth-one MTP target verification row identity mismatch");
            if (rows.has_topk) {
                self.decision.emplace(decide_mtp_depth1_topk(self.base + 1, self.anchor, self.first,
                    rows.topk[0], rows.topk[1], self.policy, self.stops));
            } else {
                self.decision.emplace(decide_mtp_depth1(self.base + 1, self.anchor, self.first,
                    rows.logits[0], rows.logits[1], self.policy, self.stops));
            }
        }
        self.publication.emplace(prepare_mtp_anchor_publication2(self.base, *self.decision));
        return 1 + self.decision->accepted;
    }
};
struct TargetDFlash2Selection {
    uint32_t base = 0, anchor = 0;
    const DFlash2DraftResult& draft;
    SamplingPolicy policy;
    std::span<const uint32_t> stops;
    std::optional<SampledPathDecision> decision;
    std::optional<MtpAnchorPublication> publication;
    static uint32_t choose(void* raw, const StackSampledVerificationRows& rows) {
        auto& self = *static_cast<TargetDFlash2Selection*>(raw);
        require(rows.position == self.base && rows.count == 1 + self.draft.count &&
            rows.residual_width == 10240 && self.draft.count >= 1 && self.draft.count <= 7,
            "DFlash2 target verification row identity mismatch");
        const uint32_t prefix = self.base + 1;
        if (rows.has_topk) {
            const SampledPathIdentity identity{1, 1, 1, prefix, rows.vocabulary, self.anchor};
            std::array<SampledTargetTopKRow, 8> target_rows{};
            for (uint32_t i = 0; i <= self.draft.count; ++i) target_rows[i] = {prefix + i, rows.topk[i]};
            self.decision.emplace(decide_sampled_path_topk(
                identity, std::span(self.draft.proposals).first(self.draft.count),
                identity, std::span(target_rows).first(1 + self.draft.count),
                self.policy, 1 + self.draft.count, self.stops,
                self.draft.random_after, self.draft.proposal_begin_words));
        } else {
            const SampledPathIdentity identity{1, 1, 1, prefix, uint32_t(rows.logits[0].size()), self.anchor};
            std::array<SampledTargetRow, 8> target_rows{};
            for (uint32_t i = 0; i <= self.draft.count; ++i) target_rows[i] = {prefix + i, rows.logits[i]};
            self.decision.emplace(decide_sampled_path(
                identity, std::span(self.draft.proposals).first(self.draft.count),
                identity, std::span(target_rows).first(1 + self.draft.count),
                self.policy, 1 + self.draft.count, self.stops,
                self.draft.random_after, self.draft.proposal_begin_words));
        }
        self.publication.emplace(prepare_mtp_anchor_publication8(self.base, *self.decision));
        return 1 + self.decision->accepted;
    }
};
#endif

class PleAsyncGatherer {
public:
    static constexpr size_t kWorkers = 8;
    static constexpr size_t kMaxSlots = 8 * 16;
    static constexpr size_t kRowWidth = 160;

    PleAsyncGatherer(EmbeddingRows& cache, const NgramSpec& spec, NgramPolicy policy,
                     std::span<const std::byte> ple_bytes, uint32_t ple_type,
                     uint64_t stored_rows, uint64_t logical_rows, PreadSource pread_source = {})
        : cache_(cache), spec_(spec), policy_(policy), ple_bytes_(ple_bytes), ple_type_(ple_type),
          stored_rows_(stored_rows), logical_rows_(logical_rows), pread_source_(pread_source) {
        try {
            for (size_t i = 0; i < kWorkers; ++i) {
                workers_[i] = std::thread([this, i] { worker_loop(i); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                stop_ = true;
            }
            cv_req_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) worker.join();
            }
            throw;
        }
    }
    ~PleAsyncGatherer() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_req_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }
    PleAsyncGatherer(const PleAsyncGatherer&) = delete;
    PleAsyncGatherer& operator=(const PleAsyncGatherer&) = delete;

    void start_tokens(NgramHistory entry_history, std::span<const uint32_t> tokens, std::span<float> dst) {
        require(!tokens.empty() && tokens.size() <= 8 && dst.size() == tokens.size() * 2560ull,
            "invalid async PLE gather token batch");
        require(!busy_, "concurrent async PLE gather overlap");

        const size_t total_slots = tokens.size() * 16u;
        NgramHistory hist = entry_history;
        for (size_t t = 0; t < tokens.size(); ++t) {
            hist = ngram_step(spec_, policy_, hist, tokens[t], std::span(active_ids_).subspan(t * 16u, 16u));
        }

        size_t miss_slots = 0;
        for (size_t s = 0; s < total_slots; ++s) {
            const uint64_t row = active_ids_[s];
            if (!cache_.try_copy(row, dst.subspan(s * kRowWidth, kRowWidth))) {
                miss_slots_[miss_slots++] = {row, uint32_t(s)};
            }
        }

        exit_history_ = hist;
        dst_ = dst;
        miss_slot_count_ = miss_slots;
        if (miss_slots == 0) {
            miss_unique_count_ = 0;
            busy_ = true;
            async_inflight_ = false;
            return;
        }

        std::sort(miss_slots_.begin(), miss_slots_.begin() + ptrdiff_t(miss_slots),
            [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first < b.first : a.second < b.second;
            });
        size_t unique_count = 0;
        for (size_t i = 0; i < miss_slots;) {
            const uint64_t row = miss_slots_[i].first;
            unique_misses_[unique_count++] = row;
            ++i;
            while (i < miss_slots && miss_slots_[i].first == row) ++i;
        }
        miss_unique_count_ = unique_count;

        const size_t launch_workers = std::min<size_t>(kWorkers, unique_count);
        for (size_t w = 0; w < launch_workers; ++w) worker_errors_[w] = nullptr;
        next_item_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mu_);
            active_workers_ = launch_workers;
            finished_workers_ = 0;
            ++epoch_;
        }
        busy_ = true;
        async_inflight_ = true;
        // Every worker must observe the epoch because a one-row job is assigned specifically to worker 0.
        cv_req_.notify_all();
    }

    NgramHistory finish() {
        require(busy_, "async PLE gather finish without active job");
        busy_ = false;
        if (!async_inflight_) {
            return exit_history_;
        }
        async_inflight_ = false;
        size_t launched = 0;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_done_.wait(lock, [this] { return finished_workers_ == active_workers_; });
            launched = active_workers_;
        }
        for (size_t w = 0; w < launched; ++w) {
            if (worker_errors_[w]) {
                auto err = std::exchange(worker_errors_[w], nullptr);
                cache_.record_failure();
                std::rethrow_exception(err);
            }
        }
        size_t u = 0;
        for (size_t i = 0; i < miss_slot_count_; ++u) {
            const uint64_t row = miss_slots_[i].first;
            const auto decoded = std::span<const float>(miss_decoded_.data() + u * kRowWidth, kRowWidth);
            cache_.insert_decoded(row, decoded, true);
            while (i < miss_slot_count_ && miss_slots_[i].first == row) {
                const uint32_t slot = miss_slots_[i].second;
                std::memcpy(dst_.data() + size_t(slot) * kRowWidth, decoded.data(), kRowWidth * sizeof(float));
                ++i;
            }
        }
        return exit_history_;
    }

    void cancel_join_nothrow() noexcept {
        if (!busy_) return;
        busy_ = false;
        if (!async_inflight_) return;
        async_inflight_ = false;
        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this] { return finished_workers_ == active_workers_; });
        for (auto& err : worker_errors_) err = nullptr;
    }

private:
    void worker_loop(size_t worker_id) {
        uint64_t seen_epoch = 0;
        for (;;) {
            size_t active = 0;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_req_.wait(lock, [&] { return stop_ || epoch_ != seen_epoch; });
                if (stop_) return;
                seen_epoch = epoch_;
                active = active_workers_;
            }
            if (worker_id < active) {
                for (;;) {
                    const size_t idx = next_item_.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= miss_unique_count_) break;
                    try {
                        dequantize_row_at(ple_bytes_, ple_type_, uint32_t(kRowWidth), stored_rows_, logical_rows_,
                            unique_misses_[idx],
                            std::span<float>(miss_decoded_.data() + idx * kRowWidth, kRowWidth),
                            pread_source_);
                    } catch (...) {
                        if (!worker_errors_[worker_id]) worker_errors_[worker_id] = std::current_exception();
                    }
                }
                std::lock_guard<std::mutex> lock(mu_);
                if (++finished_workers_ == active) cv_done_.notify_one();
            }
        }
    }

    EmbeddingRows& cache_;
    const NgramSpec& spec_;
    NgramPolicy policy_;
    std::span<const std::byte> ple_bytes_;
    uint32_t ple_type_ = 0;
    uint64_t stored_rows_ = 0, logical_rows_ = 0;
    PreadSource pread_source_{};
    std::mutex mu_;
    std::condition_variable cv_req_, cv_done_;
    bool stop_ = false, busy_ = false, async_inflight_ = false;
    uint64_t epoch_ = 0;
    size_t active_workers_ = 0, finished_workers_ = 0;
    std::atomic<size_t> next_item_{0};
    NgramHistory exit_history_{};
    std::span<float> dst_{};
    std::array<uint64_t, kMaxSlots> active_ids_{};
    std::array<std::pair<uint64_t, uint32_t>, kMaxSlots> miss_slots_{};
    std::array<uint64_t, kMaxSlots> unique_misses_{};
    std::array<float, kMaxSlots * kRowWidth> miss_decoded_{};
    size_t miss_slot_count_ = 0, miss_unique_count_ = 0;
    std::array<std::exception_ptr, kWorkers> worker_errors_{};
    std::array<std::thread, kWorkers> workers_{};
};

struct PleAsyncScopeGuard {
    PleAsyncGatherer* gatherer = nullptr;
    ~PleAsyncScopeGuard() { if (gatherer) gatherer->cancel_join_nothrow(); }
    void start(PleAsyncGatherer& g, NgramHistory h, std::span<const uint32_t> tokens, std::span<float> dst) {
        g.start_tokens(h, tokens, dst);
        gatherer = &g;
    }
    NgramHistory finish() {
        auto* g = std::exchange(gatherer, nullptr);
        require(g != nullptr, "PleAsyncScopeGuard finish without active gatherer");
        return g->finish();
    }
};

#endif
const char* profile_mode(ProfileMode mode) {
    return mode == ProfileMode::encoder_boundary ? "encoder_boundary" : mode == ProfileMode::dispatch_boundary ? "dispatch_boundary" : "disabled";
}
Json profile_json(const GraphProfile& profile) {
    if (profile.mode == ProfileMode::disabled) return Json();
    // Do not trust the flag alone when exposing diagnostic measurements.
    const bool usable = profile.timings_valid && profile_calibration_valid(profile.gpu_tick_nanoseconds, profile.cpu_clock_mismatch_fraction);
    using Key = std::tuple<std::string, uint32_t, std::array<uint32_t, 4>>;
    struct Summary { uint64_t count = 0; double total = 0, minimum = INFINITY, maximum = 0; };
    std::map<Key, Summary> groups;
    for (const auto& kernel : profile.kernels) {
        auto& s = groups[{kernel.kernel, kernel.encoding, kernel.shape}]; ++s.count;
        if (usable) {
            s.total += kernel.milliseconds;
            s.minimum = std::min(s.minimum, kernel.milliseconds); s.maximum = std::max(s.maximum, kernel.milliseconds);
        }
    }
    Json::Array result;
    for (const auto& [key, value] : groups) {
        const auto& [name, type, shape] = key; Json::Array dimensions; for (auto d : shape) dimensions.push_back(Json::integer(d));
        result.push_back(Json(Json::Object{{"kernel", Json(name)}, {"encoding", Json::integer(type)}, {"shape", Json(std::move(dimensions))}, {"calls", Json::integer(value.count)},
            {"total_ms", usable ? number(value.total) : Json()}, {"minimum_ms", usable ? number(value.minimum) : Json()}, {"maximum_ms", usable ? number(value.maximum) : Json()}}));
    }
    return Json(Json::Object{{"mode", Json(profile_mode(profile.mode))}, {"intrusive", Json(true)}, {"dispatches", Json::integer(profile.kernels.size())},
        {"timings_valid", Json(usable)}, {"sampled_sum_ms", usable ? number(profile.sampled_milliseconds) : Json()},
        {"gpu_tick_nanoseconds", usable ? number(profile.gpu_tick_nanoseconds) : Json()}, {"cpu_clock_is_mach", Json(profile.cpu_clock_is_mach)},
        {"cpu_clock_mismatch_fraction", std::isfinite(profile.cpu_clock_mismatch_fraction) ? number(profile.cpu_clock_mismatch_fraction) : Json()},
        {"scope", Json("per-operation diagnostic; encoder changes/barriers perturb execution, summed scopes need not equal unprofiled critical path")}, {"groups", Json(std::move(result))}});
}
#include "route_attribution_sink.inc"
#include "route_lock_sink.inc"
#include "kernel_search_capture_sink.inc"
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
struct TargetQsaRouteAttributionSink {
    int directory = -1, manifest = -1;
    uint32_t records = 0;
    uint64_t bytes = 0;
    std::array<std::string, 6> phase1_digests{};
    ~TargetQsaRouteAttributionSink() { if (manifest >= 0) close(manifest); if (directory >= 0) close(directory); }
    static void write_all(int fd, std::span<const std::byte> data) {
        while (!data.empty()) { const auto n = write(fd, data.data(), data.size()); if (n < 0 && errno == EINTR) continue; require(n > 0, "QSA route capture write failed"); data = data.subspan(size_t(n)); }
    }
    void open_directory(const std::filesystem::path& path) {
        require(path.is_absolute() && std::filesystem::canonical(path) == path.lexically_normal() && std::filesystem::is_empty(path), "QSA route capture needs a fresh canonical private directory");
        directory = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC); struct stat st{};
        require(directory >= 0 && fstat(directory, &st) == 0 && st.st_uid == geteuid() && (st.st_mode & 07777) == 0700, "QSA route capture directory must be owner-only 0700");
        manifest = openat(directory, "manifest.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        require(manifest >= 0, "QSA route capture manifest already exists");
    }
    static bool emit(void* context, MetalGraph& g, const QsaRouteAttributionRecord& r) noexcept {
        try { static_cast<TargetQsaRouteAttributionSink*>(context)->record(g, r); return true; } catch (...) { return false; }
    }
    void record(MetalGraph& g, const QsaRouteAttributionRecord& r) {
        constexpr std::array<uint32_t, 4> sequence{1, 2, 5, 3};
        const uint32_t call = records / 4;
        require(records < 512 && r.layer >= 0 && r.phase == sequence[records % 4], "QSA route capture phase sequence");
        const bool common = r.phase == 1 || r.phase == 5; Json::Array views;
        for (size_t i = 0; i < r.views.size(); ++i) {
            if (!r.views[i].owner) {
                if (r.phase == 1) phase1_digests[i].clear();
                if (r.phase == 5) require(phase1_digests[i].empty(), "QSA route capture common-view presence changed");
                views.emplace_back(); continue;
            }
            const auto data = g.read(r.views[i]); require(data.size() < (64ull << 20), "QSA route capture view bound");
            std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256(data.data(), CC_LONG(data.size()), digest.data());
            constexpr char hex[] = "0123456789abcdef"; std::string hash; for (auto c : digest) { hash += hex[c >> 4]; hash += hex[c & 15]; }
            if (r.phase == 1) phase1_digests[i] = hash;
            if (r.phase == 5) require(phase1_digests[i] == hash, "QSA route capture shadow changed common input or restored cache");
            Json::Object view{{"bytes", Json::integer(data.size())}, {"sha256", Json(hash)}};
            if (!common) {
                if (i == 1) {
                    require(data.size() % 4 == 0, "QSA route capture router-logit alignment");
                    std::vector<float> values(data.size() / 4); std::memcpy(values.data(), data.data(), data.size());
                    for (auto x : values) require(std::isfinite(x), "nonfinite QSA route capture router logit");
                }
                require(bytes + data.size() <= (512ull << 20), "QSA route capture disk budget");
                const auto name = std::to_string(call) + "-" + std::to_string(r.layer) + "-" + std::to_string(r.phase) + "-" + std::to_string(i) + ".bin";
                const int fd = openat(directory, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
                require(fd >= 0, "QSA route capture refuses reused output");
                try { write_all(fd, data); } catch (...) { close(fd); throw; }
                require(close(fd) == 0, "QSA route capture close failed");
                bytes += data.size(); view.emplace_back("file", Json(name));
            }
            views.emplace_back(std::move(view));
        }
        const auto line = Json(Json::Object{{"schema", Json::integer(1)}, {"record", Json::integer(records)}, {"call", Json::integer(call)},
            {"layer", Json::integer(r.layer)}, {"prefix", Json::integer(r.prefix)}, {"phase", Json::integer(r.phase)},
            {"rows", Json::integer(r.rows)}, {"channels", Json::integer(r.channels)}, {"experts", Json::integer(r.experts)},
            {"selected", Json::integer(r.selected)}, {"common_layer_input_and_accepted_state", Json(common)}, {"views", Json(std::move(views))}}).dump() + "\n";
        write_all(manifest, std::as_bytes(std::span(line))); ++records;
    }
    void finish() const { require(records > 0 && records % 4 == 0, "QSA route capture ended mid-layer"); }
};
#endif
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
struct TargetGdnRouteAttributionSink {
    int directory = -1, manifest = -1;
    uint32_t records = 0;
    uint64_t bytes = 0;
    std::array<std::string, 6> phase1_digests{};
    ~TargetGdnRouteAttributionSink() { if (manifest >= 0) close(manifest); if (directory >= 0) close(directory); }
    static void write_all(int fd, std::span<const std::byte> data) {
        while (!data.empty()) { const auto n = write(fd, data.data(), data.size()); if (n < 0 && errno == EINTR) continue; require(n > 0, "GDN route capture write failed"); data = data.subspan(size_t(n)); }
    }
    void open_directory(const std::filesystem::path& path) {
        require(path.is_absolute() && std::filesystem::canonical(path) == path.lexically_normal() && std::filesystem::is_empty(path), "GDN route capture needs a fresh canonical private directory");
        directory = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC); struct stat st{};
        require(directory >= 0 && fstat(directory, &st) == 0 && st.st_uid == geteuid() && (st.st_mode & 07777) == 0700, "GDN route capture directory must be owner-only 0700");
        manifest = openat(directory, "manifest.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        require(manifest >= 0, "GDN route capture manifest already exists");
    }
    static bool emit(void* context, MetalGraph& g, const GdnRouteAttributionRecord& r) noexcept {
        try { static_cast<TargetGdnRouteAttributionSink*>(context)->record(g, r); return true; } catch (...) { return false; }
    }
    void record(MetalGraph& g, const GdnRouteAttributionRecord& r) {
        constexpr std::array<uint32_t, 4> sequence{1, 2, 5, 3};
        const uint32_t call = records / 4;
        require(records < 1024 && r.layer >= 0 && r.phase == sequence[records % 4], "GDN route capture phase sequence");
        const bool common = r.phase == 1 || r.phase == 5; Json::Array views;
        for (size_t i = 0; i < r.views.size(); ++i) {
            if (!r.views[i].owner) {
                if (r.phase == 1) phase1_digests[i].clear();
                if (r.phase == 5) require(phase1_digests[i].empty(), "GDN route capture common-view presence changed");
                views.emplace_back(); continue;
            }
            const auto data = g.read(r.views[i]); require(data.size() < (64ull << 20), "GDN route capture view bound");
            std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256(data.data(), CC_LONG(data.size()), digest.data());
            constexpr char hex[] = "0123456789abcdef"; std::string hash; for (auto c : digest) { hash += hex[c >> 4]; hash += hex[c & 15]; }
            if (r.phase == 1) phase1_digests[i] = hash;
            if (r.phase == 5) require(phase1_digests[i] == hash, "GDN route capture shadow changed common input or accepted GDN state");
            Json::Object view{{"bytes", Json::integer(data.size())}, {"sha256", Json(hash)}};
            if (!common) {
                if (i == 1) {
                    require(data.size() % 4 == 0, "GDN route capture router-logit alignment");
                    std::vector<float> values(data.size() / 4); std::memcpy(values.data(), data.data(), data.size());
                    for (auto x : values) require(std::isfinite(x), "nonfinite GDN route capture router logit");
                }
                require(bytes + data.size() <= (1ull << 30), "GDN route capture disk budget");
                const auto name = std::to_string(call) + "-" + std::to_string(r.layer) + "-" + std::to_string(r.phase) + "-" + std::to_string(i) + ".bin";
                const int fd = openat(directory, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
                require(fd >= 0, "GDN route capture refuses reused output");
                try { write_all(fd, data); } catch (...) { close(fd); throw; }
                require(close(fd) == 0, "GDN route capture close failed");
                bytes += data.size(); view.emplace_back("file", Json(name));
            }
            views.emplace_back(std::move(view));
        }
        const auto line = Json(Json::Object{{"schema", Json::integer(1)}, {"record", Json::integer(records)}, {"call", Json::integer(call)},
            {"layer", Json::integer(r.layer)}, {"prefix", Json::integer(r.prefix)}, {"phase", Json::integer(r.phase)},
            {"rows", Json::integer(r.rows)}, {"channels", Json::integer(r.channels)}, {"experts", Json::integer(r.experts)},
            {"selected", Json::integer(r.selected)}, {"common_layer_input_and_accepted_state", Json(common)}, {"views", Json(std::move(views))}}).dump() + "\n";
        write_all(manifest, std::as_bytes(std::span(line))); ++records;
    }
    void finish() const { require(records > 0 && records % 4 == 0, "GDN route capture ended mid-layer"); }
};
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
Json::Object operator_hc_down_staged_identity(bool enabled) {
    return {{"operator_prefill_hc_down_schema", Json::integer(1)},
        {"operator_prefill_hc_down_version", Json("hc-down-staged-rows-v1")},
        {"operator_prefill_hc_down_route", Json(enabled ? "staged-f32" : "scalar-exact")},
        {"operator_prefill_hc_down_kernel", Json(enabled ? "graph_dense_q8_staged_f32" : "graph_hc_q8_exact_r1_u16")},
        {"operator_prefill_hc_down_min_rows", Json::integer(2)}, {"operator_prefill_hc_down_max_rows", Json::integer(512)},
        {"operator_prefill_hc_down_scope", Json("explicit diagnostic known-input Q8 HC down10240-to320 only; encoded calls include aborted submissions; no serving selection")}};
}
#endif
struct TargetOperatorCheckpointSink {
    int descriptor = -1;
    uint64_t bytes = 0, owner = 0, gpu_expected = 0, gpu_observed = 0, maximum_bytes = 16ull << 20;
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    std::optional<bool> router_batch; const char* router_kernel=nullptr;
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    std::optional<bool> staged_dense;
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    std::optional<bool> hc_rows;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    std::optional<bool> router_window;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    std::optional<bool> hc_down_window;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    std::optional<bool> qsa_sparse;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    std::optional<bool> gdn_ported;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    std::optional<bool> hc_down_staged;
#endif
    ~TargetOperatorCheckpointSink() { if (descriptor >= 0) close(descriptor); }
    void record(Json value) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        if (staged_dense.has_value()) {
            auto fields = value.object();
            fields.emplace_back("operator_prefill_dense_schema", Json::integer(1));
            fields.emplace_back("operator_prefill_dense_route", Json(*staged_dense ? "staged-f32" : "old-mpp"));
            fields.emplace_back("operator_prefill_dense_kernel", Json(*staged_dense ? "graph_dense_q8_staged_f32" : "graph_q8_mpp_wide"));
            fields.emplace_back("operator_prefill_dense_scope", Json("selected chunked-prefill dense Q8 only; not decode/MTP kernels or a completed dispatch counter"));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
        if(router_batch.has_value()){
            auto fields=value.object();fields.emplace_back("operator_prefill_router_schema",Json::integer(1));
            fields.emplace_back("operator_prefill_router_route",Json(*router_batch?"batched-f32":"scalar-f32"));fields.emplace_back("operator_prefill_router_kernel",Json(router_kernel));
            fields.emplace_back("operator_prefill_router_scope",Json("selected chunked-prefill raw-F32 router2560-to512 for 2..512 rows only; not ordinary/N3/MTP or a completed dispatch counter"));value=Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
        if(hc_rows.has_value()){
            auto fields=value.object();fields.emplace_back("operator_prefill_hc_rows_schema",Json::integer(1));fields.emplace_back("operator_prefill_hc_rows_route",Json(*hc_rows?"rows-split16-exact":"legacy-per-token"));
            fields.emplace_back("operator_prefill_hc_rows_scope",Json("known-input injection and residual read/write only; unchanged split16 arithmetic, no HC down or decode selection"));value=Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
        if(router_window.has_value()){
            auto fields=value.object();fields.emplace_back("operator_prefill_router_window_schema",Json::integer(1));fields.emplace_back("operator_prefill_router_window_route",Json(*router_window?"window16-simd32-exact":"legacy-per-token"));
            fields.emplace_back("operator_prefill_router_window_scope",Json("known-input raw-F32 router2560-to512, top-k route, shared-expert gate and mixture for 2..512 rows only; unchanged simd32/split16 arithmetic, no ordinary/N3/MTP selection"));value=Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
        if(hc_down_window.has_value()){
            auto fields=value.object();fields.emplace_back("operator_prefill_hc_down_window_schema",Json::integer(1));fields.emplace_back("operator_prefill_hc_down_window_route",Json(*hc_down_window?"window16-q8-simd32-exact":"scalar-hc-exact"));
            fields.emplace_back("operator_prefill_hc_down_window_scope",Json("known-input raw-Q8-0 HC down 10240-to-320 for 2..512 rows only; unchanged simd32 block-order arithmetic, no singleton/ordinary/N3/MTP selection"));value=Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        if(qsa_sparse.has_value()){
            auto fields=value.object();fields.emplace_back("operator_qsa_sparse_schema",Json::integer(1));fields.emplace_back("operator_qsa_sparse_route",Json(*qsa_sparse?"online-softmax-single-pass":"chronological-three-pass"));
            fields.emplace_back("operator_qsa_sparse_decode_kernel",Json(*qsa_sparse?"qsa_sparse_attention":"qsa_score_stage+qsa_chronological_factors+qsa_chronological_channels"));
            fields.emplace_back("operator_qsa_sparse_verify_kernel",Json(*qsa_sparse?"qsa_sparse_attention_rows":"qsa_verify_rows_score+qsa_verify_rows_factors+qsa_verify_rows_channels"));
            fields.emplace_back("operator_qsa_sparse_scope",Json("ordinary decode and verification-row QSA attention only; chunked prefill QSA unchanged"));value=Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (gdn_ported.has_value()) {
            auto fields = value.object();
            fields.emplace_back("operator_gdn_ported_schema", Json::integer(1));
            fields.emplace_back("operator_gdn_ported_route", Json(*gdn_ported ? "ported-pipelined-tb12-verify-main" : "legacy-blocked-tb16-delta-tile"));
            fields.emplace_back("operator_gdn_ported_prefill_kernel", Json(*gdn_ported ? "gdn_pipelined_tb12" : "gdn_blocked_seq_tb16"));
            fields.emplace_back("operator_gdn_ported_verify_kernel", Json(*gdn_ported ? "gdn_verify_main" : "gdn_delta_tile"));
            fields.emplace_back("operator_gdn_ported_scope", Json("chunked prefill, ordinary decode, verification-row and replay GDN recurrence only; QSA and MoE kernels unchanged"));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
        if (hc_down_staged.has_value()) {
            auto fields = value.object(); for (const auto& field : operator_hc_down_staged_identity(*hc_down_staged)) fields.push_back(field); value = Json(std::move(fields));
        }
#endif
        const auto line = value.dump(16384) + "\n"; struct stat status{};
        require(descriptor >= 0 && fstat(descriptor, &status) == 0 && S_ISREG(status.st_mode) && status.st_size >= 0 && uint64_t(status.st_size) == bytes &&
            bytes <= maximum_bytes && line.size() <= maximum_bytes - bytes, "operator checkpoint file changed or exceeded its registered byte budget");
        size_t at = 0;
        for (uint32_t attempt = 0; attempt < 64 && at < line.size(); ++attempt) {
            const auto n = write(descriptor, line.data() + at, line.size() - at); if (n < 0 && errno == EINTR) continue;
            require(n > 0, "operator checkpoint write failed"); at += size_t(n);
        }
        require(at == line.size(), "operator checkpoint short write"); bytes += line.size();
    }
    void open_file(const std::filesystem::path& path, bool enabled
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        , std::optional<bool> staged = {}
#endif
        ) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        staged_dense = staged;
#endif
        descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        require(descriptor >= 0, "operator checkpoint file must be fresh and private");
        record(Json(Json::Object{{"schema", Json::integer(1)}, {"event", Json("operator_checkpoint_settings")}, {"enabled", Json(enabled)},
            {"intrusive_command_buffer_splitting", Json(enabled)}, {"workload", Json("prefill512,generate3(one MTP2 round plus ordinary tails),forced1 continuation")},
            {"physical_clock_scope", Json("BatchResult command-buffer GPU duration; raw start/end timestamps and calibration unavailable")},
            {"rollup_scope", Json("sum of physical submitted dispatches, native batch wall and GPU durations; excludes sink and inter-segment host gaps")},
            {"critical_path_qualified", Json(false)}, {"performance_qualified", Json(false)}}));
    }
    static Json batch(const BatchResult* value, bool clock_complete = true) {
        if (!value) return Json{};
        std::ostringstream raw; raw << std::setprecision(17) << value->gpu_ms;
        return Json(Json::Object{{"dispatches", Json::integer(value->dispatches)}, {"capture_executes", Json::integer(value->capture_executes)},
            {"wall_ms", std::isfinite(value->wall_ms) ? number(value->wall_ms) : Json()}, {"gpu_ms", clock_complete && std::isfinite(value->gpu_ms) && value->gpu_ms > 0 ? number(value->gpu_ms) : Json()},
            {"gpu_ms_raw_lexeme", Json(raw.str())}, {"gpu_duration_positive", Json(std::isfinite(value->gpu_ms) && value->gpu_ms > 0)}});
    }
    static bool emit(void* context, const OperatorCheckpointRecord& tag, const BatchResult* physical, const BatchResult& total) noexcept {
        try {
            auto& self = *static_cast<TargetOperatorCheckpointSink*>(context);
            if (self.owner != tag.owner_batch) { self.owner = tag.owner_batch; self.gpu_expected = self.gpu_observed = 0; }
            if (physical) { ++self.gpu_expected; self.gpu_observed += std::isfinite(physical->gpu_ms) && physical->gpu_ms > 0; }
            const bool complete = self.gpu_expected && self.gpu_expected == self.gpu_observed;
            Json::Array dims; for (auto n : tag.dimensions) dims.push_back(Json::integer(n));
            self.record(Json(Json::Object{{"schema", Json::integer(1)}, {"event", Json("operator_checkpoint")},
                {"owner_batch", Json::integer(tag.owner_batch)}, {"segment", Json::integer(tag.segment)}, {"owner_final", Json(tag.owner_final)}, {"physical", batch(physical)}, {"rollup", batch(&total, complete)}, {"rollup_gpu_expected", Json::integer(self.gpu_expected)}, {"rollup_gpu_observed", Json::integer(self.gpu_observed)}, {"rollup_gpu_clock_complete", Json(complete)},
                {"category", Json(tag.category)}, {"tag", Json(tag.tag)}, {"phase", Json(tag.phase)}, {"layer", number(tag.layer)}, {"prefix", Json::integer(tag.prefix)}, {"rows", Json::integer(tag.rows)},
                {"dimensions", Json(std::move(dims))}, {"intrusive_command_buffer_splitting", Json(true)}, {"critical_path_qualified", Json(false)}})); return true;
        } catch (...) { return false; }
    }
    void finish() { require(descriptor >= 0 && fsync(descriptor) == 0, "operator checkpoint final sync failed"); }
};
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
bool task4_profile_stderr_writable() noexcept {
    struct stat status{}; const auto flags = fcntl(STDERR_FILENO, F_GETFL);
    return flags >= 0 && (flags & O_ACCMODE) != O_RDONLY && fstat(STDERR_FILENO, &status) == 0 && S_ISREG(status.st_mode);
}
const char* task4_phase_name(Task4ProfilePhase p) {
    switch (p) {
    case Task4ProfilePhase::prefill_layer: return "prefill_layer";
    case Task4ProfilePhase::decode_layer: return "decode_layer";
    case Task4ProfilePhase::prefill_mixer: return "prefill_mixer";
    case Task4ProfilePhase::prefill_output: return "prefill_output";
    case Task4ProfilePhase::decode_mixer: return "decode_mixer";
    case Task4ProfilePhase::decode_output: return "decode_output";
    case Task4ProfilePhase::mtp_prime: return "mtp_prime";
    case Task4ProfilePhase::mtp_advance: return "mtp_advance";
    case Task4ProfilePhase::mtp_mixer: return "mtp_mixer";
    case Task4ProfilePhase::mtp_output: return "mtp_output";
    default: return "unassigned";
    }
}
struct TargetTask4ProfileSink {
    uint32_t records = 0;
    uint64_t bytes = 0;
    Task4ProfileBoundary boundary = Task4ProfileBoundary::dispatch;
    static bool emit(void* raw, const Task4ProfileBatch& scope, const GraphProfile& profile, const BatchResult& batch) noexcept {
        try {
            auto& self = *static_cast<TargetTask4ProfileSink*>(raw);
            const bool encoder = self.boundary == Task4ProfileBoundary::encoder_late8;
            require(self.records < 8192 && scope.command_buffer_id == self.records &&
                profile.mode == (encoder ? ProfileMode::encoder_boundary : ProfileMode::dispatch_boundary) && profile.kernels.size() == batch.dispatches &&
                scope.encoder_count == (encoder ? batch.dispatches : 1) &&
                task4_profile_stderr_writable(), "invalid Task4 batch or stderr destination");
            const bool usable = profile.timings_valid && profile_calibration_valid(profile.gpu_tick_nanoseconds, profile.cpu_clock_mismatch_fraction);
            const auto dimensions = [](const auto& values) { Json::Array out; for (auto n : values) out.push_back(Json::integer(n)); return Json(std::move(out)); };
            Json::Array rows; rows.reserve(profile.kernels.size());
            for (const auto& k : profile.kernels) rows.push_back(Json(Json::Array{
                Json(task4_phase_name(k.task4_tag.phase)), number(k.task4_tag.layer), Json::integer(k.task4_tag.prefix), Json::integer(k.task4_tag.rows),
                Json::integer(k.task4_dispatch_index), Json(k.kernel), Json::integer(k.encoding), dimensions(k.shape),
                Json::integer(k.start), Json::integer(k.end), usable ? number(double(k.end - k.start) * profile.gpu_tick_nanoseconds) : Json(),
                Json(k.task4_grid_is_threadgroups ? "threadgroups" : "threads"), dimensions(k.task4_grid), dimensions(k.task4_threads_per_threadgroup),
                Json::integer(encoder ? k.task4_dispatch_index : 0)}));
            const auto record = Json(Json::Object{{"event", Json("task4_gpu_profile")}, {"backend", Json("tokspur")},
                {"command_buffer_id", Json::integer(scope.command_buffer_id)}, {"command_buffer_id_scope", Json("profile_window")},
                {"encoder_id", encoder ? Json() : Json::integer(0)}, {"encoder_count", Json::integer(scope.encoder_count)},
                {"counter_boundary", Json(encoder ? "encoder_boundary" : "dispatch_boundary")}, {"changed_encoder_topology", Json(encoder)},
                {"timing_scope", Json(encoder ? "intrusive_per_dispatch_encoder" : "intrusive_dispatch")},
                {"intrusive_serialized", Json(true)}, {"critical_path_qualified", Json(false)}, {"timings_valid", Json(usable)},
                {"gpu_tick_nanoseconds", usable ? number(profile.gpu_tick_nanoseconds) : Json()},
                {"cpu_clock_mismatch_fraction", std::isfinite(profile.cpu_clock_mismatch_fraction) ? number(profile.cpu_clock_mismatch_fraction) : Json()},
                {"cpu_clock_is_mach", Json(profile.cpu_clock_is_mach)}, {"counter_ticks_valid", Json(profile.task4_counter_ticks_valid)},
                {"calibration_gpu_start", Json::integer(profile.task4_calibration_gpu_start)}, {"calibration_gpu_end", Json::integer(profile.task4_calibration_gpu_end)},
                {"calibration_cpu_start", Json::integer(profile.task4_calibration_cpu_start)}, {"calibration_cpu_end", Json::integer(profile.task4_calibration_cpu_end)},
                {"calibration_steady_span_ns", std::isfinite(profile.task4_calibration_steady_span_ns) ? number(profile.task4_calibration_steady_span_ns) : Json()}, {"cpu_encode_ms", number(scope.encode_ms)},
                {"cpu_submit_ms", number(scope.submit_ms)}, {"cpu_wait_ms", number(scope.wait_ms)},
                {"batch_wall_ms", number(batch.wall_ms)}, {"batch_gpu_ms", number(batch.gpu_ms)}, {"dispatches", Json::integer(batch.dispatches)},
                {"node_index", Json()}, {"node_name", Json()}, {"node_op", Json()},
                {"dispatch_columns", Json(Json::Array{Json("phase"), Json("layer"), Json("prefix"), Json("rows"), Json("dispatch_index"), Json("kernel_name"), Json("encoding"), Json("shape"), Json("raw_gpu_start"), Json("raw_gpu_end"), Json("gpu_ns"), Json("grid_unit"), Json("grid"), Json("threads_per_threadgroup"), Json("encoder_id")})},
                {"ordered_dispatches", Json(std::move(rows))}}).dump() + "\n";
            struct stat status{};
            require(fstat(STDERR_FILENO, &status) == 0 && status.st_size >= 0 &&
                task4_profile_record_fits(uint64_t(status.st_size), record.size()) && task4_profile_record_fits(self.bytes, record.size()),
                "Task4 profile exceeds bounded stderr or its reserved diagnostic space");
            size_t at = 0;
            for (uint32_t attempt = 0; attempt < 64 && at < record.size(); ++attempt) {
                const auto n = write(STDERR_FILENO, record.data() + at, record.size() - at);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) return false;
                at += size_t(n);
            }
            if (at != record.size()) return false;
            ++self.records; self.bytes += record.size(); return true;
        } catch (...) { return false; }
    }
};
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
bool mtp_profile_stderr_writable() noexcept {
    struct stat status{}; const auto flags = fcntl(STDERR_FILENO, F_GETFL);
    return flags >= 0 && (flags & O_ACCMODE) != O_RDONLY &&
        fstat(STDERR_FILENO, &status) == 0 && S_ISREG(status.st_mode);
}
struct TargetMtpProfileSink {
    uint32_t records = 0;
    MtpProfileBoundary boundary = MtpProfileBoundary::dispatch;
    static bool emit(void* raw, const MtpProfileBatch& scope, const GraphProfile& profile, const BatchResult& batch) noexcept {
        try {
            auto& self = *static_cast<TargetMtpProfileSink*>(raw);
            require(self.records < mtp_profile_max_rounds * mtp_profile_max_batches_per_round &&
                profile.mode == (self.boundary == MtpProfileBoundary::dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary) &&
                profile.kernels.size() == batch.dispatches &&
                mtp_profile_stderr_writable(), "invalid MTP diagnostic record or stderr destination");
            const bool usable = profile.timings_valid && profile_calibration_valid(profile.gpu_tick_nanoseconds,
                profile.cpu_clock_mismatch_fraction);
            Json::Array ordered; ordered.reserve(profile.kernels.size());
            for (const auto& kernel : profile.kernels) {
                Json::Array shape; for (const auto value : kernel.shape) shape.push_back(Json::integer(value));
                ordered.push_back(Json(Json::Array{Json(kernel.kernel), Json::integer(kernel.encoding), Json(std::move(shape)),
                    Json::integer(kernel.start), Json::integer(kernel.end), usable ? number(kernel.milliseconds) : Json()}));
            }
            const auto record = Json(Json::Object{{"event", Json("mtp_gpu_profile")},
                {"round", Json::integer(scope.round)}, {"base", Json::integer(scope.base)},
                {"batch", Json::integer(scope.batch)}, {"phase", Json(mtp_profile_phase_name(scope.phase))},
                {"cpu_encode_ms", number(scope.encode_ms)}, {"cpu_submit_ms", number(scope.submit_ms)},
                {"cpu_wait_ms", number(scope.wait_ms)}, {"batch_wall_ms", number(batch.wall_ms)},
                {"batch_gpu_ms", number(batch.gpu_ms)}, {"prefill_profiled", Json(false)},
                {"timings_qualified", Json(false)}, {"sampling_boundary", Json(mtp_profile_boundary_name(self.boundary))},
                {"gpu_profile", profile_json(profile)},
                {"dispatch_columns", Json(Json::Array{Json("kernel"), Json("encoding"), Json("shape"),
                    Json("start_tick"), Json("end_tick"), Json("milliseconds")})},
                {"ordered_dispatches", Json(std::move(ordered))}}).dump() + "\n";
            require(record.size() <= 4'000'000, "MTP diagnostic record exceeds its byte bound");
            size_t written = 0;
            for (uint32_t attempt = 0; attempt < 64 && written < record.size(); ++attempt) {
                const auto count = write(STDERR_FILENO, record.data() + written, record.size() - written);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) return false;
                written += size_t(count);
            }
            if (written != record.size()) return false;
            ++self.records; return true;
        } catch (...) { return false; }
    }
};
struct TargetMtpProfileRound {
    MetalGraph& gpu; bool active;
    TargetMtpProfileRound(MetalGraph& graph, uint32_t base) : gpu(graph), active(graph.begin_mtp_profile_round(base)) {}
    ~TargetMtpProfileRound() { if (active) gpu.end_mtp_profile_round(); }
};
#endif
}
int target_cli(int argc, char** argv) {
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    std::optional<bool> native_raw_decode;
    MetalGraph::RawDecodeRoutes raw_decode_request_before{};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    bool q8_three_rows = false;
#endif
    enum class MtpLookupMode : uint8_t { off = 0, bypass = 1, hybrid = 2 };
    bool swiglu_fused = false, moe_three_union = false, candidate_opt_selected = false;
    Q8GemvRg4Mode q8_gemv_rg4 = Q8GemvRg4Mode::off;
    Q8ThreeMsgMode q8_three_msg = Q8ThreeMsgMode::off;
    MtpLookupMode mtp_lookup_mode = MtpLookupMode::off;
    uint32_t mtp_lookup_match = 4;
    PrefillM64Mode prefill_m64 = PrefillM64Mode::off;
    bool prefill_multiseg = false, prefill_swizzle = false, qsa_prefill_fused = false, hc_consumer_fused = false;
    bool ple_overlap = false, shared_prefix_cache_enabled = false;
    std::filesystem::path policy_path; auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    std::vector<std::filesystem::path> paths; std::set<std::string> seen; bool allow = false, dry = false, profiling = false;
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    std::optional<bool> operator_checkpoints; std::optional<std::filesystem::path> operator_checkpoint_path;
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    std::optional<bool> operator_hc_rows;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    std::optional<bool> operator_router_window;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    std::optional<bool> operator_hc_down_window;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    std::optional<bool> operator_qsa_sparse;
    std::optional<std::filesystem::path> qsa_route_attribution_path;
    TargetQsaRouteAttributionSink qsa_route_sink;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    std::optional<bool> operator_gdn_ported;
    std::optional<std::filesystem::path> gdn_route_attribution_path;
    std::optional<uint32_t> gdn_route_attribution_prefill_prefix;
    TargetGdnRouteAttributionSink gdn_route_sink;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    std::optional<bool> operator_hc_down_staged;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    std::optional<bool> operator_router_f32_batch;
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    std::optional<bool> operator_staged_dense_f32;
#endif
    TargetOperatorCheckpointSink operator_sink; uint32_t operator_requests = 0, operator_base_prefix = 512;
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
    std::optional<std::filesystem::path> route_attribution_path; TargetRouteAttributionSink route_sink;
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    std::optional<std::filesystem::path> route_lock_output, route_lock_reference; bool route_census_only = false, route_census_failed = false; TargetRouteLockSink route_lock_sink;
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    std::optional<std::filesystem::path> kernel_search_capture_path; TargetKernelSearchCaptureSink kernel_search_sink; uint32_t kernel_search_requests = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    std::optional<bool> task4_profile;
    std::optional<Task4ProfileBoundary> task4_profile_boundary;
#endif
    uint32_t small_gemv_splits = 0, block_gemv_formats = 0, known_token_capacity = 0; bool allow_kernel_experiments = false;
    uint32_t hc_q8_splits = 0;
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    std::optional<bool> grouped_prefill_ffn;
    std::optional<bool> prefill_layer_submissions;
    std::optional<bool> prefill_hc_residual_rows;
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    std::optional<GdnChunkMode> native_prefill_chunks;
    std::optional<bool> native_prefill_dense_f32;
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    bool native_prefill_dense_staged_f32 = false;
#endif
    bool native_prefill_dense_nax = false;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    bool native_prefill_final_logits = false;
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    std::optional<bool> native_prefill_hc_rows;
    std::array<uint64_t, 3> hc_rows_request_before{};
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    std::optional<bool> native_prefill_router_window;
    std::array<uint64_t, 4> router_window_request_before{};
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    std::optional<bool> native_prefill_hc_down_window;
    uint64_t hc_down_window_request_before = 0;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    std::optional<bool> native_qsa_sparse;
    std::array<uint64_t, 2> qsa_sparse_request_before{};
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    std::optional<bool> native_gdn_ported;
    std::optional<bool> native_gdn_prefill_ported;
    GdnPortedRoutes gdn_ported_request_before{};
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    bool native_prefill_hc_down_f32 = false;
    uint64_t hc_down_f32_request_before = 0;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    bool native_gdn_old_order_batch = false;
    uint64_t gdn_old_order_request_before = 0, gdn_old_order_layers = 0;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    std::optional<bool> native_prefill_grouped_f32;
    bool native_prefill_grouped_auto = false;
    bool native_prefill_grouped_nax = false;
    bool router_margin_audit_enabled = false;
    float router_margin_epsilon = 1.0e-4f;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    std::optional<bool> qsa_sparse;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    std::optional<bool> gdn_ported;
#endif
#endif
#endif
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
    std::optional<std::filesystem::path> logit_trace_path;
    uint32_t logit_trace_rows = 0;
#endif
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
    std::optional<PleGateTransform> ple_gate_transform{PleGateTransform::signed_root};
#else
    std::optional<PleGateTransform> ple_gate_transform;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
    std::optional<verification_cost::Manifest> verification_cost_manifest;
    size_t verification_cost_prefix = 0;
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
    std::optional<verification_cost::SampledManifest> sampled_cost_manifest;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    std::optional<QsaVerifyRowsMode> verification_qsa_rows;
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    std::optional<bool> known_qsa_rows;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    bool sampled_prefill_input8 = false;
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
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    bool qsa_stage_v8 = false;
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    bool qsa_stage_v8_present = false;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    std::optional<bool> qsa_channel_chronological;
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    std::optional<bool> kernel_qsa_f32;
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    std::optional<bool> kernel_grouped_fp16;
#endif
#endif
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    std::optional<bool> batch_phase_clock;
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    bool concurrent_dispatch = false, concurrent_selected = false, concurrent_state_proof = false;
#endif
    uint32_t checkpoint_slots = 0; uint64_t checkpoint_budget = 0; bool worker_broker = false, prompt_lookup = false;
    bool checkpoint_ssd = false; uint64_t checkpoint_ssd_budget = 8ull << 30; uint64_t checkpoint_ssd_floor = 20ull << 30;
    std::filesystem::path checkpoint_ssd_dir; std::string checkpoint_policy = "recompute-cost"; bool overlap_prefill = false;
    std::optional<std::filesystem::path> derived_pack_path;
    std::optional<uint32_t> derived_pack_auth_workers;
    std::string derived_pack_trust_mode = "full";
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::optional<std::filesystem::path> weight_lanes;
    std::optional<uint32_t> weight_lanes_auth_workers;
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    std::optional<bool> native_verify_nax;
    std::optional<bool> native_verify_nax_m8;
    std::optional<std::filesystem::path> target_m8_splits_path;
    std::optional<bool> native_verify_hc_gdn_fused;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    std::optional<bool> native_prefill_router_fast;
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::optional<std::filesystem::path> mtp_sidecar;
    std::optional<std::filesystem::path> dflash2_dir;
    DFlash2Precision dflash2_precision = DFlash2Precision::q8_0;
    bool dflash2_defer_context = false;
    bool dflash2_adaptive_budget = false;
    std::optional<std::filesystem::path> verify_cost_profile_path;
    std::optional<MeasuredCostProfile> verify_cost_profile;
    std::unique_ptr<MeasuredController> measured_controller;
    bool dflash2_auto_draft = false;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    CarrierDiagnosticMode carrier_mode = CarrierDiagnosticMode::off;
    std::string carrier_capture_path, carrier_identity_path;
    uint32_t carrier_actions = 4;
    bool carrier_forced_input = false, carrier_single_action = false, carrier_batch_only = false;
#endif
    std::optional<uint32_t> dflash2_max_drafts;
    std::optional<uint32_t> dflash2_vocab_slice;
    std::optional<bool> native_logit_topk;
    std::optional<bool> native_ple_async_gather;
    std::optional<bool> native_mtp_fast_round;
    std::string drafter_mode = "auto";
    bool mtp_lookahead_anchor = false, mtp_request_carry = false, native_mtp_draft_slice = false;
    bool mtp_depth_explicit = false, mtp_draft_attention_explicit = false, mtp_lookahead_explicit = false;
    uint32_t mtp_depth = 1;
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    std::optional<uint32_t> mtp_output_packet_limit;
#endif
    MtpDraftAttention mtp_draft_attention = MtpDraftAttention::serial;
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    uint32_t mtp_profile_rounds = 0;
    std::optional<MtpProfileBoundary> mtp_profile_boundary;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    bool worker_broker8 = false;
#endif
    uint64_t verification_budget = 0; bool allow_lookup_experiments = false;
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    uint32_t verification_max_tokens = 2;
    bool host_logit_views = false;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    bool verification8_branch_reuse = false;
#endif
    StackExecution execution = StackExecution::direct; bool allow_stack_experiments = false, allow_prefill_experiments = false, reuse_residual = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]); if (!option.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        require(seen.emplace(option).second && ++i < argc, "duplicate target option or missing value"); const std::string_view value(argv[i]);
        if (option == "--policy") policy_path = value;
        else if (option == "--derived-pack") derived_pack_path = std::filesystem::path(value);
        else if (option == "--derived-pack-auth-workers") {
            uint32_t workers = 0; const auto parsed = std::from_chars(value.data(), value.data() + value.size(), workers);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && workers &&
                workers <= derived_pack::max_authentication_workers, "derived pack authentication workers must be between 1 and 16");
            derived_pack_auth_workers = workers;
        }
        else if (option == "--derived-pack-trust-mode") {
            require(value == "full" || value == "local-seal", "derived pack trust mode must be full or local-seal");
            derived_pack_trust_mode = std::string(value);
        }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        else if (option == "--weight-lanes") weight_lanes = std::filesystem::path(value);
        else if (option == "--weight-lanes-auth-workers") {
            uint32_t workers = 0; const auto parsed = std::from_chars(value.data(), value.data() + value.size(), workers);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && workers &&
                workers <= lanes::sidecar_max_authentication_workers, "lane authentication workers must be between 1 and 16");
            weight_lanes_auth_workers = workers;
        }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        else if (option == "--mtp-sidecar") mtp_sidecar = std::filesystem::path(value);
        else if (option == "--dflash2-dir" || option == "--native-dflash2-dir") dflash2_dir = std::filesystem::path(value);
        else if (option == "--dflash2-precision" || option == "--native-dflash2-precision") {
            dflash2_precision = parse_dflash2_precision(value);
        }
        else if (option == "--native-verify-cost-profile") verify_cost_profile_path = std::filesystem::path(value);
        else if (option == "--native-dflash2-draft-policy") {
            require(value == "always" || value == "auto", "native DFlash2 draft policy must be always or auto");
            dflash2_auto_draft = value == "auto";
        }
        else if (option == "--native-prepared-validation") {
            require(value == "off" || value == "on", "native prepared validation must be off or on");
            configure_prepared_sampling(value == "on");
        }
        else if (option == "--native-dflash2-defer-context") {
            require(value == "off" || value == "on", "native DFlash2 deferred context must be off or on");
            dflash2_defer_context = value == "on";
        }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        else if (option == "--carrier-diagnostic-cost-tape") {
            MappedFile tape{std::filesystem::path(value)}; require(tape.size() <= 2048, "controller timing replay byte bound");
            const auto bytes = tape.bytes(); carrier_cost_replay.configure(parse_json({reinterpret_cast<const char*>(bytes.data()), bytes.size()}));
            tape.assert_unchanged();
        }
        else if (option == "--carrier-diagnostic-identity-out") carrier_identity_path = value;
        else if (option == "--carrier-diagnostic-capture") carrier_capture_path = value;
        else if (option == "--carrier-diagnostic-forced-input") {
            require(value == "off" || value == "on", "carrier forced input must be off or on"); carrier_forced_input = value == "on";
        }
        else if (option == "--carrier-diagnostic-single-action") {
            require(value == "off" || value == "on", "carrier single action must be off or on"); carrier_single_action = value == "on";
        }
        else if (option == "--carrier-diagnostic-batch-only") {
            require(value == "off" || value == "on", "carrier batch only must be off or on"); carrier_batch_only = value == "on";
        }
        else if (option == "--carrier-diagnostic-mode") carrier_mode = carrier_diagnostic_mode(value);
        else if (option == "--carrier-diagnostic-actions") {
            const auto result = std::from_chars(value.data(), value.data() + value.size(), carrier_actions);
            require(result.ec == std::errc{} && result.ptr == value.data() + value.size() && carrier_actions && carrier_actions <= 4,
                "carrier diagnostic actions must be 1..4");
        }
#endif
        else if (option == "--dflash2-verify-rows" || option == "--native-dflash2-verify-rows") {
            require(value == "all" || value == "auto", "native DFlash2 verify rows must be all or auto");
            dflash2_adaptive_budget = value == "auto";
        }
        else if (option == "--dflash2-max-drafts" || option == "--native-dflash2-max-drafts") {
            dflash2_max_drafts = parse_dflash2_max_drafts(value);
        }
        else if (option == "--dflash2-vocab-slice" || option == "--native-dflash2-vocab-slice") {
            dflash2_vocab_slice = parse_dflash2_vocab_slice(value);
        }
        else if (option == "--native-host-logit-views") {
            require(value == "off" || value == "on", "native host logit views must be off or on");
            host_logit_views = value == "on";
        }
        else if (option == "--logit-topk" || option == "--native-logit-topk") {
            require(value == "off" || value == "on", "native logit top-k must be off or on");
            native_logit_topk = value == "on";
        }
        else if (option == "--ple-async-gather" || option == "--native-ple-async-gather") {
            require(value == "off" || value == "on", "native PLE async gather must be off or on");
            native_ple_async_gather = value == "on";
        }
        else if (option == "--mtp-fast-round" || option == "--native-mtp-fast-round") {
            require(value == "off" || value == "on", "native MTP fast round must be off or on");
            native_mtp_fast_round = value == "on";
        }
        else if (option == "--drafter" || option == "--native-drafter") {
            require(value == "off" || value == "mtp" || value == "dflash2", "drafter mode must be off, mtp or dflash2");
            drafter_mode = std::string(value);
        }
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        else if (option == "--mtp-profile-mode") {
            if (value == "dispatch-boundary") mtp_profile_boundary = MtpProfileBoundary::dispatch;
            else if (value == "encoder-boundary") mtp_profile_boundary = MtpProfileBoundary::encoder;
            else throw FormatError("MTP profile mode must be dispatch-boundary or encoder-boundary");
        }
        else if (option == "--mtp-profile-rounds") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), mtp_profile_rounds);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                mtp_profile_rounds && mtp_profile_rounds <= mtp_profile_max_rounds, "MTP profile rounds must be1..4");
        }
#endif
        else if (option == "--mtp-draft-attention") {
            if (value == "serial") mtp_draft_attention = MtpDraftAttention::serial;
            else if (value == "tiled128") mtp_draft_attention = MtpDraftAttention::tiled128;
            else if (value == "tiled256") mtp_draft_attention = MtpDraftAttention::tiled256;
            else if (value == "split16") mtp_draft_attention = MtpDraftAttention::split16;
            else throw FormatError("MTP draft attention must be serial, tiled128, tiled256 or split16");
            mtp_draft_attention_explicit = true;
        }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        else if (option == "--mtp-depth") {
            mtp_depth = parse_native_speculative_depth(value);
            mtp_depth_explicit = true;
        }
#endif
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        else if (option == "--mtp-output-packet") {
            require(value == "8" || value == "64", "native MTP output packet must be 8 or 64");
            mtp_output_packet_limit = value == "8" ? 8 : 64;
        }
#endif
        else if (option == "--mtp-request-carry") {
            require(value == "on" || value == "off", "native MTP request carry must be off or on");
            mtp_request_carry = value == "on";
        }
        else if (option == "--mtp-lookahead-anchor") {
            require(value == "on" || value == "off", "native MTP lookahead anchor must be off or on");
            mtp_lookahead_anchor = value == "on";
            mtp_lookahead_explicit = true;
        }
        else if (option == "--native-mtp-draft-slice") {
            require(value == "on" || value == "off", "native MTP draft slice requires off/on");
            native_mtp_draft_slice = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        else if (option == "--verify-nax" || option == "--native-verify-nax") {
            require(value == "on" || value == "off", "native verify NAX requires on/off");
            native_verify_nax = value == "on";
        }
        else if (option == "--native-verify-m8-splits") target_m8_splits_path = std::filesystem::path(value);
        else if (option == "--verify-nax-m8" || option == "--native-verify-nax-m8") {
            require(value == "on" || value == "off", "native verify NAX m8 requires on/off");
            native_verify_nax_m8 = value == "on";
        }
        else if (option == "--verify-hc-gdn-fused" || option == "--native-verify-hc-gdn-fused") {
            require(value == "on" || value == "off", "native verify HC+GDN fused requires on/off");
            native_verify_hc_gdn_fused = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
        else if (option == "--prefill-router-fast" || option == "--native-prefill-router-fast") {
            require(value == "on" || value == "off", "native prefill router fast requires on/off");
            native_prefill_router_fast = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
        else if (option == "--verification-cost-manifest") verification_cost_manifest = verification_cost::load(std::filesystem::path(value));
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
        else if (option == "--sampled-verification-cost-manifest") sampled_cost_manifest = verification_cost::load_sampled_current(std::filesystem::path(value));
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        else if (option == "--verification-qsa-rows") verification_qsa_rows = qsa_verify_rows_mode(value);
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        else if (option == "--known-qsa-rows") {
            require(value == "off" || value == "q8", "known QSA rows must be off or q8");
            known_qsa_rows = value == "q8";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        else if (option == "--sampled-prefill-input8") {
            require(value == "off" || value == "on", "sampled input8 must be off or on");
            sampled_prefill_input8 = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
        else if (option == "--private-logit-trace-dir") { logit_trace_path = std::filesystem::path(value); }
        else if (option == "--private-logit-trace-rows") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), logit_trace_rows);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                logit_trace_rows && logit_trace_rows <= 1024, "private logit trace rows must be1..1024");
        }
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        else if (option == "--concurrent-dispatch") {
            require(value == "off" || value == "on", "private concurrent dispatch must be off or on");
            concurrent_dispatch = value == "on"; concurrent_selected = true;
        }
        else if (option == "--concurrent-state-proof") {
            require(value == "off" || value == "on", "private concurrent state proof must be off or on");
            concurrent_state_proof = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
        else if (option == "--ple-gate-transform") {
            if (value == "plain") ple_gate_transform = PleGateTransform::plain;
            else if (value == "signed-root") ple_gate_transform = PleGateTransform::signed_root;
            else throw FormatError("PLE gate transform must be plain or signed-root");
        }
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        else if (option == "--hc-exact") {
            if (value == "off") hc_exact = HcExactMode::off;
            else if (value == "scalar") hc_exact = HcExactMode::scalar;
            else if (value == "combined") hc_exact = HcExactMode::combined;
            else throw FormatError("private exact-HC mode must be off, scalar or combined");
        }
#endif
        else if (option == "--library") library = value;
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        else if (option == "--qsa-stage-v8") {
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            require(!qsa_stage_v8_present, "duplicate private QSA stage-v8 selector");
#endif
            require(value == "off" || value == "on", "private QSA stage-v8 mode must be off or on"); qsa_stage_v8 = value == "on";
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            qsa_stage_v8_present = true;
#endif
        }
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        else if (option == "--qsa-channel-chronological") {
            qsa_channel_chronological = qsa_channel_mode(value);
            require(qsa_channel_chronological.has_value(), "private chronological QSA mode must be off or on");
        }
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
        else if (option == "--batch-phase-clock") {
            batch_phase_clock = batch_phase_clock_mode(value);
            require(batch_phase_clock.has_value(), "private batch phase clock must be off or on");
        }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        else if (option == "--wide-q8-exact") {
            require(value == "off" || value == "on", "private wide-Q8 mode must be off or on"); wide_q8_exact = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        else if (option == "--wide-q8-many") {
            require(value == "off" || value == "on", "private wide-Q8-many mode must be off or on"); wide_q8_many = value == "on";
        }
#endif
        else if (option == "--worker-broker") { require(value == "true" || value == "false", "expected explicit worker broker boolean"); worker_broker = value == "true"; }
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        else if (option == "--worker-broker8") { require(value == "true" || value == "false", "expected explicit branch8 broker boolean"); worker_broker8 = value == "true"; }
#endif
        else if (option == "--prompt-lookup") { require(value == "true" || value == "false", "expected explicit prompt lookup boolean"); prompt_lookup = value == "true"; }
        else if (option == "--allow-prompt-lookup-experiments") { require(value == "true" || value == "false", "expected explicit lookup experiment boolean"); allow_lookup_experiments = value == "true"; }
        else if (option == "--verification-budget-bytes") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), verification_budget);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid combined lookup verification budget");
        }
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        else if (option == "--verification-max-tokens") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), verification_max_tokens);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                (verification_max_tokens == 2 || verification_max_tokens == 8), "private verification depth must be 2 or 8");
        }
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        else if (option == "--verification8-branch-reuse") {
            require(value == "true" || value == "false", "expected explicit branch-verifier boolean");
            verification8_branch_reuse = value == "true";
        }
#endif
        else if (option == "--checkpoint-slots") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), checkpoint_slots);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid checkpoint slot count");
        }
        else if (option == "--checkpoint-budget-bytes") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), checkpoint_budget);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid combined checkpoint byte budget");
        }
        else if (option == "--checkpoint-ssd") {
            require(value == "true" || value == "false" || value == "on" || value == "off", "expected explicit checkpoint-ssd boolean");
            checkpoint_ssd = (value == "true" || value == "on");
        }
        else if (option == "--checkpoint-ssd-budget-bytes") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), checkpoint_ssd_budget);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid checkpoint ssd budget");
        }
        else if (option == "--checkpoint-ssd-floor-bytes") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), checkpoint_ssd_floor);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid checkpoint ssd floor");
        }
        else if (option == "--checkpoint-ssd-dir") {
            checkpoint_ssd_dir = std::filesystem::path(value);
        }
        else if (option == "--checkpoint-policy") {
            require(value == "recompute-cost" || value == "lru", "checkpoint policy requires recompute-cost or lru");
            checkpoint_policy = std::string(value);
        }
        else if (option == "--overlap-prefill") {
            require(value == "true" || value == "false" || value == "on" || value == "off", "overlap prefill requires boolean");
            overlap_prefill = (value == "true" || value == "on");
        }
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        else if (option == "--native-gdn-old-order-batch") {
            require(value == "on" || value == "off", "native old-order GDN requires off/on"); native_gdn_old_order_batch = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        else if (option == "--native-prefill-hc-down-f32") {
            require(value == "on" || value == "off", "native HC down batching requires off/on"); native_prefill_hc_down_f32 = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
        else if (option == "--native-prefill-hc-rows") {
            require(value == "on" || value == "off", "native HC rows require off/on"); native_prefill_hc_rows = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
        else if (option == "--native-prefill-router-window") {
            require(value == "on" || value == "off", "native router window requires off/on"); native_prefill_router_window = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
        else if (option == "--native-prefill-hc-down-window") {
            require(value == "on" || value == "off", "native HC down window requires off/on"); native_prefill_hc_down_window = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
        else if (option == "--native-qsa-sparse") {
            require(value == "on" || value == "off", "native sparse QSA requires off/on"); native_qsa_sparse = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
        else if (option == "--native-gdn-ported") {
            require(value == "on" || value == "off", "native ported GDN requires off/on"); native_gdn_ported = value == "on";
        }
        else if (option == "--native-gdn-prefill-ported") {
            require(value == "on" || value == "off", "native prefill-only ported GDN requires off/on"); native_gdn_prefill_ported = value == "on";
        }
#endif
        else if (option == "--native-prefill-final-logits") {
            require(value == "on" || value == "off", "native final-input logits require off/on"); native_prefill_final_logits = value == "on";
        }
#endif
        else if (option == "--native-prefill-dense") {
            require(value == "f16" || value == "f32" || value == "nax"
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
                || value == "f32-staged"
#endif
                , "native prefill dense requires a compiled f16/f32/nax backend"); native_prefill_dense_f32 = value != "f16";
            native_prefill_dense_nax = value == "nax";
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            native_prefill_dense_staged_f32 = value == "f32-staged";
#endif
        }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        else if (option == "--native-prefill-grouped") {
            require(value == "serial" || value == "f32" || value == "auto" || value == "nax", "native prefill grouped requires serial, f32, auto or nax");
            native_prefill_grouped_nax = value == "nax";
            if (value == "auto") {
                native_prefill_grouped_auto = true;
                native_prefill_grouped_f32 = false;
            } else if (value == "nax") {
                native_prefill_grouped_auto = false;
                native_prefill_grouped_f32 = false;
            } else {
                native_prefill_grouped_auto = false;
                native_prefill_grouped_f32 = value == "f32";
            }
        }
        else if (option == "--router-margin-epsilon") {
            size_t consumed = 0;
            router_margin_epsilon = std::stof(std::string(value), &consumed);
            require(consumed == value.size(), "router margin epsilon contains trailing text");
            require(std::isfinite(router_margin_epsilon) && router_margin_epsilon > 0.0f, "router margin epsilon must be positive finite");
        }
        else if (option == "--router-margin-audit") {
            require(value == "on" || value == "off", "router margin audit requires on/off");
            router_margin_audit_enabled = value == "on";
        }
#endif
        else if (option == "--native-prefill-chunks") {
            require(value == "off" || value == "fused" || value == "cs64", "native prefill chunks require off/fused/cs64");
            if (value == "off") native_prefill_chunks.reset(); else native_prefill_chunks = value == "fused" ? GdnChunkMode::fused : GdnChunkMode::cs64;
        }
#endif
        else if (option == "--prefill-grouped-ffn") {
            require(value == "on" || value == "off", "private grouped FFN selector requires on/off");
            grouped_prefill_ffn = value == "on";
        }
        else if (option == "--prefill-layer-submissions") {
            require(value == "on" || value == "off", "private prefill layer submissions require on/off");
            prefill_layer_submissions = value == "on";
        }
        else if (option == "--prefill-hc-residual-rows") {
            require(value == "on" || value == "off", "private HC residual row selector requires on/off");
            prefill_hc_residual_rows = value == "on";
        }
#endif
        else if (option == "--known-token-capacity") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), known_token_capacity);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                (known_token_capacity == 0 || known_token_capacity == 8
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
                 || known_token_capacity == 512 || known_token_capacity == 1024 || known_token_capacity == 2048
#endif
                ), "unsupported explicit known-token capacity");
        }
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        else if (option == "--q8-three-rows") {
            require(value == "on" || value == "off", "three-row Q8 requires on or off"); q8_three_rows = value == "on";
        }
#endif
        else if (option == "--swiglu-fused" || option == "--native-swiglu-fused") {
            require(value == "on" || value == "off", "fused SwiGLU requires on or off");
            swiglu_fused = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--q8-gemv-rg4" || option == "--native-q8-gemv-rg4") {
            if (value == "off") q8_gemv_rg4 = Q8GemvRg4Mode::off;
            else if (value == "on") q8_gemv_rg4 = Q8GemvRg4Mode::on;
            else if (value == "wide") q8_gemv_rg4 = Q8GemvRg4Mode::wide;
            else throw FormatError("4-row/SG Q8 GEMV requires off, on or wide");
            candidate_opt_selected = true;
        }
        else if (option == "--q8-three-msg" || option == "--native-q8-three-msg") {
            if (value == "off") q8_three_msg = Q8ThreeMsgMode::off;
            else if (value == "on") q8_three_msg = Q8ThreeMsgMode::on;
            else if (value == "wide") q8_three_msg = Q8ThreeMsgMode::wide;
            else throw FormatError("multi-SG three-row Q8 requires off, on or wide");
            candidate_opt_selected = true;
        }
        else if (option == "--moe-three-union" || option == "--native-moe-three-union") {
            require(value == "on" || value == "off", "unioned three-row MoE requires on or off");
            moe_three_union = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--mtp-lookup" || option == "--native-mtp-lookup") {
            if (value == "off") mtp_lookup_mode = MtpLookupMode::off;
            else if (value == "bypass") mtp_lookup_mode = MtpLookupMode::bypass;
            else if (value == "hybrid") mtp_lookup_mode = MtpLookupMode::hybrid;
            else throw FormatError("MTP lookup bypass requires off, bypass or hybrid");
            candidate_opt_selected = true;
        }
        else if (option == "--mtp-lookup-match" || option == "--native-mtp-lookup-match") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), mtp_lookup_match);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                mtp_lookup_match >= 2 && mtp_lookup_match <= 16, "MTP lookup match width must be 2..16");
            candidate_opt_selected = true;
        }
        else if (option == "--prefill-m64" || option == "--native-prefill-m64") {
            if (value == "off") prefill_m64 = PrefillM64Mode::off;
            else if (value == "on") prefill_m64 = PrefillM64Mode::on;
            else if (value == "short") prefill_m64 = PrefillM64Mode::short_only;
            else throw FormatError("prefill M64 mode requires off, on or short");
            candidate_opt_selected = true;
        }
        else if (option == "--prefill-multiseg" || option == "--native-prefill-multiseg") {
            require(value == "on" || value == "off", "prefill multiseg requires on or off");
            prefill_multiseg = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--prefill-swizzle" || option == "--native-prefill-swizzle") {
            require(value == "on" || value == "off", "prefill swizzle requires on or off");
            prefill_swizzle = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--qsa-prefill-fused" || option == "--native-qsa-prefill-fused") {
            require(value == "on" || value == "off", "fused QSA prefill prep requires on or off");
            qsa_prefill_fused = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--hc-consumer-fused" || option == "--native-hc-consumer-fused") {
            require(value == "on" || value == "off", "fused HC prefill consumer requires on or off");
            hc_consumer_fused = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--ple-overlap" || option == "--native-ple-overlap") {
            require(value == "on" || value == "off", "PLE chunk overlap requires on or off");
            ple_overlap = value == "on"; candidate_opt_selected = true;
        }
        else if (option == "--shared-prefix-cache" || option == "--native-shared-prefix-cache") {
            require(value == "on" || value == "off", "shared prefix cache requires on or off");
            shared_prefix_cache_enabled = value == "on"; candidate_opt_selected = true;
        }
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        else if (option == "--raw-decode") {
            require(value == "on" || value == "off", "raw routed decode requires on or off"); native_raw_decode = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        else if (option == "--qsa-sparse") {
            require(value == "on" || value == "off", "sparse QSA requires on or off"); qsa_sparse = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        else if (option == "--gdn-ported") {
            require(value == "on" || value == "off", "ported GDN requires on or off"); gdn_ported = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        else if (option == "--operator-checkpoints") { require(value == "on" || value == "off", "operator checkpoints require explicit on or off"); operator_checkpoints = value == "on"; }
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        else if (option == "--operator-staged-dense-f32") { require(value == "on" || value == "off", "operator staged dense requires explicit on or off"); operator_staged_dense_f32 = value == "on"; }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
        else if(option=="--operator-router-f32-batch"){require(value=="on"||value=="off","operator router FP32 batch requires explicit on or off");operator_router_f32_batch=value=="on";}
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
        else if(option=="--operator-hc-rows"){require(value=="on"||value=="off","operator HC rows require explicit on or off");operator_hc_rows=value=="on";}
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
        else if(option=="--operator-router-window"){require(value=="on"||value=="off","operator router window requires explicit on or off");operator_router_window=value=="on";}
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
        else if(option=="--operator-hc-down-window"){require(value=="on"||value=="off","operator HC down window requires explicit on or off");operator_hc_down_window=value=="on";}
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        else if(option=="--operator-qsa-sparse"){require(value=="on"||value=="off","operator sparse QSA requires explicit on or off");operator_qsa_sparse=value=="on";}
        else if(option=="--qsa-route-attribution-dir")qsa_route_attribution_path=std::filesystem::path(value);
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        else if (option == "--operator-gdn-ported") { require(value == "on" || value == "off", "operator ported GDN requires explicit on or off"); operator_gdn_ported = value == "on"; }
        else if (option == "--gdn-route-attribution-dir") gdn_route_attribution_path = std::filesystem::path(value);
        else if (option == "--gdn-route-attribution-prefill-prefix") {
            uint32_t pfx = 0; const auto parsed = std::from_chars(value.data(), value.data() + value.size(), pfx);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && pfx <= 65024 && pfx % 512 == 0,
                "GDN route attribution prefill prefix requires a 512-aligned chunk offset");
            gdn_route_attribution_prefill_prefix = pfx;
        }
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
        else if (option == "--operator-hc-down-staged") { require(value == "on" || value == "off", "operator HC down staged requires explicit on or off"); operator_hc_down_staged = value == "on"; }
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        else if (option == "--route-attribution-dir") route_attribution_path=std::filesystem::path(value);
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        else if(option=="--route-lock-output")route_lock_output=std::filesystem::path(value);
        else if(option=="--route-census-output"){route_lock_output=std::filesystem::path(value);route_census_only=true;}
        else if(option=="--route-lock-reference")route_lock_reference=std::filesystem::path(value);
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
        else if (option == "--kernel-search-capture-dir") kernel_search_capture_path = std::filesystem::path(value);
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
        else if (option == "--kernel-qsa-f32") { require(value == "on" || value == "off", "QSA step-2 candidate requires explicit on or off"); kernel_qsa_f32 = value == "on"; }
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
        else if (option == "--kernel-grouped-fp16") { require(value == "on" || value == "off", "grouped step-2 requires explicit on or off"); kernel_grouped_fp16 = value == "on"; }
#endif
#endif
        else if (option == "--operator-checkpoints-path") { operator_checkpoint_path = std::filesystem::path(value); require(operator_checkpoint_path->is_absolute(), "operator checkpoint path must be absolute"); }
#endif
        else if (option == "--stack-execution") {
            if (value == "direct") execution = StackExecution::direct;
            else if (value == "indirect-compatible") execution = StackExecution::indirect_compatible;
            else if (value == "recurrent-replay") execution = StackExecution::recurrent_replay;
            else throw FormatError("invalid stack execution mode");
        }
        else if (option == "--hc-q8-splits") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), hc_q8_splits);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid HC Q8 split count");
        }
        else if (option == "--small-gemv-splits") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), small_gemv_splits);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                (small_gemv_splits == 0 || small_gemv_splits == 1 || small_gemv_splits == 4 || small_gemv_splits == 8 || small_gemv_splits == 16 || small_gemv_splits == 32), "invalid small GEMV split count");
        }
        else if (option == "--block-gemv-formats") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), block_gemv_formats);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && block_gemv_formats <= 3, "invalid block GEMV format mask");
        }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        else if (option == "--task4-profile") {
            require(value == "off" || value == "on", "Task4 profile must be explicitly off or on");
            task4_profile = value == "on";
        }
        else if (option == "--task4-profile-boundary") {
            require(value == "dispatch" || value == "encoder-late8", "Task4 boundary requires dispatch or explicit encoder-late8");
            task4_profile_boundary = value == "dispatch" ? Task4ProfileBoundary::dispatch : Task4ProfileBoundary::encoder_late8;
        }
#endif
        else if (option == "--allow-full-model" || option == "--preflight-only" || option == "--profile" || option == "--allow-kernel-experiments" || option == "--allow-stack-experiments" || option == "--allow-prefill-experiments" || option == "--reuse-residual-projections") {
            require(value == "true" || value == "false", "expected explicit target boolean");
            if (option == "--allow-full-model") allow = value == "true"; else if (option == "--profile") profiling = value == "true";
            else if (option == "--allow-kernel-experiments") allow_kernel_experiments = value == "true";
            else if (option == "--allow-prefill-experiments") allow_prefill_experiments = value == "true";
            else if (option == "--reuse-residual-projections") reuse_residual = value == "true";
            else if (option == "--allow-stack-experiments") allow_stack_experiments = value == "true"; else dry = value == "true";
        } else throw FormatError("unknown target option");
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (drafter_mode == "off") {
        require(!mtp_sidecar.has_value() && !dflash2_dir.has_value(), "drafter off refuses sidecar or DFlash2 directory");
    } else if (drafter_mode == "mtp") {
        require(mtp_sidecar.has_value() && !dflash2_dir.has_value(), "drafter mtp requires --mtp-sidecar without --dflash2-dir");
    } else if (drafter_mode == "dflash2") {
        require(dflash2_dir.has_value() && !mtp_sidecar.has_value(), "drafter dflash2 requires --dflash2-dir without --mtp-sidecar");
    }
    if (dflash2_dir.has_value()) {
        require(!mtp_sidecar.has_value() && !native_mtp_draft_slice, "DFlash2 is mutually exclusive with MTP sidecar and draft slice");
        if (!mtp_depth_explicit) mtp_depth = 7;
        if (!mtp_draft_attention_explicit) mtp_draft_attention = MtpDraftAttention::split16;
        if (!mtp_lookahead_explicit) mtp_lookahead_anchor = true;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (!native_verify_nax.has_value()) native_verify_nax = true;
        if (native_verify_nax.value_or(false) && !native_verify_nax_m8.has_value()) native_verify_nax_m8 = true;
#endif
    }
    const bool has_speculative = mtp_sidecar.has_value() || dflash2_dir.has_value();
    require(!host_logit_views || has_speculative, "native host logit views require a sampled speculative verifier");
    require(!native_logit_topk.value_or(false) || has_speculative, "native logit top-k requires a sampled speculative verifier");
    require(!native_ple_async_gather.value_or(false) || has_speculative, "native PLE async gather requires a sampled speculative verifier");
    require(!native_mtp_fast_round.value_or(false) || mtp_sidecar.has_value(), "native MTP fast round requires the native MTP sidecar");
    require(!dflash2_defer_context || dflash2_dir.has_value(), "native DFlash2 deferred context requires DFlash2");
    require(!dflash2_adaptive_budget || dflash2_dir.has_value(), "native DFlash2 verify rows auto requires DFlash2");
    require(!dflash2_max_drafts.has_value() || dflash2_dir.has_value(), "native DFlash2 max drafts requires DFlash2");
    require(!dflash2_vocab_slice.has_value() || dflash2_dir.has_value(), "native DFlash2 vocab slice requires DFlash2");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (native_verify_nax_m8 == true && !native_verify_nax.has_value()) native_verify_nax = true;
    require(!native_verify_nax_m8.value_or(false) || native_verify_nax.value_or(false),
        "native verify NAX m8 requires verify NAX on");
    require(!native_verify_hc_gdn_fused.value_or(false) || native_verify_nax.value_or(false),
        "native verify HC+GDN fused requires --verify-nax on");
#endif
    const bool scoped_spec_ok = ((mtp_depth == 2 || mtp_depth == 3) && mtp_request_carry && mtp_draft_attention == MtpDraftAttention::tiled128) ||
        (mtp_depth == 7 && mtp_request_carry && mtp_draft_attention == MtpDraftAttention::split16);
    const bool scoped_spec_carry = (mtp_depth == 2 || mtp_depth == 3 || mtp_depth == 7) && mtp_request_carry;
    require(!checkpoint_ssd || (checkpoint_ssd_budget > 0 && checkpoint_ssd_budget <= maximum_checkpoint_ssd_budget &&
        (checkpoint_policy == "recompute-cost" || checkpoint_policy == "lru")),
        "SSD checkpoint budget must be positive and within the 32 GiB ceiling");
#else
    require(!checkpoint_ssd, "SSD checkpoints require the native MTP target build");
    (void)checkpoint_ssd_budget; (void)checkpoint_ssd_floor; (void)checkpoint_ssd_dir; (void)checkpoint_policy;
#endif
#ifndef TOKSPUR_PRIVATE_PREFILL_CHUNK
    require(!overlap_prefill, "overlap prefill requires known-chunk prefill support");
#endif
    require(!policy_path.empty() && (dry || allow), "target requires explicit policy and --allow-full-model true, or --preflight-only true");
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
    require(logit_trace_path.has_value() == (logit_trace_rows != 0), "private logit trace requires explicit directory and row bound together");
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    final_input_trace_preparation(native_prefill_final_logits, logit_trace_path.has_value() || logit_trace_rows != 0);
#endif
    require(!logit_trace_path || (!dry && !profiling && execution == StackExecution::direct &&
        !prompt_lookup && !worker_broker && !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "private logit trace requires direct target-only semantics, no broker/profiling/request overrides");
#endif
    require(!profiling || execution == StackExecution::direct, "indirect stack experiments do not support GPU profiling");
    require(execution != StackExecution::recurrent_replay || !allow_kernel_experiments,
        "immutable recurrent replay refuses request-time kernel overrides; choose the kernel policy at startup");
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    const bool mtp_chunked_prefill = has_speculative && native_prefill_chunks.has_value();
    require(!mtp_chunked_prefill || ((known_token_capacity == 512 || known_token_capacity == 1024 || known_token_capacity == 2048) && reuse_residual &&
        native_prefill_dense_f32.has_value() && !weight_lanes && !known_qsa_rows.has_value()),
        "native MTP input512 requires explicit raw dense precision, fixed verifier reuse and no lane or known-QSA selector");
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    hc_down_f32_preparation(native_prefill_hc_down_f32, mtp_chunked_prefill, native_prefill_dense_f32 == true, (native_prefill_grouped_f32 == true || native_prefill_grouped_auto),
        (native_prefill_chunks == GdnChunkMode::fused || native_prefill_chunks == GdnChunkMode::cs64), scoped_spec_carry, allow_prefill_experiments || allow_kernel_experiments || allow_stack_experiments);
#endif
    const bool grouped_prefill_ok =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        (native_prefill_grouped_f32 == true || native_prefill_grouped_auto || native_prefill_grouped_nax);
#else
        false;
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    const bool dense_staged_or_nax = native_prefill_dense_staged_f32 || native_prefill_dense_nax;
    require(!native_prefill_dense_staged_f32 || (mtp_chunked_prefill && (native_prefill_chunks == GdnChunkMode::fused || native_prefill_chunks == GdnChunkMode::cs64) &&
        grouped_prefill_ok && !native_prefill_hc_down_f32 && !weight_lanes && execution == StackExecution::direct &&
        scoped_spec_ok &&
        !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "staged FP32 dense requires fixed raw fused/cs64 input512 grouped-FP32/auto/nax MTP2/DFlash2 and HC down off");
    require(!native_prefill_dense_nax || (native_prefill_chunks.has_value() &&
        (native_prefill_chunks == GdnChunkMode::fused || native_prefill_chunks == GdnChunkMode::cs64) &&
        (!has_speculative || (mtp_chunked_prefill && grouped_prefill_ok && !native_prefill_hc_down_f32 &&
        !weight_lanes && execution == StackExecution::direct && scoped_spec_ok))),
        "NAX dense prefill requires fused/cs64 chunked prefill and compatible MTP2/DFlash2 settings");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    require(!native_prefill_grouped_nax || (native_prefill_chunks.has_value() &&
        (native_prefill_chunks == GdnChunkMode::fused || native_prefill_chunks == GdnChunkMode::cs64) &&
        (!has_speculative || (mtp_chunked_prefill && dense_staged_or_nax && !native_prefill_hc_down_f32 &&
        !weight_lanes && execution == StackExecution::direct && scoped_spec_ok))),
        "NAX grouped prefill requires fused/cs64 chunked prefill and compatible MTP2/DFlash2 settings");
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require(!native_prefill_dense_staged_f32 || !operator_staged_dense_f32.has_value() || *operator_staged_dense_f32,
        "native staged dense conflicts with explicit operator old-MPP selection");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    require(!native_prefill_hc_rows.has_value() || (mtp_chunked_prefill && execution == StackExecution::direct && scoped_spec_carry), "HC rows require fixed input512 MTP2/DFlash2");
    require(!native_prefill_hc_rows.value_or(false) || (dense_staged_or_nax && grouped_prefill_ok && !native_prefill_hc_down_f32), "HC rows require staged dense FP32 or NAX and scalar HC down");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require(!native_prefill_hc_rows.has_value() || !operator_hc_rows.has_value(), "native and operator HC rows selectors are exclusive");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    require(!native_prefill_router_window.has_value() || (mtp_chunked_prefill && execution == StackExecution::direct && scoped_spec_carry), "router window requires fixed input512 MTP2/DFlash2");
    require(!native_prefill_router_window.value_or(false) || (dense_staged_or_nax && native_prefill_hc_rows == true && grouped_prefill_ok && !native_prefill_hc_down_f32), "router window requires staged dense FP32 or NAX, exact HC rows and scalar HC down");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require(!native_prefill_router_window.has_value() || !operator_router_window.has_value(), "native and operator router window selectors are exclusive");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    require(!native_prefill_hc_down_window.has_value() || (mtp_chunked_prefill && execution == StackExecution::direct && scoped_spec_carry), "HC down window requires fixed input512 MTP2/DFlash2");
    require(!native_prefill_hc_down_window.value_or(false) || (dense_staged_or_nax && native_prefill_hc_rows == true && native_prefill_router_window == true && grouped_prefill_ok && !native_prefill_hc_down_f32), "HC down window requires staged dense FP32 or NAX, exact HC rows, exact router window and reassociating HC down off");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require(!native_prefill_hc_down_window.has_value() || !operator_hc_down_window.has_value(), "native and operator HC down window selectors are exclusive");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    require(!native_qsa_sparse.has_value() || (mtp_chunked_prefill && execution == StackExecution::direct && scoped_spec_carry &&
        qsa_stage_v8 && qsa_channel_chronological == true && !qsa_sparse.has_value()),
        "native sparse QSA requires fixed input512 chronological MTP2/DFlash2 and no standalone selector");
    require(!native_qsa_sparse.value_or(false) || (dense_staged_or_nax && native_prefill_hc_rows == true &&
        native_prefill_router_window == true && native_prefill_hc_down_window == true && grouped_prefill_ok && !native_prefill_hc_down_f32),
        "native sparse QSA requires staged dense FP32 or NAX, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require(!native_qsa_sparse.has_value() || (!operator_qsa_sparse.has_value() && !qsa_route_attribution_path),
        "native and operator sparse QSA selectors are exclusive");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    require(!qsa_sparse.has_value() || (!has_speculative && execution == StackExecution::direct &&
        qsa_stage_v8 && qsa_channel_chronological == true && !profiling && !weight_lanes && !allow_kernel_experiments && !allow_stack_experiments
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        && !operator_checkpoints.has_value() && !operator_qsa_sparse.has_value() && !qsa_route_attribution_path
#endif
        ), "standalone sparse QSA requires direct unprofiled chronological non-MTP execution");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    require(!native_gdn_ported.has_value() || (mtp_chunked_prefill && execution == StackExecution::direct && scoped_spec_carry &&
        qsa_stage_v8 && qsa_channel_chronological == true && !gdn_ported.has_value()),
        "native ported GDN requires fixed input512 chronological MTP2/DFlash2 and no standalone selector");
    require(!native_gdn_ported.value_or(false) || (dense_staged_or_nax && native_prefill_hc_rows == true &&
        native_prefill_router_window == true && native_prefill_hc_down_window == true && grouped_prefill_ok && !native_prefill_hc_down_f32),
        "native ported GDN requires staged dense FP32 or NAX, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
    require(!native_gdn_prefill_ported.has_value() || (mtp_chunked_prefill && execution == StackExecution::direct && scoped_spec_carry &&
        qsa_stage_v8 && qsa_channel_chronological == true && !gdn_ported.has_value() && !native_gdn_ported.value_or(false)),
        "native prefill-only ported GDN requires fixed input512 chronological MTP2/DFlash2 with decode ported GDN off");
    require(!native_gdn_prefill_ported.value_or(false) || (dense_staged_or_nax && native_prefill_hc_rows == true &&
        native_prefill_router_window == true && native_prefill_hc_down_window == true && grouped_prefill_ok && !native_prefill_hc_down_f32),
        "native prefill-only ported GDN requires staged dense FP32 or NAX, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require((!native_gdn_ported.has_value() && !native_gdn_prefill_ported.has_value()) || (!operator_gdn_ported.has_value() && !gdn_route_attribution_path),
        "native and operator ported GDN selectors are exclusive");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    require(!gdn_ported.has_value() || (!has_speculative && execution == StackExecution::direct &&
        !profiling && !weight_lanes && !allow_kernel_experiments && !allow_stack_experiments
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        && !operator_checkpoints.has_value() && !operator_gdn_ported.has_value() && !gdn_route_attribution_path
#endif
        ), "standalone ported GDN requires direct unprofiled non-MTP execution");
#endif
    require(!native_prefill_final_logits || (mtp_chunked_prefill && scoped_spec_carry &&
        !prompt_lookup && !worker_broker && !checkpoint_slots), "final-input logits require fixed input512 MTP2/DFlash2 without branch or lookup state");
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    require(!native_gdn_old_order_batch || (mtp_chunked_prefill && native_prefill_chunks == GdnChunkMode::fused &&
        native_prefill_dense_f32 == true && native_prefill_grouped_f32 == true && mtp_depth == 2 && mtp_request_carry),
        "old-order GDN requires explicit fixed f32/f32 input512 fused MTP2");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    require(operator_checkpoints.has_value() == operator_checkpoint_path.has_value(), "operator checkpoints require a paired private path and explicit mode");
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    require(!operator_staged_dense_f32.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        native_prefill_dense_f32 == true && native_prefill_grouped_f32 == true && !native_prefill_hc_down_f32 && !weight_lanes),
        "operator staged dense option requires explicit operator diagnostics, raw FP32/FP32 and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    require(!operator_router_f32_batch.has_value()||(operator_checkpoints.has_value()&&operator_checkpoint_path.has_value()&&native_prefill_dense_f32==true&&native_prefill_grouped_f32==true&&!weight_lanes),"operator router option requires explicit raw FP32/FP32 operator diagnostics");
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    require(!operator_hc_rows.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && native_prefill_grouped_f32 == true && !native_prefill_hc_down_f32 && !weight_lanes), "operator HC rows require explicit staged FP32/FP32 diagnostics and scalar HC down");
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    require(!operator_router_window.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && (native_prefill_hc_rows == true || operator_hc_rows == true) &&
        native_prefill_grouped_f32 == true && !native_prefill_hc_down_f32 && !weight_lanes && !operator_router_f32_batch.value_or(false)),
        "operator router window requires explicit staged FP32/FP32 and exact HC row diagnostics with reassociating router and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    require(!operator_hc_down_window.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && (native_prefill_hc_rows == true || operator_hc_rows == true) &&
        native_prefill_grouped_f32 == true && !native_prefill_hc_down_f32 && !weight_lanes && !operator_router_f32_batch.value_or(false)
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
        && !operator_hc_down_staged.value_or(false)
#endif
        ),
        "operator HC down window requires explicit staged FP32/FP32 and exact HC row diagnostics with reassociating router and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    require(!operator_qsa_sparse.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && (native_prefill_hc_rows == true || operator_hc_rows == true) &&
        native_prefill_grouped_f32 == true && !native_prefill_hc_down_f32 && !weight_lanes && !operator_router_f32_batch.value_or(false) &&
        qsa_stage_v8 && qsa_channel_chronological == true && mtp_chunked_prefill
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
        && !operator_hc_down_staged.value_or(false)
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        && !route_attribution_path
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        && !route_lock_output
#endif
        ), "operator sparse QSA requires the fixed exact prefill and chronological QSA diagnostic carrier with other reassociating candidates off");
    require(!qsa_route_attribution_path || (operator_checkpoints == true && operator_qsa_sparse == false &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && (native_prefill_hc_rows == true || operator_hc_rows == true)),
        "QSA route attribution requires operator checkpoints on and explicit operator sparse QSA off");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    require(!gdn_route_attribution_prefill_prefix.has_value() || gdn_route_attribution_path.has_value(),
        "GDN route attribution prefill prefix requires an explicit attribution directory");
    require(!operator_gdn_ported.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && (native_prefill_hc_rows == true || operator_hc_rows == true) &&
        native_prefill_grouped_f32 == true && !native_prefill_hc_down_f32 && !weight_lanes && !operator_router_f32_batch.value_or(false) &&
        qsa_stage_v8 && qsa_channel_chronological == true && mtp_chunked_prefill
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        && !operator_qsa_sparse.value_or(false) && !qsa_route_attribution_path
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
        && !operator_hc_down_staged.value_or(false)
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        && !route_attribution_path
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        && !route_lock_output
#endif
        ), "operator ported GDN requires the fixed exact prefill and chronological QSA diagnostic carrier with other reassociating candidates off");
    require(!gdn_route_attribution_path || (operator_checkpoints == true && operator_gdn_ported == false &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && (native_prefill_hc_rows == true || operator_hc_rows == true)),
        "GDN route attribution requires operator checkpoints on and explicit operator ported GDN off");
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    require(!operator_hc_down_staged.has_value() || (operator_checkpoints.has_value() && operator_checkpoint_path.has_value() &&
        (native_prefill_dense_staged_f32 || operator_staged_dense_f32 == true) && native_prefill_grouped_f32 == true &&
        !native_prefill_hc_down_f32 && !weight_lanes
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        && (!operator_hc_down_staged.value_or(false) || !operator_router_f32_batch.value_or(false))
#else
        && !operator_router_f32_batch.value_or(false)
#endif
#endif
        ), "operator HC down staged requires explicit staged FP32 diagnostics, scalar router and native HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
    require(!route_attribution_path || (operator_checkpoints==true && operator_staged_dense_f32==true && operator_hc_rows==true && operator_hc_down_staged==false && !operator_router_f32_batch.value_or(false)), "route attribution needs explicit staged dense and exact HC rows with scalar router and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    require(!route_census_only || !route_lock_reference, "occupancy census cannot replay routes");
    require(!route_lock_reference || route_lock_output.has_value(),"route lock reference needs explicit output");
    require(!route_lock_output || (route_census_only && operator_checkpoints==true && native_prefill_dense_staged_f32 &&
        native_prefill_grouped_f32==true && native_prefill_hc_rows==true && !route_attribution_path) ||
        (operator_checkpoints==true && operator_staged_dense_f32==true && operator_hc_rows==true &&
        operator_hc_down_staged==false && operator_router_f32_batch.has_value() && !route_attribution_path &&
        (!route_lock_reference || operator_router_f32_batch==true
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
            || kernel_grouped_fp16==true
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            || native_verify_nax==true
#endif
        )),"route-lock replay needs an explicit candidate, exact HC rows, staged dense and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    require(!kernel_qsa_f32.has_value() || (operator_checkpoints.has_value() && operator_staged_dense_f32 == true && operator_hc_rows == true &&
        operator_hc_down_staged == false && !operator_router_f32_batch.value_or(false) && !route_attribution_path && !route_lock_output && !native_raw_decode.value_or(false)),
        "QSA step-2 requires the fixed staged-dense and exact-HC-row diagnostic carrier with other held candidates off");
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    require(!kernel_grouped_fp16.has_value() || (!kernel_qsa_f32.has_value() && operator_checkpoints.has_value() && operator_staged_dense_f32 == true && operator_hc_rows == true &&
        operator_hc_down_staged == false && !operator_router_f32_batch.value_or(false) && !route_attribution_path && !native_raw_decode.value_or(false)),
        "grouped step-2 requires the fixed diagnostic carrier with QSA and other held candidates off");
#endif
#endif
    const bool fixed_operator_screen = operator_checkpoints.has_value()
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        && !route_census_only
#endif
        ;
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    require(!route_census_only || (!dry && !profiling && execution==StackExecution::direct && mtp_chunked_prefill &&
        known_token_capacity>=512 && native_prefill_dense_staged_f32 && native_prefill_grouped_f32==true &&
        (mtp_depth==2 || mtp_depth==3 || mtp_depth==7) && mtp_request_carry && !prompt_lookup && !worker_broker && !checkpoint_slots &&
        !logit_trace_path && !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "occupancy census requires the fixed staged-FP32 speculative carrier without replay or trace");
#endif
    if (fixed_operator_screen) {
        require(!dry && !profiling && execution == StackExecution::direct && mtp_chunked_prefill && known_token_capacity == 512 && reuse_residual &&
            native_prefill_chunks == GdnChunkMode::fused && native_prefill_dense_f32 == true && native_prefill_grouped_f32 == true && !native_prefill_final_logits &&
            mtp_depth == 2 && mtp_request_carry && !prompt_lookup && !worker_broker && !checkpoint_slots &&
            !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments &&
            concurrent_selected && concurrent_dispatch && concurrent_state_proof && logit_trace_path.has_value() && logit_trace_rows == 4,
            "operator screen requires fixed input512 f32/f32 MTP2, concurrent state proof and private trace rows4");
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        require(!task4_profile.has_value(), "operator checkpoints and legacy Task4 counters are exclusive");
#endif
    }
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    require(!kernel_search_capture_path || (!dry && !profiling && execution == StackExecution::direct && !mtp_sidecar && known_token_capacity == 512 &&
        native_prefill_chunks == GdnChunkMode::fused && native_prefill_dense_f32 == true && native_prefill_grouped_f32 == true && !native_prefill_dense_staged_f32 &&
        !native_prefill_hc_down_f32 && !native_prefill_final_logits && !native_raw_decode.value_or(false) && !weight_lanes &&
        !operator_checkpoints && !route_attribution_path && !route_lock_output && !operator_staged_dense_f32 && !operator_hc_rows && !operator_router_f32_batch && !operator_hc_down_staged &&
        logit_trace_path && logit_trace_rows == 3 && !prompt_lookup && !worker_broker && !checkpoint_slots && !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "kernel capture requires bounded exact raw f32/f32 teacher forcing without MTP or other diagnostics");
#endif
    require(!allow_stack_experiments || execution == StackExecution::recurrent_replay,
        "request-time stack experiments require replay preparation at startup");
    require(!known_token_capacity || (execution == StackExecution::direct && (!profiling || std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE")) && !allow_kernel_experiments &&
        !allow_stack_experiments && (small_gemv_splits == 0 || small_gemv_splits == 16) && (block_gemv_formats & 1)),
        "known-token prefill requires fixed split0/16 block-Q8 direct execution, no profiling or request-time overrides");
    require(!allow_prefill_experiments || known_token_capacity == 8, "prefill experiments require startup capacity eight");
    require(!reuse_residual || known_token_capacity == 8
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
        || kernel_search_capture_path.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        || mtp_chunked_prefill
#endif
        , "residual reuse requires startup known-token capacity eight");
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    require((!grouped_prefill_ffn.has_value() && !prefill_layer_submissions.has_value() && !prefill_hc_residual_rows.has_value()) || known_token_capacity == 512 || known_token_capacity == 1024,
        "private large-prefill selectors require capacity512");
    require(!prefill_hc_residual_rows.value_or(false) || (grouped_prefill_ffn.has_value() && !*grouped_prefill_ffn),
        "private HC residual rows require explicit --prefill-grouped-ffn off");
    require((known_token_capacity != 512 && known_token_capacity != 1024 && known_token_capacity != 2048) || (execution == StackExecution::direct && !profiling &&
        small_gemv_splits == 16 && block_gemv_formats == 3 && !hc_q8_splits &&
        hc_exact == HcExactMode::combined && qsa_stage_v8_present && qsa_stage_v8 &&
        qsa_channel_chronological.value_or(false) && ple_gate_transform == PleGateTransform::signed_root &&
        !wide_q8_exact && !wide_q8_many && !checkpoint_slots && !worker_broker && !worker_broker8 && !prompt_lookup &&
        !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments &&
        !allow_lookup_experiments && (!reuse_residual
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
            || kernel_search_capture_path.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            || mtp_chunked_prefill
#endif
        ) && verification_max_tokens == 2 && !verification8_branch_reuse),
        "private grouped capacity512 requires corrected fixed direct target-only policy; no checkpoints, lookup, verifier or request overrides");
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    require(!native_prefill_dense_f32.has_value() || native_prefill_chunks.has_value(), "native prefill dense requires native prefill chunks");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    require((!native_prefill_grouped_f32.has_value() && !native_prefill_grouped_auto) || native_prefill_chunks.has_value(), "native prefill grouped requires native prefill chunks");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require((!native_prefill_grouped_f32.has_value() && !native_prefill_grouped_auto) || !weight_lanes, "native prefill grouped selection requires original GGUF layout");
#endif
#endif
    require(!native_prefill_chunks || ((known_token_capacity == 512 || known_token_capacity == 1024 || known_token_capacity == 2048) && !grouped_prefill_ffn.has_value() &&
        !prefill_layer_submissions.has_value() && !prefill_hc_residual_rows.has_value() && !verification_budget),
        "native prefill chunks require the fixed capacity>=512 target without legacy selectors or verifier stores");
#endif
    branch_preparation(checkpoint_slots, checkpoint_budget, uint32_t(execution), profiling,
        allow_kernel_experiments, allow_stack_experiments, allow_prefill_experiments);
    require(!worker_broker || checkpoint_slots, "worker broker requires explicit checkpoint preparation");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    require(!worker_broker8 || (worker_broker && verification8_branch_reuse && checkpoint_slots && prompt_lookup &&
        verification_max_tokens == 8 && known_token_capacity == 8 && reuse_residual && hc_exact == HcExactMode::combined),
        "branch8 broker requires explicit history/V8/HCcombined/cap8/residual consent");
#endif
    lookup_preparation(prompt_lookup, verification_budget, uint32_t(execution), profiling, small_gemv_splits, block_gemv_formats,
        allow_kernel_experiments, allow_stack_experiments, allow_prefill_experiments, worker_broker, allow_lookup_experiments
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , worker_broker8
#endif
        );
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    const bool history_allowed = !checkpoint_slots
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        || verification8_branch_reuse
#endif
        ;
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    require(!verification8_branch_reuse || (checkpoint_slots && prompt_lookup && verification_max_tokens == 8 &&
        known_token_capacity == 8 && reuse_residual && hc_exact == HcExactMode::combined && (!worker_broker
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            || worker_broker8
#endif
        )),
        "private branch verifier requires explicit history, V8 lookup, HCcombined/cap8/residual and no broker");
#endif
    require(verification_max_tokens == 2 || (prompt_lookup && known_token_capacity == 8 && history_allowed),
        "private verification8 requires lookup/capacity8 without persistent checkpoints");
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(verification_max_tokens != 8 || !wide_q8_exact, "private verification8 first cohort requires scalar wide off");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(verification_max_tokens != 8 || !wide_q8_many, "private verification8 first cohort requires wide many off");
#endif
#endif
    hc_q8_preparation(hc_q8_splits, uint32_t(execution), profiling, block_gemv_formats,
        allow_kernel_experiments, allow_stack_experiments, allow_prefill_experiments, prompt_lookup, reuse_residual);
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(hc_exact_startup_allowed(hc_exact, uint32_t(execution), profiling, small_gemv_splits, block_gemv_formats,
        hc_q8_splits, known_token_capacity, allow_kernel_experiments, allow_stack_experiments, allow_prefill_experiments,
        worker_broker, checkpoint_slots
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        , verification8_branch_reuse
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , worker_broker8
#endif
        ), "private exact-HC requires fixed direct split16/block3/HC0, cap0/8 and its explicit branch consent");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(wide_q8_exact_startup_allowed(wide_q8_exact, uint32_t(execution), profiling, small_gemv_splits, block_gemv_formats,
        hc_q8_splits, known_token_capacity, allow_kernel_experiments, allow_stack_experiments, allow_prefill_experiments,
        worker_broker, checkpoint_slots), "private wide-Q8 requires fixed direct split16/block3/HC0, cap0/8, no profiling/checkpoints/request overrides");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(wide_q8_many_startup_allowed(wide_q8_many, uint32_t(execution), profiling, small_gemv_splits, block_gemv_formats,
        hc_q8_splits, known_token_capacity, allow_kernel_experiments, allow_stack_experiments, allow_prefill_experiments,
        worker_broker, checkpoint_slots, wide_q8_exact), "private wide-Q8-many requires fixed direct split16/block3/HC0/cap8, scalar wide off, no profiling/checkpoints/request overrides");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    if (qsa_stage_v8_present)
#endif
    require(qsa_stage_v8_startup_allowed(uint32_t(execution), profiling, small_gemv_splits, block_gemv_formats,
        hc_q8_splits, known_token_capacity, allow_kernel_experiments, allow_stack_experiments,
        allow_prefill_experiments, allow_lookup_experiments, worker_broker,
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        wide_q8_exact,
#else
        false,
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        wide_q8_many
#else
        false
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        , worker_broker8 && verification8_branch_reuse
#endif
        ), "private staged QSA arms require fixed direct split16/block3/HC0 without request overrides, wide or an unqualified broker");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    require(qsa_channel_startup_allowed(qsa_channel_chronological.has_value(), qsa_stage_v8, uint32_t(hc_exact)),
        "explicit chronological QSA OFF/ON requires Stage-V8 ON and combined exact-HC");
#endif
    using TargetBranches = BranchSession<StackCheckpoint, StackResult>;
    require(!checkpoint_slots || TargetBranches::minimum_bytes(checkpoint_slots, 248320) < checkpoint_budget,
        "combined checkpoint budget cannot cover host session storage");
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
    require(!verification_cost_manifest || (!dry && !profiling && !worker_broker && !worker_broker8 &&
        verification8_branch_reuse && verification_max_tokens == 8 && known_token_capacity == 8 && reuse_residual &&
        prompt_lookup && !allow_lookup_experiments && checkpoint_slots == 4 &&
        ple_gate_transform == PleGateTransform::signed_root && qsa_stage_v8 &&
        qsa_channel_chronological == std::optional<bool>(true) && hc_exact == HcExactMode::combined &&
        !wide_q8_exact && !wide_q8_many && !batch_phase_clock.value_or(false)),
        "verification cost requires corrected capacity8 direct fixed branch8/channel policy with phase clock off");
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
    require(!verification_qsa_rows.has_value() || verification_cost_manifest.has_value(),
        "private verification QSA rows are admitted only by the sealed direct COST probe; no HTTP/broker policy");
#else
    require(!verification_qsa_rows.has_value(), "verification QSA selectors require the separate COST probe");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    require(!known_qsa_rows.has_value() || (!dry && !profiling && known_token_capacity == 8 && !allow_prefill_experiments &&
        !worker_broker && !worker_broker8 && !prompt_lookup && !checkpoint_slots && !verification_budget &&
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
        !verification_cost_manifest &&
#endif
        !verification_qsa_rows && small_gemv_splits == 16 && block_gemv_formats == 3 &&
        !hc_q8_splits && hc_exact == HcExactMode::combined && qsa_stage_v8 &&
        qsa_channel_chronological == std::optional<bool>(true) && !wide_q8_exact && !wide_q8_many),
        "known QSA rows require private direct capacity8/channel/combined-HC without broker, lookup or verifier");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    require(!sampled_prefill_input8 || (known_qsa_rows == std::optional<bool>(true) && known_token_capacity == 8 &&
        reuse_residual && !prompt_lookup && !verification_budget && verification_max_tokens == 2 &&
        !verification8_branch_reuse && !checkpoint_slots && !worker_broker && !worker_broker8 &&
        !allow_lookup_experiments && !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "sampled input8 requires fixed known-QSA rows without verifier, lookup, broker or overrides");
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!mtp_profile_boundary || mtp_profile_rounds, "MTP profile mode requires an explicit round budget");
    require(!mtp_profile_rounds || (!dry && !profiling && mtp_sidecar && mtp_lookahead_anchor &&
        mtp_draft_attention == MtpDraftAttention::tiled128 && weight_lanes &&
        weight_lanes_auth_workers == std::optional<uint32_t>(4) && mtp_profile_stderr_writable()),
        "MTP profiling requires the explicit live lookahead/tiled128/lane/auth4 carrier and regular writable stderr; ordinary profiler refused");
#endif
    require(mtp_depth == 1 ||
        ((mtp_depth == 2 || mtp_depth == 3) && mtp_sidecar && mtp_lookahead_anchor && mtp_draft_attention == MtpDraftAttention::tiled128 && !profiling && verification_max_tokens == 2) ||
        (mtp_depth == 7 && dflash2_dir && mtp_lookahead_anchor && mtp_draft_attention == MtpDraftAttention::split16 && !profiling && verification_max_tokens == 2),
        "speculative depth requires matching sidecar/DFlash2, lookahead and draft attention without fallback");
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(mtp_depth == 1 || !mtp_profile_rounds, "depth-two MTP does not admit depth-one profiling controls");
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
    require(mtp_depth == 1 || !sampled_cost_manifest, "depth-two MTP does not admit the target-only COST probe");
#endif
    require(!dflash2_auto_draft || (verify_cost_profile_path && dflash2_dir && dflash2_adaptive_budget),
        "automatic DFlash2 drafting requires a compatible profile and automatic row selection");
    if (verify_cost_profile_path) {
        require(mtp_request_carry && (mtp_depth == 2 || mtp_depth == 3 || mtp_depth == 7) &&
            !native_verify_nax.value_or(false) && !native_verify_nax_m8.value_or(false) &&
            !native_verify_hc_gdn_fused.value_or(false) && !native_mtp_fast_round.value_or(false) &&
            mtp_lookup_mode == MtpLookupMode::off && (!dflash2_dir || dflash2_adaptive_budget),
            "measured controllers require exact target verification, scoped v2 carry, fast round off and no lookup");
        MappedFile profile_file(*verify_cost_profile_path);
        require(profile_file.size() <= 1024 * 1024, "measured profile exceeds its byte bound");
        const auto bytes = profile_file.bytes();
        verify_cost_profile = MeasuredCostProfile::parse(parse_json({reinterpret_cast<const char*>(bytes.data()), bytes.size()}));
        MappedFile proof_file(verify_cost_profile_path->string() + ".exactness.json");
        require(proof_file.size() <= 4 * 1024 * 1024, "exactness receipt exceeds its byte bound");
        const auto proof_bytes = proof_file.bytes();
        const std::string_view proof(reinterpret_cast<const char*>(proof_bytes.data()), proof_bytes.size());
        verify_cost_profile->admit_exactness(parse_json(proof), derived_pack::hash(proof));
        const uint32_t maximum = dflash2_dir ? dflash2_max_drafts.value_or(dflash2_default_max_drafts) + 1 : mtp_depth + 1;
        require(maximum == verify_cost_profile->maximum_rows() && maximum <= 5,
            "measured profile does not cover the configured serving width");
        profile_file.assert_unchanged(); proof_file.assert_unchanged();
    }
    require(!mtp_request_carry || (has_speculative && mtp_lookahead_anchor), "MTP request carry requires explicit lookahead drafter");
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    require(!mtp_output_packet_limit || (has_speculative && scoped_spec_ok),
        "native MTP output packet requires fixed depth-two or DFlash2 scoped attention");
#endif
    require(!mtp_lookahead_anchor || has_speculative, "MTP lookahead anchor requires an explicit drafter");
    require(mtp_draft_attention == MtpDraftAttention::serial ||
        (mtp_draft_attention == MtpDraftAttention::split16 ? dflash2_dir.has_value() : mtp_sidecar.has_value()),
        "MTP draft attention requires matching sidecar or DFlash2 directory");
    require(!has_speculative || (((known_qsa_rows == std::optional<bool>(true) && known_token_capacity == 8)
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        || mtp_chunked_prefill
#endif
        ) &&
        reuse_residual && !sampled_prefill_input8 && !prompt_lookup && !checkpoint_slots && !worker_broker &&
        !worker_broker8 && !allow_lookup_experiments && !allow_kernel_experiments && !allow_stack_experiments &&
        !allow_prefill_experiments && ple_gate_transform == PleGateTransform::signed_root),
        "native speculative decoding requires corrected fixed input8/channel target without greedy lookup, broker or overrides");
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(!weight_lanes_auth_workers || (weight_lanes && !weight_lanes->empty()), "lane authentication workers require an explicit sidecar");
    require(!weight_lanes || (!dry && execution == StackExecution::direct && !allow_kernel_experiments &&
        !allow_stack_experiments && !allow_prefill_experiments && !hc_q8_splits && !wide_q8_exact && !wide_q8_many &&
        small_gemv_splits == 16 && block_gemv_formats == 3 && ple_gate_transform == PleGateTransform::signed_root),
        "lane weights require an explicit corrected direct target without other arithmetic experiments");
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
    require(!sampled_cost_manifest || (!dry && !profiling && mtp_sidecar && mtp_lookahead_anchor &&
        mtp_draft_attention == MtpDraftAttention::tiled128 && weight_lanes &&
        weight_lanes_auth_workers == std::optional<uint32_t>(4) &&
        qsa_stage_v8 && qsa_channel_chronological == std::optional<bool>(true) && hc_exact == HcExactMode::combined),
        "sampled COST requires the corrected fixed sampled-MTP1/tiled128/lane/auth4 carrier without profiling");
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!sampled_cost_manifest || !mtp_profile_rounds, "sampled COST refuses intrusive MTP profiling");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    require(!concurrent_state_proof || concurrent_selected, "private concurrent state proof requires an explicit arm selector");
    require(!concurrent_dispatch || (execution == StackExecution::direct && !profiling && !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "private concurrent dispatch requires fixed direct unprofiled execution");
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    require(!concurrent_dispatch || !mtp_profile_rounds, "private concurrent dispatch refuses intrusive MTP profiling");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    require(!q8_three_rows || (execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments),
        "three-row Q8 requires fixed direct target execution");
    require(q8_three_msg == Q8ThreeMsgMode::off || (q8_three_rows && execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments),
        "multi-SG three-row Q8 requires three-row direct target execution");
    require(!moe_three_union || (q8_three_rows && execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments),
        "unioned three-row MoE requires three-row direct target execution");
#else
    require(q8_three_msg == Q8ThreeMsgMode::off && !moe_three_union, "three-row Q8 candidates require the three-row build");
#endif
    require(!swiglu_fused || (!allow_kernel_experiments && !allow_stack_experiments),
        "fused SwiGLU requires fixed target execution without request-time kernel/stack overrides");
    require(q8_gemv_rg4 == Q8GemvRg4Mode::off || (execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments),
        "4-row/SG Q8 GEMV requires fixed direct target execution");
    require((prefill_m64 == PrefillM64Mode::off && !prefill_multiseg && !prefill_swizzle && !qsa_prefill_fused && !hc_consumer_fused && !ple_overlap) ||
        (execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments),
        "exact prefill candidates require fixed direct target execution without request-time overrides");
    require(!shared_prefix_cache_enabled || (execution == StackExecution::direct && !worker_broker && !checkpoint_slots && !prompt_lookup &&
        !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        && !operator_checkpoints.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
        && !kernel_search_capture_path.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        && !task4_profile.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
        && !verification_cost_manifest.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
        && !sampled_cost_manifest.has_value()
#endif
        ), "shared prefix cache requires fixed direct serving/target execution without broker or diagnostic captures");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    require(mtp_lookup_mode == MtpLookupMode::off || (mtp_sidecar.has_value() && (mtp_depth == 2 || mtp_depth == 3) && execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        && !operator_checkpoints.has_value()
#endif
        ), "MTP lookup bypass requires depth-two or depth-three direct native MTP");
#else
    require(mtp_lookup_mode == MtpLookupMode::off, "MTP lookup bypass requires the depth-two native MTP build");
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    require(!native_raw_decode.value_or(false) || (execution == StackExecution::direct && !allow_kernel_experiments && !allow_stack_experiments && !worker_broker && !prompt_lookup && !checkpoint_slots),
        "raw routed decode requires fixed direct target execution without branch or lookup state");
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    require(!task4_profile_boundary || task4_profile.has_value(), "Task4 boundary requires an explicit profile arm");
    if (task4_profile.has_value()) {
        require(!dry && !profiling && execution == StackExecution::direct && known_token_capacity == 8 && reuse_residual &&
            small_gemv_splits == 16 && block_gemv_formats == 3 && !hc_q8_splits && !checkpoint_slots && !worker_broker && !prompt_lookup &&
            !allow_kernel_experiments && !allow_stack_experiments && !allow_prefill_experiments && !allow_lookup_experiments &&
            logit_trace_path.has_value() && logit_trace_rows == 2 && (!*task4_profile || task4_profile_stderr_writable()),
            "Task4 requires fixed serial input8, two endpoint vectors and no request-time experiments");
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        require(known_qsa_rows == std::optional<bool>(true), "Task4 requires the frozen known-QSA q8 path");
#else
        throw FormatError("Task4 requires the known-QSA-row build");
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        require(!concurrent_dispatch, "Task4 serial diagnostic refuses concurrent dispatch");
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        require(!mtp_profile_rounds, "Task4 and MTP-round counters are exclusive");
#endif
    }
#endif
    const auto options = policy_file(policy_path); const auto chosen_policy = policy(options, 1e-6f); const auto dil = dilation(options);
    auto source = std::make_shared<ModelFiles>(paths); const auto& files = *source;
    std::optional<derived_pack::AdoptedPack> pack;
    std::optional<planar_v3::AdoptedPackV3> pack_v3;
    if (derived_pack_path) {
        std::shared_ptr<ModelFiles> mtp_for_pack;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_sidecar) mtp_for_pack = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{*mtp_sidecar});
#endif
        const auto trust_mode = (derived_pack_trust_mode == "local-seal")
            ? derived_pack::TrustMode::local_seal
            : derived_pack::TrustMode::full;
        const bool is_v3 = [&]() {
            Gguf header_probe(*derived_pack_path);
            return header_probe.find("tokspur_pack.version") &&
                   header_probe.unsigned_value("tokspur_pack.version") == planar_v3::pack_version_v3;
        }();
        if (is_v3) {
            pack_v3.emplace(planar_v3::adopt_derived_pack_v3(source, mtp_for_pack, *derived_pack_path,
                8, derived_pack::default_chunk_bytes, derived_pack_auth_workers.value_or(16), trust_mode));
        } else {
            pack.emplace(derived_pack::adopt_derived_pack(source, mtp_for_pack, *derived_pack_path,
                derived_pack::default_chunk_bytes, derived_pack_auth_workers.value_or(16), trust_mode));
        }
    }
    const std::shared_ptr<const ModelFiles> pack_model = pack ? pack->model : (pack_v3 ? pack_v3->model : nullptr);
    const bool pack_shared_mapping = pack_model && pack_model->shards().front().backing_mapping() &&
        pack_model->shards().front().backing_mapping()->mapping_mode() == ReadOnlyMapMode::shared_file;
    const auto layout = inspect_flash_next(files, pack_model.get()); admit(layout);
    const auto before = memory();
    if (dry) { std::cout << Json(Json::Object{{"event", Json("preflight")}, {"memory", describe(before)}, {"full_model_loaded", Json(false)}
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
        , {"ple_gate_transform", Json(ple_gate_transform == PleGateTransform::plain ? "plain" : "signed-root")}, {"ple_gate_transform_schema", Json::integer(1)}
#endif
        }).dump() << std::endl; return 0; }
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
    std::optional<LogitTrace> logit_trace;
    if (logit_trace_path) logit_trace.emplace(*logit_trace_path, logit_trace_rows);
#endif
    const auto start = Clock::now();
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::optional<lanes::AdoptedSidecar> lane_weights;
    if (weight_lanes) lane_weights.emplace(lanes::adopt_sidecar(source, *weight_lanes, std::span<const std::string>{}, lanes::sidecar_max_bytes,
        lanes::sidecar_default_chunk_bytes, weight_lanes_auth_workers.value_or(1)));
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::array<char, PROC_PIDPATHINFO_MAXSIZE> executable_path{};
    require(proc_pidpath(getpid(), executable_path.data(), uint32_t(executable_path.size())) > 0, "cannot identify the running worker executable");
    auto identity_worker = execution_identity_artifact(std::filesystem::path(executable_path.data()));
    auto identity_library = execution_identity_artifact(library);
#endif
    MetalGraph gpu(library, gpu_budget); std::set<std::string> consumed{"token_embd.weight", "per_layer_token_embd.weight"};
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH)
    if(operator_router_f32_batch.has_value()){gpu.configure_moe_router_f32_batch(*operator_router_f32_batch);operator_sink.router_batch=operator_router_f32_batch;operator_sink.router_kernel=gpu.moe_router_f32_projection_kernel();}
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    if (kernel_search_capture_path) gpu.configure_operator_checkpoints(TargetKernelSearchCaptureSink::checkpoint, &kernel_search_sink, 16384);
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    operator_sink.hc_rows=operator_hc_rows;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    operator_sink.router_window=operator_router_window;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    operator_sink.hc_down_window=operator_hc_down_window;
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    operator_sink.qsa_sparse=operator_qsa_sparse;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    operator_sink.gdn_ported = operator_gdn_ported;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    operator_sink.hc_down_staged = operator_hc_down_staged;
#endif
    if (operator_checkpoints.has_value()) {
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
        if (kernel_qsa_f32.has_value()) operator_sink.maximum_bytes = 32ull << 20;
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
        if (kernel_grouped_fp16.has_value()) operator_sink.maximum_bytes = 32ull << 20;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        if(route_attribution_path)operator_sink.maximum_bytes=32ull<<20;
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(route_lock_output)operator_sink.maximum_bytes=32ull<<20;
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
        if (operator_hc_down_window.has_value()) operator_sink.maximum_bytes = 64ull << 20;
#endif
        uint32_t operator_limit = 16384;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        if (operator_qsa_sparse.has_value() || qsa_route_attribution_path) { operator_sink.maximum_bytes = 256ull << 20; operator_limit = 131072; }
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (operator_gdn_ported.has_value() || gdn_route_attribution_path) { operator_sink.maximum_bytes = 256ull << 20; operator_limit = 131072; }
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(route_census_only){operator_sink.maximum_bytes=512ull<<20;operator_limit=1048576;}
        else if(route_lock_output){operator_sink.maximum_bytes=256ull<<20;operator_limit=131072;}
#endif
        operator_sink.open_file(*operator_checkpoint_path, *operator_checkpoints
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            , native_prefill_dense_staged_f32 ? std::optional<bool>(true) : operator_staged_dense_f32
#endif
            );
        if (*operator_checkpoints) gpu.configure_operator_checkpoints(TargetOperatorCheckpointSink::emit, &operator_sink, operator_limit);
    }
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    gpu.configure_raw_decode(native_raw_decode.value_or(false));
#endif
    // Counters are chunked into buffers no larger than this M5's 32 KiB limit.
    auto counter_sampling = profiling ? gpu.configure_profiling(4096) : ProfileMode::disabled;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    gpu.configure_concurrent_dispatch(concurrent_dispatch);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    gpu.configure_q8_three_rows(q8_three_rows);
#endif
    gpu.configure_swiglu_fused(swiglu_fused);
    gpu.configure_q8_gemv_rg4(q8_gemv_rg4);
    gpu.configure_q8_three_msg(q8_three_msg);
    gpu.configure_moe_three_union(moe_three_union);
    gpu.configure_prefill_m64(prefill_m64);
    gpu.configure_prefill_multiseg(prefill_multiseg);
    gpu.configure_prefill_swizzle(prefill_swizzle);
    gpu.configure_qsa_prefill_fused(qsa_prefill_fused);
    gpu.configure_hc_consumer_fused(hc_consumer_fused);
    gpu.configure_small_gemv(small_gemv_splits);
    gpu.configure_block_gemv(block_gemv_formats);
    if (hc_q8_splits) gpu.configure_hc_q8_splits(hc_q8_splits);
    const auto hc_q8 = gpu.hc_q8_policy(); // Fixed for this target lifetime.
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    gpu.configure_hc_exact(hc_exact);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    gpu.configure_wide_q8_exact(wide_q8_exact);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    gpu.configure_wide_q8_many(wide_q8_many);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    if (qsa_stage_v8_present)
#endif
    gpu.configure_qsa_stage_v8(qsa_stage_v8); // Same prepared score storage in OFF and ON.
    const auto qsa_stage_policy = gpu.qsa_stage_v8_policy();
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    if (qsa_channel_chronological.has_value()) gpu.configure_qsa_channel_chronological(*qsa_channel_chronological);
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    if (kernel_qsa_f32.has_value()) gpu.configure_kernel_qsa_f32(*kernel_qsa_f32);
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    if (kernel_grouped_fp16.has_value()) gpu.configure_kernel_qsa_f32(false);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    if (qsa_sparse.has_value()) gpu.configure_qsa_sparse(*qsa_sparse);
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    if (native_qsa_sparse.has_value()) gpu.configure_qsa_sparse(*native_qsa_sparse);
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (operator_qsa_sparse.has_value()) gpu.configure_qsa_sparse(*operator_qsa_sparse);
    if (qsa_route_attribution_path) {
        qsa_route_sink.open_directory(*qsa_route_attribution_path);
        gpu.configure_qsa_route_attribution(TargetQsaRouteAttributionSink::emit, &qsa_route_sink);
    }
#endif
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    if (gdn_ported.has_value()) gpu.configure_gdn_ported(*gdn_ported);
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    if (native_gdn_ported.has_value()) gpu.configure_gdn_ported(*native_gdn_ported);
    if (native_gdn_prefill_ported.has_value()) gpu.configure_gdn_prefill_ported(*native_gdn_prefill_ported);
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (operator_gdn_ported.has_value()) gpu.configure_gdn_ported(*operator_gdn_ported);
    if (gdn_route_attribution_path) {
        gdn_route_sink.open_directory(*gdn_route_attribution_path);
        gpu.configure_gdn_route_attribution(TargetGdnRouteAttributionSink::emit, &gdn_route_sink, gdn_route_attribution_prefill_prefix);
    }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    if (batch_phase_clock.has_value()) gpu.configure_batch_phase_clock(*batch_phase_clock);
#endif
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    require(!carrier_forced_input || carrier_mode != CarrierDiagnosticMode::off, "forced history requires a bounded diagnostic mode");
    require(!carrier_single_action || (carrier_mode == CarrierDiagnosticMode::batch && !carrier_forced_input &&
        carrier_capture_path.empty() && verify_cost_profile.has_value()),
        "single-action timing requires concurrent batch diagnostics, a measured policy and no forced-history or correctness capture");
    require(!carrier_batch_only || carrier_single_action,
        "batch-only timing requires the declared single-action cost diagnostic");
    if (carrier_mode != CarrierDiagnosticMode::off) gpu.prepare_carrier_diagnostic(carrier_mode, carrier_actions, !carrier_batch_only);
    if (!carrier_capture_path.empty()) {
        require(carrier_mode == CarrierDiagnosticMode::batch || carrier_mode == CarrierDiagnosticMode::replay,
            "complete correctness capture requires batch or replay mode and refuses serial cost profiling");
        require(carrier_mode != CarrierDiagnosticMode::off, "complete-logit capture requires a bounded diagnostic mode");
        gpu.configure_carrier_capture(carrier_capture_path);
    }
#endif
    std::vector<TensorSlice> slices; std::map<std::string, size_t> indices;
    for (const auto& [name, location] : files.tensors()) {
        static_cast<void>(location); if (consumed.contains(name)) continue;
        if (pack_model && pack_model->tensors().contains(name)) continue;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (lane_weights && lane_weights->model->tensors().contains(name)) continue;
#endif
        indices.emplace(name, slices.size()); slices.push_back({name, 0, 0});
    }
    // Replace large original Metal resources rather than layering capture aliases
    // over them. Whole tensor slices stay intact; capped edge pages are charged
    // for every actual resource. Both same-resident comparators share this plan.
    const uint64_t mapped_region_limit = (execution == StackExecution::recurrent_replay ||
        (TOKSPUR_PRIVATE_M5_RESOURCE_SCREEN && execution == StackExecution::direct)) ? uint64_t(1) << 32 : 0;
    ImportedWeights imports;
    if (!slices.empty()) {
        imports = gpu.import_weights(source, slices, mapped_region_limit);
        own_memory_guard();
    }
    size_t imported_region_count = imports.regions.size();
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::map<std::string, GpuView> packed_views;
    if (lane_weights) {
        std::vector<TensorSlice> packed_slices;
        for (const auto& tensor : lane_weights->plan.tensors) packed_slices.push_back({tensor.name, 0, 0});
        const auto packed = gpu.import_weights(lane_weights->model, packed_slices, mapped_region_limit, true);
        imported_region_count += packed.regions.size();
        for (size_t i = 0; i < lane_weights->plan.tensors.size(); ++i) {
            const auto& tensor = lane_weights->plan.tensors[i]; const auto& dims = tensor.dimensions;
            gpu.register_lane_matrix(packed.tensors.at(i), {tensor.type, uint32_t(dims.at(0)), uint32_t(dims.at(1)), dims.size() == 3 ? uint32_t(dims.at(2)) : 1});
            require(packed_views.emplace(tensor.name, packed.tensors.at(i)).second, "duplicate adopted lane tensor view");
        }
        lane_weights->model->assert_unchanged(); own_memory_guard();
    }
#endif
    std::map<std::string, GpuView> pack_views;
    std::map<std::string, uint32_t> pack_source_types;
    if (pack) {
        std::vector<TensorSlice> packed_slices;
        std::vector<size_t> packed_plan_indices;
        for (size_t i = 0; i < pack->plan.tensors.size(); ++i) {
            const auto& tensor = pack->plan.tensors[i];
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!mtp_sidecar && tensor.name.starts_with("blk.48.")) continue;
#endif
            packed_slices.push_back({tensor.name, 0, 0});
            packed_plan_indices.push_back(i);
        }
        const auto packed = gpu.import_weights(pack->model, packed_slices, mapped_region_limit
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            , true
#endif
        );
        imported_region_count += packed.regions.size();
        for (size_t k = 0; k < packed_plan_indices.size(); ++k) {
            const auto& tensor = pack->plan.tensors[packed_plan_indices[k]];
            require(pack_views.emplace(tensor.name, packed.tensors.at(k)).second, "duplicate adopted pack tensor view");
            require(pack_source_types.emplace(tensor.name, pack->model->tensor(tensor.name).type).second, "duplicate adopted pack tensor type");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
            if (tensor.layout != "plain") {
                const auto& dims = tensor.dimensions;
                gpu.register_lane_matrix(packed.tensors.at(k), {tensor.type, uint32_t(dims.at(0)), uint32_t(dims.at(1)), dims.size() == 3 ? uint32_t(dims.at(2)) : 1});
            }
#endif
        }
        pack->model->assert_unchanged(); own_memory_guard();
    } else if (pack_v3) {
        std::vector<TensorSlice> packed_slices;
        std::vector<size_t> packed_plan_indices;
        for (size_t i = 0; i < pack_v3->plan.tensors.size(); ++i) {
            const auto& tensor = pack_v3->plan.tensors[i];
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!mtp_sidecar && tensor.name.starts_with("blk.48.")) continue;
#endif
            packed_slices.push_back({tensor.name, 0, 0});
            packed_plan_indices.push_back(i);
        }
        const auto packed = gpu.import_weights(pack_v3->model, packed_slices, mapped_region_limit
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            , true
#endif
        );
        imported_region_count += packed.regions.size();
        for (size_t k = 0; k < packed_plan_indices.size(); ++k) {
            const auto& tensor = pack_v3->plan.tensors[packed_plan_indices[k]];
            const auto view = packed.tensors.at(k);
            require(pack_views.emplace(tensor.name, view).second, "duplicate adopted pack v3 tensor view");
            require(pack_source_types.emplace(tensor.name, tensor.source_type).second, "duplicate adopted pack v3 tensor type");
            if (planar_v3::is_planar_v3_kind(tensor.kind)) {
                const auto& sdims = tensor.source_dimensions;
                require(sdims.size() == 3, "planar v3 MoE tensor requires 3D source dimensions");
                gpu.register_planar_matrix(view, {
                    .source_type = tensor.source_type,
                    .kind = uint32_t(tensor.kind),
                    .columns = uint32_t(sdims[0]),
                    .rows = uint32_t(sdims[1]),
                    .experts = uint32_t(sdims[2]),
                });
            }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
            else if (tensor.kind == planar_v3::PlanarKind::q8_lanes_v1) {
                const auto& dims = tensor.dimensions;
                gpu.register_lane_matrix(view, {tensor.type, uint32_t(dims.at(0)), uint32_t(dims.at(1)),
                    dims.size() == 3 ? uint32_t(dims.at(2)) : 1});
            }
#endif
        }
        const bool prep_planar_nax =
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
            native_prefill_chunks.has_value();
#else
            false;
#endif
        const uint32_t planar_nax_cap =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            native_prefill_grouped_nax ? std::clamp(known_token_capacity, 512u, 2048u) : 512u;
#else
            512u;
#endif
        for (const auto [gu_enc, dn_enc] : {std::pair{12u, 7u}, std::pair{13u, 7u}, std::pair{8u, 8u}}) {
            gpu.prepare_planar_moe_v3(gu_enc, dn_enc, prep_planar_nax, planar_nax_cap, 512, 10, 2560, 640);
        }
        pack_v3->model->assert_unchanged(); own_memory_guard();
    }
    const auto load = [&](const std::string& name) {
        require(consumed.emplace(name).second, "duplicate full target tensor use"); own_memory_guard();
        if (const auto packed = pack_views.find(name); packed != pack_views.end()) return GpuMatrix{packed->second, pack_source_types.at(name)};
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (const auto packed = packed_views.find(name); packed != packed_views.end()) return GpuMatrix{packed->second, files.tensor(name).type};
#endif
        return GpuMatrix{imports.tensors.at(indices.at(name)), files.tensor(name).type};
    };
    StackWeights weights; weights.layers.reserve(48);
    const auto freq_data = rotary_frequencies(64, layout.rope_base); const auto frequencies = gpu.upload(std::as_bytes(std::span(freq_data)), "owned rotary frequencies");
    for (uint32_t layer = 0; layer < 48; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const auto tensor = [&](const std::string& suffix) { return load(prefix + suffix); };
        const auto residual = [&](const std::string& kind) { return ResidualWeights{tensor(kind + "_norm.weight").view, tensor(kind + "_down.weight"), tensor(kind + "_up.weight"), tensor(kind + "_inject.weight")}; };
        const auto attention_read = residual("hc_attn"), ffn_read = residual("hc_ffn");
        const MoeWeights moe{tensor("ffn_gate_inp.weight"), tensor("ffn_gate_exps.weight"), tensor("ffn_up_exps.weight"), tensor("ffn_down_exps.weight"), tensor("ffn_gate_shexp.weight"), tensor("ffn_up_shexp.weight"), tensor("ffn_down_shexp.weight"), tensor("ffn_gate_inp_shexp.weight")};
        if (layer % 4 == 3) {
            const QsaWeights attention{tensor("attn_q.weight"), tensor("attn_k.weight"), tensor("attn_v.weight"), tensor("attn_output.weight"), tensor("indexer.q_proj.weight"), tensor("indexer.k_proj.weight"),
                tensor("attn_q_norm.weight").view, tensor("attn_k_norm.weight").view, tensor("indexer.q_norm.weight").view, tensor("indexer.k_norm.weight").view, frequencies};
            weights.layers.emplace_back(AttentionLayerWeights{attention_read, ffn_read, attention, moe});
        } else {
            const GdnWeights attention{tensor("attn_qkv.weight"), tensor("attn_gate.weight"), tensor("ssm_alpha.weight"), tensor("ssm_beta.weight"), tensor("ssm_out.weight"), tensor("ssm_conv1d.weight").view,
                tensor("ssm_a").view, tensor("ssm_dt.bias").view, tensor("ssm_norm.weight").view};
            weights.layers.emplace_back(RecurrentWeights{attention_read, ffn_read, attention, moe});
        }
        if (layer == 1) weights.memory = {tensor("ple_key.weight"), tensor("ple_value.weight"), tensor("ple_norm_key.weight").view, tensor("ple_norm_query.weight").view, tensor("ple_norm_conv.weight").view, tensor("ple_conv1d.weight").view};
        std::cerr << "Owned target weight views bound: " << layer + 1 << "/48 layers\n";
    }
    weights.mixer = {load("output_hc_norm.weight").view, load("output_hc_down.weight"), load("output_hc_up.weight")}; weights.output = load("output.weight");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    const auto mtp_borrowed_output = weights.output;
#endif
    require(consumed.size() == files.tensors().size(), "full target left a declared tensor unaccounted");
#if defined(TOKSPUR_PRIVATE_RAW_DECODE) && defined(TOKSPUR_PRIVATE_GROUPED_MM)
    if (native_raw_decode.value_or(false)) for (const auto& layer : weights.layers) std::visit([&](const auto& weight) {
        for (const auto* matrix : {&weight.ffn.gate, &weight.ffn.up, &weight.ffn.down}) if (matrix->encoding == 7 || matrix->encoding == 12)
            require(gpu.prefill_weight_layout(matrix->view) == PrefillWeightLayout::gguf, "raw routed decode refuses packed expert weights");
    }, layer);
#endif
    const NgramSpec spec{3, 8, 160, 248320, 248044, 248056, layout.ple_multipliers, layout.ple_offsets, layout.ple_vocabularies};
    const auto& table = files.tensor("per_layer_token_embd.weight");
    const bool ple_async_gather_enabled =
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        native_ple_async_gather.value_or(false);
#else
        false;
#endif
    NocacheTensorFile ple_nocache_file;
    if (ple_async_gather_enabled) {
        if (const auto backing = files.tensor_backing_file(table.name)) {
            ple_nocache_file = NocacheTensorFile(backing->path, backing->offset, backing->bytes, backing->device, backing->inode);
            files.evict_tensor_pages(table.name);
        }
    }
    EmbeddingRows cache(files.tensor_bytes(table.name), table.type, 160, layout.ple_storage_rows, layout.ple_addressable_rows, cache_budget, ple_nocache_file.source(), ple_async_gather_enabled);
#if defined(TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM) && TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM
    uint64_t cache_reclamations = 0, cache_reclaimed_bytes = 0, cache_allocator_released_bytes = 0;
    std::function<void()> reclaim_shared_prefix;
    const auto reclaim_own_caches = [&]() {
        cache_reclaimed_bytes = checked_add(cache_reclaimed_bytes, cache.reclaim());
        if (reclaim_shared_prefix) reclaim_shared_prefix();
        cache_allocator_released_bytes = checked_add(cache_allocator_released_bytes, malloc_zone_pressure_relief(nullptr, 0));
        ++cache_reclamations;
    };
    const auto own_memory_guard = [&]() {
        Memory observed{};
        const bool admitted = reclaim_before_memory_refusal(native_limit, [&]() {
            observed = memory(); return MemoryGuardReading{observed.current, observed.pressure};
        }, reclaim_own_caches);
        if (!admitted) throw FormatError("target process/host memory guard refused further work: " + describe(observed).dump());
    };
#endif
    uint64_t stateless_prefill_ple_rows = 0;
    StackShape shape{{4, 2560, 320}, {2560, 16, 48, 128, 128, 4}, {2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536}, {2560, 640, 512, 10}, {4, 2560, 2560, 4, dil}, 248320, 1};
    if (ple_gate_transform) shape.memory.gate_transform = *ple_gate_transform;
    auto effective_policy = chosen_policy; effective_policy.recurrent.rms_epsilon = effective_policy.attention.rms_epsilon = float(layout.rms_epsilon);
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    if (native_gdn_old_order_batch) {
        gdn_old_order_layers = std::count_if(weights.layers.begin(), weights.layers.end(), [](const auto& layer) { return std::holds_alternative<RecurrentWeights>(layer); });
        require(gdn_old_order_layers == 36, "old-order GDN target layer geometry differs");
    }
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (native_verify_nax.has_value()) gpu.configure_verify_nax(*native_verify_nax);
    if (native_verify_nax_m8.has_value()) gpu.configure_verify_nax_m8(*native_verify_nax_m8);
    if (target_m8_splits_path) {
        require(gpu.verify_nax_m8(), "M8 split override requires explicitly enabled M8");
        MappedFile file(*target_m8_splits_path); require(file.size() <= 128 * 1024, "M8 split override byte bound");
        const auto bytes = file.bytes();
        gpu.configure_target_m8_splits(NaxSplitOverrides::parse(parse_json({reinterpret_cast<const char*>(bytes.data()), bytes.size()})));
        file.assert_unchanged();
    }
    if (native_verify_hc_gdn_fused.has_value()) gpu.configure_verify_hc_gdn_fused(*native_verify_hc_gdn_fused);
#endif
    TextStack stack(gpu, std::move(weights), shape, effective_policy, spec, execution, known_token_capacity, reuse_residual); files.assert_unchanged(); own_memory_guard();
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    stack.configure_sampled_logit_views(host_logit_views);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (native_logit_topk.has_value()) stack.configure_sampled_logit_topk(*native_logit_topk);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    if (known_qsa_rows.has_value()) stack.prepare_known_qsa_rows(*known_qsa_rows);
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    if (native_prefill_chunks) {
        gpu.configure_prefill_dense_f32(native_prefill_dense_f32.value_or(false));
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (native_prefill_dense_nax) gpu.configure_prefill_dense_nax(true);
        if (native_prefill_grouped_nax) {
            gpu.configure_prefill_grouped_nax(true);
        } else if (native_prefill_grouped_auto) {
            gpu.configure_prefill_grouped_auto(true);
            gpu.configure_router_margin_epsilon(router_margin_epsilon);
        } else {
            gpu.configure_prefill_grouped_f32(native_prefill_grouped_f32.value_or(false));
        }
        if (router_margin_audit_enabled) {
            gpu.configure_router_margin_audit(true);
            gpu.configure_router_margin_epsilon(router_margin_epsilon);
        }
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
        if (kernel_grouped_fp16.has_value()) gpu.configure_kernel_grouped_fp16(*kernel_grouped_fp16);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        gpu.configure_prefill_hc_down_f32(native_prefill_hc_down_f32);
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        if (native_prefill_dense_staged_f32) gpu.configure_prefill_dense_staged_f32(true);
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        else if (operator_staged_dense_f32.has_value()) gpu.configure_prefill_dense_staged_f32(*operator_staged_dense_f32);
#endif
#endif
#if defined(TOKSPUR_PRIVATE_HC_DOWN_STAGED) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
        if (operator_hc_down_staged.has_value()) gpu.configure_prefill_hc_down_f32(*operator_hc_down_staged);
#endif
#if defined(TOKSPUR_PRIVATE_HC_ROWS) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
        if(operator_hc_rows.has_value())gpu.configure_prefill_hc_rows(*operator_hc_rows);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
        if (native_prefill_hc_rows.has_value()) gpu.configure_prefill_hc_rows(*native_prefill_hc_rows);
#endif
#if defined(TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
        if(operator_router_window.has_value())gpu.configure_prefill_router_window(*operator_router_window);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
        if (native_prefill_router_window.has_value()) gpu.configure_prefill_router_window(*native_prefill_router_window);
#endif
#if defined(TOKSPUR_PRIVATE_HC_DOWN_WINDOW) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
        if(operator_hc_down_window.has_value())gpu.configure_prefill_hc_down_window(*operator_hc_down_window);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
        if (native_prefill_hc_down_window.has_value()) gpu.configure_prefill_hc_down_window(*native_prefill_hc_down_window);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        if (qsa_sparse.has_value() && (*native_prefill_chunks == GdnChunkMode::fused || *native_prefill_chunks == GdnChunkMode::cs64) &&
            native_prefill_dense_f32 == true && grouped_prefill_ok) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            if (!native_prefill_dense_staged_f32 && !native_prefill_dense_nax) gpu.configure_prefill_dense_staged_f32(true);
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
            gpu.configure_prefill_hc_rows(true);
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
            gpu.configure_prefill_router_window(true);
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
            gpu.configure_prefill_hc_down_window(true);
#endif
        }
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (gdn_ported.has_value() && (*native_prefill_chunks == GdnChunkMode::fused || *native_prefill_chunks == GdnChunkMode::cs64) &&
            native_prefill_dense_f32 == true && grouped_prefill_ok) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            if (!native_prefill_dense_staged_f32 && !native_prefill_dense_nax) gpu.configure_prefill_dense_staged_f32(true);
#endif
#ifdef TOKSPUR_PRIVATE_HC_ROWS
            if (!gpu.prefill_hc_rows()) gpu.configure_prefill_hc_rows(true);
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
            if (!gpu.prefill_router_window()) gpu.configure_prefill_router_window(true);
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
            if (!gpu.prefill_hc_down_window()) gpu.configure_prefill_hc_down_window(true);
#endif
        }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
        if (native_prefill_router_fast.has_value()) {
            require(!*native_prefill_router_fast || (gpu.prefill_router_window() && (gpu.prefill_dense_nax() || gpu.prefill_grouped_nax())),
                "fast BF16 prefill router requires enabled router window and NAX prefill");
            gpu.configure_prefill_router_fast(*native_prefill_router_fast);
        }
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        if(route_attribution_path){route_sink.open_directory(*route_attribution_path);gpu.configure_route_attribution(TargetRouteAttributionSink::emit,&route_sink);}
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(route_lock_output){route_lock_sink.open_directory(*route_lock_output,route_lock_reference,route_census_only);gpu.configure_route_lock(TargetRouteLockSink::emit,&route_lock_sink);}
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
        if (kernel_search_capture_path) {
            gpu.configure_prefill_dense_staged_f32(true); gpu.configure_prefill_hc_rows(true);
            kernel_search_sink.open_directory(*kernel_search_capture_path); gpu.configure_kernel_search_capture(TargetKernelSearchCaptureSink::emit, &kernel_search_sink);
        }
#endif
        const uint64_t chunked_budget_bytes = ((known_token_capacity >= 2048 || native_prefill_dense_nax || native_prefill_grouped_nax) ? 1792ull : (known_token_capacity >= 1024 ? 1024ull : 512ull)) << 20;
        stack.prepare_chunked_prefill(*native_prefill_chunks, chunked_budget_bytes
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
            , native_gdn_old_order_batch
#endif
            ); own_memory_guard();
    }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::unique_ptr<MtpSharedBinding> mtp_binding;
    std::unique_ptr<MtpHead> mtp;
    std::unique_ptr<DFlash2Drafter> dflash2;
    std::unique_ptr<PromptLookup> mtp_lookup;
    bool mtp_lookup_valid = false;
    uint64_t mtp_head_calls = 0, mtp_proposed = 0, mtp_prime_calls = 0, mtp_rounds = 0, mtp_accepted = 0, mtp_recoveries = 0;
    uint64_t mtp_lookup_rounds = 0, mtp_lookup_accepted = 0;
    uint64_t mtp_carried_anchors = 0, mtp_discarded_anchors = 0;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    uint64_t mtp_priming_batch_submits = 0;
#endif
    MtpGenerationCarry mtp_generation;
    double mtp_head_gpu_ms = 0, mtp_head_wall_ms = 0, mtp_prime_gpu_ms = 0, mtp_prime_wall_ms = 0;
    if (mtp_sidecar) {
        stack.prepare_final_hc_tap(
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            mtp_chunked_prefill ? (known_token_capacity >= 2048 ? (256ull << 20) : (known_token_capacity >= 1024 ? (128ull << 20) : (64ull << 20))) :
#endif
            1'000'000);
        stack.prepare_sampled_verification2(maximum_stack_verification_budget, {true, QsaVerifyRowsMode::core_and_q8});
        mtp_binding = std::make_unique<MtpSharedBinding>(source, *mtp_sidecar, MtpTemplatePolicy::target_rendered_ids_only, pack_model);
        mtp = std::make_unique<MtpHead>(gpu, *mtp_binding, mtp_borrowed_output, 65536, mtp_draft_attention, pack_model ? &pack_views : nullptr);
        if (native_mtp_draft_slice) {
            const auto draft_slice = gpu.suballocate_draft_output_slice(mtp_borrowed_output.view, 32768, 2560);
            static_cast<void>(gpu.allocate_draft_embedding_slice(32768, 2560));
            const GpuMatrix draft_matrix{draft_slice, mtp_borrowed_output.encoding};
            mtp->configure_draft_vocabulary(32768, draft_matrix);
        }
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        if (mtp_chunked_prefill) {
            const bool nax_priming =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                native_prefill_dense_nax || native_prefill_grouped_nax;
#else
                false;
#endif
            const uint32_t mtp_priming_cap = nax_priming ? known_token_capacity : std::min(known_token_capacity, 512u);
            mtp->prepare_priming_batch(mtp_priming_cap, ((mtp_priming_cap >= 1024 || native_prefill_dense_nax) ? 160ull : 32ull) << 20);
        }
#endif
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        if (!mtp->priming_batch_capacity() && (mtp_lookup_mode != MtpLookupMode::off || native_mtp_fast_round.value_or(false))) {
            mtp->prepare_priming_batch(4, 2ull << 20);
        }
#endif
        if (native_mtp_fast_round.value_or(false)) {
            mtp->configure_fast_round(true);
            stack.configure_gpu_final_hc_tap(true);
        }
        if (mtp_lookup_mode != MtpLookupMode::off) {
            LookupConfig lookup_cfg{};
            lookup_cfg.match_tokens = mtp_lookup_match;
            lookup_cfg.proposal_tokens = mtp_depth == 3 ? 3 : 2;
            lookup_cfg.history_tokens = 65536;
            lookup_cfg.token_limit = shape.vocabulary;
            lookup_cfg.memory_budget_bytes = 4ull << 20;
            mtp_lookup = std::make_unique<PromptLookup>(lookup_cfg);
            mtp_lookup_valid = true;
        }
        own_memory_guard();
    } else if (dflash2_dir) {
        require(mtp_borrowed_output.encoding == 8 &&
            gpu.prefill_weight_layout(mtp_borrowed_output.view) == PrefillWeightLayout::q8_lanes_v1,
            "DFlash2 requires borrowed q8_lanes_v1 output.weight from tokspur-pack-v2");
        dflash2 = std::make_unique<DFlash2Drafter>(gpu, *dflash2_dir, mtp_borrowed_output.view, known_token_capacity,
            [source](uint32_t token_id, std::span<float> dst) {
                const auto& tt = source->tensor("token_embd.weight");
                const auto rb = tt.bytes / 248320;
                dequantize(tt.type, source->tensor_bytes(tt.name).subspan(uint64_t(token_id) * rb, rb), dst);
            }, dflash2_precision, dflash2_defer_context, dflash2_checkpoint_identity_required(checkpoint_ssd || verify_cost_profile
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                || !carrier_identity_path.empty()
#endif
                ));
        dflash2->set_adaptive_budget(dflash2_adaptive_budget);
        if (dflash2_max_drafts.has_value()) dflash2->set_max_drafts(*dflash2_max_drafts);
        if (dflash2_vocab_slice.has_value()) dflash2->configure_draft_vocab_slice(*dflash2_vocab_slice);
        stack.prepare_dflash_tap(dflash2->target_layer_ids(), dflash2_tap_budget(
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            mtp_chunked_prefill ? known_token_capacity :
#endif
            8));
        stack.prepare_sampled_verification2(maximum_stack_verification_budget, {true, QsaVerifyRowsMode::core_and_q8});
        own_memory_guard();
    }
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    TargetMtpProfileSink mtp_profile_sink;
    mtp_profile_sink.boundary = mtp_profile_boundary.value_or(MtpProfileBoundary::dispatch);
    if (mtp_profile_rounds) {
        gpu.prepare_mtp_generation_profile(mtp_profile_rounds, mtp_profile_sink.boundary, TargetMtpProfileSink::emit, &mtp_profile_sink);
        own_memory_guard();
    }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    TargetTask4ProfileSink task4_sink;
    task4_sink.boundary = task4_profile_boundary.value_or(Task4ProfileBoundary::dispatch);
    if (task4_profile.value_or(false)) {
        gpu.prepare_task4_profile(16384, 8192, TargetTask4ProfileSink::emit, &task4_sink, task4_sink.boundary);
        counter_sampling = task4_sink.boundary == Task4ProfileBoundary::dispatch ? ProfileMode::dispatch_boundary : ProfileMode::encoder_boundary;
        own_memory_guard();
    }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    if (grouped_prefill_ffn) stack.configure_known_grouped_ffn(*grouped_prefill_ffn);
    if (prefill_layer_submissions) stack.configure_known_layer_submissions(*prefill_layer_submissions);
    if (prefill_hc_residual_rows) stack.configure_known_hc_residual_rows(*prefill_hc_residual_rows);
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    if (verification8_branch_reuse) stack.permit_verification8_branch_reuse();
#endif
    std::unique_ptr<TargetBranches> branches;
    if (checkpoint_slots) {
        auto prepared = std::make_unique<TargetBranches>(checkpoint_slots, shape.vocabulary);
        require(prepared->charged_bytes() < checkpoint_budget, "actual host branch storage exceeds combined budget");
        stack.prepare_checkpoints({checkpoint_slots, checkpoint_budget - prepared->charged_bytes()});
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        prepared->configure_policy_digests(gpu.kernel_policy(), gpu.target_m8_split_digest());
#endif
        branches = std::move(prepared); own_memory_guard();
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    CheckpointDrafter checkpoint_drafter{mtp.get(), dflash2.get()};
    auto* checkpoint_owner = checkpoint_drafter ? &checkpoint_drafter : nullptr;
    std::unique_ptr<SsdCheckpointManager> ssd_manager;
    std::array<uint8_t, 32> base_ssd_identity_digest{};
    std::vector<uint32_t> ssd_token_history;
    uint64_t ssd_spills = 0, ssd_restores = 0, ssd_restored_tokens = 0;
    uint32_t last_ssd_spill_prefix = 0;
    auto last_ssd_spill_time = Clock::now();
    SlotHandshake ssd_handshake;
    const auto ssd_execution_digest_for = [&](uint32_t exec_mode, bool res_reuse, std::array<uint32_t, 2> kp) -> std::array<uint8_t, 32> {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        CC_SHA256_Update(&ctx, base_ssd_identity_digest.data(), sizeof(base_ssd_identity_digest));
        CC_SHA256_Update(&ctx, &exec_mode, sizeof(exec_mode));
        const uint8_t rr = res_reuse ? 1u : 0u;
        CC_SHA256_Update(&ctx, &rr, sizeof(rr));
        CC_SHA256_Update(&ctx, kp.data(), sizeof(kp));
        std::array<uint8_t, 32> digest{};
        CC_SHA256_Final(digest.data(), &ctx);
        return digest;
    };
    if (checkpoint_ssd || verify_cost_profile
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        || !carrier_identity_path.empty()
#endif
        ) {
        CC_SHA256_CTX id_ctx;
        CC_SHA256_Init(&id_ctx);
        const auto feed_bytes = [&](std::span<const std::byte> data) {
            constexpr size_t max_chunk = 1ull << 28;
            size_t off = 0;
            while (off < data.size()) {
                const size_t n = std::min<size_t>(max_chunk, data.size() - off);
                CC_SHA256_Update(&id_ctx, data.data() + off, CC_LONG(n));
                off += n;
            }
        };
        const auto feed_pod = [&](const auto& v) {
            CC_SHA256_Update(&id_ctx, &v, CC_LONG(sizeof(v)));
        };
        const auto feed_str = [&](std::string_view s) {
            const uint64_t len = uint64_t(s.size());
            feed_pod(len);
            if (!s.empty()) feed_bytes(std::as_bytes(std::span(s.data(), s.size())));
        };
        feed_str("tokspur-execution-v3");
        for (const auto* file : {&identity_worker, &identity_library}) {
            file->assert_unchanged();
            const uint64_t size = file->size(); feed_pod(size); feed_bytes(file->bytes());
            file->assert_unchanged();
        }
        for (int i = 1; i + 1 < argc; ++i) {
            const std::string_view flag(argv[i]);
            if (flag.starts_with("--native-prefill-") || flag == "--prefill-router-fast" || flag == "--native-verify-nax" || flag == "--native-verify-nax-m8" || flag == "--native-verify-hc-gdn-fused") {
                feed_str(flag); feed_str(argv[i + 1]);
            }
        }
        const auto feed_opt_bool = [&](const std::optional<bool>& opt) {
            const uint8_t enc = !opt.has_value() ? 0xffu : (*opt ? 1u : 0u);
            feed_pod(enc);
        };
        const auto feed_file_stat = [&](const std::filesystem::path& p) {
            std::error_code ec;
            const auto canon = std::filesystem::weakly_canonical(p, ec);
            feed_str(ec ? p.string() : canon.string());
            struct stat st{};
            if (::stat(p.c_str(), &st) == 0) {
                const uint64_t dev = uint64_t(st.st_dev), ino = uint64_t(st.st_ino), sz = uint64_t(st.st_size);
                const int64_t msec = st.st_mtimespec.tv_sec, mnsec = st.st_mtimespec.tv_nsec;
                const int64_t csec = st.st_ctimespec.tv_sec, cnsec = st.st_ctimespec.tv_nsec;
                feed_pod(dev); feed_pod(ino); feed_pod(sz);
                feed_pod(msec); feed_pod(mnsec); feed_pod(csec); feed_pod(cnsec);
            }
        };
        const auto feed_model_files = [&](const ModelFiles& mf) {
            feed_str(mf.architecture());
            const uint64_t shard_count = mf.shards().size();
            feed_pod(shard_count);
            for (const auto& shard : mf.shards()) {
                feed_pod(shard.file_size());
                feed_pod(shard.data_offset());
                const uint64_t meta_count = shard.metadata().size();
                feed_pod(meta_count);
                for (const auto& [k, m] : shard.metadata()) {
                    feed_str(k);
                    feed_pod(m.type);
                    feed_pod(m.element_type);
                    feed_pod(m.elements);
                    const auto mb = shard.metadata_bytes(k);
                    const uint64_t mb_len = mb.size();
                    feed_pod(mb_len);
                    feed_bytes(mb);
                }
                const uint64_t tensor_count = shard.tensors().size();
                feed_pod(tensor_count);
                for (const auto& t : shard.tensors()) {
                    feed_str(t.name);
                    feed_pod(t.type);
                    const uint64_t nd = t.dimensions.size();
                    feed_pod(nd);
                    for (uint64_t d : t.dimensions) feed_pod(d);
                    feed_pod(t.offset);
                    feed_pod(t.bytes);
                    feed_pod(t.elements);
                    if (!shard.header_only() && t.bytes > 0) {
                        if (t.bytes <= 65536) {
                            feed_bytes(shard.tensor_bytes(t));
                        } else if (t.name == "token_embd.weight" || t.name == "per_layer_token_embd.weight" ||
                                   t.name == "output.weight" || t.name == "blk.0.attn_qkv.weight" ||
                                   t.name == "blk.3.attn_q.weight" || t.name == "blk.47.ffn_down_exps.weight" ||
                                   t.name == "blk.48.eh_proj.weight" || t.name == "blk.48.attn_output.weight" ||
                                   t.name == "blk.48.ffn_down_exps.weight") {
                            const auto tb = shard.tensor_bytes(t);
                            const size_t edge = std::min<size_t>(4096, tb.size());
                            feed_bytes(tb.first(edge));
                            feed_bytes(tb.last(edge));
                        }
                    }
                }
            }
        };

        feed_str("tokspur-ssd-execution-identity-v2");
        const uint64_t path_count = paths.size();
        feed_pod(path_count);
        for (const auto& p : paths) feed_file_stat(p);
        feed_model_files(files);
        if (ple_nocache_file.valid()) files.evict_tensor_pages(table.name);

        const uint8_t has_pack = pack ? 1u : (pack_v3 ? 3u : 0u);
        feed_pod(has_pack);
        if (pack) {
            if (derived_pack_path) feed_file_stat(*derived_pack_path);
            feed_str(pack->manifest_sha256);
            feed_str(pack->plan.recipe);
            feed_pod(pack->plan.version);
            feed_pod(pack->plan.alignment);
            feed_pod(pack->plan.file_bytes);
            feed_pod(pack->plan.payload_bytes);
            const uint64_t pt_count = pack->plan.tensors.size();
            feed_pod(pt_count);
            for (const auto& pt : pack->plan.tensors) {
                feed_str(pt.name);
                feed_pod(pt.type);
                feed_str(pt.layout);
                feed_pod(pt.bytes);
                feed_pod(pt.offset);
                feed_str(pt.source_sha256);
                feed_str(pt.packed_sha256);
            }
        } else if (pack_v3) {
            if (derived_pack_path) feed_file_stat(*derived_pack_path);
            feed_str(pack_v3->manifest_sha256);
            feed_str(pack_v3->plan.recipe);
            feed_pod(pack_v3->plan.version);
            feed_pod(pack_v3->plan.alignment);
            feed_pod(pack_v3->plan.file_bytes);
            feed_pod(pack_v3->plan.payload_bytes);
            const uint64_t pt_count = pack_v3->plan.tensors.size();
            feed_pod(pt_count);
            for (const auto& pt : pack_v3->plan.tensors) {
                feed_str(pt.name);
                feed_pod(pt.type);
                feed_pod(pt.source_type);
                feed_str(pt.layout);
                feed_pod(pt.bytes);
                feed_pod(pt.offset);
                feed_str(pt.source_sha256);
                feed_str(pt.packed_sha256);
            }
        }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        const uint8_t has_lanes = lane_weights.has_value() ? 1u : 0u;
        feed_pod(has_lanes);
        if (lane_weights) {
            if (weight_lanes) feed_file_stat(*weight_lanes);
            feed_str(lane_weights->manifest_sha256);
            feed_pod(lane_weights->plan.file_bytes);
            feed_pod(lane_weights->plan.payload_bytes);
            const uint64_t lt_count = lane_weights->plan.tensors.size();
            feed_pod(lt_count);
            for (const auto& lt : lane_weights->plan.tensors) {
                feed_str(lt.name);
                feed_pod(lt.type);
                feed_pod(lt.bytes);
                feed_str(lt.source_sha256);
                feed_str(lt.packed_sha256);
            }
        }
#endif
        const uint8_t has_mtp = mtp_sidecar.has_value() ? 1u : 0u;
        feed_pod(has_mtp);
        if (mtp_sidecar) {
            feed_file_stat(*mtp_sidecar);
            if (mtp_binding && mtp_binding->sidecar_source()) {
                feed_model_files(*mtp_binding->sidecar_source());
            }
        }

        if (dflash2) {
            feed_str("dflash2-context-checkpoint-v1");
            require(!dflash2->state_identity_sha256().empty(), "DFlash2 SSD checkpoint requires the full weight identity");
            feed_str(dflash2->state_identity_sha256());
            feed_pod(dflash2->precision());
            feed_pod(dflash2->draft_vocab_slice());
            for (auto layer : dflash2->target_layer_ids()) feed_pod(layer);
        }

        feed_str(options.dump());
        feed_str(layout.describe().dump());
        feed_pod(shape.vocabulary);
        feed_pod(shape.memory_layer);
        feed_pod(shape.memory.dilation);
        const uint32_t ple_gate_mode = uint32_t(shape.memory.gate_transform);
        feed_pod(ple_gate_mode);
        feed_pod(effective_policy.recurrent.rms_epsilon);
        feed_pod(effective_policy.attention.rms_epsilon);
        const uint32_t router_mode = uint32_t(effective_policy.router);
        feed_pod(router_mode);
        const uint8_t norm_sel = effective_policy.normalize_selected ? 1u : 0u;
        feed_pod(norm_sel);
        feed_pod(known_token_capacity);
        feed_pod(hc_q8.splits);
        const uint8_t overlap_u8 = overlap_prefill ? 1u : 0u;
        feed_pod(overlap_u8);
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        const uint32_t hc_exact_u32 = uint32_t(hc_exact);
        feed_pod(hc_exact_u32);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        const uint8_t wq8e = wide_q8_exact ? 1u : 0u;
        feed_pod(wq8e);
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        const uint8_t wq8m = wide_q8_many ? 1u : 0u;
        feed_pod(wq8m);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        const uint8_t q8_3 = q8_three_rows ? 1u : 0u;
        feed_pod(q8_3);
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        feed_opt_bool(native_raw_decode);
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        const uint8_t conc_u8 = concurrent_dispatch ? 1u : 0u;
        feed_pod(conc_u8);
#endif
#if defined(TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK) && !defined(TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC)
        feed_opt_bool(batch_phase_clock);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        const uint8_t qsa_prep = qsa_stage_policy.prepared ? 1u : 0u;
        const uint8_t qsa_en = qsa_stage_policy.enabled ? 1u : 0u;
        feed_pod(qsa_prep);
        feed_pod(qsa_en);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        feed_opt_bool(qsa_channel_chronological);
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
        feed_opt_bool(kernel_qsa_f32);
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
        feed_opt_bool(kernel_grouped_fp16);
#endif
#endif
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        feed_opt_bool(known_qsa_rows);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        const uint8_t sp8 = sampled_prefill_input8 ? 1u : 0u;
        feed_pod(sp8);
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        feed_pod(verification_max_tokens);
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        const uint8_t v8b = verification8_branch_reuse ? 1u : 0u;
        feed_pod(v8b);
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        const uint32_t vqr = verification_qsa_rows ? uint32_t(*verification_qsa_rows) : 0xffffffffu;
        feed_pod(vqr);
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        feed_opt_bool(grouped_prefill_ffn);
        feed_opt_bool(prefill_layer_submissions);
        feed_opt_bool(prefill_hc_residual_rows);
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        const uint32_t chunk_mode_u32 = native_prefill_chunks ? uint32_t(*native_prefill_chunks) : 0xffffffffu;
        feed_pod(chunk_mode_u32);
        feed_opt_bool(native_prefill_dense_f32);
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        const uint8_t dns_staged = native_prefill_dense_staged_f32 ? 1u : 0u;
        feed_pod(dns_staged);
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        feed_opt_bool(native_prefill_grouped_f32);
        const uint8_t grp_auto = native_prefill_grouped_auto ? 1u : 0u;
        feed_pod(grp_auto);
        feed_pod(router_margin_epsilon);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        feed_opt_bool(qsa_sparse);
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        feed_opt_bool(gdn_ported);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        const uint8_t pfl = native_prefill_final_logits ? 1u : 0u;
        const uint8_t mcp = mtp_chunked_prefill ? 1u : 0u;
        feed_pod(pfl);
        feed_pod(mcp);
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        const uint8_t hcd_f32 = native_prefill_hc_down_f32 ? 1u : 0u;
        feed_pod(hcd_f32);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
        feed_opt_bool(native_prefill_hc_rows);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
        feed_opt_bool(native_prefill_router_window);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
        feed_opt_bool(native_prefill_hc_down_window);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
        feed_opt_bool(native_qsa_sparse);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
        feed_opt_bool(native_gdn_ported);
        if (native_gdn_prefill_ported.has_value()) feed_opt_bool(native_gdn_prefill_ported);
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        const uint8_t gdn_old = native_gdn_old_order_batch ? 1u : 0u;
        feed_pod(gdn_old);
#endif
#endif
#endif
#endif
        feed_pod(mtp_depth);
        feed_pod(host_logit_views);
        feed_pod(dflash2_defer_context);
        if (dflash2_adaptive_budget) feed_pod(dflash2_adaptive_budget);
        if (native_logit_topk.has_value()) feed_opt_bool(native_logit_topk);
        if (native_ple_async_gather.has_value()) feed_opt_bool(native_ple_async_gather);
        if (native_mtp_fast_round.has_value()) feed_opt_bool(native_mtp_fast_round);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (native_verify_nax_m8.has_value()) feed_opt_bool(native_verify_nax_m8);
        if (native_verify_hc_gdn_fused.has_value()) feed_opt_bool(native_verify_hc_gdn_fused);
#endif
        if (dflash2_max_drafts.has_value()) feed_pod(*dflash2_max_drafts);
        if (dflash2_vocab_slice.has_value()) feed_pod(*dflash2_vocab_slice);
        const uint32_t mtp_da = uint32_t(mtp_draft_attention);
        feed_pod(mtp_da);
        const uint8_t mtp_la = mtp_lookahead_anchor ? 1u : 0u;
        const uint8_t mtp_rc = mtp_request_carry ? 1u : 0u;
        const uint8_t mtp_ds = native_mtp_draft_slice ? 1u : 0u;
        const uint32_t mtp_dv = mtp ? mtp->draft_vocabulary() : 0u;
        feed_pod(mtp_la);
        feed_pod(mtp_rc);
        feed_pod(mtp_ds);
        feed_pod(mtp_dv);
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        const uint32_t mtp_opl = mtp_output_packet_limit.value_or(0u);
        feed_pod(mtp_opl);
#endif
        if (!gpu.target_m8_split_digest().empty()) {
            feed_str("target-verification-M8-split-policy"); feed_str(gpu.target_m8_split_digest());
        }
#if defined(TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS) && TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS
        // Conservative experimental identity until complete equality qualifies
        // this kernel policy. The disabled build retains every legacy byte.
        feed_str("verify-narrow-rows-experiment-v1");
#endif
        feed_bytes(std::as_bytes(std::span(legacy_checkpoint_identity_tag)));
        CC_SHA256_Final(base_ssd_identity_digest.data(), &id_ctx);

        if (verify_cost_profile) {
            const auto digest = ssd_execution_digest_for(uint32_t(execution), reuse_residual, gpu.kernel_policy());
            std::string hex; const char* digits = "0123456789abcdef";
            for (uint8_t byte : digest) { hex += digits[byte >> 4]; hex += digits[byte & 15]; }
            verify_cost_profile->bind(hex, dflash2 ? MeasuredDrafter::dflash2 : MeasuredDrafter::mtp,
                prepared_sampling(), pack_shared_mapping);
            std::array<uint8_t, 32> active{};
            std::string effective_policy = verify_cost_profile->policy_digest() + (dflash2_auto_draft ? ";draft=auto" : ";draft=always");
            if (dflash2) {
                const auto& a = dflash2->verify_budget().acceptance();
                const auto append_float = [&](float value) {
                    const auto bits = std::bit_cast<uint32_t>(value);
                    for (unsigned shift = 0; shift < 32; shift += 8) effective_policy += char((bits >> shift) & 255);
                };
                append_float(a.slope); for (float value : a.intercepts) append_float(value);
                effective_policy += ";confidence-estimator-v1;labels=4096;refit=64";
            }
            const auto policy = derived_pack::hash(effective_policy);
            for (size_t i = 0; i < active.size(); ++i) {
                unsigned value = 0; const auto parsed = std::from_chars(policy.data() + 2 * i, policy.data() + 2 * i + 2, value, 16);
                require(parsed.ec == std::errc{}, "invalid effective controller digest"); active[i] = uint8_t(value);
            }
            mtp_generation.configure_controller_policy(active);
            if (branches) branches->configure_controller_policy(active, true);
            measured_controller = std::make_unique<MeasuredController>(*verify_cost_profile);
            if (dflash2) {
                measured_controller->set_ordinary_allowed(dflash2_auto_draft);
                dflash2->set_measured_budget([&](uint32_t context, std::span<const float, 7> confidence, uint32_t rows) {
                    const auto probabilities = dflash_measured_probabilities(*dflash2, confidence);
                    return measured_controller->after_draft_causal(context, rows, probabilities, verify_cost_profile->maximum_rows()).rows;
                });
            }
        }
        if (checkpoint_ssd) {
        const auto ssd_dir = checkpoint_ssd_dir.empty()
            ? std::filesystem::path(".tokspur/state/checkpoints")
            : checkpoint_ssd_dir;
        ssd_manager = std::make_unique<SsdCheckpointManager>(
            ssd_dir, checkpoint_ssd_budget, checkpoint_ssd_floor, checkpoint_policy);
        require(ssd_manager->enabled(), "failed to initialize SSD checkpoint directory");
        ssd_manager->set_execution_identity(
            ssd_execution_digest_for(uint32_t(execution), reuse_residual, gpu.kernel_policy()));
        ssd_token_history.reserve(65536);
        own_memory_guard();
        }
    }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (!carrier_identity_path.empty()) {
        const auto digest = ssd_execution_digest_for(uint32_t(execution), reuse_residual, gpu.kernel_policy());
        std::string hex; const char* digits = "0123456789abcdef";
        for (auto byte : digest) { hex += digits[byte >> 4]; hex += digits[byte & 15]; }
        const auto record = Json(Json::Object{{"schema", Json::integer(1)}, {"runtime_base", Json("8a72db4c202745420ec01b4bfc7d6f44e406c07d")},
            {"native_config", Json("carrier-diagnostic")}, {"execution_digest", Json(hex)}}).dump() + "\n";
        const int fd = ::open(carrier_identity_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        require(fd >= 0, "cannot exclusively create private execution identity");
        const auto written = ::write(fd, record.data(), record.size()); const auto closed = ::close(fd);
        require(written == ssize_t(record.size()) && closed == 0, "diagnostic execution identity write failed");
    }
#endif
#endif
    struct SharedPrefixAnchor {
        std::vector<uint32_t> tokens;
        std::array<uint32_t, 2> kernel_policy{};
        uint8_t chunk_mode = 0xff;
        bool has_mtp = false;
        bool has_logits = false;
        uint64_t stack_bytes = 0, mtp_bytes = 0;
        uint32_t mtp_first = 1, mtp_rows = 0;
        SamplingRng rng{};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        StageAControllerState controller_state{};
#endif
        std::vector<float> logits;
        std::vector<std::byte> payload;
        uint64_t last_access_tick = 0;
        uint64_t ram_bytes() const {
            return uint64_t(payload.size()) + uint64_t(logits.size()) * sizeof(float) + uint64_t(tokens.size()) * sizeof(uint32_t);
        }
    };
    struct SharedPrefixCacheState {
        enum : uint64_t { max_ram_bytes = 768ull << 20 };
        enum : size_t {
            max_anchors = 8,
            max_recent_sessions = 8,
            boundary_tokens = 512,
            max_prefix_tokens = 2048,
        };
        std::vector<SharedPrefixAnchor> anchors;
        std::vector<std::vector<uint32_t>> recent_prefixes;
        std::vector<uint32_t> active_tokens;
        bool active_valid = false;
        uint64_t tick = 0, occupied_bytes = 0, captures = 0, hits = 0, restored_tokens = 0;
#if defined(TOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS) && TOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS
        SharedPrefixEvictionCounters evictions{};
#endif
    } shared_prefix_cache;
    if (shared_prefix_cache_enabled) {
        shared_prefix_cache.active_tokens.reserve(SharedPrefixCacheState::max_prefix_tokens);
    }
#if defined(TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM) && TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM
    reclaim_shared_prefix = [&]() {
        uint64_t released = 0;
        for (const auto& anchor : shared_prefix_cache.anchors) released = checked_add(released, anchor.ram_bytes());
        std::vector<SharedPrefixAnchor>().swap(shared_prefix_cache.anchors);
        std::vector<std::vector<uint32_t>>().swap(shared_prefix_cache.recent_prefixes);
        std::vector<uint32_t>().swap(shared_prefix_cache.active_tokens);
        shared_prefix_cache.occupied_bytes = 0; shared_prefix_cache.active_valid = false;
        cache_reclaimed_bytes = checked_add(cache_reclaimed_bytes, released);
    };
#endif
    uint64_t ple_overlap_chunks = 0;
    std::unique_ptr<TargetLookup<StackResult>> lookup;
    if (prompt_lookup) {
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        auto prepared = std::make_unique<TargetLookup<StackResult>>(shape.vocabulary, shape.residual.channels, shape.memory.embedding, verification_budget,
            verification_max_tokens == 8 ? 7 : 2);
        if (verification_max_tokens == 8) stack.prepare_transient_verification8(verification_budget - prepared->charged_bytes()
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            , {verification_qsa_rows.has_value(), verification_qsa_rows.value_or(QsaVerifyRowsMode::off)}
#endif
            );
        else stack.prepare_transient_verification(verification_budget - prepared->charged_bytes());
#else
        auto prepared = std::make_unique<TargetLookup<StackResult>>(shape.vocabulary, shape.residual.channels, shape.memory.embedding, verification_budget);
        stack.prepare_transient_verification(verification_budget - prepared->charged_bytes());
#endif
        lookup = std::move(prepared); own_memory_guard();
    }
    bool lookup_active = bool(lookup);
    TargetSampler sampler;
    CommittedResult<StackResult> retained;
    std::unique_ptr<TargetBroker> broker;
    if (worker_broker) {
        std::random_device random; uint64_t epoch = 0;
        for (unsigned attempt = 0; !epoch && attempt < 8; ++attempt) epoch = (uint64_t(random()) << 32) ^ uint64_t(random());
        broker = std::make_unique<TargetBroker>(epoch
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            , worker_broker8
#endif
            );
    }
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    const auto branch_stage_fields = [&](Json::Object& fields) {
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        if (qsa_channel_chronological.has_value()) {
            require(ple_gate_transform == PleGateTransform::signed_root, "fast branch snapshot requires corrected PLE");
            const auto channel = gpu.qsa_channel_policy(); const auto calls = gpu.qsa_channel_calls();
            fields.emplace_back("ple_gate_transform", Json("signed-root")); fields.emplace_back("ple_gate_transform_schema", Json::integer(1));
            fields.emplace_back("qsa_channel_chronological_mode", Json(channel.enabled ? "on" : "off"));
            fields.emplace_back("qsa_channel_chronological_schema", Json::integer(1));
            fields.emplace_back("qsa_channel_factor_logical_bytes", Json::integer(channel.factor_logical_bytes));
            fields.emplace_back("qsa_channel_factor_charged_bytes", Json::integer(channel.factor_charged_bytes));
            fields.emplace_back("qsa_channel_calls", Json::integer(calls)); fields.emplace_back("qsa_channel_extra_dispatches", Json::integer(calls));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        if (qsa_stage_policy.prepared) {
            const auto calls = gpu.qsa_stage_v8_calls();
            fields.emplace_back("qsa_stage_v8_prepared", Json(true)); fields.emplace_back("qsa_stage_v8_enabled", Json(qsa_stage_policy.enabled));
            fields.emplace_back("qsa_stage_v8_scratch_logical_bytes", Json::integer(qsa_stage_policy.scratch_logical_bytes));
            fields.emplace_back("qsa_stage_v8_scratch_charged_bytes", Json::integer(qsa_stage_policy.scratch_charged_bytes));
            fields.emplace_back("qsa_stage_v8_calls", Json::integer(calls));
            fields.emplace_back("qsa_stage_v8_physical_dispatches", Json::integer(2 * calls));
            fields.emplace_back("qsa_stage_v8_extra_dispatches", Json::integer(calls));
        }
#else
        static_cast<void>(fields);
#endif
    };
    const auto branch8_snapshot = [&](const Json& current_envelope) {
        require(lookup && lookup_active && lookup->matches(stack.position()) && sampler.matches(stack.position()) &&
            branches && branches->matches(stack.position()), "branch8 snapshot requires coherent current model/lookup/sampler ancestry");
        const auto prefix = stack.position(); const auto stop = retained.stop_id(); const auto& cost = lookup->ledger();
        // A done already read/hash-checked these current logits. Reuse its
        // freshly constructed payload, never a historical reply; controls
        // compute their actual restored/current values once here.
        const auto* current_top = current_envelope.find("final_logit_top8");
        const auto* current_hash = current_envelope.find("hc_exact_final_logit_sha256");
        Json::Object fields{{"committed_prefix", Json::integer(prefix)}, {"sampling", sampling_json(sampler, prefix)},
            {"logits_available", Json(retained.matches(prefix))}, {"continuation_allowed", Json(retained.continuable(prefix))},
            {"stop_id", stop ? Json::integer(*stop) : Json()},
            {"final_logit_top8", prefix ? (current_top ? *current_top : top_logits(retained.get(prefix).logits)) : Json(Json::Array{})},
            {"hc_exact_final_logit_sha256", prefix ? (current_hash ? *current_hash : Json(hc_exact_logits_hash(retained.get(prefix).logits))) : Json()},
            {"hc_exact_routes", hc_routes_json(gpu.hc_exact_routes())}, {"lookup_history_epoch", Json::integer(lookup->epoch())},
            {"lookup_appended_tokens", Json::integer(cost.appended_tokens)}, {"lookup_indexed_grams", Json::integer(cost.indexed_grams)},
            {"lookup_truncated_grams", Json::integer(cost.truncated_grams)}, {"lookup_cleared_grams", Json::integer(cost.cleared_grams)},
            {"lookup_resets", Json::integer(cost.resets)}, {"lookup_truncations", Json::integer(cost.truncations)},
            {"lookup_queries", Json::integer(cost.queries)}, {"lookup_candidates_examined", Json::integer(cost.candidates_examined)},
            {"lookup_token_comparisons", Json::integer(cost.token_comparisons)}, {"lookup_work_limited_queries", Json::integer(cost.work_limited_queries)},
            {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)}, {"peak_metal_buffer_bytes", Json::integer(gpu.memory().peak_bytes)},
            {"gpu_allocation_count", Json::integer(gpu.memory().allocations)}, {"gpu_buffer_limit_bytes", Json::integer(gpu.memory().budget)},
            {"imported_weight_backing_bytes", Json::integer(gpu.memory().imported_backing_bytes)}};
        branch_stage_fields(fields); return Json(std::move(fields));
    };
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    const auto stage_envelope = [&](Json value) {
        if (!qsa_stage_v8_present) {
            // These are locally constructed diagnostic members, never incoming
            // wire data. An absent selector retains the historical wire shape.
            Json::Object fields;
            constexpr std::array<std::string_view, 9> names{"qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes",
                "qsa_stage_v8_scratch_charged_bytes", "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches",
                "qsa_stage_v8_request_calls", "qsa_stage_v8_round_calls"};
            for (const auto& entry : value.object()) if (std::find(names.begin(), names.end(), entry.first) == names.end()) fields.push_back(entry);
            value = Json(std::move(fields));
        }
        return value;
    };
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    // Only the explicit correctness proof allocates host scratch; hashing is outside GPU rounds.
    std::vector<std::byte> concurrent_state_scratch(concurrent_state_proof ? 1000000 : 0);
#endif
    const auto math_envelope = [&](Json value) {
#if defined(TOKSPUR_PRIVATE_MEMORY_PLAN_REPORT) && TOKSPUR_PRIVATE_MEMORY_PLAN_REPORT
        // Report-only: computed from readings the events already take; no budget reads it.
        if (const auto* event = value.find("event"); event && (event->string() == "ready" || event->string() == "done")) {
            const auto now = memory();
            const auto plan = memory_plan_room(native_limit, now.current, memory_plan_reserve_bytes);
            const uint64_t staging = ssd_manager && ssd_manager->enabled() ? uint64_t(staging_slab_bytes) : 0;
            auto fields = value.object();
            fields.emplace_back("memory_plan", Json(Json::Object{{"schema", Json::integer(1)}, {"applied", Json(false)},
                {"engine_limit_bytes", Json::integer(native_limit)}, {"physical_footprint_bytes", Json::integer(now.current)},
                {"lifetime_peak_footprint_bytes", Json::integer(now.peak)},
                {"gpu_ledger_live_bytes", Json::integer(gpu.memory().live_bytes)},
                {"gpu_ledger_peak_bytes", Json::integer(gpu.memory().peak_bytes)},
                {"imported_weight_backing_bytes", Json::integer(gpu.memory().imported_backing_bytes)},
                {"row_cache_charged_bytes", Json::integer(cache.charged_bytes())},
                {"shared_prefix_anchor_bytes", Json::integer(shared_prefix_cache.occupied_bytes)},
                {"ssd_staging_bytes", Json::integer(staging)}, {"reserve_bytes", Json::integer(memory_plan_reserve_bytes)},
                {"room_bytes", Json::integer(plan.room)}, {"rule", Json(std::string("half_of_room"))},
                {"would_allow_cache_bytes", Json::integer(plan.would_allow_cache_bytes)},
                {"unavailable", Json(Json::Array{Json(std::string("session_state_bytes")), Json(std::string("drafter_state_bytes"))})}}));
            value = Json(std::move(fields));
        }
#endif
#if defined(TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM) && TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM
        if (const auto* event = value.find("event"); event && (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object(); fields.emplace_back("native_cache_pressure_reclaim", Json(true));
            fields.emplace_back("cache_reclamations", Json::integer(cache_reclamations));
            fields.emplace_back("cache_reclaimed_accounted_bytes", Json::integer(cache_reclaimed_bytes));
            fields.emplace_back("cache_allocator_released_bytes", Json::integer(cache_allocator_released_bytes));
            value = Json(std::move(fields));
        }
#endif
        if (const auto* event = value.find("event"); pack_shared_mapping && event &&
            (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object(); fields.emplace_back("native_pack_shared_mapping", Json(true));
            value = Json(std::move(fields));
        }
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (const auto* event = value.find("event"); event && (event->string() == "ready" || event->string() == "done" || event->string() == "mtp_round")) {
            auto fields = value.object();
            fields.emplace_back("gpu_lifetime_counters", Json(Json::Object{
                {"dispatches", Json::integer(gpu.lifetime_dispatches())},
                {"command_buffers", Json::integer(gpu.lifetime_command_buffers())},
                {"barriers", Json::integer(gpu.lifetime_barriers())}}));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        const auto* concurrent_event = value.find("event");
        if (concurrent_selected && concurrent_event && (concurrent_event->string() == "ready" || concurrent_event->string() == "done")) {
            const bool terminal = concurrent_event->string() == "done"; auto fields = value.object();
            fields.emplace_back("concurrent_dispatch_schema", Json::integer(1));
            fields.emplace_back("concurrent_dispatch_mode", Json(gpu.concurrent_dispatch() ? "on" : "off"));
            fields.emplace_back("concurrent_dispatch_barriers", Json::integer(gpu.concurrent_barriers()));
            fields.emplace_back("concurrent_state_proof", Json(concurrent_state_proof));
            if (terminal && concurrent_state_proof) {
                const auto state = stack.accepted_fingerprint(concurrent_state_scratch);
                constexpr char hex[] = "0123456789abcdef"; std::string hash; hash.reserve(64);
                for (auto byte : state.sha256) { hash += hex[byte >> 4]; hash += hex[byte & 15]; }
                fields.emplace_back("concurrent_target_state_sha256", Json(hash));
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (!mtp)
#endif
                    fields.emplace_back("concurrent_state_sha256", Json(hash));
                fields.emplace_back("concurrent_target_state_bytes", Json::integer(state.bytes));
            }
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
        if (sampled_cost_manifest) {
            require(!mtp_head_calls && !mtp_prime_calls && !mtp_rounds, "sampled COST must not execute draft or prime work");
            auto fields = value.object();
            fields.emplace_back("sampled_verification_cost_protocol",Json("sampled-current-cost-v1"));
            fields.emplace_back("sampled_verification_cost_source_sha256",Json(sampled_cost_manifest->sealed_source_sha256));
            fields.emplace_back("sampled_verification_cost_max_rows",Json::integer(3));
            fields.emplace_back("production_sampled_max_rows",Json::integer(2));
            fields.emplace_back("sampled_verification_cost_head_calls",Json::integer(mtp_head_calls));
            fields.emplace_back("sampled_verification_cost_prime_calls",Json::integer(mtp_prime_calls));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        const auto* profile_event = value.find("event");
        if (mtp_profile_rounds && profile_event && (profile_event->string() == "ready" || profile_event->string() == "done")) {
            const auto window = gpu.mtp_profile_window(); auto fields = value.object();
            fields.emplace_back("mtp_profile_rounds_requested", Json::integer(window.requested));
            fields.emplace_back("mtp_profile_rounds_captured", Json::integer(window.rounds));
            fields.emplace_back("mtp_profile_batches_captured", Json::integer(window.batches));
            fields.emplace_back("mtp_profile_mode", Json(mtp_profile_boundary_name(mtp_profile_sink.boundary)));
            fields.emplace_back("mtp_profile_scope", Json("generation_only"));
            fields.emplace_back("mtp_profile_dispatch_capacity", Json::integer(mtp_profile_dispatch_capacity));
            fields.emplace_back("mtp_profile_counter_payload_bytes", Json::integer(gpu.memory().profile_counter_payload_bytes));
            fields.emplace_back("mtp_profile_prefill", Json(false));
            fields.emplace_back("mtp_profile_intrusive", Json(true));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        const auto* lane_event = value.find("event");
        if (lane_weights && lane_event && (lane_event->string() == "ready" || lane_event->string() == "done" || lane_event->string() == "control_done")) {
            lane_weights->model->assert_unchanged();
            auto fields = value.object();
            fields.emplace_back("weight_lanes_schema", Json::integer(lanes::sidecar_version));
            fields.emplace_back("weight_lanes_manifest_sha256", Json(lane_weights->manifest_sha256));
            fields.emplace_back("weight_lanes_tensor_count", Json::integer(lane_weights->plan.tensors.size()));
            fields.emplace_back("weight_lanes_file_bytes", Json::integer(lane_weights->plan.file_bytes));
            fields.emplace_back("weight_lanes_original_payload_bytes_skipped", Json::integer(lane_weights->plan.payload_bytes));
            if (weight_lanes_auth_workers) {
                fields.emplace_back("lane_auth_workers_requested", Json::integer(*weight_lanes_auth_workers));
                fields.emplace_back("lane_auth_workers", Json::integer(lane_weights->authentication_workers));
            }
            value = Json(std::move(fields));
        }
#endif
        const auto* pack_event = value.find("event");
        if ((pack || pack_v3) && pack_event && (pack_event->string() == "ready" || pack_event->string() == "done" || pack_event->string() == "control_done")) {
            pack_model->assert_unchanged();
            auto fields = value.object();
            if (pack) {
                fields.emplace_back("derived_pack_schema", Json::integer(derived_pack::pack_version));
                fields.emplace_back("derived_pack_recipe", Json(derived_pack::pack_recipe));
                fields.emplace_back("derived_pack_manifest_sha256", Json(pack->manifest_sha256));
                fields.emplace_back("derived_pack_tensor_count", Json::integer(pack->plan.tensors.size()));
                fields.emplace_back("derived_pack_file_bytes", Json::integer(pack->plan.file_bytes));
                fields.emplace_back("derived_pack_payload_bytes", Json::integer(pack->plan.payload_bytes));
                fields.emplace_back("derived_pack_auth_workers", Json::integer(pack->authentication_workers));
            } else {
                fields.emplace_back("derived_pack_schema", Json::integer(planar_v3::pack_version_v3));
                fields.emplace_back("derived_pack_recipe", Json(planar_v3::pack_recipe_v3));
                fields.emplace_back("derived_pack_manifest_sha256", Json(pack_v3->manifest_sha256));
                fields.emplace_back("derived_pack_tensor_count", Json::integer(pack_v3->plan.tensors.size()));
                fields.emplace_back("derived_pack_file_bytes", Json::integer(pack_v3->plan.file_bytes));
                fields.emplace_back("derived_pack_payload_bytes", Json::integer(pack_v3->plan.payload_bytes));
                fields.emplace_back("derived_pack_auth_workers", Json::integer(pack_v3->authentication_workers));
                fields.emplace_back("derived_pack_auth_source_read_mode", Json(std::string(pack_v3->authentication_source_read_mode)));
            }
            value = Json(std::move(fields));
        }
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
        const auto* trace_event = value.find("event");
        if (logit_trace && trace_event && (trace_event->string() == "ready" || trace_event->string() == "done")) {
            auto fields = value.object();
            fields.emplace_back("full_logit_trace_schema", Json::integer(1));
            fields.emplace_back("full_logit_trace_rows", Json::integer(logit_trace->rows()));
            fields.emplace_back("full_logit_trace_limit", Json::integer(logit_trace->limit()));
            fields.emplace_back("full_logit_trace_bytes", Json::integer(logit_trace->bytes()));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        if (!qsa_channel_chronological.has_value()) {
            // Strip only our two locally constructed optional diagnostic fields.
            // Incoming wire data is never rewritten or granted unknown fields.
            Json::Object fields;
            for (const auto& entry : value.object()) if (entry.first != "qsa_channel_request_calls" && entry.first != "qsa_channel_round_calls") fields.push_back(entry);
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        const auto* rows_event = value.find("event");
        if (verification_qsa_rows.has_value() && rows_event && (rows_event->string() == "ready" ||
            rows_event->string() == "done" || rows_event->string() == "verification_cost_case")) {
            const auto memory = stack.verification_qsa_rows_memory(); const auto routes = gpu.qsa_verify_rows_routes();
            auto fields = value.object();
            fields.emplace_back("verification_qsa_rows_schema", Json::integer(1));
            fields.emplace_back("verification_qsa_rows_mode", Json(qsa_verify_rows_name(memory.config.mode)));
            fields.emplace_back("verification_qsa_rows_logical_bytes", Json::integer(memory.logical_bytes));
            fields.emplace_back("verification_qsa_rows_charged_bytes", Json::integer(memory.charged_bytes));
            fields.emplace_back("verification_qsa_rows_allocations", Json::integer(memory.allocations));
            fields.emplace_back("verification_qsa_rows_borrowed_common_views", Json::integer(memory.borrowed_common_views));
            fields.emplace_back("verification_qsa_rows_physical_tiles", Json::integer(routes[0]));
            fields.emplace_back("verification_qsa_rows_logical_rows", Json::integer(routes[1]));
            fields.emplace_back("verification_qsa_rows_score_passes", Json::integer(routes[2]));
            fields.emplace_back("verification_qsa_rows_factor_passes", Json::integer(routes[3]));
            fields.emplace_back("verification_qsa_rows_value_passes", Json::integer(routes[4]));
            fields.emplace_back("verification_qsa_rows_http_admitted", Json(false));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        const auto* known_rows_event = value.find("event");
        if ((known_qsa_rows.has_value()
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            || mtp_chunked_prefill
#endif
            ) && known_rows_event &&
            (known_rows_event->string() == "ready" || known_rows_event->string() == "done")) {
            const auto memory = stack.known_qsa_rows_memory(); const auto routes = gpu.qsa_verify_rows_routes();
            auto fields = value.object();
            fields.emplace_back("known_qsa_rows_schema", Json::integer(1));
            fields.emplace_back("known_qsa_rows_mode", Json(memory.enabled ? "q8" : "off"));
            fields.emplace_back("known_qsa_rows_logical_bytes", Json::integer(memory.logical_bytes));
            fields.emplace_back("known_qsa_rows_charged_bytes", Json::integer(memory.charged_bytes));
            fields.emplace_back("known_qsa_rows_allocations", Json::integer(memory.allocations));
            fields.emplace_back("known_qsa_rows_physical_tiles", Json::integer(routes[0]));
            fields.emplace_back("known_qsa_rows_logical_rows", Json::integer(routes[1]));
            fields.emplace_back("known_qsa_rows_http_admitted", Json(
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
                sampled_prefill_input8
#else
                false
#endif
            ));
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        if (sampled_prefill_input8) {
            const auto* event = value.find("event");
            if (event && (event->string() == "ready" || event->string() == "done")) {
                auto fields = value.object();
                fields.emplace_back("sampled_prefill_protocol", Json("sampled-prefill-input8-v1"));
                value = Json(std::move(fields));
            }
        }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp || dflash2) {
            const auto* event = value.find("event");
            if (event && (event->string() == "ready" || event->string() == "done")) {
                auto fields = value.object();
                if (verify_cost_profile) {
                    Json::Array policy; for (auto byte : mtp_generation.controller_policy()) policy.push_back(Json::integer(byte));
                    fields.emplace_back("native_controller_policy_digest", Json(std::move(policy)));
                    fields.emplace_back("native_measured_controller", Json(true));
                    if (event->string() == "ready") {
                        Json::Array modes;
                        for (const auto sampling : {MeasuredSampling::greedy, MeasuredSampling::full, MeasuredSampling::topk, MeasuredSampling::coding})
                            if (verify_cost_profile->supports_sampling(sampling)) modes.emplace_back(std::string(measured_sampling_name(sampling)));
                        fields.emplace_back("native_controller_sampling_modes", Json(std::move(modes)));
                    }
                    fields.emplace_back("native_measured_timing_schema", Json::integer(1));
                    fields.emplace_back("native_measured_decode_observations", measured_controller->timing_packet());
                }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                if (carrier_single_action) fields.emplace_back("carrier_diagnostic_single_action", Json(true));
                if (carrier_batch_only) fields.emplace_back("carrier_diagnostic_batch_only", Json(true));
#endif
                if (mtp_request_carry) {
                    fields.emplace_back("native_mtp_carry_scope", Json("http-generation-v2"));
                    fields.emplace_back("native_mtp_generation", mtp_generation.metadata(stack.position()));
                }
                fields.emplace_back("native_mtp_protocol", Json(dflash2 ? "native-dflash2-depth7-lookahead-v2" :
                    mtp_depth == 3 ? "native-mtp-depth3-lookahead-v2" :
                    mtp_depth == 2 ? "native-mtp-depth2-lookahead-v2" : mtp_lookahead_anchor ?
                    "native-mtp-depth1-lookahead-v2" : "native-mtp-depth1-consumed-v1"));
                if (mtp_depth == 2 || mtp_depth == 3 || mtp_depth == 7) fields.emplace_back("native_mtp_tail_policy", Json("commit-through; round overflow drained across packets"));
                fields.emplace_back("native_mtp_lookahead_anchor", Json(mtp_lookahead_anchor));
                fields.emplace_back("native_mtp_depth", Json::integer(mtp_depth));
                if (prepared_sampling()) fields.emplace_back("native_prepared_validation", Json(true));
#if defined(TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS) && TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS
                fields.emplace_back("native_verify_narrow_rows", Json(true));
#endif
                if (dflash2_auto_draft) fields.emplace_back("native_dflash2_draft_policy", Json("auto"));
                if (!gpu.target_m8_split_digest().empty()) fields.emplace_back("native_target_m8_split_digest", Json(gpu.target_m8_split_digest()));
                fields.emplace_back("native_host_logit_views", Json(stack.sampled_logit_views()));
                if (native_logit_topk.has_value()) fields.emplace_back("native_logit_topk", Json(stack.sampled_logit_topk()));
                if (native_ple_async_gather.has_value()) fields.emplace_back("native_ple_async_gather", Json(*native_ple_async_gather));
                if (native_mtp_fast_round.has_value()) fields.emplace_back("native_mtp_fast_round", Json(mtp && mtp->fast_round()));
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                if (native_verify_hc_gdn_fused.has_value()) fields.emplace_back("native_verify_hc_gdn_fused", Json(gpu.verify_hc_gdn_fused_enabled()));
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
                if (native_prefill_router_fast.has_value()) fields.emplace_back("native_prefill_router_fast", Json(gpu.prefill_router_fast()));
#endif
                fields.emplace_back("native_dflash2_defer_context", Json(dflash2 && dflash2->deferred_context()));
                if (dflash2) {
                    fields.emplace_back("native_dflash2_precision", Json(dflash2->precision() == DFlash2Precision::bf16 ? "bf16" : "q8_0"));
                    fields.emplace_back("native_dflash2_verify_rows", Json(dflash2->adaptive_budget() ? "auto" : "all"));
                    if (dflash2_max_drafts.has_value()) fields.emplace_back("native_dflash2_max_drafts", Json::integer(dflash2->max_drafts()));
                    if (dflash2_vocab_slice.has_value()) fields.emplace_back("native_dflash2_vocab_slice", Json::integer(dflash2->draft_vocab_slice()));
                }
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
                if (mtp_output_packet_limit) {
                    fields.emplace_back("native_mtp_output_packet_schema", Json::integer(1));
                    fields.emplace_back("native_mtp_output_packet", Json::integer(*mtp_output_packet_limit));
                }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                if (mtp_chunked_prefill) {
                    const uint32_t mtp_input_packet_max =
                        (known_token_capacity == 2048 && ple_overlap && native_prefill_dense_nax && native_prefill_grouped_nax)
                            ? 8192u
                            : known_token_capacity;
                    fields.emplace_back("native_mtp_input_packet_schema", Json::integer(1));
                    fields.emplace_back("native_mtp_input_packet_max", Json::integer(mtp_input_packet_max));
                    fields.emplace_back("native_mtp_priming_batch_capacity", Json::integer(dflash2 ? dflash2->priming_batch_capacity() : mtp->priming_batch_capacity()));
                    fields.emplace_back("native_mtp_priming_batch_bytes", Json::integer(dflash2 ? dflash2->priming_batch_bytes() : mtp->priming_batch_bytes()));
                    fields.emplace_back("native_mtp_priming_batch_submits", Json::integer(mtp_priming_batch_submits));
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
                    if (native_gdn_old_order_batch) {
                        fields.emplace_back("native_gdn_old_order_version", Json("old-order-batch-v1"));
                        fields.emplace_back("native_gdn_old_order_kernel", Json("gdn_chunk_legacy_batch"));
                        fields.emplace_back("native_gdn_old_order_layers", Json::integer(gdn_old_order_layers));
                        fields.emplace_back("native_gdn_old_order_scratch_bytes", Json::integer(0));
                        fields.emplace_back("native_gdn_old_order_encoded_calls", Json::integer(gpu.gdn_legacy_batch_encoded_calls()));
                        fields.emplace_back("native_gdn_old_order_request_encoded_calls", Json::integer(gpu.gdn_legacy_batch_encoded_calls() - gdn_old_order_request_before));
                    }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
                    if (native_prefill_hc_down_f32) for (const auto& field : mtp_hc_down_f32_fields(gpu.prefill_hc_down_f32_encoded_calls(), gpu.prefill_hc_down_f32_encoded_calls() - hc_down_f32_request_before)) fields.push_back(field);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
                    if (native_prefill_hc_rows.value_or(false)) {
                        const auto encoded = gpu.hc_rows_encoded(); std::array<uint64_t, 3> request{};
                        for (size_t i = 0; i < 3; ++i) { require(encoded[i] >= hc_rows_request_before[i], "HC row counter rewound"); request[i] = encoded[i] - hc_rows_request_before[i]; }
                        for (const auto& field : mtp_hc_rows_fields(encoded, request)) fields.push_back(field);
                    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
                    if (native_prefill_router_window.value_or(false)) {
                        const auto encoded = gpu.router_window_encoded(); std::array<uint64_t, 4> request{};
                        for (size_t i = 0; i < 4; ++i) { require(encoded[i] >= router_window_request_before[i], "router window counter rewound"); request[i] = encoded[i] - router_window_request_before[i]; }
                        for (const auto& field : mtp_router_window_fields(encoded, request)) fields.push_back(field);
                    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
                    if (native_prefill_hc_down_window.value_or(false)) {
                        const auto encoded = gpu.hc_down_window_encoded();
                        require(encoded >= hc_down_window_request_before, "HC down window counter rewound");
                        for (const auto& field : mtp_hc_down_window_fields(encoded, encoded - hc_down_window_request_before)) fields.push_back(field);
                    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
                    if (native_qsa_sparse.value_or(false)) {
                        const std::array<uint64_t, 2> completed{gpu.qsa_sparse_decode_calls(), gpu.qsa_sparse_verify_calls()};
                        std::array<uint64_t, 2> request{};
                        for (size_t i = 0; i < 2; ++i) { require(completed[i] >= qsa_sparse_request_before[i], "sparse QSA counter rewound"); request[i] = completed[i] - qsa_sparse_request_before[i]; }
                        for (const auto& field : mtp_qsa_sparse_fields(completed, request)) fields.push_back(field);
                    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
                    if (native_gdn_ported.value_or(false)) {
                        const auto completed = gpu.gdn_ported_routes();
                        GdnPortedRoutes request{};
                        for (size_t i = 0; i < 5; ++i) { require(completed[i] >= gdn_ported_request_before[i], "ported GDN counter rewound"); request[i] = completed[i] - gdn_ported_request_before[i]; }
                        for (const auto& field : mtp_gdn_ported_fields(completed, request)) fields.push_back(field);
                    }
                    if (native_gdn_prefill_ported.has_value()) {
                        fields.emplace_back("native_prefill_gdn_tb12", Json(*native_gdn_prefill_ported));
                    }
#endif
                    if (native_prefill_final_logits) {
                        fields.emplace_back("native_prefill_output_policy", Json("final-input-v1"));
                        fields.emplace_back("logits_available", Json(retained.matches(stack.position())));
                    }
                }
#endif
                fields.emplace_back("native_mtp_attention", Json(dflash2 ? "sliding-window-2048" : "dense-compression0"));
                fields.emplace_back("native_mtp_draft_attention", Json(mtp_draft_attention == MtpDraftAttention::serial ? "serial" :
                    mtp_draft_attention == MtpDraftAttention::tiled128 ? "tiled128" :
                    mtp_draft_attention == MtpDraftAttention::tiled256 ? "tiled256" : "split16"));
                fields.emplace_back("native_mtp_first_cache_position", Json::integer(dflash2 ? 0 : 1));
                fields.emplace_back("native_mtp_head_calls", Json::integer(mtp_head_calls));
                fields.emplace_back("native_mtp_prime_calls", Json::integer(mtp_prime_calls));
                fields.emplace_back("native_mtp_head_gpu_ms", number(mtp_head_gpu_ms));
                fields.emplace_back("native_mtp_head_wall_ms", number(mtp_head_wall_ms));
                fields.emplace_back("native_mtp_prime_gpu_ms", number(mtp_prime_gpu_ms));
                fields.emplace_back("native_mtp_prime_wall_ms", number(mtp_prime_wall_ms));
                fields.emplace_back("native_mtp_rounds", Json::integer(mtp_rounds));
                fields.emplace_back("native_mtp_proposed", Json::integer(mtp_proposed));
                fields.emplace_back("native_mtp_accepted", Json::integer(mtp_accepted));
                fields.emplace_back("native_mtp_recovery_rounds", Json::integer(mtp_recoveries));
                fields.emplace_back("native_mtp_carried_anchors", Json::integer(mtp_carried_anchors));
                fields.emplace_back("native_mtp_discarded_anchors", Json::integer(mtp_discarded_anchors));
                fields.emplace_back("native_mtp_workspace_bytes", Json::integer(dflash2 ? dflash2->workspace_bytes() : mtp->workspace_bytes()));
                fields.emplace_back("native_mtp_visible_frontier", Json(mtp_lookahead_anchor ?
                    "all emitted tokens physically consumed; sampled terminal carried invisibly within request" :
                    "all emitted tokens physically consumed; terminal materialised"));
                if (dflash2) {
                    fields.emplace_back("native_drafter", Json("dflash2"));
                    fields.emplace_back("native_dflash2_checkpoint_sha256", Json(dflash2->checkpoint_sha256()));
                    fields.emplace_back("native_dflash2_state_identity_sha256", Json(dflash2->state_identity_sha256()));
                    fields.emplace_back("native_dflash2_context_position", Json::integer(dflash2->context_len()));
                    fields.emplace_back("native_dflash2_first_cache_position", Json::integer(dflash2->first_absolute_position()));
                    fields.emplace_back("native_dflash2_cache_rows", Json::integer(dflash2->cache_rows()));
                    fields.emplace_back("native_dflash2_weight_bytes", Json::integer(dflash2->weight_bytes()));
                }
                if (native_mtp_draft_slice) {
                    fields.emplace_back("native_mtp_draft_slice", Json(true));
                    fields.emplace_back("native_mtp_draft_vocabulary", Json::integer(32768));
                }
                value = Json(std::move(fields));
            }
        }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (ssd_manager && ssd_manager->enabled()) {
            const auto* ssd_event = value.find("event");
            if (ssd_event && (ssd_event->string() == "ready" || ssd_event->string() == "done" || ssd_event->string() == "control_done")) {
                auto fields = value.object();
                fields.emplace_back("checkpoint_ssd_enabled", Json(true));
                fields.emplace_back("checkpoint_ssd_policy", Json(ssd_manager->eviction_policy()));
                fields.emplace_back("checkpoint_ssd_budget_bytes", Json::integer(ssd_manager->budget_bytes()));
                fields.emplace_back("checkpoint_ssd_occupied_bytes", Json::integer(ssd_manager->occupied_bytes()));
                fields.emplace_back("checkpoint_ssd_spills", Json::integer(ssd_spills));
                fields.emplace_back("checkpoint_ssd_restores", Json::integer(ssd_restores));
                fields.emplace_back("checkpoint_ssd_restored_tokens", Json::integer(ssd_restored_tokens));
                value = Json(std::move(fields));
            }
        }
#endif
        if (overlap_prefill) {
            const auto* ov_event = value.find("event");
            if (ov_event && ov_event->string() == "ready") {
                auto fields = value.object();
                fields.emplace_back("overlap_prefill_enabled", Json(true));
                value = Json(std::move(fields));
            }
        }
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
        value = ple_gate_metadata(std::move(value), ple_gate_transform);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        value = qsa_channel_metadata(std::move(value), gpu);
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
        if (kernel_grouped_fp16.has_value()) { const auto* event = value.find("event"); if (event && (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object(); fields.emplace_back("kernel_step2_candidate", Json(*kernel_grouped_fp16 ? "grouped-fp16" : "off"));
            fields.emplace_back("kernel_step2_grouped_kernel", Json(*kernel_grouped_fp16 ? "graph_grouped_search_fp16" : "graph_grouped_mm_f32"));
            fields.emplace_back("kernel_step2_grouped_operand_precision", Json(*kernel_grouped_fp16 ? "fp16" : "fp32"));
            fields.emplace_back("kernel_step2_grouped_encoded_calls", Json::integer(gpu.kernel_grouped_fp16_encoded_calls())); value = Json(std::move(fields));
        } }
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
        if (kernel_qsa_f32.has_value()) { const auto* event = value.find("event"); if (event && (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object(); fields.emplace_back("kernel_step2_candidate", Json(*kernel_qsa_f32 ? "qsa-f32" : "off"));
            fields.emplace_back("kernel_step2_qsa_channels", Json(*kernel_qsa_f32 ? "qsa_channels_search_f32" : "qsa_chronological_channels"));
            fields.emplace_back("kernel_step2_qsa_rows_channels", Json(*kernel_qsa_f32 ? "qsa_rows_channels_search_f32" : "qsa_verify_rows_channels")); value = Json(std::move(fields));
        } }
#endif
#endif
#if defined(TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK) && !defined(TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC)
        const auto* event = value.find("event");
        if (event && (event->string() == "ready" || event->string() == "done"))
            value = batch_phase_metadata(std::move(value), gpu.batch_phase_clock_state());
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        const auto* concurrent_token_event = value.find("event");
        if (concurrent_selected && concurrent_token_event && concurrent_token_event->string() == "token"
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            && !mtp && !dflash2
#endif
            ) {
            auto fields = value.object();
            fields.emplace_back("concurrent_dispatch_schema", Json::integer(1));
            fields.emplace_back("concurrent_dispatch_mode", Json(gpu.concurrent_dispatch() ? "on" : "off"));
            fields.emplace_back("concurrent_dispatch_barriers", Json::integer(gpu.concurrent_barriers()));
            fields.emplace_back("concurrent_state_proof", Json(concurrent_state_proof));
            if (concurrent_state_proof) fields.emplace_back("concurrent_logit_sha256", retained.matches(stack.position()) ? Json(hc_exact_logits_hash(retained.get(stack.position()).logits)) : Json());
            value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (const auto* event = value.find("event"); native_raw_decode.has_value() && event && (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object(); const auto completed = gpu.raw_decode_routes();
            const auto request = event->string() == "ready" ? MetalGraph::RawDecodeRoutes{} : raw_decode_routes_delta(completed, raw_decode_request_before);
            auto metadata = raw_decode_metadata_fields(*native_raw_decode, completed, request);
            fields.insert(fields.end(), metadata.begin(), metadata.end()); value = Json(std::move(fields));
        }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if (const auto* event = value.find("event"); event && (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object();
            fields.emplace_back("q8_three_rows_schema",Json::integer(1));
            fields.emplace_back("q8_three_rows_mode",Json(gpu.q8_three_rows_enabled()?"on":"off"));
            fields.emplace_back("q8_three_rows_routes",q8_three_routes_json(gpu.q8_three_rows_routes()));
            fields.emplace_back("q8_three_rows_padding_rows",Json::integer(0));
            fields.emplace_back("q8_three_rows_hc_route",Json("generic-exact-fp32"));
            value = Json(std::move(fields));
        }
#endif
        if (candidate_opt_selected) {
            if (const auto* event = value.find("event"); event && (event->string() == "ready" || event->string() == "done")) {
                const auto rg4_routes = gpu.q8_gemv_rg4_routes();
                const auto msg_routes = gpu.q8_three_msg_routes();
                const uint64_t q8_gemv_rg4_total = rg4_routes[0] + rg4_routes[1];
                const uint64_t q8_three_msg_total = msg_routes[0] + msg_routes[1];
                const char* rg4_name = q8_gemv_rg4 == Q8GemvRg4Mode::on ? "on" :
                    q8_gemv_rg4 == Q8GemvRg4Mode::wide ? "wide" : "off";
                const char* msg_name = q8_three_msg == Q8ThreeMsgMode::on ? "on" :
                    q8_three_msg == Q8ThreeMsgMode::wide ? "wide" : "off";
                const char* lookup_name = mtp_lookup_mode == MtpLookupMode::bypass ? "bypass" :
                    mtp_lookup_mode == MtpLookupMode::hybrid ? "hybrid" : "off";
                auto fields = value.object();
                fields.emplace_back("exact_candidate_swiglu_fused", Json(gpu.swiglu_fused_enabled()));
                fields.emplace_back("exact_candidate_swiglu_calls", Json::integer(gpu.swiglu_fused_calls()));
                fields.emplace_back("exact_candidate_q8_gemv_rg4", Json(rg4_name));
                fields.emplace_back("exact_candidate_q8_gemv_rg4_calls", Json::integer(q8_gemv_rg4_total));
                fields.emplace_back("exact_candidate_q8_three_msg", Json(msg_name));
                fields.emplace_back("exact_candidate_q8_three_msg_calls", Json::integer(q8_three_msg_total));
                fields.emplace_back("exact_candidate_moe_three_union", Json(gpu.moe_three_union_enabled()));
                fields.emplace_back("exact_candidate_moe_three_union_calls", Json::integer(gpu.moe_three_union_calls()));
                fields.emplace_back("exact_candidate_mtp_lookup", Json(lookup_name));
                fields.emplace_back("exact_candidate_mtp_lookup_match", Json::integer(mtp_lookup_match));
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                fields.emplace_back("exact_candidate_mtp_lookup_rounds", Json::integer(mtp_lookup_rounds));
                fields.emplace_back("exact_candidate_mtp_lookup_accepted", Json::integer(mtp_lookup_accepted));
#endif
                const char* m64_name = prefill_m64 == PrefillM64Mode::on ? "on" :
                    prefill_m64 == PrefillM64Mode::short_only ? "short" : "off";
                fields.emplace_back("exact_candidate_prefill_m64", Json(m64_name));
                fields.emplace_back("exact_candidate_prefill_m64_calls", Json::integer(gpu.prefill_m64_calls()));
                fields.emplace_back("exact_candidate_prefill_multiseg", Json(gpu.prefill_multiseg_enabled()));
                fields.emplace_back("exact_candidate_prefill_multiseg_calls", Json::integer(gpu.prefill_multiseg_calls()));
                fields.emplace_back("exact_candidate_prefill_swizzle", Json(gpu.prefill_swizzle_enabled()));
                fields.emplace_back("exact_candidate_qsa_prefill_fused", Json(gpu.qsa_prefill_fused_enabled()));
                fields.emplace_back("exact_candidate_hc_consumer_fused", Json(gpu.hc_consumer_fused_enabled()));
                fields.emplace_back("exact_candidate_hc_consumer_fused_calls", Json::integer(gpu.hc_consumer_fused_calls()));
                fields.emplace_back("exact_candidate_ple_overlap", Json(ple_overlap));
                fields.emplace_back("exact_candidate_ple_overlap_chunks", Json::integer(ple_overlap_chunks));
                fields.emplace_back("shared_prefix_cache_enabled", Json(shared_prefix_cache_enabled));
                fields.emplace_back("shared_prefix_cache_anchors", Json::integer(shared_prefix_cache.anchors.size()));
                fields.emplace_back("shared_prefix_cache_bytes", Json::integer(shared_prefix_cache.occupied_bytes));
                fields.emplace_back("shared_prefix_cache_captures", Json::integer(shared_prefix_cache.captures));
                fields.emplace_back("shared_prefix_cache_hits", Json::integer(shared_prefix_cache.hits));
                fields.emplace_back("shared_prefix_cache_restored_tokens", Json::integer(shared_prefix_cache.restored_tokens));
#if defined(TOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS) && TOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS
                fields.emplace_back("shared_prefix_cache_evictions", Json(Json::Object{
                    {"bytes", Json::integer(shared_prefix_cache.evictions.bytes)},
                    {"count", Json::integer(shared_prefix_cache.evictions.count)},
                    {"both", Json::integer(shared_prefix_cache.evictions.both)}}));
#endif
                value = Json(std::move(fields));
            }
        }
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
        if (kernel_search_capture_path) { const auto* event = value.find("event"); if (event && (event->string() == "ready" || event->string() == "done")) {
            auto fields = value.object(); fields.emplace_back("kernel_search_capture_schema", Json::integer(1)); fields.emplace_back("kernel_search_capture_records", Json::integer(kernel_search_sink.records));
            fields.emplace_back("kernel_search_capture_dense_kernel", Json("graph_dense_q8_staged_f32")); fields.emplace_back("kernel_search_capture_hc_rows", Json(true)); fields.emplace_back("kernel_search_capture_router", Json("scalar")); value = Json(std::move(fields));
        } }
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(route_lock_output){const auto* event=value.find("event");if(event&&(event->string()=="ready"||event->string()=="done")){
            auto fields=value.object();fields.emplace_back("route_lock_schema",Json::integer(1));fields.emplace_back("route_lock_mode",Json(route_lock_reference?"replay":"observe"));
            fields.emplace_back("route_census_only",Json(route_census_only));fields.emplace_back("route_lock_records",Json::integer(route_lock_sink.records));fields.emplace_back("route_lock_rows",Json::integer(route_lock_sink.rows));value=Json(std::move(fields));}}
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        const auto* chunk_event = value.find("event");
        if ((known_token_capacity == 512 || known_token_capacity == 1024 || known_token_capacity == 2048) && chunk_event &&
            (chunk_event->string() == "ready" || chunk_event->string() == "done")) {
            auto fields = value.object();
            fields.emplace_back("prefill_chunk_schema", Json::integer(1));
#ifdef TOKSPUR_PRIVATE_GROUPED_ORDERED
            fields.emplace_back("prefill_chunk_numerical_tier", Json("exact-candidate"));
#ifdef TOKSPUR_PRIVATE_GROUPED_PACKED
            fields.emplace_back("prefill_chunk_projection", Json(stack.known_grouped_ffn() ? "packed-gemv" : "ordinary"));
#else
            fields.emplace_back("prefill_chunk_projection", Json(stack.known_grouped_ffn() ? "ordered-gemv" : "ordinary"));
#endif
#else
            fields.emplace_back("prefill_chunk_numerical_tier", Json(stack.known_grouped_ffn() ? "operator" : "exact-candidate"));
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
            fields.emplace_back("native_prefill_chunks", Json(native_prefill_chunks ? (*native_prefill_chunks == GdnChunkMode::fused ? "fused" : "cs64") : "off"));
            fields.emplace_back("native_prefill_chunks_schema", Json::integer(3));
            fields.emplace_back("native_prefill_dense", Json(native_prefill_chunks ? (gpu.prefill_dense_nax() ? "nax" : (gpu.prefill_dense_f32() ? "f32" : "f16")) : "off"));
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            if (gpu.prefill_dense_staged_f32() || gpu.prefill_dense_nax()) fields.emplace_back("native_prefill_dense_kernel", Json(gpu.prefill_dense_kernel()));
#endif
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_DENSE_Q8_STAGED)
            const auto* operator_event = value.find("event");
            if (operator_staged_dense_f32.has_value() && operator_event && (operator_event->string() == "ready" || operator_event->string() == "done")) {
                fields.emplace_back("operator_prefill_dense_schema", Json::integer(1));
                fields.emplace_back("operator_prefill_dense_route", Json(gpu.prefill_dense_staged_f32() ? "staged-f32" : "old-mpp"));
                fields.emplace_back("operator_prefill_dense_kernel", Json(gpu.prefill_dense_kernel()));
                fields.emplace_back("operator_prefill_dense_scope", Json("selected chunked-prefill dense Q8 only; not decode/MTP kernels or a completed dispatch counter"));
            }
#endif
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH)
            const auto* router_event=value.find("event");
            if(operator_router_f32_batch.has_value()&&router_event&&(router_event->string()=="ready"||router_event->string()=="done")){
                fields.emplace_back("operator_prefill_router_schema",Json::integer(1));fields.emplace_back("operator_prefill_router_route",Json(gpu.moe_router_f32_batch_enabled()?"batched-f32":"scalar-f32"));
                fields.emplace_back("operator_prefill_router_kernel",Json(gpu.moe_router_f32_projection_kernel()));fields.emplace_back("operator_prefill_router_scope",Json("selected chunked-prefill raw-F32 router2560-to512 for 2..512 rows only; not ordinary/N3/MTP or a completed dispatch counter"));
            }
#endif
#if defined(TOKSPUR_PRIVATE_HC_DOWN_STAGED) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
            const auto* hc_down_staged_event = value.find("event");
            if (operator_hc_down_staged.has_value() && hc_down_staged_event && (hc_down_staged_event->string() == "ready" || hc_down_staged_event->string() == "done")) {
                for (const auto& field : operator_hc_down_staged_identity(*operator_hc_down_staged)) fields.push_back(field);
                fields.emplace_back("operator_prefill_hc_down_encoded_calls", Json::integer(gpu.prefill_hc_down_f32_encoded_calls()));
            }
#endif
#if defined(TOKSPUR_PRIVATE_HC_ROWS) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
            const auto* hc_rows_event=value.find("event");
            if(operator_hc_rows.has_value()&&hc_rows_event&&(hc_rows_event->string()=="ready"||hc_rows_event->string()=="done")){
                fields.emplace_back("operator_prefill_hc_rows_schema",Json::integer(1));fields.emplace_back("operator_prefill_hc_rows_route",Json(gpu.prefill_hc_rows()?"rows-split16-exact":"legacy-per-token"));
                fields.emplace_back("operator_prefill_hc_rows_scratch_bytes",Json::integer(gpu.prefill_hc_rows()?512*4*16*4:0));
                Json::Array calls;for(auto n:gpu.hc_rows_encoded())calls.push_back(Json::integer(n));fields.emplace_back("operator_prefill_hc_rows_encoded",Json(std::move(calls)));
            }
#endif
#if defined(TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
            const auto* router_window_event=value.find("event");
            if(operator_router_window.has_value()&&router_window_event&&(router_window_event->string()=="ready"||router_window_event->string()=="done")){
                fields.emplace_back("operator_prefill_router_window_schema",Json::integer(1));fields.emplace_back("operator_prefill_router_window_route",Json(gpu.prefill_router_window()?"window16-simd32-exact":"legacy-per-token"));
                fields.emplace_back("operator_prefill_router_window_scratch_bytes",Json::integer(0));
                Json::Array calls;for(auto n:gpu.router_window_encoded())calls.push_back(Json::integer(n));fields.emplace_back("operator_prefill_router_window_encoded",Json(std::move(calls)));
            }
#endif
#if defined(TOKSPUR_PRIVATE_HC_DOWN_WINDOW) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
            const auto* hc_down_window_event=value.find("event");
            if(operator_hc_down_window.has_value()&&hc_down_window_event&&(hc_down_window_event->string()=="ready"||hc_down_window_event->string()=="done")){
                fields.emplace_back("operator_prefill_hc_down_window_schema",Json::integer(1));fields.emplace_back("operator_prefill_hc_down_window_route",Json(gpu.prefill_hc_down_window()?"window16-q8-simd32-exact":"scalar-hc-exact"));
                fields.emplace_back("operator_prefill_hc_down_window_threadgroup_bytes",Json::integer(gpu.prefill_hc_down_window()?32768:0));
                fields.emplace_back("operator_prefill_hc_down_window_encoded",Json::integer(gpu.hc_down_window_encoded()));
            }
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
            const auto* qsa_sparse_event = value.find("event");
            if ((qsa_sparse.has_value()
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
                || operator_qsa_sparse.has_value()
#endif
                ) && qsa_sparse_event && (qsa_sparse_event->string() == "ready" || qsa_sparse_event->string() == "done")) {
                fields.emplace_back("operator_qsa_sparse_schema", Json::integer(1));
                fields.emplace_back("operator_qsa_sparse_route", Json(gpu.qsa_sparse_enabled() ? "online-softmax-single-pass" : "chronological-three-pass"));
                fields.emplace_back("operator_qsa_sparse_decode_bn", Json::integer(TOKSPUR_QSA_SPARSE_DECODE_BN));
                fields.emplace_back("operator_qsa_sparse_verify_bn", Json::integer(TOKSPUR_QSA_SPARSE_VERIFY_BN));
                fields.emplace_back("operator_qsa_sparse_decode_calls", Json::integer(gpu.qsa_sparse_decode_calls()));
                fields.emplace_back("operator_qsa_sparse_verify_calls", Json::integer(gpu.qsa_sparse_verify_calls()));
            }
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
            if (qsa_route_attribution_path && qsa_sparse_event && (qsa_sparse_event->string() == "ready" || qsa_sparse_event->string() == "done")) {
                fields.emplace_back("qsa_route_attribution_records", Json::integer(qsa_route_sink.records));
            }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
            const auto* gdn_ported_event = value.find("event");
            if ((gdn_ported.has_value()
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
                || operator_gdn_ported.has_value()
#endif
                ) && gdn_ported_event && (gdn_ported_event->string() == "ready" || gdn_ported_event->string() == "done")) {
                const auto routes = gpu.gdn_ported_routes();
                fields.emplace_back("operator_gdn_ported_schema", Json::integer(1));
                fields.emplace_back("operator_gdn_ported_route", Json(gpu.gdn_ported_enabled() ? "ported-pipelined-tb12-verify-main" : "legacy-blocked-tb16-delta-tile"));
                fields.emplace_back("operator_gdn_ported_verify_db", Json::integer(TOKSPUR_GDN_VERIFY_DB));
                fields.emplace_back("operator_gdn_ported_prefill_calls", Json::integer(routes[0]));
                fields.emplace_back("operator_gdn_ported_decode_calls", Json::integer(routes[1]));
                fields.emplace_back("operator_gdn_ported_verify_calls", Json::integer(routes[2]));
                fields.emplace_back("operator_gdn_ported_replay_calls", Json::integer(routes[3]));
                fields.emplace_back("operator_gdn_ported_norm_gate_calls", Json::integer(routes[4]));
            }
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
            if (gdn_route_attribution_path && gdn_ported_event && (gdn_ported_event->string() == "ready" || gdn_ported_event->string() == "done")) {
                fields.emplace_back("gdn_route_attribution_records", Json::integer(gdn_route_sink.records));
            }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            fields.emplace_back("native_gdn_cooperative4", Json(gdn_cooperative4_enabled));
            fields.emplace_back("native_verify_gdn_row_batch", Json(verify_gdn_row_batch));
            if (native_verify_nax.has_value()) {
                fields.emplace_back("native_verify_nax", Json(gpu.verify_nax()));
            }
            if (native_verify_nax_m8.has_value()) {
                fields.emplace_back("native_verify_nax_m8", Json(gpu.verify_nax_m8()));
            }
            if (native_prefill_grouped_f32.has_value() || native_prefill_grouped_auto || native_prefill_grouped_nax) {
                fields.emplace_back("native_prefill_grouped", Json(gpu.prefill_grouped_nax() ? "nax" : gpu.prefill_grouped_auto() ? "auto" : gpu.prefill_grouped_f32() ? "f32" : "serial"));
                fields.emplace_back("native_prefill_grouped_kernel", Json(gpu.prefill_grouped_nax() ? "nax_grouped_moe" : gpu.prefill_grouped_auto() ? "graph_grouped_mm" : gpu.prefill_grouped_f32() ? "graph_grouped_mm_f32" : "graph_grouped_serial"));
            }
            if (native_prefill_grouped_auto || router_margin_audit_enabled) {
                const auto tm = gpu.router_margin_telemetry();
                fields.emplace_back("router_margin_min", Json::number(std::isinf(tm.min_margin) ? 0.0 : double(tm.min_margin)));
                fields.emplace_back("router_margin_mean", Json::number(double(tm.mean_margin())));
                fields.emplace_back("router_margin_violations", Json::integer(tm.violation_count));
                fields.emplace_back("router_margin_fallback_layers", Json::integer(tm.fallback_layers));
                fields.emplace_back("router_margin_total_layers", Json::integer(tm.total_layers));
            }
#endif
            fields.emplace_back("native_prefill_single_input", Json(native_prefill_chunks ? "ordinary" : "off"));
            fields.emplace_back("native_prefill_chunks_charged_bytes", Json::integer(stack.chunked_prefill_bytes()));
            fields.emplace_back("native_prefill_chunks_budget_bytes", Json::integer(native_prefill_chunks ? (((known_token_capacity >= 2048 || native_prefill_dense_nax || native_prefill_grouped_nax) ? 1600ull : (known_token_capacity >= 1024 ? 1024ull : 512ull)) << 20) : 0));
#endif
            fields.emplace_back("prefill_chunk_grouped_ffn", Json(stack.known_grouped_ffn()));
            fields.emplace_back("prefill_chunk_assembly_schema", Json::integer(2));
            fields.emplace_back("prefill_chunk_layer_submissions", Json(stack.known_layer_submissions()));
            fields.emplace_back("prefill_chunk_hc_residual_rows", Json(stack.known_hc_residual_rows()));
            fields.emplace_back("prefill_chunk_capacity", Json::integer(known_token_capacity >= 2048 ? 2048 : (known_token_capacity >= 1024 ? 1024 : 512)));
            value = Json(std::move(fields));
        }
#endif
        return value;
    };
    const auto decorate = [&](Json value) {
        value = cache_memory_metadata(std::move(value), gpu);
        value = math_envelope(std::move(value));
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        value = stage_envelope(std::move(value));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (worker_broker8) {
            const auto type = field(value, "event").string();
            if (type == "ready" || type == "control_done" || type == "done") {
                auto fields = value.object(); fields.emplace_back("branch8_session", branch8_snapshot(value)); value = Json(std::move(fields));
            }
        }
#endif
        if (broker) return broker->decorate(std::move(value)); return value;
    };
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    std::vector<float> mtp_priming_hc_rows(mtp && mtp_chunked_prefill ? uint64_t(std::max(known_token_capacity, 512u)) * 10240 : 0);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    std::vector<float> known_rows(uint64_t(known_token_capacity) * 2560), known_embeddings(known_rows.size());
    own_memory_guard();
#endif
    if (!profiling && std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE")) {
        counter_sampling = gpu.configure_profiling(16384);
    }
    const auto decorate_ready = [&](Json value) {
        return decorate(std::move(value));
    };
    std::cout << decorate_ready(Json(Json::Object{{"event", Json("ready")}, {"scope", Json("complete 48-layer text target, experimental checkpoint policies; not quality or performance qualified")},
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
        {"verification_cost_probe", Json(bool(verification_cost_manifest))},
        {"verification_cost_sealed_source_sha256", verification_cost_manifest ? Json(verification_cost_manifest->sealed_source_sha256) : Json()},
#endif
        {"policy", options}, {"weight_setup_ms", number(milliseconds(start))}, {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)},
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        {"prefill_batch_top10_enabled", Json(gpu.prefill_batch_top10())},
        {"prefill_batch_mixture_enabled", Json(gpu.prefill_batch_mixture())},
        {"prefill_top10_kernel", Json(gpu.prefill_batch_top10() ? "graph_route_batch" : "graph_route")},
        {"prefill_mixture_kernel", Json(gpu.prefill_batch_mixture() ? "graph_mixture_batch" : "graph_mixture")},
#endif
        {"peak_metal_buffer_bytes", Json::integer(gpu.memory().peak_bytes)}, {"gpu_allocation_count", Json::integer(gpu.memory().allocations)},
        {"gpu_buffer_limit_bytes", Json::integer(gpu.memory().budget)}, {"live_captures", Json::integer(gpu.memory().live_captures)},
        {"imported_weight_backing_bytes", Json::integer(gpu.memory().imported_backing_bytes)}, {"imported_regions", Json::integer(imported_region_count)}, {"weight_copy", Json(false)},
        {"mapped_weight_region_limit_bytes", Json::integer(mapped_region_limit)},
        {"weights_storage", Json("read-only mapped Metal views; first-use page faults are not hidden from request time")},
        {"gpu_profile_mode", Json(profile_mode(counter_sampling))}, {"profile_counter_payload_bytes", Json::integer(gpu.memory().profile_counter_payload_bytes)},
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        {"task4_profile_selected", Json(task4_profile.has_value())}, {"task4_profile_enabled", Json(task4_profile.value_or(false))},
        {"task4_input_ids", Json::integer(task4_profile.has_value() ? 4097 : 0)}, {"task4_counter_dispatch_capacity", Json::integer(task4_profile.value_or(false) ? 16384 : 0)},
        {"task4_profile_boundary", Json(task4_sink.boundary == Task4ProfileBoundary::dispatch ? "dispatch_boundary" : "encoder_boundary")},
        {"task4_profile_window", Json(task4_sink.boundary == Task4ProfileBoundary::dispatch ? "full_history" : "late8_decode")},
        {"task4_profile_first_prefix", Json::integer(task4_sink.boundary == Task4ProfileBoundary::dispatch ? 0 : 4088)},
#endif
        {"sampling", sampling_json(sampler, stack.position())},
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        {"qsa_stage_v8_prepared", Json(qsa_stage_policy.prepared)}, {"qsa_stage_v8_enabled", Json(qsa_stage_policy.enabled)},
        {"qsa_stage_v8_scratch_logical_bytes", Json::integer(qsa_stage_policy.scratch_logical_bytes)},
        {"qsa_stage_v8_scratch_charged_bytes", Json::integer(qsa_stage_policy.scratch_charged_bytes)},
        {"qsa_stage_v8_calls", Json::integer(gpu.qsa_stage_v8_calls())},
        {"qsa_stage_v8_physical_dispatches", Json::integer(2 * gpu.qsa_stage_v8_calls())},
        {"qsa_stage_v8_extra_dispatches", Json::integer(gpu.qsa_stage_v8_calls())},
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        {"hc_exact_mode", Json(hc_exact_name(hc_exact))}, {"hc_exact_rows_per_group", Json::integer(1)}, {"hc_exact_unroll", Json::integer(16)},
        {"hc_exact_scratch_logical_bytes", Json::integer(0)}, {"hc_exact_scratch_charged_bytes", Json::integer(0)},
        {"hc_exact_routes", hc_routes_json(gpu.hc_exact_routes())},
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        {"wide_q8_exact_enabled", Json(wide_q8_exact)}, {"wide_q8_exact_rows_per_group", Json::integer(4)}, {"wide_q8_exact_unroll", Json::integer(16)},
        {"wide_q8_exact_scratch_logical_bytes", Json::integer(0)}, {"wide_q8_exact_scratch_charged_bytes", Json::integer(0)},
        {"wide_q8_exact_routes", wide_routes_json(gpu.wide_q8_exact_routes())},
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        {"wide_q8_many_enabled", Json(wide_q8_many)}, {"wide_q8_many_rows_per_group", Json::integer(4)}, {"wide_q8_many_unroll", Json::integer(4)},
        {"wide_q8_many_scratch_logical_bytes", Json::integer(0)}, {"wide_q8_many_scratch_charged_bytes", Json::integer(0)},
        {"wide_q8_many_routes", wide_many_routes_json(gpu.wide_q8_many_routes())},
#endif
        {"checkpoint_broker", Json(bool(broker))}, {"broker_epoch", Json::integer(broker ? broker->epoch() : 0)},
        {"prompt_lookup", Json(bool(lookup))}, {"lookup_match_tokens", Json::integer(lookup ? 16 : 0)},
        {"allow_prompt_lookup_experiments", Json(allow_lookup_experiments)},
        {"lookup_history_tokens", Json::integer(lookup ? 65536 : 0)},
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        {"lookup_proposal_tokens", Json::integer(lookup ? lookup->proposal_tokens() : 0)},
        {"verification_max_tokens", Json::integer(
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            mtp_depth == 7 ? 8 : mtp_depth == 3 ? 4 : mtp_depth == 2 ? 3 :
#endif
            verification_max_tokens)},
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        {"verification8_branch_reuse", Json(verification8_branch_reuse)},
#endif
        {"verification_head_many", Json(verification_max_tokens == 8)},
        {"verification_tape_payload_bytes", Json::integer(stack.verification_memory().tape_payload_bytes)},
#else
        {"lookup_proposal_tokens", Json::integer(lookup ? 2 : 0)},
#endif
        {"lookup_verification_budget_bytes", Json::integer(verification_budget)},
        {"lookup_caller_host_bytes", Json::integer(lookup ? lookup->charged_bytes() : 0)},
        {"verification_host_bytes", Json::integer(stack.verification_memory().host_charged_bytes)},
        {"verification_gpu_bytes", Json::integer(stack.verification_memory().gpu_charged_bytes)},
        {"lookup_verification_charged_bytes", Json::integer(lookup ? lookup->charged_bytes() +
            stack.verification_memory().host_charged_bytes + stack.verification_memory().gpu_charged_bytes : 0)},
        {"checkpoint_slots", Json::integer(checkpoint_slots)}, {"checkpoint_combined_budget_bytes", Json::integer(checkpoint_budget)},
        {"checkpoint_host_bytes", Json::integer(branches ? branches->charged_bytes() : 0)},
        {"checkpoint_store_host_bytes", Json::integer(stack.checkpoint_memory().host_metadata_bytes)},
        {"checkpoint_gpu_bytes", Json::integer(stack.checkpoint_memory().gpu_charged_bytes)},
        {"small_gemv_column_splits", Json::integer(small_gemv_splits)},
        {"hc_q8_splits", Json::integer(hc_q8.splits)},
        {"hc_q8_scratch_logical_bytes", Json::integer(hc_q8.scratch_logical_bytes)},
        {"hc_q8_scratch_charged_bytes", Json::integer(hc_q8.scratch_charged_bytes)},
        {"hc_q8_numerical_tier", Json(hc_q8_numerical_tier(hc_q8.splits))},
        {"known_token_capacity", Json::integer(stack.known_token_capacity())},
        {"known_residual_reuse", Json(stack.known_residual_reuse())},
        {"allow_request_prefill_experiments", Json(allow_prefill_experiments)},
        {"stack_execution", Json(execution_name(execution))}, {"recurrent_replay_captures", Json::integer(stack.replay_capture_count())},
        {"replay_prepared", Json(stack.replay_prepared())}, {"allow_request_stack_experiments", Json(allow_stack_experiments)},
        {"capture_resource_bytes", Json::integer(gpu.memory().capture_resource_bytes)},
        {"capture_binding_aliases", Json::integer(gpu.memory().capture_binding_aliases)},
        {"capture_binding_alias_span_bytes", Json::integer(gpu.memory().capture_binding_alias_span_bytes)},
        {"capture_binding_alias_resource_bytes", Json::integer(gpu.memory().capture_binding_alias_resource_bytes)},
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        {"execution_contract", Json(
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            worker_broker8 ? "private branch8-v1 width8 broker; explicit zero-model-step cold/ancestor controls; bounded greedy model tiles; no automatic retry" :
#endif
            lookup ? (verification_max_tokens == 8 ?
            "native CLI only: ordinary steps or explicit greedy capacity-eight verified rounds; at most seven PLD proposals; no worker-broker combination" :
            "native CLI only: ordinary steps or explicit greedy depth-two verified rounds; no worker-broker combination") :
#else
        {"execution_contract", Json(lookup ? "native CLI only: ordinary steps or explicit greedy depth-two verified rounds; no worker-broker combination" :
#endif
            "one serial token batch; replay records recurrent runs and PLE, QSA/final remain dynamic; two immutable physical orientations per run")},
        {"compatible_comparator_limit", Json("indirect-compatible uses compatible PSOs for the whole batch, including dynamic QSA/final; B/C is not pure replay-only attribution")},
        {"block_gemv_formats", Json::integer(block_gemv_formats)}, {"maximum_generated_tokens", Json::integer(4096)},
        {"allow_request_kernel_experiments", Json(allow_kernel_experiments)},
        {"thermal_state", Json::integer(host_thermal_state())},
        {"host_row_cache_charged_bytes", Json::integer(cache.charged_bytes())}, {"native_limit_bytes", Json::integer(native_limit)},
        {"ple_cache", Json(Json::Object{{"hits", Json::integer(cache.stats().hits)}, {"misses", Json::integer(cache.stats().misses)}, {"evictions", Json::integer(cache.stats().evictions)}, {"failures", Json::integer(cache.stats().failures)}})},
        {"stateless_prefill_ple_rows", Json::integer(stateless_prefill_ple_rows)},
        {"memory", describe(memory())}, {"timings_qualified", Json(false)}})).dump() << std::endl;
    std::vector<float> row(2560), embedding(2560); const auto& token_table = files.tensor("token_embd.weight"); const auto row_bytes = token_table.bytes / 248320;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::vector<float> mtp_previous_hc(mtp ? 10240 : 0), mtp_candidate_rows((mtp || dflash2) ? 2560 * (mtp_depth + 1) : 0), mtp_candidate_memory((mtp || dflash2) ? 2560 * (mtp_depth + 1) : 0);
    std::vector<float> mtp_lookup_prime_hc(mtp_lookup ? 10240 * (mtp_depth + 1) : 0);
    std::optional<StackResult> mtp_spare;
    if (mtp || dflash2) { mtp_spare.emplace(); mtp_spare->logits.resize(shape.vocabulary); }
    std::unique_ptr<PleAsyncGatherer> ple_async;
    if ((mtp || dflash2) && native_ple_async_gather.value_or(false)) {
        ple_async = std::make_unique<PleAsyncGatherer>(cache, spec, effective_policy.ngram,
            files.tensor_bytes(table.name), table.type, layout.ple_storage_rows, layout.ple_addressable_rows,
            ple_nocache_file.source());
    }
    const bool mtp_fast_round = mtp && mtp->fast_round();
    const auto flush_mtp_deferred = [&]() {
        if (!mtp || !mtp->deferred_prime_rows()) return;
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        gpu.set_mtp_profile_phase(MtpProfilePhase::prime);
#endif
        const auto started = Clock::now();
        const auto batch = mtp->flush_deferred_prime();
        mtp_prime_gpu_ms += batch.gpu_ms;
        mtp_prime_wall_ms += milliseconds(started);
    };
    const auto mtp_prime = [&](std::span<const float> hc, std::span<const float> input, uint32_t absolute) {
        if (!mtp || !absolute) return;
        flush_mtp_deferred();
        require(mtp->next_absolute_position() == absolute, "native MTP cache lost chronological target alignment");
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        gpu.set_mtp_profile_phase(MtpProfilePhase::prime);
#endif
        const auto started = Clock::now();
        const auto batch = mtp->prime_pair(hc, input, absolute);
        ++mtp_prime_calls; mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(started);
    };
#endif
    const auto step = [&](uint32_t token) {
        own_memory_guard(); const auto ids = stack.lookup(token); cache.gather(ids.rows, embedding);
        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(token) * row_bytes, row_bytes), row);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        const auto base_before = stack.position();
        if (mtp && base_before) {
            const auto residual_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
            if (residual_gpu.owner) {
                flush_mtp_deferred();
                const auto started = Clock::now();
                const auto batch = mtp->prime_pair(residual_gpu, row, base_before);
                ++mtp_prime_calls; mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(started);
            } else {
                mtp_prime(stack.last_final_hc_residual(), row, base_before);
            }
        }
#endif
        auto res = stack.step(token, row, embedding);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (dflash2) {
            const auto cap = stack.last_dflash_capture();
            require(cap.rows.owner && cap.count == 1, "DFlash2 step tap capture mismatch");
            const auto started = Clock::now();
            const auto batch = dflash2->commit_context(cap.rows, 1, base_before);
            ++mtp_prime_calls; mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(started);
        }
#endif
        return res;
    };
    const auto advance = [&](uint32_t token) {
        require(sampler.matches(stack.position()), "invalid sampling prefix before model step; reset required");
        auto result = step(token);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        gpu.capture_carrier_logits(result.logits, 0, 1);
#endif
        // Vector moves and scalar publication cannot fail. If a future stack
        // implementation commits then throws, the catch below invalidates the
        // mismatched retained value instead of sampling from stale logits.
        retained.commit(std::move(result), stack.position());
        require(sampler.committed_input(stack.position()), "unexpected committed model position invalidated entropy");
        if (branches) require(branches->committed(stack.position(), std::span(&token, 1)), "committed token invalidated branch history");
        if (lookup_active) lookup->committed(stack.position(), std::span(&token, 1));
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_lookup && mtp_lookup_valid) mtp_lookup->append(mtp_lookup->epoch(), stack.position() - 1, std::span(&token, 1));
#endif
    };
    // Bounded host-owned gathers finish before encoding; no cache row is
    // borrowed across another lookup. This advances known INPUT only, never a
    // proposed/generated token or a partially accepted speculative batch.
#ifndef TOKSPUR_PRIVATE_HTTP_PREFILL512
    std::vector<float> known_rows(uint64_t(known_token_capacity) * 2560), known_embeddings(known_rows.size());
#endif
    struct ScopedJoinThread {
        std::thread thread;
        ~ScopedJoinThread() { if (thread.joinable()) thread.join(); }
        void join() { if (thread.joinable()) thread.join(); }
    };
    const uint64_t ple_host_cap = ple_overlap && known_token_capacity > 1 ? uint64_t(known_token_capacity) : 0;
    std::vector<float> ple_prefetch_rows(ple_overlap ? known_rows.size() : 0), ple_prefetch_embeddings(ple_overlap ? known_embeddings.size() : 0);
    std::vector<uint64_t> ple_chunk_ids(ple_host_cap * 16), ple_prefetch_ids(ple_host_cap * 16);
    PleGatherScratch ple_gather_scratch, ple_prefetch_scratch;
    const auto advance_known_prepared = [&](std::span<const uint32_t> tokens, uint32_t base, bool emit_logits = true,
                                            [[maybe_unused]] uint64_t remaining_after = 0) {
        auto result = stack.step_known(tokens, std::span(known_rows).first(tokens.size() * 2560),
            std::span(known_embeddings).first(tokens.size() * 2560), emit_logits);
        require(result.logits_available == emit_logits && (emit_logits ? result.logits.size() == 248320 : result.logits.empty()),
            "known-token result logits readiness differs");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp) {
            require(stack.last_final_hc_count() == tokens.size(), "native MTP known HC tap count mismatch");
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            if (mtp_chunked_prefill) {
                const uint32_t first = base ? 0 : 1, count = uint32_t(tokens.size()) - first;
                if (count) {
                    flush_mtp_deferred();
                    const uint32_t batch_cap = mtp->priming_batch_capacity();
                    const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_rows() : GpuView{};
                    if (hc_gpu.owner) {
                        constexpr uint64_t hc_row_bytes = 10240ull * sizeof(float);
                        for (uint32_t offset = 0; offset < count; offset += batch_cap) {
                            const uint32_t chunk_count = std::min(batch_cap, count - offset);
                            const auto chunk_emb = std::span(known_rows).subspan(uint64_t(first + offset) * 2560, uint64_t(chunk_count) * 2560);
                            require(mtp->next_absolute_position() == base + first + offset, "native MTP batch priming frontier differs");
                            const auto started = Clock::now();
                            const auto batch = (base && offset == 0)
                                ? mtp->prime_batch_with_prefix(
                                    mtp_previous_hc,
                                    chunk_count > 1 ? gpu.slice(hc_gpu, 0, uint64_t(chunk_count - 1) * hc_row_bytes) : GpuView{},
                                    chunk_emb, base)
                                : mtp->prime_batch(
                                    gpu.slice(hc_gpu, uint64_t(base ? (offset - 1) : offset) * hc_row_bytes, uint64_t(chunk_count) * hc_row_bytes),
                                    chunk_emb, base + first + offset);
                            mtp_prime_calls += chunk_count; ++mtp_priming_batch_submits;
                            mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(started);
                        }
                    } else {
                        const auto hc_rows = stack.last_final_hc_rows();
                        std::span<const float> prime_hc;
                        if (base) {
                            std::copy(mtp_previous_hc.begin(), mtp_previous_hc.end(), mtp_priming_hc_rows.begin());
                            std::copy_n(hc_rows.begin(), uint64_t(count - 1) * 10240, mtp_priming_hc_rows.begin() + 10240);
                            prime_hc = std::span(mtp_priming_hc_rows).first(uint64_t(count) * 10240);
                        } else prime_hc = hc_rows.first(uint64_t(count) * 10240);
                        for (uint32_t offset = 0; offset < count; offset += batch_cap) {
                            const uint32_t chunk_count = std::min(batch_cap, count - offset);
                            const auto chunk_hc = prime_hc.subspan(uint64_t(offset) * 10240, uint64_t(chunk_count) * 10240);
                            const auto chunk_emb = std::span(known_rows).subspan(uint64_t(first + offset) * 2560, uint64_t(chunk_count) * 2560);
                            require(mtp->next_absolute_position() == base + first + offset, "native MTP batch priming frontier differs");
                            const auto started = Clock::now();
                            const auto batch = mtp->prime_batch(chunk_hc, chunk_emb, base + first + offset);
                            mtp_prime_calls += chunk_count; ++mtp_priming_batch_submits;
                            mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(started);
                        }
                    }
                }
            } else
#endif
            {
                const auto hc_rows = stack.last_final_hc_rows();
                for (uint32_t i = 0; i < tokens.size(); ++i) if (base + i) {
                    const auto hc = i ? hc_rows.subspan(uint64_t(i - 1) * 10240, 10240) : std::span<const float>(mtp_previous_hc);
                    mtp_prime(hc, std::span(known_rows).subspan(uint64_t(i) * 2560, 2560), base + i);
                }
            }
        } else if (dflash2) {
            const auto cap = stack.last_dflash_capture();
            require(cap.rows.owner && cap.count == tokens.size(), "DFlash2 known tap capture mismatch");
            const auto started = Clock::now();
            // Keep every ring row while a shared anchor can still be captured. Later packets only need their final sliding window.
            const auto batch = dflash2->commit_context(cap.rows, uint32_t(tokens.size()), base,
                dflash2_prefill_remaining_after(remaining_after, shared_prefix_cache_enabled,
                    base + tokens.size(), SharedPrefixCacheState::max_prefix_tokens));
            mtp_prime_calls += tokens.size();
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            if (mtp_chunked_prefill) ++mtp_priming_batch_submits;
#endif
            mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(started);
        }
#endif
        retained.commit(std::move(result), stack.position(), emit_logits);
        // No entropy is consumed for input. The same existing one-input
        // transition is acknowledged for each position, only AFTER atomic
        // all-N model publication. Any invariant failure invalidates the session.
        for (uint32_t i = 1; i <= tokens.size(); ++i)
            require(sampler.committed_input(base + i), "known-token publication invalidated entropy prefix");
        if (branches) require(branches->committed(stack.position(), tokens), "known-token publication invalidated branch history");
        if (lookup_active) lookup->committed(stack.position(), tokens);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_lookup && mtp_lookup_valid) mtp_lookup->append(mtp_lookup->epoch(), base, tokens);
#endif
    };
    const auto gather_known_dedup = [&](std::span<const uint32_t> tokens, NgramHistory entry_history,
                                        std::span<uint64_t> ids_buf, PleGatherScratch& scratch_buf,
                                        std::span<float> out_rows, std::span<float> out_embeddings,
                                        bool parallel_ple) -> NgramHistory {
        const auto ple_bytes = files.tensor_bytes(table.name);
        const auto tok_bytes = files.tensor_bytes(token_table.name);
        auto active_ids = ids_buf.first(tokens.size() * 16);
        NgramHistory hist = entry_history;
        for (size_t t = 0; t < tokens.size(); ++t) {
            hist = ngram_step(spec, effective_policy.ngram, hist, tokens[t], active_ids.subspan(t * 16, 16));
        }
        if (parallel_ple) {
            std::exception_ptr worker_error = nullptr;
            ScopedJoinThread ple_worker{std::thread([&]() {
                try {
                    dequantize_rows_sorted_dedup(ple_bytes, table.type, 160, layout.ple_storage_rows,
                        layout.ple_addressable_rows, active_ids, out_embeddings.first(tokens.size() * 2560),
                        scratch_buf, ple_async_gather_enabled ? 16u : 1u, ple_nocache_file.source());
                } catch (...) {
                    worker_error = std::current_exception();
                }
            })};
            for (size_t i = 0; i < tokens.size(); ++i) {
                dequantize(token_table.type, tok_bytes.subspan(uint64_t(tokens[i]) * row_bytes, row_bytes),
                    out_rows.subspan(i * 2560, 2560));
            }
            ple_worker.join();
            if (worker_error) std::rethrow_exception(worker_error);
        } else {
            dequantize_rows_sorted_dedup(ple_bytes, table.type, 160, layout.ple_storage_rows,
                layout.ple_addressable_rows, active_ids, out_embeddings.first(tokens.size() * 2560),
                scratch_buf, ple_async_gather_enabled ? 12u : 1u, ple_nocache_file.source());
            for (size_t i = 0; i < tokens.size(); ++i) {
                dequantize(token_table.type, tok_bytes.subspan(uint64_t(tokens[i]) * row_bytes, row_bytes),
                    out_rows.subspan(i * 2560, 2560));
            }
        }
        return hist;
    };
    const auto advance_known = [&](std::span<const uint32_t> tokens, bool emit_logits = true) {
        own_memory_guard(); const auto base = stack.position();
        require(sampler.matches(base), "invalid sampling prefix before known-token input; reset required");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp && base) std::copy(stack.last_final_hc_residual().begin(), stack.last_final_hc_residual().end(), mtp_previous_hc.begin());
#endif
        if (tokens.size() > 1 && ple_overlap && ple_host_cap >= tokens.size()) {
            const auto entry_history = stack.committed_history();
            static_cast<void>(gather_known_dedup(tokens, entry_history, ple_chunk_ids, ple_gather_scratch,
                known_rows, known_embeddings, true));
            require(entry_history == stack.committed_history(),
                "prefill window history assertion failed: entry_history == stack_committed_history");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (ple_async) cache.seed_from_dedup(ple_gather_scratch, std::span(known_embeddings).first(tokens.size() * 2560));
#endif
            stateless_prefill_ple_rows += tokens.size() * 16;
        } else if (tokens.size() > 1) {
            const auto worker_entry_history = stack.committed_history();
            std::vector<float> alternate_embeddings(tokens.size() * 2560);
            std::exception_ptr worker_error = nullptr;
            const auto ple_bytes = files.tensor_bytes(table.name);
            const auto ple_type = table.type;
            const auto& e = encoding(ple_type);
            const uint64_t ple_row_bytes = uint64_t(160 / e.elements_per_block) * e.bytes_per_block;
            const uint64_t ple_logical = layout.ple_addressable_rows;

            auto worker_task = [&]() {
                try {
                    if (ple_async_gather_enabled) {
                        NgramHistory worker_history = worker_entry_history;
                        PleGatherScratch local_scratch;
                        std::vector<uint64_t> slice_ids;
                        for (size_t off = 0; off < tokens.size(); off += 2048) {
                            const size_t n_slice = std::min<size_t>(2048, tokens.size() - off);
                            slice_ids.resize(n_slice * 16);
                            for (size_t t = 0; t < n_slice; ++t) {
                                worker_history = ngram_step(spec, effective_policy.ngram, worker_history,
                                    tokens[off + t], std::span<uint64_t>(slice_ids).subspan(t * 16, 16));
                            }
                            dequantize_rows_sorted_dedup(ple_bytes, ple_type, 160, layout.ple_storage_rows,
                                layout.ple_addressable_rows, slice_ids,
                                std::span<float>(alternate_embeddings).subspan(off * 2560, n_slice * 2560),
                                local_scratch, 16u, ple_nocache_file.source());
                        }
                    } else {
                        const size_t n_tok = tokens.size();
                        if (n_tok >= 64) {
                            std::vector<std::array<uint64_t, 16>> all_rows(n_tok);
                            NgramHistory worker_history = worker_entry_history;
                            std::vector<uint64_t> step_rows(spec.offsets.size());
                            for (size_t t = 0; t < n_tok; ++t) {
                                worker_history = ngram_step(spec, effective_policy.ngram, worker_history, tokens[t], step_rows);
                                for (size_t h = 0; h < 16; ++h) all_rows[t][h] = step_rows[h];
                            }
                            const size_t num_workers = std::min<size_t>(16, std::max<size_t>(1, n_tok / 16));
                            std::vector<std::exception_ptr> sub_errors(num_workers);
                            std::vector<std::thread> pool;
                            pool.reserve(num_workers - 1);
                            struct PleGatherItem {
                                uint64_t row;
                                uint32_t out_offset;
                            };
                            auto slice_work = [&](size_t wid) {
                                try {
                                    const size_t begin = (n_tok * wid) / num_workers;
                                    const size_t end = (n_tok * (wid + 1)) / num_workers;
                                    if (begin >= end) return;
                                    std::vector<PleGatherItem> items;
                                    items.reserve((end - begin) * 16);
                                    for (size_t t = begin; t < end; ++t) {
                                        for (size_t h = 0; h < 16; ++h) {
                                            items.push_back({all_rows[t][h], uint32_t(t * 2560 + h * 160)});
                                        }
                                    }
                                    std::sort(items.begin(), items.end(), [](const PleGatherItem& a, const PleGatherItem& b) {
                                        return a.row != b.row ? a.row < b.row : a.out_offset < b.out_offset;
                                    });
                                    std::vector<float> scratch(160);
                                    uint64_t prev_row = UINT64_MAX;
                                    for (const auto& item : items) {
                                        if (item.row >= ple_logical) throw FormatError("stateless worker gather addresses out-of-range row");
                                        if (item.row != prev_row) {
                                            dequantize(ple_type, ple_bytes.subspan(item.row * ple_row_bytes, ple_row_bytes), scratch);
                                            prev_row = item.row;
                                        }
                                        std::memcpy(alternate_embeddings.data() + item.out_offset, scratch.data(), 160 * sizeof(float));
                                    }
                                } catch (...) {
                                    sub_errors[wid] = std::current_exception();
                                }
                            };
                            try {
                                for (size_t wid = 1; wid < num_workers; ++wid) pool.emplace_back(slice_work, wid);
                            } catch (...) {
                                for (auto& thread : pool) if (thread.joinable()) thread.join();
                                throw;
                            }
                            slice_work(0);
                            for (auto& th : pool) th.join();
                            for (const auto& err : sub_errors) if (err) std::rethrow_exception(err);
                        } else {
                            NgramHistory worker_history = worker_entry_history;
                            std::vector<uint64_t> step_rows(spec.offsets.size());
                            std::vector<float> scratch(160);
                            for (size_t t = 0; t < n_tok; ++t) {
                                worker_history = ngram_step(spec, effective_policy.ngram, worker_history, tokens[t], step_rows);
                                std::array<std::pair<uint64_t, size_t>, 16> sorted_rows;
                                for (size_t h = 0; h < 16; ++h) sorted_rows[h] = {step_rows[h], h};
                                std::sort(sorted_rows.begin(), sorted_rows.end(), [](const auto& a, const auto& b) {
                                    return a.first < b.first;
                                });
                                float* token_out = alternate_embeddings.data() + t * 2560;
                                for (const auto& [r, head_idx] : sorted_rows) {
                                    if (r >= ple_logical) throw FormatError("stateless worker gather addresses out-of-range row");
                                    dequantize(ple_type, ple_bytes.subspan(r * ple_row_bytes, ple_row_bytes), scratch);
                                    std::memcpy(token_out + head_idx * 160, scratch.data(), 160 * sizeof(float));
                                }
                            }
                        }
                    }
                } catch (...) {
                    worker_error = std::current_exception();
                }
            };

            ScopedJoinThread worker_thread{std::thread(worker_task)};
            for (size_t i = 0; i < tokens.size(); ++i) {
                dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(tokens[i]) * row_bytes, row_bytes),
                    std::span(known_rows).subspan(i * 2560, 2560));
            }
            worker_thread.join();
            if (worker_error) std::rethrow_exception(worker_error);

            require(worker_entry_history == stack.committed_history(),
                "prefill window history assertion failed: worker_entry_history == stack_committed_history");
            std::copy(alternate_embeddings.begin(), alternate_embeddings.end(), known_embeddings.begin());
            stateless_prefill_ple_rows += tokens.size() * 16;
        } else {
            const auto lookups = stack.lookups(tokens);
            for (size_t i = 0; i < tokens.size(); ++i) {
                cache.gather(lookups[i].rows, std::span(known_embeddings).subspan(i * 2560, 2560));
                dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(tokens[i]) * row_bytes, row_bytes),
                    std::span(known_rows).subspan(i * 2560, 2560));
            }
        }
        advance_known_prepared(tokens, base, emit_logits);
    };
    const auto emit_error = [&](const std::exception& error, bool fatal) {
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(route_census_only)route_census_failed=true;
#endif
        const auto position = stack.position();
        shared_prefix_cache.active_valid = false;
        shared_prefix_cache.active_tokens.clear();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        // A head may have advanced before a target refusal. No implicit retry
        // can re-use that auxiliary state or an uncertain entropy frontier.
        if (mtp || dflash2) {
            if (mtp_request_carry && mtp_generation.clear()) ++mtp_discarded_anchors;
            if (mtp_lookup && (fatal || !mtp_lookup_valid || mtp_lookup->committed_length() != position)) mtp_lookup_valid = false;
            sampler.invalidate(); retained.clear();
        }
#endif
        if (broker) broker->revoke();
        if (lookup_active && (fatal || !lookup->matches(position))) { lookup->invalidate(); retained.clear(); sampler.invalidate(); }
        if (branches && !branches->matches(position)) { branches->invalidate(); retained.clear(); sampler.invalidate(); }
        reconcile_failure(retained, sampler, position, fatal);
        if (branches && (fatal || !sampler.matches(position) || (position && !retained.matches(position)))) branches->invalidate();
        std::cout << decorate(Json(Json::Object{{"event", Json("error")}, {"message", Json(error.what())}, {"committed_prefix", Json::integer(position)}, {"automatic_retry_safe", Json(false)},
            {"status", Json("error")}, {"finish_reason", Json("error")}, {"fatal", Json(fatal)}, {"sampling", sampling_json(sampler, position)},
            {"stack_execution", Json(execution_name(stack.execution()))}, {"replay_prepared", Json(stack.replay_prepared())},
            {"logits_available", Json(retained.matches(position))}, {"continuation_allowed", Json(retained.continuable(position) && sampler.matches(position))}})).dump() << std::endl;
    };
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
#include "verification_cost_probe.inc"
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
#include "sampled_verification_cost_probe.inc"
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    const auto compute_ssd_state_digest = [&](const std::array<uint8_t, 32>& exec_digest,
                                              uint32_t pfx, uint64_t stack_bytes,
                                              uint64_t mtp_bytes, bool has_logits) -> std::array<uint8_t, 32> {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        CC_SHA256_Update(&ctx, exec_digest.data(), sizeof(exec_digest));
        CC_SHA256_Update(&ctx, &pfx, sizeof(pfx));
        CC_SHA256_Update(&ctx, &stack_bytes, sizeof(stack_bytes));
        CC_SHA256_Update(&ctx, &mtp_bytes, sizeof(mtp_bytes));
        const uint8_t hl = has_logits ? 1u : 0u;
        CC_SHA256_Update(&ctx, &hl, sizeof(hl));
        std::array<uint8_t, 32> digest{};
        CC_SHA256_Final(digest.data(), &ctx);
        return digest;
    };
    const auto ssd_filename_for_digest = [&](std::span<const uint32_t> prefix_tokens, bool has_logits,
                                             const std::array<uint8_t, 32>& exec_digest) -> std::string {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        const uint32_t fmt_ver = checkpoint_format_version;
        CC_SHA256_Update(&ctx, &fmt_ver, sizeof(fmt_ver));
        CC_SHA256_Update(&ctx, exec_digest.data(), sizeof(exec_digest));
        const uint32_t pfx = uint32_t(prefix_tokens.size());
        CC_SHA256_Update(&ctx, &pfx, sizeof(pfx));
        if (!prefix_tokens.empty()) {
            CC_SHA256_Update(&ctx, prefix_tokens.data(), CC_LONG(prefix_tokens.size_bytes()));
        }
        const uint8_t hl = has_logits ? 1 : 0;
        CC_SHA256_Update(&ctx, &hl, sizeof(hl));
        CC_SHA256_Update(&ctx, legacy_checkpoint_identity_tag.data(), sizeof(legacy_checkpoint_identity_tag));
        std::array<uint8_t, 32> digest{};
        CC_SHA256_Final(digest.data(), &ctx);
        constexpr char hex[] = "0123456789abcdef";
        std::string name = "pfx_";
        name.reserve(41);
        for (size_t i = 0; i < 16; ++i) {
            name.push_back(hex[digest[i] >> 4]);
            name.push_back(hex[digest[i] & 0x0f]);
        }
        name += ".ckpt";
        return name;
    };
    const auto ssd_filename_for = [&](std::span<const uint32_t> prefix_tokens, bool has_logits) -> std::string {
        const auto exec_digest =
            ssd_execution_digest_for(uint32_t(stack.execution()), stack.known_residual_reuse(), gpu.kernel_policy());
        return ssd_filename_for_digest(prefix_tokens, has_logits, exec_digest);
    };
    const auto spill_to_ssd = [&](std::span<const uint32_t> prefix_tokens, bool is_system_anchor) -> bool {
        if (!ssd_manager || !ssd_manager->enabled()) return false;
        const uint32_t pfx = stack.position();
        if (!pfx || prefix_tokens.size() != pfx) return false;
        if (dflash2 && dflash2->context_len() != pfx) return false;
        const auto exec_digest =
            ssd_execution_digest_for(uint32_t(stack.execution()), stack.known_residual_reuse(), gpu.kernel_policy());
        ssd_manager->set_execution_identity(exec_digest);
        const bool has_logits = retained.matches(pfx);
        const auto fname = ssd_filename_for_digest(prefix_tokens, has_logits, exec_digest);
        const uint64_t stack_bytes = stack.ssd_checkpoint_payload_bytes(pfx);
        const uint64_t mtp_bytes = checkpoint_owner ? checkpoint_owner->ssd_cache_bytes() : 0;
        CheckpointFileHeader hdr{};
        hdr.prefix_tokens = pfx;
        hdr.total_payload_bytes = checked_add(stack_bytes, mtp_bytes);
        hdr.is_system_anchor = is_system_anchor ? 1 : 0;
        hdr.state_digest = compute_ssd_state_digest(exec_digest, pfx, stack_bytes, mtp_bytes, has_logits);
        CheckpointMetadata meta{};
        meta.rng = sampler.random();
        meta.controller_state = has_speculative ? mtp_generation.controller_state() : StageAControllerState{};
        meta.mtp_first = checkpoint_owner ? checkpoint_owner->first_absolute_position() : 1;
        meta.mtp_rows = checkpoint_owner ? checkpoint_owner->cache_rows() : 0;
        if (has_speculative) {
            if (const auto* ap = mtp_generation.carry().peek(pfx)) meta.carried_anchor = *ap;
            meta.pending_overflow = mtp_generation.pending_overflow();
        }
        if (has_logits) meta.logits = retained.get(pfx).logits;
        const bool ok = ssd_manager->spill_checkpoint(
            fname, ssd_handshake, hdr, meta,
            [&](size_t chunk_idx, std::span<std::byte> target) -> bool {
                const uint64_t chunk_off = uint64_t(chunk_idx) * checkpoint_chunk_bytes;
                if (chunk_off < stack_bytes) {
                    const uint64_t n = std::min<uint64_t>(target.size(), stack_bytes - chunk_off);
                    if (!stack.read_ssd_checkpoint_span(chunk_off, target.first(size_t(n)))) return false;
                    if (n < target.size()) {
                        if (!checkpoint_owner || !checkpoint_owner->read_ssd_cache_span(0, target.subspan(size_t(n)))) return false;
                    }
                    return true;
                }
                return checkpoint_owner ? checkpoint_owner->read_ssd_cache_span(chunk_off - stack_bytes, target) : target.empty();
            });
        if (ok) {
            ++ssd_spills;
            last_ssd_spill_prefix = pfx;
            last_ssd_spill_time = Clock::now();
        }
        return ok;
    };
    const auto restore_from_ssd_with_digest = [&](std::span<const uint32_t> prefix_tokens, bool expect_logits,
                                                  const TargetSampling& sampling_opts,
                                                  const std::array<uint8_t, 32>& exec_digest) -> bool {
        if (!ssd_manager || !ssd_manager->enabled() || prefix_tokens.empty()) return false;
        const uint32_t target_pfx = uint32_t(prefix_tokens.size());
        if (target_pfx > shape.attention.capacity) return false;
        ssd_manager->set_execution_identity(exec_digest);
        const auto fname = ssd_filename_for_digest(prefix_tokens, expect_logits, exec_digest);
        if (!ssd_manager->has_checkpoint(fname)) return false;
        const uint64_t stack_bytes = stack.ssd_checkpoint_payload_bytes(target_pfx);

        const uint32_t prev_pfx = stack.position();
        const uint32_t prev_mtp_first = checkpoint_owner ? checkpoint_owner->first_absolute_position() : 1u;
        const uint32_t prev_mtp_rows = checkpoint_owner ? checkpoint_owner->cache_rows() : 0u;
        std::vector<std::byte> prev_stack_snapshot;
        std::vector<std::byte> prev_mtp_snapshot;
        bool wrote_any_chunk = false;

        CheckpointFileHeader out_hdr{};
        CheckpointMetadata out_meta{};
        const auto rollback = [&]() {
            if (!wrote_any_chunk) return;
            bool restored = true;
            try {
                if (!prev_pfx) stack.reset();
                else restored = stack.write_ssd_checkpoint_span(prev_pfx, 0, prev_stack_snapshot);
            } catch (...) { restored = false; }
            if (checkpoint_owner) {
                try {
                    if (!prev_mtp_rows) checkpoint_owner->reset(prev_mtp_first);
                    else if (!checkpoint_owner->write_ssd_cache_span(prev_mtp_first, prev_mtp_rows, 0, prev_mtp_snapshot)) restored = false;
                } catch (...) { restored = false; }
            }
            if (!restored) throw SharedPrefixRollbackError();
        };
        bool ok = false;
        try {
            ok = ssd_manager->restore_checkpoint(
                fname, out_hdr, out_meta,
                [&](size_t chunk_idx, std::span<const std::byte> source) -> bool {
                    if (!wrote_any_chunk) {
                        if (prev_pfx > 0) {
                            prev_stack_snapshot.resize(size_t(stack.ssd_checkpoint_payload_bytes(prev_pfx)));
                            if (!stack.read_ssd_checkpoint_span(0, prev_stack_snapshot)) return false;
                        }
                        if (checkpoint_owner && prev_mtp_rows > 0) {
                            prev_mtp_snapshot.resize(size_t(checkpoint_owner->ssd_cache_bytes(prev_mtp_rows)));
                            if (!checkpoint_owner->read_ssd_cache_span(0, prev_mtp_snapshot)) return false;
                        }
                        wrote_any_chunk = true;
                    }
                    const uint64_t chunk_off = uint64_t(chunk_idx) * checkpoint_chunk_bytes;
                    if (chunk_off < stack_bytes) {
                        const uint64_t n = std::min<uint64_t>(source.size(), stack_bytes - chunk_off);
                        if (!stack.write_ssd_checkpoint_span(target_pfx, chunk_off, source.first(size_t(n)))) return false;
                        if (n < source.size()) {
                            if (!checkpoint_owner || !checkpoint_owner->write_ssd_cache_span(out_meta.mtp_first, out_meta.mtp_rows, 0, source.subspan(size_t(n)))) return false;
                        }
                        return true;
                    }
                    return checkpoint_owner ? checkpoint_owner->write_ssd_cache_span(out_meta.mtp_first, out_meta.mtp_rows, chunk_off - stack_bytes, source) : source.empty();
                },
                [&](const CheckpointFileHeader& hdr, const CheckpointMetadata& meta) -> bool {
                    if (hdr.prefix_tokens != target_pfx) return false;
                    if (hdr.model_execution_digest != exec_digest) return false;
                    uint64_t expected_mtp = 0;
                    if (checkpoint_owner) {
                        if (!checkpoint_owner->matches(target_pfx, meta.mtp_first, meta.mtp_rows)) return false;
                        expected_mtp = checkpoint_owner->ssd_cache_bytes(meta.mtp_rows);
                    } else if (meta.mtp_rows != 0) {
                        return false;
                    }
                    if (hdr.total_payload_bytes != stack_bytes + expected_mtp) return false;
                    if (hdr.state_digest != compute_ssd_state_digest(exec_digest, target_pfx, stack_bytes, expected_mtp, expect_logits)) {
                        return false;
                    }
                    if (expect_logits) {
                        if (meta.logits.size() != shape.vocabulary) return false;
                        for (float v : meta.logits) {
                            if (!std::isfinite(v)) return false;
                        }
                    }
                    return true;
                });
            if (ok && checkpoint_owner && out_meta.mtp_rows == 0) {
                ok = checkpoint_owner->write_ssd_cache_span(out_meta.mtp_first, 0, 0, {});
            }
        } catch (...) {
            const auto failure = std::current_exception();
            rollback();
            std::rethrow_exception(failure);
        }
        if (!ok) { rollback(); return false; }
        StackResult restored_res{};
        restored_res.position = target_pfx;
        restored_res.logits_available = expect_logits;
        if (expect_logits) restored_res.logits = std::move(out_meta.logits);
        retained.commit(std::move(restored_res), target_pfx, expect_logits);
        if (sampling_opts.policy.temperature > 0 && sampling_opts.seed == out_meta.rng.seed()) {
            sampler = TargetSampler::restore_branch(sampling_opts, target_pfx, out_meta.rng);
        } else {
            TargetSampler fresh(sampling_opts);
            for (uint32_t p = 1; p <= target_pfx; ++p) {
                require(fresh.committed_input(p), "SSD checkpoint sampler prefix restore failed");
            }
            sampler = std::move(fresh);
        }
        if (has_speculative) {
            mtp_generation.controller_state() = restore_controller_policy(out_meta.controller_state, mtp_generation.controller_policy(), mtp_generation.controller_policy() != default_controller_policy_digest);
        }
        ssd_token_history.assign(prefix_tokens.begin(), prefix_tokens.end());
        if (mtp_lookup) {
            static_cast<void>(mtp_lookup->reset());
            mtp_lookup->append(mtp_lookup->epoch(), 0, prefix_tokens);
            mtp_lookup_valid = true;
        }
        last_ssd_spill_prefix = target_pfx;
        ++ssd_restores;
        ssd_restored_tokens += target_pfx;
        return true;
    };
    const auto restore_from_ssd = [&](std::span<const uint32_t> prefix_tokens, bool expect_logits, const TargetSampling& sampling_opts) -> bool {
        const auto exec_digest =
            ssd_execution_digest_for(uint32_t(stack.execution()), stack.known_residual_reuse(), gpu.kernel_policy());
        return restore_from_ssd_with_digest(prefix_tokens, expect_logits, sampling_opts, exec_digest);
    };
#endif
    const auto current_chunk_mode_byte = [&]() -> uint8_t {
        uint8_t cm = 0xff;
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        if (native_prefill_chunks) cm = uint8_t(*native_prefill_chunks);
#endif
        return cm;
    };
    const auto has_active_mtp = [&]() -> bool {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        return bool(mtp) || bool(dflash2);
#else
        return false;
#endif
    };
    const auto commit_recent_shared_prefix = [&]() {
        if (!shared_prefix_cache_enabled || !shared_prefix_cache.active_valid ||
            shared_prefix_cache.active_tokens.size() < SharedPrefixCacheState::boundary_tokens) {
            return;
        }
        const size_t capped = std::min<size_t>(shared_prefix_cache.active_tokens.size(), SharedPrefixCacheState::max_prefix_tokens);
        const auto cand = std::span<const uint32_t>(shared_prefix_cache.active_tokens).first(capped);
        for (size_t idx = 0; idx < shared_prefix_cache.recent_prefixes.size(); ++idx) {
            auto& prev = shared_prefix_cache.recent_prefixes[idx];
            const size_t min_len = std::min(prev.size(), cand.size());
            if (min_len >= SharedPrefixCacheState::boundary_tokens &&
                std::equal(prev.begin(), prev.begin() + min_len, cand.begin())) {
                std::vector<uint32_t> updated;
                if (cand.size() >= prev.size()) updated.assign(cand.begin(), cand.end());
                else updated = std::move(prev);
                shared_prefix_cache.recent_prefixes.erase(shared_prefix_cache.recent_prefixes.begin() + ptrdiff_t(idx));
                shared_prefix_cache.recent_prefixes.push_back(std::move(updated));
                return;
            }
        }
        if (shared_prefix_cache.recent_prefixes.size() >= SharedPrefixCacheState::max_recent_sessions) {
            shared_prefix_cache.recent_prefixes.erase(shared_prefix_cache.recent_prefixes.begin());
        }
        shared_prefix_cache.recent_prefixes.emplace_back(cand.begin(), cand.end());
    };
    const auto shared_prefix_lcp_boundary = [&](std::span<const uint32_t> cand) -> size_t {
        if (!shared_prefix_cache_enabled || cand.size() < SharedPrefixCacheState::boundary_tokens) return 0;
        const size_t limit = std::min<size_t>(cand.size(), SharedPrefixCacheState::max_prefix_tokens);
        size_t best = 0;
        for (const auto& prev : shared_prefix_cache.recent_prefixes) {
            const size_t bound = std::min(limit, prev.size());
            size_t lcp = 0;
            while (lcp < bound && prev[lcp] == cand[lcp]) ++lcp;
            best = std::max(best, lcp);
        }
        return (best / SharedPrefixCacheState::boundary_tokens) * SharedPrefixCacheState::boundary_tokens;
    };
    const auto capture_shared_prefix = [&](std::span<const uint32_t> prefix_tokens) -> bool {
        if (!shared_prefix_cache_enabled) return false;
        const uint32_t pfx = stack.position();
        if (!pfx || pfx > SharedPrefixCacheState::max_prefix_tokens ||
            pfx % SharedPrefixCacheState::boundary_tokens != 0 || prefix_tokens.size() != pfx) {
            return false;
        }
        const auto kp = gpu.kernel_policy();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (dflash2 && dflash2->context_len() != pfx) return false;
#endif
        const uint8_t cm = current_chunk_mode_byte();
        const bool with_mtp = has_active_mtp();
        const bool has_logits = retained.matches(pfx);
        for (auto& existing : shared_prefix_cache.anchors) {
            if (existing.tokens.size() == pfx && existing.kernel_policy == kp && existing.chunk_mode == cm &&
                existing.has_mtp == with_mtp && std::equal(existing.tokens.begin(), existing.tokens.end(), prefix_tokens.begin())) {
                existing.last_access_tick = ++shared_prefix_cache.tick;
                if (has_logits && !existing.has_logits) {
                    const uint64_t extra = uint64_t(shape.vocabulary) * sizeof(float);
                    if (shared_prefix_cache.occupied_bytes + extra <= SharedPrefixCacheState::max_ram_bytes) {
                        existing.logits = retained.get(pfx).logits;
                        existing.has_logits = true;
                        shared_prefix_cache.occupied_bytes += extra;
                    }
                }
                return false;
            }
        }
        const uint64_t stack_bytes = stack.ssd_checkpoint_payload_bytes(pfx);
        uint64_t mtp_bytes = 0;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (checkpoint_owner) mtp_bytes = checkpoint_owner->ssd_cache_bytes();
#endif
        const uint64_t total_payload = checked_add(stack_bytes, mtp_bytes);
        const uint64_t needed_ram = total_payload + (has_logits ? uint64_t(shape.vocabulary) * sizeof(float) : 0) + uint64_t(pfx) * sizeof(uint32_t);
        if (needed_ram > SharedPrefixCacheState::max_ram_bytes) return false;
        while ((shared_prefix_cache.occupied_bytes + needed_ram > SharedPrefixCacheState::max_ram_bytes ||
                shared_prefix_cache.anchors.size() >= SharedPrefixCacheState::max_anchors) &&
               !shared_prefix_cache.anchors.empty()) {
#if defined(TOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS) && TOKSPUR_PRIVATE_SHARED_PREFIX_EVICTION_COUNTERS
            shared_prefix_cache.evictions.record(shared_prefix_eviction_cause(
                shared_prefix_cache.occupied_bytes + needed_ram > SharedPrefixCacheState::max_ram_bytes,
                shared_prefix_cache.anchors.size() >= SharedPrefixCacheState::max_anchors));
#endif
            const size_t victim = shared_prefix_victim(SharedPrefixEvictionPolicy::lru, shared_prefix_cache.anchors,
                std::span<const uint32_t>(prefix_tokens.data(), prefix_tokens.size()),
                [](const SharedPrefixAnchor& item) -> const std::vector<uint32_t>& { return item.tokens; },
                [](const SharedPrefixAnchor& item) { return item.last_access_tick; });
            shared_prefix_cache.occupied_bytes -= shared_prefix_cache.anchors[victim].ram_bytes();
            shared_prefix_cache.anchors.erase(shared_prefix_cache.anchors.begin() + ptrdiff_t(victim));
        }
        SharedPrefixAnchor anchor{};
        anchor.tokens.assign(prefix_tokens.begin(), prefix_tokens.end());
        anchor.kernel_policy = kp;
        anchor.chunk_mode = cm;
        anchor.has_mtp = with_mtp;
        anchor.has_logits = has_logits;
        anchor.stack_bytes = stack_bytes;
        anchor.mtp_bytes = mtp_bytes;
        anchor.rng = sampler.random();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        anchor.mtp_first = checkpoint_owner ? checkpoint_owner->first_absolute_position() : 1;
        anchor.mtp_rows = checkpoint_owner ? checkpoint_owner->cache_rows() : 0;
        anchor.controller_state = has_speculative ? mtp_generation.controller_state() : StageAControllerState{};
#endif
        if (has_logits) anchor.logits = retained.get(pfx).logits;
        anchor.payload.resize(size_t(total_payload));
        if (!stack.read_ssd_checkpoint_span(0, std::span<std::byte>(anchor.payload).first(size_t(stack_bytes)))) return false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_bytes > 0 && (!checkpoint_owner || !checkpoint_owner->read_ssd_cache_span(0, std::span<std::byte>(anchor.payload).subspan(size_t(stack_bytes), size_t(mtp_bytes))))) {
            return false;
        }
#endif
        anchor.last_access_tick = ++shared_prefix_cache.tick;
        shared_prefix_cache.occupied_bytes += anchor.ram_bytes();
        shared_prefix_cache.anchors.push_back(std::move(anchor));
        ++shared_prefix_cache.captures;
        return true;
    };
    const auto restore_shared_prefix_anchor = [&](SharedPrefixAnchor& anchor, bool expect_logits, const TargetSampling& sampling_opts) -> bool {
        if (!shared_prefix_cache_enabled || anchor.tokens.empty()) return false;
        if (anchor.kernel_policy != gpu.kernel_policy() || anchor.chunk_mode != current_chunk_mode_byte() || anchor.has_mtp != has_active_mtp()) {
            return false;
        }
        if (expect_logits && (!anchor.has_logits || anchor.logits.size() != shape.vocabulary)) return false;
        const uint32_t target_pfx = uint32_t(anchor.tokens.size());
        if (stack.ssd_checkpoint_payload_bytes(target_pfx) != anchor.stack_bytes) return false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (checkpoint_owner && !checkpoint_owner->matches(target_pfx, anchor.mtp_first, anchor.mtp_rows)) return false;
        const uint64_t expected_mtp = checkpoint_owner ? checkpoint_owner->ssd_cache_bytes(anchor.mtp_rows) : 0;
        if (expected_mtp != anchor.mtp_bytes) return false;
#else
        if (anchor.mtp_bytes != 0) return false;
#endif
        if (anchor.payload.size() != anchor.stack_bytes + anchor.mtp_bytes) return false;
        const auto stack_payload = std::span<const std::byte>(anchor.payload).first(size_t(anchor.stack_bytes));
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (!restore_shared_prefix_checkpoint(stack, target_pfx, stack_payload, checkpoint_owner, anchor.mtp_first, anchor.mtp_rows,
            std::span<const std::byte>(anchor.payload).subspan(size_t(anchor.stack_bytes), size_t(anchor.mtp_bytes)))) return false;
#else
        if (!restore_shared_prefix_checkpoint(stack, target_pfx, stack_payload)) return false;
#endif
        StackResult restored_res{};
        restored_res.position = target_pfx;
        restored_res.logits_available = expect_logits;
        if (expect_logits) restored_res.logits = anchor.logits;
        retained.commit(std::move(restored_res), target_pfx, expect_logits);
        if (sampling_opts.policy.temperature > 0 && sampling_opts.seed == anchor.rng.seed()) {
            sampler = TargetSampler::restore_branch(sampling_opts, target_pfx, anchor.rng);
        } else {
            TargetSampler fresh(sampling_opts);
            for (uint32_t p = 1; p <= target_pfx; ++p) {
                require(fresh.committed_input(p), "shared prefix cache sampler prefix restore failed");
            }
            sampler = std::move(fresh);
        }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (has_speculative) {
            mtp_generation.controller_state() = restore_controller_policy(anchor.controller_state, mtp_generation.controller_policy(), mtp_generation.controller_policy() != default_controller_policy_digest);
        }
        if (ssd_manager && ssd_manager->enabled()) {
            ssd_token_history.assign(anchor.tokens.begin(), anchor.tokens.end());
            last_ssd_spill_prefix = target_pfx;
        }
        if (mtp_lookup) {
            static_cast<void>(mtp_lookup->reset());
            mtp_lookup->append(mtp_lookup->epoch(), 0, anchor.tokens);
            mtp_lookup_valid = true;
        }
#endif
        anchor.last_access_tick = ++shared_prefix_cache.tick;
        return true;
    };
    [[maybe_unused]] bool session_generated_since_reset = false;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    bool carrier_component_needs_reset = false, carrier_preparation_sealed = false;
    std::vector<float> carrier_component_rows(carrier_forced_input ? 8 * 2560 : 0), carrier_component_memory(carrier_forced_input ? 8 * 2560 : 0);
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    double last_prefill_wall_ms = 0.0;
    uint64_t last_prefill_dispatches = 0;
    PrefillTop10MixtureTelemetry last_prefill_top10_mixture_stats{};
#endif
    std::string line;
    while (true) {
        try {
            if (!read_line(std::cin, line)) break;
            const auto position = stack.position();
            const auto wire = parse_json(line, {1024 * 1024, 300000, 8});
            std::optional<BrokerCommand> ticket;
            if (broker) ticket = broker->admit(wire, position);
            const auto& body = ticket ? ticket->body : wire;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            if (const auto* inspect = body.find("carrier_warm_state")) {
                const auto operation = inspect->string();
                const bool gpu_readback = operation == "inspect-gpu";
                const bool prime_weights = operation == "inspect-ready";
                const bool reclaim_probe =
#if defined(TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM) && TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM
                    operation == "reclaim-recover" || operation == "reclaim-refuse";
#else
                    false;
#endif
                require(body.object().size() == 1 && (reclaim_probe || operation == "capture" || operation == "inspect" || gpu_readback || prime_weights) &&
                    (reclaim_probe ? carrier_mode == CarrierDiagnosticMode::off : carrier_mode != CarrierDiagnosticMode::off && !carrier_identity_path.empty()) &&
                    !broker && checkpoint_owner && position && retained.matches(position) &&
                    sampler.matches(position) && !mtp_generation.active() && !mtp_generation.pending_overflow() &&
                    !mtp_generation.carry().peek(position), "warm inspection requires an idle complete diagnostic prefix");
                files.assert_unchanged(); own_memory_guard();
                // Deferred prompt work is setup, outside every decoded-action clock.
                flush_mtp_deferred(); if (dflash2) static_cast<void>(dflash2->flush_context());
                if (gpu_readback) gpu.begin_carrier_gpu_readback();
                struct EndStateReadback {
                    MetalGraph& graph; bool enabled;
                    ~EndStateReadback() { if (enabled) graph.end_carrier_gpu_readback(); }
                } end_state_readback{gpu, gpu_readback};
                const auto hex = [](const auto& digest) {
                    std::string out; constexpr char digits[] = "0123456789abcdef";
                    for (uint8_t byte : digest) { out += digits[byte >> 4]; out += digits[byte & 15]; }
                    return out;
                };
                std::vector<std::byte> slab(checkpoint_chunk_bytes);
                const auto hash_payload = [&](uint64_t size, const auto& read) {
                    CC_SHA256_CTX ctx; CC_SHA256_Init(&ctx);
                    for (uint64_t offset = 0; offset < size; offset += slab.size()) {
                        auto part = std::span(slab).first(size_t(std::min<uint64_t>(slab.size(), size - offset)));
                        require(read(offset, part), "warm inspection could not read complete state");
                        CC_SHA256_Update(&ctx, part.data(), CC_LONG(part.size()));
                    }
                    std::array<uint8_t, 32> digest{}; CC_SHA256_Final(digest.data(), &ctx); return hex(digest);
                };
                const uint64_t stack_bytes = stack.ssd_checkpoint_payload_bytes();
                const uint64_t draft_bytes = checkpoint_owner->ssd_cache_bytes();
                const auto stack_hash = hash_payload(stack_bytes, [&](uint64_t offset, auto part) { return stack.read_ssd_checkpoint_span(offset, part); });
                const auto draft_hash = hash_payload(draft_bytes, [&](uint64_t offset, auto part) { return checkpoint_owner->read_ssd_cache_span(offset, part); });
#if defined(TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM) && TOKSPUR_PRIVATE_CACHE_PRESSURE_RECLAIM
                if (reclaim_probe) {
                    const auto before = memory();
                    const auto rng = sampler.random();
                    const auto sampling = sampling_json(sampler, position).dump();
                    const auto controller = mtp_generation.controller_state();
                    const auto logits_hash = hc_exact_logits_hash(retained.get(position).logits);
                    const auto draft_first = checkpoint_owner->first_absolute_position(), draft_rows = checkpoint_owner->cache_rows();
                    const auto row_cache_bytes = cache.charged_bytes();
                    const auto anchor_count = shared_prefix_cache.anchors.size();
                    const auto attempts = cache_reclamations, released = cache_reclaimed_bytes, allocator = cache_allocator_released_bytes;
                    require(row_cache_bytes > 0, "reclamation diagnostic requires a populated row cache");
                    unsigned readings = 0; Memory after{};
                    const bool persistent = operation == "reclaim-refuse";
                    const bool admitted = reclaim_before_memory_refusal(native_limit, [&]() {
                        after = memory(); ++readings;
                        return MemoryGuardReading{after.current, readings == 1 || persistent ? 2u : after.pressure};
                    }, reclaim_own_caches);
                    require(readings == 2 && cache_reclamations == attempts + 1 && cache_reclaimed_bytes > released &&
                        cache.charged_bytes() < row_cache_bytes && shared_prefix_cache.anchors.empty() && shared_prefix_cache.recent_prefixes.empty() &&
                        shared_prefix_cache.active_tokens.empty() && !shared_prefix_cache.active_valid,
                        "reclamation diagnostic did not release the host caches exactly once");
                    require(stack.position() == position && stack.ssd_checkpoint_payload_bytes() == stack_bytes &&
                        checkpoint_owner->ssd_cache_bytes() == draft_bytes && checkpoint_owner->first_absolute_position() == draft_first && checkpoint_owner->cache_rows() == draft_rows &&
                        hash_payload(stack_bytes, [&](uint64_t offset, auto part) { return stack.read_ssd_checkpoint_span(offset, part); }) == stack_hash &&
                        hash_payload(draft_bytes, [&](uint64_t offset, auto part) { return checkpoint_owner->read_ssd_cache_span(offset, part); }) == draft_hash &&
                        retained.matches(position) && hc_exact_logits_hash(retained.get(position).logits) == logits_hash &&
                        sampler.random().seed() == rng.seed() && sampler.random().words_consumed() == rng.words_consumed() && sampler.random().engine() == rng.engine() &&
                        sampling_json(sampler, position).dump() == sampling && mtp_generation.controller_state() == controller &&
                        !mtp_generation.active() && !mtp_generation.pending_overflow() && !mtp_generation.carry().peek(position),
                        "reclamation changed accepted model, sampling or controller state");
                    std::cout << decorate(Json(Json::Object{{"event", Json("done")}, {"generated_ids", Json(Json::Array{})},
                        {"committed_prefix", Json::integer(position)}, {"carrier_cache_reclaim", Json(Json::Object{
                            {"injected_guard_readings", Json(true)}, {"persistent_warning", Json(persistent)}, {"admitted", Json(admitted)},
                            {"guard_readings", Json::integer(readings)}, {"reclamation_attempts", Json::integer(cache_reclamations - attempts)},
                            {"row_cache_bytes_before", Json::integer(row_cache_bytes)}, {"row_cache_bytes_after", Json::integer(cache.charged_bytes())}, {"anchors_before", Json::integer(anchor_count)},
                            {"released_accounted_bytes", Json::integer(cache_reclaimed_bytes - released)}, {"allocator_released_bytes", Json::integer(cache_allocator_released_bytes - allocator)},
                            {"memory_before", describe(before)}, {"memory_after", describe(after)},
                            {"stack_bytes", Json::integer(stack_bytes)}, {"drafter_bytes", Json::integer(draft_bytes)},
                            {"stack_sha256", Json(stack_hash)}, {"drafter_sha256", Json(draft_hash)}, {"complete_logits_sha256", Json(logits_hash)},
                            {"full_rng_unchanged", Json(true)}, {"controller_unchanged", Json(true)}, {"accepted_state_unchanged", Json(true)}})},
                        {"timings_qualified", Json(false)}})).dump() << std::endl;
                    continue;
                }
#endif
                require(!gpu_readback || gpu.carrier_gpu_readback_commands() > 0, "GPU warm inspection did not read device state");
                uint32_t primed_regions = 0;
                if (prime_weights) {
                    gpu.begin_carrier_gpu_readback(); end_state_readback.enabled = true;
                    primed_regions = gpu.prime_carrier_mapped_weights();
                }
                const SamplingRng fresh(sampler.options().seed);
                const bool fresh_rng = sampler.random().words_consumed() == 0 && sampler.random().engine() == fresh.engine();
                require(fresh_rng, "warm inspection requires fresh sampling entropy");
                if (operation == "capture") {
                    require(ssd_manager && ssd_manager->enabled() && ssd_token_history.size() == position,
                        "warm capture requires prepared SSD ownership and complete history");
                    const auto name = ssd_filename_for(ssd_token_history, true);
                    if (!ssd_manager->has_checkpoint(name))
                        require(spill_to_ssd(ssd_token_history, false), "warm checkpoint capture failed");
                }
                own_memory_guard();
                std::cout << decorate(Json(Json::Object{{"event", Json("done")}, {"generated_ids", Json(Json::Array{})},
                    {"committed_prefix", Json::integer(position)}, {"carrier_warm_state", Json(Json::Object{
                        {"execution_digest", Json(hex(ssd_execution_digest_for(uint32_t(stack.execution()), stack.known_residual_reuse(), gpu.kernel_policy())))},
                        {"stack_sha256", Json(stack_hash)}, {"drafter_sha256", Json(draft_hash)},
                        {"complete_logits_sha256", Json(hc_exact_logits_hash(retained.get(position).logits))},
                        {"stack_bytes", Json::integer(stack_bytes)}, {"drafter_bytes", Json::integer(draft_bytes)}, {"prefix", Json::integer(position)},
                        {"readback", Json(gpu_readback ? "gpu-copy" : "host")},
                        {"readback_commands", Json::integer(gpu_readback ? gpu.carrier_gpu_readback_commands() : 0)},
                        {"primed_mapped_regions", Json::integer(primed_regions)},
                        {"drafter_first", Json::integer(checkpoint_owner->first_absolute_position())}, {"drafter_rows", Json::integer(checkpoint_owner->cache_rows())},
                        {"generation_active", Json(false)}, {"pending", Json::integer(0)}, {"carried", Json(false)},
                        {"fresh_rng", Json(fresh_rng)}, {"sampling", sampling_json(sampler, position)}})}, {"timings_qualified", Json(false)}})).dump() << std::endl;
                continue;
            }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (measured_controller) measured_controller->begin_packet();
#endif
            if (ticket && (ticket->begin || ticket->capture)) {
                files.assert_unchanged(); own_memory_guard();
                const auto started = Clock::now(); uint32_t common = position; std::string outcome;
                if (ticket->begin) {
                    std::vector<uint32_t> future; future.reserve(field(body, "prompt_ids").array().size());
                    for (const auto& id : field(body, "prompt_ids").array()) future.push_back(uint32_t(id.unsigned_integer()));
                    const bool reuse = field(body, "allow_restore").boolean();
                    const bool gen_req = field(body, "generation_requested").boolean();
                    const auto options = target_sampling(body, true);
                    common = reuse ? uint32_t(branches->common_prefix(future)) : 0;
                    const auto selected = reuse ? branches->select(future, gen_req) : std::optional<size_t>{};
                    broker->started(*ticket);
                    if (selected) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                        StageAControllerState restored_ctrl{};
                        uint32_t restored_mtp_rows = 0;
                        std::optional<SampledPathToken> restored_anchor{};
                        uint32_t restored_overflow = 0;
                        branches->restore(*selected, retained, sampler, options,
                            [&](StackCheckpoint handle) { return stack.restore_checkpoint(handle); },
                            &restored_ctrl, &restored_mtp_rows, &restored_anchor, &restored_overflow);
                        if (mtp) {
                            const uint32_t target_rows = stack.position() > 1 ? stack.position() - 1 : 0;
                            if (target_rows <= mtp->cache_rows()) mtp->truncate(target_rows);
                            else mtp->reset(1);
                        } else if (dflash2) {
                            if (stack.position() <= dflash2->context_len()) dflash2->truncate(stack.position());
                            else dflash2->reset();
                        }
                        if (has_speculative) {
                            mtp_generation.controller_state() = restore_controller_policy(restored_ctrl, mtp_generation.controller_policy(), mtp_generation.controller_policy() != default_controller_policy_digest);
                        }
                        if (ssd_manager && ssd_manager->enabled()) {
                            ssd_token_history.assign(future.begin(), future.begin() + stack.position());
                        }
                        if (mtp_lookup) {
                            if (mtp_lookup_valid && stack.position() <= mtp_lookup->committed_length()) {
                                static_cast<void>(mtp_lookup->truncate(mtp_lookup->epoch(), stack.position()));
                            } else {
                                static_cast<void>(mtp_lookup->reset());
                                mtp_lookup->append(mtp_lookup->epoch(), 0, std::span<const uint32_t>(future).first(stack.position()));
                                mtp_lookup_valid = true;
                            }
                        }
#else
                        branches->restore(*selected, retained, sampler, options,
                            [&](StackCheckpoint handle) { return stack.restore_checkpoint(handle); });
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                        if (worker_broker8) lookup->restore(stack.position());
#endif
                        outcome = "restored";
                        session_generated_since_reset = true;
                    } else {
                        bool ssd_hit = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                        if (reuse && ssd_manager && ssd_manager->enabled()) {
                            const auto cur_exec_digest = ssd_execution_digest_for(
                                uint32_t(stack.execution()), stack.known_residual_reuse(), gpu.kernel_policy());
                            ssd_manager->set_execution_identity(cur_exec_digest);
                            auto entries = ssd_manager->list_entries();
                            std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                                return a.prefix_tokens > b.prefix_tokens;
                            });
                            for (const auto& entry : entries) {
                                if (!entry.prefix_tokens || entry.prefix_tokens > future.size()) continue;
                                if (gen_req && entry.prefix_tokens == future.size()) continue;
                                const auto cand = std::span<const uint32_t>(future).first(entry.prefix_tokens);
                                if (entry.filename == ssd_filename_for(cand, true) && restore_from_ssd(cand, true, options)) {
                                    branches->reset();
                                    require(branches->committed(stack.position(), cand), "SSD branch_begin history sync failed");
                                    common = stack.position();
                                    outcome = "restored";
                                    ssd_hit = true;
                                    session_generated_since_reset = true;
                                    break;
                                }
                            }
                        }
#endif
                        if (!ssd_hit) {
                            reset_session(retained, sampler, options, [&] { stack.reset(); }); branches->reset(); outcome = "cold_miss";
                            session_generated_since_reset = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                            ssd_token_history.clear();
                            last_ssd_spill_prefix = 0;
                            if (mtp_lookup) { static_cast<void>(mtp_lookup->reset()); mtp_lookup_valid = true; }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                            if (worker_broker8) { lookup->reset(); lookup_active = true; }
#endif
                        }
                    }
                    broker->published(*ticket);
                } else {
                    broker->started(*ticket);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    const StageAControllerState* ctrl_p = has_speculative ? &mtp_generation.controller_state() : nullptr;
                    const uint32_t mtp_rows_val = mtp ? mtp->cache_rows() : 0;
                    const auto* anchor_p = mtp_sidecar ? mtp_generation.carry().peek(position) : nullptr;
                    const std::optional<SampledPathToken> anchor_val = anchor_p ? std::optional<SampledPathToken>(*anchor_p) : std::nullopt;
                    const uint32_t overflow_val = has_speculative ? mtp_generation.pending_overflow() : 0;
                    branches->capture(position, retained, &sampler, ctrl_p, 1, mtp_rows_val, anchor_val, overflow_val,
                        [&] { return stack.retain_checkpoint(); });
                    if (ssd_manager && ssd_manager->enabled() && ssd_token_history.size() == position) {
                        static_cast<void>(spill_to_ssd(ssd_token_history, position <= 1024));
                    }
#else
                    branches->capture(position, retained, [&] { return stack.retain_checkpoint(); });
#endif
                    broker->published(*ticket); broker->captured(); outcome = "captured";
                }
                const auto stop = retained.stop_id();
                std::cout << decorate(Json(Json::Object{{"event", Json("control_done")}, {"op", Json(ticket->begin ? "branch_begin" : "branch_capture")},
                    {"outcome", Json(outcome)}, {"model_steps", Json::integer(0)}, {"generated_ids", Json(Json::Array{})},
                    {"common_prefix", Json::integer(common)}, {"committed_prefix", Json::integer(stack.position())},
                    {"sampling", sampling_json(sampler, stack.position())}, {"stop_id", stop ? Json::integer(*stop) : Json()},
                    {"logits_available", Json(retained.matches(stack.position()))}, {"continuation_allowed", Json(retained.continuable(stack.position()))},
                    {"checkpoint_slots", Json::integer(checkpoint_slots)}, {"checkpoint_combined_budget_bytes", Json::integer(checkpoint_budget)},
                    {"checkpoint_host_bytes", Json::integer(branches->charged_bytes())}, {"checkpoint_store_host_bytes", Json::integer(stack.checkpoint_memory().host_metadata_bytes)},
                    {"checkpoint_gpu_bytes", Json::integer(stack.checkpoint_memory().gpu_charged_bytes)}, {"checkpoint_retained", Json::integer(branches->retained_count())},
                    {"control_wall_ms", number(milliseconds(started))}, {"timings_qualified", Json(false)}})).dump() << std::endl;
                continue;
            }
            const Json* request_body = &body;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            const auto* carrier_component = body.find("carrier_component");
            const auto* carrier_prepare = body.find("carrier_prepare_ordinary");
            const auto* carrier_require_warm = body.find("carrier_require_warm");
            Json carrier_filtered;
            if (carrier_component || carrier_prepare) {
                require(carrier_forced_input && carrier_mode != CarrierDiagnosticMode::off && !broker && !verify_cost_profile,
                    "target components require explicit bounded forced-history diagnostics without a live controller or broker");
                require(!carrier_component || !carrier_prepare, "target component and history preparation are separate commands");
                auto fields = body.object();
                fields.erase(std::remove_if(fields.begin(), fields.end(), [](const auto& item) { return item.first == "carrier_component" || item.first == "carrier_prepare_ordinary"; }), fields.end());
                carrier_filtered = Json(std::move(fields)); request_body = &carrier_filtered;
            }
            if (carrier_require_warm) {
                require(carrier_require_warm->boolean() && carrier_mode != CarrierDiagnosticMode::off && !broker &&
                    !carrier_component && !carrier_prepare && ssd_manager && ssd_manager->enabled(),
                    "required warm reuse needs diagnostic SSD preparation");
                auto fields = body.object();
                fields.erase(std::remove_if(fields.begin(), fields.end(), [](const auto& item) { return item.first == "carrier_require_warm"; }), fields.end());
                carrier_filtered = Json(std::move(fields)); request_body = &carrier_filtered;
            }
#endif
            const auto job = request(*request_body, position, small_gemv_splits,
                block_gemv_formats, allow_kernel_experiments, retained.matches(position), retained.continuable(position), sampler.matches(position),
                uint32_t(stack.execution()), allow_stack_experiments, stack.replay_prepared(), known_token_capacity, allow_prefill_experiments,
                stack.known_residual_reuse(), checkpoint_slots, bool(lookup), allow_lookup_experiments, lookup_active
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                , native_prefill_final_logits
#endif
                ); files.assert_unchanged();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            require(!carrier_forced_input || !job.generate, "forced-history components cannot measure live controller decisions");
            require(!carrier_component_needs_reset || job.reset, "target verification component requires an explicit reset before further work");
            if (job.reset) { carrier_component_needs_reset = false; carrier_preparation_sealed = false; }
            require(!carrier_require_warm || (job.reset && !job.generate && !job.tokens.empty() && !job.branch && !job.lookup),
                "required warm reuse needs a fresh complete known-input prefix");
            if (carrier_prepare) {
                require(!carrier_preparation_sealed && !job.reset && !body.find("known_prefill") && !job.generate && !job.branch && !job.lookup &&
                    job.stops.empty() && position && !mtp_generation.active() && (mtp || dflash2),
                    "ordinary history preparation must precede components after a fresh fixed prefix");
                const auto steps = carrier_ordinary_history_steps(*carrier_prepare, uint32_t(job.tokens.size()));
                flush_mtp_deferred(); if (dflash2) static_cast<void>(dflash2->flush_context());
                for (const auto token : job.tokens) advance(token);
                if (!carrier_capture_path.empty()) carrier_state_record(stack, "carrier_diagnostic_preparation_state", 0);
                carrier_preparation_sealed = true;
                std::cout << cache_memory_metadata(Json(Json::Object{{"event", Json("done")}, {"carrier_preparation", Json(true)}, {"component_kind", Json("ordinary-history")},
                    {"base", Json::integer(position)}, {"steps", Json::integer(steps)}, {"committed_prefix", Json::integer(stack.position())},
                    {"continuation_allowed", Json(true)}, {"timings_qualified", Json(false)}}), gpu).dump() << std::endl;
                continue;
            }
            const auto carrier_prefix = carrier_component ? carrier_component_prefix(*carrier_component, uint32_t(job.tokens.size())) : 0u;
            require(!carrier_component || (!job.reset && !body.find("known_prefill") && !job.generate && !job.branch && !job.lookup &&
                job.stops.empty() && position && !mtp_generation.active() && (mtp || dflash2)),
                "target component requires aligned nonempty fixed history and no generation, stop, cache or lookup control");
            const bool carrier_known = (carrier_forced_input || carrier_mode == CarrierDiagnosticMode::replay) &&
                ((job.tokens.size() == 1 && !job.generate && !job.reset && !job.known_prefill) || carrier_component);
            if (carrier_known) carrier_preparation_sealed = true;
            if (carrier_known) { flush_mtp_deferred(); if (dflash2) static_cast<void>(dflash2->flush_context()); }
            if (carrier_known && !carrier_capture_path.empty())
                carrier_state_record(stack, "carrier_diagnostic_initial_state", gpu.carrier_diagnostic_action() + 1);
            CarrierAction carrier_known_action(gpu, stack, carrier_known ? carrier_mode : CarrierDiagnosticMode::off,
                false, !carrier_capture_path.empty(), mtp_generation.controller_state(), carrier_known);
            if (carrier_component) {
                carrier_preparation_sealed = true;
                const auto component_started = Clock::now();
                const auto count = uint32_t(job.tokens.size());
                if (count == 1) advance(job.tokens.front());
                else {
                    const auto staged = stack.sampled_verification8_lookups(job.tokens);
                    for (uint32_t t = 0; t < count; ++t) {
                        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(job.tokens[t]) * row_bytes, row_bytes),
                            std::span(carrier_component_rows).subspan(uint64_t(t) * 2560, 2560));
                        cache.gather(staged[t].rows, std::span(carrier_component_memory).subspan(uint64_t(t) * 2560, 2560));
                    }
                    CarrierComponentSelection selector{carrier_prefix};
                    const auto result = stack.verify_sampled8(job.tokens,
                        std::span(carrier_component_rows).first(uint64_t(count) * 2560),
                        std::span(carrier_component_memory).first(uint64_t(count) * 2560), position,
                        {CarrierComponentSelection::choose, &selector, sampling_topk_eligible(sampler.options().policy, shape.vocabulary)});
                    require(result.committed == carrier_prefix && stack.position() == position + carrier_prefix, "target component physical frontier differs");
                    // This target-only branch deliberately leaves auxiliary context untouched. It cannot continue serving or seed a full-action cost profile.
                    sampler.invalidate(); retained.clear(); carrier_component_needs_reset = true;
                }
                { TOKSPUR_CARRIER_PHASE(publication);
                std::cout << cache_memory_metadata(Json(Json::Object{{"event", Json("done")}, {"carrier_component", Json(true)}, {"component_kind", Json("target-rows")},
                    {"base", Json::integer(position)}, {"rows", Json::integer(count)}, {"accepted_prefix", Json::integer(carrier_prefix)},
                    {"committed_prefix", Json::integer(stack.position())}, {"continuation_allowed", Json(count == 1)},
                    {"wall_ms", number(milliseconds(component_started))}, {"timings_qualified", Json(false)}}), gpu).dump() << std::endl; }
                continue;
            }
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
            const auto operator_rounds_before = mtp_rounds, operator_accepted_before = mtp_accepted;
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
            if(route_census_only){require(operator_requests<2048 && !job.branch && !job.lookup && job.stops.empty() &&
                job.tokens.size()<=2048 && job.generate<=256,"occupancy census request bound");route_lock_sink.request_index=operator_requests;}
#endif
            if (fixed_operator_screen) {
                const bool allow_operator_4k =
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
                    operator_qsa_sparse.has_value() || qsa_route_attribution_path.has_value() ||
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
                    operator_gdn_ported.has_value() || gdn_route_attribution_path.has_value() ||
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                    route_lock_output.has_value() ||
#endif
                    false;
                if (operator_requests == 0 && allow_operator_4k && job.tokens.size() == 4096) operator_base_prefix = 4096;
                require(operator_requests < 3 && !job.branch && !job.lookup && job.stops.empty() &&
                    ((operator_requests == 0 && position == 0 && job.reset && job.known_prefill && job.tokens.size() == operator_base_prefix && !job.generate) ||
                    (operator_requests == 1 && position == operator_base_prefix && !job.reset && job.tokens.empty() && job.generate == 3) ||
                    (operator_requests == 2 && position == operator_base_prefix + 3 && !job.reset && job.known_prefill && job.tokens.size() == 1 && !job.generate)),
                    "operator screen accepts only prefill512 (or prefill4096 for sparse QSA or ported GDN),generate3 and forced1 continuation");
            }
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
            if (kernel_search_capture_path) require(kernel_search_requests < 3 && !job.generate && !job.branch && !job.lookup && job.stops.empty() && job.known_prefill &&
                job.reset == (kernel_search_requests == 0) && job.tokens.size() == (kernel_search_requests ? 1 : 4095) && job.sampling.policy.temperature == 0,
                "kernel capture accepts prefill4095 then two forced ordinary rows only");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
            gdn_old_order_packet(native_gdn_old_order_batch, job.tokens.size());
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            if (task4_profile.has_value()) require(position <= 4096 && !job.generate && !job.branch && !job.lookup && job.stops.empty() &&
                job.sampling.policy.temperature == 0 && job.reset == (position == 0) &&
                ((position < 4096 && position % 8 == 0 && job.tokens.size() == 8) || (position == 4096 && job.tokens.size() == 1)),
                "Task4 requires exactly512 input8 packets then one forced ordinary decode, with no retry");
            if (task4_profile.value_or(false) && task4_sink.boundary == Task4ProfileBoundary::encoder_late8 && position == 4088)
                gpu.begin_task4_profile_window(position);
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            const auto* generation_field = body.find("mtp_generation");
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
            require(!mtp_output_packet_limit || job.generate <= *mtp_output_packet_limit, "native MTP output command exceeds startup packet bound");
#endif
            require(!generation_field || mtp_request_carry, "MTP generation command requires explicit startup consent");
            const auto generation_command = generation_field ? std::optional<MtpGenerationCommand>(mtp_generation_command(*generation_field)) : std::nullopt;
            if (mtp_request_carry) {
                if (!job.tokens.empty() || job.reset) {
                    require(!generation_command && !job.generate, "MTP scoped known input cannot carry output or a generation command");
                    if (mtp_generation.clear()) ++mtp_discarded_anchors;
                } else {
                    require(generation_command.has_value(), "MTP output/control requires an explicit generation lifetime");
                    if (generation_command->abort) {
                        require(!job.generate && job.stops.empty(), "MTP abort cannot execute model work");
                        if (mtp_generation.abort(*generation_command)) ++mtp_discarded_anchors;
                    } else mtp_generation.admit(*generation_command, job.generate, position, job.stops
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
                        , mtp_output_packet_limit.value_or(8)
#endif
                        );
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                    if (!generation_command->abort && generation_command->begin && carrier_mode != CarrierDiagnosticMode::off && !measured_controller) {
                        const auto priming_started = Clock::now();
                        flush_mtp_deferred();
                        if (dflash2) static_cast<void>(dflash2->flush_context());
                        const auto record = Json(Json::Object{{"event", Json("carrier_diagnostic_priming")}, {"prefix", Json::integer(stack.position())},
                            {"wall_ms", number(milliseconds(priming_started))}, {"excluded_from_decode", Json(true)}, {"timings_qualified", Json(false)}}).dump() + "\n";
                        require(std::fwrite(record.data(), 1, record.size(), stderr) == record.size() && std::fflush(stderr) == 0, "diagnostic priming record failed");
                    }
#endif
                    if (!generation_command->abort && measured_controller) {
                        require(generation_command->v2, "measured controllers refuse legacy generation packets");
                        if (generation_command->begin) {
                            flush_mtp_deferred();
                            if (dflash2) { static_cast<void>(dflash2->flush_context()); dflash2->reset_verify_budget(); }
                            const auto policy = sampler.options().policy;
                            const auto sampling = measured_sampling(policy);
                            measured_controller->begin_generation(generation_command->id, position, sampling);
                            mtp_generation.controller_state().measured = measured_controller->observation();
                            mtp_generation.controller_state().ewma_alpha1 = measured_controller->acceptance()[0];
                        }
                    }
                }
            }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            require(!sampled_prefill_input8 || ((job.tokens.size() >= 1 && job.tokens.size() <= 8 && !job.generate) ||
                (job.tokens.empty() && job.generate == 1)), "sampled input8 target packet requires1..8 known inputs OR1 output");
#endif
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
            require(!logit_trace || (!job.lookup && job.sampling.policy.temperature == 0), "private logit trace accepts target-only greedy requests");
#endif
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
            if (verification_cost_manifest) {
                require(verification_cost_prefix < verification_cost_manifest->prefixes.size(), "verification cost manifest already exhausted");
                require(job.reset && !job.branch && !job.generate && job.stops.empty() && job.known_prefill && job.lookup &&
                    job.sampling.policy.temperature == 0 && job.sampling.seed == verification_cost_manifest->seed &&
                    job.tokens == verification_cost_manifest->prefixes[verification_cost_prefix].input,
                    "verification cost request must exactly match the next frozen prefix, greedy reset and generate0");
            }
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
            if (sampled_cost_manifest) {
                require(job.reset && !job.branch && !job.generate && job.stops.empty() && job.known_prefill && !job.lookup &&
                    job.sampling.policy.temperature == 1 && job.sampling.policy.top_p == 0.95 &&
                    job.sampling.policy.top_k == 20 && job.sampling.policy.min_p == 0 &&
                    job.sampling.seed == sampled_cost_manifest->seed && job.tokens == sampled_cost_manifest->input,
                    "sampled COST request must match its frozen input, sampled policy, reset and generate0");
                // This process serves exactly this diagnostic, never resumes an
                // agent after target-only reset/reprepare bypassed head priming.
                sampler.invalidate(); retained.clear();
                run_sampled_verification_cost(*sampled_cost_manifest);
                return 0;
            }
#endif
            require(!lookup || (job.execution == 0 && job.small_splits == 16 && job.block_formats == 3), "prepared lookup policy changed");
            require(!lookup_active || job.reset || lookup->matches(position), "invalidated lookup requires explicit reset");
            require(!job.lookup || !job.branch || (lookup_active && lookup->matches(position)), "branch lookup restore requires coherent current ancestry; use explicit cold reset");
            if (broker) broker->started(*ticket);
            const auto started = Clock::now();
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
            const uint64_t dispatches_before_prefill = gpu.lifetime_dispatches();
            if (!job.tokens.empty() || job.reset) {
                gpu.reset_prefill_top10_mixture_telemetry();
            }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
            hc_down_f32_request_before = gpu.prefill_hc_down_f32_encoded_calls();
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
            hc_rows_request_before = gpu.hc_rows_encoded();
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
            router_window_request_before = gpu.router_window_encoded();
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
            hc_down_window_request_before = gpu.hc_down_window_encoded();
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
            qsa_sparse_request_before = {gpu.qsa_sparse_decode_calls(), gpu.qsa_sparse_verify_calls()};
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
            gdn_ported_request_before = gpu.gdn_ported_routes();
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
            if (native_raw_decode.has_value()) raw_decode_request_before = gpu.raw_decode_routes();
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
            gdn_old_order_request_before = gpu.gdn_legacy_batch_encoded_calls();
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
            const auto stage_request_before = gpu.qsa_stage_v8_calls();
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
            const auto channel_request_before = gpu.qsa_channel_calls();
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
            const auto hc_request_before = gpu.hc_exact_routes();
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
            const auto q8_three_request_before = gpu.q8_three_rows_routes();
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
            const auto wide_request_before = gpu.wide_q8_exact_routes();
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
            const auto wide_many_request_before = gpu.wide_q8_many_routes();
#endif
            const auto lookup_ledger_before = lookup ? lookup->ledger() : LookupLedger{};
            size_t input_at = 0; double restore_ms = 0, capture_ms = 0;
            bool ssd_restored_this_request = false;
            bool shared_prefix_restored_this_request = false;
            bool request_overlap_pipelined = false;
            std::optional<size_t> selected;
            if (job.branch) selected = branches->select(job.tokens, job.generate);
            if (selected) {
                const auto restore_started = Clock::now();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                StageAControllerState restored_ctrl{};
                uint32_t restored_mtp_rows = 0;
                std::optional<SampledPathToken> restored_anchor{};
                uint32_t restored_overflow = 0;
                input_at = branches->restore(*selected, retained, sampler, job.sampling,
                    [&](StackCheckpoint handle) { return stack.restore_checkpoint(handle); },
                    &restored_ctrl, &restored_mtp_rows, &restored_anchor, &restored_overflow);
                if (mtp) {
                    const uint32_t target_rows = stack.position() > 1 ? stack.position() - 1 : 0;
                    if (target_rows <= mtp->cache_rows()) mtp->truncate(target_rows);
                    else mtp->reset(1);
                } else if (dflash2) {
                    if (stack.position() <= dflash2->context_len()) dflash2->truncate(stack.position());
                    else dflash2->reset();
                }
                if (has_speculative) {
                    mtp_generation.controller_state() = restore_controller_policy(restored_ctrl, mtp_generation.controller_policy(), mtp_generation.controller_policy() != default_controller_policy_digest);
                }
                if (ssd_manager && ssd_manager->enabled()) {
                    ssd_token_history.assign(job.tokens.begin(), job.tokens.begin() + input_at);
                }
                if (mtp_lookup) {
                    if (mtp_lookup_valid && input_at <= mtp_lookup->committed_length()) {
                        static_cast<void>(mtp_lookup->truncate(mtp_lookup->epoch(), uint32_t(input_at)));
                    } else {
                        static_cast<void>(mtp_lookup->reset());
                        mtp_lookup->append(mtp_lookup->epoch(), 0, std::span<const uint32_t>(job.tokens).first(input_at));
                        mtp_lookup_valid = true;
                    }
                }
#else
                input_at = branches->restore(*selected, retained, sampler, job.sampling,
                    [&](StackCheckpoint handle) { return stack.restore_checkpoint(handle); });
#endif
                if (job.lookup) lookup->restore(uint32_t(input_at)); else if (lookup) lookup->invalidate();
                lookup_active = job.lookup;
                session_generated_since_reset = true;
                restore_ms = milliseconds(restore_started);
            } else if (job.branch) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (ssd_manager && ssd_manager->enabled()) {
                    const auto restore_started = Clock::now();
                    const auto job_exec_digest = ssd_execution_digest_for(
                        job.execution, job.residual_reuse, std::array<uint32_t, 2>{job.small_splits, job.block_formats});
                    ssd_manager->set_execution_identity(job_exec_digest);
                    auto entries = ssd_manager->list_entries();
                    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                        return a.prefix_tokens > b.prefix_tokens;
                    });
                    for (const auto& entry : entries) {
                        if (!entry.prefix_tokens || entry.prefix_tokens > job.tokens.size()) continue;
                        if (job.generate && entry.prefix_tokens == job.tokens.size()) continue;
                        const auto cand = std::span<const uint32_t>(job.tokens).first(entry.prefix_tokens);
                        if (entry.filename == ssd_filename_for_digest(cand, true, job_exec_digest) &&
                            restore_from_ssd_with_digest(cand, true, job.sampling, job_exec_digest)) {
                            if (branches) {
                                branches->reset();
                                require(branches->committed(stack.position(), cand), "SSD branch history sync failed");
                            }
                            input_at = stack.position();
                            ssd_restored_this_request = true;
                            session_generated_since_reset = true;
                            lookup_active = job.lookup;
                            if (lookup_active) {
                                lookup->reset();
                                lookup->committed(stack.position(), cand);
                            } else if (lookup) lookup->invalidate();
                            restore_ms = milliseconds(restore_started);
                            break;
                        }
                    }
                }
#endif
                if (!ssd_restored_this_request && job.reset) {
                    commit_recent_shared_prefix();
                    shared_prefix_cache.active_tokens.clear();
                    shared_prefix_cache.active_valid = shared_prefix_cache_enabled && !broker;
                    reset_session(retained, sampler, job.sampling, [&] {
                        stack.reset();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                        if (mtp) mtp->reset(1);
                        if (dflash2) dflash2->reset();
                        ssd_token_history.clear();
                        last_ssd_spill_prefix = 0;
                        if (mtp_lookup) { static_cast<void>(mtp_lookup->reset()); mtp_lookup_valid = true; }
#endif
                    });
                    session_generated_since_reset = false;
                    if (branches) branches->reset();
                    lookup_active = job.lookup;
                    if (lookup_active) lookup->reset(); else if (lookup) lookup->invalidate();
                }
            } else if (job.reset) {
                commit_recent_shared_prefix();
                shared_prefix_cache.active_tokens.clear();
                shared_prefix_cache.active_valid = shared_prefix_cache_enabled && !broker;
                reset_session(retained, sampler, job.sampling, [&] {
                    stack.reset();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    if (mtp) mtp->reset(1);
                    if (dflash2) dflash2->reset();
                    ssd_token_history.clear();
                    last_ssd_spill_prefix = 0;
                    if (mtp_lookup) { static_cast<void>(mtp_lookup->reset()); mtp_lookup_valid = true; }
#endif
                });
                session_generated_since_reset = false;
                if (branches) branches->reset();
                lookup_active = job.lookup;
                if (lookup_active) lookup->reset(); else if (lookup) lookup->invalidate();
            }
            if (broker) broker->published(*ticket);
            if (job.execution != uint32_t(stack.execution())) stack.configure_execution(StackExecution(job.execution));
            if (job.residual_reuse != stack.known_residual_reuse()) stack.configure_known_residual_reuse(job.residual_reuse);
            // Overrides require reset and validation above. Reset first so a
            // configuration refusal cannot leave an old prefix under a new mode.
            if (job.small_splits != small_gemv_splits) { gpu.configure_small_gemv(job.small_splits); small_gemv_splits = job.small_splits; }
            if (job.block_formats != block_gemv_formats) { gpu.configure_block_gemv(job.block_formats); block_gemv_formats = job.block_formats; }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!job.branch && !broker && !job.tokens.empty() && ssd_manager && ssd_manager->enabled() &&
                ssd_token_history.size() == stack.position()) {
                std::vector<uint32_t> candidate_history = ssd_token_history;
                candidate_history.insert(candidate_history.end(), job.tokens.begin(), job.tokens.end());
                const bool packet_emits_logits = !job.known_prefill
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                    || !job.final_logits || job.prefill_final
#endif
                    ;
                const auto restore_started = Clock::now();
                const auto opts = job.reset ? job.sampling : sampler.options();
                const auto original_prefix = stack.position();
                auto entries = ssd_manager->list_entries();
                std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.prefix_tokens > b.prefix_tokens; });
                for (const auto& entry : entries) {
                    if (entry.prefix_tokens <= original_prefix || entry.prefix_tokens > candidate_history.size()) continue;
                    const auto prefix = std::span<const uint32_t>(candidate_history).first(entry.prefix_tokens);
                    const bool needs_logits = packet_emits_logits && entry.prefix_tokens == candidate_history.size();
                    bool restored = false;
                    for (const bool logits : {true, false}) {
                        if ((!logits && needs_logits) || entry.filename != ssd_filename_for(prefix, logits)) continue;
                        restored = restore_from_ssd(prefix, logits, opts);
                        if (restored) break;
                    }
                    if (!restored) continue;
                    const uint32_t restored_delta = entry.prefix_tokens - original_prefix;
                    if (mtp || dflash2) mtp_prime_calls += restored_delta - (mtp && job.reset ? 1u : 0u);
                    if (branches) {
                        if (job.reset) branches->reset();
                        require(branches->committed(stack.position(), std::span<const uint32_t>(job.tokens).first(restored_delta)), "SSD prefill branch history sync failed");
                    }
                    if (lookup_active) {
                        if (job.reset) lookup->reset();
                        lookup->committed(stack.position(), std::span<const uint32_t>(job.tokens).first(restored_delta));
                    }
                    input_at = restored_delta;
                    ssd_restored_this_request = true;
                    session_generated_since_reset = true;
                    restore_ms = milliseconds(restore_started);
                    break;
                }
            }
#endif
            if (shared_prefix_cache_enabled && !ssd_restored_this_request && !job.branch && !broker &&
                !job.tokens.empty() && shared_prefix_cache.active_valid &&
                shared_prefix_cache.active_tokens.size() == stack.position()) {
                const uint32_t before_pos = stack.position();
                std::vector<uint32_t> candidate_history = shared_prefix_cache.active_tokens;
                candidate_history.insert(candidate_history.end(), job.tokens.begin(), job.tokens.end());
                const bool packet_emits_logits = !job.known_prefill
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                    || !job.final_logits || job.prefill_final
#endif
                    ;
                SharedPrefixAnchor* best_anchor = nullptr;
                for (auto& anchor : shared_prefix_cache.anchors) {
                    const size_t alen = anchor.tokens.size();
                    if (alen <= before_pos || alen > candidate_history.size()) continue;
                    if (alen < candidate_history.size() && !job.known_prefill) continue;
                    if (alen == candidate_history.size() && (packet_emits_logits || job.generate > 0) && !anchor.has_logits) continue;
                    if (!std::equal(anchor.tokens.begin(), anchor.tokens.end(), candidate_history.begin())) continue;
                    if (!best_anchor || alen > best_anchor->tokens.size()) best_anchor = &anchor;
                }
                if (best_anchor) {
                    const auto restore_started = Clock::now();
                    const bool full_packet_restore = best_anchor->tokens.size() == candidate_history.size();
                    const bool expect_logits = full_packet_restore ? (packet_emits_logits || job.generate > 0) : false;
                    const auto opts = job.reset ? job.sampling : sampler.options();
                    if (restore_shared_prefix_anchor(*best_anchor, expect_logits, opts)) {
                        const size_t restored_in_job = best_anchor->tokens.size() - before_pos;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                        if (mtp || dflash2) {
                            mtp_prime_calls += uint32_t(restored_in_job) - (mtp && before_pos == 0 ? 1u : 0u);
                        }
#endif
                        const auto restored_span = std::span<const uint32_t>(job.tokens).first(restored_in_job);
                        if (branches) {
                            if (job.reset) branches->reset();
                            require(branches->committed(stack.position(), restored_span), "shared prefix branch history sync failed");
                        }
                        if (lookup_active) {
                            if (job.reset) lookup->reset();
                            lookup->committed(stack.position(), restored_span);
                        }
                        shared_prefix_cache.active_tokens.assign(best_anchor->tokens.begin(), best_anchor->tokens.end());
                        ++shared_prefix_cache.hits;
                        shared_prefix_cache.restored_tokens += restored_in_job;
                        input_at = restored_in_job;
                        shared_prefix_restored_this_request = full_packet_restore;
                        session_generated_since_reset = true;
                        restore_ms += milliseconds(restore_started);
                    }
                }
            }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            require(!carrier_require_warm || (ssd_restored_this_request && input_at == job.tokens.size()),
                "required warm checkpoint was not restored; cold fallback refused");
#endif
            if (overlap_prefill && job.known_prefill
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                && job.prefill_final
#endif
                && job.tokens.size() == 16 && position >= 16) {
                request_overlap_pipelined = true;
            }
            if (job.known_prefill
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
                && !(fixed_operator_screen && operator_requests == 2)
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                && !(task4_profile.has_value() && position == 4096)
#endif
                ) {
                const bool split_boundary_16 = overlap_prefill
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
                    && !operator_checkpoints.has_value()
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
                    && !kernel_search_capture_path
#endif
                    && (job.tokens.size() - input_at > 16);
                if (split_boundary_16) request_overlap_pipelined = true;
                const size_t prefill_slice_cap =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                    (native_prefill_dense_nax && native_prefill_grouped_nax) ? size_t(known_token_capacity) : std::min<size_t>(known_token_capacity, 512u);
#else
                    std::min<size_t>(known_token_capacity, 512u);
#endif
                const size_t base_pos_at_input = stack.position();
                const auto chunk_count_at = [&](size_t offset) -> size_t {
                    if (offset >= job.tokens.size()) return 0;
                    size_t n = std::min<size_t>(prefill_slice_cap, job.tokens.size() - offset);
                    if (split_boundary_16 && job.tokens.size() - offset > 16) {
                        n = std::min<size_t>(n, (job.tokens.size() - 16) - offset);
                    } else if (ple_async_gather_enabled && ple_overlap && ple_host_cap >= 512) {
                        const size_t abs_pos = base_pos_at_input + (offset - input_at);
                        if (prefill_slice_cap <= 512 && offset == input_at && (job.tokens.size() - offset >= 1024) && n > 512) {
                            n = 512;
                        } else if (prefill_slice_cap == 2048 && (abs_pos % 512 == 0) && (abs_pos % 2048 != 0)) {
                            n = std::min<size_t>(n, 2048 - (abs_pos % 2048));
                        }
                    }
                    if (ple_async_gather_enabled && !split_boundary_16 && n > 2 && (job.tokens.size() - offset - n == 1)) --n;
                    return n;
                };
                bool has_prefetched_chunk = false;
                size_t prefetched_at = 0, prefetched_count = 0;
                NgramHistory prefetched_entry_history{}, prefetched_exit_history{};
                for (size_t at = input_at; at < job.tokens.size();) {
                    const size_t count = chunk_count_at(at);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    std::optional<uint32_t> part_chunk_mode;
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
                    if (native_prefill_chunks) part_chunk_mode = uint32_t(*native_prefill_chunks);
#endif
                    if (branches) branches->record_partition(uint32_t(count), part_chunk_mode);
#endif
                    const auto chunk_tokens = std::span(job.tokens).subspan(at, count);
                    const bool chunk_emit_logits = !job.final_logits || (job.prefill_final && at + count == job.tokens.size());
                    const size_t next_at = at + count;
                    const size_t next_count = chunk_count_at(next_at);
                    if (ple_overlap && ple_host_cap >= count && (count > 1 || next_count > 1)) {
                        own_memory_guard();
                        const auto base = stack.position();
                        require(sampler.matches(base), "invalid sampling prefix before known-token input; reset required");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                        if (mtp && base) std::copy(stack.last_final_hc_residual().begin(), stack.last_final_hc_residual().end(), mtp_previous_hc.begin());
#endif
                        NgramHistory current_exit_history{};
                        if (has_prefetched_chunk && prefetched_at == at && prefetched_count == count) {
                            require(prefetched_entry_history == stack.committed_history(),
                                "prefetched chunk entry history mismatch");
                            std::swap(known_rows, ple_prefetch_rows);
                            std::swap(known_embeddings, ple_prefetch_embeddings);
                            std::swap(ple_gather_scratch.pairs, ple_prefetch_scratch.pairs);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                            if (ple_async) cache.seed_from_dedup(ple_gather_scratch, std::span(known_embeddings).first(count * 2560));
#endif
                            stateless_prefill_ple_rows += count * 16;
                            ++ple_overlap_chunks;
                            current_exit_history = prefetched_exit_history;
                            has_prefetched_chunk = false;
                        } else if (count > 1) {
                            const auto entry_history = stack.committed_history();
                            current_exit_history = gather_known_dedup(chunk_tokens, entry_history, ple_chunk_ids,
                                ple_gather_scratch, known_rows, known_embeddings, true);
                            require(entry_history == stack.committed_history(),
                                "prefill window history assertion failed: entry_history == stack_committed_history");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                            if (ple_async) cache.seed_from_dedup(ple_gather_scratch, std::span(known_embeddings).first(count * 2560));
#endif
                            stateless_prefill_ple_rows += count * 16;
                        } else {
                            const auto lookups = stack.lookups(chunk_tokens);
                            for (size_t i = 0; i < count; ++i) {
                                cache.gather(lookups[i].rows, std::span(known_embeddings).subspan(i * 2560, 2560));
                                dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(chunk_tokens[i]) * row_bytes, row_bytes),
                                    std::span(known_rows).subspan(i * 2560, 2560));
                            }
                            if (next_count > 1 && ple_host_cap >= next_count) {
                                current_exit_history = ngram_step(spec, effective_policy.ngram, stack.committed_history(),
                                    chunk_tokens[0], std::span(ple_chunk_ids).first(16));
                            }
                        }
                        if (next_count > 1 && ple_host_cap >= next_count) {
                            const auto next_tokens = std::span(job.tokens).subspan(next_at, next_count);
                            std::exception_ptr prefetch_error = nullptr;
                            NgramHistory next_exit_history{};
                            {
                                ScopedJoinThread prefetch_worker{std::thread([&]() {
                                    try {
                                        next_exit_history = gather_known_dedup(next_tokens, current_exit_history,
                                            ple_prefetch_ids, ple_prefetch_scratch, ple_prefetch_rows, ple_prefetch_embeddings, false);
                                    } catch (...) {
                                        prefetch_error = std::current_exception();
                                    }
                                })};
                                advance_known_prepared(chunk_tokens, base, chunk_emit_logits, job.tokens.size() - next_at);
                                prefetch_worker.join();
                            }
                            if (prefetch_error) std::rethrow_exception(prefetch_error);
                            has_prefetched_chunk = true;
                            prefetched_at = next_at;
                            prefetched_count = next_count;
                            prefetched_entry_history = current_exit_history;
                            prefetched_exit_history = next_exit_history;
                        } else {
                            advance_known_prepared(chunk_tokens, base, chunk_emit_logits, job.tokens.size() - next_at);
                        }
                    } else {
                        advance_known(chunk_tokens, chunk_emit_logits);
                    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    if (ssd_manager && ssd_manager->enabled() && ssd_token_history.size() == stack.position() - count) {
                        ssd_token_history.insert(ssd_token_history.end(), job.tokens.begin() + at, job.tokens.begin() + at + count);
                    }
#endif
                    if (shared_prefix_cache_enabled && !job.branch && !broker && shared_prefix_cache.active_valid &&
                        shared_prefix_cache.active_tokens.size() == stack.position() - count &&
                        shared_prefix_cache.active_tokens.size() < SharedPrefixCacheState::max_prefix_tokens) {
                        const size_t room = SharedPrefixCacheState::max_prefix_tokens - shared_prefix_cache.active_tokens.size();
                        const size_t append_n = std::min(count, room);
                        shared_prefix_cache.active_tokens.insert(shared_prefix_cache.active_tokens.end(),
                            job.tokens.begin() + at, job.tokens.begin() + at + append_n);
                        if (stack.position() <= SharedPrefixCacheState::max_prefix_tokens &&
                            stack.position() % SharedPrefixCacheState::boundary_tokens == 0 &&
                            shared_prefix_cache.active_tokens.size() == stack.position() &&
                            shared_prefix_lcp_boundary(shared_prefix_cache.active_tokens) >= stack.position()) {
                            const auto cap_started = Clock::now();
                            if (capture_shared_prefix(shared_prefix_cache.active_tokens)) {
                                capture_ms += milliseconds(cap_started);
                            }
                        }
                    }
                    at += count;
                }
            } else for (size_t at = input_at; at < job.tokens.size(); ++at) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                std::optional<uint32_t> part_chunk_mode;
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
                if (native_prefill_chunks) part_chunk_mode = uint32_t(*native_prefill_chunks);
#endif
                if (branches) branches->record_partition(1, part_chunk_mode);
#endif
                advance(job.tokens[at]);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (ssd_manager && ssd_manager->enabled() && ssd_token_history.size() == stack.position() - 1) {
                    ssd_token_history.push_back(job.tokens[at]);
                }
#endif
                if (shared_prefix_cache_enabled && !job.branch && !broker && shared_prefix_cache.active_valid &&
                    shared_prefix_cache.active_tokens.size() == stack.position() - 1 &&
                    shared_prefix_cache.active_tokens.size() < SharedPrefixCacheState::max_prefix_tokens) {
                    shared_prefix_cache.active_tokens.push_back(job.tokens[at]);
                }
            }
            if (broker && input_at < job.tokens.size()) broker->input();
            const bool prefill_packet_complete = !job.known_prefill
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                || !job.final_logits || job.prefill_final
#endif
                ;
            if (branches && !broker && input_at < job.tokens.size() && prefill_packet_complete
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
                && !verification_cost_manifest
#endif
            ) {
                const auto capture_started = Clock::now();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                const StageAControllerState* ctrl_p = has_speculative ? &mtp_generation.controller_state() : nullptr;
                const uint32_t mtp_rows_val = mtp ? mtp->cache_rows() : 0;
                const auto* anchor_p = mtp_sidecar ? mtp_generation.carry().peek(stack.position()) : nullptr;
                const std::optional<SampledPathToken> anchor_val = anchor_p ? std::optional<SampledPathToken>(*anchor_p) : std::nullopt;
                const uint32_t overflow_val = has_speculative ? mtp_generation.pending_overflow() : 0;
                branches->capture(stack.position(), retained, &sampler, ctrl_p, 1, mtp_rows_val, anchor_val, overflow_val,
                    [&] { return stack.retain_checkpoint(); });
#else
                branches->capture(stack.position(), retained, [&] { return stack.retain_checkpoint(); });
#endif
                capture_ms = milliseconds(capture_started);
            }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!broker && input_at < job.tokens.size() && prefill_packet_complete && ssd_manager && ssd_manager->enabled() &&
                ssd_token_history.size() == stack.position() && stack.position() > last_ssd_spill_prefix) {
                const bool is_system_anchor = !session_generated_since_reset && stack.position() <= 1024;
                const bool allow_ssd_spill = !shared_prefix_cache_enabled || is_system_anchor ||
                    session_generated_since_reset ||
                    shared_prefix_lcp_boundary(ssd_token_history) >= SharedPrefixCacheState::boundary_tokens;
                if (allow_ssd_spill) {
                    const double elapsed_s = std::chrono::duration<double>(Clock::now() - last_ssd_spill_time).count();
                    const uint32_t newly_committed = stack.position() - last_ssd_spill_prefix;
                    if (ssd_manager->should_coalesce(elapsed_s, newly_committed)) {
                        const auto spill_started = Clock::now();
                        if (spill_to_ssd(ssd_token_history, is_system_anchor)) {
                            capture_ms += milliseconds(spill_started);
                        }
                    }
                }
            }
#endif
            const auto prefill_ms = milliseconds(started); Json::Array generated; double batch_wall = 0, batch_gpu = 0;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
            const uint64_t prefill_dispatches = gpu.lifetime_dispatches() - dispatches_before_prefill;
            if (!job.tokens.empty()) {
                last_prefill_wall_ms = prefill_ms;
                last_prefill_dispatches = prefill_dispatches;
                last_prefill_top10_mixture_stats = gpu.prefill_top10_mixture_telemetry();
            } else if (job.reset) {
                last_prefill_wall_ms = 0.0;
                last_prefill_dispatches = 0;
                last_prefill_top10_mixture_stats = {};
            }
            const auto prefill_top10_mixture_stats = !job.tokens.empty() ? gpu.prefill_top10_mixture_telemetry() : last_prefill_top10_mixture_stats;
            const uint64_t effective_prefill_dispatches = !job.tokens.empty() ? prefill_dispatches : last_prefill_dispatches;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
            if (verification_cost_manifest) run_verification_cost(verification_cost_manifest->prefixes[verification_cost_prefix++]);
#endif
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
            if (logit_trace && !job.generate && retained.matches(stack.position())
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                && (!task4_profile.has_value() || stack.position() == 4096 || stack.position() == 4097)
#endif
                ) logit_trace->append(stack.position(), retained.get(stack.position()).logits);
#endif
            if (lookup_active) generated.reserve(job.generate);
            uint64_t lookup_rounds = 0, lookup_verified_rounds = 0, lookup_recovery_rounds = 0, lookup_proposed = 0, lookup_accepted = 0, lookup_ordinary = 0;
            double lookup_round_wall = 0;
            std::optional<uint32_t> stop_id = job.branch && input_at == job.tokens.size() && !job.generate
                ? retained.stop_id() : std::optional<uint32_t>{};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            std::optional<MeasuredRoundCompletion> measured_packet_tail;
#endif
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            CarrierPacketTail carrier_packet_tail{generated, stop_id, job.generate, carrier_single_action};
#endif
            MtpAnchorCarry mtp_packet_anchor;
            auto& mtp_anchor = mtp_request_carry ? mtp_generation.carry() : mtp_packet_anchor;
            if (mtp_request_carry && mtp_generation.pending_overflow() > 0) {
                while (mtp_generation.pending_overflow() > 0 && generated.size() < job.generate) {
                    const auto token = mtp_generation.pop_overflow();
                    generated.push_back(Json::integer(token.id));
                    std::cout << decorate(Json(Json::Object{{"event", Json("token")},
                        {"id", Json::integer(token.id)},
                        {"source_logit", number(token.score)},
                        {"committed_prefix", Json::integer(token.committed_prefix)},
                        {"token_prefix", Json::integer(token.token_prefix)},
                        {"round_committed_prefix", Json::integer(token.committed_prefix)},
                        {"entropy_words", Json::integer(sampler.random().words_consumed())},
                        {"gpu_profile", Json()}})).dump() << std::endl;
                    if (std::find(job.stops.begin(), job.stops.end(), token.id) != job.stops.end()) {
                        stop_id = token.id;
                        retained.stopped(token.id);
                        mtp_generation.mark_stop_drained();
                        break;
                    }
                }
            }
            const auto current_allow_topk = [&]() noexcept -> bool {
                return stack.sampled_logit_topk() && sampling_topk_eligible(sampler.options().policy, shape.vocabulary);
            };
            const auto materialize_verified_logits_if_needed = [&](const StackSampledVerificationResult& verified,
                                                                   uint32_t round_emitted_count,
                                                                   bool round_stopped,
                                                                   bool has_pending_anchor) {
                if (!verified.logits.empty()) {
                    std::copy(verified.logits.begin(), verified.logits.end(), mtp_spare->logits.begin());
                    return;
                }
                const uint32_t rem_packet = job.generate - uint32_t(generated.size());
                const bool need_row = (round_emitted_count >= rem_packet) || round_stopped || !has_pending_anchor
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                    || carrier_single_action
#endif
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
                    || bool(logit_trace)
#endif
                    ;
                if (need_row) {
                    stack.read_verified_committed_logits(verified.committed, mtp_spare->logits);
                }
            };
#endif
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            const uint32_t carrier_actions_before_packet = gpu.carrier_diagnostic_action();
            require(!carrier_single_action || !job.generate || (generation_command && generation_command->begin && generation_command->v2),
                "single-action timing generation requires a fresh full v2 begin");
#endif
            for (uint32_t i = uint32_t(generated.size()); i < job.generate; ++i) {
                if (stop_id) break;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                if (carrier_single_action && carrier_single_action_yield(carrier_actions_before_packet, gpu.carrier_diagnostic_action())) break;
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
#include "dflash2_round.inc"
#include "mtp_depth2_round.inc"
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (mtp && mtp_lookahead_anchor && job.generate - i == 1) {
                    // This one remaining visible token must be materialised
                    // without evaluating a draft beyond the reserved context.
                    const auto tail_started = Clock::now(); const auto prime_before = mtp_prime_gpu_ms;
                    const auto base = stack.position(); const auto* pending = mtp_anchor.peek(base);
                    auto random = sampler.random();
                    const auto& logits = retained.get(base).logits;
                    const auto id = pending ? pending->id : sample(sampling_distribution(logits, sampler.options().policy), random);
                    const auto score = pending ? pending->source_logit : logits[id];
                    advance(id); sampler.publish(std::move(random));
                    if (pending) ++mtp_carried_anchors;
                    static_cast<void>(mtp_anchor.clear());
                    generated.push_back(Json::integer(id));
                    const auto& committed = retained.get(stack.position());
                    batch_wall += milliseconds(tail_started); batch_gpu += committed.batch.gpu_ms + mtp_prime_gpu_ms - prime_before;
                    if (std::find(job.stops.begin(), job.stops.end(), id) != job.stops.end()) { stop_id = id; retained.stopped(id); }
                    std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)},
                        {"source_logit", number(score)}, {"committed_prefix", Json::integer(stack.position())},
                        {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl;
                    break;
                }
                if (mtp && job.generate - i >= (mtp_lookahead_anchor ? 2u : 3u)) {
                    const auto round_started = Clock::now();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                    const Json round_counters_before(Json::Object{{"dispatches",Json::integer(gpu.lifetime_dispatches())},
                        {"command_buffers",Json::integer(gpu.lifetime_command_buffers())},{"barriers",Json::integer(gpu.lifetime_barriers())}});
#endif
                    const auto prime_gpu_before = mtp_prime_gpu_ms;
                    const auto base = stack.position();
                    require(sampler.matches(base) && mtp->next_absolute_position() == base,
                        "native MTP head/target/entropy frontier mismatch");
                    auto random = sampler.random();
                    const auto& current = retained.get(base);
                    const auto* pending = mtp_lookahead_anchor ? mtp_anchor.peek(base) : nullptr;
                    const auto anchor = pending ? pending->id : sample(sampling_distribution(current.logits, sampler.options().policy), random);
                    const auto anchor_score = pending ? pending->source_logit : current.logits[anchor];
                    if (std::find(job.stops.begin(), job.stops.end(), anchor) == job.stops.end()) {
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
                        TargetMtpProfileRound profile_round(gpu, base);
                        gpu.set_mtp_profile_phase(MtpProfilePhase::draft);
#endif
                        // All proposal entropy precedes target publication. V1
                        // materialises the terminal here; opt-in V2 carries it
                        // invisibly as the next anchor, without drawing it again.
                        require(mtp_spare && mtp_spare->logits.size() == shape.vocabulary,
                            "native MTP retained-row storage unavailable");
                        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(anchor) * row_bytes, row_bytes),
                            std::span(mtp_candidate_rows).first(2560));
                        PleAsyncScopeGuard ple_guard;
                        NgramHistory ple_hist{};
                        if (ple_async) {
                            ple_hist = stack.committed_history();
                            ple_guard.start(*ple_async, ple_hist, std::span(&anchor, 1), std::span(mtp_candidate_memory).first(2560));
                        }
                        const auto head_started = Clock::now();
                        double draft_gpu = 0;
                        std::optional<MtpDraftDraw> draft_opt;
                        if (mtp_fast_round) {
                            const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
                            const auto head = hc_gpu.owner
                                ? mtp->advance_fast(hc_gpu, std::span(mtp_candidate_rows).first(2560), base)
                                : mtp->advance_fast_host(stack.last_final_hc_residual(), std::span(mtp_candidate_rows).first(2560), base);
                            draft_gpu = head.batch.gpu_ms;
                            draft_opt.emplace(draw_mtp_depth1_topk(head.topk, sampler.options().policy, random, shape.vocabulary));
                        } else {
                            const auto residual_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
                            const auto head = residual_gpu.owner
                                ? mtp->advance(residual_gpu, std::span(mtp_candidate_rows).first(2560), base)
                                : mtp->advance(stack.last_final_hc_residual(), std::span(mtp_candidate_rows).first(2560), base);
                            draft_gpu = head.batch.gpu_ms;
                            draft_opt.emplace(draw_mtp_depth1(head.logits, sampler.options().policy, random, shape.vocabulary));
                        }
                        const auto draft_wall = milliseconds(head_started);
                        const auto& draft = *draft_opt;
                        ++mtp_head_calls; ++mtp_proposed; mtp_head_gpu_ms += draft_gpu; mtp_head_wall_ms += draft_wall;
                        const std::array<uint32_t, 2> candidates{anchor, draft.token};
                        if (ple_async) {
                            ple_hist = ple_guard.finish();
                            ple_guard.start(*ple_async, ple_hist, std::span(&draft.token, 1),
                                std::span(mtp_candidate_memory).subspan(2560, 2560));
                            dequantize(token_table.type,
                                files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[1]) * row_bytes, row_bytes),
                                std::span(mtp_candidate_rows).subspan(2560, 2560));
                            static_cast<void>(ple_guard.finish());
                        } else {
                            const auto staged = stack.sampled_verification_lookups(candidates);
                            for (uint32_t t = 0; t < 2; ++t) {
                                if (t) dequantize(token_table.type,
                                    files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                                    std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                                cache.gather(staged[t].rows, std::span(mtp_candidate_memory).subspan(uint64_t(t) * 2560, 2560));
                            }
                        }
                        TargetMtpSelection selector{base, anchor, draft, sampler.options().policy, job.stops, {}, {}};
                        const auto verified = stack.verify_sampled2(candidates, mtp_candidate_rows, mtp_candidate_memory,
                            base, {TargetMtpSelection::choose, &selector, current_allow_topk()});
                        require(selector.decision && verified.committed == 1 + selector.decision->accepted,
                            "native MTP target/decision commit mismatch");
                        auto& decision = *selector.decision;
                        materialize_verified_logits_if_needed(verified,
                            mtp_lookahead_anchor ? verified.committed : (1 + decision.count),
                            mtp_lookahead_anchor ? selector.anchor_publication->stopped : decision.stopped,
                            mtp_lookahead_anchor && selector.anchor_publication->pending.has_value());
                        mtp_spare->position = base;
                        mtp_spare->batch = verified.verification;
                        mtp_spare->batch.gpu_ms += verified.recovery.gpu_ms;
                        mtp_spare->batch.wall_ms += verified.recovery.wall_ms;
                        retained.exchange(mtp_spare, stack.position(), {});
                        for (uint32_t t = 1; t <= verified.committed; ++t)
                            require(sampler.committed_input(base + t), "native MTP consumed target prefix invalidated entropy");
                        if (decision.accepted) {
                            // The head already consumed the anchor. Catch up the
                            // accepted draft from authoritative HC after anchor.
                            if (mtp_fast_round) {
                                const auto prime_started = Clock::now();
                                const auto prime_hc_gpu = gpu.slice(verified.evaluated.final_hc_gpu_rows, 0, 10240 * sizeof(float));
                                mtp->defer_prime_batch(prime_hc_gpu, std::span(mtp_candidate_rows).subspan(2560, 2560), base + 1);
                                ++mtp_prime_calls;
                                mtp_prime_wall_ms += milliseconds(prime_started);
                            } else if (stack.has_last_final_hc_gpu()) {
                                const auto hc_gpu = stack.last_final_hc_gpu_rows();
                                const auto prime_started = Clock::now();
                                const auto batch = mtp->prime_pair(gpu.slice(hc_gpu, 0, 10240ull * sizeof(float)),
                                    std::span(mtp_candidate_rows).subspan(2560, 2560), base + 1);
                                ++mtp_prime_calls; mtp_prime_gpu_ms += batch.gpu_ms; mtp_prime_wall_ms += milliseconds(prime_started);
                            } else {
                                mtp_prime(verified.evaluated.final_hc_rows.first(10240),
                                    std::span(mtp_candidate_rows).subspan(2560, 2560), base + 1);
                            }
                        }
                        const auto materialize_started = Clock::now();
                        double materialize_gpu = 0;
                        const auto& terminal = decision.tokens[decision.count - 1];
                        // An accepted EOS is already consumed. In V1 every other
                        // terminal is materialised now; in V2 it becomes the next
                        // round's anchor and is emitted only after consumption.
                        if (!mtp_lookahead_anchor && !terminal.accepted_proposal) {
                            advance(terminal.id);
                            materialize_gpu = retained.get(stack.position()).batch.gpu_ms;
                        }
                        const auto materialize_wall = milliseconds(materialize_started);
                        sampler.publish(std::move(decision.random_after));
                        if (mtp_lookahead_anchor) {
                            require(selector.anchor_publication && selector.anchor_publication->prefix == stack.position(),
                                "native MTP lookahead publication differs from consumed prefix");
                            if (pending) ++mtp_carried_anchors;
                            mtp_anchor.publish(*selector.anchor_publication);
                        }
                        ++mtp_rounds; mtp_accepted += decision.accepted; mtp_recoveries += !decision.accepted;
                        const auto count = mtp_lookahead_anchor ? verified.committed : 1 + decision.count;
                        require(stack.position() == base + count && sampler.matches(stack.position()),
                            "native MTP final visible/physical frontier mismatch");
                        if (mtp_lookahead_anchor ? selector.anchor_publication->stopped : decision.stopped) {
                            stop_id = terminal.id; retained.stopped(terminal.id);
                        }
                        const auto round_wall = milliseconds(round_started);
                        batch_wall += round_wall;
                        batch_gpu += verified.verification.gpu_ms + verified.recovery.gpu_ms + materialize_gpu + draft_gpu +
                            (mtp_prime_gpu_ms - prime_gpu_before);
                        std::cout << decorate(Json(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                            {"gpu_counters_before",round_counters_before},
#endif
                            {"committed_prefix", Json::integer(stack.position())}, {"count", Json::integer(count)},
                            {"accepted", Json::integer(decision.accepted)}, {"draft_wall_ms", number(draft_wall)},
                            {"draft_gpu_ms", number(draft_gpu)}, {"verification_wall_ms", number(verified.verification.wall_ms)},
                            {"verification_gpu_ms", number(verified.verification.gpu_ms)}, {"recovery_wall_ms", number(verified.recovery.wall_ms)},
                            {"recovery_gpu_ms", number(verified.recovery.gpu_ms)}, {"materialization_wall_ms", number(materialize_wall)},
                            {"lookahead_anchor", Json(mtp_lookahead_anchor)}, {"used_carried_anchor", Json(bool(pending))},
                            {"pending_anchor", Json(mtp_lookahead_anchor && mtp_anchor.peek(stack.position()))},
                            {"round_wall_ms", number(round_wall)}, {"timings_qualified", Json(false)}})).dump() << std::endl;
                        for (uint32_t t = 0; t < count; ++t) {
                            const auto id = t ? decision.tokens[t - 1].id : anchor;
                            const auto score = t ? decision.tokens[t - 1].source_logit : anchor_score;
                            generated.push_back(Json::integer(id));
                            std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)},
                                {"source_logit", number(score)}, {"committed_prefix", Json::integer(stack.position())},
                                {"token_prefix", Json::integer(base + t + 1)},
                                {"round_committed_prefix", Json::integer(stack.position())},
                                {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl;
                        }
                        i += count - 1;
                        if (stop_id) break;
                        continue;
                    }
                    // Ordinary terminal path consumes exactly the anchor draw.
                    advance(anchor); sampler.publish(std::move(random));
                    if (pending) ++mtp_carried_anchors;
                    static_cast<void>(mtp_anchor.clear());
                    stop_id = anchor; retained.stopped(anchor); generated.push_back(Json::integer(anchor));
                    const auto& committed = retained.get(stack.position());
                    batch_wall += mtp_lookahead_anchor ? milliseconds(round_started) : committed.batch.wall_ms;
                    batch_gpu += committed.batch.gpu_ms + (mtp_lookahead_anchor ? mtp_prime_gpu_ms - prime_gpu_before : 0);
                    std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(anchor)},
                        {"source_logit", number(anchor_score)}, {"committed_prefix", Json::integer(stack.position())},
                        {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl;
                    break;
                }
#endif
                if (lookup_active && sampler.options().policy.temperature == 0) {
                    try {
                        const auto round_started = Clock::now();
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
                        const auto stage_round_before = gpu.qsa_stage_v8_calls();
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
                        const auto channel_round_before = gpu.qsa_channel_calls();
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
                        const auto hc_round_before = gpu.hc_exact_routes();
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
                        const auto wide_round_before = gpu.wide_q8_exact_routes();
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
                        const auto wide_many_round_before = gpu.wide_q8_many_routes();
#endif
                        double gather_ms = 0, verify_ms = 0, fallback_ms = 0;
                        BatchResult verification{}, recovery{}, fallback{};
                        const auto round = lookup_round(*lookup, retained, sampler, job.generate - i, job.stops,
                            [&] { return stack.position(); },
                            [&](std::span<const uint32_t> tokens, uint32_t base, std::span<const float> logits,
                                uint32_t remaining, std::span<const uint32_t> stops) {
                                const auto gather_started = Clock::now(); own_memory_guard();
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                                if (verification_max_tokens == 8) {
                                    const auto staged = stack.verification_lookups8(tokens, remaining, stops);
                                    require(staged.count >= 2, "eligible verification8 has no reserved active tile");
                                    const auto active = tokens.first(staged.count);
                                    for (size_t n = 0; n < staged.count; ++n) {
                                        cache.gather(staged.rows[n].rows, std::span(lookup->embeddings).subspan(n * 2560, 2560));
                                        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(active[n]) * row_bytes, row_bytes),
                                            std::span(lookup->rows).subspan(n * 2560, 2560));
                                    }
                                    gather_ms = milliseconds(gather_started);
                                    const auto result = stack.verify_greedy8(active, std::span(lookup->rows).first(uint64_t(staged.count) * 2560),
                                        std::span(lookup->embeddings).first(uint64_t(staged.count) * 2560), base, logits, remaining, stops);
                                    verify_ms = result.wall_ms; verification = result.verification; recovery = result.recovery; return result;
                                }
#endif
                                const auto staged = stack.verification_lookups(tokens);
                                for (size_t n = 0; n < 2; ++n) {
                                    cache.gather(staged[n].rows, std::span(lookup->embeddings).subspan(n * 2560, 2560));
                                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(tokens[n]) * row_bytes, row_bytes),
                                        std::span(lookup->rows).subspan(n * 2560, 2560));
                                }
                                gather_ms = milliseconds(gather_started);
                                const auto result = stack.verify_greedy2(tokens, lookup->rows, lookup->embeddings, base, logits, remaining, stops);
                                verify_ms = result.wall_ms; verification = result.verification; recovery = result.recovery;
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                                StackVerificationResult8 wrapped{result.accepted, result.position,
                                    result.verification.dispatches ? 2u : 0u, result.verification.dispatches ? 2u : 0u, result.logits,
                                    result.verification, result.recovery, result.wall_ms};
                                std::copy(result.source_logits.begin(), result.source_logits.end(), wrapped.source_logits.begin()); return wrapped;
#else
                                return result;
#endif
                            },
                            [&] {
                                const auto fallback_started = Clock::now(); const auto& result = retained.get(stack.position()); float score = 0;
                                const auto id = sampled_step(sampler, result.logits, stack.position(), [&](uint32_t token) {
                                    score = result.logits[token]; advance(token);
                                });
                                fallback = retained.get(stack.position()).batch; fallback_ms = milliseconds(fallback_started);
                                return LookupRoundToken{id, score, false};
                            },
                            [&](uint32_t prefix, std::span<const uint32_t> tokens) {
                                if (branches) require(branches->committed(prefix, tokens), "verified lookup invalidated branch ancestry");
                            });
                        const auto wall_ms = milliseconds(round_started);
                        ++lookup_rounds; lookup_verified_rounds += round.verified; lookup_proposed += round.proposed;
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                        lookup_recovery_rounds += bool(recovery.dispatches);
#else
                        lookup_recovery_rounds += round.accepted == 1;
#endif
                        lookup_accepted += round.accepted; lookup_ordinary += round.count - round.accepted; lookup_round_wall += wall_ms;
                        const auto round_batch_wall = verification.wall_ms + recovery.wall_ms + fallback.wall_ms;
                        const auto round_batch_gpu = verification.gpu_ms + recovery.gpu_ms + fallback.gpu_ms;
                        batch_wall += round_batch_wall; batch_gpu += round_batch_gpu;
                        if (broker) broker->generated();
                        for (uint32_t n = 0; n < round.count; ++n) {
                            const auto id = round.tokens[n].id; generated.push_back(Json::integer(id));
                            if (std::find(job.stops.begin(), job.stops.end(), id) != job.stops.end()) {
                                require(n + 1 == round.count, "lookup committed beyond first terminal token"); stop_id = id; retained.stopped(id);
                            }
                        }
                        std::cout << decorate(Json(Json::Object{{"event", Json("generation_round")}, {"round_id", Json::integer(lookup_rounds)},
                            {"round_base", Json::integer(round.base)}, {"committed_prefix", Json::integer(round.committed)},
                            {"proposed_tokens", Json::integer(round.proposed)},
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                            {"verified_tokens", Json::integer(round.evaluated)}, {"verification_max_tokens", Json::integer(
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            mtp_depth == 7 ? 8 : mtp_depth == 3 ? 4 : mtp_depth == 2 ? 3 :
#endif
            verification_max_tokens)},
                            {"verification_head_dispatches", Json::integer(round.head_dispatches)},
#else
                            {"verified_tokens", Json::integer(round.verified ? 2 : 0)},
#endif
                            {"accepted_verified_tokens", Json::integer(round.accepted)}, {"ordinary_tokens", Json::integer(round.count - round.accepted)},
                            {"useful_tokens", Json::integer(round.count)}, {"round_wall_ms", number(wall_ms)},
                            {"lookup_and_publication_ms", number(std::max(0.0, wall_ms - gather_ms - verify_ms - fallback_ms))},
                            {"gather_wall_ms", number(gather_ms)}, {"verification_including_recovery_wall_ms", number(verify_ms)},
                            {"recovery_batch_wall_ms", number(recovery.wall_ms)}, {"fallback_wall_ms", number(fallback_ms)},
                            {"batch_wall_ms", number(round_batch_wall)}, {"batch_gpu_ms", number(round_batch_gpu)},
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
                            {"qsa_stage_v8_round_calls", Json::integer(gpu.qsa_stage_v8_calls() - stage_round_before)},
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
                            {"qsa_channel_round_calls", Json::integer(gpu.qsa_channel_calls() - channel_round_before)},
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
                            {"hc_exact_round_routes", hc_routes_json(hc_routes_delta(gpu.hc_exact_routes(), hc_round_before))},
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
                            {"wide_q8_exact_round_routes", wide_routes_json(wide_routes_delta(gpu.wide_q8_exact_routes(), wide_round_before))},
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
                            {"wide_q8_many_round_routes", wide_many_routes_json(wide_many_routes_delta(gpu.wide_q8_many_routes(), wide_many_round_before))},
#endif
                            {"dispatches", Json::integer(verification.dispatches + recovery.dispatches + fallback.dispatches)},
                            {"capture_executes", Json::integer(verification.capture_executes + recovery.capture_executes + fallback.capture_executes)},
                            {"useful_tokens_per_second", number(wall_ms > 0 ? round.count * 1000.0 / wall_ms : 0)},
                            {"accepted_tokens_per_second", number(wall_ms > 0 ? round.accepted * 1000.0 / wall_ms : 0)},
                            {"timings_qualified", Json(false)}, {"gpu_profile", Json()}})).dump() << std::endl;
                        for (uint32_t n = 0; n < round.count; ++n)
                            std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(round.tokens[n].id)},
                                {"source_logit", number(round.tokens[n].source_logit)}, {"committed_prefix", Json::integer(round.committed)},
                                {"token_prefix", Json::integer(round.base + n + 1)}, {"round_id", Json::integer(lookup_rounds)},
                                {"round_base", Json::integer(round.base)}, {"round_committed_prefix", Json::integer(round.committed)},
                                {"verified", Json(round.tokens[n].verified)},
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                                {"batched_commit", Json(round.accepted >= 2)},
#else
                                {"batched_commit", Json(round.accepted == 2)},
#endif
                                {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl;
                        i += round.count - 1; if (stop_id) break; continue;
                    } catch (...) {
                        lookup->invalidate(); retained.clear(); sampler.invalidate(); if (branches) branches->invalidate(); throw;
                    }
                }
                const auto& result = retained.get(stack.position());
#ifdef TOKSPUR_PRIVATE_LOGIT_TRACE
                if (logit_trace) logit_trace->append(stack.position(), result.logits);
#endif
                float score = 0;
                const auto id = sampled_step(sampler, result.logits, stack.position(), [&](uint32_t token) {
                    score = result.logits[token]; advance(token);
                });
                if (broker) broker->generated();
                generated.push_back(Json::integer(id));
                const auto& committed = retained.get(stack.position());
                batch_wall += committed.batch.wall_ms; batch_gpu += committed.batch.gpu_ms;
                if (std::find(job.stops.begin(), job.stops.end(), id) != job.stops.end()) { stop_id = id; retained.stopped(id); }
                std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)}, {"source_logit", number(score)}, {"committed_prefix", Json::integer(stack.position())},
                    {"entropy_words", Json::integer(sampler.random().words_consumed())},
                    {"dispatches", Json::integer(committed.batch.dispatches)}, {"capture_executes", Json::integer(committed.batch.capture_executes)},
                    {"batch_wall_ms", number(committed.batch.wall_ms)}, {"batch_gpu_ms", number(committed.batch.gpu_ms)}, {"gpu_profile", profile_json(gpu.last_profile())}})).dump() << std::endl;
                if (stop_id) break;
            }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (mtp && mtp->deferred_prime_rows()) {
                const auto flush_started = Clock::now();
                const auto flush = mtp->flush_deferred_prime();
                const auto flush_wall = milliseconds(flush_started);
                batch_wall += flush_wall; batch_gpu += flush.gpu_ms;
                mtp_prime_wall_ms += flush_wall; mtp_prime_gpu_ms += flush.gpu_ms;
            }
            if (dflash2 && dflash2->deferred_context()) {
                // Drain committed context before reporting completion so the generation clock includes its final priming work.
                const auto flush_started = Clock::now();
                const auto flush = dflash2->flush_context();
                const auto flush_wall = milliseconds(flush_started);
                batch_wall += flush_wall; batch_gpu += flush.gpu_ms;
                mtp_prime_wall_ms += flush_wall; mtp_prime_gpu_ms += flush.gpu_ms;
            }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            if (measured_packet_tail) measured_packet_tail->publish();
#endif
            if (mtp_request_carry && job.generate) {
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                if (carrier_single_action) {
                    require(carrier_single_action_yield(carrier_actions_before_packet, gpu.carrier_diagnostic_action()),
                        "single-action timing did not execute exactly one action");
                    if (mtp_generation.finish_diagnostic_action(uint32_t(generated.size()), stack.position(), stop_id.has_value())) ++mtp_discarded_anchors;
                } else
#endif
                if (mtp_generation.finish(uint32_t(generated.size()), stack.position(), stop_id.has_value())) ++mtp_discarded_anchors;
            } else if (!mtp_request_carry && mtp_lookahead_anchor && mtp_anchor.clear()) ++mtp_discarded_anchors;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            if (carrier_packet_tail.open) gpu.end_carrier_diagnostic();
#endif
#endif
            files.assert_unchanged();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (mtp_binding) mtp_binding->assert_unchanged();
#endif
            own_memory_guard(); const auto stats = cache.stats();
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
            if (operator_checkpoints.has_value()) {
                const auto rounds = mtp_rounds - operator_rounds_before, accepted = mtp_accepted - operator_accepted_before;
                require(!fixed_operator_screen || (stack.position() == (operator_requests == 0 ? operator_base_prefix : operator_requests == 1 ? operator_base_prefix + 3 : operator_base_prefix + 4) &&
                    rounds == (operator_requests == 1 ? 1 : 0) && (!job.generate || (generated.size() == 3 && rounds + accepted <= generated.size()))),
                    "operator screen workload/round/continuation result differs");
                operator_sink.record(Json(Json::Object{{"schema", Json::integer(1)}, {"event", Json("operator_request_complete")}, {"request", Json::integer(operator_requests++)},
                    {"prefix", Json::integer(stack.position())}, {"rounds", Json::integer(rounds)}, {"accepted", Json::integer(accepted)}, {"generated", Json::integer(generated.size())},
                    {"ordinary_tail_tokens", fixed_operator_screen ? Json::integer(job.generate ? generated.size() - rounds - accepted : 0) : Json()}, {"performance_qualified", Json(false)}}));
            }
#endif
            const auto lookup_report = [&](Json value) {
                if (!lookup) return value;
                auto fields = value.object();
                const auto report = Json::Object{
                    {"prompt_lookup", Json(lookup_active)},
                    {"lookup_rounds", Json::integer(lookup_rounds)}, {"lookup_verified_rounds", Json::integer(lookup_verified_rounds)},
                    {"lookup_recovery_rounds", Json::integer(lookup_recovery_rounds)},
                    {"lookup_proposed_tokens", Json::integer(lookup_proposed)}, {"lookup_accepted_tokens", Json::integer(lookup_accepted)},
                    {"lookup_ordinary_tokens", Json::integer(lookup_ordinary)}, {"lookup_round_wall_ms", number(lookup_round_wall)},
                    {"lookup_useful_tokens_per_second", number(lookup_round_wall > 0 ? (lookup_accepted + lookup_ordinary) * 1000.0 / lookup_round_wall : 0)},
                    {"lookup_accepted_tokens_per_second", number(lookup_round_wall > 0 ? lookup_accepted * 1000.0 / lookup_round_wall : 0)},
                    {"lookup_history_epoch", Json::integer(lookup->epoch())},
                    {"lookup_request_appended_tokens", Json::integer(lookup->ledger().appended_tokens - lookup_ledger_before.appended_tokens)},
                    {"lookup_request_indexed_grams", Json::integer(lookup->ledger().indexed_grams - lookup_ledger_before.indexed_grams)},
                    {"lookup_request_queries", Json::integer(lookup->ledger().queries - lookup_ledger_before.queries)},
                    {"lookup_request_resets", Json::integer(lookup->ledger().resets - lookup_ledger_before.resets)},
                    {"lookup_appended_tokens", Json::integer(lookup->ledger().appended_tokens)},
                    {"lookup_indexed_grams", Json::integer(lookup->ledger().indexed_grams)},
                    {"lookup_truncated_grams", Json::integer(lookup->ledger().truncated_grams)},
                    {"lookup_cleared_grams", Json::integer(lookup->ledger().cleared_grams)},
                    {"lookup_resets", Json::integer(lookup->ledger().resets)}, {"lookup_truncations", Json::integer(lookup->ledger().truncations)},
                    {"lookup_queries", Json::integer(lookup->ledger().queries)},
                    {"lookup_candidates_examined", Json::integer(lookup->ledger().candidates_examined)},
                    {"lookup_token_comparisons", Json::integer(lookup->ledger().token_comparisons)},
                    {"lookup_work_limited_queries", Json::integer(lookup->ledger().work_limited_queries)}};
                fields.insert(fields.end(), report.begin(), report.end()); return Json(std::move(fields));
            };
            if (!generated.empty()) session_generated_since_reset = true;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (ssd_manager && ssd_manager->enabled() && !generated.empty() &&
                ssd_token_history.size() + generated.size() == stack.position()) {
                for (const auto& tok : generated) ssd_token_history.push_back(uint32_t(tok.unsigned_integer()));
            }
#endif
            std::cout << decorate(lookup_report(Json(Json::Object{{"event", Json("done")}, {"generated_ids", Json(std::move(generated))}, {"committed_prefix", Json::integer(stack.position())},
                {"status", Json(stop_id ? "stopped" : job.generate ? "length" : "prefilled")}, {"finish_reason", stop_id ? Json("stop") : job.generate ? Json("length") : Json()},
                {"stop_id", stop_id ? Json::integer(*stop_id) : Json()}, {"continuation_allowed", Json(retained.continuable(stack.position()))},
                {"prefill_wall_ms", number(prefill_ms)},
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
                {"last_prefill_wall_ms", number(last_prefill_wall_ms)},
#endif
                {"request_wall_ms", number(milliseconds(started))}, {"generation_batch_wall_ms", number(batch_wall)}, {"generation_batch_gpu_ms", number(batch_gpu)},
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
                {"prefill_batch_top10_enabled", Json(gpu.prefill_batch_top10())},
                {"prefill_batch_mixture_enabled", Json(gpu.prefill_batch_mixture())},
                {"prefill_dispatches", Json::integer(effective_prefill_dispatches)},
                {"prefill_moe_batch_calls", Json::integer(prefill_top10_mixture_stats.moe_batch_calls)},
                {"prefill_moe_batch_rows", Json::integer(prefill_top10_mixture_stats.moe_batch_rows)},
                {"prefill_moe_batch_min_rows", Json::integer(prefill_top10_mixture_stats.moe_batch_min_rows)},
                {"prefill_moe_batch_max_rows", Json::integer(prefill_top10_mixture_stats.moe_batch_max_rows)},
                {"prefill_top10_batched_calls", Json::integer(prefill_top10_mixture_stats.top10_batched_calls)},
                {"prefill_top10_batched_rows", Json::integer(prefill_top10_mixture_stats.top10_batched_rows)},
                {"prefill_top10_scalar_calls", Json::integer(prefill_top10_mixture_stats.top10_scalar_calls)},
                {"prefill_top10_fallbacks", Json::integer(prefill_top10_mixture_stats.top10_fallbacks)},
                {"prefill_mixture_batched_calls", Json::integer(prefill_top10_mixture_stats.mixture_batched_calls)},
                {"prefill_mixture_batched_rows", Json::integer(prefill_top10_mixture_stats.mixture_batched_rows)},
                {"prefill_mixture_scalar_calls", Json::integer(prefill_top10_mixture_stats.mixture_scalar_calls)},
                {"prefill_mixture_fallbacks", Json::integer(prefill_top10_mixture_stats.mixture_fallbacks)},
#endif
                {"prefill_generation_overlap", Json(request_overlap_pipelined ? "16_token_b10_pipelined" : "none")},
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
                {"qsa_stage_v8_prepared", Json(qsa_stage_policy.prepared)}, {"qsa_stage_v8_enabled", Json(qsa_stage_policy.enabled)},
                {"qsa_stage_v8_scratch_logical_bytes", Json::integer(qsa_stage_policy.scratch_logical_bytes)},
                {"qsa_stage_v8_scratch_charged_bytes", Json::integer(qsa_stage_policy.scratch_charged_bytes)},
                {"qsa_stage_v8_calls", Json::integer(gpu.qsa_stage_v8_calls())},
                {"qsa_stage_v8_physical_dispatches", Json::integer(2 * gpu.qsa_stage_v8_calls())},
                {"qsa_stage_v8_extra_dispatches", Json::integer(gpu.qsa_stage_v8_calls())},
                {"qsa_stage_v8_request_calls", Json::integer(gpu.qsa_stage_v8_calls() - stage_request_before)},
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
                {"qsa_channel_request_calls", Json::integer(gpu.qsa_channel_calls() - channel_request_before)},
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
                {"verification_max_tokens", Json::integer(
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            mtp_depth == 7 ? 8 : mtp_depth == 3 ? 4 : mtp_depth == 2 ? 3 :
#endif
            verification_max_tokens)}, {"verification_head_many", Json(verification_max_tokens == 8)},
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
                {"verification8_branch_reuse", Json(verification8_branch_reuse)},
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
                {"hc_exact_mode", Json(hc_exact_name(hc_exact))}, {"hc_exact_rows_per_group", Json::integer(1)}, {"hc_exact_unroll", Json::integer(16)},
                {"hc_exact_scratch_logical_bytes", Json::integer(0)}, {"hc_exact_scratch_charged_bytes", Json::integer(0)},
                {"hc_exact_routes", hc_routes_json(gpu.hc_exact_routes())},
                {"hc_exact_request_routes", hc_routes_json(hc_routes_delta(gpu.hc_exact_routes(), hc_request_before))},
                {"hc_exact_final_logit_sha256", retained.matches(stack.position()) ? Json(hc_exact_logits_hash(retained.get(stack.position()).logits)) : Json()},
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
                {"q8_three_rows_request_routes",q8_three_routes_json(q8_three_routes_delta(gpu.q8_three_rows_routes(),q8_three_request_before))},
#endif

#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
                {"wide_q8_exact_enabled", Json(wide_q8_exact)}, {"wide_q8_exact_rows_per_group", Json::integer(4)}, {"wide_q8_exact_unroll", Json::integer(16)},
                {"wide_q8_exact_scratch_logical_bytes", Json::integer(0)}, {"wide_q8_exact_scratch_charged_bytes", Json::integer(0)},
                {"wide_q8_exact_routes", wide_routes_json(gpu.wide_q8_exact_routes())},
                {"wide_q8_exact_request_routes", wide_routes_json(wide_routes_delta(gpu.wide_q8_exact_routes(), wide_request_before))},
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
                {"wide_q8_many_enabled", Json(wide_q8_many)}, {"wide_q8_many_rows_per_group", Json::integer(4)}, {"wide_q8_many_unroll", Json::integer(4)},
                {"wide_q8_many_scratch_logical_bytes", Json::integer(0)}, {"wide_q8_many_scratch_charged_bytes", Json::integer(0)},
                {"wide_q8_many_routes", wide_many_routes_json(gpu.wide_q8_many_routes())},
                {"wide_q8_many_request_routes", wide_many_routes_json(wide_many_routes_delta(gpu.wide_q8_many_routes(), wide_many_request_before))},
#endif
                {"checkpoint_reuse", Json(job.branch ? ((selected || ssd_restored_this_request) ? "restored" : "cold_miss") : (ssd_restored_this_request ? "ssd_restored" : "off"))},
                {"checkpoint_ssd_restored", Json(ssd_restored_this_request)},
                {"checkpoint_restored_tokens", Json::integer(input_at)}, {"replayed_input_tokens", Json::integer(job.tokens.size() - input_at)},
                {"checkpoint_capture_wall_ms", number(capture_ms)}, {"checkpoint_restore_wall_ms", number(restore_ms)},
                {"checkpoint_retained", Json::integer(branches ? branches->retained_count() : 0)},
                {"checkpoint_slots", Json::integer(checkpoint_slots)}, {"checkpoint_combined_budget_bytes", Json::integer(checkpoint_budget)},
                {"checkpoint_host_bytes", Json::integer(branches ? branches->charged_bytes() : 0)},
                {"checkpoint_store_host_bytes", Json::integer(stack.checkpoint_memory().host_metadata_bytes)},
                {"checkpoint_gpu_bytes", Json::integer(stack.checkpoint_memory().gpu_charged_bytes)},
                {"final_logit_top8", retained.matches(stack.position()) ? top_logits(retained.get(stack.position()).logits) : Json()}, {"cache_hits", Json::integer(stats.hits)}, {"cache_misses", Json::integer(stats.misses)}, {"cache_evictions", Json::integer(stats.evictions)},
                {"small_gemv_column_splits", Json::integer(small_gemv_splits)},
                {"hc_q8_splits", Json::integer(hc_q8.splits)},
                {"hc_q8_scratch_logical_bytes", Json::integer(hc_q8.scratch_logical_bytes)},
                {"hc_q8_scratch_charged_bytes", Json::integer(hc_q8.scratch_charged_bytes)},
                {"hc_q8_numerical_tier", Json(hc_q8_numerical_tier(hc_q8.splits))},
                {"known_token_capacity", Json::integer(stack.known_token_capacity())},
                {"known_residual_reuse", Json(stack.known_residual_reuse())},
                {"known_token_prefill", Json(job.known_prefill)},
                {"stack_execution", Json(execution_name(stack.execution()))}, {"recurrent_replay_captures", Json::integer(stack.replay_capture_count())},
                {"replay_prepared", Json(stack.replay_prepared())},
                {"capture_resource_bytes", Json::integer(gpu.memory().capture_resource_bytes)},
                {"capture_binding_aliases", Json::integer(gpu.memory().capture_binding_aliases)},
                {"capture_binding_alias_span_bytes", Json::integer(gpu.memory().capture_binding_alias_span_bytes)},
                {"capture_binding_alias_resource_bytes", Json::integer(gpu.memory().capture_binding_alias_resource_bytes)},
                {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)}, {"peak_metal_buffer_bytes", Json::integer(gpu.memory().peak_bytes)},
                {"gpu_allocation_count", Json::integer(gpu.memory().allocations)}, {"gpu_buffer_limit_bytes", Json::integer(gpu.memory().budget)},
                {"live_captures", Json::integer(gpu.memory().live_captures)}, {"imported_weight_backing_bytes", Json::integer(gpu.memory().imported_backing_bytes)},
                {"native_limit_bytes", Json::integer(native_limit)}, {"profile_counter_payload_bytes", Json::integer(gpu.memory().profile_counter_payload_bytes)},
                {"mapped_weight_region_limit_bytes", Json::integer(mapped_region_limit)},
                {"block_gemv_formats", Json::integer(block_gemv_formats)},
                {"sampling", sampling_json(sampler, stack.position())},
                {"thermal_state", Json::integer(host_thermal_state())},
                {"ple_cache", Json(Json::Object{{"hits", Json::integer(cache.stats().hits)}, {"misses", Json::integer(cache.stats().misses)}, {"evictions", Json::integer(cache.stats().evictions)}, {"failures", Json::integer(cache.stats().failures)}})},
                {"stateless_prefill_ple_rows", Json::integer(stateless_prefill_ple_rows)},
                {"memory", describe(memory())}, {"gpu_profile", (job.tokens.empty() || ((job.branch || ssd_restored_this_request || shared_prefix_restored_this_request) && input_at == job.tokens.size())) && !job.generate ? Json() : profile_json(gpu.last_profile())}, {"timings_qualified", Json(false)}}))).dump() << std::endl;
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
            if (fixed_operator_screen && operator_requests == 3) break;
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
            if (kernel_search_capture_path && ++kernel_search_requests == 3) break;
#endif
        } catch (const FormatError& e) {
            emit_error(e, false);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            if (carrier_mode != CarrierDiagnosticMode::off || carrier_cost_replay.enabled()) return 1;
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
            if (kernel_search_capture_path) return 1;
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
            if (operator_checkpoints.has_value()) return 1;
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            if (task4_profile.has_value()) return 1;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
            if (verification_cost_manifest) return 1; // A failed probe cannot be retried as a normal request.
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
            if (sampled_cost_manifest) return 1;
#endif
            if (!std::cin) break;
        } catch (const std::exception& e) {
            // Preserve the previous fatal-exit policy for driver/system errors,
            // but report an explicit failed termination before releasing state.
            emit_error(e, true); return 1;
        }
    }
#ifdef TOKSPUR_PRIVATE_VERIFICATION_COST
    require(!verification_cost_manifest || verification_cost_prefix == verification_cost_manifest->prefixes.size(),
        "verification cost input ended before both frozen prefixes completed");
#endif
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
    require(!sampled_cost_manifest, "sampled COST input ended before its single frozen request");
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    require(!task4_profile.has_value() || (stack.position() == 4097 && logit_trace && logit_trace->rows() == 2),
        "Task4 input ended before both frozen endpoint vectors completed");
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    if (operator_checkpoints.has_value()) {
        require(!fixed_operator_screen || (operator_requests == 3 && stack.position() == operator_base_prefix + 4 && mtp_rounds == 1), "operator screen ended before fixed workload completed");
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        require(!route_census_only || (operator_requests>0 && !route_census_failed),"occupancy census request failed or empty");
#endif
        operator_sink.record(Json(Json::Object{{"schema", Json::integer(1)}, {"event", Json("operator_checkpoint_complete")}, {"passed", Json(true)},
            {"requests", Json::integer(operator_requests)}, {"rounds", Json::integer(mtp_rounds)}, {"prefix", Json::integer(stack.position())}, {"performance_qualified", Json(false)}}));
        operator_sink.finish();
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(route_lock_output)route_lock_sink.finish();
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        if (qsa_route_attribution_path) qsa_route_sink.finish();
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (gdn_route_attribution_path) gdn_route_sink.finish();
#endif
    }
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    if (kernel_search_capture_path) { require(kernel_search_requests == 3 && stack.position() == 4097, "kernel capture ended early"); kernel_search_sink.finish(); }
#endif
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    require(carrier_cost_replay.complete(), "controller timing replay ended before all observations");
    if (carrier_mode != CarrierDiagnosticMode::off) gpu.finish_carrier_diagnostic();
#endif
    return 0;
}
} // namespace tokspur
