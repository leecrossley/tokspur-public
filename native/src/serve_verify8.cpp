#include "tokspur/serve_verify8.hpp"
#include "tokspur/gguf.hpp"
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
#include "tokspur/serve.hpp"
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#include "tokspur/attention_math.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <string>

namespace tokspur {
namespace {
void need(bool ok, const char* why) { if (!ok) throw FormatError(why); }
const Json& get(const Json& v, std::string_view k) { const auto* p = v.find(k); if (!p) throw FormatError("private V8 missing field: " + std::string(k)); return *p; }
uint64_t integer(const Json& v, std::string_view k) { return get(v, k).unsigned_integer(); }
double real(const Json& v, std::string_view k) {
    const auto s = get(v, k).number_lexeme(); double n = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), n);
    need(r.ec == std::errc{} && r.ptr == s.data() + s.size() && std::isfinite(n), "private V8 nonfinite number"); return n;
}
void close(double a, double b) { need(std::abs(a - b) <= 1e-6 * std::max({1.0, std::abs(a), std::abs(b)}), "private V8 costs do not reconcile"); }
void keys(const Json& v, std::initializer_list<std::string_view> allowed, bool branch8 = false, bool stage8 = false, bool fast_math = false) {
    for (const auto& [k, ignored] : v.object()) { static_cast<void>(ignored);
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        if (fast_math && k == "qsa_channel_round_calls") continue;
#else
        static_cast<void>(fast_math);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        if (stage8 && k == "qsa_stage_v8_round_calls") continue;
#else
        static_cast<void>(stage8);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (branch8 && (k == "broker_epoch" || k == "command_id" || k == "lease_id" || k == "broker_protocol" || k == "broker_batch_width")) continue;
#else
        static_cast<void>(branch8);
#endif
        need(std::find(allowed.begin(), allowed.end(), k) != allowed.end(), "private V8 unknown event/command field"); }
}
constexpr std::array<const char*, 5> hc{"scalar", "many1", "many2", "many4", "many8"};
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
constexpr std::array<const char*, 7> stage8_fields{"qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes",
    "qsa_stage_v8_scratch_charged_bytes", "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches"};
bool stage8_metadata(const Json& v) {
    const auto* prepared = v.find("qsa_stage_v8_prepared");
    const auto* event = v.find("event"); const bool done = event && event->string() == "done";
    for (const auto& [name, ignored] : v.object()) if (name.starts_with("qsa_stage_v8_")) {
        static_cast<void>(ignored);
        need(prepared && (std::find(stage8_fields.begin(), stage8_fields.end(), name) != stage8_fields.end() ||
            (done && name == "qsa_stage_v8_request_calls")), "private staged-QSA unknown/incomplete metadata family");
    }
    if (!prepared) return false;
    need(prepared->boolean() && integer(v, "qsa_stage_v8_scratch_logical_bytes") == 196896 &&
        integer(v, "qsa_stage_v8_scratch_charged_bytes") == 212992, "private staged-QSA must prepare the exact OFF/ON storage");
    const bool enabled = get(v, "qsa_stage_v8_enabled").boolean(); const auto calls = integer(v, "qsa_stage_v8_calls");
    need(calls % 12 == 0 && calls <= std::numeric_limits<uint64_t>::max() / 2 && integer(v, "qsa_stage_v8_physical_dispatches") == 2 * calls &&
        integer(v, "qsa_stage_v8_extra_dispatches") == calls && (enabled || !calls), "private staged-QSA physical cost identity");
    if (done) static_cast<void>(integer(v, "qsa_stage_v8_request_calls"));
    return true;
}
void stage8_same(const Json& value, const Json& prepared) {
    const bool present = stage8_metadata(prepared);
    need(stage8_metadata(value) == present, "private staged-QSA selector presence changed");
    if (present) for (size_t i = 0; i < 4; ++i)
        need(get(value, stage8_fields[i]).dump() == get(prepared, stage8_fields[i]).dump(), "private staged-QSA immutable policy changed");
}
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
constexpr std::array<const char*, 8> fast_fields{"ple_gate_transform", "ple_gate_transform_schema",
    "qsa_channel_chronological_mode", "qsa_channel_chronological_schema", "qsa_channel_factor_logical_bytes",
    "qsa_channel_factor_charged_bytes", "qsa_channel_calls", "qsa_channel_extra_dispatches"};
bool ple_metadata(const Json& value) {
    const bool present = value.find("ple_gate_transform") || value.find("ple_gate_transform_schema");
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("ple_gate_")) {
        static_cast<void>(ignored);
        need(name == "ple_gate_transform" || name == "ple_gate_transform_schema", "unknown base PLE metadata");
    }
    if (present) need((get(value, "ple_gate_transform").string() == "plain" ||
        get(value, "ple_gate_transform").string() == "signed-root") && integer(value, "ple_gate_transform_schema") == 1,
        "incomplete or invalid base PLE metadata");
    return present;
}
std::optional<bool> fast_metadata(const Json& value) {
    static_cast<void>(ple_metadata(value));
    const auto* mode = value.find("qsa_channel_chronological_mode");
    const auto* event = value.find("event"); const bool done = event && event->string() == "done";
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("qsa_channel_")) {
        static_cast<void>(ignored);
        need(mode && (std::find(fast_fields.begin(), fast_fields.end(), name) != fast_fields.end() ||
            (done && name == "qsa_channel_request_calls")), "private fast-math unknown/incomplete metadata family");
    }
    if (!mode) return std::nullopt;
    need(mode->string() == "off" || mode->string() == "on", "private fast-math channel mode");
    need(stage8_metadata(value) && get(value, "qsa_stage_v8_enabled").boolean() &&
        get(value, "ple_gate_transform").string() == "signed-root" && integer(value, "ple_gate_transform_schema") == 1 &&
        integer(value, "qsa_channel_chronological_schema") == 1 && integer(value, "qsa_channel_factor_logical_bytes") == 393792 &&
        integer(value, "qsa_channel_factor_charged_bytes") == 409600, "private fast-math fixed PLE/channel preparation");
    const bool channel = mode->string() == "on"; const auto calls = integer(value, "qsa_channel_calls");
    need(calls % 12 == 0 && integer(value, "qsa_channel_extra_dispatches") == calls &&
        (channel ? calls == integer(value, "qsa_stage_v8_calls") : calls == 0), "private fast-math physical cost identity");
    if (done) static_cast<void>(integer(value, "qsa_channel_request_calls"));
    return channel;
}
void fast_same(const Json& value, const Json& prepared) {
    const bool base = ple_metadata(prepared);
    need(ple_metadata(value) == base, "base PLE metadata presence changed");
    if (base) for (size_t i = 0; i < 2; ++i)
        need(get(value, fast_fields[i]).dump() == get(prepared, fast_fields[i]).dump(), "base PLE policy changed");
    const auto mode = fast_metadata(prepared);
    need(fast_metadata(value) == mode, "private fast-math selector presence/mode changed");
    if (mode.has_value()) for (size_t i = 0; i < 6; ++i)
        need(get(value, fast_fields[i]).dump() == get(prepared, fast_fields[i]).dump(), "private fast-math immutable policy changed");
}
#endif
void zero_routes(const Json& v, std::string_view name, size_t count) {
    const auto& r = get(v, name); need(r.object().size() == count, "private V8 route shape");
    if (name.starts_with("wide_q8_exact_")) keys(r, {"qkv_scalar", "qkv_many1", "output_scalar", "output_many1"});
    else keys(r, {"qkv_many2", "qkv_many4", "qkv_many8", "output_many2", "output_many4", "output_many8"});
    for (const auto& [k, n] : r.object()) { static_cast<void>(k); need(n.unsigned_integer() == 0, "private V8 inactive routes"); }
}
void fixed(const Json& v, bool branch8 = false) {
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    need(!fast_metadata(v).has_value() || branch8, "private fast HTTP math requires the branch8 wire");
    need(!v.find("licensed_q4_routed_mode") && !v.find("batch_phase_clock_mode"), "private fast HTTP refuses unrelated numerical/telemetry tiers");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    const bool staged = stage8_metadata(v);
    need(!staged || branch8, "private staged HTTP requires the branch8 wire");
#endif
    for (const auto& [k, n] : {std::pair{"small_gemv_column_splits", 16ull}, {"block_gemv_formats", 3ull},
        {"known_token_capacity", 8ull}, {"hc_q8_splits", 0ull}, {"hc_q8_scratch_logical_bytes", 0ull},
        {"hc_q8_scratch_charged_bytes", 0ull}, {"recurrent_replay_captures", 0ull}, {"mapped_weight_region_limit_bytes", 0ull},
        {"capture_resource_bytes", 0ull}, {"capture_binding_aliases", 0ull}, {"capture_binding_alias_span_bytes", 0ull},
        {"capture_binding_alias_resource_bytes", 0ull}, {"live_captures", 0ull}, {"profile_counter_payload_bytes", 0ull},
        {"verification_max_tokens", 8ull},
        {"native_limit_bytes", 96ull << 30}, {"hc_exact_rows_per_group", 1ull}, {"hc_exact_unroll", 16ull},
        {"hc_exact_scratch_logical_bytes", 0ull}, {"hc_exact_scratch_charged_bytes", 0ull},
        {"wide_q8_exact_rows_per_group", 4ull}, {"wide_q8_exact_unroll", 16ull}, {"wide_q8_exact_scratch_logical_bytes", 0ull},
        {"wide_q8_exact_scratch_charged_bytes", 0ull}, {"wide_q8_many_rows_per_group", 4ull}, {"wide_q8_many_unroll", 4ull},
        {"wide_q8_many_scratch_logical_bytes", 0ull}, {"wide_q8_many_scratch_charged_bytes", 0ull}})
        need(integer(v, k) == n, "private V8 fixed policy differs");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    if (branch8) {
        need(integer(v, "checkpoint_slots") && integer(v, "checkpoint_slots") <= 4 &&
            integer(v, "checkpoint_combined_budget_bytes") && integer(v, "checkpoint_combined_budget_bytes") <= (512ull << 20) &&
            get(v, "verification8_branch_reuse").boolean(), "private branch8 history consent/configuration");
        uint64_t total = 0;
        for (const auto name : {"checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"}) {
            const auto bytes = integer(v, name); need(bytes && bytes <= integer(v, "checkpoint_combined_budget_bytes") - total, "private branch8 history budget"); total += bytes;
        }
    } else
#else
    static_cast<void>(branch8);
#endif
    for (const auto name : {"checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"})
        need(!integer(v, name), "private V8 without explicit branch8 consent cannot use history");
    need(get(v, "stack_execution").string() == "direct" && get(v, "hc_q8_numerical_tier").string() == "ordinary" &&
        get(v, "hc_exact_mode").string() == "combined" && get(v, "known_residual_reuse").boolean() &&
        get(v, "prompt_lookup").boolean() && get(v, "verification_head_many").boolean() &&
        !get(v, "replay_prepared").boolean() && !get(v, "wide_q8_exact_enabled").boolean() &&
        !get(v, "wide_q8_many_enabled").boolean() && !get(v, "timings_qualified").boolean(), "private V8 policy drift");
    need(!v.find("expert_route_trace") && !v.find("trace_expert_routes") && !v.find("moe_token_tile_enabled"), "private V8 unrelated feature metadata");
    zero_routes(v, "wide_q8_exact_routes", 4); zero_routes(v, "wide_q8_many_routes", 6);
}
uint64_t ordinary(uint32_t prefix, bool stage8_enabled = false, bool channel_enabled = false) {
    // This private endpoint binds the adopted fixed 48-layer architecture:
    // 12 QSA layers, compression4 and a 2048-token selection budget.
    // 2222 includes 18 ordinary QSA dispatches per layer. Selection begins
    // at token2052 and adds sort/merge work; never bless an observed count.
    constexpr QsaShape shape{2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536};
    return 2222 + 12 * (qsa_dispatches(shape, prefix) - 18) + 12 * uint64_t(stage8_enabled) + 12 * uint64_t(channel_enabled);
}
void sampling(const Json& v, const Json& policy, uint64_t seed, uint32_t prefix) {
    const auto& s = get(v, "sampling");
    need(get(s, "valid").boolean() && integer(s, "seed") == seed && integer(s, "entropy_prefix") == prefix &&
        integer(s, "entropy_words") == 0, "private greedy V8 sampling ancestry");
    for (const auto& [k, n] : policy.object()) {
        Json pair(Json::Object{{"x", n}}); need(real(s, k) == real(pair, "x"), "private V8 sampling policy changed");
    }
}
void hc_shape(const Json& r) {
    need(r.object().size() == hc.size(), "private V8 HC route shape");
    for (const auto name : hc) static_cast<void>(integer(r, name));
}
} // namespace

namespace {
void ready(const Json& v, bool branch8) {
    fixed(v, branch8);
    const bool transport =
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        branch8 ? get(v, "checkpoint_broker").boolean() && integer(v, "broker_epoch") :
#endif
        !get(v, "checkpoint_broker").boolean() && integer(v, "broker_epoch") == 0;
    need(get(v, "event").string() == "ready" && integer(v, "maximum_generated_tokens") == 4096 &&
        transport &&
        !get(v, "weight_copy").boolean() && get(v, "gpu_profile_mode").string() == "disabled" &&
        get(v, "allow_prompt_lookup_experiments").boolean() ==
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            !stage8_metadata(v)
#else
            true
#endif
        , "private V8 ready identity");
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    if (stage8_metadata(v)) need(!integer(v, "qsa_stage_v8_calls"), "private staged-QSA pre-ready model work");
#endif
    for (const auto name : {"allow_request_kernel_experiments", "allow_request_stack_experiments", "allow_request_prefill_experiments"})
        need(!get(v, name).boolean(), "private V8 request-time experiment");
    need(integer(v, "lookup_match_tokens") == 16 && integer(v, "lookup_history_tokens") == 65536 &&
        integer(v, "lookup_proposal_tokens") == 7 && integer(v, "lookup_verification_budget_bytes") == (128ull << 20), "private V8 capability");
    uint64_t total = 0;
    for (const auto name : {"lookup_caller_host_bytes", "verification_host_bytes", "verification_gpu_bytes"}) {
        const auto bytes = integer(v, name); need(bytes && bytes <= (128ull << 20) - total, "private V8 combined budget"); total += bytes;
    }
    need(total == integer(v, "lookup_verification_charged_bytes") && integer(v, "verification_tape_payload_bytes") &&
        integer(v, "verification_tape_payload_bytes") <= integer(v, "verification_gpu_bytes"), "private V8 actual verifier charge");
    const auto live = integer(v, "tracked_metal_buffer_bytes"), peak = integer(v, "peak_metal_buffer_bytes"), budget = integer(v, "gpu_buffer_limit_bytes");
    need(budget == (96ull << 30) - (512ull << 20) - (2ull << 30) && live && live <= peak && peak <= budget &&
        integer(v, "imported_weight_backing_bytes") <= live && integer(v, "gpu_allocation_count"), "private V8 graph ledger");
    hc_shape(get(v, "hc_exact_routes")); for (const auto name : hc) need(!integer(get(v, "hc_exact_routes"), name), "private V8 pre-ready routes");
}
} // namespace

void validate_verify8_ready(const Json& v) { ready(v, false); }
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
std::optional<bool> verify8_fast_math(const Json& v) { return fast_metadata(v); }
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
namespace {
constexpr std::array<const char*, 5> scalar_resources{"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes",
    "gpu_allocation_count", "gpu_buffer_limit_bytes", "imported_weight_backing_bytes"};
void scalar_fixed(const Json& v, bool sampled_prefill = false) {
    need(fast_metadata(v) == std::optional<bool>(true), "scalar-channel requires the complete signed-root/channel-ON metadata family");
    for (const auto& [name, expected] : {std::pair{"small_gemv_column_splits", 16ull}, {"block_gemv_formats", 3ull},
        {"known_token_capacity", sampled_prefill ? 8ull : 0ull}, {"hc_q8_splits", 0ull}, {"hc_q8_scratch_logical_bytes", 0ull},
        {"hc_q8_scratch_charged_bytes", 0ull}, {"checkpoint_slots", 0ull}, {"checkpoint_combined_budget_bytes", 0ull},
        {"checkpoint_host_bytes", 0ull}, {"checkpoint_store_host_bytes", 0ull}, {"checkpoint_gpu_bytes", 0ull},
        {"recurrent_replay_captures", 0ull}, {"mapped_weight_region_limit_bytes", 0ull},
        {"capture_resource_bytes", 0ull}, {"capture_binding_aliases", 0ull}, {"capture_binding_alias_span_bytes", 0ull},
        {"capture_binding_alias_resource_bytes", 0ull}, {"live_captures", 0ull}, {"profile_counter_payload_bytes", 0ull},
        {"verification_max_tokens", 2ull}, {"native_limit_bytes", 96ull << 30},
        {"hc_exact_rows_per_group", 1ull}, {"hc_exact_unroll", 16ull},
        {"hc_exact_scratch_logical_bytes", 0ull}, {"hc_exact_scratch_charged_bytes", 0ull}})
        need(integer(v, name) == expected, "scalar-channel fixed preparation differs");
    need(get(v, "stack_execution").string() == "direct" && get(v, "hc_q8_numerical_tier").string() == "ordinary" &&
        get(v, "hc_exact_mode").string() == "combined" && get(v, "known_residual_reuse").boolean() == sampled_prefill &&
        !get(v, "verification_head_many").boolean() && !get(v, "verification8_branch_reuse").boolean() &&
        !get(v, "replay_prepared").boolean() && !get(v, "wide_q8_exact_enabled").boolean() &&
        !get(v, "wide_q8_many_enabled").boolean() && !get(v, "timings_qualified").boolean(), "scalar-channel policy drift");
    need(!v.find("licensed_q4_routed_mode") && !v.find("batch_phase_clock_mode") &&
        !v.find("branch8_session") && !v.find("broker_protocol"), "scalar-channel unrelated private policy");
    zero_routes(v, "wide_q8_exact_routes", 4); zero_routes(v, "wide_q8_many_routes", 6);
    const auto live = integer(v, "tracked_metal_buffer_bytes"), peak = integer(v, "peak_metal_buffer_bytes"), budget = integer(v, "gpu_buffer_limit_bytes");
    need(budget == (96ull << 30) - (512ull << 20) - (2ull << 30) && live && live <= peak && peak <= budget &&
        integer(v, "imported_weight_backing_bytes") <= live && integer(v, "gpu_allocation_count"), "scalar-channel graph ledger");
    hc_shape(get(v, "hc_exact_routes"));
}
}
void validate_scalar_channel_ready(const Json& v) {
    validate_worker_ready(v, "direct"); scalar_fixed(v);
    need(!integer(v, "qsa_stage_v8_calls") && !integer(v, "qsa_channel_calls") &&
        !get(v, "prompt_lookup").boolean() && !get(v, "checkpoint_broker").boolean() && !integer(v, "broker_epoch") &&
        !get(v, "weight_copy").boolean() && get(v, "gpu_profile_mode").string() == "disabled", "scalar-channel ready performed work or enabled unrelated storage");
    for (const auto name : {"lookup_verification_budget_bytes", "lookup_caller_host_bytes", "verification_host_bytes",
        "verification_gpu_bytes", "lookup_verification_charged_bytes", "verification_tape_payload_bytes"})
        need(!integer(v, name), "scalar-channel must not prepare a verifier");
    for (const auto name : {"allow_prompt_lookup_experiments", "allow_request_kernel_experiments",
        "allow_request_stack_experiments", "allow_request_prefill_experiments"})
        need(!get(v, name).boolean(), "scalar-channel request-time override");
    for (const auto name : hc) need(!integer(get(v, "hc_exact_routes"), name), "scalar-channel pre-ready HC work");
}
void validate_scalar_channel_done(const Json& ready, const Json& previous, const Json& job, const Json& done, const Json* token) {
    scalar_fixed(done); fast_same(done, ready); stage8_same(done, ready);
    need(get(done, "event").string() == "done", "scalar-channel completion event");
    for (const auto name : scalar_resources) need(get(done, name).dump() == get(ready, name).dump(), "scalar-channel resource drift");
    const auto inputs = get(job, "tokens").array().size();
    const auto outputs = integer(job, "generate");
    need(inputs <= 1 && outputs <= 1 && inputs + outputs == 1, "scalar-channel one-step command required");
    const bool reset = get(job, "reset").boolean();
    const auto before = get(previous, "event").string() == "ready" ? 0 : integer(previous, "committed_prefix");
    need(integer(done, "committed_prefix") == (reset ? 0 : before) + 1 && integer(done, "committed_prefix") <= 65536,
        "scalar-channel committed prefix");
    for (const auto name : {"qsa_stage_v8_calls", "qsa_channel_calls"}) {
        const auto prior = integer(previous, name), current = integer(done, name);
        need(current >= prior && current - prior == 12, "scalar-channel physical model-step count");
    }
    need(integer(done, "qsa_stage_v8_request_calls") == 12 && integer(done, "qsa_channel_request_calls") == 12,
        "scalar-channel per-command counter mismatch");
    const auto& ids = get(done, "generated_ids").array();
    need(ids.size() == outputs && bool(token) == bool(outputs), "scalar-channel token count");
    if (token) need(get(*token, "event").string() == "token" && integer(*token, "id") < 248320 &&
        integer(*token, "id") == ids[0].unsigned_integer() &&
        integer(*token, "committed_prefix") == integer(done, "committed_prefix"), "scalar-channel token publication mismatch");
    const auto& policy = reset ? get(job, "sampling") : get(previous, "sampling");
    const auto& sampled = get(done, "sampling");
    need(get(sampled, "valid").boolean() && integer(sampled, "entropy_prefix") == integer(done, "committed_prefix") &&
        integer(sampled, "seed") == (reset ? integer(job, "seed") : integer(policy, "seed")), "scalar-channel sampler ancestry");
    for (const auto name : {"temperature", "top_k", "top_p", "min_p"})
        need(real(sampled, name) == real(policy, name), "scalar-channel sampling policy changed");
    const auto old_words = reset ? 0 : integer(policy, "entropy_words"), words = integer(sampled, "entropy_words");
    need(words >= old_words, "scalar-channel sampler entropy rewound without reset");
    // sample() draws nothing for a point mass and otherwise one uniform. The
    // scalar target's greedy branch draws nothing, as does every input step.
    const uint64_t maximum_words = outputs && real(policy, "temperature") > 0 ? 1 : 0;
    need(words - old_words <= maximum_words, "scalar-channel sampler entropy advance exceeds one scalar step");
    if (token) need(integer(*token, "entropy_words") == words, "scalar-channel token/completion entropy mismatch");
    hc_shape(get(done, "hc_exact_request_routes"));
    for (const auto name : hc) {
        const auto prior = integer(get(previous, "hc_exact_routes"), name), current = integer(get(done, "hc_exact_routes"), name);
        need(current >= prior && current - prior == integer(get(done, "hc_exact_request_routes"), name), "scalar-channel HC route accounting");
        need(integer(get(done, "hc_exact_request_routes"), name) == (std::string_view(name) == "scalar" ? 97 : 0),
            "scalar-channel HC one-row route coverage");
    }
}
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
#ifndef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
#error Sampled prefill requires the separate sampled-capable scalar channel lineage.
#endif
namespace {
constexpr std::array<const char*, 8> sampled_rows_fields{"known_qsa_rows_schema", "known_qsa_rows_mode",
    "known_qsa_rows_logical_bytes", "known_qsa_rows_charged_bytes", "known_qsa_rows_allocations",
    "known_qsa_rows_physical_tiles", "known_qsa_rows_logical_rows", "known_qsa_rows_http_admitted"};
constexpr std::array<const char*, 14> sampled_ready_only_fields{"lookup_verification_budget_bytes",
    "lookup_caller_host_bytes", "verification_host_bytes", "verification_gpu_bytes",
    "lookup_verification_charged_bytes", "verification_tape_payload_bytes", "prompt_lookup",
    "checkpoint_broker", "broker_epoch", "gpu_profile_mode", "allow_prompt_lookup_experiments",
    "allow_request_kernel_experiments", "allow_request_stack_experiments", "allow_request_prefill_experiments"};
void sampled_prefill_fixed(const Json& v) {
    scalar_fixed(v, true);
    need(get(v, "sampled_prefill_protocol").string() == "sampled-prefill-input8-v1",
        "sampled prefill wire identity differs");
    for (const auto& [name, ignored] : v.object()) if (name.starts_with("known_qsa_rows_")) {
        static_cast<void>(ignored);
        need(std::find(sampled_rows_fields.begin(), sampled_rows_fields.end(), name) != sampled_rows_fields.end(),
            "sampled prefill unknown row metadata");
    }
    // Same fixed bound as maximum_known_qsa_rows_bytes; this validator stays CPU-only.
    constexpr uint64_t row_budget = 64ull << 20;
    const auto logical = integer(v, "known_qsa_rows_logical_bytes"), charged = integer(v, "known_qsa_rows_charged_bytes");
    need(integer(v, "known_qsa_rows_schema") == 1 && get(v, "known_qsa_rows_mode").string() == "q8" &&
        logical && logical <= charged && charged <= row_budget && integer(v, "known_qsa_rows_allocations") == 46 &&
        get(v, "known_qsa_rows_http_admitted").boolean(), "sampled prefill fixed row preparation differs");
}
void sampled_delta(const Json& previous, const Json& done, const char* name, uint64_t expected) {
    const auto prior = integer(previous, name), current = integer(done, name);
    need(current >= prior && current - prior == expected, "sampled prefill physical route count differs");
}
}
void validate_sampled_prefill_ready(const Json& v) {
    validate_worker_ready(v, "direct"); sampled_prefill_fixed(v);
    // Startup preparation is reported by READY, not repeated on DONE.
    for (const auto name : {"lookup_verification_budget_bytes", "lookup_caller_host_bytes", "verification_host_bytes",
        "verification_gpu_bytes", "lookup_verification_charged_bytes", "verification_tape_payload_bytes"})
        need(!integer(v, name), "sampled prefill must not prepare a verifier");
    need(!get(v, "prompt_lookup").boolean() && !get(v, "checkpoint_broker").boolean() && !integer(v, "broker_epoch") &&
        get(v, "gpu_profile_mode").string() == "disabled", "sampled prefill unrelated executor policy");
    for (const auto name : {"allow_prompt_lookup_experiments", "allow_request_kernel_experiments",
        "allow_request_stack_experiments", "allow_request_prefill_experiments"})
        need(!get(v, name).boolean(), "sampled prefill request-time override");
    need(!get(v, "weight_copy").boolean(), "sampled prefill ready copied weights");
    for (const auto name : {"qsa_stage_v8_calls", "qsa_channel_calls", "known_qsa_rows_physical_tiles", "known_qsa_rows_logical_rows"})
        need(!integer(v, name), "sampled prefill ready performed model work");
    for (const auto name : hc) need(!integer(get(v, "hc_exact_routes"), name), "sampled prefill pre-ready HC work");
}
void validate_sampled_prefill_done(const Json& ready, const Json& previous, const Json& job, const Json& done, const Json* token) {
    sampled_prefill_fixed(done); fast_same(done, ready); stage8_same(done, ready);
    for (const auto name : sampled_ready_only_fields)
        need(!done.find(name), "sampled prefill DONE contains startup-only metadata");
    need(get(done, "event").string() == "done" && get(done, "known_token_prefill").boolean(), "sampled prefill completion event");
    for (const auto name : scalar_resources) need(get(done, name).dump() == get(ready, name).dump(), "sampled prefill resource drift");
    for (const auto name : {"known_qsa_rows_schema", "known_qsa_rows_mode", "known_qsa_rows_logical_bytes",
        "known_qsa_rows_charged_bytes", "known_qsa_rows_allocations", "known_qsa_rows_http_admitted", "sampled_prefill_protocol"})
        need(get(done, name).dump() == get(ready, name).dump(), "sampled prefill immutable row policy changed");
    const auto inputs = get(job, "tokens").array().size();
    const auto outputs = integer(job, "generate");
    need((inputs >= 1 && inputs <= 8 && !outputs) || (!inputs && outputs == 1), "sampled prefill requires input1..8 OR output1");
    for (const auto& id : get(job, "tokens").array()) need(id.unsigned_integer() < 248320, "sampled prefill input outside vocabulary");
    const bool reset = get(job, "reset").boolean(), was_ready = get(previous, "event").string() == "ready";
    need((!was_ready || reset) && (!outputs || !reset), "sampled prefill reset/continuation ancestry differs");
    const auto before = was_ready ? 0 : integer(previous, "committed_prefix");
    const auto count = inputs + outputs;
    need(integer(done, "committed_prefix") == (reset ? 0 : before) + count && integer(done, "committed_prefix") <= 65536,
        "sampled prefill committed prefix differs");
    for (const auto name : {"qsa_stage_v8_calls", "qsa_channel_calls"}) sampled_delta(previous, done, name, outputs ? 12 : 0);
    need(integer(done, "qsa_stage_v8_request_calls") == (outputs ? 12 : 0) &&
        integer(done, "qsa_channel_request_calls") == (outputs ? 12 : 0), "sampled prefill per-command attention count differs");
    sampled_delta(previous, done, "known_qsa_rows_physical_tiles", inputs ? 12 : 0);
    sampled_delta(previous, done, "known_qsa_rows_logical_rows", 12 * inputs);
    const auto& ids = get(done, "generated_ids").array();
    need(ids.size() == outputs && bool(token) == bool(outputs), "sampled prefill token count differs");
    if (inputs) need(get(done, "status").string() == "prefilled" && get(done, "finish_reason").is_null() &&
        get(done, "stop_id").is_null(), "sampled prefill input emitted a terminal result");
    if (token) need(get(*token, "event").string() == "token" && integer(*token, "id") < 248320 &&
        integer(*token, "id") == ids[0].unsigned_integer() &&
        integer(*token, "committed_prefix") == integer(done, "committed_prefix"), "sampled prefill token publication differs");
    const auto& policy = reset ? get(job, "sampling") : get(previous, "sampling");
    const auto& sampled = get(done, "sampling");
    need(get(sampled, "valid").boolean() && integer(sampled, "entropy_prefix") == integer(done, "committed_prefix") &&
        integer(sampled, "seed") == (reset ? integer(job, "seed") : integer(policy, "seed")), "sampled prefill sampler ancestry differs");
    for (const auto name : {"temperature", "top_k", "top_p", "min_p"})
        need(real(sampled, name) == real(policy, name), "sampled prefill sampling policy changed");
    const auto old_words = reset ? 0 : integer(policy, "entropy_words"), words = integer(sampled, "entropy_words");
    const uint64_t maximum_words = outputs && real(policy, "temperature") > 0 ? 1 : 0;
    need(words >= old_words && words - old_words <= maximum_words, "sampled prefill entropy advance differs");
    if (token) need(integer(*token, "entropy_words") == words, "sampled prefill token/completion entropy differs");
    // 36 recurrent layers: two residual-down projections per width chunk.
    // 12 attention layers retain two scalar residual-downs per input row; the
    // final mixer adds one scalar projection. Decode remains the ordinary97.
    const std::array<uint64_t, 5> expected = outputs ? std::array<uint64_t, 5>{97, 0, 0, 0, 0} :
        std::array<uint64_t, 5>{24 * inputs + 1, (inputs & 1) ? 72ull : 0ull,
            (inputs & 2) ? 72ull : 0ull, (inputs & 4) ? 72ull : 0ull, (inputs & 8) ? 72ull : 0ull};
    hc_shape(get(done, "hc_exact_request_routes"));
    for (size_t i = 0; i < hc.size(); ++i) {
        const auto prior = integer(get(previous, "hc_exact_routes"), hc[i]), current = integer(get(done, "hc_exact_routes"), hc[i]);
        need(current >= prior && current - prior == expected[i] && integer(get(done, "hc_exact_request_routes"), hc[i]) == expected[i],
            "sampled prefill exact-HC coverage differs");
    }
}
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
namespace {
constexpr std::array<const char*, 5> graph_fields{"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes", "gpu_allocation_count", "gpu_buffer_limit_bytes", "imported_weight_backing_bytes"};
constexpr std::array<const char*, 10> cost_fields{"lookup_appended_tokens", "lookup_indexed_grams", "lookup_truncated_grams", "lookup_cleared_grams",
    "lookup_resets", "lookup_truncations", "lookup_queries", "lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries"};
uint64_t grams(uint64_t prefix) { return prefix < 16 ? 0 : prefix - 15; }
void branch_identity(const Json& value) {
    need(get(value, "broker_protocol").string() == "branch8-v1" && integer(value, "broker_batch_width") == 8 &&
        integer(value, "broker_epoch"), "private branch8 transport identity");
    static_cast<void>(integer(value, "command_id")); static_cast<void>(integer(value, "lease_id"));
}
void frozen_branch_storage(const Json& value, const Json& prepared) {
    for (const auto name : {"checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"})
        need(integer(value, name) == integer(prepared, name), "private branch8 prepared history changed");
    for (const auto name : graph_fields)
        need(integer(branch8_session(value), name) == integer(prepared, name), "private branch8 frozen graph changed");
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    stage8_same(branch8_session(value), prepared);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    fast_same(branch8_session(value), prepared);
#endif
}
} // namespace

const Json& branch8_session(const Json& reply) {
    branch_identity(reply); const auto& state = get(reply, "branch8_session");
    keys(state, {"committed_prefix", "sampling", "logits_available", "continuation_allowed", "stop_id", "final_logit_top8", "hc_exact_final_logit_sha256",
        "hc_exact_routes", "lookup_history_epoch", "lookup_appended_tokens", "lookup_indexed_grams", "lookup_truncated_grams", "lookup_cleared_grams",
        "lookup_resets", "lookup_truncations", "lookup_queries", "lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries",
        "tracked_metal_buffer_bytes", "peak_metal_buffer_bytes", "gpu_allocation_count", "gpu_buffer_limit_bytes", "imported_weight_backing_bytes"
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        , "qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes", "qsa_stage_v8_scratch_charged_bytes",
        "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches"
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        , "ple_gate_transform", "ple_gate_transform_schema", "qsa_channel_chronological_mode", "qsa_channel_chronological_schema",
        "qsa_channel_factor_logical_bytes", "qsa_channel_factor_charged_bytes", "qsa_channel_calls", "qsa_channel_extra_dispatches"
#endif
        });
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    static_cast<void>(stage8_metadata(state));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    static_cast<void>(fast_metadata(state));
#endif
    const auto prefix = integer(state, "committed_prefix"); need(prefix <= 65536 && integer(state, "lookup_history_epoch"), "private branch8 state prefix/epoch");
    for (const auto name : cost_fields) static_cast<void>(integer(state, name));
    for (const auto name : graph_fields) static_cast<void>(integer(state, name));
    hc_shape(get(state, "hc_exact_routes"));
    const auto& policy = get(state, "sampling");
    need(get(policy, "valid").boolean() && integer(policy, "entropy_prefix") == prefix && !integer(policy, "entropy_words") &&
        real(policy, "temperature") == 0 && integer(policy, "top_k") <= 248320 && real(policy, "top_p") > 0 && real(policy, "top_p") <= 1 &&
        real(policy, "min_p") >= 0 && real(policy, "min_p") <= 1, "private branch8 greedy sampler state");
    static_cast<void>(integer(policy, "seed"));
    const bool logits = get(state, "logits_available").boolean(); const auto& stop = get(state, "stop_id");
    need(logits == bool(prefix) && (stop.is_null() || (logits && stop.unsigned_integer() < 248320)) &&
        get(state, "continuation_allowed").boolean() == (logits && stop.is_null()), "private branch8 state logits/terminal");
    const auto& top = get(state, "final_logit_top8").array();
    need(top.size() == (prefix ? 8 : 0), "private branch8 state top8 shape");
    if (prefix) {
        const auto hash = get(state, "hc_exact_final_logit_sha256").string();
        need(hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string_view::npos, "private branch8 full-logit identity");
        double last = std::numeric_limits<double>::infinity(); uint64_t last_id = 0;
        for (size_t i = 0; i < top.size(); ++i) {
            keys(top[i], {"id", "logit"}); const auto id = integer(top[i], "id"); const auto score = real(top[i], "logit");
            need(id < 248320 && (score < last || (i && score == last && id > last_id)), "private branch8 sorted state logits");
            for (size_t j = 0; j < i; ++j) need(integer(top[j], "id") != id, "private branch8 duplicate state logit");
            last = score; last_id = id;
        }
    } else need(get(state, "hc_exact_final_logit_sha256").is_null(), "private branch8 empty state hash");
    // A fresh native snapshot must agree with every duplicate field in its
    // actual envelope. No historical packet is rewritten to impersonate it.
    for (const auto& [name, value] : state.object()) if (const auto* outside = reply.find(name))
        need(outside->dump() == value.dump(), "private branch8 snapshot/envelope disagreement");
    return state;
}

void validate_branch8_ready(const Json& value) {
    ready(value, true); const auto& state = branch8_session(value);
    need(!integer(value, "command_id") && !integer(value, "lease_id") && !integer(state, "committed_prefix") &&
        integer(state, "lookup_history_epoch") == 1, "private branch8 fresh readiness ancestry");
    for (const auto name : cost_fields) need(!integer(state, name), "private branch8 pre-ready history work");
    frozen_branch_storage(value, value);
}

void validate_branch8_control(const Json& prepared, const Json& previous, const Json& job, const Json& value, bool cold_resync) {
    branch_identity(value); const auto& state = branch8_session(value); const auto& old = branch8_session(previous);
    keys(value, {"event", "op", "outcome", "model_steps", "generated_ids", "common_prefix", "committed_prefix", "sampling", "stop_id",
        "logits_available", "continuation_allowed", "checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes", "checkpoint_store_host_bytes",
        "checkpoint_gpu_bytes", "checkpoint_retained", "control_wall_ms", "timings_qualified", "branch8_session"}, true);
    frozen_branch_storage(value, prepared);
    need(get(value, "event").string() == "control_done" && get(value, "op").string() == get(job, "op").string() && !integer(value, "model_steps") &&
        get(value, "generated_ids").array().empty() && !get(value, "timings_qualified").boolean() && real(value, "control_wall_ms") >= 0 &&
        integer(value, "checkpoint_retained") <= integer(prepared, "checkpoint_slots"), "private branch8 zero-model-step control envelope");
    const auto operation = get(job, "op").string(); const auto prefix = integer(state, "committed_prefix"), before = integer(old, "committed_prefix");
    const auto outcome = get(value, "outcome").string(); const auto common = integer(value, "common_prefix");
    const auto command = integer(previous, "command_id"); need(command < std::numeric_limits<uint64_t>::max() &&
        integer(value, "broker_epoch") == integer(prepared, "broker_epoch") &&
        (cold_resync ? integer(value, "command_id") > command : integer(value, "command_id") == command + 1), "private branch8 control command ancestry");
    const auto lease = integer(previous, "lease_id");
    need(operation == "branch_begin" ? integer(value, "lease_id") > lease : lease && integer(value, "lease_id") == lease,
        "private branch8 control lease ancestry");
    if (operation == "branch_capture") {
        keys(job, {"op"}); need(!cold_resync && prefix && common == before && prefix == before && outcome == "captured" &&
            state.dump() == old.dump(), "private branch8 capture changed current state or lifetime costs"); return;
    }
    need(operation == "branch_begin", "private branch8 unknown control");
    keys(job, {"op", "prompt_ids", "generation_requested", "allow_restore", "sampling", "seed"});
    const auto count = get(job, "prompt_ids").array().size(); const bool allow = get(job, "allow_restore").boolean();
    need(count && count <= 65536 && (!cold_resync || !allow) && (allow || !common) && common <= std::min<uint64_t>(before, count) && prefix <= common &&
        ((outcome == "restored" && prefix && allow && !cold_resync) || (outcome == "cold_miss" && !prefix)), "private branch8 ancestor restore/cold selection");
    for (const auto& id : get(job, "prompt_ids").array()) need(id.unsigned_integer() < 248320, "private branch8 future ID");
    const auto& policy = get(job, "sampling"); keys(policy, {"temperature", "top_k", "top_p", "min_p"});
    need(real(policy, "temperature") == 0 && integer(policy, "top_k") <= 248320 && real(policy, "top_p") > 0 && real(policy, "top_p") <= 1 &&
        real(policy, "min_p") >= 0 && real(policy, "min_p") <= 1, "private branch8 fresh greedy policy");
    sampling(state, get(job, "sampling"), integer(job, "seed"), uint32_t(prefix));
    need(!get(job, "generation_requested").boolean() || prefix < count || get(state, "stop_id").is_null(), "private branch8 restored terminal cannot generate");
    for (const auto name : hc) {
        const auto a = integer(get(old, "hc_exact_routes"), name), b = integer(get(state, "hc_exact_routes"), name);
        need(cold_resync ? b >= a : b == a, "private branch8 control changed GPU execution cost");
    }
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    stage8_same(state, old);
    if (stage8_metadata(state)) {
        const auto a = integer(old, "qsa_stage_v8_calls"), b = integer(state, "qsa_stage_v8_calls");
        need(cold_resync ? b >= a : b == a, "private staged-QSA zero-step control changed execution costs");
    }
#endif
    const auto epoch = integer(old, "lookup_history_epoch"); need(epoch < std::numeric_limits<uint64_t>::max() &&
        (cold_resync ? integer(state, "lookup_history_epoch") > epoch : integer(state, "lookup_history_epoch") == epoch + 1), "private branch8 fresh lookup epoch");
    for (const auto name : cost_fields) {
        const auto a = integer(old, name), b = integer(state, name); uint64_t delta = 0;
        if (std::string_view(name) == "lookup_resets") delta = outcome == "cold_miss";
        if (std::string_view(name) == "lookup_truncations") delta = outcome == "restored";
        if (std::string_view(name) == "lookup_truncated_grams" && outcome == "restored") delta = grams(before) - grams(prefix);
        // Only the exact worker-side predecessor identifies what a cold reset
        // clears. An HTTP observer may have missed a completed ancestor restore
        // before cancellation, so its stale longer prefix cannot lower-bound
        // that work. Resync still requires monotonic costs, a new reset/epoch,
        // a fresh command/lease and an explicit zero-prefix cold miss above.
        if (std::string_view(name) == "lookup_cleared_grams" && outcome == "cold_miss" && !cold_resync) delta = grams(before);
        need(b >= a && (cold_resync ? b - a >= delta : b - a == delta), "private branch8 control lifetime-history cost mismatch");
    }
}
#endif

Verify8Reply::Verify8Reply(const Json& ready, const Json& previous, const Json& job, uint32_t before, const Json& policy, uint64_t seed, bool reset_resync
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    , bool branch8
#endif
    ) : ready_(ready), previous_(
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        branch8 && !previous.is_null() ? branch8_session(previous) :
#endif
        previous), job_(job), sampling_(policy), seed_(seed), base_(before), position_(before),
      inputs_(uint32_t(get(job, "tokens").array().size())), maximum_(uint32_t(integer(job, "generate"))), reset_(get(job, "reset").boolean()), reset_resync_(reset_resync)
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
      , branch8_(branch8)
      , branch_epoch_(branch8 ? integer(ready, "broker_epoch") : 0)
      , branch_command_(branch8 ? integer(previous, "command_id") + 1 : 0)
      , branch_lease_(branch8 ? integer(previous, "lease_id") : 0)
#endif
      {
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    const auto fast = fast_metadata(ready_); fast_math_prepared_ = fast.has_value(); channel_enabled_ = fast.value_or(false);
    if (!previous_.is_null()) fast_same(previous_, ready_);
    if (fast_math_prepared_) channel_before_ = integer(previous_, "qsa_channel_calls");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    stage8_prepared_ = stage8_metadata(ready_);
    if (!previous_.is_null()) stage8_same(previous_, ready_);
    if (stage8_prepared_) {
        need(branch8_, "private staged HTTP requires branch8 admission");
        stage8_enabled_ = get(ready_, "qsa_stage_v8_enabled").boolean(); stage8_before_ = integer(previous_, "qsa_stage_v8_calls");
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    need(!branch8_ || (!reset_ && !reset_resync_ && !previous_.is_null() && branch_epoch_ && branch_command_ && branch_lease_),
        "private branch8 model work requires a separately admitted cold/restore control");
#endif
    keys(job, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed", "prompt_lookup"});
    need(get(job, "tokens").array().size() <= 8 && integer(job, "generate") <= 8 &&
        bool(inputs_) != bool(maximum_) && before <= 65536, "private V8 one bounded input OR generation batch required");
    keys(policy, {"temperature", "top_k", "top_p", "min_p"});
    need(real(policy, "temperature") == 0 && integer(policy, "top_k") <= 248320 && real(policy, "top_p") > 0 &&
        real(policy, "top_p") <= 1 && real(policy, "min_p") >= 0 && real(policy, "min_p") <= 1, "private V8 is greedy-only");
    if (reset_) { base_ = position_ = 0; need(inputs_ && !maximum_ && get(job, "sampling").dump() == policy.dump() && integer(job, "seed") == seed, "private V8 reset must install policy/seed with input"); }
    else {
        need(!job.find("sampling") && !job.find("seed") && !previous_.is_null() && integer(previous_, "committed_prefix") == before &&
            (inputs_ || get(previous_, "continuation_allowed").boolean()), "private V8 continuation policy/lineage");
        sampling(previous_, policy, seed, before);
    }
    need(base_ + inputs_ + maximum_ <= 65536 && (inputs_ || base_), "private V8 context bound");
    for (const auto& id : get(job, "tokens").array()) need(id.unsigned_integer() < 248320, "private V8 input ID");
    need((reset_ ? get(job, "prompt_lookup").boolean() : !job.find("prompt_lookup")) && (!reset_resync_ || reset_), "private V8 reset-only lookup/resync policy");
    if (maximum_) {
        const auto& stops = get(job, "stop_ids").array();
        need((stops.size() == 1 && stops[0].unsigned_integer() < 248320) ||
            (stops.size() == 2 && ((stops[0].unsigned_integer() == 248046 && stops[1].unsigned_integer() == 248044) ||
                (stops[0].unsigned_integer() == 248044 && stops[1].unsigned_integer() == 248046))), "private V8 requires one terminal or the pinned terminal pair");
    }
    else need(!job.find("stop_ids"), "private V8 prefill stop list");
}

void Verify8Reply::event(const Json& v) {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    if (branch8_) {
        branch_identity(v); need(integer(v, "broker_epoch") == branch_epoch_ && integer(v, "command_id") == branch_command_ &&
            integer(v, "lease_id") == branch_lease_, "private branch8 event epoch/command/lease mismatch");
    }
#endif
    need(!finished_ && maximum_ && !stopped_ && event_count_ < events_.size(), "private V8 unsolicited event");
    const auto type = get(v, "event").string();
    if (type == "generation_round") {
        keys(v, {"event", "round_id", "round_base", "committed_prefix", "proposed_tokens", "verified_tokens", "verification_max_tokens", "verification_head_dispatches",
            "accepted_verified_tokens", "ordinary_tokens", "useful_tokens", "round_wall_ms", "lookup_and_publication_ms", "gather_wall_ms", "verification_including_recovery_wall_ms",
            "recovery_batch_wall_ms", "fallback_wall_ms", "batch_wall_ms", "batch_gpu_ms", "hc_exact_round_routes", "wide_q8_exact_round_routes", "wide_q8_many_round_routes",
            "dispatches", "capture_executes", "useful_tokens_per_second", "accepted_tokens_per_second", "timings_qualified", "gpu_profile"}
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            , branch8_
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            , stage8_prepared_
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
            , fast_math_prepared_
#endif
#endif
#endif
            );
        need(!pending_ && ++rounds_ <= 8 && integer(v, "round_id") == rounds_ && integer(v, "round_base") == position_, "private V8 round chronology");
        const auto n = integer(v, "verified_tokens"), k = integer(v, "accepted_verified_tokens"), o = integer(v, "ordinary_tokens"), p = integer(v, "proposed_tokens");
        const auto count = integer(v, "useful_tokens");
        need(p <= 7 && (n == 0 || (n >= 2 && n <= 7)) && k <= n && n <= p && n <= maximum_ - count_ && n <= 65536 - position_ &&
            o <= 1 && count == k + o && count && count <= maximum_ - count_ &&
            (!n ? !k && o == 1 : k >= 1 && o == uint64_t(k < n)) && integer(v, "verification_max_tokens") == 8 &&
            integer(v, "verification_head_dispatches") == uint32_t(std::popcount(uint32_t(n))) && integer(v, "committed_prefix") == position_ + count,
            "private V8 round acceptance/correction/depth bound");
        accepted_ = uint32_t(k); pending_ = round_count_ = uint32_t(count);
        ++verified_rounds_; if (!n) --verified_rounds_; evaluated_total_ += uint32_t(n); proposed_total_ += uint32_t(p); ordinary_total_ += uint32_t(o);
        recovery_rounds_ += k > 0 && k < n;
        const bool stage_enabled =
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            stage8_enabled_;
        if (stage8_prepared_) need(integer(v, "qsa_stage_v8_round_calls") == (stage8_enabled_ ? 12 * (n + o) : 0),
            "private staged-QSA exact round work");
#else
            false;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        const bool channel_enabled = channel_enabled_;
        if (fast_math_prepared_) need(integer(v, "qsa_channel_round_calls") == (channel_enabled ? 12 * (n + o) : 0),
            "private fast-math exact round work");
        else need(!v.find("qsa_channel_round_calls"), "unexpected fast-math round metadata");
#else
        constexpr bool channel_enabled = false;
#endif
        uint64_t serial = 0; for (uint32_t t = 0; t < n; ++t) serial += ordinary(position_ + t, stage_enabled, channel_enabled);
        const auto work = (n ? serial - 72 * (n - 1) - 253 * (n - std::popcount(uint32_t(n))) : 0) +
            (k && k < n ? 72 : 0) + (o ? ordinary(position_ + uint32_t(k), stage_enabled, channel_enabled) : 0);
        need(integer(v, "dispatches") == work && !integer(v, "capture_executes") && !get(v, "timings_qualified").boolean() && get(v, "gpu_profile").is_null(), "private V8 physical work/profile"); dispatches_ += work;
        const auto& r = get(v, "hc_exact_round_routes"); hc_shape(r);
        for (size_t i = 0; i < hc.size(); ++i) { const auto expected = i == 0 ? 25 * n + 97 * o : (n & (1u << (i - 1)) ? 72 : 0);
            need(integer(r, hc[i]) == expected, "private V8 round HC route coverage"); routes_[i] += expected; }
        zero_routes(v, "wide_q8_exact_round_routes", 4); zero_routes(v, "wide_q8_many_round_routes", 6);
        for (const auto name : {"round_wall_ms", "lookup_and_publication_ms", "gather_wall_ms", "verification_including_recovery_wall_ms", "recovery_batch_wall_ms", "fallback_wall_ms", "batch_wall_ms", "batch_gpu_ms", "useful_tokens_per_second", "accepted_tokens_per_second"})
            need(real(v, name) >= 0, "private V8 negative clock");
        const double wall = real(v, "round_wall_ms"), verify = real(v, "verification_including_recovery_wall_ms"), recovery = real(v, "recovery_batch_wall_ms");
        close(wall, real(v, "lookup_and_publication_ms") + real(v, "gather_wall_ms") + verify + real(v, "fallback_wall_ms"));
        need(wall > 0 && real(v, "batch_wall_ms") > 0 && real(v, "batch_gpu_ms") > 0 &&
            real(v, "batch_wall_ms") <= wall + 1e-6 &&
            (n ? verify > 0 : (!real(v, "gather_wall_ms") && !verify && !recovery)) &&
            ((k && k < n) ? recovery > 0 : recovery == 0) &&
            (o ? real(v, "fallback_wall_ms") > 0 : !real(v, "fallback_wall_ms")) &&
            recovery <= verify + 1e-6 && real(v, "batch_gpu_ms") <= real(v, "batch_wall_ms") + 1e-6,
            "private V8 clock attribution");
        close(real(v, "useful_tokens_per_second"), wall ? count * 1000.0 / wall : 0); close(real(v, "accepted_tokens_per_second"), wall ? k * 1000.0 / wall : 0);
        round_wall_ += wall; batch_wall_ += real(v, "batch_wall_ms"); batch_gpu_ += real(v, "batch_gpu_ms");
    } else {
        keys(v, {"event", "id", "source_logit", "committed_prefix", "token_prefix", "round_id", "round_base", "round_committed_prefix", "verified", "batched_commit", "entropy_words", "gpu_profile"}
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            , branch8_
#endif
            );
        const auto offset = round_count_ - pending_; const auto id = integer(v, "id");
        need(type == "token" && pending_ && count_ < maximum_ && id < 248320 && integer(v, "round_id") == rounds_ &&
            integer(v, "round_base") == position_ && integer(v, "committed_prefix") == position_ + round_count_ &&
            integer(v, "round_committed_prefix") == position_ + round_count_ && integer(v, "token_prefix") == position_ + offset + 1 &&
            get(v, "verified").boolean() == (offset < accepted_) && get(v, "batched_commit").boolean() == (accepted_ >= 2) &&
            !integer(v, "entropy_words") && get(v, "gpu_profile").is_null(), "private V8 token ancestry/source");
        const auto score = real(v, "source_logit");
        if (!count_) {
            const auto& top = get(previous_, "final_logit_top8").array();
            need(top.size() == 8 && id == integer(top[0], "id") && score == real(top[0], "logit"), "private V8 first greedy ID/source differs from retained logits");
        }
        ids_[count_++] = uint32_t(id); --pending_;
        const auto& stops = get(job_, "stop_ids").array();
        stopped_ = std::any_of(stops.begin(), stops.end(), [&](const Json& stop) { return id == stop.unsigned_integer(); });
        need(!stopped_ || !pending_, "private V8 token after first terminal");
        if (!pending_) position_ += round_count_;
    }
    events_[event_count_++] = v;
}

void Verify8Reply::finish(const Json& v) {
    need(!finished_ && !pending_ && get(v, "event").string() == "done", "private V8 incomplete round/done"); fixed(v
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , branch8_
#endif
        );
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    fast_same(v, ready_);
    if (fast_math_prepared_) {
        const uint64_t expected = channel_enabled_ ? 12ull * (inputs_ + evaluated_total_ + ordinary_total_) : 0;
        need(expected <= std::numeric_limits<uint64_t>::max() - channel_before_ &&
            integer(v, "qsa_channel_request_calls") == expected && integer(v, "qsa_channel_calls") == channel_before_ + expected,
            "private fast-math exact request/lifetime cost");
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    stage8_same(v, ready_);
    if (stage8_prepared_) {
        const uint64_t expected = stage8_enabled_ ? 12ull * (inputs_ + evaluated_total_ + ordinary_total_) : 0;
        need(expected <= std::numeric_limits<uint64_t>::max() - stage8_before_ &&
            integer(v, "qsa_stage_v8_request_calls") == expected && integer(v, "qsa_stage_v8_calls") == stage8_before_ + expected,
            "private staged-QSA exact request/lifetime cost");
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    if (branch8_) {
        static_cast<void>(branch8_session(v));
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        stage8_same(branch8_session(v), ready_);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        fast_same(branch8_session(v), ready_);
#endif
        need(integer(v, "broker_epoch") == branch_epoch_ && integer(v, "command_id") == branch_command_ && integer(v, "lease_id") == branch_lease_,
            "private branch8 completion epoch/command/lease mismatch");
        for (const auto name : {"checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"})
            need(integer(v, name) == integer(ready_, name), "private branch8 persistent storage drift");
        need(integer(v, "checkpoint_retained") <= integer(ready_, "checkpoint_slots") && get(v, "checkpoint_reuse").string() == "off" &&
            !integer(v, "checkpoint_restored_tokens") && integer(v, "replayed_input_tokens") == inputs_ &&
            !real(v, "checkpoint_capture_wall_ms") && !real(v, "checkpoint_restore_wall_ms"), "private branch8 model command cannot hide control work");
    }
#endif
    if (inputs_) position_ = base_ + inputs_;
    need(integer(v, "committed_prefix") == position_ && count_ == get(v, "generated_ids").array().size() &&
        (inputs_ || (count_ && (stopped_ || count_ == maximum_))) && get(v, "known_token_prefill").boolean(), "private V8 terminal prefix/count");
    for (size_t i = 0; i < count_; ++i) need(get(v, "generated_ids").array()[i].unsigned_integer() == ids_[i], "private V8 token/done disagreement");
    need(get(v, "status").string() == (inputs_ ? "prefilled" : stopped_ ? "stopped" : "length") &&
        (inputs_ ? get(v, "finish_reason").is_null() : get(v, "finish_reason").string() == (stopped_ ? "stop" : "length")) &&
        get(v, "continuation_allowed").boolean() == !stopped_, "private V8 completion/continuation");
    if (stopped_) need(integer(v, "stop_id") == ids_[count_ - 1], "private V8 terminal ID"); else need(get(v, "stop_id").is_null(), "private V8 unexpected stop");
    sampling(v, sampling_, seed_, position_);
    for (const auto name : {"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes", "gpu_allocation_count", "gpu_buffer_limit_bytes", "imported_weight_backing_bytes"})
        need(integer(v, name) == integer(ready_, name), "private V8 frozen graph ledger drift");
    const auto hash = get(v, "hc_exact_final_logit_sha256").string(); need(hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string_view::npos, "private V8 final logit identity");
    hc_shape(get(v, "hc_exact_routes")); hc_shape(get(v, "hc_exact_request_routes"));
    if (inputs_) { routes_[0] = 24 * inputs_ + 1; for (size_t i = 1; i < hc.size(); ++i) routes_[i] = inputs_ & (1u << (i - 1)) ? 72 : 0; }
    const auto& previous = previous_.is_null() ? ready_ : previous_;
    for (size_t i = 0; i < hc.size(); ++i) { const auto old = integer(get(previous, "hc_exact_routes"), hc[i]), now = integer(get(v, "hc_exact_routes"), hc[i]);
        need(now >= old && (reset_resync_ ? now - old >= routes_[i] : now - old == routes_[i]) &&
            integer(get(v, "hc_exact_request_routes"), hc[i]) == routes_[i], "private V8 completed HC routes"); }
    zero_routes(v, "wide_q8_exact_request_routes", 4); zero_routes(v, "wide_q8_many_request_routes", 6);
    need(integer(v, "lookup_rounds") == rounds_ && integer(v, "lookup_verified_rounds") == verified_rounds_ && integer(v, "lookup_recovery_rounds") == recovery_rounds_ &&
        integer(v, "lookup_proposed_tokens") == proposed_total_ && integer(v, "lookup_accepted_tokens") == count_ - ordinary_total_ && integer(v, "lookup_ordinary_tokens") == ordinary_total_, "private V8 round summary");
    const auto epoch = previous_.is_null() ? 1 : integer(previous_, "lookup_history_epoch");
    need(epoch < std::numeric_limits<uint64_t>::max() && (reset_resync_ ? integer(v, "lookup_history_epoch") >= epoch + 1 :
        integer(v, "lookup_history_epoch") == epoch + reset_), "private V8 history epoch");
    for (const auto& [all, delta] : {std::pair{"lookup_appended_tokens", "lookup_request_appended_tokens"}, {"lookup_indexed_grams", "lookup_request_indexed_grams"},
        {"lookup_queries", "lookup_request_queries"}, {"lookup_resets", "lookup_request_resets"}}) {
        const auto old = previous_.is_null() ? 0 : integer(previous_, all), now = integer(v, all);
        need(now >= old && (reset_resync_ ? now - old >= integer(v, delta) : now - old == integer(v, delta)), "private V8 history cost delta");
    }
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    if (branch8_) {
        for (const auto name : {"lookup_truncated_grams", "lookup_cleared_grams", "lookup_truncations"})
            need(integer(v, name) == integer(previous_, name), "private branch8 model work altered ancestry-maintenance costs");
        for (const auto name : {"lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries"})
            need(integer(v, name) >= integer(previous_, name), "private branch8 lookup execution costs rewound");
    }
#endif
    const auto grams = [](uint32_t n) { return n < 16 ? 0u : n - 15; };
    need(integer(v, "lookup_request_appended_tokens") == inputs_ + count_ && integer(v, "lookup_request_indexed_grams") == grams(position_) - grams(base_) &&
        integer(v, "lookup_request_queries") == rounds_ && integer(v, "lookup_request_resets") == reset_, "private V8 committed history work");
    close(real(v, "lookup_round_wall_ms"), round_wall_); close(real(v, "generation_batch_wall_ms"), batch_wall_); close(real(v, "generation_batch_gpu_ms"), batch_gpu_);
    need(real(v, "prefill_wall_ms") >= 0 && real(v, "request_wall_ms") > 0 &&
        real(v, "prefill_wall_ms") + round_wall_ <= real(v, "request_wall_ms") + 1e-6 && get(v, "gpu_profile").is_null(), "private V8 request clock/profile");
    const auto& top = get(v, "final_logit_top8").array(); need(top.size() == 8, "private V8 final top8");
    double last = std::numeric_limits<double>::infinity(); uint64_t last_id = 0;
    for (size_t i = 0; i < top.size(); ++i) { keys(top[i], {"id", "logit"}); const auto id = integer(top[i], "id"); const auto score = real(top[i], "logit");
        need(id < 248320 && (score < last || (i && score == last && id > last_id)), "private V8 sorted unique final logits");
        for (size_t j = 0; j < i; ++j) need(integer(top[j], "id") != id, "private V8 duplicate final ID"); last = score; last_id = id; }
    finished_ = true;
}

std::span<const Json> Verify8Reply::events() const { need(finished_, "private V8 events not admitted"); return {events_.data(), event_count_}; }
} // namespace tokspur
#endif
