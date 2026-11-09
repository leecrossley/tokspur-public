#include "tokspur/mtp_wire.hpp"
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#include "tokspur/gguf.hpp"
#include "tokspur/raw_decode_wire.hpp"
#include "tokspur/serve.hpp"
#include "tokspur/serve_verify8.hpp"
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#include "tokspur/owned_worker.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>

namespace tokspur {
namespace {
void need(bool value, const char* message) { if (!value) throw FormatError(message); }
const Json& get(const Json& value, std::string_view name) {
    const auto* field = value.find(name); if (!field) throw FormatError("missing MTP wire field: " + std::string(name)); return *field;
}
uint64_t integer(const Json& value, std::string_view name) { return get(value, name).unsigned_integer(); }
double real(const Json& value, std::string_view name) {
    const auto text = get(value, name).number_lexeme(); double out = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), out);
    need(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(out), "nonfinite MTP wire number"); return out;
}
void keys(const Json& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& [name, ignored] : value.object()) {
        static_cast<void>(ignored);
        need(std::find(allowed.begin(), allowed.end(), name) != allowed.end(), "unknown MTP event/command field");
    }
}
constexpr std::array resources{"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes", "gpu_allocation_count",
    "gpu_buffer_limit_bytes", "imported_weight_backing_bytes", "native_mtp_workspace_bytes"};
constexpr std::array counters{"native_mtp_head_calls", "native_mtp_prime_calls", "native_mtp_rounds", "native_mtp_proposed",
    "native_mtp_accepted", "native_mtp_recovery_rounds", "native_mtp_carried_anchors", "native_mtp_discarded_anchors"};
constexpr std::array clocks{"native_mtp_head_gpu_ms", "native_mtp_head_wall_ms", "native_mtp_prime_gpu_ms", "native_mtp_prime_wall_ms"};
constexpr std::array hc_names{"scalar", "many1", "many2", "many4", "many8"};
constexpr std::array immutable{"native_mtp_protocol", "native_mtp_depth", "native_mtp_attention", "native_mtp_first_cache_position",
    "native_mtp_lookahead_anchor", "native_mtp_visible_frontier", "small_gemv_column_splits", "block_gemv_formats",
    "known_token_capacity", "known_residual_reuse", "known_qsa_rows_schema", "known_qsa_rows_mode", "known_qsa_rows_logical_bytes",
    "known_qsa_rows_charged_bytes", "known_qsa_rows_allocations", "known_qsa_rows_http_admitted", "ple_gate_transform", "ple_gate_transform_schema",
    "qsa_channel_chronological_mode", "qsa_channel_chronological_schema", "qsa_channel_factor_logical_bytes", "qsa_channel_factor_charged_bytes",
    "qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes", "qsa_stage_v8_scratch_charged_bytes",
    "hc_exact_mode", "hc_exact_rows_per_group", "hc_exact_unroll", "hc_exact_scratch_logical_bytes", "hc_exact_scratch_charged_bytes",
    "hc_q8_splits", "hc_q8_scratch_logical_bytes", "hc_q8_scratch_charged_bytes", "hc_q8_numerical_tier", "wide_q8_exact_enabled",
    "wide_q8_many_enabled", "stack_execution", "replay_prepared", "checkpoint_slots", "checkpoint_combined_budget_bytes",
    "checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes", "native_limit_bytes", "mapped_weight_region_limit_bytes",
    "capture_resource_bytes", "capture_binding_aliases", "capture_binding_alias_span_bytes", "capture_binding_alias_resource_bytes",
    "live_captures", "profile_counter_payload_bytes", "verification_max_tokens", "verification_head_many", "verification8_branch_reuse"};
void final_logits_state(const Json& value, bool available) {
    need(get(value, "logits_available").boolean() == available, "MTP logits readiness differs");
    const auto& hash = get(value, "hc_exact_final_logit_sha256"); const auto& top = get(value, "final_logit_top8");
    if (!available) { need(hash.is_null() && top.is_null(), "MTP unready input claimed computed logits"); return; }
    const auto digest = hash.string();
    need(digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }), "MTP ready-logits hash differs");
    need(top.array().size() == 8, "MTP ready-logits top8 differs");
    for (const auto& item : top.array()) { need(integer(item, "id") < 248320, "MTP ready-logits ID bound"); static_cast<void>(real(item, "logit")); }
}
void fixed(const Json& value) {
    static_cast<void>(raw_decode_selection(value));
    const auto input_packet = mtp_input_packet(value);
    static_cast<void>(mtp_final_input_logits(value));
    static_cast<void>(mtp_gdn_old_order(value));
    static_cast<void>(mtp_hc_down_f32(value));
    static_cast<void>(mtp_hc_rows(value));
    static_cast<void>(mtp_router_window(value));
    static_cast<void>(mtp_hc_down_window(value));
    static_cast<void>(mtp_qsa_sparse(value));
    static_cast<void>(mtp_gdn_ported(value));
    static_cast<void>(mtp_output_packet(value));
    static_cast<void>(mtp_draft_attention(value));
    static_cast<void>(mtp_request_carry(value));
    const auto depth = mtp_depth(value);
    const auto proto = get(value, "native_mtp_protocol").string();
    const bool proto_ok = depth == 7 ? (proto == "native-dflash2-depth7-lookahead-v2") :
        depth == 3 ? (proto == "native-mtp-depth3-lookahead-v2") :
        depth == 2 ? (proto == "native-mtp-depth2-lookahead-v1" || proto == "native-mtp-depth2-lookahead-v2") :
        (proto == "native-mtp-depth1-lookahead-v2");
    const bool attn_ok = depth == 7 ?
        (get(value, "native_mtp_attention").string() == "sliding-window-2048" && integer(value, "native_mtp_first_cache_position") == 0) :
        (get(value, "native_mtp_attention").string() == "dense-compression0" && integer(value, "native_mtp_first_cache_position") == 1);
    need(proto_ok && get(value, "native_mtp_lookahead_anchor").boolean() && attn_ok &&
        get(value, "native_mtp_visible_frontier").string() ==
            "all emitted tokens physically consumed; sampled terminal carried invisibly within request", "MTP wire protocol/policy differs");
    need((depth == 3 || depth == 7) ? (get(value, "native_mtp_tail_policy").string() == "commit-through; round overflow drained across packets") :
        depth == 2 ? (proto == "native-mtp-depth2-lookahead-v2" ?
            get(value, "native_mtp_tail_policy").string() == "commit-through; round overflow drained across packets" :
            get(value, "native_mtp_tail_policy").string() == "remaining1-or2 ordinary; no depth-one rounds") :
        !value.find("native_mtp_tail_policy"), "MTP tail policy differs");
    need(integer(value, "verification_max_tokens") == depth + 1, "MTP verification width differs from depth");
    need(verify8_fast_math(value) == std::optional<bool>(true), "MTP requires corrected fixed channel math");
    need(get(value, "stack_execution").string() == "direct" && get(value, "known_residual_reuse").boolean() &&
        !get(value, "replay_prepared").boolean() && !get(value, "wide_q8_exact_enabled").boolean() &&
        !get(value, "wide_q8_many_enabled").boolean() && !get(value, "verification8_branch_reuse").boolean() &&
        !get(value, "verification_head_many").boolean() && !get(value, "timings_qualified").boolean() &&
        get(value, "hc_exact_mode").string() == "combined" && get(value, "hc_q8_numerical_tier").string() == "ordinary",
        "MTP execution/numerical policy differs");
    for (const auto& [name, expected] : {std::pair{"small_gemv_column_splits", 16ull}, {"block_gemv_formats", 3ull},
        {"hc_exact_rows_per_group", 1ull}, {"hc_exact_unroll", 16ull},
        {"native_limit_bytes", 96ull << 30}, {"known_qsa_rows_schema", 1ull}})
        need(integer(value, name) == expected, "MTP fixed startup geometry differs");
    need(integer(value, "known_token_capacity") == std::min(input_packet, 2048u), "MTP known-input capacity differs from negotiated bound");
    for (const auto name : {"hc_q8_splits", "hc_q8_scratch_logical_bytes", "hc_q8_scratch_charged_bytes", "hc_exact_scratch_logical_bytes",
        "hc_exact_scratch_charged_bytes", "checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes",
        "checkpoint_store_host_bytes", "checkpoint_gpu_bytes", "recurrent_replay_captures", "mapped_weight_region_limit_bytes",
        "capture_resource_bytes", "capture_binding_aliases", "capture_binding_alias_span_bytes", "capture_binding_alias_resource_bytes",
        "live_captures", "profile_counter_payload_bytes"}) need(integer(value, name) == 0, "MTP unrelated prepared feature");
    need(!value.find("sampled_prefill_protocol") && !value.find("broker_protocol") && !value.find("branch8_session") &&
        !value.find("licensed_q4_routed_mode") && !value.find("batch_phase_clock_mode"), "MTP refuses another wire/tier");
    need(get(value, "known_qsa_rows_mode").string() == (input_packet >= 512 ? "off" : "q8") && !get(value, "known_qsa_rows_http_admitted").boolean(),
        "MTP must use distinct native known-row consent");
    const auto logical = integer(value, "known_qsa_rows_logical_bytes"), charged = integer(value, "known_qsa_rows_charged_bytes");
    if (input_packet >= 512) need(!logical && !charged && !integer(value, "known_qsa_rows_allocations"), "MTP input512 cannot prepare input-eight row storage");
    else need(logical && logical <= charged && charged <= (64ull << 20) && integer(value, "known_qsa_rows_allocations") == 46, "MTP known-row workspace bound");
    const auto live = integer(value, "tracked_metal_buffer_bytes"), peak = integer(value, "peak_metal_buffer_bytes");
    const auto budget = integer(value, "gpu_buffer_limit_bytes"), workspace = integer(value, "native_mtp_workspace_bytes");
    need(live && live <= peak && peak <= budget && budget == (96ull << 30) - (512ull << 20) - (2ull << 30) &&
        integer(value, "gpu_allocation_count") && integer(value, "imported_weight_backing_bytes") <= live &&
        workspace && workspace <= (512ull << 20), "MTP fixed GPU resource ledger");
    for (const auto name : {"wide_q8_exact_routes", "wide_q8_many_routes"})
        for (const auto& [key, count] : get(value, name).object()) { static_cast<void>(key); need(!count.unsigned_integer(), "MTP unexpected wide route"); }
    need(get(value, "hc_exact_routes").object().size() == hc_names.size(), "MTP HC route shape");
    for (const auto name : hc_names) static_cast<void>(integer(get(value, "hc_exact_routes"), name));
    for (const auto name : counters) static_cast<void>(integer(value, name));
    for (const auto name : clocks) need(real(value, name) >= 0, "MTP negative clock");
}
bool stop(uint32_t id, const Json& job) {
    const auto* ids = job.find("stop_ids"); if (!ids) return false;
    for (const auto& item : ids->array()) if (item.unsigned_integer() == id) return true;
    return false;
}
void delta(const Json& previous, const Json& done, const char* name, uint64_t expected, bool resync) {
    const auto before = integer(previous, name), after = integer(done, name);
    need(after >= before && (resync ? after - before >= expected : after - before == expected), "MTP lifetime counter differs");
}
} // namespace

std::string_view mtp_draft_attention(const Json& value) {
    const auto* field = value.find("native_mtp_draft_attention");
    const auto mode = field ? field->string() : std::string_view("serial");
    need(mode == "serial" || mode == "tiled128" || mode == "tiled256" || mode == "split16", "unknown MTP draft-attention mode");
    return mode;
}

uint32_t mtp_depth(const Json& value) {
    const auto depth = integer(value, "native_mtp_depth");
    need(depth == 1
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        || depth == 2 || depth == 3 || depth == 7
#endif
        , "unsupported native MTP depth");
    return uint32_t(depth);
}

uint32_t mtp_output_packet(const Json& value) {
    const auto* cap = value.find("native_mtp_output_packet");
    const auto* schema = value.find("native_mtp_output_packet_schema");
    for (const auto& [name, ignored] : value.object()) {
        static_cast<void>(ignored);
        if (name.starts_with("native_mtp_output_packet"))
            need(name == "native_mtp_output_packet" || name == "native_mtp_output_packet_schema", "unknown MTP output-packet metadata");
    }
    need(bool(cap) == bool(schema), "MTP output-packet metadata is incomplete");
    if (!cap) return 8;
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    const auto limit = cap->unsigned_integer();
    need(schema->unsigned_integer() == 1 && (limit == 8 || limit == 64), "MTP output-packet metadata bound/schema");
    need(mtp_request_carry(value) && (((mtp_depth(value) == 2 || mtp_depth(value) == 3) && mtp_draft_attention(value) == "tiled128") ||
        (mtp_depth(value) == 7 && mtp_draft_attention(value) == "split16")),
        "MTP output packets require fixed depth-two/three tiled128 or depth-seven split16 scoped carry");
    return uint32_t(limit);
#else
    throw FormatError("MTP output-packet extension is not compiled");
#endif
}
void validate_mtp_output_packet_selection(const Json& value, std::optional<uint32_t> expected) {
    const auto limit = mtp_output_packet(value);
    need(bool(value.find("native_mtp_output_packet")) == expected.has_value() && (!expected || limit == *expected),
        "MTP output-packet metadata differs from startup consent");
}

uint32_t mtp_input_packet(const Json& value) {
    const auto* cap = value.find("native_mtp_input_packet_max");
    const auto* schema = value.find("native_mtp_input_packet_schema");
    const auto* grouped = value.find("native_prefill_grouped");
    const auto* grouped_kernel = value.find("native_prefill_grouped_kernel");
    const auto* dense_kernel = value.find("native_prefill_dense_kernel");
    constexpr std::array<std::string_view, 3> priming_fields{"native_mtp_priming_batch_capacity", "native_mtp_priming_batch_bytes", "native_mtp_priming_batch_submits"};
    size_t priming_count = 0;
    for (const auto& [name, ignored] : value.object()) {
        static_cast<void>(ignored);
        if (name.starts_with("native_mtp_input_packet"))
            need(name == "native_mtp_input_packet_max" || name == "native_mtp_input_packet_schema", "unknown MTP input-packet metadata");
        if (name.starts_with("native_mtp_priming_batch")) {
            need(std::find(priming_fields.begin(), priming_fields.end(), name) != priming_fields.end(), "unknown MTP priming-batch metadata"); ++priming_count;
        }
        if (name.starts_with("native_prefill_dense"))
            need(name == "native_prefill_dense" || name == "native_prefill_dense_kernel", "unknown MTP dense-prefill metadata");
        if (name.starts_with("native_prefill_grouped"))
            need(name == "native_prefill_grouped" || name == "native_prefill_grouped_kernel", "unknown MTP grouped-prefill metadata");
    }
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    if (dense_kernel) {
        const auto dense_mode = get(value, "native_prefill_dense").string();
        const bool dense_pair_ok = (dense_mode == "f32" && dense_kernel->string() == "graph_dense_q8_staged_f32") ||
            (dense_mode == "nax" && dense_kernel->string() == "nax_prefill_dense_q8_m32");
        need(cap && dense_pair_ok && grouped && (grouped->string() == "f32" || grouped->string() == "auto" || grouped->string() == "nax") &&
            get(value, "native_prefill_chunks").string() == "fused",
            "MTP staged/NAX dense kernel requires fixed fused input512/2048 compatible grouped mode");
        for (const auto& [name, ignored] : value.object()) {
            static_cast<void>(ignored); need(!name.starts_with("native_prefill_hc_down_f32"), "MTP staged dense kernel refuses HC down metadata");
        }
    }
#else
    need(!dense_kernel, "MTP staged dense extension is not compiled");
#endif
    need(bool(grouped) == bool(grouped_kernel), "MTP grouped-prefill metadata is incomplete");
    if (grouped) {
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
        need(cap && (grouped->string() == "serial" || grouped->string() == "f32" || grouped->string() == "auto" || grouped->string() == "nax"), "MTP grouped prefill requires input512 and serial, f32, auto or nax");
        need(grouped_kernel->string() == (grouped->string() == "nax" ? "nax_grouped_moe" : grouped->string() == "auto" ? "graph_grouped_mm" : grouped->string() == "f32" ? "graph_grouped_mm_f32" : "graph_grouped_serial"),
            "MTP grouped-prefill kernel differs from its mode");
#else
        throw FormatError("MTP grouped FP32 extension is not compiled");
#endif
    }
    need(bool(cap) == bool(schema), "MTP input-packet metadata is incomplete");
    if (!cap) { need(!priming_count, "MTP batch priming requires explicit input512 metadata"); return 8; }
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    const auto max_cap = cap->unsigned_integer();
    need(schema->unsigned_integer() == 1 && (max_cap == 512 || max_cap == 1024 || max_cap == 2048 || max_cap == 4096 || max_cap == 8192), "MTP input-packet metadata bound/schema");
    need(mtp_request_carry(value) && (((mtp_depth(value) == 2 || mtp_depth(value) == 3) && mtp_draft_attention(value) == "tiled128") ||
        (mtp_depth(value) == 7 && mtp_draft_attention(value) == "split16")),
        "MTP input512 requires fixed depth-two/three tiled128 or depth-seven split16 scoped carry");
    const auto dense = get(value, "native_prefill_dense").string();
    need(!value.find("weight_lanes_schema") && (dense == "f16" || dense == "f32" || (dense == "nax" && dense_kernel)), "MTP input512 requires original weight views and explicit dense precision metadata");
    if (max_cap > 2048) {
        const auto* overlap = value.find("exact_candidate_ple_overlap");
        // The worker admits an external 8192-token packet only as 2048-token NAX tiles with PLE overlap.
        need(max_cap == 8192 && integer(value, "known_token_capacity") == 2048 && dense == "nax" && grouped && grouped->string() == "nax" &&
            overlap && overlap->kind() == Json::Kind::boolean && overlap->boolean(), "MTP expanded input packet requires 2048-token NAX tiles with PLE overlap");
    }
    const bool nax_priming = dense == "nax" || (grouped && grouped->string() == "nax");
    const uint64_t expected_priming_cap = (mtp_depth(value) == 7 || nax_priming) ? std::min<uint64_t>(max_cap, 2048) : std::min<uint64_t>(max_cap, 512);
    const uint64_t priming_max = (expected_priming_cap >= 1024 || dense == "nax") ? (160ull << 20) : (32ull << 20);
    need(priming_count == priming_fields.size() && integer(value, priming_fields[0]) == expected_priming_cap &&
        integer(value, priming_fields[1]) && integer(value, priming_fields[1]) <= priming_max, "MTP priming-batch capacity/workspace bound");
    static_cast<void>(integer(value, priming_fields[2]));
    return uint32_t(max_cap);
#else
    throw FormatError("MTP input512 extension is not compiled");
#endif
}

void validate_mtp_input_packet_selection(const Json& value, bool expected_input512) {
    need((mtp_input_packet(value) >= 512) == expected_input512, "MTP input-packet metadata differs from startup consent");
}

bool mtp_gdn_old_order(const Json& value) {
    constexpr std::array<std::string_view, 6> names{"native_gdn_old_order_version", "native_gdn_old_order_kernel", "native_gdn_old_order_layers", "native_gdn_old_order_scratch_bytes", "native_gdn_old_order_encoded_calls", "native_gdn_old_order_request_encoded_calls"};
    size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_gdn_old_order_")) {
        static_cast<void>(ignored); need(std::find(names.begin(), names.end(), name) != names.end(), "unknown old-order GDN metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    need(count == names.size() && mtp_input_packet(value) == 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        get(value, "native_prefill_dense").string() == "f32" && get(value, "native_prefill_grouped").string() == "f32", "old-order GDN route or metadata differs");
    need(get(value, names[0]).string() == "old-order-batch-v1" && get(value, names[1]).string() == "gdn_chunk_legacy_batch" &&
        integer(value, names[2]) == 36 && !integer(value, names[3]), "old-order GDN immutable version/kernel/resources differ");
    need(integer(value, names[5]) <= integer(value, names[4]), "old-order GDN encoded call bounds differ"); return true;
#else
    throw FormatError("old-order GDN metadata is not compiled");
#endif
}
void validate_mtp_gdn_old_order_selection(const Json& value, bool expected) {
    need(mtp_gdn_old_order(value) == expected, "old-order GDN metadata differs from startup consent");
}

Json::Object mtp_hc_rows_fields(std::array<uint64_t, 3> encoded, std::array<uint64_t, 3> request) {
    Json::Object fields{{"native_prefill_hc_rows_schema", Json::integer(1)},
        {"native_prefill_hc_rows_version", Json("hc-rows-split16-exact-v1")},
        {"native_prefill_hc_rows_kernels", Json("graph_hc_rows_inject_partials,graph_hc_rows_inject_reduce,graph_hc_rows_read,graph_hc_rows_write")},
        {"native_prefill_hc_rows_precision", Json("fp32-split16-exact")},
        {"native_prefill_hc_rows_min_rows", Json::integer(2)}, {"native_prefill_hc_rows_max_rows", Json::integer(512)},
        {"native_prefill_hc_rows_branches", Json::integer(4)}, {"native_prefill_hc_rows_channels", Json::integer(2560)},
        {"native_prefill_hc_rows_scratch_bytes", Json::integer(131072)},
        {"native_prefill_hc_rows_counter_scope", Json("encoded-inject-read-write-calls-including-aborted-submissions")}};
    for (size_t i = 0; i < 3; ++i) { fields.emplace_back(mtp_hc_rows_counter_names[i], Json::integer(encoded[i])); fields.emplace_back(mtp_hc_rows_request_names[i], Json::integer(request[i])); }
    return fields;
}
bool mtp_hc_rows(const Json& value) {
    const auto expected = Json(mtp_hc_rows_fields()); size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_prefill_hc_rows")) {
        static_cast<void>(ignored); need(expected.find(name), "unknown HC rows metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    const auto grouped_mode = get(value, "native_prefill_grouped").string();
    const bool grouped_ok = (grouped_mode == "f32" || grouped_mode == "auto" || grouped_mode == "nax");
    const auto dense_mode = get(value, "native_prefill_dense").string();
    const auto dense_k = get(value, "native_prefill_dense_kernel").string();
    const bool dense_ok = (dense_mode == "f32" && dense_k == "graph_dense_q8_staged_f32") || (dense_mode == "nax" && dense_k == "nax_prefill_dense_q8_m32");
    need(count == expected.object().size() && mtp_input_packet(value) >= 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        dense_ok && grouped_ok && !mtp_hc_down_f32(value), "HC rows require staged FP32 or NAX and scalar HC down");
    for (const auto name : mtp_hc_rows_identity_names) need(get(value, name).dump() == get(expected, name).dump(), "HC rows immutable identity differs");
    for (size_t i = 0; i < 3; ++i) need(integer(value, mtp_hc_rows_request_names[i]) <= integer(value, mtp_hc_rows_counter_names[i]), "HC rows encoded bounds differ");
    return true;
#else
    throw FormatError("HC rows serving metadata is not compiled");
#endif
}
void validate_mtp_hc_rows_selection(const Json& value, bool expected) { need(mtp_hc_rows(value) == expected, "HC rows differ from startup consent"); }
Json::Object mtp_router_window_fields(std::array<uint64_t, 4> encoded, std::array<uint64_t, 4> request) {
    Json::Object fields{{"native_prefill_router_window_schema", Json::integer(1)},
        {"native_prefill_router_window_version", Json("router-window-f32-exact-v1")},
        {"native_prefill_router_window_kernels", Json("graph_moe_router_f32_window,graph_moe_route_rows,graph_moe_shared_router_rows,graph_moe_mixture_rows")},
        {"native_prefill_router_window_precision", Json("fp32-simd32-exact")},
        {"native_prefill_router_window_min_rows", Json::integer(2)}, {"native_prefill_router_window_max_rows", Json::integer(512)},
        {"native_prefill_router_window_window_rows", Json::integer(16)}, {"native_prefill_router_window_channels", Json::integer(2560)},
        {"native_prefill_router_window_experts", Json::integer(512)}, {"native_prefill_router_window_scratch_bytes", Json::integer(0)},
        {"native_prefill_router_window_counter_scope", Json("encoded-router-route-shared-mixture-calls-including-aborted-submissions")}};
    for (size_t i = 0; i < 4; ++i) { fields.emplace_back(mtp_router_window_counter_names[i], Json::integer(encoded[i])); fields.emplace_back(mtp_router_window_request_names[i], Json::integer(request[i])); }
    return fields;
}
bool mtp_router_window(const Json& value) {
    const auto expected = Json(mtp_router_window_fields()); size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_prefill_router_window")) {
        static_cast<void>(ignored); need(expected.find(name), "unknown router window metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    const auto grouped_mode = get(value, "native_prefill_grouped").string();
    const bool grouped_ok = (grouped_mode == "f32" || grouped_mode == "auto" || grouped_mode == "nax");
    const auto dense_mode = get(value, "native_prefill_dense").string();
    const auto dense_k = get(value, "native_prefill_dense_kernel").string();
    const bool dense_ok = (dense_mode == "f32" && dense_k == "graph_dense_q8_staged_f32") || (dense_mode == "nax" && dense_k == "nax_prefill_dense_q8_m32");
    need(count == expected.object().size() && mtp_input_packet(value) >= 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        dense_ok && grouped_ok && !mtp_hc_down_f32(value), "router window requires staged FP32 or NAX and scalar HC down");
    for (const auto name : mtp_router_window_identity_names) need(get(value, name).dump() == get(expected, name).dump(), "router window immutable identity differs");
    for (size_t i = 0; i < 4; ++i) need(integer(value, mtp_router_window_request_names[i]) <= integer(value, mtp_router_window_counter_names[i]), "router window encoded bounds differ");
    return true;
#else
    throw FormatError("router window serving metadata is not compiled");
#endif
}
void validate_mtp_router_window_selection(const Json& value, bool expected) { need(mtp_router_window(value) == expected, "router window differs from startup consent"); }
Json::Object mtp_hc_down_window_fields(uint64_t encoded, uint64_t request) {
    return Json::Object{{"native_prefill_hc_down_window_schema", Json::integer(1)},
        {"native_prefill_hc_down_window_version", Json("hc-down-window-q8-exact-v1")},
        {"native_prefill_hc_down_window_kernel", Json("graph_hc_down_q8_window")},
        {"native_prefill_hc_down_window_precision", Json("q8-0-simd32-exact")},
        {"native_prefill_hc_down_window_min_rows", Json::integer(2)}, {"native_prefill_hc_down_window_max_rows", Json::integer(512)},
        {"native_prefill_hc_down_window_window_rows", Json::integer(16)}, {"native_prefill_hc_down_window_columns", Json::integer(10240)},
        {"native_prefill_hc_down_window_rows", Json::integer(320)}, {"native_prefill_hc_down_window_threads", Json::integer(512)},
        {"native_prefill_hc_down_window_threadgroup_bytes", Json::integer(32768)},
        {"native_prefill_hc_down_window_counter_scope", Json("encoded-hc-down-window-calls-including-aborted-submissions")},
        {mtp_hc_down_window_counter_name, Json::integer(encoded)}, {mtp_hc_down_window_request_name, Json::integer(request)}};
}
bool mtp_hc_down_window(const Json& value) {
    const auto expected = Json(mtp_hc_down_window_fields()); size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_prefill_hc_down_window")) {
        static_cast<void>(ignored); need(expected.find(name), "unknown HC down window metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    const auto grouped_mode = get(value, "native_prefill_grouped").string();
    const bool grouped_ok = (grouped_mode == "f32" || grouped_mode == "auto" || grouped_mode == "nax");
    const auto dense_mode = get(value, "native_prefill_dense").string();
    const auto dense_k = get(value, "native_prefill_dense_kernel").string();
    const bool dense_ok = (dense_mode == "f32" && dense_k == "graph_dense_q8_staged_f32") || (dense_mode == "nax" && dense_k == "nax_prefill_dense_q8_m32");
    need(count == expected.object().size() && mtp_input_packet(value) >= 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        dense_ok && grouped_ok && mtp_hc_rows(value) && mtp_router_window(value) && !mtp_hc_down_f32(value),
        "HC down window requires staged FP32 or NAX, exact HC rows, exact router window and reassociating HC down off");
    for (const auto name : mtp_hc_down_window_identity_names) need(get(value, name).dump() == get(expected, name).dump(), "HC down window immutable identity differs");
    need(integer(value, mtp_hc_down_window_request_name) <= integer(value, mtp_hc_down_window_counter_name), "HC down window encoded bounds differ");
    return true;
#else
    throw FormatError("HC down window serving metadata is not compiled");
#endif
}
void validate_mtp_hc_down_window_selection(const Json& value, bool expected) { need(mtp_hc_down_window(value) == expected, "HC down window differs from startup consent"); }
Json::Object mtp_qsa_sparse_fields(std::array<uint64_t, 2> completed, std::array<uint64_t, 2> request) {
    constexpr uint64_t decode_bn = TOKSPUR_QSA_SPARSE_DECODE_BN;
    constexpr uint64_t verify_bn = TOKSPUR_QSA_SPARSE_VERIFY_BN;
    return Json::Object{{"native_qsa_sparse_schema", Json::integer(1)},
        {"native_qsa_sparse_version", Json("qsa-sparse-online-softmax-v1")},
        {"native_qsa_sparse_decode_kernel", Json("qsa_sparse_attention")},
        {"native_qsa_sparse_verify_kernel", Json("qsa_sparse_attention_rows")},
        {"native_qsa_sparse_precision", Json("fp32-online-softmax-bf16-kv")},
        {"native_qsa_sparse_decode_bn", Json::integer(decode_bn)},
        {"native_qsa_sparse_verify_bn", Json::integer(verify_bn)},
        {"native_qsa_sparse_decode_threads", Json::integer(32 * decode_bn)},
        {"native_qsa_sparse_verify_threads", Json::integer(32 * verify_bn)},
        {"native_qsa_sparse_decode_threadgroup_bytes", Json::integer((decode_bn * 33 + 3 * decode_bn) * 4)},
        {"native_qsa_sparse_verify_threadgroup_bytes", Json::integer((verify_bn * 33 + 3 * verify_bn) * 4)},
        {"native_qsa_sparse_counter_scope", Json("completed-physical-dispatches-aborted-excluded")},
        {mtp_qsa_sparse_counter_names[0], Json::integer(completed[0])},
        {mtp_qsa_sparse_counter_names[1], Json::integer(completed[1])},
        {mtp_qsa_sparse_request_names[0], Json::integer(request[0])},
        {mtp_qsa_sparse_request_names[1], Json::integer(request[1])}};
}
bool mtp_qsa_sparse(const Json& value) {
    size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_qsa_sparse_")) {
        static_cast<void>(ignored);
        const bool known = std::find(mtp_qsa_sparse_identity_names.begin(), mtp_qsa_sparse_identity_names.end(), name) != mtp_qsa_sparse_identity_names.end() ||
            std::find(mtp_qsa_sparse_counter_names.begin(), mtp_qsa_sparse_counter_names.end(), name) != mtp_qsa_sparse_counter_names.end() ||
            std::find(mtp_qsa_sparse_request_names.begin(), mtp_qsa_sparse_request_names.end(), name) != mtp_qsa_sparse_request_names.end();
        need(known, "unknown sparse QSA metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    const Json expected(mtp_qsa_sparse_fields());
    const auto grouped_mode = get(value, "native_prefill_grouped").string();
    const bool grouped_ok = (grouped_mode == "f32" || grouped_mode == "auto" || grouped_mode == "nax");
    const auto dense_mode = get(value, "native_prefill_dense").string();
    const auto dense_k = get(value, "native_prefill_dense_kernel").string();
    const bool dense_ok = (dense_mode == "f32" && dense_k == "graph_dense_q8_staged_f32") || (dense_mode == "nax" && dense_k == "nax_prefill_dense_q8_m32");
    need(count == mtp_qsa_sparse_identity_names.size() + 4 && mtp_input_packet(value) >= 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        dense_ok && grouped_ok && mtp_hc_rows(value) && mtp_router_window(value) && mtp_hc_down_window(value) && !mtp_hc_down_f32(value),
        "sparse QSA requires staged FP32 or NAX, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
    for (const auto name : mtp_qsa_sparse_identity_names) need(get(value, name).dump() == get(expected, name).dump(), "sparse QSA immutable identity differs");
    for (size_t i = 0; i < 2; ++i) need(integer(value, mtp_qsa_sparse_request_names[i]) <= integer(value, mtp_qsa_sparse_counter_names[i]), "sparse QSA completed bounds differ");
    return true;
#else
    throw FormatError("sparse QSA serving metadata is not compiled");
#endif
}
void validate_mtp_qsa_sparse_selection(const Json& value, bool expected) { need(mtp_qsa_sparse(value) == expected, "sparse QSA differs from startup consent"); }
Json::Object mtp_gdn_ported_fields(std::array<uint64_t, 5> completed, std::array<uint64_t, 5> request) {
    constexpr uint64_t verify_db = TOKSPUR_GDN_VERIFY_DB;
    constexpr uint64_t verify_stride = verify_db == 8 ? 10 : 20;
    Json::Object fields{{"native_gdn_ported_schema", Json::integer(1)},
        {"native_gdn_ported_version", Json("gdn-ported-f32-recurrence-v1")},
        {"native_gdn_ported_prefill_kernel", Json("gdn_pipelined_tb12")},
        {"native_gdn_ported_prepare_kernel", Json("gdn_prepare_rows")},
        {"native_gdn_ported_verify_kernel", Json("gdn_verify_main")},
        {"native_gdn_ported_replay_kernel", Json("gdn_verify_replay")},
        {"native_gdn_ported_norm_gate_kernel", Json("gdn_norm_gate")},
        {"native_gdn_ported_precision", Json("fp32-state-fp32-accum")},
        {"native_gdn_ported_prefill_tile_rows", Json::integer(12)},
        {"native_gdn_ported_verify_db", Json::integer(verify_db)},
        {"native_gdn_ported_prefill_threadgroup_bytes", Json::integer(14032)},
        {"native_gdn_ported_verify_threadgroup_bytes", Json::integer((128 * verify_stride + 4) * 4)},
        {"native_gdn_ported_counter_scope", Json("completed-physical-dispatches-aborted-excluded")}};
    for (size_t i = 0; i < 5; ++i) {
        fields.emplace_back(mtp_gdn_ported_counter_names[i], Json::integer(completed[i]));
        fields.emplace_back(mtp_gdn_ported_request_names[i], Json::integer(request[i]));
    }
    return fields;
}
bool mtp_gdn_ported(const Json& value) {
    size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_gdn_ported_")) {
        static_cast<void>(ignored);
        const bool known = std::find(mtp_gdn_ported_identity_names.begin(), mtp_gdn_ported_identity_names.end(), name) != mtp_gdn_ported_identity_names.end() ||
            std::find(mtp_gdn_ported_counter_names.begin(), mtp_gdn_ported_counter_names.end(), name) != mtp_gdn_ported_counter_names.end() ||
            std::find(mtp_gdn_ported_request_names.begin(), mtp_gdn_ported_request_names.end(), name) != mtp_gdn_ported_request_names.end();
        need(known, "unknown ported GDN metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    const Json expected(mtp_gdn_ported_fields());
    const auto grouped_mode = get(value, "native_prefill_grouped").string();
    const bool grouped_ok = (grouped_mode == "f32" || grouped_mode == "auto" || grouped_mode == "nax");
    const auto dense_mode = get(value, "native_prefill_dense").string();
    const auto dense_k = get(value, "native_prefill_dense_kernel").string();
    const bool dense_ok = (dense_mode == "f32" && dense_k == "graph_dense_q8_staged_f32") || (dense_mode == "nax" && dense_k == "nax_prefill_dense_q8_m32");
    need(count == mtp_gdn_ported_identity_names.size() + 10 && mtp_input_packet(value) >= 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        dense_ok && grouped_ok && mtp_hc_rows(value) && mtp_router_window(value) && mtp_hc_down_window(value) && !mtp_hc_down_f32(value),
        "ported GDN requires staged FP32 or NAX, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
    for (const auto name : mtp_gdn_ported_identity_names) need(get(value, name).dump() == get(expected, name).dump(), "ported GDN immutable identity differs");
    for (size_t i = 0; i < 5; ++i) need(integer(value, mtp_gdn_ported_request_names[i]) <= integer(value, mtp_gdn_ported_counter_names[i]), "ported GDN completed bounds differ");
    return true;
#else
    throw FormatError("ported GDN serving metadata is not compiled");
#endif
}
void validate_mtp_gdn_ported_selection(const Json& value, bool expected) { need(mtp_gdn_ported(value) == expected, "ported GDN differs from startup consent"); }
Json::Object mtp_hc_down_f32_fields(uint64_t encoded, uint64_t request) {
    return Json::Object{{"native_prefill_hc_down_f32_schema", Json::integer(1)},
        {"native_prefill_hc_down_f32_version", Json("hc-down-f32-full512-v1")},
        {"native_prefill_hc_down_f32_kernel", Json("graph_q8_mpp_wide")},
        {"native_prefill_hc_down_f32_precision", Json("fp32-reassociated")},
        {"native_prefill_hc_down_f32_layout", Json("raw-gguf-q8-0")},
        {"native_prefill_hc_down_f32_input_rows", Json::integer(512)},
        {"native_prefill_hc_down_f32_input_columns", Json::integer(10240)},
        {"native_prefill_hc_down_f32_output_columns", Json::integer(320)},
        {"native_prefill_hc_down_f32_branches", Json::integer(4)},
        {"native_prefill_hc_down_f32_scratch_bytes", Json::integer(0)},
        {"native_prefill_hc_down_f32_counter_scope", Json("encoded-calls-including-aborted-submissions")},
        {"native_prefill_hc_down_f32_encoded_calls", Json::integer(encoded)}, {"native_prefill_hc_down_f32_request_encoded_calls", Json::integer(request)}};
}
bool mtp_hc_down_f32(const Json& value) {
    size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_prefill_hc_down_f32_")) {
        static_cast<void>(ignored);
        need(std::find(mtp_hc_down_f32_identity_names.begin(), mtp_hc_down_f32_identity_names.end(), name) != mtp_hc_down_f32_identity_names.end() ||
            name == "native_prefill_hc_down_f32_encoded_calls" || name == "native_prefill_hc_down_f32_request_encoded_calls", "unknown HC down batching metadata"); ++count;
    }
    if (!count) return false;
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    need(count == mtp_hc_down_f32_identity_names.size() + 2 && mtp_input_packet(value) == 512 && get(value, "native_prefill_chunks").string() == "fused" &&
        get(value, "native_prefill_dense").string() == "f32" && get(value, "native_prefill_grouped").string() == "f32" &&
        get(value, "native_prefill_grouped_kernel").string() == "graph_grouped_mm_f32", "HC down batching route or precision differs");
    need(integer(value, "native_prefill_hc_down_f32_schema") == 1 &&
        get(value, "native_prefill_hc_down_f32_version").string() == "hc-down-f32-full512-v1" &&
        get(value, "native_prefill_hc_down_f32_kernel").string() == "graph_q8_mpp_wide" &&
        get(value, "native_prefill_hc_down_f32_precision").string() == "fp32-reassociated" &&
        get(value, "native_prefill_hc_down_f32_layout").string() == "raw-gguf-q8-0" &&
        integer(value, "native_prefill_hc_down_f32_input_rows") == 512 &&
        integer(value, "native_prefill_hc_down_f32_input_columns") == 10240 &&
        integer(value, "native_prefill_hc_down_f32_output_columns") == 320 &&
        integer(value, "native_prefill_hc_down_f32_branches") == 4 &&
        integer(value, "native_prefill_hc_down_f32_scratch_bytes") == 0 &&
        get(value, "native_prefill_hc_down_f32_counter_scope").string() == "encoded-calls-including-aborted-submissions", "HC down batching immutable identity differs");
    need(integer(value, "native_prefill_hc_down_f32_request_encoded_calls") <= integer(value, "native_prefill_hc_down_f32_encoded_calls"), "HC down batching encoded bounds differ"); return true;
#else
    throw FormatError("HC down batching metadata is not compiled");
#endif
}
void validate_mtp_hc_down_f32_selection(const Json& value, bool expected) { need(mtp_hc_down_f32(value) == expected, "HC down batching metadata differs from startup consent"); }

bool mtp_final_input_logits(const Json& value) {
    for (const auto& [name, ignored] : value.object()) {
        static_cast<void>(ignored);
        if (name.starts_with("native_prefill_output_policy")) need(name == "native_prefill_output_policy", "unknown MTP final-input metadata");
    }
    const auto* policy = value.find("native_prefill_output_policy");
    if (!policy) return false;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    need(mtp_input_packet(value) >= 512 && policy->string() == "final-input-v1", "MTP final-input policy differs");
    return true;
#else
    need(false, "MTP final-input policy requires input512 build"); return false;
#endif
}
void validate_mtp_final_input_selection(const Json& value, bool expected) {
    need(mtp_final_input_logits(value) == expected, "MTP final-input selection differs from startup consent");
}

#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
void validate_mtp_prefill_selection(const Json& value, std::optional<std::string> expected_dense, std::optional<std::string> expected_grouped) {
    const auto input512 = mtp_input_packet(value) >= 512;
    need((!expected_dense && !expected_grouped) || input512, "MTP prefill precision selection requires input512");
    const bool staged = expected_dense == "f32-staged", nax_dense = expected_dense == "nax";
    need(!expected_dense || *expected_dense == "f16" || *expected_dense == "f32" || staged || nax_dense, "unknown MTP dense backend consent");
#ifndef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    need(!staged && !nax_dense, "MTP staged/NAX dense backend consent is not compiled");
#endif
    if (input512) need(get(value, "native_prefill_dense").string() == (staged ? "f32" : expected_dense.value_or("f16")), "MTP dense-prefill precision differs from startup consent");
    need(bool(value.find("native_prefill_dense_kernel")) == (staged || nax_dense), "MTP dense-kernel presence differs from startup consent");
    const auto* grouped = value.find("native_prefill_grouped");
    need(bool(grouped) == expected_grouped.has_value() && (!grouped || grouped->string() == *expected_grouped),
        "MTP grouped-prefill precision differs from startup consent");
}
#endif

bool mtp_request_carry(const Json& value) {
    const auto* scope = value.find("native_mtp_carry_scope"); const auto* generation = value.find("native_mtp_generation");
    need(bool(scope) == bool(generation), "MTP generation metadata presence differs");
    if (!scope) return false;
    const auto scope_str = scope->string();
    need(scope_str == "http-generation-v1" || scope_str == "http-generation-v2", "unknown MTP carried-terminal scope");
    if (scope_str == "http-generation-v1") {
        keys(*generation, {"id", "active", "remaining", "pending"});
    } else {
        keys(*generation, {"id", "active", "remaining", "pending", "committed", "emitted", "overflow",
                           "remaining_compute", "total_budget", "continuation_allowed", "stop_queued",
                           "stop_drained", "stage_a_controller"});
    }
    static_cast<void>(integer(*generation, "id"));
    const auto active = get(*generation, "active").boolean(), pending = get(*generation, "pending").boolean();
    const auto remaining = integer(*generation, "remaining");
    need(active ? integer(*generation, "id") && remaining && remaining <= 32768 : !remaining && !pending,
        "invalid MTP generation lifetime metadata");
    return true;
}

void validate_mtp_ready(const Json& value, std::string_view expected_draft_attention, bool expected_request_carry, uint32_t expected_depth) {
    validate_worker_ready(value, "direct"); fixed(value);
    static_cast<void>(raw_decode_selection(value, true));
    if (mtp_final_input_logits(value)) need(!get(value, "logits_available").boolean(), "MTP ready contains logits");
    if (mtp_gdn_old_order(value)) need(!integer(value, "native_gdn_old_order_encoded_calls") && !integer(value, "native_gdn_old_order_request_encoded_calls"), "MTP ready already encoded old-order GDN calls");
    if (mtp_hc_down_f32(value)) need(!integer(value, "native_prefill_hc_down_f32_encoded_calls") && !integer(value, "native_prefill_hc_down_f32_request_encoded_calls"), "MTP ready already encoded HC down batching calls");
    if (mtp_hc_rows(value)) for (size_t i = 0; i < 3; ++i)
        need(!integer(value, mtp_hc_rows_counter_names[i]) && !integer(value, mtp_hc_rows_request_names[i]), "MTP ready already encoded HC rows");
    if (mtp_router_window(value)) for (size_t i = 0; i < 4; ++i)
        need(!integer(value, mtp_router_window_counter_names[i]) && !integer(value, mtp_router_window_request_names[i]), "MTP ready already encoded router window");
    if (mtp_hc_down_window(value))
        need(!integer(value, mtp_hc_down_window_counter_name) && !integer(value, mtp_hc_down_window_request_name), "MTP ready already encoded HC down window");
    if (mtp_qsa_sparse(value)) for (size_t i = 0; i < 2; ++i)
        need(!integer(value, mtp_qsa_sparse_counter_names[i]) && !integer(value, mtp_qsa_sparse_request_names[i]), "MTP ready already executed sparse QSA");
    if (mtp_gdn_ported(value)) for (size_t i = 0; i < 5; ++i)
        need(!integer(value, mtp_gdn_ported_counter_names[i]) && !integer(value, mtp_gdn_ported_request_names[i]), "MTP ready already executed ported GDN");
    need(mtp_depth(value) == expected_depth, "MTP depth startup selection differs");
    need((expected_depth != 2 && expected_depth != 3) || (expected_request_carry && mtp_draft_attention(value) == "tiled128"),
        "depth-two/three HTTP requires scoped carry and tiled128");
    need(expected_depth != 7 || (expected_request_carry && mtp_draft_attention(value) == "split16"),
        "depth-seven DFlash2 HTTP requires scoped carry and split16");
    need(mtp_request_carry(value) == expected_request_carry, "MTP request-carry startup mode differs");
    if (expected_request_carry) {
        const auto& generation = get(value, "native_mtp_generation");
        need(!integer(generation, "id") && !get(generation, "active").boolean(), "MTP ready contains a generation");
    }
    need(mtp_draft_attention(value) == expected_draft_attention, "MTP draft-attention startup mode differs");
    need(!get(value, "prompt_lookup").boolean() && !get(value, "checkpoint_broker").boolean() && !integer(value, "broker_epoch") &&
        !get(value, "weight_copy").boolean() && get(value, "gpu_profile_mode").string() == "disabled", "MTP unrelated ready feature");
    for (const auto name : {"allow_prompt_lookup_experiments", "allow_request_kernel_experiments", "allow_request_stack_experiments",
        "allow_request_prefill_experiments"}) need(!get(value, name).boolean(), "MTP request override enabled");
    for (const auto name : {"lookup_match_tokens", "lookup_history_tokens", "lookup_proposal_tokens", "lookup_verification_budget_bytes",
        "lookup_caller_host_bytes", "lookup_verification_charged_bytes"}) need(!integer(value, name), "MTP cannot prepare greedy lookup state");
    const auto host = integer(value, "verification_host_bytes"), gpu = integer(value, "verification_gpu_bytes");
    need(host && gpu && host <= (128ull << 20) && gpu <= (128ull << 20) - host &&
        integer(value, "verification_tape_payload_bytes") && integer(value, "verification_tape_payload_bytes") <= gpu,
        "MTP verifier workspace/tape budget");
    for (const auto name : counters) need(!integer(value, name), "MTP ready executed work");
    if (mtp_input_packet(value) >= 512) need(!integer(value, "native_mtp_priming_batch_submits"), "MTP ready already submitted batch priming");
    for (const auto name : clocks) need(!real(value, name), "MTP ready contains execution clock");
    for (const auto name : {"qsa_stage_v8_calls", "qsa_channel_calls", "known_qsa_rows_physical_tiles", "known_qsa_rows_logical_rows"})
        need(!integer(value, name), "MTP ready performed attention");
    for (const auto name : hc_names) need(!integer(get(value, "hc_exact_routes"), name), "MTP ready performed HC work");
}

MtpWireReply::MtpWireReply(const Json& ready, const Json& previous, const Json& job, bool resync)
    : ready_(ready), previous_(previous), job_(job) {
    validate_mtp_ready(ready_, mtp_draft_attention(ready_), mtp_request_carry(ready_), mtp_depth(ready_));
    depth_ = mtp_depth(ready_);
    validate_raw_decode_previous(ready_, previous_);
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    validate_owned_q8_three_previous(ready_,previous_);
#endif
    scoped_ = mtp_request_carry(ready_);
    need(mtp_gdn_old_order(ready_) == mtp_gdn_old_order(previous_), "MTP previous old-order GDN selection differs");
    need(mtp_hc_down_f32(ready_) == mtp_hc_down_f32(previous_), "MTP previous HC down batching selection differs");
    if (get(previous_, "event").string() == "ready" && mtp_hc_down_f32(previous_))
        need(!integer(previous_, "native_prefill_hc_down_f32_encoded_calls") && !integer(previous_, "native_prefill_hc_down_f32_request_encoded_calls"), "MTP previous READY already encoded HC down calls");
    need(mtp_hc_rows(ready_) == mtp_hc_rows(previous_), "MTP previous HC rows selection differs");
    if (mtp_hc_rows(previous_) && get(previous_, "event").string() == "ready") for (size_t i = 0; i < 3; ++i)
        need(!integer(previous_, mtp_hc_rows_counter_names[i]) && !integer(previous_, mtp_hc_rows_request_names[i]), "MTP previous READY already encoded HC rows");
    need(mtp_router_window(ready_) == mtp_router_window(previous_), "MTP previous router window selection differs");
    if (mtp_router_window(previous_) && get(previous_, "event").string() == "ready") for (size_t i = 0; i < 4; ++i)
        need(!integer(previous_, mtp_router_window_counter_names[i]) && !integer(previous_, mtp_router_window_request_names[i]), "MTP previous READY already encoded router window");
    need(mtp_hc_down_window(ready_) == mtp_hc_down_window(previous_), "MTP previous HC down window selection differs");
    if (mtp_hc_down_window(previous_) && get(previous_, "event").string() == "ready")
        need(!integer(previous_, mtp_hc_down_window_counter_name) && !integer(previous_, mtp_hc_down_window_request_name), "MTP previous READY already encoded HC down window");
    need(mtp_qsa_sparse(ready_) == mtp_qsa_sparse(previous_), "MTP previous sparse QSA selection differs");
    if (mtp_qsa_sparse(previous_) && get(previous_, "event").string() == "ready") for (size_t i = 0; i < 2; ++i)
        need(!integer(previous_, mtp_qsa_sparse_counter_names[i]) && !integer(previous_, mtp_qsa_sparse_request_names[i]), "MTP previous READY already executed sparse QSA");
    need(mtp_gdn_ported(ready_) == mtp_gdn_ported(previous_), "MTP previous ported GDN selection differs");
    if (mtp_gdn_ported(previous_) && get(previous_, "event").string() == "ready") for (size_t i = 0; i < 5; ++i)
        need(!integer(previous_, mtp_gdn_ported_counter_names[i]) && !integer(previous_, mtp_gdn_ported_request_names[i]), "MTP previous READY already executed ported GDN");
    final_input_logits_ = mtp_final_input_logits(ready_);
    need(mtp_final_input_logits(previous_) == final_input_logits_, "MTP previous final-input policy differs");
    need(mtp_input_packet(ready_) == mtp_input_packet(previous_), "MTP previous input-packet mode differs");
    if (mtp_input_packet(ready_) >= 512) {
        for (const auto name : {"native_mtp_priming_batch_capacity", "native_mtp_priming_batch_bytes", "native_prefill_dense"})
            need(get(ready_, name).dump() == get(previous_, name).dump(), "MTP previous prefill preparation differs");
        for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel", "native_prefill_dense_kernel"})
            need(bool(ready_.find(name)) == bool(previous_.find(name)) && (!ready_.find(name) || get(ready_, name).dump() == get(previous_, name).dump()),
                "MTP previous grouped-prefill preparation differs");
    }
    need(bool(ready_.find("native_mtp_output_packet")) == bool(previous_.find("native_mtp_output_packet")) &&
        mtp_output_packet(ready_) == mtp_output_packet(previous_), "MTP previous output-packet mode differs");
    need(mtp_request_carry(previous_) == scoped_, "MTP previous carry scope differs");
    need(mtp_draft_attention(previous_) == mtp_draft_attention(ready_), "MTP previous draft-attention mode differs");
    if (scoped_) keys(job_, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed", "mtp_generation", "prefill_final"});
    else keys(job_, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed"});
    if (const auto* command = job_.find("mtp_generation")) {
        need(scoped_, "MTP generation command lacks explicit scope"); generation_ = mtp_generation_command(*command); abort_ = generation_.abort;
    }
    reset_ = get(job_, "reset").boolean(); inputs_ = uint32_t(get(job_, "tokens").array().size());
    const auto outputs = integer(job_, "generate");
    need((inputs_ >= 1 && inputs_ <= mtp_input_packet(ready_) && !outputs) || (!inputs_ && outputs >= 1 && outputs <= mtp_output_packet(ready_)) || (abort_ && !inputs_ && !outputs), "MTP packet exceeds negotiated input or output bound");
    maximum_ = uint32_t(outputs);
    const auto* final = job_.find("prefill_final");
    need(bool(final) == (final_input_logits_ && inputs_), "MTP final-input flag presence differs");
    if (final) final_input_ = final->boolean();
    const auto event = get(previous_, "event").string();
    need(event == "ready" || event == "done", "MTP previous ancestry event differs");
    need(event != "ready" || reset_, "MTP first packet must reset");
    if (final_input_logits_) {
        const bool available = get(previous_, "logits_available").boolean();
        need(event != "ready" || !available, "MTP READY contains available logits");
        if (!inputs_) need(available, "MTP output/control requires final-input logits");
        if (event == "done") {
            need(get(previous_, "continuation_allowed").boolean() == (available && get(previous_, "stop_id").is_null()), "MTP previous readiness/continuation differs");
            final_logits_state(previous_, available);
        }
    }
    need(!reset_ || inputs_, "MTP reset must contain known inputs");
    need(reset_ || (!job_.find("sampling") && !job_.find("seed")), "MTP append changed sampling parameters");
    if (reset_) {
        const auto& policy = get(job_, "sampling"); keys(policy, {"temperature", "top_k", "top_p", "min_p"});
        need(real(policy, "temperature") >= 0 && real(policy, "temperature") <= 100 && integer(policy, "top_k") <= 248320 &&
            real(policy, "top_p") > 0 && real(policy, "top_p") <= 1 && real(policy, "min_p") >= 0 && real(policy, "min_p") <= 1,
            "MTP reset sampling policy bound");
        static_cast<void>(integer(job_, "seed"));
    }
    const auto before = event == "ready" ? 0 : integer(previous_, "committed_prefix");
    need(before <= 65536, "MTP previous context bound"); base_ = position_ = reset_ ? 0 : uint32_t(before);
    const uint32_t pending_overflow = (generation_.v2 && !generation_.begin) ? generation_.pending : 0;
    if (pending_overflow) {
        need(pending_overflow <= (depth_ == 7 ? 7u : depth_) && pending_overflow <= position_, "invalid MTP pending overflow bound");
        position_ -= pending_overflow;
        pending_tokens_ = pending_overflow;
        round_end_ = base_;
    }
    need(uint64_t(base_) + inputs_ + maximum_ <= 65536, "MTP packet exceeds context reservation");
    resync_ = resync; need(!resync_ || (reset_ && inputs_), "MTP counter resync must be a cold input reset");
    if (!reset_) {
        need(get(get(previous_, "sampling"), "valid").boolean() && integer(get(previous_, "sampling"), "entropy_prefix") == base_,
            "MTP previous entropy frontier differs");
        if (maximum_) need(get(previous_, "continuation_allowed").boolean(), "MTP cannot continue stopped prefix without input");
        last_words_ = integer(get(previous_, "sampling"), "entropy_words");
    }
    if (scoped_) {
        const auto& previous_generation = get(previous_, "native_mtp_generation");
        const auto previous_id = integer(previous_generation, "id"), previous_remaining = integer(previous_generation, "remaining");
        const bool active = get(previous_generation, "active").boolean(), pending = get(previous_generation, "pending").boolean();
        if (inputs_) {
            need(!job_.find("mtp_generation"), "known inputs cannot carry MTP generation commands");
            discarded_ = !resync_ && pending ? 1 : 0;
        } else {
            need(job_.find("mtp_generation"), "MTP output/control requires a generation command");
            if (abort_) {
                need(active && generation_.id == previous_id && !job_.find("stop_ids"), "MTP abort identity differs");
                discarded_ = pending ? 1 : 0;
            } else {
                need(generation_.remaining >= maximum_, "MTP packet exceeds remaining generation budget");
                if (generation_.begin) need(!active && generation_.id > previous_id, "MTP overlapping/stale generation begin");
                else {
                    need(active && generation_.id == previous_id && generation_.remaining == previous_remaining,
                        "MTP continuation lifetime differs"); carry_ = pending;
                }
            }
        }
    }
    for (const auto& token : get(job_, "tokens").array()) need(token.unsigned_integer() < 248320, "MTP input token bound");
    if (const auto* stops = job_.find("stop_ids")) {
        need(stops->array().size() <= 16, "MTP stop set bound");
        std::array<uint32_t, 16> seen{}; size_t count = 0;
        for (const auto& token : stops->array()) {
            const auto id = token.unsigned_integer(); need(id < 248320 && std::find(seen.begin(), seen.begin() + ptrdiff_t(count), id) == seen.begin() + ptrdiff_t(count), "MTP invalid/duplicate stop ID");
            seen[count++] = uint32_t(id);
        }
    }
}

void MtpWireReply::event(const Json& value) {
    need(!finished_ && maximum_ && event_count_ < events_.size() && !stopped_, "MTP unsolicited/excess/post-terminal event");
    const auto type = get(value, "event").string();
    if (type == "mtp_round") {
        const auto proposals = ((depth_ == 2 || depth_ == 3 || depth_ == 7) && value.find("proposed")) ? uint32_t(integer(value, "proposed")) : depth_;
        if (depth_ == 2 || depth_ == 3) {
            keys(value, {"event", "base", "committed_prefix", "count", "accepted", "proposed", "head_calls", "prime_extra",
                "draft_wall_ms", "draft_gpu_ms", "verification_wall_ms",
                "verification_gpu_ms", "recovery_wall_ms", "recovery_gpu_ms", "materialization_wall_ms", "lookahead_anchor", "used_carried_anchor",
                "pending_anchor", "round_wall_ms", "timings_qualified"});
            need(proposals >= 1 && proposals <= depth_, "variable-depth round omitted actual proposal count");
        } else if (depth_ == 7) {
            keys(value, {"event", "base", "committed_prefix", "count", "accepted", "proposed", "draft_wall_ms", "draft_gpu_ms", "verification_wall_ms",
                "verification_gpu_ms", "recovery_wall_ms", "recovery_gpu_ms", "materialization_wall_ms", "lookahead_anchor", "used_carried_anchor",
                "pending_anchor", "round_wall_ms", "timings_qualified"});
            need(proposals >= 1 && proposals <= depth_, "variable-depth round omitted actual proposal count");
        } else {
        keys(value, {"event", "base", "committed_prefix", "count", "accepted", "draft_wall_ms", "draft_gpu_ms", "verification_wall_ms",
            "verification_gpu_ms", "recovery_wall_ms", "recovery_gpu_ms", "materialization_wall_ms", "lookahead_anchor", "used_carried_anchor",
            "pending_anchor", "round_wall_ms", "timings_qualified"});
        }
        const bool has_head_calls = value.find("head_calls") != nullptr;
        const bool has_prime_extra = value.find("prime_extra") != nullptr;
        need(has_head_calls == has_prime_extra && (!has_head_calls || depth_ == 2 || depth_ == 3), "MTP round lookup-bypass accounting incomplete");
        const uint32_t round_head_calls = has_head_calls ? uint32_t(integer(value, "head_calls")) : proposals;
        const uint32_t round_prime_extra = has_prime_extra ? uint32_t(integer(value, "prime_extra")) : 0u;
        need((round_head_calls == proposals && round_prime_extra == 0) || (round_head_calls == 0 && round_prime_extra == 1),
            "MTP round head/prime bypass accounting invalid");
        const auto count = integer(value, "count"), accepted = integer(value, "accepted");
        const bool count_valid = generation_.v2 ? (count_ < maximum_) : (count <= maximum_ - count_);
        need(!pending_tokens_ && count >= 1 && count <= proposals + 1 && accepted <= proposals && count == 1 + accepted &&
            count_valid && integer(value, "base") == position_ && integer(value, "committed_prefix") == position_ + count &&
            get(value, "lookahead_anchor").boolean() && get(value, "used_carried_anchor").boolean() == carry_ &&
            !get(value, "timings_qualified").boolean(), "MTP round prefix/shape/carry differs");
        if (carry_) ++carried_;
        carry_ = get(value, "pending_anchor").boolean();
        ++rounds_; proposed_ += proposals; head_calls_ += round_head_calls; prime_extra_ += round_prime_extra;
        rejected_ += accepted < proposals; accepted_ += uint32_t(accepted);
        const uint32_t tokens_in_packet = generation_.v2 ? std::min(uint32_t(count), uint32_t(maximum_ - count_)) : uint32_t(count);
        pending_tokens_ = tokens_in_packet; round_end_ = position_ + uint32_t(count);
        for (const auto name : {"draft_wall_ms", "draft_gpu_ms", "verification_wall_ms", "verification_gpu_ms", "recovery_wall_ms",
            "recovery_gpu_ms", "materialization_wall_ms", "round_wall_ms"}) need(real(value, name) >= 0, "MTP negative round clock");
    } else {
        need(type == "token", "MTP unknown native event");
        keys(value, {"event", "id", "source_logit", "committed_prefix", "token_prefix", "round_committed_prefix", "entropy_words", "gpu_profile"});
        const auto id = integer(value, "id"), words = integer(value, "entropy_words");
        need(count_ < maximum_ && id < 248320 && words >= last_words_ && get(value, "gpu_profile").is_null(), "MTP token/entropy bound");
        static_cast<void>(real(value, "source_logit"));
        if (pending_tokens_) {
            need(integer(value, "committed_prefix") == round_end_ && integer(value, "round_committed_prefix") == round_end_ &&
                integer(value, "token_prefix") == position_ + 1, "MTP batched token prefix differs");
            --pending_tokens_;
        } else {
            need(integer(value, "committed_prefix") == position_ + 1 && !value.find("round_committed_prefix") && !value.find("token_prefix"),
                "MTP ordinary token prefix differs");
            ++ordinary_; if (carry_) ++carried_; carry_ = false;
        }
        ++position_; ids_[count_] = uint32_t(id); token_words_[count_++] = words; last_words_ = words;
        stopped_ = stop(uint32_t(id), job_);
        need(!stopped_ || (!pending_tokens_ && !carry_), "MTP terminal appeared before consumed round end");
    }
    events_[event_count_++] = value;
}

void MtpWireReply::finish(const Json& done) {
    for (const char* name : {"native_controller_policy_digest", "native_prepared_validation", "native_measured_controller", "native_measured_timing_schema", "native_dflash2_draft_policy", "native_target_m8_split_digest", "native_verify_narrow_rows", "native_pack_shared_mapping"}) {
        const auto* policy = ready_.find(name), *observed = done.find(name);
        need(bool(policy) == bool(observed) && (!policy || policy->dump() == observed->dump()), "controller or validation policy differs from startup negotiation");
    }
    validate_measured_timing_packet(ready_, previous_, done, generation_.begin, generation_.id, rounds_ + ordinary_);
    const auto* dense_kernel = ready_.find("native_prefill_dense_kernel");
    need(bool(dense_kernel) == bool(done.find("native_prefill_dense_kernel")) &&
        (!dense_kernel || dense_kernel->dump() == get(done, "native_prefill_dense_kernel").dump()),
        "MTP immutable dense-kernel identity drift before counter resynchronisation");
    if (dense_kernel) static_cast<void>(mtp_input_packet(done));
    const auto input_packet = mtp_input_packet(ready_);
    const auto* dense_field = done.find("native_prefill_dense");
    const bool dense_nax = dense_field && dense_field->string() == "nax";
    const auto* grouped_field = done.find("native_prefill_grouped");
    const bool grouped_nax = grouped_field && grouped_field->string() == "nax";
    const auto* overlap_field = done.find("exact_candidate_ple_overlap");
    const bool ple_overlap = overlap_field && overlap_field->kind() == Json::Kind::boolean && overlap_field->boolean();
    const uint64_t chunk_cap = (input_packet >= 512 && dense_nax && grouped_nax)
        ? std::min<uint64_t>(input_packet, 2048ull)
        : 512ull;
    uint64_t hc_chunks = 0;
    if (input_packet >= 512 && inputs_ >= 2) {
        uint64_t at = 0;
        while (at < inputs_) {
            uint64_t n = std::min<uint64_t>(chunk_cap, inputs_ - at);
            if (ple_overlap) {
                const uint64_t abs_pos = base_ + at;
                if (chunk_cap <= 512 && at == 0 && inputs_ >= 1024 && n > 512) {
                    n = 512;
                } else if (chunk_cap == 2048 && (abs_pos % 512 == 0) && (abs_pos % 2048 != 0)) {
                    n = std::min<uint64_t>(n, 2048 - (abs_pos % 2048));
                }
            }
            if (n > 2 && (inputs_ - at - n == 1)) --n;
            if (n >= 2) ++hc_chunks;
            at += n;
        }
    }
    const bool hc_rows = mtp_hc_rows(ready_); need(mtp_hc_rows(done) == hc_rows, "MTP DONE HC rows selection differs");
    if (hc_rows) for (size_t i = 0; i < 3; ++i) {
        const uint64_t expected = 96ull * hc_chunks;
        need(integer(done, mtp_hc_rows_request_names[i]) == expected, "MTP HC rows per-packet encoded calls differ");
        delta(previous_, done, mtp_hc_rows_counter_names[i], expected, resync_);
    }
    const bool router_window = mtp_router_window(ready_); need(mtp_router_window(done) == router_window, "MTP DONE router window selection differs");
    if (router_window) for (size_t i = 0; i < 4; ++i) {
        const uint64_t router_chunks = grouped_nax ? (inputs_ <= chunk_cap ? (inputs_ >= 2 ? 1 : 0) : hc_chunks) : hc_chunks;
        const uint64_t expected = 48ull * (ple_overlap ? hc_chunks : router_chunks);
        need(integer(done, mtp_router_window_request_names[i]) == expected, "MTP router window per-packet encoded calls differ");
        delta(previous_, done, mtp_router_window_counter_names[i], expected, resync_);
    }
    const bool hc_down_window = mtp_hc_down_window(ready_); need(mtp_hc_down_window(done) == hc_down_window, "MTP DONE HC down window selection differs");
    if (hc_down_window) {
        const uint64_t expected = 96ull * hc_chunks;
        need(integer(done, mtp_hc_down_window_request_name) == expected, "MTP HC down window per-packet encoded calls differ");
        delta(previous_, done, mtp_hc_down_window_counter_name, expected, resync_);
    }
    const bool qsa_sparse = mtp_qsa_sparse(ready_); need(mtp_qsa_sparse(done) == qsa_sparse, "MTP DONE sparse QSA selection differs");
    const bool gdn_ported = mtp_gdn_ported(ready_); need(mtp_gdn_ported(done) == gdn_ported, "MTP DONE ported GDN selection differs");
    validate_raw_decode_done(ready_, previous_, done, resync_);
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    validate_owned_q8_three_done(ready_,previous_,done,resync_);
#endif
    need(mtp_final_input_logits(done) == final_input_logits_, "MTP DONE final-input policy differs");
    const bool hc_down = mtp_hc_down_f32(ready_); need(mtp_hc_down_f32(done) == hc_down, "MTP DONE HC down batching selection differs");
    if (hc_down) {
        const auto calls = integer(done, "native_prefill_hc_down_f32_request_encoded_calls");
        // Two residual reads per layer and the architecture's 128-layer ceiling bound a packet. No per-round ratio is inferred.
        need(inputs_ == 512 ? calls && calls <= 256 : !calls, "MTP HC down batching used an ineligible packet or exceeded its bound");
        delta(previous_, done, "native_prefill_hc_down_f32_encoded_calls", calls, resync_);
    }
    const bool old_order = mtp_gdn_old_order(ready_); need(mtp_gdn_old_order(done) == old_order, "MTP DONE old-order GDN selection differs");
    if (old_order) {
        const uint64_t calls = inputs_ > 1 ? 36 : 0;
        need(integer(done, "native_gdn_old_order_request_encoded_calls") == calls, "MTP old-order GDN per-packet encoded calls differ");
        delta(previous_, done, "native_gdn_old_order_encoded_calls", calls, resync_);
    }
    need(input_packet == mtp_input_packet(done), "MTP DONE input-packet mode differs");
    if (input_packet >= 512) {
        for (const auto name : {"native_mtp_priming_batch_capacity", "native_mtp_priming_batch_bytes", "native_prefill_dense"})
            need(get(ready_, name).dump() == get(done, name).dump(), "MTP immutable prefill preparation drift");
        for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel", "native_prefill_dense_kernel"})
            need(bool(ready_.find(name)) == bool(done.find(name)) && (!ready_.find(name) || get(ready_, name).dump() == get(done, name).dump()),
                "MTP immutable grouped-prefill preparation drift");
        need(integer(done, "native_mtp_priming_batch_submits") >= integer(previous_, "native_mtp_priming_batch_submits"), "MTP priming submit counter rewound");
    }
    need(bool(ready_.find("native_mtp_output_packet")) == bool(done.find("native_mtp_output_packet")) &&
        mtp_output_packet(ready_) == mtp_output_packet(done), "MTP DONE output-packet mode differs");
    need(!finished_ && !pending_tokens_ && get(done, "event").string() == "done", "MTP incomplete/duplicate completion");
    fixed(done);
    need(mtp_request_carry(done) == scoped_, "MTP carry scope changed");
    if (scoped_) {
        const auto& observed = get(done, "native_mtp_generation");
        const auto expected_id = inputs_ ? integer(get(previous_, "native_mtp_generation"), "id") : generation_.id;
        const auto observed_id = integer(observed, "id");
        // A cancelled generation may complete after the caller's last admitted DONE. Only a cold reset may catch up that monotone identity.
        const bool identity_matches = observed_id == expected_id || (resync_ && reset_ && inputs_ && observed_id >= expected_id);
        const auto remaining = maximum_ && !stopped_ ? generation_.remaining - uint32_t(count_) : 0;
        need(identity_matches && get(observed, "active").boolean() == bool(remaining) &&
            integer(observed, "remaining") == remaining && get(observed, "pending").boolean() == (remaining && carry_),
            "MTP generation completion identity/budget/pending differs");
        if (maximum_ && !remaining && carry_) ++discarded_;
    } else discarded_ = carry_ ? 1 : 0;
    need(mtp_draft_attention(done) == mtp_draft_attention(ready_), "MTP immutable draft-attention mode drift");
    for (const auto name : immutable) need(get(done, name).dump() == get(ready_, name).dump(), "MTP immutable preparation drift");
    for (const auto name : resources) need(get(done, name).dump() == get(ready_, name).dump(), "MTP fixed resource ledger drift");
    const auto expected_committed_prefix = (round_end_ && generation_.v2) ? std::max(round_end_, position_) : (base_ + inputs_ + count_);
    need(get(done, "known_token_prefill").boolean() && integer(done, "committed_prefix") == expected_committed_prefix &&
        (inputs_ ? !count_ : (stopped_ ? count_ >= 1 : count_ == maximum_)), "MTP completion prefix/count differs");
    const auto& ids = get(done, "generated_ids").array(); need(ids.size() == count_, "MTP completion ID count differs");
    for (size_t i = 0; i < count_; ++i) need(ids[i].unsigned_integer() == ids_[i], "MTP token/DONE ID disagreement");
    const auto& sampled = get(done, "sampling"), policy = reset_ ? get(job_, "sampling") : get(previous_, "sampling");
    need(get(sampled, "valid").boolean() && integer(sampled, "entropy_prefix") == expected_committed_prefix &&
        integer(sampled, "seed") == (reset_ ? integer(job_, "seed") : integer(policy, "seed")), "MTP sampler ancestry differs");
    for (const auto name : {"temperature", "top_k", "top_p", "min_p"}) need(real(sampled, name) == real(policy, name), "MTP target sampling policy changed");
    for (const auto name : {"filter_order", "rng", "ownership"}) need(get(sampled, name).dump() == get(get(ready_, "sampling"), name).dump(), "MTP sampler algorithm changed");
    const auto words = integer(sampled, "entropy_words"), before_words = reset_ ? 0 : integer(policy, "entropy_words");
    const uint64_t max_words_delta = (real(policy, "temperature") == 0 || inputs_) ? 0 :
        (depth_ == 7 ? 130ull * (depth_ * rounds_ + ordinary_) + count_ : 130ull * depth_ * rounds_ + count_);
    need(words >= before_words && words - before_words <= max_words_delta,
        "MTP entropy advance exceeds bounded rejection draws");
    if (count_) need(token_words_[count_ - 1] == words, "MTP last token/DONE entropy differs");
    const bool available = !final_input_logits_ || !inputs_ || final_input_;
    need(get(done, "continuation_allowed").boolean() == (available && !stopped_), "MTP continuation flag differs");
    if (final_input_logits_) {
        final_logits_state(done, available);
    }
    if (inputs_ || abort_) need(get(done, "status").string() == "prefilled" && get(done, "finish_reason").is_null() && get(done, "stop_id").is_null(), "MTP input emitted terminal state");
    else if (stopped_) need(get(done, "status").string() == "stopped" && get(done, "finish_reason").string() == "stop" && integer(done, "stop_id") == ids_[count_ - 1], "MTP stop completion differs");
    else need(get(done, "status").string() == "length" && get(done, "finish_reason").string() == "length" && get(done, "stop_id").is_null(), "MTP length completion differs");
    const uint64_t primes = depth_ == 7 ? (inputs_ ? inputs_ : rounds_ + accepted_ + ordinary_) :
        (inputs_ ? inputs_ - (reset_ ? 1 : 0) : accepted_ + ordinary_);
    const std::array<uint64_t, 8> expected{head_calls_, primes + prime_extra_, rounds_, proposed_, accepted_, rejected_, carried_, discarded_};
    for (size_t i = 0; i < counters.size(); ++i) delta(previous_, done, counters[i], expected[i], resync_);
    for (const auto name : clocks) need(real(done, name) >= real(previous_, name), "MTP execution clock rewound");
    const uint64_t ordinary_attention = ordinary_ + (input_packet >= 512 && inputs_ == 1 ? 1 : 0);
    const uint64_t legacy_ordinary_attention = qsa_sparse ? 0 : 12ull * ordinary_attention;
    for (const auto name : {"qsa_stage_v8_calls", "qsa_channel_calls"}) delta(previous_, done, name, legacy_ordinary_attention, resync_);
    need(integer(done, "qsa_stage_v8_request_calls") == legacy_ordinary_attention && integer(done, "qsa_channel_request_calls") == legacy_ordinary_attention, "MTP per-packet attention counters differ");
    const uint64_t verify_tiles = 12ull * (inputs_ ? (input_packet >= 512 ? 0 : 1) : rounds_);
    if (qsa_sparse) {
        const std::array<uint64_t, 2> expected_sparse{12ull * ordinary_attention, verify_tiles};
        for (size_t i = 0; i < 2; ++i) {
            need(integer(done, mtp_qsa_sparse_request_names[i]) == expected_sparse[i], "MTP sparse QSA per-packet calls differ");
            delta(previous_, done, mtp_qsa_sparse_counter_names[i], expected_sparse[i], resync_);
        }
    }
    if (gdn_ported) {
        const uint64_t prefill_gdn = 36ull * ((ple_overlap || input_packet > 2048) ? hc_chunks : (input_packet >= 512 && inputs_ >= 2 ? (inputs_ + chunk_cap - 1) / chunk_cap : 0));
        const uint64_t decode_gdn = 36ull * ordinary_attention;
        const uint64_t verify_gdn = 36ull * (inputs_ ? (input_packet >= 512 ? 0 : 1) : rounds_);
        const uint64_t replay_gdn = 36ull * (inputs_ ? 0 : rejected_);
        const std::array<uint64_t, 5> expected_gdn{prefill_gdn, decode_gdn, verify_gdn, replay_gdn, decode_gdn + verify_gdn};
        for (size_t i = 0; i < 5; ++i) {
            need(integer(done, mtp_gdn_ported_request_names[i]) == expected_gdn[i], "MTP ported GDN per-packet calls differ");
            delta(previous_, done, mtp_gdn_ported_counter_names[i], expected_gdn[i], resync_);
        }
    }
    delta(previous_, done, "known_qsa_rows_physical_tiles", verify_tiles, resync_);
    delta(previous_, done, "known_qsa_rows_logical_rows",
        12ull * (inputs_ ? (input_packet >= 512 ? 0 : inputs_) : (rounds_ + proposed_)), resync_);
    const auto& routes = get(done, "hc_exact_routes"), request_routes = get(done, "hc_exact_request_routes");
    need(request_routes.object().size() == hc_names.size(), "MTP request HC route shape");
    for (const auto name : hc_names) {
        const auto before = integer(get(previous_, "hc_exact_routes"), name), after = integer(routes, name), request = integer(request_routes, name);
        need(after >= before && (resync_ ? after - before >= request : after - before == request), "MTP HC request/lifetime counter differs");
    }
    finished_ = true;
}
std::span<const Json> MtpWireReply::events() const {
    need(finished_, "MTP events cannot publish before completion admission"); return {events_.data(), event_count_};
}
} // namespace tokspur
#endif
