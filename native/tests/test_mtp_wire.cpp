// Generated protocol controls only. Reuse the existing CPU fixture definitions;
// no inherited main, worker, model, tokenizer or Metal test is executed here.
#define TOKSPUR_HTTP_SAMPLED_PREFILL_TEST_ENTRY inherited_sampled_prefill_entry
#include "test_http_sampled_prefill.cpp"
#undef TOKSPUR_HTTP_SAMPLED_PREFILL_TEST_ENTRY
#include "tokspur/mtp_wire.hpp"
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#include "tokspur/raw_decode_wire.hpp"
#endif

namespace mtp_fixture {
constexpr std::array counter_names{"native_mtp_head_calls", "native_mtp_prime_calls", "native_mtp_rounds", "native_mtp_proposed",
    "native_mtp_accepted", "native_mtp_recovery_rounds", "native_mtp_carried_anchors", "native_mtp_discarded_anchors"};
constexpr std::array clock_names{"native_mtp_head_gpu_ms", "native_mtp_head_wall_ms", "native_mtp_prime_gpu_ms", "native_mtp_prime_wall_ms"};
Json ready() {
    auto r = without(sampled_prefill_fixture::ready(), "sampled_prefill_protocol");
    for (const auto name : {"lookup_match_tokens", "lookup_history_tokens", "lookup_proposal_tokens"}) r = with(std::move(r), name, Json::integer(0));
    r = with(std::move(r), "known_qsa_rows_http_admitted", Json(false));
    r = with(std::move(r), "verification_host_bytes", Json::integer(1'000'000));
    r = with(std::move(r), "verification_gpu_bytes", Json::integer(4'000'000));
    r = with(std::move(r), "verification_tape_payload_bytes", Json::integer(1024));
    for (const auto& [name, value] : Json::Object{{"native_mtp_protocol", Json("native-mtp-depth1-lookahead-v2")},
        {"native_mtp_depth", Json::integer(1)}, {"native_mtp_attention", Json("dense-compression0")},
        {"native_mtp_first_cache_position", Json::integer(1)}, {"native_mtp_lookahead_anchor", Json(true)},
        {"native_mtp_visible_frontier", Json("all emitted tokens physically consumed; sampled terminal carried invisibly within request")},
        {"native_mtp_workspace_bytes", Json::integer(1'000'000)}}) r = with(std::move(r), name, value);
    for (const auto name : counter_names) r = with(std::move(r), name, Json::integer(0));
    for (const auto name : clock_names) r = with(std::move(r), name, number(0));
    auto sampling = scalar_fixture::sampled_policy().object();
    for (const auto& item : Json::Object{{"valid", Json(true)}, {"seed", Json::integer(0)}, {"entropy_prefix", Json::integer(0)},
        {"entropy_words", Json::integer(0)}, {"filter_order", Json("fixture-filter")}, {"rng", Json("fixture-rng")}, {"ownership", Json("fixture-owner")}})
        sampling.push_back(item);
    return with(std::move(r), "sampling", Json(std::move(sampling)));
}
Json scoped_ready() {
    auto value = with(ready(), "native_mtp_carry_scope", Json("http-generation-v1"));
    return with(std::move(value), "native_mtp_generation", Json(Json::Object{{"id", Json::integer(0)}, {"active", Json(false)},
        {"remaining", Json::integer(0)}, {"pending", Json(false)}}));
}
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
Json prefill512_ready(Json value, std::string_view dense = "f16", std::optional<std::string_view> grouped = std::nullopt) {
    for (const auto& [name, field] : Json::Object{{"native_mtp_input_packet_schema", Json::integer(1)}, {"native_mtp_input_packet_max", Json::integer(512)},
        {"native_prefill_dense", Json(std::string(dense))},
        {"known_token_capacity", Json::integer(512)}, {"known_qsa_rows_mode", Json("off")}, {"known_qsa_rows_logical_bytes", Json::integer(0)},
        {"known_qsa_rows_charged_bytes", Json::integer(0)}, {"known_qsa_rows_allocations", Json::integer(0)},
        {"native_mtp_priming_batch_capacity", Json::integer(512)}, {"native_mtp_priming_batch_bytes", Json::integer(26'214'400)},
        {"native_mtp_priming_batch_submits", Json::integer(0)}}) value = with(std::move(value), name, field);
    if (dense == "f32-staged") {
        value = with(std::move(value), "native_prefill_dense", Json("f32"));
        value = with(std::move(value), "native_prefill_dense_kernel", Json("graph_dense_q8_staged_f32"));
        value = with(std::move(value), "native_prefill_chunks", Json("fused"));
    }
    if (grouped) {
        value = with(std::move(value), "native_prefill_grouped", Json(std::string(*grouped)));
        value = with(std::move(value), "native_prefill_grouped_kernel", Json(*grouped == "f32" ? "graph_grouped_mm_f32" : "graph_grouped_serial"));
    }
    return value;
}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
Json old_order_fields(Json value) {
    for (const auto& [name, field] : Json::Object{{"native_prefill_chunks", Json("fused")}, {"native_prefill_dense", Json("f32")}, {"native_prefill_grouped", Json("f32")}, {"native_prefill_grouped_kernel", Json("graph_grouped_mm_f32")},
        {"native_gdn_old_order_version", Json("old-order-batch-v1")}, {"native_gdn_old_order_kernel", Json("gdn_chunk_legacy_batch")}, {"native_gdn_old_order_layers", Json::integer(36)}, {"native_gdn_old_order_scratch_bytes", Json::integer(0)},
        {"native_gdn_old_order_encoded_calls", Json::integer(0)}, {"native_gdn_old_order_request_encoded_calls", Json::integer(0)}}) value = with(std::move(value), name, field);
    return value;
}
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
Json hc_down_fields(Json value, uint64_t encoded = 0, uint64_t request = 0) {
    for (const auto& [name, field] : Json::Object{{"native_prefill_chunks", Json("fused")}, {"native_prefill_dense", Json("f32")}, {"native_prefill_grouped", Json("f32")}, {"native_prefill_grouped_kernel", Json("graph_grouped_mm_f32")}}) value = with(std::move(value), name, field);
    for (const auto& [name, field] : mtp_hc_down_f32_fields(encoded, request)) value = with(std::move(value), name, field);
    return value;
}
#endif
Json hc_rows_fields(Json value, uint64_t encoded = 0, uint64_t request = 0) {
    for (const auto& [name, field] : mtp_hc_rows_fields({encoded, encoded, encoded}, {request, request, request})) value = with(std::move(value), name, field);
    return value;
}
Json router_window_fields(Json value, uint64_t encoded = 0, uint64_t request = 0) {
    for (const auto& [name, field] : mtp_router_window_fields({encoded, encoded, encoded, encoded}, {request, request, request, request})) value = with(std::move(value), name, field);
    return value;
}
Json hc_down_window_fields(Json value, uint64_t encoded = 0, uint64_t request = 0) {
    for (const auto& [name, field] : mtp_hc_down_window_fields(encoded, request)) value = with(std::move(value), name, field);
    return value;
}
Json qsa_sparse_fields(Json value, std::array<uint64_t, 2> completed = {}, std::array<uint64_t, 2> request = {}) {
    for (const auto& [name, field] : mtp_qsa_sparse_fields(completed, request)) value = with(std::move(value), name, field);
    return value;
}
Json gdn_ported_fields(Json value, std::array<uint64_t, 5> completed = {}, std::array<uint64_t, 5> request = {}) {
    for (const auto& [name, field] : mtp_gdn_ported_fields(completed, request)) value = with(std::move(value), name, field);
    return value;
}
Json scope_job(Json job, uint64_t id, bool begin, uint32_t remaining) {
    return with(std::move(job), "mtp_generation", Json(Json::Object{{"id", Json::integer(id)}, {"begin", Json(begin)}, {"remaining", Json::integer(remaining)}}));
}
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
Json q8_three_fields(Json value,bool on){
    const Json routes(Json::Object{{"raw",Json::integer(0)},{"packed",Json::integer(0)},{"hc_raw",Json::integer(0)},{"wide_qkv_raw",Json::integer(0)},{"wide_output_raw",Json::integer(0)}});
    for(const auto&[name,field]:Json::Object{{"q8_three_rows_schema",Json::integer(1)},{"q8_three_rows_mode",Json(on?"on":"off")},{"q8_three_rows_routes",routes},{"q8_three_rows_padding_rows",Json::integer(0)},{"q8_three_rows_hc_route",Json("generic-exact-fp32")}})value=with(value,name,field);
    return value;
}
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
Json decode_fields(Json value, bool enabled, const RawDecodeRouteCounts& total = {}, const RawDecodeRouteCounts& request = {}) {
    for (const auto& [name, field] : raw_decode_metadata_fields(enabled, total, request)) value = with(std::move(value), name, field);
    return value;
}
#endif
struct Fake {
    Json prepared = ready(), previous = prepared, selected = scalar_fixture::sampled_policy();
    std::array<uint64_t, 8> counters{};
    uint32_t prefix = 0; uint64_t words = 0, seed = 44, stage = 0, tiles = 0, rows = 0, hc_count = 0;
    std::vector<Json> packets, events;
    uint32_t stop_at = 0; // Relative generated position in this packet.
    bool broken_done = false, use_acceptance = true, lookup_bypass = false;
    uint32_t force_accepted = 99, force_proposed = 0;
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    RawDecodeRouteCounts decode_routes{};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    std::array<uint64_t,5> n3_routes{};
#endif
    bool generation_active = false, pending_terminal = false;
    uint64_t generation_id = 0; uint32_t generation_remaining = 0;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    uint64_t priming_submits = 0;
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    uint64_t hc_rows_encoded = 0;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    uint64_t router_window_encoded = 0;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    uint64_t hc_down_window_encoded = 0;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    std::array<uint64_t, 2> qsa_sparse_completed{};
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    std::array<uint64_t, 5> gdn_ported_completed{};
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    uint64_t hc_down_encoded = 0;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    uint64_t old_order_encoded = 0;
#endif
#endif
    Json operator()(const Json& job, const std::function<void(const Json&)>& callback) {
        packets.push_back(job); events.clear(); const bool reset = job.find("reset")->boolean();
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        const auto n3_before=n3_routes;const auto n3_rounds_before=counters[2];
#endif
        if (reset) { prefix = 0; words = 0; seed = integer(job, "seed"); selected = *job.find("sampling"); }
        const auto n = uint32_t(job.find("tokens")->array().size()), maximum = uint32_t(integer(job, "generate"));
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        const auto decode_before = decode_routes;
        // Deliberately synthetic per-packet increments test admission, never production dispatch totals.
        if (const auto* mode = prepared.find("native_raw_decode_mode"); mode && mode->string() == "on") {
            decode_routes[0] += n ? 1 : maximum ? 3 : 0; decode_routes[1] += n ? 2 : maximum ? 1 : 0;
        }
#endif
        const bool scoped = mtp_request_carry(prepared); const auto depth = mtp_depth(prepared);
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
        const auto hc_rows_before = hc_rows_encoded;
        if (n >= 2 && n <= 512 && mtp_hc_rows(prepared)) hc_rows_encoded += 96;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
        const auto router_window_before = router_window_encoded;
        if (n >= 2 && n <= 512 && mtp_router_window(prepared)) router_window_encoded += 48;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
        const auto hc_down_window_before = hc_down_window_encoded;
        if (n >= 2 && n <= 512 && mtp_hc_down_window(prepared)) hc_down_window_encoded += 96;
#endif
        const bool qsa_sparse_active =
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
            mtp_qsa_sparse(prepared);
        const auto qsa_sparse_before = qsa_sparse_completed;
#else
            false;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
        const bool gdn_ported_active = mtp_gdn_ported(prepared);
        const auto gdn_ported_before = gdn_ported_completed;
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        const auto hc_down_before = hc_down_encoded;
        if (n == 512 && mtp_hc_down_f32(prepared)) hc_down_encoded += 7; // Synthetic, not a production route estimate.
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        const auto old_order_before = old_order_encoded;
        if (n > 1 && mtp_gdn_old_order(prepared)) old_order_encoded += 36;
#endif
        const auto* command = job.find("mtp_generation");
        bool aborted = false;
        if (scoped) {
            if (n || reset) {
                if (pending_terminal) ++counters[7];
                generation_active = pending_terminal = false; generation_remaining = 0;
            } else {
                const auto selected_command = mtp_generation_command(*command); aborted = selected_command.abort;
                if (aborted) {
                    if (pending_terminal) ++counters[7];
                    generation_active = pending_terminal = false; generation_remaining = 0;
                } else if (selected_command.begin) {
                    generation_id = selected_command.id; generation_remaining = selected_command.remaining; generation_active = true;
                }
            }
        }
        const auto before_hc = hc_count; uint32_t ordinary = 0; Json::Array ids;
        bool stopped = false, carry = scoped && pending_terminal;
        const bool sampled = std::stod(std::string(selected.find("temperature")->number_lexeme())) > 0;
        const auto emit = [&](uint32_t id, uint32_t committed, bool batched) {
            ++prefix; ids.push_back(Json::integer(id));
            Json::Object token{{"event", Json("token")}, {"id", Json::integer(id)}, {"source_logit", number(0.5)},
                {"committed_prefix", Json::integer(committed)}, {"entropy_words", Json::integer(words)}, {"gpu_profile", Json()}};
            if (batched) { token.emplace_back("token_prefix", Json::integer(prefix)); token.emplace_back("round_committed_prefix", Json::integer(committed)); }
            events.emplace_back(std::move(token));
            stopped = id == 248046;
        };
        if (n) {
            prefix += n; counters[1] += n - (reset ? 1 : 0); hc_count += 100 * n;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            if (mtp_input_packet(prepared) >= 512) {
                if (n == 1) {
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
                    if (qsa_sparse_active) qsa_sparse_completed[0] += 12; else
#endif
                    stage += 12;
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
                    if (gdn_ported_active) { gdn_ported_completed[1] += 36; gdn_ported_completed[4] += 36; }
#endif
                }
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
                if (n >= 2 && n <= 512 && gdn_ported_active) gdn_ported_completed[0] += 36;
#endif
                if (n > (reset ? 1u : 0u)) ++priming_submits;
            } else
#endif
            { tiles += 12; rows += 12 * n; }
        } else while (ids.size() < maximum && !stopped) {
            const auto remaining = maximum - uint32_t(ids.size());
            if (remaining >= depth + 1 && stop_at != ids.size() + 1) {
                const uint32_t round_proposed = ((depth == 2 || depth == 3) && force_proposed) ? force_proposed : depth;
                uint32_t accepted = force_accepted <= round_proposed ? force_accepted : use_acceptance ? round_proposed : 0;
                const auto base = prefix;
                const bool draft_stop = accepted && stop_at >= ids.size() + 2 && stop_at <= ids.size() + 1 + accepted;
                if (draft_stop) accepted = stop_at - uint32_t(ids.size()) - 1;
                const auto count = 1 + accepted;
                const bool used_carry = carry;
                if (carry) ++counters[6];
                if (sampled) words += (lookup_bypass ? 0u : round_proposed) + (carry ? 1 : 2);
                carry = !draft_stop;
                counters[0] += lookup_bypass ? 0u : round_proposed;
                counters[1] += accepted + (lookup_bypass ? 1u : 0u);
                ++counters[2];
                counters[3] += round_proposed;
                counters[4] += accepted; counters[5] += accepted < round_proposed; tiles += 12; rows += 12 * (round_proposed + 1); hc_count += 100 * (round_proposed + 1);
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
                if (qsa_sparse_active) qsa_sparse_completed[1] += 12;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
                if (gdn_ported_active) {
                    gdn_ported_completed[2] += 36;
                    if (accepted < round_proposed) gdn_ported_completed[3] += 36;
                    gdn_ported_completed[4] += 36;
                }
#endif
                events.emplace_back(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
                    {"committed_prefix", Json::integer(base + count)}, {"count", Json::integer(count)}, {"accepted", Json::integer(accepted)},
                    {"draft_wall_ms", number(lookup_bypass ? 0 : 1)}, {"draft_gpu_ms", number(lookup_bypass ? 0 : 1)}, {"verification_wall_ms", number(2)}, {"verification_gpu_ms", number(2)},
                    {"recovery_wall_ms", number(0)}, {"recovery_gpu_ms", number(0)}, {"materialization_wall_ms", number(0)},
                    {"lookahead_anchor", Json(true)}, {"used_carried_anchor", Json(used_carry)}, {"pending_anchor", Json(carry)},
                    {"round_wall_ms", number(3)}, {"timings_qualified", Json(false)}});
                if (depth == 2 || depth == 3) {
                    events.back() = with(std::move(events.back()), "proposed", Json::integer(round_proposed));
                    if (lookup_bypass) {
                        events.back() = with(std::move(events.back()), "head_calls", Json::integer(0));
                        events.back() = with(std::move(events.back()), "prime_extra", Json::integer(1));
                    }
                }
                emit(42, base + count, true);
                for (uint32_t k = 0; k < accepted; ++k) emit(draft_stop && k + 1 == accepted ? 248046 : 42, base + count, true);
            } else {
                if (carry) ++counters[6]; else if (sampled) ++words;
                carry = false; ++ordinary; ++counters[1];
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
                if (qsa_sparse_active) qsa_sparse_completed[0] += 12; else
#endif
                stage += 12;
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
                if (gdn_ported_active) { gdn_ported_completed[1] += 36; gdn_ported_completed[4] += 36; }
#endif
                hc_count += 100;
                emit(stop_at == ids.size() + 1 ? 248046 : 42, prefix + 1, false);
            }
        }
        if (scoped && maximum) {
            generation_remaining -= uint32_t(ids.size());
            if (stopped || !generation_remaining) {
                if (carry) ++counters[7];
                generation_active = carry = false; generation_remaining = 0;
            }
            pending_terminal = carry;
        } else if (!scoped && carry) ++counters[7];
        auto done = with(prepared, "event", Json("done"));
        for (const auto name : sampled_prefill_fixture::ready_only_fields) done = without(done, name);
        done = with(std::move(done), "committed_prefix", Json::integer(prefix));
        done = with(std::move(done), "generated_ids", Json(std::move(ids)));
        done = with(std::move(done), "known_token_prefill", Json(true));
        done = with(std::move(done), "status", Json(n || aborted ? "prefilled" : stopped ? "stopped" : "length"));
        done = with(std::move(done), "finish_reason", n || aborted ? Json() : Json(stopped ? "stop" : "length"));
        done = with(std::move(done), "stop_id", stopped ? Json::integer(248046) : Json());
        done = with(std::move(done), "continuation_allowed", Json(!stopped));
        auto sampled_state = selected.object();
        for (const auto& item : Json::Object{{"valid", Json(true)}, {"seed", Json::integer(seed)}, {"entropy_prefix", Json::integer(prefix)},
            {"entropy_words", Json::integer(words)}, {"filter_order", Json("fixture-filter")}, {"rng", Json("fixture-rng")}, {"ownership", Json("fixture-owner")}})
            sampled_state.push_back(item);
        done = with(std::move(done), "sampling", Json(std::move(sampled_state)));
        for (size_t i = 0; i < counters.size(); ++i) done = with(std::move(done), counter_names[i], Json::integer(counters[i]));
        for (const auto name : clock_names) done = with(std::move(done), name, number(counters[0] + counters[1]));
        for (const auto name : {"qsa_stage_v8_calls", "qsa_stage_v8_extra_dispatches", "qsa_channel_calls", "qsa_channel_extra_dispatches"})
            done = with(std::move(done), name, Json::integer(stage));
        done = with(std::move(done), "qsa_stage_v8_physical_dispatches", Json::integer(2 * stage));
        const auto ordinary_attention = ordinary
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            + (mtp_input_packet(prepared) >= 512 && n == 1 ? 1u : 0u)
#endif
            ;
        for (const auto name : {"qsa_stage_v8_request_calls", "qsa_channel_request_calls"}) done = with(std::move(done), name, Json::integer(qsa_sparse_active ? 0 : 12 * ordinary_attention));
        done = with(std::move(done), "known_qsa_rows_physical_tiles", Json::integer(tiles));
        done = with(std::move(done), "known_qsa_rows_logical_rows", Json::integer(rows));
        done = with(std::move(done), "hc_exact_routes", hc({hc_count, 0, 0, 0, 0}));
        done = with(std::move(done), "hc_exact_request_routes", hc({hc_count - before_hc, 0, 0, 0, 0}));
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        if (mtp_input_packet(prepared) >= 512) done = with(std::move(done), "native_mtp_priming_batch_submits", Json::integer(priming_submits));
#endif
        if (scoped) done = with(std::move(done), "native_mtp_generation", Json(Json::Object{{"id", Json::integer(generation_id)},
            {"active", Json(generation_active)}, {"remaining", Json::integer(generation_remaining)}, {"pending", Json(pending_terminal)}}));
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if(const auto* mode=prepared.find("q8_three_rows_mode")){
            // Synthetic completed-route increments exercise transport accounting, not production operation counts.
            if(mode->string()=="on"&&depth==2)n3_routes[0]+=counters[2]-n3_rounds_before;
            Json::Object total,delta;constexpr std::array names{"raw","packed","hc_raw","wide_qkv_raw","wide_output_raw"};
            for(size_t i=0;i<names.size();++i){total.emplace_back(names[i],Json::integer(n3_routes[i]));delta.emplace_back(names[i],Json::integer(n3_routes[i]-n3_before[i]));}
            done=with(with(done,"q8_three_rows_routes",Json(std::move(total))),"q8_three_rows_request_routes",Json(std::move(delta)));
        }
#endif
        if (mtp_final_input_logits(prepared)) {
            const bool available = !n || job.find("prefill_final")->boolean();
            done = with(std::move(done), "logits_available", Json(available));
            done = with(std::move(done), "continuation_allowed", Json(available && !stopped));
            done = with(std::move(done), "hc_exact_final_logit_sha256", available ? Json(std::string(64, 'a')) : Json());
            Json::Array top; for (uint32_t i = 0; i < 8; ++i) top.emplace_back(Json::Object{{"id", Json::integer(i)}, {"logit", number(8 - i)}});
            done = with(std::move(done), "final_logit_top8", available ? Json(std::move(top)) : Json());
        }
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        if (mtp_gdn_old_order(prepared)) {
            done = with(std::move(done), "native_gdn_old_order_encoded_calls", Json::integer(old_order_encoded));
            done = with(std::move(done), "native_gdn_old_order_request_encoded_calls", Json::integer(old_order_encoded - old_order_before));
        }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        if (mtp_hc_down_f32(prepared)) done = hc_down_fields(std::move(done), hc_down_encoded, hc_down_encoded - hc_down_before);
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (const auto* mode = prepared.find("native_raw_decode_mode")) done = decode_fields(done, mode->string() == "on", decode_routes, raw_decode_routes_delta(decode_routes, decode_before));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
        if (mtp_hc_rows(prepared)) done = hc_rows_fields(std::move(done), hc_rows_encoded, hc_rows_encoded - hc_rows_before);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
        if (mtp_router_window(prepared)) done = router_window_fields(std::move(done), router_window_encoded, router_window_encoded - router_window_before);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
        if (mtp_hc_down_window(prepared)) done = hc_down_window_fields(std::move(done), hc_down_window_encoded, hc_down_window_encoded - hc_down_window_before);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
        if (qsa_sparse_active) done = qsa_sparse_fields(std::move(done), qsa_sparse_completed, {qsa_sparse_completed[0] - qsa_sparse_before[0], qsa_sparse_completed[1] - qsa_sparse_before[1]});
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
        if (gdn_ported_active) {
            std::array<uint64_t, 5> delta{};
            for (size_t i = 0; i < 5; ++i) delta[i] = gdn_ported_completed[i] - gdn_ported_before[i];
            done = gdn_ported_fields(std::move(done), gdn_ported_completed, delta);
        }
#endif
        if (broken_done) done = with(std::move(done), "native_mtp_rounds", Json::integer(999));
        for (const auto& event : events) callback(event);
        previous = done; return done;
    }
};
Json input(uint32_t n = 8, bool reset = true) {
    Json::Array ids; for (uint32_t i = 0; i < n; ++i) ids.push_back(Json::integer(65));
    Json::Object fields{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)}, {"reset", Json(reset)}};
    if (reset) { fields.emplace_back("sampling", scalar_fixture::sampled_policy()); fields.emplace_back("seed", Json::integer(44)); }
    return Json(std::move(fields));
}
void contracts() {
    const auto r = ready(); validate_mtp_ready(r); ++checks;
#ifndef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    auto uncompiled_hc = r; for (const auto& [name, value] : mtp_hc_down_f32_fields(0, 0)) uncompiled_hc = with(std::move(uncompiled_hc), name, value);
    refuses([&] { validate_mtp_ready(uncompiled_hc); });
#endif
#ifndef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    refuses([&] { validate_mtp_ready(hc_rows_fields(r)); });
#endif
#ifndef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    refuses([&] { validate_mtp_ready(gdn_ported_fields(r)); });
#endif
    check(mtp_input_packet(r) == 8, "omitted input-packet metadata changed the default bound");
#ifndef TOKSPUR_PRIVATE_HTTP_PREFILL512
    auto unsupported = with(with(r, "native_mtp_input_packet_schema", Json::integer(1)), "native_mtp_input_packet_max", Json::integer(512));
    refuses([&] { static_cast<void>(mtp_input_packet(unsupported)); });
    refuses([&] { validate_mtp_input_packet_selection(unsupported, true); });
#endif
    refuses([&] { validate_mtp_ready(with(r, "native_mtp_protocol", Json("native-mtp-depth1-consumed-v1"))); });
    refuses([&] { validate_mtp_ready(with(r, "native_mtp_lookahead_anchor", Json(false))); });
    refuses([&] { validate_mtp_ready(with(r, "verification_gpu_bytes", Json::integer(0))); });
    refuses([&] { validate_mtp_ready(with(r, "native_mtp_head_calls", Json::integer(1))); });
    for (const auto mode : {"serial", "tiled128", "tiled256"}) {
        const auto prepared = with(r, "native_mtp_draft_attention", Json(mode));
        validate_mtp_ready(prepared, mode); ++checks;
        if (std::string_view(mode) != "serial") refuses([&] { validate_mtp_ready(prepared); });
        Fake fake; fake.prepared = prepared;
        const auto job = input(); MtpWireReply reply(prepared, prepared, job);
        const auto done = fake(job, [](const Json&) {});
        reply.finish(done); ++checks;
        MtpWireReply drift(prepared, prepared, job);
        refuses([&] { drift.finish(with(done, "native_mtp_draft_attention", Json(std::string_view(mode) == "serial" ? "tiled128" : "serial"))); });
        refuses([&] { MtpWireReply wrong_previous(prepared, with(prepared, "native_mtp_draft_attention", Json("unknown")), job); });
    }
    refuses([&] { validate_mtp_ready(with(r, "native_mtp_draft_attention", Json("tiled512")), "tiled512"); });
    validate_mtp_ready(with(r, "native_mtp_draft_attention", Json("serial"))); ++checks;
    for (const bool accepted : {false, true}) for (uint32_t width = 1; width <= 8; ++width) for (uint32_t eos = 0; eos <= width; ++eos) {
        Fake f; f.use_acceptance = accepted;
        const auto start = input(); MtpWireReply prefill(r, r, start);
        auto prior = f(start, [&](const Json& e) { prefill.event(e); }); prefill.finish(prior);
        f.stop_at = eos; const auto job = generation_job(width); MtpWireReply reply(r, prior, job);
        auto done = f(job, [&](const Json& event) { reply.event(event); });
        refuses([&] { static_cast<void>(reply.events()); });
        reply.finish(done); check(!reply.events().empty(), "MTP admitted output lost events");
        check(integer(done, "committed_prefix") == 8 + (eos ? eos : width), "MTP wire visible prefix mismatch");
        refuses([&] { reply.finish(done); });
        auto bad = with(done, "gpu_allocation_count", Json::integer(999));
        MtpWireReply drift(r, prior, job); for (const auto& event : f.events) drift.event(event);
        refuses([&] { drift.finish(bad); });
        MtpWireReply counter(r, prior, job); for (const auto& event : f.events) counter.event(event);
        refuses([&] { counter.finish(with(done, "native_mtp_accepted", Json::integer(999))); });
    }
}
void sessions() {
    Fake f; ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, f.prepared);
    ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
    auto request = with(chat(std::string(17, 'A'), false, 17), "temperature", number(1));
    const auto result = session.complete(request, "mtp-input-output-eight", [&](const Json& job, const std::function<void(const Json&)>& cb) { return f(job, cb); }, sink);
    check(integer(*result.find("usage"), "completion_tokens") == 17, "MTP HTTP output budget changed");
    check(f.packets.size() == 6 && f.packets[0].find("tokens")->array().size() == 8 &&
        integer(f.packets[3], "generate") == 8 && integer(f.packets[5], "generate") == 1, "MTP HTTP did not batch input/output8");
    check(integer(*result.find("tokspur_private"), "maximum_generation_width") == 8, "MTP HTTP width diagnostics differ");
    bool cancel = false; Fake stopped; ServeSession cancelled(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, stopped.prepared);
    ServeSink interrupted{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
    refuses([&] { cancelled.complete(request, "mtp-cancel", [&](const Json& job, const std::function<void(const Json&)>& cb) { auto done = stopped(job, cb); cancel = true; return done; }, interrupted); });
    check(cancelled.committed_tokens() == 0, "MTP cancellation published serving history");
    cancel = false;
    static_cast<void>(cancelled.complete(request, "mtp-recover", [&](const Json& job, const std::function<void(const Json&)>& cb) { return stopped(job, cb); }, sink));
    check(stopped.packets[1].find("reset")->boolean(), "MTP cancelled session did not cold reset");
}
void context_budget_sessions() {
    const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
    for (uint32_t remaining : {1u,2u,9u}) {
        Fake fake;
        ServeSession session(session_codec(),"tokspur","direct",Json(),false,Json(),false,ToolEnumValidation::strict,false,fake.prepared);
        auto request=with(chat(std::string(65536-remaining,'A'),false,32768),"temperature",number(1));
        const auto result=session.complete(request,"mtp-context-tail",[&](const Json& job,const std::function<void(const Json&)>& event) { return fake(job,event); },sink);
        check(integer(*result.find("usage"),"completion_tokens")==remaining && session.committed_tokens()==65536,
            "MTP context clamp lost the final available tokens or overflowed context");
        check(result.find("choices")->array()[0].find("finish_reason")->string()=="length",
            "MTP context exhaustion must report length");
        uint32_t requested=0;
        for (const auto& job:fake.packets) {
            const auto count=uint32_t(integer(job,"generate"));
            if (count) { check(count<=8 && requested+count<=remaining,"MTP packet exceeded remaining context"); requested+=count; }
        }
        check(requested==remaining,"MTP generation budget was not clamped before packetisation");
    }
}
void generation_contracts() {
    const auto prepared = scoped_ready();
    validate_mtp_ready(prepared, "serial", true);
    refuses([&] { validate_mtp_ready(prepared); });
    refuses([&] { validate_mtp_ready(ready(), "serial", true); });
    refuses([&] { validate_mtp_ready(with(prepared, "native_mtp_carry_scope", Json("unknown")), "serial", true); });
    for (const auto name : {"native_mtp_carry_scope", "native_mtp_generation"}) {
        refuses([&] { validate_mtp_ready(without(prepared, name), "serial", true); });
        for (const auto& value : {Json(), Json(true), Json::integer(1)})
            refuses([&] { validate_mtp_ready(with(prepared, name, value), "serial", true); });
    }
    for (const auto name : {"id", "active", "remaining", "pending"}) {
        const auto generation = *prepared.find("native_mtp_generation");
        refuses([&] { validate_mtp_ready(with(prepared, "native_mtp_generation", without(generation, name)), "serial", true); });
    }
    Fake fake; fake.prepared = fake.previous = prepared;
    const auto run = [&](const Json& job) {
        const auto previous_value = fake.previous;
        MtpWireReply wire(prepared, previous_value, job);
        const auto done = fake(job, [&](const Json& event) { wire.event(event); }); wire.finish(done); return done;
    };
    auto previous = run(input());
    refuses([&] { MtpWireReply wire(prepared, previous, generation_job(2)); });
    auto first = scope_job(generation_job(2), 1, true, 4);
    auto done = run(first);
    check(done.find("native_mtp_generation")->find("pending")->boolean() && integer(done, "native_mtp_discarded_anchors") == 0,
        "packet boundary discarded a terminal inside one generation");
    const auto entropy = integer(*done.find("sampling"), "entropy_words");
    for (const auto& invalid : {scope_job(generation_job(2), 2, true, 2), scope_job(generation_job(2), 1, false, 3),
         scope_job(generation_job(2), 2, false, 2)})
        refuses([&] { MtpWireReply wire(prepared, done, invalid); });
    auto next = scope_job(generation_job(1), 1, false, 2);
    const auto tail = run(next);
    check(integer(tail, "native_mtp_carried_anchors") == 1 && integer(*tail.find("sampling"), "entropy_words") == entropy,
        "one-token continuation redrew an already-sampled terminal");
    auto final_job = scope_job(generation_job(1), 1, false, 1);
    const auto before_final = fake.previous;
    auto final = fake(final_job, [](const Json&) {});
    MtpWireReply final_wire(prepared, before_final, final_job);
    for (const auto& event : fake.events) final_wire.event(event);
    final_wire.finish(final);
    check(!final.find("native_mtp_generation")->find("active")->boolean(), "final output budget did not close generation");
    for (const auto& invalid : {without(final, "native_mtp_generation"), with(final, "native_mtp_carry_scope", Json("other")),
         with(final, "native_mtp_generation", Json(Json::Object{{"id", Json::integer(9)}, {"active", Json(false)},
            {"remaining", Json::integer(0)}, {"pending", Json(false)}}))}) {
        MtpWireReply bad(prepared, before_final, final_job);
        for (const auto& event : fake.events) bad.event(event);
        refuses([&] { bad.finish(invalid); });
    }
    done = run(scope_job(generation_job(2), 2, true, 4));
    const auto abort = Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(0)}, {"reset", Json(false)},
        {"mtp_generation", Json(Json::Object{{"id", Json::integer(2)}, {"abort", Json(true)}})}});
    const auto aborted = run(abort);
    check(integer(aborted, "committed_prefix") == integer(done, "committed_prefix") &&
        integer(*aborted.find("sampling"), "entropy_words") == integer(*done.find("sampling"), "entropy_words") &&
        integer(aborted, "native_mtp_discarded_anchors") == integer(done, "native_mtp_discarded_anchors") + 1,
        "abort advanced physical prefix/RNG or lost discard accounting");
    refuses([&] { MtpWireReply wire(prepared, aborted, abort); });
    done = run(scope_job(generation_job(2), 3, true, 4));
    const auto cleared = run(input(1, false));
    check(!cleared.find("native_mtp_generation")->find("active")->boolean() &&
        integer(cleared, "native_mtp_discarded_anchors") == integer(done, "native_mtp_discarded_anchors") + 1,
        "new known input did not terminate the generation");
}

void generation_sessions() {
    Fake fake; fake.prepared = fake.previous = scoped_ready();
    ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false,
        ToolEnumValidation::strict, false, fake.prepared);
    const auto request = with(chat(std::string(17, 'A'), false, 17), "temperature", number(1));
    const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
    const auto result = session.complete(request, "generation-lifetime", [&](const Json& job, const std::function<void(const Json&)>& cb) {
        return fake(job, cb);
    }, sink);
    size_t packets = 0; uint32_t remaining = 17;
    for (const auto& packet : fake.packets) if (integer(packet, "generate")) {
        const auto command = mtp_generation_command(*packet.find("mtp_generation"));
        check(command.id == 1 && command.begin == (packets == 0) && command.remaining == remaining,
            "HTTP generation lifetime changed at a native packet boundary");
        remaining -= uint32_t(integer(packet, "generate")); ++packets;
    }
    check(packets == 3 && !remaining && !fake.generation_active, "HTTP completion did not end the scoped native generation");
    const auto& details = *result.find("tokspur_private");
    check(details.find("native_mtp_carry_scope")->string() == "http-generation-v1" &&
        !details.find("native_mtp_generation")->find("active")->boolean(), "HTTP completion omitted carried-terminal scope");
    session.invalidate(); // ID must never be reused after cancellation/cold recovery.
    const auto first_packet = fake.packets.size();
    static_cast<void>(session.complete(request, "generation-recovery", [&](const Json& job, const std::function<void(const Json&)>& cb) {
        return fake(job, cb);
    }, sink));
    bool saw_generation = false;
    for (size_t i = first_packet; i < fake.packets.size(); ++i) if (const auto* command = fake.packets[i].find("mtp_generation")) {
        check(integer(*command, "id") == 2, "new HTTP request reused a previous generation ID"); saw_generation = true;
    }
    check(saw_generation, "recovery omitted generation packets");
}

void cold_generation_resync(const Json& prepared) {
    Fake fake; fake.prepared = fake.previous = prepared;
    const auto reset = input();
    const auto previous = fake(reset, [](const Json&) {});
    // The child completes a generation packet that cancellation prevents the caller from admitting.
    static_cast<void>(fake(scope_job(generation_job(8), 1, true, 16), [](const Json&) {}));
    check(fake.generation_id == 1 && fake.generation_active, "missed DONE did not advance the child's generation");
    const auto reset_done = fake(reset, [](const Json&) {});
    MtpWireReply recovered(prepared, previous, reset, true); recovered.finish(reset_done);
    check(recovered.events().empty() && integer(*reset_done.find("native_mtp_generation"), "id") == 1,
        "cold reset did not recover the drained monotone generation identity");
    { MtpWireReply strict(prepared, previous, reset); refuses([&] { strict.finish(reset_done); }); }
    const auto future = with(previous, "native_mtp_generation", with(*previous.find("native_mtp_generation"), "id", Json::integer(2)));
    { MtpWireReply backwards(prepared, future, reset, true); refuses([&] { backwards.finish(reset_done); }); }
    for (const auto& generation : {
        with(with(*reset_done.find("native_mtp_generation"), "active", Json(true)), "remaining", Json::integer(1)),
        with(*reset_done.find("native_mtp_generation"), "pending", Json(true))}) {
        MtpWireReply invalid(prepared, previous, reset, true);
        refuses([&] { invalid.finish(with(reset_done, "native_mtp_generation", generation)); });
        refuses([&] { static_cast<void>(invalid.events()); });
    }
    refuses([&] { MtpWireReply append(prepared, previous, input(1, false), true); });
}

int target(int argc, char** argv) {
    std::map<std::string, std::string> flags;
    for (int i = 2; i < argc; ++i) if (std::string_view(argv[i]).starts_with("--")) {
        const std::string name(argv[i]); if (++i == argc || !flags.emplace(name, argv[i]).second) return 91;
    }
    bool input512 = false;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    input512 = flags.contains("--native-prefill-chunks");
    if (input512) {
        if (flags.at("--native-prefill-chunks") != "fused" || (flags["--native-prefill-dense"] != "f16" && flags["--native-prefill-dense"] != "f32"
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            && flags["--native-prefill-dense"] != "f32-staged"
#endif
            ) || flags.contains("--known-qsa-rows")) return 101;
    } else if (flags.contains("--native-prefill-dense") || flags.contains("--native-prefill-grouped")) return 101;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (flags.contains("--native-prefill-grouped") && flags.at("--native-prefill-grouped") != "serial" && flags.at("--native-prefill-grouped") != "f32" && flags.at("--native-prefill-grouped") != "auto") return 102;
#else
    if (flags.contains("--native-prefill-grouped")) return 102;
#endif
#endif
    if (!input512 && flags["--known-qsa-rows"] != "q8") return 92;
    for (const auto& [name, value] : std::map<std::string, std::string>{{"--known-token-capacity", input512 ? "512" : "8"}, {"--reuse-residual-projections", "true"},
        {"--mtp-lookahead-anchor", "on"}, {"--hc-exact", "combined"}, {"--qsa-stage-v8", "on"},
        {"--qsa-channel-chronological", "on"}, {"--ple-gate-transform", "signed-root"}}) if (flags[name] != value) return 92;
    if (!flags.contains("--mtp-sidecar") || flags.contains("--sampled-prefill-input8") || flags.contains("--prompt-lookup")) return 93;
    const auto path = std::filesystem::path(flags.at("--policy")); Fake fake;
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    if (flags.contains("--raw-decode")) {
        const auto mode = flags.at("--raw-decode"); if (mode != "on" && mode != "off") return 104;
        if (path.filename() == "decode-selector-on" && mode != "on") return 104;
        if (path.filename() == "decode-selector-off" && mode != "off") return 104;
        fake.prepared = decode_fields(fake.prepared, mode == "on");
    } else if (path.filename().string().starts_with("decode-selector-")) return 104;
#endif
    if (flags.contains("--mtp-draft-attention")) {
        const auto& mode = flags.at("--mtp-draft-attention");
        if (mode != "serial" && mode != "tiled128" && mode != "tiled256") return 94;
        fake.prepared = with(fake.prepared, "native_mtp_draft_attention", Json(mode));
    }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (flags.contains("--weight-lanes")) {
        if (flags.at("--weight-lanes") != (path.parent_path()/"lanes").string()) return 95;
        fake.prepared = scalar_fixture::lane_fields(fake.prepared);
    }
    if (flags.contains("--weight-lanes-auth-workers")) {
        if (!flags.contains("--weight-lanes")) return 96;
        const auto count = parse_json(flags.at("--weight-lanes-auth-workers")).unsigned_integer();
        if (!count || count > 16) return 96;
        fake.prepared = scalar_fixture::lane_auth_fields(fake.prepared, count);
    }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    if (flags.contains("--mtp-depth")) {
        if (flags.at("--mtp-depth") != "1" && flags.at("--mtp-depth") != "2") return 98;
        if (flags.at("--mtp-depth") == "2") {
            fake.prepared = with(fake.prepared, "native_mtp_depth", Json::integer(2));
            fake.prepared = with(fake.prepared, "native_mtp_protocol", Json("native-mtp-depth2-lookahead-v1"));
            fake.prepared = with(fake.prepared, "verification_max_tokens", Json::integer(3));
            fake.prepared = with(fake.prepared, "native_mtp_tail_policy", Json("remaining1-or2 ordinary; no depth-one rounds"));
        }
    }
#endif
    if (flags.contains("--mtp-request-carry")) {
        if (flags.at("--mtp-request-carry") != "on") return 97;
        const auto scoped = scoped_ready();
        for (const auto name : {"native_mtp_carry_scope", "native_mtp_generation"}) fake.prepared = with(std::move(fake.prepared), name, *scoped.find(name));
    }
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
    if (flags.contains("--concurrent-dispatch") || path.filename() == "concurrent-unsolicited") {
        const std::string mode = flags.contains("--concurrent-dispatch") ? flags.at("--concurrent-dispatch") : "on";
        if (mode != "off" && mode != "on") return 99;
        for (const auto& [name, value] : Json::Object{{"concurrent_dispatch_schema", Json::integer(1)},
            {"concurrent_dispatch_mode", Json(mode)}, {"concurrent_dispatch_barriers", Json::integer(0)}, {"concurrent_state_proof", Json(false)}})
            fake.prepared = with(fake.prepared, name, value);
    }
#endif
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    if (flags.contains("--mtp-output-packet") || path.filename() == "packet64-unsolicited") {
        const auto cap = flags.contains("--mtp-output-packet") ? flags.at("--mtp-output-packet") : "64";
        if (cap != "8" && cap != "64") return 100;
        fake.prepared = with(fake.prepared, "native_mtp_output_packet_schema", Json::integer(1));
        fake.prepared = with(fake.prepared, "native_mtp_output_packet", Json::integer(cap == "64" ? 64 : 8));
    }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    if(flags.contains("--q8-three-rows")){
        const auto mode=flags.at("--q8-three-rows");if(mode!="on"&&mode!="off")return 102;
        if(path.filename()=="q8-three-on"&&mode!="on")return 102;if(path.filename()=="q8-three-off"&&mode!="off")return 102;
        fake.prepared=q8_three_fields(fake.prepared,mode=="on");
    }else if(path.filename().string().starts_with("q8-three-"))return 102;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    if (input512 || path.filename() == "prefill512-unsolicited") fake.prepared = prefill512_ready(fake.prepared,
        input512 ? std::string_view(flags.at("--native-prefill-dense")) : std::string_view("f16"),
        flags.contains("--native-prefill-grouped") ? std::optional<std::string_view>(flags.at("--native-prefill-grouped")) : std::nullopt);
    const auto mode = path.filename().string();
    if (mode.starts_with("precision-argv-")) {
        const auto expected = std::string("precision-argv-") + flags["--native-prefill-dense"] + "-" +
            (flags.contains("--native-prefill-grouped") ? flags.at("--native-prefill-grouped") : "absent");
        if (mode != expected || flags.contains("--q8-three-rows") || flags.contains("--native-residency")) return 103;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    if (flags.contains("--native-prefill-final-logits")) {
        if (!input512 || flags.at("--native-prefill-final-logits") != "on") return 104;
        fake.prepared = with(with(fake.prepared, "native_prefill_output_policy", Json("final-input-v1")), "logits_available", Json(false));
    }
    if (path.filename().string().starts_with("final-logits-argv-")) {
        const bool expected = path.filename() == "final-logits-argv-on";
        if (flags.contains("--native-prefill-final-logits") != expected) return 105;
    }
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    if (flags.contains("--native-gdn-old-order-batch")) {
        if (!input512 || flags.at("--native-gdn-old-order-batch") != "on" || (flags["--native-prefill-dense"] != "f32" && flags["--native-prefill-dense"] != "f32-staged") || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 106;
        fake.prepared = old_order_fields(fake.prepared);
    }
    if (path.filename().string().starts_with("gdn-old-order-argv-")) {
        const bool selected = path.filename() == "gdn-old-order-argv-on";
        if (flags.contains("--native-gdn-old-order-batch") != selected) return 107;
    }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    if (flags.contains("--native-prefill-hc-down-f32")) {
        const auto mode = flags.at("--native-prefill-hc-down-f32"); if (mode != "on" && mode != "off") return 108;
        if (mode == "on") { if (!input512 || flags["--native-prefill-dense"] != "f32" || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 109; fake.prepared = hc_down_fields(fake.prepared); }
    }
    if (path.filename().string().starts_with("hc-down-argv-")) {
        const auto mode = path.filename().string().substr(std::string("hc-down-argv-").size());
        if (mode == "omitted" ? flags.contains("--native-prefill-hc-down-f32") : !flags.contains("--native-prefill-hc-down-f32") || flags.at("--native-prefill-hc-down-f32") != mode) return 110;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    if (flags.contains("--native-prefill-hc-rows")) {
        const auto mode = flags.at("--native-prefill-hc-rows"); if (mode != "on" && mode != "off") return 111;
        if (mode == "on") {
            if (!input512 || flags["--native-prefill-dense"] != "f32-staged" || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 112;
            fake.prepared = hc_rows_fields(fake.prepared);
        }
    }
    if (path.filename().string().starts_with("hc-rows-argv-")) {
        const auto mode = path.filename().string().substr(std::string("hc-rows-argv-").size());
        if (mode == "omitted" ? flags.contains("--native-prefill-hc-rows") : !flags.contains("--native-prefill-hc-rows") || flags.at("--native-prefill-hc-rows") != mode) return 113;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    if (flags.contains("--native-prefill-router-window")) {
        const auto mode = flags.at("--native-prefill-router-window"); if (mode != "on" && mode != "off") return 114;
        if (mode == "on") {
            if (!input512 || flags["--native-prefill-dense"] != "f32-staged" || flags["--native-prefill-hc-rows"] != "on" || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 115;
            fake.prepared = router_window_fields(fake.prepared);
        }
    }
    if (path.filename().string().starts_with("router-window-argv-")) {
        const auto mode = path.filename().string().substr(std::string("router-window-argv-").size());
        if (mode == "omitted" ? flags.contains("--native-prefill-router-window") : !flags.contains("--native-prefill-router-window") || flags.at("--native-prefill-router-window") != mode) return 116;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    if (flags.contains("--native-prefill-hc-down-window")) {
        const auto mode = flags.at("--native-prefill-hc-down-window"); if (mode != "on" && mode != "off") return 117;
        if (mode == "on") {
            if (!input512 || flags["--native-prefill-dense"] != "f32-staged" || flags["--native-prefill-hc-rows"] != "on" || flags["--native-prefill-router-window"] != "on" || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 118;
            fake.prepared = hc_down_window_fields(fake.prepared);
        }
    }
    if (path.filename().string().starts_with("hc-down-window-argv-")) {
        const auto mode = path.filename().string().substr(std::string("hc-down-window-argv-").size());
        if (mode == "omitted" ? flags.contains("--native-prefill-hc-down-window") : !flags.contains("--native-prefill-hc-down-window") || flags.at("--native-prefill-hc-down-window") != mode) return 119;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    if (flags.contains("--native-qsa-sparse")) {
        const auto mode = flags.at("--native-qsa-sparse"); if (mode != "on" && mode != "off") return 120;
        if (mode == "on") {
            if (!input512 || flags["--native-prefill-dense"] != "f32-staged" || flags["--native-prefill-hc-rows"] != "on" ||
                flags["--native-prefill-router-window"] != "on" || flags["--native-prefill-hc-down-window"] != "on" || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 121;
            fake.prepared = qsa_sparse_fields(fake.prepared);
        }
    }
    if (path.filename().string().starts_with("qsa-sparse-argv-")) {
        const auto mode = path.filename().string().substr(std::string("qsa-sparse-argv-").size());
        if (mode == "omitted" ? flags.contains("--native-qsa-sparse") : !flags.contains("--native-qsa-sparse") || flags.at("--native-qsa-sparse") != mode) return 122;
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    if (flags.contains("--native-gdn-ported")) {
        const auto mode = flags.at("--native-gdn-ported"); if (mode != "on" && mode != "off") return 123;
        if (mode == "on") {
            if (!input512 || flags["--native-prefill-dense"] != "f32-staged" || flags["--native-prefill-hc-rows"] != "on" ||
                flags["--native-prefill-router-window"] != "on" || flags["--native-prefill-hc-down-window"] != "on" || (flags["--native-prefill-grouped"] != "f32" && flags["--native-prefill-grouped"] != "auto")) return 124;
            fake.prepared = gdn_ported_fields(fake.prepared);
        }
    }
    if (path.filename().string().starts_with("gdn-ported-argv-")) {
        auto mode = path.filename().string().substr(std::string("gdn-ported-argv-").size());
        if (mode.starts_with("qsa-")) {
            if (!flags.contains("--native-qsa-sparse") || flags.at("--native-qsa-sparse") != "on") return 125;
            mode.erase(0, 4);
        } else if (mode.starts_with("noqsa-")) {
            if (!flags.contains("--native-qsa-sparse") || flags.at("--native-qsa-sparse") != "off") return 125;
            mode.erase(0, 6);
        }
        if (mode == "omitted" ? flags.contains("--native-gdn-ported") : !flags.contains("--native-gdn-ported") || flags.at("--native-gdn-ported") != mode) return 125;
    }
#endif
    auto prepared = fake.prepared;
    if (path.filename() == "hc-rows-ready-missing") prepared = without(prepared, "native_prefill_hc_rows_kernels");
    if (path.filename() == "hc-rows-ready-unsolicited") prepared = hc_rows_fields(prepared);
    if (path.filename() == "router-window-ready-missing") prepared = without(prepared, "native_prefill_router_window_kernels");
    if (path.filename() == "router-window-ready-unsolicited") prepared = router_window_fields(prepared);
    if (path.filename() == "hc-down-window-ready-missing") prepared = without(prepared, "native_prefill_hc_down_window_kernel");
    if (path.filename() == "hc-down-window-ready-unsolicited") prepared = hc_down_window_fields(prepared);
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    if (path.filename() == "qsa-sparse-ready-missing") prepared = without(prepared, "native_qsa_sparse_decode_kernel");
    if (path.filename() == "qsa-sparse-ready-unsolicited") prepared = qsa_sparse_fields(prepared);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    if (path.filename() == "gdn-ported-ready-missing") prepared = without(prepared, "native_gdn_ported_prefill_kernel");
    if (path.filename() == "gdn-ported-ready-unsolicited") prepared = gdn_ported_fields(prepared);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    if (path.filename() == "prefill512-missing-ready") prepared = without(prepared, "native_mtp_input_packet_schema");
    if (path.filename() == "prefill512-bad-priming-ready") prepared = with(prepared, "native_mtp_priming_batch_capacity", Json::integer(8));
    if (path.filename() == "precision-dense-ready-drift") prepared = with(prepared, "native_prefill_dense", Json("f16"));
    if (path.filename() == "staged-ready-missing") prepared = without(prepared, "native_prefill_dense_kernel");
    if (path.filename() == "staged-ready-drift") prepared = with(prepared, "native_prefill_dense_kernel", Json("graph_q8_mpp_wide"));
    if (path.filename() == "staged-ready-unsolicited") prepared = prefill512_ready(prepared, "f32-staged", "f32");
    if (path.filename() == "precision-grouped-ready-missing") prepared = without(prepared, "native_prefill_grouped_kernel");
    if (path.filename() == "precision-grouped-ready-unsolicited") prepared = prefill512_ready(prepared, "f32", "f32");
#endif
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    if (path.filename() == "packet64-missing-ready") prepared = without(prepared, "native_mtp_output_packet");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
    if (path.filename() == "concurrent-missing-ready") prepared = without(prepared, "concurrent_dispatch_mode");
    if (path.filename() == "concurrent-proof-ready") prepared = with(prepared, "concurrent_state_proof", Json(true));
#endif
    if (path.filename() == "bad-ready") prepared = with(std::move(prepared), "native_mtp_lookahead_anchor", Json(false));
    if (path.filename() == "bad-draft-mode") prepared = with(std::move(prepared), "native_mtp_draft_attention", Json("serial"));
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    if(path.filename()=="bad-depth2-ready")prepared=with(std::move(prepared),"native_mtp_depth",Json::integer(1));
    if(path.filename()=="bad-depth2-width")prepared=with(std::move(prepared),"verification_max_tokens",Json::integer(2));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    if (path.filename() == "final-logits-ready-drift") prepared = without(prepared, "native_prefill_output_policy");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    if (path.filename() == "gdn-old-order-ready-drift") prepared = without(prepared, "native_gdn_old_order_kernel");
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    if (path.filename() == "hc-down-ready-missing") prepared = without(prepared, "native_prefill_hc_down_f32_kernel");
#endif
    { std::ofstream started(path.parent_path() / "started"); started << "explicit sampled MTP arguments\n"; }
    emit(prepared); std::string line;
    while (std::getline(std::cin, line)) {
        const auto job = parse_json(line);
        auto done = fake(job, [](const Json& value) { emit(value); });
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (integer(job, "generate") && path.filename() == "decode-selector-drift") done = with(done, "native_raw_decode_fp_contract", Json("fast"));
        if (integer(job, "generate") && path.filename() == "decode-selector-missing") done = without(done, "native_raw_decode_routes");
        if (integer(job, "generate") && path.filename() == "decode-selector-underreported") done = with(done, "native_raw_decode_request_routes", raw_decode_routes_json({0, 0}));
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if(integer(job,"generate")&&path.filename()=="q8-three-drift")done=with(done,"q8_three_rows_mode",Json("off"));
        if(integer(job,"generate")&&path.filename()=="q8-three-missing")done=without(done,"q8_three_rows_routes");
        if(integer(job,"generate")&&path.filename()=="q8-three-underreported")done=with(done,"q8_three_rows_request_routes",with(*done.find("q8_three_rows_request_routes"),"raw",Json::integer(0)));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        if (path.filename() == "final-logits-done-drift") done = without(done, "native_prefill_output_policy");
        if (path.filename() == "final-logits-false-ready") done = with(done, "logits_available", Json(true));
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        if (path.filename() == "gdn-old-order-done-drift") done = without(done, "native_gdn_old_order_version");
        if (path.filename() == "gdn-old-order-call-drift") done = with(done, "native_gdn_old_order_request_encoded_calls", Json::integer(37));
#endif
        if (integer(job, "generate") && path.filename() == "prefill512-drift") done = with(done, "native_mtp_input_packet_max", Json::integer(8));
        if (integer(job, "generate") && path.filename() == "prefill512-missing-done") done = without(done, "native_mtp_priming_batch_bytes");
        if (integer(job, "generate") && path.filename() == "prefill512-unknown-done") done = with(done, "native_mtp_input_packet_extra", Json(true));
        if (integer(job, "generate") && path.filename() == "precision-dense-drift") done = with(done, "native_prefill_dense", Json("f16"));
        if ((integer(job, "generate") && path.filename() == "staged-done-missing") || (job.find("reset")->boolean() && path.filename() == "staged-cold-reset-missing")) done = without(done, "native_prefill_dense_kernel");
        if (integer(job, "generate") && path.filename() == "staged-done-drift") done = with(done, "native_prefill_dense_kernel", Json("graph_q8_mpp_wide"));
        if (integer(job, "generate") && path.filename() == "staged-done-fused-drift") done = with(done, "native_prefill_chunks", Json("cs64"));
        if (integer(job, "generate") && path.filename() == "precision-grouped-drift")
            done = with(with(done, "native_prefill_grouped", Json("serial")), "native_prefill_grouped_kernel", Json("graph_grouped_serial"));
        if (integer(job, "generate") && path.filename() == "precision-grouped-missing") done = without(done, "native_prefill_grouped");
        if (integer(job, "generate") && path.filename() == "precision-grouped-unknown") done = with(done, "native_prefill_grouped_extra", Json(false));
#endif
        if (integer(job, "generate") && path.filename() == "bad-resource") done = with(std::move(done), "gpu_allocation_count", Json::integer(999));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (integer(job, "generate") && path.filename() == "lane-drift")
            done = with(std::move(done), "weight_lanes_manifest_sha256", Json(std::string(64, 'b')));
        if (integer(job, "generate") && path.filename() == "lane-auth-drift")
            done = with(std::move(done), "lane_auth_workers_requested", Json::integer(8));
        if (integer(job, "generate") && path.filename() == "lane-auth-missing-done")
            done = without(done, "lane_auth_workers");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
        if (flags.contains("--concurrent-dispatch")) done = with(done, "concurrent_dispatch_barriers", Json::integer(7));
        if (integer(job, "generate") && path.filename() == "concurrent-drift") done = with(done, "concurrent_dispatch_mode", Json("off"));
        if (integer(job, "generate") && path.filename() == "concurrent-missing-done") done = without(done, "concurrent_dispatch_schema");
        if (integer(job, "generate") && path.filename() == "concurrent-unknown-done") done = with(done, "concurrent_unregistered", Json(true));
#endif
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        if (integer(job, "generate") && path.filename() == "packet64-drift") done = with(done, "native_mtp_output_packet", Json::integer(8));
        if (integer(job, "generate") && path.filename() == "packet64-missing-done") done = without(done, "native_mtp_output_packet_schema");
        if (integer(job, "generate") && path.filename() == "packet64-unknown-done") done = with(done, "native_mtp_output_packet_unregistered", Json(true));
        if (integer(job, "generate") && path.filename() == "packet64-slow") pause_ms(100);
#endif
        if (integer(job, "generate") && path.filename() == "slow-done") pause_ms(100);
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        if (path.filename() == "hc-rows-done-drift") done = with(done, "native_prefill_hc_rows_precision", Json("fp16"));
        if (path.filename() == "hc-rows-call-underreported") done = with(done, "native_prefill_hc_rows_read_request_encoded", Json::integer(95));
        if (path.filename() == "router-window-done-drift") done = with(done, "native_prefill_router_window_precision", Json("fp16"));
        if (path.filename() == "router-window-call-underreported") done = with(done, "native_prefill_router_window_router_request_encoded", Json::integer(47));
        if (path.filename() == "hc-down-window-done-drift") done = with(done, "native_prefill_hc_down_window_precision", Json("fp16"));
        if (path.filename() == "hc-down-window-call-underreported") done = with(done, "native_prefill_hc_down_window_request_encoded", Json::integer(95));
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
        if (path.filename() == "qsa-sparse-done-drift") done = with(done, "native_qsa_sparse_precision", Json("fp16"));
        if (path.filename() == "qsa-sparse-call-underreported" && job.find("tokens")->array().size() == 1) done = with(done, "native_qsa_sparse_decode_request_calls", Json::integer(11));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
        if (path.filename() == "gdn-ported-done-drift") done = with(done, "native_gdn_ported_precision", Json("fp16"));
        if (path.filename() == "gdn-ported-call-underreported" && job.find("tokens")->array().size() == 1) done = with(done, "native_gdn_ported_decode_request_calls", Json::integer(35));
#endif
        if (path.filename() == "hc-down-done-drift") done = with(done, "native_prefill_hc_down_f32_precision", Json("fp16"));
        if (path.filename() == "hc-down-call-underreported" && job.find("tokens")->array().size() == 512) done = with(done, "native_prefill_hc_down_f32_request_encoded_calls", Json::integer(6));
#endif
        emit(done);
    }
    return 0;
}
void workers(const std::filesystem::path& executable) {
    for (const auto mode : {"normal", "bad-ready", "bad-resource", "slow-done", "normal-tiled128", "normal-tiled256", "bad-draft-mode"}) {
        scalar_fixture::Files files(executable, mode);
        files.options.mtp_sidecar = files.directory / "mtp-sidecar";
        { std::ofstream sidecar(files.options.mtp_sidecar); sidecar << "generated CPU-only sidecar\n"; }
        if (std::string_view(mode) == "normal-tiled128" || std::string_view(mode) == "bad-draft-mode") files.options.mtp_draft_attention = "tiled128";
        if (std::string_view(mode) == "normal-tiled256") files.options.mtp_draft_attention = "tiled256";
        if (std::string_view(mode) == "bad-ready" || std::string_view(mode) == "bad-draft-mode") { refuses([&] { OwnedWorker worker(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t tokens = 0;
        worker.request(input(), Clock::now() + 3s, [] { return false; });
        const auto started = Clock::now();
        if (std::string_view(mode) == "slow-done") {
            bool interrupted = false;
            try { worker.request(generation_job(8), Clock::now() + 3s, [&] { return Clock::now() - started > 10ms; }, [&](const Json&) { ++tokens; }); }
            catch (const WorkerCancelled&) { interrupted = true; }
            check(interrupted && !tokens && worker.alive(), "MTP cancel did not drain without publication");
            refuses([&] { worker.request(generation_job(1), Clock::now() + 3s, [] { return false; }); });
            worker.request(input(), Clock::now() + 3s, [] { return false; }); worker.close(); continue;
        }
        const auto generate = [&] { worker.request(generation_job(8), Clock::now() + 3s, [] { return false; },
            [&](const Json& value) { tokens += value.find("event")->string() == "token"; }); };
        if (std::string_view(mode) == "bad-resource") {
            refuses(generate); check(!tokens && !worker.alive(), "MTP invalid DONE released buffered tokens"); continue;
        }
        generate(); check(tokens == 8 && worker.alive(), "MTP worker lost a validated token batch");
        bool interrupted = false;
        try { worker.request(generation_job(8), Clock::now() + 3s, [] { return false; }, [](const Json&) { throw WorkerCancelled("fixture callback cancelled"); }); }
        catch (const WorkerCancelled&) { interrupted = true; }
        check(interrupted && worker.alive(), "MTP post-admission callback cancellation lost drain state");
        refuses([&] { worker.request(generation_job(1), Clock::now() + 3s, [] { return false; }); });
        worker.request(input(), Clock::now() + 3s, [] { return false; }); worker.close();
    }
}
void generation_workers(const std::filesystem::path& executable, uint32_t depth = 1) {
    for (const bool slow : {false, true}) {
        scalar_fixture::Files files(executable, slow ? "slow-done" : "generation-carry");
        files.options.mtp_sidecar = files.directory / "mtp-sidecar"; files.options.mtp_request_carry = true;
        { std::ofstream sidecar(files.options.mtp_sidecar); sidecar << "generated CPU-only sidecar\n"; }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        if (depth == 2) { files.options.mtp_depth = 2; files.options.mtp_draft_attention = "tiled128"; }
#endif
        OwnedWorker worker(files.options);
        worker.request(input(), Clock::now() + 3s, [] { return false; });
        const auto width = depth + 1;
        const auto started = Clock::now(); size_t tokens = 0; bool cancelled = false, reentrant_refused = false;
        try { worker.request(scope_job(generation_job(width), 1, true, 2 * width), Clock::now() + 3s,
            [&] { return slow && Clock::now() - started > 10ms; }, [&](const Json& event) {
                if (!reentrant_refused) {
                    try { worker.abort_mtp_generation(Clock::now() + 3s); }
                    catch (const std::exception&) { reentrant_refused = true; }
                    check(reentrant_refused, "generation abort bypassed the in-flight request lock");
                }
                tokens += event.find("event")->string() == "token";
            }); }
        catch (const WorkerCancelled&) { cancelled = true; }
        check(cancelled == slow && tokens == (slow ? 0 : width) && reentrant_refused == !slow,
            "generation worker changed publication/cancellation/lock ownership");
        worker.abort_mtp_generation(Clock::now() + 3s);
        check(worker.alive(), "zero-step generation abort killed a healthy drained worker");
        refuses([&] { worker.request(scope_job(generation_job(width), 1, false, width), Clock::now() + 3s, [] { return false; }); });
        worker.request(input(), Clock::now() + 3s, [] { return false; });
        const auto done = worker.request(scope_job(generation_job(width), 2, true, width), Clock::now() + 3s, [] { return false; });
        check(!done.find("native_mtp_generation")->find("active")->boolean(), "worker reset/new generation retained stale carry");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        if(depth==2) {
            refuses([&]{worker.request(scope_job(generation_job(9),3,true,9),Clock::now()+3s,[]{return false;});});
            for(uint32_t tail : {1u,2u}) {
                worker.request(input(),Clock::now()+3s,[]{return false;});
                const auto end=worker.request(scope_job(generation_job(tail),2+tail,true,tail),Clock::now()+3s,[]{return false;});
                check(end.find("generated_ids")->array().size()==tail &&
                    !end.find("native_mtp_generation")->find("active")->boolean(),"D2 remaining1/2 tail failed through owned worker");
            }
        }
#endif
        worker.close();
    }
}

#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
Json depth2_ready() {
    auto value = scoped_ready();
    for (const auto& [name, field] : Json::Object{{"native_mtp_depth", Json::integer(2)}, {"verification_max_tokens", Json::integer(3)},
        {"native_mtp_protocol", Json("native-mtp-depth2-lookahead-v1")}, {"native_mtp_draft_attention", Json("tiled128")},
        {"native_mtp_tail_policy", Json("remaining1-or2 ordinary; no depth-one rounds")}}) value = with(std::move(value), name, field);
    return value;
}
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && defined(TOKSPUR_PRIVATE_DENSE_Q8_STAGED) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
void expanded_prefill_packet_contracts() {
    auto tiled = prefill512_ready(depth2_ready(), "f32-staged", "f32");
    tiled = with(std::move(tiled), "native_prefill_dense", Json("nax"));
    tiled = with(std::move(tiled), "native_prefill_dense_kernel", Json("nax_prefill_dense_q8_m32"));
    tiled = with(std::move(tiled), "native_prefill_grouped", Json("nax"));
    tiled = with(std::move(tiled), "native_prefill_grouped_kernel", Json("nax_grouped_moe"));
    tiled = with(std::move(tiled), "exact_candidate_ple_overlap", Json(true));
    for (const uint32_t packet : {512u, 1024u, 2048u, 8192u}) {
        const uint32_t capacity = std::min(packet, 2048u);
        auto prepared = with(tiled, "native_mtp_input_packet_max", Json::integer(packet));
        prepared = with(std::move(prepared), "known_token_capacity", Json::integer(capacity));
        prepared = with(std::move(prepared), "native_mtp_priming_batch_capacity", Json::integer(capacity));
        validate_mtp_ready(prepared, "tiled128", true, 2);
        check(mtp_input_packet(prepared) == packet, "negotiated packet must retain its external bound");
        ServeSession admitted(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, prepared);
        ++checks;
        const auto job = input(packet);
        Fake fake; fake.prepared = prepared;
        const auto done = fake(job, [](const Json&) {});
        MtpWireReply reply(prepared, prepared, job);
        reply.finish(done);
        ++checks;
        refuses([&] { MtpWireReply oversized(prepared, prepared, input(packet + 1)); });
        for (const uint32_t wrong : {8u, 512u, 1024u, 2048u, 4096u, 8192u}) {
            if (wrong == capacity) continue;
            refuses([&] { validate_mtp_ready(with(prepared, "known_token_capacity", Json::integer(wrong)), "tiled128", true, 2); });
            refuses([&] { validate_mtp_ready(with(prepared, "native_mtp_priming_batch_capacity", Json::integer(wrong)), "tiled128", true, 2); });
        }
        if (packet <= 2048) {
            validate_mtp_ready(without(prepared, "exact_candidate_ple_overlap"), "tiled128", true, 2);
            ++checks;
            continue;
        }
        refuses([&] { validate_mtp_ready(with(prepared, "native_mtp_input_packet_max", Json::integer(4096)), "tiled128", true, 2); });
        for (const auto& drifted : {with(done, "known_token_capacity", Json::integer(8192)),
            with(done, "native_mtp_input_packet_max", Json::integer(2048)),
            with(done, "exact_candidate_ple_overlap", Json(false)), without(done, "exact_candidate_ple_overlap")}) {
            refuses([&] { MtpWireReply changed(prepared, prepared, job); changed.finish(drifted); });
        }
        refuses([&] { validate_mtp_ready(without(prepared, "exact_candidate_ple_overlap"), "tiled128", true, 2); });
        for (const auto& overlap : {Json(false), Json::integer(1), Json()}) {
            refuses([&] { validate_mtp_ready(with(prepared, "exact_candidate_ple_overlap", overlap), "tiled128", true, 2); });
        }
        auto dense_f32 = with(prepared, "native_prefill_dense", Json("f32"));
        dense_f32 = with(std::move(dense_f32), "native_prefill_dense_kernel", Json("graph_dense_q8_staged_f32"));
        refuses([&] { validate_mtp_ready(dense_f32, "tiled128", true, 2); });
        auto grouped_f32 = with(prepared, "native_prefill_grouped", Json("f32"));
        grouped_f32 = with(std::move(grouped_f32), "native_prefill_grouped_kernel", Json("graph_grouped_mm_f32"));
        refuses([&] { validate_mtp_ready(grouped_f32, "tiled128", true, 2); });
    }
}
#endif
void depth2_worker_refusals(const std::filesystem::path& executable) {
    for(const auto mode : {"bad-depth2-ready","bad-depth2-width","depth2-without-carry","depth2-without-tile"}) {
        scalar_fixture::Files files(executable,mode);files.options.mtp_sidecar=files.directory/"mtp-sidecar";
        {std::ofstream sidecar(files.options.mtp_sidecar);sidecar<<"generated CPU-only sidecar\n";}
        files.options.mtp_depth=2;files.options.mtp_request_carry=std::string_view(mode)!="depth2-without-carry";
        files.options.mtp_draft_attention=std::string_view(mode)=="depth2-without-tile"?"serial":"tiled128";
        refuses([&]{OwnedWorker rejected(files.options);});
        if(std::string_view(mode).starts_with("depth2-without"))
            check(!std::filesystem::exists(files.directory/"started"),"invalid D2 consent executed a child before refusal");
    }
}
void depth2_sessions() {
    for(uint32_t maximum : {1u,2u,17u}) {
        Fake f;f.prepared=depth2_ready();
        ServeSession session(session_codec(),"tokspur","direct",Json(),false,Json(),false,ToolEnumValidation::strict,false,f.prepared);
        const auto request=with(chat(std::string(17,'A'),false,maximum),"temperature",number(1));
        const ServeSink sink{[](const Json&){},[]{return false;},[]{return uint64_t(44);}};
        const auto result=session.complete(request,"depth2-session",[&](const Json& job,const auto& cb){return f(job,cb);},sink);
        const auto& details=*result.find("tokspur_private");
        check(integer(details,"native_mtp_depth")==2 && details.find("native_mtp_protocol")->string()=="native-mtp-depth2-lookahead-v1",
            "ServeSession lost explicit D2 carrier identity");
        check(integer(details,"generation_batches")==((maximum+7)/8) && !details.find("native_mtp_generation")->find("active")->boolean(),
            "ServeSession D2 multi-packet/tail did not close its generation");
        size_t generated=0;for(const auto& job:f.packets)generated+=integer(job,"generate");
        check(generated==maximum,"ServeSession D2 changed registered output budget");
    }
    const auto r=depth2_ready();
    for(const auto& invalid : {with(r,"native_mtp_depth",Json::integer(1)),with(r,"verification_max_tokens",Json::integer(2)),
         with(r,"native_mtp_protocol",Json("native-mtp-depth1-lookahead-v2"))})
        refuses([&]{ServeSession bad(session_codec(),"tokspur","direct",Json(),false,Json(),false,ToolEnumValidation::strict,false,invalid);});
}
void depth2_contracts() {
    const auto r = depth2_ready(); validate_mtp_ready(r, "tiled128", true, 2);
    refuses([&] { validate_mtp_ready(r, "tiled128", true); });
    refuses([&] { validate_mtp_ready(without(r,"native_mtp_tail_policy"), "tiled128", true, 2); });
    for (uint32_t accepted : {0u,1u,2u}) for (uint32_t width=1;width<=8;++width) for (uint32_t eos=0;eos<=width;++eos) {
        Fake f; f.prepared = r; f.force_accepted = accepted;
        const auto pre = input(); MtpWireReply start(r,r,pre); auto prior=f(pre,[](const Json&){});start.finish(prior);
        f.stop_at=eos; const auto job=scope_job(generation_job(width),1,true,width); MtpWireReply reply(r,prior,job);
        const auto done=f(job,[&](const Json& event){reply.event(event);});reply.finish(done);
        check(!done.find("native_mtp_generation")->find("active")->boolean(),"depth-two stop/budget did not close generation");
        check(reply.events().size()<=16,"depth-two event buffering exceeded its packet bound");
    }
    Fake f;f.prepared=r;f.previous=r;
    const auto pre=input();MtpWireReply initial(r,r,pre);auto prior=f(pre,[](const Json&){});initial.finish(prior);
    const auto one=scope_job(generation_job(3),1,true,6);MtpWireReply first(r,prior,one);
    auto done=f(one,[&](const Json& e){first.event(e);});first.finish(done);
    check(done.find("native_mtp_generation")->find("pending")->boolean(),"depth-two packet lost carried bonus");
    const auto next=scope_job(generation_job(3),1,false,3);MtpWireReply second(r,done,next);
    auto end=f(next,[&](const Json& e){second.event(e);});second.finish(end);
    check(integer(end,"native_mtp_carried_anchors")==1,"depth-two next packet redrew/lost its anchor");
    const auto bad=with(end,"native_mtp_depth",Json::integer(1));
    MtpWireReply drift(r,done,next);for(const auto& e:f.events)drift.event(e);refuses([&]{drift.finish(bad);});
    // Lookup-bypass rounds (head_calls=0, prime_extra=1) with 1 or 2 proposed tokens and malformed pair refusals.
    for (uint32_t prop : {1u, 2u}) for (uint32_t acc = 0; acc <= prop; ++acc) {
        Fake lf; lf.prepared = r; lf.lookup_bypass = true; lf.force_proposed = prop; lf.force_accepted = acc;
        const auto lpre = input(); MtpWireReply lstart(r, r, lpre); auto lprior = lf(lpre, [](const Json&) {}); lstart.finish(lprior);
        const auto ljob = scope_job(generation_job(3), 1, true, 6); MtpWireReply lreply(r, lprior, ljob);
        const auto ldone = lf(ljob, [&](const Json& event) { lreply.event(event); }); lreply.finish(ldone);
        check(integer(ldone, "native_mtp_head_calls") == 0, "lookup-bypass round incremented head_calls");
        check(integer(ldone, "native_mtp_proposed") >= prop, "lookup-bypass round omitted proposed count");
    }
    {
        Fake lf; lf.prepared = r; lf.lookup_bypass = true; lf.force_proposed = 2; lf.force_accepted = 2;
        const auto lpre = input(); MtpWireReply lstart(r, r, lpre); auto lprior = lf(lpre, [](const Json&) {}); lstart.finish(lprior);
        const auto ljob = scope_job(generation_job(3), 1, true, 6);
        auto ignored = lf(ljob, [](const Json&) {}); static_cast<void>(ignored);
        for (const auto& [hc, pe] : std::array<std::pair<uint64_t, uint64_t>, 3>{{{0, 0}, {1, 1}, {2, 1}}}) {
            MtpWireReply bad_reply(r, lprior, ljob);
            auto bad_round = with(with(lf.events.front(), "head_calls", Json::integer(hc)), "prime_extra", Json::integer(pe));
            refuses([&] { bad_reply.event(bad_round); });
        }
    }
}
Json depth3_ready() {
    auto value = scoped_ready();
    for (const auto& [name, field] : Json::Object{{"native_mtp_depth", Json::integer(3)}, {"verification_max_tokens", Json::integer(4)},
        {"native_mtp_protocol", Json("native-mtp-depth3-lookahead-v2")}, {"native_mtp_draft_attention", Json("tiled128")},
        {"native_mtp_tail_policy", Json("commit-through; round overflow drained across packets")}}) value = with(std::move(value), name, field);
    return value;
}
void depth3_contracts() {
    const auto r = depth3_ready();
    validate_mtp_ready(r, "tiled128", true, 3);
    refuses([&] { validate_mtp_ready(r, "tiled128", true); });
    refuses([&] { validate_mtp_ready(without(r, "native_mtp_tail_policy"), "tiled128", true, 3); });
    refuses([&] { validate_mtp_ready(with(r, "verification_max_tokens", Json::integer(3)), "tiled128", true, 3); });
    for (const bool shared : {false, true}) {
        const auto prepared = shared ? with(r, "native_pack_shared_mapping", Json(true)) : r;
        Fake fake; fake.prepared = prepared;
        const auto job = input(); const auto done = fake(job, [](const Json&) {});
        MtpWireReply unchanged(prepared, prepared, job); unchanged.finish(done);
        for (const auto& changed : {with(done, "native_pack_shared_mapping", Json(!shared)),
                                   shared ? without(done, "native_pack_shared_mapping") :
                                            with(done, "native_pack_shared_mapping", Json(false))}) {
            MtpWireReply reply(prepared, prepared, job);
            refuses([&] { reply.finish(changed); });
        }
    }
    for (uint32_t accepted : {0u, 1u, 2u, 3u}) for (uint32_t width : {1u, 4u, 8u}) {
        Fake f; f.prepared = r; f.force_accepted = accepted;
        const auto pre = input(); MtpWireReply start(r, r, pre); auto prior = f(pre, [](const Json&) {}); start.finish(prior);
        const auto job = scope_job(generation_job(width), 1, true, width); MtpWireReply reply(r, prior, job);
        const auto done = f(job, [&](const Json& event) { reply.event(event); }); reply.finish(done);
        check(!done.find("native_mtp_generation")->find("active")->boolean(), "depth-three budget did not close generation");
    }
}
#endif

#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
void lane_sessions() {
    const auto prepared = scalar_fixture::lane_fields(ready());
    const auto make = [&](const Json& mtp, const Json& lanes) {
        return ServeSession(session_codec(), "tokspur", "direct", Json(), false, Json(), false,
            ToolEnumValidation::strict, false, mtp, lanes);
    };
    // Every advertised field is mandatory, typed and bound to the admitted MTP READY.
    refuses([&] { static_cast<void>(make(prepared, Json())); });
    refuses([&] { static_cast<void>(make(ready(), prepared)); });
    for (const auto name : owned_weight_lane_fields) {
        refuses([&] { static_cast<void>(make(prepared, without(prepared, name))); });
        for (const auto& invalid : {Json(), Json(true), Json(Json::Array{})})
            refuses([&] { static_cast<void>(make(prepared, with(prepared, name, invalid))); });
        const auto changed = name == "weight_lanes_manifest_sha256" ? Json(std::string(64, 'b')) :
            Json::integer(integer(prepared, name) + 1);
        refuses([&] { static_cast<void>(make(prepared, with(prepared, name, changed))); });
    }
    auto request = with(chat(std::string(17, 'A'), false, 17), "temperature", number(1));
    for (const bool stream : {false, true}) {
        Fake fake; fake.prepared = prepared;
        // Only the declared fields may reach HTTP; unrelated READY details stay private.
        const auto private_ready = with(prepared, "fixture_private_path", Json("/private/generated/fixture"));
        auto session = make(prepared, private_ready); Json::Array packets;
        ServeSink sink{[&](const Json& value) { packets.push_back(value); }, [] { return false; }, [] { return uint64_t(44); }};
        const auto result = session.complete(with(request, "stream", Json(stream)), "lane-projection",
            [&](const Json& job, const std::function<void(const Json&)>& cb) { return fake(job, cb); }, sink);
        const auto* details = result.find("tokspur_private"); check(details && details->find("native_mtp"), "lane projection lost MTP completion diagnostics");
        for (const auto name : owned_weight_lane_fields)
            check(details->find(name) && details->find(name)->dump() == prepared.find(name)->dump(), "HTTP completion lost lane identity");
        check(!details->find("fixture_private_path"), "HTTP completion leaked unrelated READY data");
        if (stream) {
            check(!packets.empty(), "selected layout did not produce SSE chunks");
            for (size_t i = 0; i < packets.size(); ++i) {
                const auto* projected = packets[i].find("tokspur_private");
                if (i + 1 < packets.size()) { check(!projected, "layout metadata leaked before final SSE chunk"); continue; }
                check(projected && projected->object().size() == owned_weight_lane_fields.size(), "final SSE chunk lacks exact lane projection");
                for (const auto name : owned_weight_lane_fields)
                    check(projected->find(name) && projected->find(name)->dump() == prepared.find(name)->dump(), "final SSE identity differs from READY");
                check(!packets[i].find("choices")->array()[0].find("finish_reason")->is_null(), "lane metadata attached to a nonfinal chunk");
            }
        } else check(packets.empty(), "nonstream layout emitted SSE chunks");
    }
    // An omitted selector and the explicit absent value produce the same JSON,
    // apart from the ordinary wall-clock created field, for both transports.
    for (const bool stream : {false, true}) {
        Json::Array outputs;
        for (const bool explicit_absent : {false, true}) {
            Fake fake;
            auto session = explicit_absent ? make(fake.prepared, Json()) : ServeSession(session_codec(), "tokspur", "direct", Json(), false, Json(), false,
                ToolEnumValidation::strict, false, fake.prepared);
            Json::Array packets;
            ServeSink sink{[&](const Json& value) { packets.push_back(without(value, "created")); }, [] { return false; }, [] { return uint64_t(44); }};
            const auto result = session.complete(with(request, "stream", Json(stream)), "lane-absent",
                [&](const Json& job, const std::function<void(const Json&)>& cb) { return fake(job, cb); }, sink);
            for (const auto name : owned_weight_lane_fields)
                check(!result.find("tokspur_private")->find(name), "unselected layout changed completion metadata");
            for (const auto name : owned_lane_auth_fields)
                check(!result.find("tokspur_private")->find(name), "omitted authentication changed completion metadata");
            outputs.emplace_back(Json::Object{{"result", without(result, "created")}, {"packets", Json(std::move(packets))}});
        }
        check(outputs[0].dump() == outputs[1].dump(), "unselected constructor changed HTTP response bytes");
    }
    // Authentication concurrency is immutable startup consent, independent of
    // the five unchanged layout identity fields, and appears only when selected.
    for (const uint64_t workers : {1, 4, 16}) {
        const auto authentication = scalar_fixture::lane_auth_fields(prepared, workers);
        refuses([&] { static_cast<void>(make(authentication, prepared)); });
        refuses([&] { static_cast<void>(make(prepared, authentication)); });
        for (const bool stream : {false, true}) {
            Fake fake; fake.prepared = authentication;
            auto session = make(authentication, authentication); Json::Array packets;
            ServeSink sink{[&](const Json& value) { packets.push_back(value); }, [] { return false; }, [] { return uint64_t(44); }};
            const auto result = session.complete(with(request, "stream", Json(stream)), "lane-auth-projection",
                [&](const Json& job, const std::function<void(const Json&)>& cb) { return fake(job, cb); }, sink);
            const auto* details = result.find("tokspur_private"); check(details, "authentication lost private completion details");
            for (const auto name : owned_lane_auth_fields)
                check(details->find(name) && details->find(name)->dump() == authentication.find(name)->dump(), "HTTP authentication differs from READY");
            for (const auto name : owned_weight_lane_fields)
                check(details->find(name) && details->find(name)->dump() == prepared.find(name)->dump(), "authentication changed lane identity");
            if (stream) {
                check(!packets.empty(), "authentication did not produce final SSE chunk");
                for (size_t i = 0; i + 1 < packets.size(); ++i)
                    check(!packets[i].find("tokspur_private"), "authentication leaked before final SSE chunk");
                const auto* projected = packets.back().find("tokspur_private");
                check(projected && projected->object().size() == owned_weight_lane_fields.size() + owned_lane_auth_fields.size(), "final SSE authentication projection differs");
                for (const auto name : owned_lane_auth_fields)
                    check(projected->find(name) && projected->find(name)->dump() == authentication.find(name)->dump(), "SSE authentication differs from READY");
            } else check(packets.empty(), "nonstream authentication emitted SSE chunks");
        }
    }
}
void lane_workers(const std::filesystem::path& executable) {
    for (const auto mode : {"lane-normal", "lane-drift", "lane-auth-normal", "lane-auth-drift", "lane-auth-missing-done"}) {
        scalar_fixture::Files files(executable, mode);
        files.options.mtp_sidecar=files.directory/"mtp-sidecar";
        files.options.weight_lanes=files.directory/"lanes";
        for (const auto& path : {files.options.mtp_sidecar, files.options.weight_lanes}) {
            std::ofstream output(path); output<<"generated CPU-only fixture\n";
        }
        if (std::string_view(mode).starts_with("lane-auth-")) files.options.weight_lanes_auth_workers = 4;
        OwnedWorker worker(files.options);size_t callbacks=0;
        worker.request(input(),Clock::now()+3s,[]{return false;});
        const auto generate=[&]{worker.request(generation_job(8),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;});};
        if (std::string_view(mode)!="lane-normal" && std::string_view(mode)!="lane-auth-normal") {
            refuses(generate);check(!callbacks&&!worker.alive(),"unadmitted lane identity released MTP events");continue;
        }
        generate();check(callbacks>=8&&worker.alive());worker.close();
    }
}
#endif
void actual_ready(const char* path, std::string_view expected_mode, bool expected_scope = false) {
    constexpr size_t limit = 8'000'000;
    std::ifstream input(path, std::ios::binary); check(bool(input), "cannot open private MTP ready capture");
    std::string raw(limit + 1, '\0'); input.read(raw.data(), std::streamsize(raw.size()));
    const auto count = input.gcount(); check(count > 0 && uint64_t(count) <= limit && !input.bad(), "MTP capture read bound");
    raw.resize(size_t(count)); const auto value = parse_json(raw);
    const auto* ready_value = value.find("ready"); validate_mtp_ready(ready_value ? *ready_value : value, expected_mode, expected_scope); ++checks;
    std::cout << "Actual MTP READY admitted (CPU protocol scope only)\n";
}
} // namespace mtp_fixture

#ifndef TOKSPUR_MTP_WIRE_TEST_ENTRY
#define TOKSPUR_MTP_WIRE_TEST_ENTRY main
#endif
int TOKSPUR_MTP_WIRE_TEST_ENTRY(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
        if (argc > 1 && std::string_view(argv[1]) == "--check-native-ready") {
            check(argc >= 3 && argc <= 5 && (argc != 5 || std::string_view(argv[4]) == "http-generation-v1"),
                "expected private actual READY capture, optional registered draft mode and explicit carry scope");
            mtp_fixture::actual_ready(argv[2], argc >= 4 ? std::string_view(argv[3]) : std::string_view("serial"), argc == 5); return 0;
        }
        check(argc == 1, "unexpected MTP wire test arguments");
        mtp_fixture::contracts(); mtp_fixture::generation_contracts(); mtp_fixture::generation_sessions(); mtp_fixture::cold_generation_resync(mtp_fixture::scoped_ready()); mtp_fixture::sessions(); mtp_fixture::context_budget_sessions(); mtp_fixture::workers(std::filesystem::canonical(argv[0])); mtp_fixture::generation_workers(std::filesystem::canonical(argv[0]));
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && defined(TOKSPUR_PRIVATE_DENSE_Q8_STAGED) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
        mtp_fixture::expanded_prefill_packet_contracts();
#endif
        mtp_fixture::depth2_contracts(); mtp_fixture::depth2_sessions(); mtp_fixture::cold_generation_resync(mtp_fixture::depth2_ready()); mtp_fixture::depth2_worker_refusals(std::filesystem::canonical(argv[0])); mtp_fixture::generation_workers(std::filesystem::canonical(argv[0]),2);
        mtp_fixture::depth3_contracts(); mtp_fixture::cold_generation_resync(mtp_fixture::depth3_ready());
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        scalar_fixture::lane_contracts();mtp_fixture::lane_sessions();mtp_fixture::lane_workers(std::filesystem::canonical(argv[0]));
#endif
        std::cout << "MTP HTTP CPU wire controls passed: " << checks << '\n'; return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
