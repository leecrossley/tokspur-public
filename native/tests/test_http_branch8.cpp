// Original deterministic fixtures. This translation unit also runs every old
// private-V8 refusal/worker/session check; that protocol remains a distinct arm.
#define main previous_http_verify8_main
#include "test_http_verify8.cpp"
#undef main

#include <deque>
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
#include "tokspur/graph.hpp"
#endif

#ifndef TOKSPUR_PRIVATE_HTTP_BRANCH8
#error This fixture requires the explicit branch8 HTTP build.
#endif

namespace branch_fixture {
constexpr std::array<const char*, 10> costs{"lookup_appended_tokens", "lookup_indexed_grams", "lookup_truncated_grams", "lookup_cleared_grams",
    "lookup_resets", "lookup_truncations", "lookup_queries", "lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries"};
constexpr std::array<const char*, 5> graph{"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes", "gpu_allocation_count", "gpu_buffer_limit_bytes", "imported_weight_backing_bytes"};
constexpr std::array<const char*, 5> storage{"checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"};
Json sampled(const Json& policy_value, uint64_t seed, uint64_t prefix) {
    auto value = with(policy_value, "seed", Json::integer(seed)); value = with(std::move(value), "valid", Json(true));
    value = with(std::move(value), "entropy_words", Json::integer(0)); return with(std::move(value), "entropy_prefix", Json::integer(prefix));
}
Json snapshot(const Json& envelope, const Json& previous = Json()) {
    Json::Object out;
    for (const auto name : {"committed_prefix", "sampling", "logits_available", "continuation_allowed", "stop_id", "final_logit_top8",
        "hc_exact_final_logit_sha256", "hc_exact_routes", "lookup_history_epoch"}) {
        const auto* value = envelope.find(name); if (!value && !previous.is_null()) value = previous.find(name);
        if (!value) throw std::runtime_error("fixture snapshot source incomplete"); out.emplace_back(name, *value);
    }
    for (const auto name : costs) {
        const auto* value = envelope.find(name); if (!value && !previous.is_null()) value = previous.find(name);
        out.emplace_back(name, value ? *value : Json::integer(0));
    }
    for (const auto name : graph) {
        const auto* value = envelope.find(name); if (!value && !previous.is_null()) value = previous.find(name);
        if (!value) throw std::runtime_error("fixture graph snapshot source incomplete"); out.emplace_back(name, *value);
    }
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    const auto* staged = envelope.find("qsa_stage_v8_prepared");
    if (!staged && !previous.is_null()) staged = previous.find("qsa_stage_v8_prepared");
    if (staged) for (const auto name : {"qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes", "qsa_stage_v8_scratch_charged_bytes",
        "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches"}) {
        const auto* value = envelope.find(name); if (!value && !previous.is_null()) value = previous.find(name);
        if (!value) throw std::runtime_error("fixture staged-QSA snapshot incomplete"); out.emplace_back(name, *value);
    }
#endif
    return Json(std::move(out));
}
Json stamp(Json value, uint64_t command, uint64_t lease, const Json& state = Json()) {
    value = with(std::move(value), "broker_protocol", Json("branch8-v1")); value = with(std::move(value), "broker_batch_width", Json::integer(8));
    value = with(std::move(value), "broker_epoch", Json::integer(4242)); value = with(std::move(value), "command_id", Json::integer(command));
    value = with(std::move(value), "lease_id", Json::integer(lease));
    if (!state.is_null()) value = with(std::move(value), "branch8_session", state); return value;
}
Json prepared() {
    auto ready = ready_record();
    for (const auto& [name, amount] : {std::pair{"checkpoint_slots", 4ull}, {"checkpoint_combined_budget_bytes", 512ull << 20},
        {"checkpoint_host_bytes", 5229136ull}, {"checkpoint_store_host_bytes", 368ull}, {"checkpoint_gpu_bytes", 472186880ull},
        {"tracked_metal_buffer_bytes", 84277411004ull}, {"peak_metal_buffer_bytes", 84277411004ull}, {"gpu_allocation_count", 616ull}})
        ready = with(std::move(ready), name, Json::integer(amount));
    ready = with(std::move(ready), "verification8_branch_reuse", Json(true)); ready = with(std::move(ready), "checkpoint_broker", Json(true));
    Json state_source = ready;
    state_source = with(std::move(state_source), "committed_prefix", Json::integer(0)); state_source = with(std::move(state_source), "sampling", sampled(policy(), 0, 0));
    state_source = with(std::move(state_source), "logits_available", Json(false)); state_source = with(std::move(state_source), "continuation_allowed", Json(false));
    state_source = with(std::move(state_source), "stop_id", Json()); state_source = with(std::move(state_source), "final_logit_top8", Json(Json::Array{}));
    state_source = with(std::move(state_source), "hc_exact_final_logit_sha256", Json()); state_source = with(std::move(state_source), "lookup_history_epoch", Json::integer(1));
    return stamp(std::move(ready), 0, 0, snapshot(state_source));
}
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
Json staged_prepared(bool enabled) {
    auto ready = prepared();
    ready = with(std::move(ready), "allow_prompt_lookup_experiments", Json(false));
    ready = with(std::move(ready), "qsa_stage_v8_prepared", Json(true));
    ready = with(std::move(ready), "qsa_stage_v8_enabled", Json(enabled));
    for (const auto& [name, value] : {std::pair{"qsa_stage_v8_scratch_logical_bytes", 196896ull}, {"qsa_stage_v8_scratch_charged_bytes", 212992ull},
        {"qsa_stage_v8_calls", 0ull}, {"qsa_stage_v8_physical_dispatches", 0ull}, {"qsa_stage_v8_extra_dispatches", 0ull},
        {"tracked_metal_buffer_bytes", 84277623996ull}, {"peak_metal_buffer_bytes", 84277623996ull}, {"gpu_allocation_count", 617ull}})
        ready = with(std::move(ready), name, Json::integer(value));
    auto state = *ready.find("branch8_session");
    for (const auto name : graph) state = with(std::move(state), name, *ready.find(name));
    for (const auto name : {"qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes", "qsa_stage_v8_scratch_charged_bytes",
        "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches"}) state = with(std::move(state), name, *ready.find(name));
    return with(std::move(ready), "branch8_session", std::move(state));
}
#endif
Json begin_job(std::span<const uint32_t> prompt, bool reuse) {
    Json::Array ids; for (const auto id : prompt) ids.push_back(Json::integer(id));
    return Json(Json::Object{{"op", Json("branch_begin")}, {"prompt_ids", Json(std::move(ids))}, {"generation_requested", Json(true)},
        {"allow_restore", Json(reuse)}, {"sampling", policy()}, {"seed", Json::integer(0)}});
}
struct Fake {
    struct Saved { uint32_t prefix; Json state; };
    Json ready, previous, state, sampling = policy();
    explicit Fake(Json prepared_ready = prepared()) : ready(std::move(prepared_ready)), previous(ready), state(*ready.find("branch8_session")) {}
    uint64_t command = 0, lease = 0, seed = 0;
    std::vector<uint32_t> history, output = encode_text("ok<|im_end|>"); size_t next = 0;
    std::deque<Saved> saved;
    std::vector<uint32_t> input_widths, captured_prefixes, restored_prefixes;
    bool corrupt_done = false;
    void plan(std::string_view text = "ok<|im_end|>") { output = encode_text(text); next = 0; }
    Json control(const Json& job) {
        const auto op = job.find("op")->string(); const auto before = uint32_t(history.size()); uint32_t common = before;
        std::string outcome;
        if (op == "branch_begin") {
            ++lease; sampling = *job.find("sampling"); seed = integer(job, "seed"); common = 0;
            const bool reuse = job.find("allow_restore")->boolean(); const auto& prompt = job.find("prompt_ids")->array();
            if (reuse) while (common < history.size() && common < prompt.size() && history[common] == prompt[common].unsigned_integer()) ++common;
            std::optional<Saved> selected;
            if (reuse) for (const auto& record : saved) if (record.prefix <= common && (!selected || record.prefix >= selected->prefix)) selected = record;
            const auto prefix = selected ? selected->prefix : 0; auto replacement = state;
            if (selected) {
                for (const auto name : {"committed_prefix", "logits_available", "continuation_allowed", "stop_id", "final_logit_top8", "hc_exact_final_logit_sha256"})
                    replacement = with(std::move(replacement), name, *selected->state.find(name));
                replacement = with(std::move(replacement), "lookup_truncations", Json::integer(integer(state, "lookup_truncations") + 1));
                replacement = with(std::move(replacement), "lookup_truncated_grams", Json::integer(integer(state, "lookup_truncated_grams") + grams(before) - grams(prefix)));
                outcome = "restored"; restored_prefixes.push_back(prefix);
                for (auto at = saved.begin(); at != saved.end();) { if (at->prefix > prefix) at = saved.erase(at); else ++at; }
            } else {
                saved.clear(); outcome = "cold_miss";
                replacement = with(std::move(replacement), "committed_prefix", Json::integer(0)); replacement = with(std::move(replacement), "logits_available", Json(false));
                replacement = with(std::move(replacement), "continuation_allowed", Json(false)); replacement = with(std::move(replacement), "stop_id", Json());
                replacement = with(std::move(replacement), "final_logit_top8", Json(Json::Array{})); replacement = with(std::move(replacement), "hc_exact_final_logit_sha256", Json());
                replacement = with(std::move(replacement), "lookup_resets", Json::integer(integer(state, "lookup_resets") + 1));
                replacement = with(std::move(replacement), "lookup_cleared_grams", Json::integer(integer(state, "lookup_cleared_grams") + grams(before)));
            }
            replacement = with(std::move(replacement), "lookup_history_epoch", Json::integer(integer(state, "lookup_history_epoch") + 1));
            replacement = with(std::move(replacement), "sampling", sampled(sampling, seed, prefix)); state = std::move(replacement); history.resize(prefix); next = 0;
        } else {
            check(op == "branch_capture" && before > 0); if (saved.size() == 4) saved.pop_front();
            saved.push_back({before, state}); captured_prefixes.push_back(before); outcome = "captured";
        }
        Json::Object fields{{"event", Json("control_done")}, {"op", Json(std::string(op))}, {"outcome", Json(outcome)}, {"model_steps", Json::integer(0)},
            {"generated_ids", Json(Json::Array{})}, {"common_prefix", Json::integer(common)}, {"control_wall_ms", number(1)}, {"timings_qualified", Json(false)},
            {"checkpoint_retained", Json::integer(saved.size())}};
        for (const auto name : {"committed_prefix", "sampling", "stop_id", "logits_available", "continuation_allowed"}) fields.emplace_back(name, *state.find(name));
        for (const auto name : storage) fields.emplace_back(name, *ready.find(name));
        return stamp(Json(std::move(fields)), command, lease, state);
    }
    Json operator()(const Json& job, const std::function<void(const Json&)>& callback) {
        ++command;
        if (job.find("op")) { previous = control(job); return previous; }
        check(!job.find("reset")->boolean(), "new branch wire reset must be a distinct control");
        const auto n = uint32_t(integer(job, "generate")), input = uint32_t(job.find("tokens")->array().size());
        std::vector<Round> rounds; const size_t first = next;
        for (uint32_t t = 0; t < n && next < output.size(); ++t, ++next) {
            const bool terminal = output[next] == 248046; rounds.push_back({0, 0, 1, terminal}); if (terminal) { ++next; break; }
        }
        auto t = make_transcript(ready, state, job, sampling, seed, rounds);
        for (const auto name : {"lookup_truncated_grams", "lookup_cleared_grams", "lookup_truncations", "lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries"})
            t.done = with(std::move(t.done), name, *state.find(name));
        Json::Array ids; size_t at = first;
        for (auto& event : t.events) if (event.find("event")->string() == "token") {
            ids.push_back(Json::integer(output.at(at))); event = with(std::move(event), "id", Json::integer(output.at(at++)));
        }
        t.done = with(std::move(t.done), "generated_ids", Json(ids));
        auto top = t.done.find("final_logit_top8")->array(); top[0] = with(top[0], "id", Json::integer(next < output.size() ? output[next] : 200));
        t.done = with(std::move(t.done), "final_logit_top8", Json(std::move(top))); t.done = with(std::move(t.done), "logits_available", Json(true));
        t.done = with(std::move(t.done), "checkpoint_retained", Json::integer(saved.size()));
        t.done = with(std::move(t.done), "checkpoint_capture_wall_ms", number(0)); t.done = with(std::move(t.done), "checkpoint_restore_wall_ms", number(0));
        for (const auto& id : job.find("tokens")->array()) history.push_back(uint32_t(id.unsigned_integer()));
        for (const auto& id : ids) history.push_back(uint32_t(id.unsigned_integer()));
        if (input) input_widths.push_back(input);
        state = snapshot(t.done, state); previous = stamp(without(t.done, "branch8_session"), command, lease, state);
        for (auto& event : t.events) callback(stamp(std::move(event), command, lease));
        if (corrupt_done && n) return with(previous, "hc_exact_final_logit_sha256", Json("malformed"));
        return previous;
    }
};

void admission_checks(const Json& ready = prepared()) {
    const auto sampling_policy = policy(); validate_branch8_ready(ready); ++checks;
    refuses([&] { validate_verify8_ready(ready); });
    refuses([&] { ServeSession wrong(session_codec(), "tokspur", "direct", ready, true, ready); });
    refuses([&] { ServeSession wrong(session_codec(), "tokspur", "direct", ready, false, Json(), true); });
    for (const auto key : {"broker_protocol", "broker_batch_width", "broker_epoch", "command_id", "lease_id", "verification8_branch_reuse", "branch8_session"})
        refuses([&] { validate_branch8_ready(without(ready, key)); });
    for (const auto& [key, value] : Json::Object{{"broker_protocol", Json("ordinary")}, {"broker_batch_width", Json::integer(1)},
        {"checkpoint_slots", Json::integer(0)}, {"checkpoint_slots", Json::integer(5)}, {"checkpoint_combined_budget_bytes", Json::integer((512ull << 20) + 1)},
        {"checkpoint_gpu_bytes", Json::integer(512ull << 20)}, {"verification8_branch_reuse", Json(false)}, {"command_id", Json::integer(1)},
        {"lease_id", Json::integer(1)}, {"lookup_verification_budget_bytes", Json::integer(512ull << 20)}})
        refuses([&] { validate_branch8_ready(with(ready, key, value)); });
    for (const auto& [key, ignored] : ready.find("branch8_session")->object()) {
        static_cast<void>(ignored);
        refuses([&] { validate_branch8_ready(with(ready, "branch8_session", without(*ready.find("branch8_session"), key))); });
        refuses([&] { validate_branch8_ready(with(ready, "branch8_session", with(*ready.find("branch8_session"), key, Json("wrong")))); });
    }
    Fake fake(ready); const auto initial = fake.previous; const std::array<uint32_t, 8> ids{1,2,3,4,5,6,7,8};
    const auto job = begin_job(ids, false); const auto cold = fake(job, [](const Json&) { check(false); });
    validate_branch8_control(ready, initial, job, cold); ++checks;
    for (const auto key : {"model_steps", "command_id", "common_prefix", "committed_prefix", "checkpoint_retained"})
        refuses([&] { validate_branch8_control(ready, initial, job, with(cold, key, Json::integer(999))); });
    refuses([&] { validate_branch8_control(ready, initial, job, with(cold,"lease_id",Json::integer(0))); });
    // Fresh broker leases are strictly increasing, not necessarily contiguous.
    // Exact echo of the parent's issued lease is a separate worker boundary.
    validate_branch8_control(ready,initial,job,with(cold,"lease_id",Json::integer(999))); ++checks;
    for (const auto name : costs) {
        auto state = with(*cold.find("branch8_session"), name, Json::integer(integer(*cold.find("branch8_session"), name) + 1));
        refuses([&] { validate_branch8_control(ready, initial, job, with(cold, "branch8_session", state)); });
    }
    for (uint32_t width = 1; width <= 8; ++width) {
        const auto before = fake.previous; const auto input = input_job(width, false, policy(), 0);
        const auto done = fake(input, [](const Json&) { check(false); });
        Verify8Reply reply(ready, before, input, uint32_t(integer(*before.find("branch8_session"), "committed_prefix")), sampling_policy, 0, false, true);
        refuses([&] { static_cast<void>(reply.events()); }); reply.finish(done); check(reply.events().empty());
    }
    const auto capture_job = parse_json(R"({"op":"branch_capture"})"); const auto before = fake.previous;
    const auto captured = fake(capture_job, [](const Json&) { check(false); }); validate_branch8_control(ready, before, capture_job, captured); ++checks;
    const auto invalid_capture = with(captured, "branch8_session", with(*captured.find("branch8_session"), "lookup_queries", Json::integer(1)));
    refuses([&] { validate_branch8_control(ready, before, capture_job, invalid_capture); });
    fake.plan(); const auto previous = fake.previous; const auto generate = generation_job(8); std::vector<Json> events;
    const auto done = fake(generate, [&](const Json& event) { events.push_back(event); });
    Verify8Reply reply(ready, previous, generate, uint32_t(integer(*previous.find("branch8_session"), "committed_prefix")), sampling_policy, 0, false, true);
    for (const auto& event : events) { reply.event(event); refuses([&] { static_cast<void>(reply.events()); }); }
    reply.finish(done); check(reply.events().size() == 6 && done.find("status")->string() == "stopped");
    refuses([&] { Verify8Reply old(ready, previous, generate, 36, sampling_policy, 0); for (const auto& event : events) old.event(event); old.finish(done); });
    const auto restore_job = begin_job(std::span<const uint32_t>(fake.history).first(36), true);
    const auto restored = fake(restore_job, [](const Json&) { check(false); }); validate_branch8_control(ready, done, restore_job, restored); ++checks;
    check(integer(restored,"committed_prefix")==36 && integer(*restored.find("branch8_session"),"lookup_truncated_grams")==3);
    check(restored.find("branch8_session")->find("hc_exact_routes")->dump()==done.find("hc_exact_routes")->dump(), "restoring model state rewound execution cost");
    for (uint32_t n=2;n<=7;++n) for (uint32_t k=1;k<=n;++k) {
        std::vector<Round> rounds{{n,k,uint32_t(k<n)}};
        for (uint32_t used=k+(k<n);used<n;++used) rounds.push_back({0,0,1});
        auto t=make_transcript(ready,*restored.find("branch8_session"),generation_job(n),policy(),0,rounds);
        for (const auto name : {"lookup_truncated_grams","lookup_cleared_grams","lookup_truncations","lookup_candidates_examined","lookup_token_comparisons","lookup_work_limited_queries"})
            t.done=with(std::move(t.done),name,*restored.find("branch8_session")->find(name));
        t.done=with(std::move(t.done),"logits_available",Json(true)); t.done=with(std::move(t.done),"checkpoint_capture_wall_ms",number(0));
        t.done=with(std::move(t.done),"checkpoint_restore_wall_ms",number(0));
        auto output=t.done.find("generated_ids")->array(); output[0]=Json::integer(112); t.done=with(std::move(t.done),"generated_ids",Json(output));
        bool first=true; for (auto& event:t.events) if (event.find("event")->string()=="token" && first) { event=with(std::move(event),"id",Json::integer(112)); first=false; }
        const auto command=integer(restored,"command_id")+1, lease=integer(restored,"lease_id");
        const auto state=snapshot(t.done,*restored.find("branch8_session")); t.done=stamp(without(t.done,"branch8_session"),command,lease,state);
        for (auto& event:t.events) event=stamp(std::move(event),command,lease);
        Verify8Reply candidate(ready,restored,t.job,36,sampling_policy,0,false,true);
        for (const auto& event:t.events) candidate.event(event); candidate.finish(t.done); check(candidate.events().size()==t.events.size());
        for (const auto& [field,value] : Json::Object{{"broker_batch_width",Json::integer(1)},{"broker_epoch",Json::integer(43)},
            {"command_id",Json::integer(command+1)},{"lease_id",Json::integer(lease+1)}}) {
            Verify8Reply bad(ready,restored,t.job,36,sampling_policy,0,false,true);
            refuses([&]{bad.event(with(t.events[0],field,value));}); refuses([&]{static_cast<void>(bad.events());});
        }
    }
}

void session_checks(Json ready = prepared()) {
    Fake fake(std::move(ready)); ServeSession session(session_codec(), "tokspur", "direct", fake.ready, true, fake.ready, true);
    bool cancelled = false; std::vector<Json> chunks;
    const ServeSink sink{[&](const Json& event) { chunks.push_back(event); }, [&] { return cancelled; }, [] { return 0; }};
    const ServeTarget target = [&](const Json& job, const auto& event) { return fake(job, event); };
    refuses([&] { session.complete(chat(std::string(65536,'x'),false,8), "context-overflow", target, sink); }); check(fake.command==0);
    std::string prompt(2052, 'a'); auto result = session.complete(chat(prompt), "cold2052", target, sink);
    check(fake.captured_prefixes == std::vector<uint32_t>({512,1024,1536,2048,2052}), "branch8 cloned every tile or missed scheduled ancestors");
    check(integer(*result.find("tokspur_private"), "checkpoint_captures") == 5 && integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 0);
    fake.plan(); result = session.complete(chat(prompt), "exact2052", target, sink);
    check(integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 2052 && fake.restored_prefixes.back() == 2052);
    check(integer(*result.find("tokspur_private"), "known_input_batches") == 0 && integer(*result.find("tokspur_private"), "checkpoint_restores") == 1);
    prompt.back() = 'b'; fake.plan(); result = session.complete(chat(prompt), "sibling2048", target, sink);
    check(integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 2048 && fake.input_widths.back() == 4);
    fake.plan(); result = session.complete(chat(prompt.substr(0,1536)), "short1536", target, sink);
    check(integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 1536);
    const std::string append = prompt.substr(0,1536) + "ok<|im_end|>suffix";
    fake.plan(); result = session.complete(chat(append), "append", target, sink);
    check(integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 1539);
    prompt[0] = 'z'; fake.plan(); result = session.complete(chat(prompt), "deep-cold", target, sink);
    check(integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 0);
    const auto tool = with(chat("tool-root",true,128),"tools",parse_json(R"([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}])"));
    fake.plan("<tool_call><function=read_file><parameter=path>\nx\n</parameter></function></tool_call><|im_end|>");
    result=session.complete(tool,"terminal-tool",target,sink);
    check(result.find("choices")->array()[0].find("finish_reason")->string()=="tool_calls" &&
        result.find("choices")->array()[0].find("message")->find("tool_calls")->array().size()==1,"new branch path lost terminal tool admission");
    fake.plan(); const ServeTarget cancel_after_done = [&](const Json& job, const auto& event) {
        auto done = fake(job, event); if (!job.find("op") && integer(job, "generate")) cancelled = true; return done;
    };
    refuses([&] { session.complete(chat(prompt, true), "cancel", cancel_after_done, sink); }); check(session.committed_tokens() == 0);
    cancelled = false; fake.plan(); result = session.complete(chat(prompt), "cold-resync", target, sink);
    check(integer(*result.find("usage")->find("prompt_tokens_details"), "cached_tokens") == 0 && integer(*result.find("tokspur_private"), "checkpoint_cold_resyncs") == 1);
    for (const auto width : fake.input_widths) check(width && width <= 8);
    const auto command_before = fake.command;
    refuses([&] { session.complete(with(chat("x"), "temperature", number(0.5)), "non-greedy", target, sink); }); check(fake.command == command_before);
    chunks.clear(); fake.plan(); fake.corrupt_done = true;
    refuses([&] { session.complete(chat("fresh", true), "bad-snapshot", target, sink); }); check(session.committed_tokens() == 0);
    for (const auto& event : chunks) {
        const auto& delta = *event.find("choices")->array()[0].find("delta"); check(!delta.find("content") && !delta.find("tool_calls"));
    }
}

void cancelled_rewind_resync_checks(Json ready = prepared()) {
    Fake fake(std::move(ready)); ServeSession session(session_codec(),"tokspur","direct",fake.ready,true,fake.ready,true);
    bool cancelled=false;
    const ServeSink sink{[](const Json&) {},[&]{return cancelled;},[]{return 0;}};
    const ServeTarget target=[&](const Json& job,const auto& event){return fake(job,event);};
    const std::string prompt(2052,'a');
    static_cast<void>(session.complete(chat(prompt),"rewind-prime",target,sink));
    const auto stale=fake.previous;
    check(integer(*stale.find("branch8_session"),"committed_prefix")==2055 && fake.saved.size()==4,
        "rewind cancellation regression lacks an admitted long prefix and saved ancestors");
    Json restored;
    const ServeTarget cancel_after_restore=[&](const Json& job,const auto& event) {
        const auto worker_before=fake.previous; auto done=fake(job,event);
        if (job.find("op") && job.find("op")->string()=="branch_begin") {
            validate_branch8_control(fake.ready,worker_before,job,done); // Exact worker-side admission still succeeds.
            check(done.find("outcome")->string()=="restored" && integer(done,"committed_prefix")==1536,
                "cancellation happened without actually rewinding the worker");
            restored=done; cancelled=true; // HTTP's post-target cancellation check prevents reply publication.
        }
        return done;
    };
    fake.plan(); refuses([&]{session.complete(chat(prompt.substr(0,1536)),"cancel-restored-control",cancel_after_restore,sink);});
    check(cancelled && session.committed_tokens()==0 && !restored.is_null() && fake.history.size()==1536,
        "cancelled shorter control did not leave the worker restored and HTTP invalidated");
    cancelled=false; fake.plan(); bool checked=false;
    const ServeTarget cold_after_rewind=[&](const Json& job,const auto& event) {
        const auto worker_before=fake.previous; auto done=fake(job,event);
        if (job.find("op") && job.find("op")->string()=="branch_begin") {
            checked=true;
            check(!job.find("allow_restore")->boolean() && done.find("outcome")->string()=="cold_miss" && !integer(done,"committed_prefix") &&
                !integer(done,"common_prefix"),"rewind resync attempted a restore or reported a cache hit");
            const auto& now=*done.find("branch8_session"); const auto& old=*stale.find("branch8_session");
            const auto cleared=integer(now,"lookup_cleared_grams")-integer(old,"lookup_cleared_grams");
            check(cleared==grams(1536) && cleared<grams(uint32_t(integer(old,"committed_prefix"))),
                "regression does not exercise fewer cleared grams than the stale HTTP prefix");
            validate_branch8_control(fake.ready,worker_before,job,done); // Exact delta uses authoritative1536, not stale2055.
            validate_branch8_control(fake.ready,stale,job,done,true); // The explicitly cold HTTP resync can rebase.
            auto broken=with(now,"lookup_cleared_grams",Json::integer(integer(now,"lookup_cleared_grams")-1));
            refuses([&]{validate_branch8_control(fake.ready,worker_before,job,with(done,"branch8_session",broken));});
            broken=with(now,"lookup_resets",*old.find("lookup_resets"));
            refuses([&]{validate_branch8_control(fake.ready,stale,job,with(done,"branch8_session",broken),true);});
            broken=with(now,"lookup_appended_tokens",Json::integer(integer(old,"lookup_appended_tokens")-1));
            refuses([&]{validate_branch8_control(fake.ready,stale,job,with(done,"branch8_session",broken),true);});
            refuses([&]{validate_branch8_control(fake.ready,stale,with(job,"allow_restore",Json(true)),done,true);});
            refuses([&]{validate_branch8_control(fake.ready,stale,job,with(done,"outcome",Json("restored")),true);});
        }
        return done;
    };
    const auto result=session.complete(chat(prompt),"cold-after-rewind",cold_after_rewind,sink);
    check(checked && integer(*result.find("usage")->find("prompt_tokens_details"),"cached_tokens")==0 &&
        integer(*result.find("tokspur_private"),"checkpoint_cold_resyncs")==1 && !integer(*result.find("tokspur_private"),"checkpoint_restores"),
        "cancelled rewind failed to recover through an explicitly cold, zero-hit resync");
}

int target(int argc, char** argv) {
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    bool stage_present = false, stage_enabled = false;
    for (int i = 2; i + 1 < argc; ++i) if (std::string_view(argv[i]) == "--qsa-stage-v8") {
        if (stage_present || (std::string_view(argv[i+1]) != "off" && std::string_view(argv[i+1]) != "on")) return 90;
        stage_present = true; stage_enabled = std::string_view(argv[i+1]) == "on";
    }
#endif
    const std::array<std::pair<std::string_view, std::string_view>, 18> fixed{{
        {"--allow-full-model","true"},{"--small-gemv-splits","16"},{"--block-gemv-formats","3"},{"--stack-execution","direct"},
        {"--known-token-capacity","8"},{"--reuse-residual-projections","true"},{"--prompt-lookup","true"},{"--verification-budget-bytes","134217728"},
        {"--allow-prompt-lookup-experiments",
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            stage_present ? "false" :
#endif
            "true"},{"--verification-max-tokens","8"},{"--hc-exact","combined"},{"--wide-q8-exact","off"},
        {"--wide-q8-many","off"},{"--verification8-branch-reuse","true"},{"--worker-broker8","true"},{"--checkpoint-slots","4"},
        {"--checkpoint-budget-bytes","536870912"},{"--worker-broker","true"}}};
    if (argc != 2 + 2 * int(fixed.size() + 2) + 4
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        + 2 * int(stage_present)
#endif
        ) return 91;
    for (const auto& [name, value] : fixed) {
        unsigned found = 0; for (int i = 2; i + 1 < argc; ++i) if (std::string_view(argv[i]) == name && std::string_view(argv[i+1]) == value) ++found;
        if (found != 1) return 92;
    }
    std::filesystem::path policy_path;
    for (int i = 2; i + 1 < argc; ++i) if (std::string_view(argv[i]) == "--policy") policy_path = argv[i+1];
    if (policy_path.empty()) return 93;
    const auto directory = policy_path.parent_path(), mode = policy_path.filename();
    { std::ofstream file(directory / "started"); file << "explicit branch8 fake argv\n"; }
    Fake fake(
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        stage_present ? staged_prepared(stage_enabled) :
#endif
        prepared());
    Json initial = fake.ready;
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    if (mode == "wrong-stage-arm") initial = staged_prepared(!stage_enabled);
    if (mode == "unprepared-stage") initial = prepared();
#endif
    emit(initial); std::string line;
    while (std::getline(std::cin, line)) {
        { std::ofstream file(directory / "commands", std::ios::app); file << "command\n"; }
        const auto wire = parse_json(line);
        if (integer(wire,"broker_epoch") != 4242 || integer(wire,"command_id") != fake.command + 1 || integer(wire,"expected_prefix") != fake.history.size() ||
            wire.find("broker_protocol")->string() != "branch8-v1" || integer(wire,"broker_batch_width") != 8) return 94;
        Json job = wire;
        for (const auto name : {"broker_epoch","command_id","lease_id","expected_prefix","broker_protocol","broker_batch_width"}) job = without(job,name);
        const bool begin = job.find("op") && job.find("op")->string() == "branch_begin";
        if (integer(wire,"lease_id") != fake.lease + begin) return 95;
        const bool generate = !job.find("op") && integer(job,"generate");
        if (begin) fake.plan();
        auto done = fake(job, [&](Json event) {
            if (mode == "wrong-width") event = with(std::move(event),"broker_batch_width",Json::integer(1)); emit(event);
        });
        if (mode=="wrong-control-lease" && begin) done=with(std::move(done),"lease_id",Json::integer(999));
        if (generate) {
            { std::ofstream file(directory / "events-sent"); file << "events emitted, done withheld\n"; }
            if (mode == "truncated") return 0;
            if (mode == "slow-done") {
                for (unsigned wait = 0; wait < 100 && !std::filesystem::exists(directory / "cancel-observed"); ++wait) pause_ms(5);
                if (!std::filesystem::exists(directory / "cancel-observed")) return 96;
            }
            if (mode == "bad-snapshot") {
                auto broken = with(*done.find("branch8_session"),"lookup_resets",Json::integer(999));
                done = with(std::move(done),"branch8_session",std::move(broken));
            }
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            if (mode == "bad-stage-cost") done = with(std::move(done), "qsa_stage_v8_request_calls", Json::integer(999));
#endif
        }
        emit(done);
    }
    return 0;
}
struct WorkerFiles : WorkerFixture {
    WorkerFiles(const std::filesystem::path& executable, std::string_view mode, bool stage_present = false, bool stage_enabled = false) : WorkerFixture(executable,mode) {
        const auto named = directory / "branch8-target"; std::error_code error;
        std::filesystem::create_hard_link(options.executable,named,error); if (error) std::filesystem::copy_file(options.executable,named);
        options.executable = named; options.branch8 = true; options.checkpoint_slots = 4; options.checkpoint_budget_bytes = 512ull << 20;
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        options.stage8_present = stage_present; options.stage8_enabled = stage_enabled;
#else
        static_cast<void>(stage_present); static_cast<void>(stage_enabled);
#endif
    }
};
void worker_checks(const std::filesystem::path& executable, bool stage_present = false, bool stage_enabled = false) {
    const std::array<uint32_t,8> ids{100,101,102,103,104,105,106,107}; const auto cold = begin_job(ids,false), warm = begin_job(ids,true);
    const auto input = input_job(8,false,policy(),0); const auto capture = parse_json(R"({"op":"branch_capture"})");
    {
        WorkerFiles f(executable,"normal",stage_present,stage_enabled); OwnedWorker worker(f.options); size_t callbacks = 0;
        refuses([&] { worker.request(input_job(8,true,policy(),0),Clock::now()+3s,[]{return false;}); }); check(f.commands()==0);
        worker.request(cold,Clock::now()+3s,[]{return false;}); worker.request(input,Clock::now()+3s,[]{return false;});
        worker.request(capture,Clock::now()+3s,[]{return false;});
        worker.request(generation_job(8),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;}); check(callbacks==6);
        const auto restored = worker.request(warm,Clock::now()+3s,[]{return false;}); check(integer(restored,"committed_prefix")==8 && restored.find("outcome")->string()=="restored");
        worker.request(generation_job(8),Clock::now()+3s,[]{return false;}); worker.close(); check(!worker.alive());
    }
    {
        WorkerFiles f(executable,"wrong-control-lease",stage_present,stage_enabled); OwnedWorker worker(f.options); size_t callbacks=0;
        refuses([&]{worker.request(cold,Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;});});
        check(!callbacks && !worker.alive() && f.commands()==1,"worker accepted a different fresh lease than the exact issued lease");
        refuses([&]{worker.request(cold,Clock::now()+3s,[]{return false;});});
        check(f.commands()==1,"wrong control-lease echo triggered a retry"); worker.close();
    }
    for (const auto mode : {"wrong-width","bad-snapshot","truncated"}) {
        WorkerFiles f(executable,mode,stage_present,stage_enabled); OwnedWorker worker(f.options); size_t callbacks = 0;
        worker.request(cold,Clock::now()+3s,[]{return false;}); worker.request(input,Clock::now()+3s,[]{return false;});
        refuses([&]{worker.request(generation_job(8),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;});});
        check(callbacks==0 && !worker.alive() && f.commands()==3); worker.close();
    }
    {
        WorkerFiles f(executable,"slow-done",stage_present,stage_enabled); OwnedWorker worker(f.options); size_t callbacks = 0; bool cancelled = false;
        worker.request(cold,Clock::now()+3s,[]{return false;}); worker.request(input,Clock::now()+3s,[]{return false;});
        try { worker.request(generation_job(8),Clock::now()+3s,[&]{
            if (!std::filesystem::exists(f.directory/"events-sent")) return false;
            std::ofstream file(f.directory/"cancel-observed"); file<<"cancel acknowledged\n"; return true;
        },[&](const Json&){++callbacks;}); } catch (const WorkerCancelled&) { cancelled=true; }
        check(cancelled && !callbacks && worker.alive()); const auto commands=f.commands();
        refuses([&]{worker.request(warm,Clock::now()+3s,[]{return false;});}); check(f.commands()==commands);
        const auto reset=worker.request(cold,Clock::now()+3s,[]{return false;}); check(integer(reset,"committed_prefix")==0);
        worker.request(input,Clock::now()+3s,[]{return false;}); worker.close();
    }
    for (const auto mode : {"no-consent","no-verify","no-history","replay"}) {
        WorkerFiles f(executable,"normal",stage_present,stage_enabled);
        if (std::string_view(mode)=="no-consent") f.options.branch8=false;
        if (std::string_view(mode)=="no-verify") f.options.verify8=false;
        if (std::string_view(mode)=="no-history") { f.options.checkpoint_slots=0; f.options.checkpoint_budget_bytes=0; }
        if (std::string_view(mode)=="replay") f.options.stack_execution="recurrent-replay";
        refuses([&]{OwnedWorker wrong(f.options);}); check(!std::filesystem::exists(f.directory/"started") && f.commands()==0);
    }
}
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
Json stage_cost(Json value, uint64_t calls) {
    value = with(std::move(value), "qsa_stage_v8_calls", Json::integer(calls));
    value = with(std::move(value), "qsa_stage_v8_physical_dispatches", Json::integer(2 * calls));
    return with(std::move(value), "qsa_stage_v8_extra_dispatches", Json::integer(calls));
}
void stage_startup_checks() {
    check(qsa_stage_v8_startup_allowed(0,false,16,3,0,8,false,false,false,false,false,false,false));
    check(!qsa_stage_v8_startup_allowed(0,false,16,3,0,8,false,false,false,false,true,false,false));
    check(qsa_stage_v8_startup_allowed(0,false,16,3,0,8,false,false,false,false,true,false,false,true));
    for (unsigned unsafe = 0; unsafe < 13; ++unsafe) {
        check(!qsa_stage_v8_startup_allowed(unsafe==0,unsafe==1,unsafe==2?8:16,unsafe==3?1:3,unsafe==4?8:0,
            unsafe==5?0:8,unsafe==6,unsafe==7,unsafe==8,unsafe==9,true,unsafe==10,unsafe==11,unsafe!=12),
            "staged-QSA broker startup admitted an unsafe fixed-policy combination");
    }
}
void stage_admission_checks(const Json& ready) {
    const bool enabled = ready.find("qsa_stage_v8_enabled")->boolean(); validate_branch8_ready(ready); ++checks;
    for (const auto name : {"qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes", "qsa_stage_v8_scratch_charged_bytes",
        "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches"}) {
        refuses([&]{validate_branch8_ready(without(ready,name));});
        refuses([&]{validate_branch8_ready(with(ready,"branch8_session",without(*ready.find("branch8_session"),name)));});
    }
    for (const auto& [name,value] : Json::Object{{"qsa_stage_v8_prepared",Json(false)}, {"qsa_stage_v8_scratch_logical_bytes",Json::integer(0)},
        {"qsa_stage_v8_scratch_charged_bytes",Json::integer(0)}, {"qsa_stage_v8_scratch_charged_bytes",Json::integer(212991)},
        {"qsa_stage_v8_calls",Json::integer(1)}, {"qsa_stage_v8_physical_dispatches",Json::integer(1)},
        {"qsa_stage_v8_extra_dispatches",Json::integer(1)}, {"qsa_stage_v8_unknown",Json::integer(0)},
        {"allow_prompt_lookup_experiments",Json(true)}})
        refuses([&]{validate_branch8_ready(with(ready,name,value));});
    Fake fake(ready); const std::array<uint32_t,8> ids{100,101,102,103,104,105,106,107};
    const auto cold_job = begin_job(ids,false), initial = fake.previous;
    const auto cold = fake(cold_job,[](const Json&){check(false);}); validate_branch8_control(ready,initial,cold_job,cold); ++checks;
    const auto input = input_job(8,false,policy(),0), before_input = fake.previous;
    const auto filled = fake(input,[](const Json&){check(false);});
    // Keep the sampling Json alive for admission's non-owning reference.
    const auto greedy = policy();
    Verify8Reply input_reply(ready,before_input,input,0,greedy,0,false,true); input_reply.finish(filled); check(input_reply.events().empty());
    check(integer(filled,"qsa_stage_v8_request_calls") == (enabled?96:0));
    Verify8Reply corrupt_input(ready,before_input,input,0,greedy,0,false,true);
    refuses([&]{corrupt_input.finish(with(filled,"qsa_stage_v8_request_calls",Json::integer(999)));});
    const auto capture_job = parse_json(R"({"op":"branch_capture"})"), before_capture = fake.previous;
    const auto captured = fake(capture_job,[](const Json&){check(false);}); validate_branch8_control(ready,before_capture,capture_job,captured); ++checks;
    if (enabled) {
        auto bad_state = stage_cost(*captured.find("branch8_session"),integer(*captured.find("branch8_session"),"qsa_stage_v8_calls")+12);
        refuses([&]{validate_branch8_control(ready,before_capture,capture_job,with(captured,"branch8_session",bad_state));});
    }
    const auto previous = fake.previous, job = generation_job(7);
    std::vector<Round> rounds{{7,2,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
    auto transcript = make_transcript(ready,*previous.find("branch8_session"),job,greedy,0,rounds);
    // The generic V8 fixture omits these branch-only lifetime fields. Keep the
    // complete native envelope; do not weaken branch reply admission for it.
    for (const auto name : {"lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries"})
        transcript.done = with(std::move(transcript.done),name,*previous.find("branch8_session")->find(name));
    transcript.done = with(std::move(transcript.done),"logits_available",Json(true));
    transcript.done = with(std::move(transcript.done),"checkpoint_capture_wall_ms",number(0));
    transcript.done = with(std::move(transcript.done),"checkpoint_restore_wall_ms",number(0));
    auto generated = transcript.done.find("generated_ids")->array();
    const auto first_id = integer(previous.find("branch8_session")->find("final_logit_top8")->array()[0],"id");
    generated[0] = Json::integer(first_id); transcript.done = with(std::move(transcript.done),"generated_ids",Json(std::move(generated)));
    bool first = true;
    for (auto& event : transcript.events) {
        if (first && event.find("event")->string()=="token") { event=with(std::move(event),"id",Json::integer(first_id)); first=false; }
        event=stamp(std::move(event),integer(previous,"command_id")+1,integer(previous,"lease_id"));
    }
    transcript.done=stamp(transcript.done,integer(previous,"command_id")+1,integer(previous,"lease_id"),snapshot(transcript.done,*previous.find("branch8_session")));
    const auto verify = [&](const std::vector<Json>& events,const Json& done) {
        Verify8Reply reply(ready,previous,job,8,greedy,0,false,true);
        for (const auto& event : events) { reply.event(event); refuses([&]{static_cast<void>(reply.events());}); }
        reply.finish(done); check(reply.events().size()==events.size());
    };
    verify(transcript.events,transcript.done);
    check(integer(transcript.done,"qsa_stage_v8_request_calls")==(enabled?144:0),"staged cost counted accepted rows rather than evaluated rows");
    auto broken_events = transcript.events;
    broken_events[0]=with(broken_events[0],"qsa_stage_v8_round_calls",Json::integer(integer(broken_events[0],"qsa_stage_v8_round_calls")+1));
    refuses([&]{verify(broken_events,transcript.done);});
    broken_events=transcript.events; broken_events[0]=without(broken_events[0],"qsa_stage_v8_round_calls");
    refuses([&]{verify(broken_events,transcript.done);});
    broken_events=transcript.events;
    const auto old_dispatches=integer(broken_events[0],"dispatches");
    broken_events[0]=with(broken_events[0],"dispatches",Json::integer(enabled?old_dispatches-96:old_dispatches+12));
    refuses([&]{verify(broken_events,transcript.done);});
    const auto now=integer(transcript.done,"qsa_stage_v8_calls");
    auto bad_done=stage_cost(transcript.done,now+1);
    bad_done=with(std::move(bad_done),"branch8_session",stage_cost(*transcript.done.find("branch8_session"),now+1));
    refuses([&]{verify(transcript.events,bad_done);});
    refuses([&]{verify(transcript.events,without(transcript.done,"qsa_stage_v8_request_calls"));});
    bad_done=with(transcript.done,"qsa_stage_v8_enabled",Json(!enabled));
    bad_done=with(std::move(bad_done),"branch8_session",with(*transcript.done.find("branch8_session"),"qsa_stage_v8_enabled",Json(!enabled)));
    refuses([&]{verify(transcript.events,bad_done);});
    const auto generated_done=fake(generation_job(8),[](const Json&){});
    const auto restore_job=begin_job(ids,true), restored=fake(restore_job,[](const Json&){check(false);});
    validate_branch8_control(ready,generated_done,restore_job,restored); ++checks;
    check(integer(*restored.find("branch8_session"),"qsa_stage_v8_calls")==integer(generated_done,"qsa_stage_v8_calls"),
        "model restore rewound lifetime staged execution work");
    if (enabled) {
        const auto stale_cost=integer(*captured.find("branch8_session"),"qsa_stage_v8_calls");
        check(stale_cost<integer(generated_done,"qsa_stage_v8_calls"),"restore fixture did not perform intervening stage work");
        auto rewound=with(restored,"branch8_session",stage_cost(*restored.find("branch8_session"),stale_cost));
        refuses([&]{validate_branch8_control(ready,generated_done,restore_job,rewound);});
        const auto cold_before=fake.previous, cold_again=fake(cold_job,[](const Json&){check(false);});
        validate_branch8_control(ready,cold_before,cold_job,cold_again); ++checks;
        auto advanced=with(cold_again,"branch8_session",stage_cost(*cold_again.find("branch8_session"),integer(*cold_again.find("branch8_session"),"qsa_stage_v8_calls")+12));
        refuses([&]{validate_branch8_control(ready,cold_before,cold_job,advanced);}); // Exact worker predecessor.
        validate_branch8_control(ready,cold_before,cold_job,advanced,true); ++checks; // Stale HTTP observer may have missed work.
        auto fractional=with(cold_again,"branch8_session",stage_cost(*cold_again.find("branch8_session"),integer(*cold_again.find("branch8_session"),"qsa_stage_v8_calls")+1));
        refuses([&]{validate_branch8_control(ready,cold_before,cold_job,fractional,true);}); // Whole model rows execute all12 QSA layers.
    }
}
void stage_worker_checks(const std::filesystem::path& executable,bool enabled) {
    const std::array<uint32_t,8> ids{1,2,3,4,5,6,7,8};
    for (const auto mode : {"wrong-stage-arm","unprepared-stage"}) {
        WorkerFiles f(executable,mode,true,enabled); refuses([&]{OwnedWorker worker(f.options);});
        check(std::filesystem::exists(f.directory/"started") && f.commands()==0,"stage readiness mismatch was not tested at the actual child boundary");
    }
    {
        WorkerFiles f(executable,"bad-stage-cost",true,enabled); OwnedWorker worker(f.options); size_t callbacks=0;
        worker.request(begin_job(ids,false),Clock::now()+3s,[]{return false;});
        worker.request(input_job(8,false,policy(),0),Clock::now()+3s,[]{return false;});
        refuses([&]{worker.request(generation_job(8),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;});});
        check(!callbacks && !worker.alive() && f.commands()==3); worker.close();
    }
    {
        WorkerFiles f(executable,"normal",false,true); refuses([&]{OwnedWorker worker(f.options);});
        check(!std::filesystem::exists(f.directory/"started"),"enabled without selector presence launched a child");
    }
}
void stage_checks(const std::filesystem::path& executable) {
    stage_startup_checks();
    const auto off=staged_prepared(false), on=staged_prepared(true), absent=prepared();
    check(!absent.find("qsa_stage_v8_prepared") && absent.find("allow_prompt_lookup_experiments")->boolean());
    for (const auto name : {"qsa_stage_v8_scratch_logical_bytes","qsa_stage_v8_scratch_charged_bytes","tracked_metal_buffer_bytes","peak_metal_buffer_bytes","gpu_allocation_count"})
        check(integer(off,name)==integer(on,name),"prepared OFF/ON ledgers differ");
    for (const auto enabled : {false,true}) {
        const auto ready=staged_prepared(enabled);
        admission_checks(ready); stage_admission_checks(ready); session_checks(ready); cancelled_rewind_resync_checks(ready);
        worker_checks(executable,true,enabled); stage_worker_checks(executable,enabled);
    }
}
#endif
} // namespace branch_fixture

#ifndef TOKSPUR_HTTP_BRANCH8_TEST_ENTRY
#define TOKSPUR_HTTP_BRANCH8_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_BRANCH8_TEST_ENTRY(int argc,char** argv) {
    try {
        if (argc>1 && std::string_view(argv[1])=="target")
            return std::filesystem::path(argv[0]).filename()=="branch8-target" ? branch_fixture::target(argc,argv) : fake_target(argc,argv);
        const auto prior = previous_http_verify8_main(argc,argv); if (prior) return prior;
        branch_fixture::admission_checks(); branch_fixture::session_checks(); branch_fixture::cancelled_rewind_resync_checks();
        branch_fixture::worker_checks(std::filesystem::canonical(argv[0]));
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        branch_fixture::stage_checks(std::filesystem::canonical(argv[0]));
#endif
        std::cout<<"Private HTTP branch8 CPU checks="<<checks<<'\n'; return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
