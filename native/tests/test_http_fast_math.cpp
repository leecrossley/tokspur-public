// Fresh CPU-generated contracts only; no historical receipt is rewritten.
#define TOKSPUR_HTTP_BRANCH8_TEST_ENTRY inherited_branch8_checks
#include "test_http_branch8.cpp"
#undef TOKSPUR_HTTP_BRANCH8_TEST_ENTRY

#ifndef TOKSPUR_PRIVATE_HTTP_FAST_MATH
#error This fixture requires the explicit corrected-PLE/channel HTTP build.
#endif

namespace fast_fixture {
constexpr std::array<const char*, 8> names{"ple_gate_transform", "ple_gate_transform_schema", "qsa_channel_chronological_mode",
    "qsa_channel_chronological_schema", "qsa_channel_factor_logical_bytes", "qsa_channel_factor_charged_bytes",
    "qsa_channel_calls", "qsa_channel_extra_dispatches"};
Json fields(Json value, bool channel) {
    value = with(std::move(value), "ple_gate_transform", Json("signed-root"));
    value = with(std::move(value), "ple_gate_transform_schema", Json::integer(1));
    value = with(std::move(value), "qsa_channel_chronological_mode", Json(channel ? "on" : "off"));
    value = with(std::move(value), "qsa_channel_chronological_schema", Json::integer(1));
    value = with(std::move(value), "qsa_channel_factor_logical_bytes", Json::integer(393792));
    value = with(std::move(value), "qsa_channel_factor_charged_bytes", Json::integer(409600));
    const auto calls = channel ? integer(value, "qsa_stage_v8_calls") : 0;
    value = with(std::move(value), "qsa_channel_calls", Json::integer(calls));
    return with(std::move(value), "qsa_channel_extra_dispatches", Json::integer(calls));
}
Json annotate(Json value, bool channel) {
    const std::string event(value.find("event")->string());
    if (event == "generation_round") {
        const auto rows = integer(value, "verified_tokens") + integer(value, "ordinary_tokens");
        const auto extra = channel ? 12 * rows : 0;
        const auto dispatches = integer(value, "dispatches") + extra;
        value = with(std::move(value), "qsa_channel_round_calls", Json::integer(extra));
        return with(std::move(value), "dispatches", Json::integer(dispatches));
    }
    if (event == "ready" || event == "done") {
        value = fields(std::move(value), channel);
        if (event == "done") {
            const auto calls = channel ? integer(value, "qsa_stage_v8_request_calls") : 0;
            value = with(std::move(value), "qsa_channel_request_calls", Json::integer(calls));
        }
    }
    if (const auto* state = value.find("branch8_session")) {
        auto annotated = fields(*state, channel);
        value = with(std::move(value), "branch8_session", std::move(annotated));
    }
    return value;
}
Json core_ready() {
    auto ready = branch_fixture::staged_prepared(true);
    for (const auto name : {"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes"}) {
        const auto bytes = integer(ready, name) + 409600;
        ready = with(std::move(ready), name, Json::integer(bytes));
    }
    const auto allocations = integer(ready, "gpu_allocation_count") + 1;
    ready = with(std::move(ready), "gpu_allocation_count", Json::integer(allocations));
    auto state = *ready.find("branch8_session");
    for (const auto name : branch_fixture::graph) state = with(std::move(state), name, *ready.find(name));
    return with(std::move(ready), "branch8_session", std::move(state));
}
struct Fake {
    branch_fixture::Fake inner;
    bool channel;
    Json ready, previous;
    explicit Fake(bool enabled) : inner(core_ready()), channel(enabled), ready(annotate(inner.ready, enabled)), previous(ready) {}
    Json operator()(const Json& job, const std::function<void(const Json&)>& callback) {
        std::vector<Json> events;
        auto done = inner(job, [&](const Json& event) { events.push_back(annotate(event, channel)); });
        previous = annotate(std::move(done), channel);
        for (const auto& event : events) callback(event);
        return previous;
    }
};
void admission_and_rounds(bool channel) {
    Fake fake(channel); const auto ready = fake.ready;
    const auto sampling_policy = policy();
    validate_branch8_ready(ready); check(verify8_fast_math(ready) == std::optional<bool>(channel));
    for (const auto name : names) {
        refuses([&] { validate_branch8_ready(without(ready, name)); });
        refuses([&] { validate_branch8_ready(with(ready, "branch8_session", without(*ready.find("branch8_session"), name))); });
    }
    for (const auto& [name, value] : Json::Object{{"ple_gate_transform",Json("plain")}, {"ple_gate_transform_schema",Json::integer(2)},
        {"qsa_channel_chronological_schema",Json(true)}, {"qsa_channel_chronological_mode",Json("invalid")},
        {"qsa_channel_factor_logical_bytes",Json::integer(0)}, {"qsa_channel_factor_charged_bytes",Json::integer(393792)},
        {"qsa_channel_calls",Json::integer(12)}, {"qsa_channel_extra_dispatches",Json::integer(12)},
        {"qsa_channel_unknown",Json::integer(0)}, {"licensed_q4_routed_mode",Json("on")}, {"batch_phase_clock_mode",Json("on")}})
        refuses([&] { validate_branch8_ready(with(ready, name, value)); });
    const std::array<uint32_t, 8> ids{100,101,102,103,104,105,106,107};
    auto previous = ready; const auto cold_job = branch_fixture::begin_job(ids, false);
    const auto cold = fake(cold_job, [](const Json&) { check(false); }); validate_branch8_control(ready, previous, cold_job, cold); previous = cold;
    const auto input = input_job(8, false, policy(), 0);
    const auto filled = fake(input, [](const Json&) { check(false); });
    Verify8Reply prefill(ready, previous, input, 0, sampling_policy, 0, false, true); prefill.finish(filled); previous = filled;
    check(integer(filled, "qsa_channel_request_calls") == (channel ? 96 : 0));
    const auto capture_job = parse_json(R"({"op":"branch_capture"})");
    const auto captured = fake(capture_job, [](const Json&) { check(false); }); validate_branch8_control(ready, previous, capture_job, captured); previous = captured;
    const auto generation = generation_job(8); std::vector<Json> events;
    const auto done = fake(generation, [&](const Json& v) { events.push_back(v); });
    const auto verify = [&](const std::vector<Json>& stream, const Json& result) {
        Verify8Reply reply(ready, previous, generation, 8, sampling_policy, 0, false, true);
        for (const auto& e : stream) reply.event(e); reply.finish(result); check(!reply.events().empty());
    };
    verify(events, done);
    auto broken = events; broken[0] = without(broken[0], "qsa_channel_round_calls"); refuses([&] { verify(broken, done); });
    broken = events; broken[0] = with(broken[0], "qsa_channel_round_calls", Json::integer(999)); refuses([&] { verify(broken, done); });
    broken = events; broken[0] = with(broken[0], "dispatches", Json::integer(integer(broken[0], "dispatches") + 1)); refuses([&] { verify(broken, done); });
    refuses([&] { verify(events, with(done, "qsa_channel_request_calls", Json::integer(999))); });
    auto bad_state = fields(*done.find("branch8_session"), !channel);
    refuses([&] { verify(events, with(done, "branch8_session", bad_state)); });
    const auto warm_job = branch_fixture::begin_job(ids, true);
    const auto restored = fake(warm_job, [](const Json&) { check(false); }); validate_branch8_control(ready, done, warm_job, restored);
    check(integer(*restored.find("branch8_session"), "qsa_channel_calls") == integer(done, "qsa_channel_calls"), "channel lifetime counter rewound on restore");
    // Independently generated partial rejection: seven evaluated, three
    // accepted, correction plus four fallback rows =12 evaluated model rows,
    // NOT the eight committed outputs. Preserve the existing recovery budget.
    const auto& plain_previous = fake.inner.state;
    const std::array<Round,5> rounds{{{7,3,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1}}};
    auto t = make_transcript(fake.inner.ready, plain_previous, generation, policy(), 0, rounds);
    for (const auto name : {"lookup_truncated_grams", "lookup_cleared_grams", "lookup_truncations",
        "lookup_candidates_examined", "lookup_token_comparisons", "lookup_work_limited_queries"})
        t.done = with(std::move(t.done), name, *plain_previous.find(name));
    t.done = with(std::move(t.done), "logits_available", Json(true));
    t.done = with(std::move(t.done), "checkpoint_retained", Json::integer(fake.inner.saved.size()));
    t.done = with(std::move(t.done), "checkpoint_capture_wall_ms", number(0));
    t.done = with(std::move(t.done), "checkpoint_restore_wall_ms", number(0));
    auto partial_state = branch_fixture::snapshot(t.done, plain_previous);
    t.done = with(std::move(t.done), "branch8_session", std::move(partial_state));
    auto partial_done = annotate(branch_fixture::stamp(t.done, fake.inner.command + 1, fake.inner.lease), channel);
    auto before = restored;
    auto state = *before.find("branch8_session");
    auto top = state.find("final_logit_top8")->array(); top[0] = with(top[0], "id", Json::integer(200));
    state = with(std::move(state), "final_logit_top8", Json(std::move(top)));
    before = with(std::move(before), "branch8_session", state);
    std::vector<Json> partial_events;
    for (const auto& e : t.events) partial_events.push_back(annotate(branch_fixture::stamp(e, fake.inner.command + 1, fake.inner.lease), channel));
    const auto verify_partial = [&](const Json& result) {
        Verify8Reply reply(ready, before, generation, 8, sampling_policy, 0, false, true);
        for (const auto& e : partial_events) reply.event(e); reply.finish(result); check(!reply.events().empty());
    };
    verify_partial(partial_done);
    check(integer(partial_done, "qsa_channel_request_calls") == (channel ? 144 : 0));
    if (channel) refuses([&] { verify_partial(with(partial_done, "qsa_channel_request_calls", Json::integer(96))); });
}
int target(int argc, char** argv) {
    std::optional<bool> channel; std::filesystem::path policy_path; bool signed_root = false;
    for (int i = 2; i + 1 < argc; ++i) {
        const std::string_view key(argv[i]), value(argv[i+1]);
        if (key == "--qsa-channel-chronological") { if (channel.has_value() || (value != "off" && value != "on")) return 81; channel = value == "on"; }
        if (key == "--ple-gate-transform") signed_root = value == "signed-root";
        if (key == "--policy") policy_path = value;
    }
    if (!channel.has_value() || !signed_root || policy_path.empty()) return 82;
    const auto directory = policy_path.parent_path(), mode = policy_path.filename();
    { std::ofstream out(directory / "started"); out << "explicit fast-math fixture argv\n"; }
    Fake fake(*channel); Json ready = fake.ready;
    if (mode == "wrong-fast-arm") ready = Fake(!*channel).ready;
    if (mode == "missing-fast") ready = core_ready();
    emit(ready); std::string line;
    while (std::getline(std::cin, line)) {
        { std::ofstream out(directory / "commands", std::ios::app); out << "command\n"; }
        auto job = parse_json(line);
        if (integer(job,"broker_epoch") != 4242 || integer(job,"command_id") != fake.inner.command + 1 ||
            integer(job,"expected_prefix") != fake.inner.history.size() || integer(job,"broker_batch_width") != 8 ||
            job.find("broker_protocol")->string() != "branch8-v1") return 83;
        const bool begin = job.find("op") && job.find("op")->string() == "branch_begin";
        if (integer(job,"lease_id") != fake.inner.lease + begin) return 84;
        for (const auto name : {"broker_epoch","command_id","lease_id","expected_prefix","broker_protocol","broker_batch_width"}) job = without(job,name);
        auto done = fake(job, [](const Json& v) { emit(v); });
        if (mode == "bad-fast-cost" && !job.find("op") && integer(job,"generate"))
            done = with(std::move(done), "qsa_channel_request_calls", Json::integer(999));
        emit(done);
    }
    return 0;
}
void workers(const std::filesystem::path& executable, bool channel) {
    const std::array<uint32_t,8> ids{1,2,3,4,5,6,7,8};
    for (const auto mode : {"normal", "bad-fast-cost", "wrong-fast-arm", "missing-fast"}) {
        branch_fixture::WorkerFiles f(executable, mode, true, true); f.options.fast_channel = channel;
        if (std::string_view(mode) == "wrong-fast-arm" || std::string_view(mode) == "missing-fast") {
            refuses([&] { OwnedWorker worker(f.options); }); check(f.commands() == 0); continue;
        }
        OwnedWorker worker(f.options); worker.request(branch_fixture::begin_job(ids,false),Clock::now()+3s,[]{return false;});
        worker.request(input_job(8,false,policy(),0),Clock::now()+3s,[]{return false;}); size_t callbacks = 0;
        const auto generate = [&] { worker.request(generation_job(8),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;}); };
        if (std::string_view(mode) == "bad-fast-cost") { refuses(generate); check(!callbacks && !worker.alive() && f.commands() == 3); }
        else { generate(); check(callbacks == 6 && worker.alive()); }
        worker.close();
    }
    for (bool present : {false,true}) {
        branch_fixture::WorkerFiles f(executable,"normal",present,false); f.options.fast_channel = channel;
        refuses([&] { OwnedWorker worker(f.options); }); check(!std::filesystem::exists(f.directory/"started"), "fast math without staged ON launched a child");
    }
}
}
#ifndef TOKSPUR_HTTP_FAST_MATH_TEST_ENTRY
#define TOKSPUR_HTTP_FAST_MATH_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_FAST_MATH_TEST_ENTRY(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "target") {
            for (int i = 2; i < argc; ++i) if (std::string_view(argv[i]) == "--qsa-channel-chronological") return fast_fixture::target(argc,argv);
            return inherited_branch8_checks(argc,argv);
        }
        const auto prior = inherited_branch8_checks(argc,argv); if (prior) return prior;
        for (bool channel : {false,true}) { fast_fixture::admission_and_rounds(channel); fast_fixture::workers(std::filesystem::canonical(argv[0]),channel); }
        std::cout << "Private HTTP fast-math CPU checks=" << checks << '\n'; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
