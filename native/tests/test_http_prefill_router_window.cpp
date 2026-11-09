// CPU serving contract controls. Fake children never load a model or open Metal.
#define TOKSPUR_HTTP_HC_ROWS_TEST_ENTRY inherited_router_window_entry
#include "test_http_prefill_hc_rows.cpp"
#undef TOKSPUR_HTTP_HC_ROWS_TEST_ENTRY
#ifndef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
#error Router window serving controls require the explicit overlay.
#endif
namespace {
Json router_window_ready() { return mtp_fixture::router_window_fields(rows_ready()); }
void router_window_metadata_and_wire() {
    const auto ready = router_window_ready(), omitted = rows_ready();
    validate_mtp_ready(ready, "tiled128", true, 2); validate_mtp_router_window_selection(ready, true); validate_mtp_router_window_selection(omitted, false);
    refuses([&] { validate_mtp_router_window_selection(ready, false); }); refuses([&] { validate_mtp_router_window_selection(omitted, true); });
    for (const auto& [name, value] : mtp_router_window_fields()) {
        static_cast<void>(value); refuses([&] { validate_mtp_ready(without(ready, name), "tiled128", true, 2); });
    }
    for (const auto& [name, value] : Json::Object{{"native_prefill_router_window_min_rows", Json::integer(1)}, {"native_prefill_router_window_max_rows", parse_json("512.0")},
        {"native_prefill_router_window_precision", Json("fp16")}, {"native_prefill_router_window_scratch_bytes", Json::integer(4)}, {"native_prefill_router_window_extra", Json(false)},
        {"native_prefill_router_window_router_encoded", Json::integer(1)}, {"native_prefill_dense_kernel", Json("graph_q8_mpp_wide")}})
        refuses([&] { validate_mtp_ready(with(ready, name, value), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(without(ready, "native_prefill_hc_rows_version"), "tiled128", true, 2); });
    mtp_fixture::Fake fake; fake.prepared = fake.previous = ready;
    for (uint32_t n : {512u, 332u, 18u, 490u, 2u, 3u, 511u, 1u}) {
        const auto job = mtp_fixture::input(n, fake.prefix == 0), previous = fake.previous, done = fake(job, [](const Json&) {});
        MtpWireReply good(ready, previous, job); good.finish(done); checks += 1;
        for (const auto name : mtp_router_window_identity_names) {
            refuses([&] { MtpWireReply bad(ready, without(previous, name), job); });
            MtpWireReply bad(ready, previous, job); refuses([&] { bad.finish(without(done, name)); });
        }
        for (size_t i = 0; i < 4; ++i) {
            check(integer(done, mtp_router_window_request_names[i]) == (n > 1 ? 48 : 0), "router window partial-row delta differs");
            for (const auto name : {mtp_router_window_counter_names[i], mtp_router_window_request_names[i]}) {
                MtpWireReply missing(ready, previous, job); refuses([&] { missing.finish(without(done, name)); });
                MtpWireReply drift(ready, previous, job); refuses([&] { drift.finish(with(done, name, Json::integer(integer(done, name) + 1))); });
            }
        }
    }
    const auto cold = mtp_fixture::input(332), recovered = fake(cold, [](const Json&) {});
    MtpWireReply resync(ready, ready, cold, true); resync.finish(recovered);
    refuses([&] { MtpWireReply strict(ready, ready, cold); strict.finish(recovered); });
    for (const auto name : mtp_router_window_identity_names) {
        MtpWireReply bad(ready, ready, cold, true); refuses([&] { bad.finish(without(recovered, name)); });
    }
    refuses([&] { MtpWireReply append(ready, recovered, mtp_fixture::input(2, false), true); });
    const auto job = mtp_fixture::scope_job(generation_job(8), 1, true, 8), previous = fake.previous;
    MtpWireReply reply(ready, previous, job); const auto done = fake(job, [&](const Json& e) { reply.event(e); }); reply.finish(done);
    for (size_t i = 0; i < 4; ++i) check(!integer(done, mtp_router_window_request_names[i]), "output changed router window counters");
    for (const auto name : mtp_router_window_counter_names) {
        MtpWireReply drift(ready, previous, job); for (const auto& e : fake.events) drift.event(e);
        refuses([&] { drift.finish(with(done, name, Json::integer(0))); }); refuses([&] { static_cast<void>(drift.events()); });
    }
}
void router_window_workers(const std::filesystem::path& executable) {
    for (const auto mode : {"omitted", "off", "on"}) {
        scalar_fixture::Files files(executable, std::string("router-window-argv-") + mode); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32"; files.options.prefill_hc_rows = true;
        if (std::string_view(mode) != "omitted") files.options.prefill_router_window = std::string_view(mode) == "on";
        OwnedWorker worker(files.options); const auto done = worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; });
        check(mtp_router_window(worker.ready()) == (std::string_view(mode) == "on") && mtp_router_window(done) == mtp_router_window(worker.ready()), "router window child argv or READY differs"); worker.close();
    }
    for (const auto mode : {"router-window-ready-missing", "router-window-ready-unsolicited", "router-window-done-drift", "router-window-call-underreported"}) {
        scalar_fixture::Files files(executable, mode); prefill512_fixture::options(files); files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32"; files.options.prefill_hc_rows = true;
        files.options.prefill_router_window = std::string_view(mode) != "router-window-ready-unsolicited";
        if (std::string_view(mode).find("ready") != std::string_view::npos) { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t callbacks = 0;
        refuses([&] { worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(!callbacks && !worker.alive(), "router window drift admitted callbacks or retained worker");
    }
    for (unsigned invalid = 0; invalid < 5; ++invalid) {
        scalar_fixture::Files files(executable, "router-window-invalid"); prefill512_fixture::options(files, invalid != 0);
        files.options.prefill_dense = invalid == 1 ? "f32" : "f32-staged"; files.options.prefill_grouped = invalid == 2 ? "serial" : "f32";
        files.options.prefill_hc_rows = invalid != 4; files.options.prefill_router_window = true; files.options.prefill_hc_down_f32 = invalid == 3;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "invalid router window consent started a child");
    }
}
void router_window_sessions() {
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    for (bool after_output : {false, true}) {
        const auto ready = router_window_ready(); mtp_fixture::Fake fake; fake.prepared = fake.previous = ready; bool cancel = false;
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, ready);
        const ServeSink stopped{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
        refuses([&] { session.complete(request, "router-window-cancel", [&](const Json& job, const auto& cb) { auto done = fake(job, cb); if (!after_output || integer(job, "generate")) cancel = true; return done; }, stopped); });
        check(!session.committed_tokens(), "cancelled router window request committed HTTP state");
        const auto before = fake.router_window_encoded; cancel = false; std::vector<Json> trace;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { trace.push_back(row); }};
        const auto result = session.complete(request, "router-window-recovery", [&](const Json& job, const auto& cb) { return fake(job, cb); }, sink);
        const auto& details = *result.find("tokspur_private"); const auto& route = *trace.back().find("prefill_router_window");
        for (const auto name : mtp_router_window_identity_names) check(details.find(name)->dump() == ready.find(name)->dump() && route.find(name)->dump() == ready.find(name)->dump(), "router window HTTP identity differs");
        const auto& calls = details.find("native_prefill_router_window_admitted_encoded_calls")->array();
        check(calls.size() == 4 && calls[0].unsigned_integer() == fake.router_window_encoded - before && calls[0].dump() == calls[1].dump() && calls[1].dump() == calls[2].dump() && calls[2].dump() == calls[3].dump(), "router window cancelled work entered admitted totals");
        check(route.find("admitted_encoded_calls")->dump() == details.find("native_prefill_router_window_admitted_encoded_calls")->dump(), "router window private trace count differs");
    }
}
}
#ifndef TOKSPUR_HTTP_ROUTER_WINDOW_TEST_ENTRY
#define TOKSPUR_HTTP_ROUTER_WINDOW_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_ROUTER_WINDOW_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_router_window_entry(argc, argv); if (inherited) return inherited;
    try { router_window_metadata_and_wire(); router_window_workers(std::filesystem::canonical(argv[0])); router_window_sessions(); std::cout << "Router window HTTP CPU checks=" << checks << " pass; no model/GPU\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
