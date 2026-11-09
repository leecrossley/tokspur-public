// CPU serving contract controls. Fake children never load a model or open Metal.
#define TOKSPUR_HTTP_ROUTER_WINDOW_TEST_ENTRY inherited_hc_down_window_entry
#include "test_http_prefill_router_window.cpp"
#undef TOKSPUR_HTTP_ROUTER_WINDOW_TEST_ENTRY
#ifndef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
#error HC down window serving controls require the explicit overlay.
#endif
namespace {
Json hc_down_window_ready() { return mtp_fixture::hc_down_window_fields(router_window_ready()); }
void hc_down_window_metadata_and_wire() {
    const auto ready = hc_down_window_ready(), omitted = router_window_ready();
    validate_mtp_ready(ready, "tiled128", true, 2); validate_mtp_hc_down_window_selection(ready, true); validate_mtp_hc_down_window_selection(omitted, false);
    refuses([&] { validate_mtp_hc_down_window_selection(ready, false); }); refuses([&] { validate_mtp_hc_down_window_selection(omitted, true); });
    for (const auto& [name, value] : mtp_hc_down_window_fields()) {
        static_cast<void>(value); refuses([&] { validate_mtp_ready(without(ready, name), "tiled128", true, 2); });
    }
    for (const auto& [name, value] : Json::Object{{"native_prefill_hc_down_window_min_rows", Json::integer(1)}, {"native_prefill_hc_down_window_max_rows", parse_json("512.0")},
        {"native_prefill_hc_down_window_precision", Json("fp16")}, {"native_prefill_hc_down_window_threadgroup_bytes", Json::integer(16384)}, {"native_prefill_hc_down_window_extra", Json(false)},
        {"native_prefill_hc_down_window_encoded", Json::integer(1)}, {"native_prefill_dense_kernel", Json("graph_q8_mpp_wide")}})
        refuses([&] { validate_mtp_ready(with(ready, name, value), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(without(ready, "native_prefill_hc_rows_version"), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(without(ready, "native_prefill_router_window_version"), "tiled128", true, 2); });
    mtp_fixture::Fake fake; fake.prepared = fake.previous = ready;
    for (uint32_t n : {512u, 332u, 18u, 490u, 2u, 3u, 511u, 1u}) {
        const auto job = mtp_fixture::input(n, fake.prefix == 0), previous = fake.previous, done = fake(job, [](const Json&) {});
        MtpWireReply good(ready, previous, job); good.finish(done); checks += 1;
        for (const auto name : mtp_hc_down_window_identity_names) {
            refuses([&] { MtpWireReply bad(ready, without(previous, name), job); });
            MtpWireReply bad(ready, previous, job); refuses([&] { bad.finish(without(done, name)); });
        }
        check(integer(done, mtp_hc_down_window_request_name) == (n > 1 ? 96 : 0), "HC down window partial-row delta differs");
        for (const auto name : {mtp_hc_down_window_counter_name, mtp_hc_down_window_request_name}) {
            MtpWireReply missing(ready, previous, job); refuses([&] { missing.finish(without(done, name)); });
            MtpWireReply drift(ready, previous, job); refuses([&] { drift.finish(with(done, name, Json::integer(integer(done, name) + 1))); });
        }
    }
    const auto cold = mtp_fixture::input(332), recovered = fake(cold, [](const Json&) {});
    MtpWireReply resync(ready, ready, cold, true); resync.finish(recovered);
    refuses([&] { MtpWireReply strict(ready, ready, cold); strict.finish(recovered); });
    for (const auto name : mtp_hc_down_window_identity_names) {
        MtpWireReply bad(ready, ready, cold, true); refuses([&] { bad.finish(without(recovered, name)); });
    }
    refuses([&] { MtpWireReply append(ready, recovered, mtp_fixture::input(2, false), true); });
    const auto job = mtp_fixture::scope_job(generation_job(8), 1, true, 8), previous = fake.previous;
    MtpWireReply reply(ready, previous, job); const auto done = fake(job, [&](const Json& e) { reply.event(e); }); reply.finish(done);
    check(!integer(done, mtp_hc_down_window_request_name), "output changed HC down window counters");
    {
        MtpWireReply drift(ready, previous, job); for (const auto& e : fake.events) drift.event(e);
        refuses([&] { drift.finish(with(done, mtp_hc_down_window_counter_name, Json::integer(0))); }); refuses([&] { static_cast<void>(drift.events()); });
    }
}
void hc_down_window_workers(const std::filesystem::path& executable) {
    for (const auto mode : {"omitted", "off", "on"}) {
        scalar_fixture::Files files(executable, std::string("hc-down-window-argv-") + mode); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32"; files.options.prefill_hc_rows = true; files.options.prefill_router_window = true;
        if (std::string_view(mode) != "omitted") files.options.prefill_hc_down_window = std::string_view(mode) == "on";
        OwnedWorker worker(files.options); const auto done = worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; });
        check(mtp_hc_down_window(worker.ready()) == (std::string_view(mode) == "on") && mtp_hc_down_window(done) == mtp_hc_down_window(worker.ready()), "HC down window child argv or READY differs"); worker.close();
    }
    for (const auto mode : {"hc-down-window-ready-missing", "hc-down-window-ready-unsolicited", "hc-down-window-done-drift", "hc-down-window-call-underreported"}) {
        scalar_fixture::Files files(executable, mode); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32"; files.options.prefill_hc_rows = true; files.options.prefill_router_window = true;
        files.options.prefill_hc_down_window = std::string_view(mode) != "hc-down-window-ready-unsolicited";
        if (std::string_view(mode).find("ready") != std::string_view::npos) { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t callbacks = 0;
        refuses([&] { worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(!callbacks && !worker.alive(), "HC down window drift admitted callbacks or retained worker");
    }
    for (unsigned invalid = 0; invalid < 6; ++invalid) {
        scalar_fixture::Files files(executable, "hc-down-window-invalid"); prefill512_fixture::options(files, invalid != 0);
        files.options.prefill_dense = invalid == 1 ? "f32" : "f32-staged"; files.options.prefill_grouped = invalid == 2 ? "serial" : "f32";
        files.options.prefill_hc_rows = invalid != 4; files.options.prefill_router_window = invalid != 5;
        files.options.prefill_hc_down_window = true; files.options.prefill_hc_down_f32 = invalid == 3;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "invalid HC down window consent started a child");
    }
}
void hc_down_window_sessions() {
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    for (bool after_output : {false, true}) {
        const auto ready = hc_down_window_ready(); mtp_fixture::Fake fake; fake.prepared = fake.previous = ready; bool cancel = false;
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, ready);
        const ServeSink stopped{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
        refuses([&] { session.complete(request, "hc-down-window-cancel", [&](const Json& job, const auto& cb) { auto done = fake(job, cb); if (!after_output || integer(job, "generate")) cancel = true; return done; }, stopped); });
        check(!session.committed_tokens(), "cancelled HC down window request committed HTTP state");
        const auto before = fake.hc_down_window_encoded; cancel = false; std::vector<Json> trace;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { trace.push_back(row); }};
        const auto result = session.complete(request, "hc-down-window-recovery", [&](const Json& job, const auto& cb) { return fake(job, cb); }, sink);
        const auto& details = *result.find("tokspur_private"); const auto& route = *trace.back().find("prefill_hc_down_window");
        for (const auto name : mtp_hc_down_window_identity_names) check(details.find(name)->dump() == ready.find(name)->dump() && route.find(name)->dump() == ready.find(name)->dump(), "HC down window HTTP identity differs");
        const auto admitted = details.find("native_prefill_hc_down_window_admitted_encoded_calls")->unsigned_integer();
        check(admitted == fake.hc_down_window_encoded - before, "HC down window cancelled work entered admitted totals");
        check(route.find("admitted_encoded_calls")->dump() == details.find("native_prefill_hc_down_window_admitted_encoded_calls")->dump(), "HC down window private trace count differs");
    }
}
}
#ifndef TOKSPUR_HTTP_HC_DOWN_WINDOW_TEST_ENTRY
#define TOKSPUR_HTTP_HC_DOWN_WINDOW_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_HC_DOWN_WINDOW_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_hc_down_window_entry(argc, argv); if (inherited) return inherited;
    try { hc_down_window_metadata_and_wire(); hc_down_window_workers(std::filesystem::canonical(argv[0])); hc_down_window_sessions(); std::cout << "HC down window HTTP CPU checks=" << checks << " pass; no model/GPU\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
