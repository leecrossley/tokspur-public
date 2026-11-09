// CPU serving contract controls. Fake children never load a model or open Metal.
#define TOKSPUR_HTTP_PREFILL512_TEST_ENTRY inherited_hc_rows_entry
#include "test_http_prefill512.cpp"
#undef TOKSPUR_HTTP_PREFILL512_TEST_ENTRY
#ifndef TOKSPUR_PRIVATE_HTTP_HC_ROWS
#error HC row serving controls require the explicit overlay.
#endif
namespace {
Json rows_ready() { return mtp_fixture::hc_rows_fields(mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), "f32-staged", "f32")); }
void rows_metadata_and_wire() {
    const auto ready = rows_ready(), omitted = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), "f32-staged", "f32");
    validate_mtp_ready(ready, "tiled128", true, 2); validate_mtp_hc_rows_selection(ready, true); validate_mtp_hc_rows_selection(omitted, false);
    refuses([&] { validate_mtp_hc_rows_selection(ready, false); }); refuses([&] { validate_mtp_hc_rows_selection(omitted, true); });
    for (const auto& [name, value] : mtp_hc_rows_fields()) {
        static_cast<void>(value); refuses([&] { validate_mtp_ready(without(ready, name), "tiled128", true, 2); });
    }
    for (const auto& [name, value] : Json::Object{{"native_prefill_hc_rows_min_rows", Json::integer(1)}, {"native_prefill_hc_rows_max_rows", parse_json("512.0")},
        {"native_prefill_hc_rows_precision", Json("fp16")}, {"native_prefill_hc_rows_scratch_bytes", Json::integer(0)}, {"native_prefill_hc_rows_extra", Json(false)},
        {"native_prefill_hc_rows_inject_encoded", Json::integer(1)}, {"native_prefill_dense_kernel", Json("graph_q8_mpp_wide")}})
        refuses([&] { validate_mtp_ready(with(ready, name, value), "tiled128", true, 2); });
    mtp_fixture::Fake fake; fake.prepared = fake.previous = ready;
    for (uint32_t n : {512u, 332u, 18u, 490u, 2u, 3u, 511u, 1u}) {
        const auto job = mtp_fixture::input(n, fake.prefix == 0), previous = fake.previous, done = fake(job, [](const Json&) {});
        MtpWireReply good(ready, previous, job); good.finish(done); checks += 1;
        for (const auto name : mtp_hc_rows_identity_names) {
            refuses([&] { MtpWireReply bad(ready, without(previous, name), job); });
            MtpWireReply bad(ready, previous, job); refuses([&] { bad.finish(without(done, name)); });
        }
        for (size_t i = 0; i < 3; ++i) {
            check(integer(done, mtp_hc_rows_request_names[i]) == (n > 1 ? 96 : 0), "HC partial-row delta differs");
            for (const auto name : {mtp_hc_rows_counter_names[i], mtp_hc_rows_request_names[i]}) {
                MtpWireReply missing(ready, previous, job); refuses([&] { missing.finish(without(done, name)); });
                MtpWireReply drift(ready, previous, job); refuses([&] { drift.finish(with(done, name, Json::integer(integer(done, name) + 1))); });
            }
        }
    }
    const auto cold = mtp_fixture::input(332), recovered = fake(cold, [](const Json&) {});
    MtpWireReply resync(ready, ready, cold, true); resync.finish(recovered);
    refuses([&] { MtpWireReply strict(ready, ready, cold); strict.finish(recovered); });
    for (const auto name : mtp_hc_rows_identity_names) {
        MtpWireReply bad(ready, ready, cold, true); refuses([&] { bad.finish(without(recovered, name)); });
    }
    refuses([&] { MtpWireReply append(ready, recovered, mtp_fixture::input(2, false), true); });
    const auto job = mtp_fixture::scope_job(generation_job(8), 1, true, 8), previous = fake.previous;
    MtpWireReply reply(ready, previous, job); const auto done = fake(job, [&](const Json& e) { reply.event(e); }); reply.finish(done);
    for (size_t i = 0; i < 3; ++i) check(!integer(done, mtp_hc_rows_request_names[i]), "output changed HC row counters");
    for (const auto name : mtp_hc_rows_counter_names) {
        MtpWireReply drift(ready, previous, job); for (const auto& e : fake.events) drift.event(e);
        refuses([&] { drift.finish(with(done, name, Json::integer(0))); }); refuses([&] { static_cast<void>(drift.events()); });
    }
}
void rows_workers(const std::filesystem::path& executable) {
    for (const auto mode : {"omitted", "off", "on"}) {
        scalar_fixture::Files files(executable, std::string("hc-rows-argv-") + mode); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32";
        if (std::string_view(mode) != "omitted") files.options.prefill_hc_rows = std::string_view(mode) == "on";
        OwnedWorker worker(files.options); const auto done = worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; });
        check(mtp_hc_rows(worker.ready()) == (std::string_view(mode) == "on") && mtp_hc_rows(done) == mtp_hc_rows(worker.ready()), "HC child argv or READY differs"); worker.close();
    }
    for (const auto mode : {"hc-rows-ready-missing", "hc-rows-ready-unsolicited", "hc-rows-done-drift", "hc-rows-call-underreported"}) {
        scalar_fixture::Files files(executable, mode); prefill512_fixture::options(files); files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32";
        files.options.prefill_hc_rows = std::string_view(mode) != "hc-rows-ready-unsolicited";
        if (std::string_view(mode).find("ready") != std::string_view::npos) { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t callbacks = 0;
        refuses([&] { worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(!callbacks && !worker.alive(), "HC drift admitted callbacks or retained worker");
    }
    for (unsigned invalid = 0; invalid < 4; ++invalid) {
        scalar_fixture::Files files(executable, "hc-rows-invalid"); prefill512_fixture::options(files, invalid != 0);
        files.options.prefill_dense = invalid == 1 ? "f32" : "f32-staged"; files.options.prefill_grouped = invalid == 2 ? "serial" : "f32";
        files.options.prefill_hc_rows = true; files.options.prefill_hc_down_f32 = invalid == 3;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "invalid HC row consent started a child");
    }
}
void rows_sessions() {
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    for (bool after_output : {false, true}) {
        const auto ready = rows_ready(); mtp_fixture::Fake fake; fake.prepared = fake.previous = ready; bool cancel = false;
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, ready);
        const ServeSink stopped{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
        refuses([&] { session.complete(request, "rows-cancel", [&](const Json& job, const auto& cb) { auto done = fake(job, cb); if (!after_output || integer(job, "generate")) cancel = true; return done; }, stopped); });
        check(!session.committed_tokens(), "cancelled HC request committed HTTP state");
        const auto before = fake.hc_rows_encoded; cancel = false; std::vector<Json> trace;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { trace.push_back(row); }};
        const auto result = session.complete(request, "rows-recovery", [&](const Json& job, const auto& cb) { return fake(job, cb); }, sink);
        const auto& details = *result.find("tokspur_private"); const auto& route = *trace.back().find("prefill_hc_rows");
        for (const auto name : mtp_hc_rows_identity_names) check(details.find(name)->dump() == ready.find(name)->dump() && route.find(name)->dump() == ready.find(name)->dump(), "HC HTTP identity differs");
        const auto& calls = details.find("native_prefill_hc_rows_admitted_encoded_calls")->array();
        check(calls.size() == 3 && calls[0].unsigned_integer() == fake.hc_rows_encoded - before && calls[0].dump() == calls[1].dump() && calls[1].dump() == calls[2].dump(), "HC cancelled work entered admitted totals");
        check(route.find("admitted_encoded_calls")->dump() == details.find("native_prefill_hc_rows_admitted_encoded_calls")->dump(), "HC private trace count differs");
    }
}
}
#ifndef TOKSPUR_HTTP_HC_ROWS_TEST_ENTRY
#define TOKSPUR_HTTP_HC_ROWS_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_HC_ROWS_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_hc_rows_entry(argc, argv); if (inherited) return inherited;
    try { rows_metadata_and_wire(); rows_workers(std::filesystem::canonical(argv[0])); rows_sessions(); std::cout << "HC rows HTTP CPU checks=" << checks << " pass; no model/GPU\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
