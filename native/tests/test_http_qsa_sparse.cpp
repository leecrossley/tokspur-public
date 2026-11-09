// CPU serving contract controls. Fake children never load a model or open Metal.
#define TOKSPUR_HTTP_HC_DOWN_WINDOW_TEST_ENTRY inherited_qsa_sparse_entry
#include "test_http_prefill_hc_down_window.cpp"
#undef TOKSPUR_HTTP_HC_DOWN_WINDOW_TEST_ENTRY
#ifndef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
#error Sparse QSA serving controls require the explicit overlay.
#endif
namespace {
Json qsa_sparse_ready() { return mtp_fixture::qsa_sparse_fields(hc_down_window_ready()); }
void qsa_sparse_metadata_and_wire() {
    const auto ready = qsa_sparse_ready(), omitted = hc_down_window_ready();
    validate_mtp_ready(ready, "tiled128", true, 2); validate_mtp_qsa_sparse_selection(ready, true); validate_mtp_qsa_sparse_selection(omitted, false);
    refuses([&] { validate_mtp_qsa_sparse_selection(ready, false); }); refuses([&] { validate_mtp_qsa_sparse_selection(omitted, true); });
    for (const auto& [name, value] : mtp_qsa_sparse_fields()) {
        static_cast<void>(value); refuses([&] { validate_mtp_ready(without(ready, name), "tiled128", true, 2); });
    }
    for (const auto& [name, value] : Json::Object{{"native_qsa_sparse_decode_bn", Json::integer(4)}, {"native_qsa_sparse_verify_bn", parse_json("16.0")},
        {"native_qsa_sparse_precision", Json("fp16")}, {"native_qsa_sparse_decode_threads", Json::integer(128)}, {"native_qsa_sparse_extra", Json(false)},
        {"native_qsa_sparse_decode_calls", Json::integer(1)}, {"native_prefill_dense_kernel", Json("graph_q8_mpp_wide")}})
        refuses([&] { validate_mtp_ready(with(ready, name, value), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(without(ready, "native_prefill_hc_down_window_version"), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(without(ready, "native_prefill_router_window_version"), "tiled128", true, 2); });
    mtp_fixture::Fake fake; fake.prepared = fake.previous = ready;
    for (uint32_t n : {512u, 332u, 2u, 1u}) {
        const auto job = mtp_fixture::input(n, fake.prefix == 0), previous = fake.previous, done = fake(job, [](const Json&) {});
        MtpWireReply good(ready, previous, job); good.finish(done); checks += 1;
        for (const auto name : mtp_qsa_sparse_identity_names) {
            refuses([&] { MtpWireReply bad(ready, without(previous, name), job); });
            MtpWireReply bad(ready, previous, job); refuses([&] { bad.finish(without(done, name)); });
        }
        check(integer(done, mtp_qsa_sparse_request_names[0]) == (n == 1 ? 12 : 0), "sparse QSA decode input delta differs");
        check(!integer(done, mtp_qsa_sparse_request_names[1]), "sparse QSA verify input delta differs");
        check(!integer(done, "qsa_stage_v8_request_calls") && !integer(done, "qsa_channel_request_calls"), "sparse QSA incremented legacy decode counters");
        for (size_t i = 0; i < 2; ++i) for (const auto name : {mtp_qsa_sparse_counter_names[i], mtp_qsa_sparse_request_names[i]}) {
            MtpWireReply missing(ready, previous, job); refuses([&] { missing.finish(without(done, name)); });
            MtpWireReply drift(ready, previous, job); refuses([&] { drift.finish(with(done, name, Json::integer(integer(done, name) + 1))); });
        }
    }
    const auto cold = mtp_fixture::input(332), recovered = fake(cold, [](const Json&) {});
    MtpWireReply resync(ready, ready, cold, true); resync.finish(recovered);
    refuses([&] { MtpWireReply strict(ready, ready, cold); strict.finish(recovered); });
    for (const auto name : mtp_qsa_sparse_identity_names) {
        MtpWireReply bad(ready, ready, cold, true); refuses([&] { bad.finish(without(recovered, name)); });
    }
    const auto job = mtp_fixture::scope_job(generation_job(8), 1, true, 8), previous = fake.previous;
    MtpWireReply reply(ready, previous, job); const auto done = fake(job, [&](const Json& e) { reply.event(e); }); reply.finish(done);
    check(integer(done, mtp_qsa_sparse_request_names[1]) == 12ull * integer(done, "native_mtp_rounds"), "sparse QSA verify round delta differs");
    check(!integer(done, "qsa_stage_v8_request_calls") && !integer(done, "qsa_channel_request_calls"), "sparse QSA output incremented legacy counters");
    {
        MtpWireReply drift(ready, previous, job); for (const auto& e : fake.events) drift.event(e);
        refuses([&] { drift.finish(with(done, mtp_qsa_sparse_counter_names[1], Json::integer(0))); }); refuses([&] { static_cast<void>(drift.events()); });
    }
}
void qsa_sparse_workers(const std::filesystem::path& executable) {
    for (const auto mode : {"omitted", "off", "on"}) {
        scalar_fixture::Files files(executable, std::string("qsa-sparse-argv-") + mode); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32";
        files.options.prefill_hc_rows = true; files.options.prefill_router_window = true; files.options.prefill_hc_down_window = true;
        if (std::string_view(mode) != "omitted") files.options.qsa_sparse = std::string_view(mode) == "on";
        OwnedWorker worker(files.options);
        const auto first = worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; });
        const auto single = worker.request(mtp_fixture::input(1, false), Clock::now() + 3s, [] { return false; });
        check(mtp_qsa_sparse(worker.ready()) == (std::string_view(mode) == "on") && mtp_qsa_sparse(first) == mtp_qsa_sparse(worker.ready()) &&
            mtp_qsa_sparse(single) == mtp_qsa_sparse(worker.ready()), "sparse QSA child argv or READY differs");
        worker.close();
    }
    for (const auto mode : {"qsa-sparse-ready-missing", "qsa-sparse-ready-unsolicited", "qsa-sparse-done-drift", "qsa-sparse-call-underreported"}) {
        scalar_fixture::Files files(executable, mode); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32";
        files.options.prefill_hc_rows = true; files.options.prefill_router_window = true; files.options.prefill_hc_down_window = true;
        files.options.qsa_sparse = std::string_view(mode) != "qsa-sparse-ready-unsolicited";
        if (std::string_view(mode).find("ready") != std::string_view::npos) { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t callbacks = 0;
        if (std::string_view(mode) == "qsa-sparse-call-underreported")
            static_cast<void>(worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; }));
        refuses([&] {
            worker.request(std::string_view(mode) == "qsa-sparse-call-underreported" ? mtp_fixture::input(1, false) : mtp_fixture::input(332),
                Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; });
        });
        check(!callbacks && !worker.alive(), "sparse QSA drift admitted callbacks or retained worker");
    }
    for (unsigned invalid = 0; invalid < 7; ++invalid) {
        scalar_fixture::Files files(executable, "qsa-sparse-invalid"); prefill512_fixture::options(files, invalid != 0);
        files.options.prefill_dense = invalid == 1 ? "f32" : "f32-staged"; files.options.prefill_grouped = invalid == 2 ? "serial" : "f32";
        files.options.prefill_hc_rows = invalid != 4; files.options.prefill_router_window = invalid != 5; files.options.prefill_hc_down_window = invalid != 6;
        files.options.qsa_sparse = true; files.options.prefill_hc_down_f32 = invalid == 3;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "invalid sparse QSA consent started a child");
    }
#ifndef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    for (bool requested : {false, true}) {
        scalar_fixture::Files files(executable, "gdn-ported-uncompiled"); prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged"; files.options.prefill_grouped = "f32";
        files.options.prefill_hc_rows = true; files.options.prefill_router_window = true; files.options.prefill_hc_down_window = true;
        files.options.qsa_sparse = true; files.options.gdn_ported = requested;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "uncompiled ported GDN consent started a child");
    }
    refuses([&] { validate_mtp_gdn_ported_selection(mtp_fixture::gdn_ported_fields(qsa_sparse_ready()), true); });
#endif
}
void qsa_sparse_sessions() {
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    for (bool after_output : {false, true}) {
        const auto ready = qsa_sparse_ready(); mtp_fixture::Fake fake; fake.prepared = fake.previous = ready; bool cancel = false;
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, ready);
        const ServeSink stopped{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
        refuses([&] { session.complete(request, "qsa-sparse-cancel", [&](const Json& job, const auto& cb) { auto done = fake(job, cb); if (!after_output || integer(job, "generate")) cancel = true; return done; }, stopped); });
        check(!session.committed_tokens(), "cancelled sparse QSA request committed HTTP state");
        const auto before = fake.qsa_sparse_completed; cancel = false; std::vector<Json> trace;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { trace.push_back(row); }};
        const auto result = session.complete(request, "qsa-sparse-recovery", [&](const Json& job, const auto& cb) { return fake(job, cb); }, sink);
        const auto& details = *result.find("tokspur_private"); const auto& route = *trace.back().find("qsa_sparse");
        for (const auto name : mtp_qsa_sparse_identity_names) check(details.find(name)->dump() == ready.find(name)->dump() && route.find(name)->dump() == ready.find(name)->dump(), "sparse QSA HTTP identity differs");
        const auto& calls = details.find("native_qsa_sparse_admitted_completed_calls")->array();
        check(calls.size() == 2 && calls[0].unsigned_integer() == fake.qsa_sparse_completed[0] - before[0] &&
            calls[1].unsigned_integer() == fake.qsa_sparse_completed[1] - before[1], "sparse QSA cancelled work entered admitted totals");
        check(route.find("admitted_completed_calls")->dump() == details.find("native_qsa_sparse_admitted_completed_calls")->dump(), "sparse QSA private trace count differs");
    }
}
}
#ifndef TOKSPUR_HTTP_QSA_SPARSE_TEST_ENTRY
#define TOKSPUR_HTTP_QSA_SPARSE_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_QSA_SPARSE_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_qsa_sparse_entry(argc, argv); if (inherited) return inherited;
    try { qsa_sparse_metadata_and_wire(); qsa_sparse_workers(std::filesystem::canonical(argv[0])); qsa_sparse_sessions(); std::cout << "Sparse QSA HTTP CPU checks=" << checks << " pass; no model/GPU\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
