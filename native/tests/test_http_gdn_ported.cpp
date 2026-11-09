// CPU serving contract controls. Fake children never load a model or open Metal.
#define TOKSPUR_HTTP_QSA_SPARSE_TEST_ENTRY inherited_gdn_ported_entry
#include "test_http_qsa_sparse.cpp"
#undef TOKSPUR_HTTP_QSA_SPARSE_TEST_ENTRY
#ifndef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
#error Ported GDN serving controls require the explicit overlay.
#endif
namespace {
Json gdn_ported_ready(bool with_qsa_sparse = true) {
    return mtp_fixture::gdn_ported_fields(with_qsa_sparse ? qsa_sparse_ready() : hc_down_window_ready());
}
void gdn_ported_metadata_and_wire() {
    for (bool with_qsa_sparse : {true, false}) {
        const auto base = with_qsa_sparse ? qsa_sparse_ready() : hc_down_window_ready();
        const auto ready = gdn_ported_ready(with_qsa_sparse);
        validate_mtp_ready(ready, "tiled128", true, 2);
        validate_mtp_gdn_ported_selection(ready, true);
        validate_mtp_gdn_ported_selection(base, false);
        validate_mtp_qsa_sparse_selection(ready, with_qsa_sparse);
        refuses([&] { validate_mtp_gdn_ported_selection(ready, false); });
        refuses([&] { validate_mtp_gdn_ported_selection(base, true); });
        for (const auto& [name, value] : mtp_gdn_ported_fields()) {
            static_cast<void>(value);
            refuses([&] { validate_mtp_ready(without(ready, name), "tiled128", true, 2); });
        }
        for (const auto& [name, value] : Json::Object{
                 {"native_gdn_ported_prefill_tile_rows", Json::integer(16)},
                 {"native_gdn_ported_verify_db", Json::integer(TOKSPUR_GDN_VERIFY_DB == 8 ? 16 : 8)},
                 {"native_gdn_ported_verify_db", parse_json("8.0")},
                 {"native_gdn_ported_precision", Json("fp16")},
                 {"native_gdn_ported_prefill_threadgroup_bytes", Json::integer(12288)},
                 {"native_gdn_ported_verify_threadgroup_bytes", Json::integer(4096)},
                 {"native_gdn_ported_prefill_kernel", Json("graph_gdn_chunk_fused")},
                 {"native_gdn_ported_verify_kernel", Json("graph_gdn_delta_tile")},
                 {"native_gdn_ported_extra", Json(false)},
                 {"native_gdn_ported_prefill_calls", Json::integer(1)},
                 {"native_prefill_dense_kernel", Json("graph_q8_mpp_wide")}}) {
            refuses([&] { validate_mtp_ready(with(ready, name, value), "tiled128", true, 2); });
        }
        refuses([&] { validate_mtp_ready(without(ready, "native_prefill_hc_down_window_version"), "tiled128", true, 2); });
        refuses([&] { validate_mtp_ready(without(ready, "native_prefill_router_window_version"), "tiled128", true, 2); });

        mtp_fixture::Fake fake;
        fake.prepared = fake.previous = ready;
        for (uint32_t n : {512u, 332u, 2u, 1u}) {
            const auto job = mtp_fixture::input(n, fake.prefix == 0), previous = fake.previous;
            const auto done = fake(job, [](const Json&) {});
            MtpWireReply good(ready, previous, job);
            good.finish(done);
            checks += 1;
            for (const auto name : mtp_gdn_ported_identity_names) {
                refuses([&] { MtpWireReply bad(ready, without(previous, name), job); });
                MtpWireReply bad(ready, previous, job);
                refuses([&] { bad.finish(without(done, name)); });
            }
            const std::array<uint64_t, 5> expected_req{
                n >= 2 ? 36ull : 0ull, n == 1 ? 36ull : 0ull, 0ull, 0ull, n == 1 ? 36ull : 0ull};
            for (size_t i = 0; i < 5; ++i) {
                check(integer(done, mtp_gdn_ported_request_names[i]) == expected_req[i],
                    "ported GDN input packet request counter differs");
                for (const auto name : {mtp_gdn_ported_counter_names[i], mtp_gdn_ported_request_names[i]}) {
                    MtpWireReply missing(ready, previous, job);
                    refuses([&] { missing.finish(without(done, name)); });
                    MtpWireReply drift(ready, previous, job);
                    refuses([&] { drift.finish(with(done, name, Json::integer(integer(done, name) + 1))); });
                }
            }
        }

        const auto cold = mtp_fixture::input(332), recovered = fake(cold, [](const Json&) {});
        MtpWireReply resync(ready, ready, cold, true);
        resync.finish(recovered);
        refuses([&] {
            MtpWireReply strict(ready, ready, cold);
            strict.finish(recovered);
        });
        for (const auto name : mtp_gdn_ported_identity_names) {
            MtpWireReply bad(ready, ready, cold, true);
            refuses([&] { bad.finish(without(recovered, name)); });
        }

        uint64_t gen_id = 1;
        for (uint32_t force_accepted : {2u, 1u, 0u}) {
            fake.force_accepted = force_accepted;
            const auto job = mtp_fixture::scope_job(generation_job(8), gen_id++, true, 8), previous = fake.previous;
            MtpWireReply reply(ready, previous, job);
            const auto done = fake(job, [&](const Json& e) { reply.event(e); });
            reply.finish(done);
            const uint64_t rounds = integer(done, "native_mtp_rounds") - integer(previous, "native_mtp_rounds");
            const uint64_t rejected = integer(done, "native_mtp_recovery_rounds") - integer(previous, "native_mtp_recovery_rounds");
            const uint64_t accepted_tok = integer(done, "native_mtp_accepted") - integer(previous, "native_mtp_accepted");
            const uint64_t primes = integer(done, "native_mtp_prime_calls") - integer(previous, "native_mtp_prime_calls");
            const uint64_t ordinary = primes - accepted_tok;
            check(integer(done, mtp_gdn_ported_request_names[0]) == 0, "ported GDN output prefill delta differs");
            check(integer(done, mtp_gdn_ported_request_names[1]) == 36ull * ordinary, "ported GDN output decode delta differs");
            check(integer(done, mtp_gdn_ported_request_names[2]) == 36ull * rounds, "ported GDN output verify delta differs");
            check(integer(done, mtp_gdn_ported_request_names[3]) == 36ull * rejected, "ported GDN output replay delta differs");
            check(integer(done, mtp_gdn_ported_request_names[4]) == 36ull * (ordinary + rounds), "ported GDN output norm_gate delta differs");
            if (force_accepted < 2u) check(rejected > 0, "ported GDN rejected test did not exercise replay accounting");
            {
                MtpWireReply drift(ready, previous, job);
                for (const auto& e : fake.events) drift.event(e);
                refuses([&] { drift.finish(with(done, mtp_gdn_ported_counter_names[2], Json::integer(0))); });
                refuses([&] { static_cast<void>(drift.events()); });
            }
        }
    }
}
void gdn_ported_workers(const std::filesystem::path& executable) {
    for (bool qsa_on : {false, true}) {
        for (const auto mode : {"omitted", "off", "on"}) {
            scalar_fixture::Files files(executable, std::string("gdn-ported-argv-") + (qsa_on ? "qsa-" : "noqsa-") + mode);
            prefill512_fixture::options(files);
            files.options.prefill_dense = "f32-staged";
            files.options.prefill_grouped = "f32";
            files.options.prefill_hc_rows = true;
            files.options.prefill_router_window = true;
            files.options.prefill_hc_down_window = true;
            files.options.qsa_sparse = qsa_on;
            if (std::string_view(mode) != "omitted") files.options.gdn_ported = std::string_view(mode) == "on";
            OwnedWorker worker(files.options);
            const auto first = worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; });
            const auto single = worker.request(mtp_fixture::input(1, false), Clock::now() + 3s, [] { return false; });
            const bool expected = std::string_view(mode) == "on";
            check(mtp_gdn_ported(worker.ready()) == expected && mtp_gdn_ported(first) == expected &&
                mtp_gdn_ported(single) == expected && mtp_qsa_sparse(worker.ready()) == qsa_on,
                "ported GDN child argv or READY differs");
            worker.close();
        }
    }
    for (const auto mode : {"gdn-ported-ready-missing", "gdn-ported-ready-unsolicited", "gdn-ported-done-drift", "gdn-ported-call-underreported"}) {
        scalar_fixture::Files files(executable, mode);
        prefill512_fixture::options(files);
        files.options.prefill_dense = "f32-staged";
        files.options.prefill_grouped = "f32";
        files.options.prefill_hc_rows = true;
        files.options.prefill_router_window = true;
        files.options.prefill_hc_down_window = true;
        files.options.qsa_sparse = true;
        files.options.gdn_ported = std::string_view(mode) != "gdn-ported-ready-unsolicited";
        if (std::string_view(mode).find("ready") != std::string_view::npos) {
            refuses([&] { OwnedWorker bad(files.options); });
            continue;
        }
        OwnedWorker worker(files.options);
        size_t callbacks = 0;
        if (std::string_view(mode) == "gdn-ported-call-underreported")
            static_cast<void>(worker.request(mtp_fixture::input(332), Clock::now() + 3s, [] { return false; }));
        refuses([&] {
            worker.request(std::string_view(mode) == "gdn-ported-call-underreported" ? mtp_fixture::input(1, false) : mtp_fixture::input(332),
                Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; });
        });
        check(!callbacks && !worker.alive(), "ported GDN drift admitted callbacks or retained worker");
    }
    for (unsigned invalid = 0; invalid < 7; ++invalid) {
        scalar_fixture::Files files(executable, "gdn-ported-invalid");
        prefill512_fixture::options(files, invalid != 0);
        files.options.prefill_dense = invalid == 1 ? "f32" : "f32-staged";
        files.options.prefill_grouped = invalid == 2 ? "serial" : "f32";
        files.options.prefill_hc_rows = invalid != 4;
        files.options.prefill_router_window = invalid != 5;
        files.options.prefill_hc_down_window = invalid != 6;
        files.options.gdn_ported = true;
        files.options.prefill_hc_down_f32 = invalid == 3;
        refuses([&] { OwnedWorker bad(files.options); });
        check(!std::filesystem::exists(files.directory / "started"), "invalid ported GDN consent started a child");
    }
}
void gdn_ported_sessions() {
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    for (bool after_output : {false, true}) {
        const auto ready = gdn_ported_ready(true);
        mtp_fixture::Fake fake;
        fake.prepared = fake.previous = ready;
        fake.force_accepted = 1;
        bool cancel = false;
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, ready);
        const ServeSink stopped{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
        refuses([&] {
            session.complete(request, "gdn-ported-cancel", [&](const Json& job, const auto& cb) {
                auto done = fake(job, cb);
                if (!after_output || integer(job, "generate")) cancel = true;
                return done;
            }, stopped);
        });
        check(!session.committed_tokens(), "cancelled ported GDN request committed HTTP state");
        const auto before = fake.gdn_ported_completed;
        cancel = false;
        std::vector<Json> trace;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { trace.push_back(row); }};
        const auto result = session.complete(request, "gdn-ported-recovery", [&](const Json& job, const auto& cb) { return fake(job, cb); }, sink);
        const auto& details = *result.find("tokspur_private");
        const auto& route = *trace.back().find("gdn_ported");
        for (const auto name : mtp_gdn_ported_identity_names) {
            check(details.find(name)->dump() == ready.find(name)->dump() && route.find(name)->dump() == ready.find(name)->dump(),
                "ported GDN HTTP identity differs");
        }
        const auto& calls = details.find("native_gdn_ported_admitted_completed_calls")->array();
        check(calls.size() == 5, "ported GDN admitted completed calls size differs");
        for (size_t i = 0; i < 5; ++i) {
            check(calls[i].unsigned_integer() == fake.gdn_ported_completed[i] - before[i],
                "ported GDN cancelled work entered admitted totals");
        }
        check(calls[0].unsigned_integer() > 0 && calls[2].unsigned_integer() > 0 && calls[3].unsigned_integer() > 0 &&
            calls[4].unsigned_integer() == calls[1].unsigned_integer() + calls[2].unsigned_integer(),
            "ported GDN session recovery did not exercise prefill, verify, replay and norm_gate counters");
        check(route.find("admitted_completed_calls")->dump() == details.find("native_gdn_ported_admitted_completed_calls")->dump(),
            "ported GDN private trace count differs");
    }
}
}
#ifndef TOKSPUR_HTTP_GDN_PORTED_TEST_ENTRY
#define TOKSPUR_HTTP_GDN_PORTED_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_GDN_PORTED_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_gdn_ported_entry(argc, argv);
    if (inherited) return inherited;
    try {
        gdn_ported_metadata_and_wire();
        gdn_ported_workers(std::filesystem::canonical(argv[0]));
        gdn_ported_sessions();
        std::cout << "Ported GDN HTTP CPU checks=" << checks << " pass; no model/GPU\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
