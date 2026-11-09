// Generated CPU protocol controls. No model, tokenizer or Metal is opened.
#define TOKSPUR_MTP_WIRE_TEST_ENTRY inherited_mtp_wire_entry
#include "test_mtp_wire.cpp"
#undef TOKSPUR_MTP_WIRE_TEST_ENTRY

#ifndef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
#error This fixture needs the explicit current-MTP concurrent HTTP carrier.
#endif

namespace {
Json concurrent_fields(Json value, bool on) {
    for (const auto& [name, field] : Json::Object{{"concurrent_dispatch_schema", Json::integer(1)},
        {"concurrent_dispatch_mode", Json(on ? "on" : "off")}, {"concurrent_dispatch_barriers", Json::integer(0)}, {"concurrent_state_proof", Json(false)}})
        value = with(value, name, field);
    return value;
}
void metadata() {
    const auto absent = mtp_fixture::ready();
    validate_owned_concurrent_ready(absent, std::nullopt); ++checks;
    for (const bool mode : {false, true}) {
        const auto ready = concurrent_fields(absent, mode);
        validate_owned_concurrent_ready(ready, mode); ++checks;
        validate_owned_concurrent_done(ready, with(ready, "concurrent_dispatch_barriers", Json::integer(999))); ++checks;
        refuses([&] { validate_owned_concurrent_ready(ready, std::nullopt); });
        refuses([&] { validate_owned_concurrent_ready(absent, mode); });
        refuses([&] { validate_owned_concurrent_ready(ready, !mode); });
        for (const auto name : {"concurrent_dispatch_schema", "concurrent_dispatch_mode", "concurrent_dispatch_barriers", "concurrent_state_proof"}) {
            refuses([&] { validate_owned_concurrent_ready(without(ready, name), mode); });
            refuses([&] { validate_owned_concurrent_done(ready, without(ready, name)); });
        }
        for (const auto& [name, value] : Json::Object{{"concurrent_dispatch_schema", Json::integer(2)},
            {"concurrent_dispatch_schema", parse_json("1.0")}, {"concurrent_dispatch_barriers", parse_json("-1")},
            {"concurrent_dispatch_barriers", Json::integer(1)}, {"concurrent_state_proof", Json(true)},
            {"concurrent_dispatch_mode", Json(true)}, {"concurrent_unregistered", Json(false)}})
            refuses([&] { validate_owned_concurrent_ready(with(ready, name, value), mode); });
        refuses([&] { validate_owned_concurrent_done(ready, concurrent_fields(absent, !mode)); });
        refuses([&] { validate_owned_concurrent_done(absent, ready); });
    }
}
void workers(const std::filesystem::path& executable) {
    for (const auto name : {"concurrent-on", "concurrent-off", "concurrent-unsolicited", "concurrent-missing-ready", "concurrent-proof-ready",
        "concurrent-drift", "concurrent-missing-done", "concurrent-unknown-done"}) {
        scalar_fixture::Files files(executable, name);
        files.options.mtp_sidecar = files.directory / "mtp-sidecar";
        { std::ofstream sidecar(files.options.mtp_sidecar); sidecar << "generated fixture\n"; }
        files.options.mtp_depth = 2; files.options.mtp_request_carry = true; files.options.mtp_draft_attention = "tiled128";
        if (std::string_view(name) != "concurrent-unsolicited") files.options.concurrent_dispatch = std::string_view(name) != "concurrent-off";
        if (std::string_view(name) == "concurrent-unsolicited" || std::string_view(name) == "concurrent-missing-ready" || std::string_view(name) == "concurrent-proof-ready") {
            refuses([&] { OwnedWorker refused(files.options); }); continue;
        }
        OwnedWorker worker(files.options); size_t callbacks = 0;
        worker.request(mtp_fixture::input(), Clock::now()+3s, [] { return false; });
        const auto generate = [&] { worker.request(mtp_fixture::scope_job(generation_job(8), 1, true, 8), Clock::now()+3s, [] { return false; }, [&](const Json&) { ++callbacks; }); };
        if (std::string_view(name) == "concurrent-on" || std::string_view(name) == "concurrent-off") {
            generate(); check(callbacks >= 8 && worker.alive(), "selected concurrent worker lost admitted tokens"); worker.close();
        } else { refuses(generate); check(!callbacks && !worker.alive(), "concurrent drift released tokens or left worker alive"); }
    }
    for (const bool mode : {false, true}) {
        scalar_fixture::Files files(executable, "unselected-transport"); files.options.concurrent_dispatch = mode;
        refuses([&] { OwnedWorker refused(files.options); });
        check(!std::filesystem::exists(files.directory/"started"), "dormant concurrent option launched a target");
    }
}
}
#ifndef TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY
#define TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto prior = inherited_mtp_wire_entry(argc, argv); if (prior) return prior;
    try { metadata(); workers(std::filesystem::canonical(argv[0]));
        std::cout << "concurrent HTTP CPU checks=" << checks << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
