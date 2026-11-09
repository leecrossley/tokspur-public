// CPU-only packet boundaries, publication and owned-child controls. No model is opened.
#define TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY inherited_http_concurrent_entry
#include "test_http_concurrent.cpp"
#undef TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY

#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
namespace packet_fixture {
Json prepared(uint32_t cap) {
    return with(with(mtp_fixture::depth2_ready(), "native_mtp_output_packet_schema", Json::integer(1)),
        "native_mtp_output_packet", Json::integer(cap));
}
void metadata() {
    const auto absent = mtp_fixture::depth2_ready();
    check(mtp_output_packet(absent) == 8, "omitted output selection changed the eight-token default");
    validate_mtp_output_packet_selection(absent, std::nullopt);
    for (const auto cap : {8u, 64u}) {
        const auto r = prepared(cap);
        validate_mtp_ready(r, "tiled128", true, 2);
        validate_mtp_output_packet_selection(r, cap);
        check(mtp_output_packet(r) == cap, "explicit output packet was not retained");
        refuses([&] { validate_mtp_output_packet_selection(r, std::nullopt); });
        refuses([&] { validate_mtp_output_packet_selection(absent, cap); });
        refuses([&] { validate_mtp_output_packet_selection(r, cap == 8 ? 64 : 8); });
        refuses([&] { static_cast<void>(mtp_output_packet(without(r, "native_mtp_output_packet"))); });
        refuses([&] { static_cast<void>(mtp_output_packet(without(r, "native_mtp_output_packet_schema"))); });
        for (const auto& bad : {Json::integer(0), Json::integer(7), Json::integer(9), Json::integer(65), parse_json("-1"), parse_json("64.0"), Json(true), Json("64")})
            refuses([&] { static_cast<void>(mtp_output_packet(with(r, "native_mtp_output_packet", bad))); });
        refuses([&] { static_cast<void>(mtp_output_packet(with(r, "native_mtp_output_packet_schema", Json::integer(2)))); });
        refuses([&] { static_cast<void>(mtp_output_packet(with(r, "native_mtp_output_packet_extra", Json(false)))); });
        refuses([&] { static_cast<void>(mtp_output_packet(with(r, "native_mtp_depth", Json::integer(1)))); });
        refuses([&] { static_cast<void>(mtp_output_packet(with(r, "native_mtp_draft_attention", Json("serial")))); });
        refuses([&] { static_cast<void>(mtp_output_packet(without(without(r, "native_mtp_carry_scope"), "native_mtp_generation"))); });
    }
}
void wire() {
    for (const auto cap : {8u, 64u}) for (const auto accepted : {0u, 1u, 2u}) {
        for (const auto width : {1u, 2u, 7u, 8u, 9u, 63u, 64u}) {
            mtp_fixture::Fake fake; fake.prepared = fake.previous = prepared(cap); fake.force_accepted = accepted;
            const auto input = mtp_fixture::input(); MtpWireReply initial(fake.prepared, fake.prepared, input);
            const auto prior = fake(input, [](const Json&) {}); initial.finish(prior);
            const auto job = mtp_fixture::scope_job(generation_job(width), 1, true, width);
            if (width > cap) { refuses([&] { MtpWireReply refused(fake.prepared, prior, job); }); continue; }
            MtpWireReply reply(fake.prepared, prior, job); refuses([&] { static_cast<void>(reply.events()); });
            const auto done = fake(job, [&](const Json& event) { reply.event(event); }); reply.finish(done);
            check(done.find("generated_ids")->array().size() == width && reply.events().size() <= 2 * cap,
                "packet output/event bound changed");
            if (cap == 64 && width == 64 && accepted == 0)
                check(reply.events().size() == 126, "maximum rejection event boundary was not exercised");
            refuses([&] { reply.event(fake.events.back()); });
            refuses([&] { MtpWireReply bad(fake.prepared, prior, mtp_fixture::scope_job(generation_job(cap + 1), 1, true, cap + 1)); });
            refuses([&] { MtpWireReply bad(fake.prepared, prior, mtp_fixture::input(9)); });
            auto drift = with(prior, "native_mtp_output_packet", Json::integer(cap == 8 ? 64 : 8));
            refuses([&] { MtpWireReply bad(fake.prepared, drift, job); });
            MtpWireReply invalid(fake.prepared, prior, job);
            for (const auto& event : fake.events) invalid.event(event);
            refuses([&] { invalid.finish(without(done, "native_mtp_output_packet_schema")); });
            refuses([&] { static_cast<void>(invalid.events()); });
            refuses([&] { invalid.finish(with(done, "native_mtp_output_packet", Json::integer(cap == 8 ? 64 : 8))); });
            refuses([&] { invalid.finish(with(done, "generated_ids", Json(Json::Array{}))); });
        }
    }
    for (const auto stop : {1u, 2u, 3u, 31u, 63u, 64u}) {
        mtp_fixture::Fake fake; fake.prepared = fake.previous = prepared(64); fake.stop_at = stop;
        const auto pre = mtp_fixture::input(); MtpWireReply initial(fake.prepared, fake.prepared, pre);
        const auto prior = fake(pre, [](const Json&) {}); initial.finish(prior);
        const auto job = mtp_fixture::scope_job(generation_job(64), 1, true, 64);
        MtpWireReply reply(fake.prepared, prior, job);
        const auto done = fake(job, [&](const Json& event) { reply.event(event); }); reply.finish(done);
        check(done.find("generated_ids")->array().size() == stop && done.find("status")->string() == "stopped",
            "large output packet lost an early stop");
    }
}
void sessions() {
    for (const auto cap : {8u, 64u}) for (const auto maximum : {1u, 8u, 9u, 64u, 65u, 129u}) {
        mtp_fixture::Fake fake; fake.prepared = fake.previous = prepared(cap);
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, fake.prepared);
        const auto request = with(chat(std::string(17, 'A'), false, maximum), "temperature", number(1));
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
        const auto result = session.complete(request, "output-packet", [&](const Json& job, const auto& callback) { return fake(job, callback); }, sink);
        size_t outputs = 0; uint32_t remaining = maximum;
        for (const auto& job : fake.packets) {
            const auto width = uint32_t(integer(job, "generate"));
            if (!width) { check(job.find("tokens")->array().size() <= 8, "larger output packet widened known input"); continue; }
            const auto command = mtp_generation_command(*job.find("mtp_generation"));
            check(width == std::min(cap, remaining) && command.remaining == remaining && command.begin == (outputs == 0),
                "HTTP scheduler lost the negotiated output bound or lifetime");
            remaining -= width; ++outputs;
        }
        const auto& details = *result.find("tokspur_private");
        check(!remaining && outputs == (maximum + cap - 1) / cap && integer(details, "generation_batches") == outputs && !fake.generation_active,
            "large-packet request did not consume and close its exact budget");
        check(integer(details, "native_mtp_output_packet") == cap && integer(details, "native_mtp_output_packet_schema") == 1 &&
            integer(details, "maximum_generation_width") == std::min(cap, maximum), "HTTP response lost its selected packet identity");
        session.invalidate();
        static_cast<void>(session.complete(request, "cold-output-packet", [&](const Json& job, const auto& callback) { return fake(job, callback); }, sink));
        check(fake.generation_id == 2, "large-packet cold recovery reused a generation ID");
    }
    {
        mtp_fixture::Fake fake; fake.prepared = fake.previous = prepared(64);
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, fake.prepared,
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
            Json(),
#endif
            uint32_t(8));
        size_t chunks_after_first_packet = 0, streamed_chunks = 0;
        const auto request = with(chat(std::string(17, 'A'), true, 80), "temperature", number(1));
        const ServeSink sink{[&](const Json& chunk) {
            if (chunk.find("choices") && !chunk.find("choices")->array().empty() &&
                chunk.find("choices")->array()[0].find("delta")->find("content")) ++streamed_chunks;
        }, [] { return false; }, [] { return uint64_t(44); }};
        const auto result = session.complete(request, "first-packet-8", [&](const Json& job, const auto& callback) {
            auto done = fake(job, callback);
            if (integer(job, "generate") == 8 && chunks_after_first_packet == 0 && streamed_chunks == 0) {
                // Stream chunks flush immediately after run_mtp returns for packet 0.
            } else if (integer(job, "generate") == 64 && chunks_after_first_packet == 0) {
                chunks_after_first_packet = streamed_chunks;
            }
            return done;
        }, sink);
        std::vector<uint32_t> gen_widths;
        for (const auto& job : fake.packets) {
            const auto width = uint32_t(integer(job, "generate"));
            if (width) gen_widths.push_back(width);
        }
        check(gen_widths == std::vector<uint32_t>{8, 64, 8}, "first-output-packet=8 over output_packet=64 did not emit 8 -> 64 -> 8 packet schedule");
        check(chunks_after_first_packet > 0, "first-output-packet=8 did not flush SSE content chunks before steady-state 64-token packet");
        const auto& details = *result.find("tokspur_private");
        check(integer(details, "native_mtp_first_output_packet") == 8 && integer(details, "maximum_generation_width") == 64,
            "first-output-packet metadata missing from tokspur_private");
        refuses([&] {
            ServeSession bad(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, prepared(8),
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
                Json(),
#endif
                uint32_t(16));
        });
    }
}
void options(scalar_fixture::Files& files, std::optional<uint32_t> cap) {
    files.options.mtp_sidecar = files.directory / "mtp-sidecar";
    { std::ofstream sidecar(files.options.mtp_sidecar); sidecar << "generated CPU-only fixture\n"; }
    files.options.mtp_depth = 2; files.options.mtp_request_carry = true; files.options.mtp_draft_attention = "tiled128";
    files.options.mtp_output_packet = cap;
}
void workers(const std::filesystem::path& executable) {
    for (const auto mode : {"packet64-ok", "packet64-drift", "packet64-missing-done", "packet64-unknown-done", "packet64-slow",
        "packet64-unsolicited", "packet64-missing-ready"}) {
        scalar_fixture::Files files(executable, mode); options(files, std::string_view(mode) == "packet64-unsolicited" ? std::nullopt : std::optional<uint32_t>(64));
        if (std::string_view(mode) == "packet64-unsolicited" || std::string_view(mode) == "packet64-missing-ready") {
            refuses([&] { OwnedWorker bad(files.options); }); continue;
        }
        OwnedWorker worker(files.options); worker.request(mtp_fixture::input(), Clock::now() + 3s, [] { return false; });
        size_t callbacks = 0;
        const auto job = mtp_fixture::scope_job(generation_job(64), 1, true, 64);
        const auto began = Clock::now();
        const auto generate = [&] { return worker.request(job, Clock::now() + 3s,
            [&] { return std::string_view(mode) == "packet64-slow" && Clock::now() - began > 25ms; }, [&](const Json&) { ++callbacks; }); };
        if (std::string_view(mode) == "packet64-slow") {
            bool interrupted = false;
            try { static_cast<void>(generate()); } catch (const WorkerCancelled&) { interrupted = true; }
            check(interrupted && !callbacks && worker.alive(), "cancelled large packet did not drain without publication");
            refuses([&] { worker.request(job, Clock::now() + 3s, [] { return false; }); });
            worker.request(mtp_fixture::input(), Clock::now() + 3s, [] { return false; });
            worker.close();
        } else if (std::string_view(mode) == "packet64-ok") {
            const auto result = generate(); check(result.find("generated_ids")->array().size() == 64 && callbacks >= 64 && worker.alive(), "owned large packet lost output"); worker.close();
        } else { refuses(generate); check(!callbacks && !worker.alive(), "unadmitted or cancelled packet published tokens or kept its worker"); }
    }
    for (const auto cap : {8u, 64u}) {
        scalar_fixture::Files files(executable, "packet-explicit"); options(files, cap);
        OwnedWorker worker(files.options); validate_mtp_output_packet_selection(worker.ready(), cap); worker.close();
    }
    for (const auto invalid : {0u, 9u, 65u}) {
        scalar_fixture::Files files(executable, "packet-invalid"); options(files, invalid);
        refuses([&] { OwnedWorker bad(files.options); });
        check(!std::filesystem::exists(files.directory / "started"), "invalid output cap executed a child");
    }
    scalar_fixture::Files files(executable, "packet-incompatible"); options(files, 64); files.options.mtp_request_carry = false;
    refuses([&] { OwnedWorker bad(files.options); });
    check(!std::filesystem::exists(files.directory / "started"), "incompatible packet mode executed a child");
}
}
#endif
int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_http_concurrent_entry(argc, argv); if (inherited) return inherited;
    try {
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        packet_fixture::metadata(); packet_fixture::wire(); packet_fixture::sessions(); packet_fixture::workers(std::filesystem::canonical(argv[0]));
#else
        const auto ready = mtp_fixture::depth2_ready();
        check(mtp_output_packet(ready) == 8, "legacy output bound changed");
        validate_mtp_output_packet_selection(ready, std::nullopt);
        for (const auto cap : {8u, 64u}) {
            const auto advertised = with(with(ready, "native_mtp_output_packet_schema", Json::integer(1)), "native_mtp_output_packet", Json::integer(cap));
            refuses([&] { static_cast<void>(mtp_output_packet(advertised)); });
            refuses([&] { validate_mtp_output_packet_selection(ready, cap); });
        }
#endif
        std::cout << "MTP packet controls passed: " << checks << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
