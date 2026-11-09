// Generated CPU wire/session/owned-child controls. No model, tokenizer or Metal is opened.
#define TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY inherited_http_concurrent_entry
#include "test_http_concurrent.cpp"
#undef TOKSPUR_HTTP_CONCURRENT_TEST_ENTRY

#ifndef TOKSPUR_PRIVATE_HTTP_PREFILL512
#error This fixture requires the isolated HTTP input512 carrier.
#endif
namespace prefill512_fixture {
void options(scalar_fixture::Files& files, bool selected = true);
Json ready(uint32_t output = 8) {
    auto value = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready());
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    value = with(with(std::move(value), "native_mtp_output_packet_schema", Json::integer(1)), "native_mtp_output_packet", Json::integer(output));
#else
    check(output == 8, "output64 fixture requires its explicit build");
#endif
    return value;
}
void metadata() {
    const auto legacy = mtp_fixture::depth2_ready(), r = ready();
    check(mtp_input_packet(legacy) == 8 && mtp_input_packet(r) == 512, "input packet default or selection differs");
    validate_mtp_input_packet_selection(legacy, false); validate_mtp_input_packet_selection(r, true); validate_mtp_ready(r, "tiled128", true, 2);
    refuses([&] { validate_mtp_input_packet_selection(legacy, true); }); refuses([&] { validate_mtp_input_packet_selection(r, false); });
    for (const auto name : {"native_mtp_input_packet_max", "native_mtp_input_packet_schema", "native_mtp_priming_batch_capacity", "native_mtp_priming_batch_bytes", "native_mtp_priming_batch_submits", "native_prefill_dense"})
        refuses([&] { validate_mtp_ready(without(r, name), "tiled128", true, 2); });
    for (const auto& [name, field] : Json::Object{{"native_mtp_input_packet_max", Json::integer(8)}, {"native_mtp_input_packet_max", Json::integer(513)},
        {"native_mtp_input_packet_max", parse_json("512.0")}, {"native_mtp_input_packet_schema", Json::integer(2)},
        {"native_mtp_input_packet_extra", Json(true)}, {"native_mtp_priming_batch_capacity", Json::integer(8)},
        {"native_mtp_priming_batch_bytes", Json::integer(0)}, {"native_mtp_priming_batch_bytes", Json::integer((32ull << 20) + 1)},
        {"native_mtp_priming_batch_submits", Json::integer(1)}, {"native_mtp_priming_batch_extra", Json(false)},
        {"known_token_capacity", Json::integer(8)}, {"known_qsa_rows_mode", Json("q8")}, {"known_qsa_rows_logical_bytes", Json::integer(1)},
        {"known_qsa_rows_charged_bytes", Json::integer(1)}, {"known_qsa_rows_allocations", Json::integer(46)},
        {"native_prefill_dense", Json("unknown")}, {"native_mtp_depth", Json::integer(1)}, {"native_mtp_draft_attention", Json("serial")}})
        refuses([&] { validate_mtp_ready(with(r, name, field), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(scalar_fixture::lane_fields(r), "tiled128", true, 2); });
    refuses([&] { static_cast<void>(mtp_input_packet(with(legacy, "native_mtp_priming_batch_capacity", Json::integer(512)))); });
    refuses([&] { MtpWireReply reply(legacy, legacy, mtp_fixture::input(9)); });
    refuses([&] { MtpWireReply reply(r, r, mtp_fixture::input(513)); });
}
void staged_dense_controls(const std::filesystem::path& executable) {
    const auto staged = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), "f32-staged", "f32");
#ifndef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    refuses([&]{validate_mtp_ready(staged,"tiled128",true,2);});
    scalar_fixture::Files files(executable,"staged-uncompiled");options(files);files.options.prefill_dense="f32-staged";
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    files.options.prefill_grouped="f32";
#endif
    refuses([&]{OwnedWorker invalid(files.options);});check(!std::filesystem::exists(files.directory/"started"),"uncompiled staged selector launched a child");
#else
    validate_mtp_ready(staged,"tiled128",true,2);validate_mtp_prefill_selection(staged,"f32-staged","f32");++checks;
    for (const uint32_t capacity : {1024u, 2048u}) {
        const auto wide = with(with(staged, "native_mtp_input_packet_max", Json::integer(capacity)), "known_token_capacity", Json::integer(capacity));
        check(mtp_input_packet(wide) == capacity, "staged FP32 wide input packet cap differs");
        validate_mtp_ready(wide, "tiled128", true, 2);
        { MtpWireReply reply(wide, wide, mtp_fixture::input(capacity)); ++checks; }
        refuses([&] { validate_mtp_ready(with(wide, "native_mtp_priming_batch_capacity", Json::integer(capacity)), "tiled128", true, 2); });
        auto nax = with(with(wide, "native_prefill_dense", Json("nax")), "native_prefill_dense_kernel", Json("nax_prefill_dense_q8_m32"));
        nax = with(with(nax, "native_prefill_grouped", Json("nax")), "native_prefill_grouped_kernel", Json("nax_grouped_moe"));
        nax = with(with(nax, "native_mtp_priming_batch_capacity", Json::integer(capacity)), "native_mtp_priming_batch_bytes", Json::integer(160ull << 20));
        validate_mtp_ready(nax, "tiled128", true, 2); ++checks;
        refuses([&] { validate_mtp_ready(with(nax, "native_mtp_priming_batch_capacity", Json::integer(512)), "tiled128", true, 2); });
        refuses([&] { validate_mtp_ready(with(nax, "native_mtp_priming_batch_bytes", Json::integer((160ull << 20) + 1)), "tiled128", true, 2); });
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
        // These generated wire records test counter admission, not physical GPU dispatch counts.
        nax = mtp_fixture::hc_rows_fields(nax);
        mtp_fixture::Fake wide_target; wide_target.prepared = wide_target.previous = nax;
        const auto wide_job = mtp_fixture::input(capacity);
        auto wide_done = wide_target(wide_job, [](const Json&) {});
        wide_done = with(with(wide_done, "known_qsa_rows_physical_tiles", Json::integer(0)), "known_qsa_rows_logical_rows", Json::integer(0));
        wide_done = mtp_fixture::hc_rows_fields(wide_done, 96, 96);
        MtpWireReply admitted(nax, nax, wide_job); admitted.finish(wide_done); ++checks;
        MtpWireReply wrong_chunks(nax, nax, wide_job);
        refuses([&] { wrong_chunks.finish(mtp_fixture::hc_rows_fields(wide_done, 192, 192)); });
#endif
    }
    refuses([&]{validate_mtp_prefill_selection(staged,"f32","f32");});
    refuses([&]{validate_mtp_prefill_selection(without(staged,"native_prefill_dense_kernel"),"f32-staged","f32");});
    for(const auto& [name,value]:Json::Object{{"native_prefill_dense_kernel",Json("graph_q8_mpp_wide")},{"native_prefill_dense_kernel",Json(true)},{"native_prefill_dense_kernel",Json()},{"native_prefill_dense",Json("f16")},{"native_prefill_chunks",Json("cs64")},{"native_prefill_grouped",Json("serial")},{"native_prefill_dense_extra",Json(true)}})
        refuses([&]{validate_mtp_ready(with(staged,name,value),"tiled128",true,2);});
    mtp_fixture::Fake target;target.prepared=target.previous=staged;const auto cold=mtp_fixture::input(3);MtpWireReply input(staged,staged,cold);const auto prior=target(cold,[](const Json&){});input.finish(prior);
    const auto job=mtp_fixture::scope_job(generation_job(8),1,true,8);MtpWireReply good(staged,prior,job);const auto done=target(job,[&](const Json& event){good.event(event);});good.finish(done);
    for(const auto& changed:std::vector<Json>{without(done,"native_prefill_dense_kernel"),with(done,"native_prefill_dense_kernel",Json("graph_q8_mpp_wide")),with(done,"native_prefill_chunks",Json("cs64"))}){
        MtpWireReply bad(staged,prior,job);for(const auto& e:target.events)bad.event(e);refuses([&]{bad.finish(changed);});refuses([&]{static_cast<void>(bad.events());});
    }
    // The inherited inactive cold reset remains the only counter resync. It cannot change the pinned dense kernel.
    refuses([&]{MtpWireReply bad(staged,without(prior,"native_prefill_dense_kernel"),cold);});
    refuses([&]{MtpWireReply bad(staged,with(prior,"native_prefill_dense_kernel",Json("graph_q8_mpp_wide")),cold);});
    mtp_fixture::cold_generation_resync(staged);
    mtp_fixture::Fake endpoint;endpoint.prepared=endpoint.previous=staged;std::vector<Json> traces;
    ServeSession session(session_codec(),"tokspur","direct",Json(),false,Json(),false,ToolEnumValidation::strict,false,staged);
    const ServeSink sink{[](const Json&){},[]{return false;},[]{return uint64_t(44);},[&](const Json& v){traces.push_back(v);}};
    const auto result=session.complete(with(chat("staged",false,8),"temperature",number(1)),"staged-dense",[&](const Json& packet,const auto& callback){return endpoint(packet,callback);},sink);
    const auto& details=*result.find("tokspur_private");check(details.find("native_prefill_dense")->string()=="f32"&&details.find("native_prefill_dense_kernel")->string()=="graph_dense_q8_staged_f32","HTTP completion lost staged precision/kernel identity");
    check(traces.back().find("native_prefill_dense_kernel")->string()=="graph_dense_q8_staged_f32","private completion trace lost dense kernel identity");
    for(const auto mode:{"staged-argv","staged-ready-missing","staged-ready-drift","staged-ready-unsolicited","staged-done-missing","staged-done-drift","staged-done-fused-drift","staged-cold-reset-missing"}){
        scalar_fixture::Files files(executable,mode);options(files);files.options.prefill_dense=std::string_view(mode)=="staged-ready-unsolicited"?"f32":"f32-staged";files.options.prefill_grouped="f32";
        if(std::string_view(mode).find("ready")!=std::string_view::npos){refuses([&]{OwnedWorker invalid(files.options);});continue;}
        OwnedWorker worker(files.options);
        if(std::string_view(mode)=="staged-cold-reset-missing"){refuses([&]{worker.request(cold,Clock::now()+3s,[]{return false;});});check(!worker.alive(),"cold reset kernel drift left worker alive");continue;}
        worker.request(cold,Clock::now()+3s,[]{return false;});size_t callbacks=0;
        const auto generate=[&]{worker.request(job,Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;});};
        if(std::string_view(mode)=="staged-argv"){generate();check(callbacks>=8&&worker.ready().find("native_prefill_dense")->string()=="f32"&&worker.ready().find("native_prefill_dense_kernel"),"staged child argv/identity differs");worker.close();}
        else {refuses(generate);check(!callbacks&&!worker.alive(),"staged drift released unadmitted callbacks");}
    }
    for(unsigned invalid=0;invalid<3;++invalid){scalar_fixture::Files files(executable,"staged-invalid-consent");options(files,invalid!=0);files.options.prefill_dense="f32-staged";files.options.prefill_grouped=invalid==1?"serial":"f32";files.options.prefill_hc_down_f32=invalid==2;
        refuses([&]{OwnedWorker bad(files.options);});check(!std::filesystem::exists(files.directory/"started"),"invalid staged startup consent launched a child");}
#endif
}
void precision_metadata() {
#ifndef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    for (bool on : {false, true}) {
        scalar_fixture::Files files("/usr/bin/false", "hc-rows-uncompiled"); options(files); files.options.prefill_hc_rows = on;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "uncompiled HC rows started a child");
    }
#endif
#ifndef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    for (bool on : {false, true}) {
        scalar_fixture::Files files("/usr/bin/false", "router-window-uncompiled"); options(files); files.options.prefill_router_window = on;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "uncompiled router window started a child");
    }
#endif
#ifndef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    for (bool on : {false, true}) {
        scalar_fixture::Files files("/usr/bin/false", "hc-down-window-uncompiled"); options(files); files.options.prefill_hc_down_window = on;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "uncompiled HC down window started a child");
    }
#endif
    const auto legacy = mtp_fixture::depth2_ready(), base = ready();
    validate_mtp_prefill_selection(legacy); validate_mtp_prefill_selection(base);
    refuses([&] { validate_mtp_prefill_selection(legacy, "f32"); });
    refuses([&] { validate_mtp_prefill_selection(base, "unknown"); });
    for (const auto dense : {"f16", "f32"}) {
        const auto r = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), dense);
        validate_mtp_ready(r, "tiled128", true, 2); validate_mtp_prefill_selection(r, dense); ++checks;
        if (std::string_view(dense) == "f32") refuses([&] { validate_mtp_prefill_selection(r); });
        const auto grouped = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), dense, "f32");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        for (const auto mode : {"serial", "f32"}) {
            const auto selected = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), dense, mode);
            validate_mtp_ready(selected, "tiled128", true, 2); validate_mtp_prefill_selection(selected, dense, mode); ++checks;
            refuses([&] { validate_mtp_prefill_selection(selected, dense); });
            refuses([&] { validate_mtp_prefill_selection(selected, dense, std::string_view(mode) == "f32" ? "serial" : "f32"); });
        }
        for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel"})
            refuses([&] { validate_mtp_ready(without(grouped, name), "tiled128", true, 2); });
        for (const auto& [name, field] : Json::Object{{"native_prefill_grouped", Json("unknown")}, {"native_prefill_grouped", Json(true)},
            {"native_prefill_grouped_kernel", Json("graph_grouped_serial")}, {"native_prefill_grouped_kernel", Json()},
            {"native_prefill_grouped_extra", Json(false)}})
            refuses([&] { validate_mtp_ready(with(grouped, name, field), "tiled128", true, 2); });
        refuses([&] { validate_mtp_ready(with(with(legacy, "native_prefill_grouped", Json("f32")),
            "native_prefill_grouped_kernel", Json("graph_grouped_mm_f32")), "tiled128", true, 2); });
#else
        refuses([&] { validate_mtp_ready(grouped, "tiled128", true, 2); });
#endif
    }
}
void precision_wire_sessions() {
    for (const auto dense : {"f16", "f32"}) {
        std::vector<std::optional<std::string_view>> modes{std::nullopt};
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        modes.push_back("serial"); modes.push_back("f32");
#endif
        for (const auto grouped : modes) {
            mtp_fixture::Fake f; f.prepared = f.previous = mtp_fixture::prefill512_ready(mtp_fixture::depth2_ready(), dense, grouped);
            const auto input = mtp_fixture::input(512); MtpWireReply first(f.prepared, f.prepared, input);
            const auto prior = f(input, [](const Json&) {}); first.finish(prior);
            const auto job = mtp_fixture::scope_job(generation_job(8), 1, true, 8);
            MtpWireReply output(f.prepared, prior, job); const auto done = f(job, [&](const Json& event) { output.event(event); }); output.finish(done);
            check(output.events().size() >= 8, "selected prefill precision lost admitted generation");
            const auto opposite_dense = std::string_view(dense) == "f16" ? "f32" : "f16";
            { MtpWireReply bad(f.prepared, prior, job); for (const auto& event : f.events) bad.event(event);
                refuses([&] { bad.finish(with(done, "native_prefill_dense", Json(opposite_dense))); }); refuses([&] { static_cast<void>(bad.events()); }); }
            refuses([&] { MtpWireReply bad(f.prepared, with(prior, "native_prefill_dense", Json(opposite_dense)), job); });
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel"}) {
                const auto changed = grouped ? without(done, name) : with(done, name, Json("f32"));
                MtpWireReply bad(f.prepared, prior, job); for (const auto& event : f.events) bad.event(event);
                refuses([&] { bad.finish(changed); }); refuses([&] { static_cast<void>(bad.events()); });
                refuses([&] { MtpWireReply invalid(f.prepared, grouped ? without(prior, name) : with(prior, name, Json("f32")), job); });
            }
            if (grouped) {
                const auto mode = *grouped == "f32" ? "serial" : "f32";
                const auto changed = with(with(done, "native_prefill_grouped", Json(mode)), "native_prefill_grouped_kernel", Json(*grouped == "f32" ? "graph_grouped_serial" : "graph_grouped_mm_f32"));
                MtpWireReply bad(f.prepared, prior, job); for (const auto& event : f.events) bad.event(event);
                refuses([&] { bad.finish(changed); }); refuses([&] { static_cast<void>(bad.events()); });
            }
#endif
            mtp_fixture::Fake session_target; session_target.prepared = session_target.previous = f.prepared;
            ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, f.prepared);
            const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
            const auto result = session.complete(with(chat(std::string(513, 'A'), false, 8), "temperature", number(1)), "precision",
                [&](const Json& packet, const auto& callback) { return session_target(packet, callback); }, sink);
            const auto& details = *result.find("tokspur_private");
            check(details.find("native_prefill_dense")->string() == dense, "HTTP completion lost effective dense precision");
            for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel"})
                check(bool(details.find(name)) == grouped.has_value() && (!grouped || details.find(name)->dump() == f.prepared.find(name)->dump()),
                    "HTTP completion lost effective grouped precision identity");
            check(!details.find("q8_three_rows_mode") && !details.find("native_residency_mode"), "precision forwarding enabled an unrelated experiment");
            mtp_fixture::cold_generation_resync(f.prepared);
        }
    }
}
void private_prefix_trace() {
    mtp_fixture::Fake fake; fake.prepared = fake.previous = ready();
    ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, fake.prepared);
    const ServeSink ordinary{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
    check(!ordinary.prefix_trace, "private prefix trace enabled without a callback");
    std::vector<Json> records; bool admitted = false;
    const ServeSink traced{[](const Json& chunk) { check(chunk.dump().find("prompt_ids") == std::string::npos, "private IDs reached an HTTP chunk"); },
        [] { return false; }, [] { return uint64_t(44); }, [&](const Json& record) {
            if (record.find("event")->string() == "prompt_prefix") { check(!admitted, "duplicate prompt admission trace"); admitted = true; }
            records.push_back(record);
        }};
    const auto run = [&](std::string text, std::string id, bool stream) {
        admitted = false;
        const auto result = session.complete(with(chat(std::move(text), stream, 1), "temperature", number(1)), id,
            [&](const Json& job, const auto& callback) { check(admitted, "target work preceded private prefix trace"); return fake(job, callback); }, traced);
        check(result.dump().find("prompt_ids") == std::string::npos && result.dump().find("prior_committed_ids") == std::string::npos, "private IDs reached HTTP completion");
        const auto& completed = records.back(); const auto& usage = *result.find("usage");
        check(completed.find("event")->string() == "completion" && completed.find("response_id")->string() == id &&
            integer(completed, "prompt_tokens") == integer(usage, "prompt_tokens") && integer(completed, "completion_tokens") == integer(usage, "completion_tokens") &&
            integer(completed, "cached_tokens") == integer(*usage.find("prompt_tokens_details"), "cached_tokens"), "private completion trace differs from admitted usage");
    };
    run("AB", "trace-cold", false); run("AB)C", "trace-growing", true); run("AB?D", "trace-diverged", false);
    check(records.size() == 6, "private prefix trace omitted an admission or completion");
    const auto& cold = records[0]; const auto& growing = records[2]; const auto& diverged = records[4];
    check(cold.find("prompt_ids")->dump() == "[66,67]" && cold.find("prior_committed_ids")->array().empty() &&
        !cold.find("reusable")->boolean() && !cold.find("append")->boolean() && !integer(cold, "common_prefix") && !integer(cold, "cached_tokens") && integer(cold, "new_tokens") == 2,
        "cold trace lost exact canonical IDs or cold counts");
    check(growing.find("prompt_ids")->dump() == "[66,67,42,68]" && growing.find("prior_committed_ids")->dump() == "[66,67,42]" &&
        growing.find("reusable")->boolean() && growing.find("append")->boolean() && integer(growing, "prior_committed_tokens") == 3 &&
        integer(growing, "common_prefix") == 3 && integer(growing, "cached_tokens") == 3 && integer(growing, "new_tokens") == 1,
        "growing trace lost actual committed IDs or append reuse");
    check(diverged.find("prompt_ids")->dump() == "[66,67,64,69]" && diverged.find("prior_committed_ids")->dump() == "[66,67,42,68,42]" &&
        diverged.find("reusable")->boolean() && !diverged.find("append")->boolean() && integer(diverged, "prior_committed_tokens") == 5 &&
        integer(diverged, "common_prefix") == 2 && !integer(diverged, "cached_tokens") && integer(diverged, "new_tokens") == 4,
        "diverged trace confused actual common prefix with admitted cache reuse");
    const auto before = fake.packets.size();
    const auto before_tokens = session.committed_tokens();
    auto refused = traced; refused.prefix_trace = [](const Json&) { throw FormatError("generated private trace failure"); };
    refuses([&] { session.complete(chat("AB", false, 1), "trace-refused", [&](const Json& job, const auto& callback) { return fake(job, callback); }, refused); });
    check(fake.packets.size() == before && session.committed_tokens() == before_tokens, "failed private trace started target work or changed session state");
}
void private_native_timing_trace() {
    for (const auto mode : {"normal", "missing-clock", "counter-drift", "cancel-resync", "ordinary-stop"}) {
        mtp_fixture::Fake fake; fake.prepared = fake.previous = ready();
        if (std::string_view(mode) == "ordinary-stop") fake.stop_at = 1;
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, fake.prepared);
        std::vector<Json> records; bool cancel = false, cancel_output = false;
        uint32_t input_packet = 0, output_packet = 0;
        const ServeSink sink{[](const Json& chunk) { check(chunk.dump().find("native_worker") == std::string::npos, "private native clocks reached SSE"); },
            [&] { return cancel; }, [] { return uint64_t(44); }, [&](const Json& record) { records.push_back(record); }};
        const auto target = [&](const Json& packet, const auto& callback) {
            const bool input = !packet.find("tokens")->array().empty();
            if (input) ++input_packet;
            else { ++output_packet; fake.force_accepted = output_packet == 1 ? 2 : output_packet == 2 ? 1 : 0; }
            auto done = fake(packet, [&](const Json& event) {
                auto observed = event;
                if (event.find("event")->string() == "mtp_round") {
                    const double scale = output_packet, recovery = integer(event, "accepted") < 2 ? scale : 0;
                    for (const auto& [name, value] : std::array<std::pair<std::string_view, double>, 8>{{
                        {"draft_wall_ms", scale}, {"draft_gpu_ms", scale / 2}, {"verification_wall_ms", 2 * scale}, {"verification_gpu_ms", 1.5 * scale},
                        {"recovery_wall_ms", recovery / 4}, {"recovery_gpu_ms", recovery / 8}, {"materialization_wall_ms", 0.75 * scale}, {"round_wall_ms", 4 * scale}}})
                        observed = with(std::move(observed), name, number(value));
                }
                callback(observed);
            });
            done = with(std::move(done), "request_wall_ms", number(input ? 9 + 2 * input_packet : 100 + output_packet));
            done = with(std::move(done), "prefill_wall_ms", number(input ? 6 + input_packet : 0));
            done = with(std::move(done), "generation_batch_wall_ms", number(input ? 0 : 50 + output_packet));
            done = with(std::move(done), "generation_batch_gpu_ms", number(input ? 0 : 20 + output_packet));
            if (!input && std::string_view(mode) == "missing-clock" && output_packet == 2) done = without(done, "generation_batch_gpu_ms");
            if (!input && std::string_view(mode) == "counter-drift") {
                const auto proposed = integer(done, "native_mtp_proposed"); done = with(std::move(done), "native_mtp_proposed", Json::integer(proposed + 1));
            }
            if (!input && cancel_output) cancel = true;
            return done;
        };
        const auto request = with(chat(std::string(513, 'A'), std::string_view(mode) == "normal", 19), "temperature", number(1));
        if (std::string_view(mode) == "cancel-resync") {
            cancel_output = true;
            refuses([&] { session.complete(with(request, "max_tokens", Json::integer(8)), "native-trace-cancelled", target, sink); });
            check(!session.committed_tokens() && records.size() == 1 && records.back().find("event")->string() == "prompt_prefix",
                "cancelled native trace published a completion aggregate");
            cancel = cancel_output = false; input_packet = output_packet = 0;
        }
        if (std::string_view(mode) == "counter-drift") {
            refuses([&] { session.complete(request, "native-trace-drift", target, sink); });
            check(records.size() == 1 && !session.committed_tokens(), "unadmitted counter drift published native timing aggregates"); continue;
        }
        const auto result = session.complete(request, "native-trace-complete", target, sink);
        check(result.dump().find("native_worker") == std::string::npos, "private native timing aggregates reached HTTP completion");
        const auto& aggregate = *records.back().find("native_worker");
        const auto& inputs = *aggregate.find("input_packets"); const auto& outputs = *aggregate.find("output_packets"); const auto& mtp = *aggregate.find("mtp");
        const auto clock = [&](const Json& section, std::string_view name, double expected, uint64_t samples) {
            const auto& value = *section.find(name);
            check(integer(value, "observed_samples") == samples && integer(value, "expected_samples") == samples &&
                std::stod(std::string(value.find("total_ms")->number_lexeme())) == expected, "private native clock total or sample scope differs");
        };
        if (std::string_view(mode) == "ordinary-stop") {
            check(integer(inputs, "count") == 2 && integer(outputs, "count") == 1 && !integer(mtp, "rounds") && integer(mtp, "ordinary_tokens") == 1 &&
                !integer(mtp, "ordinary_tail_tokens") && integer(mtp, "ordinary_non_tail_tokens") == 1, "ordinary terminal token was mislabelled as a packet tail");
            clock(inputs, "request_wall_ms", 24, 2); clock(outputs, "request_wall_ms", 101, 1);
            const auto& no_rounds = *mtp.find("round_wall_ms");
            check(no_rounds.find("total_ms")->is_null() && !integer(no_rounds, "expected_samples"), "unobserved MTP round clock became a measured zero"); continue;
        }
        check(integer(inputs, "count") == 2 && integer(outputs, "count") == 3 && !aggregate.find("isolated_decode_measured")->boolean() &&
            !aggregate.find("timings_qualified")->boolean(), "private native packet scope or qualification differs");
        clock(inputs, "request_wall_ms", 24, 2); clock(inputs, "prefill_wall_ms", 15, 2);
        clock(outputs, "request_wall_ms", 306, 3); clock(outputs, "generation_batch_wall_ms", 156, 3);
        if (std::string_view(mode) == "missing-clock") {
            const auto& missing = *outputs.find("generation_batch_gpu_ms");
            check(missing.find("total_ms")->is_null() && integer(missing, "observed_samples") == 2 && integer(missing, "expected_samples") == 3,
                "missing native clock became a zero or complete total");
        } else clock(outputs, "generation_batch_gpu_ms", 66, 3);
        check(integer(mtp, "rounds") == 6 && integer(mtp, "proposed_tokens") == 12 && integer(mtp, "accepted_tokens") == 7 &&
            integer(mtp, "recovery_rounds") == 4 && integer(mtp, "ordinary_tokens") == 6 && integer(mtp, "ordinary_tail_tokens") == 6 &&
            !integer(mtp, "ordinary_non_tail_tokens"), "private native counts included stale counters or omitted ordinary tails");
        clock(mtp, "draft_wall_ms", 11, 6); clock(mtp, "draft_gpu_ms", 5.5, 6);
        clock(mtp, "verification_wall_ms", 22, 6); clock(mtp, "verification_gpu_ms", 16.5, 6);
        clock(mtp, "recovery_wall_ms", 2.25, 6); clock(mtp, "recovery_gpu_ms", 1.125, 6);
        clock(mtp, "materialization_wall_ms", 8.25, 6); clock(mtp, "round_wall_ms", 44, 6);
        check(mtp.find("materialization_gpu_ms")->is_null() && mtp.find("round_gpu_ms")->is_null(), "unavailable round GPU clocks were invented");
    }
}
void wire() {
    for (uint32_t n : {1, 2, 3, 8, 9, 63, 511, 512}) {
        mtp_fixture::Fake f; f.prepared = f.previous = ready();
        const auto input = mtp_fixture::input(n); MtpWireReply first(f.prepared, f.prepared, input);
        const auto prior = f(input, [](const Json&) {}); first.finish(prior); check(first.events().empty(), "known inputs released token events");
        check(integer(prior, "native_mtp_prime_calls") == n - 1 && integer(prior, "native_mtp_priming_batch_submits") == (n > 1), "input512 prime rows or submits differ");
        check(!integer(prior, "known_qsa_rows_physical_tiles") && !integer(prior, "known_qsa_rows_logical_rows"), "known input entered verifier row counters");
        check(integer(prior, "qsa_stage_v8_request_calls") == (n == 1 ? 12 : 0), "singleton input did not use ordinary attention");
        const auto append = mtp_fixture::input(3, false); MtpWireReply second(f.prepared, prior, append); const auto appended = f(append, [](const Json&) {}); second.finish(appended);
        check(integer(appended, "native_mtp_prime_calls") == n + 2 && integer(appended, "native_mtp_priming_batch_submits") == (n > 1 ? 2 : 1), "cached append lost batched priming chronology");
        for (const auto& [name, field] : Json::Object{{"native_mtp_input_packet_max", Json::integer(8)}, {"native_mtp_priming_batch_capacity", Json::integer(511)},
            {"native_mtp_priming_batch_bytes", Json::integer(26'214'404)},
            {"known_qsa_rows_physical_tiles", Json::integer(12)}, {"known_qsa_rows_logical_rows", Json::integer(36)},
            {"qsa_stage_v8_request_calls", Json::integer(12)}, {"native_prefill_dense", Json("f32")}}) {
            MtpWireReply bad(f.prepared, prior, append); refuses([&] { bad.finish(with(appended, name, field)); }); refuses([&] { static_cast<void>(bad.events()); });
        }
        if (integer(prior, "native_mtp_priming_batch_submits")) { MtpWireReply bad(f.prepared, prior, append); refuses([&] { bad.finish(with(appended, "native_mtp_priming_batch_submits", Json::integer(0))); }); }
        refuses([&] { MtpWireReply bad(f.prepared, mtp_fixture::depth2_ready(), append); });
        const auto job = mtp_fixture::scope_job(generation_job(8), 1, true, 8); MtpWireReply output(f.prepared, appended, job);
        const auto done = f(job, [&](const Json& event) { output.event(event); }); output.finish(done);
        check(output.events().size() >= 8 && integer(done, "known_qsa_rows_physical_tiles") == 12 * integer(done, "native_mtp_rounds") &&
            integer(done, "known_qsa_rows_logical_rows") == 36 * integer(done, "native_mtp_rounds"), "input512 changed the three-row verifier counters");
    }
}
void sessions() {
    for (uint32_t output : {8u, 64u}) for (uint32_t length : {1u, 3u, 511u, 512u, 513u, 1025u}) {
#ifndef TOKSPUR_PRIVATE_MTP_OUTPUT64
        if (output == 64) continue;
#endif
        mtp_fixture::Fake f; f.prepared = f.previous = ready(output);
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, f.prepared);
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
        const auto request = with(chat(std::string(length, 'A'), false, 65), "temperature", number(1));
        const auto result = session.complete(request, "input512", [&](const Json& job, const auto& callback) { return f(job, callback); }, sink);
        size_t inputs = 0, outputs = 0; uint32_t seen = 0;
        for (const auto& packet : f.packets) {
            const auto n = packet.find("tokens")->array().size();
            if (n) { check(n == std::min(512u, length - seen), "HTTP known inputs were not packetised at512"); seen += uint32_t(n); ++inputs; }
            else { check(integer(packet, "generate") <= output, "input512 widened output packets"); ++outputs; }
        }
        const auto& details = *result.find("tokspur_private");
        check(seen == length && inputs == (length + 511) / 512 && outputs == (65 + output - 1) / output, "HTTP input/output packet counts differ");
        check(integer(details, "native_mtp_input_packet_max") == 512 && integer(details, "maximum_known_input_width") == std::min(length, 512u) &&
            details.find("native_prefill_dense")->string() == "f16" && !details.find("performance_qualified")->boolean(), "HTTP experimental input512 identity differs");
        session.invalidate(); const auto before = f.packets.size();
        static_cast<void>(session.complete(request, "input512-cold-recovery", [&](const Json& job, const auto& callback) { return f(job, callback); }, sink));
        check(f.packets[before].find("reset")->boolean() && f.generation_id == 2, "input512 cold recovery lost reset/new generation identity");
    }
    mtp_fixture::Fake stopped; stopped.prepared = stopped.previous = ready();
    ServeSession cancelled(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, stopped.prepared);
    bool cancel = false;
    const ServeSink interrupted{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
    const ServeSink resumed{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    refuses([&] { cancelled.complete(request, "input512-cancel", [&](const Json& job, const auto& callback) { auto done = stopped(job, callback); cancel = true; return done; }, interrupted); });
    check(!cancelled.committed_tokens() && stopped.packets.front().find("tokens")->array().size() == 512, "cancelled chunk published serving history or used the wrong input bound");
    cancel = false; const auto before = stopped.packets.size();
    static_cast<void>(cancelled.complete(request, "input512-after-drain", [&](const Json& job, const auto& callback) { return stopped(job, callback); }, resumed));
    check(stopped.packets[before].find("reset")->boolean(), "cancelled input512 request did not cold reset before retry");
}
Json final_ready() {
    return with(with(ready(), "native_prefill_output_policy", Json("final-input-v1")), "logits_available", Json(false));
}
Json final_input(uint32_t n, bool reset, bool final) { return with(mtp_fixture::input(n, reset), "prefill_final", Json(final)); }
void final_logits_contract() {
    const auto r = final_ready(); validate_mtp_ready(r, "tiled128", true, 2);
    validate_mtp_final_input_selection(r, true); validate_mtp_final_input_selection(ready(), false);
    refuses([&] { validate_mtp_final_input_selection(r, false); });
    refuses([&] { validate_mtp_final_input_selection(ready(), true); });
    refuses([&] { validate_mtp_ready(with(r, "native_prefill_output_policy", Json("unknown")), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(with(r, "logits_available", Json(true)), "tiled128", true, 2); });
    refuses([&] { validate_mtp_ready(with(r, "native_prefill_output_policy_extra", Json(true)), "tiled128", true, 2); });
    for (uint32_t n : {1, 511, 512}) {
        mtp_fixture::Fake f; f.prepared = f.previous = r;
        const auto invoke = [&](Json job) {
            const auto previous = f.previous; MtpWireReply reply(r, previous, job);
            auto done = f(job, [&](const Json& event) { reply.event(event); }); reply.finish(done); return done;
        };
        refuses([&] { MtpWireReply bad(r, r, mtp_fixture::input(n)); });
        refuses([&] { MtpWireReply bad(r, r, with(final_input(n, true, false), "prefill_final", Json::integer(0))); });
        const auto intermediate = invoke(final_input(n, true, false));
        check(!intermediate.find("logits_available")->boolean() && !intermediate.find("continuation_allowed")->boolean() &&
            intermediate.find("hc_exact_final_logit_sha256")->is_null() && intermediate.find("final_logit_top8")->is_null(), "intermediate input claimed ready logits");
        refuses([&] { MtpWireReply bad(r, intermediate, mtp_fixture::scope_job(generation_job(3), 1, true, 3)); });
        for (const auto& bad : std::vector<Json>{with(intermediate, "hc_exact_final_logit_sha256", Json(std::string(64, 'a'))), with(intermediate, "final_logit_top8", Json(Json::Array{})), with(intermediate, "continuation_allowed", Json(true))})
            refuses([&] { const auto job = final_input(n, true, false); MtpWireReply rejected(r, r, job); rejected.finish(bad); });
        refuses([&] { MtpWireReply bad(r, without(intermediate, "native_prefill_output_policy"), final_input(1, false, true)); });
        const auto next = final_input(1, false, true), previous = f.previous;
        MtpWireReply reply(r, previous, next); const auto done = f(next, [](const Json&) {}); reply.finish(done);
        check(done.find("logits_available")->boolean() && done.find("continuation_allowed")->boolean(), "final input failed to publish ready logits");
        for (const auto& bad : std::vector<Json>{without(done, "native_prefill_output_policy"), with(done, "logits_available", Json(false)), with(done, "continuation_allowed", Json(false))})
            refuses([&] { MtpWireReply check_reply(r, previous, next); check_reply.finish(bad); });
        refuses([&] { MtpWireReply bad(r, with(intermediate, "logits_available", Json(true)), final_input(1, false, true)); });
        const auto output = invoke(mtp_fixture::scope_job(generation_job(3), 1, true, 3));
        check(output.find("logits_available")->boolean() && integer(output, "committed_prefix") == n + 4, "generation after final input lost prefix/readiness");
    }
    const auto unselected = ready();
    refuses([&] { MtpWireReply bad(unselected, unselected, final_input(1, true, false)); });
    refuses([&] { MtpWireReply bad(r, r, with(mtp_fixture::scope_job(generation_job(3), 1, true, 3), "prefill_final", Json(true))); });
}
void final_logits_sessions() {
    for (uint32_t n : {1, 511, 512, 513, 1024, 1025}) {
        mtp_fixture::Fake f; f.prepared = f.previous = final_ready();
        ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, f.prepared);
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
        const auto call = [&](std::string text) { return session.complete(with(chat(std::move(text), false, 8), "temperature", number(1)), "final-input-fixture", [&](const Json& job, const auto& callback) { return f(job, callback); }, sink); };
        const auto result = call(std::string(n, 'A')); size_t used = 0;
        for (const auto& job : f.packets) if (!job.find("tokens")->array().empty()) {
            used += job.find("tokens")->array().size();
            check(job.find("prefill_final") && job.find("prefill_final")->boolean() == (used == n), "HTTP final flag inferred from packet length rather than complete prompt boundary");
        } else check(!job.find("prefill_final"), "HTTP output carried an input-final flag");
        const auto& details = *result.find("tokspur_private");
        check(details.find("native_prefill_output_policy")->string() == "final-input-v1" && details.find("logits_available")->boolean(), "HTTP completion omitted final-input identity/readiness");
        const auto before = f.packets.size(); call(std::string(n, 'A') + std::string(8, ')'));
        check(f.packets[before].find("tokens")->array().empty(), "exact committed prompt needlessly replayed input");
        const auto append_at = f.packets.size(); const auto appended = call(std::string(n, 'A') + std::string(16, ')') + "C");
        check(f.packets[append_at].find("tokens")->array().size() == 1 && f.packets[append_at].find("prefill_final")->boolean() &&
            integer(*appended.find("usage")->find("prompt_tokens_details"), "cached_tokens") == n + 16, "final-input append lost exact prefix reuse");
    }
    mtp_fixture::Fake f; f.prepared = f.previous = final_ready(); bool cancel = false;
    ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, f.prepared);
    const ServeSink interrupted{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
    const ServeSink resumed{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }};
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    refuses([&] { session.complete(request, "deferred-cancel", [&](const Json& job, const auto& callback) { auto done = f(job, callback); cancel = true; return done; }, interrupted); });
    check(!session.committed_tokens() && !f.previous.find("logits_available")->boolean(), "cancelled intermediate input exposed history/logits");
    cancel = false; const auto before = f.packets.size();
    session.complete(request, "deferred-reset", [&](const Json& job, const auto& callback) { return f(job, callback); }, resumed);
    check(f.packets[before].find("reset")->boolean() && !f.packets[before].find("prefill_final")->boolean() && f.previous.find("logits_available")->boolean(), "deferred cancellation failed cold reset/final recovery");
}
void options(scalar_fixture::Files& files, bool selected) {
    files.options.mtp_sidecar = files.directory / "mtp-sidecar"; { std::ofstream sidecar(files.options.mtp_sidecar); sidecar << "generated CPU-only fixture\n"; }
    files.options.mtp_depth = 2; files.options.mtp_request_carry = true; files.options.mtp_draft_attention = "tiled128"; files.options.prefill_input512 = selected;
}
#ifndef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
void hc_down_uncompiled_workers(const std::filesystem::path& executable) {
    for (bool mode : {false,true}) {
        scalar_fixture::Files files(executable,mode?"hc-down-uncompiled-on":"hc-down-uncompiled-off"); options(files); files.options.prefill_hc_down_f32=mode;
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory/"started"),"uncompiled HC selector presence started a child");
    }
}
#endif
void workers(const std::filesystem::path& executable) {
    for (const auto mode : {"prefill512-ok", "prefill512-drift", "prefill512-missing-done", "prefill512-unknown-done", "prefill512-unsolicited", "prefill512-missing-ready", "prefill512-bad-priming-ready"}) {
        scalar_fixture::Files files(executable, mode); options(files, std::string_view(mode) != "prefill512-unsolicited");
        if (std::string_view(mode) == "prefill512-unsolicited" || std::string_view(mode) == "prefill512-missing-ready" || std::string_view(mode) == "prefill512-bad-priming-ready") { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); worker.request(mtp_fixture::input(512), Clock::now() + 3s, [] { return false; });
        worker.request(mtp_fixture::input(1, false), Clock::now() + 3s, [] { return false; }); size_t callbacks = 0;
        const auto generate = [&] { worker.request(mtp_fixture::scope_job(generation_job(8), 1, true, 8), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); };
        if (std::string_view(mode) == "prefill512-ok") { generate(); check(callbacks >= 8 && worker.alive(), "owned input512 lost admitted output"); worker.close(); }
        else { refuses(generate); check(!callbacks && !worker.alive(), "input512 metadata drift released events or retained child"); }
    }
    for (uint32_t incompatible : {0, 1, 2, 3}) {
        scalar_fixture::Files files(executable, "prefill512-incompatible"); options(files);
        if (!incompatible) files.options.mtp_sidecar.clear();
        if (incompatible == 1) files.options.sampled_prefill = true;
        if (incompatible == 2) files.options.verify8 = true;
        if (incompatible == 3) files.options.weight_lanes = files.directory / "lanes";
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "incompatible input512 launched native work");
    }
}
void final_logits_workers(const std::filesystem::path& executable) {
    for (bool selected : {false, true}) {
        scalar_fixture::Files files(executable, selected ? "final-logits-argv-on" : "final-logits-argv-off"); options(files); files.options.prefill_final_logits = selected;
        OwnedWorker worker(files.options); const auto first = selected ? final_input(512, true, false) : mtp_fixture::input(512);
        const auto intermediate = worker.request(first, Clock::now() + 3s, [] { return false; });
        check(mtp_final_input_logits(worker.ready()) == selected && (!selected || !intermediate.find("logits_available")->boolean()), "actual worker final-input argv or readiness differs");
        worker.request(selected ? final_input(1, false, true) : mtp_fixture::input(1, false), Clock::now() + 3s, [] { return false; }); size_t callbacks = 0;
        worker.request(mtp_fixture::scope_job(generation_job(8), 1, true, 8), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; });
        check(callbacks >= 8, "final-input worker lost admitted generation"); worker.close();
    }
    for (const auto mode : {"final-logits-ready-drift", "final-logits-done-drift", "final-logits-false-ready"}) {
        scalar_fixture::Files files(executable, mode); options(files); files.options.prefill_final_logits = true;
        if (std::string_view(mode) == "final-logits-ready-drift") { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t callbacks = 0;
        refuses([&] { worker.request(final_input(512, true, false), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(!callbacks && !worker.alive(), "unready/drifting completion left admitted callbacks or live child");
    }
    scalar_fixture::Files files(executable, "final-logits-incompatible"); options(files, false); files.options.prefill_final_logits = true;
    refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "final-input mode without input512 launched a child");
}
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
void old_order_controls(const std::filesystem::path& executable) {
    const auto r = mtp_fixture::old_order_fields(final_ready()); validate_mtp_ready(r, "tiled128", true, 2);
    validate_mtp_gdn_old_order_selection(r, true); validate_mtp_gdn_old_order_selection(ready(), false);
    refuses([&] { validate_mtp_gdn_old_order_selection(r, false); });
    for (const auto& bad : std::vector<Json>{without(r, "native_gdn_old_order_version"), with(r, "native_gdn_old_order_extra", Json(true)), with(r, "native_prefill_dense", Json("f16")), with(r, "native_prefill_chunks", Json("cs64")), with(r, "native_gdn_old_order_encoded_calls", Json::integer(1))}) refuses([&] { validate_mtp_ready(bad, "tiled128", true, 2); });
    mtp_fixture::Fake f; f.prepared = f.previous = r;
    const auto lost_job = final_input(512, true, false); const auto lost = f(lost_job, [](const Json&) {});
    check(integer(lost, "native_gdn_old_order_request_encoded_calls") == 36, "cancelled packet did not retain actual encoded calls");
    const auto reset_job = final_input(512, true, false); const auto reset = f(reset_job, [](const Json&) {});
    MtpWireReply recovered(r, r, reset_job, true); recovered.finish(reset);
    check(integer(reset, "native_gdn_old_order_encoded_calls") == 72 && integer(reset, "native_gdn_old_order_request_encoded_calls") == 36, "cold reset attributed stale lifetime calls to current packet");
    refuses([&] { MtpWireReply strict(r, r, reset_job); strict.finish(reset); });
    refuses([&] { MtpWireReply invalid(r, reset, final_input(1, false, true), true); });
    refuses([&] { MtpWireReply drift(r, without(reset, "native_gdn_old_order_kernel"), final_input(1, false, true)); });
    const auto final_job = final_input(1, false, true); const auto previous = f.previous; const auto final = f(final_job, [](const Json&) {});
    MtpWireReply final_reply(r, previous, final_job); final_reply.finish(final);
    check(!integer(final, "native_gdn_old_order_request_encoded_calls"), "ordinary one-token input was redirected to old-order recurrence");
    for (bool selected : {false, true}) {
        scalar_fixture::Files files(executable, selected ? "gdn-old-order-argv-on" : "gdn-old-order-argv-off"); options(files); files.options.prefill_dense = "f32"; files.options.prefill_grouped = "f32"; files.options.prefill_final_logits = true; files.options.gdn_old_order_batch = selected;
        OwnedWorker worker(files.options); worker.request(final_input(512, true, false), Clock::now() + 3s, [] { return false; }); worker.request(final_input(1, false, true), Clock::now() + 3s, [] { return false; });
        const auto out = worker.request(mtp_fixture::scope_job(generation_job(8), 1, true, 8), Clock::now() + 3s, [] { return false; });
        check(mtp_gdn_old_order(worker.ready()) == selected && (!selected || !integer(out, "native_gdn_old_order_request_encoded_calls")), "actual old-order argv or ordinary output route differs"); worker.close();
    }
    for (const auto mode : {"gdn-old-order-ready-drift", "gdn-old-order-done-drift", "gdn-old-order-call-drift"}) {
        scalar_fixture::Files files(executable, mode); options(files); files.options.prefill_dense = "f32"; files.options.prefill_grouped = "f32"; files.options.gdn_old_order_batch = true;
        if (std::string_view(mode) == "gdn-old-order-ready-drift") { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); size_t callbacks = 0; refuses([&] { worker.request(mtp_fixture::input(512), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(!callbacks && !worker.alive(), "old-order drift released callbacks or retained child");
    }
    for (uint32_t invalid : {0, 1, 2}) {
        scalar_fixture::Files files(executable, "gdn-old-order-invalid"); options(files, invalid != 0); files.options.gdn_old_order_batch = true; files.options.prefill_dense = invalid == 1 ? "f16" : "f32"; files.options.prefill_grouped = invalid == 2 ? "serial" : "f32";
        refuses([&] { OwnedWorker bad(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "unsupported old-order route started a child");
    }
    mtp_fixture::Fake endpoint; endpoint.prepared = endpoint.previous = r; bool cancel = false;
    ServeSession session(session_codec(), "tokspur", "direct", Json(), false, Json(), false, ToolEnumValidation::strict, false, r);
    const auto request = with(chat(std::string(513, 'A'), false, 8), "temperature", number(1));
    const ServeSink interrupted{[](const Json&) {}, [&] { return cancel; }, [] { return uint64_t(44); }};
    refuses([&] { session.complete(request, "old-order-cancel", [&](const Json& job, const auto& callback) { auto done = endpoint(job, callback); cancel = true; return done; }, interrupted); });
    cancel = false; std::vector<Json> traces; const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { traces.push_back(row); }};
    const auto result = session.complete(request, "old-order-recovery", [&](const Json& job, const auto& callback) { return endpoint(job, callback); }, sink); const auto& details = *result.find("tokspur_private");
    check(integer(details, "native_gdn_old_order_admitted_encoded_calls") == 36 && endpoint.old_order_encoded == 72 && integer(*traces.back().find("gdn_old_order"), "admitted_encoded_calls") == 36, "HTTP/private trace counted cancelled lifetime work as admitted request work");
}
#endif
void precision_workers(const std::filesystem::path& executable) {
    for (const auto dense : {"f16", "f32"}) {
        std::vector<std::optional<std::string>> modes{std::nullopt};
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        modes.push_back("serial"); modes.push_back("f32");
#endif
        for (const auto& grouped : modes) {
            scalar_fixture::Files files(executable, std::string("precision-argv-") + dense + "-" + grouped.value_or("absent")); options(files);
            files.options.prefill_dense = dense;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            files.options.prefill_grouped = grouped;
#endif
            OwnedWorker worker(files.options);
            check(worker.ready().find("native_prefill_dense")->string() == dense, "owned child did not receive selected dense precision");
            worker.request(mtp_fixture::input(3), Clock::now() + 3s, [] { return false; });
            size_t callbacks = 0;
            worker.request(mtp_fixture::scope_job(generation_job(8), 1, true, 8), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; });
            check(callbacks >= 8 && worker.alive(), "selected precision child lost generation callbacks"); worker.close();
        }
    }
    for (const auto mode : {"precision-dense-ready-drift", "precision-dense-drift"
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        , "precision-grouped-ready-missing", "precision-grouped-ready-unsolicited", "precision-grouped-drift", "precision-grouped-missing", "precision-grouped-unknown"
#endif
    }) {
        scalar_fixture::Files files(executable, mode); options(files); files.options.prefill_dense = "f32";
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (std::string_view(mode).starts_with("precision-grouped") && std::string_view(mode) != "precision-grouped-ready-unsolicited") files.options.prefill_grouped = "f32";
#endif
        if (std::string_view(mode).find("ready") != std::string_view::npos) { refuses([&] { OwnedWorker bad(files.options); }); continue; }
        OwnedWorker worker(files.options); worker.request(mtp_fixture::input(3), Clock::now() + 3s, [] { return false; }); size_t callbacks = 0;
        refuses([&] { worker.request(mtp_fixture::scope_job(generation_job(8), 1, true, 8), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(!callbacks && !worker.alive(), "precision drift released unadmitted callbacks or left child alive");
    }
    for (const auto value : {"f16", "f32", "unknown"}) {
        scalar_fixture::Files files(executable, "precision-invalid-dense"); options(files, std::string_view(value) == "unknown"); files.options.prefill_dense = value;
        refuses([&] { OwnedWorker invalid(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "invalid dense consent launched a child");
    }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    for (const auto value : {"serial", "f32", "unknown"}) {
        scalar_fixture::Files files(executable, "precision-invalid-grouped"); options(files, std::string_view(value) == "unknown"); files.options.prefill_grouped = value;
        refuses([&] { OwnedWorker invalid(files.options); }); check(!std::filesystem::exists(files.directory / "started"), "invalid grouped consent launched a child");
    }
#endif
}
}
#ifndef TOKSPUR_HTTP_PREFILL512_TEST_ENTRY
#define TOKSPUR_HTTP_PREFILL512_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_PREFILL512_TEST_ENTRY(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return mtp_fixture::target(argc, argv);
    const auto inherited = inherited_http_concurrent_entry(argc, argv); if (inherited) return inherited;
    try {
#ifndef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        prefill512_fixture::hc_down_uncompiled_workers(std::filesystem::canonical(argv[0]));
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        prefill512_fixture::old_order_controls(std::filesystem::canonical(argv[0]));
#endif
        prefill512_fixture::staged_dense_controls(std::filesystem::canonical(argv[0]));
        prefill512_fixture::final_logits_contract(); prefill512_fixture::final_logits_sessions(); prefill512_fixture::final_logits_workers(std::filesystem::canonical(argv[0])); prefill512_fixture::metadata(); prefill512_fixture::precision_metadata(); prefill512_fixture::wire(); prefill512_fixture::precision_wire_sessions(); prefill512_fixture::private_prefix_trace(); prefill512_fixture::private_native_timing_trace(); prefill512_fixture::sessions(); prefill512_fixture::workers(std::filesystem::canonical(argv[0])); prefill512_fixture::precision_workers(std::filesystem::canonical(argv[0]));
        std::cout << "HTTP input512 CPU checks=" << checks << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
