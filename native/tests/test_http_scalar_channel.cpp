// CPU-generated transport records only. No model, Metal or tokenizer assets.
#define TOKSPUR_HTTP_FAST_MATH_TEST_ENTRY inherited_fast_math_checks
#include "test_http_fast_math.cpp"
#undef TOKSPUR_HTTP_FAST_MATH_TEST_ENTRY
#include <map>

#ifndef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
#error This fixture requires the separate scalar-channel HTTP build.
#endif
#if defined(TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK) && !defined(TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC)
#error The closed scalar-channel build must omit phase-clock instrumentation.
#endif

namespace scalar_fixture {
Json ready() {
    auto value = fast_fixture::fields(fast_fixture::core_ready(), true);
    for (const auto name : {"branch8_session", "broker_protocol", "broker_batch_width"}) value = without(value, name);
    for (const auto name : {"known_token_capacity", "checkpoint_slots", "checkpoint_combined_budget_bytes", "checkpoint_host_bytes",
        "checkpoint_store_host_bytes", "checkpoint_gpu_bytes", "broker_epoch", "lookup_verification_budget_bytes",
        "lookup_caller_host_bytes", "verification_host_bytes", "verification_gpu_bytes", "lookup_verification_charged_bytes",
        "verification_tape_payload_bytes"}) value = with(std::move(value), name, Json::integer(0));
    for (const auto name : {"known_residual_reuse", "prompt_lookup", "verification_head_many", "verification8_branch_reuse",
        "checkpoint_broker", "allow_prompt_lookup_experiments"}) value = with(std::move(value), name, Json(false));
    return with(std::move(value), "verification_max_tokens", Json::integer(2));
}
Json sampled_policy() { return parse_json(R"({"temperature":1,"top_k":20,"top_p":0.95,"min_p":0})"); }
Json input(bool reset = true) {
    auto job = parse_json(R"({"tokens":[1],"generate":0,"reset":false})");
    job = with(std::move(job), "reset", Json(reset));
    if (reset) { job = with(std::move(job), "sampling", sampled_policy()); job = with(std::move(job), "seed", Json::integer(44)); }
    return job;
}
Json generation() { return parse_json(R"({"tokens":[],"generate":1,"reset":false,"stop_ids":[1000]})"); }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
Json lane_fields(Json value) {
    for (const auto& [name, field] : Json::Object{{"weight_lanes_schema", Json::integer(1)},
        {"weight_lanes_manifest_sha256", Json(std::string(64, 'a'))}, {"weight_lanes_tensor_count", Json::integer(2)},
        {"weight_lanes_file_bytes", Json::integer(4096)}, {"weight_lanes_original_payload_bytes_skipped", Json::integer(2048)}})
        value = with(std::move(value), name, field);
    return value;
}
Json lane_auth_fields(Json value, uint64_t workers = 4) {
    const auto effective = std::min(workers, integer(value, "weight_lanes_tensor_count"));
    value = with(std::move(value), "lane_auth_workers_requested", Json::integer(workers));
    return with(std::move(value), "lane_auth_workers", Json::integer(effective));
}
void lane_contracts() {
    const auto absent = ready(), selected = lane_fields(absent);
    validate_owned_weight_lanes_ready(absent, false); ++checks;
    validate_owned_weight_lanes_ready(selected, true); ++checks;
    refuses([&] { validate_owned_weight_lanes_ready(absent, true); });
    refuses([&] { validate_owned_weight_lanes_ready(selected, false); });
    for (const auto name : owned_weight_lane_fields) {
        refuses([&] { validate_owned_weight_lanes_ready(without(selected, name), true); });
        for (const auto& invalid : {Json(), Json(true), Json(Json::Array{})})
            refuses([&] { validate_owned_weight_lanes_ready(with(selected, name, invalid), true); });
        const auto changed = name == "weight_lanes_manifest_sha256" ? Json(std::string(64, 'b')) :
            Json::integer(integer(selected, name) + 1);
        refuses([&] { validate_owned_weight_lanes_done(selected, with(selected, name, changed)); });
    }
    for (const auto& invalid : {Json(""), Json(std::string(63, 'a')), Json(std::string(64, 'A')), Json(std::string(64, 'g')), Json::integer(9)})
        refuses([&] { validate_owned_weight_lanes_ready(with(selected, "weight_lanes_manifest_sha256", invalid), true); });
    for (const auto name : {"weight_lanes_schema", "weight_lanes_tensor_count", "weight_lanes_file_bytes", "weight_lanes_original_payload_bytes_skipped"})
        for (const auto& invalid : {parse_json("-1"), parse_json("1.0"), parse_json("1e0"), Json("1")})
            refuses([&] { validate_owned_weight_lanes_ready(with(selected, name, invalid), true); });
    for (const auto& [name, bad] : std::initializer_list<std::pair<const char*, uint64_t>>{
        {"weight_lanes_schema", 0}, {"weight_lanes_schema", 2}, {"weight_lanes_tensor_count", 0}, {"weight_lanes_tensor_count", 1025},
        {"weight_lanes_file_bytes", 0}, {"weight_lanes_file_bytes", 32'000'000'001},
        {"weight_lanes_original_payload_bytes_skipped", 0}, {"weight_lanes_original_payload_bytes_skipped", 4096}})
        refuses([&] { validate_owned_weight_lanes_ready(with(selected, name, Json::integer(bad)), true); });
    refuses([&] { validate_owned_weight_lanes_ready(with(selected, "weight_lanes_unknown", Json(true)), true); });
    refuses([&] { validate_owned_weight_lanes_ready(with(absent, "weight_lanes_unknown", Json(true)), false); });
    for (const auto event : {"done", "control_done"}) {
        validate_owned_weight_lanes_done(selected, with(selected, "event", Json(event))); ++checks;
        validate_owned_weight_lanes_done(absent, with(absent, "event", Json(event))); ++checks;
        refuses([&] { validate_owned_weight_lanes_done(absent, with(selected, "event", Json(event))); });
        refuses([&] { validate_owned_weight_lanes_done(selected, with(absent, "event", Json(event))); });
    }
    for (const uint32_t workers : {1u, 4u, 16u}) {
        const auto auth = lane_auth_fields(selected, workers);
        validate_owned_weight_lanes_ready(auth, true, workers); ++checks;
        validate_owned_weight_lanes_done(auth, with(auth, "event", Json("done"))); ++checks;
        refuses([&] { validate_owned_weight_lanes_ready(auth, true); });
        refuses([&] { validate_owned_weight_lanes_ready(auth, false, workers); });
        refuses([&] { validate_owned_weight_lanes_ready(selected, true, workers); });
        for (const auto name : owned_lane_auth_fields) {
            refuses([&] { validate_owned_weight_lanes_ready(without(auth, name), true, workers); });
            for (const auto& bad : {Json(), Json(true), Json("4"), parse_json("4.0"), parse_json("-1")})
                refuses([&] { validate_owned_weight_lanes_ready(with(auth, name, bad), true, workers); });
            refuses([&] { validate_owned_weight_lanes_done(auth, without(auth, name)); });
            refuses([&] { validate_owned_weight_lanes_done(auth, with(auth, name, Json::integer(3))); });
        }
        refuses([&] { validate_owned_weight_lanes_done(selected, auth); });
        refuses([&] { validate_owned_weight_lanes_done(auth, selected); });
        refuses([&] { validate_owned_weight_lanes_ready(with(auth, "lane_auth_unknown", Json(true)), true, workers); });
    }
    for (const auto invalid : {0u, 17u}) {
        refuses([&] { validate_owned_weight_lanes_ready(selected, true, invalid); });
        refuses([&] { validate_owned_weight_lanes_ready(lane_auth_fields(selected, invalid), true, invalid); });
    }
}
#endif
struct Fake {
    Json prepared = ready(), previous = prepared, selected = sampled_policy();
    uint64_t calls = 0, hc_calls = 0, words = 0, seed = 44;
    uint32_t prefix = 0;
    Json operator()(const Json& job, const std::function<void(const Json&)>& callback) {
        if (job.find("reset")->boolean()) { prefix = 0; selected = *job.find("sampling"); seed = integer(job, "seed"); words = 0; }
        const auto outputs = integer(job, "generate"); prefix += uint32_t(job.find("tokens")->array().size() + outputs);
        calls += 12; hc_calls += 97;
        if (outputs && std::stod(std::string(selected.find("temperature")->number_lexeme())) > 0) ++words;
        auto sampled = selected.object(); sampled.emplace_back("valid", Json(true));
        sampled.emplace_back("seed", Json::integer(seed)); sampled.emplace_back("entropy_prefix", Json::integer(prefix));
        sampled.emplace_back("entropy_words", Json::integer(words));
        auto done = with(prepared, "event", Json("done"));
        done = with(std::move(done), "committed_prefix", Json::integer(prefix));
        for (const auto name : {"qsa_stage_v8_calls", "qsa_stage_v8_extra_dispatches", "qsa_channel_calls", "qsa_channel_extra_dispatches"})
            done = with(std::move(done), name, Json::integer(calls));
        done = with(std::move(done), "qsa_stage_v8_physical_dispatches", Json::integer(2 * calls));
        for (const auto name : {"qsa_stage_v8_request_calls", "qsa_channel_request_calls"}) done = with(std::move(done), name, Json::integer(12));
        done = with(std::move(done), "sampling", Json(std::move(sampled)));
        done = with(std::move(done), "hc_exact_routes", hc({hc_calls, 0, 0, 0, 0}));
        done = with(std::move(done), "hc_exact_request_routes", hc({97, 0, 0, 0, 0}));
        done = with(std::move(done), "status", Json(outputs ? "length" : "prefilled"));
        done = with(std::move(done), "finish_reason", outputs ? Json("length") : Json());
        done = with(std::move(done), "stop_id", Json()); done = with(std::move(done), "continuation_allowed", Json(true));
        done = with(std::move(done), "prefill_wall_ms", number(0));
        done = with(std::move(done), "generated_ids", Json(outputs ? Json::Array{Json::integer(42)} : Json::Array{}));
        if (outputs) callback(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(42)},
            {"committed_prefix", Json::integer(prefix)}, {"entropy_words", Json::integer(words)}}));
        previous = done; return done;
    }
};
void contracts() {
    const auto original = ready(); validate_scalar_channel_ready(original); ++checks;
    // Valid base PLE metadata does not imply the optional channel family.
    for (const auto mode : {"plain", "signed-root"}) {
        const auto base = parse_json(std::string("{\"event\":\"ready\",\"ple_gate_transform\":\"") + mode + "\",\"ple_gate_transform_schema\":1}");
        check(!verify8_fast_math(base).has_value());
        refuses([&] { verify8_fast_math(with(base, "ple_gate_unknown", Json::integer(0))); });
        refuses([&] { verify8_fast_math(without(base, "ple_gate_transform_schema")); });
    }
    for (const auto name : fast_fixture::names) refuses([&] { validate_scalar_channel_ready(without(original, name)); });
    for (const auto& [name, value] : Json::Object{{"known_token_capacity",Json::integer(8)}, {"known_residual_reuse",Json(true)},
        {"checkpoint_slots",Json::integer(1)}, {"prompt_lookup",Json(true)}, {"verification_max_tokens",Json::integer(8)},
        {"qsa_stage_v8_calls",Json::integer(12)}, {"hc_exact_mode",Json("off")}, {"qsa_channel_chronological_mode",Json("off")},
        {"gpu_allocation_count",Json::integer(0)}, {"allow_request_kernel_experiments",Json(true)}})
        refuses([&] { validate_scalar_channel_ready(with(original, name, value)); });
    Fake fake; const auto job = input(); const auto first = fake(job, [](const Json&) { check(false); });
    validate_scalar_channel_done(original, original, job, first, nullptr); ++checks;
    std::optional<Json> token; const auto next = generation();
    const auto second = fake(next, [&](const Json& v) { token = v; });
    validate_scalar_channel_done(original, first, next, second, &*token); ++checks;
    for (const auto& [name, value] : Json::Object{{"qsa_channel_request_calls",Json::integer(11)},
        {"qsa_stage_v8_calls",Json::integer(12)}, {"gpu_allocation_count",Json::integer(integer(original,"gpu_allocation_count")+1)},
        {"committed_prefix",Json::integer(99)}, {"hc_exact_request_routes",hc()}, {"qsa_channel_unregistered",Json::integer(0)}})
        refuses([&] { validate_scalar_channel_done(original, first, next, with(second, name, value), &*token); });
    refuses([&] { validate_scalar_channel_done(original, first, next, second, nullptr); });
    const auto wrong_token = with(*token, "id", Json::integer(43));
    refuses([&] { validate_scalar_channel_done(original, first, next, second, &wrong_token); });
    auto wrong_sampling = with(*second.find("sampling"), "temperature", number(0));
    refuses([&] { validate_scalar_channel_done(original, first, next, with(second,"sampling",wrong_sampling), &*token); });
    // T>0 may still have deterministic support and therefore use zero words.
    const auto pointmass = with(second, "sampling", with(*second.find("sampling"), "entropy_words", Json::integer(0)));
    const auto zero_token = with(*token, "entropy_words", Json::integer(0));
    validate_scalar_channel_done(original, first, next, pointmass, &zero_token); ++checks;
    for (const auto jump : {2u, 100u}) {
        const auto changed = with(second, "sampling", with(*second.find("sampling"), "entropy_words", Json::integer(jump)));
        const auto changed_token = with(*token, "entropy_words", Json::integer(jump));
        refuses([&] { validate_scalar_channel_done(original, first, next, changed, &changed_token); });
    }
    const auto greedy_before = with(first, "sampling", with(*first.find("sampling"), "temperature", number(0)));
    const auto greedy_jump = with(second, "sampling", with(*second.find("sampling"), "temperature", number(0)));
    refuses([&] { validate_scalar_channel_done(original, greedy_before, next, greedy_jump, &*token); });
    const auto greedy_zero = with(pointmass, "sampling", with(*pointmass.find("sampling"), "temperature", number(0)));
    validate_scalar_channel_done(original, greedy_before, next, greedy_zero, &zero_token); ++checks;
    const auto input_jump = with(first, "sampling", with(*first.find("sampling"), "entropy_words", Json::integer(1)));
    refuses([&] { validate_scalar_channel_done(original, original, job, input_jump, nullptr); });
    const auto entropy_mismatch = with(*token, "entropy_words", Json::integer(0));
    refuses([&] { validate_scalar_channel_done(original, first, next, second, &entropy_mismatch); });
    const auto no_entropy = without(*token, "entropy_words");
    refuses([&] { validate_scalar_channel_done(original, first, next, second, &no_entropy); });
    // Reset rewinds model/RNG ancestry, never lifetime physical work counters.
    const auto reset = input(); const auto third = fake(reset, [](const Json&) { check(false); });
    validate_scalar_channel_done(original, second, reset, third, nullptr); ++checks;
}
int target(int argc, char** argv) {
    std::filesystem::path path; std::map<std::string,std::string> flags;
    for (int i=2;i<argc;++i) if (std::string_view(argv[i]).starts_with("--")) {
        const std::string key(argv[i]); if (++i==argc || !flags.emplace(key,argv[i]).second) return 91;
        if (key=="--policy") path=argv[i];
    }
    if (path.empty()) return 92;
    for (const auto& [key,value] : std::map<std::string,std::string>{{"--stack-execution","direct"},{"--small-gemv-splits","16"},
        {"--block-gemv-formats","3"},{"--hc-exact","combined"},{"--ple-gate-transform","signed-root"},
        {"--qsa-stage-v8","on"},{"--qsa-channel-chronological","on"}}) if (flags[key]!=value) return 93;
    for (const auto key : {"--prompt-lookup","--worker-broker","--worker-broker8","--known-token-capacity","--checkpoint-slots"})
        if (flags.contains(key)) return 94;
    { std::ofstream out(path.parent_path()/"started"); out<<"explicit scalar-channel arguments\n"; }
    Fake fake;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (flags.contains("--weight-lanes")) {
        if (flags.at("--weight-lanes") != (path.parent_path()/"lanes").string()) return 95;
        fake.prepared = lane_fields(fake.prepared);
    }
    if (flags.contains("--weight-lanes-auth-workers")) {
        if (!flags.contains("--weight-lanes")) return 96;
        const auto count = parse_json(flags.at("--weight-lanes-auth-workers")).unsigned_integer();
        if (!count || count > 16) return 96;
        fake.prepared = lane_auth_fields(fake.prepared, count);
    }
    if (path.filename()=="lane-auth-unsolicited") fake.prepared=lane_auth_fields(fake.prepared);
    if (path.filename()=="lane-unsolicited") fake.prepared=lane_fields(fake.prepared);
#endif
    auto prepared=fake.prepared;
    if (path.filename()=="bad-ready") prepared=with(std::move(prepared),"known_token_capacity",Json::integer(8));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (path.filename()=="lane-missing-ready") prepared=without(prepared,"weight_lanes_manifest_sha256");
    if (path.filename()=="lane-auth-missing-ready") prepared=without(prepared,"lane_auth_workers");
#endif
    emit(prepared); std::string line;
    while (std::getline(std::cin,line)) {
        { std::ofstream out(path.parent_path()/"commands",std::ios::app); out<<"step\n"; }
        const auto job=parse_json(line);auto done=fake(job,[](const Json& v){emit(v);});
        if (integer(job,"generate")&&path.filename()=="bad-cost") done=with(std::move(done),"qsa_channel_request_calls",Json::integer(11));
        if (integer(job,"generate")&&path.filename()=="bad-resource") done=with(std::move(done),"gpu_allocation_count",Json::integer(integer(done,"gpu_allocation_count")+1));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (integer(job,"generate")&&path.filename()=="lane-drift") done=with(std::move(done),"weight_lanes_manifest_sha256",Json(std::string(64,'b')));
        if (integer(job,"generate")&&path.filename()=="lane-missing-done") done=without(done,"weight_lanes_schema");
        if (integer(job,"generate")&&path.filename()=="lane-unknown-done") done=with(std::move(done),"weight_lanes_unknown",Json(true));
        if (integer(job,"generate")&&path.filename()=="lane-appeared-done") done=lane_fields(std::move(done));
        if (integer(job,"generate")&&path.filename()=="lane-auth-drift") done=with(std::move(done),"lane_auth_workers_requested",Json::integer(8));
        if (integer(job,"generate")&&path.filename()=="lane-auth-missing-done") done=without(done,"lane_auth_workers");
        if (integer(job,"generate")&&path.filename()=="lane-auth-wrongtype-done") done=with(std::move(done),"lane_auth_workers",Json(true));
        if (integer(job,"generate")&&path.filename()=="lane-auth-appeared-done") done=lane_auth_fields(std::move(done));
#endif
        emit(done);
    }
    return 0;
}
struct Files {
    std::filesystem::path directory;
    OwnedWorkerOptions options;
    Files(const std::filesystem::path& executable,std::string_view mode) {
        auto pattern=(std::filesystem::temp_directory_path()/"tokspur-scalar-channel-XXXXXX").string();
        std::vector<char> data(pattern.begin(),pattern.end());data.push_back('\0');
        const auto* created=mkdtemp(data.data());if(!created)throw std::runtime_error("scalar fixture directory");directory=created;
        options.executable=directory/"scalar-channel-target";
        std::filesystem::copy_file(executable,options.executable);
        options.policy=directory/mode;options.library=directory/"library";
        for(const auto& file:{options.policy,options.library}){std::ofstream out(file);out<<"generated CPU fixture\n";}
        for(unsigned i=0;i<4;++i){auto file=directory/("shard"+std::to_string(i));{std::ofstream out(file);out<<i;}options.shards.push_back(file);}
        options.scalar_channel=true;options.stage8_present=true;options.stage8_enabled=true;options.fast_channel=true;
        options.startup_timeout=3s;options.step_timeout=3s;options.drain_timeout=300ms;
    }
    ~Files(){std::error_code ignored;std::filesystem::remove_all(directory,ignored);}
};
void workers(const std::filesystem::path& executable) {
    for(const auto mode:{"normal","bad-ready","bad-cost","bad-resource"}) {
        Files files(executable,mode);
        if(std::string_view(mode)=="bad-ready"){refuses([&]{OwnedWorker worker(files.options);});continue;}
        OwnedWorker worker(files.options);size_t callbacks=0;
        const auto submit=[&]{worker.request(input(),Clock::now()+3s,[]{return false;});};
        submit();
        const auto generate=[&]{worker.request(generation(),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;});};
        if(std::string_view(mode)!="normal"){refuses(generate);check(!worker.alive()&&!callbacks,"unadmitted scalar token was published");continue;}
        generate();
        check(callbacks==1&&worker.alive());
        refuses([&]{worker.request(with(generation(),"known_prefill",Json(true)),Clock::now()+3s,[]{return false;});});
        check(worker.alive());worker.close();
    }
    // A callback can abort after physical advancement but before successful
    // publication. Its dedicated cancellation catch must still require reset.
    {
        Files files(executable,"normal");OwnedWorker worker(files.options);
        worker.request(input(),Clock::now()+3s,[]{return false;});bool cancelled=false;
        try{worker.request(generation(),Clock::now()+3s,[]{return false;},[](const Json&){throw WorkerCancelled("generated publication cancellation");});}
        catch(const WorkerCancelled&){cancelled=true;}
        check(cancelled&&worker.alive());
        refuses([&]{worker.request(generation(),Clock::now()+3s,[]{return false;});});
        worker.request(input(),Clock::now()+3s,[]{return false;});size_t published=0;
        worker.request(generation(),Clock::now()+3s,[]{return false;},[&](const Json&){++published;});
        check(published==1&&worker.alive());worker.close();
    }
    for(unsigned change=0;change<8;++change){
        Files files(executable,"normal");auto& o=files.options;
        if(change==0)o.verify8=true;else if(change==1)o.branch8=true;else if(change==2){o.checkpoint_slots=1;o.checkpoint_budget_bytes=512;}
        else if(change==3)o.stack_execution="recurrent-replay";else if(change==4)o.stage8_present=false;
        else if(change==5)o.stage8_enabled=false;else if(change==6)o.fast_channel=false;else o.scalar_channel=false;
        check(!scalar_channel_transport(o));refuses([&]{OwnedWorker worker(o);});
        check(!std::filesystem::exists(files.directory/"started"),"invalid scalar startup launched child");
    }
}
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
void lane_workers(const std::filesystem::path& executable) {
    for (const auto mode : {"lane-normal", "lane-unsolicited", "lane-missing-ready", "lane-drift", "lane-missing-done", "lane-unknown-done", "lane-appeared-done"}) {
        Files files(executable, mode);
        if (std::string_view(mode) != "lane-unsolicited" && std::string_view(mode) != "lane-appeared-done") {
            files.options.weight_lanes=files.directory/"lanes";
            std::ofstream output(files.options.weight_lanes); output<<"generated CPU-only lane identity\n";
        }
        if (std::string_view(mode)=="lane-unsolicited" || std::string_view(mode)=="lane-missing-ready") {
            refuses([&] { OwnedWorker worker(files.options); }); continue;
        }
        OwnedWorker worker(files.options); size_t callbacks=0;
        worker.request(input(),Clock::now()+3s,[]{return false;});
        const auto generate=[&] { worker.request(generation(),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;}); };
        if (std::string_view(mode)!="lane-normal") {
            refuses(generate);check(!worker.alive()&&!callbacks,"unadmitted lane identity published a scalar token"); continue;
        }
        generate();check(callbacks==1&&worker.alive());worker.close();
    }
    for (unsigned invalid=0;invalid<4;++invalid) {
        Files files(executable,"normal");
        files.options.weight_lanes=invalid==0?std::filesystem::path("relative-lanes"):invalid==1?files.directory/"missing":files.directory;
        if (invalid==3) { files.options.weight_lanes=files.options.library;files.options.scalar_channel=false; }
        refuses([&]{OwnedWorker worker(files.options);});
        check(!std::filesystem::exists(files.directory/"started"),"invalid lane consent launched a target");
    }
    for (const auto mode : {"lane-auth-normal", "lane-auth-unsolicited", "lane-auth-missing-ready", "lane-auth-drift",
            "lane-auth-missing-done", "lane-auth-wrongtype-done", "lane-auth-appeared-done"}) {
        Files files(executable,mode); files.options.weight_lanes=files.directory/"lanes";
        { std::ofstream output(files.options.weight_lanes); output<<"generated CPU-only lane identity\n"; }
        if (std::string_view(mode)!="lane-auth-unsolicited" && std::string_view(mode)!="lane-auth-appeared-done")
            files.options.weight_lanes_auth_workers=4;
        if (std::string_view(mode)=="lane-auth-unsolicited" || std::string_view(mode)=="lane-auth-missing-ready") {
            refuses([&] { OwnedWorker worker(files.options); }); continue;
        }
        OwnedWorker worker(files.options); size_t callbacks=0;
        worker.request(input(),Clock::now()+3s,[]{return false;});
        const auto generate=[&] { worker.request(generation(),Clock::now()+3s,[]{return false;},[&](const Json&){++callbacks;}); };
        if (std::string_view(mode)!="lane-auth-normal") {
            refuses(generate);check(!worker.alive()&&!callbacks,"unadmitted authentication metadata published a scalar token");continue;
        }
        generate();check(callbacks==1&&worker.alive());worker.close();
    }
    for (const auto invalid : {0u, 17u, 4u}) {
        Files files(executable,"normal"); files.options.weight_lanes_auth_workers=invalid;
        if (invalid!=4) { files.options.weight_lanes=files.directory/"lanes";std::ofstream output(files.options.weight_lanes);output<<"generated\n"; }
        refuses([&] { OwnedWorker worker(files.options); });
        check(!std::filesystem::exists(files.directory/"started"),"dormant or unbounded authentication option launched a target");
    }
}
#endif
void requests() {
    auto request=with(chat("A"),"temperature",number(1));
    request=with(std::move(request),"max_tokens",Json::integer(32768));
    check(admit_chat(request,"tokspur").maximum_tokens==32768);
    Fake fake;ServeSession session(session_codec(),"tokspur");
    request=with(std::move(request),"max_tokens",Json::integer(2));
    ServeSink sink{[](const Json&){},[]{return false;},[]{return uint64_t(44);}};
    const auto result=session.complete(request,"sampled-scalar",[&](const Json& j,const std::function<void(const Json&)>& cb){return fake(j,cb);},sink);
    check(integer(*result.find("usage"),"completion_tokens")==2);
    check(fake.selected.find("temperature")->number_lexeme()=="1");
    fast_fixture::Fake verifier(true);ServeSession greedy(session_codec(),"tokspur","direct",verifier.ready,true,verifier.ready,true);
    refuses([&]{greedy.complete(request,"still-greedy",[&](const Json& j,const std::function<void(const Json&)>& cb){return verifier(j,cb);},sink);});
}
}
#ifndef TOKSPUR_HTTP_SCALAR_CHANNEL_TEST_ENTRY
#define TOKSPUR_HTTP_SCALAR_CHANNEL_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_SCALAR_CHANNEL_TEST_ENTRY(int argc,char** argv) {
    try {
        if(argc>1&&std::string_view(argv[1])=="target") {
            if(std::filesystem::path(argv[0]).filename()=="scalar-channel-target")return scalar_fixture::target(argc,argv);
            return inherited_fast_math_checks(argc,argv);
        }
        const auto prior=inherited_fast_math_checks(argc,argv);if(prior)return prior;
        scalar_fixture::contracts();scalar_fixture::workers(std::filesystem::canonical(argv[0]));scalar_fixture::requests();
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        scalar_fixture::lane_contracts();scalar_fixture::lane_workers(std::filesystem::canonical(argv[0]));
#endif
        std::cout<<"Private sampled scalar-channel CPU checks="<<checks<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
