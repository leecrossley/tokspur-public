// CPU-generated transport records only. Source controls; no model/Metal assets.
#define TOKSPUR_HTTP_SCALAR_CHANNEL_TEST_ENTRY inherited_scalar_channel_entry
#include "test_http_scalar_channel.cpp"
#undef TOKSPUR_HTTP_SCALAR_CHANNEL_TEST_ENTRY
#include <fstream>

#ifndef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
#error This fixture requires the distinct sampled input8/scalar-output build.
#endif

namespace sampled_prefill_fixture {
constexpr std::array<const char*, 14> ready_only_fields{"lookup_verification_budget_bytes",
    "lookup_caller_host_bytes", "verification_host_bytes", "verification_gpu_bytes",
    "lookup_verification_charged_bytes", "verification_tape_payload_bytes", "prompt_lookup",
    "checkpoint_broker", "broker_epoch", "gpu_profile_mode", "allow_prompt_lookup_experiments",
    "allow_request_kernel_experiments", "allow_request_stack_experiments", "allow_request_prefill_experiments"};
Json ready() {
    auto v = scalar_fixture::ready();
    v = with(std::move(v), "sampled_prefill_protocol", Json("sampled-prefill-input8-v1"));
    v = with(std::move(v), "known_token_capacity", Json::integer(8));
    v = with(std::move(v), "known_residual_reuse", Json(true));
    v = with(std::move(v), "known_qsa_rows_schema", Json::integer(1));
    v = with(std::move(v), "known_qsa_rows_mode", Json("q8"));
    v = with(std::move(v), "known_qsa_rows_logical_bytes", Json::integer(8'000'000));
    v = with(std::move(v), "known_qsa_rows_charged_bytes", Json::integer(8'003'584));
    v = with(std::move(v), "known_qsa_rows_allocations", Json::integer(46));
    v = with(std::move(v), "known_qsa_rows_physical_tiles", Json::integer(0));
    v = with(std::move(v), "known_qsa_rows_logical_rows", Json::integer(0));
    return with(std::move(v), "known_qsa_rows_http_admitted", Json(true));
}
Json input(size_t count, bool reset) {
    Json::Array ids; for (size_t i = 0; i < count; ++i) ids.push_back(Json::integer(i + 1));
    auto v = with(scalar_fixture::input(reset), "tokens", Json(std::move(ids)));
    return v;
}
struct Fake {
    Json prepared = ready(), previous = prepared, selected = scalar_fixture::sampled_policy();
    uint64_t stage = 0, tiles = 0, rows = 0, words = 0, seed = 44;
    std::array<uint64_t, 5> routes{};
    uint32_t prefix = 0;
    std::vector<Json> packets;
    Json operator()(const Json& job, const std::function<void(const Json&)>& callback) {
        packets.push_back(job);
        if (job.find("reset")->boolean()) { prefix = 0; words = 0; selected = *job.find("sampling"); seed = integer(job,"seed"); }
        const auto n = job.find("tokens")->array().size();
        const auto outputs = integer(job,"generate");
        prefix += uint32_t(n + outputs);
        const std::array<uint64_t, 5> step = outputs ? std::array<uint64_t, 5>{97,0,0,0,0} :
            std::array<uint64_t, 5>{24*n+1,(n&1)?72ull:0ull,(n&2)?72ull:0ull,(n&4)?72ull:0ull,(n&8)?72ull:0ull};
        for (size_t i=0;i<routes.size();++i) routes[i]+=step[i];
        stage += outputs ? 12 : 0; tiles += n ? 12 : 0; rows += 12*n;
        if (outputs && std::stod(std::string(selected.find("temperature")->number_lexeme())) > 0) ++words;
        auto sampled=selected.object(); sampled.emplace_back("valid",Json(true)); sampled.emplace_back("seed",Json::integer(seed));
        sampled.emplace_back("entropy_prefix",Json::integer(prefix)); sampled.emplace_back("entropy_words",Json::integer(words));
        auto done=with(prepared,"event",Json("done"));
        for (const auto name : ready_only_fields) done=without(std::move(done),name);
        for (const auto name : {"compatible_comparator_limit","execution_contract","host_row_cache_charged_bytes",
            "imported_regions","lookup_history_tokens","lookup_match_tokens","lookup_proposal_tokens",
            "maximum_generated_tokens","policy","scope","weight_copy","weight_setup_ms","weights_storage"})
            done=without(std::move(done),name);
        done=with(std::move(done),"committed_prefix",Json::integer(prefix));
        done=with(std::move(done),"sampling",Json(std::move(sampled)));
        done=with(std::move(done),"known_token_prefill",Json(true));
        for (const auto name : {"qsa_stage_v8_calls","qsa_stage_v8_extra_dispatches","qsa_channel_calls","qsa_channel_extra_dispatches"})
            done=with(std::move(done),name,Json::integer(stage));
        done=with(std::move(done),"qsa_stage_v8_physical_dispatches",Json::integer(2*stage));
        for (const auto name : {"qsa_stage_v8_request_calls","qsa_channel_request_calls"})
            done=with(std::move(done),name,Json::integer(outputs?12:0));
        done=with(std::move(done),"known_qsa_rows_physical_tiles",Json::integer(tiles));
        done=with(std::move(done),"known_qsa_rows_logical_rows",Json::integer(rows));
        done=with(std::move(done),"hc_exact_routes",hc(routes));
        done=with(std::move(done),"hc_exact_request_routes",hc(step));
        done=with(std::move(done),"status",Json(outputs?"length":"prefilled"));
        done=with(std::move(done),"finish_reason",outputs?Json("length"):Json());
        done=with(std::move(done),"stop_id",Json()); done=with(std::move(done),"continuation_allowed",Json(true));
        done=with(std::move(done),"generated_ids",Json(outputs?Json::Array{Json::integer(42)}:Json::Array{}));
        if (outputs) callback(Json(Json::Object{{"event",Json("token")},{"id",Json::integer(42)},
            {"committed_prefix",Json::integer(prefix)},{"entropy_words",Json::integer(words)}}));
        previous=done; return done;
    }
};
void contracts() {
    const auto prepared=ready(); validate_sampled_prefill_ready(prepared); ++checks;
    for (const auto name : ready_only_fields)
        refuses([&]{validate_sampled_prefill_ready(without(prepared,name));});
    refuses([&]{validate_scalar_channel_ready(prepared);});
    refuses([&]{validate_sampled_prefill_ready(scalar_fixture::ready());});
    for (const auto& [name,value] : Json::Object{{"known_token_capacity",Json::integer(0)},
        {"known_residual_reuse",Json(false)},{"known_qsa_rows_mode",Json("off")},
        {"known_qsa_rows_http_admitted",Json(false)},{"known_qsa_rows_allocations",Json::integer(45)},
        {"known_qsa_rows_charged_bytes",Json::integer((64ull<<20)+1)},{"prompt_lookup",Json(true)},
        {"verification_max_tokens",Json::integer(8)},{"checkpoint_slots",Json::integer(1)},
        {"sampled_prefill_protocol",Json("scalar-channel-v1")}})
        refuses([&]{validate_sampled_prefill_ready(with(prepared,name,value));});
    for (size_t n=1;n<=8;++n) {
        Fake fake; const auto job=input(n,true);
        const auto first=fake(job,[](const Json&){check(false);});
        validate_sampled_prefill_done(prepared,prepared,job,first,nullptr); ++checks;
        for (const auto name : ready_only_fields) {
            check(!first.find(name));
            refuses([&]{validate_sampled_prefill_done(prepared,prepared,job,with(first,name,*prepared.find(name)),nullptr);});
        }
        const auto bad_entropy=with(first,"sampling",with(*first.find("sampling"),"entropy_words",Json::integer(1)));
        refuses([&]{validate_sampled_prefill_done(prepared,prepared,job,bad_entropy,nullptr);});
        std::optional<Json> token; const auto output=scalar_fixture::generation();
        const auto second=fake(output,[&](const Json& v){token=v;});
        validate_sampled_prefill_done(prepared,first,output,second,&*token); ++checks;
        const auto pointmass=with(second,"sampling",with(*second.find("sampling"),"entropy_words",Json::integer(0)));
        const auto pointmass_token=with(*token,"entropy_words",Json::integer(0));
        validate_sampled_prefill_done(prepared,first,output,pointmass,&pointmass_token); ++checks;
        const auto double_draw=with(second,"sampling",with(*second.find("sampling"),"entropy_words",Json::integer(2)));
        const auto double_draw_token=with(*token,"entropy_words",Json::integer(2));
        refuses([&]{validate_sampled_prefill_done(prepared,first,output,double_draw,&double_draw_token);});
        const auto append=input(9-n,false); const auto third=fake(append,[](const Json&){check(false);});
        validate_sampled_prefill_done(prepared,second,append,third,nullptr); ++checks;
        check(integer(*third.find("sampling"),"entropy_words")==1);
        refuses([&]{validate_sampled_prefill_done(prepared,second,append,
            with(third,"sampling",with(*third.find("sampling"),"seed",Json::integer(99))),nullptr);});
        const auto reset=input(n,true); const auto fourth=fake(reset,[](const Json&){check(false);});
        validate_sampled_prefill_done(prepared,third,reset,fourth,nullptr); ++checks;
        check(integer(*fourth.find("sampling"),"entropy_words")==0);
        for (const auto name : {"known_qsa_rows_physical_tiles","known_qsa_rows_logical_rows","qsa_channel_calls","gpu_allocation_count"})
            refuses([&]{validate_sampled_prefill_done(prepared,third,reset,with(fourth,name,Json::integer(999)),nullptr);});
    }
    Fake fake; const auto valid=input(1,true); const auto done=fake(valid,[](const Json&){check(false);});
    for (const auto& bad : {input(0,true),input(9,true),with(valid,"generate",Json::integer(1)),input(1,false)})
        refuses([&]{validate_sampled_prefill_done(prepared,prepared,bad,done,nullptr);});
}
void sessions() {
    ServeSink sink{[](const Json&){},[]{return false;},[]{return uint64_t(44);}};
    for (size_t tail=1;tail<=8;++tail) {
        Fake fake; ServeSession session(session_codec(),"tokspur","direct",Json(),false,Json(),false,ToolEnumValidation::strict,true);
        auto request=with(chat(std::string(8+tail,'A'),false,1),"temperature",number(1));
        const auto result=session.complete(request,"input8-tail",[&](const Json& j,const std::function<void(const Json&)>& cb){return fake(j,cb);},sink);
        check(integer(*result.find("usage"),"completion_tokens")==1);
        check(fake.packets.size()==3 && fake.packets[0].find("tokens")->array().size()==8 &&
            fake.packets[1].find("tokens")->array().size()==tail && integer(fake.packets[2],"generate")==1);
        check(fake.packets[0].find("reset")->boolean() && !fake.packets[1].find("reset")->boolean() && fake.words==1);
        request=with(chat(std::string(8+tail,'A')+")B",false,1),"temperature",number(1));
        static_cast<void>(session.complete(request,"input8-append",[&](const Json& j,const std::function<void(const Json&)>& cb){return fake(j,cb);},sink));
        check(fake.packets.size()==5 && !fake.packets[3].find("reset")->boolean() &&
            fake.packets[3].find("tokens")->array().size()==1 && fake.words==2);
    }
    Fake fake; ServeSession session(session_codec(),"tokspur","direct",Json(),false,Json(),false,ToolEnumValidation::strict,true);
    bool cancelled=false; ServeSink cancelling{[](const Json&){},[&]{return cancelled;},[]{return uint64_t(44);}};
    auto request=with(chat(std::string(9,'A'),false,1),"temperature",number(1));
    refuses([&]{session.complete(request,"input8-cancel",[&](const Json& j,const std::function<void(const Json&)>& cb){
        auto done=fake(j,cb); cancelled=true; return done;},cancelling);});
    check(session.committed_tokens()==0 && fake.packets.size()==1 && fake.prefix==8 && fake.words==0);
    static_cast<void>(session.complete(request,"input8-reset-after-cancel",[&](const Json& j,const std::function<void(const Json&)>& cb){return fake(j,cb);},sink));
    check(fake.packets[1].find("reset")->boolean());
}
const Json& capture_field(const Json& value,const char* name) {
    const auto* field=value.find(name);
    if(!field) throw FormatError("native capture is missing a required field");
    return *field;
}
void capture_need(bool valid,const char* message) { if(!valid) throw FormatError(message); }
void validate_capture(const Json& capture) {
    capture_need(capture_field(capture,"returncode").unsigned_integer()==0,"native capture command failed");
    const auto& packets=capture_field(capture,"packets").array();
    const auto& events=capture_field(capture,"events").array();
    capture_need(!packets.empty() && packets.size()<=1024 && !events.empty() && events.size()<=1+2*packets.size(),"native capture size/order bound");
    const auto& prepared=events.front();
    capture_need(capture_field(prepared,"event").string()=="ready","native capture must begin with READY");
    validate_sampled_prefill_ready(prepared);
    auto previous=prepared; size_t at=1;
    for(const auto& packet:packets) {
        const auto outputs=capture_field(packet,"generate").unsigned_integer();
        capture_need(outputs<=1,"native capture output bound");
        const Json* token=nullptr;
        if(outputs) {
            capture_need(at<events.size() && capture_field(events[at],"event").string()=="token","native capture requires exactly one ordered token");
            token=&events[at++];
        }
        capture_need(at<events.size() && capture_field(events[at],"event").string()=="done","native capture requires an ordered completion");
        const auto& done=events[at++];
        validate_sampled_prefill_done(prepared,previous,packet,done,token);
        previous=done;
    }
    capture_need(at==events.size(),"native capture has trailing/unsolicited events");
}
void capture_controls() {
    Fake fake; Json::Array packets{input(8,true),input(3,false),scalar_fixture::generation()};
    Json::Array events{fake.prepared};
    for(const auto& packet:packets) {
        const auto done=fake(packet,[&](const Json& token){events.push_back(token);}); events.push_back(done);
    }
    const auto capture=Json(Json::Object{{"returncode",Json::integer(0)},{"packets",Json(packets)},{"events",Json(events)}});
    validate_capture(capture); ++checks;
    refuses([&]{validate_capture(with(capture,"returncode",Json::integer(1)));});
    const auto rejects=[&](Json::Array changed){refuses([&]{validate_capture(with(capture,"events",Json(std::move(changed))));});};
    auto changed=events; changed.erase(changed.begin()); rejects(std::move(changed));
    changed=events; changed.erase(changed.begin()+3); rejects(std::move(changed));
    changed=events; std::swap(changed[3],changed[4]); rejects(std::move(changed));
    changed=events; changed.push_back(events.back()); rejects(std::move(changed));
    changed=events; changed[1]=fake.prepared; rejects(std::move(changed));
    changed=events; changed[1]=with(changed[1],"event",Json("error")); rejects(std::move(changed));
    changed=events; changed.insert(changed.begin()+4,events[3]); rejects(std::move(changed));
    changed=events; changed.pop_back(); rejects(std::move(changed));
}
void check_native_capture(const std::filesystem::path& path) {
    constexpr size_t maximum=8*1024*1024;
    std::ifstream stream(path,std::ios::binary);
    check(bool(stream),"native capture cannot be opened");
    std::string raw(maximum+1,'\0'); stream.read(raw.data(),std::streamsize(raw.size()));
    const auto size=stream.gcount();
    check(!stream.bad() && size>0 && uint64_t(size)<=maximum,"native capture exceeds bounded JSON input");
    raw.resize(size_t(size)); validate_capture(parse_json(raw));
    std::cout<<"Native sampled input8 capture admitted by CPU validators (no model or GPU executed)\n";
}
}
#ifndef TOKSPUR_HTTP_SAMPLED_PREFILL_TEST_ENTRY
#define TOKSPUR_HTTP_SAMPLED_PREFILL_TEST_ENTRY main
#endif
int TOKSPUR_HTTP_SAMPLED_PREFILL_TEST_ENTRY(int argc,char** argv) {
    try {
        if(argc>1 && std::string_view(argv[1])=="target") return inherited_scalar_channel_entry(argc,argv);
        if(argc>1 && std::string_view(argv[1])=="--check-native-capture") {
            check(argc==3,"usage: test-http_sampled_prefill --check-native-capture <private-receipt.json>");
            sampled_prefill_fixture::check_native_capture(argv[2]); return 0;
        }
        check(argc==1,"unknown sampled prefill fixture arguments");
        const auto prior=inherited_scalar_channel_entry(argc,argv); if(prior) return prior;
        sampled_prefill_fixture::contracts(); sampled_prefill_fixture::sessions(); sampled_prefill_fixture::capture_controls();
        std::cout<<"Sampled input8/scalar-output CPU controls="<<checks<<'\n'; return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
