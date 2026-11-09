#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
struct RouteControlSink {
    std::array<std::vector<std::byte>,6> common;
    std::vector<std::byte> exact_input;
    uint32_t records=0, refuse_phase=0;
    static bool emit(void* context,MetalGraph& g,const RouteAttributionRecord& r) noexcept {
        try {
            auto& s=*static_cast<RouteControlSink*>(context);++s.records;
            if(r.phase==s.refuse_phase)return false;
            if(r.phase==1){for(size_t i=0;i<r.views.size();++i)s.common[i]=r.views[i].owner?g.read(r.views[i]):std::vector<std::byte>{};}
            if(r.phase==5){for(size_t i=0;i<r.views.size();++i)require(s.common[i]==(r.views[i].owner?g.read(r.views[i]):std::vector<std::byte>{}),"shadow changed common layer input, accepted state or restored cache");}
            if(r.phase==3)s.exact_input=g.read(r.views[0]);
            if(r.phase==4)require(g.read(r.views[0])==s.exact_input,"router comparison changed common input");
            if(r.phase==2||r.phase==3||r.phase==4){require(r.experts==512&&r.selected==10,"route test needs actual top10 expert geometry");const auto bytes=g.read(r.views[1]);std::vector<float> values(bytes.size()/4);std::memcpy(values.data(),bytes.data(),bytes.size());for(auto x:values)require(std::isfinite(x),"nonfinite generated common-input router row");}
            return true;
        }catch(...){return false;}
    }
};
#include "route_lock_control_sink.inc"
OperatorStateEvidence route_control_run(const char* library,bool enabled,bool concurrent
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    ,bool router=false,RouteLockControl* lock=nullptr
#endif
) {
    StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,16,1024},{2560,32,512,10},{4,2560,4,4,3},13,1};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    MetalGraph g(library,2ull<<30);g.configure_small_gemv(16);g.configure_block_gemv(3);g.configure_hc_exact(HcExactMode::combined);g.configure_concurrent_dispatch(concurrent);
    g.configure_prefill_dense_f32(true);g.configure_prefill_grouped_f32(true);g.configure_prefill_dense_staged_f32(true);g.configure_prefill_hc_rows(true);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    g.configure_moe_router_f32_batch(router);
    g.configure_q8_three_rows(true);
    g.configure_qsa_stage_v8(true);g.configure_qsa_channel_chronological(true);
#endif
    OperatorStateSink checkpoints;g.configure_operator_checkpoints(OperatorStateSink::emit,&checkpoints,16384);RouteControlSink sink;
    if(enabled)g.configure_route_attribution(RouteControlSink::emit,&sink);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    if(lock)g.configure_route_lock(RouteLockControl::emit,lock);
#endif
    Assets a{g,std::mt19937(44),{},{},{}};auto small=shape;small.experts.experts=2;small.experts.selected=1;const std::array<uint32_t,1> attention_layers{1};auto f=fixture(a,small,2,attention_layers);
    for(auto& layer:f.weights.layers)std::visit([&](auto& w){
        for(auto* read:{&w.attention_read,&w.ffn_read}){read->down=stack_q8(a,10240,320);read->up=stack_q8(a,320,10240);}
        auto& m=w.ffn;m.router=a.matrix(2560*512).first;m.gate=stack_q8(a,2560,32,512);m.up=stack_q8(a,2560,32,512);m.down=stack_q8(a,32,2560,512);m.shared_gate=stack_q8(a,2560,32);m.shared_up=stack_q8(a,2560,32);m.shared_down=stack_q8(a,32,2560);
        if constexpr(std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>){w.attention.qkv=stack_q8(a,2560,uint32_t(gdn_qkv_count(shape.recurrent)));w.attention.gate=stack_q8(a,2560,64);w.attention.output=stack_q8(a,64,2560);}
        else {auto& q=w.attention;q.query_gate=stack_q8(a,2560,256);q.key=stack_q8(a,2560,64);q.value=stack_q8(a,2560,64);q.output=stack_q8(a,128,2560);q.index_query=stack_bf16(a,q.index_query);q.index_key=stack_bf16(a,q.index_key);}
    },layer);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    f.weights.output=stack_q8(a,2560,shape.vocabulary);
#endif
    TextStack stack(g,f.weights,shape,policy,spec,StackExecution::direct,512,true);stack.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20);stack.prepare_final_hc_tap(64ull<<20);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    stack.prepare_sampled_verification2(maximum_stack_verification_budget,{true,QsaVerifyRowsMode::core_and_q8});
#endif
    std::vector<uint32_t> tokens(512);std::vector<float> rows(512*2560),memory(512*4);auto history=empty_ngram_history(spec);
    for(uint32_t t=0;t<512;++t){tokens[t]=(t*3)%13;const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]);history=ids.next;for(uint32_t c=0;c<2560;++c)rows[t*2560+c]=float(int((t*13+c*7)%131)-65)/997;for(uint32_t c=0;c<4;++c)memory[t*4+c]=float(int(ids.rows[c]%17)-8)/32;}
    OperatorStateEvidence evidence;
    const auto run=[&](uint32_t n){return stack.step_known(std::span(tokens).first(n),std::span(rows).first(n*2560),std::span(memory).first(n*4));};
    for(auto n:{3u,37u,512u}) {
        stack.reset();const auto initial=stack.accepted_snapshot(),physical=stack.chunked_cache_snapshot();
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(lock){lock->refuse=true;lock->refused_calls=0;reject([&]{static_cast<void>(run(n));});lock->refuse=false;require(stack.position()==0&&stack.accepted_snapshot()==initial&&stack.chunked_cache_snapshot()==physical,"route lock failure lost physical rollback");}
#endif
        if(enabled){sink.refuse_phase=2;reject([&]{static_cast<void>(run(n));});sink.refuse_phase=0;require(stack.position()==0&&stack.accepted_snapshot()==initial&&stack.chunked_cache_snapshot()==physical,"refused route capture failed physical rollback");}
        const auto allocations=g.memory().allocations;const auto result=run(n);require(g.memory().allocations==allocations,"route capture allocated during layer encoding");
        const auto bytes=std::as_bytes(std::span(result.logits));evidence.logits.emplace_back(bytes.begin(),bytes.end());evidence.accepted.push_back(stack.accepted_snapshot());evidence.physical.push_back(stack.chunked_cache_snapshot());
        const auto expected=evidence.accepted.back();stack.reset();const auto repeat=run(n);require(result.logits==repeat.logits&&stack.accepted_snapshot()==expected,"route capture own repeat changed full logits or accepted state");
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        if(n==512){
            for(uint32_t count:{2u,3u}){
                const auto base=stack.position();const auto proposals=std::span(tokens).first(count);const auto lookups=stack.sampled_verification8_lookups(proposals);std::vector<float> verification_memory(count*4);
                for(uint32_t t=0;t<count;++t)for(uint32_t j=0;j<4;++j)verification_memory[t*4+j]=float(int(lookups[t].rows[j]%17)-8)/32;
                struct AcceptAll{static uint32_t choose(void* context,const StackSampledVerificationRows& r){const auto expected=*static_cast<uint32_t*>(context);require(r.count==expected,"route lock verification rows");return expected;}};
                const StackSampledVerificationSelector selector{AcceptAll::choose,&count};
                const auto verified=count==2?stack.verify_sampled2(proposals,std::span(rows).first(count*2560),verification_memory,base,selector):
                    stack.verify_sampled_depth2(proposals,std::span(rows).first(count*2560),verification_memory,base,selector);
                require(verified.committed==count&&stack.position()==base+count,"route lock verification frontier");const auto bytes=std::as_bytes(std::span(verified.logits));evidence.logits.emplace_back(bytes.begin(),bytes.end());evidence.accepted.push_back(stack.accepted_snapshot());evidence.physical.push_back(stack.chunked_cache_snapshot());
            }
            const auto last=stack.step(tokens[3],std::span(rows).subspan(3*2560,2560),std::span(memory).subspan(3*4,4));const auto final=std::as_bytes(std::span(last.logits));evidence.logits.emplace_back(final.begin(),final.end());evidence.accepted.push_back(stack.accepted_snapshot());evidence.physical.push_back(stack.chunked_cache_snapshot());
        }
#endif
    }
    if(enabled)require(sink.records>=60,"route control missed common-input rows");return evidence;
}
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
void route_lock_controls(const char* library,bool concurrent) {
    const auto exact=route_control_run(library,false,concurrent);
    RouteLockControl captured;const auto observed=route_control_run(library,false,concurrent,false,&captured);
    require(exact==observed,"observing route tape changed exact state or full logits");
    std::array<std::array<bool,2>,2> two_row_coverage{};uint32_t two_row_records=0;
    for(const auto& row:captured.tape)if(row.phase=="verification_layer"&&row.context_rows==2){
        require(row.rows==1&&row.prefix==512&&row.layer>=0&&row.layer<2&&row.offset<2,"scalar width-two route owner differs");
        auto& seen=two_row_coverage[size_t(row.layer)][row.offset];require(!seen,"scalar width-two route rows overlap");seen=true;++two_row_records;
    }
    require(two_row_records==4,"scalar width-two route capture omitted a GDN or QSA row");
    static_cast<void>(route_control_run(library,false,concurrent,true));
    RouteLockControl replay;replay.reference=&captured.tape;const auto locked=route_control_run(library,false,concurrent,true,&replay);
    require(replay.cursor==captured.tape.size()&&locked==exact,"candidate matmul with exact routes did not recover exact vectors/state/cache");
    std::cout<<"Generated router route-lock pass: actual top10/E512 GDN/QSA n3,37,512 plus width2/N3 verification and ordinary continuation; scalar row offsets0/1 checked for both layers, candidate matmul executes, exact IDs/weights replay restores exact logits/state/cache, sink-failure rollback; concurrent="<<concurrent<<"\n";
}
#endif
void route_attribution_controls(const char* library,bool concurrent) {
    const auto off=route_control_run(library,false,concurrent),on=route_control_run(library,true,concurrent);
    require(off==on,"route attribution changed exact full logits, accepted state or physical cache");
    std::cout<<"Common-input route capture top10/E512 HC/GDN/QSA controls pass: diagnostic off/on exact full logits/state/cache, own repeat, sink-failure rollback; concurrent="<<concurrent<<"\n";
}
#endif
