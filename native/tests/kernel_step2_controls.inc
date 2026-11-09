#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
struct KernelStep2Evidence {
    OperatorStateEvidence state;
    uint64_t scalar_calls=0, row_core_calls=0, rollback_checks=0, grouped_encoded_calls=0;
};
KernelStep2Evidence kernel_step2_run(const char* library,bool enabled,bool concurrent,bool grouped=false,RouteLockControl* lock=nullptr) {
    StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,2048,8192},{2560,32,512,10},{4,2560,4,4,3},13,1};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    MetalGraph g(library,4ull<<30); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined); g.configure_concurrent_dispatch(concurrent);
    g.configure_prefill_dense_f32(true); g.configure_prefill_grouped_f32(true); g.configure_prefill_dense_staged_f32(true); g.configure_prefill_hc_rows(true); g.configure_qsa_stage_v8(true); g.configure_qsa_channel_chronological(true);
    g.configure_q8_three_rows(true); g.configure_kernel_qsa_f32(enabled && !grouped);
    require(g.kernel_qsa_f32_enabled() == (enabled && !grouped), "QSA candidate startup identity");
    reject([&]{g.configure_kernel_qsa_f32(!enabled);});
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    if (grouped) { g.configure_kernel_grouped_fp16(enabled); require(g.kernel_grouped_fp16_enabled() == enabled, "grouped candidate startup identity"); reject([&]{g.configure_kernel_grouped_fp16(!enabled);}); }
#endif
    OperatorStateSink checkpoints;
    if(lock){require(grouped,"step-2 route tape requires the grouped candidate");g.configure_operator_checkpoints(OperatorStateSink::emit,&checkpoints,16384);g.configure_route_lock(RouteLockControl::emit,lock);}
    Assets a{g,std::mt19937(44),{},{},{}};auto small=shape;small.experts.experts=2;small.experts.selected=1;const std::array<uint32_t,1> attention_layers{1};auto f=fixture(a,small,2,attention_layers);
    for(auto& layer:f.weights.layers)std::visit([&](auto& w){
        for(auto* read:{&w.attention_read,&w.ffn_read}){read->down=stack_q8(a,10240,320);read->up=stack_q8(a,320,10240);}
        auto& m=w.ffn;m.router=a.matrix(2560*512).first;m.gate=stack_raw_routed(a,12,2560,32,512);m.up=stack_raw_routed(a,12,2560,32,512);m.down=stack_raw_routed(a,7,32,2560,512);m.shared_gate=stack_q8(a,2560,32);m.shared_up=stack_q8(a,2560,32);m.shared_down=stack_q8(a,32,2560);
        if constexpr(std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>){w.attention.qkv=stack_q8(a,2560,uint32_t(gdn_qkv_count(shape.recurrent)));w.attention.gate=stack_q8(a,2560,64);w.attention.output=stack_q8(a,64,2560);}
        else {auto& q=w.attention;q.query_gate=stack_q8(a,2560,256);q.key=stack_q8(a,2560,64);q.value=stack_q8(a,2560,64);q.output=stack_q8(a,128,2560);q.index_query=stack_bf16(a,q.index_query);q.index_key=stack_bf16(a,q.index_key);}
    },layer);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    const auto original_head=stack_q8(a,2560,shape.vocabulary);
    const auto mutable_head=g.upload(g.read(original_head.view),"step-2 mutable vocabulary head",false); a.views.push_back(mutable_head); f.weights.output={mutable_head,8};
#endif


    TextStack stack(g,f.weights,shape,policy,spec,StackExecution::direct,512,true);
    stack.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20); stack.prepare_final_hc_tap(64ull<<20);
    stack.prepare_sampled_verification2(maximum_stack_verification_budget,{true,QsaVerifyRowsMode::core_and_q8});
    constexpr uint32_t count=4103; std::vector<uint32_t> tokens(count); std::vector<float> rows(count*2560),memory(count*4); auto history=empty_ngram_history(spec);
    for(uint32_t t=0;t<count;++t){tokens[t]=(t*3)%13;const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]);history=ids.next;
        for(uint32_t c=0;c<2560;++c)rows[t*2560+c]=float(int((t*13+c*7)%131)-65)/997;
        for(uint32_t c=0;c<4;++c)memory[t*4+c]=float(int(ids.rows[c]%17)-8)/32;
    }
    KernelStep2Evidence result; auto& evidence=result.state;
    const auto logits=[&](std::span<const float> values){for(auto v:values)require(std::isfinite(v),"step-2 nonfinite full vocabulary");const auto bytes=std::as_bytes(values);evidence.logits.emplace_back(bytes.begin(),bytes.end());};
    const auto retain=[&](std::span<const float> values){logits(values);evidence.prefixes.push_back(stack.position());evidence.accepted.push_back(stack.accepted_snapshot());evidence.physical.push_back(stack.chunked_cache_snapshot());const auto hc=std::as_bytes(stack.last_final_hc_rows());evidence.hc.emplace_back(hc.begin(),hc.end());};
    const auto run=[&](uint32_t at,uint32_t n){return stack.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4));};
    if(lock){
        const auto accepted=stack.accepted_snapshot(),physical=stack.chunked_cache_snapshot();lock->refuse=true;lock->refused_calls=0;
        reject([&]{static_cast<void>(run(0,37));});lock->refuse=false;
        require(stack.position()==0&&stack.accepted_snapshot()==accepted&&stack.chunked_cache_snapshot()==physical,"grouped route-tape refusal lost accepted state or physical cache");++result.rollback_checks;
    }
    const auto head=g.read(f.weights.output.view);auto bad=head;const uint16_t inf=0x7c00;std::memcpy(bad.data(),&inf,2);
    uint32_t at=0;
    for(uint32_t n:{512u,37u,475u,512u,512u,2u,1u,512u,512u,512u,506u}) {
        const auto accepted=stack.accepted_snapshot(),physical=stack.chunked_cache_snapshot();const auto calls=g.qsa_channel_calls();g.write(f.weights.output.view,bad);
        reject([&]{static_cast<void>(run(at,n));});g.write(f.weights.output.view,head);
        require(stack.position()==at&&stack.accepted_snapshot()==accepted&&stack.chunked_cache_snapshot()==physical,"step-2 failed chunk lost accepted state or full physical cache");
        require(g.qsa_channel_calls()==calls,"step-2 failed chunk published successful QSA calls");++result.rollback_checks;
        const auto allocations=g.memory().allocations;const auto actual=run(at,n);at+=n;
        require(stack.position()==at&&g.memory().allocations==allocations,"step-2 chunk frontier or allocation changed");retain(actual.logits);
    }
    require(at==4093,"step-2 generated prefix schedule");
    for(uint32_t consumed:{1u,2u,3u}) {
        const auto proposals=std::span(tokens).subspan(at,3);const auto lookups=stack.sampled_verification_depth2_lookups(proposals);std::vector<float> input_memory(12);
        for(uint32_t t=0;t<3;++t)for(uint32_t c=0;c<4;++c)input_memory[t*4+c]=float(int(lookups[t].rows[c]%17)-8)/32;
        const auto input_rows=std::span(rows).subspan(uint64_t(at)*2560,3*2560);
        struct Choice {
            uint32_t consumed;bool throwing;std::vector<std::vector<float>> rows;
            static uint32_t select(void* context,const StackSampledVerificationRows& r){auto& self=*static_cast<Choice*>(context);require(r.count==3&&r.final_hc_rows.size()==3*10240,"step-2 verification geometry");self.rows.clear();for(uint32_t t=0;t<3;++t){for(auto v:r.logits[t])require(std::isfinite(v),"step-2 verification full logits finite");self.rows.emplace_back(r.logits[t].begin(),r.logits[t].end());}if(self.throwing)throw FormatError("step-2 selector refusal");return self.consumed;}
        } choice{consumed,true,{}};
        const auto accepted=stack.accepted_snapshot(),physical=stack.chunked_cache_snapshot();const std::vector<float> hc(stack.last_final_hc_rows().begin(),stack.last_final_hc_rows().end());
        const auto verify=[&](const StackVerificationCancellation& cancellation=StackVerificationCancellation{}){return stack.verify_sampled_depth2(proposals,input_rows,input_memory,at,{Choice::select,&choice},cancellation);};
        reject([&]{static_cast<void>(verify());});const auto refused_rows=choice.rows;
        const bool accepted_equal=stack.accepted_snapshot()==accepted,physical_equal=stack.chunked_cache_snapshot()==physical,hc_equal=std::equal(hc.begin(),hc.end(),stack.last_final_hc_rows().begin(),stack.last_final_hc_rows().end());
        if(!accepted_equal||!physical_equal||!hc_equal)std::cerr<<"step-2 refusal candidate="<<enabled<<" prefix="<<at<<" accepted_equal="<<accepted_equal<<" physical_equal="<<physical_equal<<" hc_equal="<<hc_equal<<"\n";
        require(stack.position()==at&&accepted_equal&&physical_equal&&hc_equal,"step-2 selector refusal lost accepted state, HC or physical cache");++result.rollback_checks;
        choice.throwing=false;
        struct Cancel {uint32_t polls=0;static bool requested(void* context){return ++static_cast<Cancel*>(context)->polls==2;}} cancel;
        reject([&]{static_cast<void>(verify({Cancel::requested,&cancel}));});
        require(stack.position()==at&&stack.accepted_snapshot()==accepted&&stack.chunked_cache_snapshot()==physical,"step-2 cancellation lost accepted state or physical cache");++result.rollback_checks;
        const auto allocations=g.memory().allocations;const auto actual=verify();require(actual.committed==consumed&&stack.position()==at+consumed&&g.memory().allocations==allocations,"step-2 verification frontier or allocations");
        const auto after=stack.chunked_cache_snapshot();require(after.size()==physical.size(),"step-2 physical cache extent changed");
        const uint64_t kv_row=uint64_t(shape.attention.kv_heads)*shape.attention.head_width*2,kv_capacity=uint64_t(shape.attention.capacity)*kv_row,index_row=uint64_t(shape.attention.index_width)*2;
        for(size_t byte=0;byte<physical.size();++byte){const bool accepted_key=byte>=uint64_t(at)*kv_row&&byte<uint64_t(at+consumed)*kv_row;const bool accepted_value=byte>=kv_capacity+uint64_t(at)*kv_row&&byte<kv_capacity+uint64_t(at+consumed)*kv_row;const bool accepted_index=byte>=2*kv_capacity+uint64_t(at/shape.attention.compression)*index_row&&byte<2*kv_capacity+uint64_t((at+consumed)/shape.attention.compression)*index_row;require(accepted_key||accepted_value||accepted_index||after[byte]==physical[byte],"step-2 accepted prefix changed rejected or unrelated physical cache bytes");}
        require(choice.rows==refused_rows,"step-2 retry changed candidate full verification rows");for(const auto& row:choice.rows)logits(row);at+=consumed;retain(actual.logits);
    }
    const auto accepted=stack.accepted_snapshot(),physical=stack.chunked_cache_snapshot();g.write(f.weights.output.view,bad);reject([&]{static_cast<void>(run(at,1));});g.write(f.weights.output.view,head);
    require(stack.position()==at&&stack.accepted_snapshot()==accepted&&stack.chunked_cache_snapshot()==physical,"step-2 ordinary failure lost physical rollback");++result.rollback_checks;
    retain(run(at,1).logits);result.scalar_calls=g.qsa_channel_calls();result.row_core_calls=g.qsa_verify_rows_routes()[0];
    require(result.scalar_calls>0&&result.row_core_calls>0&&result.rollback_checks==(lock?19u:18u),"step-2 bypassed scalar or verification QSA coverage");
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    result.grouped_encoded_calls=g.kernel_grouped_fp16_encoded_calls();
    require(result.grouped_encoded_calls==(grouped&&enabled?(lock?123u:120u):0u), "step-2 grouped candidate dispatch count differs from chunks and their refused attempts");
#endif
    return result;
}
void kernel_step2_controls(const char* library,bool concurrent,bool grouped) {
    const auto exact=kernel_step2_run(library,false,concurrent,grouped),candidate=kernel_step2_run(library,true,concurrent,grouped),repeat=kernel_step2_run(library,true,concurrent,grouped);
    require(candidate.state==repeat.state&&candidate.grouped_encoded_calls==repeat.grouped_encoded_calls,"step-2 candidate own repeat changed full logits, accepted state, HC or complete physical cache");
    require(exact.state.logits.size()==candidate.state.logits.size()&&exact.state.prefixes==candidate.state.prefixes,"step-2 observation alignment");
    for(size_t i=0;i<exact.state.logits.size();++i){const auto& a=candidate.state.logits[i];const auto& b=exact.state.logits[i];std::vector<float> x(a.size()/4),y(b.size()/4);std::memcpy(x.data(),a.data(),a.size());std::memcpy(y.data(),b.data(),b.size());stack_compare(x,y);}
    const StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,2048,8192},{2560,32,512,10},{4,2560,4,4,3},13,1};
    uint32_t changed=0;for(size_t i=0;i<exact.state.accepted.size();++i){hc_down_state_compare(candidate.state.accepted[i],exact.state.accepted[i],shape,exact.state.prefixes[i]);changed+=candidate.state.accepted[i]!=exact.state.accepted[i];}
    std::cout<<"PASS kernel "<<(grouped?"grouped FP16":"QSA")<<" step-2 generated controls concurrent="<<concurrent<<" full_logit_rows="<<candidate.state.logits.size()<<" accepted_snapshots="<<candidate.state.accepted.size()<<" exact_state_differences="<<changed<<" own_repeat=byte-exact rollback_checks_per_arm="<<candidate.rollback_checks<<" scalar_calls="<<candidate.scalar_calls<<" row_core_calls="<<candidate.row_core_calls<<" grouped_encoded_calls="<<candidate.grouped_encoded_calls<<" worst_fixed_nmse="<<worst<<" generated_nmse_bound=1e-6 no model or timing admission\n";
}
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
void kernel_grouped_route_lock_controls(const char* library,bool concurrent) {
    const auto exact=kernel_step2_run(library,false,concurrent,true);
    RouteLockControl captured;const auto observed=kernel_step2_run(library,false,concurrent,true,&captured);
    require(exact.state==observed.state,"grouped exact route observation changed full logits, accepted state, HC or physical cache");
    RouteLockControl replay;replay.reference=&captured.tape;const auto locked=kernel_step2_run(library,true,concurrent,true,&replay);
    RouteLockControl replay_again;replay_again.reference=&captured.tape;const auto repeat=kernel_step2_run(library,true,concurrent,true,&replay_again);
    require(!captured.tape.empty()&&replay.cursor==captured.tape.size()&&replay_again.cursor==captured.tape.size(),"grouped exact route tape was not completely consumed");
    require(locked.state==repeat.state&&locked.grouped_encoded_calls==123&&repeat.grouped_encoded_calls==123,"grouped locked candidate own repeat changed or matmul was bypassed");
    require(locked.state.prefixes==exact.state.prefixes&&locked.state.logits.size()==exact.state.logits.size(),"grouped locked observation alignment");
    for(size_t i=0;i<exact.state.logits.size();++i){const auto& a=locked.state.logits[i];const auto& b=exact.state.logits[i];std::vector<float> x(a.size()/4),y(b.size()/4);std::memcpy(x.data(),a.data(),a.size());std::memcpy(y.data(),b.data(),b.size());stack_compare(x,y);}
    const StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,2048,8192},{2560,32,512,10},{4,2560,4,4,3},13,1};
    uint32_t changed=0;for(size_t i=0;i<exact.state.accepted.size();++i){hc_down_state_compare(locked.state.accepted[i],exact.state.accepted[i],shape,exact.state.prefixes[i]);changed+=locked.state.accepted[i]!=exact.state.accepted[i];}
    std::cout<<"PASS grouped FP16 exact-route lock concurrent="<<concurrent<<" full_logit_rows="<<locked.state.logits.size()<<" accepted_snapshots="<<locked.state.accepted.size()<<" exact_state_differences="<<changed<<" route_records="<<captured.tape.size()<<" replay_complete=1 own_repeat=byte-exact rollback_checks_per_arm="<<locked.rollback_checks<<" grouped_encoded_calls="<<locked.grouped_encoded_calls<<" worst_fixed_nmse="<<worst<<" generated_nmse_bound=1e-6 no model or speed admission\n";
}
#endif
#endif
