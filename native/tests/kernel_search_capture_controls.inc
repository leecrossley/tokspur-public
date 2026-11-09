#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
struct KernelCaptureControl {
    uint32_t grouped = 0, qsa = 0; bool refuse_qsa = false;
    static bool emit(void* context, MetalGraph& g, const KernelSearchCaptureRecord& r) noexcept {
        try {
            auto& self = *static_cast<KernelCaptureControl*>(context);
            if (r.kind == 2 && self.refuse_qsa) return false;
            require(r.parameter_bytes && r.parameter_bytes <= 128, "kernel capture parameter extent");
            for (auto v : r.views) if (v.owner) require(g.read(v).size() == v.bytes, "kernel capture immutable view extent");
            if (r.kind == 1) ++self.grouped; else { require(r.kind == 2, "kernel capture kind"); ++self.qsa; }
            return true;
        } catch (...) { return false; }
    }
};
OperatorStateEvidence kernel_capture_run(const char* library, bool capture, bool concurrent) {
    StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,16,8192},{2560,32,512,10},{4,2560,4,4,3},13,1};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    MetalGraph g(library,4ull<<30); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined); g.configure_concurrent_dispatch(concurrent);
    g.configure_prefill_dense_f32(true); g.configure_prefill_grouped_f32(true); g.configure_prefill_dense_staged_f32(true); g.configure_prefill_hc_rows(true); g.configure_qsa_stage_v8(true); g.configure_qsa_channel_chronological(true);
    OperatorStateSink checkpoints; KernelCaptureControl sink;
    if (capture) { g.configure_operator_checkpoints(OperatorStateSink::emit,&checkpoints,16384); g.configure_kernel_search_capture(KernelCaptureControl::emit,&sink); }
    Assets a{g,std::mt19937(44),{},{},{}};auto small=shape;small.experts.experts=2;small.experts.selected=1;const std::array<uint32_t,2> attention_layers{3,7};auto f=fixture(a,small,8,attention_layers);
    for(auto& layer:f.weights.layers)std::visit([&](auto& w){
        for(auto* read:{&w.attention_read,&w.ffn_read}){read->down=stack_q8(a,10240,320);read->up=stack_q8(a,320,10240);}
        auto& m=w.ffn;m.router=a.matrix(2560*512).first;m.gate=stack_raw_routed(a,12,2560,32,512);m.up=stack_raw_routed(a,12,2560,32,512);m.down=stack_raw_routed(a,7,32,2560,512);m.shared_gate=stack_q8(a,2560,32);m.shared_up=stack_q8(a,2560,32);m.shared_down=stack_q8(a,32,2560);
        if constexpr(std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>){w.attention.qkv=stack_q8(a,2560,uint32_t(gdn_qkv_count(shape.recurrent)));w.attention.gate=stack_q8(a,2560,64);w.attention.output=stack_q8(a,64,2560);}
        else {auto& q=w.attention;q.query_gate=stack_q8(a,2560,256);q.key=stack_q8(a,2560,64);q.value=stack_q8(a,2560,64);q.output=stack_q8(a,128,2560);q.index_query=stack_bf16(a,q.index_query);q.index_key=stack_bf16(a,q.index_key);}
    },layer);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    f.weights.output=stack_q8(a,2560,shape.vocabulary);
#endif

    TextStack stack(g,f.weights,shape,policy,spec,StackExecution::direct,512,true); stack.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20);
    std::vector<uint32_t> tokens(4097); std::vector<float> rows(4097*2560), memory(4097*4); auto history = empty_ngram_history(spec);
    for (uint32_t t=0;t<4097;++t) { tokens[t]=(t*3)%13; const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]); history=ids.next;
        for(uint32_t c=0;c<2560;++c) rows[t*2560+c]=float(int((t*13+c*7)%131)-65)/997;
        for(uint32_t c=0;c<4;++c) memory[t*4+c]=float(int(ids.rows[c]%17)-8)/32;
    }
    OperatorStateEvidence evidence;
    const auto retain = [&](const StackResult& r) { const auto bytes=std::as_bytes(std::span(r.logits)); evidence.logits.emplace_back(bytes.begin(),bytes.end()); evidence.accepted.push_back(stack.accepted_snapshot()); evidence.physical.push_back(stack.chunked_cache_snapshot()); };
    for(uint32_t at=0;at<4095;) { const auto n=std::min(512u,4095-at); const auto allocations=g.memory().allocations;
        const auto result=stack.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(at*2560,n*2560),std::span(memory).subspan(at*4,n*4));
        require(g.memory().allocations==allocations,"kernel capture changed prepared GPU allocations"); retain(result); at+=n;
    }
    for(uint32_t at=4095;at<4097;++at) {
        // The existing known-input singleton transaction executes ordinary arithmetic and owns physical cache rollback across diagnostic submits.
        const auto step=[&]{return stack.step_known(std::span(tokens).subspan(at,1),std::span(rows).subspan(at*2560,2560),std::span(memory).subspan(at*4,4));};
        if(capture&&at==4095) { const auto accepted=stack.accepted_snapshot(),physical=stack.chunked_cache_snapshot(); sink.refuse_qsa=true; reject([&]{static_cast<void>(step());}); sink.refuse_qsa=false;
            require(stack.position()==at&&stack.accepted_snapshot()==accepted&&stack.chunked_cache_snapshot()==physical,"kernel capture QSA refusal lost full physical rollback"); }
        retain(step());
    }
    if(capture)require(sink.grouped==4&&sink.qsa==4,"kernel capture missed grouped or QSA train/holdout records");
    return evidence;
}
void kernel_capture_controls(const char* library, bool concurrent) {
    const auto off=kernel_capture_run(library,false,concurrent),on=kernel_capture_run(library,true,concurrent);
    require(off==on,"kernel slice capture changed complete logits, accepted state or physical cache");
    std::cout<<"PASS kernel capture generated eight-layer full-logit/state/cache controls, actual E512/top10 Q4_K/Q5_1, 512 and partial chunks, ordinary4096/4097 and physical QSA rollback; concurrent="<<concurrent<<"\n";
}
#endif
