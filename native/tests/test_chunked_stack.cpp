#define main inherited_stack_entry
#include "test_stack.cpp"
#undef main
#include "tokspur/gdn_chunk.hpp"
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#include "tokspur/raw_decode.hpp"
#endif

namespace {
GpuMatrix stack_q8(Assets& a, uint32_t c, uint32_t r, uint32_t experts = 1) {
    std::vector<std::byte> data(uint64_t(c / 32) * r * experts * 34);
    for (uint64_t b = 0; b < data.size() / 34; ++b) {
        const uint16_t d = 0x1400; std::memcpy(data.data() + b * 34, &d, 2);
        for (uint32_t j = 0; j < 32; ++j) { const int8_t q = int8_t(int((b * 17 + j * 11) % 15) - 7); std::memcpy(data.data() + b * 34 + 2 + j, &q, 1); }
    }
    const auto v = a.g.upload(data, "generated chunked stack Q8", true); a.views.push_back(v); return {v, 8};
}
GpuMatrix stack_bf16(Assets& a, GpuMatrix w) {
    const auto bytes = a.g.read(w.view); std::vector<uint16_t> values(bytes.size() / 4);
    for (size_t i = 0; i < values.size(); ++i) { float x; std::memcpy(&x, bytes.data() + i * 4, 4); values[i] = finite_bf16(x); }
    const auto v = a.g.upload(std::as_bytes(std::span(values)), "generated chunked stack BF16", true); a.views.push_back(v); return {v, 30};
}
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
GpuMatrix stack_raw_routed(Assets& a, uint32_t type, uint32_t c, uint32_t r, uint32_t experts) {
    namespace raw = tokspur::raw_decode;
    const auto format = static_cast<raw::Format>(type); const auto qk = raw::block_columns(format), qb = raw::block_bytes(format);
    std::vector<uint8_t> bytes(uint64_t(c / qk) * r * experts * qb);
    for (size_t at = 0; at < bytes.size(); at += qb) {
        raw::write16(bytes.data() + at, type == 12 ? 0x0400 : 0x1400); raw::write16(bytes.data() + at + 2, type == 12 ? 0x0400 : 0x9400);
        if (type == 12) for (unsigned group = 0; group < 8; ++group) raw::set_q4_group(bytes.data() + at, group, 1 + group % 4, 1 + group % 3);
        for (unsigned column = 0; column < qk; ++column) raw::set_quant(format, bytes.data() + at, column, unsigned((at * 17 + column * 11) % (type == 12 ? 16 : 32)));
    }
    const auto v = a.g.upload(std::as_bytes(std::span(bytes)), "generated raw Q4_K/Q5_1 expert matrix", true); a.views.push_back(v); return {v, type};
}
#endif
void stack_compare(std::span<const float> a, std::span<const float> b) {
    require(a.size() == b.size(), "chunked stack vocabulary extent"); double error = 0, signal = 0;
    for (size_t i = 0; i < a.size(); ++i) { require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite chunked stack logits"); const double d = double(a[i]) - b[i]; error += d * d; signal += double(b[i]) * b[i]; }
    require(signal > 0 && error / signal <= 1e-6, "chunked generated stack fixed NMSE gate"); worst = std::max(worst, error / signal);
}
void chunked_case(MetalGraph& g, GdnChunkMode mode, bool legacy_order = false) {
    const StackShape shape{{2,256,32},{256,1,2,128,32,4},{256,4,2,32,2,32,32,4,16,1024},{256,64,16,10},{2,256,4,4,3},13,1};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    Assets a{g,std::mt19937(44),{},{},{}}; const std::array<uint32_t,2> attention_layers{1,2}; auto f=fixture(a,shape,3,attention_layers);
    for(auto& layer:f.weights.layers)std::visit([&](auto& w){
        for(auto* read:{&w.attention_read,&w.ffn_read}){read->down=stack_q8(a,512,32);read->up=stack_q8(a,32,512);}
        auto& m=w.ffn;m.gate=stack_q8(a,256,64,16);m.up=stack_q8(a,256,64,16);m.down=stack_q8(a,64,256,16);
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (g.raw_decode_enabled()) { m.gate=stack_raw_routed(a,12,256,64,16);m.up=stack_raw_routed(a,12,256,64,16);m.down=stack_raw_routed(a,7,64,256,16); }
#endif
        m.shared_gate=stack_q8(a,256,64);m.shared_up=stack_q8(a,256,64);m.shared_down=stack_q8(a,64,256);
        if constexpr(std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>){w.attention.qkv=stack_q8(a,256,uint32_t(gdn_qkv_count(shape.recurrent)));w.attention.gate=stack_q8(a,256,64);w.attention.output=stack_q8(a,64,256);}
        else{auto& q=w.attention;q.query_gate=stack_q8(a,256,256);q.key=stack_q8(a,256,64);q.value=stack_q8(a,256,64);q.output=stack_q8(a,128,256);q.index_query=stack_bf16(a,q.index_query);q.index_key=stack_bf16(a,q.index_key);}
    },layer);
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    auto& recurrent_weight=std::get<RecurrentWeights>(f.weights.layers[0]).attention.qkv;
    const auto mutable_trunk=g.upload(g.read(recurrent_weight.view),"generated mutable trunk Q8",false);a.views.push_back(mutable_trunk);recurrent_weight.view=mutable_trunk;
    const auto original_q8=stack_q8(a,256,shape.vocabulary);const auto mutable_head=g.upload(g.read(original_q8.view),"generated mutable Q8 vocabulary",false);a.views.push_back(mutable_head);f.weights.output={mutable_head,8};
#endif
    const auto base=g.memory().live_bytes;
    {
        TextStack serial(g,f.weights,shape,policy,spec),candidate(g,f.weights,shape,policy,spec,StackExecution::direct,512
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            ,true
#endif
            );
        if(legacy_order)reject([&]{candidate.prepare_chunked_prefill(GdnChunkMode::cs64,512ull<<20,true);});
        candidate.prepare_chunked_prefill(mode,512ull<<20,legacy_order); require(candidate.chunked_prefill_enabled()&&candidate.chunked_prefill_bytes()>0,"chunked preparation unavailable");
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        candidate.prepare_final_hc_tap(64ull<<20);
        require(candidate.final_hc_tap_memory().capacity==512,"MTP input512 tap capacity");
        candidate.prepare_sampled_verification2(maximum_stack_verification_budget);
#endif
        reject([&]{candidate.prepare_transient_verification(64ull<<20);});reject([&]{candidate.prepare_transient_verification8(64ull<<20);});
        reject([&]{candidate.prepare_checkpoints({1,64ull<<20});});reject([&]{candidate.configure_known_layer_submissions(true);});
        reject([&]{candidate.configure_known_grouped_ffn(false);});reject([&]{candidate.prepare_chunked_prefill(mode,512ull<<20);});
        constexpr uint32_t count=1024;std::vector<uint32_t> tokens(count);std::vector<float> rows(uint64_t(count)*256),memory(uint64_t(count)*4);auto history=empty_ngram_history(spec);
        for(uint32_t t=0;t<count;++t){tokens[t]=(t*3)%13;const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]);history=ids.next;
            for(uint32_t j=0;j<256;++j)rows[uint64_t(t)*256+j]=float(int((t*13+j*7)%131)-65)/997;
            for(uint32_t j=0;j<4;++j)memory[uint64_t(t)*4+j]=float(int(ids.rows[j]%17)-8)/32;}
        const auto run=[&](uint32_t at,uint32_t n){return candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*256,uint64_t(n)*256),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4));};
        const auto original_head=g.read(f.weights.output.view);auto bad=original_head;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        const uint16_t inf=0x7c00;std::memcpy(bad.data(),&inf,2);
#else
        const float inf=INFINITY;std::memcpy(bad.data(),&inf,4);
#endif
        for(uint32_t at:{0u,512u}){
            const uint32_t n=at?37:512;const auto accepted=candidate.accepted_snapshot(),physical=candidate.chunked_cache_snapshot();g.write(f.weights.output.view,bad);
            reject([&]{static_cast<void>(run(at,n));});require(candidate.position()==at&&candidate.accepted_snapshot()==accepted,"failed chunked stack published state/history");
            require(candidate.chunked_cache_snapshot()==physical,"failed chunked stack did not restore full physical cache capacity");g.write(f.weights.output.view,original_head);
            const auto allocations=g.memory().allocations;const auto result=run(at,n);require(result.position==at&&candidate.position()==at+n&&g.memory().allocations==allocations,"chunked commit/allocation invariant");
            StackResult expected;for(uint32_t t=at;t<at+n;++t)expected=serial.step(tokens[t],std::span(rows).subspan(uint64_t(t)*256,256),std::span(memory).subspan(uint64_t(t)*4,4));
            stack_compare(result.logits,expected.logits);
        }
        candidate.reset();require(candidate.position()==0,"chunked reset prefix");const auto rerun=run(0,3);require(rerun.position==0&&candidate.position()==3,"chunked reset reuse");
        const auto single_accepted=candidate.accepted_snapshot(),single_physical=candidate.chunked_cache_snapshot();g.write(f.weights.output.view,bad);
        reject([&]{static_cast<void>(run(3,1));});require(candidate.position()==3&&candidate.accepted_snapshot()==single_accepted&&candidate.chunked_cache_snapshot()==single_physical,"ordinary single-input path lost physical rollback");
        g.write(f.weights.output.view,original_head);const auto single=run(3,1);require(single.position==3&&candidate.position()==4,"ordinary single-input publication");
        serial.reset();StackResult single_reference;for(uint32_t t=0;t<4;++t)single_reference=serial.step(tokens[t],std::span(rows).subspan(uint64_t(t)*256,256),std::span(memory).subspan(uint64_t(t)*4,4));
        stack_compare(single.logits,single_reference.logits);
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        const auto ids=std::span(tokens).subspan(4,3);const auto proposed=candidate.sampled_verification_depth2_lookups(ids);
        std::vector<float> verify_memory(12);for(uint32_t t=0;t<3;++t)for(uint32_t j=0;j<4;++j)verify_memory[t*4+j]=float(int(proposed[t].rows[j]%17)-8)/32;
        const auto verify_rows=std::span(rows).subspan(4*256,3*256);const auto accepted=candidate.accepted_snapshot();
        struct Choose { bool throwing=false;static uint32_t run(void* context,const StackSampledVerificationRows& r){
            require(r.count==3&&r.residual_width==512&&r.final_hc_rows.size()==3*512,"input512 verifier row geometry");
            for(auto logits:r.logits)for(auto v:logits)require(std::isfinite(v),"input512 nonfinite verifier logits");
            if(static_cast<Choose*>(context)->throwing)throw FormatError("generated selector refusal");return 1;}} choice{true};
        reject([&]{static_cast<void>(candidate.verify_sampled_depth2(ids,verify_rows,verify_memory,4,{Choose::run,&choice}));});
        require(candidate.position()==4&&candidate.accepted_snapshot()==accepted&&candidate.last_final_hc_count()==1,"input512 verifier refusal published state");
        choice.throwing=false;const auto allocations=g.memory().allocations;
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        const auto decode_before = g.raw_decode_routes();
#endif
        const auto verified=candidate.verify_sampled_depth2(ids,verify_rows,verify_memory,4,{Choose::run,&choice});
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (g.raw_decode_enabled()) { const auto routes=g.raw_decode_routes();require(routes[0]>decode_before[0]&&routes[1]>decode_before[1],"sampled input512 verifier bypassed raw Q4_K/Q5_1 routes"); }
#endif
        require(verified.committed==1&&candidate.position()==5&&candidate.last_final_hc_count()==1&&g.memory().allocations==allocations,"input512 verifier publication/allocation");
        const auto reference=serial.step(tokens[4],std::span(rows).subspan(4*256,256),std::span(memory).subspan(4*4,4));stack_compare(verified.logits,reference.logits);
        const auto continued=candidate.step(tokens[5],std::span(rows).subspan(5*256,256),std::span(memory).subspan(5*4,4));
        const auto ordinary=serial.step(tokens[5],std::span(rows).subspan(5*256,256),std::span(memory).subspan(5*4,4));stack_compare(continued.logits,ordinary.logits);

        // Compare the same trunk/chunk arithmetic with only final output work omitted. All HC rows remain host-visible for unchanged MTP priming.
        TextStack eager(g,f.weights,shape,policy,spec,StackExecution::direct,512,true);eager.prepare_chunked_prefill(mode,512ull<<20,legacy_order);eager.prepare_final_hc_tap(64ull<<20);
        // Reset ignores old suffix bytes, so initialise both complete physical capacities through the same valid lineage before comparing them.
        candidate.reset();for(uint32_t at:{0u,512u}){const auto input=std::span(tokens).subspan(at,512);const auto embedding_rows=std::span(rows).subspan(uint64_t(at)*256,512*256);const auto memory_rows=std::span(memory).subspan(uint64_t(at)*4,512*4);static_cast<void>(candidate.step_known(input,embedding_rows,memory_rows));static_cast<void>(eager.step_known(input,embedding_rows,memory_rows));}
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if(g.q8_three_rows_enabled())for(uint32_t consumed:{1u,2u,3u}){
            candidate.reset();eager.reset();const auto deferred=candidate.step_known(std::span(tokens).first(512),std::span(rows).first(512*256),std::span(memory).first(512*4),false);
            require(!deferred.logits_available&&deferred.logits.empty()&&candidate.position()==512,"N3 capacity512 deferred prefix claimed logits");
            const auto final=candidate.step_known(std::span(tokens).subspan(512,1),std::span(rows).subspan(512*256,256),std::span(memory).subspan(512*4,4),true);
            require(final.logits_available&&candidate.position()==513,"N3 capacity512 final tail omitted logits");
            static_cast<void>(eager.step_known(std::span(tokens).first(512),std::span(rows).first(512*256),std::span(memory).first(512*4),false));
            static_cast<void>(eager.step_known(std::span(tokens).subspan(512,1),std::span(rows).subspan(512*256,256),std::span(memory).subspan(512*4,4),true));
            require(candidate.accepted_snapshot()==eager.accepted_snapshot(),"N3 verifier control starts from different input512 state");
            const auto proposals=std::span(tokens).subspan(513,3);const auto lookup=candidate.sampled_verification_depth2_lookups(proposals);std::vector<float> input_memory(12);
            for(uint32_t t=0;t<3;++t)for(uint32_t j=0;j<4;++j)input_memory[t*4+j]=float(int(lookup[t].rows[j]%17)-8)/32;
            const auto input_rows=std::span(rows).subspan(513*256,3*256);const auto before=candidate.accepted_snapshot();const std::vector<float> hc(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end());
            struct SelectN3{uint32_t consumed;bool throwing=false;static uint32_t choose(void* p,const StackSampledVerificationRows& r){auto& self=*static_cast<SelectN3*>(p);require(r.count==3,"N3 capacity512 verifier lost row count");if(self.throwing)throw FormatError("generated N3 selector failure");return self.consumed;}} selection{consumed,true};
            reject([&]{static_cast<void>(candidate.verify_sampled_depth2(proposals,input_rows,input_memory,513,{SelectN3::choose,&selection}));});
            require(candidate.position()==513&&candidate.accepted_snapshot()==before&&std::equal(hc.begin(),hc.end(),candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end()),"N3 selector failure published accepted state/HC");
            selection.throwing=false;struct CancelN3{uint32_t polls=0;static bool requested(void* p){return ++static_cast<CancelN3*>(p)->polls==2;}} cancel;
            reject([&]{static_cast<void>(candidate.verify_sampled_depth2(proposals,input_rows,input_memory,513,{SelectN3::choose,&selection},{CancelN3::requested,&cancel}));});
            require(candidate.position()==513&&candidate.accepted_snapshot()==before,"N3 post-execution cancellation published accepted state");
            const auto routes_before=g.q8_three_rows_routes();const auto out=candidate.verify_sampled_depth2(proposals,input_rows,input_memory,513,{SelectN3::choose,&selection});const auto routes=g.q8_three_rows_routes();
            require(out.committed==consumed&&candidate.position()==513+consumed&&routes[0]>routes_before[0],"N3 input512 verification did not complete genuine three-row dispatches");
            StackResult expected;std::vector<float> expected_hc;for(uint32_t t=513;t<513+consumed;++t){expected=eager.step(tokens[t],std::span(rows).subspan(uint64_t(t)*256,256),std::span(memory).subspan(uint64_t(t)*4,4));const auto tap=eager.last_final_hc_residual();expected_hc.insert(expected_hc.end(),tap.begin(),tap.end());}
            require(std::equal(out.logits.begin(),out.logits.end(),expected.logits.begin(),expected.logits.end())&&candidate.accepted_snapshot()==eager.accepted_snapshot()&&std::equal(expected_hc.begin(),expected_hc.end(),candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end()),"N3 input512 sampled commit changed exact logits, accepted state or HC rows");
            const uint32_t next=513+consumed;const auto continued=candidate.step(tokens[next],std::span(rows).subspan(uint64_t(next)*256,256),std::span(memory).subspan(uint64_t(next)*4,4));const auto control=eager.step(tokens[next],std::span(rows).subspan(uint64_t(next)*256,256),std::span(memory).subspan(uint64_t(next)*4,4));require(continued.logits==control.logits&&candidate.accepted_snapshot()==eager.accepted_snapshot(),"N3 rejected-tail overwrite or later ordinary continuation changed");
        }
#endif
        const auto original_trunk=g.read(mutable_trunk);auto broken_trunk=original_trunk;const uint16_t invalid_scale=0x7c00;std::memcpy(broken_trunk.data(),&invalid_scale,2);
        const uint64_t final_dispatches=3+g.gemv_dispatches(f.weights.mixer.down.encoding,512,32)+g.gemv_dispatches(f.weights.mixer.up.encoding,32,512)+g.gemv_dispatches(f.weights.output.encoding,256,shape.vocabulary);
        for(const auto& plan:std::vector<std::vector<uint32_t>>{{1,511,512},{512,512},{512,37}}){
            candidate.reset();eager.reset();uint32_t at=0;
            for(size_t packet=0;packet<plan.size();++packet){const auto n=plan[packet];const bool final=packet+1==plan.size();
                const auto input=std::span(tokens).subspan(at,n);const auto embeddings=std::span(rows).subspan(uint64_t(at)*256,uint64_t(n)*256);const auto memories=std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4);
                const auto expected=eager.step_known(input,embeddings,memories);const auto state=candidate.accepted_snapshot(),physical=candidate.chunked_cache_snapshot();const std::vector<float> old_hc(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end());
                g.write(mutable_trunk,broken_trunk);reject([&]{static_cast<void>(candidate.step_known(input,embeddings,memories,final));});g.write(mutable_trunk,original_trunk);
                require(candidate.position()==at&&candidate.accepted_snapshot()==state&&candidate.chunked_cache_snapshot()==physical,"deferred/final failure lost published state or physical QSA rollback");
                require(std::equal(old_hc.begin(),old_hc.end(),candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end()),"deferred failure published candidate final HC rows");
                if(!final)g.write(f.weights.output.view,bad);const auto allocations=g.memory().allocations;
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
                const auto encoded_before=g.gdn_legacy_batch_encoded_calls();
#endif
                const auto actual=candidate.step_known(input,embeddings,memories,final);if(!final)g.write(f.weights.output.view,original_head);
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
                require(g.gdn_legacy_batch_encoded_calls()-encoded_before==(legacy_order&&n>1?1u:0u),"generated old-order route did not encode exactly the actual recurrent layer count");
#endif
                require(actual.logits_available==final&&(final?actual.logits==expected.logits:actual.logits.empty()),"deferred/final logits readiness or exact final result differs");
                require(actual.batch.dispatches+(final?0:final_dispatches)==expected.batch.dispatches&&g.memory().allocations==allocations,"deferred dispatch/allocation accounting differs from independent final output plan");
                require(candidate.accepted_snapshot()==eager.accepted_snapshot()&&candidate.chunked_cache_snapshot()==eager.chunked_cache_snapshot(),"omitting final output changed accepted trunk state or physical cache");
                require(candidate.last_final_hc_count()==n&&std::equal(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end(),eager.last_final_hc_rows().begin(),eager.last_final_hc_rows().end()),"omitting final output changed HC rows used for MTP priming");
                at+=n;
            }
            if(at<tokens.size()){const auto actual=candidate.step(tokens[at],std::span(rows).subspan(uint64_t(at)*256,256),std::span(memory).subspan(uint64_t(at)*4,4));const auto expected=eager.step(tokens[at],std::span(rows).subspan(uint64_t(at)*256,256),std::span(memory).subspan(uint64_t(at)*4,4));require(actual.logits_available&&actual.logits==expected.logits,"ordinary continuation after deferred prefill changed");}
        }

#endif
    }
    require(g.memory().live_bytes==base,"chunked stack resource leak");
}
#if defined(TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32) && defined(TOKSPUR_PRIVATE_HTTP_PREFILL512)
// The accepted snapshot is typed in the existing stable order: exact prefix/history, F32 state, then visible BF16 QSA cache rows.
void hc_down_state_compare(std::span<const std::byte> actual, std::span<const std::byte> expected, StackShape shape, uint32_t prefix) {
    require(actual.size() == expected.size(), "HC down accepted-state snapshot extent");
    size_t offset = sizeof(uint32_t) + sizeof(NgramHistory{}.previous);
    require(actual.size() >= offset && std::equal(actual.begin(), actual.begin() + offset, expected.begin()), "HC down prefix/history changed");
    const auto part = [&](uint64_t count, bool bf16) {
        const auto bytes = checked_mul(count, bf16 ? 2 : 4); require(bytes <= actual.size() - offset, "HC down state section extent");
        double error = 0, signal = 0;
        for (uint64_t i = 0; i < count; ++i) {
            float x, y;
            if (bf16) { uint16_t a, b; std::memcpy(&a, actual.data() + offset + i * 2, 2); std::memcpy(&b, expected.data() + offset + i * 2, 2); x = read_bf16(a); y = read_bf16(b); }
            else { std::memcpy(&x, actual.data() + offset + i * 4, 4); std::memcpy(&y, expected.data() + offset + i * 4, 4); }
            require(std::isfinite(x) && std::isfinite(y), "nonfinite HC down accepted state"); const double d = double(x) - y; error += d * d; signal += double(y) * y;
        }
        const auto nmse = signal ? error / signal : error; require(nmse <= 1e-6, "HC down accepted-state fixed NMSE gate"); worst = std::max(worst, nmse); offset += bytes;
    };
    part(uint64_t(shape.memory.branches) * shape.memory.channels * (shape.memory.kernel - 1) * shape.memory.dilation, false);
    part(gdn_state_count(shape.recurrent), false); part(gdn_qkv_count(shape.recurrent) * (shape.recurrent.kernel - 1), false);
    part(shape.attention.index_width, false);
    part(uint64_t(prefix) * shape.attention.kv_heads * shape.attention.head_width, true);
    part(uint64_t(prefix) * shape.attention.kv_heads * shape.attention.head_width, true);
    part(uint64_t(prefix / shape.attention.compression) * shape.attention.index_width, true);
    require(offset == actual.size(), "HC down accepted-state decoder omitted bytes");
}
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
struct OperatorStateEvidence {
    std::vector<std::vector<std::byte>> logits, hc, accepted, physical;
    std::vector<uint32_t> prefixes;
    std::vector<bool> ready;
    bool operator==(const OperatorStateEvidence&) const = default;
};
struct OperatorStateSink {
    uint64_t owner=0,physical_batches=0,closed_owners=0; BatchResult sum{};
    static bool emit(void* p,const OperatorCheckpointRecord& tag,const BatchResult* physical,const BatchResult& total) noexcept {
        try {
            auto& self=*static_cast<OperatorStateSink*>(p);if(self.owner!=tag.owner_batch){self.owner=tag.owner_batch;self.sum={};}
            if(physical){++self.physical_batches;self.sum.dispatches+=physical->dispatches;self.sum.wall_ms+=physical->wall_ms;self.sum.gpu_ms+=physical->gpu_ms;}
            require(total.dispatches==self.sum.dispatches&&total.wall_ms==self.sum.wall_ms&&total.gpu_ms==self.sum.gpu_ms,"stack diagnostic owner rollup differs from its physical batches");
            if(tag.owner_final)++self.closed_owners;return true;
        }catch(...){return false;}
    }
};
#endif
void hc_down_actual_state_case(const char* library, bool enabled, bool concurrent
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    ,OperatorStateEvidence* evidence=nullptr,bool diagnostic=false
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    ,bool staged_dense=false
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    ,bool hc_rows=false
#endif
#endif
#endif
    ) {
    // Actual HC width/rank, coupled trunk channels and minimal supported core/FFN dimensions. Old F32 fixture weights remain owned.
    const StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,16,1024},{2560,32,2,1},{4,2560,4,4,3},13,1};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    constexpr uint64_t graph_budget = 1ull << 30;
    MetalGraph g(library, graph_budget); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    g.configure_prefill_dense_f32(true); g.configure_prefill_grouped_f32(true); g.configure_prefill_hc_down_f32(enabled);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    g.configure_concurrent_dispatch(concurrent);
#else
    require(!concurrent, "HC down concurrent fixture requires the concurrent build");
#endif
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    OperatorStateSink checkpoint_sink;if(diagnostic)g.configure_operator_checkpoints(OperatorStateSink::emit,&checkpoint_sink,16384);
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    if(staged_dense){
#ifndef TOKSPUR_PRIVATE_HC_DOWN_STAGED
        require(!enabled,"operator staged state control requires HC down off");
#endif
        g.configure_prefill_dense_staged_f32(true);
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    if(hc_rows){require(staged_dense,"HC row state control requires staged dense");g.configure_prefill_hc_rows(true);}
#endif
#endif
#endif
    const auto expected_hc_down = [&](uint32_t n) {
#if defined(TOKSPUR_PRIVATE_HC_DOWN_STAGED) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS)
        return enabled && (staged_dense ? n >= 2 && n <= 512 : n == 512);
#else
        return enabled && n == 512;
#endif
    };
    const auto initial = g.memory().live_bytes;
    {
        Assets a{g,std::mt19937(44),{},{},{}}; const std::array<uint32_t,1> attention_layers{1}; auto f = fixture(a,shape,2,attention_layers);
        for (auto& layer : f.weights.layers) std::visit([&](auto& w) {
            for (auto* read : {&w.attention_read,&w.ffn_read}) { read->down=stack_q8(a,10240,320); read->up=stack_q8(a,320,10240); }
            auto& m=w.ffn; m.gate=stack_q8(a,2560,32,2); m.up=stack_q8(a,2560,32,2); m.down=stack_q8(a,32,2560,2);
            m.shared_gate=stack_q8(a,2560,32); m.shared_up=stack_q8(a,2560,32); m.shared_down=stack_q8(a,32,2560);
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>) {
                w.attention.qkv=stack_q8(a,2560,uint32_t(gdn_qkv_count(shape.recurrent))); w.attention.gate=stack_q8(a,2560,64); w.attention.output=stack_q8(a,64,2560);
            } else {
                auto& q=w.attention; q.query_gate=stack_q8(a,2560,256); q.key=stack_q8(a,2560,64); q.value=stack_q8(a,2560,64); q.output=stack_q8(a,128,2560);
                q.index_query=stack_bf16(a,q.index_query); q.index_key=stack_bf16(a,q.index_key);
            }
        }, layer);
        auto& recurrent_weight=std::get<RecurrentWeights>(f.weights.layers[0]).attention.qkv;
        const auto mutable_trunk=g.upload(g.read(recurrent_weight.view),"generated actual-HC mutable trunk Q8",false); a.views.push_back(mutable_trunk); recurrent_weight.view=mutable_trunk;
        const auto q8_head=stack_q8(a,2560,shape.vocabulary); const auto mutable_head=g.upload(g.read(q8_head.view),"generated actual-HC mutable head Q8",false); a.views.push_back(mutable_head); f.weights.output={mutable_head,8};
        const auto original_head=g.read(mutable_head), original_trunk=g.read(mutable_trunk); auto bad_head=original_head,bad_trunk=original_trunk; const uint16_t invalid=0x7c00;
        std::memcpy(bad_head.data(),&invalid,2); std::memcpy(bad_trunk.data(),&invalid,2);
        uint64_t retained_host_payload=0; for (const auto& v:a.floats) retained_host_payload+=v.capacity()*sizeof(float); for (const auto& v:a.doubles) retained_host_payload+=v.capacity()*sizeof(double);
        const auto base=g.memory().live_bytes;
        constexpr uint32_t count=550; std::vector<uint32_t> tokens(count); std::vector<float> rows(uint64_t(count)*2560),memory(uint64_t(count)*4); auto history=empty_ngram_history(spec);
        for(uint32_t t=0;t<count;++t) { tokens[t]=(t*3)%13; const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]); history=ids.next;
            for(uint32_t j=0;j<2560;++j) rows[uint64_t(t)*2560+j]=float(int((t*13+j*7)%131)-65)/997;
            for(uint32_t j=0;j<4;++j) memory[uint64_t(t)*4+j]=float(int(ids.rows[j]%17)-8)/32;
        }
        {
            TextStack candidate(g,f.weights,shape,policy,spec,StackExecution::direct,512,true); candidate.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20); candidate.prepare_final_hc_tap(64ull<<20);
            const auto run=[&](uint32_t at,uint32_t n,bool emit=true) {
                const auto routes=g.hc_exact_routes(); const auto allocations=g.memory().allocations;
                const auto encoded_before=g.prefill_hc_down_f32_encoded_calls();
                auto out=candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4),emit);
                const auto after=g.hc_exact_routes(); require(after[0]-routes[0]==(expected_hc_down(n)?0:4ull*n),"actual-HC active/fallback physical route count");
                require(g.prefill_hc_down_f32_encoded_calls()-encoded_before==(expected_hc_down(n)?4u:0u),"actual-HC encoded route count differs from its four eligible call sites");
                require(g.memory().allocations==allocations,"actual-HC chunk allocated during encoding");
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
                if(evidence){const auto lb=std::as_bytes(std::span(out.logits)),hb=std::as_bytes(candidate.last_final_hc_rows());evidence->logits.emplace_back(lb.begin(),lb.end());evidence->hc.emplace_back(hb.begin(),hb.end());evidence->accepted.push_back(candidate.accepted_snapshot());evidence->physical.push_back(candidate.chunked_cache_snapshot());evidence->prefixes.push_back(candidate.position());evidence->ready.push_back(out.logits_available);}
#endif
                return out;
            };
            const auto fail=[&](uint32_t at,uint32_t n,bool emit,bool trunk) {
                const auto accepted=candidate.accepted_snapshot(),physical=candidate.chunked_cache_snapshot(); const std::vector<float> hc(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end()); const auto routes=g.hc_exact_routes();
                const auto encoded_before=g.prefill_hc_down_f32_encoded_calls();
                g.write(trunk?mutable_trunk:mutable_head,trunk?bad_trunk:bad_head);
                reject([&] { static_cast<void>(candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4),emit)); });
                g.write(trunk?mutable_trunk:mutable_head,trunk?original_trunk:original_head);
                const auto encoded_after=g.prefill_hc_down_f32_encoded_calls();
                require(encoded_after>=encoded_before&&encoded_after-encoded_before<=(expected_hc_down(n)?4u:0u),"failed actual-HC encoded calls rewound or crossed their eligible boundary");
                require(candidate.position()==at&&candidate.accepted_snapshot()==accepted&&candidate.chunked_cache_snapshot()==physical,"actual-HC failure lost accepted state or physical QSA rollback");
                require(std::equal(hc.begin(),hc.end(),candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end())&&g.hc_exact_routes()==routes,"actual-HC failure published HC rows or successful routes");
            };
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
            if(evidence){
                fail(0,512,true,false);static_cast<void>(run(0,512));fail(512,37,true,false);static_cast<void>(run(512,37));fail(549,1,true,false);static_cast<void>(run(549,1));
                const auto encoded=g.prefill_hc_down_f32_encoded_calls();candidate.reset();require(g.prefill_hc_down_f32_encoded_calls()==encoded,"diagnostic reset rewound encoded lifetime");
                fail(0,512,false,true);static_cast<void>(run(0,512,false));static_cast<void>(run(512,1,true));
                require(evidence->prefixes==std::vector<uint32_t>({512,549,550,512,513}),"diagnostic state evidence lost its bounded lineage");
            }else
#endif
            {
            {
                TextStack serial(g,f.weights,shape,policy,spec); serial.prepare_final_hc_tap(1ull<<20);
                for (uint32_t at : {0u,512u}) {
                    const uint32_t n=at?37:512; fail(at,n,true,false); const auto actual=run(at,n); std::vector<float> hc; hc.reserve(uint64_t(n)*10240); StackResult expected;
                    for(uint32_t t=at;t<at+n;++t) { expected=serial.step(tokens[t],std::span(rows).subspan(uint64_t(t)*2560,2560),std::span(memory).subspan(uint64_t(t)*4,4)); const auto tap=serial.last_final_hc_residual(); hc.insert(hc.end(),tap.begin(),tap.end()); }
                    require(actual.logits_available&&candidate.position()==at+n&&candidate.last_final_hc_count()==n,"actual-HC chunk publication/readiness");
                    stack_compare(actual.logits,expected.logits); stack_compare(candidate.last_final_hc_rows(),hc); hc_down_state_compare(candidate.accepted_snapshot(),serial.accepted_snapshot(),shape,at+n);
                }
                fail(549,1,true,false); const auto actual=run(549,1); const auto expected=serial.step(tokens[549],std::span(rows).subspan(549*2560,2560),std::span(memory).subspan(549*4,4)); stack_compare(actual.logits,expected.logits); hc_down_state_compare(candidate.accepted_snapshot(),serial.accepted_snapshot(),shape,550);
                const auto encoded_reset_before=g.prefill_hc_down_f32_encoded_calls(); candidate.reset(); require(g.prefill_hc_down_f32_encoded_calls()==encoded_reset_before,"reset rewound the HC down encoded lifetime"); serial.reset(); require(candidate.position()==0,"actual-HC reset prefix"); const auto reset=run(0,3); StackResult ordinary;
                for(uint32_t t=0;t<3;++t) ordinary=serial.step(tokens[t],std::span(rows).subspan(uint64_t(t)*2560,2560),std::span(memory).subspan(uint64_t(t)*4,4));
                stack_compare(reset.logits,ordinary.logits); stack_compare(candidate.last_final_hc_residual(),serial.last_final_hc_residual()); hc_down_state_compare(candidate.accepted_snapshot(),serial.accepted_snapshot(),shape,3);
            }
            {
                candidate.reset(); TextStack eager(g,f.weights,shape,policy,spec,StackExecution::direct,512,true); eager.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20); eager.prepare_final_hc_tap(64ull<<20);
                for(uint32_t at : {0u,512u}) {
                    const uint32_t n=at?1:512; const bool final=at!=0; fail(at,n,final,!final);
                    const auto expected=eager.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4));
                    if(!final)g.write(mutable_head,bad_head); const auto actual=run(at,n,final); if(!final)g.write(mutable_head,original_head);
                    require(actual.logits_available==final&&(final?actual.logits==expected.logits:actual.logits.empty()),"actual-HC deferred/final readiness changed");
                    require(candidate.accepted_snapshot()==eager.accepted_snapshot()&&candidate.last_final_hc_count()==n&&std::equal(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end(),eager.last_final_hc_rows().begin(),eager.last_final_hc_rows().end()),"actual-HC deferred path changed exact same-policy state or HC rows");
                }
            }
            }
        }
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        if(diagnostic)require(checkpoint_sink.physical_batches>checkpoint_sink.closed_owners&&checkpoint_sink.closed_owners>=5,"diagnostic stack did not exercise genuine checkpointed owners");
#endif
        require(g.memory().live_bytes==base,"actual-HC stack fixture leaked resources");
        std::cout<<"Actual-HC generated state mode="<<(enabled?"on":"off")<<" concurrent="<<concurrent<<" graph_budget="<<graph_budget<<" retained_fixture_gpu_bytes="<<base-initial<<" retained_host_reference_payload_bytes="<<retained_host_payload<<" graph_peak_bytes="<<g.memory().peak_bytes<<" worst_fixed_nmse="<<worst<<" no model/fidelity or timing admission\n";
    }
    require(g.memory().live_bytes==initial,"actual-HC fixture assets leaked");
}
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
#ifdef TOKSPUR_PRIVATE_HC_ROWS
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
#include "route_attribution_controls.inc"
#include "kernel_search_capture_controls.inc"
#include "kernel_step2_controls.inc"
void operator_hc_down_staged_state_pair(const char* library, bool concurrent) {
    const StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,16,1024},{2560,32,2,1},{4,2560,4,4,3},13,1};
    OperatorStateEvidence baseline, ordinary;
    hc_down_actual_state_case(library, false, concurrent, &baseline, false, true, true);
    for (bool diagnostic : {false, true}) {
        OperatorStateEvidence candidate; hc_down_actual_state_case(library, true, concurrent, &candidate, diagnostic, true, true);
        require(candidate.prefixes == baseline.prefixes && candidate.ready == baseline.ready && candidate.logits.size() == baseline.logits.size(), "staged HC down changed publication boundaries");
        const auto compare = [](const std::vector<std::byte>& actual, const std::vector<std::byte>& expected) {
            require(actual.size() == expected.size() && actual.size() % sizeof(float) == 0, "staged HC down F32 evidence extent");
            if (actual.empty()) return;
            std::vector<float> a(actual.size() / sizeof(float)), b(expected.size() / sizeof(float));
            std::memcpy(a.data(), actual.data(), actual.size()); std::memcpy(b.data(), expected.data(), expected.size()); stack_compare(a, b);
        };
        for (size_t i = 0; i < candidate.logits.size(); ++i) {
            compare(candidate.logits[i], baseline.logits[i]); compare(candidate.hc[i], baseline.hc[i]);
            hc_down_state_compare(candidate.accepted[i], baseline.accepted[i], shape, candidate.prefixes[i]);
        }
        if (!diagnostic) ordinary = candidate;
        else require(candidate == ordinary, "staged HC down diagnostic changed exact same-policy state or physical cache");
    }
    std::cout << "Staged HC down off/on generated full-logit and accepted-state NMSE gates pass; failed submissions restore exact physical cache within each policy; concurrent=" << concurrent << " worst_fixed_nmse=" << worst << " no model or latency admission\n";
}
#endif
void operator_hc_rows_state_pair(const char* library,bool concurrent){
    OperatorStateEvidence baseline;hc_down_actual_state_case(library,false,concurrent,&baseline,false,true,false);
    for(bool diagnostic:{false,true}){OperatorStateEvidence candidate;hc_down_actual_state_case(library,false,concurrent,&candidate,diagnostic,true,true);require(candidate==baseline,"HC rows changed exact logits, HC, accepted state or physical rollback");}
    std::cout<<"Generated exact HC row batching full-state and physical-rollback pair pass; concurrent="<<concurrent<<" no model or timing admission\n";
}
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
OperatorStateEvidence router_window_actual_state_case(const char* library,bool router_window,bool concurrent,bool diagnostic) {
    const StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,16,1024},{2560,32,512,10},{4,2560,4,4,3},13,1};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    MetalGraph g(library,2ull<<30);g.configure_small_gemv(16);g.configure_block_gemv(3);g.configure_hc_exact(HcExactMode::combined);
    g.configure_prefill_dense_f32(true);g.configure_prefill_grouped_f32(true);g.configure_prefill_dense_staged_f32(true);g.configure_prefill_hc_rows(true);
    g.configure_prefill_router_window(router_window);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    g.configure_concurrent_dispatch(concurrent);
#else
    require(!concurrent,"router window concurrent fixture requires the concurrent build");
#endif
    OperatorStateSink checkpoint_sink;if(diagnostic)g.configure_operator_checkpoints(OperatorStateSink::emit,&checkpoint_sink,16384);
    const auto initial=g.memory().live_bytes;
    OperatorStateEvidence evidence;
    {
        Assets a{g,std::mt19937(44),{},{},{}};auto small=shape;small.experts.experts=2;small.experts.selected=1;const std::array<uint32_t,1> attention_layers{1};auto f=fixture(a,small,2,attention_layers);
        for(auto& layer:f.weights.layers)std::visit([&](auto& w){
            for(auto* read:{&w.attention_read,&w.ffn_read}){read->down=stack_q8(a,10240,320);read->up=stack_q8(a,320,10240);}
            auto& m=w.ffn;m.router=a.matrix(2560*512).first;m.gate=stack_q8(a,2560,32,512);m.up=stack_q8(a,2560,32,512);m.down=stack_q8(a,32,2560,512);m.shared_gate=stack_q8(a,2560,32);m.shared_up=stack_q8(a,2560,32);m.shared_down=stack_q8(a,32,2560);
            if constexpr(std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>){w.attention.qkv=stack_q8(a,2560,uint32_t(gdn_qkv_count(shape.recurrent)));w.attention.gate=stack_q8(a,2560,64);w.attention.output=stack_q8(a,64,2560);}
            else {auto& q=w.attention;q.query_gate=stack_q8(a,2560,256);q.key=stack_q8(a,2560,64);q.value=stack_q8(a,2560,64);q.output=stack_q8(a,128,2560);q.index_query=stack_bf16(a,q.index_query);q.index_key=stack_bf16(a,q.index_key);}
        },layer);
        auto& recurrent_weight=std::get<RecurrentWeights>(f.weights.layers[0]).attention.qkv;
        const auto mutable_trunk=g.upload(g.read(recurrent_weight.view),"generated router-window mutable trunk Q8",false);a.views.push_back(mutable_trunk);recurrent_weight.view=mutable_trunk;
        const auto q8_head=stack_q8(a,2560,shape.vocabulary);const auto mutable_head=g.upload(g.read(q8_head.view),"generated router-window mutable head Q8",false);a.views.push_back(mutable_head);f.weights.output={mutable_head,8};
        const auto original_head=g.read(mutable_head),original_trunk=g.read(mutable_trunk);auto bad_head=original_head,bad_trunk=original_trunk;const uint16_t invalid=0x7c00;
        std::memcpy(bad_head.data(),&invalid,2);std::memcpy(bad_trunk.data(),&invalid,2);
        const auto base=g.memory().live_bytes;
        constexpr uint32_t count=550;std::vector<uint32_t> tokens(count);std::vector<float> rows(uint64_t(count)*2560),memory(uint64_t(count)*4);auto history=empty_ngram_history(spec);
        for(uint32_t t=0;t<count;++t){tokens[t]=(t*3)%13;const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]);history=ids.next;
            for(uint32_t j=0;j<2560;++j)rows[uint64_t(t)*2560+j]=float(int((t*13+j*7)%131)-65)/997;
            for(uint32_t j=0;j<4;++j)memory[uint64_t(t)*4+j]=float(int(ids.rows[j]%17)-8)/32;
        }
        {
            TextStack candidate(g,f.weights,shape,policy,spec,StackExecution::direct,512,true);candidate.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20);candidate.prepare_final_hc_tap(64ull<<20);
            const auto run=[&](uint32_t at,uint32_t n,bool emit=true){
                const auto allocations=g.memory().allocations;const auto encoded_before=g.router_window_encoded();
                auto out=candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4),emit);
                const auto encoded_after=g.router_window_encoded();const uint64_t expected_delta=(router_window&&n>=2&&n<=512)?2u:0u;
                for(size_t i=0;i<4;++i)require(encoded_after[i]-encoded_before[i]==expected_delta,"router window encoded call delta differs");
                require(g.memory().allocations==allocations,"router window chunk allocated during encoding");
                const auto lb=std::as_bytes(std::span(out.logits)),hb=std::as_bytes(candidate.last_final_hc_rows());
                evidence.logits.emplace_back(lb.begin(),lb.end());evidence.hc.emplace_back(hb.begin(),hb.end());evidence.accepted.push_back(candidate.accepted_snapshot());evidence.physical.push_back(candidate.chunked_cache_snapshot());evidence.prefixes.push_back(candidate.position());evidence.ready.push_back(out.logits_available);
                return out;
            };
            const auto fail=[&](uint32_t at,uint32_t n,bool emit,bool trunk){
                const auto accepted=candidate.accepted_snapshot(),physical=candidate.chunked_cache_snapshot();const std::vector<float> hc(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end());
                const auto encoded_before=g.router_window_encoded();
                g.write(trunk?mutable_trunk:mutable_head,trunk?bad_trunk:bad_head);
                reject([&]{static_cast<void>(candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4),emit));});
                g.write(trunk?mutable_trunk:mutable_head,trunk?original_trunk:original_head);
                const auto encoded_after=g.router_window_encoded();const uint64_t max_delta=(router_window&&n>=2&&n<=512)?2u:0u;
                for(size_t i=0;i<4;++i)require(encoded_after[i]>=encoded_before[i]&&encoded_after[i]-encoded_before[i]<=max_delta,"failed router window encoded calls rewound or crossed boundary");
                require(candidate.position()==at&&candidate.accepted_snapshot()==accepted&&candidate.chunked_cache_snapshot()==physical,"router window failure lost accepted state or physical QSA rollback");
                require(std::equal(hc.begin(),hc.end(),candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end()),"router window failure published HC rows");
            };
            fail(0,512,true,false);static_cast<void>(run(0,512));fail(512,37,true,false);static_cast<void>(run(512,37));fail(549,1,true,false);static_cast<void>(run(549,1));
            const auto encoded=g.router_window_encoded();candidate.reset();require(g.router_window_encoded()==encoded,"router window reset rewound encoded lifetime");
            fail(0,512,false,true);static_cast<void>(run(0,512,false));static_cast<void>(run(512,1,true));
            require(evidence.prefixes==std::vector<uint32_t>({512,549,550,512,513}),"router window state evidence lost its bounded lineage");
        }
        if(diagnostic)require(checkpoint_sink.physical_batches>checkpoint_sink.closed_owners&&checkpoint_sink.closed_owners>=5,"router window diagnostic stack did not exercise genuine checkpointed owners");
        require(g.memory().live_bytes==base,"router window stack fixture leaked resources");
    }
    require(g.memory().live_bytes==initial,"router window fixture assets leaked");
    return evidence;
}
void operator_router_window_state_pair(const char* library,bool concurrent){
    const auto baseline=router_window_actual_state_case(library,false,concurrent,false);
    for(bool diagnostic:{false,true}){const auto candidate=router_window_actual_state_case(library,true,concurrent,diagnostic);require(candidate==baseline,"router window changed exact logits, HC, accepted state or physical rollback");}
    std::cout<<"Generated exact MoE router window full-state and physical-rollback pair pass; concurrent="<<concurrent<<" no model or timing admission\n";
}
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
OperatorStateEvidence hc_down_window_actual_state_case(const char* library,bool hc_down_window,bool concurrent,bool diagnostic) {
    const StackShape shape{{4,2560,320},{2560,1,2,128,32,4},{2560,4,2,32,2,32,32,4,16,1024},{2560,32,512,10},{4,2560,4,4,3},13,1};
    const StackPolicy policy{{0.08838835f,1e-6f,L2Mode::add_squared_epsilon,1e-6f},{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},RouterMode::softmax,true,{HashOrder::newest_first,NgramBoundary(0),true}};
    const NgramSpec spec{3,2,1,13,11,12,{3,5,7},{0,7,18,31},{7,11,13,17}};
    MetalGraph g(library,2ull<<30);g.configure_small_gemv(16);g.configure_block_gemv(3);g.configure_hc_exact(HcExactMode::combined);
    g.configure_prefill_dense_f32(true);g.configure_prefill_grouped_f32(true);g.configure_prefill_dense_staged_f32(true);g.configure_prefill_hc_rows(true);
    g.configure_prefill_router_window(true);g.configure_prefill_hc_down_window(hc_down_window);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    g.configure_concurrent_dispatch(concurrent);
#else
    require(!concurrent,"HC down window concurrent fixture requires the concurrent build");
#endif
    OperatorStateSink checkpoint_sink;if(diagnostic)g.configure_operator_checkpoints(OperatorStateSink::emit,&checkpoint_sink,16384);
    const auto initial=g.memory().live_bytes;
    OperatorStateEvidence evidence;
    {
        Assets a{g,std::mt19937(44),{},{},{}};auto small=shape;small.experts.experts=2;small.experts.selected=1;const std::array<uint32_t,1> attention_layers{1};auto f=fixture(a,small,2,attention_layers);
        for(auto& layer:f.weights.layers)std::visit([&](auto& w){
            for(auto* read:{&w.attention_read,&w.ffn_read}){read->down=stack_q8(a,10240,320);read->up=stack_q8(a,320,10240);}
            auto& m=w.ffn;m.router=a.matrix(2560*512).first;m.gate=stack_q8(a,2560,32,512);m.up=stack_q8(a,2560,32,512);m.down=stack_q8(a,32,2560,512);m.shared_gate=stack_q8(a,2560,32);m.shared_up=stack_q8(a,2560,32);m.shared_down=stack_q8(a,32,2560);
            if constexpr(std::is_same_v<std::decay_t<decltype(w)>,RecurrentWeights>){w.attention.qkv=stack_q8(a,2560,uint32_t(gdn_qkv_count(shape.recurrent)));w.attention.gate=stack_q8(a,2560,64);w.attention.output=stack_q8(a,64,2560);}
            else {auto& q=w.attention;q.query_gate=stack_q8(a,2560,256);q.key=stack_q8(a,2560,64);q.value=stack_q8(a,2560,64);q.output=stack_q8(a,128,2560);q.index_query=stack_bf16(a,q.index_query);q.index_key=stack_bf16(a,q.index_key);}
        },layer);
        auto& recurrent_weight=std::get<RecurrentWeights>(f.weights.layers[0]).attention.qkv;
        const auto mutable_trunk=g.upload(g.read(recurrent_weight.view),"generated hc-down-window mutable trunk Q8",false);a.views.push_back(mutable_trunk);recurrent_weight.view=mutable_trunk;
        const auto q8_head=stack_q8(a,2560,shape.vocabulary);const auto mutable_head=g.upload(g.read(q8_head.view),"generated hc-down-window mutable head Q8",false);a.views.push_back(mutable_head);f.weights.output={mutable_head,8};
        const auto original_head=g.read(mutable_head),original_trunk=g.read(mutable_trunk);auto bad_head=original_head,bad_trunk=original_trunk;const uint16_t invalid=0x7c00;
        std::memcpy(bad_head.data(),&invalid,2);std::memcpy(bad_trunk.data(),&invalid,2);
        const auto base=g.memory().live_bytes;
        constexpr uint32_t count=550;std::vector<uint32_t> tokens(count);std::vector<float> rows(uint64_t(count)*2560),memory(uint64_t(count)*4);auto history=empty_ngram_history(spec);
        for(uint32_t t=0;t<count;++t){tokens[t]=(t*3)%13;const auto ids=ngram_step(spec,policy.ngram,history,tokens[t]);history=ids.next;
            for(uint32_t j=0;j<2560;++j)rows[uint64_t(t)*2560+j]=float(int((t*13+j*7)%131)-65)/997;
            for(uint32_t j=0;j<4;++j)memory[uint64_t(t)*4+j]=float(int(ids.rows[j]%17)-8)/32;
        }
        {
            TextStack candidate(g,f.weights,shape,policy,spec,StackExecution::direct,512,true);candidate.prepare_chunked_prefill(GdnChunkMode::fused,512ull<<20);candidate.prepare_final_hc_tap(64ull<<20);
            const auto run=[&](uint32_t at,uint32_t n,bool emit=true){
                const auto allocations=g.memory().allocations;const auto encoded_before=g.hc_down_window_encoded();
                auto out=candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4),emit);
                const auto encoded_after=g.hc_down_window_encoded();const uint64_t expected_delta=(hc_down_window&&n>=2&&n<=512)?4u:0u;
                require(encoded_after-encoded_before==expected_delta,"HC down window encoded call delta differs");
                require(g.memory().allocations==allocations,"HC down window chunk allocated during encoding");
                const auto lb=std::as_bytes(std::span(out.logits)),hb=std::as_bytes(candidate.last_final_hc_rows());
                evidence.logits.emplace_back(lb.begin(),lb.end());evidence.hc.emplace_back(hb.begin(),hb.end());evidence.accepted.push_back(candidate.accepted_snapshot());evidence.physical.push_back(candidate.chunked_cache_snapshot());evidence.prefixes.push_back(candidate.position());evidence.ready.push_back(out.logits_available);
                return out;
            };
            const auto fail=[&](uint32_t at,uint32_t n,bool emit,bool trunk){
                const auto accepted=candidate.accepted_snapshot(),physical=candidate.chunked_cache_snapshot();const std::vector<float> hc(candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end());
                const auto encoded_before=g.hc_down_window_encoded();
                g.write(trunk?mutable_trunk:mutable_head,trunk?bad_trunk:bad_head);
                reject([&]{static_cast<void>(candidate.step_known(std::span(tokens).subspan(at,n),std::span(rows).subspan(uint64_t(at)*2560,uint64_t(n)*2560),std::span(memory).subspan(uint64_t(at)*4,uint64_t(n)*4),emit));});
                g.write(trunk?mutable_trunk:mutable_head,trunk?original_trunk:original_head);
                const auto encoded_after=g.hc_down_window_encoded();const uint64_t max_delta=(hc_down_window&&n>=2&&n<=512)?4u:0u;
                require(encoded_after>=encoded_before&&encoded_after-encoded_before<=max_delta,"failed HC down window encoded calls rewound or crossed boundary");
                require(candidate.position()==at&&candidate.accepted_snapshot()==accepted&&candidate.chunked_cache_snapshot()==physical,"HC down window failure lost accepted state or physical QSA rollback");
                require(std::equal(hc.begin(),hc.end(),candidate.last_final_hc_rows().begin(),candidate.last_final_hc_rows().end()),"HC down window failure published HC rows");
            };
            fail(0,512,true,false);static_cast<void>(run(0,512));fail(512,37,true,false);static_cast<void>(run(512,37));fail(549,1,true,false);static_cast<void>(run(549,1));
            const auto encoded=g.hc_down_window_encoded();candidate.reset();require(g.hc_down_window_encoded()==encoded,"HC down window reset rewound encoded lifetime");
            fail(0,512,false,true);static_cast<void>(run(0,512,false));static_cast<void>(run(512,1,true));
            require(evidence.prefixes==std::vector<uint32_t>({512,549,550,512,513}),"HC down window state evidence lost its bounded lineage");
        }
        if(diagnostic)require(checkpoint_sink.physical_batches>checkpoint_sink.closed_owners&&checkpoint_sink.closed_owners>=5,"HC down window diagnostic stack did not exercise genuine checkpointed owners");
        require(g.memory().live_bytes==base,"HC down window stack fixture leaked resources");
    }
    require(g.memory().live_bytes==initial,"HC down window fixture assets leaked");
    return evidence;
}
void operator_hc_down_window_state_pair(const char* library,bool concurrent){
    const auto baseline=hc_down_window_actual_state_case(library,false,concurrent,false);
    for(bool diagnostic:{false,true}){const auto candidate=hc_down_window_actual_state_case(library,true,concurrent,diagnostic);require(candidate==baseline,"HC down window changed exact logits, HC, accepted state or physical rollback");}
    std::cout<<"Generated exact HC down window full-state and physical-rollback pair pass; concurrent="<<concurrent<<" no model or timing admission\n";
}
#include "qsa_sparse_controls.inc"
#include "gdn_ported_controls.inc"
#endif
#endif
#endif
void operator_staged_dense_state_pair(const char* library,bool concurrent){
    OperatorStateEvidence baseline;hc_down_actual_state_case(library,false,concurrent,&baseline,false,false);
    for(bool diagnostic:{false,true}){OperatorStateEvidence candidate;hc_down_actual_state_case(library,false,concurrent,&candidate,diagnostic,true);require(candidate==baseline,"operator staged dense changed exact full logits, HC, accepted state, readiness or complete physical rollback");}
    std::cout<<"Generated operator staged dense exact state/physical rollback OFF/ON pair pass; concurrent="<<concurrent<<" HCdown=off no model or timing admission\n";
}
#endif
void operator_checkpoint_state_pair(const char* library,bool concurrent){
    for(bool enabled:{false,true}){OperatorStateEvidence ordinary,diagnostic;hc_down_actual_state_case(library,enabled,concurrent,&ordinary,false);hc_down_actual_state_case(library,enabled,concurrent,&diagnostic,true);require(ordinary==diagnostic,"diagnostic OFF/ON changed exact full logits, HC rows, accepted state or complete physical cache");}
    std::cout<<"Generated diagnostic OFF/ON exact full-state and physical-rollback pair pass; concurrent="<<concurrent<<" no model or timing admission\n";
}
#endif
#endif
}
int main(int argc,char**argv){try{
#if defined(TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32) && defined(TOKSPUR_PRIVATE_HTTP_PREFILL512)
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
#ifdef TOKSPUR_PRIVATE_HC_ROWS
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    if(argc>=3&&std::string(argv[2])=="kernel-search-capture"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"kernel capture controls usage");kernel_capture_controls(argv[1],argc==4);return 0;}
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    if(argc>=3&&std::string(argv[2])=="kernel-qsa-step2"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"QSA step-2 controls usage");kernel_step2_controls(argv[1],argc==4,false);return 0;}
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    if(argc>=3&&std::string(argv[2])=="kernel-grouped-step2"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"grouped step-2 controls usage");kernel_step2_controls(argv[1],argc==4,true);return 0;}
    if(argc>=3&&std::string(argv[2])=="kernel-grouped-route-lock"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"grouped route-lock controls usage");kernel_grouped_route_lock_controls(argv[1],argc==4);return 0;}
#endif
#endif
    if(argc>=3&&std::string(argv[2])=="route-lock"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"route-lock controls usage");route_lock_controls(argv[1],argc==4);return 0;}
#endif
    if(argc>=3&&std::string(argv[2])=="route-attribution"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"route controls usage");route_attribution_controls(argv[1],argc==4);return 0;}
#endif
    if(argc>=3&&std::string(argv[2])=="operator-hc-down-staged-state"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB operator-hc-down-staged-state [concurrent]");operator_hc_down_staged_state_pair(argv[1],argc==4);return 0;}
#endif
    if(argc>=3&&std::string(argv[2])=="operator-hc-rows-state"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB operator-hc-rows-state [concurrent]");operator_hc_rows_state_pair(argv[1],argc==4);return 0;}
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    if(argc>=3&&std::string(argv[2])=="operator-router-window-state"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB operator-router-window-state [concurrent]");operator_router_window_state_pair(argv[1],argc==4);return 0;}
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    if(argc>=3&&std::string(argv[2])=="operator-hc-down-window-state"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB operator-hc-down-window-state [concurrent]");operator_hc_down_window_state_pair(argv[1],argc==4);return 0;}
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    if(argc>=3&&std::string(argv[2])=="qsa-sparse-step5"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB qsa-sparse-step5 [concurrent]");qsa_sparse_step5_controls(argv[1],argc==4);return 0;}
    if(argc>=3&&std::string(argv[2])=="qsa-route-attribution"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB qsa-route-attribution [concurrent]");qsa_route_attribution_controls(argv[1],argc==4);return 0;}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    if(argc>=3&&std::string(argv[2])=="gdn-ported-step5"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB gdn-ported-step5 [concurrent]");gdn_ported_step5_controls(argv[1],argc==4);return 0;}
    if(argc>=3&&std::string(argv[2])=="gdn-route-attribution"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB gdn-route-attribution [concurrent]");gdn_route_attribution_controls(argv[1],argc==4);return 0;}
#endif
#endif
#endif
#endif
    if(argc>=3&&std::string(argv[2])=="operator-staged-dense-state"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB operator-staged-dense-state [concurrent]");operator_staged_dense_state_pair(argv[1],argc==4);return 0;}
#endif
    if(argc>=3&&std::string(argv[2])=="operator-checkpoints-state"){require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB operator-checkpoints-state [concurrent]");operator_checkpoint_state_pair(argv[1],argc==4);return 0;}
#endif
    if(argc>=3&&(std::string(argv[2])=="hc-down-f32-state-off"||std::string(argv[2])=="hc-down-f32-state-on")) {
        require(argc==3||(argc==4&&std::string(argv[3])=="concurrent"),"usage: test-chunked_stack OWNED_METALLIB hc-down-f32-state-off|hc-down-f32-state-on [concurrent]");
        hc_down_actual_state_case(argv[1],std::string(argv[2])=="hc-down-f32-state-on",argc==4); return 0;
    }
#endif
if(argc<2||argc>6)throw std::runtime_error("usage: test-chunked_stack OWNED_METALLIB [f32|matmul-f32|old-order] [n3] [concurrent] [raw-decode]");MetalGraph g(argv[1],512ull<<20);bool legacy_order=false;
    for(int i=2;i<argc;++i){
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if(std::string(argv[i])=="raw-decode"){g.configure_raw_decode(true);continue;}
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if(std::string(argv[i])=="n3"){g.configure_q8_three_rows(true);continue;}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        if(std::string(argv[i])=="old-order"){legacy_order=true;g.configure_prefill_dense_f32(true);g.configure_prefill_grouped_f32(true);continue;}
#endif
        if(std::string(argv[i])=="f32")g.configure_prefill_dense_f32(true);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        else if(std::string(argv[i])=="matmul-f32"){g.configure_prefill_dense_f32(true);g.configure_prefill_grouped_f32(true);}
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        else if(std::string(argv[i])=="concurrent")g.configure_concurrent_dispatch(true);
#endif
        else throw FormatError("unsupported test mode");}g.configure_small_gemv(16);g.configure_block_gemv(3);
    chunked_case(g,GdnChunkMode::fused,legacy_order);if(!legacy_order)chunked_case(g,GdnChunkMode::cs64);std::cout<<"Generated chunked stack checks "<<checks<<", worst logit NMSE "<<worst<<" pass; full physical cache rollback, no model admission\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
