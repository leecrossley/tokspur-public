// Generated small owned assets only; no original model/weights are opened.
#define main inherited_stack_entry_not_run
#include "test_stack.cpp"
#undef main
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
#define main inherited_sampled_entry_not_run
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
#include <CommonCrypto/CommonDigest.h>
#endif

#ifndef TOKSPUR_PRIVATE_VERIFY_EIGHT
#error "sampled stack test requires the private V8 build"
#endif

namespace {
uint32_t nonfinite_output_refusals = 0;
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
uint32_t cost_width3_cases = 0, production_three_refusals = 0, cost_three_refusals = 0;
#endif
bool same_bytes(std::span<const float> a, std::span<const float> b) {
    const auto x = std::as_bytes(a), y = std::as_bytes(b);
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin());
}
struct SampledFrame {
    std::vector<float> logits, hc;
    std::vector<std::byte> state;
};
struct Select {
    TextStack* target;
    std::span<const SampledFrame> expected;
    uint32_t commit, calls = 0;
    bool throwing = false;
    static uint32_t choose(void* raw, const StackSampledVerificationRows& rows) {
        auto& s = *static_cast<Select*>(raw); ++s.calls;
        require(rows.count == s.expected.size() && rows.residual_width == s.expected[0].hc.size() &&
            rows.final_hc_rows.size() == uint64_t(rows.count) * rows.residual_width, "selector HC geometry");
        for (uint32_t i = 0; i < rows.count; ++i) {
            require(same_bytes(rows.logits[i], s.expected[i].logits), "selector target row differs from ordinary");
            require(same_bytes(rows.final_hc_rows.subspan(uint64_t(i) * rows.residual_width, rows.residual_width),
                s.expected[i].hc), "selector full-stream HC row differs from ordinary");
        }
        for (size_t i = rows.count; i < rows.logits.size(); ++i)
            require(rows.logits[i].empty(), "unused target row is not empty");
        reject([&] { static_cast<void>(s.target->position()); });
        reject([&] { s.target->reset(); });
        reject([&] { static_cast<void>(s.target->last_final_hc_rows()); });
        if (s.throwing) throw FormatError("selector deliberately refused");
        return s.commit;
    }
};

void sampled_stack(const char* library, bool reuse, [[maybe_unused]] bool concurrent, bool host_views = false) {
    MetalGraph graph(library, 128ull << 20);
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    graph.configure_q8_three_rows(true);
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    graph.configure_concurrent_dispatch(concurrent);
#endif
    graph.configure_small_gemv(16); graph.configure_block_gemv(3);
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    graph.configure_hc_exact(HcExactMode::combined);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    graph.configure_qsa_stage_v8(true);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    graph.configure_qsa_channel_chronological(true);
#endif
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    constexpr uint32_t sampled_hidden = 32;
#else
    constexpr uint32_t sampled_hidden = 8;
#endif
    const StackShape shape{{4, 32, 3}, {32, 1, 2, 128, 32, 4},
        {32, 4, 2, 16, 2, 4, 4, 4, 8, 64}, {32, sampled_hidden, 4, 2}, {4, 32, 4, 4, 3}, 13, 1};
    const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true,
        {HashOrder::newest_first, NgramBoundary::literal, true}};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    Assets assets{graph, std::mt19937(0x73616d70), {}, {}, {}};
    auto f = fixture(assets, shape);
    std::deque<std::vector<std::byte>> encoded;
    const auto q8 = [&](uint32_t columns, uint32_t rows, bool immutable = true) {
        encoded.emplace_back(uint64_t(columns / 32) * rows * 34); auto& bytes = encoded.back();
        for (uint32_t row = 0; row < rows; ++row) for (uint32_t block = 0; block < columns / 32; ++block) {
            const size_t at = (size_t(row) * (columns / 32) + block) * 34;
            const uint16_t scale = 0x1403; std::memcpy(bytes.data() + at, &scale, 2);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const int8_t value = int8_t(int((row * 17 + block * 11 + lane * 7) % 31) - 15);
                std::memcpy(bytes.data() + at + 2 + lane, &value, 1);
            }
        }
        const auto view = graph.upload(bytes, "generated sampled-stack Q8", immutable); assets.views.push_back(view);
        return GpuMatrix{view, 8};
    };
    const auto bf16 = [&](uint32_t columns, uint32_t rows) {
        const auto values = assets.values(uint64_t(columns) * rows); encoded.emplace_back(values.size() * 2);
        auto& bytes = encoded.back();
        for (size_t i = 0; i < values.size(); ++i) {
            const auto value = finite_bf16(values[i]); std::memcpy(bytes.data() + i * 2, &value, 2);
        }
        const auto view = graph.upload(bytes, "generated sampled-stack BF16"); assets.views.push_back(view);
        return GpuMatrix{view, 30};
    };
    for (auto& layer : f.weights.layers) {
        if (auto* w = std::get_if<RecurrentWeights>(&layer)) {
            w->attention.qkv = q8(32, uint32_t(gdn_qkv_count(shape.recurrent)));
            w->attention.gate = q8(32, 64); w->attention.output = q8(64, 32);
        } else {
            auto& a = std::get<AttentionLayerWeights>(layer).attention;
            a.query_gate = q8(32, 128); a.key = q8(32, 32); a.value = q8(32, 32); a.output = q8(64, 32);
            a.index_query = bf16(32, 8); a.index_key = bf16(32, 4);
        }
    }
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    for(auto& layer:f.weights.layers)std::visit([&](auto& w){w.ffn.shared_gate=q8(32,sampled_hidden);
        w.ffn.shared_up=q8(32,sampled_hidden);w.ffn.shared_down=q8(sampled_hidden,32);},layer);
#endif
    // This generated output matrix is writable only for GPU nonfinite failure injection below.
    f.weights.output = q8(32, shape.vocabulary, false);
    const auto weights_only = graph.memory();
    {
        TextStack target(graph, f.weights, shape, policy, spec, StackExecution::direct, 8, reuse);
        TextStack serial(graph, f.weights, shape, policy, spec);
        require(!target.sampled_logit_views(), "sampled shared views enabled by default");
        target.configure_sampled_logit_views(host_views);
        require(target.sampled_logit_views() == host_views, "sampled shared view selector differs");
        require(!target.final_hc_tap_memory().prepared, "tap prepared by default");
        reject([&] { static_cast<void>(target.last_final_hc_rows()); });
        reject([&] { target.prepare_final_hc_tap(1); });
        reject([&] { target.prepare_sampled_verification2(32ull << 20); });
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        // Separate known-prefill resources remain prepared and enabled.
        // The new sampled preparation alone consents to their composition.
        target.prepare_known_qsa_rows(true);
#endif
        target.prepare_final_hc_tap(1ull << 20); serial.prepare_final_hc_tap(1ull << 20);
        const auto tap = target.final_hc_tap_memory();
        require(tap.prepared && tap.capacity == 8 && tap.residual_width == 128 && tap.host_charged_bytes <= (1ull << 20),
            "bounded full-four-stream host tap");
        reject([&] { target.prepare_final_hc_tap(1ull << 20); });
        reject([&] { target.prepare_checkpoints({1, 16ull << 20}); });
        target.prepare_sampled_verification2(32ull << 20
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            , {reuse, reuse ? QsaVerifyRowsMode::core_and_q8 : QsaVerifyRowsMode::off}
#endif
            );
        reject([&] { target.prepare_sampled_verification2(32ull << 20); });
        reject([&] { target.configure_sampled_logit_views(!host_views); });
        const auto resident = graph.memory();
        const auto ledger = [&] { const auto now = graph.memory(); require(now.live_bytes == resident.live_bytes &&
            now.allocations == resident.allocations, "sampled request changed prepared GPU ledger"); };
        const auto row = [&](uint32_t token) { std::vector<float> out(shape.residual.channels);
            for (size_t i = 0; i < out.size(); ++i) out[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16; return out; };
        const auto memory = [&](std::span<const uint64_t> ids) { std::vector<float> out(shape.memory.embedding);
            for (size_t i = 0; i < out.size(); ++i) out[i] = float(int(ids[i / spec.row_width] % 17) - 8) / 32; return out; };
        const auto advance = [&](TextStack& stack, uint32_t token) { const auto lookup = stack.lookup(token); ++batches;
            const auto out = stack.step(token, row(token), memory(lookup.rows));
            require(stack.last_final_hc_count() == 1 && stack.last_final_hc_residual().size() == 128, "ordinary HC tap geometry");
            auto state = stack.accepted_snapshot();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
            std::array<std::byte, 4096> scratch{};
            const auto actual = stack.accepted_fingerprint(scratch);
            std::array<unsigned char, 32> expected{};
            CC_SHA256(state.data(), CC_LONG(state.size()), expected.data());
            require(actual.bytes == state.size() && actual.sha256 == expected, "streaming target fingerprint differs from snapshot");
#endif
            return SampledFrame{out.logits, {stack.last_final_hc_residual().begin(), stack.last_final_hc_residual().end()}, std::move(state)}; };
        const auto seed = [&](TextStack& stack, uint32_t n) { stack.reset(); SampledFrame out;
            require(!stack.last_final_hc_count() && stack.last_final_hc_rows().empty(), "reset retained HC history");
            reject([&] { static_cast<void>(stack.last_final_hc_residual()); });
            for (uint32_t i = 0; i < n; ++i) out = advance(stack, i % shape.vocabulary); return out; };
        struct Inputs { std::vector<uint32_t> tokens; std::vector<float> rows, memory; };
        const auto inputs = [&](std::span<const uint32_t> ids) {
            Inputs in{{ids.begin(), ids.end()}, {}, {}};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            const auto lookups = target.sampled_verification8_lookups(ids);
#elif defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
            const auto lookups = target.sampled_verification_cost_lookups(ids);
#else
            const auto lookups = target.sampled_verification_lookups(ids);
#endif
            for (size_t i = 0; i < ids.size(); ++i) {
                const auto r = row(ids[i]), m = memory(lookups[i].rows);
                in.rows.insert(in.rows.end(), r.begin(), r.end()); in.memory.insert(in.memory.end(), m.begin(), m.end());
            }
            return in;
        };
        for (uint32_t prefix : {1, 3, 4, 7}) {
            const auto base = seed(serial, prefix);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            constexpr uint32_t limit = 8;
#elif defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
            constexpr uint32_t limit = 3;
#else
            constexpr uint32_t limit = 2;
#endif
            std::array<uint32_t, limit> ids{}; std::array<SampledFrame, limit> frames;
            auto logits = base.logits;
            for (uint32_t i = 0; i < limit; ++i) {
                ids[i] = uint32_t((std::max_element(logits.begin(), logits.end()) - logits.begin() + 1) % shape.vocabulary);
                frames[i] = advance(serial, ids[i]); logits = frames[i].logits;
            }
            for (uint32_t n = 1; n <= limit; ++n) for (uint32_t k = 1; k <= n; ++k) {
                const auto target_base = seed(target, prefix); require(target_base.state == base.state, "sampled base differs");
                const auto in = inputs(std::span(ids).first(n)); Select selection{&target, std::span(frames).first(n), k};
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
                const auto out =
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                    n > 3 ? target.verify_sampled8(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection}) :
#endif
                    n == 3 ?
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                    target.verify_sampled_depth2(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection}) :
#else
                    target.verify_sampled_cost3(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection}) :
#endif
                    target.verify_sampled2(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection});
                cost_width3_cases += n == 3;
#else
                const auto out = target.verify_sampled2(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection});
#endif
                batches += 1 + bool(out.recovery.dispatches);
                require(selection.calls == 1 && out.position == prefix && out.committed == k && out.evaluated.count == n &&
                    target.position() == prefix + k && same_bytes(out.logits, frames[k-1].logits) &&
                    target.accepted_snapshot() == frames[k-1].state, "arbitrary sampled prefix/state differs from ordinary");
                require(target.last_final_hc_count() == k && same_bytes(target.last_final_hc_residual(), frames[k-1].hc) &&
                    same_bytes(out.evaluated.final_hc_rows.last(128), frames[n-1].hc), "committed/evaluated HC confusion");
                static_cast<void>(seed(serial, prefix));
                for (uint32_t i = 0; i < k; ++i) static_cast<void>(advance(serial, ids[i]));
                const auto a = advance(target, 6), b = advance(serial, 6);
                require(a.state == b.state && same_bytes(a.logits,b.logits) && same_bytes(a.hc,b.hc), "rejected tail contaminated append");
                ledger();
            }
        }
        // Known-input prefill must retain every row, including the first row a
        // head needs when priming from a prior target HC residual.
        static_cast<void>(seed(target, 1)); static_cast<void>(seed(serial, 1));
        const std::array<uint32_t, 8> known{2, 3, 4, 5, 6, 7, 8, 9};
        const auto lookups = target.lookups(known); std::vector<float> kr, km, expected_hc; SampledFrame last;
        for (size_t i = 0; i < known.size(); ++i) {
            const auto r = row(known[i]), m = memory(lookups[i].rows);
            kr.insert(kr.end(),r.begin(),r.end()); km.insert(km.end(),m.begin(),m.end());
            last = advance(serial,known[i]); expected_hc.insert(expected_hc.end(),last.hc.begin(),last.hc.end());
        }
        const auto known_out = target.step_known(known,kr,km); ++batches;
        require(target.last_final_hc_count()==8 && same_bytes(target.last_final_hc_rows(),expected_hc) &&
            same_bytes(known_out.logits,last.logits) && target.accepted_snapshot()==last.state,"known prefill lost/reordered HC rows");
        for (uint32_t mode = 0; mode < 7; ++mode) {
            const auto base = seed(target,3); static_cast<void>(seed(serial,3));
            const std::array<uint32_t,2> ids{5,8}; std::array<SampledFrame,2> frames{advance(serial,5),advance(serial,8)};
            const auto in=inputs(ids); Select selection{&target,frames,mode==0?0u:mode==1?3u:1u}; selection.throwing=mode==2;
            struct Cancel { uint32_t at,polls=0; static bool poll(void* p) { auto& c=*static_cast<Cancel*>(p); return ++c.polls==c.at; } };
            Cancel cancel{mode>=3?mode-2:99};
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens,in.rows,in.memory,3,{Select::choose,&selection},{Cancel::poll,&cancel})); });
            require(target.position()==3 && target.accepted_snapshot()==base.state &&
                target.last_final_hc_count()==1 && same_bytes(target.last_final_hc_residual(),base.hc),"refused sampled transaction published state/tap");
            ledger();
        }
        {
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            const std::array<uint32_t, 2> ids{5, 8};
            std::array<SampledFrame, 2> frames{advance(serial, 5), advance(serial, 8)};
            const auto in = inputs(ids); Select selection{&target, frames, 1};
            const auto after_selection = [](void* value) { return static_cast<Select*>(value)->calls != 0; };
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection}, {after_selection, &selection})); });
            require(selection.calls == 1 && target.position() == 3 && target.accepted_snapshot() == base.state && same_bytes(target.last_final_hc_residual(), base.hc), "post-selection cancellation published state or HC");
            selection.calls = 0;
            const auto retry = target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection});
            require(retry.committed == 1 && same_bytes(retry.logits, frames[0].logits) && target.accepted_snapshot() == frames[0].state, "cancelled shared view corrupted retry");
            ledger();
        }
        for (const uint16_t nonfinite_scale : {uint16_t(0x7e00), uint16_t(0x7c00)}) {
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            const std::array<uint32_t, 2> ids{5, 8};
            std::array<SampledFrame, 2> frames{advance(serial, 5), advance(serial, 8)};
            const auto in = inputs(ids); Select selection{&target, frames, 1};
            const auto saved_output = graph.read(f.weights.output.view);
            auto poisoned_output = saved_output;
            std::memcpy(poisoned_output.data(), &nonfinite_scale, sizeof(nonfinite_scale));
            graph.write(f.weights.output.view, poisoned_output);
            // Inputs remain finite. The evaluated output projection must reject its NaN/Inf before the selector or state publication.
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection})); });
            require(selection.calls == 0 && target.position() == 3 && target.accepted_snapshot() == base.state && same_bytes(target.last_final_hc_residual(), base.hc), "nonfinite evaluated output published state, HC or selector input");
            ++nonfinite_output_refusals;
            graph.write(f.weights.output.view, saved_output);
            const auto retry = target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection});
            require(selection.calls == 1 && retry.committed == 1 && same_bytes(retry.logits, frames[0].logits) && target.accepted_snapshot() == frames[0].state && same_bytes(target.last_final_hc_residual(), frames[0].hc), "nonfinite output refusal corrupted valid retry");
            ledger();
        }
        {
            target.reset();
            target.configure_gpu_final_hc_tap(true);
            require(target.gpu_final_hc_tap(), "GPU final HC tap toggle failed");
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            (void)base;
            const std::array<uint32_t, 2> ids{5, 8};
            std::array<SampledFrame, 2> frames{advance(serial, 5), advance(serial, 8)};
            const auto in = inputs(ids);
            struct GpuTapSelect {
                bool saw_gpu_only = false;
                static uint32_t choose(void* ctx, const StackSampledVerificationRows& rows) {
                    auto& self = *static_cast<GpuTapSelect*>(ctx);
                    self.saw_gpu_only = rows.count == 2 && rows.final_hc_rows.empty() &&
                        rows.final_hc_gpu_rows.owner != 0 && rows.final_hc_gpu_rows.bytes == 2 * 128 * sizeof(float);
                    return 2;
                }
            };
            GpuTapSelect gpu_sel{false};
            const auto res = target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {GpuTapSelect::choose, &gpu_sel});
            require(gpu_sel.saw_gpu_only && res.evaluated.final_hc_rows.empty() && res.evaluated.final_hc_gpu_rows.owner != 0 &&
                target.has_last_final_hc_gpu(), "GPU final HC tap did not suppress eager host copy");
            require(same_bytes(target.last_final_hc_residual(), frames[1].hc) &&
                same_bytes(target.last_final_hc_rows().first(frames[0].hc.size()), frames[0].hc),
                "GPU final HC tap lazy host materialisation differs");
            target.reset();
            target.configure_gpu_final_hc_tap(false);
            ledger();
        }
        reject([&] { static_cast<void>(target.sampled_verification_lookups({})); });
        reject([&] { static_cast<void>(target.sampled_verification_lookups(std::array<uint32_t,3>{1,2,3})); });
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        for (uint32_t width : {1u,2u}) {
            const std::array<uint32_t,2> ids{1,2};const auto in=inputs(std::span(ids).first(width));Select selector{&target,{},1};
            reject([&]{static_cast<void>(target.verify_sampled_depth2(in.tokens,in.rows,in.memory,target.position(),{Select::choose,&selector}));});
            require(!selector.calls,"wrong-width depth-two entry reached selector");
        }
#endif
        reject([&] { static_cast<void>(target.sampled_verification_cost_lookups({})); });
        reject([&] { static_cast<void>(target.sampled_verification_cost_lookups(std::array<uint32_t,4>{1,2,3,4})); });
        { const auto in = inputs(std::array<uint32_t,3>{1,2,3}); Select selection{&target, {}, 1};
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens,in.rows,in.memory,target.position(),{Select::choose,&selection})); });
            require(selection.calls == 0, "production width-three refusal reached selector"); ++production_three_refusals; }
        for (uint32_t mode : {0,1}) {
            const auto base = seed(target,3); static_cast<void>(seed(serial,3));
            const std::array<uint32_t,3> ids{5,8,4};
            std::array<SampledFrame,3> frames{advance(serial,5),advance(serial,8),advance(serial,4)};
            const auto in = inputs(ids); Select selection{&target,frames,4}; selection.throwing = mode == 1;
            reject([&] { static_cast<void>(
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                target.verify_sampled_depth2(in.tokens,in.rows,in.memory,3,{Select::choose,&selection})
#else
                target.verify_sampled_cost3(in.tokens,in.rows,in.memory,3,{Select::choose,&selection})
#endif
                ); });
            ++cost_three_refusals;
            require(target.position()==3 && target.accepted_snapshot()==base.state &&
                same_bytes(target.last_final_hc_residual(),base.hc),"refused COST width-three transaction published state"); ledger();
        }
#endif
        ledger();
    }
    require(graph.memory().live_bytes==weights_only.live_bytes,"sampled stack leaked prepared GPU resources");
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    const auto n3_routes=graph.q8_three_rows_routes();require(n3_routes[0]>0,"sampled state test never dispatched genuine N3");
#endif
}
} // namespace

int main(int argc,char** argv) {
    try {
        bool concurrent = false;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (argc == 3) { require(std::string_view(argv[2]) == "--concurrent", "unknown sampled stack scheduling argument"); concurrent = true; }
        else
#endif
        if(argc!=2) throw std::runtime_error("usage: test-sampled-stack <metallib> [--concurrent]");
        for (const bool host_views : {false, true}) {
            sampled_stack(argv[1],false,concurrent,host_views);
            sampled_stack(argv[1],true,concurrent,host_views);
        }
        require(nonfinite_output_refusals == 8, "copy/shared-view NaN and infinity coverage differs");
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
        require(cost_width3_cases == 48 && production_three_refusals == 4 && cost_three_refusals == 8,
            "sampled COST width-three control coverage differs");
        std::cout << "sampled copy/shared-view width3 cases=48 production-three-refusals=4 cost-three-refusals=8\n";
#endif
        std::cout<<"sampled stack generated checks="<<checks<<" batches="<<batches<<" nonfinite-output-refusals="<<nonfinite_output_refusals<<"; no model/serving qualification\n";
    } catch(const std::exception& e) { std::cerr<<"sampled stack: "<<e.what()<<'\n'; return 1; }
    return 0;
}
