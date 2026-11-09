#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
struct QsaSparseStep5Evidence {
    OperatorStateEvidence state;
    uint64_t sparse_decode_calls = 0, sparse_verify_calls = 0;
    uint64_t legacy_stage_calls = 0, legacy_channel_calls = 0;
    QsaVerifyRowsRoutes row_routes{};
    uint64_t rollback_checks = 0;
};

QsaSparseStep5Evidence qsa_sparse_step5_run(const char* library, bool enabled, bool concurrent, bool diagnostic = false) {
    const StackShape shape{{4, 2560, 320}, {2560, 1, 2, 128, 32, 4}, {2560, 4, 2, 32, 2, 32, 32, 4, 2048, 8192}, {2560, 32, 512, 10}, {4, 2560, 4, 4, 3}, 13, 1};
    const StackPolicy policy{{0.08838835f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true, {HashOrder::newest_first, NgramBoundary(0), true}};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};

    MetalGraph g(library, 4ull << 30);
    g.configure_small_gemv(16);
    g.configure_block_gemv(3);
    g.configure_hc_exact(HcExactMode::combined);
    g.configure_concurrent_dispatch(concurrent);
    g.configure_prefill_dense_f32(true);
    g.configure_prefill_grouped_f32(true);
    g.configure_prefill_dense_staged_f32(true);
    g.configure_prefill_hc_rows(true);
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    g.configure_prefill_router_window(true);
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    g.configure_prefill_hc_down_window(true);
#endif
#endif
    g.configure_q8_three_rows(true);
    g.configure_qsa_stage_v8(true);
    g.configure_qsa_channel_chronological(true);
    g.configure_qsa_sparse(enabled);
    require(g.qsa_sparse_enabled() == enabled && g.qsa_sparse_rollback_enabled(), "QSA sparse Step 5 startup identity");
    reject([&] { g.configure_qsa_sparse(!enabled); });

    OperatorStateSink checkpoints;
    if (diagnostic) g.configure_operator_checkpoints(OperatorStateSink::emit, &checkpoints, 16384);

    const auto initial = g.memory().live_bytes;
    QsaSparseStep5Evidence result;
    {
        Assets a{g, std::mt19937(44), {}, {}, {}};
        auto small = shape;
        small.experts.experts = 2;
        small.experts.selected = 1;
        const std::array<uint32_t, 1> attention_layers{1};
        auto f = fixture(a, small, 2, attention_layers);
        for (auto& layer : f.weights.layers) {
            std::visit([&](auto& w) {
                for (auto* read : {&w.attention_read, &w.ffn_read}) {
                    read->down = stack_q8(a, 10240, 320);
                    read->up = stack_q8(a, 320, 10240);
                }
                auto& m = w.ffn;
                m.router = a.matrix(2560 * 512).first;
                m.gate = stack_raw_routed(a, 12, 2560, 32, 512);
                m.up = stack_raw_routed(a, 12, 2560, 32, 512);
                m.down = stack_raw_routed(a, 7, 32, 2560, 512);
                m.shared_gate = stack_q8(a, 2560, 32);
                m.shared_up = stack_q8(a, 2560, 32);
                m.shared_down = stack_q8(a, 32, 2560);
                if constexpr (std::is_same_v<std::decay_t<decltype(w)>, RecurrentWeights>) {
                    w.attention.qkv = stack_q8(a, 2560, uint32_t(gdn_qkv_count(shape.recurrent)));
                    w.attention.gate = stack_q8(a, 2560, 64);
                    w.attention.output = stack_q8(a, 64, 2560);
                } else {
                    auto& q = w.attention;
                    q.query_gate = stack_q8(a, 2560, 256);
                    q.key = stack_q8(a, 2560, 64);
                    q.value = stack_q8(a, 2560, 64);
                    q.output = stack_q8(a, 128, 2560);
                    q.index_query = stack_bf16(a, q.index_query);
                    q.index_key = stack_bf16(a, q.index_key);
                }
            }, layer);
        }
        auto& recurrent_weight = std::get<RecurrentWeights>(f.weights.layers[0]).attention.qkv;
        const auto mutable_trunk = g.upload(g.read(recurrent_weight.view), "qsa-sparse step5 mutable trunk Q8", false);
        a.views.push_back(mutable_trunk);
        recurrent_weight.view = mutable_trunk;
        const auto original_head_mat = stack_q8(a, 2560, shape.vocabulary);
        const auto mutable_head = g.upload(g.read(original_head_mat.view), "qsa-sparse step5 mutable vocabulary head", false);
        a.views.push_back(mutable_head);
        f.weights.output = {mutable_head, 8};

        const auto head_ok = g.read(mutable_head), trunk_ok = g.read(mutable_trunk);
        auto head_bad = head_ok, trunk_bad = trunk_ok;
        const uint16_t inf = 0x7c00;
        std::memcpy(head_bad.data(), &inf, 2);
        std::memcpy(trunk_bad.data(), &inf, 2);
        const auto base = g.memory().live_bytes;

        {
            TextStack stack(g, f.weights, shape, policy, spec, StackExecution::direct, 512, true);
            stack.prepare_chunked_prefill(GdnChunkMode::fused, 512ull << 20);
            stack.prepare_final_hc_tap(64ull << 20);
            stack.prepare_sampled_verification2(maximum_stack_verification_budget, {true, QsaVerifyRowsMode::core_and_q8});

            constexpr uint32_t count = 4105;
            std::vector<uint32_t> tokens(count);
            std::vector<float> rows(uint64_t(count) * 2560), memory(uint64_t(count) * 4);
            auto history = empty_ngram_history(spec);
            for (uint32_t t = 0; t < count; ++t) {
                tokens[t] = (t * 3) % 13;
                const auto ids = ngram_step(spec, policy.ngram, history, tokens[t]);
                history = ids.next;
                for (uint32_t c = 0; c < 2560; ++c) rows[uint64_t(t) * 2560 + c] = float(int((t * 13 + c * 7) % 131) - 65) / 997.0f;
                for (uint32_t c = 0; c < 4; ++c) memory[uint64_t(t) * 4 + c] = float(int(ids.rows[c] % 17) - 8) / 32.0f;
            }

            auto& evidence = result.state;
            const auto logits = [&](std::span<const float> values) {
                for (auto v : values) require(std::isfinite(v), "QSA sparse step-5 nonfinite full vocabulary");
                const auto bytes = std::as_bytes(values);
                evidence.logits.emplace_back(bytes.begin(), bytes.end());
            };
            const auto retain = [&](std::span<const float> values) {
                logits(values);
                evidence.prefixes.push_back(stack.position());
                evidence.accepted.push_back(stack.accepted_snapshot());
                evidence.physical.push_back(stack.chunked_cache_snapshot());
                const auto hc = std::as_bytes(stack.last_final_hc_rows());
                evidence.hc.emplace_back(hc.begin(), hc.end());
            };
            const auto run = [&](uint32_t at, uint32_t n) {
                return stack.step_known(
                    std::span(tokens).subspan(at, n),
                    std::span(rows).subspan(uint64_t(at) * 2560, uint64_t(n) * 2560),
                    std::span(memory).subspan(uint64_t(at) * 4, uint64_t(n) * 4));
            };

            uint32_t at = 0;
            for (uint32_t n : {512u, 37u, 475u, 512u, 512u, 2u, 1u, 512u, 512u, 512u, 506u}) {
                const auto accepted = stack.accepted_snapshot(), physical = stack.chunked_cache_snapshot();
                const auto dec_before = g.qsa_sparse_decode_calls(), ch_before = g.qsa_channel_calls();
                const bool corrupt_trunk = (n == 1u);
                g.write(corrupt_trunk ? mutable_trunk : mutable_head, corrupt_trunk ? trunk_bad : head_bad);
                reject([&] { static_cast<void>(run(at, n)); });
                g.write(corrupt_trunk ? mutable_trunk : mutable_head, corrupt_trunk ? trunk_ok : head_ok);
                require(stack.position() == at && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical,
                    "QSA sparse step-5 failed chunk lost accepted state or full physical cache");
                if (!diagnostic) {
                    require(g.qsa_sparse_decode_calls() == dec_before && g.qsa_channel_calls() == ch_before,
                        "QSA sparse step-5 failed chunk published decode calls");
                }
                ++result.rollback_checks;
                const auto allocations = g.memory().allocations;
                const auto actual_out = run(at, n);
                at += n;
                require(stack.position() == at && g.memory().allocations == allocations,
                    "QSA sparse step-5 chunk frontier or allocation changed");
                retain(actual_out.logits);
            }
            require(at == 4093, "QSA sparse step-5 generated prefix schedule");

            for (uint32_t consumed : {1u, 2u, 3u}) {
                const auto proposals = std::span(tokens).subspan(at, 3);
                const auto lookups = stack.sampled_verification_depth2_lookups(proposals);
                std::vector<float> input_memory(12);
                for (uint32_t t = 0; t < 3; ++t) {
                    for (uint32_t c = 0; c < 4; ++c) input_memory[t * 4 + c] = float(int(lookups[t].rows[c] % 17) - 8) / 32.0f;
                }
                const auto input_rows = std::span(rows).subspan(uint64_t(at) * 2560, 3 * 2560);
                struct Choice {
                    uint32_t consumed;
                    bool throwing;
                    std::vector<std::vector<float>> rows;
                    static uint32_t select(void* context, const StackSampledVerificationRows& r) {
                        auto& self = *static_cast<Choice*>(context);
                        require(r.count == 3 && r.final_hc_rows.size() == 3 * 10240, "QSA sparse step-5 verification geometry");
                        self.rows.clear();
                        for (uint32_t t = 0; t < 3; ++t) {
                            for (auto v : r.logits[t]) require(std::isfinite(v), "QSA sparse step-5 verification full logits finite");
                            self.rows.emplace_back(r.logits[t].begin(), r.logits[t].end());
                        }
                        if (self.throwing) throw FormatError("QSA sparse step-5 selector refusal");
                        return self.consumed;
                    }
                } choice{consumed, true, {}};

                const auto accepted = stack.accepted_snapshot(), physical = stack.chunked_cache_snapshot();
                const std::vector<float> hc(stack.last_final_hc_rows().begin(), stack.last_final_hc_rows().end());
                const auto verify = [&](const StackVerificationCancellation& cancellation = StackVerificationCancellation{}) {
                    return stack.verify_sampled_depth2(proposals, input_rows, input_memory, at, {Choice::select, &choice}, cancellation);
                };

                // GPU failure rollback during verification.
                const auto ver_before = g.qsa_sparse_verify_calls();
                g.write(mutable_head, head_bad);
                reject([&] { static_cast<void>(verify()); });
                g.write(mutable_head, head_ok);
                if (!diagnostic) require(g.qsa_sparse_verify_calls() == ver_before, "QSA sparse step-5 failed GPU verification published verify calls");
                require(stack.position() == at && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical,
                    "QSA sparse step-5 GPU verification failure lost accepted state or physical cache");
                ++result.rollback_checks;

                // Selector refusal rollback after GPU completion.
                reject([&] { static_cast<void>(verify()); });
                const auto refused_rows = choice.rows;
                require(stack.position() == at && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical &&
                    std::equal(hc.begin(), hc.end(), stack.last_final_hc_rows().begin(), stack.last_final_hc_rows().end()),
                    "QSA sparse step-5 selector refusal lost accepted state, HC or physical cache");
                ++result.rollback_checks;

                choice.throwing = false;
                struct Cancel {
                    uint32_t polls = 0;
                    static bool requested(void* context) { return ++static_cast<Cancel*>(context)->polls == 2; }
                } cancel;
                reject([&] { static_cast<void>(verify({Cancel::requested, &cancel})); });
                require(stack.position() == at && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical,
                    "QSA sparse step-5 cancellation lost accepted state or physical cache");
                ++result.rollback_checks;

                const auto allocations = g.memory().allocations;
                const auto actual_out = verify();
                require(actual_out.committed == consumed && stack.position() == at + consumed && g.memory().allocations == allocations,
                    "QSA sparse step-5 verification frontier or allocations");
                const auto after = stack.chunked_cache_snapshot();
                require(after.size() == physical.size(), "QSA sparse step-5 physical cache extent changed");
                const uint64_t kv_row = uint64_t(shape.attention.kv_heads) * shape.attention.head_width * 2;
                const uint64_t kv_capacity = uint64_t(shape.attention.capacity) * kv_row;
                const uint64_t index_row = uint64_t(shape.attention.index_width) * 2;
                for (size_t byte = 0; byte < physical.size(); ++byte) {
                    const bool accepted_key = byte >= uint64_t(at) * kv_row && byte < uint64_t(at + consumed) * kv_row;
                    const bool accepted_value = byte >= kv_capacity + uint64_t(at) * kv_row && byte < kv_capacity + uint64_t(at + consumed) * kv_row;
                    const bool accepted_index = byte >= 2 * kv_capacity + uint64_t(at / shape.attention.compression) * index_row &&
                        byte < 2 * kv_capacity + uint64_t((at + consumed) / shape.attention.compression) * index_row;
                    require(accepted_key || accepted_value || accepted_index || after[byte] == physical[byte],
                        "QSA sparse step-5 accepted prefix changed rejected or unrelated physical cache bytes");
                }
                require(choice.rows == refused_rows, "QSA sparse step-5 retry changed candidate full verification rows");
                for (const auto& row : choice.rows) logits(row);
                at += consumed;
                retain(actual_out.logits);
            }

            // Ordinary 1-token continuation via step_known(..., 1) with physical rollback check, then direct step().
            const auto accepted = stack.accepted_snapshot(), physical = stack.chunked_cache_snapshot();
            g.write(mutable_head, head_bad);
            reject([&] { static_cast<void>(run(at, 1)); });
            g.write(mutable_head, head_ok);
            require(stack.position() == at && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical,
                "QSA sparse step-5 ordinary failure lost physical rollback");
            ++result.rollback_checks;

            retain(run(at, 1).logits);
            ++at;
            const auto direct = stack.step(tokens[at], std::span(rows).subspan(uint64_t(at) * 2560, 2560), std::span(memory).subspan(uint64_t(at) * 4, 4));
            ++at;
            require(stack.position() == at, "QSA sparse step-5 direct step frontier");
            retain(direct.logits);
        }
        if (diagnostic) {
            require(checkpoints.physical_batches > checkpoints.closed_owners && checkpoints.closed_owners >= 15,
                "QSA sparse step-5 diagnostic stack did not exercise checkpointed owners");
        }
        require(g.memory().live_bytes == base, "QSA sparse step-5 stack fixture leaked resources");
    }
    require(g.memory().live_bytes == initial, "QSA sparse step-5 fixture assets leaked");

    result.sparse_decode_calls = g.qsa_sparse_decode_calls();
    result.sparse_verify_calls = g.qsa_sparse_verify_calls();
    result.legacy_stage_calls = g.qsa_stage_v8_calls();
    result.legacy_channel_calls = g.qsa_channel_calls();
    result.row_routes = g.qsa_verify_rows_routes();

    // Note: selector refusal and cancellation run GPU batches that complete before host refusal/poll, so 3 committed + 6 refused/cancelled = 9 verify GPU completions in non-diagnostic runs!
    if (enabled) {
        require(result.sparse_decode_calls >= 3 && result.sparse_verify_calls >= 9 &&
            result.legacy_stage_calls == 0 && result.legacy_channel_calls == 0 &&
            result.row_routes[0] == result.sparse_verify_calls && result.row_routes[2] == 0 &&
            result.row_routes[3] == 0 && result.row_routes[4] == 0,
            "QSA sparse step-5 candidate did not route exclusively through single-pass decode and verify kernels");
    } else {
        require(result.sparse_decode_calls == 0 && result.sparse_verify_calls == 0 &&
            result.legacy_stage_calls >= 3 && result.legacy_channel_calls == result.legacy_stage_calls &&
            result.row_routes[0] >= 9 && result.row_routes[2] == result.row_routes[0] &&
            result.row_routes[3] == result.row_routes[0] && result.row_routes[4] == result.row_routes[0],
            "QSA sparse step-5 baseline did not route exclusively through three-pass decode and verify kernels");
    }
    require(result.rollback_checks == 21u, "QSA sparse step-5 rollback check count changed");
    return result;
}

void qsa_sparse_step5_controls(const char* library, bool concurrent) {
    const auto exact = qsa_sparse_step5_run(library, false, concurrent, false);
    const auto candidate = qsa_sparse_step5_run(library, true, concurrent, false);
    const auto repeat = qsa_sparse_step5_run(library, true, concurrent, false);
    const auto diag = qsa_sparse_step5_run(library, true, concurrent, true);
    const auto opposite = qsa_sparse_step5_run(library, true, !concurrent, false);

    require(candidate.state == repeat.state &&
        candidate.sparse_decode_calls == repeat.sparse_decode_calls &&
        candidate.sparse_verify_calls == repeat.sparse_verify_calls,
        "QSA sparse step-5 own repeat changed full logits, accepted state, HC or physical cache");
    require(candidate.state == diag.state,
        "QSA sparse step-5 operator checkpoints changed full logits, accepted state, HC or physical cache");
    require(candidate.state == opposite.state,
        "QSA sparse step-5 serial vs concurrent changed full logits, accepted state, HC or physical cache");
    require(exact.state.logits.size() == candidate.state.logits.size() && exact.state.prefixes == candidate.state.prefixes,
        "QSA sparse step-5 observation alignment");

    for (size_t i = 0; i < exact.state.logits.size(); ++i) {
        const auto& a = candidate.state.logits[i];
        const auto& b = exact.state.logits[i];
        std::vector<float> x(a.size() / 4), y(b.size() / 4);
        std::memcpy(x.data(), a.data(), a.size());
        std::memcpy(y.data(), b.data(), b.size());
        stack_compare(x, y);
    }
    const StackShape shape{{4, 2560, 320}, {2560, 1, 2, 128, 32, 4}, {2560, 4, 2, 32, 2, 32, 32, 4, 2048, 8192}, {2560, 32, 512, 10}, {4, 2560, 4, 4, 3}, 13, 1};
    uint32_t changed = 0;
    for (size_t i = 0; i < exact.state.accepted.size(); ++i) {
        hc_down_state_compare(candidate.state.accepted[i], exact.state.accepted[i], shape, exact.state.prefixes[i]);
        changed += candidate.state.accepted[i] != exact.state.accepted[i];
    }

    std::cout << "PASS QSA sparse step-5 generated controls concurrent=" << concurrent
              << " full_logit_rows=" << candidate.state.logits.size()
              << " accepted_snapshots=" << candidate.state.accepted.size()
              << " exact_state_differences=" << changed
              << " own_repeat=byte-exact diagnostic_off_on=byte-exact serial_concurrent=byte-exact"
              << " rollback_checks_per_arm=" << candidate.rollback_checks
              << " sparse_decode_calls=" << candidate.sparse_decode_calls
              << " sparse_verify_calls=" << candidate.sparse_verify_calls
              << " worst_fixed_nmse=" << worst << " generated_nmse_bound=1e-6 no model or timing admission\n";
}

struct QsaRouteControlSink {
    std::array<std::vector<std::byte>, 6> common{};
    uint32_t expected_index = 0, records = 0, pairs = 0, refuse_phase = 0;
    static bool emit(void* context, MetalGraph& g, const QsaRouteAttributionRecord& r) noexcept {
        try {
            auto& s = *static_cast<QsaRouteControlSink*>(context);
            constexpr std::array<uint32_t, 4> sequence{1, 2, 5, 3};
            if (r.phase != sequence[s.expected_index]) return false;
            ++s.records;
            if (r.phase == s.refuse_phase) {
                s.expected_index = 0;
                return false;
            }
            if (r.phase == 1) {
                for (size_t i = 0; i < r.views.size(); ++i) {
                    s.common[i] = r.views[i].owner ? g.read(r.views[i]) : std::vector<std::byte>{};
                }
            } else if (r.phase == 5) {
                for (size_t i = 0; i < r.views.size(); ++i) {
                    const auto cur = r.views[i].owner ? g.read(r.views[i]) : std::vector<std::byte>{};
                    require(s.common[i] == cur, "QSA route shadow changed common input, accepted index sum or restored KV/index cache");
                }
            } else {
                require(r.rows >= 1 && r.rows <= 8 && r.channels == 2560 && r.experts == 512 && r.selected == 10,
                    "QSA route capture geometry mismatch");
                const auto logits = g.read(r.views[1]);
                const auto ids = g.read(r.views[2]);
                const auto weights = g.read(r.views[3]);
                require(logits.size() == uint64_t(r.rows) * 512 * 4 && ids.size() == uint64_t(r.rows) * 10 * 4 && weights.size() == ids.size(),
                    "QSA route capture view byte sizes differ");
                std::vector<float> lv(logits.size() / 4), wv(weights.size() / 4);
                std::vector<uint32_t> iv(ids.size() / 4);
                std::memcpy(lv.data(), logits.data(), logits.size());
                std::memcpy(iv.data(), ids.data(), ids.size());
                std::memcpy(wv.data(), weights.data(), weights.size());
                for (float v : lv) require(std::isfinite(v), "nonfinite QSA route router logit");
                for (uint32_t id : iv) require(id < 512, "out-of-range QSA route expert ID");
                for (float w : wv) require(std::isfinite(w) && w > 0.0f, "invalid QSA route expert weight");
                if (r.phase == 3) ++s.pairs;
            }
            s.expected_index = (s.expected_index + 1) % 4;
            return true;
        } catch (...) {
            return false;
        }
    }
};

OperatorStateEvidence qsa_route_control_run(const char* library, bool enabled, bool concurrent) {
    const StackShape shape{{4, 2560, 320}, {2560, 1, 2, 128, 32, 4}, {2560, 4, 2, 32, 2, 32, 32, 4, 16, 8192}, {2560, 32, 512, 10}, {4, 2560, 4, 4, 3}, 13, 1};
    const StackPolicy policy{{0.08838835f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true, {HashOrder::newest_first, NgramBoundary(0), true}};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};

    MetalGraph g(library, 2ull << 30);
    g.configure_small_gemv(16);
    g.configure_block_gemv(3);
    g.configure_hc_exact(HcExactMode::combined);
    g.configure_concurrent_dispatch(concurrent);
    g.configure_prefill_dense_f32(true);
    g.configure_prefill_grouped_f32(true);
    g.configure_prefill_dense_staged_f32(true);
    g.configure_prefill_hc_rows(true);
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    g.configure_prefill_router_window(true);
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    g.configure_prefill_hc_down_window(true);
#endif
#endif
    g.configure_q8_three_rows(true);
    g.configure_qsa_stage_v8(true);
    g.configure_qsa_channel_chronological(true);
    g.configure_qsa_sparse(false);

    OperatorStateSink checkpoints;
    g.configure_operator_checkpoints(OperatorStateSink::emit, &checkpoints, 16384);
    QsaRouteControlSink sink;
    if (enabled) g.configure_qsa_route_attribution(QsaRouteControlSink::emit, &sink);

    Assets a{g, std::mt19937(44), {}, {}, {}};
    auto small = shape;
    small.experts.experts = 2;
    small.experts.selected = 1;
    const std::array<uint32_t, 1> attention_layers{1};
    auto f = fixture(a, small, 2, attention_layers);
    for (auto& layer : f.weights.layers) {
        std::visit([&](auto& w) {
            for (auto* read : {&w.attention_read, &w.ffn_read}) {
                read->down = stack_q8(a, 10240, 320);
                read->up = stack_q8(a, 320, 10240);
            }
            auto& m = w.ffn;
            m.router = a.matrix(2560 * 512).first;
            m.gate = stack_q8(a, 2560, 32, 512);
            m.up = stack_q8(a, 2560, 32, 512);
            m.down = stack_q8(a, 32, 2560, 512);
            m.shared_gate = stack_q8(a, 2560, 32);
            m.shared_up = stack_q8(a, 2560, 32);
            m.shared_down = stack_q8(a, 32, 2560);
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>, RecurrentWeights>) {
                w.attention.qkv = stack_q8(a, 2560, uint32_t(gdn_qkv_count(shape.recurrent)));
                w.attention.gate = stack_q8(a, 2560, 64);
                w.attention.output = stack_q8(a, 64, 2560);
            } else {
                auto& q = w.attention;
                q.query_gate = stack_q8(a, 2560, 256);
                q.key = stack_q8(a, 2560, 64);
                q.value = stack_q8(a, 2560, 64);
                q.output = stack_q8(a, 128, 2560);
                q.index_query = stack_bf16(a, q.index_query);
                q.index_key = stack_bf16(a, q.index_key);
            }
        }, layer);
    }
    f.weights.output = stack_q8(a, 2560, shape.vocabulary);

    TextStack stack(g, f.weights, shape, policy, spec, StackExecution::direct, 512, true);
    stack.prepare_chunked_prefill(GdnChunkMode::fused, 512ull << 20);
    stack.prepare_final_hc_tap(64ull << 20);
    stack.prepare_sampled_verification2(maximum_stack_verification_budget, {true, QsaVerifyRowsMode::core_and_q8});

    constexpr uint32_t count = 530;
    std::vector<uint32_t> tokens(count);
    std::vector<float> rows(uint64_t(count) * 2560), memory(uint64_t(count) * 4);
    auto history = empty_ngram_history(spec);
    for (uint32_t t = 0; t < count; ++t) {
        tokens[t] = (t * 3) % 13;
        const auto ids = ngram_step(spec, policy.ngram, history, tokens[t]);
        history = ids.next;
        for (uint32_t c = 0; c < 2560; ++c) rows[uint64_t(t) * 2560 + c] = float(int((t * 13 + c * 7) % 131) - 65) / 997.0f;
        for (uint32_t c = 0; c < 4; ++c) memory[uint64_t(t) * 4 + c] = float(int(ids.rows[c] % 17) - 8) / 32.0f;
    }

    OperatorStateEvidence evidence;
    const auto retain = [&](std::span<const float> values) {
        const auto bytes = std::as_bytes(values);
        evidence.logits.emplace_back(bytes.begin(), bytes.end());
        evidence.prefixes.push_back(stack.position());
        evidence.accepted.push_back(stack.accepted_snapshot());
        evidence.physical.push_back(stack.chunked_cache_snapshot());
        const auto hc = std::as_bytes(stack.last_final_hc_rows());
        evidence.hc.emplace_back(hc.begin(), hc.end());
    };

    // Prefill 512 (exceeds token_budget=16 so sparse block selection is active!).
    const auto pre = stack.step_known(std::span(tokens).first(512), std::span(rows).first(512 * 2560), std::span(memory).first(512 * 4));
    retain(pre.logits);

    const auto proposals = std::span(tokens).subspan(512, 3);
    const auto lookups = stack.sampled_verification_depth2_lookups(proposals);
    std::vector<float> vmem(12);
    for (uint32_t t = 0; t < 3; ++t) {
        for (uint32_t c = 0; c < 4; ++c) vmem[t * 4 + c] = float(int(lookups[t].rows[c] % 17) - 8) / 32.0f;
    }
    struct AcceptThree {
        static uint32_t choose(void*, const StackSampledVerificationRows& r) {
            require(r.count == 3, "QSA route verification rows");
            return 3;
        }
    };
    int ctx = 0;

    if (enabled) {
        for (uint32_t bad_phase : {2u, 5u, 3u}) {
            const auto accepted = stack.accepted_snapshot(), physical = stack.chunked_cache_snapshot();
            sink.refuse_phase = bad_phase;
            sink.expected_index = 0;
            reject([&] {
                static_cast<void>(stack.verify_sampled_depth2(
                    proposals, std::span(rows).subspan(512ull * 2560, 3 * 2560), vmem, 512, {AcceptThree::choose, &ctx}));
            });
            sink.refuse_phase = 0;
            require(stack.position() == 512 && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical,
                "QSA route verification sink refusal failed physical rollback");
        }
    }

    const auto allocations_v = g.memory().allocations;
    const auto verified = stack.verify_sampled_depth2(
        proposals, std::span(rows).subspan(512ull * 2560, 3 * 2560), vmem, 512, {AcceptThree::choose, &ctx});
    require(verified.committed == 3 && stack.position() == 515 && g.memory().allocations == allocations_v,
        "QSA route N3 frontier or allocation");
    retain(verified.logits);

    if (enabled) {
        for (uint32_t bad_phase : {2u, 5u, 3u}) {
            const auto accepted = stack.accepted_snapshot(), physical = stack.chunked_cache_snapshot();
            sink.refuse_phase = bad_phase;
            sink.expected_index = 0;
            reject([&] {
                static_cast<void>(stack.step(tokens[515], std::span(rows).subspan(515ull * 2560, 2560), std::span(memory).subspan(515ull * 4, 4)));
            });
            sink.refuse_phase = 0;
            require(stack.position() == 515 && stack.accepted_snapshot() == accepted && stack.chunked_cache_snapshot() == physical,
                "QSA route decode sink refusal failed physical rollback");
        }
    }

    const auto allocations_d = g.memory().allocations;
    const auto last = stack.step(tokens[515], std::span(rows).subspan(515ull * 2560, 2560), std::span(memory).subspan(515ull * 4, 4));
    require(stack.position() == 516 && g.memory().allocations == allocations_d, "QSA route decode frontier or allocation");
    retain(last.logits);

    // Also test step_known(..., 1) crossing a compression block boundary (515 -> 516 crossed, now 516 -> 517).
    const auto next_one = stack.step_known(std::span(tokens).subspan(516, 1), std::span(rows).subspan(516ull * 2560, 2560), std::span(memory).subspan(516ull * 4, 4));
    require(stack.position() == 517, "QSA route step_known(1) frontier");
    retain(next_one.logits);

    const uint64_t expected_channel = 2u;
    const QsaVerifyRowsRoutes expected_rows{1, 3, 1, 1, 1};
    require(g.qsa_sparse_decode_calls() == 0 && g.qsa_sparse_verify_calls() == 0 &&
        g.qsa_channel_calls() == expected_channel && g.qsa_verify_rows_routes() == expected_rows,
        "QSA route shadow polluted outer serving route counters");
    if (enabled) require(sink.pairs == 3 && sink.expected_index == 0, "QSA route control missed common-input pairs");
    return evidence;
}

void qsa_route_attribution_controls(const char* library, bool concurrent) {
    const auto off = qsa_route_control_run(library, false, concurrent);
    const auto on = qsa_route_control_run(library, true, concurrent);
    const auto repeat = qsa_route_control_run(library, true, concurrent);
    require(off == on && on == repeat,
        "QSA route attribution changed exact full logits, accepted state, HC rows or physical cache");
    std::cout << "PASS QSA common-input route attribution controls concurrent=" << concurrent
              << " off_on=byte-exact own_repeat=byte-exact sink_refusal_rollback=phases_2_5_3\n";
}
#endif
