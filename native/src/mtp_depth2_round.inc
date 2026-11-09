// Separate explicit depth-two/depth-three controller. The depth-one controller is unchanged.
if (mtp && (mtp_depth == 2 || mtp_depth == 3)) {
    const auto round_started = Clock::now(); const auto prime_before = mtp_prime_gpu_ms;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    const Json round_counters_before(Json::Object{{"dispatches",Json::integer(gpu.lifetime_dispatches())},
        {"command_buffers",Json::integer(gpu.lifetime_command_buffers())},{"barriers",Json::integer(gpu.lifetime_barriers())}});
#endif
    const auto base = stack.position(); const auto* pending = mtp_anchor.peek(base);
    require(sampler.matches(base) && mtp->next_absolute_position() == base, "depth-two MTP frontier differs");
    const bool is_v2 = mtp_request_carry && mtp_generation.is_v2();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    CarrierAction carrier_action(gpu, stack, carrier_mode, is_v2, !carrier_capture_path.empty(), mtp_generation.controller_state(), false, &carrier_packet_tail);
#endif
    auto random = sampler.random(); const auto& current = retained.get(base);
    const auto anchor = pending ? pending->id : sample(sampling_distribution(current.logits, sampler.options().policy), random);
    const auto anchor_score = pending ? pending->source_logit : current.logits[anchor];
    const bool anchor_stop = std::find(job.stops.begin(), job.stops.end(), anchor) != job.stops.end();
    MeasuredRoundObserver measured_observer(
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        carrier_mode == CarrierDiagnosticMode::off ? measured_controller.get() : nullptr,
#else
        measured_controller.get(),
#endif
        &mtp_generation.controller_state(), base, round_started, mtp_proposed, mtp_accepted,
        measured_packet_tail, generated, stop_id, job.generate);
    const uint32_t compute_remaining = is_v2 ? mtp_generation.compute_remaining() : (job.generate - i);
    const auto context_length = base + 1;
    const bool use_verify_nax_costs = gpu.verify_nax();
    auto& controller = mtp_generation.controller_state();

    const auto measured_before = measured_controller ? measured_controller->before_draft(base, std::min(mtp_depth + 1, compute_remaining)) : MeasuredChoice{};

    if (measured_controller) measured_observer.probabilities = measured_controller->acceptance();

    if (!measured_controller && is_v2 && controller.current_mode == uint8_t(StageAMode::Speculative)) {
        const double r_ord = 1.0 / double(StageACostTable::lookup_t_ord_us(context_length, use_verify_nax_costs));
        const double spec_tp = StageACostTable::max_speculative_throughput(controller.ewma_alpha1, use_verify_nax_costs);
        if (spec_tp < (1.0 + StageACostTable::eta_exit) * r_ord) {
            controller.current_mode = uint8_t(StageAMode::OrdinaryK0);
            controller.consecutive_ordinary_steps = 0;
        }
    }

    const bool is_probe_round = !measured_controller && (is_v2 && controller.current_mode == uint8_t(StageAMode::OrdinaryK0) &&
                                 controller.consecutive_ordinary_steps >= StageACostTable::probe_interval - 1 &&
                                 compute_remaining >= 2);
    const bool is_ordinary_step = measured_controller ? (compute_remaining < 2 || anchor_stop || measured_before.rows == 1) : (is_v2 ? (is_probe_round ? compute_remaining < 2 : compute_remaining < 3) : job.generate - i < mtp_depth + 1) ||
        anchor_stop || (is_v2 && controller.current_mode == uint8_t(StageAMode::OrdinaryK0) && !is_probe_round);

    // This declared tail/ordinary policy uses only ordinary target rows. It is not an
    // implicit depth-one fallback and does not assert seeded cross-depth identity.
    if (is_ordinary_step) {
        advance(anchor); sampler.publish(std::move(random));
        if (pending) ++mtp_carried_anchors;
        static_cast<void>(mtp_anchor.clear()); generated.push_back(Json::integer(anchor));
        const auto& committed = retained.get(stack.position());
        batch_wall += milliseconds(round_started); batch_gpu += committed.batch.gpu_ms + mtp_prime_gpu_ms - prime_before;
        { TOKSPUR_CARRIER_PHASE(publication);
        std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(anchor)},
            {"source_logit", number(anchor_score)}, {"committed_prefix", Json::integer(stack.position())},
            {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl; }
        if (is_v2) {
            mtp_generation.commit_round(1, std::span(&anchor, 1), std::span(&anchor_score, 1),
                                       base, stack.position(), 1, anchor_stop);
            if (!measured_controller && controller.current_mode == uint8_t(StageAMode::OrdinaryK0) && !anchor_stop) {
                stage_a_step_ordinary(controller, context_length, use_verify_nax_costs);
            }
        }
        if (anchor_stop) {
            stop_id = anchor;
            retained.stopped(anchor);
            if (is_v2) mtp_generation.mark_stop_drained();
            break;
        }
        continue;
    }
    uint32_t lookup_bypass_depth = 0;
    std::array<uint32_t, 3> lookup_proposed{};
    if (mtp_lookup && mtp_lookup_valid) {
        require(mtp_lookup->committed_length() == base, "depth-two MTP lookup prefix differs from target frontier");
        mtp_lookup->append(mtp_lookup->epoch(), base, std::span(&anchor, 1));
        if (!is_probe_round && mtp_lookup_mode != MtpLookupMode::off) {
            const auto prop = mtp_lookup->propose(mtp_lookup->epoch(), base + 1);
            if (prop.outcome == LookupOutcome::proposed && prop.count >= 1) {
                lookup_proposed[0] = prop.tokens[0];
                if (prop.count >= 2) lookup_proposed[1] = prop.tokens[1];
                if (prop.count >= 3) lookup_proposed[2] = prop.tokens[2];
                const bool p0_stop = std::find(job.stops.begin(), job.stops.end(), lookup_proposed[0]) != job.stops.end();
                const bool p1_stop = prop.count >= 2 && std::find(job.stops.begin(), job.stops.end(), lookup_proposed[1]) != job.stops.end();
                if (mtp_depth == 3 && compute_remaining >= 4 && prop.count >= 3 && !p0_stop && !p1_stop) {
                    lookup_bypass_depth = 3;
                } else if (compute_remaining >= 3 && prop.count >= 2 && !p0_stop) {
                    lookup_bypass_depth = 2;
                } else if (mtp_lookup_mode == MtpLookupMode::hybrid && prop.count >= 1) {
                    lookup_bypass_depth = 1;
                }
            }
        }
    }

    uint32_t count = 0;
    std::optional<MtpDraftDraw> second;
    std::optional<MtpDraftDraw> third;
    double draft_gpu = 0, draft_wall = 0;
    std::array<uint32_t, 4> round_ids{};
    std::array<float, 4> round_scores{};
    std::optional<TargetMtpSelection2> selector_storage;

    if (lookup_bypass_depth > 0) {
        flush_mtp_deferred();
        std::copy(stack.last_final_hc_residual().begin(), stack.last_final_hc_residual().end(), mtp_lookup_prime_hc.begin());
        const auto lookup_draft_started = Clock::now();
        const auto first = draw_deterministic_proposal(lookup_proposed[0], random, shape.vocabulary);
        if (lookup_bypass_depth >= 2) {
            second.emplace(draw_deterministic_proposal(lookup_proposed[1], first.random_after, shape.vocabulary));
        }
        if (lookup_bypass_depth == 3) {
            third.emplace(draw_deterministic_proposal(lookup_proposed[2], second->random_after, shape.vocabulary));
        }
        draft_wall = milliseconds(lookup_draft_started);
        draft_gpu = 0.0;
        StackSampledVerificationResult verified{};
        const bool allow_topk = current_allow_topk();
        if (lookup_bypass_depth == 3) {
            const std::array<uint32_t, 4> candidates{anchor, first.token, second->token, third->token};
            if (ple_async) {
                PleAsyncScopeGuard ple_guard;
                ple_guard.start(*ple_async, stack.committed_history(), candidates, std::span(mtp_candidate_memory).first(4 * 2560));
                for (uint32_t t = 0; t < 4; ++t) {
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                }
                static_cast<void>(ple_guard.finish());
            } else {
                const auto staged = stack.sampled_verification_depth3_lookups(candidates);
                std::array<uint64_t, 64> batch_ple_rows{};
                for (uint32_t t = 0; t < 4; ++t) {
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                    std::copy(staged[t].rows.begin(), staged[t].rows.end(), batch_ple_rows.begin() + ptrdiff_t(t * 16));
                }
                cache.gather(batch_ple_rows, std::span(mtp_candidate_memory).first(4 * 2560));
            }
            selector_storage.emplace(TargetMtpSelection2{base, anchor, first, &*second, &*third, sampler.options().policy, job.stops, {}, {}});
            verified = stack.verify_sampled_depth3(candidates,
                std::span(mtp_candidate_rows).first(4 * 2560),
                std::span(mtp_candidate_memory).first(4 * 2560),
                base, {TargetMtpSelection2::choose, &*selector_storage, allow_topk});
        } else if (lookup_bypass_depth == 2) {
            const std::array<uint32_t, 3> candidates{anchor, first.token, second->token};
            if (ple_async) {
                PleAsyncScopeGuard ple_guard;
                ple_guard.start(*ple_async, stack.committed_history(), candidates, std::span(mtp_candidate_memory).first(3 * 2560));
                for (uint32_t t = 0; t < 3; ++t) {
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                }
                static_cast<void>(ple_guard.finish());
            } else {
                const auto staged = stack.sampled_verification_depth2_lookups(candidates);
                std::array<uint64_t, 48> batch_ple_rows{};
                for (uint32_t t = 0; t < 3; ++t) {
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                    std::copy(staged[t].rows.begin(), staged[t].rows.end(), batch_ple_rows.begin() + ptrdiff_t(t * 16));
                }
                cache.gather(batch_ple_rows, std::span(mtp_candidate_memory).first(3 * 2560));
            }
            selector_storage.emplace(TargetMtpSelection2{base, anchor, first, &*second, nullptr, sampler.options().policy, job.stops, {}, {}});
            verified = stack.verify_sampled_depth2(candidates,
                std::span(mtp_candidate_rows).first(3 * 2560),
                std::span(mtp_candidate_memory).first(3 * 2560),
                base, {TargetMtpSelection2::choose, &*selector_storage, allow_topk});
        } else {
            const std::array<uint32_t, 2> candidates{anchor, first.token};
            if (ple_async) {
                PleAsyncScopeGuard ple_guard;
                ple_guard.start(*ple_async, stack.committed_history(), candidates, std::span(mtp_candidate_memory).first(2 * 2560));
                for (uint32_t t = 0; t < 2; ++t) {
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                }
                static_cast<void>(ple_guard.finish());
            } else {
                const auto staged = stack.sampled_verification_lookups(std::span(candidates));
                std::array<uint64_t, 32> batch_ple_rows{};
                for (uint32_t t = 0; t < 2; ++t) {
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                    std::copy(staged[t].rows.begin(), staged[t].rows.end(), batch_ple_rows.begin() + ptrdiff_t(t * 16));
                }
                cache.gather(batch_ple_rows, std::span(mtp_candidate_memory).first(2 * 2560));
            }
            selector_storage.emplace(TargetMtpSelection2{base, anchor, first, nullptr, nullptr, sampler.options().policy, job.stops, {}, {}});
            verified = stack.verify_sampled2(std::span(candidates),
                std::span(mtp_candidate_rows).first(2 * 2560),
                std::span(mtp_candidate_memory).first(2 * 2560),
                base, {TargetMtpSelection2::choose, &*selector_storage, allow_topk});
        }
        require(selector_storage->decision && selector_storage->publication && verified.committed == 1 + selector_storage->decision->accepted &&
            selector_storage->publication->prefix == stack.position(), "lookup-bypass target/decision publication differs");
        auto& decision = *selector_storage->decision;
        measured_observer.stopped = decision.stopped;
        materialize_verified_logits_if_needed(verified, verified.committed,
            selector_storage->publication->stopped, selector_storage->publication->pending.has_value());
        mtp_spare->position = base; mtp_spare->batch = verified.verification;
        mtp_spare->batch.gpu_ms += verified.recovery.gpu_ms; mtp_spare->batch.wall_ms += verified.recovery.wall_ms;
        retained.exchange(mtp_spare, stack.position(), {});
        for (uint32_t t = 1; t <= verified.committed; ++t)
            require(sampler.committed_input(base + t), "lookup-bypass target publication invalidated entropy");
        if (mtp_lookup && mtp_lookup_valid && decision.accepted > 0) {
            mtp_lookup->append(mtp_lookup->epoch(), base + 1, std::span(lookup_proposed).first(decision.accepted));
        }
        const uint32_t prime_rows = 1 + decision.accepted;
        if (decision.accepted > 0) {
            if (!verified.evaluated.final_hc_rows.empty()) {
                std::copy_n(verified.evaluated.final_hc_rows.begin(), uint64_t(decision.accepted) * 10240,
                    mtp_lookup_prime_hc.begin() + 10240);
            } else {
                const auto* src = gpu.host_floats(verified.evaluated.final_hc_gpu_rows);
                std::copy_n(src, uint64_t(decision.accepted) * 10240, mtp_lookup_prime_hc.begin() + 10240);
            }
        }
        const auto prime_hc = std::span<const float>(mtp_lookup_prime_hc).first(uint64_t(prime_rows) * 10240);
        const auto prime_emb = std::span<const float>(mtp_candidate_rows).first(uint64_t(prime_rows) * 2560);
        if (mtp_fast_round) {
            const auto prime_started = Clock::now();
            mtp->defer_prime_batch_host(prime_hc, prime_emb, base);
            mtp_prime_calls += prime_rows;
            mtp_prime_wall_ms += milliseconds(prime_started);
        } else
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        if (mtp->priming_batch_capacity() >= prime_rows) {
            const auto prime_started = Clock::now();
            const auto batch = mtp->prime_batch(prime_hc, prime_emb, base);
            mtp_prime_calls += prime_rows;
            mtp_prime_gpu_ms += batch.gpu_ms;
            mtp_prime_wall_ms += milliseconds(prime_started);
        } else
#endif
        {
            for (uint32_t t = 0; t < prime_rows; ++t)
                mtp_prime(prime_hc.subspan(uint64_t(t) * 10240, 10240), prime_emb.subspan(uint64_t(t) * 2560, 2560), base + t);
        }
        sampler.publish(std::move(decision.random_after));
        if (pending) ++mtp_carried_anchors;
        mtp_anchor.publish(*selector_storage->publication);
        ++mtp_rounds; ++mtp_lookup_rounds;
        mtp_proposed += lookup_bypass_depth;
        mtp_accepted += decision.accepted;
        mtp_lookup_accepted += decision.accepted;
        mtp_recoveries += decision.accepted < lookup_bypass_depth;
        count = verified.committed;
        require(stack.position() == base + count && sampler.matches(stack.position()) && mtp->next_absolute_position() == stack.position(),
            "lookup-bypass visible/physical/head frontier differs");
        batch_wall += milliseconds(round_started);
        batch_gpu += verified.verification.gpu_ms + verified.recovery.gpu_ms + mtp_prime_gpu_ms - prime_before;
        std::cout << decorate(Json(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                {"gpu_counters_before",round_counters_before},
#endif
            {"committed_prefix", Json::integer(stack.position())}, {"count", Json::integer(count)}, {"accepted", Json::integer(decision.accepted)},
            {"proposed", Json::integer(lookup_bypass_depth)}, {"head_calls", Json::integer(0)}, {"prime_extra", Json::integer(1)},
            {"draft_wall_ms", number(draft_wall)}, {"draft_gpu_ms", number(draft_gpu)},
            {"verification_wall_ms", number(verified.verification.wall_ms)}, {"verification_gpu_ms", number(verified.verification.gpu_ms)},
            {"recovery_wall_ms", number(verified.recovery.wall_ms)}, {"recovery_gpu_ms", number(verified.recovery.gpu_ms)},
            {"materialization_wall_ms", number(0)}, {"lookahead_anchor", Json(true)}, {"used_carried_anchor", Json(bool(pending))},
            {"pending_anchor", Json(bool(mtp_anchor.peek(stack.position())))}, {"round_wall_ms", number(milliseconds(round_started))},
            {"timings_qualified", Json(false)}})).dump() << std::endl;
        for (uint32_t t = 0; t < count; ++t) {
            round_ids[t] = t ? decision.tokens[t - 1].id : anchor;
            round_scores[t] = t ? decision.tokens[t - 1].source_logit : anchor_score;
        }
    } else {
        const auto cache_before = mtp->cache_rows();
        const bool allow_topk = current_allow_topk();
        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(anchor) * row_bytes, row_bytes),
            std::span(mtp_candidate_rows).first(2560));
        PleAsyncScopeGuard ple_guard;
        NgramHistory ple_hist{};
        if (ple_async) {
            ple_hist = stack.committed_history();
            ple_guard.start(*ple_async, ple_hist, std::span(&anchor, 1), std::span(mtp_candidate_memory).first(2560));
        }
        const auto head_started = Clock::now();
        double head1_gpu = 0, head2_gpu = 0, head3_gpu = 0;
        std::optional<MtpDraftDraw> first_opt;
        std::optional<MtpHeadFastChainResult> greedy_chain;
        std::optional<MtpHeadSampledChainResult> sampled_chain;
        const auto& draft_policy = sampler.options().policy;
        const auto draw_sampled_chain_law = [&](const GpuMtpSampledLawRow& law_row, const SamplingRng& rng_before) -> MtpDraftDraw {
            require(law_row.count >= 1 && law_row.count <= sampled_path_maximum_support,
                "MTP sampled chain returned invalid proposal support size");
            std::array<uint64_t, 20> grid_units{};
            for (uint32_t j = 0; j < law_row.count; ++j) {
                grid_units[j] = uint64_t(law_row.units24[j]) << 29u;
            }
            auto law = SampledSparseLaw::from_grid_units(
                shape.vocabulary,
                std::span(law_row.ids.data(), law_row.count),
                std::span(grid_units.data(), law_row.count));
            auto rng = rng_before;
            const auto begin = rng.words_consumed();
            const auto ticket = law.deterministic() ? uint64_t(0) : uint64_t(std::ldexp(rng.uniform(), 53));
            require(uint32_t(!law.deterministic()) == law_row.used_ticket,
                "MTP sampled chain determinism mismatch");
            const auto token = law.sample_at(ticket);
            require(token == law_row.chosen_token,
                "MTP sampled chain GPU/CPU token mismatch");
            return {std::move(law), token, begin, std::move(rng)};
        };
        const auto measured_after = measured_controller ? measured_controller->after_draft(base,
            std::min(mtp_depth + 1, compute_remaining), measured_controller->acceptance(), 2) : MeasuredChoice{};
        const bool want_d2_pre = measured_controller ? measured_after.rows >= 3 : !is_probe_round && compute_remaining >= 3 &&
            (StageACostTable::expected_speculative_throughput(controller.ewma_alpha1, 2, use_verify_nax_costs) >
             StageACostTable::expected_speculative_throughput(controller.ewma_alpha1, 1, use_verify_nax_costs));
        const bool want_d3_pre = measured_controller ? measured_after.rows >= 4 : want_d2_pre && mtp_depth == 3 && compute_remaining >= 4 &&
            (StageACostTable::expected_speculative_throughput(controller.ewma_alpha1, 3, use_verify_nax_costs) >
             StageACostTable::expected_speculative_throughput(controller.ewma_alpha1, 2, use_verify_nax_costs));
        const uint32_t greedy_chain_depth = (mtp_fast_round && mtp->can_greedy_chain() && want_d2_pre &&
            token_table.type == 8 && (draft_policy.temperature == 0.0 || draft_policy.top_k == 1))
            ? (want_d3_pre ? 3u : 2u) : 0u;
        const float sampled_inv_temp = (draft_policy.temperature > 0.0 && std::isfinite(draft_policy.temperature))
            ? float(1.0 / draft_policy.temperature) : 0.0f;
        const uint32_t sampled_chain_depth = (!greedy_chain_depth && mtp_fast_round && mtp->can_greedy_chain() &&
            want_d2_pre && token_table.type == 8 &&
            std::isfinite(sampled_inv_temp) && sampled_inv_temp > 0.0f &&
            draft_policy.top_k != 1 && draft_policy.top_k <= maximum_sampling_vocabulary &&
            std::isfinite(draft_policy.top_p) && draft_policy.top_p > 0.0 && draft_policy.top_p <= 1.0 &&
            std::isfinite(draft_policy.min_p) && draft_policy.min_p >= 0.0 && draft_policy.min_p <= 1.0)
            ? (want_d3_pre ? 3u : 2u) : 0u;
        if (greedy_chain_depth >= 2) {
            const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
            greedy_chain.emplace(hc_gpu.owner
                ? mtp->advance_fast_greedy_chain(hc_gpu, std::span(mtp_candidate_rows).first(2560), base, greedy_chain_depth)
                : mtp->advance_fast_greedy_chain_host(stack.last_final_hc_residual(), std::span(mtp_candidate_rows).first(2560), base, greedy_chain_depth));
            head1_gpu = greedy_chain->batch.gpu_ms;
            first_opt.emplace(draw_mtp_depth1_topk(greedy_chain->topks[0], draft_policy, random, shape.vocabulary));
        } else if (sampled_chain_depth >= 2) {
            auto ticket_rng = random;
            const std::array<uint64_t, 3> pre_tickets{
                uint64_t(std::ldexp(ticket_rng.uniform(), 53)),
                uint64_t(std::ldexp(ticket_rng.uniform(), 53)),
                uint64_t(std::ldexp(ticket_rng.uniform(), 53)),
            };
            const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
            sampled_chain.emplace(hc_gpu.owner
                ? mtp->advance_fast_sampled_chain(hc_gpu, std::span(mtp_candidate_rows).first(2560), base, sampled_chain_depth, draft_policy, pre_tickets)
                : mtp->advance_fast_sampled_chain_host(stack.last_final_hc_residual(), std::span(mtp_candidate_rows).first(2560), base, sampled_chain_depth, draft_policy, pre_tickets));
            head1_gpu = sampled_chain->batch.gpu_ms;
            first_opt.emplace(draw_sampled_chain_law(sampled_chain->laws[0], random));
        } else if (mtp_fast_round) {
            const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
            const auto head1 = hc_gpu.owner
                ? mtp->advance_fast(hc_gpu, std::span(mtp_candidate_rows).first(2560), base)
                : mtp->advance_fast_host(stack.last_final_hc_residual(), std::span(mtp_candidate_rows).first(2560), base);
            head1_gpu = head1.batch.gpu_ms;
            first_opt.emplace(draw_mtp_depth1_topk(head1.topk, draft_policy, random, shape.vocabulary));
        } else {
            const auto residual_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_residual() : GpuView{};
            const auto head1 = residual_gpu.owner
                ? mtp->advance(residual_gpu, std::span(mtp_candidate_rows).first(2560), base)
                : mtp->advance(stack.last_final_hc_residual(), std::span(mtp_candidate_rows).first(2560), base);
            head1_gpu = head1.batch.gpu_ms;
            first_opt.emplace(draw_mtp_depth1(head1.logits, draft_policy, random, shape.vocabulary));
        }
        const auto& first = *first_opt;
        if (ple_async) ple_hist = ple_guard.finish();

        const bool first_stop = std::find(job.stops.begin(), job.stops.end(), first.token) != job.stops.end();
        const bool allow_depth2 = want_d2_pre && !first_stop;
        const bool fast_chained = bool(greedy_chain) || bool(sampled_chain);
        if (fast_chained && !allow_depth2) {
            mtp->truncate(cache_before + 1);
        }

        if (allow_depth2) {
            if (greedy_chain) {
                second = draw_mtp_depth1_topk(greedy_chain->topks[1], draft_policy, first.random_after, shape.vocabulary);
            } else if (sampled_chain) {
                second = draw_sampled_chain_law(sampled_chain->laws[1], first.random_after);
            } else {
                if (ple_async) {
                    ple_guard.start(*ple_async, ple_hist, std::span(&first.token, 1),
                        std::span(mtp_candidate_memory).subspan(2560, 2560));
                }
                dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(first.token) * row_bytes, row_bytes),
                    std::span(mtp_candidate_rows).subspan(2560, 2560));
                if (mtp_fast_round) {
                    const auto head2 = mtp->advance_fast_chain(std::span(mtp_candidate_rows).subspan(2560, 2560), base + 1);
                    head2_gpu = head2.batch.gpu_ms;
                    second = draw_mtp_depth1_topk(head2.topk, draft_policy, first.random_after, shape.vocabulary);
                } else {
                    const auto head2 = mtp->advance(mtp->hidden_gpu(), std::span(mtp_candidate_rows).subspan(2560, 2560), base + 1);
                    head2_gpu = head2.batch.gpu_ms;
                    second = draw_mtp_depth1(head2.logits, draft_policy, first.random_after, shape.vocabulary);
                }
                if (ple_async) ple_hist = ple_guard.finish();
            }
            const bool second_stop = std::find(job.stops.begin(), job.stops.end(), second->token) != job.stops.end();
            const bool allow_depth3 = want_d3_pre && !second_stop;
            if (allow_depth3) {
                if (greedy_chain) {
                    third = draw_mtp_depth1_topk(greedy_chain->topks[2], draft_policy, second->random_after, shape.vocabulary);
                } else if (sampled_chain) {
                    third = draw_sampled_chain_law(sampled_chain->laws[2], second->random_after);
                } else {
                    if (ple_async) {
                        ple_guard.start(*ple_async, ple_hist, std::span(&second->token, 1),
                            std::span(mtp_candidate_memory).subspan(5120, 2560));
                    }
                    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(second->token) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(5120, 2560));
                    if (mtp_fast_round) {
                        const auto head3 = mtp->advance_fast_chain(std::span(mtp_candidate_rows).subspan(5120, 2560), base + 2);
                        head3_gpu = head3.batch.gpu_ms;
                        third = draw_mtp_depth1_topk(head3.topk, draft_policy, second->random_after, shape.vocabulary);
                    } else {
                        const auto head3 = mtp->advance(mtp->hidden_gpu(), std::span(mtp_candidate_rows).subspan(5120, 2560), base + 2);
                        head3_gpu = head3.batch.gpu_ms;
                        third = draw_mtp_depth1(head3.logits, draft_policy, second->random_after, shape.vocabulary);
                    }
                    if (ple_async) ple_hist = ple_guard.finish();
                }
                draft_wall = milliseconds(head_started);
                draft_gpu = head1_gpu + head2_gpu + head3_gpu;
                mtp_head_calls += 3; mtp_proposed += 3; mtp_head_gpu_ms += draft_gpu; mtp_head_wall_ms += draft_wall;
                mtp->truncate(cache_before + 1);
                const std::array<uint32_t, 4> candidates{anchor, first.token, second->token, third->token};
                if (ple_async) {
                    if (fast_chained) {
                        ple_guard.start(*ple_async, ple_hist, std::span(candidates).subspan(1, 3),
                            std::span(mtp_candidate_memory).subspan(2560, 3 * 2560));
                        for (uint32_t t = 1; t < 4; ++t) {
                            dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                                std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                        }
                        static_cast<void>(ple_guard.finish());
                    } else {
                        ple_guard.start(*ple_async, ple_hist, std::span(&third->token, 1),
                            std::span(mtp_candidate_memory).subspan(7680, 2560));
                        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(third->token) * row_bytes, row_bytes),
                            std::span(mtp_candidate_rows).subspan(7680, 2560));
                        static_cast<void>(ple_guard.finish());
                    }
                } else {
                    const auto staged = stack.sampled_verification_depth3_lookups(candidates);
                    std::array<uint64_t, 64> batch_ple_rows{};
                    for (uint32_t t = 0; t < 4; ++t) {
                        if (t == 3 || (fast_chained && t >= 1)) dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                            std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                        std::copy(staged[t].rows.begin(), staged[t].rows.end(), batch_ple_rows.begin() + ptrdiff_t(t * 16));
                    }
                    cache.gather(batch_ple_rows, std::span(mtp_candidate_memory).first(4 * 2560));
                }
                selector_storage.emplace(TargetMtpSelection2{base, anchor, first, &*second, &*third, sampler.options().policy, job.stops, {}, {}});
                const auto verified = stack.verify_sampled_depth3(candidates,
                    std::span(mtp_candidate_rows).first(4 * 2560),
                    std::span(mtp_candidate_memory).first(4 * 2560),
                    base, {TargetMtpSelection2::choose, &*selector_storage, allow_topk});
                require(selector_storage->decision && selector_storage->publication && verified.committed == 1 + selector_storage->decision->accepted &&
                    selector_storage->publication->prefix == stack.position(), "depth-three target/decision publication differs");
                auto& decision = *selector_storage->decision;
                measured_observer.stopped = decision.stopped;
                materialize_verified_logits_if_needed(verified, verified.committed,
                    selector_storage->publication->stopped, selector_storage->publication->pending.has_value());
                mtp_spare->position = base; mtp_spare->batch = verified.verification;
                mtp_spare->batch.gpu_ms += verified.recovery.gpu_ms; mtp_spare->batch.wall_ms += verified.recovery.wall_ms;
                retained.exchange(mtp_spare, stack.position(), {});
                for (uint32_t t = 1; t <= verified.committed; ++t)
                    require(sampler.committed_input(base + t), "depth-three target publication invalidated entropy");
                if (mtp_lookup && mtp_lookup_valid && decision.accepted > 0) {
                    mtp_lookup->append(mtp_lookup->epoch(), base + 1, std::span(candidates).subspan(1, decision.accepted));
                }
                if (mtp_fast_round) {
                    if (decision.accepted > 0) {
                        const auto prime_started = Clock::now();
                        const auto prime_hc_gpu = gpu.slice(verified.evaluated.final_hc_gpu_rows, 0, uint64_t(decision.accepted) * 10240 * sizeof(float));
                        const auto prime_emb = std::span(mtp_candidate_rows).subspan(2560, uint64_t(decision.accepted) * 2560);
                        mtp->defer_prime_batch(prime_hc_gpu, prime_emb, base + 1);
                        mtp_prime_calls += decision.accepted;
                        mtp_prime_wall_ms += milliseconds(prime_started);
                    }
                } else
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
                if (decision.accepted > 0 && mtp->priming_batch_capacity() >= decision.accepted) {
                    const auto prime_emb = std::span(mtp_candidate_rows).subspan(2560, uint64_t(decision.accepted) * 2560);
                    const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_rows() : GpuView{};
                    const auto prime_started = Clock::now();
                    const auto batch = hc_gpu.owner
                        ? mtp->prime_batch(gpu.slice(hc_gpu, 0, uint64_t(decision.accepted) * 10240 * sizeof(float)), prime_emb, base + 1)
                        : mtp->prime_batch(verified.evaluated.final_hc_rows.subspan(0, uint64_t(decision.accepted) * 10240), prime_emb, base + 1);
                    mtp_prime_calls += decision.accepted;
                    mtp_prime_gpu_ms += batch.gpu_ms;
                    mtp_prime_wall_ms += milliseconds(prime_started);
                } else
#endif
                {
                    for (uint32_t t = 0; t < decision.accepted; ++t)
                        mtp_prime(verified.evaluated.final_hc_rows.subspan(uint64_t(t) * 10240, 10240),
                            std::span(mtp_candidate_rows).subspan(uint64_t(t + 1) * 2560, 2560), base + t + 1);
                }
                sampler.publish(std::move(decision.random_after));
                if (pending) ++mtp_carried_anchors;
                mtp_anchor.publish(*selector_storage->publication);
                ++mtp_rounds; mtp_accepted += decision.accepted; mtp_recoveries += decision.accepted < 3;
                count = verified.committed;
                require(stack.position() == base + count && sampler.matches(stack.position()) && mtp->next_absolute_position() == stack.position(),
                    "depth-three visible/physical/head frontier differs");
                batch_wall += milliseconds(round_started);
                batch_gpu += verified.verification.gpu_ms + verified.recovery.gpu_ms + draft_gpu + mtp_prime_gpu_ms - prime_before;
                std::cout << decorate(Json(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                    {"gpu_counters_before",round_counters_before},
#endif
                    {"committed_prefix", Json::integer(stack.position())}, {"count", Json::integer(count)}, {"accepted", Json::integer(decision.accepted)},
                    {"proposed", Json::integer(3)}, {"draft_wall_ms", number(draft_wall)}, {"draft_gpu_ms", number(draft_gpu)},
                    {"verification_wall_ms", number(verified.verification.wall_ms)}, {"verification_gpu_ms", number(verified.verification.gpu_ms)},
                    {"recovery_wall_ms", number(verified.recovery.wall_ms)}, {"recovery_gpu_ms", number(verified.recovery.gpu_ms)},
                    {"materialization_wall_ms", number(0)}, {"lookahead_anchor", Json(true)}, {"used_carried_anchor", Json(bool(pending))},
                    {"pending_anchor", Json(bool(mtp_anchor.peek(stack.position())))}, {"round_wall_ms", number(milliseconds(round_started))},
                    {"timings_qualified", Json(false)}})).dump() << std::endl;
                for (uint32_t t = 0; t < count; ++t) {
                    round_ids[t] = t ? decision.tokens[t - 1].id : anchor;
                    round_scores[t] = t ? decision.tokens[t - 1].source_logit : anchor_score;
                }
                if (is_v2 && !measured_controller) {
                    const double y1 = double(decision.accepted > 0 ? 1 : 0);
                    stage_a_commit_speculative(controller, y1, context_length, use_verify_nax_costs);
                }
            } else {
                draft_wall = milliseconds(head_started); draft_gpu = head1_gpu + head2_gpu;
                mtp_head_calls += 2; mtp_proposed += 2; mtp_head_gpu_ms += draft_gpu; mtp_head_wall_ms += draft_wall;
                mtp->truncate(cache_before + 1);
                const std::array<uint32_t, 3> candidates{anchor, first.token, second->token};
                if (ple_async) {
                    if (fast_chained) {
                        ple_guard.start(*ple_async, ple_hist, std::span(candidates).subspan(1, 2),
                            std::span(mtp_candidate_memory).subspan(2560, 2 * 2560));
                        for (uint32_t t = 1; t < 3; ++t) {
                            dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                                std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                        }
                        static_cast<void>(ple_guard.finish());
                    } else {
                        ple_guard.start(*ple_async, ple_hist, std::span(&second->token, 1),
                            std::span(mtp_candidate_memory).subspan(5120, 2560));
                        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(second->token) * row_bytes, row_bytes),
                            std::span(mtp_candidate_rows).subspan(5120, 2560));
                        static_cast<void>(ple_guard.finish());
                    }
                } else {
                    const auto staged = stack.sampled_verification_depth2_lookups(candidates);
                    for (uint32_t t = 0; t < 3; ++t) {
                        if (t == 2 || (fast_chained && t >= 1)) dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                            std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                        cache.gather(staged[t].rows, std::span(mtp_candidate_memory).subspan(uint64_t(t) * 2560, 2560));
                    }
                }
                selector_storage.emplace(TargetMtpSelection2{base, anchor, first, &*second, nullptr, sampler.options().policy, job.stops, {}, {}});
                const auto verified = stack.verify_sampled_depth2(candidates,
                    std::span(mtp_candidate_rows).first(3 * 2560),
                    std::span(mtp_candidate_memory).first(3 * 2560),
                    base, {TargetMtpSelection2::choose, &*selector_storage, allow_topk});
                require(selector_storage->decision && selector_storage->publication && verified.committed == 1 + selector_storage->decision->accepted &&
                    selector_storage->publication->prefix == stack.position(), "depth-two target/decision publication differs");
                auto& decision = *selector_storage->decision;
                measured_observer.stopped = decision.stopped;
                materialize_verified_logits_if_needed(verified, verified.committed,
                    selector_storage->publication->stopped, selector_storage->publication->pending.has_value());
                mtp_spare->position = base; mtp_spare->batch = verified.verification;
                mtp_spare->batch.gpu_ms += verified.recovery.gpu_ms; mtp_spare->batch.wall_ms += verified.recovery.wall_ms;
                retained.exchange(mtp_spare, stack.position(), {});
                for (uint32_t t = 1; t <= verified.committed; ++t)
                    require(sampler.committed_input(base + t), "depth-two target publication invalidated entropy");
                if (mtp_lookup && mtp_lookup_valid && decision.accepted > 0) {
                    mtp_lookup->append(mtp_lookup->epoch(), base + 1, std::span(candidates).subspan(1, decision.accepted));
                }
                if (mtp_fast_round) {
                    if (decision.accepted > 0) {
                        const auto prime_started = Clock::now();
                        const auto prime_hc_gpu = gpu.slice(verified.evaluated.final_hc_gpu_rows, 0, uint64_t(decision.accepted) * 10240 * sizeof(float));
                        const auto prime_emb = std::span(mtp_candidate_rows).subspan(2560, uint64_t(decision.accepted) * 2560);
                        mtp->defer_prime_batch(prime_hc_gpu, prime_emb, base + 1);
                        mtp_prime_calls += decision.accepted;
                        mtp_prime_wall_ms += milliseconds(prime_started);
                    }
                } else
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
                if (decision.accepted > 0 && mtp->priming_batch_capacity() >= decision.accepted) {
                    const auto prime_emb = std::span(mtp_candidate_rows).subspan(2560, uint64_t(decision.accepted) * 2560);
                    const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_rows() : GpuView{};
                    const auto prime_started = Clock::now();
                    const auto batch = hc_gpu.owner
                        ? mtp->prime_batch(gpu.slice(hc_gpu, 0, uint64_t(decision.accepted) * 10240 * sizeof(float)), prime_emb, base + 1)
                        : mtp->prime_batch(verified.evaluated.final_hc_rows.subspan(0, uint64_t(decision.accepted) * 10240), prime_emb, base + 1);
                    mtp_prime_calls += decision.accepted;
                    mtp_prime_gpu_ms += batch.gpu_ms;
                    mtp_prime_wall_ms += milliseconds(prime_started);
                } else
#endif
                {
                    for (uint32_t t = 0; t < decision.accepted; ++t)
                        mtp_prime(verified.evaluated.final_hc_rows.subspan(uint64_t(t) * 10240, 10240),
                            std::span(mtp_candidate_rows).subspan(uint64_t(t + 1) * 2560, 2560), base + t + 1);
                }
                sampler.publish(std::move(decision.random_after));
                if (pending) ++mtp_carried_anchors;
                mtp_anchor.publish(*selector_storage->publication);
                ++mtp_rounds; mtp_accepted += decision.accepted; mtp_recoveries += decision.accepted < 2;
                count = verified.committed;
                require(stack.position() == base + count && sampler.matches(stack.position()) && mtp->next_absolute_position() == stack.position(),
                    "depth-two visible/physical/head frontier differs");
                batch_wall += milliseconds(round_started);
                batch_gpu += verified.verification.gpu_ms + verified.recovery.gpu_ms + draft_gpu + mtp_prime_gpu_ms - prime_before;
                std::cout << decorate(Json(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                    {"gpu_counters_before",round_counters_before},
#endif
                    {"committed_prefix", Json::integer(stack.position())}, {"count", Json::integer(count)}, {"accepted", Json::integer(decision.accepted)},
                    {"proposed", Json::integer(2)}, {"draft_wall_ms", number(draft_wall)}, {"draft_gpu_ms", number(draft_gpu)},
                    {"verification_wall_ms", number(verified.verification.wall_ms)}, {"verification_gpu_ms", number(verified.verification.gpu_ms)},
                    {"recovery_wall_ms", number(verified.recovery.wall_ms)}, {"recovery_gpu_ms", number(verified.recovery.gpu_ms)},
                    {"materialization_wall_ms", number(0)}, {"lookahead_anchor", Json(true)}, {"used_carried_anchor", Json(bool(pending))},
                    {"pending_anchor", Json(bool(mtp_anchor.peek(stack.position())))}, {"round_wall_ms", number(milliseconds(round_started))},
                    {"timings_qualified", Json(false)}})).dump() << std::endl;
                for (uint32_t t = 0; t < count; ++t) {
                    round_ids[t] = t ? decision.tokens[t - 1].id : anchor;
                    round_scores[t] = t ? decision.tokens[t - 1].source_logit : anchor_score;
                }
                if (is_v2 && !measured_controller) {
                    const double y1 = double(decision.accepted > 0 ? 1 : 0);
                    stage_a_commit_speculative(controller, y1, context_length, use_verify_nax_costs);
                }
            }
        } else {
            draft_wall = milliseconds(head_started); draft_gpu = head1_gpu;
            mtp_head_calls += 1; mtp_proposed += 1; mtp_head_gpu_ms += draft_gpu; mtp_head_wall_ms += draft_wall;
            const std::array<uint32_t, 2> candidates{anchor, first.token};
            if (ple_async) {
                ple_guard.start(*ple_async, ple_hist, std::span(&first.token, 1),
                    std::span(mtp_candidate_memory).subspan(2560, 2560));
                dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[1]) * row_bytes, row_bytes),
                    std::span(mtp_candidate_rows).subspan(2560, 2560));
                static_cast<void>(ple_guard.finish());
            } else {
                const auto staged = stack.sampled_verification_lookups(std::span(candidates));
                std::array<uint64_t, 32> batch_ple_rows{};
                for (uint32_t t = 0; t < 2; ++t) {
                    if (t) dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
                    std::copy(staged[t].rows.begin(), staged[t].rows.end(), batch_ple_rows.begin() + ptrdiff_t(t * 16));
                }
                cache.gather(batch_ple_rows, std::span(mtp_candidate_memory).first(2 * 2560));
            }
            selector_storage.emplace(TargetMtpSelection2{base, anchor, first, nullptr, nullptr, sampler.options().policy, job.stops, {}, {}});
            const auto verified = stack.verify_sampled2(std::span(candidates),
                std::span(mtp_candidate_rows).first(2 * 2560),
                std::span(mtp_candidate_memory).first(2 * 2560),
                base, {TargetMtpSelection2::choose, &*selector_storage, allow_topk});
            require(selector_storage->decision && selector_storage->publication && verified.committed == 1 + selector_storage->decision->accepted &&
                selector_storage->publication->prefix == stack.position(), "depth-one target/decision publication differs");
            auto& decision = *selector_storage->decision;
            measured_observer.stopped = decision.stopped;
            materialize_verified_logits_if_needed(verified, verified.committed,
                selector_storage->publication->stopped, selector_storage->publication->pending.has_value());
            mtp_spare->position = base; mtp_spare->batch = verified.verification;
            mtp_spare->batch.gpu_ms += verified.recovery.gpu_ms; mtp_spare->batch.wall_ms += verified.recovery.wall_ms;
            retained.exchange(mtp_spare, stack.position(), {});
            for (uint32_t t = 1; t <= verified.committed; ++t)
                require(sampler.committed_input(base + t), "depth-one target publication invalidated entropy");
            if (mtp_lookup && mtp_lookup_valid && decision.accepted > 0) {
                mtp_lookup->append(mtp_lookup->epoch(), base + 1, std::span(candidates).subspan(1, decision.accepted));
            }
            if (mtp_fast_round) {
                if (decision.accepted > 0) {
                    const auto prime_started = Clock::now();
                    const auto prime_hc_gpu = gpu.slice(verified.evaluated.final_hc_gpu_rows, 0, uint64_t(decision.accepted) * 10240 * sizeof(float));
                    const auto prime_emb = std::span(mtp_candidate_rows).subspan(2560, uint64_t(decision.accepted) * 2560);
                    mtp->defer_prime_batch(prime_hc_gpu, prime_emb, base + 1);
                    mtp_prime_calls += decision.accepted;
                    mtp_prime_wall_ms += milliseconds(prime_started);
                }
            } else
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
            if (decision.accepted > 0 && mtp->priming_batch_capacity() >= decision.accepted) {
                const auto prime_emb = std::span(mtp_candidate_rows).subspan(2560, uint64_t(decision.accepted) * 2560);
                const auto hc_gpu = stack.has_last_final_hc_gpu() ? stack.last_final_hc_gpu_rows() : GpuView{};
                const auto prime_started = Clock::now();
                const auto batch = hc_gpu.owner
                    ? mtp->prime_batch(gpu.slice(hc_gpu, 0, uint64_t(decision.accepted) * 10240 * sizeof(float)), prime_emb, base + 1)
                    : mtp->prime_batch(verified.evaluated.final_hc_rows.subspan(0, uint64_t(decision.accepted) * 10240), prime_emb, base + 1);
                mtp_prime_calls += decision.accepted;
                mtp_prime_gpu_ms += batch.gpu_ms;
                mtp_prime_wall_ms += milliseconds(prime_started);
            } else
#endif
            {
                for (uint32_t t = 0; t < decision.accepted; ++t)
                    mtp_prime(verified.evaluated.final_hc_rows.subspan(uint64_t(t) * 10240, 10240),
                        std::span(mtp_candidate_rows).subspan(uint64_t(t + 1) * 2560, 2560), base + t + 1);
            }
            sampler.publish(std::move(decision.random_after));
            if (pending) ++mtp_carried_anchors;
            mtp_anchor.publish(*selector_storage->publication);
            ++mtp_rounds; mtp_accepted += decision.accepted; mtp_recoveries += decision.accepted < 1;
            count = verified.committed;
            require(stack.position() == base + count && sampler.matches(stack.position()) && mtp->next_absolute_position() == stack.position(),
                "depth-one visible/physical/head frontier differs");
            batch_wall += milliseconds(round_started);
            batch_gpu += verified.verification.gpu_ms + verified.recovery.gpu_ms + draft_gpu + mtp_prime_gpu_ms - prime_before;
            std::cout << decorate(Json(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
                {"gpu_counters_before",round_counters_before},
#endif
                {"committed_prefix", Json::integer(stack.position())}, {"count", Json::integer(count)}, {"accepted", Json::integer(decision.accepted)},
                {"proposed", Json::integer(1)}, {"draft_wall_ms", number(draft_wall)}, {"draft_gpu_ms", number(draft_gpu)},
                {"verification_wall_ms", number(verified.verification.wall_ms)}, {"verification_gpu_ms", number(verified.verification.gpu_ms)},
                {"recovery_wall_ms", number(verified.recovery.wall_ms)}, {"recovery_gpu_ms", number(verified.recovery.gpu_ms)},
                {"materialization_wall_ms", number(0)}, {"lookahead_anchor", Json(true)}, {"used_carried_anchor", Json(bool(pending))},
                {"pending_anchor", Json(bool(mtp_anchor.peek(stack.position())))}, {"round_wall_ms", number(milliseconds(round_started))},
                {"timings_qualified", Json(false)}})).dump() << std::endl;
            for (uint32_t t = 0; t < count; ++t) {
                round_ids[t] = t ? decision.tokens[t - 1].id : anchor;
                round_scores[t] = t ? decision.tokens[t - 1].source_logit : anchor_score;
            }
            if (is_v2 && !measured_controller) {
                const double y1 = double(decision.accepted > 0 ? 1 : 0);
                if (is_probe_round) {
                    stage_a_commit_probe(controller, y1, context_length, use_verify_nax_costs);
                } else {
                    stage_a_commit_speculative(controller, y1, context_length, use_verify_nax_costs);
                }
            }
        }
    }
    const uint32_t remaining_in_packet = job.generate - uint32_t(generated.size());
    const uint32_t emitted_now = is_v2 ? std::min(uint32_t(count), remaining_in_packet) : uint32_t(count);
    if (is_v2) {
        mtp_generation.commit_round(count, std::span(round_ids).first(count),
                                   std::span(round_scores).first(count),
                                   base, stack.position(), emitted_now,
                                   selector_storage->publication->stopped,
                                   mtp_depth);
    }
    // An accepted stop is published only when this packet emits it. A stop left in round overflow stays queued until the next packet's overflow drain publishes it, as in the DFlash2 round.
    for (uint32_t t = 0; t < emitted_now; ++t) {
        TOKSPUR_CARRIER_PHASE(publication);
        const auto id = round_ids[t];
        const auto score = round_scores[t];
        generated.push_back(Json::integer(id));
        std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)}, {"source_logit", number(score)},
            {"committed_prefix", Json::integer(stack.position())}, {"token_prefix", Json::integer(base + t + 1)},
            {"round_committed_prefix", Json::integer(stack.position())}, {"entropy_words", Json::integer(sampler.random().words_consumed())},
            {"gpu_profile", Json()}})).dump() << std::endl;
        if (selector_storage->publication->stopped && t == count - 1) {
            stop_id = id;
            retained.stopped(id);
            if (is_v2) mtp_generation.mark_stop_drained();
        }
    }
    i += emitted_now - 1;
    if (stop_id) break;
    continue;
}
