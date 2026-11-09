// Explicit DFlash2 block-8 non-causal masked speculative controller.
if (dflash2 && mtp_depth == 7) {
    const auto round_started = Clock::now();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    const auto round_dispatches_before = gpu.lifetime_dispatches();
    const auto round_commands_before = gpu.lifetime_command_buffers();
    const auto round_barriers_before = gpu.lifetime_barriers();
#endif
    const auto prime_before = mtp_prime_gpu_ms;
    const auto base = stack.position();
    const auto* pending = mtp_anchor.peek(base);
    require(sampler.matches(base) && dflash2->context_len() == base, "DFlash2 frontier differs");
    const bool is_v2 = mtp_request_carry && mtp_generation.is_v2();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    CarrierAction carrier_action(gpu, stack, carrier_mode, is_v2, !carrier_capture_path.empty(), mtp_generation.controller_state(), false, &carrier_packet_tail);
#endif
    auto random = sampler.random();
    const auto& current = retained.get(base);
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
    const uint32_t max_context_steps = base < shape.attention.capacity ? (shape.attention.capacity - base) : 0u;
    const auto measured_action = measured_controller && dflash2_auto_draft && compute_remaining >= 2 && max_context_steps >= 2
        ? measured_controller->before_draft(base, std::min({verify_cost_profile->maximum_rows(), compute_remaining, max_context_steps})).rows : 0u;
    const bool is_ordinary_step = compute_remaining < 2 || max_context_steps < 2 || anchor_stop || measured_action == 1;

    if (is_ordinary_step) {
        advance(anchor);
        sampler.publish(std::move(random));
        if (pending) ++mtp_carried_anchors;
        static_cast<void>(mtp_anchor.clear());
        generated.push_back(Json::integer(anchor));
        const auto& committed = retained.get(stack.position());
        batch_wall += milliseconds(round_started);
        batch_gpu += committed.batch.gpu_ms + mtp_prime_gpu_ms - prime_before;
        { TOKSPUR_CARRIER_PHASE(publication);
        std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(anchor)},
            {"source_logit", number(anchor_score)}, {"committed_prefix", Json::integer(stack.position())},
            {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl; }
        if (is_v2) {
            mtp_generation.commit_round(1, std::span(&anchor, 1), std::span(&anchor_score, 1),
                                       base, stack.position(), 1, anchor_stop, 7);
        }
        if (anchor_stop) {
            stop_id = anchor;
            retained.stopped(anchor);
            if (is_v2) mtp_generation.mark_stop_drained();
            break;
        }
        continue;
    }

    const uint32_t max_k = std::min({7u, dflash2->max_drafts(), compute_remaining - 1u, max_context_steps - 1u});
    dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(anchor) * row_bytes, row_bytes),
        std::span(mtp_candidate_rows).first(2560));
    PleAsyncScopeGuard ple_guard;
    NgramHistory ple_hist{};
    if (ple_async) {
        ple_hist = stack.committed_history();
        ple_guard.start(*ple_async, ple_hist, std::span(&anchor, 1), std::span(mtp_candidate_memory).first(2560));
    }
    const auto head_started = Clock::now();
    measured_observer.drafted_rows = max_k + 1;
    const auto draft = dflash2->draft_block8(max_k, base, anchor, std::span(mtp_candidate_rows).first(2560),
        sampler.options().policy, random, dflash2->adaptive_budget());
    if (measured_controller) measured_observer.probabilities = dflash_measured_probabilities(*dflash2, draft.confidences);
    const double draft_wall = milliseconds(head_started);
    const double draft_gpu = draft.batch.gpu_ms;
    if (ple_async) {
        ple_hist = ple_guard.finish();
    }
    const uint32_t k = draft.count;
    mtp_head_calls += k;
    mtp_proposed += k;
    mtp_head_gpu_ms += draft_gpu;
    mtp_head_wall_ms += draft_wall;

    if (k == 0u) {
        advance(anchor);
        sampler.publish(std::move(random));
        if (pending) ++mtp_carried_anchors;
        static_cast<void>(mtp_anchor.clear());
        generated.push_back(Json::integer(anchor));
        const auto& committed = retained.get(stack.position());
        const double round_wall = milliseconds(round_started);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        if (carrier_mode == CarrierDiagnosticMode::off) {
#endif
        dflash2->observe_verify_round(draft.confidences, 1u, 0u, measured_controller ? 0 : controller_observed_decode_ms(round_wall), true);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        }
#endif
        batch_wall += round_wall;
        batch_gpu += committed.batch.gpu_ms + draft_gpu + mtp_prime_gpu_ms - prime_before;
        { TOKSPUR_CARRIER_PHASE(publication);
        std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(anchor)},
            {"source_logit", number(anchor_score)}, {"committed_prefix", Json::integer(stack.position())},
            {"entropy_words", Json::integer(sampler.random().words_consumed())}, {"gpu_profile", Json()}})).dump() << std::endl; }
        if (is_v2) {
            mtp_generation.commit_round(1, std::span(&anchor, 1), std::span(&anchor_score, 1),
                                       base, stack.position(), 1, false, 7);
        }
        continue;
    }

    std::array<uint32_t, 8> candidates{};
    candidates[0] = anchor;
    for (uint32_t t = 0; t < k; ++t) candidates[1 + t] = draft.proposals[t].selected;
    const auto cand_span = std::span<const uint32_t>(candidates).first(1 + k);
    if (ple_async) {
        ple_guard.start(*ple_async, ple_hist, cand_span.subspan(1),
            std::span(mtp_candidate_memory).subspan(2560, uint64_t(k) * 2560));
        for (uint32_t t = 1; t <= k; ++t) {
            dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
        }
        static_cast<void>(ple_guard.finish());
    } else {
        const auto staged = stack.sampled_verification8_lookups(cand_span);
        std::array<uint64_t, 128> batch_ple_rows{};
        for (uint32_t t = 0; t <= k; ++t) {
            if (t) {
                dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(candidates[t]) * row_bytes, row_bytes),
                    std::span(mtp_candidate_rows).subspan(uint64_t(t) * 2560, 2560));
            }
            std::copy(staged[t].rows.begin(), staged[t].rows.end(), batch_ple_rows.begin() + ptrdiff_t(t * 16));
        }
        cache.gather(std::span<const uint64_t>(batch_ple_rows).first(size_t(1 + k) * 16),
                     std::span(mtp_candidate_memory).first(uint64_t(1 + k) * 2560));
    }

    TargetDFlash2Selection selector{base, anchor, draft, sampler.options().policy, job.stops, {}, {}};
    const auto verified = stack.verify_sampled8(cand_span,
        std::span(mtp_candidate_rows).first(uint64_t(1 + k) * 2560),
        std::span(mtp_candidate_memory).first(uint64_t(1 + k) * 2560),
        base, {TargetDFlash2Selection::choose, &selector, current_allow_topk()});
    require(selector.decision && selector.publication &&
        verified.committed == 1 + selector.decision->accepted &&
        selector.publication->prefix == stack.position(), "DFlash2 target/decision publication differs");
    auto& decision = *selector.decision;
    measured_observer.stopped = decision.stopped;
    materialize_verified_logits_if_needed(verified, verified.committed,
        selector.publication->stopped, selector.publication->pending.has_value());
    mtp_spare->position = base;
    mtp_spare->batch = verified.verification;
    mtp_spare->batch.gpu_ms += verified.recovery.gpu_ms;
    mtp_spare->batch.wall_ms += verified.recovery.wall_ms;
    retained.exchange(mtp_spare, stack.position(), {});
    for (uint32_t t = 1; t <= verified.committed; ++t) {
        require(sampler.committed_input(base + t), "DFlash2 target publication invalidated entropy");
    }

    const auto cap = stack.last_dflash_capture();
    require(cap.rows.owner && cap.count == verified.committed, "DFlash2 verification tap capture mismatch");
    const auto prime_started = Clock::now();
    if (dflash2->deferred_context()) dflash2->stage_context_commit(cap.rows, verified.committed, base);
    else mtp_prime_gpu_ms += dflash2->commit_context(cap.rows, verified.committed, base).gpu_ms;
    mtp_prime_calls += verified.committed;
    mtp_prime_wall_ms += milliseconds(prime_started);

    sampler.publish(std::move(decision.random_after));
    if (pending) ++mtp_carried_anchors;
    mtp_anchor.publish(*selector.publication);
    ++mtp_rounds;
    mtp_accepted += decision.accepted;
    mtp_recoveries += decision.accepted < k;
    const uint32_t count = verified.committed;
    require(stack.position() == base + count && sampler.matches(stack.position()) && dflash2->context_len() == stack.position(),
        "DFlash2 visible/physical/drafter frontier differs");
    const double round_wall = milliseconds(round_started);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (carrier_mode == CarrierDiagnosticMode::off) {
#endif
    dflash2->observe_verify_round(draft.confidences, 1u + k, decision.accepted, measured_controller ? 0 : controller_observed_decode_ms(round_wall), !decision.stopped);
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    }
#endif
    batch_wall += round_wall;
    batch_gpu += verified.verification.gpu_ms + verified.recovery.gpu_ms + draft_gpu + mtp_prime_gpu_ms - prime_before;
    std::cout << decorate(Json(Json::Object{{"event", Json("mtp_round")}, {"base", Json::integer(base)},
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        {"gpu_counters_before", Json(Json::Object{{"dispatches", Json::integer(round_dispatches_before)}, {"command_buffers", Json::integer(round_commands_before)}, {"barriers", Json::integer(round_barriers_before)}})},
#endif
        {"committed_prefix", Json::integer(stack.position())}, {"count", Json::integer(count)},
        {"accepted", Json::integer(decision.accepted)}, {"proposed", Json::integer(k)},
        {"draft_wall_ms", number(draft_wall)}, {"draft_gpu_ms", number(draft_gpu)},
        {"verification_wall_ms", number(verified.verification.wall_ms)},
        {"verification_gpu_ms", number(verified.verification.gpu_ms)},
        {"recovery_wall_ms", number(verified.recovery.wall_ms)},
        {"recovery_gpu_ms", number(verified.recovery.gpu_ms)},
        {"materialization_wall_ms", number(0)}, {"lookahead_anchor", Json(true)},
        {"used_carried_anchor", Json(bool(pending))},
        {"pending_anchor", Json(bool(mtp_anchor.peek(stack.position())))},
        {"round_wall_ms", number(round_wall)},
        {"timings_qualified", Json(false)}})).dump() << std::endl;

    std::array<uint32_t, 8> round_ids{};
    std::array<float, 8> round_scores{};
    for (uint32_t t = 0; t < count; ++t) {
        round_ids[t] = t ? decision.tokens[t - 1].id : anchor;
        round_scores[t] = t ? decision.tokens[t - 1].source_logit : anchor_score;
    }
    const uint32_t remaining_in_packet = job.generate - uint32_t(generated.size());
    const uint32_t emitted_now = is_v2 ? std::min(count, remaining_in_packet) : count;
    if (is_v2) {
        mtp_generation.commit_round(count, std::span(round_ids).first(count),
                                   std::span(round_scores).first(count),
                                   base, stack.position(), emitted_now,
                                   selector.publication->stopped, 7);
    }
    for (uint32_t t = 0; t < emitted_now; ++t) {
        TOKSPUR_CARRIER_PHASE(publication);
        const auto id = round_ids[t];
        const auto score = round_scores[t];
        generated.push_back(Json::integer(id));
        std::cout << decorate(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)},
            {"source_logit", number(score)}, {"committed_prefix", Json::integer(stack.position())},
            {"token_prefix", Json::integer(base + t + 1)},
            {"round_committed_prefix", Json::integer(stack.position())},
            {"entropy_words", Json::integer(sampler.random().words_consumed())},
            {"gpu_profile", Json()}})).dump() << std::endl;
        if (selector.publication->stopped && t == count - 1) {
            stop_id = id;
            retained.stopped(id);
            if (is_v2) mtp_generation.mark_stop_drained();
        }
    }
    i += emitted_now - 1;
    if (stop_id) break;
    continue;
}
