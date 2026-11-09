// Included inside target_cli: reuse the one loaded target, real asset gather and
// existing transaction APIs. No serving request fields or kernels are added.
const auto run_verification_cost = [&](const verification_cost::Prefix& frozen) {
    const auto base = uint32_t(frozen.input.size());
    require(stack.position() == base && retained.matches(base), "verification cost base is not prepared");
    const auto equal = [](std::span<const float> a, std::span<const float> b) {
        return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size_bytes());
    };
    std::vector<float> rows(8 * 2560), memories(rows.size());
    const auto gather = [&](std::span<const uint32_t> tokens, bool verification) {
        if (verification) {
            const auto staged = stack.verification_lookups8(tokens, uint32_t(tokens.size()), {});
            require(staged.count == tokens.size(), "verification cost input was trimmed");
            for (size_t i = 0; i < tokens.size(); ++i)
                cache.gather(staged.rows[i].rows, std::span(memories).subspan(i * 2560, 2560));
        } else {
            const auto staged = stack.lookups(tokens);
            for (size_t i = 0; i < tokens.size(); ++i)
                cache.gather(staged[i].rows, std::span(memories).subspan(i * 2560, 2560));
        }
        for (size_t i = 0; i < tokens.size(); ++i)
            dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(tokens[i]) * row_bytes, row_bytes),
                std::span(rows).subspan(i * 2560, 2560));
    };
    const auto ordinary = [&](uint32_t token) {
        const auto staged = stack.lookup(token); cache.gather(staged.rows, std::span(memories).first(2560));
        dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(token) * row_bytes, row_bytes), std::span(rows).first(2560));
        return stack.step(token, std::span(rows).first(2560), std::span(memories).first(2560));
    };
    const auto seal = [&](std::span<const float> logits, uint32_t offset) {
        require(hc_exact_logits_hash(logits) == frozen.logit_hashes.at(offset), "verification cost ordinary path differs from sealed full vocabulary");
    };
    auto base_logits = retained.get(base).logits; seal(base_logits, 0);
    std::vector<std::byte> state_scratch(1000000);
    double fingerprint_ms = 0;
    const auto state = [&] {
        const auto started = Clock::now(); const auto out = stack.accepted_fingerprint(state_scratch);
        fingerprint_ms += milliseconds(started); return out;
    };
    const auto state_hash = [](const TextStack::AcceptedFingerprint& value) {
        constexpr char hex[] = "0123456789abcdef"; std::string out; out.reserve(64);
        for (auto byte : value.sha256) { out += hex[byte >> 4]; out += hex[byte & 15]; } return out;
    };
    const auto base_state = state();
    // Eight vocabulary copies plus a bounded streaming-state scratch.
    // Fingerprints and vector comparisons are outside every timed round.
    const uint64_t host_bound = state_scratch.capacity() + 8 * uint64_t(base_logits.capacity()) * sizeof(float) +
        uint64_t(rows.capacity() + memories.capacity()) * sizeof(float) + verification_cost_manifest->bytes;
    require(host_bound <= 512000000, "verification cost private host scratch exceeds512MB");
    const auto resident = gpu.memory();
    // An independent cold ordinary preparation in this same owner checks the
    // checkpoint's starting state. It uses exactly the existing capacity8 route.
    const auto cold_started = Clock::now(); stack.reset(); StackResult cold{};
    for (size_t at = 0; at < frozen.input.size(); at += 8) {
        const auto tokens = std::span(frozen.input).subspan(at, std::min<size_t>(8, frozen.input.size() - at));
        gather(tokens, false); cold = stack.step_known(tokens, std::span(rows).first(tokens.size() * 2560), std::span(memories).first(tokens.size() * 2560));
    }
    require(equal(cold.logits, base_logits) && state() == base_state,
        "verification cost cold preparation differs from starting state"); seal(cold.logits, 0);
    const auto cold_ms = milliseconds(cold_started); const auto checkpoint = stack.retain_checkpoint();
    const auto restore = [&] {
        const auto started = Clock::now(); require(stack.restore_checkpoint(checkpoint) == base, "verification cost restore prefix differs");
        const auto ms = milliseconds(started);
        require(state() == base_state, "verification cost restore changed accepted state"); return ms;
    };
    for (size_t arm_index = 0; arm_index < frozen.order.size(); ++arm_index) {
        const auto& arm = frozen.order[arm_index]; const auto fingerprint_before = fingerprint_ms; const auto restore_ms = restore();
        BatchResult verification{}, recovery{}, correction{}; uint32_t accepted = 0, evaluated = 0, heads = 0;
        std::vector<float> result_logits; result_logits.reserve(base_logits.size());
        // Full round includes candidate-ID handling, real embedding gathers,
        // verifier host readback/publication, compact rollback and correction.
        const auto started = Clock::now(); const auto proposals = std::span(arm.proposals);
        double gather_ms = 0, verify_ms = 0, correction_ms = 0;
        if (arm.rows == 1) {
            require(TargetLookup<StackResult>::greedy(base_logits) == proposals[0], "verification cost N1 is not greedy");
            const auto ordinary_started = Clock::now(); auto result = ordinary(proposals[0]);
            correction_ms = milliseconds(ordinary_started); correction = result.batch; result_logits = std::move(result.logits); evaluated = 1; heads = 1;
        } else {
            const auto gather_started = Clock::now(); gather(proposals, true); gather_ms = milliseconds(gather_started);
            const auto result = stack.verify_greedy8(proposals, std::span(rows).first(arm.rows * 2560), std::span(memories).first(arm.rows * 2560),
                base, base_logits, arm.rows, {});
            accepted = result.accepted; evaluated = result.evaluated; heads = result.head_dispatches;
            verification = result.verification; recovery = result.recovery; verify_ms = result.wall_ms;
            require(accepted == arm.accepted, "verification cost acceptance differs from frozen arm");
            if (accepted < arm.rows) {
                const auto ordinary_started = Clock::now();
                const auto id = TargetLookup<StackResult>::greedy(result.logits);
                require(id == frozen.continuation[accepted], "verification cost correction differs from sealed continuation");
                auto next = ordinary(id); correction = next.batch; result_logits = std::move(next.logits); correction_ms = milliseconds(ordinary_started);
            } else result_logits.assign(result.logits.begin(), result.logits.end());
        }
        const auto round_ms = milliseconds(started); const auto validation_started = Clock::now();
        const uint32_t corrected = arm.rows == 1 || accepted < arm.rows;
        const uint32_t emitted = arm.rows == 1 ? 1 : accepted + corrected;
        require(stack.position() == base + emitted, "verification cost published prefix differs");
        seal(result_logits, emitted); auto candidate_state = state();
        const auto result_hash = hc_exact_logits_hash(result_logits);
        // Rejected future rows must also be invisible to a real ordinary append.
        const auto next_id = frozen.continuation[emitted];
        require(TargetLookup<StackResult>::greedy(result_logits) == next_id, "verification cost continuation is not sealed greedy");
        auto candidate_next = ordinary(next_id); seal(candidate_next.logits, emitted + 1);
        auto candidate_next_state = state(); static_cast<void>(restore());
        StackResult serial{};
        for (uint32_t i = 0; i < emitted; ++i) {
            serial = ordinary(frozen.continuation[i]); seal(serial.logits, i + 1);
        }
        require(equal(serial.logits, result_logits) && state() == candidate_state,
            "verification cost full vocabulary or accepted state differs from ordinary replay");
        auto serial_next = ordinary(next_id);
        require(equal(serial_next.logits, candidate_next.logits) && state() == candidate_next_state,
            "verification cost rejected tail contaminated ordinary continuation");
        require(gpu.memory().live_bytes == resident.live_bytes && gpu.memory().allocations == resident.allocations,
            "verification cost changed prepared GPU allocations");
        std::cout << decorate(Json(Json::Object{{"event", Json("verification_cost_case")}, {"schema", Json::integer(1)},
            {"prefix", Json::integer(base)}, {"arm_index", Json::integer(arm_index)}, {"seed", Json::integer(verification_cost_manifest->seed)},
            {"scope", Json("direct-proposal target round; excludes lookup search, learned head, session envelopes and streaming")},
            {"requested_target_input_rows", Json::integer(arm.rows)}, {"candidate_id_count", Json::integer(arm.rows)},
            {"draft_candidate_count", Json::integer(arm.rows - 1)}, {"evaluated_target_rows", Json::integer(evaluated)},
            {"accepted_verified_tokens", Json::integer(accepted)}, {"accepted_draft_tokens", Json::integer(accepted ? accepted - 1 : 0)},
            {"emitted_tokens", Json::integer(emitted)}, {"correction_tokens", Json::integer(arm.rows > 1 ? corrected : 0)},
            {"ordinary_tokens", Json::integer(corrected)}, {"bonus_tokens", Json::integer(0)}, {"head_dispatches", Json::integer(heads)},
            {"round_wall_ms", number(round_ms)}, {"gather_wall_ms", number(gather_ms)}, {"verification_including_recovery_wall_ms", number(verify_ms)},
            {"correction_wall_ms", number(correction_ms)}, {"restore_wall_ms_excluded", number(restore_ms)},
            {"cold_control_wall_ms_excluded", number(cold_ms)}, {"validation_wall_ms_excluded", number(milliseconds(validation_started))},
            {"state_fingerprint_wall_ms_excluded", number(fingerprint_ms - fingerprint_before)}, {"verification_gpu_ms", number(verification.gpu_ms)},
            {"recovery_gpu_ms", number(recovery.gpu_ms)}, {"correction_gpu_ms", number(correction.gpu_ms)},
            {"round_gpu_ms", number(verification.gpu_ms + recovery.gpu_ms + correction.gpu_ms)},
            {"verification_batch_wall_ms", number(verification.wall_ms)}, {"recovery_batch_wall_ms", number(recovery.wall_ms)},
            {"correction_batch_wall_ms", number(correction.wall_ms)},
            {"dispatches", Json::integer(verification.dispatches + recovery.dispatches + correction.dispatches)},
            {"final_logit_sha256", Json(result_hash)}, {"accepted_state_sha256", Json(state_hash(candidate_state))},
            {"ordinary_continuation_state_sha256", Json(state_hash(candidate_next_state))}, {"full_vocabulary_exact", Json(true)}, {"accepted_state_exact", Json(true)},
            {"ordinary_continuation_exact", Json(true)}, {"accepted_state_bytes", Json::integer(base_state.bytes)}, {"private_host_scratch_bound_bytes", Json::integer(host_bound)},
            {"timings_qualified", Json(false)}, {"learned_head", Json(false)}, {"drafting_cost_included", Json(false)}})).dump() << std::endl;
    }
    static_cast<void>(restore());
};
