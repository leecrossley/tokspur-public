// Private current-carrier sibling of verification_cost_probe.inc. Same exact
// fingerprint/full-vocabulary/ordinary-continuation checks; reset/reprepare
// replaces persistent checkpoints. No RNG, learned-head or HTTP path is tested.
const auto run_sampled_verification_cost = [&](const verification_cost::SampledManifest& frozen) {
    constexpr auto measured = verification_cost::sampled_current_rows;
    const auto base = uint32_t(frozen.input.size());
    const auto equal = [](std::span<const float> a, std::span<const float> b) {
        return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size_bytes());
    };
    struct Frame { std::vector<float> logits, hc; TextStack::AcceptedFingerprint state; };
    struct Totals { double wall = 0, gpu = 0, batch_wall = 0, recovery_gpu = 0, recovery_wall = 0;
        uint64_t dispatches = 0, batches = 0, heads = 0; };
    std::vector<Frame> reference(frozen.teacher.size() + 1);
    std::vector<float> rows(8 * 2560), memories(rows.size());
    std::vector<std::byte> state_scratch(1000000);
    const auto resident = gpu.memory();
    const auto ledger = [&] { const auto now = gpu.memory();
        require(now.live_bytes == resident.live_bytes && now.allocations == resident.allocations &&
            !now.profile_counter_payload_bytes && !now.live_captures, "sampled COST prepared GPU ledger changed"); };
    const auto fingerprint = [&] { return stack.accepted_fingerprint(state_scratch); };
    const auto hash_state = [](const TextStack::AcceptedFingerprint& value) {
        constexpr char hex[] = "0123456789abcdef"; std::string out; out.reserve(64);
        for (auto byte : value.sha256) { out += hex[byte >> 4]; out += hex[byte & 15]; } return out;
    };
    constexpr uint64_t host_bound = 64000000;
    const auto host_ledger = [&] {
        uint64_t bytes = state_scratch.capacity() + reference.capacity() * sizeof(Frame) +
            uint64_t(rows.capacity() + memories.capacity()) * sizeof(float) + frozen.bytes;
        for (const auto& item : reference) bytes += uint64_t(item.logits.capacity() + item.hc.capacity()) * sizeof(float);
        require(bytes <= host_bound, "sampled COST private host scratch exceeds64MB"); return bytes;
    };
    const auto gather = [&](std::span<const uint32_t> ids, bool verifier) {
        if (verifier) {
            const auto lookups = stack.sampled_verification_cost_lookups(ids);
            for (size_t i = 0; i < ids.size(); ++i)
                cache.gather(lookups[i].rows, std::span(memories).subspan(i * 2560, 2560));
        } else {
            const auto lookups = stack.lookups(ids);
            for (size_t i = 0; i < ids.size(); ++i)
                cache.gather(lookups[i].rows, std::span(memories).subspan(i * 2560, 2560));
        }
        for (size_t i = 0; i < ids.size(); ++i)
            dequantize(token_table.type, files.tensor_bytes(token_table.name).subspan(uint64_t(ids[i]) * row_bytes, row_bytes),
                std::span(rows).subspan(i * 2560, 2560));
    };
    const auto ordinary = [&](uint32_t id) {
        gather(std::span(&id, 1), false);
        return stack.step(id, std::span(rows).first(2560), std::span(memories).first(2560));
    };
    const auto frame = [&](std::span<const float> logits) {
        const auto hc = stack.last_final_hc_residual();
        require(hc.size() == 10240, "sampled COST requires the complete four-stream HC tap");
        return Frame{{logits.begin(), logits.end()}, {hc.begin(), hc.end()}, fingerprint()};
    };
    const auto check_frame = [&](std::span<const float> logits, uint32_t consumed) {
        require(stack.position() == base + consumed && consumed < reference.size(), "sampled COST accepted prefix differs");
        require(equal(logits, reference[consumed].logits) &&
            equal(stack.last_final_hc_residual(), reference[consumed].hc) && fingerprint() == reference[consumed].state,
            "sampled COST full vocabulary, HC or accepted state differs from ordinary reference"); ledger();
    };
    const auto prepare = [&] {
        const auto started = Clock::now(); stack.reset(); StackResult out{};
        for (size_t at = 0; at < frozen.input.size(); at += 8) {
            const auto ids = std::span(frozen.input).subspan(at, std::min<size_t>(8, frozen.input.size() - at));
            gather(ids, false); out = stack.step_known(ids, std::span(rows).first(ids.size() * 2560),
                std::span(memories).first(ids.size() * 2560));
        }
        const auto ms = milliseconds(started); require(stack.position() == base, "sampled COST reprepare prefix differs");
        if (reference[0].logits.empty()) reference[0] = frame(out.logits); else check_frame(out.logits, 0);
        ledger(); return ms;
    };
    const auto emit = [&](std::string kind, uint32_t width, uint32_t committed, uint32_t start,
        uint32_t count, const Totals& total, double prefill_ms, uint32_t end) {
        Json::Array hashes, hc_hashes;
        for (uint32_t i = start; i < start + count; ++i) {
            const auto seal = sampled_cost::row_seal(reference[i + 1].logits,reference[i + 1].hc,hash_state(reference[i + 1].state));
            hashes.emplace_back(*seal.find("logit")); hc_hashes.emplace_back(*seal.find("hc"));
        }
        const bool ordinary_reference = kind == "ordinary";
        auto fields = Json::Object{{"event",Json("verification_cost_case")},{"schema",Json::integer(3)},
            {"kind",Json(std::move(kind))},{"prefix",Json::integer(base + start)},
            {"requested_target_input_rows",Json::integer(width)},{"consumed_per_batch",Json::integer(committed)},
            {"evaluated_rows",Json::integer(count)},{"emitted_tokens",Json::integer(end - start)},
            {"batches",Json::integer(total.batches)},{"head_dispatches",Json::integer(total.heads)},
            {"round_wall_ms",number(total.wall)},{"round_gpu_ms",number(total.gpu)},
            {"verification_gpu_ms",number(total.gpu - total.recovery_gpu)},
            {"recovery_gpu_ms",number(total.recovery_gpu)},{"recovery_batch_wall_ms",number(total.recovery_wall)},
            {"verification_batch_wall_ms",number(total.batch_wall)},{"dispatches",Json::integer(total.dispatches)},
            {"cold_control_wall_ms_excluded",number(prefill_ms)},{"validation_outside_timed_rounds",Json(true)},
            {"row_logit_sha256",Json(std::move(hashes))},{"row_hc_sha256",Json(std::move(hc_hashes))},
            {"final_logit_sha256",Json(hc_exact_logits_hash(reference[end].logits))},
            {"accepted_state_sha256",Json(hash_state(reference[end].state))},
            {"ordinary_continuation_state_sha256",Json(hash_state(reference[end + 1].state))},
            {"accepted_state_bytes",Json::integer(reference[end].state.bytes)},
            {"full_vocabulary_exact",Json(true)},{"full_hc_exact",Json(true)},{"accepted_state_exact",Json(true)},
            {"ordinary_continuation_exact",Json(true)},{"private_host_scratch_bound_bytes",Json::integer(host_bound)},
            {"private_host_scratch_bytes",Json::integer(host_ledger())},
            {"timings_qualified",Json(false)},{"learned_head",Json(false)},{"drafting_cost_included",Json(false)},
            {"sampled_law_tested",Json(false)},{"sampler_entropy_consumed",Json::integer(0)},
            {"tracked_metal_buffer_bytes",Json::integer(gpu.memory().live_bytes)},
            {"gpu_allocation_count",Json::integer(gpu.memory().allocations)}};
        if (ordinary_reference) {
            Json::Array seals;
            for (const auto& item : reference)
                seals.emplace_back(sampled_cost::row_seal(item.logits,item.hc,hash_state(item.state)));
            fields.emplace_back("ordinary_reference",Json(std::move(seals)));
        }
        std::cout << decorate(Json(std::move(fields))).dump() << std::endl;
    };
    // Ordinary reference first; twelve measured rows, twenty-four retained
    // frames for partial-commit/continuation witnesses. Hashing is never timed.
    const auto ordinary_prefill = prepare(); Totals ordinary_total;
    for (uint32_t i = 0; i < frozen.teacher.size(); ++i) {
        const auto started = Clock::now(); auto result = ordinary(frozen.teacher[i]); const auto wall = milliseconds(started);
        if (i < measured) { ordinary_total.wall += wall; ordinary_total.gpu += result.batch.gpu_ms;
            ordinary_total.batch_wall += result.batch.wall_ms; ordinary_total.dispatches += result.batch.dispatches;
            ++ordinary_total.batches; ++ordinary_total.heads; }
        reference[i + 1] = frame(result.logits); ledger();
    }
    emit("ordinary",1,1,0,measured,ordinary_total,ordinary_prefill,measured);
    struct Choice { uint32_t count, calls = 0;
        static uint32_t choose(void* raw, const StackSampledVerificationRows&) {
            auto& choice = *static_cast<Choice*>(raw); ++choice.calls; return choice.count;
        } };
    const auto transaction = [&](uint32_t at, uint32_t width, uint32_t commit, Totals& total) {
        const auto ids = std::span(frozen.teacher).subspan(at, width); Choice choice{commit};
        const auto started = Clock::now(); gather(ids,true);
        const auto result = width == 3 ? stack.verify_sampled_cost3(ids,std::span(rows).first(width * 2560),
            std::span(memories).first(width * 2560),base + at,{Choice::choose,&choice}) :
            stack.verify_sampled2(ids,std::span(rows).first(width * 2560),std::span(memories).first(width * 2560),
                base + at,{Choice::choose,&choice});
        total.wall += milliseconds(started); total.gpu += result.verification.gpu_ms + result.recovery.gpu_ms;
        total.batch_wall += result.verification.wall_ms; total.recovery_gpu += result.recovery.gpu_ms;
        total.recovery_wall += result.recovery.wall_ms; total.dispatches += result.verification.dispatches + result.recovery.dispatches;
        total.heads += width == 3 ? 2 : 1; ++total.batches;
        require(choice.calls == 1 && result.evaluated.count == width && result.committed == commit,
            "sampled COST selector count differs");
        // Borrowed rows are checked immediately, before an ordinary append or
        // next transaction can overwrite their fixed host storage.
        for (uint32_t i = 0; i < width; ++i)
            require(equal(result.evaluated.logits[i],reference[at + i + 1].logits) &&
                equal(result.evaluated.final_hc_rows.subspan(uint64_t(i) * 10240,10240),reference[at + i + 1].hc),
                "sampled COST evaluated full vocabulary/HC differs from ordinary reference");
        check_frame(result.logits,at + commit);
    };
    for (uint32_t width : {1,2,3}) {
        const auto prefill = prepare(); Totals total;
        for (uint32_t at = 0; at < measured; at += width) transaction(at,width,width,total);
        auto next = ordinary(frozen.teacher[measured]); check_frame(next.logits,measured + 1);
        emit("verifier",width,width,0,measured,total,prefill,measured);
        if (width == 3) {
            uint32_t at = measured + 1;
            for (uint32_t commit : {1,2,3}) {
                Totals partial; transaction(at,3,commit,partial);
                auto append = ordinary(frozen.teacher[at + commit]); check_frame(append.logits,at + commit + 1);
                emit("partial",3,commit,at,3,partial,0,at + commit); at += commit + 1;
            }
        }
    }
    ledger(); files.assert_unchanged();
    std::cout << decorate(Json(Json::Object{{"event",Json("verification_cost_done")},{"schema",Json::integer(3)},
        {"passed",Json(true)},{"cases",Json::integer(7)},{"prefill_runs",Json::integer(4)},
        {"reference_rows",Json::integer(frozen.teacher.size())},{"measured_rows_per_full_arm",Json::integer(measured)},
        {"committed_prefix",Json::integer(stack.position())},{"continuation_allowed",Json(false)},
        {"sampler_entropy_consumed",Json::integer(0)},{"timings_qualified",Json(false)},
        {"tracked_metal_buffer_bytes",Json::integer(gpu.memory().live_bytes)},
        {"gpu_allocation_count",Json::integer(gpu.memory().allocations)}})).dump() << std::endl;
};
