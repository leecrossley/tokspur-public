#include "tokspur/mtp_anchor_state.hpp"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tokspur;
size_t checks = 0;
void check(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void refused(const std::function<void()>& action) {
    ++checks; try { action(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid MTP anchor transition accepted");
}
SampledPathDecision decision(uint32_t base, bool accepted, bool stopped, bool terminal_is_draft) {
    const SamplingRng random(44); SampledPathDecision out(random);
    out.identity = {1, 1, 1, base + 1, 10, 2};
    out.accepted = accepted; out.stopped = stopped;
    out.count = accepted && !terminal_is_draft ? 2 : 1;
    out.tokens[0] = {3, 0.75f, accepted};
    if (out.count == 2) out.tokens[1] = {4, 0.625f, false};
    return out;
}
void transitions() {
    MtpAnchorCarry carry;
    check(!carry.peek(100), "fresh carry has a pending anchor");
    auto accepted = decision(100, true, false, false);
    const auto accepted_rng = accepted.random_after;
    const auto a = prepare_mtp_anchor_publication(100, accepted);
    check(a.emitted == 2 && a.prefix == 102 && !a.stopped, "accepted round consumed-prefix shape");
    check(a.pending && a.pending->id == 4 && !a.pending->accepted_proposal, "target bonus was not carried");
    carry.publish(a);
    check(carry.peek(102)->id == 4 && carry.peek(102)->source_logit == 0.625f, "carried bonus identity/score changed");
    refused([&] { static_cast<void>(carry.peek(101)); });
    check(accepted.random_after == accepted_rng, "carry advanced or rewound transactional RNG");
    // Rejected draft commits/emits only the anchor. The correction was sampled
    // already, and later replaces it as the next anchor without an extra draw.
    auto rejected = decision(102, false, false, false);
    const auto r = prepare_mtp_anchor_publication(102, rejected);
    check(r.emitted == 1 && r.prefix == 103 && r.pending && r.pending->id == 3, "rejected correction transition");
    carry.publish(r);
    check(carry.peek(103)->id == 3, "rejected correction was dropped");
    check(carry.clear() && !carry.clear() && !carry.peek(103), "ordinary tail did not consume carry exactly once");
    // A draft EOS is already in the consumed target state and stops now.
    const auto eos = prepare_mtp_anchor_publication(103, decision(103, true, true, true));
    check(eos.stopped && eos.emitted == 2 && !eos.pending, "accepted draft EOS did not stop immediately");
    // A sampled EOS bonus/correction is still invisible: it stops only if a
    // subsequent target step consumes/emits it. Output budget may discard it.
    const auto bonus_eos = prepare_mtp_anchor_publication(103, decision(103, true, true, false));
    check(!bonus_eos.stopped && bonus_eos.pending && bonus_eos.emitted == 2, "bonus EOS stopped before consumption");
    const auto correction_eos = prepare_mtp_anchor_publication(103, decision(103, false, true, false));
    check(!correction_eos.stopped && correction_eos.pending && correction_eos.emitted == 1, "correction EOS stopped before consumption");
    carry.publish(bonus_eos);
    check(carry.clear(), "output-budget boundary failed to discard an invisible terminal");
    const auto boundary = prepare_mtp_anchor_publication(65534, decision(65534, true, false, false));
    check(boundary.prefix == 65536 && boundary.pending, "last reserved two-token boundary rejected");
    refused([&] { static_cast<void>(prepare_mtp_anchor_publication(65535, decision(65535, true, false, false))); });
    auto bad = accepted; bad.identity.prefix = 100;
    refused([&] { static_cast<void>(prepare_mtp_anchor_publication(100, bad)); });
    bad = accepted; bad.tokens[1].source_logit = std::numeric_limits<float>::infinity();
    refused([&] { static_cast<void>(prepare_mtp_anchor_publication(100, bad)); });
    bad = accepted; bad.count = 1;
    refused([&] { static_cast<void>(prepare_mtp_anchor_publication(100, bad)); });
}
void generation_lifetime() {
    const std::array<uint32_t, 1> stops{9};
    const MtpGenerationCommand first{1, 4, true, false}, next{1, 2, false, false};
    MtpGenerationCarry state;
    state.admit(first, 2, 100, stops);
    auto accepted = prepare_mtp_anchor_publication(100, decision(100, true, true, false));
    accepted.pending->id = 9; // Already-drawn bonus EOS must survive a packet.
    state.carry().publish(accepted);
    check(!state.finish(2, 102, false) && state.active() && state.remaining() == 2, "packet discarded the pending bonus EOS");
    check(state.pending(102) && state.carry().peek(102)->id == 9, "pending EOS identity changed across packets");
    refused([&] { state.admit({2, 2, true, false}, 2, 102, stops); });
    refused([&] { state.admit({1, 3, false, false}, 2, 102, stops); });
    refused([&] { state.admit(next, 2, 101, stops); });
    refused([&] { state.admit(next, 2, 102, std::array<uint32_t, 1>{8}); });
    state.admit(next, 2, 102, stops);
    check(state.carry().clear(), "ordinary consumed-EOS transition lost its pending token");
    check(!state.finish(1, 103, true) && !state.active() && !state.pending(103), "consumed EOS failed to close generation");
    refused([&] { state.admit(next, 2, 103, stops); });
    refused([&] { state.admit({1, 2, true, false}, 2, 103, stops); });
    MtpGenerationCarry rejected;
    rejected.admit({1, 4, true, false}, 1, 200, stops);
    const auto publication = prepare_mtp_anchor_publication(200, decision(200, false, false, false));
    rejected.carry().publish(publication);
    check(!rejected.finish(1, 201, false) && rejected.carry().peek(201)->id == 3, "correction was not retained exactly");
    refused([&] { static_cast<void>(rejected.abort({2, 0, false, true})); });
    check(rejected.abort({1, 0, false, true}) && !rejected.active() && !rejected.pending(201), "abort failed to discard invisible correction");
    refused([&] { static_cast<void>(rejected.abort({1, 0, false, true})); });
    rejected.admit({2, 1, true, false}, 1, 201, stops);
    check(!rejected.finish(1, 202, false) && !rejected.active(), "final budget did not close generation");
    rejected.admit({3, 4, true, false}, 2, 202, stops);
    rejected.carry().publish(prepare_mtp_anchor_publication(202, decision(202, true, false, false)));
    check(rejected.clear() && !rejected.active(), "known input/reset/error invalidation retained carry");
    MtpGenerationCarry correction_eos;
    correction_eos.admit({1, 4, true, false}, 1, 300, stops);
    auto eos_decision = decision(300, false, true, false); eos_decision.tokens[0].id = 9;
    correction_eos.carry().publish(prepare_mtp_anchor_publication(300, eos_decision));
    check(!correction_eos.finish(1, 301, false) && correction_eos.pending(301), "correction EOS stopped before physical consumption");
    correction_eos.admit({1, 3, false, false}, 1, 301, stops);
    check(correction_eos.carry().peek(301)->id == 9 && correction_eos.carry().clear(), "correction EOS was redrawn or dropped");
    check(!correction_eos.finish(1, 302, true) && !correction_eos.active(), "consumed correction EOS failed to stop");
    MtpGenerationCarry final_budget;
    final_budget.admit({1, 2, true, false}, 2, 400, stops);
    final_budget.carry().publish(prepare_mtp_anchor_publication(400, decision(400, true, false, false)));
    check(final_budget.finish(2, 402, false) && !final_budget.active() && !final_budget.pending(402), "final budget retained an invisible bonus");
    const auto command = mtp_generation_command(Json(Json::Object{{"id", Json::integer(4)}, {"begin", Json(true)}, {"remaining", Json::integer(8)}}));
    check(command.id == 4 && command.begin && command.remaining == 8 && !command.abort, "generation packet parsing changed identity");
    for (const auto text : {"{}", "{\"id\":0,\"abort\":true}", "{\"id\":1,\"abort\":false}",
         "{\"id\":1,\"begin\":true,\"remaining\":0}", "{\"id\":1,\"begin\":true,\"remaining\":32769}",
         "{\"id\":1,\"abort\":true,\"remaining\":1}", "{\"id\":1,\"begin\":true,\"other\":1}"})
        refused([&] { static_cast<void>(mtp_generation_command(parse_json(text))); });
}

void generation_packet_limits() {
    const std::array<uint32_t, 1> stops{9};
    for (const uint32_t maximum : {9u, 64u, 65u}) {
        MtpGenerationCarry state;
        const auto before = state.metadata(100).dump();
        refused([&] { state.admit({1, 128, true, false}, maximum, 100, stops); });
        refused([&] { state.admit({1, 128, true, false}, maximum, 100, stops, 8); });
        check(state.metadata(100).dump() == before, "refused default/eight packet changed generation state");
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        if (maximum <= 64) {
            state.admit({1, maximum, true, false}, maximum, 100, stops, 64);
            check(state.active() && state.remaining() == maximum, "explicit wide packet was not admitted");
            check(!state.finish(maximum, 100 + maximum, false) && !state.active(), "wide packet did not close its exact budget");
        } else {
            refused([&] { state.admit({1, 128, true, false}, maximum, 100, stops, 64); });
            check(state.metadata(100).dump() == before, "oversize wide packet changed generation state");
        }
#else
        refused([&] { state.admit({1, 128, true, false}, maximum, 100, stops, 64); });
        check(state.metadata(100).dump() == before, "legacy build admitted wide packet consent");
#endif
    }
    for (const uint32_t invalid_limit : {0u, 1u, 9u, 63u, 65u}) {
        MtpGenerationCarry state;
        refused([&] { state.admit({1, 128, true, false}, 1, 100, stops, invalid_limit); });
        check(!state.active() && !state.id(), "invalid packet limit changed generation identity");
    }
#ifndef TOKSPUR_PRIVATE_MTP_OUTPUT64
    MtpGenerationCarry legacy;
    refused([&] { legacy.admit({1, 128, true, false}, 1, 100, stops, 64); });
#endif
}

#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
void wide_generation_lifetime() {
    const std::array<uint32_t, 1> stops{9};
    const SampledPathToken terminal{4, 0.625f, false};
    MtpGenerationCarry state;
    state.admit({1, 129, true, false}, 64, 100, stops, 64);
    state.carry().publish({164, 64, false, terminal});
    check(!state.finish(64, 164, false) && state.remaining() == 65 && state.pending(164), "first wide packet lost its pending terminal");
    const auto prior = state.metadata(164).dump();
    refused([&] { state.admit({1, 65, false, false}, 65, 164, stops, 64); });
    refused([&] { state.admit({1, 64, false, false}, 64, 164, stops, 64); });
    refused([&] { state.admit({1, 65, false, false}, 64, 163, stops, 64); });
    refused([&] { state.admit({1, 65, false, false}, 64, 164, std::array<uint32_t, 1>{8}, 64); });
    check(state.metadata(164).dump() == prior, "refused wide continuation changed committed generation state");
    state.admit({1, 65, false, false}, 64, 164, stops, 64);
    refused([&] { static_cast<void>(state.finish(63, 227, false)); });
    refused([&] { static_cast<void>(state.finish(65, 229, false)); });
    check(state.metadata(164).dump() == prior, "refused wide publication changed committed generation state");
    state.carry().publish({228, 64, false, terminal});
    check(!state.finish(64, 228, false) && state.remaining() == 1 && state.pending(228), "second wide packet lost its exact remaining budget");
    state.admit({1, 1, false, false}, 1, 228, stops, 64);
    state.carry().publish({229, 1, false, terminal});
    check(state.finish(1, 229, false) && !state.active() && !state.remaining() && !state.pending(229), "final wide-generation tail retained its terminal");
    state.admit({2, 128, true, false}, 64, 229, stops, 64);
    state.carry().publish({233, 4, true, {}});
    check(!state.finish(4, 233, true) && !state.active() && !state.remaining(), "early stop failed to close a wide generation");
    state.admit({3, 128, true, false}, 64, 233, stops, 64);
    state.carry().publish({297, 64, false, terminal});
    check(!state.finish(64, 297, false), "wide packet discarded its terminal before abort");
    check(state.abort({3, 0, false, true}) && !state.active() && !state.pending(297), "wide-generation abort retained its terminal");
    state.admit({4, 8, true, false}, 8, 297, stops);
    check(!state.finish(8, 305, false) && !state.active(), "wide-generation error recovery lost default-eight reuse");
}
#endif

#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
void depth2_publication() {
    for (uint32_t active : {1u, 2u, 3u}) for (uint32_t accepted = 0; accepted <= active; ++accepted) for (const bool stop : {false, true}) {
        if (stop && !accepted) continue;
        SampledPathDecision d(SamplingRng(44)); d.identity={1,1,1,101,10,2}; d.accepted=accepted; d.active_proposals=active;
        d.terminal = stop ? SampledPathTerminal::accepted_stop : accepted == active ? SampledPathTerminal::target_bonus : SampledPathTerminal::correction;
        d.stopped=stop; d.count=accepted+(stop?0:1);
        for (uint32_t i=0;i<d.count;++i)d.tokens[i]={i+3,0.25f,i<accepted};
        const auto out=prepare_mtp_anchor_publication2(100,d);
        check(out.prefix==101+accepted && out.emitted==1+accepted && out.stopped==stop,"depth-two consumed frontier differs");
        check(bool(out.pending)==!stop,"depth-two correction/bonus lifetime differs");
        MtpGenerationCarry state; state.admit({1,8,true,false},std::max(3u, active + 1),100,std::array<uint32_t,1>{9});state.carry().publish(out);
        if (stop) check(!state.finish(out.emitted,out.prefix,true)&&!state.active(),"depth-two accepted stop did not close generation");
        else {
            check(state.carry().peek(out.prefix)->id==d.tokens[d.count-1].id,"depth-two pending terminal identity differs");
            check(state.clear(),"depth-two cancellation/reset failed to clear pending terminal");
        }
    }
}

void stage_a_controller_transitions() {
    // 1. CostTable lookups
    check(StageACostTable::lookup_t_ord_us(100) == 34430, "CostTable t_ord 100 differs");
    check(StageACostTable::lookup_t_ord_us(512) == 34430, "CostTable t_ord 512 differs");
    check(StageACostTable::lookup_t_ord_us(1024) == 36324, "CostTable t_ord 1024 differs");
    check(StageACostTable::lookup_t_ord_us(2048) == 44652, "CostTable t_ord 2048 differs");
    check(StageACostTable::lookup_t_ord_us(4096) == 44673, "CostTable t_ord 4096 differs");

    check(std::abs(StageACostTable::lookup_alpha2_hat(0.0) - 0.182) < 1e-6, "CostTable alpha2 0 differs");
    check(std::abs(StageACostTable::lookup_alpha2_hat(1.0) - 0.915) < 1e-6, "CostTable alpha2 1 differs");
    check(StageACostTable::expected_speculative_throughput(0.75, 3) >
          StageACostTable::expected_speculative_throughput(0.75, 2),
          "CostTable k=3 throughput should exceed k=2 at alpha1=0.75");

    // 2. StageAControllerState default values and digest
    StageAControllerState state{};
    check(std::abs(state.ewma_alpha1 - 0.45) < 1e-6, "default ewma_alpha1 differs");
    check(state.current_mode == 0, "default current_mode differs");
    check(state.consecutive_ordinary_steps == 0, "default consecutive_ordinary_steps differs");
    check(state.controller_policy_digest == default_controller_policy_digest, "default digest differs");

    // 3. Digest mismatch refusal in admit
    MtpGenerationCarry carry;
    MtpGenerationCommand bad_cmd{
        .id = 1,
        .remaining = 8,
        .begin = true,
        .stage_a_controller = StageAControllerState{
            .controller_policy_digest = {}
        },
        .v2 = true
    };
    bool refused = false;
    try {
        carry.admit(bad_cmd, 8, 10, std::array<uint32_t, 1>{0});
    } catch (const FormatError&) {
        refused = true;
    }
    check(refused, "tampered digest was not refused");

    // 4. Successful admit and zero mutation during pop_overflow
    MtpGenerationCommand good_cmd{
        .id = 2,
        .remaining = 8,
        .begin = true,
        .total_budget = 8,
        .stage_a_controller = state,
        .v2 = true
    };
    carry.admit(good_cmd, 8, 10, std::array<uint32_t, 1>{0});
    check(carry.controller_state() == state, "admitted controller state differs");

    std::array<uint32_t, 2> ids{101, 102};
    std::array<float, 2> scores{1.0f, 2.0f};
    carry.commit_round(2, ids, scores, 10, 12, 1, false);
    check(carry.pending_overflow() == 1, "pending overflow expected 1");
    const auto before_pop = carry.controller_state();
    static_cast<void>(carry.pop_overflow());
    check(carry.controller_state() == before_pop, "pop_overflow mutated StageAControllerState");

    // 5. Ordinary k=0 step: immediate re-entry at high alpha1 (0.8) and preserved EWMA at low alpha1 (0.2)
    state.current_mode = uint8_t(StageAMode::OrdinaryK0);
    state.consecutive_ordinary_steps = 4;
    state.ewma_alpha1 = 0.8;
    stage_a_step_ordinary(state, 512);
    check(state.current_mode == uint8_t(StageAMode::Speculative), "high alpha1 (0.8) failed to re-enter Speculative");
    check(state.consecutive_ordinary_steps == 0, "consecutive_ordinary_steps reset failed on re-entry");
    check(std::abs(state.ewma_alpha1 - 0.8) < 1e-12, "ordinary step mutated ewma_alpha1 on re-entry");

    state.current_mode = uint8_t(StageAMode::OrdinaryK0);
    state.consecutive_ordinary_steps = 0;
    state.ewma_alpha1 = 0.2;
    stage_a_step_ordinary(state, 512);
    check(state.current_mode == uint8_t(StageAMode::OrdinaryK0), "low alpha1 (0.2) unexpectedly left OrdinaryK0");
    check(state.consecutive_ordinary_steps == 1, "consecutive_ordinary_steps increment failed");
    check(std::abs(state.ewma_alpha1 - 0.2) < 1e-12, "unobserved ordinary step decayed ewma_alpha1");

    // 6. Speculative exit transition when alpha1 is low
    state.current_mode = uint8_t(StageAMode::Speculative);
    state.ewma_alpha1 = 0.1;
    stage_a_commit_speculative(state, 0.0, 512);
    check(state.current_mode == uint8_t(StageAMode::OrdinaryK0), "exit to OrdinaryK0 failed");
    check(state.consecutive_ordinary_steps == 0, "consecutive_ordinary_steps reset failed on exit");

    // 7. Periodic probe rounds recover from low alpha1 (0.2) into Speculative without lock-out
    state.current_mode = uint8_t(StageAMode::OrdinaryK0);
    state.consecutive_ordinary_steps = 0;
    state.ewma_alpha1 = 0.2;
    uint32_t probes_until_reentry = 0;
    while (state.current_mode == uint8_t(StageAMode::OrdinaryK0) && probes_until_reentry < 16) {
        for (uint32_t ord = 0; ord < StageACostTable::probe_interval - 1; ++ord) {
            check(state.consecutive_ordinary_steps == ord, "ordinary step counter drifted before probe");
            stage_a_step_ordinary(state, 512);
            check(state.current_mode == uint8_t(StageAMode::OrdinaryK0), "premature re-entry during ordinary steps");
        }
        check(state.consecutive_ordinary_steps == StageACostTable::probe_interval - 1, "probe threshold not reached");
        const double before_probe = state.ewma_alpha1;
        stage_a_commit_probe(state, 1.0, 512);
        ++probes_until_reentry;
        check(state.consecutive_ordinary_steps == 0, "probe consecutive_ordinary_steps reset failed");
        check(state.ewma_alpha1 > before_probe, "accepted probe did not raise ewma_alpha1");
    }
    check(state.current_mode == uint8_t(StageAMode::Speculative), "Stage A locked out in OrdinaryK0 despite accepted probes");
    check(probes_until_reentry == 8, "expected 8 accepted periodic probes to cross re-entry threshold from 0.2");

    // 8. Multi-packet v2 generation lifetime and prefix publication
    MtpGenerationCarry multi;
    const std::array<uint32_t, 1> m_stops{0};
    MtpGenerationCommand p0_cmd{
        .id = 5,
        .remaining = 16,
        .begin = true,
        .committed = 0,
        .emitted = 0,
        .pending = 0,
        .remaining_compute = 16,
        .total_budget = 16,
        .v2 = true
    };
    multi.admit(p0_cmd, 8, 100, m_stops);
    std::array<uint32_t, 8> p0_ids{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, 8> p0_scores{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    multi.commit_round(8, p0_ids, p0_scores, 100, 108, 8, false);
    check(!multi.finish(8, 108, false) && multi.active(), "packet 0 finish failed");
    check(multi.remaining() == 8 && multi.committed() == 8 && multi.emitted() == 8, "packet 0 state mismatch");

    MtpGenerationCommand p1_cmd{
        .id = 5,
        .remaining = 8,
        .begin = false,
        .committed = 8,
        .emitted = 8,
        .pending = 0,
        .remaining_compute = 8,
        .total_budget = 16,
        .v2 = true
    };
    multi.admit(p1_cmd, 8, 108, m_stops);
    std::array<uint32_t, 8> p1_ids{9, 10, 11, 12, 13, 14, 15, 16};
    std::array<float, 8> p1_scores{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    multi.commit_round(8, p1_ids, p1_scores, 108, 116, 8, false);
    check(!multi.finish(8, 116, false) && !multi.active(), "packet 1 finish failed to close generation");
    check(multi.committed() == 0 && multi.remaining() == 0, "generation clear failed after final packet");
}

// A round whose accepted stop lands past the packet end finishes that packet as length, keeps the stop queued and publishes it from the next packet's drain.
void deferred_stop_drain() {
    MtpGenerationCarry carry;
    const std::array<uint32_t, 1> stops{9};
    carry.admit({.id = 7, .remaining = 16, .begin = true, .remaining_compute = 16, .total_budget = 16, .v2 = true}, 8, 100, stops);
    const std::array<uint32_t, 7> first_ids{1, 2, 3, 4, 5, 6, 7};
    const std::array<float, 7> first_scores{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    carry.commit_round(7, first_ids, first_scores, 100, 107, 7, false);
    const std::array<uint32_t, 2> stop_ids{42, 9};
    const std::array<float, 2> stop_scores{0.5f, 0.25f};
    carry.commit_round(2, stop_ids, stop_scores, 107, 109, 1, true);
    check(carry.pending_overflow() == 1 && carry.stop_queued() && !carry.stop_drained(), "overflowing stop was not queued");
    check(!carry.finish(8, 109, false) && carry.active() && carry.remaining() == 8, "packet with a queued stop did not finish as length");
    carry.admit({.id = 7, .remaining = 8, .begin = false, .committed = 9, .emitted = 8, .pending = 1, .remaining_compute = 7,
        .total_budget = 16, .v2 = true}, 8, 109, stops);
    const auto stop = carry.pop_overflow();
    check(stop.id == 9 && stop.token_prefix == 109 && stop.committed_prefix == 109, "queued stop identity or frontier changed");
    check(!carry.pending_overflow() && carry.emitted() == 9, "drained stop left overflow behind");
    carry.mark_stop_drained();
    check(carry.stop_drained() && !carry.finish(1, 109, true) && !carry.active(), "drained stop did not close the generation");
}
#endif

}

int main() {
    try { transitions(); generation_lifetime(); generation_packet_limits();
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        wide_generation_lifetime();
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        depth2_publication();
        stage_a_controller_transitions();
        deferred_stop_drain();
#endif
        std::cout << "MTP carried-anchor transition checks passed: " << checks << '\n'; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
