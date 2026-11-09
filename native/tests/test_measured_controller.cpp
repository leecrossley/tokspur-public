#include "tokspur/measured_controller.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/derived_pack.hpp"
#include "tokspur/sampling.hpp"
#include "tokspur/mtp_anchor_state.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace tokspur;
namespace {
void check(bool b) { if (!b) throw std::runtime_error("measured controller check failed"); }
void near(double a, double b) { check(std::abs(a - b) < 1e-10); }
template<class F> void refused(F fn) { try { fn(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid measured state accepted"); }
MeasuredCostProfile profile(double slow = 0, std::string proof = std::string(64, 'b'), std::string mode = "greedy",
                            std::string mapping = "") {
    return MeasuredCostProfile::parse(parse_json(std::string(R"({"schema":1,"drafter":"mtp","maximum_rows":4,"execution_digest":")") + std::string(64, 'a') +
        R"(","exactness_receipt":")" + proof + "\"" + mapping + R"(,"modes":[{"sampling":")" + mode + R"(","anchors":[
        {"context":4096,"ordinary_ms":10,"draft_ms":[0,2,4,6],"remaining_ms":[[10],[11,10],[12,11,10],[)" + std::to_string(slow + 13) + R"(,12,11,10]],"conditional_acceptance":[0.8,0.6,0.4]},
        {"context":4352,"ordinary_ms":20,"draft_ms":[0,4,8,12],"remaining_ms":[[20],[22,20],[24,22,20],[)" + std::to_string(slow + 26) + R"(,24,22,20]],"conditional_acceptance":[0.8,0.6,0.4]}]}]})"));
}
}
int main() {
    try {
        const auto p = profile(); const auto c = p.costs(4096, MeasuredSampling::greedy);
        check(measured_sampling({}) == MeasuredSampling::greedy);
        check(measured_sampling({1, 0, 1, 0}) == MeasuredSampling::full);
        check(measured_sampling({1, 20, 1, 0}) == MeasuredSampling::topk);
        check(measured_sampling({1, 20, .95, 0}) == MeasuredSampling::coding);
        check(measured_sampling_name(MeasuredSampling::coding) == "coding");
        for (const auto policy : {SamplingPolicy{1, 20, .9, 0}, SamplingPolicy{1, 0, .95, 0},
                SamplingPolicy{.9, 20, .95, 0}, SamplingPolicy{1, 20, .95, .1}})
            refused([&] { static_cast<void>(measured_sampling(policy)); });
        const auto coding = profile(0, std::string(64, 'b'), "coding");
        const auto topk = profile(0, std::string(64, 'b'), "topk");
        check(coding.policy_digest() != topk.policy_digest());
        refused([&] { static_cast<void>(topk.costs(4096, MeasuredSampling::coding)); });
        refused([&] { static_cast<void>(coding.costs(4096, MeasuredSampling::topk)); });
        MeasuredController coding_controller(coding);
        coding_controller.begin_generation(1, 4096, MeasuredSampling::coding);
        check(coding_controller.before_draft(4096, 4).converged);
        AcceptanceRates a{0.8, 0.6, 0.4};
        near(MeasuredController::reward(4, a), 1 + .8 + .48 + .192);
        const auto remaining = .2 * 13 + .8 * .4 * 12 + .48 * .6 * 11 + .192 * 10;
        near(MeasuredController::expected_cost(4, a, c, true), remaining);
        near(MeasuredController::expected_cost(4, a, c, false), 6 + remaining);
        near(p.costs(4224, MeasuredSampling::greedy).ordinary_ms, 15);
        check(p.costs(4224, MeasuredSampling::greedy).calibrated);
        check(!p.costs(5000, MeasuredSampling::greedy).calibrated);
        near(p.costs(5000, MeasuredSampling::greedy).ordinary_ms, 20);
        p.bind(std::string(64, 'a'), MeasuredDrafter::mtp);
        refused([&] { p.bind(std::string(64, 'b'), MeasuredDrafter::mtp); });
        refused([&] { p.bind(std::string(64, 'a'), MeasuredDrafter::mtp, true); });
        const auto shared = profile(0, std::string(64, 'b'), "greedy", ",\"shared_pack_mapping\":true");
        const auto explicit_private = profile(0, std::string(64, 'b'), "greedy", ",\"shared_pack_mapping\":false");
        check(shared.policy_digest() != p.policy_digest());
        check(explicit_private.policy_digest() == p.policy_digest());
        shared.bind(std::string(64, 'a'), MeasuredDrafter::mtp, false, true);
        refused([&] { shared.bind(std::string(64, 'a'), MeasuredDrafter::mtp); });
        refused([&] { p.bind(std::string(64, 'a'), MeasuredDrafter::mtp, false, true); });
        refused([&] { static_cast<void>(profile(0, std::string(64, 'b'), "greedy", ",\"shared_pack_mapping\":1")); });
        refused([&] { static_cast<void>(p.costs(4096, MeasuredSampling::full)); });
        MeasuredController m(p); refused([&] { static_cast<void>(m.before_draft(4096, 4)); });
        m.begin_generation(1, 4096, MeasuredSampling::greedy);
        auto choice = m.before_draft(4096, 4); check(choice.converged);
        double optimal_rate = 0.1;
        for (uint32_t width = 2; width <= 4; ++width)
            optimal_rate = std::max(optimal_rate, MeasuredController::reward(width, a) / MeasuredController::expected_cost(width, a, c, false));
        near(choice.lambda, optimal_rate);
        // Drift affects costs and rate inversely and leaves action ordering unchanged.
        m.observe(4096, 1, 0, 20, a); const auto drift = m.cost_multiplier();
        near(drift, 1.03); near(m.before_draft(4096, 4).lambda, optimal_rate / drift);
        for (unsigned i = 0; i < 40; ++i) m.observe(4096, 2, 1, 12, a);
        check(m.history_size() == 32);
        m.begin_generation(2, 4096, MeasuredSampling::greedy);
        check(m.history_size() == 1 && m.cost_multiplier() == 1);
        near(m.acceptance()[1], .6); near(m.acceptance()[2], .4);
        // Only actually reached conditional positions change. A failed first proposal provides no p2/p3 label.
        m.observe(4096, 4, 0, 19, a);
        near(m.acceptance()[1], .6); near(m.acceptance()[2], .4);
        const auto old_second = m.acceptance()[1];
        m.observe(4096, 4, 1, 18, a, true, 4, true);
        near(m.acceptance()[1], old_second); // An accepted stop is not a rejection at the next position.
        const auto observation = m.observation();
        check(MeasuredObservation::parse(parse_json(observation.json().dump())) == observation);
        m.begin_generation(3, 4096, MeasuredSampling::greedy);
        m.observe(4096, 1, 0, 1.0 / 3.0, a); m.observe(4096, 1, 0, 71.12345678901234, a);
        const Json timing_ready(Json::Object{{"native_measured_controller", Json(true)}, {"native_measured_timing_schema", Json::integer(1)}});
        const auto timing_done = [&](Json packet) { return Json(Json::Object{{"native_measured_timing_schema", Json::integer(1)},
            {"native_measured_decode_observations", std::move(packet)},
            {"native_mtp_generation", Json(Json::Object{{"stage_a_controller", Json(Json::Object{{"measured", m.observation().json()}})}})}}); };
        const auto complete_times = m.timing_packet();
        validate_measured_timing_packet(timing_ready, Json(), timing_done(complete_times), true, 3, 2);
        refused([&] { validate_measured_timing_packet(timing_ready, Json(), timing_done(complete_times), true, 3, 1); });
        refused([&] { validate_measured_timing_packet(timing_ready, Json(), timing_done(parse_json(R"({"first_action":2,"milliseconds":[0.33333333333333331,71.12345678901234]})")), true, 3, 2); });
        refused([&] { validate_measured_timing_packet(timing_ready, Json(), timing_done(complete_times), true, 4, 2); });
        refused([&] { validate_measured_timing_packet(Json(Json::Object{}), Json(), timing_done(complete_times), true, 3, 2); });
        const auto previous_times = timing_done(complete_times);
        m.begin_packet(); m.observe(4096, 1, 0, 12, a);
        validate_measured_timing_packet(timing_ready, previous_times, timing_done(m.timing_packet()), false, 3, 1);
        m.begin_packet();
        validate_measured_timing_packet(timing_ready, previous_times, timing_done(m.timing_packet()), false, 3, 0);
        for (uint32_t i = 0; i < 64; ++i) m.observe(4096, 1, 0, 12, a);
        refused([&] { m.observe(4096, 1, 0, 12, a); });
        check(m.observation().actions == 67);
        m.begin_packet(); m.observe(4096, 1, 0, 12, a);
        // A deliberately hostile deepest-depth prior/cost cannot permanently starve its estimate.
        const auto hostile = profile(10000); MeasuredController h(hostile);
        h.begin_generation(1, 4096, MeasuredSampling::greedy);
        for (unsigned i = 0; i < 16; ++i) {
            check(!h.before_draft(4096, 4).probe); h.observe(4096, 2, 0, 13, a);
        }
        choice = h.before_draft(4096, 4); check(choice.probe && choice.rows == 4);
        h.observe(4096, 4, 3, 16, a); check(!h.before_draft(4096, 4).probe);
        check(p.policy_digest() == profile().policy_digest()); check(p.policy_digest() != hostile.policy_digest());
        Json::Array comparisons;
        const Json arm(Json::Object{{"initial_state", Json(std::string(64, 'b'))}, {"complete_logits", Json(std::string(64, 'c'))}, {"accepted_state", Json(std::string(64, 'd'))},
            {"sampling_distribution", Json(std::string(64, 'e'))}, {"routes", Json(std::string(64, 'f'))}});
        for (const uint32_t context : {4096u, 4352u}) for (uint32_t rows = 2; rows <= 4; ++rows) for (uint32_t prefix = 1; prefix <= rows; ++prefix)
            comparisons.push_back(Json(Json::Object{{"context", Json::integer(context)}, {"rows", Json::integer(rows)}, {"accepted_prefix", Json::integer(prefix)},
                {"sampling", Json("greedy")}, {"capture_ids", Json(Json::Object{{"ordinary", Json(std::string(32, '1'))}, {"verification", Json(std::string(32, '2'))}})},
                {"source_identity", Json(Json::Object{{"worker_sha256", Json(std::string(64, 'a'))}, {"library_sha256", Json(std::string(64, 'b'))}, {"capture_sha256", Json(std::string(64, 'c'))}, {"source_revision", Json(std::string(40, 'd'))}})},
                {"artifact_sizes", Json(Json::Object{{"ordinary", Json(Json::Object{{"initial_state", Json::integer(256)}, {"complete_logits", Json::integer(248320 * 4)}, {"accepted_state", Json::integer(256)}, {"sampling_distribution", Json::integer(20)}, {"routes", Json::integer(rows * 4)}})}, {"verification", Json(Json::Object{{"initial_state", Json::integer(256)}, {"complete_logits", Json::integer(248320 * 4)}, {"accepted_state", Json::integer(256)}, {"sampling_distribution", Json::integer(20)}, {"routes", Json::integer(rows * 4)}})}})},
                {"ordinary", arm}, {"verification", arm}}));
        const auto receipt = [&](Json::Array cases) { return Json(Json::Object{{"schema", Json::integer(3)}, {"runtime_base", Json("tokspur-execution-v3")},
            {"execution_digest", Json(std::string(64, 'a'))}, {"comparisons", Json(std::move(cases))}}); };
        auto good = receipt(comparisons); const auto proof_hash = derived_pack::hash(good.dump()); const auto admitted = profile(0, proof_hash);
        admitted.admit_exactness(good, proof_hash); check(admitted.policy_digest() == p.policy_digest());
        refused([&] { coding.admit_exactness(good, std::string(64, 'b')); });
        auto coding_comparisons = comparisons;
        for (auto& item : coding_comparisons) {
            auto fields = item.object(); fields[3].second = Json("coding"); item = Json(std::move(fields));
        }
        const auto coding_receipt = receipt(coding_comparisons);
        const auto coding_hash = derived_pack::hash(coding_receipt.dump());
        profile(0, coding_hash, "coding").admit_exactness(coding_receipt, coding_hash);
        refused([&] { topk.admit_exactness(coding_receipt, std::string(64, 'b')); });
        refused([&] { admitted.admit_exactness(good, std::string(64, '0')); });
        auto missing = comparisons; missing.pop_back();
        refused([&] { admitted.admit_exactness(receipt(missing), proof_hash); });
        auto duplicate = comparisons; duplicate.push_back(comparisons.front());
        refused([&] { admitted.admit_exactness(receipt(duplicate), proof_hash); });
        for (size_t field_index : {0u, 1u}) {
            auto bad_case = comparisons.front().object(); auto bad_arm = arm.object(); bad_arm[field_index].second = Json(std::string(64, '0'));
            bad_case.back().second = Json(std::move(bad_arm)); auto mismatch = comparisons; mismatch[0] = Json(std::move(bad_case));
            refused([&] { admitted.admit_exactness(receipt(mismatch), proof_hash); });
        }
        auto legacy_proof = good.object(); legacy_proof[0].second = Json::integer(1);
        refused([&] { admitted.admit_exactness(Json(legacy_proof), proof_hash); });
        // A non-causal truncation can bias even byte-identical target rows. Reusing the discarded proposal's uniform does not repair it when target and proposal CDF orders differ.
        const auto target = sampling_distribution(std::array<double, 2>{std::log(.1), std::log(.9)}, {1, 0, 1, 0});
        uint32_t ordinary_zero = 0, noncausal_zero = 0;
        for (uint32_t i = 0; i < 16384; ++i) {
            const double unit = double(i) / 16384;
            const uint32_t proposal = unit < .9 ? 0 : 1;
            ordinary_zero += sample_at(target, unit) == 0;
            // Discard q0, but include q1, whose p1/q1 >= 1 acceptance is certain.
            noncausal_zero += (proposal == 0 ? sample_at(target, unit) : proposal) == 0;
        }
        check(ordinary_zero > 1600 && noncausal_zero == 0);
        const auto d = MeasuredCostProfile::parse(parse_json(std::string(R"({"schema":1,"drafter":"dflash2","maximum_rows":3,"execution_digest":")") + std::string(64, 'a') +
            R"(","exactness_receipt":")" + std::string(64, 'b') + R"(","modes":[{"sampling":"greedy","anchors":[{"context":4096,"ordinary_ms":10,"draft_ms":[0,2,2],"remaining_ms":[[10],[13,13],[12,12,12]],"conditional_acceptance":[0.2,0.01]}]}]})"));
        MeasuredController causal(d); causal.set_ordinary_allowed(false); causal.begin_generation(1, 4096, MeasuredSampling::greedy);
        AcceptanceRates low{.15, .01}, high{.15, 1};
        check(causal.after_draft(4096, 3, low, 3).rows == 1);
        check(causal.after_draft(4096, 3, high, 3).rows == 3);
        check(causal.after_draft_causal(4096, 3, low, 3).rows == 1);
        check(causal.after_draft_causal(4096, 3, high, 3).rows == 1);
        MtpGenerationCarry live; auto active_policy = legacy_checkpoint_identity_tag; active_policy[0] ^= 0x55;
        live.configure_controller_policy(active_policy);
        refused([&] { live.admit({.id = 44, .remaining = 16, .begin = true}, 1, 100, {}); });
        live.admit({.id = 44, .remaining = 16, .begin = true, .remaining_compute = 16, .total_budget = 16, .v2 = true}, 1, 100, {});
        auto live_observation = observation; live_observation.generation = 44;
        live.controller_state().measured = live_observation;
        const std::array<uint32_t, 2> ids{5, 9}; const std::array<float, 2> scores{.5f, .25f};
        live.commit_round(2, ids, scores, 100, 102, 1, false);
        check(!live.finish(1, 102, false) && live.pending_overflow() == 1);
        MtpGenerationCommand continuation{.id = 44, .remaining = 15, .begin = false, .committed = 2, .emitted = 1, .pending = 1, .remaining_compute = 14, .total_budget = 16, .v2 = true};
        refused([&] { live.admit(continuation, 1, 102, {}); });
        continuation.stage_a_controller = live.controller_state();
        continuation.stage_a_controller->measured->cost_multiplier += .1;
        refused([&] { live.admit(continuation, 1, 102, {}); });
        refused([&] { live.configure_controller_policy(legacy_checkpoint_identity_tag); });
        check(live.pending_overflow() == 1 && live.committed() == 2 && live.emitted() == 1 && live.controller_state().measured == live_observation);
        continuation.stage_a_controller = live.controller_state();
        live.admit(continuation, 1, 102, {});
        const auto overflow = live.pop_overflow(); check(overflow.id == 9 && overflow.score == .25f && overflow.token_prefix == 102);
        check(!live.finish(1, 102, false));
        static_cast<void>(live.abort({.id = 44, .abort = true}));
        check(!live.active() && !live.pending_overflow());
        live.configure_controller_policy(legacy_checkpoint_identity_tag);
        std::cout << "measured controller checks passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
