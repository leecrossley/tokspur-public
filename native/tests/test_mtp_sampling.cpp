#include "tokspur/mtp_sampling.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
size_t checks = 0;
void check(bool value) { ++checks; if (!value) throw std::runtime_error("MTP sampling bridge check failed"); }
}

int main() {
    using namespace tokspur;
    try {
        // Greedy acceptance includes the actual target bonus, not a resampled
        // or silently discarded token. Point masses consume no RNG words.
        const std::array<float, 3> q{0, 3, -2}, bonus{4, 0, -2};
        const SamplingRng random(44);
        const auto yes = draw_mtp_depth1(q, {}, random);
        const auto accepted = decide_mtp_depth1(17, 2, yes, q, bonus, {}, {});
        check(yes.token == 1 && yes.random_after == random);
        check(accepted.accepted == 1 && accepted.count == 2);
        check(accepted.tokens[0].id == 1 && accepted.tokens[0].accepted_proposal);
        check(accepted.tokens[1].id == 0 && !accepted.tokens[1].accepted_proposal);
        check(accepted.random_after == random);
        // Rejection uses the complete positive residual and commits no draft.
        const auto rejected = decide_mtp_depth1(17, 2, yes, bonus, q, {}, {});
        check(rejected.accepted == 0 && rejected.count == 1);
        check(rejected.tokens[0].id == 0 && rejected.terminal == SampledPathTerminal::correction);
        check(rejected.random_after == random);
        const std::array<uint32_t, 1> stops{1};
        const auto stopped = decide_mtp_depth1(17, 2, yes, q, bonus, {}, stops);
        check(stopped.accepted == 1 && stopped.count == 1 && stopped.stopped);
        // Production filtering: equal p/q must accept for every draw. Future
        // target bonus draws remain part of the returned transactional RNG.
        std::vector<float> logits(248320, -100);
        for (uint32_t i = 0; i < 20; ++i) logits[i] = float(20 - i) * 0.025f;
        const SamplingPolicy policy{1, 20, 0.95, 0};
        for (uint64_t seed = 0; seed < 32; ++seed) {
            const SamplingRng before(seed);
            const auto draft = draw_mtp_depth1(logits, policy, before);
            auto ordinary = before;
            check(draft.token == sample(sampling_distribution(logits, policy), ordinary));
            check(draft.random_after == ordinary && draft.law.size() <= 20);
            const auto decision = decide_mtp_depth1(3072, 248044, draft, logits, logits, policy, {});
            check(decision.accepted == 1 && decision.count == 2 && decision.tokens[0].id == draft.token);
            check(decision.random_after.words_consumed() > draft.random_after.words_consumed());
        }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        const std::array<float, 3> q2{0, -2, 4};
        const auto first = draw_mtp_depth1(q, {}, random);
        const auto second = draw_mtp_depth1(q2, {}, first.random_after);
        for (uint32_t k : {0u, 1u, 2u}) {
            const auto d = decide_mtp_depth2(17, 2, first, second, k ? std::span<const float>(q) : std::span<const float>(bonus),
                k == 2 ? std::span<const float>(q2) : std::span<const float>(bonus), bonus, {}, {});
            check(d.accepted == k && d.count == k + 1 && d.active_proposals == 2 && d.random_after == random);
            check(d.tokens[k].id == 0 && !d.tokens[k].accepted_proposal);
        }
        for (uint32_t stop : {0u, 1u, 2u}) {
            const std::array<uint32_t, 1> ids{stop};
            const auto d = decide_mtp_depth2(17, 2, first, second, q, q2, bonus, {}, ids);
            check(d.stopped && d.accepted == (stop == 1 ? 1 : 2));
            check(d.count == (stop ? d.accepted : 3) && d.tokens[d.count - 1].id == stop);
        }
        // The actual two proposal laws are conditional and consume consecutive
        // RNG words. Equal target/draft laws must accept both, not redraw them.
        const SamplingPolicy sampled{1, 3, 1, 0};
        const std::array<float, 3> a{0.2f, 0.5f, -0.1f}, b{-0.4f, 0.1f, 0.6f};
        uint32_t outcomes = 0;
        for (uint64_t seed = 0; seed < 128; ++seed) {
            const SamplingRng start(seed);
            const auto x = draw_mtp_depth1(a, sampled, start), y = draw_mtp_depth1(b, sampled, x.random_after);
            const auto d = decide_mtp_depth2(20, 0, x, y, a, b, a, sampled, {});
            check(d.accepted == 2 && d.tokens[0].id == x.token && d.tokens[1].id == y.token);
            check(d.proposal_words == 2 && d.random_after.words_consumed() == 3 && start.words_consumed() == 0);
            const auto mixed = decide_mtp_depth2(20,0,x,y,b,a,b,sampled,{});
            const SampledPathIdentity identity{1,1,1,20,3,0};
            const std::array<SampledProposalRow,2> proposals{{{20,0,x.token,x.law},{21,x.token,y.token,y.law}}};
            const std::array<SampledTargetRow,3> rows{{{20,b},{21,a},{22,b}}};
            const auto reference = decide_sampled_path(identity,proposals,identity,rows,sampled,3,{},y.random_after,x.proposal_begin_words);
            outcomes |= 1u << mixed.accepted;
            check(mixed.identity==reference.identity && mixed.count==reference.count && mixed.accepted==reference.accepted &&
                mixed.terminal==reference.terminal && mixed.random_after==reference.random_after);
            check(mixed.proposal_words==2 && mixed.acceptance_words==reference.acceptance_words && mixed.terminal_words==reference.terminal_words);
            for(uint32_t i=0;i<mixed.count;++i)check(mixed.tokens[i].id==reference.tokens[i].id &&
                mixed.tokens[i].accepted_proposal==reference.tokens[i].accepted_proposal && mixed.tokens[i].source_logit==reference.tokens[i].source_logit);
        }
        check(outcomes==7);
        auto stale=second;++stale.proposal_begin_words;bool refused=false;
        try { static_cast<void>(decide_mtp_depth2(17,2,first,stale,q,q2,bonus,{},{})); } catch(const FormatError&) { refused=true; }
        check(refused);
        uint32_t outcomes3 = 0;
        for (uint64_t seed = 0; seed < 128; ++seed) {
            const SamplingRng start3(seed);
            const auto x3 = draw_mtp_depth1(a, sampled, start3);
            const auto y3 = draw_mtp_depth1(b, sampled, x3.random_after);
            const auto z3 = draw_mtp_depth1(a, sampled, y3.random_after);
            const auto eq3 = decide_mtp_depth3(30, 0, x3, y3, z3, a, b, a, b, sampled, {});
            check(eq3.accepted == 3 && eq3.count == 4 && eq3.proposal_words == 3 && eq3.random_after.words_consumed() == 4);
            const auto mixed3 = decide_mtp_depth3(30, 0, x3, y3, z3, b, a, b, a, sampled, {});
            const SampledPathIdentity id3{1, 1, 1, 30, 3, 0};
            const std::array<SampledProposalRow, 3> props3{{{30, 0, x3.token, x3.law}, {31, x3.token, y3.token, y3.law}, {32, y3.token, z3.token, z3.law}}};
            const std::array<SampledTargetRow, 4> rows3{{{30, b}, {31, a}, {32, b}, {33, a}}};
            const auto ref3 = decide_sampled_path(id3, props3, id3, rows3, sampled, 4, {}, z3.random_after, x3.proposal_begin_words);
            outcomes3 |= 1u << mixed3.accepted;
            check(mixed3.count == ref3.count && mixed3.accepted == ref3.accepted && mixed3.terminal == ref3.terminal && mixed3.random_after == ref3.random_after);
            for (uint32_t i = 0; i < mixed3.count; ++i) check(mixed3.tokens[i].id == ref3.tokens[i].id && mixed3.tokens[i].accepted_proposal == ref3.tokens[i].accepted_proposal);
        }
        check(outcomes3 == 15);
        auto stale3 = draw_mtp_depth1(a, sampled, second.random_after); ++stale3.proposal_begin_words;
        bool refused3 = false;
        try { static_cast<void>(decide_mtp_depth3(17, 2, first, second, stale3, q, q2, bonus, q, {}, {})); } catch (const FormatError&) { refused3 = true; }
        check(refused3);
#endif
        std::vector<float> draft_slice(32768, -100.0f);
        draft_slice[10] = 5.0f; draft_slice[20] = 3.0f;
        const auto sliced_draw = draw_mtp_depth1(draft_slice, {}, random, 248320);
        check(sliced_draw.law.vocabulary() == 248320);
        check(sliced_draw.token == 10);
        std::vector<float> full_target1(248320, 0.0f), full_target2(248320, 0.0f);
        full_target1[10] = 10.0f; full_target2[100] = 10.0f;
        const auto sliced_decision = decide_mtp_depth1(50, 0, sliced_draw, full_target1, full_target2, {}, {});
        check(sliced_decision.accepted == 1 && sliced_decision.count == 2);
        // Deterministic lookup proposals consume 0 proposal RNG words and preserve exact target rejection sampling.
        {
            const SamplingRng base_rng(999);
            const auto det1 = draw_deterministic_proposal(1, base_rng, 3);
            check(det1.token == 1 && det1.law.vocabulary() == 3 && det1.law.size() == 1);
            check(det1.random_after == base_rng && det1.proposal_begin_words == base_rng.words_consumed());
            bool bad_det = false;
            try { static_cast<void>(draw_deterministic_proposal(3, base_rng, 3)); } catch (const FormatError&) { bad_det = true; }
            check(bad_det);
            const auto det_greedy_acc = decide_mtp_depth1(17, 2, det1, q, bonus, {}, {});
            check(det_greedy_acc.accepted == 1 && det_greedy_acc.count == 2 && det_greedy_acc.proposal_words == 0 && det_greedy_acc.random_after == base_rng);
            const auto det_greedy_rej = decide_mtp_depth1(17, 2, det1, bonus, q, {}, {});
            check(det_greedy_rej.accepted == 0 && det_greedy_rej.count == 1 && det_greedy_rej.tokens[0].id == 0 && det_greedy_rej.proposal_words == 0);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            const auto det2 = draw_deterministic_proposal(2, det1.random_after, 3);
            check(det2.random_after == base_rng && det2.proposal_begin_words == 0);
            const auto det_d2_greedy = decide_mtp_depth2(17, 2, det1, det2, q, q2, bonus, {}, {});
            check(det_d2_greedy.accepted == 2 && det_d2_greedy.count == 3 && det_d2_greedy.proposal_words == 0 && det_d2_greedy.random_after == base_rng);
            const SamplingPolicy sp{1, 3, 1, 0};
            uint32_t det_outcomes = 0;
            for (uint64_t seed = 0; seed < 64; ++seed) {
                const SamplingRng s_rng(seed);
                const auto p_first = draw_deterministic_proposal(1, s_rng, 3);
                const auto p_second = draw_deterministic_proposal(2, p_first.random_after, 3);
                const auto d = decide_mtp_depth2(20, 0, p_first, p_second, a, b, a, sp, {});
                check(d.proposal_words == 0 && s_rng.words_consumed() == 0);
                const SampledPathIdentity id{1, 1, 1, 20, 3, 0};
                const std::array<SampledProposalRow, 2> props{{{20, 0, 1, p_first.law}, {21, 1, 2, p_second.law}}};
                const std::array<SampledTargetRow, 3> t_rows{{{20, a}, {21, b}, {22, a}}};
                const auto ref = decide_sampled_path(id, props, id, t_rows, sp, 3, {}, p_second.random_after, p_first.proposal_begin_words);
                det_outcomes |= 1u << d.accepted;
                check(d.count == ref.count && d.accepted == ref.accepted && d.terminal == ref.terminal && d.random_after == ref.random_after);
                for (uint32_t i = 0; i < d.count; ++i) check(d.tokens[i].id == ref.tokens[i].id);
            }
            check(det_outcomes == 7);
#endif
        }
        // GPU top-k draft sampling parity across greedy, top-k, unbounded top-k=0, top-p, min-p, ties and draft slice.
        {
            const auto make_topk_row = [](std::span<const float> row) {
                GpuLogitTopKRow out{};
                out.vocabulary = uint32_t(row.size());
                out.first_nonfinite_col = 0xffffffffu;
                out.first_subnormal_col = 0xffffffffu;
                std::vector<uint32_t> order(row.size());
                for (uint32_t i = 0; i < row.size(); ++i) {
                    order[i] = i;
                    if (!std::isfinite(row[i]) && out.first_nonfinite_col == 0xffffffffu) out.first_nonfinite_col = i;
                    if (std::fpclassify(row[i]) == FP_SUBNORMAL && out.first_subnormal_col == 0xffffffffu) out.first_subnormal_col = i;
                }
                const uint32_t k = uint32_t(std::min<size_t>(row.size(), gpu_logit_topk_capacity));
                out.count = k;
                std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](uint32_t lhs, uint32_t rhs) {
                    if (row[lhs] != row[rhs]) return row[lhs] > row[rhs];
                    return lhs < rhs;
                });
                for (uint32_t i = 0; i < k; ++i) {
                    out.ids[i] = order[i];
                    out.logits[i] = row[order[i]];
                }
                return out;
            };
            std::vector<float> vocab_logits(4096, -50.0f);
            for (uint32_t i = 0; i < 64; ++i) vocab_logits[i * 31] = float(64 - int(i / 2)) * 0.125f;
            const auto topk_full = make_topk_row(vocab_logits);
            const std::array<SamplingPolicy, 6> policies{{
                {0.0, 0, 1.0, 0.0},
                {0.8, 1, 1.0, 0.0},
                {0.7, 5, 0.9, 0.0},
                {1.0, 20, 0.95, 0.02},
                {1.0, 0, 0.85, 0.05},
                {1.2, 64, 1.0, 0.0},
            }};
            for (const auto& pol : policies) {
                for (uint64_t seed = 0; seed < 32; ++seed) {
                    const SamplingRng rng(seed);
                    const auto ref = draw_mtp_depth1(vocab_logits, pol, rng, 248320);
                    const auto fast = draw_mtp_depth1_topk(topk_full, pol, rng, 248320);
                    check(fast.token == ref.token && fast.random_after == ref.random_after);
                    check(fast.proposal_begin_words == ref.proposal_begin_words);
                    check(fast.law.vocabulary() == 248320 && fast.law.size() == ref.law.size());
                    for (size_t s = 0; s < ref.law.size(); ++s) {
                        check(fast.law.identifiers()[s] == ref.law.identifiers()[s]);
                        check(fast.law.mass_units()[s] == ref.law.mass_units()[s]);
                    }
                }
            }
            auto bad_topk = topk_full;
            bad_topk.first_nonfinite_col = 7;
            bool rejected_nonfinite = false;
            try { static_cast<void>(draw_mtp_depth1_topk(bad_topk, {}, SamplingRng(1), 248320)); }
            catch (const FormatError&) { rejected_nonfinite = true; }
            check(rejected_nonfinite);
        }
        std::cout << "MTP sampling bridge checks passed: " << checks << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
