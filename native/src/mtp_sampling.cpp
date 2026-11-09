#include "tokspur/mtp_sampling.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tokspur {

MtpDraftDraw draw_mtp_depth1(std::span<const float> logits, SamplingPolicy policy,
                            const SamplingRng& after_anchor, uint32_t target_vocabulary) {
    policy.top_k = policy.top_k ? std::min(policy.top_k, sampled_path_maximum_support) : sampled_path_maximum_support;
    const auto distribution = sampling_distribution(logits, policy);
#ifndef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    std::vector<double> probabilities;
    probabilities.reserve(distribution.support().size());
    for (const auto id : distribution.support()) probabilities.push_back(distribution.probability(id));
#endif
    const auto vocab = target_vocabulary ? target_vocabulary : uint32_t(distribution.vocabulary());
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    auto law = SampledSparseLaw::from_masses(vocab, distribution.support(), distribution.support_probabilities());
#else
    auto law = SampledSparseLaw::from_masses(vocab, distribution.support(), probabilities);
#endif
    auto random = after_anchor;
    const auto begin = random.words_consumed();
    const auto ticket = law.deterministic() ? uint64_t(0) : uint64_t(std::ldexp(random.uniform(), 53));
    const auto token = law.sample_at(ticket);
    return {std::move(law), token, begin, std::move(random)};
}

MtpDraftDraw draw_mtp_depth1_topk(const GpuLogitTopKRow& draft_topk, SamplingPolicy policy,
                                  const SamplingRng& after_anchor, uint32_t target_vocabulary) {
    policy.top_k = policy.top_k ? std::min(policy.top_k, sampled_path_maximum_support) : sampled_path_maximum_support;
    const auto distribution = sampling_distribution_from_topk(draft_topk, policy);
    const auto vocab = target_vocabulary ? target_vocabulary : uint32_t(distribution.vocabulary());
    auto law = SampledSparseLaw::from_masses(vocab, distribution.support(), distribution.support_probabilities());
    auto random = after_anchor;
    const auto begin = random.words_consumed();
    const auto ticket = law.deterministic() ? uint64_t(0) : uint64_t(std::ldexp(random.uniform(), 53));
    const auto token = law.sample_at(ticket);
    return {std::move(law), token, begin, std::move(random)};
}

MtpDraftDraw draw_deterministic_proposal(uint32_t proposed_token,
                                         const SamplingRng& after_previous,
                                         uint32_t target_vocabulary) {
    if (!target_vocabulary || proposed_token >= target_vocabulary)
        throw FormatError("deterministic proposal token outside target vocabulary");
    const uint64_t unit = sampled_path_grid_size;
    auto law = SampledSparseLaw::from_grid_units(target_vocabulary, std::span(&proposed_token, 1), std::span(&unit, 1));
    const auto begin = after_previous.words_consumed();
    return {std::move(law), proposed_token, begin, after_previous};
}

SampledPathDecision decide_mtp_depth1(uint32_t prefix, uint32_t anchor, const MtpDraftDraw& draft,
    std::span<const float> p1, std::span<const float> p2, SamplingPolicy policy,
    std::span<const uint32_t> stops) {
    // Fixed process-local identity is scoped to this synchronous, non-retained
    // call. The stack adapter independently binds the actual prefix and rows.
    const SampledPathIdentity identity{1, 1, 1, prefix, uint32_t(p1.size()), anchor};
    const SampledProposalRow proposal{prefix, anchor, draft.token, draft.law};
    const std::array<SampledTargetRow, 2> rows{{{prefix, p1}, {prefix + 1, p2}}};
    return decide_sampled_path(identity, std::span(&proposal, 1), identity, rows, policy,
        2, stops, draft.random_after, draft.proposal_begin_words);
}

SampledPathDecision decide_mtp_depth1_topk(uint32_t prefix, uint32_t anchor, const MtpDraftDraw& draft,
    const GpuLogitTopKRow& p1, const GpuLogitTopKRow& p2, SamplingPolicy policy,
    std::span<const uint32_t> stops) {
    const SampledPathIdentity identity{1, 1, 1, prefix, p1.vocabulary, anchor};
    const SampledProposalRow proposal{prefix, anchor, draft.token, draft.law};
    const std::array<SampledTargetTopKRow, 2> rows{{{prefix, p1}, {prefix + 1, p2}}};
    return decide_sampled_path_topk(identity, std::span(&proposal, 1), identity, rows, policy,
        2, stops, draft.random_after, draft.proposal_begin_words);
}

#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
SampledPathDecision decide_mtp_depth2(uint32_t prefix, uint32_t anchor,
    const MtpDraftDraw& first, const MtpDraftDraw& second,
    std::span<const float> p1, std::span<const float> p2, std::span<const float> p3,
    SamplingPolicy policy, std::span<const uint32_t> stops) {
    if (second.proposal_begin_words != first.random_after.words_consumed())
        throw FormatError("depth-two draft entropy is not contiguous");
    const SampledPathIdentity identity{1, 1, 1, prefix, uint32_t(p1.size()), anchor};
    const std::array<SampledProposalRow, 2> proposals{{{prefix, anchor, first.token, first.law},
        {prefix + 1, first.token, second.token, second.law}}};
    const std::array<SampledTargetRow, 3> rows{{{prefix, p1}, {prefix + 1, p2}, {prefix + 2, p3}}};
    return decide_sampled_path(identity, proposals, identity, rows, policy, 3, stops,
        second.random_after, first.proposal_begin_words);
}

SampledPathDecision decide_mtp_depth2_topk(uint32_t prefix, uint32_t anchor,
    const MtpDraftDraw& first, const MtpDraftDraw& second,
    const GpuLogitTopKRow& p1, const GpuLogitTopKRow& p2, const GpuLogitTopKRow& p3,
    SamplingPolicy policy, std::span<const uint32_t> stops) {
    if (second.proposal_begin_words != first.random_after.words_consumed())
        throw FormatError("depth-two draft entropy is not contiguous");
    const SampledPathIdentity identity{1, 1, 1, prefix, p1.vocabulary, anchor};
    const std::array<SampledProposalRow, 2> proposals{{{prefix, anchor, first.token, first.law},
        {prefix + 1, first.token, second.token, second.law}}};
    const std::array<SampledTargetTopKRow, 3> rows{{{prefix, p1}, {prefix + 1, p2}, {prefix + 2, p3}}};
    return decide_sampled_path_topk(identity, proposals, identity, rows, policy, 3, stops,
        second.random_after, first.proposal_begin_words);
}

SampledPathDecision decide_mtp_depth3(uint32_t prefix, uint32_t anchor,
    const MtpDraftDraw& first, const MtpDraftDraw& second, const MtpDraftDraw& third,
    std::span<const float> p1, std::span<const float> p2, std::span<const float> p3, std::span<const float> p4,
    SamplingPolicy policy, std::span<const uint32_t> stops) {
    if (second.proposal_begin_words != first.random_after.words_consumed() ||
        third.proposal_begin_words != second.random_after.words_consumed())
        throw FormatError("depth-three draft entropy is not contiguous");
    const SampledPathIdentity identity{1, 1, 1, prefix, uint32_t(p1.size()), anchor};
    const std::array<SampledProposalRow, 3> proposals{{{prefix, anchor, first.token, first.law},
        {prefix + 1, first.token, second.token, second.law},
        {prefix + 2, second.token, third.token, third.law}}};
    const std::array<SampledTargetRow, 4> rows{{{prefix, p1}, {prefix + 1, p2}, {prefix + 2, p3}, {prefix + 3, p4}}};
    return decide_sampled_path(identity, proposals, identity, rows, policy, 4, stops,
        third.random_after, first.proposal_begin_words);
}

SampledPathDecision decide_mtp_depth3_topk(uint32_t prefix, uint32_t anchor,
    const MtpDraftDraw& first, const MtpDraftDraw& second, const MtpDraftDraw& third,
    const GpuLogitTopKRow& p1, const GpuLogitTopKRow& p2, const GpuLogitTopKRow& p3, const GpuLogitTopKRow& p4,
    SamplingPolicy policy, std::span<const uint32_t> stops) {
    if (second.proposal_begin_words != first.random_after.words_consumed() ||
        third.proposal_begin_words != second.random_after.words_consumed())
        throw FormatError("depth-three draft entropy is not contiguous");
    const SampledPathIdentity identity{1, 1, 1, prefix, p1.vocabulary, anchor};
    const std::array<SampledProposalRow, 3> proposals{{{prefix, anchor, first.token, first.law},
        {prefix + 1, first.token, second.token, second.law},
        {prefix + 2, second.token, third.token, third.law}}};
    const std::array<SampledTargetTopKRow, 4> rows{{{prefix, p1}, {prefix + 1, p2}, {prefix + 2, p3}, {prefix + 3, p4}}};
    return decide_sampled_path_topk(identity, proposals, identity, rows, policy, 4, stops,
        third.random_after, first.proposal_begin_words);
}
#endif

} // namespace tokspur
