#pragma once

#include "tokspur/sampled_path.hpp"

namespace tokspur {

// Actual depth-one proposal law. The target policy remains unchanged. Only the
// auxiliary drafter is restricted to at most20 outcomes, exactly represented on
// the same U53 CDF grid as the already qualified standalone rejection component.
struct MtpDraftDraw {
    SampledSparseLaw law;
    uint32_t token;
    uint64_t proposal_begin_words;
    SamplingRng random_after;
};

[[nodiscard]] MtpDraftDraw draw_mtp_depth1(std::span<const float> draft_logits,
                                         SamplingPolicy target_policy,
                                         const SamplingRng& after_anchor,
                                         uint32_t target_vocabulary = 0);

[[nodiscard]] MtpDraftDraw draw_mtp_depth1_topk(const GpuLogitTopKRow& draft_topk,
                                              SamplingPolicy target_policy,
                                              const SamplingRng& after_anchor,
                                              uint32_t target_vocabulary = 0);

[[nodiscard]] MtpDraftDraw draw_deterministic_proposal(uint32_t proposed_token,
                                                       const SamplingRng& after_previous,
                                                       uint32_t target_vocabulary);

// The anchor was independently sampled from the retained target distribution.
// p1/p2 are the target rows after [anchor,draft]. The result is a correction, or
// an accepted draft followed by a target bonus. The first serving bridge must
// physically materialise that terminal correction/bonus before publication;
// there is no unconsumed visible token or discarded terminal entropy draw.
[[nodiscard]] SampledPathDecision decide_mtp_depth1(
    uint32_t anchor_prefix, uint32_t anchor, const MtpDraftDraw& draft,
    std::span<const float> p1, std::span<const float> p2,
    SamplingPolicy target_policy, std::span<const uint32_t> stops);

[[nodiscard]] SampledPathDecision decide_mtp_depth1_topk(
    uint32_t anchor_prefix, uint32_t anchor, const MtpDraftDraw& draft,
    const GpuLogitTopKRow& p1, const GpuLogitTopKRow& p2,
    SamplingPolicy target_policy, std::span<const uint32_t> stops);

#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
// Two actual conditional draws. All proposal entropy, including an unused tail
// after a stop/rejection, precedes the unchanged rejection-sampling decision.
[[nodiscard]] SampledPathDecision decide_mtp_depth2(
    uint32_t anchor_prefix, uint32_t anchor, const MtpDraftDraw& first, const MtpDraftDraw& second,
    std::span<const float> p1, std::span<const float> p2, std::span<const float> p3,
    SamplingPolicy target_policy, std::span<const uint32_t> stops);

[[nodiscard]] SampledPathDecision decide_mtp_depth2_topk(
    uint32_t anchor_prefix, uint32_t anchor, const MtpDraftDraw& first, const MtpDraftDraw& second,
    const GpuLogitTopKRow& p1, const GpuLogitTopKRow& p2, const GpuLogitTopKRow& p3,
    SamplingPolicy target_policy, std::span<const uint32_t> stops);

[[nodiscard]] SampledPathDecision decide_mtp_depth3(
    uint32_t anchor_prefix, uint32_t anchor, const MtpDraftDraw& first, const MtpDraftDraw& second,
    const MtpDraftDraw& third, std::span<const float> p1, std::span<const float> p2,
    std::span<const float> p3, std::span<const float> p4,
    SamplingPolicy target_policy, std::span<const uint32_t> stops);

[[nodiscard]] SampledPathDecision decide_mtp_depth3_topk(
    uint32_t anchor_prefix, uint32_t anchor, const MtpDraftDraw& first, const MtpDraftDraw& second,
    const MtpDraftDraw& third, const GpuLogitTopKRow& p1, const GpuLogitTopKRow& p2,
    const GpuLogitTopKRow& p3, const GpuLogitTopKRow& p4,
    SamplingPolicy target_policy, std::span<const uint32_t> stops);
#endif

} // namespace tokspur
