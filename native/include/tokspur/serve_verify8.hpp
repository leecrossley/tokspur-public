#pragma once

#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#if defined(TOKSPUR_PRIVATE_HTTP_BRANCH8) && !defined(TOKSPUR_PRIVATE_VERIFY8_BRANCH)
#error The branch8 HTTP wire requires the separately consented branch verifier.
#endif
#include "tokspur/json.hpp"

#include <array>
#include <optional>
#include <span>

namespace tokspur {
// Private CPU-only admission. No target, tokenizer, IO or model work occurs here.
void validate_verify8_ready(const Json&);
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
std::optional<bool> verify8_fast_math(const Json&); // Absent or explicit staged(false)/channel(true).
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
// Reuse only math/resource checks; no greedy verifier or branch claims.
void validate_scalar_channel_ready(const Json&);
void validate_scalar_channel_done(const Json& ready, const Json& previous,
                                  const Json& job, const Json& done, const Json* token);
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
// Distinct immutable known-input8/scalar-output transport, never the greedy V8 wire.
void validate_sampled_prefill_ready(const Json&);
void validate_sampled_prefill_done(const Json& ready, const Json& previous,
                                 const Json& job, const Json& done, const Json* token);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
// New immutable wire, never admitted by validate_verify8_ready. All model state
// below is produced afresh by native execution; a control is not a fake done.
void validate_branch8_ready(const Json&);
const Json& branch8_session(const Json& reply);
void validate_branch8_control(const Json& ready, const Json& previous, const Json& job,
                              const Json& reply, bool cold_resync = false);
#endif
class Verify8Reply {
public:
    Verify8Reply(const Json& ready, const Json& previous, const Json& job,
                 uint32_t before, const Json& sampling, uint64_t seed,
                 bool reset_resync = false
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                 , bool branch8 = false
#endif
                 );
    void event(const Json&); // Ordered round then tokens; buffers at most 16 records.
    void finish(const Json&); // Must succeed before events() may be published.
    [[nodiscard]] std::span<const Json> events() const;
private:
    const Json &ready_, &previous_, &job_, &sampling_;
    uint64_t seed_;
    uint32_t base_, position_, inputs_, maximum_;
    std::array<Json, 16> events_{};
    std::array<uint32_t, 8> ids_{};
    std::array<uint64_t, 5> routes_{};
    size_t event_count_ = 0, count_ = 0;
    uint32_t rounds_ = 0, pending_ = 0, round_count_ = 0, accepted_ = 0;
    uint32_t evaluated_total_ = 0, proposed_total_ = 0, ordinary_total_ = 0, recovery_rounds_ = 0, verified_rounds_ = 0;
    uint64_t dispatches_ = 0;
    double round_wall_ = 0, batch_wall_ = 0, batch_gpu_ = 0;
    bool reset_, reset_resync_, stopped_ = false, finished_ = false;
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    const bool branch8_;
    const uint64_t branch_epoch_, branch_command_, branch_lease_;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    bool stage8_prepared_ = false, stage8_enabled_ = false;
    uint64_t stage8_before_ = 0;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    bool fast_math_prepared_ = false, channel_enabled_ = false;
    uint64_t channel_before_ = 0;
#endif
};
} // namespace tokspur
#endif
