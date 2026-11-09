#pragma once
#include "tokspur/gguf.hpp"
#include <cstdint>

namespace tokspur {
inline constexpr uint32_t mtp_profile_max_rounds = 4;
inline constexpr uint32_t mtp_profile_max_batches_per_round = 3;
inline constexpr uint32_t mtp_profile_dispatch_capacity = 8192;
enum class MtpProfileBoundary : uint32_t { dispatch, encoder };
inline constexpr bool mtp_profile_boundary_supported(MtpProfileBoundary boundary, bool dispatch, bool encoder) {
    return boundary == MtpProfileBoundary::dispatch ? dispatch : boundary == MtpProfileBoundary::encoder && encoder;
}
inline const char* mtp_profile_boundary_name(MtpProfileBoundary boundary) {
    return boundary == MtpProfileBoundary::dispatch ? "dispatch_boundary" : "encoder_boundary";
}
enum class MtpProfilePhase : uint32_t { none, draft, verification, recovery, prime };
inline const char* mtp_profile_phase_name(MtpProfilePhase phase) {
    switch (phase) {
    case MtpProfilePhase::draft: return "draft";
    case MtpProfilePhase::verification: return "verification";
    case MtpProfilePhase::recovery: return "recovery";
    case MtpProfilePhase::prime: return "prime";
    default: return "none";
    }
}

// Lifetime budget, deliberately not reset by a request, prompt reset or EOS.
// Only the caller's actual MTP generation branch may open a window.
struct MtpProfileWindow {
    uint32_t requested = 0, rounds = 0, batches = 0, round_batches = 0;
    uint32_t base = 0;
    bool active = false;
    MtpProfilePhase phase = MtpProfilePhase::none;
    void prepare(uint32_t count) {
        if (requested || !count || count > mtp_profile_max_rounds)
            throw FormatError("MTP profile requires one immutable lifetime budget of 1..4 rounds");
        requested = count;
    }
    bool begin(uint32_t prefix) {
        if (active) throw FormatError("MTP profile round is already active");
        if (!requested || rounds == requested) return false;
        ++rounds; base = prefix; round_batches = 0; phase = MtpProfilePhase::none; active = true;
        return true;
    }
    bool next_batch_allowed() const noexcept {
        return active && ((round_batches == 0 && phase == MtpProfilePhase::draft) ||
            (round_batches == 1 && phase == MtpProfilePhase::verification) ||
            (round_batches == 2 && (phase == MtpProfilePhase::recovery || phase == MtpProfilePhase::prime)));
    }
    void submitted() {
        if (!next_batch_allowed()) throw FormatError("MTP profile batch is outside its bounded ordered generation phase");
        ++round_batches; ++batches;
    }
    void end() noexcept { active = false; phase = MtpProfilePhase::none; }
};
} // namespace tokspur
