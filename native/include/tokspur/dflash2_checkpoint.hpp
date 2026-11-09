#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

namespace tokspur {

// A prefill packet may publish an intermediate shared-prefix checkpoint before its final sliding window is complete.
[[nodiscard]] inline uint64_t dflash2_prefill_remaining_after(uint64_t remaining_after, bool shared_prefix_cache,
    uint64_t end_position, uint64_t max_prefix_tokens) noexcept {
#if defined(TOKSPUR_PRIVATE_DFLASH2_CACHE_WORK) && TOKSPUR_PRIVATE_DFLASH2_CACHE_WORK
    return shared_prefix_cache && end_position <= max_prefix_tokens ? 0 : remaining_after;
#else
    (void)end_position; (void)max_prefix_tokens;
    return shared_prefix_cache ? 0 : remaining_after;
#endif
}

[[nodiscard]] inline bool dflash2_checkpoint_identity_required(bool checkpoint_ssd) noexcept {
#if defined(TOKSPUR_PRIVATE_DFLASH2_CACHE_WORK) && TOKSPUR_PRIVATE_DFLASH2_CACHE_WORK
    return checkpoint_ssd;
#else
    (void)checkpoint_ssd;
    return true;
#endif
}

// RAM anchors stay within the owner of the loaded weights. Persistent checkpoints require the full configuration and weight identity.
template<class Hash>
[[nodiscard]] std::string dflash2_checkpoint_identity(bool enabled, std::string_view config_sha256,
    std::string_view weights, Hash&& hash) {
    if (!enabled) return {};
    return hash("dflash2-context-v1:" + std::string(config_sha256) + ":" + hash(weights));
}

struct DFlash2ContextCommitPlan {
    uint32_t keep_from = 0;
    bool discard_pending = false;
};

[[nodiscard]] inline DFlash2ContextCommitPlan dflash2_context_commit_plan(uint32_t rows, uint64_t remaining_after, uint32_t window) noexcept {
    const uint64_t total_future = uint64_t(rows) + remaining_after;
    return {total_future > window ? uint32_t(std::min<uint64_t>(rows, total_future - window)) : 0u, total_future >= window};
}

// Store valid ring rows in chronological order. Each layer contributes keys followed by values; unused and overwritten physical rows are never checkpoint state.
struct DFlash2CheckpointLayout {
    uint32_t first = 0, rows = 0, window = 2048, layers = 5, heads = 8, width = 128;

    [[nodiscard]] bool valid() const noexcept {
        return window && layers && heads && width && first <= 65536 && rows <= 65536 - first &&
            rows == std::min(first + rows, window) && layers <= 64 && heads <= 128 && width <= 1024;
    }
    [[nodiscard]] uint64_t bytes() const noexcept { return uint64_t(rows) * heads * width * 2 * 2 * layers; }

    template<class Copy>
    bool visit(uint64_t offset, uint64_t length, Copy&& copy) const {
        if (!valid() || offset > bytes() || length > bytes() - offset) return false;
        if (!length) return true;
        const uint64_t plane = uint64_t(rows) * heads * width * 2;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (uint32_t value = 0; value < 2; ++value) {
                const uint64_t base = (uint64_t(layer) * 2 + value) * plane;
                // Metal stores K as [head, slot, channel] and V as [head, channel, slot]. Preserve each layout while removing unused slots and unwrapping time.
                const uint32_t streams = value ? heads * width : heads;
                const uint64_t row_bytes = value ? 2 : uint64_t(width) * 2;
                const uint64_t first_bytes = uint64_t(std::min(rows, window - first % window)) * row_bytes;
                for (uint32_t stream = 0; stream < streams; ++stream) for (uint32_t part = 0; part < 2; ++part) {
                    const uint64_t begin = base + uint64_t(stream) * rows * row_bytes + (part ? first_bytes : 0);
                    const uint64_t end = begin + (part ? uint64_t(rows) * row_bytes - first_bytes : first_bytes);
                    const uint64_t lo = std::max(begin, offset), hi = std::min(end, offset + length);
                    const uint64_t physical = uint64_t(stream) * window * row_bytes + (part ? 0 : uint64_t(first % window) * row_bytes);
                    if (lo < hi && !copy(layer, value != 0, physical + lo - begin, lo - offset, hi - lo)) return false;
                }
            }
        }
        return true;
    }
};

} // namespace tokspur
