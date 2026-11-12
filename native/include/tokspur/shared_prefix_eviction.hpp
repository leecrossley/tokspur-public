#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tokspur {

// The shared-prefix anchor victim choice, kept pure so a fixture test can pin it.
// `lru` is the only policy and is the default route's rule: the least recently used anchor
// that is not a strict prefix of the prefix being captured; when every anchor is such a
// prefix, the least recently used anchor overall. Ties keep the lowest index.
enum class SharedPrefixEvictionPolicy : uint8_t { lru };

// Why a capture had to evict: the byte cap, the anchor count cap, or both at once.
enum class SharedPrefixEvictionCause : uint8_t { bytes, count, both };

inline SharedPrefixEvictionCause shared_prefix_eviction_cause(bool over_bytes, bool over_count) {
    if (over_bytes && over_count) return SharedPrefixEvictionCause::both;
    return over_bytes ? SharedPrefixEvictionCause::bytes : SharedPrefixEvictionCause::count;
}

struct SharedPrefixEvictionCounters {
    uint64_t bytes = 0, count = 0, both = 0;
    void record(SharedPrefixEvictionCause cause) {
        switch (cause) {
        case SharedPrefixEvictionCause::bytes: ++bytes; break;
        case SharedPrefixEvictionCause::count: ++count; break;
        case SharedPrefixEvictionCause::both: ++both; break;
        }
    }
};

// `anchors` must not be empty. `tokens_of` returns a contiguous range of uint32_t token IDs
// and `tick_of` the anchor's last access tick.
template<class Anchors, class TokensOf, class TickOf>
size_t shared_prefix_victim(SharedPrefixEvictionPolicy, const Anchors& anchors, std::span<const uint32_t> prefix,
    TokensOf&& tokens_of, TickOf&& tick_of) {
    const size_t n = anchors.size();
    size_t victim = n;
    for (size_t idx = 0; idx < n; ++idx) {
        const auto& tokens = tokens_of(anchors[idx]);
        const bool is_own_prefix = tokens.size() < prefix.size() && std::equal(tokens.begin(), tokens.end(), prefix.begin());
        if (!is_own_prefix && (victim == n || tick_of(anchors[idx]) < tick_of(anchors[victim]))) victim = idx;
    }
    if (victim == n) {
        victim = 0;
        for (size_t idx = 1; idx < n; ++idx) {
            if (tick_of(anchors[idx]) < tick_of(anchors[victim])) victim = idx;
        }
    }
    return victim;
}

} // namespace tokspur
