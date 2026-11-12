#include "tokspur/shared_prefix_eviction.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
size_t checks = 0;
void require(bool b, const char* m) { ++checks; if (!b) throw std::runtime_error(m); }

struct Anchor { std::vector<uint32_t> tokens; uint64_t last_access_tick = 0; };

// Oracle: the victim loop as it stood inline in native/src/target.cpp at 33cfd20 (lines 6370-6387),
// copied verbatim apart from the container name.
size_t inline_loop_victim(const std::vector<Anchor>& anchors, const std::vector<uint32_t>& prefix_tokens) {
    size_t victim = anchors.size();
    for (size_t idx = 0; idx < anchors.size(); ++idx) {
        const auto& item = anchors[idx];
        const bool is_own_prefix = item.tokens.size() < prefix_tokens.size() &&
            std::equal(item.tokens.begin(), item.tokens.end(), prefix_tokens.begin());
        if (!is_own_prefix && (victim == anchors.size() ||
            item.last_access_tick < anchors[victim].last_access_tick)) {
            victim = idx;
        }
    }
    if (victim == anchors.size()) {
        victim = 0;
        for (size_t idx = 1; idx < anchors.size(); ++idx) {
            if (anchors[idx].last_access_tick < anchors[victim].last_access_tick) {
                victim = idx;
            }
        }
    }
    return victim;
}

size_t header_victim(const std::vector<Anchor>& anchors, const std::vector<uint32_t>& prefix) {
    return tokspur::shared_prefix_victim(tokspur::SharedPrefixEvictionPolicy::lru, anchors, prefix,
        [](const Anchor& a) -> const std::vector<uint32_t>& { return a.tokens; },
        [](const Anchor& a) { return a.last_access_tick; });
}
}

int main() {
    try {
        using namespace tokspur;
        // Hand cases: own prefixes are skipped, all-own falls back to oldest, ties keep the lowest index.
        const std::vector<uint32_t> prefix{1, 2, 3, 4, 5, 6};
        require(header_victim({{{1, 2}, 1}, {{9}, 5}, {{1, 2, 3}, 0}}, prefix) == 1, "own prefixes are kept");
        require(header_victim({{{1, 2}, 4}, {{1, 2, 3}, 2}}, prefix) == 1, "all own prefixes: oldest overall");
        require(header_victim({{{7}, 3}, {{8}, 3}}, prefix) == 0, "ties keep the lowest index");
        require(header_victim({{{1, 2, 3, 4, 5, 6}, 0}, {{9}, 1}}, prefix) == 0, "an equal prefix is not an own prefix");
        require(header_victim({{{1, 2, 3, 4, 5, 6, 7}, 0}}, prefix) == 0, "a longer anchor is not an own prefix");

        // Randomised equivalence with the inline loop, including ties and shared stems.
        std::mt19937_64 rng(20261111);
        for (int round = 0; round < 200000; ++round) {
            const size_t count = 1 + rng() % 8;
            std::vector<uint32_t> capture(1 + rng() % 12);
            for (auto& t : capture) t = uint32_t(rng() % 3);
            std::vector<Anchor> anchors(count);
            for (auto& anchor : anchors) {
                const size_t length = 1 + rng() % 12;
                if (rng() % 2) {
                    anchor.tokens.assign(capture.begin(), capture.begin() + std::ptrdiff_t(std::min(length, capture.size())));
                } else {
                    anchor.tokens.resize(length);
                    for (auto& t : anchor.tokens) t = uint32_t(rng() % 3);
                }
                anchor.last_access_tick = rng() % 6;
            }
            require(header_victim(anchors, capture) == inline_loop_victim(anchors, capture), "lru reproduces the inline loop");
        }

        require(shared_prefix_eviction_cause(true, false) == SharedPrefixEvictionCause::bytes, "byte cause");
        require(shared_prefix_eviction_cause(false, true) == SharedPrefixEvictionCause::count, "count cause");
        require(shared_prefix_eviction_cause(true, true) == SharedPrefixEvictionCause::both, "both causes");
        SharedPrefixEvictionCounters counters;
        counters.record(SharedPrefixEvictionCause::bytes);
        counters.record(SharedPrefixEvictionCause::count);
        counters.record(SharedPrefixEvictionCause::count);
        counters.record(SharedPrefixEvictionCause::both);
        require(counters.bytes == 1 && counters.count == 2 && counters.both == 1, "eviction counters by cause");
        std::cout << "Shared-prefix eviction checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Shared-prefix eviction check failed: " << error.what() << '\n';
        return 1;
    }
}
