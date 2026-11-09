#include "tokspur/recurrent_replay_timing.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <string>

namespace {
size_t checks = 0;
void require(bool value) { ++checks; if (!value) throw std::runtime_error("recurrent replay timing protocol check"); }
void refuse(std::string_view text) {
    ++checks;
    try { static_cast<void>(tokspur::parse_recurrent_replay_timing_request(text)); }
    catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid recurrent replay timing request admitted");
}
void refuse_poison(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid freshness extent admitted");
}
void freshness_bytes() {
    using namespace tokspur;
    std::vector<std::byte> expected(1024);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = std::byte(i % 256);
    const auto saved = expected, poison = recurrent_replay_poison_bytes(expected);
    require(expected == saved && poison.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) require((poison[i] ^ expected[i]) == std::byte{255});
    // Repeated no-op validation of populated bytes is vacuous until poisoned.
    auto actual = expected; require(actual == expected);
    actual = poison; require(actual != expected); // No execution.
    actual = expected;
    std::copy_n(poison.begin(), 4, actual.begin()); require(actual != expected); // One unwritten word.
    actual = expected; require(actual == expected); // Complete write/recovery.
    for (const size_t bytes : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(5), recurrent_replay_poison_view_limit + 4}) {
        const std::vector<std::byte> invalid(bytes);
        refuse_poison([&] { static_cast<void>(recurrent_replay_poison_bytes(invalid)); });
    }
    const std::vector<std::byte> maximum(recurrent_replay_poison_view_limit);
    const auto bounded = recurrent_replay_poison_bytes(maximum);
    require(bounded.size() == maximum.size() && bounded.front() == std::byte{255} && bounded.back() == std::byte{255});
}
}
int main() {
    try {
        using namespace tokspur;
        const std::array modes{RecurrentReplayTimingMode::direct, RecurrentReplayTimingMode::compatible, RecurrentReplayTimingMode::capture};
        const std::array<const char*, 3> names{"direct", "compatible", "capture"};
        for (size_t i = 0; i < modes.size(); ++i) {
            const auto text = std::string("{\"mode\":\"") + names[i] + "\",\"cycles\":4}";
            const auto request = parse_recurrent_replay_timing_request(text);
            require(!request.close && request.cycles == 4 && request.mode == modes[i]);
            const auto reordered = parse_recurrent_replay_timing_request(" {\"cycles\":4,\"mode\":\"" + std::string(names[i]) + "\"}\r ");
            require(!reordered.close && reordered.cycles == 4 && reordered.mode == modes[i]);
        }
        const auto close = parse_recurrent_replay_timing_request(" {\"op\":\"close\"}\r ");
        require(close.close && close.cycles == 0);
        for (const auto* text : {"", "null", "[]", "{}", "true", "4", "\"capture\"", "{\"op\":\"stop\"}", "{\"op\":false}",
            "{\"op\":\"close\",\"cycles\":4}", "{\"op\":\"close\",\"mode\":\"capture\"}", "{\"op\":\"close\",\"op\":\"close\"}",
            "{\"cycles\":4}", "{\"mode\":\"capture\"}", "{\"mode\":\"capture\",\"cycles\":4,\"tokens\":1}",
            "{\"mode\":\"capture\",\"cycles\":4,\"unknown\":true}", "{\"mode\":\"capture\",\"unknown\":4}",
            "{\"mode\":\"capture\",\"mode\":\"capture\",\"cycles\":4}", "{\"mode\":\"capture\",\"cycles\":4,\"cycles\":4}",
            "{\"mode\":true,\"cycles\":4}", "{\"mode\":null,\"cycles\":4}", "{\"mode\":[],\"cycles\":4}",
            "{\"mode\":\"CAPTURE\",\"cycles\":4}", "{\"mode\":\"A\",\"cycles\":4}", "{\"mode\":\"serial\",\"cycles\":4}",
            "{\"mode\":\"direct \",\"cycles\":4}", "{\"mode\":\"\",\"cycles\":4}",
            "{\"mode\":\"capture\",\"cycles\":4.0}", "{\"mode\":\"capture\",\"cycles\":4e0}",
            "{\"mode\":\"capture\",\"cycles\":-4}", "{\"mode\":\"capture\",\"cycles\":true}",
            "{\"mode\":\"capture\",\"cycles\":\"4\"}", "{\"mode\":\"capture\",\"cycles\":null}",
            "{\"mode\":\"capture\",\"cycles\":18446744073709551616}", "{\"mode\":\"capture\",\"cycles\":4} {}",
            "{\"mode\":\"capture\",\"cycles\":4,}", "{\"mode\":\"capture\",\"cycles\":4"}) refuse(text);
        for (const auto cycles : {0, 1, 2, 3, 5, 8, 256, 1024})
            refuse("{\"mode\":\"direct\",\"cycles\":" + std::to_string(cycles) + "}");
        refuse(std::string(4097, ' '));
        refuse(std::string(4096, ' ') + "{\"op\":\"close\"}");
        freshness_bytes();
        std::cout << "Owned recurrent replay timing protocol checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned recurrent replay timing protocol check failed: " << error.what() << '\n'; return 1;
    }
}
