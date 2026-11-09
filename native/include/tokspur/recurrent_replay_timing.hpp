#pragma once

#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"

namespace tokspur {
inline constexpr std::string_view recurrent_replay_freshness_condition = "complement-expected-bank-v1";
inline constexpr size_t recurrent_replay_poison_view_limit = 4 * 1024 * 1024;

// Pure, bounded setup helper shared with generated checks. Every expected byte
// differs, including zero values and integer route IDs. Callers provide only
// semantic write extents, never allocator padding or immutable input/base data.
inline std::vector<std::byte> recurrent_replay_poison_bytes(std::span<const std::byte> expected) {
    if (expected.empty() || expected.size() % 4 || expected.size() > recurrent_replay_poison_view_limit)
        throw FormatError("invalid recurrent replay freshness extent");
    std::vector<std::byte> result(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) result[i] = ~expected[i];
    return result;
}

enum class RecurrentReplayTimingMode { direct, compatible, capture };
struct RecurrentReplayTimingRequest {
    bool close = false;
    uint32_t cycles = 0;
    RecurrentReplayTimingMode mode = RecurrentReplayTimingMode::direct;
};

// CPU-only, model-free object admission. The worker separately requires a final
// newline, explicit close, <=4096 bytes/line and <=256 measured requests.
inline RecurrentReplayTimingRequest parse_recurrent_replay_timing_request(std::string_view text) {
    const auto value = parse_json(text, {4096, 32, 4});
    const auto& object = value.object();
    if (const auto* op = value.find("op")) {
        if (object.size() != 1 || op->string() != "close") throw FormatError("invalid recurrent replay timing close request");
        return {.close = true};
    }
    if (object.size() != 2) throw FormatError("recurrent replay timing requires exactly two request fields");
    for (const auto& [key, unused] : object) {
        static_cast<void>(unused);
        if (key != "mode" && key != "cycles") throw FormatError("unknown recurrent replay timing request field");
    }
    const auto* mode_field = value.find("mode"); const auto* cycles_field = value.find("cycles");
    if (!mode_field || !cycles_field) throw FormatError("missing recurrent replay timing request field");
    const auto mode = mode_field->string(); const auto cycles = cycles_field->unsigned_integer();
    if (cycles != 4 || (mode != "direct" && mode != "compatible" && mode != "capture"))
        throw FormatError("unsupported recurrent replay timing mode or cycle count");
    return {false, uint32_t(cycles), mode == "direct" ? RecurrentReplayTimingMode::direct :
        mode == "compatible" ? RecurrentReplayTimingMode::compatible : RecurrentReplayTimingMode::capture};
}

int recurrent_replay_timing_cli(int argc, char** argv);
} // namespace tokspur
