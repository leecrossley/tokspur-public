#pragma once

#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"

namespace tokspur {
enum class RecurrentTimingMode { serial, tile };
struct RecurrentTimingRequest {
    bool close = false;
    uint32_t tokens = 0, block_mask = 0, cycles = 0;
    RecurrentTimingMode mode = RecurrentTimingMode::serial;
};

// The protocol is deliberately model-free and available to CPU-only checks.
// Framing (a required terminal newline, EOF and the request-count bound) belongs
// to the persistent worker, not to this single-object parser.
inline RecurrentTimingRequest parse_recurrent_timing_request(std::string_view text) {
    const auto value = parse_json(text, {4096, 32, 4});
    const auto& object = value.object();
    if (const auto* op = value.find("op")) {
        if (object.size() != 1 || op->string() != "close") throw FormatError("invalid recurrent timing close request");
        return {.close = true};
    }
    if (object.size() != 4) throw FormatError("recurrent timing requires exactly four request fields");
    for (const auto& [key, ignored] : object) {
        static_cast<void>(ignored);
        if (key != "tokens" && key != "mode" && key != "block_mask" && key != "cycles")
            throw FormatError("unknown recurrent timing request field");
    }
    const auto field = [&](std::string_view key) -> const Json& {
        const auto* result = value.find(key);
        if (!result) throw FormatError("missing recurrent timing request field");
        return *result;
    };
    const auto tokens = field("tokens").unsigned_integer(), mask = field("block_mask").unsigned_integer();
    const auto cycles = field("cycles").unsigned_integer(); const auto mode = field("mode").string();
    if ((tokens != 2 && tokens != 8 && tokens != 17 && tokens != 32) || (mask != 1 && mask != 3) || cycles != 4 ||
        (mode != "serial" && mode != "tile")) throw FormatError("unsupported recurrent timing request geometry or policy");
    return {false, uint32_t(tokens), uint32_t(mask), uint32_t(cycles),
        mode == "serial" ? RecurrentTimingMode::serial : RecurrentTimingMode::tile};
}

int recurrent_timing_cli(int argc, char** argv);
} // namespace tokspur
