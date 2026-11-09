#include "tokspur/recurrent_timing.hpp"

#include <iostream>
#include <string>

namespace {
size_t checks = 0;
void require(bool ok) { ++checks; if (!ok) throw std::runtime_error("recurrent timing protocol check"); }
void refuse(std::string_view text) {
    ++checks;
    try { static_cast<void>(tokspur::parse_recurrent_timing_request(text)); }
    catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("recurrent timing invalid request admitted");
}
}
int main() {
    try {
        for (const auto n : {2, 8, 17, 32}) for (const auto mask : {1, 3}) for (const auto* mode : {"serial", "tile"}) {
            const auto text = "{\"tokens\":" + std::to_string(n) + ",\"mode\":\"" + mode + "\",\"block_mask\":" + std::to_string(mask) + ",\"cycles\":4}";
            const auto r = tokspur::parse_recurrent_timing_request(text);
            require(!r.close && r.tokens == uint32_t(n) && r.block_mask == uint32_t(mask) && r.cycles == 4 &&
                r.mode == (std::string_view(mode) == "serial" ? tokspur::RecurrentTimingMode::serial : tokspur::RecurrentTimingMode::tile));
        }
        require(tokspur::parse_recurrent_timing_request(" {\"op\":\"close\"}\r ").close);
        for (const auto* text : {"", "null", "[]", "{}", "{\"op\":\"stop\"}", "{\"op\":true}", "{\"op\":\"close\",\"cycles\":4}",
            "{\"tokens\":2,\"mode\":\"serial\",\"block_mask\":1}",
            "{\"tokens\":2,\"mode\":\"serial\",\"block_mask\":1,\"cycles\":4,\"extra\":0}",
            "{\"tokens\":2,\"mode\":\"serial\",\"block_mask\":1,\"cycles\":4,\"tokens\":2}",
            "{\"tokens\":2,\"mode\":\"serial\",\"block_mask\":1,\"cycles\":4.0}",
            "{\"tokens\":2,\"mode\":\"serial\",\"block_mask\":1,\"cycles\":4e0}",
            "{\"tokens\":-2,\"mode\":\"serial\",\"block_mask\":1,\"cycles\":4}",
            "{\"tokens\":2,\"mode\":\"SERIAL\",\"block_mask\":1,\"cycles\":4}",
            "{\"tokens\":2,\"mode\":true,\"block_mask\":1,\"cycles\":4}",
            "{\"tokens\":2,\"mode\":\"serial\",\"block_mask\":1,\"cycles\":4} {}"}) refuse(text);
        for (const auto n : {0, 1, 3, 4, 7, 16, 31, 33, 65535})
            refuse("{\"tokens\":" + std::to_string(n) + ",\"mode\":\"tile\",\"block_mask\":1,\"cycles\":4}");
        for (const auto mask : {0, 2, 4, 255})
            refuse("{\"tokens\":8,\"mode\":\"tile\",\"block_mask\":" + std::to_string(mask) + ",\"cycles\":4}");
        for (const auto cycles : {0, 1, 3, 5, 256})
            refuse("{\"tokens\":8,\"mode\":\"tile\",\"block_mask\":1,\"cycles\":" + std::to_string(cycles) + "}");
        refuse(std::string(4097, ' '));
        std::cout << "Owned recurrent timing protocol checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned recurrent timing protocol check failed: " << error.what() << '\n'; return 1;
    }
}
