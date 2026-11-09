#include "tokspur/model.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid model geometry was accepted");
}
uint64_t value(const tokspur::Json& object, std::string_view key) {
    const auto* item = object.find(key);
    if (!item) throw std::runtime_error("missing layout field");
    return item->unsigned_integer();
}
}

int main() {
    try {
        const auto requirements = tokspur::flash_next_requirements(320001536);
        require(requirements.size() == 1224, "closed tensor count");
        std::set<std::string> unique;
        size_t recurrent = 0, attention = 0, ple = 0;
        for (const auto& item : requirements) {
            require(unique.insert(item.name).second, "duplicate name");
            require(!item.encodings.empty() && !item.shape.empty(), "empty requirement");
            for (const auto type : item.encodings) {
                const auto& encoding = tokspur::encoding(type);
                require(item.shape.front() % encoding.elements_per_block == 0, "unreadable row geometry");
            }
            if (item.name.ends_with("ssm_a")) ++recurrent;
            if (item.name.ends_with("indexer.q_proj.weight")) ++attention;
            if (item.name.find(".ple_") != std::string::npos) ++ple;
        }
        require(recurrent == 36 && attention == 12 && ple == 6, "layer subgraph counts");
        const auto table = std::find_if(requirements.begin(), requirements.end(), [](const auto& item) {
            return item.name == "per_layer_token_embd.weight";
        });
        require(table != requirements.end() && table->shape == std::vector<uint64_t>({160, 320001536}), "PLE table shape");
        rejected([] { static_cast<void>(tokspur::flash_next_requirements(0)); });
        rejected([] { static_cast<void>(tokspur::flash_next_requirements(uint64_t(-1))); });
        tokspur::FlashLayout layout;
        layout.tensors = 1224;
        layout.resident_matrix_bytes = 82523491840;
        layout.ple_source_bytes = 28800138240;
        layout.rms_epsilon = 1e-6;
        layout.rope_base = 1e7;
        const auto description = layout.describe();
        require(value(description, "kv_bf16_bytes") == 1610612736, "64K BF16 KV accounting");
        require(value(description, "gdn_fp32_bytes") == 113246208, "GDN FP32 state accounting");
        require(value(description, "indexer_bf16_keys_bytes") == 50331648, "index state accounting");
        // Nine PLE history vectors (kernel=4,dilation=3), not three unit-dilation vectors.
        require(value(description, "known_payload_subtotal_bytes") == 84839351808, "payload subtotal");
        require(value(tokspur::parse_json(description.dump()), "tensors_consumed") == 1224, "layout protocol roundtrip");
        std::cout << "Model geometry checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Model geometry check failed: " << error.what() << '\n';
        return 1;
    }
}
