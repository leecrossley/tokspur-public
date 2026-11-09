#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace tokspur {

struct PipelineKeyView {
    std::string_view name;
    uint32_t type, minimum_threads, layout;
    bool indirect;
    bool operator==(const PipelineKeyView&) const = default;
};

// Stored keys own their names. Lookup views borrow only for the duration of find(), including callers with temporary names.
struct PipelineKey {
    std::string name;
    uint32_t type, minimum_threads, layout;
    bool indirect;
    operator PipelineKeyView() const noexcept { return {name, type, minimum_threads, layout, indirect}; }
};

struct PipelineKeyHash {
    using is_transparent = void;
    size_t operator()(PipelineKeyView key) const noexcept {
        size_t value = std::hash<std::string_view>{}(key.name);
        const auto mix = [&](uint32_t part) { value ^= size_t(part) + size_t(0x9e3779b9u) + (value << 6) + (value >> 2); };
        mix(key.type); mix(key.minimum_threads); mix(key.layout); mix(key.indirect);
        return value;
    }
};

struct PipelineKeyEqual {
    using is_transparent = void;
    bool operator()(PipelineKeyView a, PipelineKeyView b) const noexcept { return a == b; }
};

} // namespace tokspur
