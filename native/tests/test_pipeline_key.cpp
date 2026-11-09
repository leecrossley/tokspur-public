#include "tokspur/pipeline_key.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace tokspur;
namespace {
unsigned checks = 0;
void check(bool value, const char* why) { ++checks; if (!value) throw std::runtime_error(why); }
struct CollisionHash {
    using is_transparent = void;
    size_t operator()(PipelineKeyView) const noexcept { return 0; }
};

template<class Hash>
void exercise() {
    std::unordered_map<PipelineKey, size_t, Hash, PipelineKeyEqual> cache;
    std::vector<PipelineKey> keys;
    for (const std::string name : {"nax_grouped_moe_gate_up_swiglu_v3", "nax_grouped_moe_down_v3"})
        for (uint32_t type : {0u, 8u, 12u, 13u, uint32_t(-1)}) for (uint32_t threads : {32u, 64u, 128u})
            for (uint32_t layout : {0u, 1u, uint32_t(-1)}) for (bool indirect : {false, true}) {
                PipelineKey key{name, type, threads, layout, indirect};
                const auto [at, inserted] = cache.emplace(key, keys.size());
                check(inserted && at->second == keys.size(), "pipeline specialisations collided");
                keys.push_back(key);
            }
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string borrowed = keys[i].name;
        const PipelineKeyView view{borrowed, keys[i].type, keys[i].minimum_threads, keys[i].layout, keys[i].indirect};
        check(PipelineKeyHash{}(view) == PipelineKeyHash{}(keys[i]), "owned and borrowed keys hash differently");
        const auto at = cache.find(view);
        check(at != cache.end() && at->second == i, "pipeline lookup ignored a specialisation field");
        borrowed[0] = 'x';
        check(cache.find(view) == cache.end(), "pipeline lookup used a name pointer instead of its content");
        check(cache.find(PipelineKeyView(keys[i]))->second == i, "borrowed-name mutation changed a stored key");
    }
    cache.clear();
    {
        std::string temporary = "temporary_kernel_name_longer_than_small_string_storage";
        cache.emplace(PipelineKey{temporary, 8, 128, uint32_t(-1), false}, 42);
        temporary.assign(temporary.size(), 'x');
    }
    const auto retained = cache.find(PipelineKeyView{"temporary_kernel_name_longer_than_small_string_storage", 8, 128, uint32_t(-1), false});
    check(retained != cache.end() && retained->second == 42, "pipeline cache did not own its stored name");
}
}

int main() {
    try {
        exercise<PipelineKeyHash>();
        exercise<CollisionHash>();
        std::cout << "pipeline key checks=" << checks << "; owned names and complete specialisation identity\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
