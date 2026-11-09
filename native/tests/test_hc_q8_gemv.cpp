#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/q8_split.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace tokspur;
size_t checks = 0, batches = 0;
void check(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void refuses(const std::function<void()>& fn) {
    ++checks; try { fn(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid HC Q8 policy admitted");
}
void same_resources(GraphMemory a, GraphMemory b) {
    check(a.live_bytes == b.live_bytes && a.allocations == b.allocations &&
        a.profile_counter_payload_bytes == b.profile_counter_payload_bytes && a.live_captures == b.live_captures,
        "HC Q8 execution/preparation changed fixed resource ledger");
}
void run(MetalGraph& graph) {
    const auto empty = graph.memory(); const auto off = graph.hc_q8_policy();
    check(!off.splits && !off.scratch_logical_bytes && !off.scratch_charged_bytes, "HC Q8 is not default off");
    graph.configure_hc_q8_splits(0); same_resources(empty, graph.memory());
    for (uint32_t invalid : {3, 7, 33}) refuses([&] { graph.configure_hc_q8_splits(invalid); });
    refuses([&] { graph.configure_hc_q8_splits(1); }); // No block-Q8 base policy.
    graph.configure_block_gemv(3);
    std::vector<std::byte> bytes(10240ull / 32 * 320 * 34);
    for (uint32_t r = 0; r < 320; ++r) for (uint32_t b = 0; b < 320; ++b) {
        const auto at = (uint64_t(r) * 320 + b) * 34;
        bytes[at] = std::byte{0}; bytes[at + 1] = std::byte{0x1c};
        for (uint32_t lane = 0; lane < 32; ++lane) {
            const int8_t q = int8_t(int((r * 17 + b * 11 + lane * 7) % 127) - 63);
            std::memcpy(bytes.data() + at + 2 + lane, &q, 1);
        }
    }
    std::vector<float> input(10240);
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 31) % 127) - 63) / 128;
    const auto matrix = graph.upload(bytes, "generated HC Q8 weights");
    const auto x = graph.upload(std::as_bytes(std::span(input)), "generated HC Q8 input");
    const auto y = graph.zeros(320 * 4, "generated HC Q8 output");
    const auto submit = [&](uint32_t expected, const GraphMemory& before) {
        const auto done = graph.submit(); ++batches;
        check(done.dispatches == expected && !done.capture_executes, "HC Q8 physical dispatch count differs");
        same_resources(before, graph.memory());
    };
    auto before = graph.memory(); graph.begin(); graph.gemv(8, matrix, x, y, 10240, 320); submit(1, before);
    const auto ordinary = graph.read(y); const auto reference = reference_gemv(8, bytes, 10240, 320, input);
    const auto prior = graph.memory(); graph.configure_hc_q8_splits(1); const auto fixed = graph.memory();
    auto hc = graph.hc_q8_policy();
    check(hc.splits == 1 && hc.scratch_logical_bytes == 40960 && hc.scratch_charged_bytes >= 40960 &&
        fixed.allocations == prior.allocations + 1 && fixed.live_bytes == prior.live_bytes + hc.scratch_charged_bytes,
        "HC Q8 maximum scratch ledger is not exact");
    refuses([&] { graph.configure_profiling(8); });
    refuses([&] { graph.configure_block_gemv(2); });
    refuses([&] { graph.begin_indirect_compatible(); });
    refuses([&] { graph.begin_capture(8); });
    bool foreign = false;
    std::thread other([&] { try { static_cast<void>(graph.hc_q8_policy()); } catch (const FormatError&) { foreign = true; } });
    other.join(); check(foreign, "HC Q8 policy accepted another thread");
    for (uint32_t count : {1, 2, 4, 8, 16, 32, 1}) {
        graph.configure_hc_q8_splits(count); same_resources(fixed, graph.memory());
        check(graph.gemv_dispatches(8, 10240, 320) == 2 && graph.gemv_dispatches(8, 10240, 319) == 1 &&
            graph.gemv_dispatches(8, 10208, 320) == 1 && graph.gemv_dispatches(0, 10240, 320) == 1,
            "HC Q8 dispatch family escaped its exact bounds");
        before = graph.memory(); graph.begin();
        refuses([&] { graph.configure_hc_q8_splits(0); });
        graph.gemv(8, matrix, x, y, 10240, 320); submit(2, before);
        const auto result = graph.read(y);
        if (count == 1) check(result == ordinary, "HC split1 differs from ordinary fast block-Q8");
        std::vector<float> values(320); std::memcpy(values.data(), result.data(), result.size());
        for (uint32_t row = 0; row < 320; ++row)
            check(std::isfinite(values[row]) && std::abs(values[row] - reference[row]) / (1 + std::abs(reference[row])) <= 5e-5,
                "HC Q8 exceeds independent scalar-double fixture error bound");
        // Same resident policy must not route an adjacent non-HC geometry.
        before = graph.memory(); graph.begin();
        graph.gemv(8, graph.slice(matrix, 0, 10240ull / 32 * 319 * 34), x, graph.slice(y, 0, 319 * 4), 10240, 319);
        submit(1, before);
    }
    graph.configure_hc_q8_splits(0);
    check(graph.memory().live_bytes == prior.live_bytes && !graph.hc_q8_policy().scratch_charged_bytes,
        "disabling HC Q8 did not release maximum scratch charge");
    graph.begin_capture(1);
    graph.gemv(8, matrix, x, y, 10240, 320);
    const auto capture = graph.finish_capture(); const auto capture_resources = graph.memory();
    refuses([&] { graph.configure_hc_q8_splits(1); }); same_resources(capture_resources, graph.memory());
    graph.release_capture(capture);
    graph.release(y); graph.release(x); graph.release(matrix);
    check(graph.memory().live_bytes == empty.live_bytes, "HC Q8 fixture leaked resources");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph graph(argv[1], 32ull << 20); run(graph);
        MetalGraph limited(argv[1], 16384); limited.configure_block_gemv(3);
        const auto before = limited.memory(); refuses([&] { limited.configure_hc_q8_splits(1); });
        same_resources(before, limited.memory());
        check(!limited.hc_q8_policy().splits && !limited.hc_q8_policy().scratch_logical_bytes &&
            !limited.hc_q8_policy().scratch_charged_bytes, "failed HC preparation published policy or scratch");
        std::cout << "Owned HC Q8 fixture checks passed: " << checks << ", batches: " << batches << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned HC Q8 check failed: " << e.what() << '\n'; return 1; }
}
