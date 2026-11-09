#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>

namespace {
size_t checks = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) {
    ++checks; try { f(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("native-load invalid control accepted");
}
struct Buffers {
    tokspur::MetalGraph& g;
    std::vector<tokspur::GpuView> owned;
    tokspur::GpuView bytes(std::span<const std::byte> data, bool immutable = true) {
        std::vector<std::byte> padded(data.size() + 8, std::byte{0x57});
        std::copy(data.begin(), data.end(), padded.begin() + 4);
        auto v = g.upload(padded, "plain-load component", immutable); owned.push_back(v);
        return g.slice(v, 4, data.size());
    }
    tokspur::GpuView floats(std::span<const float> x, bool immutable = true) { return bytes(std::as_bytes(x), immutable); }
    ~Buffers() { try { g.abort(); } catch (const tokspur::FormatError&) {} for (auto v : owned) g.release(v); }
};
std::vector<std::byte> weights(uint32_t type, uint32_t c, uint32_t r, std::mt19937& rng) {
    std::vector<std::byte> result(uint64_t(c) * r * (type == 0 ? 4 : 2));
    for (uint64_t i = 0; i < uint64_t(c) * r; ++i) {
        uint32_t bits = std::bit_cast<uint32_t>(float(int(rng() % 129) - 64) / 1024);
        if (i % 71 == 0) bits = 0x80000000u;
        if (i % 73 == 0) bits = 0x00800000u;
        if (i % 79 == 0) bits = 0x00010000u;
        if (type == 0) std::memcpy(result.data() + i * 4, &bits, 4);
        else { uint16_t b = uint16_t(bits >> 16); std::memcpy(result.data() + i * 2, &b, 2); }
    }
    return result;
}
void compare(tokspur::MetalGraph& g, uint32_t type, uint32_t c, uint32_t r, std::mt19937& rng) {
    Buffers b{g, {}}; auto w = weights(type, c, r, rng);
    std::vector<float> x(c); for (auto& v : x) v = float(int(rng() % 129) - 64) / 2048;
    auto matrix = b.bytes(w), input = b.floats(x, false), output = b.floats(std::vector<float>(r), false);
    const auto expected = tokspur::reference_gemv(type, w, c, r, x);
    std::vector<std::byte> baseline;
    for (bool on : {false, true, false}) {
        g.configure_native_plain_loads(on); require(g.native_plain_loads() == on, "selector differs");
        const auto before = g.memory().allocations; g.begin();
        rejects([&] { g.configure_native_plain_loads(!on); });
        g.gemv(type, matrix, input, output, c, r); const auto batch = g.submit();
        require(batch.dispatches == 1 && g.memory().allocations == before, "load route changed dispatch/allocation count");
        auto actual = g.read(output);
        if (baseline.empty()) baseline = actual; else require(actual == baseline, "native loads change output bytes");
        for (uint32_t i = 0; i < r; ++i) {
            float value; std::memcpy(&value, actual.data() + uint64_t(i) * 4, 4);
            require(std::isfinite(value) && std::abs(value - expected[i]) / std::max(1.0, std::abs(expected[i])) <= 5e-5,
                "native load independent FP64 reference differs");
        }
    }
    x[0] = std::numeric_limits<float>::infinity(); g.write(input, std::as_bytes(std::span(x)));
    for (bool on : {false, true}) {
        g.configure_native_plain_loads(on); g.begin(); g.gemv(type, matrix, input, output, c, r);
        // submit already drains and closes the failed batch.
        rejects([&] { static_cast<void>(g.submit()); });
    }
}
void benchmark(tokspur::MetalGraph& g, uint32_t type, uint32_t c, uint32_t r, std::mt19937& rng) {
    Buffers b{g, {}}; const auto w = weights(type, c, r, rng);
    std::vector<float> x(c); for (auto& v : x) v = float(int(rng() % 129) - 64) / 2048;
    auto matrix = b.bytes(w), input = b.floats(x), output = b.floats(std::vector<float>(r), false);
    constexpr uint32_t repeats = 128;
    // A short pipeline warm-up left the first F32 blocks on a changing clock. Prospectively use the same sustained, alternating warm-up for both arms.
    for (uint32_t warm = 0; warm < 64; ++warm) {
        g.configure_native_plain_loads(bool(warm & 1)); g.begin();
        for (uint32_t i = 0; i < repeats; ++i) g.gemv(type, matrix, input, output, c, r);
        static_cast<void>(g.submit());
    }
    std::array<std::vector<double>, 2> times;
    for (uint32_t round = 0; round < 3; ++round) for (bool on : {false, true, true, false}) {
        g.configure_native_plain_loads(on); g.begin();
        for (uint32_t i = 0; i < repeats; ++i) g.gemv(type, matrix, input, output, c, r);
        const auto result = g.submit(); require(result.dispatches == repeats, "benchmark dispatch count differs");
        times[on].push_back(result.gpu_ms / repeats);
    }
    for (uint32_t arm = 0; arm < 2; ++arm) {
        auto sorted = times[arm]; std::sort(sorted.begin(), sorted.end());
        std::cout << "{\"type\":" << type << ",\"columns\":" << c << ",\"rows\":" << r << ",\"on\":" << (arm ? "true" : "false")
            << ",\"samples\":6,\"median_gpu_ms\":" << (sorted[2] + sorted[3]) / 2 << ",\"gpu_ms\":[";
        for (size_t i = 0; i < times[arm].size(); ++i) std::cout << (i ? "," : "") << times[arm][i];
        std::cout << "]}\n";
    }
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--bench"), "usage: test-native_plain_loads library [--bench]");
        tokspur::MetalGraph g(argv[1], 64'000'000); require(!g.native_plain_loads(), "native loads enabled by default");
        std::mt19937 rng(4403);
        for (uint32_t type : {0u, 30u}) for (uint32_t c : {1u, 31u, 32u, 33u, 127u, 128u, 129u, 640u, 2560u})
            for (uint32_t r : {1u, 3u, 4u, 5u, 17u}) compare(g, type, c, r, rng);
        if (argc == 3) for (auto shape : std::array<std::array<uint32_t, 3>, 3>{{{0, 2560, 512}, {30, 2560, 512}, {30, 2560, 128}}})
            benchmark(g, shape[0], shape[1], shape[2], rng);
        std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"full_model\":false,\"performance_qualified\":false}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
