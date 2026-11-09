#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/q8_layout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
using namespace tokspur;
size_t checks = 0, batches = 0; double worst = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid layout operation accepted"); }
struct Arena {
    MetalGraph& g; std::vector<GpuView> all;
    GpuView bytes(std::span<const std::byte> data, bool immutable = true) {
        std::vector<std::byte> padded(data.size() + 8); std::copy(data.begin(), data.end(), padded.begin() + 4);
        auto whole = g.upload(padded, "layout fixture", immutable); all.push_back(whole); return g.slice(whole, 4, data.size());
    }
    template<class T> GpuView values(std::span<const T> data, bool immutable = true) { return bytes(std::as_bytes(data), immutable); }
    GpuView zero(size_t count) { return values<float>(std::vector<float>(count), false); }
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (auto v : all) g.release(v); }
};
void submit(MetalGraph& g, uint64_t allocations) {
    const auto b = g.submit(); ++batches;
    require(b.dispatches == 1 && g.memory().allocations == allocations, "layout changed physical dispatch/allocation count");
}
void close(MetalGraph& g, GpuView out, std::span<const double> reference) {
    const auto bytes = g.read(out); require(bytes.size() == reference.size() * 4, "layout output shape");
    for (size_t i = 0; i < reference.size(); ++i) {
        float v; std::memcpy(&v, bytes.data() + i * 4, 4);
        const auto error = std::abs(v - reference[i]) / std::max(1.0, std::abs(reference[i])); worst = std::max(worst, error);
        require(std::isfinite(v) && error <= 5e-5, "layout independent reference error");
    }
}
void exercise(MetalGraph& g, uint32_t c, uint32_t r, std::vector<std::byte> weights, std::vector<float> x) {
    Arena a{g, {}};
    const auto soa = q8_soa(weights, c, r, 32 * 1024 * 1024);
    const auto expanded = q8_expand_f32(weights, c, r, 32 * 1024 * 1024);
    require(q8_reconstruct(soa, 32 * 1024 * 1024) == weights, "layout bytes reconstruction");
    const auto reference = reference_gemv(8, weights, c, r, x);
    const auto w = a.bytes(weights), scales = a.values<uint16_t>(soa.scales), values = a.values<int8_t>(soa.values);
    const auto f32 = a.values<float>(expanded), input = a.values<float>(x, false), out = a.zero(r);
    const auto allocations = g.memory().allocations; std::vector<std::byte> ordinary;
    const auto encode = [&](uint32_t mode) {
        if (mode < 2) { g.configure_block_gemv(mode); g.begin(); g.gemv(8, w, input, out, c, r); }
        else { g.begin(); if (mode == 2) g.q8_soa_gemv(scales, values, input, out, c, r); else g.typed_f32_gemv(f32, input, out, c, r); }
    };
    for (uint32_t mode = 0; mode < 4; ++mode) {
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            encode(mode); submit(g, allocations); close(g, out, reference);
            if (!mode) ordinary = g.read(out); else require(g.read(out) == ordinary, "lossless layout changed ordinary result bits");
        }
    }
    require(g.read(w) == weights, "layout modified original weights");
    g.begin(); reject([&] { g.q8_soa_gemv(scales, values, input, out, c - 1, r); }); g.abort();
    g.begin(); reject([&] { g.q8_soa_gemv(scales, values, input, out, c, r + 1); }); g.abort();
    g.begin(); reject([&] { g.typed_f32_gemv(f32, input, out, c + 1, r); }); g.abort();
    std::fill(x.begin(), x.end(), std::numeric_limits<float>::infinity()); g.write(input, std::as_bytes(std::span(x)));
    for (uint32_t mode = 2; mode < 4; ++mode) { encode(mode); reject([&] { static_cast<void>(g.submit()); }); }
    std::fill(x.begin(), x.end(), 0); g.write(input, std::as_bytes(std::span(x)));
    for (uint32_t mode = 2; mode < 4; ++mode) { encode(mode); submit(g, allocations); close(g, out, std::vector<double>(r, 0)); }
}
void aliases(MetalGraph& g) {
    Arena a{g,{}}; constexpr uint32_t c = 32, r = 32;
    const auto values = a.zero(256), input = a.zero(c), output = a.zero(r), scales_backing = a.zero(r);
    const auto scales = g.slice(scales_backing, 4, r * 2), f32 = a.zero(c * r);
    const auto allocations = g.memory().allocations;
    // Every operand shape and mutability is valid: these reach the common
    // overlap guard, not an earlier shape or immutable-output refusal.
    g.begin(); reject([&] { g.q8_soa_gemv(scales, values, input, input, c, r); }); g.abort();
    g.begin(); reject([&] { g.q8_soa_gemv(scales, values, input, g.slice(values, 4, r * 4), c, r); }); g.abort();
    g.begin(); reject([&] { g.q8_soa_gemv(scales, values, input, scales_backing, c, r); }); g.abort();
    g.begin(); reject([&] { g.typed_f32_gemv(f32, input, input, c, r); }); g.abort();
    g.begin(); reject([&] { g.typed_f32_gemv(f32, input, g.slice(f32, 4, r * 4), c, r); }); g.abort();
    g.begin(); g.q8_soa_gemv(scales, values, input, output, c, r); submit(g, allocations); close(g, output, std::vector<double>(r, 0));
    for (const uint16_t bits : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e01), uint16_t(0xfe01)}) {
        std::vector<uint16_t> bad(r, bits); const auto bad_scales = a.values<uint16_t>(bad);
        g.begin(); g.q8_soa_gemv(bad_scales, values, input, output, c, r); reject([&] { static_cast<void>(g.submit()); });
        g.begin(); g.q8_soa_gemv(scales, values, input, output, c, r); submit(g, g.memory().allocations); close(g, output, std::vector<double>(r, 0));
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph g(argv[1], 64 * 1024 * 1024); const auto initial = g.memory().live_bytes; std::mt19937 rng(0x51534f41);
        for (const auto shape : std::array<std::array<uint32_t,2>,7>{{{32,1},{64,3},{96,5},{256,7},{2560,48},{10240,320},{320,10240}}}) {
            const auto c = shape[0], r = shape[1]; std::vector<std::byte> w(uint64_t(c / 32) * r * 34);
            for (auto& v : w) v = std::byte(rng() & 255);
            for (size_t at = 0; at < w.size(); at += 34) { w[at] = std::byte{0}; w[at + 1] = std::byte{0x18}; }
            std::vector<float> x(c); for (auto& v : x) v = (int(rng() % 2049) - 1024) / 4096.0f;
            exercise(g, c, r, std::move(w), std::move(x));
        }
        std::vector<std::byte> w(63488 * 34); uint32_t row = 0;
        for (uint32_t bits = 0; bits < 65536; ++bits) if ((bits & 0x7c00) != 0x7c00) {
            w[row * 34] = std::byte(bits & 255); w[row * 34 + 1] = std::byte(bits >> 8);
            for (uint32_t q = 0; q < 32; ++q) w[row * 34 + 2 + q] = std::byte(q ? 127 : 128);
            ++row;
        }
        exercise(g, 32, row, std::move(w), std::vector<float>(32, 0.03125f)); aliases(g);
        g.configure_block_gemv(0); require(g.memory().live_bytes == initial, "layout fixture leaked GPU backing");
        std::cout << "Owned layout GEMV checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned layout GEMV failed: " << e.what() << '\n'; return 1; }
}
