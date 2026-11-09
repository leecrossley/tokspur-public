#include "tokspur/ple.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
using namespace tokspur;
size_t checks = 0, batches = 0; double worst = 0;
void require(bool b, const char* m) { ++checks; if (!b) throw std::runtime_error(m); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid PLE/mixer operation accepted"); }
std::vector<double> real(std::span<const float> x) { return {x.begin(), x.end()}; }
struct Arena {
    MetalGraph& g; std::mt19937 rng{0x706c6531}; std::deque<std::vector<float>> data; std::vector<GpuView> owned;
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (const auto v : owned) g.release(v); }
    std::vector<float> values(size_t n, float offset = 0) { std::vector<float> out(n); for (auto& x : out) x = float(int(rng() % 2001) - 1000) / 100000 + offset; return out; }
    GpuView floats(std::span<const float> x, bool immutable = true) { const auto v = g.upload(std::as_bytes(x), "PLE test data", immutable); owned.push_back(v); return v; }
    GpuView zero(size_t n) { const auto v = g.zeros(n * 4, "PLE test state"); owned.push_back(v); return v; }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(size_t n) {
        data.push_back(values(n)); return {{floats(data.back()), 0}, {std::as_bytes(std::span(data.back())), 0}};
    }
};
std::vector<float> read(MetalGraph& g, GpuView v) { const auto bytes = g.read(v); std::vector<float> out(bytes.size() / 4); std::memcpy(out.data(), bytes.data(), bytes.size()); return out; }
void close(MetalGraph& g, GpuView view, std::span<const double> expected, double tolerance = 5e-5) {
    const auto actual = read(g, view); require(actual.size() == expected.size(), "PLE/mixer comparison shape");
    for (size_t i = 0; i < actual.size(); ++i) { const auto error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i])); worst = std::max(worst, error); require(std::isfinite(actual[i]) && error <= tolerance, "PLE/mixer double comparison failed"); }
}
void connected(MetalGraph& g) {
    for (const auto transform : {PleGateTransform::plain, PleGateTransform::signed_root})
    for (auto shape : std::array<PleShape, 3>{{{1, 1, 1, 4, 3}, {4, 7, 16, 4, 3}, {4, 2560, 2560, 4, 3}}}) {
        shape.gate_transform = transform;
        Arena a{g, std::mt19937(0x706c6531), {}, {}};
        const uint32_t width = shape.branches * shape.channels, rank = shape.channels == 2560 ? 320 : 3, vocab = 17;
        const auto key = a.matrix(width * shape.embedding), value = a.matrix(shape.channels * shape.embedding), down = a.matrix(width * rank), up = a.matrix(rank * width), head = a.matrix(shape.channels * vocab);
        const auto gains = a.values(width, 1), conv_weights = a.values(width * shape.kernel), mix_gains = a.values(width, 1);
        const auto norm = a.floats(gains), conv = a.floats(conv_weights);
        const PleWeights weights{key.first, value.first, norm, norm, norm, conv};
        const auto dg = real(gains), dc = real(conv_weights), dm = real(mix_gains);
        const ReferencePleWeights reference{key.second, value.second, dg, dg, dg, dc};
        const MixerWeights mixer{a.floats(mix_gains), down.first, up.first}; const ResidualShape mixshape{shape.branches, shape.channels, rank};
        const auto scratch = allocate_ple_scratch(g, shape); const auto ms = allocate_mixer_scratch(g, mixshape);
        auto history = a.zero(width * (shape.kernel - 1) * shape.dilation), next = a.zero(width * (shape.kernel - 1) * shape.dilation);
        const auto embedding = a.zero(shape.embedding), streams = a.zero(width), output = a.zero(width), logits = a.zero(vocab);
        std::vector<double> cpu_history(history.bytes / 4, 0);
        const auto allocations = g.memory().allocations;
        for (size_t t = 0; t < 12; ++t) {
            const auto e = a.values(shape.embedding), x = a.values(width);
            g.write(embedding, std::as_bytes(std::span(e))); g.write(streams, std::as_bytes(std::span(x)));
            const auto expected = ple_reference(reference, real(e), real(x), cpu_history, shape, 1e-6f);
            const auto mixed = mixer_reference(dm, down.second, up.second, expected.output, mixshape, 1e-6f);
            const auto expected_logits = reference_gemv_doubleinput(head.second.encoding, head.second.bytes, shape.channels, vocab, mixed.mixed);
            const auto accepted = g.read(history);
            g.begin(); encode_ple(g, weights, scratch, embedding, streams, history, next, output, shape, 1e-6f);
            encode_mixer(g, mixer, ms, output, mixshape, 1e-6f); g.gemv(0, head.first.view, ms.mixed, logits, shape.channels, vocab);
            const auto result = g.submit(); ++batches;
            require(result.dispatches == 14 && g.memory().allocations == allocations && g.read(history) == accepted, "PLE batch/accepted-state invariant");
            close(g, scratch.key, expected.key); close(g, scratch.value, expected.value); close(g, scratch.normalized_key, expected.normalized_key);
            close(g, scratch.normalized_query, expected.normalized_query); close(g, scratch.gates, expected.gates); close(g, scratch.gated, expected.gated);
            close(g, scratch.normalized_gated, expected.normalized_gated); close(g, scratch.convolved, expected.convolution.output); close(g, output, expected.output);
            close(g, next, expected.convolution.history); close(g, ms.normalized, mixed.normalized); close(g, ms.bottleneck, mixed.bottleneck);
            close(g, ms.scores, mixed.scores); close(g, ms.mixed, mixed.mixed); close(g, logits, expected_logits);
            cpu_history = expected.convolution.history; std::swap(history, next);
        }
        const auto accepted = g.read(history), candidate = g.read(next), prior = g.read(output);
        g.begin(); encode_ple(g, weights, scratch, embedding, streams, history, next, output, shape, 1e-6f); g.abort();
        require(g.read(history) == accepted && g.read(next) == candidate && g.read(output) == prior, "aborted PLE executed writes");
        g.begin(); reject([&] { encode_ple(g, weights, scratch, embedding, streams, history, history, output, shape, 1e-6f); });
        auto bad = weights; bad.value.encoding = 0xffffffffu;
        g.begin(); reject([&] { encode_ple(g, bad, scratch, embedding, streams, history, next, output, shape, 1e-6f); });
        require(g.read(next) == candidate, "late PLE encode error executed a partial batch");
        release_ple_scratch(g, scratch); release_mixer_scratch(g, ms);
    }
}
void gate_modes(MetalGraph& g) {
    const std::array<float, 21> scores{-1000, -16, -4, -1, -.25f, -1e-6f, -1e-10f, -0.f, 0.f, 1e-10f, 1e-6f, .25f, 1, 4, 16, 1000, -.75f, .75f, -.01f, .01f, 0};
    for (const uint32_t channels : {1u, 32u, 128u, 2560u}) {
        Arena a{g, std::mt19937(13), {}, {}};
        constexpr uint32_t branches = 7;
        const uint32_t width = branches * channels;
        std::vector<float> query(width), key(width, 0), values(channels);
        for (uint32_t b = 0; b < branches; ++b) key[b * channels] = 1;
        for (uint32_t c = 0; c < channels; ++c) values[c] = float(int(c % 19) - 9) / 16;
        const auto q = a.zero(width), k = a.floats(key), v = a.floats(values);
        const std::vector<float> sentinel(width + 4, -99), gs(branches + 4, -99);
        const auto storage = a.floats(sentinel, false), gate_storage = a.floats(gs, false);
        const auto output = g.slice(storage, 8, width * 4), gates = g.slice(gate_storage, 8, branches * 4);
        const auto allocations = g.memory().allocations;
        for (size_t offset = 0; offset < scores.size(); offset += branches) {
            for (uint32_t b = 0; b < branches; ++b) query[b * channels] = scores[offset + b] * std::sqrt(float(channels));
            g.write(q, std::as_bytes(std::span(query)));
            std::vector<std::byte> checkpoint_default;
            for (const auto mode : {PleGateTransform::plain, PleGateTransform::signed_root}) {
                g.begin(); g.ple_gate(q, k, v, output, gates, branches, channels, mode);
                require(g.submit().dispatches == 1 && g.memory().allocations == allocations, "gate changed allocation/dispatch contract"); ++batches;
                std::vector<double> expected_gates(branches), expected_output(width);
                for (uint32_t b = 0; b < branches; ++b) {
                    expected_gates[b] = ple_gate_reference(double(query[b * channels]) / std::sqrt(double(channels)), mode);
                    for (uint32_t c = 0; c < channels; ++c) expected_output[b * channels + c] = expected_gates[b] * values[c];
                }
                close(g, gates, expected_gates, 3e-7); close(g, output, expected_output, 3e-7);
                const auto padded = read(g, storage), padded_gates = read(g, gate_storage);
                require(padded[0] == -99 && padded[1] == -99 && padded[width + 2] == -99 && padded[width + 3] == -99, "gate output canary changed");
                require(padded_gates[0] == -99 && padded_gates[1] == -99 && padded_gates[branches + 2] == -99 && padded_gates[branches + 3] == -99, "gate scalar canary changed");
                require(read(g, q) == query && read(g, k) == key && read(g, v) == values, "gate changed an input");
                if (mode == PleShape{1, 1, 1, 4, 3}.gate_transform) checkpoint_default = g.read(output);
            }
            g.begin(); g.ple_gate(q, k, v, output, gates, branches, channels); static_cast<void>(g.submit()); ++batches;
            require(g.read(output) == checkpoint_default, "implicit PLE gate default differs from its compiled policy");
        }
        g.begin(); reject([&] { g.ple_gate(q, k, v, output, gates, branches, channels, PleGateTransform(2)); }); g.abort();
        g.begin(); reject([&] { g.ple_gate(q, k, v, q, gates, branches, channels, PleGateTransform::signed_root); }); g.abort();
        require(g.memory().allocations == allocations, "gate guard allocated storage");
    }
}
void guards(MetalGraph& g) {
    Arena a{g, std::mt19937(1), {}, {}};
    const std::array<float, 4> one{1, 1, 1, 1}, inf{1, 1, std::numeric_limits<float>::infinity(), 1};
    const auto valid = a.floats(one), invalid = a.floats(inf), output = a.zero(4), scales = a.zero(1), inject = a.zero(1);
    g.begin(); g.multiply(valid, invalid, output, true); reject([&] { static_cast<void>(g.submit()); });
    g.begin(); g.residual_read(valid, invalid, inject, output, scales, 1, 4); reject([&] { static_cast<void>(g.submit()); });
    const auto invalid_inject = g.slice(invalid, 8, 4);
    g.begin(); g.residual_read(valid, valid, invalid_inject, output, scales, 1, 4); reject([&] { static_cast<void>(g.submit()); });
    g.begin(); g.gated_mix(valid, invalid, output, 1, 4); reject([&] { static_cast<void>(g.submit()); });
    g.begin(); g.ple_write(valid, valid, invalid, output); reject([&] { static_cast<void>(g.submit()); });
    for (const auto mode : {PleGateTransform::plain, PleGateTransform::signed_root}) {
        g.begin(); g.ple_gate(valid, invalid, valid, output, scales, 1, 4, mode); reject([&] { static_cast<void>(g.submit()); });
        g.begin(); g.ple_gate(valid, valid, invalid, output, scales, 1, 4, mode); reject([&] { static_cast<void>(g.submit()); });
    }
    // A bad value between dilated taps must not be silently shifted into state.
    const auto history = a.floats(std::array<float, 3>{1, std::numeric_limits<float>::infinity(), 1});
    const auto scalar = g.slice(valid, 0, 4), weights = g.slice(valid, 0, 8), next = a.zero(3), out = a.zero(1);
    g.begin(); g.convolution(history, scalar, weights, next, out, 2, 3); reject([&] { static_cast<void>(g.submit()); });
    g.begin(); g.multiply(valid, valid, output, true); require(g.submit().dispatches == 1, "guard refusal did not allow valid reuse"); ++batches;
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        tokspur::MetalGraph g(argv[1], 512 * 1024 * 1024); const auto base = g.memory().live_bytes;
        connected(g); gate_modes(g); guards(g); require(g.memory().live_bytes == base, "PLE/mixer allocation leak");
        tokspur::MetalGraph limited(argv[1], 65536); const auto small_base = limited.memory().live_bytes;
        reject([&] { static_cast<void>(tokspur::allocate_ple_scratch(limited, {4, 2560, 2560, 4, 3})); });
        require(limited.memory().live_bytes == small_base, "partial PLE allocation leak");
        std::cout << "Owned PLE/mixer checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned PLE check failed: " << e.what() << '\n'; return 1; }
}
