#include "tokspur/recurrent_check.hpp"
#include "tokspur/gguf.hpp"

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
size_t checks = 0, batches = 0, compared = 0;
double worst = 0;
void require(bool b, const char* s) { ++checks; if (!b) throw std::runtime_error(s); }
void reject(const std::function<void()>& f) {
    ++checks; try { f(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid recurrent operation accepted");
}
struct Fixture {
    MetalGraph& g;
    std::mt19937 rng{0x72656331};
    std::deque<std::vector<float>> data;
    std::deque<std::vector<double>> real_data;
    std::vector<GpuView> owned;
    ~Fixture() {
        try { g.abort(); } catch (const FormatError&) {}
        for (const auto v : owned) g.release(v);
    }
    std::vector<float> values(size_t count, float scale = 0.15f) {
        std::vector<float> out(count);
        for (auto& x : out) x = (int(rng() % 2001) - 1000) / 1000.0f * scale;
        return out;
    }
    std::pair<GpuView, std::span<const double>> vector(size_t count, float scale = 0.15f, float offset = 0, bool negative = false) {
        data.push_back(values(count, scale));
        for (auto& x : data.back()) x = negative ? -std::abs(x) - offset : x + offset;
        real_data.emplace_back(data.back().begin(), data.back().end());
        const auto v = g.upload(std::as_bytes(std::span(data.back())), "generated recurrent weight"); owned.push_back(v);
        return {v, real_data.back()};
    }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(size_t count) {
        const auto [view, unused] = vector(count); static_cast<void>(unused);
        return {{view, 0}, {std::as_bytes(std::span(data.back())), 0}};
    }
    std::pair<ResidualWeights, ReferenceResidualWeights> residual(ResidualShape s) {
        const auto width = s.branches * s.channels;
        const auto norm = vector(width, 0.1f, 1);
        const auto down = matrix(width * s.rank), up = matrix(s.rank * width), inject = matrix(width * s.branches);
        return {{norm.first, down.first, up.first, inject.first}, {norm.second, down.second, up.second, inject.second}};
    }
    std::pair<GdnWeights, ReferenceGdnWeights> gdn(GdnShape s) {
        const auto qkv = matrix(s.channels * gdn_qkv_count(s)), gate = matrix(s.channels * s.value_heads * s.value_width),
            alpha = matrix(s.channels * s.value_heads), beta = matrix(s.channels * s.value_heads), output = matrix(s.value_heads * s.value_width * s.channels);
        const auto conv = vector(gdn_qkv_count(s) * s.kernel), rates = vector(s.value_heads, 0.5f, 0.05f, true),
            bias = vector(s.value_heads), gains = vector(s.value_width, 0.1f, 1);
        return {{qkv.first, gate.first, alpha.first, beta.first, output.first, conv.first, rates.first, bias.first, gains.first},
            {qkv.second, gate.second, alpha.second, beta.second, output.second, conv.second, rates.second, bias.second, gains.second}};
    }
    std::pair<MoeWeights, ReferenceMoeWeights> moe(MoeShape s) {
        const auto router = matrix(s.channels * s.experts), gate = matrix(s.channels * s.hidden * s.experts), up = matrix(s.channels * s.hidden * s.experts),
            down = matrix(s.hidden * s.channels * s.experts), sg = matrix(s.channels * s.hidden), su = matrix(s.channels * s.hidden),
            sd = matrix(s.hidden * s.channels), sr = matrix(s.channels);
        return {{router.first, gate.first, up.first, down.first, sg.first, su.first, sd.first, sr.first},
            {router.second, gate.second, up.second, down.second, sg.second, su.second, sd.second, sr.second}};
    }
    GpuView mutable_data(std::span<const float> input) {
        const auto v = g.upload(std::as_bytes(input), "recurrent test input", false); owned.push_back(v); return v;
    }
};

void complete_layers(MetalGraph& g) {
    for (const RecurrentShape shape : std::array<RecurrentShape, 3>{{{{4, 7, 3}, {7, 2, 6, 5, 3, 4}, {7, 9, 5, 3}},
        {{1, 1, 1}, {1, 1, 1, 1, 1, 2}, {1, 1, 1, 1}}, {{4, 32, 16}, {32, 4, 8, 31, 17, 3}, {32, 32, 17, 4}}}}) {
        Fixture f{g, std::mt19937(0x72656331), {}, {}, {}};
        const auto r1 = f.residual(shape.residual), r2 = f.residual(shape.residual);
        const auto a = f.gdn(shape.gdn);
        const auto m = f.moe(shape.moe);
        const RecurrentWeights w{r1.first, r2.first, a.first, m.first};
        const ReferenceLayerWeights reference{r1.second, r2.second, a.second, m.second};
        const auto scratch = allocate_recurrent_scratch(g, shape);
        auto current = allocate_gdn_state(g, shape.gdn), next = allocate_gdn_state(g, shape.gdn);
        auto input = f.mutable_data(f.values(shape.residual.branches * shape.residual.channels));
        const auto output = f.mutable_data(std::vector<float>(shape.residual.branches * shape.residual.channels));
        for (const auto mode : {RouterMode::softmax, RouterMode::sigmoid}) for (bool normalize : {false, true})
        for (const auto l2 : {L2Mode::add_squared_epsilon, L2Mode::floor_squared_epsilon}) for (float scale : {1.0f, 0.17677669f}) {
            const GdnPolicy policy{scale, 1e-6f, l2, 1e-6f};
            std::vector<double> cpu_state(gdn_state_count(shape.gdn), 0), cpu_history(gdn_qkv_count(shape.gdn) * (shape.gdn.kernel - 1), 0);
            const std::vector<float> state_zero(cpu_state.size()), history_zero(cpu_history.size());
            g.write(current.matrix, std::as_bytes(std::span(state_zero))); g.write(current.convolution, std::as_bytes(std::span(history_zero)));
            auto x = f.values(shape.residual.branches * shape.residual.channels);
            const auto allocations = g.memory().allocations;
            for (size_t t = 0; t < 4; ++t) {
                g.write(input, std::as_bytes(std::span(x)));
                const auto before_state = g.read(current.matrix), before_history = g.read(current.convolution), before_input = g.read(input);
                const auto e = recurrent_reference(reference, std::vector<double>(x.begin(), x.end()), cpu_state, cpu_history, shape, policy, mode, normalize);
                g.begin(); encode_recurrent_layer(g, w, scratch, current, next, input, output, shape, policy, mode, normalize);
                const auto result = g.submit(); ++batches;
                require(result.dispatches == 40 && g.memory().allocations == allocations, "layer dispatch/allocation invariant");
                require(g.read(current.matrix) == before_state && g.read(current.convolution) == before_history && g.read(input) == before_input, "accepted state or input mutated before publication");
                const auto comparison = compare_recurrent_layer(g, scratch, next, output, e);
                compared += comparison.compared_values; worst = std::max(worst, comparison.maximum);
                if (!comparison.passed) for (const auto& [name, error] : comparison.scaled_errors) if (error > 5e-5) std::cerr << name << ": " << error << '\n';
                require(comparison.passed && comparison.scaled_errors.size() == 31, "connected recurrent double comparison failed");
                cpu_state = e.attention.delta.state; cpu_history = e.attention.convolution.history;
                std::swap(current, next);
                // Perturb the next token's branch state rather than repeatedly
                // testing an identical projection or only an empty history.
                for (size_t i = 0; i < x.size(); ++i) x[i] = float(e.output_streams[i]) + float(int(f.rng() % 7) - 3) * 0.02f;
            }
        }
        // Discarded encoding must leave both accepted and candidate buffers unchanged.
        const auto accepted = g.read(current.matrix), candidate = g.read(next.matrix), old_output = g.read(output);
        g.begin(); encode_recurrent_layer(g, w, scratch, current, next, input, output, shape, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true); g.abort();
        require(g.read(current.matrix) == accepted && g.read(next.matrix) == candidate && g.read(output) == old_output, "aborted layer executed writes");
        g.begin(); reject([&] { encode_recurrent_layer(g, w, scratch, current, current, input, output, shape, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true); });
        require(g.read(current.matrix) == accepted, "alias refusal damaged accepted state");
        auto bad = w; bad.ffn.down.encoding = 0xffffffffu; // Fail late, after earlier dispatches were encoded.
        g.begin(); reject([&] { encode_recurrent_layer(g, bad, scratch, current, next, input, output, shape, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true); });
        require(g.read(next.matrix) == candidate && g.read(output) == old_output, "late encode refusal executed a partial batch");
        // A device-side failure may dirty candidate buffers but not current state.
        auto positive_rate = f.mutable_data(std::vector<float>(shape.gdn.value_heads, 1));
        bad = w; bad.attention.negative_rates = positive_rate;
        g.begin(); encode_recurrent_layer(g, bad, scratch, current, next, input, output, shape, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true);
        reject([&] { static_cast<void>(g.submit()); });
        require(g.read(current.matrix) == accepted, "failed device batch mutated accepted state");
        g.begin(); encode_recurrent_layer(g, w, scratch, current, next, input, output, shape, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true);
        require(g.submit().dispatches == 40, "valid layer reuse after device error"); ++batches;
        release_recurrent_scratch(g, scratch); release_gdn_state(g, current); release_gdn_state(g, next);
    }
}

void preparation_guards(MetalGraph& g) {
    Fixture f{g, std::mt19937(1), {}, {}, {}};
    const GdnShape s{1, 1, 1, 2, 3, 2}; const GdnPolicy p{1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
    const auto input = f.mutable_data(std::vector<float>(7, 1e30f)), q = f.mutable_data(std::vector<float>(2)), k = f.mutable_data(std::vector<float>(2)), v = f.mutable_data(std::vector<float>(3));
    g.begin(); g.gdn_prepare(input, q, k, v, s, p); reject([&] { static_cast<void>(g.submit()); });
    const std::array<float, 7> valid{0, 0, 0, 0, 1, 2, 3};
    g.write(input, std::as_bytes(std::span(valid)));
    g.begin(); g.gdn_prepare(input, q, k, v, s, p); require(g.submit().dispatches == 1, "valid preparation reuse"); ++batches;
    g.begin(); reject([&] { g.gdn_prepare(input, q, q, v, s, p); }); g.abort();
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        tokspur::MetalGraph g(argv[1], 256 * 1024 * 1024);
        const auto initial = g.memory().live_bytes;
        complete_layers(g); preparation_guards(g);
        require(g.memory().live_bytes == initial, "recurrent graph allocation leak");
        tokspur::MetalGraph limited(argv[1], 131072);
        const auto base = limited.memory().live_bytes;
        reject([&] { static_cast<void>(tokspur::allocate_recurrent_scratch(limited, {{4, 2560, 320}, {2560, 16, 48, 128, 128, 4}, {2560, 640, 512, 10}})); });
        require(limited.memory().live_bytes == base, "partial recurrent allocation leaked");
        std::cout << "Owned recurrent checks passed: " << checks << ", compared values: " << compared << ", batches: " << batches << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned recurrent check failed: " << e.what() << '\n'; return 1; }
}
