#include "tokspur/recurrent_check.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/recurrent_replay_timing.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Generated component data only. No model, service, timing qualification or
// automatic TextStack replay is exercised by this executable.
namespace {
using namespace tokspur;
size_t checks = 0, batches = 0, compared = 0;
double worst = 0;
void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
void reject(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid replay operation accepted");
}
void abort_if_active(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
template<class T> std::vector<T> read(MetalGraph& g, GpuView view) {
    const auto bytes = g.read(view);
    require(bytes.size() % sizeof(T) == 0, "unaligned replay readback");
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size()); return out;
}
void close(MetalGraph& g, GpuView view, std::span<const double> expected) {
    const auto actual = read<float>(g, view);
    require(actual.size() == expected.size(), "replay reference size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i]));
        worst = std::max(worst, error); ++compared;
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]) && error <= 5e-5, "replay double reference mismatch");
    }
}
void exact(MetalGraph& g, GpuView a, GpuView b) {
    const auto first = g.read(a), second = g.read(b); compared += first.size() / 4;
    require(first == second, "A/B/C replay bytes differ");
}
template<size_t N> void exact(MetalGraph& g, const std::array<GpuView, N>& a, const std::array<GpuView, N>& b) {
    for (size_t i = 0; i < N; ++i) exact(g, a[i], b[i]);
}
void done(MetalGraph& g, uint64_t dispatches, uint64_t executes, uint64_t allocations) {
    const auto result = g.submit(); ++batches;
    require(result.dispatches == dispatches && result.capture_executes == executes, "replay physical/execute count mismatch");
    require(g.memory().allocations == allocations, "replay allocated resources during encoding/execution");
    require(std::isfinite(result.wall_ms) && result.wall_ms > 0 && std::isfinite(result.gpu_ms) && result.gpu_ms > 0,
        "invalid replay diagnostic timestamps");
}
void begin_arm(MetalGraph& g, size_t arm) {
    if (arm == 1) g.begin_indirect_compatible(); else g.begin();
}

struct Guarded { GpuView full, body; };
struct Fixture {
    MetalGraph& g;
    std::mt19937 rng{0x69636231};
    std::deque<std::vector<float>> data;
    std::deque<std::vector<double>> real_data;
    std::vector<GpuView> owned;
    std::vector<GraphCapture> captures;
    ~Fixture() {
        abort_if_active(g);
        for (auto i = captures.rbegin(); i != captures.rend(); ++i) { try { g.release_capture(*i); } catch (const FormatError&) {} }
        for (auto i = owned.rbegin(); i != owned.rend(); ++i) { try { g.release(*i); } catch (const FormatError&) {} }
    }
    std::vector<float> values(size_t count, float scale = 0.025f) {
        std::vector<float> out(count);
        for (auto& x : out) x = (int(rng() % 2001) - 1000) / 1000.0f * scale;
        return out;
    }
    GpuView upload(std::span<const float> input, bool immutable = true) {
        const auto v = g.upload(std::as_bytes(input), "generated replay data", immutable); owned.push_back(v); return v;
    }
    GpuView bytes(std::span<const std::byte> input) {
        const auto v = g.upload(input, "generated replay encoded matrix"); owned.push_back(v); return v;
    }
    GpuView zero(size_t count) {
        const auto v = g.zeros(count * sizeof(float), "generated replay scratch"); owned.push_back(v); return v;
    }
    template<class Range> void keep(const Range& views) { for (const auto v : views) owned.push_back(v); }
    Guarded guarded(size_t count) {
        const auto full = upload(std::vector<float>(count + 8, 91.25f), false);
        return {full, g.slice(full, 4 * sizeof(float), count * sizeof(float))};
    }
    void canaries(Guarded view) {
        const auto all = read<float>(g, view.full);
        for (size_t i = 0; i < 4; ++i) require(all[i] == 91.25f && all[all.size() - 1 - i] == 91.25f, "replay damaged boundary canary");
    }
    GraphCapture capture(uint32_t capacity, const std::function<void()>& encode) {
        g.begin_capture(capacity); encode(); const auto result = g.finish_capture(); captures.push_back(result); return result;
    }
    void release(GraphCapture capture) {
        g.release_capture(capture);
        const auto found = std::find_if(captures.begin(), captures.end(), [&](auto h) { return h.owner == capture.owner && h.slot == capture.slot; });
        require(found != captures.end(), "test lost capture ownership"); captures.erase(found);
    }
    std::pair<GpuView, std::span<const double>> vector(size_t count, float scale = 0.025f, float offset = 0, bool negative = false) {
        data.push_back(values(count, scale));
        for (auto& x : data.back()) x = negative ? -std::abs(x) - offset : x + offset;
        real_data.emplace_back(data.back().begin(), data.back().end());
        return {upload(data.back()), real_data.back()};
    }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(size_t count) {
        const auto [view, unused] = vector(count); static_cast<void>(unused);
        return {{view, 0}, {std::as_bytes(std::span(data.back())), 0}};
    }
    std::pair<ResidualWeights, ReferenceResidualWeights> residual(ResidualShape s) {
        const auto width = s.branches * s.channels;
        const auto norm = vector(width, 0.05f, 1);
        const auto down = matrix(width * s.rank), up = matrix(s.rank * width), inject = matrix(width * s.branches);
        return {{norm.first, down.first, up.first, inject.first}, {norm.second, down.second, up.second, inject.second}};
    }
    std::pair<GdnWeights, ReferenceGdnWeights> gdn(GdnShape s) {
        const auto qkv = matrix(s.channels * gdn_qkv_count(s)), gate = matrix(s.channels * s.value_heads * s.value_width),
            alpha = matrix(s.channels * s.value_heads), beta = matrix(s.channels * s.value_heads), output = matrix(s.value_heads * s.value_width * s.channels);
        const auto conv = vector(gdn_qkv_count(s) * s.kernel), rates = vector(s.value_heads, 0.5f, 0.05f, true),
            bias = vector(s.value_heads), gains = vector(s.value_width, 0.05f, 1);
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
};

void chains(MetalGraph& g) {
    Fixture f{g, std::mt19937(7), {}, {}, {}, {}};
    constexpr size_t count = 131; // Nonuniform dispatch tail and padded subviews.
    const auto x = f.guarded(count), first = f.guarded(count), middle = f.guarded(count), other = f.guarded(count), last = f.guarded(count);
    const auto part1 = [&] {
        float local_scale = 0.5f;
        g.activate(first.body, middle.body, Activation::identity, local_scale);
        local_scale = 0.25f; g.activate(middle.body, middle.body, Activation::identity, local_scale);
    };
    const auto part2 = [&] {
        float local_scale = 0.75f;
        g.activate(middle.body, other.body, Activation::identity, local_scale);
        local_scale = 1; g.activate(other.body, middle.body, Activation::identity, local_scale);
    };
    const auto whole = f.capture(4, [&] { part1(); part2(); });
    const auto left = f.capture(2, part1), right = f.capture(2, part2);
    // Caller locals and temporary parameter arrays have died before replay.
    // The second capture reads the first's result, then overwrites that resource
    // after its read: capture->capture RAW/WAR/WAW, including exact in-place use.
    const auto allocations = g.memory().allocations;
    for (size_t repeat = 0; repeat < 3; ++repeat) {
        const auto input = f.values(count, 2); g.write(x.body, std::as_bytes(std::span(input)));
        std::vector<std::vector<std::byte>> expected;
        for (size_t arm = 0; arm < 5; ++arm) {
            begin_arm(g, arm);
            g.activate(x.body, first.body, Activation::identity, 2);
            if (arm < 2) { part1(); part2(); }
            else if (arm == 2) g.execute_capture(whole);
            else if (arm == 3) { g.execute_capture(left); g.execute_capture(right); }
            else { g.execute_capture(whole); g.execute_capture(whole); }
            g.activate(middle.body, last.body, Activation::identity, 4);
            done(g, arm == 4 ? 10 : 6, arm < 2 ? 0 : arm == 2 ? 1 : 2, allocations);
            std::vector<std::vector<std::byte>> actual;
            for (auto v : {first.body, middle.body, other.body, last.body}) actual.push_back(g.read(v));
            if (!arm) expected = actual; else require(actual == expected, "direct/replay dependency chain differs");
            std::vector<double> reference(input.begin(), input.end()); for (auto& value : reference) value *= 0.75;
            close(g, last.body, reference);
            for (auto v : {x, first, middle, other, last}) f.canaries(v);
        }
        const auto old = g.read(last.full);
        g.begin(); g.execute_capture(whole); g.abort();
        require(g.read(last.full) == old, "aborted replay executed GPU writes");
    }
}

void encoded_formats(MetalGraph& g) {
    Fixture f{g, std::mt19937(8), {}, {}, {}, {}};
    for (uint32_t type : {8u, 12u}) {
        const auto& format = encoding(type); constexpr uint32_t rows = 7;
        const uint32_t columns = format.elements_per_block * 3;
        std::vector<std::byte> encoded(rows * 3 * format.bytes_per_block);
        for (size_t i = 0; i < encoded.size(); ++i) encoded[i] = std::byte((i * 37 + 19) & 255);
        for (size_t offset = 0; offset < encoded.size(); offset += format.bytes_per_block) {
            encoded[offset] = std::byte(0); encoded[offset + 1] = std::byte(0x24);
            if (type == 12) { encoded[offset + 2] = std::byte(0); encoded[offset + 3] = std::byte(0x20); }
        }
        const auto matrix = f.bytes(encoded);
        const auto input = f.guarded(columns), output = f.guarded(rows);
        const auto encode = [&] { g.gemv(type, matrix, input.body, output.body, columns, rows); };
        const auto capture = f.capture(1, encode); const auto allocations = g.memory().allocations;
        for (size_t repeat = 0; repeat < 2; ++repeat) {
            const auto values = f.values(columns); g.write(input.body, std::as_bytes(std::span(values)));
            const auto reference = reference_gemv(type, encoded, columns, rows, values);
            std::vector<std::byte> expected;
            for (size_t arm = 0; arm < 3; ++arm) {
                begin_arm(g, arm); if (arm == 2) g.execute_capture(capture); else encode();
                done(g, 1, arm == 2, allocations);
                if (!arm) expected = g.read(output.body); else require(g.read(output.body) == expected, "block3 replay differs");
                close(g, output.body, reference); f.canaries(input); f.canaries(output);
            }
        }
    }
}

void dynamic_routes(MetalGraph& g) {
    Fixture f{g, std::mt19937(9), {}, {}, {}, {}};
    constexpr uint32_t experts = 5, selected = 3, columns = 32, rows = 7;
    const auto matrices = f.upload(f.values(experts * columns * rows));
    const auto input = f.upload(f.values(columns)), logits = f.zero(experts), indices = f.zero(selected), weights = f.zero(selected);
    const auto output = f.guarded(selected * rows);
    const auto encode = [&] {
        g.route(logits, indices, weights, selected, RouterMode::softmax, true);
        g.routed_gemv(0, matrices, indices, input, output.body, columns, rows, experts, true);
    };
    const auto capture = f.capture(2, encode); const auto allocations = g.memory().allocations;
    std::vector<uint32_t> previous;
    for (const auto& values : std::array<std::array<float, experts>, 3>{{{5, 4, 3, 2, 1}, {-3, 6, -2, 5, 4}, {1, 1, 1, 1, 1}}}) {
        g.write(logits, std::as_bytes(std::span(values)));
        const auto reference = route_reference(std::vector<double>(values.begin(), values.end()), selected, RouterMode::softmax, true);
        std::vector<std::byte> expected;
        for (size_t arm = 0; arm < 3; ++arm) {
            begin_arm(g, arm); if (arm == 2) g.execute_capture(capture); else encode();
            done(g, 2, arm == 2, allocations);
            require(read<uint32_t>(g, indices) == reference.indices, "captured router froze IDs or tie order");
            close(g, weights, reference.weights);
            if (!arm) expected = g.read(output.body); else require(g.read(output.body) == expected, "dynamic routed projection replay differs");
            f.canaries(output);
        }
        if (!previous.empty()) require(previous != reference.indices, "dynamic route fixture failed to change selected IDs");
        previous = reference.indices;
    }
}

void compare_gdn(MetalGraph& g, const GdnScratch& s, const GdnState& next, GpuView output, const ReferenceGdn& e) {
    close(g, s.projected, e.projected); close(g, s.gate, e.gate); close(g, s.alpha, e.alpha); close(g, s.beta, e.beta);
    auto activated = e.convolution.output; for (auto& value : activated) value = stable_silu(value);
    close(g, s.convolved, activated); close(g, s.query, e.prepared.query); close(g, s.key, e.prepared.key); close(g, s.value, e.prepared.value);
    close(g, s.decay, e.gates.decay); close(g, s.write_gate, e.gates.beta); close(g, s.raw, e.delta.output);
    close(g, s.normalized, e.normalized); close(g, s.gated, e.gated); close(g, output, e.output);
    close(g, next.matrix, e.delta.state); close(g, next.convolution, e.convolution.history);
}

// Simulate actual pair publication/reset (including reset's swap) rather than
// choosing a capture from prefix parity. This is not a TextStack integration test.
void recurrent_components(MetalGraph& g, bool full_layer) {
    Fixture f{g, std::mt19937(full_layer ? 11 : 10), {}, {}, {}, {}};
    const RecurrentShape shape{{2, 1024, 5}, {1024, 2, 4, 8, 7, 4}, {1024, 9, 5, 3}};
    const GdnPolicy policy{1 / std::sqrt(8.0f), 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
    const auto attention = f.gdn(shape.gdn);
    RecurrentWeights layer{}; ReferenceLayerWeights reference_layer{};
    if (full_layer) {
        const auto r1 = f.residual(shape.residual), r2 = f.residual(shape.residual);
        const auto moe = f.moe(shape.moe);
        layer = {r1.first, r2.first, attention.first, moe.first};
        reference_layer = {r1.second, r2.second, attention.second, moe.second};
    }
    struct Arm { RecurrentScratch scratch{}; std::array<GdnState, 2> state{}; Guarded input{}, output{}; };
    std::array<Arm, 3> arms;
    const auto input_count = full_layer ? shape.residual.channels * shape.residual.branches : shape.gdn.channels;
    for (auto& arm : arms) {
        if (full_layer) { arm.scratch = allocate_recurrent_scratch(g, shape); f.keep(arm.scratch.views()); }
        else { arm.scratch.gdn = allocate_gdn_scratch(g, shape.gdn); f.keep(arm.scratch.gdn.views()); }
        for (auto& state : arm.state) { state = allocate_gdn_state(g, shape.gdn); f.keep(state.views()); }
        arm.input = f.guarded(input_count); arm.output = f.guarded(input_count);
    }
    const auto encode = [&](size_t a, size_t parity) {
        auto& arm = arms[a];
        if (full_layer) encode_recurrent_layer(g, layer, arm.scratch, arm.state[parity], arm.state[1 - parity], arm.input.body, arm.output.body,
            shape, policy, RouterMode::softmax, true);
        else encode_gdn(g, attention.first, arm.scratch.gdn, arm.state[parity], arm.state[1 - parity], arm.input.body, arm.output.body, shape.gdn, policy);
    };
    // Four GDN projections expand at split16. Two residual reads add four
    // passes; MoE router/shared gate/shared up/shared scalar add another four.
    const uint64_t expected_dispatches = full_layer ? 52 : 16;
    const auto pristine_output = g.read(arms[2].output.full);
    const auto pristine_state = g.read(arms[2].state[1].matrix);
    const auto cap0 = f.capture(uint32_t(expected_dispatches), [&] { encode(2, 0); });
    const auto cap1 = f.capture(uint32_t(expected_dispatches), [&] { encode(2, 1); });
    const std::array<GraphCapture, 2> captures{cap0, cap1};
    require(g.read(arms[2].output.full) == pristine_output && g.read(arms[2].state[1].matrix) == pristine_state,
        "capture preparation executed component work");
    const auto allocations = g.memory().allocations;
    std::vector<double> cpu_state(gdn_state_count(shape.gdn), 0), cpu_history(gdn_qkv_count(shape.gdn) * 3, 0);
    const std::vector<float> state_zero(cpu_state.size()), history_zero(cpu_history.size());
    size_t parity = 0; uint32_t prefix = 0; bool parity_differs = false;
    const auto reset = [&] {
        for (auto& arm : arms) {
            g.write(arm.state[1 - parity].matrix, std::as_bytes(std::span(state_zero)));
            g.write(arm.state[1 - parity].convolution, std::as_bytes(std::span(history_zero)));
        }
        parity = 1 - parity; prefix = 0;
        std::fill(cpu_state.begin(), cpu_state.end(), 0); std::fill(cpu_history.begin(), cpu_history.end(), 0);
    };
    for (size_t step = 0; step < 7; ++step) {
        if (step == 1 || step == 4 || step == 5) reset();
        if (step == 5) reset(); // Repeated reset must not assume parity zero.
        parity_differs = parity_differs || parity != prefix % 2;
        const auto values = f.values(input_count, 0.3f);
        for (auto& arm : arms) g.write(arm.input.body, std::as_bytes(std::span(values)));
        const std::vector<double> x(values.begin(), values.end());
        ReferenceGdn expected_gdn; ReferenceLayer expected_layer;
        if (full_layer) expected_layer = recurrent_reference(reference_layer, x, cpu_state, cpu_history, shape, policy, RouterMode::softmax, true);
        else expected_gdn = gdn_projection_reference(attention.second, x, cpu_state, cpu_history, shape.gdn, policy);
        for (size_t a = 0; a < arms.size(); ++a) {
            auto& arm = arms[a]; const auto accepted = g.read(arm.state[parity].matrix), history = g.read(arm.state[parity].convolution);
            begin_arm(g, a); if (a == 2) g.execute_capture(captures[parity]); else encode(a, parity);
            done(g, expected_dispatches, a == 2, allocations);
            require(g.read(arm.state[parity].matrix) == accepted && g.read(arm.state[parity].convolution) == history,
                "component mutated accepted state before publication");
            if (full_layer) {
                const auto comparison = compare_recurrent_layer(g, arm.scratch, arm.state[1 - parity], arm.output.body, expected_layer);
                compared += comparison.compared_values; worst = std::max(worst, comparison.maximum);
                require(comparison.passed && comparison.scaled_errors.size() == 31, "replayed full layer double reference failed");
            } else compare_gdn(g, arm.scratch.gdn, arm.state[1 - parity], arm.output.body, expected_gdn);
            if (a) {
                if (full_layer) exact(g, arms[0].scratch.views(), arm.scratch.views());
                else exact(g, arms[0].scratch.gdn.views(), arm.scratch.gdn.views());
                exact(g, arms[0].state[1 - parity].views(), arm.state[1 - parity].views());
                exact(g, arms[0].output.body, arm.output.body);
            }
            f.canaries(arm.input); f.canaries(arm.output);
        }
        if (full_layer && step == 0) {
            // Same logical 32 scratch + output/matrix/history extents as the
            // N1 timing worker. Guarded output.body excludes boundary padding.
            const auto written = [&](const Arm& arm) {
                std::array<GpuView, 35> result; const auto scratch = arm.scratch.views();
                std::copy(scratch.begin(), scratch.end(), result.begin());
                result[32] = arm.output.body; result[33] = arm.state[1 - parity].matrix;
                result[34] = arm.state[1 - parity].convolution; return result;
            };
            std::array<std::vector<std::byte>, 35> expected, poison;
            const auto baseline_views = written(arms[0]);
            for (size_t i = 0; i < baseline_views.size(); ++i) {
                expected[i] = g.read(baseline_views[i]); poison[i] = recurrent_replay_poison_bytes(expected[i]);
            }
            const auto fixed = g.memory();
            for (size_t a = 0; a < arms.size(); ++a) {
                auto& arm = arms[a]; const auto views = written(arm);
                const auto input = g.read(arm.input.full), accepted = g.read(arm.state[parity].matrix), history = g.read(arm.state[parity].convolution);
                const auto unchanged = [&] {
                    require(g.read(arm.input.full) == input && g.read(arm.state[parity].matrix) == accepted &&
                        g.read(arm.state[parity].convolution) == history, "freshness reset/execution mutated input or accepted base");
                    const auto now = g.memory();
                    require(now.live_bytes == fixed.live_bytes && now.peak_bytes == fixed.peak_bytes && now.allocations == fixed.allocations &&
                        now.capture_resource_bytes == fixed.capture_resource_bytes && now.live_captures == fixed.live_captures &&
                        now.imported_backing_bytes == fixed.imported_backing_bytes && now.profile_counter_payload_bytes == fixed.profile_counter_payload_bytes,
                        "freshness reset/execution changed buffer/capture ledger");
                    f.canaries(arm.input); f.canaries(arm.output);
                };
                const auto poison_all = [&] {
                    for (size_t i = 0; i < views.size(); ++i) g.write(views[i], poison[i]);
                };
                const auto exact_current = [&] {
                    for (size_t i = 0; i < views.size(); ++i) {
                        const auto actual = g.read(views[i]);
                        if (actual != expected[i]) throw FormatError("freshness current-write bits differ");
                    }
                };
                // Negative controls are fixture-only host writes/no execution;
                // there are no production debug flags or missing GPU commands.
                poison_all(); reject(exact_current); unchanged();
                for (size_t i = 0; i < views.size(); ++i) if (i != 32) g.write(views[i], expected[i]);
                reject(exact_current); unchanged(); // Complete output view unwritten.
                g.write(views[32], expected[32]); exact_current();
                const auto last_word = g.slice(views[33], views[33].bytes - 4, 4);
                g.write(last_word, std::span<const std::byte>(poison[33]).last(4));
                reject(exact_current); unchanged(); // One candidate-matrix word unwritten.
                // Twice exercises both recovery from the negative controls and
                // repeated request invalidation after a previously correct bank.
                for (uint32_t repeat = 0; repeat < 2; ++repeat) {
                    poison_all(); reject(exact_current);
                    begin_arm(g, a); if (a == 2) g.execute_capture(captures[parity]); else encode(a, parity);
                    done(g, expected_dispatches, a == 2, allocations); exact_current(); unchanged();
                    const auto comparison = compare_recurrent_layer(g, arm.scratch, arm.state[1 - parity], arm.output.body, expected_layer);
                    compared += comparison.compared_values; worst = std::max(worst, comparison.maximum);
                    require(comparison.passed && comparison.scaled_errors.size() == 31, "freshness recovery lost independent layer reference");
                }
            }
        }
        const auto& expected = full_layer ? expected_layer.attention : expected_gdn;
        cpu_state = expected.delta.state; cpu_history = expected.convolution.history;
        parity = 1 - parity; ++prefix; // Publish only after every check succeeds.
    }
    require(parity_differs, "reset fixture did not distinguish physical/prefix parity");
    const auto accepted = g.read(arms[2].state[parity].matrix), candidate = g.read(arms[2].state[1 - parity].matrix), old_output = g.read(arms[2].output.full);
    g.begin(); g.execute_capture(captures[parity]); g.abort();
    require(g.read(arms[2].state[parity].matrix) == accepted && g.read(arms[2].state[1 - parity].matrix) == candidate &&
        g.read(arms[2].output.full) == old_output, "aborted component replay executed");
    if (full_layer) {
        auto bad = layer; bad.ffn.down.encoding = 0xffffffffu;
        const auto before = g.memory(); g.begin_capture(64);
        reject([&] { encode_recurrent_layer(g, bad, arms[2].scratch, arms[2].state[parity], arms[2].state[1 - parity], arms[2].input.body,
            arms[2].output.body, shape, policy, RouterMode::softmax, true); });
        reject([&] { static_cast<void>(g.finish_capture()); });
        require(g.memory().live_bytes == before.live_bytes && g.memory().live_captures == before.live_captures &&
            g.memory().capture_resource_bytes == before.capture_resource_bytes, "late composite refusal retained draft resources");
        require(g.read(arms[2].state[parity].matrix) == accepted && g.read(arms[2].output.full) == old_output,
            "late capture refusal performed GPU work");
    }
}

void safety(MetalGraph& g, const std::filesystem::path& library) {
    Fixture f{g, std::mt19937(12), {}, {}, {}, {}};
    const std::array<float, 7> values{1, 2, 3, 4, 5, 6, 7};
    const auto x = f.upload(values, false), y = f.zero(values.size()), immutable = f.upload(values), scratch = f.zero(values.size() + 1);
    const auto before = g.memory();
    reject([&] { g.begin_capture(0); }); reject([&] { g.begin_capture(1025); });
    reject([&] { static_cast<void>(g.finish_capture()); }); reject([&] { g.execute_capture({}); }); reject([&] { g.release_capture({}); });
    g.begin(); reject([&] { g.begin_capture(1); }); g.abort();
    g.begin_capture(1); reject([&] { static_cast<void>(g.finish_capture()); });
    require(g.memory().live_bytes == before.live_bytes && g.memory().capture_resource_bytes == before.capture_resource_bytes, "empty capture leaked");
    const auto old_output = g.read(y);
    g.begin_capture(1); g.activate(x, y, Activation::identity);
    reject([&] { g.activate(x, y, Activation::identity); }); // Primitive refusal requires explicit abort.
    g.abort(); require(g.read(y) == old_output, "overflowed/aborted capture executed");
    require(g.memory().live_bytes == before.live_bytes && g.memory().live_captures == before.live_captures, "aborted draft leaked");
    g.begin_capture(1);
    reject([&] { g.begin_capture(1); }); reject([&] { g.begin(); }); reject([&] { g.begin_indirect_compatible(); });
    reject([&] { static_cast<void>(g.submit()); }); reject([&] { static_cast<void>(g.read(x)); });
    reject([&] { g.write(x, std::as_bytes(std::span(values))); }); reject([&] { g.release(x); });
    reject([&] { static_cast<void>(g.zeros(4, "during capture")); }); reject([&] { g.configure_block_gemv(0); });
    reject([&] { g.activate(immutable, immutable, Activation::identity); });
    reject([&] { g.activate(g.slice(scratch, 0, 28), g.slice(scratch, 4, 28), Activation::identity); });
    auto foreign_view = y; ++foreign_view.owner;
    reject([&] { g.activate(x, foreign_view, Activation::identity); });
    g.abort();
    const auto encode = [&] { g.activate(x, y, Activation::identity); };
    const auto capture = f.capture(1, encode), second = f.capture(1, encode);
    require(g.memory().live_captures == before.live_captures + 2 && g.memory().capture_resource_bytes > before.capture_resource_bytes,
        "capture resource ledger missing");
    reject([&] { g.release(x); }); reject([&] { g.release(y); });
    const auto unleased = g.zeros(4, "unleased"); g.release(unleased);
    auto foreign = capture; ++foreign.owner; auto invalid = capture; invalid.slot = std::numeric_limits<size_t>::max();
    g.begin(); reject([&] { g.execute_capture(foreign); }); reject([&] { g.execute_capture(invalid); });
    reject([&] { g.release_capture(capture); }); reject([&] { static_cast<void>(g.finish_capture()); });
    g.execute_capture(capture); g.abort(); require(g.read(y) == old_output, "discarded captured batch executed");
    reject([&] { g.release_capture(foreign); }); reject([&] { g.release_capture(invalid); });
    g.begin_indirect_compatible(); reject([&] { g.execute_capture(capture); }); g.abort();
    bool thread_refused = false;
    std::thread other([&] { try { g.release_capture(capture); } catch (const FormatError&) { thread_refused = true; } });
    other.join(); require(thread_refused, "capture cross-thread release accepted");
    g.begin(); thread_refused = false;
    std::thread executor([&] { try { g.execute_capture(capture); } catch (const FormatError&) { thread_refused = true; } });
    executor.join(); require(thread_refused, "capture cross-thread replay accepted"); g.abort();
    const auto check_replay = [&] {
        const auto allocations = g.memory().allocations; g.begin(); g.execute_capture(capture); done(g, 1, 1, allocations);
        require(g.read(y) == g.read(x), "restored policy replay failed");
    };
    g.configure_block_gemv(1); g.begin(); reject([&] { g.execute_capture(capture); }); g.abort(); g.configure_block_gemv(3); check_replay();
    g.configure_small_gemv(8); g.begin(); reject([&] { g.execute_capture(capture); }); g.abort(); g.configure_small_gemv(16); check_replay();
    require(g.configure_profiling(8) != ProfileMode::disabled, "fixture lacks diagnostic profiling support");
    reject([&] { g.begin_capture(1); }); reject([&] { g.begin_indirect_compatible(); });
    g.begin(); reject([&] { g.execute_capture(capture); }); g.abort(); static_cast<void>(g.configure_profiling(0)); check_replay();
    // A failing device command may dirty candidates, never an immutable input;
    // the next ordinary begin must clear its error flag before valid replay.
    auto nonfinite = values; nonfinite[3] = std::numeric_limits<float>::infinity();
    g.write(x, std::as_bytes(std::span(nonfinite))); g.begin(); g.execute_capture(capture); reject([&] { static_cast<void>(g.submit()); });
    require(g.read(immutable) == std::vector<std::byte>(std::as_bytes(std::span(values)).begin(), std::as_bytes(std::span(values)).end()),
        "failed replay mutated protected input");
    g.write(x, std::as_bytes(std::span(values))); check_replay();
    f.release(capture); reject([&] { g.release_capture(capture); });
    g.begin(); reject([&] { g.execute_capture(capture); }); g.abort(); reject([&] { g.release(x); });
    f.release(second);
    const auto newer = f.capture(1, encode); require(newer.slot > second.slot, "released capture slot was recycled"); f.release(newer);
    require(g.memory().live_captures == before.live_captures && g.memory().capture_resource_bytes == before.capture_resource_bytes,
        "released capture ledger did not drain");
    MetalGraph limited(library, 16384); const auto limited_before = limited.memory();
    reject([&] { limited.begin_capture(1); });
    require(limited.memory().live_bytes == limited_before.live_bytes && !limited.memory().live_captures && !limited.memory().capture_resource_bytes,
        "capture budget refusal leaked");
}

// A tiny original GGUF data fixture, following the owned parser test's declared
// wire format. It is not a checkpoint, converter or external implementation.
class GeneratedMappedFile {
public:
    std::filesystem::path directory, path;
    GeneratedMappedFile() {
        const auto prefix = (std::filesystem::temp_directory_path() / "tokspur-replay-mmap-XXXXXX").string();
        std::vector<char> pattern(prefix.begin(), prefix.end()); pattern.push_back('\0');
        const auto* made = mkdtemp(pattern.data());
        if (!made) throw std::runtime_error("cannot create generated replay fixture directory");
        directory = made; path = directory / "fixture.gguf";
        std::vector<std::byte> bytes;
        const auto number = [&](uint64_t value, size_t width) {
            for (size_t i = 0; i < width; ++i) bytes.push_back(std::byte((value >> (8 * i)) & 255));
        };
        const auto text = [&](std::string_view value) {
            number(value.size(), 8); for (const unsigned char c : value) bytes.push_back(std::byte(c));
        };
        number(0x46554747, 4); number(3, 4); number(1, 8); number(1, 8);
        text("general.architecture"); number(8, 4); text("tokspur-generated-replay");
        text("weight"); number(2, 4); number(4, 8); number(3, 8); number(0, 4); number(0, 8);
        bytes.resize((bytes.size() + 31) / 32 * 32);
        const std::array<float, 12> weights{1, 2, 3, 4, -1, 0, 1, 2, 0.5f, -0.5f, 0.25f, -0.25f};
        const auto payload = std::as_bytes(std::span(weights)); bytes.insert(bytes.end(), payload.begin(), payload.end());
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("cannot write generated mapped replay fixture");
    }
    ~GeneratedMappedFile() {
        std::error_code ignored; std::filesystem::remove(path, ignored); std::filesystem::remove(directory, ignored);
    }
};

void mapped_lifetime(MetalGraph& g) {
    GeneratedMappedFile file;
    Fixture f{g, std::mt19937(13), {}, {}, {}, {}};
    const std::array<float, 4> values{0.5f, -1, 2, -0.5f};
    const auto input = f.upload(values), output = f.zero(3);
    const auto before = g.memory();
    auto source = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{file.path});
    std::weak_ptr<ModelFiles> weak_source = source;
    const std::array<TensorSlice, 1> slices{{{"weight", 0, 0}}};
    const auto page = uint64_t(getpagesize());
    reject([&] { static_cast<void>(g.import_weights(source, slices, page - 4)); });
    reject([&] { static_cast<void>(g.import_weights(source, slices, 1)); });
    require(g.memory().live_bytes == before.live_bytes && g.memory().allocations == before.allocations,
        "invalid bounded import changed the resource ledger");
    const auto imported = g.import_weights(source, slices, page); f.keep(imported.regions);
    require(imported.regions.size() == 1 && imported.tensors.size() == 1 && g.memory().imported_backing_bytes > before.imported_backing_bytes,
        "generated mapped import was not ledgered");
    const auto capture = f.capture(1, [&] { g.gemv(0, imported.tensors[0], input, output, 4, 3); });
    source.reset();
    require(!source && !weak_source.expired() && weak_source.use_count() >= 1, "mapped source owner died after caller reset");
    const auto retained = g.memory();
    reject([&] { g.release(imported.regions[0]); });
    require(!weak_source.expired() && g.memory().live_bytes == retained.live_bytes &&
        g.memory().imported_backing_bytes == retained.imported_backing_bytes, "leased region release changed source ownership or ledger");
    const auto allocations = g.memory().allocations;
    g.begin(); g.execute_capture(capture); done(g, 1, 1, allocations);
    close(g, output, std::array<double, 3>{2.5, 0.5, 1.375});
    f.release(capture);
    require(!weak_source.expired(), "capture release prematurely dropped the still-imported source");
    for (const auto region : imported.regions) {
        g.release(region);
        std::erase_if(f.owned, [&](GpuView v) { return v.owner == region.owner && v.slot == region.slot; });
    }
    require(weak_source.expired(), "released mapped source owner did not drain");
    require(g.memory().live_bytes == before.live_bytes && g.memory().imported_backing_bytes == before.imported_backing_bytes &&
        g.memory().capture_resource_bytes == before.capture_resource_bytes && g.memory().live_captures == before.live_captures,
        "mapped replay resource ledger did not drain");
}

// Adjacent dense Q8, plain and expert-shaped Q8 tensors exercise import grouping and shared boundary pages without model data.
class GeneratedImportIsolationFile {
public:
    std::filesystem::path directory, path;
    explicit GeneratedImportIsolationFile(bool shared_pages) {
        const auto prefix = (std::filesystem::temp_directory_path() / "tokspur-import-isolation-XXXXXX").string();
        std::vector<char> pattern(prefix.begin(), prefix.end()); pattern.push_back('\0');
        const auto* made = mkdtemp(pattern.data());
        if (!made) throw std::runtime_error("cannot create import isolation fixture directory");
        directory = made; path = directory / "fixture.gguf";
        const uint64_t page = getpagesize();
        const std::array<uint64_t, 4> offsets = shared_pages ? std::array<uint64_t, 4>{0, 96, page - 64, page + 32} : std::array<uint64_t, 4>{0, page, 2 * page, 3 * page};
        std::vector<std::byte> bytes;
        const auto number = [&](uint64_t value, size_t width) {
            for (size_t i = 0; i < width; ++i) bytes.push_back(std::byte((value >> (8 * i)) & 255));
        };
        const auto text = [&](std::string_view value) {
            number(value.size(), 8); for (const unsigned char c : value) bytes.push_back(std::byte(c));
        };
        number(0x46554747, 4); number(3, 4); number(4, 8); number(3, 8);
        text("general.architecture"); number(8, 4); text("tokspur-generated-import");
        text("general.alignment"); number(4, 4); number(32, 4);
        text("tokspur.generated.padding"); number(8, 4);
        const auto padding_size_at = bytes.size(); number(0, 8); const auto padding_at = bytes.size();
        for (size_t i = 0; i < offsets.size(); ++i) {
            text("weight" + std::to_string(i)); number(i == 3 ? 3 : 2, 4);
            number(i == 2 ? 4 : 32, 8); number(i == 2 ? 3 : i == 3 ? 1 : 2, 8);
            if (i == 3) number(2, 8);
            number(i == 2 ? 0 : 8, 4); number(offsets[i], 8);
        }
        require(bytes.size() <= page, "generated import catalogue exceeds its page");
        const auto padding = page - bytes.size();
        for (size_t i = 0; i < 8; ++i) bytes[padding_size_at + i] = std::byte((padding >> (8 * i)) & 255);
        bytes.insert(bytes.begin() + std::ptrdiff_t(padding_at), padding, std::byte{32});
        const auto start = page;
        bytes.resize(start + (shared_pages ? 2 : 4) * page);
        for (size_t i = 0; i < offsets.size(); ++i) {
            auto* data = bytes.data() + start + offsets[i];
            if (i == 2) {
                const std::array<float, 12> values{1, 2, 3, 4, -1, 0, 1, 2, 0.5f, -0.5f, 0.25f, -0.25f};
                std::memcpy(data, values.data(), sizeof(values));
            } else {
                for (size_t row = 0; row < 2; ++row) {
                    data[34 * row] = std::byte{0}; data[34 * row + 1] = std::byte{0x3c};
                    for (size_t col = 0; col < 32; ++col) data[34 * row + 2 + col] = std::byte((col + 3 * row + i) % 17);
                }
            }
        }
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("cannot write import isolation fixture");
    }
    ~GeneratedImportIsolationFile() {
        std::error_code ignored; std::filesystem::remove(path, ignored); std::filesystem::remove(directory, ignored);
    }
};

void mapped_q8_isolation(MetalGraph& g, bool shared_pages) {
    GeneratedImportIsolationFile file(shared_pages);
    Fixture f{g, std::mt19937(15), {}, {}, {}, {}};
    auto source = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{file.path});
    std::weak_ptr<ModelFiles> weak_source = source;
    const std::array<TensorSlice, 4> slices{{{"weight0", 0, 0}, {"weight1", 0, 0}, {"weight2", 0, 0}, {"weight3", 0, 0}}};
    const auto before = g.memory();
    const auto imported = g.import_weights(source, slices); f.keep(imported.regions);
    constexpr bool isolated =
#if defined(TOKSPUR_PRIVATE_Q8_IMPORT_ISOLATION) && TOKSPUR_PRIVATE_Q8_IMPORT_ISOLATION
        true;
#else
        false;
#endif
    const uint64_t pages = shared_pages ? (isolated ? 3 : 2) : 4;
    require(imported.regions.size() == (isolated ? 2 : 1) && imported.tensors.size() == slices.size(), "Q8 import region count differs");
    require(g.memory().imported_backing_bytes - before.imported_backing_bytes == pages * getpagesize() &&
        g.memory().live_bytes - before.live_bytes == pages * getpagesize(), "Q8 import shared-page charges differ");
    require(imported.tensors[0].slot == imported.tensors[1].slot && imported.tensors[2].slot == imported.tensors[3].slot &&
        ((imported.tensors[0].slot != imported.tensors[2].slot) == isolated), "Q8 matrix and expert-shaped import classes differ");
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto data = source->tensor_bytes(slices[i].name);
        require(g.read(imported.tensors[i]) == std::vector<std::byte>(data.begin(), data.end()), "Q8 isolation changed imported bytes or order");
        const uint32_t columns = i == 2 ? 4 : 32, rows = i == 2 ? 3 : 2, type = i == 2 ? 0 : 8;
        const std::vector<float> values(columns, 0.5f);
        const auto expected = reference_gemv(type, data, columns, rows, values);
        const auto input = f.upload(values), output = f.zero(rows);
        const auto allocations = g.memory().allocations;
        g.begin(); g.gemv(type, imported.tensors[i], input, output, columns, rows); done(g, 1, 0, allocations);
        close(g, output, expected);
    }
    source.reset(); require(!weak_source.expired(), "Q8 imports lost source ownership");
    for (const auto region : imported.regions) {
        g.release(region);
        std::erase_if(f.owned, [&](GpuView v) { return v.owner == region.owner && v.slot == region.slot; });
    }
    require(weak_source.expired() && g.memory().imported_backing_bytes == before.imported_backing_bytes, "Q8 imports did not release the source and mapped charges");
}

// Explicit six-GiB logical-budget API regression. Sparse backing touches only
// the header and seven tiny sentinel matrices, not a model or a full payload.
// One mapped tensor/region spans the 32-bit binding boundary; importing two tiny
// disjoint regions would not exercise the original high-offset capability.
class GeneratedSparseOffsetFile {
public:
    static constexpr uint64_t boundary = uint64_t(1) << 32;
    static constexpr uint64_t start = 16384, logical = boundary + 65536;
    static constexpr uint64_t payload = logical - start;
    struct Site { uint64_t offset; std::array<float, 16> matrix{}; };
    std::array<Site, 7> sites{{{64}, {4096}, {32768}, {boundary - 64},
        {boundary + 64}, {boundary + 4096}, {boundary + 32768}}};
    std::filesystem::path directory, path;
    GeneratedSparseOffsetFile() {
        try {
            require(std::endian::native == std::endian::little && sizeof(float) == 4 &&
                std::numeric_limits<float>::is_iec559, "sparse offset fixture requires little-endian binary32");
            require(getpagesize() == 16384,
                "this registered sparse alias-sharing fixture requires sixteen-KiB host pages");
            require(logical <= uint64_t(std::numeric_limits<off_t>::max()), "64-bit sparse file offsets required");
            const auto prefix = (std::filesystem::temp_directory_path() / "tokspur-replay-sparse-XXXXXX").string();
            std::vector<char> pattern(prefix.begin(), prefix.end()); pattern.push_back('\0');
            const auto* made = mkdtemp(pattern.data());
            if (!made) throw std::runtime_error("cannot create private sparse replay fixture directory");
            directory = made; path = directory / "fixture.gguf";
            struct stat directory_status{};
            require(lstat(directory.c_str(), &directory_status) == 0 && S_ISDIR(directory_status.st_mode) &&
                directory_status.st_uid == getuid() && (directory_status.st_mode & 0077) == 0,
                "generated sparse fixture directory is not private/owned");
            struct Descriptor { int value; ~Descriptor() { if (value >= 0) ::close(value); } };
            const Descriptor fd{open(path.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY, 0600)};
            require(fd.value >= 0 && ftruncate(fd.value, static_cast<off_t>(logical)) == 0,
                "cannot exclusively create/truncate generated sparse fixture");
            const auto put = [&](std::span<const std::byte> data, uint64_t offset) {
                require(offset <= logical && data.size() <= logical - offset, "sparse fixture write exceeds bounded file");
                while (!data.empty()) {
                    const auto count = pwrite(fd.value, data.data(), data.size(), static_cast<off_t>(offset));
                    if (count < 0 && errno == EINTR) continue;
                    require(count > 0, "generated sparse fixture write failed");
                    offset += uint64_t(count); data = data.subspan(size_t(count));
                }
            };
            std::vector<std::byte> header;
            const auto number = [&](uint64_t value, size_t width) {
                for (size_t i = 0; i < width; ++i) header.push_back(std::byte((value >> (8 * i)) & 255));
            };
            const auto text = [&](std::string_view value) {
                number(value.size(), 8); for (const unsigned char c : value) header.push_back(std::byte(c));
            };
            number(0x46554747, 4); number(3, 4); number(1, 8); number(2, 8);
            text("general.architecture"); number(8, 4); text("tokspur-generated-sparse-replay");
            text("general.alignment"); number(4, 4); number(start, 4);
            text("weight"); number(1, 4); number(payload / 4, 8); number(0, 4); number(0, 8);
            require(header.size() < start, "bounded sparse GGUF header required"); put(header, 0);
            for (size_t i = 0; i < sites.size(); ++i) {
                for (size_t j = 0; j < sites[i].matrix.size(); ++j) sites[i].matrix[j] = float(i * 64 + j + 1);
                require(sites[i].offset + sizeof(sites[i].matrix) <= payload, "sparse sentinel exceeds tensor");
                put(std::as_bytes(std::span(sites[i].matrix)), start + sites[i].offset);
            }
            require(fsync(fd.value) == 0 && fchmod(fd.value, 0400) == 0, "cannot freeze sparse source read-only");
            static_cast<void>(disk_bytes()); // Fail closed before any GPU import.
        } catch (...) { cleanup(); throw; }
    }
    uint64_t disk_bytes() const {
        struct stat status{};
        require(lstat(path.c_str(), &status) == 0 && S_ISREG(status.st_mode) && status.st_nlink == 1 &&
            status.st_size == static_cast<off_t>(logical) && status.st_blocks >= 0,
            "generated sparse fixture identity/size differs");
        const auto physical = uint64_t(status.st_blocks) * 512;
        require(physical > 0 && physical < 1024 * 1024, "sparse replay fixture materialized one MiB or more");
        return physical;
    }
    void cleanup() noexcept {
        // Only the two exact artifacts created by this instance, never recursive.
        std::error_code ignored;
        if (!path.empty()) std::filesystem::remove(path, ignored);
        if (!directory.empty()) std::filesystem::remove(directory, ignored);
    }
    ~GeneratedSparseOffsetFile() { cleanup(); }
};

void sparse_offsets(MetalGraph& g) {
    const auto initial = g.memory();
    require(initial.budget == 6 * (uint64_t(1) << 30) && initial.max_buffer_bytes >= GeneratedSparseOffsetFile::payload,
        "sparse replay requires explicit six-GiB budget and supported maxBufferLength");
    require(!initial.capture_binding_aliases && !initial.capture_binding_alias_span_bytes &&
        !initial.capture_binding_alias_resource_bytes, "sparse replay starts with no cached aliases");
    {
        GeneratedSparseOffsetFile file;
        const auto disk = file.disk_bytes();
        Fixture f{g, std::mt19937(14), {}, {}, {}, {}};
        auto source = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{file.path});
        std::weak_ptr<ModelFiles> weak_source = source;
        require(source->shards().size() == 1 && source->tensors().size() == 1 &&
            source->tensor("weight").bytes == GeneratedSparseOffsetFile::payload,
            "one greater-than-four-GiB generated tensor required");
        for (const auto& site : file.sites)
            require(std::memcmp(source->tensor_bytes("weight").data() + site.offset, site.matrix.data(), sizeof(site.matrix)) == 0,
                "tiny sparse source sentinel differs"); // No full payload scan/copy.
        const std::array<TensorSlice, 1> slices{{{"weight", 0, 0}}};
        reject([&] { static_cast<void>(g.import_weights(source, slices, uint64_t(1) << 32)); });
        require(g.memory().live_bytes == initial.live_bytes && g.memory().allocations == initial.allocations,
            "oversized whole tensor cap refusal allocated or copied weights");
        const auto imported = g.import_weights(source, slices); f.keep(imported.regions);
        require(imported.regions.size() == 1 && imported.tensors.size() == 1 && imported.tensors[0].offset == 0 &&
            imported.regions[0].bytes == GeneratedSparseOffsetFile::payload &&
            g.memory().imported_backing_bytes == initial.imported_backing_bytes + GeneratedSparseOffsetFile::payload,
            "sparse offset test must retain one large original backing region");
        const auto input = f.guarded(4), middle = f.guarded(4), output = f.guarded(4);
        const auto page = uint64_t(getpagesize());
        std::array<GpuView, 7> views{}; std::array<GraphCapture, 7> captures{};
        for (size_t i = 0; i < views.size(); ++i) {
            views[i] = g.slice(imported.tensors[0], file.sites[i].offset, sizeof(file.sites[i].matrix));
            require(views[i].owner == imported.regions[0].owner && views[i].slot == imported.regions[0].slot &&
                views[i].offset == file.sites[i].offset, "sparse capture changed the requested original view identity");
            if (i >= 4) {
                require(views[i].offset > UINT32_MAX, "sparse high view lost its original full-width offset");
                const auto base = views[i].offset / page * page;
                const auto end = (views[i].offset + views[i].bytes + page - 1) / page * page;
                require(views[i].offset - base < page && end - base == page &&
                    views[i].bytes <= page - (views[i].offset - base) &&
                    base + page <= imported.regions[0].bytes,
                    "tiny high binding must fit within exactly its one-page immutable alias");
                require(base == GeneratedSparseOffsetFile::boundary + (i == 6 ? 32768 : 0),
                    "registered high sentinel page-sharing plan differs");
            }
        }
        const auto encode = [&](size_t index) {
            g.gemv(0, views[index], input.body, middle.body, 4, 4);
            g.activate(middle.body, output.body, Activation::identity, 0.5f);
        };
        const auto plain = g.memory(); const auto untouched = g.read(output.full);
        g.begin_capture(2); encode(4); g.abort();
        const auto cached = g.memory();
        require(cached.capture_binding_aliases == 1 && cached.capture_binding_alias_span_bytes == page &&
            cached.live_bytes == plain.live_bytes + cached.capture_binding_alias_resource_bytes &&
            cached.imported_backing_bytes == plain.imported_backing_bytes &&
            cached.capture_resource_bytes == plain.capture_resource_bytes && cached.live_captures == plain.live_captures &&
            cached.allocations == plain.allocations + 3 && g.read(output.full) == untouched,
            "aborted capture must retain exactly one bounded/charged source alias cache and perform no GPU work");
        const auto alias_accounting = [&](const GraphMemory& value, uint64_t expected_count) {
            require(value.capture_binding_aliases == expected_count &&
                value.capture_binding_alias_span_bytes == expected_count * page &&
                value.capture_binding_alias_resource_bytes >= cached.capture_binding_alias_resource_bytes &&
                value.capture_resource_bytes >= plain.capture_resource_bytes &&
                value.live_bytes == plain.live_bytes + value.capture_binding_alias_resource_bytes +
                    (value.capture_resource_bytes - plain.capture_resource_bytes) &&
                value.imported_backing_bytes == cached.imported_backing_bytes,
                "one-page alias count/span/excess accounting differs or shared backing counted twice");
        };
        // A fresh capture needs exactly its constants+ICB resources; it must not
        // allocate another alias for the same original allocation/page span.
        const auto prior = g.memory(); const auto repeated = f.capture(2, [&] { encode(5); });
        alias_accounting(g.memory(), 1);
        require(g.memory().allocations == prior.allocations + 2 &&
            g.memory().capture_binding_alias_resource_bytes == cached.capture_binding_alias_resource_bytes,
            "recapture allocated another cached source alias");
        f.release(repeated); alias_accounting(g.memory(), 1);
        for (size_t i = 0; i < views.size(); ++i) {
            const auto before = g.memory(); captures[i] = f.capture(2, [&] { encode(i); });
            // The first aborted capture preregistered the page shared by high64
            // and high4096. Only high32768 needs a second sixteen-KiB alias.
            const bool new_alias = i == 6;
            alias_accounting(g.memory(), new_alias ? 2 : 1);
            require(g.memory().allocations == before.allocations + (new_alias ? 3 : 2),
                "capture preparation did not follow the independently registered alias-cache increments");
            if (!new_alias) require(g.memory().capture_binding_alias_resource_bytes == cached.capture_binding_alias_resource_bytes,
                "reusing an existing page alias changed its reported resource charge");
        }
        source->assert_unchanged(); source.reset();
        require(!weak_source.expired(), "sparse mapped owner died after caller reset");
        const auto resident = g.memory(); reject([&] { g.release(imported.regions[0]); });
        alias_accounting(g.memory(), 2);
        require(g.memory().live_bytes == resident.live_bytes && !weak_source.expired(),
            "capture alias failed to lease the original source allocation");
        std::cout << "Owned sparse replay source: logical=" << GeneratedSparseOffsetFile::logical << ", physical=" << disk
            << ", unique_import=" << resident.imported_backing_bytes << ", alias_count=" << resident.capture_binding_aliases
            << ", alias_span=" << resident.capture_binding_alias_span_bytes << ", alias_resource=" << resident.capture_binding_alias_resource_bytes << '\n';
        constexpr std::array<std::array<float, 4>, 3> inputs{{{0.5f, -1, 2, -0.5f}, {-2, 1, 0.25f, 4}, {1, -0.5f, 0.25f, -2}}};
        const auto first_batch = batches;
        for (const auto& values : inputs) {
            g.write(input.body, std::as_bytes(std::span(values)));
            for (size_t i = 0; i < views.size(); ++i) {
                std::array<double, 4> cpu{};
                for (size_t row = 0; row < cpu.size(); ++row)
                    for (size_t col = 0; col < values.size(); ++col) cpu[row] += double(file.sites[i].matrix[row * 4 + col]) * values[col];
                std::array<float, 4> want_middle{}, want_output{};
                for (size_t j = 0; j < cpu.size(); ++j) { want_middle[j] = float(cpu[j]); want_output[j] = float(cpu[j] * 0.5); }
                std::vector<std::byte> baseline_middle, baseline_output;
                for (size_t arm = 0; arm < 3; ++arm) {
                    // Poison tiny bodies, not the sparse backing or its holes.
                    const std::array<float, 4> poison{91.25f, 91.25f, 91.25f, 91.25f};
                    g.write(middle.body, std::as_bytes(std::span(poison))); g.write(output.body, std::as_bytes(std::span(poison)));
                    begin_arm(g, arm); if (arm == 2) g.execute_capture(captures[i]); else encode(i);
                    done(g, 2, arm == 2, resident.allocations);
                    const auto actual_middle = g.read(middle.body), actual_output = g.read(output.body); compared += 8;
                    require(actual_middle.size() == sizeof(want_middle) && actual_output.size() == sizeof(want_output) &&
                        std::memcmp(actual_middle.data(), want_middle.data(), sizeof(want_middle)) == 0 &&
                        std::memcmp(actual_output.data(), want_output.data(), sizeof(want_output)) == 0,
                        "sparse full-width A/B/C result differs from exact independent CPU bits");
                    if (!arm) { baseline_middle = actual_middle; baseline_output = actual_output; }
                    else require(actual_middle == baseline_middle && actual_output == baseline_output, "sparse replay bits differ from ordinary direct");
                    alias_accounting(g.memory(), 2);
                    require(g.memory().live_bytes == resident.live_bytes && g.memory().peak_bytes == resident.peak_bytes &&
                        g.memory().capture_resource_bytes == resident.capture_resource_bytes && g.memory().live_captures == resident.live_captures &&
                        g.memory().capture_binding_alias_resource_bytes == resident.capture_binding_alias_resource_bytes,
                        "sparse execution changed the frozen same-resident ledger");
                    f.canaries(input); f.canaries(middle); f.canaries(output);
                }
            }
        }
        require(batches - first_batch == 63, "all seven offsets/three inputs/three arms must complete");
        for (auto capture : captures) f.release(capture);
        alias_accounting(g.memory(), 2);
        require(g.memory().capture_binding_alias_resource_bytes == resident.capture_binding_alias_resource_bytes && !weak_source.expired(),
            "cached alias excess changed or dropped the still-imported mmap owner");
        g.release(imported.regions[0]);
        std::erase_if(f.owned, [&](auto v) { return v.owner == imported.regions[0].owner && v.slot == imported.regions[0].slot; });
        require(weak_source.expired() && !g.memory().capture_binding_aliases && !g.memory().capture_binding_alias_span_bytes &&
            !g.memory().capture_binding_alias_resource_bytes && g.memory().imported_backing_bytes == initial.imported_backing_bytes &&
            !g.memory().capture_resource_bytes && !g.memory().live_captures,
            "source release did not drain cached aliases/captures and its mmap owner");
        require(file.disk_bytes() == disk, "sparse backing physical allocation changed during GPU test");
    }
    require(g.memory().live_bytes == initial.live_bytes && !g.memory().capture_binding_aliases &&
        !g.memory().capture_binding_alias_span_bytes && !g.memory().capture_binding_alias_resource_bytes,
        "sparse replay fixture failed to drain its owned resources");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool sparse = argc == 3 && std::string_view(argv[2]) == "--sparse-offsets-only";
        if (argc != 2 && !sparse) throw std::runtime_error("expected owned Metal library [--sparse-offsets-only]");
        MetalGraph g(argv[1], sparse ? 6 * (uint64_t(1) << 30) : 256 * 1024 * 1024);
        g.configure_small_gemv(16); g.configure_block_gemv(3);
        const auto initial = g.memory().live_bytes;
        if (sparse) sparse_offsets(g);
        else {
            chains(g); encoded_formats(g); dynamic_routes(g); recurrent_components(g, false); recurrent_components(g, true); safety(g, argv[1]);
            mapped_lifetime(g); mapped_q8_isolation(g, false); mapped_q8_isolation(g, true);
        }
        require(g.memory().live_bytes == initial && !g.memory().live_captures && !g.memory().capture_resource_bytes &&
            !g.memory().capture_binding_aliases && !g.memory().capture_binding_alias_span_bytes && !g.memory().capture_binding_alias_resource_bytes,
            "graph replay resource leak");
        require(g.kernel_policy() == std::array<uint32_t, 2>{16, 3}, "replay tests changed frozen numerical policy");
        std::cout << "Owned graph replay checks passed: " << checks << ", compared values: " << compared << ", batches: " << batches
                  << ", max scaled error: " << worst << "; generated components only, not timing or target qualification\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "Owned graph replay check failed: " << error.what() << '\n'; return 1; }
}
