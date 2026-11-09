#include "tokspur/stack.hpp"
#include "tokspur/attention_reference.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <thread>

namespace {
using namespace tokspur;
size_t checks = 0, batches = 0; double worst = 0;
void require(bool value, const char* why) { ++checks; if (!value) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid stack operation accepted"); }
struct Assets {
    MetalGraph& g; std::mt19937 random{0x73746131};
    std::deque<std::vector<float>> floats; std::deque<std::vector<double>> doubles; std::vector<GpuView> views;
    ~Assets() { for (auto v : views) g.release(v); }
    std::vector<float> values(size_t n, float bias = 0) { std::vector<float> out(n); for (auto& x : out) x = float(int(random() % 2001) - 1000) / 50000 + bias; return out; }
    GpuView upload(std::span<const float> data, bool immutable = true) { const auto v = g.upload(std::as_bytes(data), "stack fixture", immutable); views.push_back(v); return v; }
    std::pair<GpuView, std::span<const double>> parameter(size_t n, float bias = 0) {
        auto data = values(n, bias); doubles.emplace_back(data.begin(), data.end()); return {upload(data), doubles.back()};
    }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(size_t n, bool immutable = true) {
        floats.push_back(values(n)); return {{upload(floats.back(), immutable), 0}, {std::as_bytes(std::span(floats.back())), 0}};
    }
    std::pair<ResidualWeights, ReferenceResidualWeights> residual(ResidualShape s) {
        const auto gain = parameter(s.branches * s.channels, 1); const auto down = matrix(s.branches * s.channels * s.rank);
        const auto up = matrix(s.rank * s.branches * s.channels); const auto inject = matrix(s.branches * s.channels * s.branches);
        return {{gain.first, down.first, up.first, inject.first}, {gain.second, down.second, up.second, inject.second}};
    }
    std::pair<MoeWeights, ReferenceMoeWeights> moe(MoeShape s) {
        const auto router = matrix(s.channels * s.experts), gate = matrix(s.channels * s.hidden * s.experts), up = matrix(s.channels * s.hidden * s.experts), down = matrix(s.hidden * s.channels * s.experts);
        const auto sg = matrix(s.channels * s.hidden), su = matrix(s.channels * s.hidden), sd = matrix(s.hidden * s.channels), sr = matrix(s.channels);
        return {{router.first, gate.first, up.first, down.first, sg.first, su.first, sd.first, sr.first},
                {router.second, gate.second, up.second, down.second, sg.second, su.second, sd.second, sr.second}};
    }
    std::pair<GdnWeights, ReferenceGdnWeights> recurrent(GdnShape s) {
        const auto qkv = matrix(s.channels * gdn_qkv_count(s)), gate = matrix(s.channels * s.value_heads * s.value_width), alpha = matrix(s.channels * s.value_heads), beta = matrix(s.channels * s.value_heads), output = matrix(s.value_heads * s.value_width * s.channels);
        const auto conv = parameter(gdn_qkv_count(s) * s.kernel), rates = parameter(s.value_heads, -1), dt = parameter(s.value_heads), gain = parameter(s.value_width, 1);
        return {{qkv.first, gate.first, alpha.first, beta.first, output.first, conv.first, rates.first, dt.first, gain.first},
                {qkv.second, gate.second, alpha.second, beta.second, output.second, conv.second, rates.second, dt.second, gain.second}};
    }
    std::pair<QsaWeights, ReferenceQsaWeights> attention(QsaShape s) {
        const auto query = matrix(s.channels * s.query_heads * s.head_width * 2), key = matrix(s.channels * s.kv_heads * s.head_width), value = matrix(s.channels * s.kv_heads * s.head_width), output = matrix(s.query_heads * s.head_width * s.channels), iq = matrix(s.channels * s.index_heads * s.index_width), ik = matrix(s.channels * s.index_width);
        const auto qn = parameter(s.head_width, 1), kn = parameter(s.head_width, 1), iqn = parameter(s.index_width, 1), ikn = parameter(s.index_width, 1);
        floats.push_back(rotary_frequencies(s.rotary_width, 10000000)); const auto freq = upload(floats.back()); const std::span<const float> rf = floats.back();
        return {{query.first, key.first, value.first, output.first, iq.first, ik.first, qn.first, kn.first, iqn.first, ikn.first, freq},
                {query.second, key.second, value.second, output.second, iq.second, ik.second, qn.second, kn.second, iqn.second, ikn.second, rf}};
    }
};
using ReferenceLayer = std::variant<ReferenceLayerWeights, ReferenceAttentionLayerWeights>;
struct CpuState {
    std::vector<double> matrix, conv, sum;
    ReferenceQsaCache attention;
};
struct Fixture {
    StackWeights weights;
    std::vector<ReferenceLayer> reference;
    ReferencePleWeights memory;
    std::span<const double> gains;
    ReferenceMatrix down, up, output;
};
Fixture fixture(Assets& a, StackShape s, uint32_t layers, std::span<const uint32_t> attention_layers) {
    Fixture f;
    for (uint32_t i = 0; i < layers; ++i) {
        const auto r1 = a.residual(s.residual), r2 = a.residual(s.residual); const auto moe = a.moe(s.experts);
        if (std::find(attention_layers.begin(), attention_layers.end(), i) != attention_layers.end()) {
            const auto at = a.attention(s.attention);
            f.weights.layers.emplace_back(AttentionLayerWeights{r1.first, r2.first, at.first, moe.first});
            f.reference.emplace_back(ReferenceAttentionLayerWeights{r1.second, r2.second, at.second, moe.second});
        } else {
            const auto r = a.recurrent(s.recurrent);
            f.weights.layers.emplace_back(RecurrentWeights{r1.first, r2.first, r.first, moe.first});
            f.reference.emplace_back(ReferenceLayerWeights{r1.second, r2.second, r.second, moe.second});
        }
    }
    const uint32_t width = s.residual.branches * s.residual.channels;
    const auto key = a.matrix(s.memory.embedding * width), value = a.matrix(s.memory.embedding * s.memory.channels);
    const auto kn = a.parameter(width, 1), qn = a.parameter(width, 1), cn = a.parameter(width, 1), conv = a.parameter(width * s.memory.kernel);
    f.weights.memory = {key.first, value.first, kn.first, qn.first, cn.first, conv.first};
    f.memory = {key.second, value.second, kn.second, qn.second, cn.second, conv.second};
    const auto norm = a.parameter(width, 1); const auto down = a.matrix(width * s.residual.rank), up = a.matrix(s.residual.rank * width), head = a.matrix(s.vocabulary * s.residual.channels, false);
    f.weights.mixer = {norm.first, down.first, up.first}; f.weights.output = head.first;
    f.gains = norm.second; f.down = down.second; f.up = up.second; f.output = head.second; return f;
}
Fixture fixture(Assets& a, StackShape s) {
    constexpr std::array<uint32_t, 1> attention_layers{2};
    return fixture(a, s, 4, attention_layers);
}
std::vector<double> cpu_step(const Fixture& f, std::vector<CpuState>& states, std::vector<double>& memory_history,
                             std::span<const float> row, std::span<const float> memory, uint32_t position, StackShape s, StackPolicy p,
                             std::vector<uint32_t>* routes = nullptr) {
    std::vector<double> streams; for (uint32_t b = 0; b < s.residual.branches; ++b) streams.insert(streams.end(), row.begin(), row.end());
    for (size_t i = 0; i < f.reference.size(); ++i) {
        if (i == s.memory_layer) {
            const auto mem = ple_reference(f.memory, std::vector<double>(memory.begin(), memory.end()), streams, memory_history, s.memory, p.attention.rms_epsilon);
            streams = mem.output; memory_history = mem.convolution.history;
        }
        if (const auto* r = std::get_if<ReferenceLayerWeights>(&f.reference[i])) {
            const auto layer = recurrent_reference(*r, streams, states[i].matrix, states[i].conv,
                {s.residual, s.recurrent, s.experts}, p.recurrent, p.router, p.normalize_selected);
            if (routes) routes->insert(routes->end(), layer.ffn.routes.indices.begin(), layer.ffn.routes.indices.end());
            streams = layer.output_streams; states[i].matrix = layer.attention.delta.state; states[i].conv = layer.attention.convolution.history;
        } else {
            const auto layer = attention_layer_reference(std::get<ReferenceAttentionLayerWeights>(f.reference[i]), states[i].attention,
                states[i].sum, streams, position, {s.residual, s.attention, s.experts}, p.attention, p.router, p.normalize_selected);
            if (routes) routes->insert(routes->end(), layer.ffn.routes.indices.begin(), layer.ffn.routes.indices.end());
            streams = layer.output_streams; states[i].sum = layer.attention.pooling.next_sum;
        }
    }
    const auto mixed = mixer_reference(f.gains, f.down, f.up, streams, s.residual, p.attention.rms_epsilon);
    return reference_gemv_doubleinput(f.output.encoding, f.output.bytes, s.residual.channels, s.vocabulary, mixed.mixed);
}
std::vector<std::vector<float>> connected(MetalGraph& graph, bool wide = false,
                                         const std::vector<std::vector<float>>* exact = nullptr) {
    const uint32_t channels = wide ? 1024 : 8;
    const StackShape shape{{2, channels, 3}, {channels, 2, 4, 4, 4, 4}, {channels, 4, 2, 4, 2, 4, 4, 4, 8, 64},
        {channels, 8, 4, 2}, {2, channels, 4, 4, 3}, 13, 1};
    std::vector<std::vector<float>> trajectory;
    const auto record = [&](const std::vector<float>& values) {
        if (exact) require(trajectory.size() < exact->size() && values == (*exact)[trajectory.size()], "aligned FP32 stack trajectory is not byte-exact");
        trajectory.push_back(values);
    };
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    Assets assets{graph, std::mt19937(0x73746131), {}, {}, {}}; const auto f = fixture(assets, shape);
    const auto base = graph.memory().live_bytes; const auto head_bytes = graph.read(f.weights.output.view);
    for (uint32_t variant = 0; variant < 4; ++variant) {
        const StackPolicy policy{{0.5f, 1e-6f, variant % 2 ? L2Mode::floor_squared_epsilon : L2Mode::add_squared_epsilon, 1e-6f},
            {variant % 2 ? QueryPacking::all_queries_then_gates : QueryPacking::head_query_gate, variant / 2 ? RotaryPairs::adjacent : RotaryPairs::split_half, 1e-6f},
            variant % 2 ? RouterMode::sigmoid : RouterMode::softmax, bool(variant / 2),
            {variant % 2 ? HashOrder::oldest_first : HashOrder::newest_first, NgramBoundary(variant % 3), true}};
        {
            TextStack stack(graph, f.weights, shape, policy, spec); const auto empty = stack.accepted_snapshot();
            std::vector<CpuState> cpu(4);
            for (auto& state : cpu) {
                state.matrix.resize(gdn_state_count(shape.recurrent)); state.conv.resize(gdn_qkv_count(shape.recurrent) * (shape.recurrent.kernel - 1)); state.sum.resize(shape.attention.index_width);
                state.attention.keys.resize(shape.attention.capacity * shape.attention.kv_heads * shape.attention.head_width);
                state.attention.values.resize(state.attention.keys.size()); state.attention.index_keys.resize(qsa_block_capacity(shape.attention) * shape.attention.index_width);
            }
            auto ngram = empty_ngram_history(spec);
            std::vector<double> memory_history(shape.memory.branches * shape.memory.channels * (shape.memory.kernel - 1) * shape.memory.dilation);
            std::vector<float> first;
            for (uint32_t position = 0; position < shape.attention.capacity; ++position) {
                const uint32_t token = position % shape.vocabulary; const auto ids = ngram_step(spec, policy.ngram, ngram, token);
                require(stack.lookup(token).rows == ids.rows && stack.position() == position, "composite prefix/hash publication differs");
                std::vector<float> row(channels), memory(4); for (size_t i = 0; i < row.size(); ++i) row[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16;
                for (size_t i = 0; i < memory.size(); ++i) memory[i] = float(int(ids.rows[i] % 17) - 8) / 32;
                if (position == 1) {
                    const auto before = stack.accepted_snapshot(); const auto accepted_policy = graph.kernel_policy();
                    graph.configure_block_gemv(accepted_policy[1] ^ 1);
                    reject([&] { static_cast<void>(stack.lookup(token)); });
                    reject([&] { static_cast<void>(stack.step(token, row, memory)); });
                    require(stack.accepted_snapshot() == before && stack.position() == position, "kernel-policy refusal damaged accepted prefix");
                    graph.configure_block_gemv(accepted_policy[1]);
                    require(stack.lookup(token).rows == ids.rows, "restored kernel policy did not recover accepted prefix");
                }
                if (position == 3) {
                    const auto before = stack.accepted_snapshot();
                    std::vector<float> bad(shape.vocabulary * shape.residual.channels); std::memcpy(bad.data(), head_bytes.data(), head_bytes.size()); bad[0] = std::numeric_limits<float>::infinity();
                    graph.write(f.weights.output.view, std::as_bytes(std::span(bad)));
                    reject([&] { static_cast<void>(stack.step(token, row, memory)); });
                    require(stack.accepted_snapshot() == before && stack.position() == position, "device failure published/damaged a composite prefix");
                    require(stack.lookup(token).rows == ids.rows, "failed GPU step advanced n-gram history");
                    graph.write(f.weights.output.view, head_bytes);
                    row[0] = std::numeric_limits<float>::infinity(); reject([&] { static_cast<void>(stack.step(token, row, memory)); }); row[0] = float(int((token * 7) % 23) - 11) / 16;
                    reject([&] { static_cast<void>(stack.step(token, {}, memory)); });
                    reject([&] { static_cast<void>(stack.lookup(shape.vocabulary)); });
                    require(stack.accepted_snapshot() == before, "host refusal damaged accepted stack state");
                }
                const auto reference = cpu_step(f, cpu, memory_history, row, memory, position, shape, policy);
                const auto result = stack.step(token, row, memory); ++batches;
                record(result.logits);
                require(result.position == position && stack.position() == position + 1, "successful stack step did not publish exactly once");
                for (size_t i = 0; i < reference.size(); ++i) {
                    const auto error = std::abs(result.logits[i] - reference[i]) / std::max(1.0, std::abs(reference[i])); worst = std::max(worst, error);
                    require(std::isfinite(result.logits[i]) && error <= 5e-5, "stack output differs from complete double-reference chain");
                }
                if (!position) first = result.logits;
                ngram = ids.next;
            }
            const auto full = stack.accepted_snapshot(); reject([&] { static_cast<void>(stack.lookup(0)); });
            require(stack.accepted_snapshot() == full, "capacity refusal changed accepted stack");
            bool denied = false; std::thread other([&] { try { static_cast<void>(stack.position()); } catch (const FormatError&) { denied = true; } }); other.join(); require(denied, "foreign thread read stack state");
            graph.begin(); reject([&] { stack.reset(); }); graph.abort(); require(stack.accepted_snapshot() == full, "refused reset changed accepted state");
            stack.reset(); require(stack.position() == 0 && stack.accepted_snapshot() == empty, "reset did not publish a complete empty prefix");
            const auto ids = stack.lookup(0); std::vector<float> row(channels), memory(4);
            for (size_t i = 0; i < row.size(); ++i) row[i] = float(int((i * 3) % 23) - 11) / 16;
            for (size_t i = 0; i < memory.size(); ++i) memory[i] = float(int(ids.rows[i] % 17) - 8) / 32;
            const auto again = stack.step(0, row, memory); ++batches; record(again.logits); require(again.logits == first, "reset/replay is not deterministic after dirty cache candidates");
        }
        require(graph.memory().live_bytes == base, "stack state/scratch leak");
    }
    {
        auto bad = f.weights; bad.output.encoding = 0xffffffffu;
        const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true, {HashOrder::newest_first, NgramBoundary::literal, true}};
        TextStack stack(graph, bad, shape, policy, spec); const auto before = stack.accepted_snapshot();
        reject([&] { static_cast<void>(stack.step(0, std::vector<float>(channels, 1), std::vector<float>(4, 1))); });
        require(stack.accepted_snapshot() == before, "late full-stack encode refusal executed partial state writes");
    }
    require(graph.memory().live_bytes == base, "failed stack leaked allocations");
    if (exact) require(trajectory.size() == exact->size(), "aligned stack trajectory length differs");
    return trajectory;
}
void replay_connected(MetalGraph& graph, bool wide) {
    const uint32_t channels = wide ? 1024 : 8;
    const StackShape shape{{2, channels, 3}, {channels, 2, 4, 4, 4, 4}, {channels, 4, 2, 4, 2, 4, 4, 4, 8, 64},
        {channels, 8, 4, 2}, {2, channels, 4, 4, 3}, 13, 1};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true,
        {HashOrder::newest_first, NgramBoundary::literal, true}};
    constexpr std::array modes{StackExecution::direct, StackExecution::indirect_compatible, StackExecution::recurrent_replay};
    Assets assets{graph, std::mt19937(0x73746131), {}, {}, {}}; const auto f = fixture(assets, shape);
    const auto baseline = graph.memory(); const auto head_bytes = graph.read(f.weights.output.view);
    const auto same_ledger = [&](const GraphMemory& a, const GraphMemory& b) {
        require(a.live_bytes == b.live_bytes && a.allocations == b.allocations && a.live_captures == b.live_captures &&
            a.capture_resource_bytes == b.capture_resource_bytes, "stack replay resource ledger changed");
    };
    reject([&] { TextStack invalid(graph, f.weights, shape, policy, spec, static_cast<StackExecution>(0xffffffffu)); });
    require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
        graph.memory().capture_resource_bytes == baseline.capture_resource_bytes, "invalid execution mode retained resources");
    {
        const auto first_projection = std::get<RecurrentWeights>(f.weights.layers[0]).attention.qkv.view;
        const auto original = graph.read(first_projection);
        auto bad = f.weights; std::get<RecurrentWeights>(bad.layers[3]).attention.qkv.encoding = 0xffffffffu;
        // The first run publishes both orientations before encoding layer3 of
        // the later run fails. All published leases and partial draft must drain.
        reject([&] { TextStack partial(graph, bad, shape, policy, spec, StackExecution::recurrent_replay); });
        require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
            graph.memory().capture_resource_bytes == baseline.capture_resource_bytes,
            "partial replay construction retained earlier run captures/state");
        require(graph.read(first_projection) == original, "partial replay construction damaged original bound weights");
    }
    {
        std::array<std::unique_ptr<TextStack>, 3> arms;
        for (size_t i = 0; i < arms.size(); ++i) {
            arms[i] = std::make_unique<TextStack>(graph, f.weights, shape, policy, spec, modes[i]);
            require(arms[i]->execution() == modes[i] && arms[i]->replay_capture_count() == (i == 2 ? 4 : 0),
                "full-stack execution mode/capture introspection differs");
            require(arms[i]->replay_prepared() == (i == 2), "stack replay preparation identity differs");
            if (i != 2) {
                const auto before = arms[i]->accepted_snapshot(); const auto resident = graph.memory();
                for (const auto mode : modes) reject([&] { arms[i]->configure_execution(mode); });
                require(arms[i]->execution() == modes[i] && !arms[i]->replay_prepared() && arms[i]->accepted_snapshot() == before,
                    "unprepared execution-mode refusal changed stack identity/state");
                same_ledger(graph.memory(), resident);
            }
        }
        const auto resident = graph.memory();
        require(resident.live_captures == baseline.live_captures + 4 &&
            resident.capture_resource_bytes > baseline.capture_resource_bytes, "two recurrent runs/orientations were not retained");
        const auto empty = arms[0]->accepted_snapshot();
        require(arms[1]->accepted_snapshot() == empty && arms[2]->accepted_snapshot() == empty,
            "three-arm empty stack snapshots differ");
        const auto fixed_empty = graph.kernel_policy(); graph.configure_block_gemv(fixed_empty[1] ^ 1);
        reject([&] { static_cast<void>(arms[2]->lookup(0)); });
        reject([&] { static_cast<void>(arms[2]->step(0, std::vector<float>(channels), std::vector<float>(shape.memory.embedding))); });
        reject([&] { arms[2]->reset(); });
        require(arms[2]->accepted_snapshot() == empty && arms[2]->position() == 0,
            "empty-prefix policy refusal damaged frozen replay bindings");
        graph.configure_block_gemv(fixed_empty[1]); same_ledger(graph.memory(), resident);
        std::vector<CpuState> cpu(4);
        const auto clear_cpu = [&] {
            cpu.assign(4, {});
            for (auto& state : cpu) {
                state.matrix.resize(gdn_state_count(shape.recurrent)); state.conv.resize(gdn_qkv_count(shape.recurrent) * (shape.recurrent.kernel - 1));
                state.sum.resize(shape.attention.index_width); state.attention.keys.resize(shape.attention.capacity * shape.attention.kv_heads * shape.attention.head_width);
                state.attention.values.resize(state.attention.keys.size()); state.attention.index_keys.resize(qsa_block_capacity(shape.attention) * shape.attention.index_width);
            }
        };
        clear_cpu();
        std::vector<double> memory_history(shape.memory.branches * shape.memory.channels * (shape.memory.kernel - 1) * shape.memory.dilation);
        std::set<std::vector<uint32_t>> route_patterns;
        std::vector<float> first_logits; std::vector<std::byte> first_snapshot;
        const auto step = [&](uint32_t token, bool inject_failure) {
            const auto position = arms[0]->position(); const auto ids = arms[0]->lookup(token);
            std::vector<float> row(channels), memory(shape.memory.embedding);
            for (size_t i = 0; i < row.size(); ++i) row[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16;
            for (size_t i = 0; i < memory.size(); ++i) memory[i] = float(int(ids.rows[i] % 17) - 8) / 32;
            const auto before = arms[0]->accepted_snapshot();
            for (auto& arm : arms) require(arm->position() == position && arm->lookup(token).rows == ids.rows && arm->accepted_snapshot() == before,
                "three-arm accepted prefix diverged before replay");
            if (inject_failure) {
                std::vector<float> bad(shape.vocabulary * shape.residual.channels); std::memcpy(bad.data(), head_bytes.data(), head_bytes.size());
                bad[0] = std::numeric_limits<float>::infinity(); graph.write(f.weights.output.view, std::as_bytes(std::span(bad)));
                for (auto& arm : arms) {
                    reject([&] { static_cast<void>(arm->step(token, row, memory)); });
                    require(arm->position() == position && arm->accepted_snapshot() == before && arm->lookup(token).rows == ids.rows,
                        "failed replay GPU batch published accepted composite state");
                }
                graph.write(f.weights.output.view, head_bytes);
                for (auto& arm : arms) {
                    auto bad_row = row; bad_row[0] = std::numeric_limits<float>::infinity();
                    reject([&] { static_cast<void>(arm->step(token, bad_row, memory)); });
                    auto bad_memory = memory; bad_memory[0] = std::numeric_limits<float>::quiet_NaN();
                    reject([&] { static_cast<void>(arm->step(token, row, bad_memory)); });
                    reject([&] { static_cast<void>(arm->step(token, {}, memory)); });
                    reject([&] { static_cast<void>(arm->step(token, row, {})); });
                    reject([&] { static_cast<void>(arm->lookup(shape.vocabulary)); });
                    require(arm->accepted_snapshot() == before, "replay host refusal changed accepted composite state");
                }
                same_ledger(graph.memory(), resident);
            }
            std::vector<uint32_t> routes;
            const auto reference = cpu_step(f, cpu, memory_history, row, memory, position, shape, policy, &routes);
            route_patterns.insert(std::move(routes));
            std::array<StackResult, 3> results;
            for (size_t i = 0; i < arms.size(); ++i) { results[i] = arms[i]->step(token, row, memory); ++batches; }
            const auto snapshot = arms[0]->accepted_snapshot();
            for (size_t i = 0; i < arms.size(); ++i) {
                const auto& result = results[i];
                require(result.position == position && arms[i]->position() == position + 1 && arms[i]->accepted_snapshot() == snapshot,
                    "replay full-stack accepted snapshots differ");
                require(result.logits.size() == results[0].logits.size() &&
                    std::memcmp(result.logits.data(), results[0].logits.data(), result.logits.size() * sizeof(float)) == 0,
                    "replay full-stack logits are not bit-exact");
                require(result.batch.dispatches == results[0].batch.dispatches && result.batch.capture_executes == (i == 2 ? 2 : 0),
                    "replay full-stack physical/capture dispatch counts differ");
                for (size_t j = 0; j < reference.size(); ++j) {
                    const auto error = std::abs(result.logits[j] - reference[j]) / std::max(1.0, std::abs(reference[j])); worst = std::max(worst, error);
                    require(std::isfinite(result.logits[j]) && error <= 5e-5, "replay stack differs from complete double-reference chain");
                }
            }
            same_ledger(graph.memory(), resident);
            if (!position && token == 0) {
                if (first_logits.empty()) { first_logits = results[0].logits; first_snapshot = snapshot; }
                else require(std::memcmp(first_logits.data(), results[0].logits.data(), first_logits.size() * sizeof(float)) == 0 && snapshot == first_snapshot,
                    "replay reset selected bindings from prefix parity instead of physical state orientation");
            }
        };
        // Exercise every QSA compression/selection boundary through capacity,
        // with changing inputs and independently observed changing expert routes.
        for (uint32_t position = 0; position < shape.attention.capacity; ++position) {
            step(position % shape.vocabulary, position == 3 || position == 4);
            if (position == 0) {
                const auto before = arms[0]->accepted_snapshot(); const auto fixed = graph.kernel_policy();
                graph.configure_block_gemv(fixed[1] ^ 1);
                for (auto& arm : arms) {
                    reject([&] { static_cast<void>(arm->lookup(0)); });
                    reject([&] { static_cast<void>(arm->step(0, std::vector<float>(channels), std::vector<float>(shape.memory.embedding))); });
                    require(arm->accepted_snapshot() == before, "changed-policy refusal damaged accepted replay state");
                }
                reject([&] { arms[2]->reset(); });
                graph.configure_block_gemv(fixed[1]); same_ledger(graph.memory(), resident);
            }
        }
        require(route_patterns.size() > 1, "replay changing-input fixture did not exercise dynamic expert routes");
        for (auto& arm : arms) {
            const auto full = arm->accepted_snapshot(); reject([&] { static_cast<void>(arm->lookup(0)); });
            require(arm->accepted_snapshot() == full, "replay capacity refusal damaged accepted prefix");
            bool denied = false; std::thread other([&] { try { static_cast<void>(arm->execution()); } catch (const FormatError&) { denied = true; } });
            other.join(); require(denied, "foreign thread read replay execution identity");
            graph.begin(); reject([&] { arm->reset(); }); graph.abort();
            require(arm->accepted_snapshot() == full, "active-batch reset refusal damaged replay state");
        }
        // Reset always swaps the physical banks, independently of token prefix.
        // Start after even64, then reset odd1 and even2; every first step must
        // reproduce the original bytes despite the opposite captured bindings.
        for (const uint32_t length : {1, 2, 1}) {
            for (auto& arm : arms) { arm->reset(); require(arm->position() == 0 && arm->accepted_snapshot() == empty, "replay reset did not publish complete empty state"); }
            clear_cpu(); std::fill(memory_history.begin(), memory_history.end(), 0);
            for (uint32_t position = 0; position < length; ++position) step(position, false);
        }
    }
    require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
        graph.memory().capture_resource_bytes == baseline.capture_resource_bytes, "full-stack replay captures/state did not drain");
    {
        // The timing comparator owns the same resident captures in every mode;
        // only the encoding policy changes, never preparation or allocation.
        TextStack stack(graph, f.weights, shape, policy, spec, StackExecution::recurrent_replay);
        const auto resident = graph.memory(); const auto empty = stack.accepted_snapshot();
        require(stack.replay_prepared() && stack.replay_capture_count() == 4 && resident.live_captures == baseline.live_captures + 4,
            "same-resident comparator lacks prepared captures");
        std::vector<std::vector<float>> expected_logits;
        std::vector<std::vector<std::byte>> expected_snapshots;
        std::vector<uint64_t> expected_dispatches;
        constexpr std::array sequence{StackExecution::recurrent_replay, StackExecution::direct,
            StackExecution::indirect_compatible, StackExecution::recurrent_replay, StackExecution::direct};
        constexpr std::array<uint32_t, 5> lengths{3, 2, 3, 2, 3};
        for (size_t round = 0; round < sequence.size(); ++round) {
            if (round) stack.reset();
            require(stack.position() == 0 && stack.accepted_snapshot() == empty,
                "same-resident reset changed complete empty prefix");
            stack.configure_execution(sequence[round]);
            require(stack.execution() == sequence[round] && stack.replay_prepared() && stack.replay_capture_count() == 4 &&
                stack.accepted_snapshot() == empty, "same-resident mode switch changed bindings or state");
            reject([&] { stack.configure_execution(static_cast<StackExecution>(0xffffffffu)); });
            require(stack.execution() == sequence[round], "invalid same-resident mode changed execution policy");
            graph.begin(); for (const auto mode : modes) reject([&] { stack.configure_execution(mode); }); graph.abort();
            require(stack.execution() == sequence[round] && stack.accepted_snapshot() == empty,
                "active-batch mode refusal changed resident execution/state");
            const auto fixed = graph.kernel_policy(); graph.configure_block_gemv(fixed[1] ^ 1);
            reject([&] { static_cast<void>(stack.lookup(0)); });
            reject([&] { static_cast<void>(stack.step(0, std::vector<float>(channels), std::vector<float>(shape.memory.embedding))); });
            reject([&] { stack.reset(); });
            for (const auto mode : modes) reject([&] { stack.configure_execution(mode); });
            require(stack.position() == 0 && stack.execution() == sequence[round] && stack.accepted_snapshot() == empty,
                "prepared direct/compatible mode accepted changed kernel policy");
            graph.configure_block_gemv(fixed[1]); same_ledger(graph.memory(), resident);
            for (uint32_t position = 0; position < lengths[round]; ++position) {
                const auto ids = stack.lookup(position); std::vector<float> row(channels), memory(shape.memory.embedding);
                for (size_t i = 0; i < row.size(); ++i) row[i] = float(int((position * 7 + i * 3) % 23) - 11) / 16;
                for (size_t i = 0; i < memory.size(); ++i) memory[i] = float(int(ids.rows[i] % 17) - 8) / 32;
                const auto result = stack.step(position, row, memory); ++batches; const auto snapshot = stack.accepted_snapshot();
                if (!round) {
                    expected_logits.push_back(result.logits); expected_snapshots.push_back(snapshot); expected_dispatches.push_back(result.batch.dispatches);
                }
                require(result.position == position && stack.position() == position + 1 && snapshot == expected_snapshots[position] &&
                    std::memcmp(result.logits.data(), expected_logits[position].data(), result.logits.size() * sizeof(float)) == 0,
                    "same-resident mode/reset parity changed logits or accepted state bits");
                require(result.batch.dispatches == expected_dispatches[position] &&
                    result.batch.capture_executes == (sequence[round] == StackExecution::recurrent_replay ? 2 : 0),
                    "same-resident mode changed physical dispatch count");
                for (const auto mode : modes) reject([&] { stack.configure_execution(mode); });
                require(stack.execution() == sequence[round] && stack.position() == position + 1 && stack.accepted_snapshot() == snapshot &&
                    stack.replay_prepared() && stack.replay_capture_count() == 4,
                    "nonempty-prefix mode refusal changed resident bindings/state");
                same_ledger(graph.memory(), resident);
            }
        }
        stack.reset(); require(stack.position() == 0 && stack.accepted_snapshot() == empty, "same-resident final reset changed accepted empty state");
        bool getter_denied = false, setter_denied = false;
        std::thread other([&] {
            try { static_cast<void>(stack.replay_prepared()); } catch (const FormatError&) { getter_denied = true; }
            try { stack.configure_execution(StackExecution::direct); } catch (const FormatError&) { setter_denied = true; }
        });
        other.join(); require(getter_denied && setter_denied, "foreign thread accessed resident replay policy");
        require(stack.execution() == StackExecution::direct && stack.accepted_snapshot() == empty, "foreign thread changed resident replay identity/state");
        same_ledger(graph.memory(), resident);
    }
    require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
        graph.memory().capture_resource_bytes == baseline.capture_resource_bytes,
        "same-resident captures did not drain when last execution mode was direct");
    for (const auto mode : modes) {
        auto bad = f.weights; bad.output.encoding = 0xffffffffu;
        {
            TextStack stack(graph, bad, shape, policy, spec, mode); const auto before = stack.accepted_snapshot(); const auto resident = graph.memory();
            reject([&] { static_cast<void>(stack.step(0, std::vector<float>(channels, 1), std::vector<float>(shape.memory.embedding, 1))); });
            require(stack.accepted_snapshot() == before && stack.position() == 0, "late replay encode refusal published partial stack state");
            same_ledger(graph.memory(), resident);
        }
        require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
            graph.memory().capture_resource_bytes == baseline.capture_resource_bytes, "failed replay stack leaked capture/state resources");
    }
    static_cast<void>(graph.configure_profiling(1024));
    {
        TextStack compatible(graph, f.weights, shape, policy, spec, StackExecution::indirect_compatible);
        const auto before = compatible.accepted_snapshot(); const auto resident = graph.memory();
        reject([&] { static_cast<void>(compatible.step(0, std::vector<float>(channels, 1), std::vector<float>(shape.memory.embedding, 1))); });
        require(compatible.accepted_snapshot() == before && compatible.position() == 0, "profiled compatible refusal published a prefix");
        same_ledger(graph.memory(), resident);
    }
    reject([&] { TextStack replay(graph, f.weights, shape, policy, spec, StackExecution::recurrent_replay); });
    static_cast<void>(graph.configure_profiling(0));
    require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
        graph.memory().capture_resource_bytes == baseline.capture_resource_bytes, "profiled replay refusal leaked captures/state");
}
void replay_generic(MetalGraph& graph) {
    // Extend, rather than replace, the historical four-layer/wide cohort above.
    // These small F32 shapes test scheduling boundaries, not model performance.
    require(graph.kernel_policy() == std::array<uint32_t, 2>{0, 0}, "generic replay fixtures require ordinary kernel policy");
    struct Case {
        uint32_t layers, memory_layer, captures, dispatches;
        std::vector<uint32_t> attention_layers;
    };
    const std::array cases{
        // 25*40=1,000 commands fit in chunk one; PLE+layer25 requires48 more,
        // forcing a second chunk and putting PLE exactly at that chunk's start.
        Case{26, 25, 4, 1054, {}},
        Case{4, 2, 4, 180, {2}}, // PLE is dynamic, directly before the QSA layer.
        Case{3, 1, 0, 152, {0, 1, 2}} // Prepared replay with no recorded commands.
    };
    constexpr std::array modes{StackExecution::direct, StackExecution::indirect_compatible, StackExecution::recurrent_replay};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true,
        {HashOrder::newest_first, NgramBoundary::literal, true}};
    for (const auto& c : cases) {
        const StackShape shape{{2, 8, 3}, {8, 2, 4, 4, 4, 4}, {8, 4, 2, 4, 2, 4, 4, 4, 8, 64},
            {8, 8, 4, 2}, {2, 8, 4, 4, 3}, 13, c.memory_layer};
        Assets assets{graph, std::mt19937(0x67656e31), {}, {}, {}};
        const auto f = fixture(assets, shape, c.layers, c.attention_layers); const auto baseline = graph.memory();
        {
            std::array<std::unique_ptr<TextStack>, 3> arms;
            for (size_t i = 0; i < arms.size(); ++i) {
                arms[i] = std::make_unique<TextStack>(graph, f.weights, shape, policy, spec, modes[i]);
                require(arms[i]->execution() == modes[i] && arms[i]->replay_prepared() == (i == 2) &&
                    arms[i]->replay_capture_count() == (i == 2 ? c.captures : 0), "generic replay preparation/run partition differs");
            }
            const auto resident = graph.memory(); const auto empty = arms[0]->accepted_snapshot();
            require(resident.live_captures == baseline.live_captures + c.captures &&
                (c.captures ? resident.capture_resource_bytes > baseline.capture_resource_bytes : resident.capture_resource_bytes == baseline.capture_resource_bytes),
                "generic replay capture resources/count differ");
            require(arms[1]->accepted_snapshot() == empty && arms[2]->accepted_snapshot() == empty, "generic replay empty states differ");
            // A no-recurrent stack is still prepared and permits the same
            // empty-prefix scheduler choices without inventing a capture.
            for (const auto mode : modes) {
                arms[2]->configure_execution(mode);
                require(arms[2]->execution() == mode && arms[2]->replay_prepared() && arms[2]->replay_capture_count() == c.captures &&
                    arms[2]->accepted_snapshot() == empty, "generic empty-prefix mode switch changed bindings/state");
            }
            std::vector<std::vector<float>> first_logits;
            std::vector<std::vector<std::byte>> first_snapshots;
            // Reset after odd5, odd1 and even2 steps. Every step also alternates
            // physical banks; the last reset gives opposite prefix-zero bindings.
            constexpr std::array<uint32_t, 4> lengths{5, 1, 2, 5};
            for (size_t round = 0; round < lengths.size(); ++round) {
                if (round) for (auto& arm : arms) arm->reset();
                for (auto& arm : arms) require(!arm->position() && arm->accepted_snapshot() == empty, "generic reset did not publish complete empty state");
                std::vector<CpuState> cpu(c.layers);
                for (auto& state : cpu) {
                    state.matrix.resize(gdn_state_count(shape.recurrent)); state.conv.resize(gdn_qkv_count(shape.recurrent) * (shape.recurrent.kernel - 1));
                    state.sum.resize(shape.attention.index_width); state.attention.keys.resize(shape.attention.capacity * shape.attention.kv_heads * shape.attention.head_width);
                    state.attention.values.resize(state.attention.keys.size()); state.attention.index_keys.resize(qsa_block_capacity(shape.attention) * shape.attention.index_width);
                }
                auto history = empty_ngram_history(spec);
                std::vector<double> memory_history(shape.memory.branches * shape.memory.channels * (shape.memory.kernel - 1) * shape.memory.dilation);
                for (uint32_t position = 0; position < lengths[round]; ++position) {
                    const auto token = (position * 3) % shape.vocabulary; const auto ids = ngram_step(spec, policy.ngram, history, token);
                    std::vector<float> row(shape.residual.channels), memory(shape.memory.embedding);
                    for (size_t i = 0; i < row.size(); ++i) row[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16;
                    for (size_t i = 0; i < memory.size(); ++i) memory[i] = float(int(ids.rows[i] % 17) - 8) / 32;
                    const auto reference = cpu_step(f, cpu, memory_history, row, memory, position, shape, policy);
                    std::array<StackResult, 3> results;
                    for (size_t i = 0; i < arms.size(); ++i) {
                        require(arms[i]->position() == position && arms[i]->lookup(token).rows == ids.rows, "generic replay changed prefix/ngram lookup");
                        results[i] = arms[i]->step(token, row, memory); ++batches;
                    }
                    const auto snapshot = arms[0]->accepted_snapshot();
                    // All inputs are narrower than the split threshold. QSA's
                    // only extra work in these five positions is3 dispatches per
                    // layer on the fourth token's completed compression block.
                    const uint64_t dispatches = c.dispatches + ((position + 1) % 4 == 0 ? 3 * c.attention_layers.size() : 0);
                    for (size_t i = 0; i < arms.size(); ++i) {
                        const auto& result = results[i];
                        require(result.position == position && arms[i]->position() == position + 1 && arms[i]->accepted_snapshot() == snapshot,
                            "generic replay accepted composite state differs");
                        require(result.logits.size() == shape.vocabulary && std::memcmp(result.logits.data(), results[0].logits.data(), result.logits.size() * sizeof(float)) == 0,
                            "generic replay logits differ in bits");
                        require(result.batch.dispatches == dispatches && result.batch.capture_executes == (i == 2 ? c.captures / 2 : 0),
                            "generic replay fixed physical/capture dispatch counts differ");
                        for (size_t j = 0; j < reference.size(); ++j) {
                            const auto error = std::abs(result.logits[j] - reference[j]) / std::max(1.0, std::abs(reference[j])); worst = std::max(worst, error);
                            require(std::isfinite(result.logits[j]) && error <= 5e-5, "generic replay differs from independent double-reference chain");
                        }
                    }
                    if (!round) { first_logits.push_back(results[0].logits); first_snapshots.push_back(snapshot); }
                    else require(first_snapshots[position] == snapshot && std::memcmp(first_logits[position].data(), results[0].logits.data(), results[0].logits.size() * sizeof(float)) == 0,
                        "generic odd/even reset changed physical replay results");
                    history = ids.next;
                    const auto now = graph.memory();
                    require(now.live_bytes == resident.live_bytes && now.allocations == resident.allocations && now.live_captures == resident.live_captures &&
                        now.capture_resource_bytes == resident.capture_resource_bytes, "generic replay step/reset changed resident ledger");
                }
            }
        }
        require(graph.memory().live_bytes == baseline.live_bytes && graph.memory().live_captures == baseline.live_captures &&
            graph.memory().capture_resource_bytes == baseline.capture_resource_bytes, "generic replay captures/state did not drain");
    }
}
void checkpoint_cases(MetalGraph& graph, const Fixture& f, StackShape shape, StackPolicy policy,
                      const NgramSpec& spec, bool known) {
    // Generated state-only differential checks. No target/session logits or RNG
    // restoration is implied by receiving a model-state checkpoint handle.
    const auto baseline = graph.memory();
    {
        TextStack stack(graph, f.weights, shape, policy, spec, StackExecution::direct, known ? 8 : 0);
        TextStack cold(graph, f.weights, shape, policy, spec);
        const auto initial = graph.memory(); const auto empty = stack.accepted_snapshot();
        const auto none = stack.checkpoint_memory();
        require(!none.slots && !none.retained && !none.gpu_charged_bytes && !none.host_metadata_bytes,
            "ordinary construction silently prepared checkpoint resources");
        reject([&] { static_cast<void>(stack.retain_checkpoint()); });
        reject([&] { static_cast<void>(stack.restore_checkpoint({})); });
        for (const auto config : {StackCheckpointConfig{0, 1 << 20}, {5, 1 << 20}, {1, 0},
                                  {1, maximum_stack_checkpoint_budget + 1}, {1, 1}})
            reject([&] { stack.prepare_checkpoints(config); });
        graph.begin(); reject([&] { stack.prepare_checkpoints({4, 16 << 20}); }); graph.abort();
        if (!known) {
            const auto fixed = graph.kernel_policy(); const auto before = stack.lookup(0).rows;
            graph.configure_block_gemv(fixed[1] ^ 1);
            require(stack.lookup(0).rows == before, "unprepared empty stack lost existing policy-change admission");
            stack.reset(); graph.configure_block_gemv(fixed[1]); stack.reset();
        }
        require(stack.accepted_snapshot() == empty && graph.memory().live_bytes == initial.live_bytes,
            "checkpoint preparation refusal changed accepted state/resources");
        stack.prepare_checkpoints({4, 16 << 20});
        const auto prepared = stack.checkpoint_memory();
        require(prepared.slots == 4 && !prepared.retained && prepared.payload_bytes_per_slot &&
            prepared.host_metadata_bytes && prepared.gpu_charged_bytes == graph.memory().live_bytes - initial.live_bytes,
            "checkpoint charge does not match actual graph ledger");
        reject([&] { stack.prepare_checkpoints({1, 16 << 20}); });
        StackCheckpoint destroyed_origin;
        {
            TextStack other(graph, f.weights, shape, policy, spec);
            const auto before = graph.memory().live_bytes;
            // Exercise actual-allocation padding when this device reports it;
            // never assume the driver's allocation granularity from payloads.
            const auto refused_budget = prepared.gpu_charged_bytes > prepared.payload_bytes_per_slot * 4
                ? prepared.payload_bytes_per_slot * 2 + prepared.host_metadata_bytes : 1;
            reject([&] { other.prepare_checkpoints({2, refused_budget}); });
            require(!other.checkpoint_memory().slots && graph.memory().live_bytes == before,
                "rounded checkpoint budget failure leaked a partial store");
            other.prepare_checkpoints({1, 16 << 20});
            const auto foreign = other.retain_checkpoint();
            reject([&] { static_cast<void>(stack.restore_checkpoint(foreign)); });
            destroyed_origin = foreign;
        }
        {
            TextStack replacement(graph, f.weights, shape, policy, spec);
            replacement.prepare_checkpoints({1, 16 << 20});
            const auto own = replacement.retain_checkpoint();
            const auto before = replacement.accepted_snapshot();
            reject([&] { static_cast<void>(replacement.restore_checkpoint(destroyed_origin)); });
            require(replacement.position() == 0 && replacement.accepted_snapshot() == before &&
                replacement.checkpoint_memory().retained == 1, "destroyed-origin handle damaged replacement state or checkpoint");
            require(replacement.restore_checkpoint(own) == 0, "destroyed-origin refusal invalidated replacement checkpoint");
        }
        for (const auto execution : {StackExecution::indirect_compatible, StackExecution::recurrent_replay}) {
            TextStack incompatible(graph, f.weights, shape, policy, spec, execution);
            reject([&] { incompatible.prepare_checkpoints({1, 16 << 20}); });
            if (execution == StackExecution::recurrent_replay) {
                incompatible.configure_execution(StackExecution::direct);
                reject([&] { incompatible.prepare_checkpoints({1, 16 << 20}); });
            }
        }
        const auto resident = graph.memory();
        const auto ledger = [&] {
            const auto now = graph.memory();
            require(now.live_bytes == resident.live_bytes && now.allocations == resident.allocations,
                "prepared retain/restore/step/reset changed GPU resource ledger");
        };
        const auto advance = [&](TextStack& target, std::span<const uint32_t> tokens, bool tiled) {
            std::vector<NgramStep> ids;
            if (tiled) ids = target.lookups(tokens);
            else { require(tokens.size() == 1, "serial checkpoint fixture step width"); ids.push_back(target.lookup(tokens[0])); }
            std::vector<float> rows(tokens.size() * shape.residual.channels), memory(tokens.size() * shape.memory.embedding);
            for (size_t t = 0; t < tokens.size(); ++t) {
                for (uint32_t i = 0; i < shape.residual.channels; ++i)
                    rows[t * shape.residual.channels + i] = float(int((tokens[t] * 7 + i * 3) % 23) - 11) / 16;
                for (uint32_t i = 0; i < shape.memory.embedding; ++i)
                    memory[t * shape.memory.embedding + i] = float(int(ids[t].rows[i / spec.row_width] % 17) - 8) / 32;
            }
            ++batches;
            return tiled ? target.step_known(tokens, rows, memory) : target.step(tokens[0], rows, memory);
        };
        const std::array<uint32_t, 12> original{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0};
        const auto seed_cold = [&](uint32_t prefix) {
            cold.reset();
            for (uint32_t i = 0; i < prefix; ++i) static_cast<void>(advance(cold, std::span(original).subspan(i, 1), false));
        };
        const auto compare = [&] {
            require(stack.position() == cold.position() && stack.accepted_snapshot() == cold.accepted_snapshot() &&
                stack.lookup(6).rows == cold.lookup(6).rows, "restored composite state/ngram history differs from cold replay");
        };
        const auto zero_prefix = stack.retain_checkpoint();
        if (known) static_cast<void>(advance(stack, std::span(original).first(5), true));
        else for (size_t i = 0; i < 5; ++i) static_cast<void>(advance(stack, std::span(original).subspan(i, 1), false));
        require(stack.position() == 5 && stack.restore_checkpoint(zero_prefix) == 0 && stack.accepted_snapshot() == empty,
            "empty-prefix restore from nonempty state retained model history");
        compare(); ledger();
        const std::array<uint32_t, 5> zero_branch{8, 7, 6, 5, 4};
        StackResult zero_actual{}, zero_expected{};
        if (known) zero_actual = advance(stack, zero_branch, true);
        else for (size_t i = 0; i < zero_branch.size(); ++i) zero_actual = advance(stack, std::span(zero_branch).subspan(i, 1), false);
        for (size_t i = 0; i < zero_branch.size(); ++i) zero_expected = advance(cold, std::span(zero_branch).subspan(i, 1), false);
        require(zero_actual.logits == zero_expected.logits, "new branch after empty-prefix restore diverged from cold replay");
        compare(); ledger();
        stack.reset(); cold.reset(); compare(); ledger();
        const auto h0 = stack.retain_checkpoint();
        std::array<StackCheckpoint, 3> boundary;
        for (uint32_t i = 0; i < 8; ++i) {
            static_cast<void>(advance(stack, std::span(original).subspan(i, 1), false));
            if (i >= 2 && i <= 4) boundary[i - 2] = stack.retain_checkpoint(); // P=3,4,5 around compression=4.
        }
        const auto h8 = stack.retain_checkpoint(); // FIFO eviction of h0.
        reject([&] { static_cast<void>(stack.restore_checkpoint(h0)); });
        reject([&] { stack.prepare_checkpoints({1, 16 << 20}); });
        const auto full = stack.accepted_snapshot();
        reject([&] { static_cast<void>(stack.step(0, {}, {})); });
        if (known) reject([&] { static_cast<void>(stack.step_known(std::span(original).first(2), {}, {})); });
        graph.begin();
        reject([&] { static_cast<void>(stack.retain_checkpoint()); });
        reject([&] { static_cast<void>(stack.restore_checkpoint(boundary[2])); });
        reject([&] { stack.reset(); }); graph.abort();
        require(stack.accepted_snapshot() == full && stack.checkpoint_memory().retained == 4,
            "active-batch checkpoint refusal changed state or handles");
        bool refused_thread = false;
        std::thread worker([&] {
            try { static_cast<void>(stack.restore_checkpoint(boundary[2])); } catch (const FormatError&) { refused_thread = true; }
        });
        worker.join(); require(refused_thread, "foreign thread restored checkpoint state");
        for (uint32_t prefix : {5, 5, 4, 3, 3}) {
            require(stack.restore_checkpoint(boundary[prefix - 3]) == prefix, "checkpoint restore returned wrong prefix");
            seed_cold(prefix); compare(); ledger();
        }
        reject([&] { static_cast<void>(stack.restore_checkpoint(h8)); });
        reject([&] { static_cast<void>(stack.restore_checkpoint(boundary[1])); });
        reject([&] { static_cast<void>(stack.restore_checkpoint(boundary[2])); });
        const auto equal_prefix = stack.retain_checkpoint();
        const std::array<uint32_t, 8> branch{8, 7, 6, 5, 4, 3, 2, 1};
        const auto before_failure = stack.accepted_snapshot(); const auto retained = stack.checkpoint_memory().retained;
        const auto head = graph.read(f.weights.output.view); auto poisoned = head;
        const float invalid = std::numeric_limits<float>::infinity();
        std::memcpy(poisoned.data(), &invalid, sizeof(invalid)); graph.write(f.weights.output.view, poisoned);
        // Dirty a discarded continuation across compression=4, then restore
        // and append different tokens so stale future rows cannot mask a loss.
        reject([&] {
            if (known) static_cast<void>(advance(stack, std::span(original).subspan(3, 8), true));
            else static_cast<void>(advance(stack, std::span(original).subspan(3, 1), false));
        });
        graph.write(f.weights.output.view, head);
        require(stack.position() == 3 && stack.accepted_snapshot() == before_failure &&
            stack.checkpoint_memory().retained == retained, "late GPU failure damaged accepted state or retained checkpoints");
        require(stack.restore_checkpoint(equal_prefix) == 3, "late GPU failure invalidated retained checkpoint");
        compare(); ledger();
        StackResult actual{};
        if (known) actual = advance(stack, branch, true);
        else for (size_t i = 0; i < branch.size(); ++i) actual = advance(stack, std::span(branch).subspan(i, 1), false);
        StackResult expected{};
        for (size_t i = 0; i < branch.size(); ++i) expected = advance(cold, std::span(branch).subspan(i, 1), false);
        require(actual.logits == expected.logits, "branch append after checkpoint changed target logits"); compare(); ledger();
        const auto branch_handle = stack.retain_checkpoint();
        require(stack.restore_checkpoint(boundary[0]) == 3, "ancestor did not survive divergent append");
        require(stack.restore_checkpoint(equal_prefix) == 3, "equal-prefix handle did not survive divergent append");
        seed_cold(3); compare();
        reject([&] { static_cast<void>(stack.restore_checkpoint(branch_handle)); });
        // A fresh branch after both odd and even bank swaps must retain exact
        // QSA completed keys and the partial compression sum at P=3.
        const auto continued = advance(stack, std::span(original).subspan(3, 1), false);
        const auto replayed = advance(cold, std::span(original).subspan(3, 1), false);
        require(continued.logits == replayed.logits, "ordinary continuation after restore diverged"); compare();
        const auto snap_before_ssd = stack.accepted_snapshot();
        std::vector<std::byte> ssd_payload(size_t(stack.ssd_checkpoint_payload_bytes(4)));
        require(stack.read_ssd_checkpoint_span(0, ssd_payload), "SSD checkpoint span read failed");
        std::vector<std::byte> bad_ssd = ssd_payload;
        const uint32_t wrong_prefix = 99;
        std::memcpy(bad_ssd.data(), &wrong_prefix, sizeof(wrong_prefix));
        require(!stack.write_ssd_checkpoint_span(2, 0, std::span(bad_ssd).first(size_t(stack.ssd_checkpoint_payload_bytes(2)))) &&
            stack.position() == 4 && stack.accepted_snapshot() == snap_before_ssd &&
            stack.restore_checkpoint(boundary[0]) == 3,
            "refused SSD checkpoint span write mutated live stack or invalidated ancestor checkpoint");
        require(stack.write_ssd_checkpoint_span(4, 0, ssd_payload) &&
            stack.position() == 4 && stack.accepted_snapshot() == snap_before_ssd,
            "SSD checkpoint span round-trip diverged");
        stack.reset(); require(stack.accepted_snapshot() == empty && !stack.checkpoint_memory().retained,
            "reset failed to invalidate checkpoint lineage");
        reject([&] { static_cast<void>(stack.restore_checkpoint(boundary[0])); });
        const auto empty_handle = stack.retain_checkpoint(); const auto fixed = graph.kernel_policy();
        graph.configure_block_gemv(fixed[1] ^ 1);
        reject([&] { static_cast<void>(stack.lookup(0)); });
        reject([&] { static_cast<void>(stack.retain_checkpoint()); });
        reject([&] { static_cast<void>(stack.restore_checkpoint(empty_handle)); });
        reject([&] { stack.reset(); });
        graph.configure_block_gemv(fixed[1]);
        if (known) reject([&] { stack.configure_known_residual_reuse(true); });
        require(stack.restore_checkpoint(empty_handle) == 0 && stack.accepted_snapshot() == empty,
            "refused policy change damaged an empty-prefix checkpoint"); ledger();
    }
    require(graph.memory().live_bytes == baseline.live_bytes, "checkpoint store did not release its graph charges");
}
void checkpoint_connected(MetalGraph& graph) {
    const StackShape shape{{2, 8, 3}, {8, 2, 4, 4, 4, 4}, {8, 4, 2, 4, 2, 4, 4, 4, 8, 64},
        {8, 8, 4, 2}, {2, 8, 4, 4, 3}, 13, 1};
    const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true,
        {HashOrder::newest_first, NgramBoundary::literal, true}};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    Assets assets{graph, std::mt19937(0x63706f69), {}, {}, {}}; const auto f = fixture(assets, shape);
    checkpoint_cases(graph, f, shape, policy, spec, false);
}
void verification_cases(MetalGraph& graph, const Fixture& fixture, StackShape shape, StackPolicy policy, const NgramSpec& spec) {
    // Generated synchronous verifier differential only; no target, sampler,
    // cancellation transport, real-model or performance qualification.
    const auto baseline = graph.memory();
    {
        Assets mutable_weights{graph, std::mt19937(0x76657269), {}, {}, {}}; auto weights = fixture.weights;
        auto& first = std::get<RecurrentWeights>(weights.layers.front()).attention;
        const auto convolution = graph.read(first.convolution);
        first.convolution = graph.upload(convolution, "verification mutable convolution fixture", false);
        mutable_weights.views.push_back(first.convolution);
        const auto head = graph.read(weights.output.view);
        auto poisoned_convolution = convolution; const float infinity = std::numeric_limits<float>::infinity();
        std::memcpy(poisoned_convolution.data(), &infinity, sizeof(infinity));
        TextStack verified(graph, weights, shape, policy, spec), serial(graph, weights, shape, policy, spec);
        const auto empty = verified.accepted_snapshot(); const auto unprepared = graph.memory();
        const auto off = verified.verification_memory();
        require(!off.prepared && !off.gpu_charged_bytes && !off.host_charged_bytes && !off.tape_payload_bytes,
            "ordinary construction silently prepared verification resources");
        reject([&] { static_cast<void>(verified.verification_lookups({})); });
        reject([&] { static_cast<void>(verified.verify_greedy2({}, {}, {}, 0, {}, 2)); });
        for (uint64_t budget : {uint64_t(0), uint64_t(1), maximum_stack_verification_budget + 1})
            reject([&] { verified.prepare_transient_verification(budget); });
        graph.begin(); reject([&] { verified.prepare_transient_verification(64ull << 20); }); graph.abort();
        require(graph.memory().live_bytes == unprepared.live_bytes && verified.accepted_snapshot() == empty,
            "verification preparation refusal changed state/resources");
        for (const auto mode : {StackExecution::indirect_compatible, StackExecution::recurrent_replay}) {
            const auto before = graph.memory();
            {
                TextStack incompatible(graph, weights, shape, policy, spec, mode); const auto resident = graph.memory();
                reject([&] { incompatible.prepare_transient_verification(64ull << 20); });
                if (mode == StackExecution::recurrent_replay) {
                    incompatible.configure_execution(StackExecution::direct);
                    reject([&] { incompatible.prepare_transient_verification(64ull << 20); });
                }
                require(!incompatible.verification_memory().prepared && graph.memory().live_bytes == resident.live_bytes,
                    "incompatible execution prepared verifier resources");
            }
            require(graph.memory().live_bytes == before.live_bytes && graph.memory().live_captures == before.live_captures,
                "incompatible verifier preparation leaked stack/capture resources");
        }
        {
            const auto before = graph.memory();
            {
                TextStack nonempty(graph, weights, shape, policy, spec);
                static_cast<void>(nonempty.step(0, std::vector<float>(shape.residual.channels), std::vector<float>(shape.memory.embedding))); ++batches;
                const auto accepted = nonempty.accepted_snapshot(); const auto resident = graph.memory();
                reject([&] { nonempty.prepare_transient_verification(64ull << 20); });
                require(nonempty.position() == 1 && nonempty.accepted_snapshot() == accepted && !nonempty.verification_memory().prepared &&
                    graph.memory().live_bytes == resident.live_bytes, "nonempty verifier preparation changed accepted state/resources");
            }
            require(graph.memory().live_bytes == before.live_bytes, "nonempty verifier preparation leaked stack resources");
        }
        verified.prepare_checkpoints({1, 16ull << 20});
        const auto before_prepare = graph.memory(); verified.prepare_transient_verification(64ull << 20);
        const auto prepared = verified.verification_memory();
        require(prepared.prepared && prepared.tape_payload_bytes && prepared.host_charged_bytes &&
            prepared.gpu_charged_bytes == graph.memory().live_bytes - before_prepare.live_bytes &&
            prepared.gpu_charged_bytes + prepared.host_charged_bytes <= (64ull << 20),
            "verification charges differ from graph ledger or explicit budget");
        reject([&] { verified.prepare_transient_verification(64ull << 20); });
        {
            TextStack refused(graph, weights, shape, policy, spec); const auto before = graph.memory();
            // Admit the same host capacities, then force GPU admission cleanup.
            reject([&] { refused.prepare_transient_verification(prepared.host_charged_bytes + 1); });
            require(!refused.verification_memory().prepared && graph.memory().live_bytes == before.live_bytes,
                "partial verification preparation leaked resources");
        }
        const auto resident = graph.memory();
        const auto ledger = [&] {
            const auto now = graph.memory();
            require(now.live_bytes == resident.live_bytes && now.allocations == resident.allocations &&
                now.live_captures == resident.live_captures, "verification changed resident GPU resource ledger");
        };
        const auto token_row = [&](uint32_t token) {
            std::vector<float> row(shape.residual.channels);
            for (size_t i = 0; i < row.size(); ++i) row[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16;
            return row;
        };
        const auto memory_row = [&](std::span<const uint64_t> ids) {
            std::vector<float> row(shape.memory.embedding);
            for (size_t i = 0; i < row.size(); ++i) row[i] = float(int(ids[i / spec.row_width] % 17) - 8) / 32;
            return row;
        };
        const auto advance = [&](TextStack& target, uint32_t token) {
            const auto ids = target.lookup(token); const auto row = token_row(token), memory = memory_row(ids.rows);
            ++batches; return target.step(token, row, memory);
        };
        const auto seed = [&](uint32_t prefix) {
            verified.reset(); serial.reset(); StackResult result{};
            for (uint32_t i = 0; i < prefix; ++i) {
                result = advance(verified, i % shape.vocabulary); const auto cold = advance(serial, i % shape.vocabulary);
                require(result.logits == cold.logits, "verification seed differs from serial");
            }
            require(verified.accepted_snapshot() == serial.accepted_snapshot(), "verification seed composite state differs");
            return result;
        };
        const auto argmax = [](std::span<const float> logits) {
            return uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        };
        struct Inputs { std::array<uint32_t, 2> tokens; std::vector<float> rows, memory; };
        const auto inputs = [&](TextStack& target, std::array<uint32_t, 2> tokens) {
            Inputs out{tokens, {}, {}}; const auto staged = target.verification_lookups(tokens);
            const auto first_lookup = target.lookup(tokens[0]);
            const auto second_lookup = ngram_step(spec, policy.ngram, first_lookup.next, tokens[1]);
            const std::array<const NgramStep*, 2> expected{&first_lookup, &second_lookup};
            for (uint32_t t = 0; t < 2; ++t) {
                require(staged[t].next.previous == expected[t]->next.previous &&
                    std::equal(staged[t].rows.begin(), staged[t].rows.end(), expected[t]->rows.begin()),
                    "verification borrowed lookup rows are not chronological");
                const auto row = token_row(tokens[t]), memory = memory_row(staged[t].rows);
                out.rows.insert(out.rows.end(), row.begin(), row.end()); out.memory.insert(out.memory.end(), memory.begin(), memory.end());
            }
            return out;
        };
        const auto same_logits = [&](std::span<const float> actual, const std::vector<float>& expected) {
            require(actual.size() == expected.size() && std::equal(actual.begin(), actual.end(), expected.begin()),
                "verification returned logits for another committed prefix");
        };
        const auto same_state = [&](TextStack& target) {
            require(target.position() == serial.position() && target.accepted_snapshot() == serial.accepted_snapshot() &&
                target.lookup(6).rows == serial.lookup(6).rows, "verification composite state/ngram differs from serial");
        };
        const auto empty_inputs = inputs(verified, {0, 1}); const std::vector<float> empty_logits(shape.vocabulary, 0);
        reject([&] { static_cast<void>(verified.verify_greedy2(empty_inputs.tokens, empty_inputs.rows, empty_inputs.memory, 0, empty_logits, 2)); });
        for (uint32_t prefix : {1, 2, 3, 4, 7, 11}) for (uint32_t accepted : {0, 1, 2}) {
            const auto base = seed(prefix); std::array<uint32_t, 2> tokens{argmax(base.logits), 0}; auto expected = base.logits;
            const auto first_source = base.logits[tokens[0]]; float second_source = 0;
            if (!accepted) tokens[0] = (tokens[0] + 1) % shape.vocabulary;
            else {
                const auto first_result = advance(serial, tokens[0]); expected = first_result.logits; tokens[1] = argmax(expected);
                second_source = expected[tokens[1]];
                if (accepted == 1) tokens[1] = (tokens[1] + 1) % shape.vocabulary;
                else expected = advance(serial, tokens[1]).logits;
            }
            const auto proposal = inputs(verified, tokens);
            const auto result = verified.verify_greedy2(tokens, proposal.rows, proposal.memory, prefix, base.logits, 2);
            batches += bool(result.verification.dispatches) + bool(result.recovery.dispatches);
            require(result.accepted == accepted && result.position == prefix && verified.position() == prefix + accepted &&
                result.source_logits[0] == first_source && (!accepted || accepted == 1 || result.source_logits[1] == second_source) &&
                bool(result.verification.dispatches) == bool(accepted) && !result.verification.capture_executes &&
                result.recovery.dispatches == (accepted == 1 ? 6U : 0U) && !result.recovery.capture_executes &&
                std::isfinite(result.wall_ms) && result.wall_ms >= 0, "verification publication/dispatch contract differs");
            same_logits(result.logits, expected); same_state(verified); ledger();
            const auto continued = advance(verified, 6), cold = advance(serial, 6);
            require(continued.logits == cold.logits, "ordinary continuation after verification diverged"); same_state(verified); ledger();
        }
        auto base = seed(3); const uint32_t first_id = argmax(base.logits);
        auto proposal = inputs(verified, {first_id, (first_id + 1) % shape.vocabulary});
        const auto before = verified.accepted_snapshot();
        for (uint32_t remaining : {0, 1}) {
            const auto result = verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, remaining);
            require(!result.accepted && !result.verification.dispatches && !result.recovery.dispatches &&
                verified.accepted_snapshot() == before, "short output reservation executed verifier work"); same_logits(result.logits, base.logits);
        }
        const std::array<uint32_t, 1> first_stop{first_id};
        const auto stopped = verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2, first_stop);
        require(!stopped.accepted && !stopped.verification.dispatches && verified.accepted_snapshot() == before,
            "first-token stop did not fall back without GPU work");
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, {}, proposal.memory, 3, base.logits, 2)); });
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, {}, 3, base.logits, 2)); });
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, base.position, base.logits, 2)); });
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, {}, 2)); });
        auto bad_rows = proposal.rows; bad_rows.back() = infinity;
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, bad_rows, proposal.memory, 3, base.logits, 2)); });
        auto bad_memory = proposal.memory; bad_memory.back() = infinity;
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, bad_memory, 3, base.logits, 2)); });
        auto bad_logits = base.logits; bad_logits.back() = infinity;
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, bad_logits, 2)); });
        const std::array<uint32_t, 1> wrong_width{0}; const std::array<uint32_t, 2> bad_token{0, shape.vocabulary};
        const std::array<uint32_t, 3> too_many{0, 1, 2};
        reject([&] { static_cast<void>(verified.verification_lookups({})); });
        reject([&] { static_cast<void>(verified.verification_lookups(wrong_width)); });
        reject([&] { static_cast<void>(verified.verification_lookups(too_many)); });
        reject([&] { static_cast<void>(verified.verification_lookups(bad_token)); });
        const std::array<uint32_t, 1> bad_stop{shape.vocabulary};
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2, bad_stop)); });
        graph.begin();
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2)); }); graph.abort();
        bool foreign_thread = false;
        std::thread worker([&] { try { static_cast<void>(verified.verification_lookups(proposal.tokens)); } catch (const FormatError&) { foreign_thread = true; } });
        worker.join(); require(foreign_thread, "foreign thread staged verification");
        const auto fixed = graph.kernel_policy(); graph.configure_block_gemv(fixed[1] ^ 1);
        reject([&] { static_cast<void>(verified.verification_lookups(proposal.tokens)); }); reject([&] { verified.reset(); });
        graph.configure_block_gemv(fixed[1]);
        require(verified.accepted_snapshot() == before && verified.position() == 3, "verification refusal changed accepted state"); ledger();

        struct Probe {
            TextStack* stack; uint32_t at, polls = 0; bool throw_error = false;
            MetalGraph* graph = nullptr; GpuView convolution{}; const std::vector<std::byte>* poison = nullptr;
            static bool requested(void* opaque) {
                auto& p = *static_cast<Probe*>(opaque); ++p.polls;
                if (p.polls == 1) {
                    reject([&] { static_cast<void>(p.stack->position()); }); reject([&] { p.stack->reset(); });
                }
                if (p.polls != p.at) return false;
                if (p.poison) { p.graph->write(p.convolution, *p.poison); return false; }
                if (p.throw_error) throw FormatError("fixture cancellation probe failed");
                return true;
            }
        };
        for (uint32_t cancel_at : {1, 2, 3}) {
            base = seed(3); const auto correct = argmax(base.logits); const auto first_result = advance(serial, correct);
            proposal = inputs(verified, {correct, (argmax(first_result.logits) + 1) % shape.vocabulary});
            const auto untouched = verified.accepted_snapshot(); const auto checkpoint = verified.retain_checkpoint();
            Probe probe{&verified, cancel_at};
            reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2,
                {}, {Probe::requested, &probe})); });
            require(probe.polls == cancel_at && verified.position() == 3 && verified.accepted_snapshot() == untouched &&
                verified.restore_checkpoint(checkpoint) == 3, "cancelled verification damaged base/checkpoint");
            serial.reset(); for (uint32_t i = 0; i < 3; ++i) static_cast<void>(advance(serial, i));
            require(advance(verified, 6).logits == advance(serial, 6).logits, "cancelled candidate tail affected ordinary append");
            same_state(verified); ledger();
        }
        base = seed(3); const auto full_first = argmax(base.logits); const auto full_first_result = advance(serial, full_first);
        proposal = inputs(verified, {full_first, argmax(full_first_result.logits)});
        const auto full_base = verified.accepted_snapshot(); Probe full_cancel{&verified, 2};
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2,
            {}, {Probe::requested, &full_cancel})); });
        require(full_cancel.polls == 2 && verified.position() == 3 && verified.accepted_snapshot() == full_base,
            "postforward cancellation published fully matching candidates"); ledger();
        base = seed(3); auto first_result = advance(serial, argmax(base.logits));
        proposal = inputs(verified, {argmax(base.logits), (argmax(first_result.logits) + 1) % shape.vocabulary});
        const auto intact = verified.accepted_snapshot(); Probe throwing{&verified, 2}; throwing.throw_error = true;
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2,
            {}, {Probe::requested, &throwing})); });
        require(verified.accepted_snapshot() == intact && throwing.polls == 2, "throwing cancellation probe published state");
        auto poisoned_head = head; std::memcpy(poisoned_head.data(), &infinity, sizeof(infinity)); graph.write(weights.output.view, poisoned_head);
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2)); });
        graph.write(weights.output.view, head);
        require(verified.accepted_snapshot() == intact, "late forward GPU failure published verifier state");
        Probe recovery_failure{&verified, 2}; recovery_failure.graph = &graph; recovery_failure.convolution = first.convolution;
        recovery_failure.poison = &poisoned_convolution;
        reject([&] { static_cast<void>(verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2,
            {}, {Probe::requested, &recovery_failure})); });
        graph.write(first.convolution, convolution);
        require(verified.accepted_snapshot() == intact && recovery_failure.polls == 2, "recovery GPU failure damaged accepted composite state");
        serial.reset(); for (uint32_t i = 0; i < 3; ++i) static_cast<void>(advance(serial, i));
        require(advance(verified, 6).logits == advance(serial, 6).logits, "failed recovery tail affected ordinary append"); same_state(verified); ledger();

        // Exact ties use a genuine all-zero output head and authoritative logits.
        graph.write(weights.output.view, std::vector<std::byte>(head.size())); base = seed(3);
        require(argmax(base.logits) == 0, "zero head did not produce lowest-ID tie");
        proposal = inputs(verified, {0, 1}); const auto tie_expected = advance(serial, 0);
        const std::array<uint32_t, 1> zero_stop{0};
        const auto stopped_zero = verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2, zero_stop);
        require(!stopped_zero.accepted && !stopped_zero.verification.dispatches && verified.position() == 3,
            "zero-valued first stop did not bypass GPU verification");
        const auto tied = verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2);
        batches += bool(tied.verification.dispatches) + bool(tied.recovery.dispatches);
        require(tied.accepted == 1, "exact greedy tie did not choose lowest token ID"); same_logits(tied.logits, tie_expected.logits); same_state(verified);
        graph.write(weights.output.view, head); ledger();
        base = seed(3); const auto checkpoint = verified.retain_checkpoint(); const auto checkpoint_state = verified.accepted_snapshot();
        const auto id0 = argmax(base.logits); first_result = advance(serial, id0); const auto id1 = argmax(first_result.logits);
        proposal = inputs(verified, {id0, id1}); const auto second_result = advance(serial, id1);
        const auto both = verified.verify_greedy2(proposal.tokens, proposal.rows, proposal.memory, 3, base.logits, 2);
        batches += bool(both.verification.dispatches) + bool(both.recovery.dispatches);
        require(both.accepted == 2, "checkpoint coexistence verification did not accept matching pair"); same_logits(both.logits, second_result.logits); same_state(verified);
        require(verified.restore_checkpoint(checkpoint) == 3 && verified.accepted_snapshot() == checkpoint_state,
            "verification invalidated or overwrote ancestor checkpoint"); ledger();
        base = seed(shape.attention.capacity - 1);
        reject([&] { static_cast<void>(verified.verification_lookups(std::array<uint32_t, 2>{0, 1})); }); ledger();

        // Narrow same-resident known-input comparator: verifier resources must
        // not change ordinary all-N publication or its immutable reuse policy.
        {
            TextStack known(graph, weights, shape, policy, spec, StackExecution::direct, 8);
            known.prepare_transient_verification(64ull << 20); const auto known_resident = graph.memory();
            reject([&] { known.configure_known_residual_reuse(true); });
            for (uint32_t accepted : {1, 2}) {
                known.reset(); serial.reset(); const auto prefill = inputs(known, {0, 1});
                const auto known_base = known.step_known(prefill.tokens, prefill.rows, prefill.memory);
                ++batches;
                static_cast<void>(advance(serial, 0)); const auto serial_base = advance(serial, 1);
                require(known_base.logits == serial_base.logits && known.position() == 2, "resident verifier changed known-input prefill"); same_state(known);
                const auto t0 = argmax(known_base.logits); auto expected = advance(serial, t0).logits; auto t1 = argmax(expected);
                if (accepted == 1) t1 = (t1 + 1) % shape.vocabulary; else expected = advance(serial, t1).logits;
                const auto pair = inputs(known, {t0, t1});
                const auto result = known.verify_greedy2(pair.tokens, pair.rows, pair.memory, known.position(), known_base.logits, 2);
                batches += bool(result.verification.dispatches) + bool(result.recovery.dispatches);
                require(result.accepted == accepted, "known-input/verifier coexistence accepted wrong prefix"); same_logits(result.logits, expected); same_state(known);
                require(advance(known, 6).logits == advance(serial, 6).logits, "ordinary continuation after known/verifier coexistence differs"); same_state(known);
                require(graph.memory().live_bytes == known_resident.live_bytes && graph.memory().allocations == known_resident.allocations,
                    "known/verifier coexistence allocated during execution");
            }
        }
        require(graph.memory().live_bytes == resident.live_bytes && graph.memory().live_captures == resident.live_captures,
            "known/verifier coexistence did not release its separate resource charges");
    }
    require(graph.memory().live_bytes == baseline.live_bytes, "verification fixture leaked owned resources");
}
void known_connected(MetalGraph& graph, uint32_t channels, uint32_t memory_layer) {
    // Generated known-token prefill, not real-model quality or timing evidence.
    // Q8 GDN projections retain their stored scale/payload and scalar order;
    // all remaining operators use the same split16/block3 sequential policy.
    const StackShape shape{{2, channels, 3}, {channels, 1, 2, 128, 32, 4},
        {channels, 4, 2, 4, 2, 4, 4, 4, 8, 32},
        {channels, 8, 4, 2}, {2, channels, 4, 4, 3}, 13, memory_layer};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    Assets assets{graph, std::mt19937(0x6b6e6f77), {}, {}, {}};
    auto f = fixture(assets, shape);
    std::deque<std::vector<std::byte>> encoded;
    const auto q8 = [&](uint32_t columns, uint32_t rows) {
        encoded.emplace_back(uint64_t(columns / 32) * rows * 34);
        auto& bytes = encoded.back();
        for (uint32_t r = 0; r < rows; ++r) for (uint32_t b = 0; b < columns / 32; ++b) {
            const auto offset = (uint64_t(r) * (columns / 32) + b) * 34;
            bytes[offset] = std::byte{0}; bytes[offset + 1] = std::byte{0x1c}; // exact half 1/256
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const int8_t value = int8_t(int((r * 17 + b * 11 + lane * 7) % 31) - 15);
                std::memcpy(bytes.data() + offset + 2 + lane, &value, 1);
            }
        }
        const auto view = graph.upload(bytes, "known-token generated Q8", true); assets.views.push_back(view);
        return std::pair{GpuMatrix{view, 8}, ReferenceMatrix{bytes, 8}};
    };
    for (size_t layer = 0; layer < f.weights.layers.size(); ++layer) {
        if (auto* w = std::get_if<RecurrentWeights>(&f.weights.layers[layer])) {
            auto& ref = std::get<ReferenceLayerWeights>(f.reference[layer]).attention;
            const auto raw = uint32_t(gdn_qkv_count(shape.recurrent));
            const auto value = shape.recurrent.value_heads * shape.recurrent.value_width;
            const auto a = q8(channels, raw), b = q8(channels, value), c = q8(value, channels);
            w->attention.qkv = a.first; ref.qkv = a.second;
            w->attention.gate = b.first; ref.gate = b.second;
            w->attention.output = c.first; ref.output = c.second;
        }
    }
    const auto base = graph.memory(); const auto head = graph.read(f.weights.output.view);
    const auto ledger = [&](const GraphMemory& expected, bool transaction = true) {
        const auto now = graph.memory();
        require(now.live_bytes == expected.live_bytes && (!transaction || now.allocations == expected.allocations) &&
            now.live_captures == expected.live_captures && now.capture_resource_bytes == expected.capture_resource_bytes,
            "known-token transaction changed prepared resource ledger");
    };
    for (uint32_t variant = 0; variant < 2; ++variant) {
        const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
            {variant ? QueryPacking::all_queries_then_gates : QueryPacking::head_query_gate,
                variant ? RotaryPairs::adjacent : RotaryPairs::split_half, 1e-6f},
            variant ? RouterMode::sigmoid : RouterMode::softmax, true,
            {variant ? HashOrder::oldest_first : HashOrder::newest_first, NgramBoundary(variant), true}};
        for (const uint32_t invalid_capacity : {1, 2, 7, 9})
            reject([&] { TextStack invalid(graph, f.weights, shape, policy, spec, StackExecution::direct, invalid_capacity); });
        for (const auto mode : {StackExecution::indirect_compatible, StackExecution::recurrent_replay})
            reject([&] { TextStack invalid(graph, f.weights, shape, policy, spec, mode, 8); });
        std::vector<uint32_t> tokens(shape.attention.capacity);
        std::vector<std::vector<float>> rows, memories, logits;
        std::vector<std::vector<std::byte>> snapshots;
        std::vector<std::array<uint64_t, 4>> lookup_rows;
        auto history = empty_ngram_history(spec);
        for (uint32_t p = 0; p < tokens.size(); ++p) {
            tokens[p] = p % shape.vocabulary;
            const auto ids = ngram_step(spec, policy.ngram, history, tokens[p]);
            std::array<uint64_t, 4> ids4{}; std::copy(ids.rows.begin(), ids.rows.end(), ids4.begin()); lookup_rows.push_back(ids4);
            history = ids.next;
            rows.emplace_back(channels); memories.emplace_back(shape.memory.embedding);
            for (size_t i = 0; i < rows.back().size(); ++i) rows.back()[i] = float(int((p * 7 + i * 3) % 23) - 11) / 16;
            for (size_t i = 0; i < memories.back().size(); ++i) memories.back()[i] = float(int(ids.rows[i] % 17) - 8) / 32;
        }
        {
            TextStack serial(graph, f.weights, shape, policy, spec);
            require(serial.known_token_capacity() == 0 && !serial.known_residual_reuse(), "ordinary stack silently prepared known-token tiles/reuse");
            reject([&] { serial.configure_known_residual_reuse(false); });
            reject([&] { serial.configure_known_residual_reuse(true); });
            reject([&] { static_cast<void>(serial.lookups(std::span(tokens).first(1))); });
            reject([&] { static_cast<void>(serial.step_known(std::span(tokens).first(1), rows[0], memories[0])); });
            std::vector<CpuState> cpu(f.weights.layers.size());
            for (auto& state : cpu) {
                state.matrix.resize(gdn_state_count(shape.recurrent)); state.conv.resize(gdn_qkv_count(shape.recurrent) * 3);
                state.sum.resize(shape.attention.index_width);
                state.attention.keys.resize(shape.attention.capacity * shape.attention.kv_heads * shape.attention.head_width);
                state.attention.values.resize(state.attention.keys.size());
                state.attention.index_keys.resize(qsa_block_capacity(shape.attention) * shape.attention.index_width);
            }
            std::vector<double> memory_history(shape.memory.branches * shape.memory.channels * 3 * shape.memory.dilation);
            for (uint32_t p = 0; p < tokens.size(); ++p) {
                const auto expected = cpu_step(f, cpu, memory_history, rows[p], memories[p], p, shape, policy);
                const auto result = serial.step(tokens[p], rows[p], memories[p]); ++batches;
                for (size_t i = 0; i < expected.size(); ++i) {
                    const auto error = std::abs(result.logits[i] - expected[i]) / std::max(1.0, std::abs(expected[i])); worst = std::max(worst, error);
                    require(std::isfinite(result.logits[i]) && error <= 5e-5, "known-token serial reference differs");
                }
                logits.push_back(result.logits); snapshots.push_back(serial.accepted_snapshot());
            }
        }
        ledger(base, false);
        reject([&] { TextStack invalid(graph, f.weights, shape, policy, spec, StackExecution::direct, 0, true); });
        ledger(base, false);
        for (const bool reuse : {false, true}) {
            TextStack tiled(graph, f.weights, shape, policy, spec, StackExecution::direct, 8, reuse);
            require(tiled.known_token_capacity() == 8 && !tiled.replay_prepared() && tiled.known_residual_reuse() == reuse, "explicit known-token preparation differs");
            const auto resident = graph.memory(); const auto empty = tiled.accepted_snapshot();
            for (const uint32_t prefix : {0, 1, 2, 3, 7, 11, 19}) for (uint32_t n = 1; n <= 8; ++n) {
                tiled.reset(); require(tiled.accepted_snapshot() == empty, "known-token reset did not restore empty state");
                tiled.configure_known_residual_reuse(!reuse);
                require(tiled.known_residual_reuse() == !reuse, "empty-prefix residual reuse switch ignored");
                tiled.configure_known_residual_reuse(reuse);
                require(graph.memory().allocations == resident.allocations, "residual selection allocated a new GPU buffer");
                for (uint32_t p = 0; p < prefix; ++p) { static_cast<void>(tiled.step(tokens[p], rows[p], memories[p])); ++batches; }
                const auto before = tiled.accepted_snapshot();
                if (prefix) {
                    reject([&] { tiled.configure_known_residual_reuse(!reuse); });
                    require(tiled.known_residual_reuse() == reuse && tiled.accepted_snapshot() == before,
                        "nonempty-prefix residual switch changed state/policy");
                }
                const auto input = std::span(tokens).subspan(prefix, n);
                const auto staged = tiled.lookups(input);
                require(staged.size() == n && tiled.position() == prefix && tiled.accepted_snapshot() == before,
                    "known-token lookup advanced accepted state");
                std::vector<float> flat_rows, flat_memory;
                for (uint32_t i = 0; i < n; ++i) {
                    require(std::equal(staged[i].rows.begin(), staged[i].rows.end(), lookup_rows[prefix + i].begin()),
                        "known-token ngram history is not chronological");
                    flat_rows.insert(flat_rows.end(), rows[prefix + i].begin(), rows[prefix + i].end());
                    flat_memory.insert(flat_memory.end(), memories[prefix + i].begin(), memories[prefix + i].end());
                }
                reject([&] { static_cast<void>(tiled.step_known(input, {}, flat_memory)); });
                reject([&] { static_cast<void>(tiled.step_known(input, flat_rows, {})); });
                auto bad = flat_rows; bad.back() = std::numeric_limits<float>::infinity();
                reject([&] { static_cast<void>(tiled.step_known(input, bad, flat_memory)); });
                require(tiled.accepted_snapshot() == before && tiled.position() == prefix, "known-token input refusal changed accepted state");
                if (prefix == 3 && (n == 2 || n == 3)) {
                    auto poisoned = head; const float invalid = std::numeric_limits<float>::infinity();
                    std::memcpy(poisoned.data(), &invalid, sizeof(invalid)); graph.write(f.weights.output.view, poisoned);
                    reject([&] { static_cast<void>(tiled.step_known(input, flat_rows, flat_memory)); });
                    require(tiled.accepted_snapshot() == before && tiled.position() == prefix,
                        "failed known-token GPU batch published partial recurrent/QSA/PLE/ngram state");
                    graph.write(f.weights.output.view, head);
                }
                const auto result = tiled.step_known(input, flat_rows, flat_memory); ++batches;
                require(result.position == prefix && tiled.position() == prefix + n && result.batch.capture_executes == 0,
                    "known-token publication advanced by the wrong count");
                require(result.logits == logits[prefix + n - 1] && tiled.accepted_snapshot() == snapshots[prefix + n - 1],
                    "known-token tile differs from identical-policy serial logits or composite state bits");
                ledger(resident);
                const auto after = tiled.step(tokens[prefix + n], rows[prefix + n], memories[prefix + n]); ++batches;
                require(after.logits == logits[prefix + n] && tiled.accepted_snapshot() == snapshots[prefix + n],
                    "ordinary continuation after known-token tile diverged");
            }
            // The real input loop chains tiles (not reset/serial seeding between
            // them), with alternating physical state/work orientations.
            tiled.reset();
            uint32_t chained = 0;
            for (const uint32_t n : {8, 7, 2, 8}) {
                std::vector<float> flat_rows, flat_memory;
                for (uint32_t i = 0; i < n; ++i) {
                    flat_rows.insert(flat_rows.end(), rows[chained + i].begin(), rows[chained + i].end());
                    flat_memory.insert(flat_memory.end(), memories[chained + i].begin(), memories[chained + i].end());
                }
                const auto result = tiled.step_known(std::span(tokens).subspan(chained, n), flat_rows, flat_memory); ++batches;
                require(result.position == chained && result.logits == logits[chained + n - 1], "chained known-token logits/base differ");
                chained += n;
                require(tiled.position() == chained && tiled.accepted_snapshot() == snapshots[chained - 1],
                    "chained known-token transaction changed composite state");
                ledger(resident);
            }
            const auto continued = tiled.step(tokens[chained], rows[chained], memories[chained]); ++batches;
            require(continued.logits == logits[chained] && tiled.accepted_snapshot() == snapshots[chained],
                "ordinary continuation after chained known-token tiles differs");
            tiled.reset();
            const auto before = tiled.accepted_snapshot();
            reject([&] { static_cast<void>(tiled.lookups({})); });
            reject([&] { static_cast<void>(tiled.lookups(std::span(tokens).first(9))); });
            const std::array<uint32_t, 1> bad_token{shape.vocabulary};
            reject([&] { static_cast<void>(tiled.lookups(bad_token)); });
            graph.configure_block_gemv(1);
            reject([&] { tiled.configure_known_residual_reuse(!reuse); });
            reject([&] { static_cast<void>(tiled.lookups(std::span(tokens).first(1))); });
            reject([&] { tiled.reset(); }); graph.configure_block_gemv(3);
            require(tiled.position() == 0 && tiled.accepted_snapshot() == before, "known-token policy refusal changed accepted state");
            for (uint32_t p = 0; p < 28; ++p) { static_cast<void>(tiled.step(tokens[p], rows[p], memories[p])); ++batches; }
            const auto near_full = tiled.accepted_snapshot();
            reject([&] { static_cast<void>(tiled.lookups(std::span(tokens).first(8))); });
            require(tiled.position() == 28 && tiled.accepted_snapshot() == near_full, "known-token capacity refusal advanced prefix");
            ledger(resident);
        }
        ledger(base, false);
        if (!variant && channels == 32) checkpoint_cases(graph, f, shape, policy, spec, true);
        if (!variant) verification_cases(graph, f, shape, policy, spec);
    }
}
void hc_q8_connected(MetalGraph& graph) {
    // Two generated layers include GDN, QSA, PLE and all five HC-down calls.
    // Not a model or quality fixture. Keep every non-HC arithmetic choice fixed.
    const StackShape shape{{4, 2560, 320}, {2560, 1, 2, 128, 32, 4},
        {2560, 4, 2, 4, 2, 4, 4, 4, 8, 16},
        {2560, 8, 4, 2}, {4, 2560, 4, 4, 3}, 13, 1};
    const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true,
        {HashOrder::newest_first, NgramBoundary::literal, true}};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    Assets assets{graph, std::mt19937(0x68637138), {}, {}, {}};
    const std::array<uint32_t, 1> attention_layers{1};
    auto f = fixture(assets, shape, 2, attention_layers);
    const auto q8 = [&](uint32_t columns, uint32_t rows) {
        std::vector<std::byte> bytes(uint64_t(columns / 32) * rows * 34);
        for (uint32_t r = 0; r < rows; ++r) for (uint32_t b = 0; b < columns / 32; ++b) {
            const auto at = (uint64_t(r) * (columns / 32) + b) * 34;
            bytes[at] = std::byte{0}; bytes[at + 1] = std::byte{0x18}; // Exact 1/512.
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const int8_t value = int8_t(int((r * 17 + b * 11 + lane * 7) % 31) - 15);
                std::memcpy(bytes.data() + at + 2 + lane, &value, 1);
            }
        }
        const auto v = graph.upload(bytes, "HC connected generated Q8"); assets.views.push_back(v); return GpuMatrix{v, 8};
    };
    for (auto& layer : f.weights.layers) {
        std::visit([&](auto& w) {
            w.attention_read.down = q8(10240, 320); w.ffn_read.down = q8(10240, 320);
        }, layer);
        if (auto* w = std::get_if<RecurrentWeights>(&layer)) {
            w->attention.qkv = q8(2560, uint32_t(gdn_qkv_count(shape.recurrent)));
            w->attention.gate = q8(2560, 64); w->attention.output = q8(64, 2560);
        }
    }
    f.weights.mixer.down = q8(10240, 320);
    const auto baseline = graph.memory();
    const std::array<uint32_t, 4> tokens{1, 3, 4, 2};
    std::vector<std::vector<float>> rows, memory;
    for (uint32_t p = 0; p < tokens.size(); ++p) {
        rows.push_back(assets.values(2560)); memory.push_back(assets.values(4));
    }
    std::vector<StackResult> ordinary;
    std::vector<std::vector<std::byte>> ordinary_state;
    {
        TextStack serial(graph, f.weights, shape, policy, spec);
        for (uint32_t p = 0; p < tokens.size(); ++p) {
            ordinary.push_back(serial.step(tokens[p], rows[p], memory[p])); ++batches;
            ordinary_state.push_back(serial.accepted_snapshot());
        }
        graph.configure_hc_q8_splits(1);
        reject([&] { static_cast<void>(serial.lookup(tokens[0])); });
        reject([&] { serial.reset(); });
        graph.configure_hc_q8_splits(0);
        require(serial.position() == tokens.size() && serial.accepted_snapshot() == ordinary_state.back(),
            "HC0 stack did not pin default-off arithmetic under an accepted prefix");
    }
    for (uint32_t count : {1, 2, 4, 8, 16, 32}) {
        graph.configure_hc_q8_splits(count);
        const auto prepared = graph.memory();
        for (const auto mode : {StackExecution::indirect_compatible, StackExecution::recurrent_replay})
            reject([&] { TextStack bad(graph, f.weights, shape, policy, spec, mode); });
        reject([&] { TextStack bad(graph, f.weights, shape, policy, spec, StackExecution::direct, 8, true); });
        require(graph.memory().live_bytes == prepared.live_bytes && graph.memory().allocations == prepared.allocations,
            "HC policy constructor refusal allocated stack resources");
        {
            TextStack serial(graph, f.weights, shape, policy, spec);
            TextStack known(graph, f.weights, shape, policy, spec, StackExecution::direct, 8, false);
            known.prepare_checkpoints({1, 8ull << 20});
            const auto fixed = graph.memory();
            reject([&] { known.configure_known_residual_reuse(true); });
            reject([&] { known.prepare_transient_verification(64ull << 20); });
            std::vector<StackResult> expected;
            std::vector<std::vector<std::byte>> states;
            for (uint32_t p = 0; p < tokens.size(); ++p) {
                expected.push_back(serial.step(tokens[p], rows[p], memory[p])); ++batches;
                states.push_back(serial.accepted_snapshot());
                require(expected.back().batch.dispatches == ordinary[p].batch.dispatches + 5,
                    "HC complete-stack dispatch accounting did not charge every reduction pass");
                if (count == 1) require(expected.back().logits == ordinary[p].logits && states.back() == ordinary_state[p],
                    "HC split1 changed complete-stack fast-block trajectory");
            }
            std::vector<float> flat_rows, flat_memory;
            for (uint32_t p = 0; p < 2; ++p) {
                flat_rows.insert(flat_rows.end(), rows[p].begin(), rows[p].end());
                flat_memory.insert(flat_memory.end(), memory[p].begin(), memory[p].end());
            }
            const auto result = known.step_known(std::span(tokens).first(2), flat_rows, flat_memory); ++batches;
            require(result.logits == expected[1].logits && known.accepted_snapshot() == states[1],
                "HC known-input preserved path differs from serial same-tier execution");
            const auto checkpoint = known.retain_checkpoint();
            for (uint32_t p = 2; p < tokens.size(); ++p) {
                const auto next = known.step(tokens[p], rows[p], memory[p]); ++batches;
                require(next.logits == expected[p].logits && known.accepted_snapshot() == states[p],
                    "HC continuation differs from serial same-tier execution");
            }
            require(known.restore_checkpoint(checkpoint) == 2 && known.accepted_snapshot() == states[1],
                "HC checkpoint restore changed accepted state");
            const auto replay = known.step(tokens[2], rows[2], memory[2]); ++batches;
            require(replay.logits == expected[2].logits && known.accepted_snapshot() == states[2],
                "HC restored continuation differs from cold same-tier prefix");
            const auto accepted = known.accepted_snapshot();
            graph.configure_hc_q8_splits(count == 32 ? 16 : 32);
            reject([&] { static_cast<void>(known.position()); });
            reject([&] { static_cast<void>(known.lookup(tokens[0])); });
            reject([&] { static_cast<void>(known.step(tokens[0], rows[0], memory[0])); });
            reject([&] { known.reset(); });
            reject([&] { static_cast<void>(known.restore_checkpoint(checkpoint)); });
            graph.configure_hc_q8_splits(count);
            require(known.position() == 3 && known.accepted_snapshot() == accepted,
                "HC immutable-policy refusal changed accepted state");
            known.reset();
            require(known.position() == 0 && !known.known_residual_reuse(), "HC reset changed fixed residual policy");
            require(graph.memory().live_bytes == fixed.live_bytes && graph.memory().allocations == fixed.allocations,
                "HC stack execution/reset/refusal allocated GPU resources");
        }
        require(graph.memory().live_bytes == prepared.live_bytes, "HC connected stacks leaked resources");
    }
    graph.configure_hc_q8_splits(0);
    require(graph.memory().live_bytes == baseline.live_bytes, "HC connected fixture leaked split scratch");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph graph(argv[1], 256 * 1024 * 1024); const auto base = graph.memory().live_bytes; static_cast<void>(connected(graph));
        const auto ordinary = connected(graph, true);
        for (const uint32_t splits : {1, 4, 8, 16, 32}) {
            graph.configure_small_gemv(splits);
            static_cast<void>(connected(graph, true, splits == 1 ? &ordinary : nullptr));
        }
        graph.configure_small_gemv(0);
        replay_connected(graph, false);
        graph.configure_small_gemv(16); graph.configure_block_gemv(3);
        replay_connected(graph, true);
        graph.configure_small_gemv(0); graph.configure_block_gemv(0);
        checkpoint_connected(graph);
        replay_generic(graph);
        graph.configure_small_gemv(16); graph.configure_block_gemv(3);
        known_connected(graph, 32, 1);
        known_connected(graph, 1024, 2);
        hc_q8_connected(graph);
        graph.configure_small_gemv(0); graph.configure_block_gemv(0);
        require(graph.memory().live_bytes == base, "stack fixture leak");
        std::cout << "Owned complete-stack fixture checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "Owned stack check failed: " << e.what() << '\n'; return 1; }
}
