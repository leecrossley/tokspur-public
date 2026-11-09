#include "tokspur/recurrent_tile.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>

namespace {
using namespace tokspur;
uint64_t checks = 0, batches = 0, compared = 0; double worst = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid recurrent tile admitted"); }
std::vector<float> values(uint64_t count, float scale = 0.01f) {
    std::vector<float> result(count); for (size_t i = 0; i < result.size(); ++i) result[i] = float(int((i * 17 + 13) % 127) - 63) / 64 * scale; return result;
}
struct Fixture {
    MetalGraph& g; std::deque<std::vector<std::byte>> data; std::deque<std::vector<double>> real; std::vector<GpuView> owned;
    GpuView floats(std::span<const float> v, bool immutable = true) {
        const auto view = g.upload(std::as_bytes(v), "generated recurrent tile data", immutable); owned.push_back(view); return view;
    }
    std::pair<GpuView, std::span<const double>> vector(uint64_t count, float scale = 0.01f, float offset = 0) {
        auto v = values(count, scale); for (auto& value : v) value += offset; real.emplace_back(v.begin(), v.end()); return {floats(v), real.back()};
    }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(uint32_t columns, uint32_t rows, uint32_t type, uint32_t experts = 1) {
        const auto& format = encoding(type); require(columns % format.elements_per_block == 0, "generated matrix block alignment");
        data.emplace_back(uint64_t(columns / format.elements_per_block) * rows * experts * format.bytes_per_block);
        if (!type) {
            const auto v = values(uint64_t(columns) * rows * experts); const auto raw = std::as_bytes(std::span(v)); std::copy(raw.begin(), raw.end(), data.back().begin());
        } else for (size_t at = 0; at < data.back().size(); at += format.bytes_per_block) {
            auto block = std::span(data.back()).subspan(at, format.bytes_per_block);
            block[0] = std::byte{0}; block[1] = std::byte(type == 12 || type == 13 ? 0x14 : 0x18);
            if (type == 8) {
                for (size_t i = 0; i < 32; ++i) block[2 + i] = std::byte(uint8_t(int((at / format.bytes_per_block * 13 + i * 7) % 31) - 15));
            } else if (type == 7) {
                for (size_t i = 4; i < 24; ++i) block[i] = std::byte(uint8_t((i * 17 + at / 24) % 256));
            } else if (type == 12 || type == 13) {
                for (size_t i = 4; i < 12; ++i) block[i] = std::byte{1};
                for (size_t i = 12; i < 16; ++i) block[i] = std::byte{0x11};
                for (size_t i = 16; i < block.size(); ++i) block[i] = std::byte(uint8_t((i * 17 + at / format.bytes_per_block) % 256));
            } else throw std::runtime_error("unsupported generated recurrent encoding");
        }
        const auto v = g.upload(data.back(), "generated recurrent tile matrix"); owned.push_back(v); return {{v, type}, {data.back(), type}};
    }
    std::pair<ResidualWeights, ReferenceResidualWeights> residual(ResidualShape s, uint32_t type) {
        const auto width = s.branches * s.channels;
        const auto gains = vector(width, 0.1f, 1);
        const auto down = matrix(width, s.rank, type), up = matrix(s.rank, width, type), inject = matrix(width, s.branches, 0);
        return {{gains.first, down.first, up.first, inject.first}, {gains.second, down.second, up.second, inject.second}};
    }
    std::pair<GdnWeights, ReferenceGdnWeights> core(GdnShape s) {
        const auto raw = static_cast<uint32_t>(gdn_qkv_count(s)), v = s.value_heads * s.value_width;
        const auto qkv = matrix(s.channels, raw, 8), gate = matrix(s.channels, v, 8), alpha = matrix(s.channels, s.value_heads, 0), beta = matrix(s.channels, s.value_heads, 0), output = matrix(v, s.channels, 8);
        const auto conv = vector(uint64_t(raw) * 4, 0.1f), rates = vector(s.value_heads, 0, -0.25f), bias = vector(s.value_heads), gains = vector(s.value_width, 0.1f, 1);
        return {{qkv.first, gate.first, alpha.first, beta.first, output.first, conv.first, rates.first, bias.first, gains.first},
            {qkv.second, gate.second, alpha.second, beta.second, output.second, conv.second, rates.second, bias.second, gains.second}};
    }
    std::pair<MoeWeights, ReferenceMoeWeights> moe(MoeShape s, uint32_t gate_type, uint32_t down_type) {
        const auto router = matrix(s.channels, s.experts, 0), gate = matrix(s.channels, s.hidden, gate_type, s.experts), up = matrix(s.channels, s.hidden, gate_type, s.experts), down = matrix(s.hidden, s.channels, down_type, s.experts);
        const auto sg = matrix(s.channels, s.hidden, 8), su = matrix(s.channels, s.hidden, 8), sd = matrix(s.hidden, s.channels, 8), sr = matrix(s.channels, 1, 0);
        return {{router.first, gate.first, up.first, down.first, sg.first, su.first, sd.first, sr.first},
            {router.second, gate.second, up.second, down.second, sg.second, su.second, sd.second, sr.second}};
    }
    std::pair<RecurrentWeights, ReferenceLayerWeights> weights(RecurrentShape s, uint32_t residual_type = 8, uint32_t gate_type = 0, uint32_t down_type = 7) {
        const auto r1 = residual(s.residual, residual_type), r2 = residual(s.residual, residual_type);
        const auto a = core(s.gdn); const auto m = moe(s.moe, gate_type, down_type);
        return {{r1.first, r2.first, a.first, m.first}, {r1.second, r2.second, a.second, m.second}};
    }
    void keep(const RecurrentScratch& scratch) { for (auto view : scratch.views()) owned.push_back(view); }
    void keep(const GdnState& state) { for (auto view : state.views()) owned.push_back(view); }
    ~Fixture() { try { g.abort(); } catch (const FormatError&) {} for (auto view : owned) g.release(view); }
};
constexpr RecurrentShape small{{2, 32, 32}, {32, 1, 2, 128, 16, 4}, {32, 32, 5, 2}};
RecurrentTileComparison exercise(MetalGraph& g, RecurrentShape shape, StateTile tile, bool every_prefix, bool zero,
    GdnPolicy policy = {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode mode = RouterMode::softmax, bool normalize = true,
    uint32_t residual_type = 8, uint32_t gate_type = 0, uint32_t down_type = 7, bool reuse_residual = false) {
    Fixture f{g, {}, {}, {}}; const auto w = f.weights(shape, residual_type, gate_type, down_type);
    const uint64_t width = uint64_t(shape.residual.branches) * shape.residual.channels;
    auto inputs = values(width * tile.capacity, 0.05f), state = values(gdn_state_count(shape.gdn)), history = values(gdn_qkv_count(shape.gdn) * 3);
    if (zero) { std::fill(state.begin(), state.end(), 0); std::fill(history.begin(), history.end(), 0); }
    std::fill(inputs.begin() + width * tile.steps, inputs.end(), std::numeric_limits<float>::quiet_NaN());
    const auto before = g.memory().live_bytes;
    const auto result = check_recurrent_tile(g, w.first, w.second, shape, policy, mode, normalize, tile, inputs, state, history, every_prefix, reuse_residual);
    checks += result.checks; batches += result.batches; compared += result.compared_values; worst = std::max(worst, result.maximum_scaled_error);
    require(g.memory().live_bytes == before, "recurrent tile comparator leaked buffers");
    return result;
}
void refusal(MetalGraph& g, const char* library, bool reuse_residual = false) {
    const uint64_t expected_dispatches = (reuse_residual ? 109 : 113) - (gdn_core_batch_elementwise ? 6 : 0);
    Fixture f{g, {}, {}, {}}; const auto w = f.weights(small).first; constexpr StateTile tile{4, 3};
    const GdnPolicy policy{1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
    auto x = values(uint64_t(tile.capacity) * small.residual.branches * small.residual.channels);
    const auto input = f.floats(x, false), output = f.floats(std::vector<float>(x.size()), false), immutable = f.floats(std::vector<float>(x.size()));
    const auto s = allocate_recurrent_tile_scratch(g, small, tile.capacity); f.keep(s);
    const auto base = allocate_gdn_state(g, small.gdn), candidate = allocate_gdn_state(g, small.gdn); f.keep(base); f.keep(candidate);
    const auto accepted_matrix = g.read(base.matrix), accepted_history = g.read(base.convolution), untouched = g.read(output);
    const auto encode = [&](const RecurrentWeights& weights, const RecurrentScratch& scratch, const GdnState& next, GpuView out) {
        encode_recurrent_layer_tile(g, weights, scratch, base, next, input, out, small, policy, RouterMode::softmax, true, tile, reuse_residual);
    };
    g.begin(); reject([&] { encode(w, s, base, output); });
    g.begin(); reject([&] { encode(w, s, candidate, input); });
    g.begin(); reject([&] { encode(w, s, candidate, immutable); });
    auto bad = w; bad.ffn.shared_router.encoding = 0xffffffffu;
    g.begin(); reject([&] { encode(bad, s, candidate, output); });
    bad = w; bad.ffn.shared_router.view = g.slice(w.ffn.shared_router.view, 0, w.ffn.shared_router.view.bytes - 4);
    g.begin(); reject([&] { encode(bad, s, candidate, output); });
    bad = w; bad.ffn.shared_router.view = g.slice(output, 0, uint64_t(small.moe.channels) * 4);
    g.begin(); reject([&] { encode(bad, s, candidate, output); });
    auto forged = s; forged.moe.weights = s.moe.indices;
    g.begin(); reject([&] { encode(w, forged, candidate, output); });
    forged = s; forged.intermediate_streams = output;
    g.begin(); reject([&] { encode(w, forged, candidate, output); });
    MetalGraph foreign(library, 4 * 1024 * 1024);
    bad = w; bad.ffn.shared_router.view = foreign.zeros(w.ffn.shared_router.view.bytes, "foreign late weight");
    g.begin(); reject([&] { encode(bad, s, candidate, output); });
    auto released = g.zeros(w.ffn.shared_router.view.bytes, "released late weight"); g.release(released);
    bad = w; bad.ffn.shared_router.view = released;
    g.begin(); reject([&] { encode(bad, s, candidate, output); });
    forged = s; forged.ffn_output = foreign.zeros(s.ffn_output.bytes, "foreign late scratch");
    g.begin(); reject([&] { encode(w, forged, candidate, output); });
    released = g.zeros(s.ffn_output.bytes, "released late scratch"); g.release(released);
    forged = s; forged.ffn_output = released;
    g.begin(); reject([&] { encode(w, forged, candidate, output); });
    auto invalid_policy = policy; invalid_policy.l2_epsilon = std::numeric_limits<float>::quiet_NaN();
    g.begin(); reject([&] { encode_recurrent_layer_tile(g, w, s, base, candidate, input, output, small, invalid_policy, RouterMode::softmax, true, tile); });
    auto invalid_shape = small; invalid_shape.gdn.key_width = 64;
    g.begin(); reject([&] { encode_recurrent_layer_tile(g, w, s, base, candidate, input, output, invalid_shape, policy, RouterMode::softmax, true, tile); });
    require(g.read(output) == untouched, "recurrent host refusal executed work");
    g.begin(); encode(w, s, candidate, output); reject([&] { encode(bad, s, candidate, output); });
    require(g.read(output) == untouched, "late recurrent encode refusal failed to abort whole batch");
    g.begin(); encode(w, s, candidate, output); g.abort(); require(g.read(output) == untouched, "aborted recurrent tile executed writes");
    const auto ledger = g.memory(); const auto valid = [&] {
        g.begin(); encode(w, s, candidate, output); const auto b = g.submit(); ++batches;
        require(b.dispatches == expected_dispatches && g.memory().allocations == ledger.allocations && g.memory().live_bytes == ledger.live_bytes, "recurrent valid dispatch/allocation invariant");
    };
    valid(); const auto expected = g.read(output), expected_matrix = g.read(candidate.matrix), expected_history = g.read(candidate.convolution);
    for (uint32_t step : {0, 1, 2}) {
        auto invalid = x; invalid[uint64_t(step) * 64] = std::numeric_limits<float>::quiet_NaN(); g.write(input, std::as_bytes(std::span(invalid)));
        g.begin(); encode(w, s, candidate, output); reject([&] { static_cast<void>(g.submit()); });
        require(g.read(base.matrix) == accepted_matrix && g.read(base.convolution) == accepted_history, "failed recurrent tile published accepted state");
        g.write(input, std::as_bytes(std::span(x))); valid(); require(g.read(output) == expected && g.read(candidate.matrix) == expected_matrix && g.read(candidate.convolution) == expected_history, "recurrent input recovery changed bits");
    }
    for (auto destination : base.views()) for (uint64_t at : {uint64_t(0), destination.bytes / 8, destination.bytes - 4}) {
        const auto old = g.read(destination); auto invalid = old; const float nan = std::numeric_limits<float>::quiet_NaN(); std::memcpy(invalid.data() + at, &nan, 4); g.write(destination, invalid);
        const auto before_matrix = g.read(base.matrix), before_history = g.read(base.convolution);
        g.begin(); encode(w, s, candidate, output); reject([&] { static_cast<void>(g.submit()); });
        require(g.read(base.matrix) == before_matrix && g.read(base.convolution) == before_history, "nonfinite accepted recurrent state modified");
        g.write(destination, old); valid(); require(g.read(output) == expected, "recurrent accepted-state recovery failed");
    }
    // Fail after the GDN candidate is complete, in the unchanged FFN router.
    auto invalid_router = values(uint64_t(small.moe.channels) * small.moe.experts); invalid_router.back() = std::numeric_limits<float>::quiet_NaN();
    bad = w; bad.ffn.router.view = f.floats(invalid_router);
    const auto after_weight = g.memory(); g.begin(); encode(bad, s, candidate, output); reject([&] { static_cast<void>(g.submit()); });
    require(g.read(base.matrix) == accepted_matrix && g.read(base.convolution) == accepted_history, "late FFN device failure modified accepted pair");
    g.begin(); encode(w, s, candidate, output); const auto recovery = g.submit(); ++batches;
    require(recovery.dispatches == expected_dispatches && g.memory().allocations == after_weight.allocations && g.memory().live_bytes == after_weight.live_bytes &&
        g.read(output) == expected && g.read(candidate.matrix) == expected_matrix && g.read(candidate.convolution) == expected_history, "late FFN device recovery failed");
    const auto replay = allocate_gdn_core_replay_scratch(g, small.gdn, tile.capacity); f.owned.push_back(replay.convolution_output); f.owned.push_back(replay.delta_output);
    g.begin(); reject([&] { encode_recurrent_tile_prefix(g, w, s, base, base, output, replay, small, tile); });
    auto bad_replay = replay; bad_replay.delta_output = s.gdn.raw;
    g.begin(); reject([&] { encode_recurrent_tile_prefix(g, w, s, base, candidate, output, bad_replay, small, tile); });
    bad_replay = replay; bad_replay.delta_output = g.slice(output, 0, replay.delta_output.bytes);
    g.begin(); reject([&] { encode_recurrent_tile_prefix(g, w, s, base, candidate, output, bad_replay, small, tile); });
    const auto query = g.read(s.gdn.query); auto invalid_query = query; const float nan = std::numeric_limits<float>::quiet_NaN(); std::memcpy(invalid_query.data(), &nan, 4); g.write(s.gdn.query, invalid_query);
    g.begin(); encode_recurrent_tile_prefix(g, w, s, base, candidate, output, replay, small, tile); reject([&] { static_cast<void>(g.submit()); });
    require(g.read(s.gdn.query) == invalid_query && g.read(output) == expected && g.read(base.matrix) == accepted_matrix &&
        g.read(base.convolution) == accepted_history, "failed recurrent replay mutated protected views");
    g.write(s.gdn.query, query); g.begin(); encode_recurrent_tile_prefix(g, w, s, base, candidate, output, replay, small, tile);
    require(g.submit().dispatches == 2 && g.read(candidate.matrix) == expected_matrix && g.read(candidate.convolution) == expected_history, "recurrent prefix recovery failed"); ++batches;
    g.configure_small_gemv(16);
    for (uint32_t mask : {0, 2}) {
        g.configure_block_gemv(mask); g.begin(); reject([&] { encode(w, s, candidate, output); });
        require(g.read(output) == expected && g.read(candidate.matrix) == expected_matrix &&
            g.read(candidate.convolution) == expected_history && g.read(base.matrix) == accepted_matrix &&
            g.read(base.convolution) == accepted_history, "recurrent policy refusal changed retained outputs/state");
        g.configure_block_gemv(3); g.begin(); encode(w, s, candidate, output);
        require(g.submit().dispatches == expected_dispatches && g.read(output) == expected && g.read(candidate.matrix) == expected_matrix &&
            g.read(candidate.convolution) == expected_history, "recurrent policy refusal recovery changed bits"); ++batches;
    }
    for (uint32_t splits : {2, 3, 64}) reject([&] { g.configure_small_gemv(splits); });
    require(g.kernel_policy() == std::array<uint32_t, 2>{16, 3}, "recurrent invalid split admission changed policy");
    g.configure_small_gemv(0); g.configure_block_gemv(1);
    g.begin(); reject([&] { encode_recurrent_layer_tile(g, w, s, base, candidate, input, output, small, policy, RouterMode(2), true, tile); });
    g.begin(); reject([&] { encode_recurrent_layer_tile(g, w, s, base, candidate, input, output, small, policy, RouterMode::softmax, true, {4, 0}); });
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph g(argv[1], 1024 * 1024 * 1024); const auto initial = g.memory().live_bytes;
        g.configure_block_gemv(1); for (uint32_t width : {1, 2, 4, 8}) g.prepare_q8_matvec_many(width);
        for (uint32_t group : {32, 64, 128}) {
            try { g.prepare_gdn_tiles(group); } catch (const FormatError& e) { std::cout << "recurrent tile group refused: " << group << " (" << e.what() << ")\n"; continue; }
            std::cout << "recurrent tile supported group: " << group << '\n';
            for (uint32_t capacity : {2, 4, 8, 16, 32}) for (uint32_t steps : {1u, capacity}) exercise(g, small, {capacity, steps}, capacity <= 8, steps == 1);
        }
        g.prepare_gdn_tiles(32);
        for (auto mode : {RouterMode::softmax, RouterMode::sigmoid}) for (bool normalize : {false, true})
        for (auto l2 : {L2Mode::add_squared_epsilon, L2Mode::floor_squared_epsilon}) for (float scale : {1.0f, 0.088388346f})
            exercise(g, small, {4, 3}, true, false, {scale, 1e-6f, l2, 1e-6f}, mode, normalize);
        for (uint32_t steps : {7, 9, 17, 31}) exercise(g, small, {32, steps}, false, false);
        const RecurrentShape quantized{{2, 256, 256}, {256, 1, 2, 128, 16, 4}, {256, 256, 5, 2}};
        g.configure_block_gemv(3);
        exercise(g, quantized, {4, 3}, true, false, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true, 12, 12, 7);
        exercise(g, quantized, {4, 3}, true, false, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true, 12, 13, 8);
        const RecurrentShape wide{{2, 1024, 32}, {1024, 1, 2, 128, 16, 4}, {1024, 32, 5, 2}};
        g.prepare_gdn_tiles(128);
        const auto split_case = [&](StateTile tile, bool every_prefix, uint32_t residual_type) {
            const auto result = exercise(g, wide, tile, every_prefix, false, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
                RouterMode::softmax, true, residual_type);
            // Two injects, alpha/beta, router and shared-router split; F32 residual
            // down projections add two more. All other generated GEMVs remain single.
            const uint64_t extra = g.kernel_policy()[0] > 1 ? (residual_type == 0 ? 8 : 6) : 0;
            uint32_t chunks = tile.steps / 8; for (uint32_t n = tile.steps % 8; n; n >>= 1) chunks += n & 1;
            require(result.serial_dispatches == uint64_t(tile.steps) * (40 + extra) &&
                result.tile_dispatches == uint64_t(tile.steps) * (35 + extra) + 3 * chunks + 2 -
                    (gdn_core_batch_elementwise ? uint64_t(tile.steps - 1) * 3 : 0),
                "wide recurrent precomputed split dispatch formula differs");
        };
        for (uint32_t splits : {0, 1, 4, 8, 16, 32}) { g.configure_small_gemv(splits); split_case({4, 3}, true, 0); }
        g.configure_small_gemv(16);
        for (uint32_t capacity : {2, 4, 8, 16, 32}) for (uint32_t steps : {1u, capacity})
            split_case({capacity, steps}, capacity <= 8, 0);
        for (uint32_t steps : {3, 5, 7}) split_case({8, steps}, true, 0);
        for (uint32_t steps : {9, 17, 31}) split_case({32, steps}, false, 8);
        const RecurrentShape wide_quantized{{2, 1024, 256}, {1024, 1, 2, 128, 16, 4}, {1024, 256, 5, 2}};
        const auto block_split = exercise(g, wide_quantized, {4, 3}, true, false,
            {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true, 12, 12, 7);
        require(block_split.serial_dispatches == 46 * 3 && block_split.tile_dispatches == 41 * 3 + 3 * 2 + 2 -
            (gdn_core_batch_elementwise ? 6 : 0),
            "split16/Q4_K recurrent precomputed dispatch formula differs");
        for (const uint32_t residual_type : {0, 8, 12}) {
            const auto shape = residual_type == 12 ? wide_quantized : wide;
            for (const uint32_t steps : {1, 2, 3, 7, 8, 9, 17, 31, 32}) {
                const StateTile tile{steps <= 8 ? 8u : 32u, steps};
                const auto ordinary = exercise(g, shape, tile, false, false,
                    {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true, residual_type);
                const auto reused = exercise(g, shape, tile, false, false,
                    {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, RouterMode::softmax, true, residual_type, 0, 7, true);
                require(ordinary.output_sha256 == reused.output_sha256 && ordinary.state_sha256 == reused.state_sha256 &&
                    ordinary.history_sha256 == reused.history_sha256 && ordinary.serial_dispatches == reused.serial_dispatches,
                    "residual reuse changed unchanged-policy outputs or accepted-state bits");
                uint32_t chunks = steps / 8; for (auto n = steps % 8; n; n >>= 1) chunks += n & 1;
                const uint64_t saved = residual_type == 8 ? 4 * uint64_t(steps - chunks) : 0;
                require(reused.tile_dispatches + saved == ordinary.tile_dispatches,
                    "residual reuse independent dispatch reduction differs");
            }
        }
        for (uint32_t mask = 0; mask < 16; ++mask) {
            Fixture f{g, {}, {}, {}}; auto w = f.weights(small);
            std::array<GpuMatrix*, 4> gpu{&w.first.attention_read.down, &w.first.attention_read.up,
                &w.first.ffn_read.down, &w.first.ffn_read.up};
            std::array<ReferenceMatrix*, 4> cpu{&w.second.attention_read.down, &w.second.attention_read.up,
                &w.second.ffn_read.down, &w.second.ffn_read.up};
            for (uint32_t i = 0; i < 4; ++i) if (!(mask & (1u << i))) {
                const auto m = f.matrix(i % 2 ? small.residual.rank : small.residual.branches * small.residual.channels,
                    i % 2 ? small.residual.branches * small.residual.channels : small.residual.rank, 0);
                *gpu[i] = m.first; *cpu[i] = m.second;
            }
            const auto inputs = values(uint64_t(8) * small.residual.branches * small.residual.channels, 0.05f);
            const auto state = values(gdn_state_count(small.gdn)), history = values(gdn_qkv_count(small.gdn) * 3);
            const GdnPolicy policy{1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
            for (const uint32_t n : {3, 8}) {
                const auto before = g.memory().live_bytes;
                const auto ordinary = check_recurrent_tile(g, w.first, w.second, small, policy, RouterMode::softmax, true,
                    {8, n}, inputs, state, history, false);
                const auto reused = check_recurrent_tile(g, w.first, w.second, small, policy, RouterMode::softmax, true,
                    {8, n}, inputs, state, history, false, true);
                for (const auto& r : {ordinary, reused}) { checks += r.checks; batches += r.batches; compared += r.compared_values; worst = std::max(worst, r.maximum_scaled_error); }
                uint32_t q8 = 0; for (auto bits = mask; bits; bits >>= 1) q8 += bits & 1;
                require(ordinary.output_sha256 == reused.output_sha256 && ordinary.state_sha256 == reused.state_sha256 &&
                    ordinary.history_sha256 == reused.history_sha256 && reused.tile_dispatches + q8 * (n - (n == 3 ? 2 : 1)) == ordinary.tile_dispatches &&
                    g.memory().live_bytes == before, "mixed residual encoding reuse bits/dispatch/ledger differ");
            }
        }
        g.configure_small_gemv(0);
        g.configure_block_gemv(1); refusal(g, argv[1]);
        g.configure_block_gemv(1); refusal(g, argv[1], true);
        MetalGraph limited(argv[1], 65536); const auto limited_base = limited.memory().live_bytes;
        reject([&] { static_cast<void>(allocate_recurrent_tile_scratch(limited, small, 32)); }); require(limited.memory().live_bytes == limited_base, "partial recurrent allocation leaked");
        require(g.memory().live_bytes == initial, "recurrent tile fixtures leaked buffers"); const auto m = g.memory();
        std::cout << "Owned full recurrent tile checks passed: " << checks << ", batches: " << batches << ", compared values: " << compared << ", max scaled error: " << worst
            << ", GPU buffer peak bytes: " << m.peak_bytes << ", GPU buffer budget bytes: " << m.budget << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned full recurrent tile failed: " << e.what() << '\n'; return 1; }
}
