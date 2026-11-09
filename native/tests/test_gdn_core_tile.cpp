#include "tokspur/gdn_core_compare.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/q8_many.hpp"

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
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid GDN core operation admitted"); }
std::vector<float> values(uint64_t count, float scale = 0.01f) {
    std::vector<float> out(count); for (size_t i = 0; i < out.size(); ++i) out[i] = float(int((i * 17 + 13) % 127) - 63) / 64 * scale; return out;
}
struct Fixture {
    MetalGraph& g; std::deque<std::vector<std::byte>> encoded; std::deque<std::vector<double>> real;
    std::vector<GpuView> owned;
    GpuView floats(std::span<const float> v, bool immutable = true) {
        const auto view = g.upload(std::as_bytes(v), "generated GDN core data", immutable); owned.push_back(view); return view;
    }
    std::pair<GpuView, std::span<const double>> vector(uint64_t n, float scale = 0.01f, float offset = 0) {
        auto v = values(n, scale); for (auto& x : v) x += offset; real.emplace_back(v.begin(), v.end());
        return {floats(v), real.back()};
    }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(uint32_t columns, uint32_t rows, bool q8) {
        if (!q8) {
            const auto v = values(uint64_t(columns) * rows); const auto raw = std::as_bytes(std::span(v)); encoded.emplace_back(raw.begin(), raw.end());
        } else {
            encoded.emplace_back(q8_many_bytes({columns, rows, 1}).matrix);
            for (size_t at = 0; at < encoded.back().size(); at += 34) {
                encoded.back()[at] = std::byte{0}; encoded.back()[at + 1] = std::byte{0x18}; // Exact half 2^-9.
                for (size_t lane = 0; lane < 32; ++lane) encoded.back()[at + 2 + lane] = std::byte(uint8_t(int((at / 34 * 13 + lane * 7) % 31) - 15));
            }
        }
        const auto view = g.upload(encoded.back(), "generated GDN core matrix"); owned.push_back(view);
        return {{view, q8 ? 8u : 0u}, {encoded.back(), q8 ? 8u : 0u}};
    }
    std::pair<GdnWeights, ReferenceGdnWeights> weights(GdnShape s) {
        const auto raw = static_cast<uint32_t>(gdn_qkv_count(s)), v = s.value_heads * s.value_width;
        const auto qkv = matrix(s.channels, raw, true), gate = matrix(s.channels, v, true), output = matrix(v, s.channels, true);
        const auto alpha = matrix(s.channels, s.value_heads, false), beta = matrix(s.channels, s.value_heads, false);
        const auto conv = vector(uint64_t(raw) * 4, 0.1f), rates = vector(s.value_heads, 0, -0.25f), bias = vector(s.value_heads), gains = vector(s.value_width, 0.1f, 1);
        return {{qkv.first, gate.first, alpha.first, beta.first, output.first, conv.first, rates.first, bias.first, gains.first},
            {qkv.second, gate.second, alpha.second, beta.second, output.second, conv.second, rates.second, bias.second, gains.second}};
    }
    ~Fixture() { try { g.abort(); } catch (const FormatError&) {} for (auto view : owned) g.release(view); }
};
GdnCoreTileComparison exercise(MetalGraph& g, GdnShape shape, StateTile tile, bool all_prefixes, bool zero,
              GdnPolicy policy = {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}) {
    Fixture f{g, {}, {}, {}}; const auto weights = f.weights(shape);
    auto x = values(uint64_t(tile.capacity) * shape.channels, 0.05f), state = values(gdn_state_count(shape)), history = values(gdn_qkv_count(shape) * 3);
    if (zero) { std::fill(state.begin(), state.end(), 0); std::fill(history.begin(), history.end(), 0); }
    std::fill(x.begin() + uint64_t(tile.steps) * shape.channels, x.end(), std::numeric_limits<float>::quiet_NaN());
    const auto before = g.memory().live_bytes;
    const auto result = check_gdn_core_tile(g, weights.first, weights.second, shape, policy, tile, x, state, history, all_prefixes);
    checks += result.checks; batches += result.batches; compared += result.compared_values; worst = std::max(worst, result.maximum_scaled_error);
    require(g.memory().live_bytes == before, "connected GDN diagnostic leaked buffers");
    return result;
}
void refusal(MetalGraph& g) {
    const GdnShape shape{32, 1, 2, 128, 16, 4}; constexpr StateTile tile{4, 3};
    constexpr uint64_t expected_dispatches = gdn_core_batch_elementwise ? 23 : 29;
    Fixture f{g, {}, {}, {}}; const auto pair = f.weights(shape); const auto weights = pair.first;
    auto input_data = values(uint64_t(tile.capacity) * shape.channels);
    const auto input = f.floats(input_data, false), output = f.floats(std::vector<float>(input_data.size()), false);
    const auto immutable_output = f.floats(std::vector<float>(input_data.size()));
    const auto scratch = allocate_gdn_core_tile_scratch(g, shape, tile.capacity), scratch2 = allocate_gdn_core_tile_scratch(g, shape, tile.capacity);
    for (auto view : scratch.views()) f.owned.push_back(view); for (auto view : scratch2.views()) f.owned.push_back(view);
    const auto base = allocate_gdn_state(g, shape), candidate = allocate_gdn_state(g, shape);
    for (auto view : base.views()) f.owned.push_back(view); for (auto view : candidate.views()) f.owned.push_back(view);
    const GdnPolicy policy{1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
    const auto accepted_matrix = g.read(base.matrix), accepted_history = g.read(base.convolution), original_output = g.read(output);
    const auto invoke = [&](const GdnWeights& w, const GdnScratch& s, const GdnState& next, GpuView out) {
        encode_gdn_core_tile(g, w, s, base, next, input, out, shape, policy, tile);
    };
    g.begin(); reject([&] { invoke(weights, scratch, base, output); });
    g.begin(); reject([&] { invoke(weights, scratch, candidate, input); });
    g.begin(); reject([&] { invoke(weights, scratch, candidate, immutable_output); });
    auto bad_weights = weights; bad_weights.output.encoding = 0;
    g.begin(); reject([&] { invoke(bad_weights, scratch, candidate, output); });
    bad_weights = weights; bad_weights.gains = g.slice(weights.gains, 0, weights.gains.bytes - 4);
    g.begin(); reject([&] { invoke(bad_weights, scratch, candidate, output); });
    auto bad_scratch = scratch; bad_scratch.gated = scratch.raw;
    g.begin(); reject([&] { invoke(weights, bad_scratch, candidate, output); });
    bad_scratch = scratch; bad_scratch.raw = g.slice(scratch.raw, 0, scratch.raw.bytes - 4);
    g.begin(); reject([&] { invoke(weights, bad_scratch, candidate, output); });
    require(g.read(output) == original_output && g.read(base.matrix) == accepted_matrix && g.read(base.convolution) == accepted_history, "host refusal executed a partial core");
    // A late failure in a second composite must discard earlier encoded work.
    g.begin(); invoke(weights, scratch, candidate, output); reject([&] { invoke(bad_weights, scratch2, candidate, output); });
    require(g.read(output) == original_output, "late core refusal executed previous dispatches");
    g.begin(); invoke(weights, scratch, candidate, output); require(g.submit().dispatches == expected_dispatches, "initial valid core failed"); ++batches;
    const auto correct_output = g.read(output), correct_matrix = g.read(candidate.matrix), correct_history = g.read(candidate.convolution);
    for (uint32_t step : {0, 1, 2}) {
        auto invalid = input_data; invalid[uint64_t(step) * shape.channels] = std::numeric_limits<float>::quiet_NaN(); g.write(input, std::as_bytes(std::span(invalid)));
        g.begin(); invoke(weights, scratch, candidate, output); reject([&] { static_cast<void>(g.submit()); });
        require(g.read(base.matrix) == accepted_matrix && g.read(base.convolution) == accepted_history, "device core failure mutated accepted pair");
        g.write(input, std::as_bytes(std::span(input_data))); g.begin(); invoke(weights, scratch, candidate, output); require(g.submit().dispatches == expected_dispatches, "valid core reuse failed"); ++batches;
        require(g.read(output) == correct_output && g.read(candidate.matrix) == correct_matrix && g.read(candidate.convolution) == correct_history, "core recovery changed valid output/state bits");
    }
    for (auto destination : base.views()) for (uint64_t position : {uint64_t(0), destination.bytes / 8, destination.bytes - 4}) {
        const auto original = g.read(destination); auto invalid = original; const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(invalid.data() + position, &nan, 4); g.write(destination, invalid);
        const auto invalid_matrix = g.read(base.matrix), invalid_history = g.read(base.convolution);
        g.begin(); invoke(weights, scratch, candidate, output); reject([&] { static_cast<void>(g.submit()); });
        require(g.read(base.matrix) == invalid_matrix && g.read(base.convolution) == invalid_history, "nonfinite accepted core state was modified");
        g.write(destination, original); g.begin(); invoke(weights, scratch, candidate, output); require(g.submit().dispatches == expected_dispatches, "core state recovery failed"); ++batches;
        require(g.read(output) == correct_output && g.read(candidate.matrix) == correct_matrix && g.read(candidate.convolution) == correct_history, "accepted-state recovery changed core bits");
    }
    const auto replay = allocate_gdn_core_replay_scratch(g, shape, tile.capacity);
    f.owned.push_back(replay.convolution_output); f.owned.push_back(replay.delta_output);
    g.begin(); reject([&] { encode_gdn_core_prefix(g, weights.convolution, scratch, output, base, base, replay, shape, tile); });
    auto invalid_replay = replay; invalid_replay.delta_output = scratch.raw;
    g.begin(); reject([&] { encode_gdn_core_prefix(g, weights.convolution, scratch, output, base, candidate, invalid_replay, shape, tile); });
    invalid_replay = replay; invalid_replay.delta_output = output;
    g.begin(); reject([&] { encode_gdn_core_prefix(g, weights.convolution, scratch, output, base, candidate, invalid_replay, shape, tile); });
    const auto query = g.read(scratch.query); auto bad_query = query; const float nan = std::numeric_limits<float>::quiet_NaN(); std::memcpy(bad_query.data(), &nan, 4);
    g.write(scratch.query, bad_query); g.begin(); encode_gdn_core_prefix(g, weights.convolution, scratch, output, base, candidate, replay, shape, tile);
    reject([&] { static_cast<void>(g.submit()); }); require(g.read(base.matrix) == accepted_matrix && g.read(base.convolution) == accepted_history && g.read(scratch.query) == bad_query, "failed prefix replay modified base/tape");
    g.write(scratch.query, query); g.begin(); encode_gdn_core_prefix(g, weights.convolution, scratch, output, base, candidate, replay, shape, tile);
    require(g.submit().dispatches == 2 && g.read(candidate.matrix) == correct_matrix && g.read(candidate.convolution) == correct_history, "prefix replay recovery differs"); ++batches;
    const GdnCoreTape compact{scratch.projected, scratch.query, scratch.key, scratch.value, scratch.decay, scratch.write_gate};
    const auto rebound = bind_gdn_core_tape(scratch2, compact);
    require(rebound.projected.slot == compact.projected.slot && rebound.query.slot == compact.query.slot &&
        rebound.convolved.slot == scratch2.convolved.slot && rebound.raw.slot == scratch2.raw.slot,
        "compact tape binding changed forward-only scratch");
    const auto resident = g.memory();
    for (uint32_t k = 1; k <= tile.steps; ++k) {
        g.begin(); encode_gdn_core_prefix(g, weights.convolution, scratch, output, base, candidate, replay, shape, {tile.capacity, k});
        require(g.submit().dispatches == 2, "full prefix comparison failed"); ++batches;
        const auto matrix = g.read(candidate.matrix), history = g.read(candidate.convolution);
        g.begin(); encode_gdn_core_tape_prefix(g, weights.convolution, compact, base, candidate, replay, shape, {tile.capacity, k});
        require(g.submit().dispatches == 2 && g.read(candidate.matrix) == matrix && g.read(candidate.convolution) == history &&
            g.read(base.matrix) == accepted_matrix && g.read(base.convolution) == accepted_history && g.read(output) == correct_output,
            "compact six-array recovery differs from protected full tape"); ++batches;
    }
    auto bad_compact = compact; bad_compact.query.owner += 1;
    g.begin(); reject([&] { encode_gdn_core_tape_prefix(g, weights.convolution, bad_compact, base, candidate, replay, shape, tile); });
    bad_compact = compact; bad_compact.value = g.slice(compact.value, 0, compact.value.bytes - 4);
    g.begin(); reject([&] { encode_gdn_core_tape_prefix(g, weights.convolution, bad_compact, base, candidate, replay, shape, tile); });
    g.begin(); reject([&] { encode_gdn_core_tape_prefix(g, weights.convolution, compact, base, base, replay, shape, tile); });
    auto aliased_replay = replay; aliased_replay.convolution_output = compact.projected;
    g.begin(); reject([&] { encode_gdn_core_tape_prefix(g, weights.convolution, compact, base, candidate, aliased_replay, shape, tile); });
    require(g.memory().allocations == resident.allocations && g.memory().live_bytes == resident.live_bytes &&
        g.read(candidate.matrix) == correct_matrix && g.read(candidate.convolution) == correct_history,
        "compact refusal changed candidate or resource ledger");
    g.configure_small_gemv(16);
    for (uint32_t mask : {0, 2}) {
        g.configure_block_gemv(mask); g.begin(); reject([&] { invoke(weights, scratch, candidate, output); });
        require(g.read(output) == correct_output && g.read(candidate.matrix) == correct_matrix &&
            g.read(candidate.convolution) == correct_history && g.read(base.matrix) == accepted_matrix &&
            g.read(base.convolution) == accepted_history, "core policy refusal changed retained outputs/state");
        g.configure_block_gemv(3); g.begin(); invoke(weights, scratch, candidate, output);
        require(g.submit().dispatches == expected_dispatches && g.read(output) == correct_output && g.read(candidate.matrix) == correct_matrix &&
            g.read(candidate.convolution) == correct_history, "core policy refusal recovery changed bits"); ++batches;
    }
    for (uint32_t splits : {2, 3, 64}) reject([&] { g.configure_small_gemv(splits); });
    require(g.kernel_policy() == std::array<uint32_t, 2>{16, 3}, "core invalid split admission changed policy");
    g.configure_small_gemv(0); g.configure_block_gemv(1);
    const auto ledger = g.memory();
    reject([&] { static_cast<void>(g.zeros(256 * 1024 * 1024, "must exceed core budget")); });
    require(g.memory().live_bytes == ledger.live_bytes, "core budget refusal leaked allocation");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph g(argv[1], 256 * 1024 * 1024); const auto initial = g.memory().live_bytes;
        g.configure_block_gemv(1); for (uint32_t width : {1, 2, 4, 8}) g.prepare_q8_matvec_many(width);
        for (uint32_t group : {32, 64, 128}) {
            try { g.prepare_gdn_tiles(group); } catch (const FormatError& e) { std::cout << "GDN core group refused: " << group << " (" << e.what() << ")\n"; continue; }
            std::cout << "GDN core supported group: " << group << '\n';
            for (uint32_t capacity : {2, 4, 8, 16, 32}) for (uint32_t steps : {1u, capacity})
                exercise(g, {32, 1, 2, 128, 16, 4}, {capacity, steps}, capacity <= 8, steps == 1);
        }
        g.prepare_gdn_tiles(32);
        for (auto mode : {L2Mode::add_squared_epsilon, L2Mode::floor_squared_epsilon}) for (float scale : {1.0f, 0.088388346f})
            exercise(g, {32, 2, 4, 128, 16, 4}, {4, 3}, true, false, {scale, 1e-6f, mode, 1e-6f});
        for (uint32_t steps : {3, 7, 9, 15, 17, 31}) exercise(g, {32, 2, 4, 128, 16, 4}, {32, steps}, false, false);
        exercise(g, {32, 1, 32, 128, 127, 4}, {4, 3}, true, false);
        // Actual head geometry with small generated projections, partial rows,
        // all intermediate bytes, inactive tails and every rollback prefix.
        exercise(g, {32, 16, 48, 128, 128, 4}, {8, 7}, true, false);
        const GdnShape wide{1024, 1, 2, 128, 16, 4};
        g.configure_block_gemv(3); g.prepare_gdn_tiles(128);
        const auto split_case = [&](StateTile tile, bool every_prefix) {
            const auto result = exercise(g, wide, tile, every_prefix, false);
            const uint64_t extra = g.kernel_policy()[0] > 1 ? 2 : 0;
            uint32_t chunks = tile.steps / 8; for (uint32_t n = tile.steps % 8; n; n >>= 1) chunks += n & 1;
            require(result.serial_dispatches == uint64_t(tile.steps) * (12 + extra) &&
                result.tile_dispatches == uint64_t(tile.steps) * (7 + extra) + 3 * chunks + 2 -
                    (gdn_core_batch_elementwise ? uint64_t(tile.steps - 1) * 3 : 0),
                "wide core precomputed split dispatch formula differs");
        };
        for (uint32_t splits : {0, 1, 4, 8, 16, 32}) { g.configure_small_gemv(splits); split_case({4, 3}, true); }
        g.configure_small_gemv(16);
        for (uint32_t capacity : {2, 4, 8, 16, 32}) for (uint32_t steps : {1u, capacity})
            split_case({capacity, steps}, capacity <= 8);
        for (uint32_t steps : {3, 5, 7}) split_case({8, steps}, true);
        for (uint32_t steps : {9, 17, 31}) split_case({32, steps}, false);
        g.configure_small_gemv(0); g.configure_block_gemv(1); refusal(g);
        MetalGraph limited(argv[1], 65536); const auto limited_base = limited.memory().live_bytes;
        reject([&] { static_cast<void>(allocate_gdn_core_tile_scratch(limited, {32, 1, 2, 128, 16, 4}, 32)); });
        require(limited.memory().live_bytes == limited_base, "partial core scratch allocation leaked");
        require(g.memory().live_bytes == initial, "GDN core fixtures leaked Metal buffers");
        const auto m = g.memory();
        std::cout << "Owned connected GDN core tile checks passed: " << checks << ", batches: " << batches << ", compared values: " << compared
            << ", max scaled error: " << worst << ", GPU buffer peak bytes: " << m.peak_bytes << ", GPU buffer budget bytes: " << m.budget << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned connected GDN core tile failed: " << e.what() << '\n'; return 1; }
}
