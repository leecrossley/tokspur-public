#include "tokspur/gdn_core_compare.hpp"
#include "tokspur/gguf.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
constexpr float sentinel = -12345.25f;
constexpr std::array<const char*, 13> stages{"projected", "convolved_activated", "query", "key", "value", "gate",
    "alpha", "beta_logits", "decay", "write_gate", "raw", "normalized", "gated"};
std::string checksum(std::span<const std::byte> bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    constexpr char digits[] = "0123456789abcdef"; std::string result;
    for (auto x : digest) { result += digits[x >> 4]; result += digits[x & 15]; }
    return result;
}
struct Arena {
    MetalGraph& g; std::vector<GpuView> owned;
    GpuView values(std::span<const float> values, bool immutable = true) {
        const auto v = g.upload(std::as_bytes(values), "GDN connected diagnostic", immutable); owned.push_back(v); return v;
    }
    GpuView output(uint64_t count) { return values(std::vector<float>(count, sentinel), false); }
    GdnState state(GdnShape s) {
        return {output(gdn_state_count(s)), output(gdn_qkv_count(s) * 3)};
    }
    GdnScratch tile(GdnShape s, uint32_t capacity) {
        const auto v = allocate_gdn_core_tile_scratch(g, s, capacity);
        for (const auto view : v.views()) owned.push_back(view);
        for (const auto view : v.views()) {
            const std::vector<float> cleared(view.bytes / 4, sentinel); g.write(view, std::as_bytes(std::span(cleared)));
        }
        return v;
    }
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (auto v : owned) g.release(v); }
};
GpuView row(MetalGraph& g, GpuView v, uint64_t n, uint32_t step) { return g.slice(v, uint64_t(step) * n * 4, n * 4); }
GdnScratch token(MetalGraph& g, const GdnScratch& s, uint32_t capacity, uint32_t step) {
    std::array<GpuView, 13> out{}; const auto views = s.views();
    for (size_t i = 0; i < out.size(); ++i) out[i] = g.slice(views[i], views[i].bytes / capacity * step, views[i].bytes / capacity);
    return {out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12]};
}
std::vector<float> floats(MetalGraph& g, GpuView v) {
    const auto bytes = g.read(v); std::vector<float> result(bytes.size() / 4); std::memcpy(result.data(), bytes.data(), bytes.size()); return result;
}
}
GdnCoreTileComparison check_gdn_core_tile(MetalGraph& g, const GdnWeights& w, const ReferenceGdnWeights& reference,
    GdnShape shape, GdnPolicy policy, StateTile tile, std::span<const float> inputs,
    std::span<const float> matrix, std::span<const float> history, bool every_prefix) {
    require(inputs.size() == uint64_t(tile.capacity) * shape.channels && matrix.size() == gdn_state_count(shape) &&
        history.size() == gdn_qkv_count(shape) * 3, "GDN diagnostic input geometry mismatch");
    // Derive the expected work from the frozen policy and matrix geometry before encoding.
    // Only these two F32 projections can split; Q8-many retains one dispatch per chunk.
    const uint64_t extra = uint64_t(g.gemv_dispatches(w.alpha.encoding, shape.channels, shape.value_heads) - 1) +
        g.gemv_dispatches(w.beta.encoding, shape.channels, shape.value_heads) - 1;
    const uint64_t serial_per_token = 12 + extra, tile_per_token = 7 + extra;
    const auto initial = g.memory(); Arena a{g, {}}; GdnCoreTileComparison result;
    const auto input = a.values(inputs), output = a.output(inputs.size()), serial_output = a.output(inputs.size());
    const GdnState base{a.values(matrix), a.values(history)}, candidate = a.state(shape), recovered = a.state(shape);
    const std::array<GdnState, 2> ping{a.state(shape), a.state(shape)};
    const auto scratch = a.tile(shape, tile.capacity), baseline = a.tile(shape, tile.capacity);
    const auto replay = allocate_gdn_core_replay_scratch(g, shape, tile.capacity);
    a.owned.push_back(replay.convolution_output); a.owned.push_back(replay.delta_output);
    const auto ledger = g.memory(); const auto saved_matrix = g.read(base.matrix), saved_history = g.read(base.convolution);
    const auto check = [&](bool ok, const char* why) { ++result.checks; require(ok, why); };
    const auto submit = [&](uint64_t expected) {
        const auto batch = g.submit(); ++result.batches; const auto current = g.memory();
        check(batch.dispatches == expected && current.live_bytes == ledger.live_bytes && current.allocations == ledger.allocations &&
            current.imported_backing_bytes == ledger.imported_backing_bytes, "GDN core batch dispatch/buffer invariant");
        check(g.read(base.matrix) == saved_matrix && g.read(base.convolution) == saved_history, "GDN core mutated accepted state");
    };
    const auto serial = [&](uint32_t steps) {
        auto current = base; g.begin();
        for (uint32_t i = 0; i < steps; ++i) {
            encode_gdn(g, w, token(g, baseline, tile.capacity, i), current, ping[i % 2], row(g, input, shape.channels, i),
                row(g, serial_output, shape.channels, i), shape, policy); current = ping[i % 2];
        }
        submit(uint64_t(steps) * serial_per_token); return current;
    };
    const auto final = serial(tile.steps); result.serial_dispatches = uint64_t(tile.steps) * serial_per_token;
    g.begin(); encode_gdn_core_tile(g, w, scratch, base, candidate, input, output, shape, policy, tile);
    uint32_t chunks = tile.steps / 8; for (uint32_t remainder = tile.steps % 8; remainder; remainder >>= 1) chunks += remainder & 1;
    result.tile_dispatches = uint64_t(tile.steps) * tile_per_token + 3 * chunks + 2 -
        gdn_core_elementwise_saved_dispatches(tile.steps); submit(result.tile_dispatches);
    check(g.read(candidate.matrix) == g.read(final.matrix) && g.read(candidate.convolution) == g.read(final.convolution), "GDN core final state bits differ");
    check(g.read(output) == g.read(serial_output), "GDN core output bits or padding differ");
    const auto tiled_views = scratch.views(), serial_views = baseline.views();
    for (size_t i = 0; i < tiled_views.size(); ++i) check(g.read(tiled_views[i]) == g.read(serial_views[i]), "GDN core stage bits or padding differ");
    std::vector<double> cpu_matrix(matrix.begin(), matrix.end()), cpu_history(history.begin(), history.end());
    const auto close = [&](GpuView view, std::span<const double> expected, const char* stage, uint32_t step) {
        const auto actual = floats(g, view); check(actual.size() == expected.size(), "GDN core double-reference geometry");
        for (size_t i = 0; i < actual.size(); ++i) {
            const auto error = std::abs(double(actual[i]) - expected[i]) / std::max(1.0, std::abs(expected[i]));
            result.maximum_scaled_error = std::max(result.maximum_scaled_error, error); ++result.compared_values;
            ++result.checks;
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error > 5e-5) {
                std::ostringstream message;
                message << "GDN core independent double-reference error: stage=" << stage << " token=" << step
                    << " index=" << i << std::setprecision(17) << " actual=" << actual[i] << " expected=" << expected[i]
                    << " scaled_error=" << error << " limit=5e-5";
                throw FormatError(message.str());
            }
        }
    };
    for (uint32_t step = 0; step < tile.steps; ++step) {
        const auto x = inputs.subspan(uint64_t(step) * shape.channels, shape.channels);
        const auto e = gdn_projection_reference(reference, std::vector<double>(x.begin(), x.end()), cpu_matrix, cpu_history, shape, policy);
        const auto t = token(g, scratch, tile.capacity, step); auto activated = e.convolution.output;
        for (auto& value : activated) value = stable_silu(value);
        const std::array<std::span<const double>, 13> expected{e.projected, activated, e.prepared.query, e.prepared.key,
            e.prepared.value, e.gate, e.alpha, e.beta, e.gates.decay, e.gates.beta, e.delta.output, e.normalized, e.gated};
        const auto views = t.views(); for (size_t i = 0; i < views.size(); ++i) close(views[i], expected[i], stages[i], step);
        close(row(g, output, shape.channels, step), e.output, "output", step); cpu_matrix = e.delta.state; cpu_history = e.convolution.history;
    }
    close(candidate.matrix, cpu_matrix, "final_matrix", tile.steps - 1); close(candidate.convolution, cpu_history, "final_history", tile.steps - 1);
    std::vector<std::vector<std::byte>> tape; for (auto view : scratch.views()) tape.push_back(g.read(view));
    const auto saved_output = g.read(output);
    for (uint32_t k = 1; k <= tile.steps; ++k) if (every_prefix || k == 1 || k == (tile.steps + 1) / 2 || k == tile.steps) {
        const auto prefix = serial(k); g.begin(); encode_gdn_core_prefix(g, w.convolution, scratch, output, base, recovered, replay, shape, {tile.capacity, k}); submit(2);
        check(g.read(recovered.matrix) == g.read(prefix.matrix) && g.read(recovered.convolution) == g.read(prefix.convolution), "GDN core prefix recovery bits differ");
        const auto views = scratch.views(); for (size_t i = 0; i < views.size(); ++i) check(g.read(views[i]) == tape[i], "GDN core prefix recovery mutated tape");
        check(g.read(output) == saved_output, "GDN core prefix recovery mutated outputs");
    }
    result.output_sha256 = checksum(saved_output); result.state_sha256 = checksum(g.read(candidate.matrix)); result.history_sha256 = checksum(g.read(candidate.convolution));
    check(g.memory().live_bytes >= initial.live_bytes, "GDN core diagnostic ledger underflow");
    return result;
}
} // namespace tokspur
