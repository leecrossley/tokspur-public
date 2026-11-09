#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#include "tokspur/moe_q8_three_rows.hpp"
#endif
#include "tokspur/recurrent_tile.hpp"
#include "tokspur/gguf.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace tokspur {
namespace {
constexpr float sentinel = -12345.25f;
constexpr std::array<const char*, 32> names{"ffn_read_normalized", "ffn_read_bottleneck", "ffn_read_scores", "ffn_read_inject", "ffn_read_input", "ffn_read_scales",
    "gdn_projected", "gdn_convolved_activated", "gdn_query", "gdn_key", "gdn_value", "gdn_gate", "gdn_alpha", "gdn_beta_logits", "gdn_decay", "gdn_write_gate",
    "gdn_raw", "gdn_normalized", "gdn_gated", "moe_logits", "moe_ids", "moe_weights", "moe_activated", "moe_up", "moe_down", "moe_shared_activated", "moe_shared_up",
    "moe_shared_down", "moe_shared_logit", "intermediate_streams", "attention_output", "ffn_output"};
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
std::string checksum(std::span<const std::byte> bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    constexpr char digits[] = "0123456789abcdef"; std::string result;
    for (auto value : digest) { result += digits[value >> 4]; result += digits[value & 15]; } return result;
}
struct Arena {
    MetalGraph& g; std::vector<GpuView> owned;
    GpuView values(std::span<const float> values, bool immutable = true) {
        const auto view = g.upload(std::as_bytes(values), "recurrent tile diagnostic", immutable); owned.push_back(view); return view;
    }
    GpuView output(uint64_t count) { return values(std::vector<float>(count, sentinel), false); }
    GdnState state(GdnShape s) { return {output(gdn_state_count(s)), output(gdn_qkv_count(s) * 3)}; }
    RecurrentScratch tile(RecurrentShape s, uint32_t capacity) {
        const auto result = allocate_recurrent_tile_scratch(g, s, capacity);
        for (auto view : result.views()) owned.push_back(view);
        for (auto view : result.views()) { const std::vector<float> cleared(view.bytes / 4, sentinel); g.write(view, std::as_bytes(std::span(cleared))); }
        return result;
    }
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (auto view : owned) g.release(view); }
};
GpuView row(MetalGraph& g, GpuView view, uint64_t count, uint32_t step) { return g.slice(view, uint64_t(step) * count * 4, count * 4); }
template<class T> std::vector<T> read(MetalGraph& g, GpuView view) {
    const auto raw = g.read(view); require(raw.size() % sizeof(T) == 0, "recurrent tile diagnostic read shape");
    std::vector<T> result(raw.size() / sizeof(T)); std::memcpy(result.data(), raw.data(), raw.size()); return result;
}
}
RecurrentTileComparison check_recurrent_tile(MetalGraph& g, const RecurrentWeights& w, const ReferenceLayerWeights& reference,
    RecurrentShape shape, GdnPolicy policy, RouterMode mode, bool normalize, StateTile tile, std::span<const float> inputs,
    std::span<const float> matrix, std::span<const float> history, bool every_prefix, bool reuse_residual_projections) {
    validate_recurrent_shape(shape); validate_state_tile(tile);
    const uint64_t width = uint64_t(shape.residual.branches) * shape.residual.channels;
    require(inputs.size() == width * tile.capacity && matrix.size() == gdn_state_count(shape.gdn) && history.size() == gdn_qkv_count(shape.gdn) * 3,
        "recurrent tile diagnostic input shape");
    // Count unchanged per-token GEMV splits independently before any encoding. Routed
    // expert projections remain single dispatches; the three Q8 core projections batch.
    const auto gemv_extra = [&](GpuMatrix matrix, uint32_t columns, uint32_t rows) {
        return uint64_t(g.gemv_dispatches(matrix.encoding, columns, rows) - 1);
    };
    uint64_t extra = 0;
    const auto c = shape.residual.channels, b = shape.residual.branches, r = shape.residual.rank, bc = b * c;
    for (const auto& part : {w.attention_read, w.ffn_read}) {
        extra += gemv_extra(part.down, bc, r) + gemv_extra(part.up, r, bc) + gemv_extra(part.inject, bc, b);
    }
    const auto& attention = w.attention; const auto value = shape.gdn.value_heads * shape.gdn.value_width;
    extra += gemv_extra(attention.qkv, c, static_cast<uint32_t>(gdn_qkv_count(shape.gdn))) +
        gemv_extra(attention.gate, c, value) + gemv_extra(attention.alpha, c, shape.gdn.value_heads) +
        gemv_extra(attention.beta, c, shape.gdn.value_heads) + gemv_extra(attention.output, value, c);
    const auto& ffn = w.ffn; const auto h = shape.moe.hidden;
    extra += gemv_extra(ffn.router, c, shape.moe.experts) + gemv_extra(ffn.shared_gate, c, h) +
        gemv_extra(ffn.shared_up, c, h) + gemv_extra(ffn.shared_down, h, c) + gemv_extra(ffn.shared_router, c, 1);
    const uint64_t swiglu_saved = g.swiglu_fused_enabled() ? 2u : 0u;
    const uint64_t serial_per_token = 40 - swiglu_saved + extra, tile_per_token = 35 - swiglu_saved + extra;
    Arena a{g, {}}; RecurrentTileComparison result;
    const auto input = a.values(inputs), output = a.output(inputs.size()), baseline_output = a.output(inputs.size());
    const GdnState base{a.values(matrix), a.values(history)}, candidate = a.state(shape.gdn), recovered = a.state(shape.gdn);
    const std::array<GdnState, 2> ping{a.state(shape.gdn), a.state(shape.gdn)};
    const auto scratch = a.tile(shape, tile.capacity), baseline = a.tile(shape, tile.capacity);
    const auto replay = allocate_gdn_core_replay_scratch(g, shape.gdn, tile.capacity);
    a.owned.push_back(replay.convolution_output); a.owned.push_back(replay.delta_output);
    const auto ledger = g.memory(); const auto saved_matrix = g.read(base.matrix), saved_history = g.read(base.convolution), saved_input = g.read(input);
    const auto check = [&](bool ok, const char* why) { ++result.checks; require(ok, why); };
    const auto submit = [&](uint64_t dispatches) {
        const auto batch = g.submit(); ++result.batches; const auto after = g.memory();
        check(batch.dispatches == dispatches && after.live_bytes == ledger.live_bytes && after.allocations == ledger.allocations &&
            after.imported_backing_bytes == ledger.imported_backing_bytes, "recurrent tile dispatch/allocation invariant");
        check(g.read(base.matrix) == saved_matrix && g.read(base.convolution) == saved_history && g.read(input) == saved_input,
            "recurrent tile changed accepted state/input");
    };
    const auto serial = [&](uint32_t steps) {
        auto current = base; g.begin();
        for (uint32_t step = 0; step < steps; ++step) {
            encode_recurrent_layer(g, w, recurrent_tile_token(g, baseline, shape, tile.capacity, step), current, ping[step % 2],
                row(g, input, width, step), row(g, baseline_output, width, step), shape, policy, mode, normalize); current = ping[step % 2];
        }
        submit(uint64_t(steps) * serial_per_token); return current;
    };
    const auto final = serial(tile.steps); result.serial_dispatches = uint64_t(tile.steps) * serial_per_token;
    uint32_t chunks = tile.steps / 8; for (uint32_t n = tile.steps % 8; n; n >>= 1) chunks += n & 1;
    result.tile_dispatches = uint64_t(tile.steps) * tile_per_token + chunks * 3 + 2 -
        gdn_core_elementwise_saved_dispatches(tile.steps);
    if (reuse_residual_projections) {
        uint64_t projections = 0;
        for (const auto& part : {w.attention_read, w.ffn_read})
            projections += uint64_t(part.down.encoding == 8) + uint64_t(part.up.encoding == 8);
        result.tile_dispatches -= projections * (tile.steps - chunks);
    }
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    const bool union_moe = g.q8_three_rows_enabled() && tile.steps == 3 && moe_three_union_layer_supported(g, w.ffn, shape.moe);
    if (g.q8_three_rows_enabled() && tile.steps == 3)
        result.tile_dispatches -= moe_q8_three_rows_saved_dispatches(g, w.ffn, shape.moe);
#else
    const bool union_moe = false;
#endif
    g.begin(); encode_recurrent_layer_tile(g, w, scratch, base, candidate, input, output, shape, policy, mode, normalize, tile, reuse_residual_projections); submit(result.tile_dispatches);
    check(g.read(output) == g.read(baseline_output), "recurrent tile output bits/padding differ");
    check(g.read(candidate.matrix) == g.read(final.matrix) && g.read(candidate.convolution) == g.read(final.convolution), "recurrent tile final-state bits differ");
    const auto views = scratch.views(), ordinary = baseline.views();
    for (size_t i = 0; i < views.size(); ++i) {
        if (union_moe && i == 23) continue; // moe_up is fused in registers inside moe_three_gate_up_swiglu.
        ++result.checks;
        if (g.read(views[i]) != g.read(ordinary[i])) throw FormatError(std::string("recurrent tile exact stage differs: ") + names[i]);
    }
    std::vector<double> cpu_state(matrix.begin(), matrix.end()), cpu_history(history.begin(), history.end());
    const auto close = [&](const char* stage, uint32_t step, GpuView view, std::span<const double> expected) {
        const auto actual = read<float>(g, view); check(actual.size() == expected.size(), "recurrent tile double-reference shape");
        for (size_t i = 0; i < actual.size(); ++i) {
            const auto error = std::abs(double(actual[i]) - expected[i]) / std::max(1.0, std::abs(expected[i]));
            result.maximum_scaled_error = std::max(result.maximum_scaled_error, error); ++result.checks; ++result.compared_values;
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error > 5e-5) {
                std::ostringstream message; message << "recurrent tile double-reference error: stage=" << stage << " token=" << step << " index=" << i
                    << std::setprecision(17) << " actual=" << actual[i] << " expected=" << expected[i] << " scaled_error=" << error << " limit=5e-5";
                throw FormatError(message.str());
            }
        }
    };
    for (uint32_t step = 0; step < tile.steps; ++step) {
        const auto x = inputs.subspan(uint64_t(step) * width, width);
        const auto e = recurrent_reference(reference, std::vector<double>(x.begin(), x.end()), cpu_state, cpu_history, shape, policy, mode, normalize);
        const auto t = recurrent_tile_token(g, scratch, shape, tile.capacity, step);
        close("output_streams", step, row(g, output, width, step), e.output_streams);
        close("intermediate_streams", step, t.intermediate_streams, e.intermediate_streams);
        close("attention_output", step, t.attention_output, e.attention.output); close("ffn_output", step, t.ffn_output, e.ffn.output);
        const std::array<std::span<const double>, 6> residual{e.ffn_read.normalized, e.ffn_read.bottleneck, e.ffn_read.scores,
            e.ffn_read.inject_logits, e.ffn_read.read.input, e.ffn_read.read.write_scales};
        const auto residual_views = t.residual.views(); for (size_t i = 0; i < residual_views.size(); ++i) close(names[i], step, residual_views[i], residual[i]);
        auto activated = e.attention.convolution.output; for (auto& value : activated) value = stable_silu(value);
        const std::array<std::span<const double>, 13> gdn{e.attention.projected, activated, e.attention.prepared.query, e.attention.prepared.key,
            e.attention.prepared.value, e.attention.gate, e.attention.alpha, e.attention.beta, e.attention.gates.decay, e.attention.gates.beta,
            e.attention.delta.output, e.attention.normalized, e.attention.gated};
        const auto gdn_views = t.gdn.views(); for (size_t i = 0; i < gdn_views.size(); ++i) close(names[i + 6], step, gdn_views[i], gdn[i]);
        close("moe_logits", step, t.moe.logits, e.ffn.logits); close("moe_weights", step, t.moe.weights, e.ffn.routes.weights);
        close("moe_activated", step, t.moe.gate, e.ffn.activated); close("moe_routed", step, t.moe.down, e.ffn.routed);
        close("moe_shared_activated", step, t.moe.shared_gate, e.ffn.shared_activated); close("moe_shared_output", step, t.moe.shared_down, e.ffn.shared_output);
        check(read<uint32_t>(g, t.moe.indices) == e.ffn.routes.indices, "recurrent tile CPU expert-ID comparison differs");
        cpu_state = e.attention.delta.state; cpu_history = e.attention.convolution.history;
    }
    close("final_matrix", tile.steps - 1, candidate.matrix, cpu_state); close("final_history", tile.steps - 1, candidate.convolution, cpu_history);
    const auto saved_output = g.read(output); std::vector<std::vector<std::byte>> saved_tape;
    for (auto view : scratch.views()) saved_tape.push_back(g.read(view));
    for (uint32_t k = 1; k <= tile.steps; ++k) if (every_prefix || k == 1 || k == (tile.steps + 1) / 2 || k == tile.steps) {
        const auto prefix = serial(k); g.begin(); encode_recurrent_tile_prefix(g, w, scratch, base, recovered, output, replay, shape, {tile.capacity, k}); submit(2);
        check(g.read(recovered.matrix) == g.read(prefix.matrix) && g.read(recovered.convolution) == g.read(prefix.convolution), "recurrent tile prefix-state bits differ");
        check(g.read(row(g, output, width * k, 0)) == g.read(row(g, baseline_output, width * k, 0)), "recurrent tile retained prefix-output bits differ");
        for (size_t i = 0; i < views.size(); ++i) check(g.read(views[i]) == saved_tape[i], "recurrent tile prefix replay mutated tape");
        check(g.read(output) == saved_output, "recurrent tile prefix replay mutated full output");
    }
    result.output_sha256 = checksum(saved_output); result.state_sha256 = checksum(g.read(candidate.matrix)); result.history_sha256 = checksum(g.read(candidate.convolution));
    return result;
}
} // namespace tokspur
