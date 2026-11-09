#include "tokspur/prefill_gdn.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>

namespace tokspur {
namespace {
void need(bool ok, const char* why) { if (!ok) throw FormatError(why); }
MoeMmLayouts layouts(const MetalGraph& g, const MoeWeights& w) {
    return {g.prefill_weight_layout(w.gate.view), g.prefill_weight_layout(w.up.view), g.prefill_weight_layout(w.down.view),
        g.prefill_weight_layout(w.shared_gate.view), g.prefill_weight_layout(w.shared_up.view), g.prefill_weight_layout(w.shared_down.view)};
}
bool prefill_nax_2k(const MetalGraph& g) {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    return g.prefill_dense_nax() && g.prefill_grouped_nax();
#else
    static_cast<void>(g); return false;
#endif
}
void project(MetalGraph& g, const GpuMatrix& w, GpuRowsF32 input, GpuRowsF32 output) {
    need(w.encoding == 8, "GDN prefill callback requires Q8");
    g.dense_mm_q8(g.prefill_weight_layout(w.view), w.view, input, output, g.can_reuse_nax_dense_bf16(input));
}
GpuRowsF32 rows(GpuView view, uint32_t n, uint32_t columns) { return {view, n, columns, uint64_t(columns) * 4}; }
std::array<GpuView, 25> weights(const RecurrentWeights& w) {
    return {w.attention_read.gains, w.attention_read.down.view, w.attention_read.up.view, w.attention_read.inject.view,
        w.ffn_read.gains, w.ffn_read.down.view, w.ffn_read.up.view, w.ffn_read.inject.view,
        w.attention.qkv.view, w.attention.gate.view, w.attention.alpha.view, w.attention.beta.view, w.attention.output.view,
        w.attention.convolution, w.attention.negative_rates, w.attention.dt_bias, w.attention.gains,
        w.ffn.router.view, w.ffn.gate.view, w.ffn.up.view, w.ffn.down.view, w.ffn.shared_gate.view,
        w.ffn.shared_up.view, w.ffn.shared_down.view, w.ffn.shared_router.view};
}
void check_layouts(const MetalGraph& g, const RecurrentWeights& w, GdnPrefillLayouts p) {
    const auto actual = gdn_prefill_layouts(g, w);
    need(actual.qkv == p.qkv && actual.gate == p.gate && actual.output == p.output, "GDN prefill projection layout changed");
    const auto a = actual.ffn, b = p.ffn;
    need(a.gate == b.gate && a.up == b.up && a.down == b.down && a.shared_gate == b.shared_gate &&
        a.shared_up == b.shared_up && a.shared_down == b.shared_down, "GDN prefill FFN layout changed");
}
void abort(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
}
GdnPrefillLayouts gdn_prefill_layouts(const MetalGraph& g, const RecurrentWeights& w) {
    return {g.prefill_weight_layout(w.attention.qkv.view), g.prefill_weight_layout(w.attention.gate.view),
        g.prefill_weight_layout(w.attention.output.view), layouts(g, w.ffn)};
}
void prepare_gdn_prefill_layer(MetalGraph& g, const RecurrentWeights& w, RecurrentShape shape, GdnPrefillLayouts p, bool legacy_order) {
    validate_recurrent_shape(shape); check_layouts(g, w, p);
    need(w.attention.qkv.encoding == 8 && w.attention.gate.encoding == 8 && w.attention.output.encoding == 8,
        "GDN prefill requires Q8 core projections");
    prepare_prefill_residual(g, w.attention_read, shape.residual); prepare_prefill_residual(g, w.ffn_read, shape.residual);
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    if (legacy_order) g.prepare_gdn_legacy_batch();
#else
    need(!legacy_order, "old-order recurrence is not compiled");
#endif
    g.prepare_gdn_chunk(); for (const auto layout : {p.qkv, p.gate, p.output}) g.prepare_dense_mm_q8(layout);
    prepare_moe_mm_many(g, w.ffn, shape.moe, p.ffn);
}
uint64_t gdn_prefill_layer_dispatches(const MetalGraph& g, const RecurrentWeights& w, RecurrentShape shape, uint32_t n, GdnChunkMode mode) {
    validate_recurrent_shape(shape); need(n && n <= 2048 && uint32_t(mode) <= 1, "GDN prefill layer rows or mode");
    const uint32_t slice_cap = prefill_nax_2k(g) ? 2048u : 512u;
    const uint64_t extra_q8 = (g.dense_mm_q8_dispatches(shape.gdn.channels) - 1) +
        (g.dense_mm_q8_dispatches(shape.gdn.value_heads * shape.gdn.value_width) - 1);
    const auto qkv_layout = g.prefill_weight_layout(w.attention.qkv.view);
    const auto gate_layout = g.prefill_weight_layout(w.attention.gate.view);
    uint64_t core = 0;
    for (uint32_t offset = 0; offset < n; offset += slice_cap) {
        const uint32_t chunk = std::min(slice_cap, n - offset);
        const uint64_t qkv_gate_saved = (w.attention.qkv.encoding == 8 && w.attention.gate.encoding == 8 &&
            g.can_dense_mm_q8_multiseg({qkv_layout, gate_layout}, chunk)) ? 1ull : 0ull;
        const uint64_t slice_core = (mode == GdnChunkMode::fused ? 7ull : (6ull + 4ull * ((chunk + 63) / 64))) + extra_q8 - qkv_gate_saved;
        core = checked_add(core, slice_core);
    }
    const uint64_t hc_saved = can_fuse_prefill_hc_consumer(g, shape.residual, n) ? 1ull : 0ull;
    const auto writes = prefill_residual_write_dispatches(g, shape.residual, n);
    return prefill_residual_dispatches(g, w.attention_read, shape.residual, n) + core + writes +
        prefill_residual_dispatches(g, w.ffn_read, shape.residual, n) - hc_saved +
        moe_mm_many_dispatches(g, w.ffn, shape.moe, n) + writes;
}
void encode_gdn_prefill_layer(MetalGraph& g, const RecurrentWeights& w, const PrefillLayerScratch& s,
    const GdnChunkScratch& core, const MoeMmManyScratch& moe, const GdnState& accepted, const GdnState& candidate,
    GpuRowsF32 input, GpuRowsF32 output, RecurrentShape shape, GdnPolicy policy,
    RouterMode router, bool normalize, GdnChunkMode mode, GdnPrefillLayouts p, bool legacy_order,
    const GdnState* work_state) {
    try {
        validate_recurrent_shape(shape); check_layouts(g, w, p); const auto n = input.rows, c = shape.residual.channels, width = shape.residual.branches * c;
        const uint32_t slice_cap = prefill_nax_2k(g) ? 2048u : 512u;
        need(n && n <= s.residual.capacity && core.capacity == std::min(s.residual.capacity, slice_cap) && moe.capacity && moe.capacity <= s.residual.capacity &&
            output.rows == n && input.columns == width && output.columns == width &&
            s.intermediate_streams.bytes == uint64_t(s.residual.capacity) * width * 4 &&
            s.attention_output.bytes == uint64_t(s.residual.capacity) * c * 4 && s.ffn_output.bytes == s.attention_output.bytes,
            "GDN prefill layer scratch or row geometry");
        const uint32_t num_slices = (n + slice_cap - 1) / slice_cap;
        need(num_slices == 1 || (work_state && work_state->matrix.owner && work_state->convolution.owner),
            "multi-slice GDN prefill requires temporary state workspace");
        std::array<GpuView, 28> reads{input.view, accepted.matrix, accepted.convolution}; const auto wv = weights(w);
        std::copy(wv.begin(), wv.end(), reads.begin() + 3);
        std::array<GpuView, 45> writes{}; size_t used = 0;
        const auto append = [&](auto views) { for (auto v : views) if (v.owner) writes[used++] = v; };
        append(s.views()); append(core.views()); append(moe.views()); append(std::array{candidate.matrix, candidate.convolution, output.view});
        if (num_slices > 1) append(std::array{work_state->matrix, work_state->convolution});
        g.validate_disjoint(reads, std::span(writes).first(used));
        const auto intermediate = rows(s.intermediate_streams, n, width), attention = rows(s.attention_output, n, c), ffn = rows(s.ffn_output, n, c);
        encode_prefill_residual(g, w.attention_read, s.residual, input, shape.residual, policy.rms_epsilon);
        const auto res_in = prefill_residual_input(s.residual, shape.residual, n);
        for (uint32_t idx = 0; idx < num_slices; ++idx) {
            const uint32_t offset = idx * slice_cap, chunk = std::min(slice_cap, n - offset), rem = num_slices - 1 - idx;
            const GdnState& in_state = idx == 0 ? accepted : ((rem % 2 == 1) ? candidate : *work_state);
            const GdnState& out_state = (rem % 2 == 1) ? *work_state : candidate;
            const auto in_slice = rows(g.slice(res_in.view, uint64_t(offset) * c * 4, uint64_t(chunk) * c * 4), chunk, c);
            const auto out_slice = rows(g.slice(s.attention_output, uint64_t(offset) * c * 4, uint64_t(chunk) * c * 4), chunk, c);
            encode_gdn_prefill_chunk(g, w.attention, core, in_state, out_state, in_slice, out_slice, shape.gdn, policy, mode, project, legacy_order, true);
        }
        encode_prefill_residual_write_and_read(g, w.ffn_read, s.residual, input, attention, intermediate, shape.residual, policy.rms_epsilon);
        encode_moe_mm_many(g, w.ffn, moe, prefill_residual_input(s.residual, shape.residual, n), ffn,
            shape.moe, router, normalize, p.ffn);
        encode_prefill_residual_write(g, s.residual, intermediate, ffn, output, shape.residual);
    } catch (...) { abort(g); throw; }
}
}
