#include "tokspur/dflash2_drafter.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/sampled_path.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <utility>
#include <vector>

using namespace tokspur;

namespace {

void check(bool ok, const char* why) {
    if (!ok) throw FormatError(why);
}

template <class F>
void rejects(F&& action, const char* why) {
    try {
        action();
    } catch (const FormatError&) {
        return;
    }
    throw FormatError(why);
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return uint16_t(bits >> 16);
}

float bf16_to_f32(uint16_t raw) {
    const uint32_t bits = uint32_t(raw) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = int32_t((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return uint16_t(sign);
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    mant = (mant + 0x00001000u) >> 13;
    if (mant == 0x0400u) return uint16_t(sign | uint32_t(exp + 1) << 10);
    return uint16_t(sign | (uint32_t(exp) << 10) | (mant & 0x03ffu));
}

SampledSparseLaw make_shortlist_law(uint32_t full_vocab, std::span<const uint32_t, 16> cands,
                                    std::span<const float, 16> scores, double temperature) {
    float max_s = scores[0];
    for (size_t i = 1; i < 16; ++i) max_s = std::max(max_s, scores[i]);
    std::array<double, 16> probs{};
    double sum = 0.0;
    for (size_t i = 0; i < 16; ++i) {
        probs[i] = std::exp((double(scores[i]) - double(max_s)) / temperature);
        sum += probs[i];
    }
    std::array<uint32_t, 16> ids{};
    std::array<uint64_t, 16> units{};
    size_t out_n = 0;
    double cum = 0.0;
    uint64_t prev = 0;
    for (size_t i = 0; i < 16; ++i) {
        cum += probs[i] / sum;
        const uint64_t end = (i == 15)
            ? sampled_path_grid_size
            : std::min<uint64_t>(sampled_path_grid_size, uint64_t(std::llround(std::min(1.0, cum) * double(sampled_path_grid_size))));
        if (end > prev) {
            ids[out_n] = cands[i];
            units[out_n] = end - prev;
            prev = end;
            ++out_n;
        }
    }
    if (out_n == 0) {
        ids[0] = cands[0];
        units[0] = sampled_path_grid_size;
        out_n = 1;
    } else if (prev < sampled_path_grid_size) {
        units[out_n - 1] += sampled_path_grid_size - prev;
    }
    return SampledSparseLaw::from_grid_units(full_vocab, std::span(ids).first(out_n), std::span(units).first(out_n));
}

} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: test-dflash2_shortlist METALLIB");
        MetalGraph g(argv[1], 1024ull << 20);
        g.configure_verify_nax(true);
        const std::array<std::pair<uint32_t, PrefillWeightLayout>, 0> no_moe{};
        g.prepare_verify_nax(no_moe);
        g.prepare_dflash2();
        const auto initial_bytes = g.memory().live_bytes;

        // First verify exact anchor >= slice_vocab codebook indexing and sparse law soundness at synthetic scale,
        // then benchmark real 2560 -> 248320 (k=7 vs k=3) vs 2560 -> 32768 (k=3) output projection + selector on GPU.
        {
            constexpr uint32_t hidden = 256u;
            constexpr uint32_t full_vocab = 4096u;
            constexpr uint32_t slice_vocab = 1024u;
            constexpr uint32_t rank = 256u;
            constexpr uint32_t k = 3u;
            constexpr uint32_t high_anchor = 3500u; // Strictly >= slice_vocab (1024) and < full_vocab (4096).

            const auto layout = lanes::row_layout(8, hidden);
            std::vector<std::byte> gguf_out(size_t(full_vocab) * layout.row_bytes);
            std::mt19937 rng(0xdf1a52u);
            std::uniform_int_distribution<int> qdist(-80, 80);
            for (uint32_t r = 0; r < full_vocab; ++r) {
                for (uint32_t b = 0; b < hidden / 32u; ++b) {
                    std::byte* blk = gguf_out.data() + size_t(r) * layout.row_bytes + size_t(b) * 34u;
                    const uint16_t d = f32_to_f16_bits(0.015625f);
                    std::memcpy(blk, &d, sizeof(d));
                    for (uint32_t q = 0; q < 32u; ++q) {
                        const int8_t v = int8_t(qdist(rng));
                        std::memcpy(blk + 2 + q, &v, 1);
                    }
                }
            }
            std::vector<std::byte> packed_out(gguf_out.size());
            lanes::pack_rows(layout, gguf_out, packed_out);

            const auto full_w = g.upload(packed_out, "shortlist.full_out", true);
            g.register_lane_matrix(full_w, {8u, hidden, full_vocab, 1u});
            const auto sliced_w = g.suballocate_draft_output_slice(full_w, slice_vocab, hidden);
            check(g.prefill_weight_layout(sliced_w) == PrefillWeightLayout::q8_lanes_v1,
                  "suballocate_draft_output_slice must preserve q8_lanes_v1 layout");
            const auto second_slice_w = g.upload(
                std::span<const std::byte>(packed_out).subspan(size_t(slice_vocab) * layout.row_bytes, size_t(slice_vocab) * layout.row_bytes),
                "shortlist.second_slice", true);
            g.register_lane_matrix(second_slice_w, {8u, hidden, slice_vocab, 1u});

            std::vector<float> final_h(size_t(k) * hidden);
            for (size_t i = 0; i < final_h.size(); ++i) final_h[i] = float(int(i % 23) - 11) * 0.125f;
            std::vector<float> sel_h(size_t(k) * rank);
            for (size_t i = 0; i < sel_h.size(); ++i) sel_h[i] = float(int((i * 3 + 1) % 19) - 9) * 0.0625f;

            std::vector<uint16_t> pred_cb(size_t(full_vocab) * rank);
            std::vector<uint16_t> succ_cb(size_t(full_vocab) * rank);
            for (size_t i = 0; i < pred_cb.size(); ++i) {
                pred_cb[i] = f32_to_bf16(float(int((i * 5 + 7) % 17) - 8) * 0.0625f);
                succ_cb[i] = f32_to_bf16(float(int((i * 11 + 3) % 21) - 10) * 0.0625f);
            }
            for (uint32_t d = 0; d < rank; ++d) {
                pred_cb[size_t(high_anchor) * rank + d] = f32_to_bf16(1.75f);
                pred_cb[size_t(slice_vocab - 1u) * rank + d] = f32_to_bf16(-1.5f);
            }

            const auto h_gpu = g.upload(std::as_bytes(std::span(final_h)), "shortlist.h");
            const auto sel_h_gpu = g.upload(std::as_bytes(std::span(sel_h)), "shortlist.sel_h");
            const auto pred_gpu = g.upload(std::as_bytes(std::span(pred_cb)), "shortlist.pred", true);
            const auto succ_gpu = g.upload(std::as_bytes(std::span(succ_cb)), "shortlist.succ", true);
            const auto logits_full_gpu = g.zeros(size_t(k) * full_vocab * 4u, "shortlist.logits_full");
            const auto logits_slice_gpu = g.zeros(size_t(k) * slice_vocab * 4u, "shortlist.logits_slice");
            const auto top_ids_gpu = g.zeros(7u * 8u * 16u * 4u, "shortlist.top_ids");
            const auto top_vals_gpu = g.zeros(7u * 8u * 16u * 4u, "shortlist.top_vals");
            const size_t sel_out_bytes = 7u * 16u * 8u + 7u * 256u * 4u;
            const auto sel_out_gpu = g.zeros(sel_out_bytes, "shortlist.sel_out");

            // Calling dflash2_select without codebook_vocabulary when anchor >= slice_vocab must fail!
            rejects([&] {
                g.begin();
                g.set_dflash2_scope(true);
                try {
                    g.dflash2_select(logits_slice_gpu, top_ids_gpu, top_vals_gpu, sel_out_gpu, sel_h_gpu,
                                     pred_gpu, succ_gpu, k, slice_vocab, high_anchor);
                } catch (...) {
                    g.set_dflash2_scope(false);
                    g.abort();
                    throw;
                }
                g.set_dflash2_scope(false);
                static_cast<void>(g.submit());
            }, "dflash2_select accepted anchor >= vocabulary without codebook_vocabulary");

            g.begin();
            g.set_dflash2_scope(true);
            g.verify_nax_dense_q8(full_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                                  GpuRowsF32{logits_full_gpu, k, full_vocab, full_vocab * 4u}, false);
            g.verify_nax_dense_q8(sliced_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                                  GpuRowsF32{logits_slice_gpu, k, slice_vocab, slice_vocab * 4u}, false);
            g.dflash2_select(logits_slice_gpu, top_ids_gpu, top_vals_gpu, sel_out_gpu, sel_h_gpu,
                             pred_gpu, succ_gpu, k, slice_vocab, high_anchor, full_vocab);
            g.set_dflash2_scope(false);
            static_cast<void>(g.submit());

            const auto logits_staged_full = g.zeros(size_t(k) * full_vocab * 4u, "shortlist.logits_staged_full");
            const auto logits_staged_slice = g.zeros(size_t(k) * slice_vocab * 4u, "shortlist.logits_staged_slice");
            const auto gu_ref_gpu = g.zeros(size_t(k) * slice_vocab * 4u, "shortlist.gu_ref");
            const auto gu_fused_gpu = g.zeros(size_t(k) * slice_vocab * 4u, "shortlist.gu_fused");

            g.begin();
            g.set_dflash2_scope(true);
            g.dflash2_dense_q8(full_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                               GpuRowsF32{logits_staged_full, k, full_vocab, full_vocab * 4u}, false);
            g.dflash2_dense_q8(sliced_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                               GpuRowsF32{logits_staged_slice, k, slice_vocab, slice_vocab * 4u}, false);
            g.verify_nax_dense_q8(sliced_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                                  GpuRowsF32{gu_ref_gpu, k, slice_vocab, slice_vocab * 4u}, false);
            g.verify_nax_dense_q8(second_slice_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                                  GpuRowsF32{gu_ref_gpu, k, slice_vocab, slice_vocab * 4u}, true);
            g.dflash2_gate_up_swiglu(sliced_w, second_slice_w, GpuRowsF32{h_gpu, k, hidden, hidden * 4u},
                                     GpuRowsF32{gu_fused_gpu, k, slice_vocab, slice_vocab * 4u}, false);
            g.set_dflash2_scope(false);
            static_cast<void>(g.submit());

            check(g.read(logits_staged_full) == g.read(logits_full_gpu),
                  "dflash2_dense_q8_r4_m8 differed bit-for-bit from verify_nax_dense_q8");
            check(g.read(logits_staged_slice) == g.read(logits_slice_gpu),
                  "dflash2_dense_q8_m8 differed bit-for-bit from verify_nax_dense_q8");
            check(g.read(gu_fused_gpu) == g.read(gu_ref_gpu),
                  "dflash2_gate_up_swiglu_q8_m8 differed bit-for-bit from unfused gate+up+SwiGLU");
            g.release(gu_fused_gpu);
            g.release(gu_ref_gpu);
            g.release(logits_staged_slice);
            g.release(logits_staged_full);

            // Also verify dflash2_conv_finish_rms_f32 bit-exactness against dflash2_conv_f32(finish=true) + rms_rows!
            {
                constexpr uint32_t c_rows = 8u, c_hid = 2560u, c_dyn = 640u;
                std::vector<float> c_in(c_rows * c_hid), c_dy(c_rows * c_dyn), c_base(4u * c_hid), c_res(c_rows * c_hid), c_gains(c_hid);
                for (size_t i = 0; i < c_in.size(); ++i) c_in[i] = float(int((i * 7 + 3) % 29) - 14) * 0.0625f;
                for (size_t i = 0; i < c_dy.size(); ++i) c_dy[i] = float(int((i * 5 + 1) % 19) - 9) * 0.03125f;
                for (size_t i = 0; i < c_base.size(); ++i) c_base[i] = float(int((i * 11 + 2) % 23) - 11) * 0.03125f;
                for (size_t i = 0; i < c_res.size(); ++i) c_res[i] = float(int((i * 13 + 5) % 31) - 15) * 0.0625f;
                for (size_t i = 0; i < c_gains.size(); ++i) c_gains[i] = 1.0f + float(int(i % 17) - 8) * 0.0625f;

                const auto g_in = g.upload(std::as_bytes(std::span(c_in)), "c.in");
                const auto g_dy = g.upload(std::as_bytes(std::span(c_dy)), "c.dy");
                const auto g_base = g.upload(std::as_bytes(std::span(c_base)), "c.base", true);
                const auto g_res = g.upload(std::as_bytes(std::span(c_res)), "c.res");
                const auto g_gains = g.upload(std::as_bytes(std::span(c_gains)), "c.gains", true);
                const auto ref_rout = g.zeros(c_rows * c_hid * 4u, "c.ref_rout");
                const auto ref_nout = g.zeros(c_rows * c_hid * 4u, "c.ref_nout");
                const auto opt_rout = g.zeros(c_rows * c_hid * 4u, "c.opt_rout");
                const auto opt_nout = g.zeros(c_rows * c_hid * 4u, "c.opt_nout");

                g.begin();
                g.set_dflash2_scope(true);
                g.dflash2_conv_f32(g_in, g_dy, g_base, g_res, ref_rout, c_hid, true);
                g.rms_rows(GpuRowsF32{ref_rout, c_rows, c_hid, c_hid * 4u}, g_gains,
                           GpuRowsF32{ref_nout, c_rows, c_hid, c_hid * 4u}, c_hid, 1e-6f, true);
                g.dflash2_conv_finish_rms_f32(g_in, g_dy, g_base, g_res, g_gains, opt_rout, opt_nout, c_hid, 1e-6f);
                g.set_dflash2_scope(false);
                static_cast<void>(g.submit());

                check(g.read(opt_rout) == g.read(ref_rout), "dflash2_conv_finish_rms_f32 residual_out bit mismatch");
                check(g.read(opt_nout) == g.read(ref_nout), "dflash2_conv_finish_rms_f32 norm_out bit mismatch");

                g.release(opt_nout); g.release(opt_rout); g.release(ref_nout); g.release(ref_rout);
                g.release(g_gains); g.release(g_res); g.release(g_base); g.release(g_dy); g.release(g_in);
            }

            std::vector<float> full_host(size_t(k) * full_vocab);
            std::vector<float> slice_host(size_t(k) * slice_vocab);
            std::vector<std::byte> sel_host(sel_out_bytes);
            g.read(logits_full_gpu, std::as_writable_bytes(std::span(full_host)));
            g.read(logits_slice_gpu, std::as_writable_bytes(std::span(slice_host)));
            g.read(sel_out_gpu, sel_host);

            for (uint32_t pos = 0; pos < k; ++pos) {
                for (uint32_t c = 0; c < slice_vocab; ++c) {
                    check(full_host[size_t(pos) * full_vocab + c] == slice_host[size_t(pos) * slice_vocab + c],
                          "sliced verify_nax_dense_q8 logit differed from full projection prefix");
                }
            }

            const auto* cands = reinterpret_cast<const uint32_t*>(sel_host.data());
            const auto* unary = reinterpret_cast<const float*>(sel_host.data() + 7u * 16u * sizeof(uint32_t));
            const auto* tables = reinterpret_cast<const float*>(sel_host.data() + 7u * 16u * (sizeof(uint32_t) + sizeof(float)));

            for (uint32_t j = 0; j < 16u; ++j) {
                const uint32_t cid = cands[j];
                check(cid < slice_vocab, "shortlist candidate exceeded slice_vocab");
                float ref_edge = 0.0f;
                for (uint32_t d = 0; d < rank; ++d) {
                    const float p = bf16_to_f32(pred_cb[size_t(high_anchor) * rank + d]);
                    const float s = bf16_to_f32(succ_cb[size_t(cid) * rank + d]);
                    ref_edge += (p * sel_h[d]) * s;
                }
                check(std::fabs(tables[j] - ref_edge) <= 1e-4f, "row 0 edge score did not use high_anchor codebook row");
            }

            const SamplingPolicy pol{0.8, 32u, 0.95, 0.0};
            const SampledPathIdentity ident{1ull, 1ull, 1ull, 100u, full_vocab, high_anchor};
            SamplingRng prop_rng(77ull);
            const uint64_t begin_words = prop_rng.words_consumed();
            std::array<SampledProposalRow, k> proposals{};
            uint32_t prev = high_anchor;
            uint32_t pred_idx = 0u;
            for (uint32_t pos = 0; pos < k; ++pos) {
                std::array<uint32_t, 16> pc{};
                std::array<float, 16> ps{};
                for (uint32_t j = 0; j < 16u; ++j) {
                    pc[j] = cands[pos * 16u + j];
                    ps[j] = unary[pos * 16u + j] + tables[(pos * 16u + pred_idx) * 16u + j];
                }
                auto law = make_shortlist_law(full_vocab, pc, ps, pol.temperature);
                check(law.vocabulary() == full_vocab, "selector law must span full target vocabulary");
                const uint64_t ticket = law.deterministic() ? 0ull : uint64_t(std::ldexp(prop_rng.uniform(), 53));
                const uint32_t selected = law.sample_at(ticket);
                for (uint32_t j = 0; j < 16u; ++j) {
                    if (pc[j] == selected) { pred_idx = j; break; }
                }
                proposals[pos] = SampledProposalRow{100u + pos, prev, selected, std::move(law)};
                prev = selected;
            }
            // Target logits strongly favour token 3999 (>= slice_vocab!) at position 100: rejection recovery must sample 3999!
            std::vector<float> target_logits(size_t(k + 1u) * full_vocab, -20.0f);
            target_logits[3999u] = 25.0f;
            for (uint32_t r = 1; r <= k; ++r) target_logits[size_t(r) * full_vocab + 2000u + r] = 25.0f;
            std::array<SampledTargetRow, k + 1u> targets{};
            for (uint32_t r = 0; r <= k; ++r) {
                targets[r] = SampledTargetRow{100u + r, std::span<const float>(target_logits.data() + size_t(r) * full_vocab, full_vocab)};
            }
            const auto dec = decide_sampled_path(ident, proposals, ident, targets, pol, 16u, {}, prop_rng, begin_words);
            check(dec.accepted == 0u && dec.count == 1u && dec.tokens[0].id == 3999u,
                  "32K draft slice blocked out-of-slice target token 3999 during rejection recovery");

            g.release(sel_out_gpu);
            g.release(top_vals_gpu);
            g.release(top_ids_gpu);
            g.release(logits_slice_gpu);
            g.release(logits_full_gpu);
            g.release(succ_gpu);
            g.release(pred_gpu);
            g.release(sel_h_gpu);
            g.release(h_gpu);
            g.release(second_slice_w);
            g.release(full_w);
        }

        check(g.memory().live_bytes == initial_bytes, "test_dflash2_shortlist synthetic block leaked GPU buffers");

        // Now measure actual Gemma-4-26B DFlash2 output projection + selector GPU time on M5 Max:
        // (A) k=7 x 248,320  (B) k=3 x 248,320  (C) k=3 x 32,768.
        {
            MetalGraph gb(argv[1], 1024ull << 20);
            gb.configure_verify_nax(true);
            gb.prepare_verify_nax(no_moe);
            gb.prepare_dflash2();
            const auto bench_initial_bytes = gb.memory().live_bytes;

            constexpr uint32_t hidden = 2560u;
            constexpr uint32_t full_vocab = 248320u;
            constexpr uint32_t slice_vocab = 32768u;
            constexpr uint32_t rank = 256u;

            const auto layout = lanes::row_layout(8, hidden);
            std::vector<std::byte> row_template(layout.row_bytes, std::byte{1});
            const uint16_t d_bits = f32_to_f16_bits(0.0078125f);
            for (uint32_t b = 0; b < hidden / 32u; ++b) {
                std::memcpy(row_template.data() + size_t(b) * 2u, &d_bits, sizeof(d_bits));
            }
            GpuView w_gpu{};
            {
                std::vector<std::byte> full_packed(size_t(full_vocab) * layout.row_bytes);
                for (uint32_t r = 0; r < full_vocab; ++r) {
                    std::memcpy(full_packed.data() + size_t(r) * layout.row_bytes, row_template.data(), layout.row_bytes);
                }
                w_gpu = gb.upload(full_packed, "bench.out_weight", true);
            }
            gb.register_lane_matrix(w_gpu, {8u, hidden, full_vocab, 1u});
            const auto w_slice_gpu = gb.suballocate_draft_output_slice(w_gpu, slice_vocab, hidden);

            const std::vector<float> h7(7u * hidden, 0.0625f);
            const std::vector<float> sh7(7u * rank, 0.0625f);
            const auto h_gpu = gb.upload(std::as_bytes(std::span(h7)), "bench.h7");
            const auto sh_gpu = gb.upload(std::as_bytes(std::span(sh7)), "bench.sh7");
            const auto pred_gpu = gb.zeros(uint64_t(full_vocab) * rank * 2u, "bench.pred");
            const auto succ_gpu = gb.zeros(uint64_t(full_vocab) * rank * 2u, "bench.succ");
            const auto logits_gpu = gb.zeros(7ull * full_vocab * 4u, "bench.logits");
            const auto top_ids_gpu = gb.zeros(7u * 8u * 16u * 4u, "bench.top_ids");
            const auto top_vals_gpu = gb.zeros(7u * 8u * 16u * 4u, "bench.top_vals");
            const auto sel_out_gpu = gb.zeros(7u * 16u * 8u + 7u * 256u * 4u, "bench.sel_out");

            auto bench_case = [&](GpuView weight, uint32_t k, uint32_t out_vocab, bool staged) -> double {
                const auto h_k = gb.slice(h_gpu, 0, uint64_t(k) * hidden * 4u);
                const auto sh_k = gb.slice(sh_gpu, 0, uint64_t(k) * rank * 4u);
                const auto log_k = gb.slice(logits_gpu, 0, uint64_t(k) * out_vocab * 4u);
                for (uint32_t w = 0; w < 3u; ++w) {
                    gb.begin();
                    gb.set_dflash2_scope(true);
                    if (staged) gb.dflash2_dense_q8(weight, GpuRowsF32{h_k, k, hidden, hidden * 4u},
                                                    GpuRowsF32{log_k, k, out_vocab, uint64_t(out_vocab) * 4u}, false);
                    else gb.verify_nax_dense_q8(weight, GpuRowsF32{h_k, k, hidden, hidden * 4u},
                                                GpuRowsF32{log_k, k, out_vocab, uint64_t(out_vocab) * 4u}, false);
                    gb.dflash2_select(log_k, top_ids_gpu, top_vals_gpu, sel_out_gpu, sh_k, pred_gpu, succ_gpu,
                                      k, out_vocab, 200000u, full_vocab);
                    gb.set_dflash2_scope(false);
                    static_cast<void>(gb.submit());
                }
                double sum_gpu_ms = 0.0;
                constexpr uint32_t iters = 12u;
                for (uint32_t it = 0; it < iters; ++it) {
                    gb.begin();
                    gb.set_dflash2_scope(true);
                    if (staged) gb.dflash2_dense_q8(weight, GpuRowsF32{h_k, k, hidden, hidden * 4u},
                                                    GpuRowsF32{log_k, k, out_vocab, uint64_t(out_vocab) * 4u}, false);
                    else gb.verify_nax_dense_q8(weight, GpuRowsF32{h_k, k, hidden, hidden * 4u},
                                                GpuRowsF32{log_k, k, out_vocab, uint64_t(out_vocab) * 4u}, false);
                    gb.dflash2_select(log_k, top_ids_gpu, top_vals_gpu, sel_out_gpu, sh_k, pred_gpu, succ_gpu,
                                      k, out_vocab, 200000u, full_vocab);
                    gb.set_dflash2_scope(false);
                    sum_gpu_ms += gb.submit().gpu_ms;
                }
                return sum_gpu_ms / iters;
            };

            const double ms_7x248k_old = bench_case(w_gpu, 7u, full_vocab, false);
            const double ms_7x248k_new = bench_case(w_gpu, 7u, full_vocab, true);
            const double ms_3x248k_new = bench_case(w_gpu, 3u, full_vocab, true);
            const double ms_3x32k_new = bench_case(w_slice_gpu, 3u, slice_vocab, true);

            std::cout << "dflash2_shortlist: anchor>=slice codebook indexing, sparse law closure, out-of-slice recovery and staged bit-parity passed"
                      << " | 2560->vocab+select gpu_ms: old_7x248320=" << ms_7x248k_old
                      << " staged_7x248320=" << ms_7x248k_new
                      << " staged_3x248320=" << ms_3x248k_new
                      << " staged_3x32768=" << ms_3x32k_new << '\n';

            gb.release(sel_out_gpu);
            gb.release(top_vals_gpu);
            gb.release(top_ids_gpu);
            gb.release(logits_gpu);
            gb.release(succ_gpu);
            gb.release(pred_gpu);
            gb.release(sh_gpu);
            gb.release(h_gpu);
            gb.release(w_gpu);
            check(gb.memory().live_bytes == bench_initial_bytes, "test_dflash2_shortlist bench block leaked GPU buffers");
        }

        check(g.memory().live_bytes == initial_bytes, "test_dflash2_shortlist leaked GPU buffers");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
