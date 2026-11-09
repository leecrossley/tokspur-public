#include "tokspur/mtp_head.hpp"
#include "tokspur/operator_checkpoints.hpp"
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
#include "tokspur/prefill_residual.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>

namespace tokspur {
namespace {
void require(bool value, const char* message) { if (!value) throw FormatError(message); }
void finite(std::span<const float> values) {
    uint32_t bad = 0;
    for (const float value : values) bad |= uint32_t((std::bit_cast<uint32_t>(value) & 0x7f800000u) == 0x7f800000u);
    require(!bad, "nonfinite MTP input/output");
}
void validate(MtpHeadShape s, MtpHeadPolicy p) {
    validate_residual_shape(s.residual);
    require(s.experts.channels == s.residual.channels && s.experts.hidden && s.experts.hidden <= 65536 &&
        s.experts.experts && s.experts.experts <= 4096 && s.experts.selected &&
        s.experts.selected <= s.experts.experts && s.experts.selected <= 64 &&
        s.query_heads && s.query_heads <= 64 && s.kv_heads && s.kv_heads <= s.query_heads &&
        s.query_heads % s.kv_heads == 0 && s.head_width && s.head_width <= 256 && s.head_width % 2 == 0 &&
        s.rotary_width && s.rotary_width <= s.head_width && s.rotary_width % 2 == 0 &&
        s.vocabulary && s.vocabulary <= (1u << 20) && s.capacity && s.capacity <= 65536,
        "unsupported MTP head geometry");
    require(uint32_t(p.query_packing) <= 1 && uint32_t(p.rotary_pairs) <= 1 && uint32_t(p.router) <= 1 &&
        uint32_t(p.draft_attention) <= 2 &&
        std::isfinite(p.rms_epsilon) && p.rms_epsilon > 0, "invalid MTP head policy");
}
} // namespace

struct MtpHead::Impl {
    MetalGraph& graph;
    MtpHeadWeights weights{};
    MtpHeadShape shape{};
    MtpHeadPolicy policy{};
    std::vector<GpuView> owned;
    std::shared_ptr<const ModelFiles> sidecar, target;
    GpuView hc_input{}, embedding_input{}, normalized_hc{}, normalized_embedding{}, concat{}, projected{},
        intermediate{}, final_hc{}, packed{}, raw_key{}, raw_value{}, query{}, gate{}, normalized_query{},
        normalized_key{}, rotary_query{}, rotary_key{}, keys{}, values{}, attention{}, gated{},
        attention_output{}, ffn_output{}, output_logits{}, attention_partials{};
    ResidualScratch residual{};
    MoeScratch moe{};
    MixerScratch mixer{};
    std::vector<float> host_logits, host_hc;
    uint32_t draft_vocabulary = 0, fast_token_embd_vocab = 0;
    GpuMatrix draft_output{};
    GpuView topk_scratch{}, topk_row{}, topk_chain_rows{}, sampled_tickets_gpu{}, sampled_chain_laws{}, fast_token_embd{}, deferred_hc{}, deferred_emb{};
    uint32_t first = 1, rows = 0, pending_prime_count = 0, pending_prime_pos = 0;
    uint64_t charged = 0;
    bool valid = true, logits_valid = false, hidden_valid = false, hidden_gpu_valid = false, fast_round = false, greedy_chain_ready = false, busy = false;
    [[nodiscard]] uint32_t logical_rows() const noexcept { return rows + pending_prime_count; }
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
    GpuView priming_hc{}, priming_embeddings{};
    GpuView priming_norm_hc{}, priming_norm_emb{}, priming_concat{}, priming_projected{},
        priming_raw_key{}, priming_raw_value{}, priming_norm_key{};
    PrefillResidualScratch priming_residual{};
    uint32_t priming_capacity = 0, priming_slice_cap = 0;
    uint64_t priming_bytes = 0;
    bool priming_prepare_locked = false, priming_batch_ready = false;
#endif

    explicit Impl(MetalGraph& g) : graph(g) { owned.reserve(96); }
    ~Impl() {
        if (busy) try { graph.abort(); } catch (...) {}
        for (auto it = owned.rbegin(); it != owned.rend(); ++it) {
            try { graph.release(*it); } catch (...) {}
        }
    }
    GpuView floats(uint64_t count, const char* label) {
        const auto view = graph.zeros(checked_mul(count, 4), label); owned.push_back(view); return view;
    }
    void retain(std::span<const GpuView> views) { for (const auto view : views) owned.push_back(view); }

    void import(const MtpSharedBinding& binding, GpuMatrix output, uint32_t capacity, const std::map<std::string, GpuView>* pack_views = nullptr) {
        sidecar = binding.sidecar_source(); target = binding.target_source();
        const auto& metadata = sidecar->shards().front();
        shape = {{4, 2560, 320}, {2560, 640, 512, 10}, 24, 2, 256, 64, 248320, capacity};
        policy.rms_epsilon = float(metadata.floating_value("qwen4exp.attention.layer_norm_rms_epsilon"));
        std::map<std::string, GpuView> adopted_views;
        const bool use_pack = pack_views && pack_views->contains("blk.48.nextn.hnorm.weight");
        if (!use_pack) {
            std::vector<TensorSlice> slices;
            std::map<std::string, size_t> indices;
            for (const auto& requirement : mtp_shared_q8_requirements()) {
                // This profile explicitly declares compression0. Indexer tensors
                // are admitted/accounted metadata but no dense-head operation uses them.
                if (requirement.name.starts_with("blk.48.indexer.")) continue;
                indices.emplace(requirement.name, slices.size());
                slices.push_back({requirement.name, 0, 0});
            }
            require(slices.size() == 28, "MTP dense weight catalogue mismatch");
            const auto imports = graph.import_weights(sidecar, slices, 0, true);
            retain(imports.regions);
            for (const auto& [name, idx] : indices) adopted_views.emplace(name, imports.tensors.at(idx));
        }
        const auto tensor = [&](const std::string& suffix) {
            const auto name = "blk.48." + suffix;
            const auto view = use_pack ? pack_views->at(name) : adopted_views.at(name);
            return GpuMatrix{view, sidecar->tensor(name).type};
        };
        const auto read = [&](const std::string& kind) {
            return ResidualWeights{tensor(kind + "_norm.weight").view, tensor(kind + "_down.weight"),
                tensor(kind + "_up.weight"), tensor(kind + "_inject.weight")};
        };
        const auto frequencies = rotary_frequencies(shape.rotary_width, metadata.floating_value("qwen4exp.rope.freq_base"));
        const auto frequency_view = graph.upload(std::as_bytes(std::span(frequencies)), "MTP rotary frequencies");
        owned.push_back(frequency_view);
        weights = {tensor("nextn.hnorm.weight").view, tensor("nextn.enorm.weight").view,
            tensor("nextn.eh_proj.weight"), read("hc_attn"), read("hc_ffn"),
            {tensor("attn_q.weight"), tensor("attn_k.weight"), tensor("attn_v.weight"), tensor("attn_output.weight"),
                tensor("attn_q_norm.weight").view, tensor("attn_k_norm.weight").view, frequency_view},
            {tensor("ffn_gate_inp.weight"), tensor("ffn_gate_exps.weight"), tensor("ffn_up_exps.weight"),
                tensor("ffn_down_exps.weight"), tensor("ffn_gate_shexp.weight"), tensor("ffn_up_shexp.weight"),
                tensor("ffn_down_shexp.weight"), tensor("ffn_gate_inp_shexp.weight")},
            {tensor("nextn.hc_head_norm.weight").view, tensor("nextn.hc_head_down.weight"), tensor("nextn.hc_head_up.weight")},
            output};
        binding.assert_unchanged();
    }

    void prepare() {
        validate(shape, policy); graph.prepare_mtp_head();
        const auto tile_tokens = mtp_draft_attention_tile_tokens(policy.draft_attention);
        if (tile_tokens) graph.prepare_mtp_head_tiled();
        const auto before = graph.memory().live_bytes;
        const uint64_t c = shape.residual.channels, h = shape.residual.branches * c;
        const uint64_t q = uint64_t(shape.query_heads) * shape.head_width;
        const uint64_t k = uint64_t(shape.kv_heads) * shape.head_width;
        const uint32_t vocab_size = draft_vocabulary ? draft_vocabulary : shape.vocabulary;
        host_logits.resize(vocab_size); host_hc.resize(h);
        hc_input = floats(h, "MTP HC input"); embedding_input = floats(c, "MTP embedding input");
        normalized_hc = floats(h, "MTP normalized HC"); normalized_embedding = floats(c, "MTP normalized embedding");
        concat = floats(h * 2, "MTP concatenated EH"); projected = floats(h, "MTP projected EH");
        intermediate = floats(h, "MTP post-attention HC"); final_hc = floats(h, "MTP final HC");
        packed = floats(q * 2, "MTP packed query gate"); raw_key = floats(k, "MTP raw key"); raw_value = floats(k, "MTP raw value");
        query = floats(q, "MTP query"); gate = floats(q, "MTP gate");
        normalized_query = floats(q, "MTP normalized query"); normalized_key = floats(k, "MTP normalized key");
        rotary_query = floats(q, "MTP rotary query"); rotary_key = floats(k, "MTP rotary key");
        keys = graph.zeros(checked_mul(uint64_t(shape.capacity) * k, 2), "MTP dense BF16 keys"); owned.push_back(keys);
        values = graph.zeros(keys.bytes, "MTP dense BF16 values"); owned.push_back(values);
        attention = floats(q, "MTP dense attention"); gated = floats(q, "MTP gated attention");
        if (tile_tokens) {
            const auto tiles = (uint64_t(shape.capacity) + tile_tokens - 1) / tile_tokens;
            const auto partial_count = checked_mul(checked_mul(shape.query_heads, tiles), uint64_t(shape.head_width) + 2);
            require(checked_mul(partial_count, 4) <= 64'000'000, "MTP draft attention workspace exceeds fixed bound");
            attention_partials = floats(partial_count, "MTP draft-only tiled attention partials");
        }
        attention_output = floats(c, "MTP attention output"); ffn_output = floats(c, "MTP FFN output");
        output_logits = floats(vocab_size, "MTP vocabulary logits");
        residual = allocate_residual_scratch(graph, shape.residual); retain(residual.views());
        moe = allocate_moe_scratch(graph, shape.experts); retain(moe.views());
        mixer = allocate_mixer_scratch(graph, shape.residual); retain(mixer.views());
        charged = graph.memory().live_bytes - before;
    }

    void encode_prefix(uint32_t absolute_position, GpuView hc_row, GpuView embedding_row, uint32_t cache_row) {
        const auto c = shape.residual.channels, branches = shape.residual.branches;
        graph.rms(hc_row, weights.hidden_gains, normalized_hc, c, policy.rms_epsilon);
        graph.rms(embedding_row, weights.embedding_gains, normalized_embedding, c, policy.rms_epsilon);
        graph.mtp_concat(normalized_embedding, normalized_hc, concat, branches, c);
        if (fast_round && weights.embedding_hidden.encoding == 8 &&
            (branches == 1 || branches == 2 || branches == 4 || branches == 8) &&
            (2 * c) % 32 == 0) {
            graph.q8_matvec_many(weights.embedding_hidden.view, concat, projected, 2 * c, c, branches);
        } else {
            for (uint32_t branch = 0; branch < branches; ++branch)
                graph.gemv(weights.embedding_hidden.encoding, weights.embedding_hidden.view,
                    graph.slice(concat, uint64_t(branch) * 2 * c * 4, uint64_t(2) * c * 4),
                    graph.slice(projected, uint64_t(branch) * c * 4, uint64_t(c) * 4), 2 * c, c);
        }
        encode_residual_read(graph, weights.attention_read, residual, projected, shape.residual, policy.rms_epsilon);
        const auto k = shape.kv_heads * shape.head_width;
        graph.gemv(weights.attention.key.encoding, weights.attention.key.view, residual.input, raw_key, c, k);
        graph.gemv(weights.attention.value.encoding, weights.attention.value.view, residual.input, raw_value, c, k);
        graph.rms(raw_key, weights.attention.key_gains, normalized_key, shape.head_width, policy.rms_epsilon);
        graph.rotary(normalized_key, weights.attention.frequencies, rotary_key, shape.kv_heads, shape.head_width,
            absolute_position, policy.rotary_pairs);
        graph.store_bf16(rotary_key, graph.slice(keys, uint64_t(cache_row) * k * 2, uint64_t(k) * 2));
        graph.store_bf16(raw_value, graph.slice(values, uint64_t(cache_row) * k * 2, uint64_t(k) * 2));
    }

    void encode_full_tail(uint32_t position, uint32_t active_kv_rows) {
        const auto q = shape.query_heads * shape.head_width;
        graph.gemv(weights.attention.query_gate.encoding, weights.attention.query_gate.view, residual.input,
            packed, shape.residual.channels, q * 2);
        graph.split_query_gate(packed, query, gate, shape.query_heads, shape.head_width, policy.query_packing);
        graph.rms(query, weights.attention.query_gains, normalized_query, shape.head_width, policy.rms_epsilon);
        graph.rotary(normalized_query, weights.attention.frequencies, rotary_query, shape.query_heads,
            shape.head_width, position, policy.rotary_pairs);
        const auto tile_tokens = mtp_draft_attention_tile_tokens(policy.draft_attention);
        if (tile_tokens)
            graph.mtp_dense_attention_tiled(rotary_query, keys, values, attention, attention_partials, active_kv_rows,
                shape.query_heads, shape.kv_heads, shape.head_width, tile_tokens);
        else
            graph.mtp_dense_attention(rotary_query, keys, values, attention, active_kv_rows,
                shape.query_heads, shape.kv_heads, shape.head_width);
        graph.multiply(attention, gate, gated, true);
        graph.gemv(weights.attention.output.encoding, weights.attention.output.view, gated,
            attention_output, q, shape.residual.channels);
        graph.residual_write(projected, attention_output, residual.scales, intermediate);
        encode_residual_read(graph, weights.ffn_read, residual, intermediate, shape.residual, policy.rms_epsilon);
        encode_moe(graph, weights.ffn, moe, residual.input, ffn_output, shape.experts,
            policy.router, policy.normalize_selected);
        graph.residual_write(intermediate, ffn_output, residual.scales, final_hc);
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        graph.set_task4_profile_tag({Task4ProfilePhase::mtp_mixer, -1, position, 1});
#endif
        encode_mixer(graph, weights.mixer, mixer, final_hc, shape.residual, policy.rms_epsilon);
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        graph.set_task4_profile_tag({Task4ProfilePhase::mtp_output, -1, position, 1});
#endif
        const auto out_vocab = draft_vocabulary ? draft_vocabulary : shape.vocabulary;
        const auto out_matrix = draft_vocabulary ? draft_output : weights.output;
        graph.gemv(out_matrix.encoding, out_matrix.view, mixer.mixed, output_logits,
            shape.residual.channels, out_vocab);
    }

    void encode_batch_prefix_rows(GpuView src_hc, GpuView hc_tail_source, bool split_first_row,
                                  GpuView src_emb, uint32_t position, uint32_t base_row, uint32_t count) {
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
#if defined(TOKSPUR_PRIVATE_MTP_PRIMING_BATCH) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
        if (priming_batch_ready && count >= 2) {
            const uint32_t k = shape.kv_heads * shape.head_width, branches = shape.residual.branches;
            for (uint32_t offset = 0; offset < count; offset += priming_slice_cap) {
                const uint32_t chunk = std::min(priming_slice_cap, count - offset);
                const auto hc_slice = (!split_first_row)
                    ? graph.slice(src_hc, uint64_t(offset) * h * 4, uint64_t(chunk) * h * 4)
                    : (offset == 0
                        ? graph.slice(src_hc, 0, uint64_t(chunk) * h * 4)
                        : graph.slice(hc_tail_source, uint64_t(offset - 1) * h * 4, uint64_t(chunk) * h * 4));
                if (chunk == 1) {
                    encode_prefix(position + offset, hc_slice,
                        graph.slice(src_emb, uint64_t(offset) * c * 4, c * 4), base_row + offset);
                    continue;
                }
                const auto emb_slice = graph.slice(src_emb, uint64_t(offset) * c * 4, uint64_t(chunk) * c * 4);
                const auto norm_hc_slice = graph.slice(priming_norm_hc, 0, uint64_t(chunk) * h * 4);
                const auto norm_emb_slice = graph.slice(priming_norm_emb, 0, uint64_t(chunk) * c * 4);
                const auto concat_slice = graph.slice(priming_concat, 0, uint64_t(chunk) * 2 * h * 4);
                const auto proj_slice = graph.slice(priming_projected, 0, uint64_t(chunk) * h * 4);
                const auto raw_k_slice = graph.slice(priming_raw_key, 0, uint64_t(chunk) * k * 4);
                const auto raw_v_slice = graph.slice(priming_raw_value, 0, uint64_t(chunk) * k * 4);
                const auto norm_k_slice = graph.slice(priming_norm_key, 0, uint64_t(chunk) * k * 4);
                graph.rms_rows({hc_slice, chunk, uint32_t(h), h * 4}, weights.hidden_gains,
                    {norm_hc_slice, chunk, uint32_t(h), h * 4}, uint32_t(c), policy.rms_epsilon, false);
                graph.rms_rows({emb_slice, chunk, uint32_t(c), c * 4}, weights.embedding_gains,
                    {norm_emb_slice, chunk, uint32_t(c), c * 4}, uint32_t(c), policy.rms_epsilon, true);
                graph.mtp_concat_rows(norm_emb_slice, norm_hc_slice, concat_slice, branches, uint32_t(c), chunk);
                graph.dense_mm_q8(graph.prefill_weight_layout(weights.embedding_hidden.view), weights.embedding_hidden.view,
                    {concat_slice, chunk * branches, uint32_t(2 * c), 2 * c * 4},
                    {proj_slice, chunk * branches, uint32_t(c), c * 4});
                graph.set_mtp_priming_scope(true);
                try {
                    encode_prefill_residual(graph, weights.attention_read, priming_residual,
                        {proj_slice, chunk, uint32_t(h), h * 4}, shape.residual, policy.rms_epsilon);
                    graph.set_mtp_priming_scope(false);
                } catch (...) {
                    graph.set_mtp_priming_scope(false);
                    throw;
                }
                const auto res_in = prefill_residual_input(priming_residual, shape.residual, chunk);
                graph.dense_mm_q8(graph.prefill_weight_layout(weights.attention.key.view), weights.attention.key.view,
                    res_in, {raw_k_slice, chunk, k, uint64_t(k) * 4});
                graph.dense_mm_q8(graph.prefill_weight_layout(weights.attention.value.view), weights.attention.value.view,
                    res_in, {raw_v_slice, chunk, k, uint64_t(k) * 4}, graph.can_reuse_nax_dense_bf16(res_in));
                graph.rms_rows({raw_k_slice, chunk, k, uint64_t(k) * 4}, weights.attention.key_gains,
                    {norm_k_slice, chunk, k, uint64_t(k) * 4}, shape.head_width, policy.rms_epsilon, true);
                graph.mtp_rope_store_kv_bf16(norm_k_slice, raw_v_slice, weights.attention.frequencies,
                    graph.slice(keys, uint64_t(base_row + offset) * k * 2, uint64_t(chunk) * k * 2),
                    graph.slice(values, uint64_t(base_row + offset) * k * 2, uint64_t(chunk) * k * 2),
                    shape.kv_heads, shape.head_width, position + offset, policy.rotary_pairs, chunk);
            }
            return;
        }
#endif
        for (uint32_t i = 0; i < count; ++i) {
            const auto hc_row = (!split_first_row)
                ? graph.slice(src_hc, uint64_t(i) * h * 4, h * 4)
                : (i == 0
                    ? graph.slice(src_hc, 0, h * 4)
                    : graph.slice(hc_tail_source, uint64_t(i - 1) * h * 4, h * 4));
            encode_prefix(position + i, hc_row,
                graph.slice(src_emb, uint64_t(i) * c * 4, c * 4), base_row + i);
        }
    }

    BatchResult flush_deferred() {
        require(!busy && valid, "MTP deferred priming flush on invalid or busy head");
        if (!pending_prime_count) return {};
        require(pending_prime_pos == first + rows, "MTP deferred priming frontier drift");
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
        static_cast<void>(graph.last_profile());
        const auto count = pending_prime_count;
        TOKSPUR_OPERATOR_CONTEXT(graph, "mtp_prime_batch", -1, pending_prime_pos, count);
        busy = true;
        try {
            graph.begin();
            encode_batch_prefix_rows(deferred_hc, {}, false, deferred_emb, pending_prime_pos, rows, count);
            const auto batch = graph.submit();
            rows += count; pending_prime_count = 0; pending_prime_pos = 0;
            logits_valid = hidden_valid = hidden_gpu_valid = false; busy = false;
            return batch;
        } catch (...) {
            try { graph.abort(); } catch (...) {}
            busy = false; valid = false; logits_valid = hidden_valid = hidden_gpu_valid = false; throw;
        }
    }

    void stage_deferred_prime(const GpuView* hc_gpu, std::span<const float> hc_host,
                              std::span<const float> emb_rows, uint32_t position) {
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        require(!busy && valid && fast_round && emb_rows.size() % c == 0, "MTP deferred priming requires idle fast-round head");
        const auto count = uint32_t(emb_rows.size() / c);
        if (!count) return;
        require(count <= 4 && position == first + logical_rows() && logical_rows() + count <= shape.capacity &&
            position + count <= 65536, "MTP deferred priming input/frontier ownership mismatch");
        if (hc_gpu) {
            require(hc_gpu->owner && hc_gpu->bytes == checked_mul(checked_mul(count, h), 4), "MTP deferred GPU HC span mismatch");
        } else {
            require(hc_host.size() == checked_mul(count, h), "MTP deferred host HC span mismatch");
            finite(hc_host);
        }
        finite(emb_rows);
        if (pending_prime_count && pending_prime_count + count > 4) {
            static_cast<void>(flush_deferred());
        }
        const uint32_t dst_off = pending_prime_count;
        const auto dst_hc = graph.slice(deferred_hc, uint64_t(dst_off) * h * 4, checked_mul(checked_mul(count, h), 4));
        const auto dst_emb = graph.slice(deferred_emb, uint64_t(dst_off) * c * 4, checked_mul(checked_mul(count, c), 4));
        if (hc_gpu) {
            const GraphCopy copy_op{*hc_gpu, dst_hc};
            graph.copy(std::span(&copy_op, 1));
        } else {
            graph.write(dst_hc, std::as_bytes(hc_host));
        }
        graph.write(dst_emb, std::as_bytes(emb_rows));
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        priming_prepare_locked = true;
#endif
        if (!pending_prime_count) pending_prime_pos = position;
        pending_prime_count += count;
        logits_valid = hidden_valid = hidden_gpu_valid = false;
    }

    MtpHeadFastResult run_fast(const GpuView* hc_gpu, std::span<const float> hc_host, bool chain_from_final_hc,
                               std::span<const float> embedding, uint32_t position) {
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        require(!busy && valid && fast_round && position == first + logical_rows() && position < 65536 &&
            logical_rows() < shape.capacity && embedding.size() == c, "MTP fast-round input/prefix ownership mismatch");
        GpuView hc_source{};
        if (chain_from_final_hc) {
            require(hidden_gpu_valid && !pending_prime_count, "MTP fast chain requires valid GPU hidden state");
            hc_source = final_hc;
        } else if (hc_gpu) {
            require(hc_gpu->owner && hc_gpu->bytes == h * 4, "MTP fast advance GPU HC span mismatch");
            hc_source = *hc_gpu;
        } else {
            require(hc_host.size() == h, "MTP fast advance host HC span mismatch");
            finite(hc_host);
        }
        finite(embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        priming_prepare_locked = true;
#endif
        TOKSPUR_OPERATOR_CONTEXT(graph, "mtp_advance", -1, position, 1);
        if (!chain_from_final_hc && !hc_gpu) {
            graph.write(hc_input, std::as_bytes(hc_host));
            hc_source = hc_input;
        }
        graph.write(embedding_input, std::as_bytes(embedding));
        busy = true;
        try {
            const uint32_t primed = pending_prime_count;
            require(!primed || pending_prime_pos == first + rows, "MTP deferred priming frontier drift");
            graph.begin();
            if (primed) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                graph.set_task4_profile_tag({Task4ProfilePhase::mtp_prime, -1, pending_prime_pos, primed});
#endif
                encode_batch_prefix_rows(deferred_hc, {}, false, deferred_emb, pending_prime_pos, rows, primed);
            }
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            graph.set_task4_profile_tag({Task4ProfilePhase::mtp_advance, -1, position, 1});
#endif
            const uint32_t target_row = rows + primed;
            encode_prefix(position, hc_source, embedding_input, target_row);
            encode_full_tail(position, target_row + 1);
            const auto out_vocab = draft_vocabulary ? draft_vocabulary : shape.vocabulary;
            graph.logit_topk(
                GpuRowsF32{graph.slice(output_logits, 0, uint64_t(out_vocab) * sizeof(float)),
                           1, out_vocab, uint64_t(out_vocab) * sizeof(float)},
                topk_scratch, topk_row);
            const auto batch = graph.submit();
            GpuLogitTopKRow topk{};
            graph.read(topk_row, std::as_writable_bytes(std::span(&topk, 1)));
            require(topk.vocabulary == out_vocab && topk.first_nonfinite_col == 0xffffffffu, "nonfinite MTP input/output");
            rows += primed + 1;
            pending_prime_count = 0; pending_prime_pos = 0;
            logits_valid = false; hidden_valid = false; hidden_gpu_valid = true; busy = false;
            return {topk, batch, position, rows};
        } catch (...) {
            try { graph.abort(); } catch (...) {}
            busy = false; valid = false; logits_valid = hidden_valid = hidden_gpu_valid = false; throw;
        }
    }

    MtpHeadFastChainResult run_fast_greedy_chain(const GpuView* hc_gpu, std::span<const float> hc_host,
                                                 std::span<const float> anchor_embedding,
                                                 uint32_t position, uint32_t depth) {
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        require(!busy && valid && fast_round && greedy_chain_ready && depth >= 2 && depth <= 3 &&
            position == first + logical_rows() && position <= 65536 - depth &&
            logical_rows() <= shape.capacity - depth && anchor_embedding.size() == c,
            "MTP fast greedy chain input/prefix ownership mismatch");
        GpuView hc_source{};
        if (hc_gpu) {
            require(hc_gpu->owner && hc_gpu->bytes == h * 4, "MTP fast greedy chain GPU HC span mismatch");
            hc_source = *hc_gpu;
        } else {
            require(hc_host.size() == h, "MTP fast greedy chain host HC span mismatch");
            finite(hc_host);
        }
        finite(anchor_embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        priming_prepare_locked = true;
#endif
        TOKSPUR_OPERATOR_CONTEXT(graph, "mtp_advance", -1, position, depth);
        if (!hc_gpu) {
            graph.write(hc_input, std::as_bytes(hc_host));
            hc_source = hc_input;
        }
        graph.write(embedding_input, std::as_bytes(anchor_embedding));
        busy = true;
        try {
            const uint32_t primed = pending_prime_count;
            require(!primed || pending_prime_pos == first + rows, "MTP deferred priming frontier drift");
            const auto out_vocab = draft_vocabulary ? draft_vocabulary : shape.vocabulary;
            graph.begin();
            if (primed) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                graph.set_task4_profile_tag({Task4ProfilePhase::mtp_prime, -1, pending_prime_pos, primed});
#endif
                encode_batch_prefix_rows(deferred_hc, {}, false, deferred_emb, pending_prime_pos, rows, primed);
            }
            const uint32_t target_row = rows + primed;
            for (uint32_t s = 0; s < depth; ++s) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                graph.set_task4_profile_tag({Task4ProfilePhase::mtp_advance, -1, position + s, 1});
#endif
                if (s > 0) {
                    require(fast_token_embd_vocab >= out_vocab, "MTP greedy chain token embedding table smaller than draft vocabulary");
                    const auto prev_row = graph.slice(topk_chain_rows, uint64_t(s - 1) * sizeof(GpuLogitTopKRow), sizeof(GpuLogitTopKRow));
                    graph.mtp_embed_from_topk_q8(fast_token_embd, prev_row, embedding_input, uint32_t(c), fast_token_embd_vocab);
                }
                const GpuView step_hc = (s == 0) ? hc_source : final_hc;
                encode_prefix(position + s, step_hc, embedding_input, target_row + s);
                encode_full_tail(position + s, target_row + s + 1);
                const auto dst_row = graph.slice(topk_chain_rows, uint64_t(s) * sizeof(GpuLogitTopKRow), sizeof(GpuLogitTopKRow));
                graph.logit_topk(
                    GpuRowsF32{graph.slice(output_logits, 0, uint64_t(out_vocab) * sizeof(float)),
                               1, out_vocab, uint64_t(out_vocab) * sizeof(float)},
                    topk_scratch, dst_row);
            }
            const auto batch = graph.submit();
            std::array<GpuLogitTopKRow, 3> topks{};
            graph.read(graph.slice(topk_chain_rows, 0, uint64_t(depth) * sizeof(GpuLogitTopKRow)),
                       std::as_writable_bytes(std::span(topks).first(depth)));
            for (uint32_t s = 0; s < depth; ++s) {
                require(topks[s].vocabulary == out_vocab && topks[s].first_nonfinite_col == 0xffffffffu,
                    "nonfinite MTP input/output");
            }
            rows += primed + depth;
            pending_prime_count = 0; pending_prime_pos = 0;
            logits_valid = false; hidden_valid = false; hidden_gpu_valid = true; busy = false;
            return {topks, depth, batch, position, rows};
        } catch (...) {
            try { graph.abort(); } catch (...) {}
            busy = false; valid = false; logits_valid = hidden_valid = hidden_gpu_valid = false; throw;
        }
    }

    MtpHeadSampledChainResult run_fast_sampled_chain(const GpuView* hc_gpu, std::span<const float> hc_host,
                                                     std::span<const float> anchor_embedding,
                                                     uint32_t position, uint32_t depth,
                                                     SamplingPolicy draft_policy,
                                                     std::span<const uint64_t, 3> tickets) {
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        const float inv_temp = (draft_policy.temperature > 0.0 && std::isfinite(draft_policy.temperature))
            ? float(1.0 / draft_policy.temperature) : 0.0f;
        require(!busy && valid && fast_round && greedy_chain_ready && depth >= 2 && depth <= 3 &&
            position == first + logical_rows() && position <= 65536 - depth &&
            logical_rows() <= shape.capacity - depth && anchor_embedding.size() == c &&
            std::isfinite(inv_temp) && inv_temp > 0.0f &&
            std::isfinite(draft_policy.top_p) && draft_policy.top_p > 0.0 && draft_policy.top_p <= 1.0 &&
            std::isfinite(draft_policy.min_p) && draft_policy.min_p >= 0.0 && draft_policy.min_p <= 1.0 &&
            draft_policy.top_k <= maximum_sampling_vocabulary &&
            tickets[0] < (1ull << 53) && tickets[1] < (1ull << 53) && tickets[2] < (1ull << 53),
            "MTP fast sampled chain input/prefix ownership mismatch");
        GpuView hc_source{};
        if (hc_gpu) {
            require(hc_gpu->owner && hc_gpu->bytes == h * 4, "MTP fast sampled chain GPU HC span mismatch");
            hc_source = *hc_gpu;
        } else {
            require(hc_host.size() == h, "MTP fast sampled chain host HC span mismatch");
            finite(hc_host);
        }
        finite(anchor_embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        priming_prepare_locked = true;
#endif
        TOKSPUR_OPERATOR_CONTEXT(graph, "mtp_advance", -1, position, depth);
        if (!hc_gpu) {
            graph.write(hc_input, std::as_bytes(hc_host));
            hc_source = hc_input;
        }
        graph.write(embedding_input, std::as_bytes(anchor_embedding));
        struct TicketUpload {
            uint32_t cursor;
            std::array<uint32_t, 3> tickets24;
            std::array<uint32_t, 4> reserved;
        } tbuf{0u, {uint32_t(tickets[0] >> 29u), uint32_t(tickets[1] >> 29u), uint32_t(tickets[2] >> 29u)}, {0u, 0u, 0u, 0u}};
        static_assert(sizeof(TicketUpload) == 32);
        graph.write(sampled_tickets_gpu, std::as_bytes(std::span(&tbuf, 1)));
        const uint32_t max_k = draft_policy.top_k ? std::min<uint32_t>(draft_policy.top_k, 20u) : 20u;
        const float min_p_f = float(draft_policy.min_p);
        const float top_p_f = float(draft_policy.top_p);
        busy = true;
        try {
            const uint32_t primed = pending_prime_count;
            require(!primed || pending_prime_pos == first + rows, "MTP deferred priming frontier drift");
            const auto out_vocab = draft_vocabulary ? draft_vocabulary : shape.vocabulary;
            require(fast_token_embd_vocab >= out_vocab, "MTP sampled chain token embedding table smaller than draft vocabulary");
            graph.begin();
            if (primed) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                graph.set_task4_profile_tag({Task4ProfilePhase::mtp_prime, -1, pending_prime_pos, primed});
#endif
                encode_batch_prefix_rows(deferred_hc, {}, false, deferred_emb, pending_prime_pos, rows, primed);
            }
            const uint32_t target_row = rows + primed;
            for (uint32_t s = 0; s < depth; ++s) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                graph.set_task4_profile_tag({Task4ProfilePhase::mtp_advance, -1, position + s, 1});
#endif
                const GpuView step_hc = (s == 0) ? hc_source : final_hc;
                encode_prefix(position + s, step_hc, embedding_input, target_row + s);
                encode_full_tail(position + s, target_row + s + 1);
                const auto dst_row = graph.slice(topk_chain_rows, uint64_t(s) * sizeof(GpuLogitTopKRow), sizeof(GpuLogitTopKRow));
                graph.logit_topk(
                    GpuRowsF32{graph.slice(output_logits, 0, uint64_t(out_vocab) * sizeof(float)),
                               1, out_vocab, uint64_t(out_vocab) * sizeof(float)},
                    topk_scratch, dst_row);
                const auto dst_law = graph.slice(sampled_chain_laws, uint64_t(s) * sizeof(GpuMtpSampledLawRow), sizeof(GpuMtpSampledLawRow));
                graph.mtp_sample_and_embed_from_topk_q8(
                    fast_token_embd, dst_row, sampled_tickets_gpu, dst_law, embedding_input,
                    uint32_t(c), fast_token_embd_vocab, max_k, (s + 1 < depth),
                    inv_temp, min_p_f, top_p_f);
            }
            const auto batch = graph.submit();
            std::array<GpuLogitTopKRow, 3> topks{};
            std::array<GpuMtpSampledLawRow, 3> laws{};
            graph.read(graph.slice(topk_chain_rows, 0, uint64_t(depth) * sizeof(GpuLogitTopKRow)),
                       std::as_writable_bytes(std::span(topks).first(depth)));
            graph.read(graph.slice(sampled_chain_laws, 0, uint64_t(depth) * sizeof(GpuMtpSampledLawRow)),
                       std::as_writable_bytes(std::span(laws).first(depth)));
            for (uint32_t s = 0; s < depth; ++s) {
                require(topks[s].vocabulary == out_vocab && topks[s].first_nonfinite_col == 0xffffffffu &&
                    laws[s].count >= 1 && laws[s].count <= 20,
                    "nonfinite MTP input/output");
            }
            rows += primed + depth;
            pending_prime_count = 0; pending_prime_pos = 0;
            logits_valid = false; hidden_valid = false; hidden_gpu_valid = true; busy = false;
            return {topks, laws, depth, batch, position, rows};
        } catch (...) {
            try { graph.abort(); } catch (...) {}
            busy = false; valid = false; logits_valid = hidden_valid = hidden_gpu_valid = false; throw;
        }
    }

    BatchResult run_gpu(GpuView hc_view, std::span<const float> embedding, uint32_t position, bool full) {
        if (pending_prime_count) static_cast<void>(flush_deferred());
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels;
        require(!busy && valid && position == first + rows && position < 65536 && rows < shape.capacity &&
            hc_view.owner && hc_view.bytes == h * 4 &&
            embedding.size() == shape.residual.channels, "MTP head input/prefix ownership mismatch");
        finite(embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        priming_prepare_locked = true;
#endif
        TOKSPUR_OPERATOR_CONTEXT(graph,full ? "mtp_advance" : "mtp_prime",-1,position,1);
        graph.write(embedding_input, std::as_bytes(embedding));
        busy = true;
        try {
            graph.begin();
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            graph.set_task4_profile_tag({full ? Task4ProfilePhase::mtp_advance : Task4ProfilePhase::mtp_prime, -1, position, 1});
#endif
            encode_prefix(position, hc_view, embedding_input, rows);
            if (full) encode_full_tail(position, rows + 1);
            const auto batch = graph.submit();
            if (full) {
                const auto out_vocab = draft_vocabulary ? draft_vocabulary : shape.vocabulary;
                graph.read(output_logits, std::as_writable_bytes(std::span(host_logits).first(out_vocab)));
                finite(std::span(host_logits).first(out_vocab));
                if (!fast_round) {
                    graph.read(final_hc, std::as_writable_bytes(std::span(host_hc)));
                    finite(host_hc);
                }
            }
            // No fallible work follows publication. Cache-only calls deliberately
            // invalidate old outputs instead of presenting them as this prefix.
            ++rows; logits_valid = full; hidden_valid = full && !fast_round; hidden_gpu_valid = full; busy = false;
            return batch;
        } catch (...) {
            try { graph.abort(); } catch (...) {}
            busy = false; valid = false; logits_valid = hidden_valid = hidden_gpu_valid = false; throw;
        }
    }

    BatchResult run(std::span<const float> hc, std::span<const float> embedding, uint32_t position, bool full) {
        if (pending_prime_count) static_cast<void>(flush_deferred());
        require(!busy && valid && hc.size() == uint64_t(shape.residual.branches) * shape.residual.channels,
            "MTP head input/prefix ownership mismatch");
        finite(hc);
        graph.write(hc_input, std::as_bytes(hc));
        return run_gpu(hc_input, embedding, position, full);
    }
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
    void prepare_batch(uint32_t capacity, uint64_t budget) {
        const uint32_t max_cap =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            (graph.prefill_dense_nax() || graph.prefill_grouped_nax() || capacity <= 2048u) ? 2048u : 512u;
#else
            capacity <= 2048u ? 2048u : 512u;
#endif
        require(!busy && valid && !rows && !priming_capacity && !priming_prepare_locked && capacity && capacity <= max_cap &&
            capacity <= shape.capacity && budget, "MTP priming batch requires idle startup, bounded capacity and explicit GPU budget");
        static_cast<void>(graph.last_profile());
        const uint64_t c = shape.residual.channels, h = uint64_t(shape.residual.branches) * c;
        const uint64_t k = uint64_t(shape.kv_heads) * shape.head_width;
        const auto hc_bytes = checked_mul(checked_mul(h, capacity), 4);
        const auto embedding_bytes = checked_mul(checked_mul(c, capacity), 4);
        const auto base_bytes = checked_add(hc_bytes, embedding_bytes);
        require(base_bytes <= budget, "MTP priming rows exceed explicit GPU scratch budget");
        bool want_batch_nax = false;
        uint32_t slice_cap = 0;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        const uint64_t remaining = budget - base_bytes;
        if (graph.prefill_dense_nax() && capacity >= 2 && shape.residual.branches == 4 &&
            weights.embedding_hidden.encoding == 8 && weights.attention.key.encoding == 8 &&
            weights.attention.value.encoding == 8 && weights.attention_read.up.encoding == 8 &&
            (2 * c) % 64 == 0 && c % 64 == 0 && shape.residual.rank % 64 == 0 &&
            remaining >= (36ull << 20)) {
            slice_cap = std::min(capacity, 128u);
            want_batch_nax = true;
            graph.prepare_gdn_chunk();
            prepare_prefill_residual(graph, weights.attention_read, shape.residual);
            graph.prepare_dense_mm_q8(graph.prefill_weight_layout(weights.embedding_hidden.view));
            graph.prepare_dense_mm_q8(graph.prefill_weight_layout(weights.attention.key.view));
            graph.prepare_dense_mm_q8(graph.prefill_weight_layout(weights.attention.value.view));
        }
#endif
        owned.reserve(owned.size() + (want_batch_nax ? 16 : 2));
        const auto before = graph.memory().live_bytes;
        GpuView hc{}, embedding{};
        std::array<GpuView, 7> extra{};
        PrefillResidualScratch res{};
        try {
            hc = graph.zeros(hc_bytes, "MTP priming HC rows");
            embedding = graph.zeros(embedding_bytes, "MTP priming embedding rows");
            if (want_batch_nax) {
                extra[0] = graph.zeros(checked_mul(checked_mul(h, slice_cap), 4), "MTP priming normalized HC");
                extra[1] = graph.zeros(checked_mul(checked_mul(c, slice_cap), 4), "MTP priming normalized embedding");
                extra[2] = graph.zeros(checked_mul(checked_mul(2 * h, slice_cap), 4), "MTP priming concat EH");
                extra[3] = graph.zeros(checked_mul(checked_mul(h, slice_cap), 4), "MTP priming projected EH");
                extra[4] = graph.zeros(checked_mul(checked_mul(k, slice_cap), 4), "MTP priming raw key");
                extra[5] = graph.zeros(checked_mul(checked_mul(k, slice_cap), 4), "MTP priming raw value");
                extra[6] = graph.zeros(checked_mul(checked_mul(k, slice_cap), 4), "MTP priming normalized key");
                res = allocate_prefill_residual_scratch(graph, shape.residual, slice_cap);
            }
            const auto actual = graph.memory().live_bytes - before;
            require(actual <= budget, "MTP priming allocated GPU scratch exceeds budget");
            const auto updated = checked_add(charged, actual);
            owned.push_back(hc); owned.push_back(embedding);
            if (want_batch_nax) {
                for (auto v : extra) owned.push_back(v);
                retain(res.rows.views());
                priming_norm_hc = extra[0]; priming_norm_emb = extra[1]; priming_concat = extra[2];
                priming_projected = extra[3]; priming_raw_key = extra[4]; priming_raw_value = extra[5];
                priming_norm_key = extra[6]; priming_residual = res;
                priming_slice_cap = slice_cap; priming_batch_ready = true;
            }
            priming_hc = hc; priming_embeddings = embedding; priming_capacity = capacity; priming_bytes = actual;
            charged = updated;
        } catch (...) {
            for (auto v : res.rows.views()) if (v.owner) graph.release(v);
            for (auto v : extra) if (v.owner) graph.release(v);
            if (embedding.owner) graph.release(embedding);
            if (hc.owner) graph.release(hc);
            throw;
        }
    }
    BatchResult encode_and_submit_batch(GpuView hc_first_chunk, GpuView hc_tail_source, bool split_first_row,
                                        uint32_t count, uint32_t position) {
        priming_prepare_locked = true; busy = true;
        try {
            graph.begin();
            encode_batch_prefix_rows(hc_first_chunk, hc_tail_source, split_first_row, priming_embeddings, position, rows, count);
            const auto batch = graph.submit();
            // All input and GPU finite/error validation completed. No prefix, output-validity or cache-row publication occurs inside the loop.
            rows += count; logits_valid = hidden_valid = hidden_gpu_valid = false; busy = false;
            return batch;
        } catch (...) {
            try { graph.abort(); } catch (...) {}
            busy = false; valid = false; logits_valid = hidden_valid = hidden_gpu_valid = false; throw;
        }
    }
    BatchResult run_batch_gpu(GpuView hc_rows, std::span<const float> embedding, uint32_t position) {
        if (pending_prime_count) static_cast<void>(flush_deferred());
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        const uint64_t row_bytes = h * 4;
        require(!busy && valid && priming_capacity && position == first + rows && position < 65536 &&
            hc_rows.owner && hc_rows.bytes >= row_bytes && hc_rows.bytes % row_bytes == 0,
            "MTP priming batch GPU input/frontier ownership mismatch");
        const auto count = uint32_t(hc_rows.bytes / row_bytes);
        require(count <= priming_capacity && count <= shape.capacity - rows && count <= 65536 - position &&
            embedding.size() == checked_mul(count, c),
            "MTP priming batch input/frontier ownership mismatch");
        finite(embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
        static_cast<void>(graph.last_profile());
        TOKSPUR_OPERATOR_CONTEXT(graph,"mtp_prime_batch",-1,position,count);
        graph.write(graph.slice(priming_embeddings, 0, uint64_t(embedding.size()) * 4), std::as_bytes(embedding));
        return encode_and_submit_batch(hc_rows, {}, false, count, position);
    }
    BatchResult run_batch_with_prefix(std::span<const float> first_hc, GpuView tail_hc, std::span<const float> embedding, uint32_t position) {
        if (pending_prime_count) static_cast<void>(flush_deferred());
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        const uint64_t row_bytes = h * 4;
        require(!busy && valid && priming_capacity && position == first + rows && position < 65536 &&
            first_hc.size() == h && (tail_hc.bytes == 0 || (tail_hc.owner && tail_hc.bytes % row_bytes == 0)),
            "MTP priming batch prefix/tail input mismatch");
        const uint32_t tail_count = uint32_t(tail_hc.bytes / row_bytes);
        const uint32_t count = 1u + tail_count;
        require(count <= priming_capacity && count <= shape.capacity - rows && count <= 65536 - position &&
            embedding.size() == checked_mul(count, c),
            "MTP priming batch input/frontier ownership mismatch");
        finite(first_hc); finite(embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
        static_cast<void>(graph.last_profile());
        TOKSPUR_OPERATOR_CONTEXT(graph,"mtp_prime_batch",-1,position,count);
        graph.write(graph.slice(priming_hc, 0, row_bytes), std::as_bytes(first_hc));
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (priming_batch_ready && count >= 2) {
            const uint32_t head_tail = std::min(tail_count, priming_slice_cap - 1u);
            if (head_tail) {
                const GraphCopy copy{graph.slice(tail_hc, 0, uint64_t(head_tail) * row_bytes),
                                     graph.slice(priming_hc, row_bytes, uint64_t(head_tail) * row_bytes)};
                graph.copy(std::span(&copy, 1));
            }
        }
#endif
        graph.write(graph.slice(priming_embeddings, 0, uint64_t(embedding.size()) * 4), std::as_bytes(embedding));
        return encode_and_submit_batch(priming_hc, tail_hc, true, count, position);
    }
    BatchResult run_batch(std::span<const float> hc, std::span<const float> embedding, uint32_t position) {
        if (pending_prime_count) static_cast<void>(flush_deferred());
        const uint64_t h = uint64_t(shape.residual.branches) * shape.residual.channels, c = shape.residual.channels;
        require(!busy && valid && priming_capacity && position == first + rows && position < 65536 && hc.size() % h == 0 &&
            hc.size() / h && hc.size() / h <= priming_capacity && hc.size() / h <= shape.capacity - rows &&
            hc.size() / h <= 65536 - position && embedding.size() == checked_mul(hc.size() / h, c),
            "MTP priming batch input/frontier ownership mismatch");
        finite(hc); finite(embedding);
        if (sidecar) { sidecar->assert_unchanged(); target->assert_unchanged(); }
        static_cast<void>(graph.last_profile());
        const auto count = uint32_t(hc.size() / h);
        TOKSPUR_OPERATOR_CONTEXT(graph,"mtp_prime_batch",-1,position,count);
        const auto hc_dst = graph.slice(priming_hc, 0, uint64_t(hc.size()) * 4);
        graph.write(hc_dst, std::as_bytes(hc));
        graph.write(graph.slice(priming_embeddings, 0, uint64_t(embedding.size()) * 4), std::as_bytes(embedding));
        return encode_and_submit_batch(hc_dst, {}, false, count, position);
    }
#endif
};

MtpHead::MtpHead(MetalGraph& graph, const MtpSharedBinding& binding, GpuMatrix output, uint32_t capacity,
                 MtpDraftAttention draft_attention, const std::map<std::string, GpuView>* pack_views)
    : impl_(std::make_unique<Impl>(graph)) {
    impl_->policy.draft_attention = draft_attention; impl_->import(binding, output, capacity, pack_views); impl_->prepare();
}
MtpHead::MtpHead(MetalGraph& graph, MtpHeadWeights weights, MtpHeadShape shape, MtpHeadPolicy policy)
    : impl_(std::make_unique<Impl>(graph)) {
    impl_->weights = weights; impl_->shape = shape; impl_->policy = policy; impl_->prepare();
}
MtpHead::~MtpHead() = default;
void MtpHead::reset(uint32_t first) {
    auto& p = *impl_; require(!p.busy && first < 65536, "invalid MTP reset position");
    p.first = first; p.rows = 0; p.pending_prime_count = 0; p.pending_prime_pos = 0;
    p.valid = true; p.logits_valid = p.hidden_valid = p.hidden_gpu_valid = false;
}
void MtpHead::truncate(uint32_t target_rows) {
    auto& p = *impl_; require(!p.busy && p.valid && target_rows <= p.logical_rows(), "invalid MTP cache truncation");
    if (target_rows <= p.rows) {
        p.pending_prime_count = 0; p.pending_prime_pos = 0;
        p.rows = target_rows;
    } else {
        p.pending_prime_count = target_rows - p.rows;
    }
    p.logits_valid = p.hidden_valid = p.hidden_gpu_valid = false;
}
uint32_t MtpHead::cache_rows() const { return impl_->logical_rows(); }
uint32_t MtpHead::next_absolute_position() const { return impl_->first + impl_->logical_rows(); }
MtpHeadResult MtpHead::advance(std::span<const float> hc, std::span<const float> embedding, uint32_t position) {
    auto& p = *impl_; const auto batch = p.run(hc, embedding, position, true);
    const auto vocab = p.draft_vocabulary ? p.draft_vocabulary : p.shape.vocabulary;
    return {std::span(p.host_logits).first(vocab), batch, position, p.rows};
}
MtpHeadResult MtpHead::advance(GpuView hc, std::span<const float> embedding, uint32_t position) {
    auto& p = *impl_; const auto batch = p.run_gpu(hc, embedding, position, true);
    const auto vocab = p.draft_vocabulary ? p.draft_vocabulary : p.shape.vocabulary;
    return {std::span(p.host_logits).first(vocab), batch, position, p.rows};
}
BatchResult MtpHead::prime_pair(std::span<const float> hc, std::span<const float> embedding, uint32_t position) {
    return impl_->run(hc, embedding, position, false);
}
BatchResult MtpHead::prime_pair(GpuView hc, std::span<const float> embedding, uint32_t position) {
    return impl_->run_gpu(hc, embedding, position, false);
}
void MtpHead::prime(std::span<const float> hc, std::span<const float> embedding, uint32_t first) {
    const auto& s = impl_->shape; const uint64_t width = uint64_t(s.residual.branches) * s.residual.channels;
    require(hc.size() % width == 0 && embedding.size() == hc.size() / s.residual.branches &&
        hc.size() / width <= s.capacity && first < 65536 && hc.size() / width <= 65536 - first,
        "MTP prime pair geometry mismatch");
    finite(hc); finite(embedding); reset(first);
    for (uint32_t i = 0; i < hc.size() / width; ++i)
        static_cast<void>(prime_pair(hc.subspan(uint64_t(i) * width, width),
            embedding.subspan(uint64_t(i) * s.residual.channels, s.residual.channels), first + i));
}
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
void MtpHead::prepare_priming_batch(uint32_t capacity, uint64_t budget) { impl_->prepare_batch(capacity, budget); }
BatchResult MtpHead::prime_batch(std::span<const float> hc, std::span<const float> embedding, uint32_t position) {
    return impl_->run_batch(hc, embedding, position);
}
BatchResult MtpHead::prime_batch(GpuView hc, std::span<const float> embedding, uint32_t position) {
    return impl_->run_batch_gpu(hc, embedding, position);
}
BatchResult MtpHead::prime_batch_with_prefix(std::span<const float> first_hc, GpuView tail_hc, std::span<const float> embedding, uint32_t position) {
    return impl_->run_batch_with_prefix(first_hc, tail_hc, embedding, position);
}
uint32_t MtpHead::priming_batch_capacity() const { return impl_->priming_capacity; }
uint64_t MtpHead::priming_batch_bytes() const { return impl_->priming_bytes; }
uint64_t MtpHead::priming_cache_prefix_bytes() const {
    const auto& p = *impl_; require(!p.busy && p.valid, "MTP cache prefix is invalid or busy");
    static_cast<void>(p.graph.last_profile());
    return checked_mul(checked_mul(uint64_t(p.logical_rows()), uint64_t(p.shape.kv_heads) * p.shape.head_width), 4);
}
void MtpHead::read_priming_cache_prefix(std::span<std::byte> output) const {
    auto& p = *impl_;
    if (p.pending_prime_count) static_cast<void>(p.flush_deferred());
    const auto bytes = priming_cache_prefix_bytes(); require(output.size() == bytes, "MTP cache prefix readback extent");
    if (!bytes) return;
    const auto half = bytes / 2;
    p.graph.read(p.graph.slice(p.keys, 0, half), output.first(half));
    p.graph.read(p.graph.slice(p.values, 0, half), output.subspan(half));
}
#endif
std::span<const float> MtpHead::logits() const {
    require(impl_->valid && impl_->logits_valid, "MTP logits are unavailable at this cache prefix");
    const auto vocab = impl_->draft_vocabulary ? impl_->draft_vocabulary : impl_->shape.vocabulary;
    return std::span(impl_->host_logits).first(vocab);
}
std::span<const float> MtpHead::hidden() const {
    auto& p = *impl_;
    require(p.valid && (p.hidden_valid || (p.hidden_gpu_valid && !p.pending_prime_count)),
        "MTP hidden output unavailable at this cache prefix");
    if (!p.hidden_valid) {
        p.graph.read(p.final_hc, std::as_writable_bytes(std::span(p.host_hc)));
        finite(p.host_hc);
        p.hidden_valid = true;
    }
    return p.host_hc;
}
GpuView MtpHead::hidden_gpu() const {
    const auto& p = *impl_;
    require(p.valid && (p.hidden_valid || (p.hidden_gpu_valid && !p.pending_prime_count)),
        "MTP hidden output unavailable at this cache prefix");
    return p.final_hc;
}
uint64_t MtpHead::workspace_bytes() const { return impl_->charged; }
MtpDraftAttention MtpHead::draft_attention() const { return impl_->policy.draft_attention; }
void MtpHead::configure_draft_vocabulary(uint32_t draft_vocab, GpuMatrix draft_out) {
    auto& p = *impl_;
    require(!p.busy && p.valid && !p.logical_rows(), "MTP draft vocabulary configuration must occur before execution");
    require(draft_vocab && draft_vocab <= p.shape.vocabulary, "invalid draft vocabulary size");
    require(draft_out.view.bytes && draft_out.encoding == p.weights.output.encoding, "invalid draft output matrix");
    p.draft_vocabulary = draft_vocab;
    p.draft_output = draft_out;
    for (auto it = p.owned.begin(); it != p.owned.end(); ++it) {
        if (it->slot == p.output_logits.slot && it->offset == p.output_logits.offset) {
            try { p.graph.release(*it); } catch (...) {}
            p.owned.erase(it);
            break;
        }
    }
    p.output_logits = p.floats(p.draft_vocabulary, "MTP draft vocabulary logits (32K)");
    p.host_logits.resize(p.draft_vocabulary);
}
uint32_t MtpHead::draft_vocabulary() const noexcept { return impl_->draft_vocabulary; }
bool MtpHead::has_draft_slice() const noexcept { return impl_->draft_vocabulary != 0; }
void MtpHead::configure_fast_round(bool enabled, GpuMatrix token_embedding) {
    auto& p = *impl_;
    require(!p.busy && p.valid && !p.logical_rows(), "MTP fast-round configuration must occur at an empty idle prefix");
    if (!enabled) {
        p.fast_round = false;
        return;
    }
    if (p.weights.embedding_hidden.encoding == 8 &&
        (p.shape.residual.branches == 1 || p.shape.residual.branches == 2 ||
         p.shape.residual.branches == 4 || p.shape.residual.branches == 8) &&
        (2 * p.shape.residual.channels) % 32 == 0) {
        p.graph.prepare_q8_matvec_many(p.shape.residual.branches);
    }
    p.graph.prepare_logit_topk();
    const uint64_t c = p.shape.residual.channels, h = uint64_t(p.shape.residual.branches) * c;
    if (!p.topk_row.owner) {
        const auto before = p.graph.memory().live_bytes;
        const auto scratch = p.graph.zeros(MetalGraph::logit_topk_scratch_bytes(1, p.shape.vocabulary), "MTP fast-round top-k scratch");
        p.owned.push_back(scratch);
        const auto row = p.graph.zeros(MetalGraph::logit_topk_output_bytes(1), "MTP fast-round top-k row");
        p.owned.push_back(row);
        const auto chain_rows = p.graph.zeros(MetalGraph::logit_topk_output_bytes(3), "MTP fast greedy chain top-k rows");
        p.owned.push_back(chain_rows);
        const auto tickets = p.graph.zeros(32, "MTP fast sampled chain tickets");
        p.owned.push_back(tickets);
        const auto laws = p.graph.zeros(3 * sizeof(GpuMtpSampledLawRow), "MTP fast sampled chain laws");
        p.owned.push_back(laws);
        const auto dhc = p.graph.zeros(checked_mul(checked_mul(4u, h), 4), "MTP deferred priming HC");
        p.owned.push_back(dhc);
        const auto demb = p.graph.zeros(checked_mul(checked_mul(4u, c), 4), "MTP deferred priming embeddings");
        p.owned.push_back(demb);
        p.topk_scratch = scratch;
        p.topk_row = row;
        p.topk_chain_rows = chain_rows;
        p.sampled_tickets_gpu = tickets;
        p.sampled_chain_laws = laws;
        p.deferred_hc = dhc;
        p.deferred_emb = demb;
        p.charged = checked_add(p.charged, p.graph.memory().live_bytes - before);
    }
    const uint32_t out_vocab = p.draft_vocabulary ? p.draft_vocabulary : p.shape.vocabulary;
    if (token_embedding.view.owner) {
        const uint64_t row_bytes = uint64_t(c / 32) * 34;
        require(token_embedding.encoding == 8 && c % 32 == 0 && row_bytes > 0 &&
            token_embedding.view.bytes % row_bytes == 0 &&
            token_embedding.view.bytes / row_bytes >= out_vocab &&
            token_embedding.view.bytes / row_bytes <= p.shape.vocabulary,
            "invalid MTP fast-round Q8_0 token embedding matrix");
        p.fast_token_embd = token_embedding.view;
        p.fast_token_embd_vocab = uint32_t(token_embedding.view.bytes / row_bytes);
        p.greedy_chain_ready = true;
    } else if (!p.fast_token_embd.owner && p.target && c % 32 == 0) {
        const auto& tt = p.target->tensor("token_embd.weight");
        const uint64_t row_bytes = uint64_t(c / 32) * 34;
        const uint64_t expected_bytes = row_bytes * p.shape.vocabulary;
        const uint64_t slice_bytes = row_bytes * out_vocab;
        const auto mem = p.graph.memory();
        if (tt.type == 8 && tt.bytes == expected_bytes) {
            if (p.draft_vocabulary && p.graph.draft_embedding_slice().owner &&
                p.graph.draft_embedding_slice().bytes == slice_bytes) {
                p.graph.write(p.graph.draft_embedding_slice(), p.target->tensor_bytes("token_embd.weight").first(slice_bytes));
                p.fast_token_embd = p.graph.draft_embedding_slice();
                p.fast_token_embd_vocab = out_vocab;
                p.greedy_chain_ready = true;
            } else if (checked_add(mem.live_bytes, checked_add(slice_bytes, 2ull << 30)) <= mem.budget) {
                const auto emb = p.graph.upload(p.target->tensor_bytes("token_embd.weight").first(slice_bytes),
                    "MTP fast-round Q8_0 token embeddings", true);
                p.owned.push_back(emb);
                p.fast_token_embd = emb;
                p.fast_token_embd_vocab = out_vocab;
                p.greedy_chain_ready = true;
            }
        }
    }
    p.fast_round = true;
}
bool MtpHead::fast_round() const noexcept { return impl_->fast_round; }
bool MtpHead::can_greedy_chain() const noexcept { return impl_->fast_round && impl_->greedy_chain_ready; }
MtpHeadFastResult MtpHead::advance_fast(GpuView hc_gpu, std::span<const float> embedding, uint32_t position) {
    return impl_->run_fast(&hc_gpu, {}, false, embedding, position);
}
MtpHeadFastResult MtpHead::advance_fast_host(std::span<const float> hc, std::span<const float> embedding, uint32_t position) {
    return impl_->run_fast(nullptr, hc, false, embedding, position);
}
MtpHeadFastResult MtpHead::advance_fast_chain(std::span<const float> embedding, uint32_t position) {
    return impl_->run_fast(nullptr, {}, true, embedding, position);
}
MtpHeadFastChainResult MtpHead::advance_fast_greedy_chain(GpuView hc_gpu, std::span<const float> anchor_embedding,
                                                          uint32_t first_position, uint32_t depth) {
    return impl_->run_fast_greedy_chain(&hc_gpu, {}, anchor_embedding, first_position, depth);
}
MtpHeadFastChainResult MtpHead::advance_fast_greedy_chain_host(std::span<const float> hc, std::span<const float> anchor_embedding,
                                                               uint32_t first_position, uint32_t depth) {
    return impl_->run_fast_greedy_chain(nullptr, hc, anchor_embedding, first_position, depth);
}
MtpHeadSampledChainResult MtpHead::advance_fast_sampled_chain(GpuView hc_gpu, std::span<const float> anchor_embedding,
                                                              uint32_t first_position, uint32_t depth,
                                                              SamplingPolicy draft_policy,
                                                              std::span<const uint64_t, 3> tickets) {
    return impl_->run_fast_sampled_chain(&hc_gpu, {}, anchor_embedding, first_position, depth, draft_policy, tickets);
}
MtpHeadSampledChainResult MtpHead::advance_fast_sampled_chain_host(std::span<const float> hc, std::span<const float> anchor_embedding,
                                                                   uint32_t first_position, uint32_t depth,
                                                                   SamplingPolicy draft_policy,
                                                                   std::span<const uint64_t, 3> tickets) {
    return impl_->run_fast_sampled_chain(nullptr, hc, anchor_embedding, first_position, depth, draft_policy, tickets);
}
void MtpHead::defer_prime_batch(GpuView hc_gpu, std::span<const float> embedding_rows, uint32_t first_position) {
    impl_->stage_deferred_prime(&hc_gpu, {}, embedding_rows, first_position);
}
void MtpHead::defer_prime_batch_host(std::span<const float> hc_rows, std::span<const float> embedding_rows, uint32_t first_position) {
    impl_->stage_deferred_prime(nullptr, hc_rows, embedding_rows, first_position);
}
BatchResult MtpHead::flush_deferred_prime() { return impl_->flush_deferred(); }
uint32_t MtpHead::deferred_prime_rows() const noexcept { return impl_->pending_prime_count; }
uint32_t MtpHead::first_absolute_position() const noexcept { return impl_->first; }
uint64_t MtpHead::ssd_cache_bytes(uint32_t cache_rows) const {
    const auto& p = *impl_;
    require(cache_rows <= p.shape.capacity, "MTP SSD cache rows exceed capacity");
    return checked_mul(checked_mul(uint64_t(cache_rows), uint64_t(p.shape.kv_heads) * p.shape.head_width), 4);
}
uint64_t MtpHead::ssd_cache_bytes() const { return ssd_cache_bytes(impl_->logical_rows()); }
bool MtpHead::read_ssd_cache_span(uint64_t offset, std::span<std::byte> target) const {
    auto& p = *impl_;
    if (p.busy || !p.valid) return false;
    if (p.pending_prime_count) static_cast<void>(p.flush_deferred());
    const uint64_t total = ssd_cache_bytes(p.rows);
    if (offset > total || target.size() > total - offset) return false;
    if (target.empty()) return true;
    const uint64_t half = total / 2;
    const uint64_t req_end = offset + target.size();
    const auto copy_gpu = [&](GpuView src, uint64_t seg_start, uint64_t seg_len) {
        const uint64_t seg_end = seg_start + seg_len;
        if (seg_end <= offset || seg_start >= req_end) return;
        const uint64_t ov_start = std::max(seg_start, offset);
        const uint64_t ov_len = std::min(seg_end, req_end) - ov_start;
        p.graph.read(p.graph.slice(src, ov_start - seg_start, ov_len),
                     target.subspan(size_t(ov_start - offset), size_t(ov_len)));
    };
    if (half > 0) {
        copy_gpu(p.keys, 0, half);
        copy_gpu(p.values, half, half);
    }
    return true;
}
bool MtpHead::write_ssd_cache_span(uint32_t first_pos, uint32_t cache_rows, uint64_t offset, std::span<const std::byte> source) {
    auto& p = *impl_;
    if (p.busy || first_pos >= 65536 || cache_rows > p.shape.capacity || cache_rows > 65536 - first_pos) return false;
    const uint64_t total = ssd_cache_bytes(cache_rows);
    if (offset > total || source.size() > total - offset) return false;
    if (offset == 0) {
        p.first = first_pos;
        p.rows = cache_rows;
        p.pending_prime_count = 0;
        p.pending_prime_pos = 0;
        p.valid = true;
        p.logits_valid = false;
        p.hidden_valid = false;
        p.hidden_gpu_valid = false;
    }
    if (source.empty()) return true;
    const uint64_t half = total / 2;
    const uint64_t req_end = offset + source.size();
    const auto write_gpu = [&](GpuView dst, uint64_t seg_start, uint64_t seg_len) {
        const uint64_t seg_end = seg_start + seg_len;
        if (seg_end <= offset || seg_start >= req_end) return;
        const uint64_t ov_start = std::max(seg_start, offset);
        const uint64_t ov_len = std::min(seg_end, req_end) - ov_start;
        p.graph.write(p.graph.slice(dst, ov_start - seg_start, ov_len),
                      source.subspan(size_t(ov_start - offset), size_t(ov_len)));
    };
    if (half > 0) {
        write_gpu(p.keys, 0, half);
        write_gpu(p.values, half, half);
    }
    return true;
}

} // namespace tokspur
