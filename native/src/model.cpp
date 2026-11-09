#include "tokspur/model.hpp"
#include "tokspur/derived_pack.hpp"
#include "tokspur/planar_pack_v3.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <set>
#include <stdexcept>

namespace tokspur {
namespace {
void require(bool condition, const char* message) { if (!condition) throw FormatError(message); }

Json number(double value) {
    std::array<char, 64> bytes{};
    const auto result = std::to_chars(bytes.data(), bytes.data() + bytes.size(), value);
    require(result.ec == std::errc{}, "cannot encode model scalar");
    return parse_json({bytes.data(), static_cast<size_t>(result.ptr - bytes.data())});
}

Json list(const std::vector<uint64_t>& values) {
    Json::Array output;
    for (const auto value : values) output.push_back(Json::integer(value));
    return Json(std::move(output));
}
} // namespace

std::vector<TensorRequirement> flash_next_requirements(uint64_t ple_rows) {
    require(ple_rows > 0 && ple_rows <= 2'000'000'000, "invalid PLE row count");
    std::vector<TensorRequirement> result;
    const auto add = [&](std::string name, std::initializer_list<uint64_t> shape, std::initializer_list<uint32_t> types) {
        result.push_back({std::move(name), shape, types});
    };
    add("token_embd.weight", {2560, 248320}, {8});
    add("output.weight", {2560, 248320}, {8});
    add("output_hc_norm.weight", {10240}, {0});
    add("output_hc_down.weight", {10240, 320}, {8});
    add("output_hc_up.weight", {320, 10240}, {8});
    add("per_layer_token_embd.weight", {160, ple_rows}, {20});
    for (size_t layer = 0; layer < 48; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        for (const auto* path : {"hc_attn", "hc_ffn"}) {
            const auto root = prefix + path;
            add(root + "_norm.weight", {10240}, {0});
            add(root + "_down.weight", {10240, 320}, {8});
            add(root + "_up.weight", {320, 10240}, {8});
            add(root + "_inject.weight", {10240, 4}, {0});
        }
        add(prefix + "ffn_gate_inp.weight", {2560, 512}, {0});
        add(prefix + "ffn_gate_inp_shexp.weight", {2560}, {0});
        add(prefix + "ffn_gate_exps.weight", {2560, 640, 512}, {12, 13});
        add(prefix + "ffn_up_exps.weight", {2560, 640, 512}, {12, 13});
        add(prefix + "ffn_down_exps.weight", {640, 2560, 512}, {7, 8});
        add(prefix + "ffn_gate_shexp.weight", {2560, 640}, {8});
        add(prefix + "ffn_up_shexp.weight", {2560, 640}, {8});
        add(prefix + "ffn_down_shexp.weight", {640, 2560}, {8});
        if (layer % 4 == 3) {
            add(prefix + "attn_q.weight", {2560, 12288}, {8});
            add(prefix + "attn_k.weight", {2560, 512}, {8});
            add(prefix + "attn_v.weight", {2560, 512}, {8});
            add(prefix + "attn_output.weight", {6144, 2560}, {8});
            add(prefix + "attn_q_norm.weight", {256}, {0});
            add(prefix + "attn_k_norm.weight", {256}, {0});
            add(prefix + "indexer.q_proj.weight", {2560, 512}, {30});
            add(prefix + "indexer.k_proj.weight", {2560, 128}, {30});
            add(prefix + "indexer.q_norm.weight", {128}, {0});
            add(prefix + "indexer.k_norm.weight", {128}, {0});
        } else {
            add(prefix + "attn_qkv.weight", {2560, 10240}, {8});
            add(prefix + "attn_gate.weight", {2560, 6144}, {8});
            add(prefix + "ssm_a", {48}, {0});
            add(prefix + "ssm_dt.bias", {48}, {0});
            add(prefix + "ssm_alpha.weight", {2560, 48}, {0});
            add(prefix + "ssm_beta.weight", {2560, 48}, {0});
            add(prefix + "ssm_conv1d.weight", {4, 10240}, {0});
            add(prefix + "ssm_norm.weight", {128}, {0});
            add(prefix + "ssm_out.weight", {6144, 2560}, {8});
        }
        if (layer == 1) {
            add(prefix + "ple_key.weight", {2560, 10240}, {8});
            add(prefix + "ple_value.weight", {2560, 2560}, {8});
            add(prefix + "ple_norm_key.weight", {10240}, {0});
            add(prefix + "ple_norm_query.weight", {10240}, {0});
            add(prefix + "ple_norm_conv.weight", {10240}, {0});
            add(prefix + "ple_conv1d.weight", {4, 10240}, {0});
        }
    }
    return result;
}

FlashLayout inspect_flash_next(const ModelFiles& files, const ModelFiles* pack) {
    require(files.architecture() == "qwen4exp", "model does not match the Flash-Next architecture profile");
    const Gguf* metadata = nullptr;
    for (const auto& shard : files.shards()) if (shard.find("qwen4exp.block_count")) {
        require(!metadata, "ambiguous model architecture metadata across shards");
        metadata = &shard;
    }
    require(metadata, "missing model architecture metadata");
    for (const auto& [key, value] : std::vector<std::pair<const char*, uint64_t>>{
        {"block_count", 48}, {"embedding_length", 2560}, {"attention.head_count", 24}, {"attention.head_count_kv", 2},
        {"attention.key_length", 256}, {"attention.value_length", 256}, {"expert_count", 512}, {"expert_used_count", 10},
        {"expert_feed_forward_length", 640}, {"expert_shared_feed_forward_length", 640}, {"ssm.conv_kernel", 4},
        {"ssm.state_size", 128}, {"ssm.group_count", 16}, {"ssm.time_step_rank", 48}, {"ssm.inner_size", 6144},
        {"full_attention_interval", 4}, {"rope.dimension_count", 64}, {"hyper_connection.count", 4}, {"hyper_connection.low_rank", 320},
        {"attention.indexer.head_count", 4}, {"attention.indexer.key_length", 128}, {"attention.indexer.top_k", 2048},
        {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 8}, {"ple.conv_kernel", 4}, {"embedding_length_per_layer_input", 160},
        {"ple.eos_token_id", 248044}, {"ple.image_token_id", 248056}}) {
        require(metadata->unsigned_value(std::string("qwen4exp.") + key) == value, "unsupported Flash-Next architecture parameter");
    }
    require(metadata->unsigned_value("qwen4exp.context_length") >= 65536, "model context cannot cover the initial profile");
    require(metadata->unsigned_array("qwen4exp.rope.dimension_sections") == std::vector<uint64_t>({11, 11, 10, 0}), "unsupported RoPE sections");
    require(metadata->unsigned_array("qwen4exp.ple.layers") == std::vector<uint64_t>({1}), "unsupported PLE placement");
    const auto ratios = metadata->unsigned_array("qwen4exp.attention.compress_ratios");
    require(ratios.size() == 48, "invalid QSA layer count");
    for (size_t i = 0; i < ratios.size(); ++i) require(ratios[i] == (i % 4 == 3 ? 4 : 0), "unsupported QSA compression schedule");
    FlashLayout result;
    result.rms_epsilon = metadata->floating_value("qwen4exp.attention.layer_norm_rms_epsilon");
    result.rope_base = metadata->floating_value("qwen4exp.rope.freq_base");
    require(result.rms_epsilon > 0 && result.rms_epsilon <= 0.01 && result.rope_base > 1, "invalid model normalization/RoPE scalars");
    result.ple_multipliers = metadata->unsigned_array("qwen4exp.ple.layer_multipliers");
    result.ple_offsets = metadata->unsigned_array("qwen4exp.ple.head_offsets");
    result.ple_vocabularies = metadata->unsigned_array("qwen4exp.ple.head_vocab_sizes");
    require(result.ple_multipliers.size() == 3 && result.ple_offsets.size() == 16 && result.ple_vocabularies.size() == 16, "invalid PLE hash geometry");
    uint64_t ple_rows = 0;
    for (size_t i = 0; i < 16; ++i) {
        require(result.ple_offsets[i] == ple_rows && result.ple_vocabularies[i] >= 1'000'000 && result.ple_vocabularies[i] <= 100'000'000, "invalid PLE table partitions");
        ple_rows = checked_add(ple_rows, result.ple_vocabularies[i]);
    }
    for (const auto value : result.ple_multipliers) require(value > 0 && (value & 1), "invalid PLE hash multiplier");
    // GGUF storage rounds the concatenated hash table up to 256 rows. Padding
    // is not another partition and must never become an addressable hash bucket.
    result.ple_addressable_rows = ple_rows;
    result.ple_storage_rows = checked_add(ple_rows, 255) & ~uint64_t(255);
    const auto requirements = flash_next_requirements(result.ple_storage_rows);
    require(requirements.size() == files.tensors().size(), "unaccounted or missing tensors in Flash-Next variant");
    const bool pack_is_v3 = pack && pack->shards().size() == 1 &&
        pack->shards()[0].find("tokspur_pack.version") &&
        pack->shards()[0].unsigned_value("tokspur_pack.version") == planar_v3::pack_version_v3;
    std::set<std::string> consumed;
    for (const auto& item : requirements) {
        const bool is_cpu = derived_pack::is_cpu_pinned_embedding(item.name);
        const bool use_pack = pack && !is_cpu;
        if (use_pack && !pack->tensors().contains(item.name)) {
            throw std::runtime_error("Derived pack missing required GPU tensor: " + item.name);
        }
        const auto& tensor = use_pack ? pack->tensor(item.name) : files.tensor(item.name);
        if (use_pack && pack_is_v3 && tensor.type == 24) {
            const auto& raw_tensor = files.tensor(item.name);
            if (raw_tensor.dimensions != item.shape) throw FormatError("Flash-Next tensor shape mismatch: " + item.name);
            if (std::find(item.encodings.begin(), item.encodings.end(), raw_tensor.type) == item.encodings.end())
                throw FormatError("unsupported tensor encoding for this model profile: " + item.name);
            require((raw_tensor.type == 12 || raw_tensor.type == 13) && item.shape.size() == 3,
                "unexpected byte-container tensor in Pack v3");
            const auto spec = planar_v3::make_expert_spec(raw_tensor.type, uint32_t(item.shape[0]), uint32_t(item.shape[1]));
            const std::vector<uint64_t> expected_dims{spec.planar_row_bytes, item.shape[1], item.shape[2]};
            if (tensor.dimensions != expected_dims || tensor.bytes != checked_mul(spec.planar_expert_bytes, item.shape[2]))
                throw FormatError("Pack v3 planar tensor geometry mismatch: " + item.name);
        } else {
            if (tensor.dimensions != item.shape) throw FormatError("Flash-Next tensor shape mismatch: " + item.name);
            if (std::find(item.encodings.begin(), item.encodings.end(), tensor.type) == item.encodings.end()) throw FormatError("unsupported tensor encoding for this model profile: " + item.name);
        }
        require(consumed.insert(item.name).second, "duplicate model tensor requirement");
        if (item.name == "per_layer_token_embd.weight") result.ple_source_bytes = tensor.bytes;
        else result.resident_matrix_bytes = checked_add(result.resident_matrix_bytes, tensor.bytes);
    }
    for (const auto& [name, location] : files.tensors()) {
        static_cast<void>(location);
        require(consumed.contains(name), "unconsumed model tensor");
    }
    if (pack) pack->assert_unchanged();
    result.tensors = consumed.size();
    files.assert_unchanged();
    return result;
}

Json FlashLayout::describe() const {
    constexpr uint64_t context = 65536;
    constexpr uint64_t kv = 12 * context * 2 * 256 * 2 * 2;
    constexpr uint64_t gdn = 36 * 48 * 128 * 128 * 4;
    constexpr uint64_t gdn_conv = 36 * 3 * 10240 * 4;
    // Published maximum-order dilation=3 requires nine historical vectors.
    // Still storage arithmetic, not checkpoint semantics or measured fit.
    constexpr uint64_t ple_conv = 9 * 10240 * 4;
    constexpr uint64_t index_keys = 12 * (context / 4) * 128 * 2;
    constexpr uint64_t index_partial = 12 * 128 * 4;
    constexpr uint64_t ple_cache = 512 * 1024 * 1024;
    const auto subtotal = checked_add(resident_matrix_bytes, kv + gdn + gdn_conv + ple_conv + index_keys + index_partial + ple_cache);
    return Json(Json::Object{
        {"schema", Json::integer(1)}, {"scope", Json("closed geometry/storage inventory; not executable graph or measured memory")},
        {"tensors_consumed", Json::integer(tensors)}, {"layers", Json::integer(48)}, {"gdn_layers", Json::integer(36)},
        {"qsa_layers", Json::integer(12)}, {"separate_mtp_tensors", Json::integer(0)}, {"ple_layer_zero_based", Json::integer(1)},
        {"resident_matrix_payload_bytes", Json::integer(resident_matrix_bytes)}, {"ple_source_bytes", Json::integer(ple_source_bytes)},
        {"ple_addressable_rows", Json::integer(ple_addressable_rows)}, {"ple_storage_rows", Json::integer(ple_storage_rows)},
        {"rms_epsilon", number(rms_epsilon)}, {"rope_base", number(rope_base)},
        {"ple_multipliers", list(ple_multipliers)}, {"ple_offsets", list(ple_offsets)}, {"ple_vocabularies", list(ple_vocabularies)},
        {"context_tokens", Json::integer(context)}, {"kv_bf16_bytes", Json::integer(kv)}, {"gdn_fp32_bytes", Json::integer(gdn)},
        {"gdn_conv_fp32_bytes", Json::integer(gdn_conv)}, {"ple_conv_fp32_bytes", Json::integer(ple_conv)},
        {"indexer_bf16_keys_bytes", Json::integer(index_keys)}, {"indexer_fp32_partial_bytes", Json::integer(index_partial)},
        {"ple_row_cache_budget_bytes", Json::integer(ple_cache)}, {"known_payload_subtotal_bytes", Json::integer(subtotal)},
        {"unbudgeted", Json("draft, scratch, historical/in-flight checkpoints, allocator/driver/page overhead; no fit qualification")},
        {"numerical_semantics", Json("projection ordering, norm/gate checkpoint transforms and full layer/logit equivalence remain unqualified")}
    });
}

} // namespace tokspur
