#include "tokspur/mtp.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <utility>

namespace tokspur {
namespace {
void require(bool condition, const char* message) { if (!condition) throw FormatError(message); }

constexpr std::array tokenizer_keys{
    "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.add_bos_token",
};

// Actual shared-Q8 metadata declarations; these are container constraints, not
// an inference of MTP graph order, target taps or QSA/cache semantics.
const std::vector<std::pair<const char*, uint64_t>>& common_scalars() {
    static const std::vector<std::pair<const char*, uint64_t>> values{
        {"context_length", 262144}, {"embedding_length", 2560}, {"attention.head_count", 24},
        {"attention.head_count_kv", 2}, {"expert_count", 512}, {"expert_used_count", 10},
        {"attention.key_length", 256}, {"attention.value_length", 256},
        {"expert_feed_forward_length", 640}, {"expert_shared_feed_forward_length", 640},
        {"ssm.conv_kernel", 4}, {"ssm.state_size", 128}, {"ssm.group_count", 16},
        {"ssm.time_step_rank", 48}, {"ssm.inner_size", 6144}, {"full_attention_interval", 4},
        {"rope.dimension_count", 64}, {"hyper_connection.count", 4}, {"hyper_connection.low_rank", 320},
        {"attention.indexer.head_count", 4}, {"attention.indexer.key_length", 128}, {"attention.indexer.top_k", 2048},
    };
    return values;
}

void same_metadata(const Gguf& a, const Gguf& b, std::string_view key) {
    const auto* left = a.find(key);
    const auto* right = b.find(key);
    if (!left || !right || left->type != right->type || left->element_type != right->element_type ||
        left->elements != right->elements || !std::ranges::equal(a.metadata_bytes(key), b.metadata_bytes(key))) {
        throw FormatError("MTP target metadata identity mismatch: " + std::string(key));
    }
}

void metadata_shape(const Gguf& sidecar) {
    std::map<std::string, uint32_t, std::less<>> expected{
        {"general.architecture", 8}, {"general.type", 8}, {"general.sampling.top_k", 5},
        {"general.sampling.top_p", 6}, {"general.sampling.temp", 6}, {"general.name", 8},
        {"general.size_label", 8}, {"general.license", 8}, {"general.license.name", 8},
        {"general.license.link", 8}, {"general.tags", 9}, {"general.quantization_version", 4}, {"general.file_type", 4},
        {"qwen4exp.nextn_shared_target_tensors", 7}, {"qwen4exp.block_count", 4}, {"qwen4exp.nextn_predict_layers", 4},
        {"qwen4exp.rope.dimension_sections", 9}, {"qwen4exp.rope.freq_base", 6},
        {"qwen4exp.attention.layer_norm_rms_epsilon", 6}, {"qwen4exp.attention.compress_ratios", 9},
        {"tokenizer.ggml.model", 8}, {"tokenizer.ggml.pre", 8}, {"tokenizer.ggml.tokens", 9},
        {"tokenizer.ggml.token_type", 9}, {"tokenizer.ggml.merges", 9}, {"tokenizer.ggml.eos_token_id", 4},
        {"tokenizer.ggml.padding_token_id", 4}, {"tokenizer.ggml.bos_token_id", 4}, {"tokenizer.ggml.add_bos_token", 7},
        {"tokenizer.chat_template", 8},
    };
    for (const auto& [key, value] : common_scalars()) {
        static_cast<void>(value);
        expected.emplace(std::string("qwen4exp.") + key, 4);
    }
    require(expected.size() == 52 && sidecar.metadata().size() == expected.size(), "unaccounted MTP metadata records");
    for (const auto& [key, type] : expected) {
        const auto* field = sidecar.find(key);
        if (!field || field->type != type) throw FormatError("unsupported MTP metadata shape: " + key);
    }
}

void array_shape(const Gguf& file, std::string_view key, uint32_t type, uint64_t count) {
    const auto* value = file.find(key);
    if (!value || value->type != 9 || value->element_type != type || value->elements != count) {
        throw FormatError("unsupported MTP metadata array: " + std::string(key));
    }
}

void borrowed_shape(const Tensor& tensor, std::string_view name) {
    require(tensor.name == name && tensor.dimensions == std::vector<uint64_t>({2560, 248320}) &&
            tensor.type == 8 && tensor.elements == 635'699'200 && tensor.bytes == 675'430'400,
            "unsupported borrowed MTP tensor");
}
} // namespace

std::vector<MtpTensorRequirement> mtp_shared_q8_requirements() {
    // Descriptor order and offsets are pinned to the actual local catalogue.
    // No tensor values, quantization conversion or external engine code used.
    std::vector<MtpTensorRequirement> result;
    uint64_t offset = mtp_shared_q8_header_bytes;
    const auto add = [&](const char* name, std::initializer_list<uint64_t> shape, uint32_t type) {
        uint64_t elements = 1;
        for (const auto dimension : shape) elements = checked_mul(elements, dimension);
        const auto& format = encoding(type);
        const auto bytes = checked_mul(elements / format.elements_per_block, format.bytes_per_block);
        result.push_back({std::string("blk.48.") + name, shape, type, offset, bytes});
        offset = checked_add(offset, bytes);
    };
    add("attn_k.weight", {2560, 512}, 8);
    add("attn_k_norm.weight", {256}, 0);
    add("attn_output.weight", {6144, 2560}, 8);
    add("attn_q.weight", {2560, 12288}, 8);
    add("attn_q_norm.weight", {256}, 0);
    add("attn_v.weight", {2560, 512}, 8);
    add("ffn_down_exps.weight", {640, 2560, 512}, 8);
    add("ffn_down_shexp.weight", {640, 2560}, 8);
    add("ffn_gate_exps.weight", {2560, 640, 512}, 8);
    add("ffn_gate_inp.weight", {2560, 512}, 0);
    add("ffn_gate_inp_shexp.weight", {2560}, 0);
    add("ffn_gate_shexp.weight", {2560, 640}, 8);
    add("ffn_up_exps.weight", {2560, 640, 512}, 8);
    add("ffn_up_shexp.weight", {2560, 640}, 8);
    add("hc_attn_down.weight", {10240, 320}, 8);
    add("hc_attn_inject.weight", {10240, 4}, 8);
    add("hc_attn_norm.weight", {10240}, 0);
    add("hc_attn_up.weight", {320, 10240}, 8);
    add("hc_ffn_down.weight", {10240, 320}, 8);
    add("hc_ffn_inject.weight", {10240, 4}, 8);
    add("hc_ffn_norm.weight", {10240}, 0);
    add("hc_ffn_up.weight", {320, 10240}, 8);
    add("indexer.k_norm.weight", {128}, 0);
    add("indexer.k_proj.weight", {2560, 128}, 30);
    add("indexer.q_norm.weight", {128}, 0);
    add("indexer.q_proj.weight", {2560, 512}, 30);
    add("nextn.eh_proj.weight", {5120, 2560}, 8);
    add("nextn.enorm.weight", {2560}, 0);
    add("nextn.hc_head_down.weight", {10240, 320}, 8);
    add("nextn.hc_head_norm.weight", {10240}, 0);
    add("nextn.hc_head_up.weight", {320, 10240}, 8);
    add("nextn.hnorm.weight", {10240}, 0);
    require(offset == mtp_shared_q8_file_bytes, "invalid compiled MTP extent contract");
    return result;
}

MtpSharedLayout inspect_mtp_shared_q8_catalogue(
    const Gguf& sidecar, const Gguf& target_metadata,
    const Tensor& token_embedding, const Tensor& output, MtpTemplatePolicy policy) {
    require(policy == MtpTemplatePolicy::require_exact_identity || policy == MtpTemplatePolicy::target_rendered_ids_only,
            "unknown MTP template policy");
    sidecar.assert_unchanged();
    target_metadata.assert_unchanged();
    require(sidecar.file_size() == mtp_shared_q8_file_bytes && sidecar.data_offset() == mtp_shared_q8_header_bytes,
            "MTP sidecar extent does not match the closed shared-Q8 profile");
    metadata_shape(sidecar);
    require(sidecar.string_value("general.architecture") == "qwen4exp" &&
            target_metadata.string_value("general.architecture") == "qwen4exp", "unsupported MTP architecture");
    require(sidecar.unsigned_value("qwen4exp.block_count") == 49 &&
            target_metadata.unsigned_value("qwen4exp.block_count") == 48 &&
            sidecar.unsigned_value("qwen4exp.nextn_predict_layers") == 1 &&
            sidecar.boolean_value("qwen4exp.nextn_shared_target_tensors"), "unsupported MTP sharing/layer profile");
    require(sidecar.unsigned_value("general.quantization_version") == 2 && sidecar.unsigned_value("general.file_type") == 7,
            "unsupported MTP quantization declarations");
    array_shape(sidecar, "general.tags", 8, 1);
    array_shape(sidecar, "qwen4exp.rope.dimension_sections", 5, 4);
    for (const auto& [suffix, value] : common_scalars()) {
        const auto key = std::string("qwen4exp.") + suffix;
        require(sidecar.unsigned_value(key) == value, "unsupported MTP architecture scalar");
        same_metadata(sidecar, target_metadata, key);
    }
    for (const auto key : {"qwen4exp.rope.freq_base", "qwen4exp.attention.layer_norm_rms_epsilon", "qwen4exp.rope.dimension_sections"}) {
        same_metadata(sidecar, target_metadata, key);
    }
    require(sidecar.floating_value("qwen4exp.rope.freq_base") == 10'000'000 &&
            sidecar.floating_value("qwen4exp.attention.layer_norm_rms_epsilon") == double(1e-6f) &&
            sidecar.unsigned_array("qwen4exp.rope.dimension_sections") == std::vector<uint64_t>({11, 11, 10, 0}),
            "unsupported MTP RoPE/normalization declarations");
    array_shape(sidecar, "qwen4exp.attention.compress_ratios", 5, 49);
    auto ratios = sidecar.unsigned_array("qwen4exp.attention.compress_ratios");
    require(ratios.back() == 0, "unsupported MTP final compression declaration");
    ratios.pop_back();
    require(ratios == target_metadata.unsigned_array("qwen4exp.attention.compress_ratios"), "MTP target compression schedule mismatch");
    for (size_t i = 0; i < ratios.size(); ++i) require(ratios[i] == (i % 4 == 3 ? 4 : 0), "unsupported target compression schedule");

    std::set<std::string_view> declared_tokenizer_keys;
    for (const auto& [key, value] : target_metadata.metadata()) {
        static_cast<void>(value);
        if (key.starts_with("tokenizer.")) declared_tokenizer_keys.insert(key);
    }
    std::set<std::string_view> expected_tokenizer_keys(tokenizer_keys.begin(), tokenizer_keys.end());
    expected_tokenizer_keys.insert("tokenizer.chat_template");
    require(declared_tokenizer_keys == expected_tokenizer_keys,
            "unaccounted target tokenizer records");
    for (const auto key : tokenizer_keys) same_metadata(sidecar, target_metadata, key);
    require(sidecar.string_value("tokenizer.ggml.model") == "gpt2" && sidecar.string_value("tokenizer.ggml.pre") == "qwen35" &&
            sidecar.unsigned_value("tokenizer.ggml.eos_token_id") == 248046 &&
            sidecar.unsigned_value("tokenizer.ggml.padding_token_id") == 248044 &&
            sidecar.unsigned_value("tokenizer.ggml.bos_token_id") == 248044 &&
            !sidecar.boolean_value("tokenizer.ggml.add_bos_token"), "unsupported MTP tokenizer policy");
    array_shape(sidecar, "tokenizer.ggml.tokens", 8, 248320);
    array_shape(sidecar, "tokenizer.ggml.token_type", 5, 248320);
    array_shape(sidecar, "tokenizer.ggml.merges", 8, 247587);
    borrowed_shape(token_embedding, "token_embd.weight");
    borrowed_shape(output, "output.weight");

    MtpSharedLayout layout;
    layout.template_policy = policy;
    layout.identical_tokenizer_records = tokenizer_keys.size();
    const auto side_template = sidecar.string_value("tokenizer.chat_template");
    const auto target_template = target_metadata.string_value("tokenizer.chat_template");
    require(!side_template.empty() && !target_template.empty(), "empty MTP/target chat template");
    layout.sidecar_template_bytes = side_template.size();
    layout.target_template_bytes = target_template.size();
    layout.templates_identical = std::ranges::equal(sidecar.metadata_bytes("tokenizer.chat_template"), target_metadata.metadata_bytes("tokenizer.chat_template"));
    if (policy == MtpTemplatePolicy::require_exact_identity) require(layout.templates_identical, "MTP sidecar chat template differs from target");
    const auto requirements = mtp_shared_q8_requirements();
    require(sidecar.tensors().size() == requirements.size(), "unaccounted MTP tensor descriptors");
    for (size_t i = 0; i < requirements.size(); ++i) {
        const auto& wanted = requirements[i];
        const auto& actual = sidecar.tensors()[i];
        if (actual.name != wanted.name || actual.dimensions != wanted.shape || actual.type != wanted.encoding ||
            actual.offset != wanted.offset || actual.bytes != wanted.bytes) {
            throw FormatError("unsupported MTP tensor descriptor: " + wanted.name);
        }
        layout.owned_tensor_bytes = checked_add(layout.owned_tensor_bytes, actual.bytes);
    }
    layout.owned_tensor_count = requirements.size();
    layout.borrowed_tensor_bytes = checked_add(token_embedding.bytes, output.bytes);
    sidecar.assert_unchanged();
    target_metadata.assert_unchanged();
    return layout;
}

Json MtpSharedLayout::describe() const {
    return Json(Json::Object{
        {"profile", Json("mtp-shared-q8-catalogue-v1")}, {"scope", Json("metadata and retained target descriptor ownership only")},
        {"expected_revision", Json(std::string(mtp_shared_q8_revision))}, {"expected_sidecar_sha256", Json(std::string(mtp_shared_q8_sha256))},
        {"expected_sidecar_file_bytes", Json::integer(mtp_shared_q8_file_bytes)},
        {"owned_tensor_count", Json::integer(owned_tensor_count)}, {"owned_tensor_bytes", Json::integer(owned_tensor_bytes)},
        {"borrowed_tensor_count", Json::integer(2)}, {"borrowed_tensor_bytes", Json::integer(borrowed_tensor_bytes)},
        {"identical_tokenizer_records", Json::integer(identical_tokenizer_records)},
        {"templates_identical", Json(templates_identical)}, {"sidecar_template_bytes", Json::integer(sidecar_template_bytes)},
        {"target_template_bytes", Json::integer(target_template_bytes)},
        {"template_policy", Json(template_policy == MtpTemplatePolicy::require_exact_identity ? "require-exact-identity" : "target-rendered-ids-only")},
        {"full_target_profile_checked", Json(full_target_profile_checked)}, {"full_weight_digest_verified", Json(false)},
        {"model_licence_qualified", Json(false)}, {"graph_implemented", Json(false)}, {"execution_qualified", Json(false)},
    });
}

MtpSharedBinding::MtpSharedBinding(std::shared_ptr<const ModelFiles> target,
                                 const std::filesystem::path& sidecar, MtpTemplatePolicy policy,
                                 std::shared_ptr<const ModelFiles> pack)
    : target_(std::move(target)), sidecar_(std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{sidecar})),
      pack_(std::move(pack)) {
    require(bool(target_), "MTP target ownership is required");
    static_cast<void>(inspect_flash_next(*target_, pack_.get()));
    require(target_->shards().size() == 4, "MTP target must use the pinned four-shard asset profile");
    constexpr std::array<uint64_t, 4> sizes{10'946'624, 49'859'583'136, 49'376'141'504, 12'087'983'520};
    const Gguf* metadata = nullptr;
    for (const auto& shard : target_->shards()) {
        const auto number = shard.unsigned_value("split.no");
        require(number < sizes.size() && shard.file_size() == sizes[number], "MTP target shard extent differs from pinned assets");
        if (shard.find("qwen4exp.block_count")) {
            require(!metadata, "ambiguous MTP target metadata");
            metadata = &shard;
        }
    }
    require(metadata, "missing MTP target metadata");
    require(sidecar_->shards().size() == 1, "MTP sidecar requires one admitted file");
    const auto& out_tensor = (pack_ && pack_->tensors().contains("output.weight")) ? pack_->tensor("output.weight") : target_->tensor("output.weight");
    layout_ = inspect_mtp_shared_q8_catalogue(sidecar_->shards().front(), *metadata, target_->tensor("token_embd.weight"), out_tensor, policy);
    layout_.full_target_profile_checked = true;
    assert_unchanged();
}

const Tensor& MtpSharedBinding::borrowed_tensor(std::string_view name) const {
    require(name == "token_embd.weight" || name == "output.weight", "unregistered borrowed MTP tensor");
    assert_unchanged();
    if (pack_ && name == "output.weight" && pack_->tensors().contains(name)) {
        return pack_->tensor(name);
    }
    return target_->tensor(name);
}

const Tensor& MtpSharedBinding::sidecar_tensor(std::string_view name) const {
    assert_unchanged();
    if (pack_ && pack_->tensors().contains(name)) {
        return pack_->tensor(name);
    }
    return sidecar_->tensor(name);
}

std::shared_ptr<const ModelFiles> MtpSharedBinding::sidecar_source() const {
    assert_unchanged();
    return sidecar_;
}

std::shared_ptr<const ModelFiles> MtpSharedBinding::target_source() const {
    assert_unchanged();
    return target_;
}

void MtpSharedBinding::assert_unchanged() const {
    target_->assert_unchanged();
    sidecar_->assert_unchanged();
    if (pack_) pack_->assert_unchanged();
}

} // namespace tokspur
