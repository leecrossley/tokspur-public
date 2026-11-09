#include "tokspur/mtp.hpp"

#include <algorithm>
#include <bit>
#include <functional>
#include <iostream>
#include <map>

namespace {
using Bytes = std::vector<std::byte>;
struct Field { uint32_t type; Bytes bytes; };
using Fields = std::map<std::string, Field>;
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid MTP catalogue was accepted");
}
void number(Bytes& bytes, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) bytes.push_back(std::byte((value >> (8 * i)) & 255));
}
Bytes integer(uint64_t value, size_t width = 4) { Bytes bytes; number(bytes, value, width); return bytes; }
void string(Bytes& bytes, std::string_view text) {
    number(bytes, text.size(), 8);
    for (const unsigned char c : text) bytes.push_back(std::byte(c));
}
Field text(std::string_view value) { Bytes bytes; string(bytes, value); return {8, std::move(bytes)}; }
Field array(uint32_t type, uint64_t count, std::string_view value = "x") {
    Bytes bytes;
    number(bytes, type, 4); number(bytes, count, 8);
    for (uint64_t i = 0; i < count; ++i) {
        if (type == 8) string(bytes, value);
        else number(bytes, 3, 4);
    }
    return {9, std::move(bytes)};
}
Field ratios(size_t count) {
    Bytes bytes; number(bytes, 5, 4); number(bytes, count, 8);
    for (size_t i = 0; i < count; ++i) number(bytes, i % 4 == 3 ? 4 : 0, 4);
    return {9, std::move(bytes)};
}

// Deliberately synthetic metadata: repeated token spellings are only identity
// controls for this catalogue component, never a usable model/tokenizer fixture.
Fields fields() {
    Fields out;
    for (const auto& [key, value] : std::vector<std::pair<const char*, const char*>>{
            {"general.architecture", "qwen4exp"}, {"general.type", "model"},
            {"general.name", "synthetic"}, {"general.size_label", "synthetic"},
            {"general.license", "other"}, {"general.license.name", "synthetic"}, {"general.license.link", "LICENSE"},
            {"tokenizer.ggml.model", "gpt2"}, {"tokenizer.ggml.pre", "qwen35"}, {"tokenizer.chat_template", "target-template"}}) out.emplace(key, text(value));
    out.emplace("general.tags", array(8, 1, ""));
    out.emplace("general.sampling.top_k", Field{5, integer(20)});
    for (const auto key : {"general.sampling.top_p", "general.sampling.temp"}) out.emplace(key, Field{6, integer(std::bit_cast<uint32_t>(1.0f))});
    out.emplace("general.quantization_version", Field{4, integer(2)});
    out.emplace("general.file_type", Field{4, integer(7)});
    for (const auto& [key, value] : std::vector<std::pair<const char*, uint64_t>>{
            {"block_count",49}, {"context_length",262144}, {"embedding_length",2560},
            {"attention.head_count",24}, {"attention.head_count_kv",2}, {"expert_count",512}, {"expert_used_count",10},
            {"attention.key_length",256}, {"attention.value_length",256}, {"expert_feed_forward_length",640},
            {"expert_shared_feed_forward_length",640}, {"nextn_predict_layers",1}, {"ssm.conv_kernel",4},
            {"ssm.state_size",128}, {"ssm.group_count",16}, {"ssm.time_step_rank",48}, {"ssm.inner_size",6144},
            {"full_attention_interval",4}, {"rope.dimension_count",64}, {"hyper_connection.count",4},
            {"hyper_connection.low_rank",320}, {"attention.indexer.head_count",4},
            {"attention.indexer.key_length",128}, {"attention.indexer.top_k",2048}}) out.emplace(std::string("qwen4exp.")+key, Field{4, integer(value)});
    out.emplace("qwen4exp.nextn_shared_target_tensors", Field{7, integer(1,1)});
    out.emplace("qwen4exp.rope.freq_base", Field{6, integer(std::bit_cast<uint32_t>(1e7f))});
    out.emplace("qwen4exp.attention.layer_norm_rms_epsilon", Field{6, integer(std::bit_cast<uint32_t>(1e-6f))});
    Bytes sections; number(sections,5,4); number(sections,4,8);
    for (const auto value : {11,11,10,0}) number(sections,value,4);
    out.emplace("qwen4exp.rope.dimension_sections",Field{9,std::move(sections)});
    out.emplace("qwen4exp.attention.compress_ratios",ratios(49));
    out.emplace("tokenizer.ggml.tokens",array(8,248320));
    out.emplace("tokenizer.ggml.token_type",array(5,248320));
    out.emplace("tokenizer.ggml.merges",array(8,247587));
    for (const auto& [key,value] : std::vector<std::pair<const char*,uint64_t>>{
            {"eos_token_id",248046},{"padding_token_id",248044},{"bos_token_id",248044}}) out.emplace(std::string("tokenizer.ggml.")+key,Field{4,integer(value)});
    out.emplace("tokenizer.ggml.add_bos_token",Field{7,integer(0,1)});
    return out;
}

Bytes header(Fields fields, const std::vector<tokspur::MtpTensorRequirement>& tensors,
             uint64_t extent = 0) {
    const auto encode = [&]() {
        Bytes out;
        number(out,0x46554747,4); number(out,3,4); number(out,tensors.size(),8); number(out,fields.size(),8);
        for (const auto& [key, field] : fields) {
            string(out,key); number(out,field.type,4); out.insert(out.end(),field.bytes.begin(),field.bytes.end());
        }
        for (const auto& tensor : tensors) {
            string(out,tensor.name); number(out,tensor.shape.size(),4);
            for (const auto dimension : tensor.shape) number(out,dimension,8);
            number(out,tensor.encoding,4); number(out,tensor.offset-tokspur::mtp_shared_q8_header_bytes,8);
        }
        out.resize((out.size()+31)/32*32);
        return out;
    };
    auto out = encode();
    if (extent) {
        require(extent >= out.size(),"fixture metadata bound");
        fields["general.tags"] = array(8,1,std::string(static_cast<size_t>(extent-out.size()),'p'));
        out = encode();
        require(out.size()==extent,"fixture header extent");
    }
    return out;
}

tokspur::Tensor borrowed(const char* name) { return {name,{2560,248320},8,0,675'430'400,635'699'200}; }
constexpr auto target_policy = tokspur::MtpTemplatePolicy::target_rendered_ids_only;

void controls() {
    const auto requirements = tokspur::mtp_shared_q8_requirements();
    require(requirements.size()==32,"closed descriptor count");
    require(requirements.front().offset==10'946'624 && requirements.back().offset==2'786'527'296,"independent boundary offsets");
    require(requirements[6].bytes==891'289'600 && requirements[15].encoding==8 && requirements[23].encoding==30,"actual format exceptions");
    const auto embedding = borrowed("token_embd.weight"), output = borrowed("output.weight");
    const auto original = fields();
    auto target_fields = original;
    target_fields["qwen4exp.block_count"] = {4,integer(48)};
    target_fields["qwen4exp.attention.compress_ratios"] = ratios(48);
    const auto target_bytes = header(target_fields,{});
    tokspur::Gguf target(std::span(target_bytes),target_bytes.size());
    const auto sidecar_bytes = header(original,requirements,tokspur::mtp_shared_q8_header_bytes);
    tokspur::Gguf sidecar(std::span(sidecar_bytes),tokspur::mtp_shared_q8_file_bytes);
    const auto inspect = [&](const tokspur::Gguf& s, const tokspur::Gguf& t, const tokspur::Tensor* e = nullptr, const tokspur::Tensor* o = nullptr) {
        return tokspur::inspect_mtp_shared_q8_catalogue(s,t,e ? *e : embedding,o ? *o : output,target_policy);
    };
    const auto layout = tokspur::inspect_mtp_shared_q8_catalogue(sidecar,target,embedding,output,tokspur::MtpTemplatePolicy::require_exact_identity);
    require(layout.templates_identical && layout.identical_tokenizer_records==9,"typed identity including template");
    require(layout.owned_tensor_bytes==2'775'621'632 && layout.borrowed_tensor_bytes==1'350'860'800,"descriptor-only byte accounting");
    const auto description = tokspur::parse_json(layout.describe().dump());
    require(!description.find("execution_qualified")->boolean() && !description.find("full_weight_digest_verified")->boolean() &&
            !description.find("full_target_profile_checked")->boolean(),"catalogue cannot grant execution/asset/target qualification");
    rejected([&] { static_cast<void>(sidecar.tensor_bytes(sidecar.tensors().front())); });

    auto different_template = target_fields;
    different_template["tokenizer.chat_template"] = text("different-target-template");
    const auto differing_bytes = header(different_template,{});
    tokspur::Gguf differing_target(std::span(differing_bytes),differing_bytes.size());
    require(!inspect(sidecar,differing_target).templates_identical,"explicit target renderer policy reports mismatch");
    rejected([&] { static_cast<void>(tokspur::inspect_mtp_shared_q8_catalogue(sidecar,differing_target,embedding,output,tokspur::MtpTemplatePolicy::require_exact_identity)); });
    rejected([&] { static_cast<void>(tokspur::inspect_mtp_shared_q8_catalogue(sidecar,target,embedding,output,static_cast<tokspur::MtpTemplatePolicy>(99))); });

    // Independently mutate every vocabulary/merge/type/special-ID policy record.
    for (const auto& [key,field] : target_fields) if (key.starts_with("tokenizer.ggml.")) {
        auto changed = target_fields;
        size_t at = field.type==8 ? 8 : (field.type==9 ? (key=="tokenizer.ggml.token_type" ? 12 : 20) : 0);
        changed[key].bytes.at(at) ^= std::byte{1};
        const auto bytes = header(changed,{});
        tokspur::Gguf candidate(std::span(bytes),bytes.size());
        rejected([&] { static_cast<void>(inspect(sidecar,candidate)); });
    }
    auto extra_tokenizer = target_fields;
    extra_tokenizer["tokenizer.ggml.extra"] = {4,integer(0)};
    const auto extra_bytes = header(extra_tokenizer,{});
    tokspur::Gguf extra_target(std::span(extra_bytes),extra_bytes.size());
    rejected([&] { static_cast<void>(inspect(sidecar,extra_target)); });

    for (size_t mutation=0; mutation<7; ++mutation) {
        auto changed = original;
        auto tensors = requirements;
        if (mutation==0) changed["qwen4exp.nextn_shared_target_tensors"] = {7,integer(0,1)};
        if (mutation==1) changed.erase("qwen4exp.nextn_predict_layers");
        if (mutation==2) changed["unregistered.metadata"] = {4,integer(0)};
        if (mutation==3) tensors.front().shape.front() = 2528;
        if (mutation==4) tensors[15].encoding = 7; // Valid generic encoding, wrong closed profile.
        if (mutation==5) std::swap(tensors[0],tensors[1]);
        if (mutation==6) tensors.pop_back();
        const auto bytes = header(changed,tensors,tokspur::mtp_shared_q8_header_bytes);
        tokspur::Gguf candidate(std::span(bytes),tokspur::mtp_shared_q8_file_bytes);
        rejected([&] { static_cast<void>(inspect(candidate,target)); });
    }
    auto wrong_borrow = embedding;
    for (size_t mutation=0; mutation<5; ++mutation) {
        wrong_borrow = embedding;
        if (mutation==0) wrong_borrow.name = "other.weight";
        if (mutation==1) wrong_borrow.dimensions[1] -= 1;
        if (mutation==2) wrong_borrow.type = 7;
        if (mutation==3) wrong_borrow.bytes -= 1;
        if (mutation==4) wrong_borrow.elements -= 1;
        rejected([&] { static_cast<void>(inspect(sidecar,target,&wrong_borrow,&output)); });
    }
    rejected([&] { tokspur::Gguf truncated(std::span(sidecar_bytes).first(sidecar_bytes.size()-1),tokspur::mtp_shared_q8_file_bytes); });
    rejected([&] { tokspur::Gguf short_extent(std::span(sidecar_bytes),tokspur::mtp_shared_q8_file_bytes-1); });
}
} // namespace

int main() {
    try {
        controls();
        std::cout << "MTP catalogue checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "MTP catalogue check failed: " << error.what() << '\n';
        return 1;
    }
}
