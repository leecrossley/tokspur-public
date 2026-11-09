// DFlash2VerifyBudget adapts runtime/model/VerifyBudget.cpp from incoai/splash 1.4.0,
// commit 1482f6bbc2abf12375332725302019594150b6cb, under Apache-2.0.
// Tokspur supplies the graph, sampling and worker integration. See NOTICE and native/third_party/splash/.
#include "tokspur/dflash2_drafter.hpp"
#include "tokspur/dflash2_checkpoint.hpp"

#include "tokspur/derived_pack.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/json.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/operator_checkpoints.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <ranges>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace tokspur {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw FormatError(message);
}

inline bool finite_bits(float v) noexcept {
    return (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u;
}

struct ScopedFd {
    int fd = -1;
    explicit ScopedFd(int v) : fd(v) {}
    ~ScopedFd() { if (fd >= 0) ::close(fd); }
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
};

struct SafetensorMmap {
    const std::byte* data = nullptr;
    size_t size = 0;
    explicit SafetensorMmap(const std::filesystem::path& path) {
        std::error_code ec;
        const auto resolved = std::filesystem::canonical(path, ec);
        require(!ec && !resolved.empty(), "cannot resolve DFlash2 safetensors path");
        ScopedFd fd(::open(resolved.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        require(fd.fd >= 0, "cannot open DFlash2 safetensors file");
        struct stat st{};
        require(::fstat(fd.fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 16,
            "DFlash2 safetensors path is not a valid regular file");
        size = size_t(st.st_size);
        void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.fd, 0);
        require(p != MAP_FAILED, "cannot mmap DFlash2 safetensors file");
        data = static_cast<const std::byte*>(p);
    }
    ~SafetensorMmap() {
        if (data && size) ::munmap(const_cast<std::byte*>(data), size);
    }
    SafetensorMmap(const SafetensorMmap&) = delete;
    SafetensorMmap& operator=(const SafetensorMmap&) = delete;
};

void validate_policy(const SamplingPolicy& policy) {
    if (!std::isfinite(policy.temperature) || policy.temperature < 0 ||
        policy.top_k > maximum_sampling_vocabulary ||
        !std::isfinite(policy.top_p) || policy.top_p <= 0 || policy.top_p > 1 ||
        !std::isfinite(policy.min_p) || policy.min_p < 0 || policy.min_p > 1) {
        throw FormatError("invalid sampling policy");
    }
}

const Json& json_at(const Json& obj, std::string_view key) {
    const Json* found = obj.find(key);
    require(found != nullptr, "missing required JSON key in DFlash2 metadata");
    return *found;
}

double json_double(const Json& val) {
    const std::string s(val.number_lexeme());
    require(!s.empty(), "expected JSON number in DFlash2 metadata");
    char* end = nullptr;
    const double d = std::strtod(s.c_str(), &end);
    require(end == s.c_str() + s.size() && std::isfinite(d), "invalid JSON number in DFlash2 metadata");
    return d;
}

struct SafetensorEntry {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

inline float bf16_to_f32(uint16_t bits) noexcept {
    const uint32_t u = uint32_t(bits) << 16u;
    float out = 0.0f;
    std::memcpy(&out, &u, sizeof(float));
    return out;
}

inline uint16_t f32_to_f16_bits(float v) noexcept {
    _Float16 h = static_cast<_Float16>(v);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
}

inline float f16_bits_to_f32(uint16_t bits) noexcept {
    _Float16 h{};
    std::memcpy(&h, &bits, sizeof(uint16_t));
    return static_cast<float>(h);
}

void quantize_bf16_rows_to_q8_lanes(const uint16_t* src_bf16, uint32_t rows, uint32_t cols, std::span<std::byte> dst_packed) {
    require(cols >= 32 && cols % 32 == 0, "DFlash2 Q8_0 columns must be a multiple of 32");
    const auto layout = lanes::row_layout(8, cols);
    require(dst_packed.size() == uint64_t(rows) * layout.row_bytes, "DFlash2 Q8_0 packed byte mismatch");
    const uint32_t blocks = cols / 32u;
    const uint32_t hw = std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
    const uint32_t workers = rows >= 256u ? std::min<uint32_t>(hw, (rows + 63u) / 64u) : 1u;

    std::atomic<uint32_t> next_row{0};
    std::exception_ptr failure;
    std::mutex fail_mu;

    auto worker = [&]() {
        try {
            std::vector<std::byte> raw_row(layout.row_bytes);
            while (true) {
                const uint32_t r_begin = next_row.fetch_add(32u, std::memory_order_relaxed);
                if (r_begin >= rows) break;
                const uint32_t r_end = std::min(rows, r_begin + 32u);
                for (uint32_t r = r_begin; r < r_end; ++r) {
                    const uint16_t* row_src = src_bf16 + uint64_t(r) * cols;
                    for (uint32_t b = 0; b < blocks; ++b) {
                        const uint16_t* blk = row_src + uint64_t(b) * 32u;
                        float vals[32];
                        float amax = 0.0f;
                        for (uint32_t j = 0; j < 32u; ++j) {
                            const float f = bf16_to_f32(blk[j]);
                            require(std::isfinite(f), "nonfinite BF16 weight in DFlash2 checkpoint");
                            vals[j] = f;
                            amax = std::max(amax, std::fabs(f));
                        }
                        const uint16_t d_bits = f32_to_f16_bits(amax / 127.0f);
                        const float d = f16_bits_to_f32(d_bits);
                        require(std::isfinite(d) && d >= 0.0f, "nonfinite Q8_0 scale in DFlash2 quantisation");
                        const float inv_d = d > 0.0f ? (1.0f / d) : 0.0f;
                        std::byte* dst_blk = raw_row.data() + uint64_t(b) * 34u;
                        dst_blk[0] = std::byte(d_bits & 0xffu);
                        dst_blk[1] = std::byte((d_bits >> 8u) & 0xffu);
                        for (uint32_t j = 0; j < 32u; ++j) {
                            const int q = std::clamp(int(std::nearbyint(vals[j] * inv_d)), -127, 127);
                            dst_blk[2u + j] = static_cast<std::byte>(static_cast<int8_t>(q));
                        }
                    }
                    lanes::pack_rows(layout, raw_row, dst_packed.subspan(uint64_t(r) * layout.row_bytes, layout.row_bytes));
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(fail_mu);
            if (!failure) failure = std::current_exception();
        }
    };

    if (workers <= 1u) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (uint32_t w = 0; w < workers; ++w) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
    }
    if (failure) std::rethrow_exception(failure);
}

std::string read_small_text_file(const std::filesystem::path& path, size_t max_bytes) {
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(path, ec);
    require(!ec && !resolved.empty(), "cannot resolve DFlash2 config.json path");
    ScopedFd fd(::open(resolved.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    require(fd.fd >= 0, "cannot open DFlash2 config.json");
    struct stat st{};
    require(::fstat(fd.fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0 && size_t(st.st_size) <= max_bytes,
        "invalid DFlash2 config.json file");
    std::string text(size_t(st.st_size), '\0');
    size_t got_total = 0;
    while (got_total < text.size()) {
        const ssize_t got = ::read(fd.fd, text.data() + got_total, text.size() - got_total);
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "cannot read DFlash2 config.json");
        got_total += size_t(got);
    }
    return text;
}

SampledSparseLaw build_selector_law(uint32_t vocab_size, std::span<const uint32_t, 16> cands,
                                    std::span<const float, 16> scores, const SamplingPolicy& policy) {
    validate_policy(policy);
    const auto canon = [](float v) -> float { return v == 0.0f ? 0.0f : v; };
    for (size_t i = 0; i < 16; ++i) {
        require(cands[i] < vocab_size, "DFlash2 selector candidate exceeds vocabulary");
        require(std::isfinite(scores[i]), "DFlash2 selector produced nonfinite score");
    }
    std::array<uint32_t, 16> order{};
    for (uint32_t i = 0; i < 16; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const float sa = canon(scores[a]);
        const float sb = canon(scores[b]);
        return sa > sb || (sa == sb && cands[a] < cands[b]);
    });
    for (size_t i = 1; i < 16; ++i) {
        require(cands[order[i]] != cands[order[i - 1]], "DFlash2 selector emitted duplicate candidate");
    }

    if (policy.temperature == 0.0) {
        const uint32_t best = cands[order[0]];
        const uint64_t unit = sampled_path_grid_size;
        return SampledSparseLaw::from_grid_units(vocab_size, std::span(&best, 1), std::span(&unit, 1));
    }

    const double max_s = double(canon(scores[order[0]]));
    std::array<uint32_t, 16> ids{};
    std::array<double, 16> weights{};
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < 16; ++i) {
        const uint32_t idx = order[i];
        const double w = (idx == order[0]) ? 1.0 : std::exp((double(canon(scores[idx])) - max_s) / policy.temperature);
        if (!(w > 0.0)) continue;
        ids[count] = cands[idx];
        weights[count] = w;
        sum += w;
        ++count;
    }
    require(count > 0 && std::isfinite(sum) && sum > 0.0, "DFlash2 selector weight normalisation failed");

    std::array<uint32_t, 16> final_ids{};
    std::array<uint64_t, 16> units{};
    size_t out_n = 0;
    double cumulative_prob = 0.0;
    uint64_t prev_end = 0;
    for (size_t i = 0; i < count; ++i) {
        const double p = weights[i] / sum;
        cumulative_prob += p;
        const uint64_t end = (i + 1 == count)
            ? sampled_path_grid_size
            : std::min<uint64_t>(sampled_path_grid_size, uint64_t(std::llround(std::min(1.0, cumulative_prob) * double(sampled_path_grid_size))));
        if (end > prev_end) {
            final_ids[out_n] = ids[i];
            units[out_n] = end - prev_end;
            prev_end = end;
            ++out_n;
        }
    }
    if (out_n == 0) {
        final_ids[0] = ids[0];
        units[0] = sampled_path_grid_size;
        out_n = 1;
    } else if (prev_end < sampled_path_grid_size) {
        units[out_n - 1] += (sampled_path_grid_size - prev_end);
    }
    return SampledSparseLaw::from_grid_units(vocab_size, std::span(final_ids).first(out_n), std::span(units).first(out_n));
}

} // namespace

void DFlash2ContextStage::prepare(GpuView storage, uint32_t width) {
    require(!storage_.owner && storage.owner && width && width <= 16u * 2560u && storage.bytes == uint64_t(8) * width * sizeof(float), "DFlash2 pending context storage bound");
    storage_ = storage;
    width_ = width;
}

void DFlash2ContextStage::stage(MetalGraph& gpu, GpuView source, uint32_t rows, uint32_t start) {
    require(storage_.owner && !rows_ && rows && rows <= 8 && start <= UINT32_MAX - rows, "DFlash2 pending context stage bound or undrained rows");
    const auto bytes = uint64_t(rows) * width_ * sizeof(float);
    const auto input = gpu.slice(source, 0, bytes);
    const auto values = std::span(gpu.host_floats(input), uint64_t(rows) * width_);
    for (const auto value : values) require(finite_bits(value), "nonfinite DFlash2 pending target context");
    const GraphCopy copy{input, gpu.slice(storage_, 0, bytes)};
    gpu.copy(std::span(&copy, 1));
    rows_ = rows;
    start_ = start;
}

void DFlash2ContextStage::truncate(uint32_t position) noexcept {
    if (!rows_ || position <= start_) clear();
    else rows_ = std::min(rows_, position - start_);
}

DFlash2Drafter::DFlash2Drafter(MetalGraph& gpu, const std::filesystem::path& directory,
                               GpuView borrowed_output_weight, uint32_t known_token_capacity,
                               const EmbedFn& embed_token, DFlash2Precision precision, bool defer_context,
                               bool bind_checkpoint_identity)
    : precision_(precision), defer_context_(defer_context), gpu_(gpu), borrowed_output_weight_(borrowed_output_weight) {
    try {
        require(precision == DFlash2Precision::q8_0 || precision == DFlash2Precision::bf16, "unsupported DFlash2 precision");
        require(borrowed_output_weight_.bytes > 0, "DFlash2 requires borrowed target output weight");
        require(bool(embed_token), "DFlash2 requires target embedding callback");
        const auto cfg_text = read_small_text_file(directory / "config.json", 1024 * 1024);
        checkpoint_sha256_ = derived_pack::hash(cfg_text);
        const auto cfg = parse_json(cfg_text, {1024 * 1024, 16384, 16});

        const auto* archs = cfg.find("architectures");
        require(archs && archs->kind() == Json::Kind::array && archs->array().size() == 1 &&
            archs->array()[0].string() == "DFlash2DraftModel", "unsupported DFlash2 architecture");
        require(!json_at(cfg, "is_causal").boolean() && !json_at(cfg, "attention_bias").boolean(),
            "DFlash2 requires non-causal unbiassed attention");
        require(json_at(cfg, "hidden_act").string() == "silu", "DFlash2 requires silu MLP activation");

        hidden_size_ = uint32_t(json_at(cfg, "hidden_size").unsigned_integer());
        intermediate_size_ = uint32_t(json_at(cfg, "intermediate_size").unsigned_integer());
        num_layers_ = uint32_t(json_at(cfg, "num_hidden_layers").unsigned_integer());
        q_heads_ = uint32_t(json_at(cfg, "num_attention_heads").unsigned_integer());
        kv_heads_ = uint32_t(json_at(cfg, "num_key_value_heads").unsigned_integer());
        head_dim_ = uint32_t(json_at(cfg, "head_dim").unsigned_integer());
        sliding_window_ = uint32_t(json_at(cfg, "sliding_window").unsigned_integer());
        vocab_size_ = uint32_t(json_at(cfg, "vocab_size").unsigned_integer());
        rms_eps_ = float(json_double(json_at(cfg, "rms_norm_eps")));
        rope_theta_ = float(json_double(json_at(json_at(cfg, "rope_parameters"), "rope_theta")));

        const auto& df = json_at(cfg, "dflash_config");
        require(json_at(df, "block_size").unsigned_integer() == 8 &&
            json_at(df, "conv_group_size").unsigned_integer() == 16 &&
            json_at(df, "conv_kernel_size").unsigned_integer() == 2 &&
            json_at(df, "selector_rank").unsigned_integer() == 256 &&
            json_at(df, "selector_top_k").unsigned_integer() == 16,
            "DFlash2 block/conv/selector config mismatch");
        mask_token_id_ = uint32_t(json_at(df, "mask_token_id").unsigned_integer());
        selector_rank_ = 256;
        selector_top_k_ = 16;

        const auto& layer_ids = json_at(df, "target_layer_ids").array();
        require(layer_ids.size() == 5, "DFlash2 requires 5 target_layer_ids");
        target_layer_ids_.reserve(layer_ids.size());
        for (const auto& item : layer_ids) {
            const uint32_t lid = uint32_t(item.unsigned_integer());
            require(lid < 48 && (target_layer_ids_.empty() || lid > target_layer_ids_.back()),
                "DFlash2 target_layer_ids must be strictly ascending within [0, 48)");
            target_layer_ids_.push_back(lid);
        }
        require(target_layer_ids_ == std::vector<uint32_t>{5, 15, 25, 35, 45}, "DFlash2 requires target layers [5, 15, 25, 35, 45]");

        require(hidden_size_ == 2560 && intermediate_size_ == 7680 && num_layers_ == 5 &&
            q_heads_ == 32 && kv_heads_ == 8 && head_dim_ == 128 && sliding_window_ == 2048 &&
            vocab_size_ == 248320 && mask_token_id_ < vocab_size_ && rms_eps_ > 0.0f && rope_theta_ > 0.0f,
            "DFlash2 geometry differs from compiled Metal 4 kernel specialisation");

        const uint64_t expected_out_bytes = uint64_t(hidden_size_ / 32u) * 34ull * vocab_size_;
        require(borrowed_output_weight_.bytes == expected_out_bytes, "borrowed output weight shape mismatch for DFlash2");
        active_output_weight_ = borrowed_output_weight_;
        active_output_vocab_ = vocab_size_;
        max_drafts_ = dflash2_default_max_drafts;

        mask_embedding_.resize(hidden_size_);
        embed_token(mask_token_id_, mask_embedding_);
        for (float v : mask_embedding_) require(finite_bits(v), "nonfinite DFlash2 mask token embedding");
        host_input_8_.resize(8ull * hidden_size_);
        for (uint32_t r = 1; r < 8u; ++r) {
            std::copy(mask_embedding_.begin(), mask_embedding_.end(), host_input_8_.begin() + ptrdiff_t(r * hidden_size_));
        }
        host_selector_out_.resize(7ull * 16ull * 8ull + 7ull * 256ull * 4ull);

        SafetensorMmap st_file(directory / "model.safetensors");
        state_identity_sha256_ = dflash2_checkpoint_identity(bind_checkpoint_identity, checkpoint_sha256_,
            std::string_view(reinterpret_cast<const char*>(st_file.data), size_t(st_file.size)), derived_pack::hash);
        uint64_t header_len = 0;
        std::memcpy(&header_len, st_file.data, sizeof(uint64_t));
        require(header_len > 2 && header_len <= 16ull * 1024ull * 1024ull && 8ull + header_len <= st_file.size,
            "invalid DFlash2 safetensors header length");
        const std::string_view header_sv(reinterpret_cast<const char*>(st_file.data + 8), size_t(header_len));
        const auto header_json = parse_json(header_sv, {16 * 1024 * 1024, 262144, 16});
        require(header_json.kind() == Json::Kind::object, "DFlash2 safetensors header must be a JSON object");

        const uint64_t payload_offset = 8ull + header_len;
        const uint64_t payload_bytes = st_file.size - payload_offset;
        std::map<std::string, SafetensorEntry> tensors;
        for (const auto& [name, meta] : header_json.object()) {
            if (name == "__metadata__") continue;
            SafetensorEntry e;
            e.dtype = std::string(json_at(meta, "dtype").string());
            for (const auto& dim : json_at(meta, "shape").array()) e.shape.push_back(dim.unsigned_integer());
            const auto& offs = json_at(meta, "data_offsets").array();
            require(offs.size() == 2, "invalid safetensors data_offsets");
            e.begin = offs[0].unsigned_integer();
            e.end = offs[1].unsigned_integer();
            require(e.begin <= e.end && e.end <= payload_bytes && (e.begin % 2ull) == 0ull,
                "DFlash2 safetensors tensor span out of bounds");
            tensors.emplace(name, std::move(e));
        }

        const auto require_bf16 = [&](const std::string& name, std::initializer_list<uint64_t> expected_shape) -> const uint16_t* {
            const auto it = tensors.find(name);
            require(it != tensors.end(), ("missing DFlash2 tensor: " + name).c_str());
            const auto& e = it->second;
            require(e.dtype == "BF16" && std::ranges::equal(e.shape, expected_shape),
                ("DFlash2 tensor dtype/shape mismatch: " + name).c_str());
            uint64_t elems = 1;
            for (uint64_t d : expected_shape) elems = checked_mul(elems, d);
            require(e.end - e.begin == checked_mul(elems, 2ull), ("DFlash2 tensor byte size mismatch: " + name).c_str());
            return reinterpret_cast<const uint16_t*>(st_file.data + payload_offset + e.begin);
        };

        const auto track_weight = [&](GpuView v) -> GpuView {
            owned_views_.push_back(v);
            weight_bytes_ = checked_add(weight_bytes_, v.bytes);
            return v;
        };
        const auto track_workspace = [&](GpuView v) -> GpuView {
            owned_views_.push_back(v);
            workspace_bytes_ = checked_add(workspace_bytes_, v.bytes);
            return v;
        };
        const auto track_priming = [&](GpuView v) -> GpuView {
            owned_views_.push_back(v);
            priming_batch_bytes_ = checked_add(priming_batch_bytes_, v.bytes);
            return v;
        };

        const auto upload_q8_lanes = [&](const uint16_t* bf16_src, uint32_t rows, uint32_t cols, const char* label) -> GpuView {
            const auto layout = lanes::row_layout(8, cols);
            std::vector<std::byte> packed(uint64_t(rows) * layout.row_bytes);
            quantize_bf16_rows_to_q8_lanes(bf16_src, rows, cols, packed);
            const auto view = track_weight(gpu_.upload(packed, label, /*immutable=*/true));
            gpu_.register_lane_matrix(view, {8, cols, rows, 1});
            return view;
        };

        const auto upload_bf16_matrix = [&](const uint16_t* bf16_src, uint32_t rows, uint32_t cols, const char* label) -> GpuView {
            const uint64_t elems = checked_mul(uint64_t(rows), uint64_t(cols));
            for (uint64_t i = 0; i < elems; ++i) {
                require(std::isfinite(bf16_to_f32(bf16_src[i])), "nonfinite BF16 weight in DFlash2 checkpoint");
            }
            return track_weight(gpu_.upload(std::as_bytes(std::span(bf16_src, size_t(elems))), label, /*immutable=*/true));
        };

        const auto upload_matrix = [&](const uint16_t* bf16_src, uint32_t rows, uint32_t cols, const char* label) -> GpuView {
            return precision_ == DFlash2Precision::bf16
                ? upload_bf16_matrix(bf16_src, rows, cols, label)
                : upload_q8_lanes(bf16_src, rows, cols, label);
        };

        const auto upload_f32_from_bf16 = [&](const uint16_t* bf16_src, size_t count, const char* label) -> GpuView {
            std::vector<float> f32(count);
            for (size_t i = 0; i < count; ++i) {
                f32[i] = bf16_to_f32(bf16_src[i]);
                require(std::isfinite(f32[i]), "nonfinite BF16 parameter in DFlash2 checkpoint");
            }
            return track_weight(gpu_.upload(std::as_bytes(std::span(f32)), label, /*immutable=*/true));
        };

        gpu_.prepare_dflash2();

        const uint32_t target_concat = uint32_t(target_layer_ids_.size()) * hidden_size_; // 12800
        fc_proj_ = upload_matrix(require_bf16("fc.weight", {hidden_size_, target_concat}), hidden_size_, target_concat, "dflash2.fc");
        hidden_norm_ = upload_f32_from_bf16(require_bf16("hidden_norm.weight", {hidden_size_}), hidden_size_, "dflash2.hidden_norm");
        final_norm_ = upload_f32_from_bf16(require_bf16("norm.weight", {hidden_size_}), hidden_size_, "dflash2.norm");
        selector_proj_ = upload_matrix(require_bf16("candidate_selector.hidden_projection.weight", {selector_rank_, hidden_size_}),
            selector_rank_, hidden_size_, "dflash2.selector_proj");

        const uint64_t cb_elems = uint64_t(vocab_size_) * selector_rank_;
        const auto* pred_cb = require_bf16("candidate_selector.predecessor_codebook", {vocab_size_, selector_rank_});
        const auto* succ_cb = require_bf16("candidate_selector.successor_codebook", {vocab_size_, selector_rank_});
        predecessor_codebook_ = track_weight(gpu_.upload(std::as_bytes(std::span(pred_cb, size_t(cb_elems))), "dflash2.pred_cb", /*immutable=*/true));
        successor_codebook_ = track_weight(gpu_.upload(std::as_bytes(std::span(succ_cb, size_t(cb_elems))), "dflash2.succ_cb", /*immutable=*/true));

        std::array<float, 64> inv_freq{};
        for (uint32_t p = 0; p < 64u; ++p) {
            inv_freq[p] = float(std::pow(double(rope_theta_), -2.0 * double(p) / double(head_dim_)));
        }

        layers_.resize(num_layers_);
        const uint64_t ring_bytes = uint64_t(kv_heads_) * sliding_window_ * head_dim_ * sizeof(uint16_t);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            const std::string pfx = "layers." + std::to_string(l) + ".";
            auto& layer = layers_[l];
            layer.input_norm = upload_f32_from_bf16(require_bf16(pfx + "input_layernorm.weight", {hidden_size_}), hidden_size_, "dflash2.in_norm");
            layer.attn_conv_base = upload_f32_from_bf16(require_bf16(pfx + "attention_conv.base_kernel", {2, 2, hidden_size_}), 4ull * hidden_size_, "dflash2.attn_conv_base");
            layer.attn_conv_proj = upload_matrix(require_bf16(pfx + "attention_conv.kernel_projection.weight", {640, hidden_size_}), 640, hidden_size_, "dflash2.attn_conv_proj");

            const auto* q_src = require_bf16(pfx + "self_attn.q_proj.weight", {4096, hidden_size_});
            const auto* k_src = require_bf16(pfx + "self_attn.k_proj.weight", {1024, hidden_size_});
            const auto* v_src = require_bf16(pfx + "self_attn.v_proj.weight", {1024, hidden_size_});
            {
                std::vector<uint16_t> qkv_bf16(6144ull * hidden_size_);
                std::memcpy(qkv_bf16.data(), q_src, 4096ull * hidden_size_ * sizeof(uint16_t));
                std::memcpy(qkv_bf16.data() + 4096ull * hidden_size_, k_src, 1024ull * hidden_size_ * sizeof(uint16_t));
                std::memcpy(qkv_bf16.data() + 5120ull * hidden_size_, v_src, 1024ull * hidden_size_ * sizeof(uint16_t));
                layer.qkv_proj = upload_matrix(qkv_bf16.data(), 6144, hidden_size_, "dflash2.qkv_proj");
                layer.kv_proj = upload_matrix(qkv_bf16.data() + 4096ull * hidden_size_, 2048, hidden_size_, "dflash2.kv_proj");
            }

            {
                const auto* qn = require_bf16(pfx + "self_attn.q_norm.weight", {head_dim_});
                const auto* kn = require_bf16(pfx + "self_attn.k_norm.weight", {head_dim_});
                std::array<float, 320> qk_freq{};
                for (uint32_t i = 0; i < 128u; ++i) qk_freq[i] = bf16_to_f32(qn[i]);
                for (uint32_t i = 0; i < 128u; ++i) qk_freq[128u + i] = bf16_to_f32(kn[i]);
                for (uint32_t i = 0; i < 64u; ++i) qk_freq[256u + i] = inv_freq[i];
                layer.qk_norm_freq = track_weight(gpu_.upload(std::as_bytes(std::span(qk_freq)), "dflash2.qk_norm_freq", /*immutable=*/true));
            }

            layer.o_proj = upload_matrix(require_bf16(pfx + "self_attn.o_proj.weight", {hidden_size_, 4096}), hidden_size_, 4096, "dflash2.o_proj");
            layer.post_attn_norm = upload_f32_from_bf16(require_bf16(pfx + "post_attention_layernorm.weight", {hidden_size_}), hidden_size_, "dflash2.post_norm");
            layer.mlp_conv_base = upload_f32_from_bf16(require_bf16(pfx + "mlp_conv.base_kernel", {2, 2, hidden_size_}), 4ull * hidden_size_, "dflash2.mlp_conv_base");
            layer.mlp_conv_proj = upload_matrix(require_bf16(pfx + "mlp_conv.kernel_projection.weight", {640, hidden_size_}), 640, hidden_size_, "dflash2.mlp_conv_proj");
            layer.gate_proj = upload_matrix(require_bf16(pfx + "mlp.gate_proj.weight", {intermediate_size_, hidden_size_}), intermediate_size_, hidden_size_, "dflash2.gate_proj");
            layer.up_proj = upload_matrix(require_bf16(pfx + "mlp.up_proj.weight", {intermediate_size_, hidden_size_}), intermediate_size_, hidden_size_, "dflash2.up_proj");
            layer.down_proj = upload_matrix(require_bf16(pfx + "mlp.down_proj.weight", {hidden_size_, intermediate_size_}), hidden_size_, intermediate_size_, "dflash2.down_proj");

            layer.ring_k = track_workspace(gpu_.zeros(ring_bytes, "dflash2.ring_k"));
            layer.ring_v = track_workspace(gpu_.zeros(ring_bytes, "dflash2.ring_v"));
        }

        priming_cap_ = std::min<uint32_t>(2048u, std::max<uint32_t>(8u, known_token_capacity ? known_token_capacity : 512u));
        prime_projected_ = track_priming(gpu_.zeros(uint64_t(priming_cap_) * hidden_size_ * sizeof(float), "dflash2.prime_proj"));
        prime_hidden_ = track_priming(gpu_.zeros(uint64_t(priming_cap_) * hidden_size_ * sizeof(float), "dflash2.prime_hidden"));
        prime_kv_ = track_priming(gpu_.zeros(uint64_t(priming_cap_) * 2048ull * sizeof(float), "dflash2.prime_kv"));
        if (defer_context_) pending_context_.prepare(track_workspace(gpu_.zeros(8ull * target_concat * sizeof(float), "dflash2.pending_context")), target_concat);

        draft_hidden_[0] = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.hidden0"));
        draft_hidden_[1] = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.hidden1"));
        draft_norm_ = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.norm_rows"));
        draft_dynamic_ = track_workspace(gpu_.zeros(8ull * 640ull * sizeof(float), "dflash2.dynamic"));
        draft_conv_ = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.conv"));
        draft_qkv_ = track_workspace(gpu_.zeros(8ull * 6144ull * sizeof(float), "dflash2.qkv"));
        draft_queries_ = track_workspace(gpu_.zeros(32ull * 8ull * 128ull * sizeof(uint16_t), "dflash2.queries"));
        draft_query_keys_ = track_workspace(gpu_.zeros(8ull * 8ull * 128ull * sizeof(uint16_t), "dflash2.qkeys"));
        draft_query_values_ = track_workspace(gpu_.zeros(8ull * 128ull * 8ull * sizeof(uint16_t), "dflash2.qvals"));
        draft_partials_ = track_workspace(gpu_.zeros(8ull * 16ull * (32ull * 128ull + 64ull) * sizeof(float), "dflash2.partials"));
        draft_attn_out_ = track_workspace(gpu_.zeros(8ull * 4096ull * sizeof(float), "dflash2.attn_out"));
        draft_proj_ = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.proj"));
        draft_residual_ = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.residual"));
        draft_inter_ = track_workspace(gpu_.zeros(8ull * intermediate_size_ * sizeof(float), "dflash2.inter"));
        draft_final_ = track_workspace(gpu_.zeros(8ull * hidden_size_ * sizeof(float), "dflash2.final"));
        draft_logits_7_ = track_workspace(gpu_.zeros(7ull * vocab_size_ * sizeof(float), "dflash2.logits7"));
        draft_selector_hidden_7_ = track_workspace(gpu_.zeros(7ull * selector_rank_ * sizeof(float), "dflash2.sel_hidden7"));
        draft_top_ids_ = track_workspace(gpu_.zeros(7ull * 8ull * 16ull * sizeof(uint32_t), "dflash2.top_ids"));
        draft_top_vals_ = track_workspace(gpu_.zeros(7ull * 8ull * 16ull * sizeof(float), "dflash2.top_vals"));
        draft_selector_out_ = track_workspace(gpu_.zeros(host_selector_out_.size(), "dflash2.selector_out"));
    } catch (...) {
        for (auto it = owned_views_.rbegin(); it != owned_views_.rend(); ++it) {
            try { gpu_.release(*it); } catch (...) {}
        }
        throw;
    }
}

DFlash2Drafter::~DFlash2Drafter() {
    for (auto it = owned_views_.rbegin(); it != owned_views_.rend(); ++it) {
        try { gpu_.release(*it); } catch (...) {}
    }
}

void DFlash2Drafter::reset() noexcept {
    context_len_ = 0;
    pending_context_.clear();
}

uint64_t DFlash2Drafter::ssd_cache_bytes(uint32_t rows) const {
    require(rows <= sliding_window_, "DFlash2 checkpoint rows exceed ring capacity");
    return uint64_t(rows) * kv_heads_ * head_dim_ * sizeof(uint16_t) * 2 * num_layers_;
}

bool DFlash2Drafter::read_ssd_cache_span(uint64_t offset, std::span<std::byte> target) {
    // Materialise pending accepted features before recording the context frontier. The same flush is already required before the next draft.
    static_cast<void>(flush_context());
    const DFlash2CheckpointLayout layout{first_absolute_position(), cache_rows(), sliding_window_, num_layers_, kv_heads_, head_dim_};
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    if (gpu_.carrier_gpu_readback_enabled()) {
        std::vector<CarrierStateRead> reads; reads.reserve(8192);
        const bool valid = layout.visit(offset, target.size(), [&](uint32_t layer, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
            if (reads.size() == 8192) return false;
            reads.push_back({value ? layers_[layer].ring_v : layers_[layer].ring_k, physical, bytes, request}); return true;
        });
        if (!valid) return false;
        if (!target.empty()) gpu_.read_carrier_state_segments(reads, target);
        return true;
    }
#endif
    return layout.visit(offset, target.size(), [&](uint32_t layer, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
        gpu_.read_bytes(value ? layers_[layer].ring_v : layers_[layer].ring_k, physical, target.subspan(size_t(request), size_t(bytes)));
        return true;
    });
}

bool DFlash2Drafter::write_ssd_cache_span(uint32_t first, uint32_t rows, uint64_t offset, std::span<const std::byte> source) {
    const DFlash2CheckpointLayout layout{first, rows, sliding_window_, num_layers_, kv_heads_, head_dim_};
    if (!layout.valid() || offset > layout.bytes() || source.size() > layout.bytes() - offset) return false;
    if (offset && (context_len_ != first + rows || pending_context_.rows())) return false;
    if (!offset) {
        context_len_ = first + rows;
        pending_context_.clear();
    }
    return layout.visit(offset, source.size(), [&](uint32_t layer, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
        gpu_.write_bytes(value ? layers_[layer].ring_v : layers_[layer].ring_k, physical, source.subspan(size_t(request), size_t(bytes)));
        return true;
    });
}

void DFlash2Drafter::truncate(uint32_t position) {
    require(position <= context_len_, "cannot extend DFlash2 context via truncate");
    context_len_ = position;
    pending_context_.truncate(position);
}

void DFlash2Drafter::set_max_drafts(uint32_t max_drafts) {
    require(max_drafts >= 1u && max_drafts <= 7u, "DFlash2 max_drafts must be in 1..7");
    max_drafts_ = max_drafts;
}

void DFlash2Drafter::configure_draft_vocab_slice(uint32_t slice_vocab) {
    if (slice_vocab == 0u || slice_vocab == vocab_size_) {
        active_output_weight_ = borrowed_output_weight_;
        active_output_vocab_ = vocab_size_;
        return;
    }
    require(slice_vocab >= 256u && slice_vocab < vocab_size_ && (slice_vocab % 32u) == 0u,
        "DFlash2 draft_vocab_slice must be 0, full vocabulary, or a multiple of 32 in [256, vocab_size)");
    active_output_weight_ = gpu_.suballocate_draft_output_slice(borrowed_output_weight_, slice_vocab, hidden_size_);
    active_output_vocab_ = slice_vocab;
}

void DFlash2Drafter::dense_proj(GpuView matrix, GpuRowsF32 in, GpuRowsF32 out, bool swiglu) {
    if (precision_ == DFlash2Precision::bf16) gpu_.dflash2_dense_bf16(matrix, in, out, swiglu);
    else gpu_.dflash2_dense_q8(matrix, in, out, swiglu);
}

void DFlash2Drafter::encode_context_commit_chunk(GpuView captured_target_hidden, uint32_t rows, uint32_t start_position) {
    const uint32_t target_concat = uint32_t(target_layer_ids_.size()) * hidden_size_;
    const uint64_t row_bytes = uint64_t(target_concat) * sizeof(float);
    const auto in_slice = gpu_.slice(captured_target_hidden, 0, uint64_t(rows) * row_bytes);
    const auto proj_slice = gpu_.slice(prime_projected_, 0, uint64_t(rows) * hidden_size_ * sizeof(float));
    const auto hid_slice = gpu_.slice(prime_hidden_, 0, uint64_t(rows) * hidden_size_ * sizeof(float));
    const auto kv_slice = gpu_.slice(prime_kv_, 0, uint64_t(rows) * 2048ull * sizeof(float));

    const GpuRowsF32 in_rows{in_slice, rows, target_concat, row_bytes};
    const GpuRowsF32 proj_rows{proj_slice, rows, hidden_size_, uint64_t(hidden_size_) * sizeof(float)};
    const GpuRowsF32 hid_rows{hid_slice, rows, hidden_size_, uint64_t(hidden_size_) * sizeof(float)};
    const GpuRowsF32 kv_rows{kv_slice, rows, 2048u, 2048ull * sizeof(float)};

    dense_proj(fc_proj_, in_rows, proj_rows, false);
    gpu_.rms_rows(proj_rows, hidden_norm_, hid_rows, hidden_size_, rms_eps_, true);
    for (uint32_t l = 0; l < num_layers_; ++l) {
        dense_proj(layers_[l].kv_proj, hid_rows, kv_rows, false);
        gpu_.dflash2_context_kv_write(kv_slice, layers_[l].qk_norm_freq, layers_[l].ring_k, layers_[l].ring_v,
            rows, start_position, sliding_window_, rms_eps_);
    }
}

void DFlash2Drafter::stage_context_commit(GpuView captured_target_hidden, uint32_t rows, uint32_t start_position) {
    require(defer_context_, "DFlash2 deferred context is not enabled");
    require(rows > 0 && rows <= 8u, "DFlash2 stage_context_commit requires 1..8 rows");
    require(start_position == context_len_, "DFlash2 context continuity mismatch in stage_context_commit");
    const uint32_t target_concat = uint32_t(target_layer_ids_.size()) * hidden_size_;
    const uint64_t row_bytes = uint64_t(target_concat) * sizeof(float);
    require(captured_target_hidden.bytes >= uint64_t(rows) * row_bytes, "DFlash2 captured target hidden buffer too small");
    pending_context_.stage(gpu_, captured_target_hidden, rows, start_position);
    context_len_ = start_position + rows;
}

BatchResult DFlash2Drafter::flush_context() {
    if (!pending_context_.rows()) return {};
    struct ScopeGuard {
        MetalGraph& g;
        ~ScopeGuard() { try { g.set_dflash2_scope(false); } catch (...) {} }
    };
    TOKSPUR_OPERATOR_CONTEXT(gpu_, "dflash2_context", -1, pending_context_.start(), pending_context_.rows());
    gpu_.begin();
    ScopeGuard guard{gpu_};
    gpu_.set_dflash2_scope(true);
    try {
        encode_context_commit_chunk(pending_context_.storage(), pending_context_.rows(), pending_context_.start());
        TOKSPUR_OPERATOR_CHECKPOINT(gpu_, "dflash2", "flush_context", {pending_context_.rows(), hidden_size_, num_layers_, 0});
        const auto batch = gpu_.submit();
        pending_context_.clear();
        return batch;
    } catch (...) {
        try { gpu_.abort(); } catch (...) {}
        throw;
    }
}

BatchResult DFlash2Drafter::commit_context(GpuView captured_target_hidden, uint32_t rows, uint32_t start_position,
                                           uint64_t remaining_after) {
    require(rows > 0, "DFlash2 commit_context requires at least one row");
    require(start_position <= context_len_, "DFlash2 commit_context start_position leaves a context gap");
    const uint32_t target_concat = uint32_t(target_layer_ids_.size()) * hidden_size_;
    const uint64_t row_bytes = uint64_t(target_concat) * sizeof(float);
    require(captured_target_hidden.bytes >= uint64_t(rows) * row_bytes, "DFlash2 captured target hidden buffer too small");
    const uint32_t max_chunk = gpu_.prefill_dense_nax() ? priming_cap_ : std::min<uint32_t>(priming_cap_, 512u);
    const auto plan = dflash2_context_commit_plan(rows, remaining_after, sliding_window_);
    BatchResult total{};
    if (plan.discard_pending) pending_context_.clear();
    else total = flush_context();
    struct ScopeGuard {
        MetalGraph& g;
        ~ScopeGuard() { try { g.set_dflash2_scope(false); } catch (...) {} }
    };

    for (uint32_t offset = plan.keep_from; offset < rows;) {
        const uint32_t chunk = std::min<uint32_t>(max_chunk, rows - offset);
        const uint32_t pos = start_position + offset;
        const auto in_slice = gpu_.slice(captured_target_hidden, uint64_t(offset) * row_bytes, uint64_t(chunk) * row_bytes);
        const auto proj_slice = gpu_.slice(prime_projected_, 0, uint64_t(chunk) * hidden_size_ * sizeof(float));
        const auto hid_slice = gpu_.slice(prime_hidden_, 0, uint64_t(chunk) * hidden_size_ * sizeof(float));
        const auto kv_slice = gpu_.slice(prime_kv_, 0, uint64_t(chunk) * 2048ull * sizeof(float));

        const GpuRowsF32 in_rows{in_slice, chunk, target_concat, row_bytes};
        const GpuRowsF32 proj_rows{proj_slice, chunk, hidden_size_, uint64_t(hidden_size_) * sizeof(float)};
        const GpuRowsF32 hid_rows{hid_slice, chunk, hidden_size_, uint64_t(hidden_size_) * sizeof(float)};
        const GpuRowsF32 kv_rows{kv_slice, chunk, 2048u, 2048ull * sizeof(float)};

        TOKSPUR_OPERATOR_CONTEXT(gpu_, "dflash2_context", -1, pos, chunk);
        gpu_.begin();
        ScopeGuard guard{gpu_};
        gpu_.set_dflash2_scope(true);
        BatchResult part{};
        try {
            if (precision_ == DFlash2Precision::bf16 || chunk <= 8u) {
                dense_proj(fc_proj_, in_rows, proj_rows, false);
                gpu_.rms_rows(proj_rows, hidden_norm_, hid_rows, hidden_size_, rms_eps_, true);
                for (uint32_t l = 0; l < num_layers_; ++l) {
                    dense_proj(layers_[l].kv_proj, hid_rows, kv_rows, false);
                    gpu_.dflash2_context_kv_write(kv_slice, layers_[l].qk_norm_freq, layers_[l].ring_k, layers_[l].ring_v,
                        chunk, pos, sliding_window_, rms_eps_);
                }
            } else {
                gpu_.dense_mm_q8(PrefillWeightLayout::q8_lanes_v1, fc_proj_, in_rows, proj_rows, false);
                gpu_.rms_rows(proj_rows, hidden_norm_, hid_rows, hidden_size_, rms_eps_, true);
                const bool can_reuse = gpu_.prefill_dense_nax();
                for (uint32_t l = 0; l < num_layers_; ++l) {
                    gpu_.dense_mm_q8(PrefillWeightLayout::q8_lanes_v1, layers_[l].kv_proj, hid_rows, kv_rows, can_reuse && l > 0u);
                    gpu_.dflash2_context_kv_write(kv_slice, layers_[l].qk_norm_freq, layers_[l].ring_k, layers_[l].ring_v,
                        chunk, pos, sliding_window_, rms_eps_);
                }
            }
            TOKSPUR_OPERATOR_CHECKPOINT(gpu_, "dflash2", "commit_context_chunk", {chunk, target_concat, hidden_size_, num_layers_});
            part = gpu_.submit();
        } catch (...) {
            try { gpu_.abort(); } catch (...) {}
            throw;
        }
        total.dispatches += part.dispatches;
        total.wall_ms += part.wall_ms;
        total.gpu_ms += part.gpu_ms;
        total.capture_executes += part.capture_executes;
        offset += chunk;
    }

    context_len_ = start_position + rows;
    return total;
}

DFlash2DraftResult DFlash2Drafter::draft_block8(uint32_t k, uint32_t base_position, uint32_t anchor_id,
                                                std::span<const float> anchor_row, SamplingPolicy policy,
                                                const SamplingRng& after_anchor, bool adaptive_budget) {
    require(base_position == context_len_, "DFlash2 context frontier differs from target base position");
    require(k >= 1u && k <= 7u, "DFlash2 draft_block8 proposal count must be in 1..7");
    k = std::min(k, max_drafts_);
    require(anchor_id < vocab_size_ && anchor_row.size() == hidden_size_, "DFlash2 anchor token or embedding mismatch");
    for (float v : anchor_row) require(finite_bits(v), "nonfinite DFlash2 anchor embedding");

    std::copy(anchor_row.begin(), anchor_row.end(), host_input_8_.begin());
    gpu_.write(draft_hidden_[0], std::as_bytes(std::span(host_input_8_)));

    const uint64_t hid_stride = uint64_t(hidden_size_) * sizeof(float);
    const GpuRowsF32 norm_rows{draft_norm_, 8u, hidden_size_, hid_stride};
    const GpuRowsF32 dyn_rows{draft_dynamic_, 8u, 640u, 640ull * sizeof(float)};
    const GpuRowsF32 conv_rows{draft_conv_, 8u, hidden_size_, hid_stride};
    const GpuRowsF32 qkv_rows{draft_qkv_, 8u, 6144u, 6144ull * sizeof(float)};
    const GpuRowsF32 attn_rows{draft_attn_out_, 8u, 4096u, 4096ull * sizeof(float)};
    const GpuRowsF32 proj_rows{draft_proj_, 8u, hidden_size_, hid_stride};
    const GpuRowsF32 inter_rows{draft_inter_, 8u, intermediate_size_, uint64_t(intermediate_size_) * sizeof(float)};

    const uint32_t out_vocab = active_output_vocab_;
    const auto final_slice_k = gpu_.slice(draft_final_, hid_stride, uint64_t(k) * hid_stride);
    const auto logits_slice_k = gpu_.slice(draft_logits_7_, 0, uint64_t(k) * out_vocab * sizeof(float));
    const auto sel_hid_slice_k = gpu_.slice(draft_selector_hidden_7_, 0, uint64_t(k) * selector_rank_ * sizeof(float));
    const GpuRowsF32 final_rows_k{final_slice_k, k, hidden_size_, hid_stride};
    const GpuRowsF32 logits_rows_k{logits_slice_k, k, out_vocab, uint64_t(out_vocab) * sizeof(float)};
    const GpuRowsF32 sel_hid_rows_k{sel_hid_slice_k, k, selector_rank_, uint64_t(selector_rank_) * sizeof(float)};

    struct ScopeGuard {
        MetalGraph& g;
        ~ScopeGuard() { try { g.set_dflash2_scope(false); } catch (...) {} }
    };

    TOKSPUR_OPERATOR_CONTEXT(gpu_, "dflash2_draft", -1, base_position, k);
    gpu_.begin();
    ScopeGuard guard{gpu_};
    gpu_.set_dflash2_scope(true);
    try {
        if (pending_context_.rows() > 0) {
            encode_context_commit_chunk(pending_context_.storage(), pending_context_.rows(), pending_context_.start());
            TOKSPUR_OPERATOR_CHECKPOINT(gpu_, "dflash2", "deferred_context_commit", {pending_context_.rows(), hidden_size_, num_layers_, 0});
        }
        gpu_.rms_rows(GpuRowsF32{draft_hidden_[0], 8u, hidden_size_, hid_stride}, layers_[0].input_norm, norm_rows, hidden_size_, rms_eps_, true);
        for (uint32_t l = 0; l < num_layers_; ++l) {
            const uint32_t cur = l & 1u;
            const uint32_t nxt = cur ^ 1u;
            const auto& layer = layers_[l];

            dense_proj(layer.attn_conv_proj, norm_rows, dyn_rows, false);
            gpu_.dflash2_conv_f32(draft_norm_, draft_dynamic_, layer.attn_conv_base, draft_hidden_[cur], draft_conv_, hidden_size_, false);
            dense_proj(layer.qkv_proj, conv_rows, qkv_rows, false);
            gpu_.dflash2_qkv_prepare(draft_qkv_, layer.qk_norm_freq, draft_queries_, draft_query_keys_, draft_query_values_, base_position, 8u, rms_eps_);
            gpu_.dflash2_attention(draft_queries_, layer.ring_k, layer.ring_v, draft_query_keys_, draft_query_values_, draft_partials_, draft_attn_out_, base_position, sliding_window_);
            dense_proj(layer.o_proj, attn_rows, proj_rows, false);
            gpu_.dflash2_conv_finish_rms_f32(draft_proj_, draft_dynamic_, layer.attn_conv_base, draft_hidden_[cur],
                layer.post_attn_norm, draft_residual_, draft_norm_, hidden_size_, rms_eps_);

            dense_proj(layer.mlp_conv_proj, norm_rows, dyn_rows, false);
            gpu_.dflash2_conv_f32(draft_norm_, draft_dynamic_, layer.mlp_conv_base, draft_residual_, draft_conv_, hidden_size_, false);
            gpu_.dflash2_gate_up_swiglu(layer.gate_proj, layer.up_proj, conv_rows, inter_rows, precision_ == DFlash2Precision::bf16);
            dense_proj(layer.down_proj, inter_rows, proj_rows, false);
            const GpuView next_gains = (l + 1u < num_layers_) ? layers_[l + 1u].input_norm : final_norm_;
            const GpuView next_norm_out = (l + 1u < num_layers_) ? draft_norm_ : draft_final_;
            gpu_.dflash2_conv_finish_rms_f32(draft_proj_, draft_dynamic_, layer.mlp_conv_base, draft_residual_,
                next_gains, draft_hidden_[nxt], next_norm_out, hidden_size_, rms_eps_);
        }
        TOKSPUR_OPERATOR_CHECKPOINT(gpu_, "dflash2", "draft_layers", {8u, hidden_size_, intermediate_size_, num_layers_});
        gpu_.dflash2_dense_q8(active_output_weight_, final_rows_k, logits_rows_k, false);
        TOKSPUR_OPERATOR_CHECKPOINT(gpu_, "dense_q8", "dflash2_output_logits", {k, hidden_size_, out_vocab, 8});
        dense_proj(selector_proj_, final_rows_k, sel_hid_rows_k, false);
        gpu_.dflash2_select(logits_slice_k, draft_top_ids_, draft_top_vals_, draft_selector_out_,
            sel_hid_slice_k, predecessor_codebook_, successor_codebook_, k, out_vocab, anchor_id, vocab_size_);
        TOKSPUR_OPERATOR_CHECKPOINT(gpu_, "dflash2", "draft_selector", {k, selector_rank_, out_vocab, 16});
    } catch (...) {
        try { gpu_.abort(); } catch (...) {}
        throw;
    }
    BatchResult batch{};
    try {
        batch = gpu_.submit();
        if (gpu_.last_profile().timings_valid && std::getenv("TOKSPUR_DUMP_VERIFY_PROFILE")) {
            std::map<std::string, std::pair<uint32_t, double>> agg;
            for (const auto& kt : gpu_.last_profile().kernels) {
                auto& e = agg[kt.kernel];
                e.first += 1;
                e.second += kt.milliseconds;
            }
            std::cerr << "DRAFT_PROFILE sampled_ms=" << gpu_.last_profile().sampled_milliseconds
                      << " gpu_ms=" << batch.gpu_ms
                      << " dispatches=" << batch.dispatches;
            for (const auto& [kname, kv] : agg) {
                std::cerr << " " << kname << "=" << kv.first << ":" << kv.second;
            }
            std::cerr << "\n";
        }
        pending_context_.clear();
    } catch (...) {
        try { gpu_.abort(); } catch (...) {}
        throw;
    }
    gpu_.read(draft_selector_out_, host_selector_out_);

    const auto* candidates = reinterpret_cast<const uint32_t*>(host_selector_out_.data());
    const auto* unary = reinterpret_cast<const float*>(host_selector_out_.data() + 7ull * 16ull * sizeof(uint32_t));
    const auto* tables = reinterpret_cast<const float*>(host_selector_out_.data() + 7ull * 16ull * (sizeof(uint32_t) + sizeof(float)));

    DFlash2DraftResult out{};
    out.evaluated_proposals = k;
    out.proposal_begin_words = after_anchor.words_consumed();
    out.batch = batch;

    SamplingRng trial_rng = after_anchor;
    uint32_t pred_idx = 0;
    uint32_t prev_token = anchor_id;
    for (uint32_t pos = 0; pos < k; ++pos) {
        std::array<uint32_t, 16> cands{};
        std::array<float, 16> scores{};
        const uint32_t* pos_cands = candidates + pos * 16u;
        const float* pos_unary = unary + pos * 16u;
        const float* pos_edges = tables + (pos * 16u + pred_idx) * 16u;
        float best_score = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < 16u; ++i) {
            cands[i] = pos_cands[i];
            scores[i] = pos_unary[i] + pos_edges[i];
            require(finite_bits(scores[i]), "nonfinite DFlash2 candidate score");
            best_score = std::max(best_score, scores[i]);
        }
        double mass = 0.0;
        for (uint32_t i = 0; i < 16u; ++i) {
            mass += std::exp(double(scores[i] - best_score));
        }
        out.confidences[pos] = mass > 0.0 ? float(1.0 / mass) : 0.0f;

        auto law = build_selector_law(vocab_size_, cands, scores, policy);
        const uint64_t ticket = law.deterministic() ? 0ull : uint64_t(std::ldexp(trial_rng.uniform(), 53));
        const uint32_t selected = law.sample_at(ticket);

        uint32_t next_pred_idx = 16u;
        for (uint32_t i = 0; i < 16u; ++i) {
            if (cands[i] == selected) {
                next_pred_idx = i;
                break;
            }
        }
        require(next_pred_idx < 16u, "DFlash2 selected token missing from candidate support");

        out.proposals[pos] = SampledProposalRow{base_position + 1u + pos, prev_token, selected, std::move(law)};
        prev_token = selected;
        pred_idx = next_pred_idx;
    }

    const uint32_t verify_rows = measured_budget_ ? measured_budget_(base_position, out.confidences, 1u + k) : adaptive_budget
        ? verify_budget_.choose_verify_rows(out.confidences, 1u + k)
        : (1u + k);
    require(verify_rows >= 1u && verify_rows <= 1u + k, "DFlash2 verify budget selected out-of-bounds row count");
    const uint32_t active_k = verify_rows - 1u;
    out.count = active_k;
    if (active_k == k) {
        out.random_after = std::move(trial_rng);
    } else {
        out.random_after = after_anchor;
        for (uint32_t pos = 0; pos < active_k; ++pos) {
            if (!out.proposals[pos].actual_q.deterministic()) {
                static_cast<void>(out.random_after.uniform());
            }
        }
    }

    return out;
}

void DFlash2Drafter::observe_verify_round(
    std::span<const float, DFlash2VerifyBudget::kPositions> confidences,
    uint32_t verify_rows, uint32_t accepted, double milliseconds, bool record_labels) {
    verify_budget_.observe(confidences, verify_rows, accepted, milliseconds, record_labels);
}

namespace {

constexpr uint32_t kBudgetParams = 1u + DFlash2VerifyBudget::kPositions;
constexpr uint32_t kBudgetMaxIters = 16u;
constexpr std::array<double, 4> kSeedKnots{1.0, 2.0, 4.0, 8.0};
constexpr std::array<double, 4> kSeedShape{0.60, 0.69, 0.85, 1.0};

using BudgetVector = std::array<double, kBudgetParams>;

double budget_logit(double p) noexcept {
    p = std::clamp(p, 1.0e-6, 1.0 - 1.0e-6);
    return std::log(p / (1.0 - p));
}

double budget_prior_precision(uint32_t index) noexcept {
    return index == 0u ? DFlash2VerifyBudget::kSlopePrecision : DFlash2VerifyBudget::kInterceptPrecision;
}

BudgetVector budget_parameters_of(const DFlash2VerifyBudget::Acceptance& a) noexcept {
    BudgetVector out{};
    out[0] = double(a.slope);
    for (uint32_t pos = 0; pos < DFlash2VerifyBudget::kPositions; ++pos) {
        out[1u + pos] = double(a.intercepts[pos]);
    }
    return out;
}

double budget_objective(std::span<const DFlash2VerifyBudget::Label> labels,
                        const BudgetVector& theta, const BudgetVector& prior) noexcept {
    double value = 0.0;
    for (const auto& label : labels) {
        const double z = theta[0] * double(label.feature) + theta[1u + label.position];
        const double margin = label.accepted ? z : -z;
        value += margin > 0.0 ? std::log1p(std::exp(-margin)) : (-margin + std::log1p(std::exp(margin)));
    }
    for (uint32_t i = 0; i < kBudgetParams; ++i) {
        const double diff = theta[i] - prior[i];
        value += 0.5 * budget_prior_precision(i) * diff * diff;
    }
    return value;
}

BudgetVector solve_spd_system(std::array<BudgetVector, kBudgetParams> h, BudgetVector g) noexcept {
    for (uint32_t col = 0; col < kBudgetParams; ++col) {
        uint32_t pivot = col;
        for (uint32_t row = col + 1u; row < kBudgetParams; ++row) {
            if (std::fabs(h[row][col]) > std::fabs(h[pivot][col])) pivot = row;
        }
        std::swap(h[col], h[pivot]);
        std::swap(g[col], g[pivot]);
        const double diag = std::fabs(h[col][col]) > 1.0e-12 ? h[col][col] : 1.0e-12;
        for (uint32_t row = col + 1u; row < kBudgetParams; ++row) {
            const double factor = h[row][col] / diag;
            for (uint32_t k = col; k < kBudgetParams; ++k) {
                h[row][k] -= factor * h[col][k];
            }
            g[row] -= factor * g[col];
        }
    }
    BudgetVector x{};
    for (uint32_t row = kBudgetParams; row-- > 0u;) {
        double sum = g[row];
        for (uint32_t k = row + 1u; k < kBudgetParams; ++k) {
            sum -= h[row][k] * x[k];
        }
        const double diag = std::fabs(h[row][row]) > 1.0e-12 ? h[row][row] : 1.0e-12;
        x[row] = sum / diag;
    }
    return x;
}

} // namespace

double DFlash2VerifyBudget::seed_cost(uint32_t rows) {
    require(rows >= 1u && rows <= kBlockRows, "invalid DFlash2VerifyBudget row count");
    const double r = double(rows);
    uint32_t upper = 1u;
    while (upper < 3u && r > kSeedKnots[upper]) ++upper;
    const double t = (r - kSeedKnots[upper - 1u]) / (kSeedKnots[upper] - kSeedKnots[upper - 1u]);
    return kSeedShape[upper - 1u] + t * (kSeedShape[upper] - kSeedShape[upper - 1u]);
}

std::optional<std::pair<double, double>> DFlash2VerifyBudget::fit_cost_line() const noexcept {
    if (cost_.weight <= 0.0) return std::nullopt;
    const double mean_rows = cost_.rows / cost_.weight;
    const double mean_time = cost_.time / cost_.weight;
    const double spread = std::max(cost_.rows_squared / cost_.weight - mean_rows * mean_rows, 0.0);
    const double cov = cost_.rows_time / cost_.weight - mean_rows * mean_time;
    const double seed_slope = (seed_cost(kBlockRows) - seed_cost(1u)) / double(kBlockRows - 1u);
    const uint32_t rounded_rows = uint32_t(std::clamp<long>(std::lround(mean_rows), 1L, long(kBlockRows)));
    const double prior_slope = seed_slope * mean_time / seed_cost(rounded_rows);
    const double slope = std::max((cost_.weight * cov + kPriorCycles * prior_slope) / (cost_.weight * spread + kPriorCycles), 0.0);
    return std::pair{mean_time - slope * mean_rows, slope};
}

std::array<float, DFlash2VerifyBudget::kBlockRows + 1> DFlash2VerifyBudget::costs() const {
    std::array<float, kBlockRows + 1> result{};
    const auto fit_line = fit_cost_line();
    if (!fit_line) {
        for (uint32_t r = 1u; r <= kBlockRows; ++r) {
            result[r] = float(seed_cost(r));
        }
        return result;
    }
    const double floor_cost = 0.05 * (cost_.time / cost_.weight);
    for (uint32_t r = 1u; r <= kBlockRows; ++r) {
        result[r] = float(std::max(fit_line->first + fit_line->second * double(r), floor_cost));
    }
    return result;
}

uint32_t DFlash2VerifyBudget::choose_verify_rows(
    std::span<const float, kPositions> confidences, uint32_t max_rows) const {
    require(max_rows >= 1u && max_rows <= kBlockRows, "invalid DFlash2VerifyBudget max_rows");
    if (max_rows == 1u || explores()) return max_rows;
    const auto row_costs = costs();
    double survival = 1.0;
    double expected_tokens = 1.0;
    double best_efficiency = expected_tokens / std::max(double(row_costs[1]), 1.0e-9);
    uint32_t best_rows = 1u;
    for (uint32_t pos = 0; pos + 1u < max_rows; ++pos) {
        const double q = std::clamp(double(confidences[pos]), 1.0e-6, 1.0 - 1.0e-6);
        const double z = double(acceptance_.slope) * std::log(q / (1.0 - q)) + double(acceptance_.intercepts[pos]);
        const double p = 1.0 / (1.0 + std::exp(-z));
        survival *= p;
        expected_tokens += survival;
        const uint32_t rows = pos + 2u;
        const double efficiency = expected_tokens / std::max(double(row_costs[rows]), 1.0e-9);
        if (efficiency > best_efficiency) {
            best_efficiency = efficiency;
            best_rows = rows;
        }
    }
    return best_rows;
}

std::vector<DFlash2VerifyBudget::Label> DFlash2VerifyBudget::labels() const {
    std::vector<Label> ordered(labels_.begin() + std::ptrdiff_t(next_label_), labels_.end());
    ordered.insert(ordered.end(), labels_.begin(), labels_.begin() + std::ptrdiff_t(next_label_));
    return ordered;
}

DFlash2VerifyBudget::Acceptance DFlash2VerifyBudget::fit(
    std::span<const Label> labels, const Acceptance& start) {
    for (const auto& label : labels) {
        require(label.position < kPositions && std::isfinite(label.feature), "invalid DFlash2VerifyBudget label");
    }
    const BudgetVector prior = budget_parameters_of(Acceptance{});
    BudgetVector theta = budget_parameters_of(start);
    double current = budget_objective(labels, theta, prior);
    for (uint32_t iter = 0; iter < kBudgetMaxIters; ++iter) {
        BudgetVector grad{};
        std::array<BudgetVector, kBudgetParams> hess{};
        for (const auto& label : labels) {
            const uint32_t b = 1u + label.position;
            const double feat = double(label.feature);
            const double p = 1.0 / (1.0 + std::exp(-(theta[0] * feat + theta[b])));
            const double residual = p - (label.accepted ? 1.0 : 0.0);
            const double w = p * (1.0 - p);
            grad[0] += residual * feat;
            grad[b] += residual;
            hess[0][0] += w * feat * feat;
            hess[0][b] += w * feat;
            hess[b][0] += w * feat;
            hess[b][b] += w;
        }
        for (uint32_t i = 0; i < kBudgetParams; ++i) {
            const double prec = budget_prior_precision(i);
            grad[i] += prec * (theta[i] - prior[i]);
            hess[i][i] += prec;
        }
        const BudgetVector step = solve_spd_system(hess, grad);
        BudgetVector next{};
        double value = current;
        double scale = 1.0;
        bool moved = false;
        for (uint32_t halving = 0; halving < 30u && !moved; ++halving, scale *= 0.5) {
            for (uint32_t i = 0; i < kBudgetParams; ++i) {
                next[i] = theta[i] - scale * step[i];
            }
            value = budget_objective(labels, next, prior);
            moved = std::isfinite(value) && value <= current;
        }
        if (!moved) break;
        double largest = 0.0;
        for (uint32_t i = 0; i < kBudgetParams; ++i) {
            largest = std::max(largest, std::fabs(next[i] - theta[i]));
        }
        theta = next;
        current = value;
        if (largest < 1.0e-9) break;
    }
    Acceptance out{};
    out.slope = float(theta[0]);
    for (uint32_t pos = 0; pos < kPositions; ++pos) {
        out.intercepts[pos] = float(theta[1u + pos]);
    }
    return out;
}

void DFlash2VerifyBudget::observe(
    std::span<const float, kPositions> confidences,
    uint32_t verify_rows, uint32_t accepted, double milliseconds, bool record_labels) {
    require(verify_rows >= 1u && verify_rows <= kBlockRows && accepted < verify_rows,
        "invalid DFlash2VerifyBudget round outcome");
    if (std::isfinite(milliseconds) && milliseconds > 0.0) {
        double clamped_ms = milliseconds;
        if (const auto fit_line = fit_cost_line()) {
            const double expected = std::max(fit_line->first + fit_line->second * double(verify_rows), 1.0e-3);
            clamped_ms = std::clamp(milliseconds, 0.5 * expected, 2.0 * expected);
        }
        cost_.weight = kDecay * cost_.weight + 1.0;
        cost_.rows = kDecay * cost_.rows + double(verify_rows);
        cost_.rows_squared = kDecay * cost_.rows_squared + double(verify_rows) * double(verify_rows);
        cost_.time = kDecay * cost_.time + clamped_ms;
        cost_.rows_time = kDecay * cost_.rows_time + double(verify_rows) * clamped_ms;
    }
    // Only positions actually verified against the target row (0 .. verify_rows - 2, up to the first rejection) receive labels.
    if (record_labels && verify_rows >= 2u) {
        const uint32_t last = std::min({accepted, verify_rows - 2u, kPositions - 1u});
        for (uint32_t pos = 0; pos <= last; ++pos) {
            const float c = confidences[pos];
            if (!(std::isfinite(c) && c >= 0.0f && c <= 1.0f)) continue;
            const Label label{pos, float(budget_logit(double(c))), pos < accepted};
            if (labels_.size() < kLabels) {
                labels_.push_back(label);
            } else {
                labels_[next_label_] = label;
                next_label_ = (next_label_ + 1u) % kLabels;
            }
        }
    }
    if ((++cycles_ % kRefitCycles) == 0u && !labels_.empty()) {
        acceptance_ = fit(labels(), acceptance_);
    }
}

} // namespace tokspur
