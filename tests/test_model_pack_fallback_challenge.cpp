#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/model.hpp"
#include "tokspur/mtp.hpp"
#include "tokspur/numeric.hpp"

#include <bit>
#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace tokspur;
using namespace tokspur::derived_pack;

namespace {

using Bytes = std::vector<std::byte>;
size_t total_checks = 0;

void require(bool condition, const char* message) {
    ++total_checks;
    if (!condition) {
        std::cerr << "Assertion failed: " << message << std::endl;
        throw std::runtime_error(message);
    }
}

void number(Bytes& bytes, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        bytes.push_back(std::byte((value >> (i * 8)) & 255));
    }
}

void string(Bytes& bytes, std::string_view value) {
    number(bytes, value.size(), 8);
    for (const unsigned char c : value) {
        bytes.push_back(std::byte(c));
    }
}

void meta_string(Bytes& b, const char* key, std::string_view value) {
    string(b, key);
    number(b, 8, 4); // STRING type
    string(b, value);
}

void meta_uint64(Bytes& b, const char* key, uint64_t value) {
    string(b, key);
    number(b, 10, 4); // UINT64 type
    number(b, value, 8);
}

void meta_float32(Bytes& b, const char* key, float value) {
    string(b, key);
    number(b, 6, 4); // FLOAT32 type
    number(b, std::bit_cast<uint32_t>(value), 4);
}

void meta_uint64_array(Bytes& b, const char* key, const std::vector<uint64_t>& values) {
    string(b, key);
    number(b, 9, 4); // ARRAY type
    number(b, 10, 4); // element type UINT64
    number(b, values.size(), 8);
    for (auto v : values) {
        number(b, v, 8);
    }
}

struct TempFileGuard {
    std::filesystem::path path;
    int fd = -1;

    explicit TempFileGuard(const std::string& prefix) {
        const auto dir = std::filesystem::canonical(std::filesystem::temp_directory_path());
        path = dir / (prefix + "_" + std::to_string(::getpid()) + "_" + std::to_string(arc4random()) + ".gguf");
        fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("failed to create temp file");
    }

    ~TempFileGuard() {
        if (fd >= 0) ::close(fd);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

void pwrite_all(int fd, std::span<const std::byte> bytes, off_t offset) {
    for (size_t n = 0; n < bytes.size();) {
        const auto got = ::pwrite(fd, bytes.data() + n, bytes.size() - n, offset + off_t(n));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) throw std::runtime_error("pwrite failed");
        n += size_t(got);
    }
}

// Build synthetic Flash-Next model GGUF file
std::unique_ptr<TempFileGuard> make_model_files_fixture() {
    auto fixture = std::make_unique<TempFileGuard>("tokspur_model_fixture");
    constexpr uint64_t ple_storage_rows = 320001536;
    const auto requirements = flash_next_requirements(ple_storage_rows);

    Bytes meta;
    uint64_t meta_count = 0;
    auto add_meta = [&](auto&& fn) {
        fn();
        ++meta_count;
    };

    add_meta([&] { meta_string(meta, "general.architecture", "qwen4exp"); });
    add_meta([&] { meta_uint64(meta, "general.alignment", 32); });

    for (const auto& [key, value] : std::vector<std::pair<const char*, uint64_t>>{
        {"block_count", 48}, {"embedding_length", 2560}, {"attention.head_count", 24}, {"attention.head_count_kv", 2},
        {"attention.key_length", 256}, {"attention.value_length", 256}, {"expert_count", 512}, {"expert_used_count", 10},
        {"expert_feed_forward_length", 640}, {"expert_shared_feed_forward_length", 640}, {"ssm.conv_kernel", 4},
        {"ssm.state_size", 128}, {"ssm.group_count", 16}, {"ssm.time_step_rank", 48}, {"ssm.inner_size", 6144},
        {"full_attention_interval", 4}, {"rope.dimension_count", 64}, {"hyper_connection.count", 4}, {"hyper_connection.low_rank", 320},
        {"attention.indexer.head_count", 4}, {"attention.indexer.key_length", 128}, {"attention.indexer.top_k", 2048},
        {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 8}, {"ple.conv_kernel", 4}, {"embedding_length_per_layer_input", 160},
        {"ple.eos_token_id", 248044}, {"ple.image_token_id", 248056}}) {
        add_meta([&] { meta_uint64(meta, (std::string("qwen4exp.") + key).c_str(), value); });
    }
    add_meta([&] { meta_uint64(meta, "qwen4exp.context_length", 65536); });
    add_meta([&] { meta_uint64_array(meta, "qwen4exp.rope.dimension_sections", {11, 11, 10, 0}); });
    add_meta([&] { meta_uint64_array(meta, "qwen4exp.ple.layers", {1}); });

    std::vector<uint64_t> compress_ratios(48);
    for (size_t i = 0; i < 48; ++i) compress_ratios[i] = (i % 4 == 3 ? 4 : 0);
    add_meta([&] { meta_uint64_array(meta, "qwen4exp.attention.compress_ratios", compress_ratios); });

    add_meta([&] { meta_float32(meta, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f); });
    add_meta([&] { meta_float32(meta, "qwen4exp.rope.freq_base", 1e7f); });
    add_meta([&] { meta_uint64_array(meta, "qwen4exp.ple.layer_multipliers", {1, 3, 5}); });

    std::vector<uint64_t> ple_offsets(16);
    std::vector<uint64_t> ple_vocabs(16, 20000096);
    for (size_t i = 0; i < 16; ++i) ple_offsets[i] = i * 20000096ULL;
    add_meta([&] { meta_uint64_array(meta, "qwen4exp.ple.head_offsets", ple_offsets); });
    add_meta([&] { meta_uint64_array(meta, "qwen4exp.ple.head_vocab_sizes", ple_vocabs); });

    Bytes hdr;
    number(hdr, 0x46554747, 4); // "GGUF"
    number(hdr, 3, 4);          // version 3
    number(hdr, requirements.size(), 8); // tensor count
    number(hdr, meta_count, 8); // exact metadata count
    hdr.insert(hdr.end(), meta.begin(), meta.end());

    uint64_t current_offset = 0;
    for (const auto& req : requirements) {
        string(hdr, req.name);
        number(hdr, req.shape.size(), 4);
        uint64_t elements = 1;
        for (auto d : req.shape) {
            number(hdr, d, 8);
            elements *= d;
        }
        const uint32_t type = req.encodings[0];
        number(hdr, type, 4);
        number(hdr, current_offset, 8);

        const auto& enc = encoding(type);
        const uint64_t bytes = (elements / enc.elements_per_block) * enc.bytes_per_block;
        const uint64_t aligned_bytes = (bytes + 31) & ~31ULL;
        current_offset += aligned_bytes;
    }

    const uint64_t data_offset = (hdr.size() + 31) & ~31ULL;
    hdr.resize(data_offset, std::byte(0));
    pwrite_all(fixture->fd, hdr, 0);

    // Write extent marker so sparse file covers all tensor offsets
    const uint64_t total_extent = data_offset + current_offset;
    const std::byte extent_byte{0xaa};
    pwrite_all(fixture->fd, std::span(&extent_byte, 1), off_t(total_extent));

    return fixture;
}

// Build a derived pack GGUF fixture with a caller-selected subset of tensors
std::unique_ptr<TempFileGuard> make_pack_fixture(
    const std::vector<tokspur::TensorRequirement>& requirements,
    const std::set<std::string>& omit_names = {},
    bool include_cpu_embeddings = false
) {
    auto fixture = std::make_unique<TempFileGuard>("tokspur_pack_fixture");

    std::vector<tokspur::TensorRequirement> included;
    for (const auto& req : requirements) {
        const bool is_cpu = is_cpu_pinned_embedding(req.name);
        if (is_cpu && !include_cpu_embeddings) continue;
        if (omit_names.contains(req.name)) continue;
        included.push_back(req);
    }

    Bytes hdr;
    number(hdr, 0x46554747, 4); // "GGUF"
    number(hdr, 3, 4);          // version 3
    number(hdr, included.size(), 8); // tensor count
    number(hdr, 2, 8);          // metadata count
    meta_string(hdr, "general.architecture", "tokspur_pack");
    meta_uint64(hdr, "general.alignment", 32);

    uint64_t current_offset = 0;
    for (const auto& req : included) {
        string(hdr, req.name);
        number(hdr, req.shape.size(), 4);
        uint64_t elements = 1;
        for (auto d : req.shape) {
            number(hdr, d, 8);
            elements *= d;
        }
        const uint32_t type = req.encodings[0];
        number(hdr, type, 4);
        number(hdr, current_offset, 8);

        const auto& enc = encoding(type);
        const uint64_t bytes = (elements / enc.elements_per_block) * enc.bytes_per_block;
        const uint64_t aligned_bytes = (bytes + 31) & ~31ULL;
        current_offset += aligned_bytes;
    }

    const uint64_t data_offset = (hdr.size() + 31) & ~31ULL;
    hdr.resize(data_offset, std::byte(0));
    pwrite_all(fixture->fd, hdr, 0);

    const uint64_t total_extent = data_offset + current_offset;
    const std::byte extent_byte{0xbb};
    pwrite_all(fixture->fd, std::span(&extent_byte, 1), off_t(total_extent));

    return fixture;
}

// Build synthetic MTP sidecar GGUF file with 32 blk.48 tensors
std::unique_ptr<TempFileGuard> make_mtp_files_fixture() {
    auto fixture = std::make_unique<TempFileGuard>("tokspur_mtp_fixture");
    const auto requirements = mtp_shared_q8_requirements();

    Bytes hdr;
    number(hdr, 0x46554747, 4); // "GGUF"
    number(hdr, 3, 4);          // version 3
    number(hdr, requirements.size(), 8); // tensor count (32)
    number(hdr, 2, 8);          // metadata count
    meta_string(hdr, "general.architecture", "qwen4exp");
    meta_uint64(hdr, "general.alignment", 32);

    uint64_t current_offset = 0;
    for (const auto& req : requirements) {
        string(hdr, req.name);
        number(hdr, req.shape.size(), 4);
        for (auto d : req.shape) number(hdr, d, 8);
        number(hdr, req.encoding, 4);
        number(hdr, current_offset, 8);
        current_offset += (req.bytes + 31) & ~31ULL;
    }

    const uint64_t data_offset = (hdr.size() + 31) & ~31ULL;
    hdr.resize(data_offset, std::byte(0));
    pwrite_all(fixture->fd, hdr, 0);

    const uint64_t total_extent = data_offset + current_offset;
    const std::byte extent_byte{0xdd};
    pwrite_all(fixture->fd, std::span(&extent_byte, 1), off_t(total_extent));

    return fixture;
}

} // namespace

int main() {
    std::cout << "=== Running Model Loader Fallback Challenge Suite (Challenger 1) ===" << std::endl;

    // Unit checks on authoritative CPU embedding table predicates
    require(is_cpu_pinned_embedding("token_embd.weight") == true, "token_embd.weight is CPU pinned");
    require(is_cpu_pinned_embedding("per_layer_token_embd.weight") == true, "per_layer_token_embd.weight is CPU pinned");
    require(is_cpu_pinned_embedding("per_layer_token_embd") == true, "per_layer_token_embd is CPU pinned");
    require(is_cpu_pinned_embedding("blk.0.attn_qkv.weight") == false, "blk.0.attn_qkv.weight is not CPU pinned");
    require(is_cpu_pinned_embedding("output.weight") == false, "output.weight is not CPU pinned");
    require(is_cpu_pinned_embedding("output_hc_norm.weight") == false, "output_hc_norm.weight is not CPU pinned");

    require(is_gpu_pack_tensor("token_embd.weight") == false, "token_embd.weight excluded from GPU pack");
    require(is_gpu_pack_tensor("per_layer_token_embd.weight") == false, "per_layer_token_embd.weight excluded from GPU pack");
    require(is_gpu_pack_tensor("per_layer_token_embd") == false, "per_layer_token_embd excluded from GPU pack");
    require(is_gpu_pack_tensor("blk.0.attn_qkv.weight") == true, "blk.0.attn_qkv.weight included in GPU pack");
    require(is_gpu_pack_tensor("output.weight") == true, "output.weight included in GPU pack");
    std::cout << "[Pass] Authoritative CPU embedding classification checks passed." << std::endl;

    // Build synthetic model fixture
    std::cout << "Building synthetic model fixture..." << std::endl;
    auto model_guard = make_model_files_fixture();
    ModelFiles files({model_guard->path});
    require(files.tensors().size() == 1224, "model files has exactly 1,224 tensors");
    const auto requirements = flash_next_requirements(320001536);

    // Test 1: Baseline inspect_flash_next without pack (pack = nullptr)
    {
        std::cout << "Testing baseline loading (pack = nullptr)..." << std::endl;
        const auto layout = inspect_flash_next(files, nullptr);
        require(layout.tensors == 1224, "layout tensors == 1224");
        std::cout << "[Pass] Baseline inspect_flash_next(files, nullptr) succeeds." << std::endl;
    }

    // Test 2: Valid pack containing all 1,222 non-CPU GPU tensors and omitting CPU embeddings
    {
        std::cout << "Testing pack with all GPU tensors and CPU embeddings omitted..." << std::endl;
        auto pack_guard = make_pack_fixture(requirements, {}, false);
        ModelFiles pack({pack_guard->path});
        require(pack.tensors().size() == 1222, "pack has 1,222 GPU tensors");
        require(!pack.tensors().contains("token_embd.weight"), "token_embd.weight omitted from pack");
        require(!pack.tensors().contains("per_layer_token_embd.weight"), "per_layer_token_embd.weight omitted from pack");

        const auto layout = inspect_flash_next(files, &pack);
        require(layout.tensors == 1224, "layout tensors == 1224");
        std::cout << "[Pass] Pack with omitted CPU embeddings succeeds without error (CPU retained)." << std::endl;
    }

    // Test 3: Attack: Pack missing 1 GPU tensor (blk.0.attn_qkv.weight)
    // MUST throw std::runtime_error("Derived pack missing required GPU tensor: blk.0.attn_qkv.weight")
    {
        std::cout << "Testing attack: pack missing blk.0.attn_qkv.weight..." << std::endl;
        auto pack_guard = make_pack_fixture(requirements, {"blk.0.attn_qkv.weight"}, false);
        ModelFiles pack({pack_guard->path});
        require(pack.tensors().size() == 1221, "incomplete pack has 1,221 tensors");

        bool threw_expected = false;
        try {
            inspect_flash_next(files, &pack);
        } catch (const std::runtime_error& err) {
            const std::string msg = err.what();
            if (msg.find("Derived pack missing required GPU tensor: blk.0.attn_qkv.weight") != std::string::npos) {
                threw_expected = true;
            } else {
                std::cerr << "Unexpected error message: " << msg << std::endl;
            }
        }
        require(threw_expected, "missing blk.0.attn_qkv.weight unconditionally throws std::runtime_error");
        std::cout << "[Pass] Refusal on missing blk.0.attn_qkv.weight confirmed (zero silent fallback)." << std::endl;
    }

    // Test 4: Attack: Pack missing output.weight
    {
        std::cout << "Testing attack: pack missing output.weight..." << std::endl;
        auto pack_guard = make_pack_fixture(requirements, {"output.weight"}, false);
        ModelFiles pack({pack_guard->path});

        bool threw_expected = false;
        try {
            inspect_flash_next(files, &pack);
        } catch (const std::runtime_error& err) {
            const std::string msg = err.what();
            if (msg.find("Derived pack missing required GPU tensor: output.weight") != std::string::npos) {
                threw_expected = true;
            }
        }
        require(threw_expected, "missing output.weight unconditionally throws std::runtime_error");
        std::cout << "[Pass] Refusal on missing output.weight confirmed (zero silent fallback)." << std::endl;
    }

    // Test 5: Attack: Pack missing an inner layer tensor (blk.23.attn_output.weight)
    {
        std::cout << "Testing attack: pack missing blk.23.attn_output.weight..." << std::endl;
        auto pack_guard = make_pack_fixture(requirements, {"blk.23.attn_output.weight"}, false);
        ModelFiles pack({pack_guard->path});

        bool threw_expected = false;
        try {
            inspect_flash_next(files, &pack);
        } catch (const std::runtime_error& err) {
            const std::string msg = err.what();
            if (msg.find("Derived pack missing required GPU tensor: blk.23.attn_output.weight") != std::string::npos) {
                threw_expected = true;
            }
        }
        require(threw_expected, "missing blk.23.attn_output.weight unconditionally throws std::runtime_error");
        std::cout << "[Pass] Refusal on missing blk.23.attn_output.weight confirmed (zero silent fallback)." << std::endl;
    }

    // Test 6: Derived pack local-seal authentication hardening (Track T-Auth / Milestone M1)
    {
        std::cout << "Testing Test 6: derived pack local-seal authentication hardening..." << std::endl;
        auto mtp_guard = make_mtp_files_fixture();
        auto source_ptr = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{model_guard->path});
        auto mtp_ptr = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{mtp_guard->path});

        auto plan = plan_derived_pack(*source_ptr, mtp_ptr.get());
        const auto canonical_header = build_header(plan);

        auto adopted_pack_guard = std::make_unique<TempFileGuard>("tokspur_adopted_pack");
        pwrite_all(adopted_pack_guard->fd, canonical_header, 0);
        const std::byte extent_byte{0};
        pwrite_all(adopted_pack_guard->fd, std::span(&extent_byte, 1), off_t(plan.file_bytes - 1));
        ::chmod(adopted_pack_guard->path.c_str(), 0444);

        // Load pack GGUF to read manifest string and compute its hash
        auto pack_model = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{adopted_pack_guard->path});
        const auto& file = pack_model->shards()[0];
        const auto manifest_sha256 = hash(file.string_value("tokspur_pack.manifest"));

        const auto seal_path = adopted_pack_guard->path.parent_path() / (adopted_pack_guard->path.stem().string() + ".seal");
        auto write_seal = [&](const std::string& content) {
            std::error_code ec;
            std::filesystem::remove(seal_path, ec);
            int sfd = ::open(seal_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0444);
            require(sfd >= 0, "create seal file");
            const std::string full = content + "\n";
            pwrite_all(sfd, std::as_bytes(std::span(full.data(), full.size())), 0);
            ::close(sfd);
            ::chmod(seal_path.c_str(), 0444);
        };

        // 6a: Seal mismatch falls back to full verification and throws
        {
            std::cout << "  [6a] Testing seal mismatch fallback..." << std::endl;
            write_seal("0000000000000000000000000000000000000000000000000000000000000000");
            bool threw = false;
            try {
                static_cast<void>(adopt_derived_pack(source_ptr, mtp_ptr, adopted_pack_guard->path, default_chunk_bytes, 1, TrustMode::local_seal));
            } catch (const FormatError&) {
                threw = true;
            } catch (const std::exception&) {
                threw = true;
            }
            require(threw, "seal mismatch must trigger full verification and throw on invalid pack proofs");
            std::cout << "  [Pass] 6a: Seal mismatch gracefully fell back to full verification and threw." << std::endl;
        }

        // 6b: Valid seal fast-starts with authentication_workers == 1
        {
            std::cout << "  [6b] Testing valid seal fast-start..." << std::endl;
            write_seal(manifest_sha256);
            auto adopted = adopt_derived_pack(source_ptr, mtp_ptr, adopted_pack_guard->path, default_chunk_bytes, 1, TrustMode::local_seal);
            require(adopted.authentication_workers == 1, "adopted authentication_workers == 1");
            require(adopted.manifest_sha256 == manifest_sha256, "manifest sha256 matched");
            require(adopted.model != nullptr, "adopted model not null");
            std::cout << "  [Pass] 6b: Valid seal fast-start succeeded with 1 worker." << std::endl;
        }

        // 6c: Backdoor seal (mtp_sidecar_sha256) throws
        {
            std::cout << "  [6c] Testing backdoor seal (mtp_sidecar_sha256) refusal..." << std::endl;
            write_seal(mtp_sidecar_sha256);
            bool threw = false;
            try {
                static_cast<void>(adopt_derived_pack(source_ptr, mtp_ptr, adopted_pack_guard->path, default_chunk_bytes, 1, TrustMode::local_seal));
            } catch (const FormatError&) {
                threw = true;
            } catch (const std::exception&) {
                threw = true;
            }
            require(threw, "backdoor mtp_sidecar_sha256 seal must NOT bypass verification");
            std::cout << "  [Pass] 6c: Backdoor seal strictly refused (fell back and threw)." << std::endl;
        }

        std::error_code ec;
        std::filesystem::remove(seal_path, ec);
    }

    std::cout << "SUCCESS: All " << total_checks << " model loader fallback challenge checks passed!" << std::endl;
    return 0;
}
