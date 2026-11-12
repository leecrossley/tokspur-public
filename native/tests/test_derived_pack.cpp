#include "tokspur/derived_pack.hpp"
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "grouped_mm_fixture.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

using namespace tokspur;
using namespace tokspur::derived_pack;

namespace {
using Bytes = std::vector<std::byte>;
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename F>
void refuses(F&& func, const char* message) {
    ++checks;
    bool caught = false;
    try {
        func();
    } catch (const FormatError&) {
        caught = true;
    } catch (const std::exception&) {
        caught = true;
    }
    if (!caught) {
        throw std::runtime_error(message);
    }
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

void push_u(Bytes& out, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        out.push_back(std::byte((value >> (8 * i)) & 0xff));
    }
}

void push_str(Bytes& out, std::string_view value) {
    push_u(out, value.size(), 8);
    for (unsigned char c : value) {
        out.push_back(std::byte(c));
    }
}

void push_meta_str(Bytes& out, std::string_view key, std::string_view value) {
    push_str(out, key);
    push_u(out, 8, 4);
    push_str(out, value);
}

void push_meta_u16(Bytes& out, std::string_view key, uint16_t value) {
    push_str(out, key);
    push_u(out, 2, 4);
    push_u(out, value, 2);
}

void push_meta_u32(Bytes& out, std::string_view key, uint32_t value) {
    push_str(out, key);
    push_u(out, 4, 4);
    push_u(out, value, 4);
}

struct SyntheticTensorSpec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dimensions;
    Bytes payload;
};

Bytes make_valid_q8_rows(uint32_t columns, uint32_t rows, uint8_t seed) {
    const uint32_t blocks = columns / 32;
    Bytes out(size_t(rows) * blocks * 34);
    size_t pos = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            // Finite F16 scale 1.0 (0x3c00) varied slightly in low mantissa bits so exponent != 31.
            const uint16_t scale = uint16_t(0x3800 | ((r + b + seed) & 0x00ff));
            out[pos++] = std::byte(scale & 0xff);
            out[pos++] = std::byte((scale >> 8) & 0xff);
            for (uint32_t q = 0; q < 32; ++q) {
                out[pos++] = std::byte((r * 13 + b * 7 + q * 3 + seed) & 0xff);
            }
        }
    }
    return out;
}

Bytes make_valid_q51_rows(uint32_t columns, uint32_t rows, uint8_t seed) {
    const uint32_t blocks = columns / 32;
    Bytes out(size_t(rows) * blocks * 24);
    size_t pos = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint16_t d = uint16_t(0x3800 | ((r + b + seed) & 0x00ff));
            const uint16_t m = uint16_t(0x3400 | ((r * 3 + b + seed) & 0x00ff));
            out[pos++] = std::byte(d & 0xff);
            out[pos++] = std::byte((d >> 8) & 0xff);
            out[pos++] = std::byte(m & 0xff);
            out[pos++] = std::byte((m >> 8) & 0xff);
            for (uint32_t h = 0; h < 4; ++h) {
                out[pos++] = std::byte((r + b + h + seed) & 0xff);
            }
            for (uint32_t q = 0; q < 16; ++q) {
                out[pos++] = std::byte((r * 17 + b * 5 + q + seed) & 0xff);
            }
        }
    }
    return out;
}

Bytes make_f32_bytes(size_t elements, uint8_t seed) {
    Bytes out(elements * 4);
    for (size_t i = 0; i < elements; ++i) {
        const float value = float(int((i + seed) % 97) - 48) * 0.125f;
        std::memcpy(out.data() + i * 4, &value, 4);
    }
    return out;
}

void write_file_bytes(const std::filesystem::path& path, const Bytes& bytes, mode_t mode = 0600) {
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, mode);
    if (fd < 0) throw std::runtime_error("failed to create synthetic GGUF file");
    for (size_t at = 0; at < bytes.size();) {
        const auto n = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            ::close(fd);
            throw std::runtime_error("failed to write synthetic GGUF file");
        }
        at += size_t(n);
    }
    ::close(fd);
}

void write_gguf_shard(
    const std::filesystem::path& path,
    std::string_view architecture,
    const std::vector<SyntheticTensorSpec>& specs,
    uint16_t split_no = 0,
    uint16_t split_count = 1,
    uint32_t split_tensors_count = 0
) {
    constexpr uint64_t alignment = 32;
    Bytes header;
    push_u(header, 0x46554747, 4);
    push_u(header, 3, 4);
    push_u(header, specs.size(), 8);
    const uint64_t meta_count = split_count > 1 ? 5 : 2;
    push_u(header, meta_count, 8);
    push_meta_str(header, "general.architecture", architecture);
    push_meta_u32(header, "general.alignment", uint32_t(alignment));
    if (split_count > 1) {
        push_meta_u16(header, "split.no", split_no);
        push_meta_u16(header, "split.count", split_count);
        push_meta_u32(header, "split.tensors.count", split_tensors_count);
    }

    std::vector<uint64_t> offsets;
    uint64_t rel = 0;
    for (const auto& s : specs) {
        rel = align_up(rel, alignment);
        offsets.push_back(rel);
        rel += s.payload.size();
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        push_str(header, specs[i].name);
        push_u(header, specs[i].dimensions.size(), 4);
        for (uint64_t d : specs[i].dimensions) push_u(header, d, 8);
        push_u(header, specs[i].type, 4);
        push_u(header, offsets[i], 8);
    }
    const uint64_t data_offset = align_up(header.size(), alignment);
    header.resize(size_t(data_offset), std::byte(0));
    Bytes file = std::move(header);
    file.resize(size_t(data_offset + rel), std::byte(0));
    for (size_t i = 0; i < specs.size(); ++i) {
        std::memcpy(
            file.data() + size_t(data_offset + offsets[i]),
            specs[i].payload.data(),
            specs[i].payload.size()
        );
    }
    write_file_bytes(path, file);
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto base = std::filesystem::canonical(std::filesystem::temp_directory_path());
        path = base /
               ("tokspur_test_derived_pack_" + std::to_string(::getpid()) + "_" + std::to_string(arc4random()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(
                 path, std::filesystem::directory_options::skip_permission_denied)) {
            if (!entry.is_symlink()) {
                ::chmod(entry.path().c_str(), entry.is_directory() ? 0700 : 0600);
            }
        }
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

size_t count_entries(const std::filesystem::path& dir) {
    size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        static_cast<void>(entry);
        ++n;
    }
    return n;
}

int run_cli_verify(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test-derived_pack --verify-pack <pack> <source...> [--mtp <mtp...>]" << std::endl;
        return 2;
    }
    const std::filesystem::path pack_path = argv[2];
    std::vector<std::filesystem::path> source_paths;
    std::vector<std::filesystem::path> mtp_paths;
    bool in_mtp = false;
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--mtp") {
            in_mtp = true;
            continue;
        }
        if (in_mtp) {
            mtp_paths.emplace_back(arg);
        } else {
            source_paths.emplace_back(arg);
        }
    }
    if (source_paths.empty()) {
        std::cerr << "missing source GGUF paths for --verify-pack" << std::endl;
        return 2;
    }
    try {
        auto source = std::make_shared<const ModelFiles>(source_paths);
        std::shared_ptr<const ModelFiles> mtp;
        if (!mtp_paths.empty()) {
            mtp = std::make_shared<const ModelFiles>(mtp_paths);
        }
        const auto single = adopt_derived_pack(source, mtp, pack_path, default_chunk_bytes, 1, TrustMode::full, false);
        const auto multi = adopt_derived_pack(source, mtp, pack_path, default_chunk_bytes, 2, TrustMode::full, false);
        require(single.manifest_sha256 == multi.manifest_sha256, "manifest digest mismatch across worker counts");
        require(single.model->tensors().size() == single.plan.tensors.size(), "adopted model tensor count mismatch");
        for (const auto& t : single.plan.tensors) {
            require(t.source_sha256.size() == 64 && t.packed_sha256.size() == 64, "missing tensor SHA-256 digest");
        }
        std::cout << "PASS verified pack tensors=" << single.plan.tensors.size()
                  << " file_bytes=" << single.plan.file_bytes
                  << " manifest_sha256=" << single.manifest_sha256 << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL verify-pack: " << e.what() << std::endl;
        return 1;
    }
}
void test_v3_authentication_sources(const std::filesystem::path& dir) {
    using namespace tokspur::planar_v3;
    std::filesystem::create_directory(dir);
    std::vector<SyntheticTensorSpec> tensors{
        {"token_embd.weight", 8, {256, 4}, make_valid_q8_rows(256, 4, 1)},
        {"per_layer_token_embd.weight", 0, {64, 2}, make_f32_bytes(128, 2)},
        {"output.weight", 8, {512, 256}, make_valid_q8_rows(512, 256, 3)},
        {"output_hc_norm.weight", 0, {20000}, make_f32_bytes(20000, 4)},
    };
    // Exercise every v3 authentication branch, including both halves of a
    // co-located gate/up unit and reads spanning the 32 KiB chunk boundary.
    for (uint32_t block = 0; block < 3; ++block) {
        const uint32_t type = block == 0 ? 12 : (block == 1 ? 13 : 8);
        const uint32_t down_type = block == 2 ? 8 : 7;
        const auto prefix = "blk." + std::to_string(block) + ".";
        for (const auto* part : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
            tensors.push_back({prefix + part, type, {256, 32, 4},
                               grouped_fixture::weights({type, 256, 32, 4, 1, 1, true})});
        }
        tensors.push_back({prefix + "ffn_down_exps.weight", down_type, {32, 256, 4},
                           grouped_fixture::weights({down_type, 32, 256, 4, 1, 1, true})});
    }
    const std::vector<SyntheticTensorSpec> mtp_tensors{
        {"blk.48.attn_q.weight", 8, {512, 128}, make_valid_q8_rows(512, 128, 5)},
    };
    const auto source_path = dir / "source.gguf", mtp_path = dir / "mtp.gguf";
    const auto pack_path = dir / "pack.asa.gguf";
    write_gguf_shard(source_path, "qwen35moe", tensors);
    write_gguf_shard(mtp_path, "qwen35moe", mtp_tensors);
    const auto source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{source_path});
    const auto mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{mtp_path});
    static_cast<void>(build_derived_pack_v3(source, mtp, pack_path, 2, 0, 32768));
    const auto adopt = [&](std::shared_ptr<const ModelFiles> origin, std::shared_ptr<const ModelFiles> sidecar,
                           uint32_t workers) {
        return adopt_derived_pack_v3(std::move(origin), std::move(sidecar), pack_path,
                                    2, 32768, workers, TrustMode::full, false);
    };
    const auto single = adopt(source, mtp, 1), parallel = adopt(source, mtp, 4);
    require(single.manifest_sha256 == parallel.manifest_sha256 && parallel.authentication_workers == 4,
            "v3 single and parallel authentication differ");
    const std::string_view expected_read_mode =
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        "bounded_nocache";
#else
        "mapped";
#endif
    require(single.authentication_source_read_mode == expected_read_mode &&
            parallel.authentication_source_read_mode == expected_read_mode,
            "v3 full authentication source-read mode mismatch");
    uint32_t layouts = 0;
    for (const auto& tensor : single.plan.tensors) layouts |= uint32_t{1} << uint32_t(tensor.kind);
    require(layouts == 0x3ffu, "v3 CPU authentication fixture must cover all ten tensor layouts");
    const MappedFile packed_file(pack_path);
    const auto packed_bytes = packed_file.bytes();
    std::cout << "v3_auth_source_read_mode=" << expected_read_mode
              << " manifest_sha256=" << single.manifest_sha256
              << " pack_sha256=" << hash(std::string_view(reinterpret_cast<const char*>(packed_bytes.data()), packed_bytes.size()))
              << '\n';
    for (size_t i = 0; i < 2; ++i) {
        require(std::ranges::equal(source->tensor_bytes(tensors[i].name), tensors[i].payload),
                "CPU embedding mapping changed during v3 authentication");
    }
    for (size_t i = 2; i < tensors.size(); ++i) {
        auto changed = tensors;
        changed[i].payload.back() ^= std::byte{1};
        const auto changed_path = dir / ("changed-" + std::to_string(i) + ".gguf");
        write_gguf_shard(changed_path, "qwen35moe", changed);
        const auto origin = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{changed_path});
        refuses([&] { static_cast<void>(adopt(origin, mtp, 4)); },
                "v3 full authentication accepted changed source tensor tail");
    }
    auto changed_mtp = mtp_tensors;
    changed_mtp[0].payload.back() ^= std::byte{1};
    const auto changed_mtp_path = dir / "changed-mtp.gguf";
    write_gguf_shard(changed_mtp_path, "qwen35moe", changed_mtp);
    const auto sidecar = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{changed_mtp_path});
    refuses([&] { static_cast<void>(adopt(source, sidecar, 4)); },
            "v3 full authentication accepted changed MTP source tail");
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
    // The bounded descriptor must bind the same file as the retained mapping.
    std::filesystem::rename(source_path, dir / "original-source.gguf");
    write_gguf_shard(source_path, "qwen35moe", tensors);
    refuses([&] { static_cast<void>(adopt(source, mtp, 4)); },
            "v3 nocache authentication accepted a replaced source inode");
#endif
}

void test_v3_publication(const std::filesystem::path& dir, std::shared_ptr<const ModelFiles> source,
                         std::shared_ptr<const ModelFiles> mtp) {
    using namespace tokspur::planar_v3;
    std::filesystem::create_directory(dir);
    const auto output = dir / "block.asa.gguf";
    const auto seal = dir / "block.asa.seal";
    const auto full = plan_derived_pack_v3(*source, mtp.get());
    refuses([&] { static_cast<void>(build_or_verify_pack_v3(full, source, mtp, output, 2, 0, 32768)); },
            "partial target cannot publish a full v3 recipe");
    const auto block = plan_block_pack_v3(*source, mtp.get(), 0);
    for (const auto* reserved : {"tokspur-pack-v2.asa.gguf", "TokSpur-Pack-V2.Gguf", "wrong.seal"}) {
        refuses([&] { static_cast<void>(build_or_verify_pack_v3(block, source, mtp, dir / reserved, 2, 0, 32768)); },
                "reserved pack or seal destination refused");
    }
    const auto budget = measure_pack_v3_budget(block, dir, {}, 2, 0, 32768);
    require(budget.planned_pack_v3_file_bytes == block.file_bytes && budget.streaming_ring_buffer_bytes >= 65536,
            "pack v3 budget accounts for configured Q8 write and readback buffers");
    const auto publish = [&](uint64_t reserve = 0) {
        return build_or_verify_pack_v3(block, source, mtp, output, 2, reserve, 32768);
    };
    refuses([&] {
        static_cast<void>(build_or_verify_pack_v3(block, source, mtp, output, 2, 0, 32768,
                                                  [] { throw FormatError("interrupted test build"); }));
    }, "interruption refused");
    require(count_entries(dir) == 0, "interrupted write left no partial pack or seal");

    const int lock = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(lock >= 0 && ::flock(lock, LOCK_EX | LOCK_NB) == 0, "acquire competing output directory lock");
    refuses([&] { static_cast<void>(publish()); }, "competing v3 publisher refused");
    ::close(lock);
    require(count_entries(dir) == 0, "competing writer left no output");

    write_file_bytes(seal, Bytes(65, std::byte{'0'}), 0444);
    refuses([&] { static_cast<void>(publish()); }, "orphan v3 seal refused");
    require(!std::filesystem::exists(output) && std::filesystem::remove(seal), "orphan seal preserved");
    require(::symlink("missing.gguf", output.c_str()) == 0, "create dangling pack symlink");
    refuses([&] { static_cast<void>(publish()); }, "dangling v3 output symlink refused");
    require(std::filesystem::is_symlink(output) && std::filesystem::remove(output), "output symlink preserved");

    write_file_bytes(output, Bytes(64, std::byte{'x'}), 0444);
    refuses([&] { static_cast<void>(publish()); }, "incompatible existing pack refused");
    require(std::filesystem::file_size(output) == 64 && !std::filesystem::exists(seal) &&
            std::filesystem::remove(output), "incompatible pack never replaced or sealed");
    const auto built = publish();
    require(!built.existed && built.verified.manifest_sha256.size() == 64, "built and fully authenticated new v3 pack");
    struct stat pack_st{}, seal_st{};
    require(::stat(output.c_str(), &pack_st) == 0 && ::stat(seal.c_str(), &seal_st) == 0 &&
            (pack_st.st_mode & 0777) == 0444 && (seal_st.st_mode & 0777) == 0444, "v3 pack and seal immutable");
    const auto existing = publish(std::numeric_limits<uint64_t>::max());
    struct stat same{};
    require(existing.existed && existing.verified.manifest_sha256 == built.verified.manifest_sha256 &&
            ::stat(seal.c_str(), &same) == 0 && same.st_ino == seal_st.st_ino, "existing v3 seal was reused");
    require(std::filesystem::remove(seal), "simulate interruption after pack publication");
    const auto repaired = publish();
    require(repaired.existed && repaired.verified.manifest_sha256 == built.verified.manifest_sha256 &&
            ::stat(output.c_str(), &same) == 0 && same.st_ino == pack_st.st_ino, "missing v3 seal repaired in place");

    require(std::filesystem::remove(seal), "remove valid seal for mismatch test");
    write_file_bytes(seal, Bytes(64, std::byte{'0'}), 0444);
    require(::stat(seal.c_str(), &seal_st) == 0, "stat wrong seal");
    refuses([&] { static_cast<void>(publish()); }, "mismatched v3 seal refused");
    require(::stat(seal.c_str(), &same) == 0 && same.st_ino == seal_st.st_ino, "mismatched seal never replaced");
    require(std::filesystem::remove(seal), "remove wrong seal");
    require(::symlink("missing.seal", seal.c_str()) == 0, "create dangling seal symlink");
    refuses([&] { static_cast<void>(publish()); }, "dangling v3 seal symlink refused");
    require(std::filesystem::is_symlink(seal) && std::filesystem::remove(seal), "seal symlink preserved");
    static_cast<void>(publish());

    // A matching manifest seal must not bypass full source verification on retry.
    require(::chmod(output.c_str(), 0600) == 0, "unseal fixture");
    const int fd = ::open(output.c_str(), O_RDWR | O_CLOEXEC);
    std::byte byte{};
    const auto offset = static_cast<off_t>(block.data_offset);
    require(fd >= 0 && ::pread(fd, &byte, 1, offset) == 1, "read v3 test byte");
    byte ^= std::byte{1};
    require(::pwrite(fd, &byte, 1, offset) == 1 && ::fchmod(fd, 0444) == 0, "alter v3 test byte");
    ::close(fd);
    refuses([&] { static_cast<void>(publish()); }, "tampered v3 payload refused even with a matching seal");
    require(std::filesystem::remove(seal), "remove seal on corrupt fixture");
    refuses([&] { static_cast<void>(publish()); }, "tampered v3 payload cannot be resealed");
    require(!std::filesystem::exists(seal) && ::stat(output.c_str(), &same) == 0 &&
            same.st_ino == pack_st.st_ino, "corrupt pack preserved and no new seal issued");
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string_view(argv[1]) == "--verify-pack" || std::string_view(argv[1]) == "--adopt")) {
        return run_cli_verify(argc, argv);
    }

    try {
        TempDir root;
        const auto source_dir = root.path / "source";
        const auto out_dir = root.path / "out";
        std::filesystem::create_directories(source_dir);
        std::filesystem::create_directories(out_dir);

        // Multi-chunk Q8_0 tensor (> 64 KiB so 32 KiB chunks split across 3 iterations).
        const std::vector<SyntheticTensorSpec> shard0 = {
            {"token_embd.weight", 8, {256, 4}, make_valid_q8_rows(256, 4, 1)},
            {"per_layer_token_embd.weight", 0, {64, 2}, make_f32_bytes(128, 2)},
            {"blk.0.attn_q.weight", 8, {256, 256}, make_valid_q8_rows(256, 256, 3)},
            {"blk.0.ffn_down_exps.weight", 7, {640, 4, 2}, make_valid_q51_rows(640, 8, 4)},
        };
        const std::vector<SyntheticTensorSpec> shard1 = {
            {"blk.1.ple_key.weight", 8, {256, 4}, make_valid_q8_rows(256, 4, 5)},
            {"output_hc_norm.weight", 0, {128}, make_f32_bytes(128, 6)},
        };
        const std::vector<SyntheticTensorSpec> mtp_specs = {
            {"blk.48.attn_q.weight", 8, {256, 8}, make_valid_q8_rows(256, 8, 7)},
            {"blk.48.ffn_gate_inp.weight", 0, {64, 2}, make_f32_bytes(128, 8)},
        };

        const auto shard0_path = source_dir / "shard-00001-of-00002.gguf";
        const auto shard1_path = source_dir / "shard-00002-of-00002.gguf";
        const auto mtp_path = source_dir / "mtp.gguf";
        write_gguf_shard(shard0_path, "qwen4exp", shard0, 0, 2, uint32_t(shard0.size() + shard1.size()));
        write_gguf_shard(shard1_path, "qwen4exp", shard1, 1, 2, uint32_t(shard0.size() + shard1.size()));
        write_gguf_shard(mtp_path, "qwen4exp", mtp_specs);

        auto source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{shard0_path, shard1_path});
        auto mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{mtp_path});

        test_v3_publication(root.path / "v3", source, mtp);
        test_v3_authentication_sources(root.path / "v3-auth");

        // 1. Planning checks
        const auto planned = plan_derived_pack(*source, mtp.get());
        require(planned.tensors.size() == 6, "CPU embeddings excluded and GPU + MTP tensors retained");
        for (const auto& t : planned.tensors) {
            require(!is_cpu_pinned_embedding(t.name), "CPU-pinned tensor leaked into pack plan");
            require(t.offset % pack_alignment == 0, "tensor offset aligned to 16 KiB");
            if (t.name == "blk.0.attn_q.weight" || t.name == "blk.48.attn_q.weight") {
                require(t.layout == "q8_lanes_v1", "eligible 2D Q8_0 tensor assigned q8_lanes_v1");
            } else if (t.name == "blk.0.ffn_down_exps.weight") {
                require(t.layout == "q51_lanes_v1", "eligible 3D Q5_1 down_exps assigned q51_lanes_v1");
            } else {
                require(t.layout == "plain", "non-eligible tensor kept plain");
            }
        }
        refuses([&] {
            static_cast<void>(plan_derived_pack(*source, mtp.get(), planned.file_bytes - 1));
        }, "byte_budget below planned file size refused");

        // 2. Reserve refusal (preflight, overflow, and deterministic mid-stream chunk drop)
        struct statvfs vfs{};
        require(::statvfs(out_dir.c_str(), &vfs) == 0, "statvfs on out_dir");
        const uint64_t real_avail = uint64_t(vfs.f_bavail) * uint64_t(vfs.f_frsize);
        const auto refused_out = out_dir / "refused-pack.gguf";
        // Reserve 1 TiB beyond the measured free space, so space that other jobs on a shared runner free before the build's own fstatvfs read is too small to admit the build.
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, refused_out, real_avail + (1ULL << 40), 32'768));
        }, "preflight reserve deficit refused");
        require(count_entries(out_dir) == 0, "no temporary files left after preflight reserve refusal");

        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, refused_out, std::numeric_limits<uint64_t>::max(), 32'768));
        }, "overflowing reserve requirement refused");
        require(count_entries(out_dir) == 0, "no temporary files left after overflow reserve refusal");

        size_t space_calls = 0;
        const uint64_t test_reserve = 50ULL * 1024 * 1024 * 1024;
        refuses([&] {
            static_cast<void>(build_derived_pack_with_space_query(
                source,
                mtp,
                refused_out,
                test_reserve,
                32'768,
                [&](int) -> uint64_t {
                    ++space_calls;
                    // Allow preflight (call 1) and chunk 0 of tensor 0 (call 2), then drop below reserve on chunk 1 (call 3).
                    if (space_calls <= 2) {
                        return planned.file_bytes + test_reserve + (1ULL << 20);
                    }
                    return test_reserve + 1024;
                }
            ));
        }, "mid-stream chunk reserve drop refused before consuming reserve");
        require(space_calls == 3, "space query ran on preflight and every chunk write");
        require(count_entries(out_dir) == 0, "RAII temporary unlinked on mid-stream reserve refusal");

        // 3. Build and exclusive publication checks
        const auto pack_path = out_dir / "tokspur-pack-v2.asa.gguf";
        const auto built = build_derived_pack(source, mtp, pack_path, 1ULL << 20, 32'768);
        require(built.file_bytes == planned.file_bytes, "built file_bytes matches plan");
        require(count_entries(out_dir) == 1, "only sealed pack present in out_dir");

        struct stat st{};
        require(::lstat(pack_path.c_str(), &st) == 0, "lstat on built pack");
        require(S_ISREG(st.st_mode) && (st.st_mode & 0777) == 0444, "built pack sealed 0444");
        require(uint64_t(st.st_size) == planned.file_bytes, "built pack exact byte length without ftruncate");

        // Refuses to overwrite existing regular file
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, pack_path, 1ULL << 20, 32'768));
        }, "existing pack file refused without overwrite");
        require(count_entries(out_dir) == 1, "no leaked temporary after existing-file refusal");

        // Refuses to overwrite existing symlink (including dangling symlink)
        const auto symlink_out = out_dir / "dangling-link.gguf";
        require(::symlink("nonexistent-target", symlink_out.c_str()) == 0, "create dangling symlink");
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, symlink_out, 1ULL << 20, 32'768));
        }, "existing dangling symlink refused");
        require(::unlink(symlink_out.c_str()) == 0, "remove dangling symlink");

        // Refuses symlink parent directory component via O_NOFOLLOW traversal
        const auto sym_dir = root.path / "sym_out";
        require(::symlink(out_dir.c_str(), sym_dir.c_str()) == 0, "create directory symlink");
        refuses([&] {
            static_cast<void>(build_derived_pack(source, mtp, sym_dir / "via-symlink.gguf", 1ULL << 20, 32'768));
        }, "symlink directory component refused");
        require(::unlink(sym_dir.c_str()) == 0, "remove directory symlink");

        // 4. Adoption and tampering checks
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, pack_path, 32'768, 1));
        }, "default adopt_derived_pack enforces full 1,254-tensor census on partial fixture");
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, pack_path, 32'768, 1, TrustMode::local_seal, false));
        }, "local-seal always enforces full 1,254-tensor census");
        const auto adopted1 = adopt_derived_pack(source, mtp, pack_path, 32'768, 1, TrustMode::full, false);
        const auto adopted4 = adopt_derived_pack(source, mtp, pack_path, 32'768, 4, TrustMode::full, false);
        require(adopted1.manifest_sha256 == adopted4.manifest_sha256, "single and multi-worker adoption match");
        require(adopted1.model->architecture() == pack_architecture, "adopted pack architecture");
        auto altered_source = shard0;
        altered_source[2].payload.back() ^= std::byte(0x01);
        const auto altered_source_path = source_dir / "altered-source.gguf";
        write_gguf_shard(altered_source_path, "qwen4exp", altered_source, 0, 2, uint32_t(shard0.size() + shard1.size()));
        const auto changed_source = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{altered_source_path, shard1_path});
        refuses([&] {
            static_cast<void>(adopt_derived_pack(changed_source, mtp, pack_path, 32'768, 4, TrustMode::full, false));
        }, "corrupted original tail byte refused across authentication chunks");
        auto altered_mtp = mtp_specs;
        altered_mtp.back().payload.back() ^= std::byte(0x01);
        const auto altered_mtp_path = source_dir / "altered-mtp.gguf";
        write_gguf_shard(altered_mtp_path, "qwen4exp", altered_mtp);
        const auto changed_mtp = std::make_shared<const ModelFiles>(std::vector<std::filesystem::path>{altered_mtp_path});
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, changed_mtp, pack_path, 32'768, 4, TrustMode::full, false));
        }, "corrupted original MTP bytes refused during authentication");
        refuses([&] {
            static_cast<void>(plan_derived_pack(*adopted1.model, nullptr));
        }, "repacking tokspur_pack refused");

        // Writable pack refused
        const auto bad_pack = out_dir / "tampered.gguf";
        std::filesystem::copy_file(pack_path, bad_pack);
        require(::chmod(bad_pack.c_str(), 0644) == 0, "make copy writable");
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, bad_pack, 32'768, 1, TrustMode::full, false));
        }, "writable pack refused by adopt_derived_pack");

        // Flip 1 byte in first tensor payload -> refused even when re-sealed 0444
        {
            const int fd = ::open(bad_pack.c_str(), O_RDWR | O_CLOEXEC);
            require(fd >= 0, "open tampered pack");
            std::byte b{};
            require(::pread(fd, &b, 1, off_t(built.data_offset)) == 1, "read payload byte");
            b ^= std::byte(0x01);
            require(::pwrite(fd, &b, 1, off_t(built.data_offset)) == 1, "flip payload byte");
            require(::fchmod(fd, 0444) == 0, "re-seal tampered pack");
            ::close(fd);
        }
        refuses([&] {
            static_cast<void>(adopt_derived_pack(source, mtp, bad_pack, 32'768, 2, TrustMode::full, false));
        }, "corrupted payload byte refused by adopt_derived_pack");

        std::cout << "PASS derived pack checks=" << checks << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
