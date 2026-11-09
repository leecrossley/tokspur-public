// Streaming builder and local-seal publisher for tokspur-pack-v3.asa.gguf.
#include "tokspur/json.hpp"
#include "tokspur/planar_pack_v3.hpp"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw tokspur::FormatError(msg);
}

std::string slurp_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    require(bool(in), "cannot open file: " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t statvfs_avail(const std::filesystem::path& dir) {
    struct statvfs vfs{};
    require(::statvfs(dir.c_str(), &vfs) == 0, "statvfs failed on " + dir.string());
    return static_cast<uint64_t>(vfs.f_bavail) * static_cast<uint64_t>(vfs.f_frsize);
}

void publish_immutable_seal(const std::filesystem::path& seal_path, const std::string& sha256_hex) {
    require(sha256_hex.size() == 64, "invalid manifest_sha256 length");
    const auto dir = seal_path.parent_path();
    const int dir_fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    require(dir_fd >= 0, "cannot open derived directory for seal");
    struct Guard {
        int fd;
        ~Guard() { if (fd >= 0) ::close(fd); }
    } dir_guard{dir_fd};

    const std::string tmp_name = ".tokspur-pack-v3-seal-" + std::to_string(::getpid()) + ".tmp";
    ::unlinkat(dir_fd, tmp_name.c_str(), 0);
    const int tmp_fd = ::openat(dir_fd, tmp_name.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY | O_CLOEXEC, 0600);
    require(tmp_fd >= 0, "cannot create temporary seal file");
    bool published = false;
    struct TmpGuard {
        int dir_fd;
        int fd;
        std::string name;
        bool* published;
        ~TmpGuard() {
            if (fd >= 0) ::close(fd);
            if (!*published) ::unlinkat(dir_fd, name.c_str(), 0);
        }
    } tmp_guard{dir_fd, tmp_fd, tmp_name, &published};

    const std::string payload = sha256_hex + "\n";
    const ssize_t wrote = ::write(tmp_fd, payload.data(), payload.size());
    require(wrote == static_cast<ssize_t>(payload.size()), "short write on seal temporary");
    require(::fchmod(tmp_fd, 0444) == 0 && ::fsync(tmp_fd) == 0, "seal chmod/fsync failed");
    require(::renameatx_np(dir_fd, tmp_name.c_str(), dir_fd, seal_path.filename().c_str(), RENAME_EXCL) == 0,
            "exclusive seal renameatx_np failed");
    published = true;
    require(::fsync(dir_fd) == 0, "directory fsync after seal failed");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path adoption_path = ".tokspur/adoption.json";
        std::filesystem::path output_path = ".tokspur/derived/tokspur-pack-v3.asa.gguf";
        std::filesystem::path pack_v2_path = ".tokspur/derived/tokspur-pack-v2.asa.gguf";
        if (!std::filesystem::is_regular_file(pack_v2_path) &&
            std::filesystem::is_regular_file(".tokspur/derived/tokspur-pack-v2.gguf")) {
            pack_v2_path = ".tokspur/derived/tokspur-pack-v2.gguf";
        }
        std::filesystem::path receipt_path = ".tokspur/receipts/pack-v3-build/receipt.json";
        uint32_t experts_per_chunk = 32;
        uint64_t chunk_bytes = 16ULL * 1024 * 1024;

        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            if (arg == "--adoption" && i + 1 < argc) adoption_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
            else if (arg == "--pack-v2" && i + 1 < argc) pack_v2_path = argv[++i];
            else if (arg == "--receipt" && i + 1 < argc) receipt_path = argv[++i];
            else if (arg == "--experts-per-chunk" && i + 1 < argc) experts_per_chunk = static_cast<uint32_t>(std::stoul(argv[++i]));
            else throw tokspur::FormatError("unknown option: " + std::string(arg));
        }

        const auto jget = [](const tokspur::Json& j, std::string_view key) -> const tokspur::Json& {
            const auto* p = j.find(key);
            require(p != nullptr, "missing JSON key: " + std::string(key));
            return *p;
        };
        const auto adoption = tokspur::parse_json(slurp_file(adoption_path));
        std::vector<std::filesystem::path> target_shards;
        for (const auto& asset : jget(jget(adoption, "target"), "assets").array()) {
            target_shards.emplace_back(jget(asset, "path").string());
        }
        const auto& draft_assets = jget(jget(adoption, "draft"), "assets").array();
        require(draft_assets.size() == 1, "expected 1 draft asset in adoption.json");
        const std::filesystem::path draft_path = jget(draft_assets[0], "path").string();

        auto source = std::make_shared<tokspur::ModelFiles>(target_shards);
        auto mtp = std::make_shared<tokspur::ModelFiles>(std::vector<std::filesystem::path>{draft_path});

        const auto budget = tokspur::planar_v3::measure_pack_v3_budget(
            *source, mtp.get(), output_path.parent_path(), pack_v2_path, experts_per_chunk);
        std::cout << "[pack-v3] preflight avail_bytes=" << budget.filesystem_available_bytes
                  << " planned_file_bytes=" << budget.planned_pack_v3_file_bytes
                  << " free_after_bytes=" << budget.free_after_pack_v3_bytes
                  << " ring_buffer_bytes=" << budget.streaming_ring_buffer_bytes
                  << " reserve_ok=" << (budget.reserve_satisfied ? "true" : "false")
                  << " ram_ok=" << (budget.ram_ceiling_satisfied ? "true" : "false") << std::endl;
        require(budget.reserve_satisfied, "50 GiB reserve check failed before pack v3 build");
        require(budget.ram_ceiling_satisfied, "streaming RAM ceiling check failed before pack v3 build");

        const auto t0 = std::chrono::steady_clock::now();
        const auto plan = tokspur::planar_v3::build_derived_pack_v3(
            source, mtp, output_path, experts_per_chunk,
            tokspur::derived_pack::default_reserve_bytes, chunk_bytes);
        const auto t1 = std::chrono::steady_clock::now();
        const double build_s = std::chrono::duration<double>(t1 - t0).count();

        tokspur::ModelFiles built_pack({output_path});
        const std::string manifest_sha256 = tokspur::derived_pack::hash(
            built_pack.shards()[0].string_value("tokspur_pack.manifest"));

        const std::filesystem::path seal_path = output_path.parent_path() / (output_path.stem().string() + ".seal");
        publish_immutable_seal(seal_path, manifest_sha256);

        const auto t2 = std::chrono::steady_clock::now();
        const auto adopted_local = tokspur::planar_v3::adopt_derived_pack_v3(
            source, mtp, output_path, experts_per_chunk, chunk_bytes, 16,
            tokspur::derived_pack::TrustMode::local_seal, true);
        const auto t3 = std::chrono::steady_clock::now();
        const double local_seal_s = std::chrono::duration<double>(t3 - t2).count();
        require(adopted_local.manifest_sha256 == manifest_sha256, "local-seal manifest_sha256 mismatch");

        const uint64_t free_after = statvfs_avail(output_path.parent_path());
        std::filesystem::create_directories(receipt_path.parent_path());
        tokspur::Json receipt(tokspur::Json::Object{
            {"schema", tokspur::Json::integer(1)},
            {"recipe", tokspur::Json(plan.recipe)},
            {"version", tokspur::Json::integer(plan.version)},
            {"output_path", tokspur::Json(output_path.string())},
            {"seal_path", tokspur::Json(seal_path.string())},
            {"manifest_sha256", tokspur::Json(manifest_sha256)},
            {"tensor_count", tokspur::Json::integer(plan.tensors.size())},
            {"source_payload_bytes", tokspur::Json::integer(plan.source_payload_bytes)},
            {"payload_bytes", tokspur::Json::integer(plan.payload_bytes)},
            {"file_bytes", tokspur::Json::integer(plan.file_bytes)},
            {"data_offset", tokspur::Json::integer(plan.data_offset)},
            {"q4k_superblocks", tokspur::Json::integer(plan.q4k_superblocks)},
            {"q5k_superblocks", tokspur::Json::integer(plan.q5k_superblocks)},
            {"original_12b_scale_header_bytes", tokspur::Json::integer(plan.original_12b_scale_header_bytes)},
            {"expanded_16b_scale_header_bytes", tokspur::Json::integer(plan.expanded_16b_scale_header_bytes)},
            {"streaming_ring_buffer_bytes", tokspur::Json::integer(budget.streaming_ring_buffer_bytes)},
            {"existing_pack_v2_file_bytes", tokspur::Json::integer(budget.existing_pack_v2_file_bytes)},
            {"free_before_bytes", tokspur::Json::integer(budget.filesystem_available_bytes)},
            {"free_after_bytes", tokspur::Json::integer(free_after)},
            {"required_reserve_bytes", tokspur::Json::integer(budget.required_reserve_bytes)},
            {"build_seconds", tokspur::Json::number(build_s)},
            {"local_seal_adopt_seconds", tokspur::Json::number(local_seal_s)},
        });
        {
            std::ofstream out(receipt_path, std::ios::binary | std::ios::trunc);
            require(bool(out), "cannot write receipt: " + receipt_path.string());
            out << receipt.dump() << "\n";
        }
        std::cout << "[pack-v3] DONE file_bytes=" << plan.file_bytes
                  << " manifest_sha256=" << manifest_sha256
                  << " build_s=" << build_s
                  << " local_seal_adopt_s=" << local_seal_s
                  << " free_after_bytes=" << free_after << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[pack-v3] ERROR: " << e.what() << std::endl;
        return 1;
    }
}
