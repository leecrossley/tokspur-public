#include "tokspur/derived_pack.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/json.hpp"
#include "tokspur/lane_sidecar.hpp"
#include "tokspur/numeric.hpp"

#ifndef __APPLE__
#error "The private derived weight pack requires Darwin and CommonCrypto"
#endif
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>

namespace tokspur::derived_pack {
namespace {
using Bytes = std::vector<std::byte>;
constexpr uint64_t max_header_bytes = 16 * 1024 * 1024; // 16 MiB manifest header ceiling

void require(bool condition, const char* message) {
    if (!condition) throw FormatError(message);
}

uint64_t aligned(uint64_t n, uint64_t boundary = pack_alignment) {
    return checked_add(n, boundary - 1) / boundary * boundary;
}

struct Fd {
    int value = -1;
    explicit Fd(int n) : value(n) { require(n >= 0, "derived pack file open refused"); }
    ~Fd() { if (value >= 0) ::close(value); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& o) noexcept : value(o.value) { o.value = -1; }
    Fd& operator=(Fd&& o) noexcept {
        if (this != &o) {
            if (value >= 0) ::close(value);
            value = o.value;
            o.value = -1;
        }
        return *this;
    }
};

struct Hash {
    CC_SHA256_CTX state{};
    Hash() { require(CC_SHA256_Init(&state) == 1, "SHA-256 initialisation failed"); }
    void add(std::span<const std::byte> data) {
        require(data.size() <= std::numeric_limits<CC_LONG>::max(), "SHA-256 chunk too large");
        require(CC_SHA256_Update(&state, data.data(), CC_LONG(data.size())) == 1, "SHA-256 update failed");
    }
    std::string finish() {
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{};
        require(CC_SHA256_Final(bytes.data(), &state) == 1, "SHA-256 finalisation failed");
        constexpr char hex[] = "0123456789abcdef";
        std::string out;
        for (auto b : bytes) { out += hex[b >> 4]; out += hex[b & 15]; }
        return out;
    }
};

} // namespace

std::string hash(std::string_view text) {
    Hash h;
    h.add(std::as_bytes(std::span(text.data(), text.size())));
    return h.finish();
}

namespace {

void number(Bytes& b, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) b.push_back(std::byte((value >> (8 * i)) & 255));
}

void string(Bytes& b, std::string_view value) {
    number(b, value.size(), 8);
    for (unsigned char c : value) b.push_back(std::byte(c));
}

void metadata_string(Bytes& b, const char* key, std::string_view value) {
    string(b, key); number(b, 8, 4); string(b, value);
}

void metadata_uint(Bytes& b, const char* key, uint32_t value) {
    string(b, key); number(b, 4, 4); number(b, value, 4);
}

std::string build_manifest(const PackPlan& plan) {
    Json::Array tensors;
    for (const auto& t : plan.tensors) {
        Json::Array dimensions;
        for (auto d : t.dimensions) dimensions.push_back(Json::integer(d));
        tensors.push_back(Json(Json::Object{
            {"name", Json(t.name)},
            {"type", Json::integer(t.type)},
            {"layout", Json(t.layout)},
            {"dimensions", Json(std::move(dimensions))},
            {"bytes", Json::integer(t.bytes)},
            {"offset", Json::integer(t.offset)},
            {"source_sha256", Json(t.source_sha256.empty() ? std::string(64, '0') : t.source_sha256)},
            {"packed_sha256", Json(t.packed_sha256.empty() ? std::string(64, '0') : t.packed_sha256)}
        }));
    }
    return Json(Json::Object{
        {"schema", Json::integer(pack_version)},
        {"recipe", Json(pack_recipe)},
        {"source_architecture", Json(source_architecture)},
        {"target_revision", Json(target_revision)},
        {"mtp_sidecar_sha256", Json(mtp_sidecar_sha256)},
        {"tensors", Json(std::move(tensors))}
    }).dump(max_header_bytes);
}

} // namespace

Bytes build_header(const PackPlan& plan) {
    Bytes b;
    number(b, 0x46554747, 4); // "GGUF" magic
    number(b, 3, 4);          // GGUF v3
    number(b, plan.tensors.size(), 8);
    number(b, 5, 8);          // 5 metadata fields
    metadata_string(b, "general.architecture", pack_architecture);
    metadata_uint(b, "general.alignment", uint32_t(pack_alignment));
    metadata_uint(b, "tokspur_pack.version", pack_version);
    metadata_string(b, "tokspur_pack.recipe", pack_recipe);
    metadata_string(b, "tokspur_pack.manifest", build_manifest(plan));

    for (const auto& t : plan.tensors) {
        string(b, t.name);
        number(b, t.dimensions.size(), 4);
        for (auto d : t.dimensions) number(b, d, 8);
        number(b, t.type, 4);
        number(b, t.offset, 8);
    }
    require(b.size() <= max_header_bytes, "derived pack header exceeds bound");
    b.resize(aligned(b.size(), pack_alignment));
    return b;
}

namespace {

uint64_t chunk_row_bound(const PackTensor& tensor, uint64_t maximum) {
    if (tensor.layout == "plain") return maximum;
    const auto row = lanes::row_layout(tensor.type, uint32_t(tensor.dimensions[0])).row_bytes;
    require(row <= maximum, "streaming chunk cannot hold a complete row");
    return maximum / row * row;
}

void write_at(int fd, std::span<const std::byte> bytes, uint64_t offset) {
    for (size_t n = 0; n < bytes.size();) {
        const auto got = ::pwrite(fd, bytes.data() + n, bytes.size() - n, off_t(checked_add(offset, n)));
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "derived pack pwrite failed");
        n += size_t(got);
    }
}

void read_into(int fd, std::span<std::byte> out, uint64_t offset) {
    for (size_t n = 0; n < out.size();) {
        const auto got = ::pread(fd, out.data() + n, out.size() - n, off_t(checked_add(offset, n)));
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "derived pack pread failed");
        n += size_t(got);
    }
}

Bytes read_at(int fd, size_t bytes, uint64_t offset) {
    Bytes out(bytes);
    read_into(fd, out, offset);
    return out;
}

Fd open_dir_traverse(const std::filesystem::path& dir_path) {
    const std::filesystem::path p = dir_path.empty() ? std::filesystem::path(".") : dir_path;
    int current_fd = -1;
    if (p.is_absolute()) {
        current_fd = ::open("/", O_RDONLY | O_CLOEXEC | O_DIRECTORY);
        require(current_fd >= 0, "cannot open root directory");
    } else {
        current_fd = ::open(".", O_RDONLY | O_CLOEXEC | O_DIRECTORY);
        require(current_fd >= 0, "cannot open current directory");
    }
    Fd cur(current_fd);
    for (const auto& part : p) {
        const std::string s = part.string();
        if (s.empty() || s == "/" || s == ".") continue;
        int next_fd = ::openat(cur.value, s.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
        if (next_fd < 0) {
            if (errno == ELOOP) {
                throw FormatError("symlink directory component rejected via O_NOFOLLOW: ELOOP");
            }
            throw FormatError("cannot open directory component: " + s);
        }
        cur = Fd(next_fd);
    }
    return cur;
}

struct Location {
    std::string name;
    Fd directory;
    explicit Location(const std::filesystem::path& path)
        : name(path.filename().string()),
          directory(open_dir_traverse(path.parent_path())) {
        require(!name.empty() && name != "." && name != ".." && name.find('\0') == std::string::npos,
                "invalid derived pack filename");
    }
};

void check_absent(const Location& location) {
    struct stat s{};
    const int found = ::fstatat(location.directory.value, location.name.c_str(), &s, AT_SYMLINK_NOFOLLOW);
    require(found < 0 && errno == ENOENT, "derived pack output already exists or is inaccessible");
}

struct TemporaryFile {
    int directory, fd = -1;
    std::string name;
    bool published = false;
    explicit TemporaryFile(int parent) : directory(parent) {
        for (unsigned tries = 0; tries < 16; ++tries) {
            std::array<unsigned char, 16> random{};
            arc4random_buf(random.data(), random.size());
            constexpr char hex[] = "0123456789abcdef";
            name = ".tokspur-pack-v2-";
            for (auto x : random) { name += hex[x >> 4]; name += hex[x & 15]; }
            name += ".tmp";
            fd = ::openat(directory, name.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_RDWR | O_CLOEXEC, 0600);
            if (fd >= 0) return;
            require(errno == EEXIST, "cannot create exclusive derived pack temporary");
        }
        throw FormatError("cannot reserve derived pack temporary file");
    }
    ~TemporaryFile() {
        if (fd >= 0) ::close(fd);
        if (!published) ::unlinkat(directory, name.c_str(), 0);
    }
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
};

void parse_manifest_into_plan(PackPlan& plan, const Gguf& file) {
    const std::map<std::string, uint32_t> keys{
        {"general.architecture", 8},
        {"general.alignment", 4},
        {"tokspur_pack.version", 4},
        {"tokspur_pack.recipe", 8},
        {"tokspur_pack.manifest", 8}
    };
    require(file.metadata().size() == keys.size(), "unexpected metadata entries in pack GGUF");
    for (const auto& [key, type] : keys) {
        const auto* m = file.find(key);
        require(m && m->type == type, "wrong pack GGUF metadata type");
    }
    require(file.string_value("general.architecture") == pack_architecture &&
            file.unsigned_value("general.alignment") == pack_alignment &&
            file.unsigned_value("tokspur_pack.version") == pack_version &&
            file.string_value("tokspur_pack.recipe") == pack_recipe,
            "stale or incompatible derived pack metadata");

    const auto json = parse_json(file.string_value("tokspur_pack.manifest"), {max_header_bytes, 500'000, 16});
    const auto* tensors_val = json.find("tensors");
    require(tensors_val && tensors_val->kind() == Json::Kind::array, "manifest missing tensors array");
    const auto& entries = tensors_val->array();
    require(entries.size() == plan.tensors.size(), "manifest tensor count mismatch");

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& v = entries[i];
        auto& t = plan.tensors[i];
        const auto* name_p = v.find("name");
        require(name_p && name_p->string() == t.name, "manifest tensor name mismatch or out of order");
        const auto* type_p = v.find("type");
        const auto* bytes_p = v.find("bytes");
        const auto* offset_p = v.find("offset");
        const auto* src_hash_p = v.find("source_sha256");
        const auto* pack_hash_p = v.find("packed_sha256");
        require(type_p && type_p->unsigned_integer() == t.type, "manifest tensor type mismatch");
        require(bytes_p && bytes_p->unsigned_integer() == t.bytes, "manifest tensor byte mismatch");
        require(offset_p && offset_p->unsigned_integer() == t.offset, "manifest tensor offset mismatch");
        require(src_hash_p && pack_hash_p, "manifest missing SHA-256 digests");
        t.source_sha256 = src_hash_p->string();
        t.packed_sha256 = pack_hash_p->string();
    }
}
} // namespace

bool is_gpu_pack_tensor(std::string_view name) {
    return !is_cpu_pinned_embedding(name);
}

bool verify_gpu_pack_census(const PackPlan& plan) {
    if (plan.tensors.size() != expected_gpu_tensors) return false;
    bool has_output = false;
    bool has_output_hc = false;
    bool has_gdn = false;
    bool has_qsa_l3 = false;
    bool has_ple_l1 = false;
    size_t mtp_count = 0;
    std::set<std::string> seen;

    for (const auto& t : plan.tensors) {
        if (!is_gpu_pack_tensor(t.name)) return false;
        if (!seen.insert(t.name).second) return false;
        if (t.name == "output.weight") has_output = true;
        if (t.name == "output_hc_norm.weight") has_output_hc = true;
        if (t.name == "blk.0.ssm_a") has_gdn = true;
        if (t.name == "blk.3.attn_q.weight") has_qsa_l3 = true;
        if (t.name == "blk.1.ple_key.weight") has_ple_l1 = true;
        if (t.name.rfind("blk.48.", 0) == 0) ++mtp_count;
    }
    return has_output && has_output_hc && has_gdn && has_qsa_l3 && has_ple_l1 && mtp_count == 32;
}

bool check_disk_reserve(uint64_t available_bytes, uint64_t peak_additional_bytes, uint64_t reserve_bytes) {
    if (available_bytes == 0) return false;
    try {
        const auto required = checked_add(peak_additional_bytes, reserve_bytes);
        return available_bytes >= required;
    } catch (const FormatError&) {
        return false;
    }
}

PackPlan plan_derived_pack(const ModelFiles& source, const ModelFiles* mtp_sidecar, uint64_t byte_budget) {
    source.assert_unchanged();
    require(source.architecture() != pack_architecture, "cannot repack an existing derived pack");

    PackPlan plan;
    std::set<std::string> seen;

    // Collect eligible GPU tensors from main source model
    for (const auto& [name, loc] : source.tensors()) {
        static_cast<void>(loc);
        if (!is_gpu_pack_tensor(name)) continue;
        const auto& t = source.tensor(name);
        std::string layout = "plain";
        if (lanes::lane_sidecar_eligible(t)) {
            if (t.type == 8) layout = "q8_lanes_v1";
            else if (t.type == 7) layout = "q51_lanes_v1";
        }
        require(seen.insert(name).second, "duplicate tensor in source");
        plan.tensors.push_back({name, t.type, layout, t.dimensions, t.bytes, 0, {}, {}});
        plan.payload_bytes = checked_add(plan.payload_bytes, t.bytes);
    }

    // Collect MTP blk.48 sidecar tensors if provided
    if (mtp_sidecar) {
        mtp_sidecar->assert_unchanged();
        for (const auto& [name, loc] : mtp_sidecar->tensors()) {
            static_cast<void>(loc);
            if (!is_gpu_pack_tensor(name)) continue;
            const auto& t = mtp_sidecar->tensor(name);
            std::string layout = "plain";
            if (lanes::lane_sidecar_eligible(t)) {
                if (t.type == 8) layout = "q8_lanes_v1";
                else if (t.type == 7) layout = "q51_lanes_v1";
            }
            if (seen.insert(name).second) {
                plan.tensors.push_back({name, t.type, layout, t.dimensions, t.bytes, 0, {}, {}});
                plan.payload_bytes = checked_add(plan.payload_bytes, t.bytes);
            }
        }
    }

    require(!plan.tensors.empty(), "empty derived pack tensor selection");

    // Compute 16 KiB page-aligned offsets for each tensor
    uint64_t relative_offset = 0;
    for (auto& t : plan.tensors) {
        relative_offset = aligned(relative_offset, pack_alignment);
        t.offset = relative_offset;
        relative_offset = checked_add(relative_offset, t.bytes);
    }

    // Compute header and total file size
    const auto initial_header = build_header(plan);
    plan.data_offset = initial_header.size();
    plan.file_bytes = checked_add(plan.data_offset, relative_offset);

    require(plan.file_bytes <= byte_budget, "planned derived pack exceeds byte budget");
    source.assert_unchanged();
    if (mtp_sidecar) mtp_sidecar->assert_unchanged();
    return plan;
}

PackPlan build_derived_pack_with_space_query(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes,
    const std::function<uint64_t(int)>& query_available_bytes
) {
    require(bool(source), "missing source model files");
    require(bool(query_available_bytes), "missing directory space query");
    require(chunk_bytes >= 32'768 && chunk_bytes <= 67'108'864, "invalid chunk byte bound");

    auto plan = plan_derived_pack(*source, mtp_sidecar.get());
    Location location(output);
    check_absent(location);

    // Verify filesystem reserve before starting construction without crediting any existing pack
    const uint64_t available_bytes = query_available_bytes(location.directory.value);
    require(check_disk_reserve(available_bytes, plan.file_bytes, reserve_bytes),
            "insufficient disk space: cannot satisfy peak allocation + 50 GB reserve");

    TemporaryFile temporary(location.directory.value);
    require(::fcntl(temporary.fd, F_NOCACHE, 1) == 0, "cannot disable buffer cache on temporary pack");

    const auto initial_header = build_header(plan);
    write_at(temporary.fd, initial_header, 0);
    uint64_t written_end = initial_header.size();

    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        auto& t = plan.tensors[i];

        std::span<const std::byte> original;
        if (source->tensors().contains(t.name)) {
            original = source->tensor_bytes(t.name);
        } else if (mtp_sidecar && mtp_sidecar->tensors().contains(t.name)) {
            original = mtp_sidecar->tensor_bytes(t.name);
        } else {
            throw FormatError("tensor missing from both source and MTP sidecar");
        }

        const auto tensor_start = checked_add(plan.data_offset, t.offset);
        require(tensor_start >= written_end, "nonmonotonic derived pack tensor offset");
        const uint64_t pending_pad = tensor_start - written_end;
        require(pending_pad < pack_alignment, "excessive alignment padding before derived pack tensor");

        const auto chunk_size = chunk_row_bound(t, chunk_bytes);
        Hash src_hash, pack_hash;
        Bytes packed_buf(chunk_size), restored_buf(chunk_size);

        for (uint64_t at = 0; at < t.bytes;) {
            source->assert_unchanged();
            if (mtp_sidecar) mtp_sidecar->assert_unchanged();

            const auto n = std::min(chunk_size, t.bytes - at);
            const uint64_t pad_bytes = (at == 0) ? pending_pad : 0ULL;
            const auto upcoming = checked_add(pad_bytes, n);
            const auto remaining_pack_bytes = plan.file_bytes - written_end;
            const uint64_t current_avail = query_available_bytes(location.directory.value);
            require(check_disk_reserve(current_avail, upcoming, reserve_bytes) &&
                    check_disk_reserve(current_avail, remaining_pack_bytes, reserve_bytes),
                    "disk space dropped below 50 GB reserve during streaming");

            if (pad_bytes > 0) {
                const Bytes zero_pad(size_t(pad_bytes), std::byte(0));
                write_at(temporary.fd, zero_pad, written_end);
                written_end = checked_add(written_end, pad_bytes);
            }

            const auto input = original.subspan(at, n);
            const auto tensor_file_pos = checked_add(plan.data_offset, checked_add(t.offset, at));
            require(written_end == tensor_file_pos, "derived pack sequential write cursor mismatch");

            if (t.layout == "plain") {
                write_at(temporary.fd, input, tensor_file_pos);
                auto restored = std::span(restored_buf).first(n);
                read_into(temporary.fd, restored, tensor_file_pos);
                require(std::ranges::equal(input, restored), "plain tensor readback byte proof failed");
                src_hash.add(input);
                pack_hash.add(input);
            } else {
                const auto layout = lanes::row_layout(t.type, uint32_t(t.dimensions[0]));
                auto out = std::span(packed_buf).first(n);
                auto inverse = std::span(restored_buf).first(n);
                lanes::pack_rows(layout, input, out);
                write_at(temporary.fd, out, tensor_file_pos);
                read_into(temporary.fd, out, tensor_file_pos);
                lanes::unpack_rows(layout, out, inverse);
                require(std::ranges::equal(input, inverse), "lane packing inverse byte proof failed");
                src_hash.add(input);
                pack_hash.add(out);
            }
            written_end = checked_add(written_end, n);
            at += n;
        }

        t.source_sha256 = src_hash.finish();
        t.packed_sha256 = pack_hash.finish();
    }

    require(written_end == plan.file_bytes, "derived pack final size mismatch");

    const auto final_header = build_header(plan);
    require(final_header.size() == initial_header.size(), "manifest changed header length");
    write_at(temporary.fd, final_header, 0);
    require(read_at(temporary.fd, final_header.size(), 0) == final_header, "pack header readback differs");
    require(::fchmod(temporary.fd, 0444) == 0 && ::fsync(temporary.fd) == 0, "pack seal/fsync failed");

    source->assert_unchanged();
    if (mtp_sidecar) mtp_sidecar->assert_unchanged();
    check_absent(location);

    require(::renameatx_np(location.directory.value, temporary.name.c_str(),
                           location.directory.value, location.name.c_str(), RENAME_EXCL) == 0,
            "exclusive pack publication refused");
    temporary.published = true;
    require(::fsync(location.directory.value) == 0, "directory fsync failed after publication");
    return plan;
}

PackPlan build_derived_pack(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes
) {
    return build_derived_pack_with_space_query(
        std::move(source),
        std::move(mtp_sidecar),
        output,
        reserve_bytes,
        chunk_bytes,
        [](int dir_fd) -> uint64_t {
            struct statvfs vfs{};
            require(::fstatvfs(dir_fd, &vfs) == 0, "statvfs on target directory failed");
            return checked_mul(uint64_t(vfs.f_bavail), uint64_t(vfs.f_frsize));
        }
    );
}

AdoptedPack adopt_derived_pack(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint64_t chunk_bytes,
    uint32_t authentication_workers,
    TrustMode trust_mode,
    bool require_full_census
) {
    require(authentication_workers && authentication_workers <= max_authentication_workers,
            "authentication workers must be between 1 and 16");
    require(bool(source), "missing source model files");
    require(chunk_bytes >= 32'768 && chunk_bytes <= 67'108'864, "invalid chunk byte bound");

    auto plan = plan_derived_pack(*source, mtp_sidecar.get());
    Location location(path);
    Fd fd(::openat(location.directory.value, location.name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    require(fd.value >= 0, "cannot open derived pack file");

    struct stat before{};
    require(::fstat(fd.value, &before) == 0 && S_ISREG(before.st_mode) && !(before.st_mode & 0222) &&
            before.st_nlink == 1 && before.st_size > 0,
            "derived pack must be an immutable regular file");

    constexpr auto mapping =
#if defined(TOKSPUR_PRIVATE_PACK_SHARED_MAPPING) && TOKSPUR_PRIVATE_PACK_SHARED_MAPPING
        ReadOnlyMapMode::shared_file;
#else
        ReadOnlyMapMode::private_copy_on_write;
#endif
    auto model = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{"/dev/fd/" + std::to_string(fd.value)}, mapping);
    require(model->architecture() == pack_architecture && model->shards().size() == 1,
            "adopted file is not a valid tokspur_pack GGUF");

    const auto& file = model->shards()[0];
    if (!mtp_sidecar && model->tensors().size() == expected_gpu_tensors) {
        const auto mjson = parse_json(file.string_value("tokspur_pack.manifest"), {max_header_bytes, 500'000, 16});
        const auto* mtensors = mjson.find("tensors");
        require(mtensors && mtensors->kind() == Json::Kind::array && mtensors->array().size() == expected_gpu_tensors,
                "manifest tensor count mismatch");
        std::set<std::string> seen;
        for (const auto& t : plan.tensors) seen.insert(t.name);
        for (size_t i = plan.tensors.size(); i < expected_gpu_tensors; ++i) {
            const auto& entry = mtensors->array()[i];
            const std::string tname(entry.find("name") ? entry.find("name")->string() : "");
            require(tname.starts_with("blk.48.") && is_gpu_pack_tensor(tname) && seen.insert(tname).second &&
                    model->tensors().contains(tname), "unexpected trailing pack tensor without MTP sidecar");
            const auto& meta = model->tensor(tname);
            std::string layout = "plain";
            if (lanes::lane_sidecar_eligible(meta)) {
                if (meta.type == 8) layout = "q8_lanes_v1";
                else if (meta.type == 7) layout = "q51_lanes_v1";
            }
            plan.tensors.push_back({tname, meta.type, layout, meta.dimensions, meta.bytes, 0, {}, {}});
            plan.payload_bytes = checked_add(plan.payload_bytes, meta.bytes);
        }
        uint64_t relative_offset = 0;
        for (auto& t : plan.tensors) {
            relative_offset = aligned(relative_offset, pack_alignment);
            t.offset = relative_offset;
            relative_offset = checked_add(relative_offset, t.bytes);
        }
        const auto hdr = build_header(plan);
        plan.data_offset = hdr.size();
        plan.file_bytes = checked_add(plan.data_offset, relative_offset);
    }
    require(uint64_t(before.st_size) == plan.file_bytes,
            "derived pack must be an immutable regular file of the exact planned size");

    parse_manifest_into_plan(plan, file);

    const auto canonical_header = build_header(plan);
    require(file.data_offset() == canonical_header.size() &&
            read_at(fd.value, canonical_header.size(), 0) == canonical_header,
            "noncanonical or corrupt derived pack header");

    if (require_full_census || trust_mode == TrustMode::local_seal) {
        require(verify_gpu_pack_census(plan), "derived pack GPU census verification failed");
    }

    bool seal_valid = false;
    if (trust_mode == TrustMode::local_seal) {
        std::string seal_name = std::filesystem::path(location.name).stem().string() + ".seal";
        int seal_fd = ::openat(location.directory.value, seal_name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (seal_fd < 0) {
            if (errno == ELOOP) {
                throw FormatError("symlink seal path rejected via O_NOFOLLOW: ELOOP");
            }
            struct stat lchk{};
            if (::fstatat(location.directory.value, seal_name.c_str(), &lchk, AT_SYMLINK_NOFOLLOW) == 0 && S_ISLNK(lchk.st_mode)) {
                throw FormatError("symlink seal path rejected via O_NOFOLLOW: ELOOP");
            }
            // Missing seal file triggers graceful fallback to full verification
        } else {
            Fd seal_guard(seal_fd);
            struct stat sstat{};
            require(::fstat(seal_fd, &sstat) == 0, "cannot stat seal file");
            if (!S_ISREG(sstat.st_mode)) {
                throw FormatError("seal file must be a regular file");
            }
            if (sstat.st_mode & 0222) {
                throw FormatError("writable seal file rejected: permissions must be immutable (!(st_mode & 0222))");
            }
            if (sstat.st_nlink != 1) {
                throw FormatError("multi-hardlink seal file rejected: st_nlink > 1");
            }
            if (sstat.st_size > 0 && sstat.st_size <= 4096) {
                std::string seal_content(size_t(sstat.st_size), '\0');
                read_into(seal_fd, std::as_writable_bytes(std::span(seal_content)), 0);
                while (!seal_content.empty() && (seal_content.back() == '\n' || seal_content.back() == '\r' || seal_content.back() == ' ')) {
                    seal_content.pop_back();
                }
                auto manifest_sha256 = hash(file.string_value("tokspur_pack.manifest"));
                if (seal_content == manifest_sha256) {
                    seal_valid = true;
                }
            }
            // Empty seal or content mismatch triggers graceful fallback to full verification
        }
    }

    const auto authenticate_tensor = [&](size_t i) {
        const auto& t = plan.tensors[i];
        const auto& packed_meta = model->tensor(t.name);
        require(packed_meta.type == t.type && packed_meta.dimensions == t.dimensions &&
                packed_meta.bytes == t.bytes &&
                packed_meta.offset == checked_add(canonical_header.size(), t.offset),
                "derived pack tensor metadata/offset mismatch");

        std::span<const std::byte> original;
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) || defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM)
        const ModelFiles* origin_files = nullptr;
#endif
        bool pack_only_trailing_mtp = false;
        if (source->tensors().contains(t.name)) {
            original = source->tensor_bytes(t.name);
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) || defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM)
            origin_files = source.get();
#endif
        } else if (mtp_sidecar && mtp_sidecar->tensors().contains(t.name)) {
            original = mtp_sidecar->tensor_bytes(t.name);
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) || defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM)
            origin_files = mtp_sidecar.get();
#endif
        } else if (!mtp_sidecar && t.name.starts_with("blk.48.")) {
            pack_only_trailing_mtp = true;
        } else {
            throw FormatError("tensor missing during adoption");
        }

        const auto packed_bytes = model->tensor_bytes(t.name);
        const auto chunk_size = chunk_row_bound(t, chunk_bytes);
        Bytes restored_buf(chunk_size);
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        std::optional<TensorBackingFile> original_backing;
        std::optional<Fd> original_reader;
        Bytes original_buffer;
        if (!pack_only_trailing_mtp) {
            require(origin_files != nullptr, "missing original tensor file owner");
            original_backing = origin_files->tensor_backing_file(t.name);
            require(original_backing && original_backing->bytes == t.bytes, "missing original tensor backing file");
            original_reader.emplace(::open(original_backing->path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK));
            struct stat info{};
            require(::fstat(original_reader->value, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
                    uint64_t(info.st_dev) == original_backing->device && uint64_t(info.st_ino) == original_backing->inode &&
                    checked_add(original_backing->offset, original_backing->bytes) <= uint64_t(info.st_size),
                    "original tensor backing file changed");
            require(::fcntl(original_reader->value, F_NOCACHE, 1) == 0, "cannot disable original authentication read cache");
            original_buffer.resize(std::min(chunk_size, t.bytes));
        }
#endif
        Hash src_hash, pack_hash;

        for (uint64_t at = 0; at < t.bytes;) {
            source->assert_unchanged();
            if (mtp_sidecar) mtp_sidecar->assert_unchanged();
            model->assert_unchanged();

            const auto n = std::min(chunk_size, t.bytes - at);
            const auto packed_chunk = packed_bytes.subspan(at, n);
            if (!pack_only_trailing_mtp) {
                auto input = original.subspan(at, n);
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
                auto buffer = std::span(original_buffer).first(n);
                read_into(original_reader->value, buffer, checked_add(original_backing->offset, at));
                input = buffer;
#endif
                if (t.layout == "plain") {
                    require(std::ranges::equal(input, packed_chunk), "plain tensor byte proof failed");
                } else {
                    const auto layout = lanes::row_layout(t.type, uint32_t(t.dimensions[0]));
                    auto inverse = std::span(restored_buf).first(n);
                    lanes::unpack_rows(layout, packed_chunk, inverse);
                    require(std::ranges::equal(input, inverse), "lane unpacking inverse byte proof failed");
                }
                src_hash.add(input);
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) || defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM)
                if (origin_files) origin_files->evict_tensor_pages(t.name, at, n);
#endif
            }
            pack_hash.add(packed_chunk);
            at += n;
        }

        require((pack_only_trailing_mtp || src_hash.finish() == t.source_sha256) && pack_hash.finish() == t.packed_sha256,
                "stale source or corrupt packed tensor digest");
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) || defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM)
        if (origin_files) origin_files->evict_tensor_pages(t.name, 0, t.bytes);
#endif

        // Verify zero-padding in intervals between consecutive tensors
        if (i + 1 < plan.tensors.size()) {
            const auto end = checked_add(t.offset, t.bytes);
            const auto next_start = plan.tensors[i + 1].offset;
            if (next_start > end) {
                const auto pad_len = next_start - end;
                const auto padding = read_at(fd.value, size_t(pad_len), checked_add(canonical_header.size(), end));
                require(std::all_of(padding.begin(), padding.end(), [](std::byte b) { return b == std::byte(0); }),
                        "nonzero padding between derived pack tensors");
            }
        }
    };

    uint32_t effective_workers = 1;
    if (!seal_valid) {
        effective_workers = std::min(authentication_workers, uint32_t(plan.tensors.size()));
        if (effective_workers == 1) {
            for (size_t i = 0; i < plan.tensors.size(); ++i) authenticate_tensor(i);
        } else {
            std::atomic<size_t> next{0};
            std::vector<std::exception_ptr> failures(plan.tensors.size());
            const auto work = [&] {
                for (;;) {
                    const auto i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= plan.tensors.size()) return;
                    try { authenticate_tensor(i); } catch (...) { failures[i] = std::current_exception(); }
                }
            };
            std::vector<std::thread> pool;
            pool.reserve(effective_workers - 1);
            std::exception_ptr launch_failure;
            try { for (uint32_t i = 1; i < effective_workers; ++i) pool.emplace_back(work); }
            catch (...) { launch_failure = std::current_exception(); }
            if (!launch_failure) work();
            for (auto& thread : pool) thread.join();
            if (launch_failure) std::rethrow_exception(launch_failure);
            for (const auto& f : failures) if (f) std::rethrow_exception(f);
        }
    }

    source->assert_unchanged();
    if (mtp_sidecar) mtp_sidecar->assert_unchanged();
    model->assert_unchanged();

    struct stat after{};
    require(::fstat(fd.value, &after) == 0 && before.st_dev == after.st_dev &&
            before.st_ino == after.st_ino && before.st_size == after.st_size &&
            before.st_mode == after.st_mode && before.st_nlink == after.st_nlink,
            "derived pack identity changed during adoption");

    auto manifest_sha256 = hash(file.string_value("tokspur_pack.manifest"));
    return {std::move(model), std::move(plan), std::move(manifest_sha256), effective_workers};
}

} // namespace tokspur::derived_pack
