#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/json.hpp"
#include "tokspur/lane_sidecar.hpp"
#include "tokspur/numeric.hpp"

#ifndef __APPLE__
#error "tokspur-pack-v3 requires Darwin and CommonCrypto"
#endif
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>

namespace tokspur::planar_v3 {
namespace {

using Bytes = std::vector<std::byte>;
constexpr uint64_t max_header_bytes = 24 * 1024 * 1024; // 24 MiB header ceiling for v3 manifest

void require(bool condition, const char* message) {
    if (!condition) throw FormatError(message);
}

uint64_t aligned(uint64_t n, uint64_t boundary = derived_pack::pack_alignment) {
    return checked_add(n, boundary - 1) / boundary * boundary;
}

uint16_t load_u16_le(const std::byte* p) noexcept {
    return static_cast<uint16_t>(std::to_integer<uint16_t>(p[0]) |
                                 (std::to_integer<uint16_t>(p[1]) << 8));
}

uint32_t load_u32_le(const std::byte* p) noexcept {
    return static_cast<uint32_t>(load_u16_le(p)) |
           (static_cast<uint32_t>(load_u16_le(p + 2)) << 16);
}

struct Fd {
    int value = -1;
    explicit Fd(int n) : value(n) { require(n >= 0, "pack v3 file open refused"); }
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

struct Sha256 {
    CC_SHA256_CTX state{};
    Sha256() { require(CC_SHA256_Init(&state) == 1, "SHA-256 init failed"); }
    void add(std::span<const std::byte> data) {
        size_t pos = 0;
        while (pos < data.size()) {
            const size_t step = std::min<size_t>(data.size() - pos, 1ULL << 30);
            require(CC_SHA256_Update(&state, data.data() + pos, static_cast<CC_LONG>(step)) == 1,
                    "SHA-256 update failed");
            pos += step;
        }
    }
    std::string finish() {
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
        require(CC_SHA256_Final(digest.data(), &state) == 1, "SHA-256 final failed");
        constexpr char hex[] = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (unsigned char b : digest) {
            out.push_back(hex[b >> 4]);
            out.push_back(hex[b & 15]);
        }
        return out;
    }
};

void push_number(Bytes& b, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) {
        b.push_back(std::byte((value >> (8 * i)) & 255));
    }
}

void push_string(Bytes& b, std::string_view value) {
    push_number(b, value.size(), 8);
    for (unsigned char c : value) b.push_back(std::byte(c));
}

void push_meta_string(Bytes& b, const char* key, std::string_view value) {
    push_string(b, key);
    push_number(b, 8, 4);
    push_string(b, value);
}

void push_meta_uint(Bytes& b, const char* key, uint32_t value) {
    push_string(b, key);
    push_number(b, 4, 4);
    push_number(b, value, 4);
}

std::string build_manifest_v3(const PackPlanV3& plan) {
    Json::Array tensors;
    tensors.reserve(plan.tensors.size());
    for (const auto& t : plan.tensors) {
        Json::Array dims;
        for (uint64_t d : t.dimensions) dims.push_back(Json::integer(d));
        Json::Array src_dims;
        for (uint64_t d : t.source_dimensions) src_dims.push_back(Json::integer(d));
        tensors.push_back(Json(Json::Object{
            {"name", Json(t.name)},
            {"type", Json::integer(t.type)},
            {"source_type", Json::integer(t.source_type)},
            {"layout", Json(t.layout)},
            {"dimensions", Json(std::move(dims))},
            {"source_dimensions", Json(std::move(src_dims))},
            {"source_bytes", Json::integer(t.source_bytes)},
            {"bytes", Json::integer(t.bytes)},
            {"offset", Json::integer(t.offset)},
            {"partner_name", Json(t.partner_name)},
            {"source_sha256", Json(t.source_sha256.empty() ? std::string(64, '0') : t.source_sha256)},
            {"packed_sha256", Json(t.packed_sha256.empty() ? std::string(64, '0') : t.packed_sha256)},
        }));
    }
    return Json(Json::Object{
        {"schema", Json::integer(plan.version)},
        {"recipe", Json(plan.recipe)},
        {"source_architecture", Json(derived_pack::source_architecture)},
        {"target_revision", Json(derived_pack::target_revision)},
        {"mtp_sidecar_sha256", Json(derived_pack::mtp_sidecar_sha256)},
        {"source_payload_bytes", Json::integer(plan.source_payload_bytes)},
        {"payload_bytes", Json::integer(plan.payload_bytes)},
        {"tensors", Json(std::move(tensors))},
    }).dump(max_header_bytes);
}

void write_at(int fd, std::span<const std::byte> bytes, uint64_t offset) {
    for (size_t n = 0; n < bytes.size();) {
        const auto got = ::pwrite(fd, bytes.data() + n, bytes.size() - n,
                                  static_cast<off_t>(checked_add(offset, n)));
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "pack v3 pwrite failed");
        n += static_cast<size_t>(got);
    }
}

void read_into(int fd, std::span<std::byte> out, uint64_t offset) {
    for (size_t n = 0; n < out.size();) {
        const auto got = ::pread(fd, out.data() + n, out.size() - n,
                                 static_cast<off_t>(checked_add(offset, n)));
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "pack v3 pread failed");
        n += static_cast<size_t>(got);
    }
}

Bytes read_at(int fd, size_t bytes, uint64_t offset) {
    Bytes out(bytes);
    read_into(fd, out, offset);
    return out;
}

Fd open_dir_traverse(const std::filesystem::path& dir_path) {
    const std::filesystem::path p = dir_path.empty() ? std::filesystem::path(".") : dir_path;
    int current_fd = p.is_absolute()
        ? ::open("/", O_RDONLY | O_CLOEXEC | O_DIRECTORY)
        : ::open(".", O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    require(current_fd >= 0, "cannot open base directory");
    Fd cur(current_fd);
    for (const auto& part : p) {
        const std::string s = part.string();
        if (s.empty() || s == "/" || s == ".") continue;
        const int next_fd = ::openat(cur.value, s.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
        if (next_fd < 0) {
            if (errno == ELOOP) throw FormatError("symlink directory component rejected via O_NOFOLLOW: ELOOP");
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
                "invalid pack v3 filename");
    }
};

void check_absent(const Location& location) {
    struct stat s{};
    const int found = ::fstatat(location.directory.value, location.name.c_str(), &s, AT_SYMLINK_NOFOLLOW);
    require(found < 0 && errno == ENOENT, "pack v3 output already exists or is inaccessible");
}

struct TemporaryFile {
    int directory = -1;
    int fd = -1;
    std::string name;
    bool published = false;
    explicit TemporaryFile(int parent) : directory(parent) {
        for (unsigned tries = 0; tries < 16; ++tries) {
            std::array<unsigned char, 16> random{};
            arc4random_buf(random.data(), random.size());
            constexpr char hex[] = "0123456789abcdef";
            name = ".tokspur-pack-v3-";
            for (unsigned char x : random) {
                name.push_back(hex[x >> 4]);
                name.push_back(hex[x & 15]);
            }
            name += ".tmp";
            fd = ::openat(directory, name.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_RDWR | O_CLOEXEC, 0600);
            if (fd >= 0) return;
            require(errno == EEXIST, "cannot create exclusive pack v3 temporary");
        }
        throw FormatError("cannot reserve pack v3 temporary file");
    }
    ~TemporaryFile() {
        if (fd >= 0) ::close(fd);
        if (!published && directory >= 0) ::unlinkat(directory, name.c_str(), 0);
    }
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
};

uint64_t query_dir_avail_bytes(int dir_fd) {
    struct statvfs vfs{};
    require(::fstatvfs(dir_fd, &vfs) == 0, "statvfs on target directory failed");
    return checked_mul(static_cast<uint64_t>(vfs.f_bavail), static_cast<uint64_t>(vfs.f_frsize));
}

bool ends_with(std::string_view s, std::string_view suffix) noexcept {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

const Tensor* lookup_tensor_meta(const ModelFiles& source, const ModelFiles* mtp_sidecar, std::string_view name) {
    if (source.tensors().contains(name)) return &source.tensor(name);
    if (mtp_sidecar && mtp_sidecar->tensors().contains(name)) return &mtp_sidecar->tensor(name);
    return nullptr;
}

std::span<const std::byte> lookup_tensor_bytes(const ModelFiles& source, const ModelFiles* mtp_sidecar, std::string_view name) {
    if (source.tensors().contains(name)) return source.tensor_bytes(name);
    if (mtp_sidecar && mtp_sidecar->tensors().contains(name)) return mtp_sidecar->tensor_bytes(name);
    throw FormatError("tensor missing from source and MTP sidecar: " + std::string(name));
}

// Keep the original model mapping alive for CPU embeddings and identity checks.
// The opt-in path mirrors v2 authentication: read only the original tensor
// through an identity-checked nocache descriptor, then advise its clean pages
// away after comparison/hash. The adopted pack is never an eviction source.
class AuthenticationSource {
public:
    AuthenticationSource(const ModelFiles& source, const ModelFiles* mtp_sidecar,
                         std::string_view name, bool present, uint64_t chunk_bytes)
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        : name_(name), chunk_bytes_(chunk_bytes)
#endif
    {
#if !defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) || !TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        static_cast<void>(chunk_bytes);
#endif
        if (!present) return; // A trailing MTP tensor may exist only in the pack.
        mapped_ = lookup_tensor_bytes(source, mtp_sidecar, name);
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        owner_ = source.tensors().contains(name) ? &source : mtp_sidecar;
        require(owner_ != nullptr && derived_pack::is_gpu_pack_tensor(name),
                "missing GPU tensor authentication source");
        backing_ = owner_->tensor_backing_file(name);
        require(backing_ && backing_->bytes == mapped_.size(), "missing original tensor backing file");
        reader_.emplace(::open(backing_->path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK));
        struct stat info{};
        require(::fstat(reader_->value, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
                uint64_t(info.st_dev) == backing_->device && uint64_t(info.st_ino) == backing_->inode &&
                checked_add(backing_->offset, backing_->bytes) <= uint64_t(info.st_size),
                "original tensor backing file changed");
        require(::fcntl(reader_->value, F_NOCACHE, 1) == 0,
                "cannot disable original authentication read cache");
#endif
    }

    std::span<const std::byte> read(uint64_t offset, size_t bytes) {
        require(offset <= mapped_.size() && bytes <= mapped_.size() - offset,
                "original authentication read exceeds tensor");
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        require(reader_.has_value(), "missing original authentication reader");
        if (offset < buffered_offset_ || offset - buffered_offset_ > buffer_.size() ||
            bytes > buffer_.size() - (offset - buffered_offset_)) {
            // Coalesce consecutive expert reads instead of issuing one syscall
            // per small expert. A large expert uses its existing proof span as
            // the bound; ordinary tensors use the configured streaming chunk.
            const auto count = std::min(std::max(chunk_bytes_, uint64_t(bytes)), uint64_t(mapped_.size()) - offset);
            buffer_.resize(static_cast<size_t>(count));
            read_into(reader_->value, buffer_, checked_add(backing_->offset, offset));
            buffered_offset_ = offset;
        }
        return std::span<const std::byte>(buffer_).subspan(static_cast<size_t>(offset - buffered_offset_), bytes);
#else
        return mapped_.subspan(static_cast<size_t>(offset), bytes);
#endif
    }

    void evict(uint64_t offset = 0, uint64_t bytes = 0) const noexcept {
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        if (owner_) owner_->evict_tensor_pages(name_, offset, bytes);
#else
        static_cast<void>(offset); static_cast<void>(bytes);
#endif
    }

private:
    std::span<const std::byte> mapped_;
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
    std::string_view name_;
    uint64_t chunk_bytes_ = 0, buffered_offset_ = 0;
    const ModelFiles* owner_ = nullptr;
    std::optional<TensorBackingFile> backing_;
    std::optional<Fd> reader_;
    Bytes buffer_;
#endif
};

bool coloc_eligible(const Tensor& gate, const Tensor& up) {
    if (gate.dimensions.size() != 3 || gate.dimensions != up.dimensions || gate.type != up.type) return false;
    if (gate.type != 12 && gate.type != 13 && gate.type != 8) return false;
    const uint64_t cols = gate.dimensions[0], rows = gate.dimensions[1], experts = gate.dimensions[2];
    if (experts < 2 || experts % 2 != 0 || rows < 4 || rows % 4 != 0) return false;
    if (gate.type == 12 || gate.type == 13) return cols >= 256 && cols % 256 == 0;
    return cols >= 32 && cols % 32 == 0 && ((rows * (cols / 32)) % 8 == 0);
}

bool down_planar_eligible(const Tensor& down) {
    if (down.dimensions.size() != 3 || (down.type != 7 && down.type != 8)) return false;
    const uint64_t cols = down.dimensions[0], rows = down.dimensions[1], experts = down.dimensions[2];
    if (experts < 1 || cols < 32 || cols % 32 != 0 || rows < 4 || rows % 4 != 0) return false;
    if (down.type == 8 && ((rows * (cols / 32)) % 8 != 0)) return false;
    return true;
}

PackTensorV3 make_coloc_entry(const Tensor& t, bool is_gate, std::string partner) {
    const auto spec = make_expert_spec(t.type, static_cast<uint32_t>(t.dimensions[0]),
                                       static_cast<uint32_t>(t.dimensions[1]));
    const uint64_t experts = t.dimensions[2];
    const uint64_t packed_bytes = checked_mul(spec.planar_expert_bytes, experts);
    PlanarKind kind = PlanarKind::plain;
    if (t.type == 12) kind = is_gate ? PlanarKind::q4k_coloc_gate_v3 : PlanarKind::q4k_coloc_up_v3;
    else if (t.type == 13) kind = is_gate ? PlanarKind::q5k_coloc_gate_v3 : PlanarKind::q5k_coloc_up_v3;
    else kind = is_gate ? PlanarKind::q8_coloc_gate_v3 : PlanarKind::q8_coloc_up_v3;

    const bool expanded = packed_bytes != t.bytes;
    const uint32_t dir_type = expanded ? 24u : t.type;
    std::vector<uint64_t> dir_dims = expanded
        ? std::vector<uint64_t>{spec.planar_row_bytes, t.dimensions[1], experts}
        : t.dimensions;
    return PackTensorV3{
        t.name,
        dir_type,
        t.type,
        kind,
        planar_kind_name(kind),
        std::move(dir_dims),
        t.dimensions,
        t.bytes,
        packed_bytes,
        0,
        std::move(partner),
        {},
        {}
    };
}

PackPlanV3 plan_filtered_v3(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar,
    std::string_view prefix_filter,
    std::string recipe_name,
    uint64_t byte_budget
) {
    source.assert_unchanged();
    require(source.architecture() != derived_pack::pack_architecture, "cannot repack an existing derived pack");
    if (mtp_sidecar) mtp_sidecar->assert_unchanged();

    std::vector<std::string> candidate_names;
    std::set<std::string> seen;
    const auto collect = [&](const ModelFiles& files) {
        for (const auto& [name, loc] : files.tensors()) {
            static_cast<void>(loc);
            if (!derived_pack::is_gpu_pack_tensor(name)) continue;
            if (!prefix_filter.empty() && std::string_view(name).substr(0, prefix_filter.size()) != prefix_filter) continue;
            if (seen.insert(name).second) candidate_names.push_back(name);
        }
    };
    collect(source);
    if (mtp_sidecar) collect(*mtp_sidecar);
    require(!candidate_names.empty(), "empty pack v3 tensor selection");

    PackPlanV3 plan;
    plan.recipe = std::move(recipe_name);
    std::set<std::string> emitted;

    for (const auto& name : candidate_names) {
        if (emitted.contains(name)) continue;
        const Tensor* meta = lookup_tensor_meta(source, mtp_sidecar, name);
        require(meta != nullptr, "missing tensor metadata while planning pack v3");

        // Pair ffn_gate_exps.weight immediately with ffn_up_exps.weight for co-located adjacency.
        if (ends_with(name, ".ffn_gate_exps.weight")) {
            const std::string up_name = name.substr(0, name.size() - std::strlen(".ffn_gate_exps.weight")) + ".ffn_up_exps.weight";
            const Tensor* up_meta = lookup_tensor_meta(source, mtp_sidecar, up_name);
            if (up_meta && seen.contains(up_name) && !emitted.contains(up_name) && coloc_eligible(*meta, *up_meta)) {
                auto g_entry = make_coloc_entry(*meta, true, up_name);
                auto u_entry = make_coloc_entry(*up_meta, false, name);
                const uint64_t sbs_each = (meta->dimensions[0] / 256) * meta->dimensions[1] * meta->dimensions[2];
                if (meta->type == 12) plan.q4k_superblocks = checked_add(plan.q4k_superblocks, sbs_each * 2);
                else if (meta->type == 13) plan.q5k_superblocks = checked_add(plan.q5k_superblocks, sbs_each * 2);

                plan.source_payload_bytes = checked_add(plan.source_payload_bytes, checked_add(g_entry.source_bytes, u_entry.source_bytes));
                plan.payload_bytes = checked_add(plan.payload_bytes, checked_add(g_entry.bytes, u_entry.bytes));
                plan.tensors.push_back(std::move(g_entry));
                plan.tensors.push_back(std::move(u_entry));
                emitted.insert(name);
                emitted.insert(up_name);
                continue;
            }
        }
        if (ends_with(name, ".ffn_up_exps.weight")) {
            const std::string gate_name = name.substr(0, name.size() - std::strlen(".ffn_up_exps.weight")) + ".ffn_gate_exps.weight";
            if (seen.contains(gate_name) && !emitted.contains(gate_name)) {
                // Will be emitted together when gate_name is reached; gate_name precedes up_name lexically.
            }
        }

        if (ends_with(name, ".ffn_down_exps.weight") && down_planar_eligible(*meta)) {
            const PlanarKind kind = meta->type == 7 ? PlanarKind::q51_planar_v3 : PlanarKind::q8_planar_v3;
            plan.source_payload_bytes = checked_add(plan.source_payload_bytes, meta->bytes);
            plan.payload_bytes = checked_add(plan.payload_bytes, meta->bytes);
            plan.tensors.push_back(PackTensorV3{
                name, meta->type, meta->type, kind, planar_kind_name(kind),
                meta->dimensions, meta->dimensions, meta->bytes, meta->bytes, 0, {}, {}, {}
            });
            emitted.insert(name);
            continue;
        }

        // Dense Q8_0 matrices retain q8_lanes_v1 where eligible; all other non-MoE tensors stay plain.
        PlanarKind kind = PlanarKind::plain;
        if (meta->type == 8 && lanes::lane_sidecar_eligible(*meta)) {
            kind = PlanarKind::q8_lanes_v1;
        }
        plan.source_payload_bytes = checked_add(plan.source_payload_bytes, meta->bytes);
        plan.payload_bytes = checked_add(plan.payload_bytes, meta->bytes);
        plan.tensors.push_back(PackTensorV3{
            name, meta->type, meta->type, kind, planar_kind_name(kind),
            meta->dimensions, meta->dimensions, meta->bytes, meta->bytes, 0, {}, {}, {}
        });
        emitted.insert(name);
    }

    const uint64_t total_k_sbs = checked_add(plan.q4k_superblocks, plan.q5k_superblocks);
    plan.original_12b_scale_header_bytes = checked_mul(total_k_sbs, 12);
    plan.expanded_16b_scale_header_bytes = checked_mul(total_k_sbs, 16);

    uint64_t relative_offset = 0;
    for (auto& t : plan.tensors) {
        relative_offset = aligned(relative_offset);
        t.offset = relative_offset;
        relative_offset = checked_add(relative_offset, t.bytes);
    }
    const auto hdr = build_header_v3(plan);
    plan.data_offset = hdr.size();
    plan.file_bytes = checked_add(plan.data_offset, relative_offset);
    require(plan.file_bytes <= byte_budget, "planned pack v3 exceeds byte budget");

    source.assert_unchanged();
    if (mtp_sidecar) mtp_sidecar->assert_unchanged();
    return plan;
}

void parse_manifest_v3_into_plan(PackPlanV3& plan, const Gguf& file) {
    const std::map<std::string, uint32_t> keys{
        {"general.architecture", 8},
        {"general.alignment", 4},
        {"tokspur_pack.version", 4},
        {"tokspur_pack.recipe", 8},
        {"tokspur_pack.manifest", 8},
    };
    require(file.metadata().size() == keys.size(), "unexpected metadata entries in pack v3 GGUF");
    for (const auto& [key, type] : keys) {
        const auto* m = file.find(key);
        require(m && m->type == type, "wrong pack v3 GGUF metadata type");
    }
    require(file.string_value("general.architecture") == derived_pack::pack_architecture &&
            file.unsigned_value("general.alignment") == derived_pack::pack_alignment &&
            file.unsigned_value("tokspur_pack.version") == plan.version &&
            file.string_value("tokspur_pack.recipe") == plan.recipe,
            "stale or incompatible pack v3 metadata");

    const auto json = parse_json(file.string_value("tokspur_pack.manifest"), {max_header_bytes, 600'000, 16});
    const auto* tensors_val = json.find("tensors");
    require(tensors_val && tensors_val->kind() == Json::Kind::array, "pack v3 manifest missing tensors array");
    const auto& entries = tensors_val->array();
    require(entries.size() == plan.tensors.size(), "pack v3 manifest tensor count mismatch");

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& v = entries[i];
        auto& t = plan.tensors[i];
        const auto* name_p = v.find("name");
        const auto* type_p = v.find("type");
        const auto* src_type_p = v.find("source_type");
        const auto* layout_p = v.find("layout");
        const auto* src_bytes_p = v.find("source_bytes");
        const auto* bytes_p = v.find("bytes");
        const auto* offset_p = v.find("offset");
        const auto* src_hash_p = v.find("source_sha256");
        const auto* pack_hash_p = v.find("packed_sha256");
        require(name_p && name_p->string() == t.name, "pack v3 manifest tensor name mismatch");
        require(type_p && type_p->unsigned_integer() == t.type, "pack v3 manifest tensor type mismatch");
        require(src_type_p && src_type_p->unsigned_integer() == t.source_type, "pack v3 manifest source_type mismatch");
        require(layout_p && layout_p->string() == t.layout, "pack v3 manifest layout mismatch");
        require(src_bytes_p && src_bytes_p->unsigned_integer() == t.source_bytes, "pack v3 manifest source_bytes mismatch");
        require(bytes_p && bytes_p->unsigned_integer() == t.bytes, "pack v3 manifest bytes mismatch");
        require(offset_p && offset_p->unsigned_integer() == t.offset, "pack v3 manifest offset mismatch");
        require(src_hash_p && pack_hash_p &&
                src_hash_p->string().size() == 64 && pack_hash_p->string().size() == 64,
                "pack v3 manifest missing SHA-256 digests");
        t.source_sha256 = src_hash_p->string();
        t.packed_sha256 = pack_hash_p->string();
    }
}

} // namespace

const char* planar_kind_name(PlanarKind kind) {
    switch (kind) {
        case PlanarKind::plain: return "plain";
        case PlanarKind::q8_lanes_v1: return "q8_lanes_v1";
        case PlanarKind::q4k_coloc_gate_v3: return "q4k_coloc_gate_v3";
        case PlanarKind::q4k_coloc_up_v3: return "q4k_coloc_up_v3";
        case PlanarKind::q5k_coloc_gate_v3: return "q5k_coloc_gate_v3";
        case PlanarKind::q5k_coloc_up_v3: return "q5k_coloc_up_v3";
        case PlanarKind::q8_coloc_gate_v3: return "q8_coloc_gate_v3";
        case PlanarKind::q8_coloc_up_v3: return "q8_coloc_up_v3";
        case PlanarKind::q51_planar_v3: return "q51_planar_v3";
        case PlanarKind::q8_planar_v3: return "q8_planar_v3";
    }
    throw FormatError("unknown PlanarKind");
}

PlanarKind parse_planar_kind(std::string_view name) {
    if (name == "plain") return PlanarKind::plain;
    if (name == "q8_lanes_v1") return PlanarKind::q8_lanes_v1;
    if (name == "q4k_coloc_gate_v3") return PlanarKind::q4k_coloc_gate_v3;
    if (name == "q4k_coloc_up_v3") return PlanarKind::q4k_coloc_up_v3;
    if (name == "q5k_coloc_gate_v3") return PlanarKind::q5k_coloc_gate_v3;
    if (name == "q5k_coloc_up_v3") return PlanarKind::q5k_coloc_up_v3;
    if (name == "q8_coloc_gate_v3") return PlanarKind::q8_coloc_gate_v3;
    if (name == "q8_coloc_up_v3") return PlanarKind::q8_coloc_up_v3;
    if (name == "q51_planar_v3") return PlanarKind::q51_planar_v3;
    if (name == "q8_planar_v3") return PlanarKind::q8_planar_v3;
    throw FormatError("unrecognised planar v3 layout string: " + std::string(name));
}

bool is_planar_v3_kind(PlanarKind kind) noexcept {
    return kind != PlanarKind::plain && kind != PlanarKind::q8_lanes_v1;
}

bool is_coloc_gate_kind(PlanarKind kind) noexcept {
    return kind == PlanarKind::q4k_coloc_gate_v3 ||
           kind == PlanarKind::q5k_coloc_gate_v3 ||
           kind == PlanarKind::q8_coloc_gate_v3;
}

bool is_coloc_up_kind(PlanarKind kind) noexcept {
    return kind == PlanarKind::q4k_coloc_up_v3 ||
           kind == PlanarKind::q5k_coloc_up_v3 ||
           kind == PlanarKind::q8_coloc_up_v3;
}

PlanarExpertSpec make_expert_spec(uint32_t source_type, uint32_t columns, uint32_t rows) {
    require(rows >= 4 && rows % 4 == 0, "planar v3 requires rows >= 4 and divisible by 4");
    PlanarExpertSpec s{};
    s.source_type = source_type;
    s.columns = columns;
    s.rows = rows;
    if (source_type == 12 || source_type == 13) {
        require(columns >= 256 && columns % 256 == 0, "Q4_K/Q5_K planar v3 requires columns divisible by 256");
        s.blocks_per_row = columns / 256;
        const uint64_t total_blocks = checked_mul(rows, s.blocks_per_row);
        if (source_type == 12) {
            s.source_row_bytes = checked_mul(s.blocks_per_row, 144);
            s.planar_row_bytes = checked_mul(s.blocks_per_row, 148);
            s.qs_offset = 0;
            s.sm_offset = checked_mul(total_blocks, 128);
            s.dm_offset = checked_mul(total_blocks, 144);
        } else {
            s.source_row_bytes = checked_mul(s.blocks_per_row, 176);
            s.planar_row_bytes = checked_mul(s.blocks_per_row, 180);
            s.qs_offset = 0;
            s.qh_offset = checked_mul(total_blocks, 128);
            s.sm_offset = checked_mul(total_blocks, 160);
            s.dm_offset = checked_mul(total_blocks, 176);
        }
    } else if (source_type == 7) {
        require(columns >= 32 && columns % 32 == 0, "Q5_1 planar v3 requires columns divisible by 32");
        s.blocks_per_row = columns / 32;
        const uint64_t total_blocks = checked_mul(rows, s.blocks_per_row);
        s.source_row_bytes = checked_mul(s.blocks_per_row, 24);
        s.planar_row_bytes = checked_mul(s.blocks_per_row, 24);
        s.qs_offset = 0;
        s.qh_offset = checked_mul(total_blocks, 16);
        s.dm_offset = checked_mul(total_blocks, 20);
    } else if (source_type == 8) {
        require(columns >= 32 && columns % 32 == 0, "Q8_0 planar v3 requires columns divisible by 32");
        s.blocks_per_row = columns / 32;
        const uint64_t total_blocks = checked_mul(rows, s.blocks_per_row);
        require(total_blocks % 8 == 0, "Q8_0 planar v3 requires total_blocks divisible by 8 for 16-byte expert alignment");
        s.source_row_bytes = checked_mul(s.blocks_per_row, 34);
        s.planar_row_bytes = checked_mul(s.blocks_per_row, 34);
        s.qs_offset = 0;
        s.dm_offset = checked_mul(total_blocks, 32);
    } else {
        throw FormatError("unsupported encoding for planar v3 expert layout");
    }
    s.source_expert_bytes = checked_mul(rows, s.source_row_bytes);
    s.planar_expert_bytes = checked_mul(rows, s.planar_row_bytes);
    require(s.planar_expert_bytes % 16 == 0, "planar v3 expert byte stride must be 16-byte aligned");
    return s;
}

void pack_sm16(std::span<const std::byte, 12> g_scales, std::span<std::byte, 16> out_sm) noexcept {
    const auto* s12 = reinterpret_cast<const uint8_t*>(g_scales.data());
    auto* sm = reinterpret_cast<uint8_t*>(out_sm.data());
    for (size_t slot = 0; slot < 4; ++slot) {
        sm[slot] = static_cast<uint8_t>(s12[slot] & 0x3fu);
        sm[4 + slot] = static_cast<uint8_t>((s12[8 + slot] & 0x0fu) | ((s12[slot] >> 6) << 4));
        sm[8 + slot] = static_cast<uint8_t>(s12[4 + slot] & 0x3fu);
        sm[12 + slot] = static_cast<uint8_t>((s12[8 + slot] >> 4) | ((s12[4 + slot] >> 6) << 4));
    }
}

void unpack_sm16(std::span<const std::byte, 16> in_sm, std::span<std::byte, 12> out_scales) {
    const auto* sm = reinterpret_cast<const uint8_t*>(in_sm.data());
    for (size_t i = 0; i < 16; ++i) {
        require((sm[i] & 0xc0u) == 0u, "corrupt planar v3 scale/min factor exceeds 6-bit [0, 63] range");
    }
    auto* s12 = reinterpret_cast<uint8_t*>(out_scales.data());
    for (size_t slot = 0; slot < 4; ++slot) {
        s12[slot] = static_cast<uint8_t>((sm[slot] & 0x3fu) | ((sm[4 + slot] >> 4) << 6));
        s12[4 + slot] = static_cast<uint8_t>((sm[8 + slot] & 0x3fu) | ((sm[12 + slot] >> 4) << 6));
        s12[8 + slot] = static_cast<uint8_t>((sm[4 + slot] & 0x0fu) | ((sm[12 + slot] & 0x0fu) << 4));
    }
}

void pack_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> gguf_expert, std::span<std::byte> planar_expert) {
    require(gguf_expert.size() == spec.source_expert_bytes && planar_expert.size() == spec.planar_expert_bytes,
            "pack_expert_planar_v3 span size mismatch");
    const uint64_t total_blocks = static_cast<uint64_t>(spec.rows) * spec.blocks_per_row;
    const std::byte* src = gguf_expert.data();
    std::byte* dst = planar_expert.data();

    if (spec.source_type == 12) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const std::byte* sb = src + blk * 144;
            std::memcpy(dst + spec.qs_offset + blk * 128, sb + 16, 128);
            pack_sm16(std::span<const std::byte, 12>(sb + 4, 12),
                      std::span<std::byte, 16>(dst + spec.sm_offset + blk * 16, 16));
            std::memcpy(dst + spec.dm_offset + blk * 4, sb, 4);
        }
    } else if (spec.source_type == 13) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const std::byte* sb = src + blk * 176;
            std::memcpy(dst + spec.qs_offset + blk * 128, sb + 48, 128);
            std::memcpy(dst + spec.qh_offset + blk * 32, sb + 16, 32);
            pack_sm16(std::span<const std::byte, 12>(sb + 4, 12),
                      std::span<std::byte, 16>(dst + spec.sm_offset + blk * 16, 16));
            std::memcpy(dst + spec.dm_offset + blk * 4, sb, 4);
        }
    } else if (spec.source_type == 7) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const std::byte* sb = src + blk * 24;
            std::memcpy(dst + spec.qs_offset + blk * 16, sb + 8, 16);
            std::memcpy(dst + spec.qh_offset + blk * 4, sb + 4, 4);
            std::memcpy(dst + spec.dm_offset + blk * 4, sb, 4);
        }
    } else if (spec.source_type == 8) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const std::byte* sb = src + blk * 34;
            std::memcpy(dst + spec.qs_offset + blk * 32, sb + 2, 32);
            std::memcpy(dst + spec.dm_offset + blk * 2, sb, 2);
        }
    } else {
        throw FormatError("unsupported encoding in pack_expert_planar_v3");
    }
}

void unpack_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<std::byte> gguf_expert) {
    require(planar_expert.size() == spec.planar_expert_bytes && gguf_expert.size() == spec.source_expert_bytes,
            "unpack_expert_planar_v3 span size mismatch");
    const uint64_t total_blocks = static_cast<uint64_t>(spec.rows) * spec.blocks_per_row;
    const std::byte* src = planar_expert.data();
    std::byte* dst = gguf_expert.data();

    if (spec.source_type == 12) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            std::byte* sb = dst + blk * 144;
            std::memcpy(sb, src + spec.dm_offset + blk * 4, 4);
            unpack_sm16(std::span<const std::byte, 16>(src + spec.sm_offset + blk * 16, 16),
                        std::span<std::byte, 12>(sb + 4, 12));
            std::memcpy(sb + 16, src + spec.qs_offset + blk * 128, 128);
        }
    } else if (spec.source_type == 13) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            std::byte* sb = dst + blk * 176;
            std::memcpy(sb, src + spec.dm_offset + blk * 4, 4);
            unpack_sm16(std::span<const std::byte, 16>(src + spec.sm_offset + blk * 16, 16),
                        std::span<std::byte, 12>(sb + 4, 12));
            std::memcpy(sb + 16, src + spec.qh_offset + blk * 32, 32);
            std::memcpy(sb + 48, src + spec.qs_offset + blk * 128, 128);
        }
    } else if (spec.source_type == 7) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            std::byte* sb = dst + blk * 24;
            std::memcpy(sb, src + spec.dm_offset + blk * 4, 4);
            std::memcpy(sb + 4, src + spec.qh_offset + blk * 4, 4);
            std::memcpy(sb + 8, src + spec.qs_offset + blk * 16, 16);
        }
    } else if (spec.source_type == 8) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            std::byte* sb = dst + blk * 34;
            std::memcpy(sb, src + spec.dm_offset + blk * 2, 2);
            std::memcpy(sb + 2, src + spec.qs_offset + blk * 32, 32);
        }
    } else {
        throw FormatError("unsupported encoding in unpack_expert_planar_v3");
    }
}

void dequantize_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<float> output) {
    require(planar_expert.size() == spec.planar_expert_bytes &&
            output.size() == checked_mul(spec.rows, spec.columns),
            "dequantize_expert_planar_v3 span size mismatch");
    const uint64_t total_blocks = static_cast<uint64_t>(spec.rows) * spec.blocks_per_row;
    const std::byte* base = planar_expert.data();

    if (spec.source_type == 12 || spec.source_type == 13) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const std::byte* qs = base + spec.qs_offset + blk * 128;
            const std::byte* qh = spec.source_type == 13 ? (base + spec.qh_offset + blk * 32) : nullptr;
            const auto* sm = reinterpret_cast<const uint8_t*>(base + spec.sm_offset + blk * 16);
            const std::byte* dm = base + spec.dm_offset + blk * 4;
            const double d = binary16(load_u16_le(dm));
            const double dmin = binary16(load_u16_le(dm + 2));
            if (!std::isfinite(d) || !std::isfinite(dmin)) throw FormatError("nonfinite planar v3 Q4_K/Q5_K super-scale");
            float* out_blk = output.data() + blk * 256;
            for (uint32_t grp = 0; grp < 8; ++grp) {
                const uint32_t pair = grp >> 1;
                const uint32_t shift = (grp & 1u) * 4u;
                const double dl = d * sm[grp];
                const double ml = dmin * sm[8 + grp];
                const auto* q_pair = reinterpret_cast<const uint8_t*>(qs + pair * 32);
                for (uint32_t l = 0; l < 32; ++l) {
                    uint32_t q = (q_pair[l] >> shift) & 15u;
                    if (qh) {
                        const uint8_t hb = std::to_integer<uint8_t>(qh[l]);
                        q |= ((static_cast<uint32_t>(hb) >> grp) & 1u) << 4u;
                    }
                    const float val = static_cast<float>(dl * q - ml);
                    if (!std::isfinite(val)) throw FormatError("planar v3 Q4_K/Q5_K weight overflow");
                    out_blk[grp * 32 + l] = val;
                }
            }
        }
    } else if (spec.source_type == 7) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const auto* qs = reinterpret_cast<const uint8_t*>(base + spec.qs_offset + blk * 16);
            const uint32_t qh = load_u32_le(base + spec.qh_offset + blk * 4);
            const std::byte* dm = base + spec.dm_offset + blk * 4;
            const double d = binary16(load_u16_le(dm));
            const double m = binary16(load_u16_le(dm + 2));
            if (!std::isfinite(d) || !std::isfinite(m)) throw FormatError("nonfinite planar v3 Q5_1 scale");
            float* out_blk = output.data() + blk * 32;
            for (uint32_t i = 0; i < 32; ++i) {
                const uint32_t low = (qs[i % 16] >> (4 * (i / 16))) & 15u;
                const uint32_t high = (qh >> i) & 1u;
                const float val = static_cast<float>(d * (low | (high << 4u)) + m);
                if (!std::isfinite(val)) throw FormatError("planar v3 Q5_1 weight overflow");
                out_blk[i] = val;
            }
        }
    } else if (spec.source_type == 8) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const auto* qs = reinterpret_cast<const int8_t*>(base + spec.qs_offset + blk * 32);
            const double d = binary16(load_u16_le(base + spec.dm_offset + blk * 2));
            if (!std::isfinite(d)) throw FormatError("nonfinite planar v3 Q8_0 scale");
            float* out_blk = output.data() + blk * 32;
            for (uint32_t i = 0; i < 32; ++i) {
                const float val = static_cast<float>(d * qs[i]);
                if (!std::isfinite(val)) throw FormatError("planar v3 Q8_0 weight overflow");
                out_blk[i] = val;
            }
        }
    } else {
        throw FormatError("unsupported encoding in dequantize_expert_planar_v3");
    }
}

void dequantize_expert_planar_v3_f32(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<float> output) {
    require(planar_expert.size() == spec.planar_expert_bytes &&
            output.size() == checked_mul(spec.rows, spec.columns),
            "dequantize_expert_planar_v3_f32 span size mismatch");
    const uint64_t total_blocks = static_cast<uint64_t>(spec.rows) * spec.blocks_per_row;
    const std::byte* base = planar_expert.data();

    if (spec.source_type == 12 || spec.source_type == 13) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const std::byte* qs = base + spec.qs_offset + blk * 128;
            const std::byte* qh = spec.source_type == 13 ? (base + spec.qh_offset + blk * 32) : nullptr;
            const auto* sm = reinterpret_cast<const uint8_t*>(base + spec.sm_offset + blk * 16);
            const std::byte* dm = base + spec.dm_offset + blk * 4;
            const float d = binary16(load_u16_le(dm));
            const float dmin = binary16(load_u16_le(dm + 2));
            if (!std::isfinite(d) || !std::isfinite(dmin)) throw FormatError("nonfinite planar v3 Q4_K/Q5_K super-scale");
            float* out_blk = output.data() + blk * 256;
            for (uint32_t grp = 0; grp < 8; ++grp) {
                const uint32_t pair = grp >> 1;
                const uint32_t shift = (grp & 1u) * 4u;
                const float dl = d * static_cast<float>(sm[grp]);
                const float ml = dmin * static_cast<float>(sm[8 + grp]);
                const auto* q_pair = reinterpret_cast<const uint8_t*>(qs + pair * 32);
                for (uint32_t l = 0; l < 32; ++l) {
                    uint32_t q = (q_pair[l] >> shift) & 15u;
                    if (qh) {
                        const uint8_t hb = std::to_integer<uint8_t>(qh[l]);
                        q |= ((static_cast<uint32_t>(hb) >> grp) & 1u) << 4u;
                    }
                    const float val = dl * static_cast<float>(q) - ml;
                    if (!std::isfinite(val)) throw FormatError("planar v3 Q4_K/Q5_K FP32 weight overflow");
                    out_blk[grp * 32 + l] = val;
                }
            }
        }
    } else if (spec.source_type == 7) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const auto* qs = reinterpret_cast<const uint8_t*>(base + spec.qs_offset + blk * 16);
            const uint32_t qh = load_u32_le(base + spec.qh_offset + blk * 4);
            const std::byte* dm = base + spec.dm_offset + blk * 4;
            const float d = binary16(load_u16_le(dm));
            const float m = binary16(load_u16_le(dm + 2));
            if (!std::isfinite(d) || !std::isfinite(m)) throw FormatError("nonfinite planar v3 Q5_1 scale");
            float* out_blk = output.data() + blk * 32;
            for (uint32_t i = 0; i < 32; ++i) {
                const uint32_t low = (qs[i % 16] >> (4 * (i / 16))) & 15u;
                const uint32_t high = (qh >> i) & 1u;
                const float val = d * static_cast<float>(low | (high << 4u)) + m;
                if (!std::isfinite(val)) throw FormatError("planar v3 Q5_1 FP32 weight overflow");
                out_blk[i] = val;
            }
        }
    } else if (spec.source_type == 8) {
        for (uint64_t blk = 0; blk < total_blocks; ++blk) {
            const auto* qs = reinterpret_cast<const int8_t*>(base + spec.qs_offset + blk * 32);
            const float d = binary16(load_u16_le(base + spec.dm_offset + blk * 2));
            if (!std::isfinite(d)) throw FormatError("nonfinite planar v3 Q8_0 scale");
            float* out_blk = output.data() + blk * 32;
            for (uint32_t i = 0; i < 32; ++i) {
                const float val = d * static_cast<float>(qs[i]);
                if (!std::isfinite(val)) throw FormatError("planar v3 Q8_0 FP32 weight overflow");
                out_blk[i] = val;
            }
        }
    } else {
        throw FormatError("unsupported encoding in dequantize_expert_planar_v3_f32");
    }
}

std::vector<std::byte> build_header_v3(const PackPlanV3& plan) {
    Bytes b;
    push_number(b, 0x46554747, 4); // "GGUF"
    push_number(b, 3, 4);          // v3
    push_number(b, plan.tensors.size(), 8);
    push_number(b, 5, 8);          // 5 metadata keys
    push_meta_string(b, "general.architecture", derived_pack::pack_architecture);
    push_meta_uint(b, "general.alignment", plan.alignment);
    push_meta_uint(b, "tokspur_pack.version", plan.version);
    push_meta_string(b, "tokspur_pack.recipe", plan.recipe);
    push_meta_string(b, "tokspur_pack.manifest", build_manifest_v3(plan));

    for (const auto& t : plan.tensors) {
        push_string(b, t.name);
        push_number(b, t.dimensions.size(), 4);
        for (uint64_t d : t.dimensions) push_number(b, d, 8);
        push_number(b, t.type, 4);
        push_number(b, t.offset, 8);
    }
    require(b.size() <= max_header_bytes, "pack v3 header exceeds bound");
    b.resize(aligned(b.size(), plan.alignment), std::byte(0));
    return b;
}

PackPlanV3 plan_derived_pack_v3(const ModelFiles& source, const ModelFiles* mtp_sidecar, uint64_t byte_budget) {
    return plan_filtered_v3(source, mtp_sidecar, {}, pack_recipe_v3, byte_budget);
}

PackPlanV3 plan_block_pack_v3(const ModelFiles& source, const ModelFiles* mtp_sidecar, uint32_t block_id, uint64_t byte_budget) {
    const std::string prefix = "blk." + std::to_string(block_id) + ".";
    const std::string recipe = std::string(block_recipe_prefix_v3) + std::to_string(block_id);
    return plan_filtered_v3(source, mtp_sidecar, prefix, recipe, byte_budget);
}

BudgetReportV3 measure_pack_v3_budget(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar,
    const std::filesystem::path& output_directory,
    const std::filesystem::path& existing_pack_v2_path,
    uint32_t experts_per_chunk,
    uint64_t reserve_bytes
) {
    return measure_pack_v3_budget(plan_derived_pack_v3(source, mtp_sidecar), output_directory,
                                  existing_pack_v2_path, experts_per_chunk, reserve_bytes,
                                  derived_pack::default_chunk_bytes);
}

BudgetReportV3 measure_pack_v3_budget(
    const PackPlanV3& plan, const std::filesystem::path& output_directory,
    const std::filesystem::path& existing_pack_v2_path, uint32_t experts_per_chunk,
    uint64_t reserve_bytes, uint64_t chunk_bytes
) {
    require(experts_per_chunk >= 1 && experts_per_chunk <= 64, "experts_per_chunk must be in 1..64");
    require(chunk_bytes >= 32'768 && chunk_bytes <= 67'108'864, "invalid chunk_bytes bound");
    Fd dir_fd = open_dir_traverse(output_directory);
    const uint64_t avail = query_dir_avail_bytes(dir_fd.value);

    uint64_t pack_v2_bytes = 0;
    if (!existing_pack_v2_path.empty()) {
        struct stat st{};
        if (::stat(existing_pack_v2_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            pack_v2_bytes = static_cast<uint64_t>(st.st_size);
        }
    }

    BudgetReportV3 r{};
    r.filesystem_available_bytes = avail;
    r.existing_pack_v2_file_bytes = pack_v2_bytes;
    r.planned_pack_v3_payload_bytes = plan.payload_bytes;
    r.planned_pack_v3_file_bytes = plan.file_bytes;
    r.required_reserve_bytes = reserve_bytes;
    r.original_12b_scale_header_bytes = plan.original_12b_scale_header_bytes;
    r.expanded_16b_scale_header_bytes = plan.expanded_16b_scale_header_bytes;

    uint64_t max_expert_pair_bytes = chunk_bytes * 2;
    for (const auto& t : plan.tensors) {
        if (t.kind == PlanarKind::q4k_coloc_gate_v3 || t.kind == PlanarKind::q4k_coloc_up_v3) {
            r.q4k_planar_bytes = checked_add(r.q4k_planar_bytes, t.bytes);
        } else if (t.kind == PlanarKind::q5k_coloc_gate_v3 || t.kind == PlanarKind::q5k_coloc_up_v3) {
            r.q5k_planar_bytes = checked_add(r.q5k_planar_bytes, t.bytes);
        } else if (t.kind == PlanarKind::q51_planar_v3) {
            r.q51_planar_bytes = checked_add(r.q51_planar_bytes, t.bytes);
        } else if (t.kind == PlanarKind::q8_planar_v3 || t.kind == PlanarKind::q8_coloc_gate_v3 || t.kind == PlanarKind::q8_coloc_up_v3) {
            r.q8_planar_moe_bytes = checked_add(r.q8_planar_moe_bytes, t.bytes);
        }
        if (is_planar_v3_kind(t.kind)) {
            const auto spec = make_expert_spec(t.source_type,
                static_cast<uint32_t>(t.source_dimensions[0]),
                static_cast<uint32_t>(t.source_dimensions[1]));
            // Peak host conversion + inverse-proof buffers for one streaming chunk of experts.
            const uint64_t pair_factor = is_coloc_gate_kind(t.kind) ? 2ULL : 1ULL;
            const uint64_t scratch = checked_mul(
                checked_add(spec.planar_expert_bytes, spec.source_expert_bytes * 2),
                checked_mul(experts_per_chunk, pair_factor));
            max_expert_pair_bytes = std::max(max_expert_pair_bytes, scratch);
        }
    }
    r.streaming_ring_buffer_bytes = max_expert_pair_bytes;
    r.reserve_satisfied = derived_pack::check_disk_reserve(avail, plan.file_bytes, reserve_bytes);
    r.free_after_pack_v3_bytes = avail >= plan.file_bytes ? (avail - plan.file_bytes) : 0;
    r.ram_ceiling_satisfied = r.streaming_ring_buffer_bytes <= r.streaming_ram_ceiling_bytes;
    return r;
}

PackPlanV3 build_pack_v3_from_plan(
    PackPlanV3 plan,
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes,
    void (*check_cancelled)()
) {
    require(bool(source), "missing source model files for pack v3");
    require(experts_per_chunk >= 1 && experts_per_chunk <= 64, "experts_per_chunk must be in 1..64");
    require(chunk_bytes >= 32'768 && chunk_bytes <= 67'108'864, "invalid chunk_bytes bound");

    Location location(output);
    check_absent(location);

    const uint64_t avail = query_dir_avail_bytes(location.directory.value);
    require(derived_pack::check_disk_reserve(avail, plan.file_bytes, reserve_bytes),
            "insufficient disk space: cannot satisfy pack v3 allocation + reserve");

    TemporaryFile temporary(location.directory.value);
    require(::fcntl(temporary.fd, F_NOCACHE, 1) == 0, "cannot enable F_NOCACHE on pack v3 temporary");

    const auto initial_header = build_header_v3(plan);
    write_at(temporary.fd, initial_header, 0);
    uint64_t written_end = initial_header.size();

    const auto pad_to = [&](uint64_t target_file_pos) {
        require(target_file_pos >= written_end, "nonmonotonic pack v3 write cursor");
        const uint64_t pad = target_file_pos - written_end;
        require(pad < derived_pack::pack_alignment, "excessive alignment padding before pack v3 tensor");
        if (pad > 0) {
            const Bytes zeros(static_cast<size_t>(pad), std::byte(0));
            write_at(temporary.fd, zeros, written_end);
            written_end = checked_add(written_end, pad);
        }
    };

    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        source->assert_unchanged();
        if (mtp_sidecar) mtp_sidecar->assert_unchanged();

        auto& t = plan.tensors[i];
        const uint64_t cur_avail = query_dir_avail_bytes(location.directory.value);
        require(derived_pack::check_disk_reserve(cur_avail, plan.file_bytes - written_end, reserve_bytes),
                "disk space dropped below reserve during pack v3 streaming");

        if (is_coloc_gate_kind(t.kind)) {
            require(i + 1 < plan.tensors.size() && is_coloc_up_kind(plan.tensors[i + 1].kind) &&
                    plan.tensors[i + 1].partner_name == t.name && t.partner_name == plan.tensors[i + 1].name,
                    "co-located gate tensor missing adjacent up partner in plan");
            auto& t_up = plan.tensors[i + 1];
            const auto gate_orig = lookup_tensor_bytes(*source, mtp_sidecar.get(), t.name);
            const auto up_orig = lookup_tensor_bytes(*source, mtp_sidecar.get(), t_up.name);
            const auto spec = make_expert_spec(t.source_type,
                static_cast<uint32_t>(t.source_dimensions[0]),
                static_cast<uint32_t>(t.source_dimensions[1]));
            const uint32_t experts = static_cast<uint32_t>(t.source_dimensions[2]);
            const uint32_t half_experts = experts / 2;

            Sha256 gate_src_h, up_src_h, lo_pack_h, hi_pack_h;
            const uint32_t step_cap = std::min(experts_per_chunk, half_experts);
            const uint64_t pair_bytes = checked_mul(spec.planar_expert_bytes, 2);
            Bytes packed_buf(static_cast<size_t>(checked_mul(pair_bytes, step_cap)));
            Bytes readback_buf(packed_buf.size());
            Bytes restored_gate(static_cast<size_t>(spec.source_expert_bytes));
            Bytes restored_up(static_cast<size_t>(spec.source_expert_bytes));

            for (uint32_t half = 0; half < 2; ++half) {
                auto& cur_t = (half == 0) ? t : t_up;
                auto& pack_h = (half == 0) ? lo_pack_h : hi_pack_h;
                const uint64_t tensor_file_start = checked_add(plan.data_offset, cur_t.offset);
                pad_to(tensor_file_start);

                const uint32_t e_begin = half * half_experts;
                const uint32_t e_end = e_begin + half_experts;
                for (uint32_t e = e_begin; e < e_end;) {
                    if (check_cancelled) check_cancelled();
                    const uint32_t count = std::min(step_cap, e_end - e);
                    const size_t chunk_len = static_cast<size_t>(count * pair_bytes);
                    for (uint32_t k = 0; k < count; ++k) {
                        const uint32_t exp_id = e + k;
                        const auto g_in = gate_orig.subspan(static_cast<size_t>(exp_id * spec.source_expert_bytes),
                                                            static_cast<size_t>(spec.source_expert_bytes));
                        const auto u_in = up_orig.subspan(static_cast<size_t>(exp_id * spec.source_expert_bytes),
                                                          static_cast<size_t>(spec.source_expert_bytes));
                        auto g_out = std::span(packed_buf).subspan(static_cast<size_t>((2 * k + 0) * spec.planar_expert_bytes),
                                                                   static_cast<size_t>(spec.planar_expert_bytes));
                        auto u_out = std::span(packed_buf).subspan(static_cast<size_t>((2 * k + 1) * spec.planar_expert_bytes),
                                                                   static_cast<size_t>(spec.planar_expert_bytes));
                        pack_expert_planar_v3(spec, g_in, g_out);
                        pack_expert_planar_v3(spec, u_in, u_out);
                        gate_src_h.add(g_in);
                        up_src_h.add(u_in);
                    }
                    const auto write_span = std::span<const std::byte>(packed_buf.data(), chunk_len);
                    write_at(temporary.fd, write_span, written_end);
                    auto rb_span = std::span<std::byte>(readback_buf.data(), chunk_len);
                    read_into(temporary.fd, rb_span, written_end);
                    for (uint32_t k = 0; k < count; ++k) {
                        const uint32_t exp_id = e + k;
                        const auto g_rb = std::span<const std::byte>(rb_span).subspan(
                            static_cast<size_t>((2 * k + 0) * spec.planar_expert_bytes),
                            static_cast<size_t>(spec.planar_expert_bytes));
                        const auto u_rb = std::span<const std::byte>(rb_span).subspan(
                            static_cast<size_t>((2 * k + 1) * spec.planar_expert_bytes),
                            static_cast<size_t>(spec.planar_expert_bytes));
                        unpack_expert_planar_v3(spec, g_rb, restored_gate);
                        unpack_expert_planar_v3(spec, u_rb, restored_up);
                        const auto g_in = gate_orig.subspan(static_cast<size_t>(exp_id * spec.source_expert_bytes),
                                                            static_cast<size_t>(spec.source_expert_bytes));
                        const auto u_in = up_orig.subspan(static_cast<size_t>(exp_id * spec.source_expert_bytes),
                                                          static_cast<size_t>(spec.source_expert_bytes));
                        require(std::ranges::equal(g_in, restored_gate) && std::ranges::equal(u_in, restored_up),
                                "co-located gate/up planar v3 inverse byte proof failed");
                    }
                    pack_h.add(rb_span);
                    written_end = checked_add(written_end, chunk_len);
                    e += count;
                }
            }
            t.source_sha256 = gate_src_h.finish();
            t_up.source_sha256 = up_src_h.finish();
            t.packed_sha256 = lo_pack_h.finish();
            t_up.packed_sha256 = hi_pack_h.finish();
            ++i; // Consumed both gate and up
            continue;
        }

        const uint64_t tensor_file_start = checked_add(plan.data_offset, t.offset);
        pad_to(tensor_file_start);
        const auto orig = lookup_tensor_bytes(*source, mtp_sidecar.get(), t.name);
        Sha256 src_h, pack_h;

        if (t.kind == PlanarKind::q51_planar_v3 || t.kind == PlanarKind::q8_planar_v3) {
            const auto spec = make_expert_spec(t.source_type,
                static_cast<uint32_t>(t.source_dimensions[0]),
                static_cast<uint32_t>(t.source_dimensions[1]));
            const uint32_t experts = static_cast<uint32_t>(t.source_dimensions[2]);
            const uint32_t step_cap = std::min(experts_per_chunk, experts);
            Bytes packed_buf(static_cast<size_t>(checked_mul(spec.planar_expert_bytes, step_cap)));
            Bytes readback_buf(packed_buf.size());
            Bytes restored_exp(static_cast<size_t>(spec.source_expert_bytes));

            for (uint32_t e = 0; e < experts;) {
                if (check_cancelled) check_cancelled();
                const uint32_t count = std::min(step_cap, experts - e);
                const size_t chunk_len = static_cast<size_t>(count * spec.planar_expert_bytes);
                for (uint32_t k = 0; k < count; ++k) {
                    const uint32_t exp_id = e + k;
                    const auto exp_in = orig.subspan(static_cast<size_t>(exp_id * spec.source_expert_bytes),
                                                     static_cast<size_t>(spec.source_expert_bytes));
                    auto exp_out = std::span(packed_buf).subspan(static_cast<size_t>(k * spec.planar_expert_bytes),
                                                                 static_cast<size_t>(spec.planar_expert_bytes));
                    pack_expert_planar_v3(spec, exp_in, exp_out);
                    src_h.add(exp_in);
                }
                const auto write_span = std::span<const std::byte>(packed_buf.data(), chunk_len);
                write_at(temporary.fd, write_span, written_end);
                auto rb_span = std::span<std::byte>(readback_buf.data(), chunk_len);
                read_into(temporary.fd, rb_span, written_end);
                for (uint32_t k = 0; k < count; ++k) {
                    const uint32_t exp_id = e + k;
                    const auto exp_rb = std::span<const std::byte>(rb_span).subspan(
                        static_cast<size_t>(k * spec.planar_expert_bytes),
                        static_cast<size_t>(spec.planar_expert_bytes));
                    unpack_expert_planar_v3(spec, exp_rb, restored_exp);
                    const auto exp_in = orig.subspan(static_cast<size_t>(exp_id * spec.source_expert_bytes),
                                                     static_cast<size_t>(spec.source_expert_bytes));
                    require(std::ranges::equal(exp_in, restored_exp), "down planar v3 inverse byte proof failed");
                }
                pack_h.add(rb_span);
                written_end = checked_add(written_end, chunk_len);
                e += count;
            }
        } else if (t.kind == PlanarKind::q8_lanes_v1) {
            const auto layout = lanes::row_layout(t.source_type, static_cast<uint32_t>(t.source_dimensions[0]));
            const uint64_t step = (chunk_bytes / layout.row_bytes) * layout.row_bytes;
            require(step > 0, "chunk_bytes smaller than single Q8 lane row");
            Bytes packed_buf(static_cast<size_t>(step));
            Bytes restored_buf(static_cast<size_t>(step));
            for (uint64_t at = 0; at < t.bytes;) {
                if (check_cancelled) check_cancelled();
                const size_t n = static_cast<size_t>(std::min(step, t.bytes - at));
                const auto in_span = orig.subspan(static_cast<size_t>(at), n);
                auto out_span = std::span(packed_buf).first(n);
                auto inv_span = std::span(restored_buf).first(n);
                lanes::pack_rows(layout, in_span, out_span);
                write_at(temporary.fd, out_span, written_end);
                read_into(temporary.fd, out_span, written_end);
                lanes::unpack_rows(layout, out_span, inv_span);
                require(std::ranges::equal(in_span, inv_span), "q8_lanes_v1 inverse byte proof failed in pack v3");
                src_h.add(in_span);
                pack_h.add(out_span);
                written_end = checked_add(written_end, n);
                at += n;
            }
        } else {
            Bytes restored_buf(static_cast<size_t>(chunk_bytes));
            for (uint64_t at = 0; at < t.bytes;) {
                if (check_cancelled) check_cancelled();
                const size_t n = static_cast<size_t>(std::min(chunk_bytes, t.bytes - at));
                const auto in_span = orig.subspan(static_cast<size_t>(at), n);
                auto rb_span = std::span(restored_buf).first(n);
                write_at(temporary.fd, in_span, written_end);
                read_into(temporary.fd, rb_span, written_end);
                require(std::ranges::equal(in_span, rb_span), "plain tensor readback byte proof failed in pack v3");
                src_h.add(in_span);
                pack_h.add(rb_span);
                written_end = checked_add(written_end, n);
                at += n;
            }
        }
        t.source_sha256 = src_h.finish();
        t.packed_sha256 = pack_h.finish();
    }

    require(written_end == plan.file_bytes, "pack v3 final byte size mismatch");
    const auto final_header = build_header_v3(plan);
    require(final_header.size() == initial_header.size(), "pack v3 manifest changed header size");
    write_at(temporary.fd, final_header, 0);
    require(read_at(temporary.fd, final_header.size(), 0) == final_header, "pack v3 header readback mismatch");
    require(::fchmod(temporary.fd, 0444) == 0 && ::fsync(temporary.fd) == 0, "pack v3 chmod/fsync failed");

    source->assert_unchanged();
    if (mtp_sidecar) mtp_sidecar->assert_unchanged();
    if (check_cancelled) check_cancelled();
    check_absent(location);
    require(::renameatx_np(location.directory.value, temporary.name.c_str(),
                           location.directory.value, location.name.c_str(), RENAME_EXCL) == 0,
            "exclusive pack v3 publication refused");
    temporary.published = true;
    require(::fsync(location.directory.value) == 0, "directory fsync failed after pack v3 publication");
    return plan;
}

PackPlanV3 build_derived_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes
) {
    require(bool(source), "missing source model files");
    auto plan = plan_derived_pack_v3(*source, mtp_sidecar.get());
    return build_pack_v3_from_plan(std::move(plan), std::move(source), std::move(mtp_sidecar),
                                   output, experts_per_chunk, reserve_bytes, chunk_bytes);
}

PackPlanV3 build_block_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    uint32_t block_id,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes
) {
    require(bool(source), "missing source model files");
    auto plan = plan_block_pack_v3(*source, mtp_sidecar.get(), block_id);
    return build_pack_v3_from_plan(std::move(plan), std::move(source), std::move(mtp_sidecar),
                                   output, experts_per_chunk, reserve_bytes, chunk_bytes);
}

AdoptedPackV3 adopt_pack_v3_with_plan(
    PackPlanV3 plan,
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk,
    uint64_t chunk_bytes,
    uint32_t authentication_workers,
    derived_pack::TrustMode trust_mode,
    bool require_full_census,
    void (*check_cancelled)()
) {
    require(bool(source), "missing source model files");
    require(authentication_workers >= 1 && authentication_workers <= derived_pack::max_authentication_workers,
            "authentication_workers must be in 1..16");
    require(experts_per_chunk >= 1 && experts_per_chunk <= 64, "experts_per_chunk must be in 1..64");
    require(chunk_bytes >= 32'768 && chunk_bytes <= 67'108'864, "invalid chunk_bytes bound");

    Location location(path);
    Fd fd(::openat(location.directory.value, location.name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));

    struct stat before{};
    require(::fstat(fd.value, &before) == 0 && S_ISREG(before.st_mode) && !(before.st_mode & 0222) &&
            before.st_nlink == 1 && before.st_size > 0,
            "pack v3 must be an immutable regular file");

    // Metal can materialise private anonymous backing for file mappings made
    // with MAP_PRIVATE. Share only the immutable adopted pack;
    // original model mappings retain their existing private, read-only mode.
    constexpr auto mapping =
#if defined(TOKSPUR_PRIVATE_PACK_SHARED_MAPPING) && TOKSPUR_PRIVATE_PACK_SHARED_MAPPING
        ReadOnlyMapMode::shared_file;
#else
        ReadOnlyMapMode::private_copy_on_write;
#endif
    auto model = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{"/dev/fd/" + std::to_string(fd.value)}, mapping);
    require(model->architecture() == derived_pack::pack_architecture && model->shards().size() == 1,
            "adopted file is not a valid tokspur_pack GGUF");

    const auto& file = model->shards()[0];
    if (!mtp_sidecar && model->tensors().size() == derived_pack::expected_gpu_tensors &&
        plan.tensors.size() < derived_pack::expected_gpu_tensors) {
        const auto mjson = parse_json(file.string_value("tokspur_pack.manifest"), {max_header_bytes, 600'000, 16});
        const auto* mtensors = mjson.find("tensors");
        require(mtensors && mtensors->kind() == Json::Kind::array &&
                mtensors->array().size() == derived_pack::expected_gpu_tensors,
                "pack v3 manifest tensor count mismatch");
        std::set<std::string> seen;
        for (const auto& t : plan.tensors) seen.insert(t.name);
        const auto parse_dims = [](const Json* arr_p) {
            require(arr_p && arr_p->kind() == Json::Kind::array && !arr_p->array().empty(),
                    "pack v3 manifest tensor dimensions missing");
            std::vector<uint64_t> out;
            out.reserve(arr_p->array().size());
            for (const auto& d : arr_p->array()) out.push_back(d.unsigned_integer());
            return out;
        };
        for (size_t i = plan.tensors.size(); i < derived_pack::expected_gpu_tensors; ++i) {
            const auto& entry = mtensors->array()[i];
            const auto* name_p = entry.find("name");
            const std::string tname(name_p ? name_p->string() : "");
            require(tname.starts_with("blk.48.") && derived_pack::is_gpu_pack_tensor(tname) &&
                    seen.insert(tname).second && model->tensors().contains(tname),
                    "unexpected trailing pack v3 tensor without MTP sidecar");
            const auto* type_p = entry.find("type");
            const auto* src_type_p = entry.find("source_type");
            const auto* layout_p = entry.find("layout");
            const auto* src_bytes_p = entry.find("source_bytes");
            const auto* bytes_p = entry.find("bytes");
            const auto* partner_p = entry.find("partner_name");
            require(type_p && src_type_p && layout_p && src_bytes_p && bytes_p,
                    "incomplete trailing pack v3 tensor manifest entry");
            const uint32_t dir_type = static_cast<uint32_t>(type_p->unsigned_integer());
            const uint32_t src_type = static_cast<uint32_t>(src_type_p->unsigned_integer());
            const std::string layout_str(layout_p->string());
            const PlanarKind kind = parse_planar_kind(layout_str);
            auto dir_dims = parse_dims(entry.find("dimensions"));
            auto src_dims = parse_dims(entry.find("source_dimensions"));
            const uint64_t src_bytes = src_bytes_p->unsigned_integer();
            const uint64_t pk_bytes = bytes_p->unsigned_integer();
            const std::string partner_str(partner_p ? partner_p->string() : "");
            const auto& meta = model->tensor(tname);
            require(meta.type == dir_type && meta.dimensions == dir_dims && meta.bytes == pk_bytes,
                    "trailing pack v3 tensor directory mismatch");
            plan.source_payload_bytes = checked_add(plan.source_payload_bytes, src_bytes);
            plan.payload_bytes = checked_add(plan.payload_bytes, pk_bytes);
            plan.tensors.push_back(PackTensorV3{
                tname, dir_type, src_type, kind, layout_str,
                std::move(dir_dims), std::move(src_dims),
                src_bytes, pk_bytes, 0, partner_str, {}, {}
            });
        }
        uint64_t relative_offset = 0;
        for (auto& t : plan.tensors) {
            relative_offset = aligned(relative_offset);
            t.offset = relative_offset;
            relative_offset = checked_add(relative_offset, t.bytes);
        }
        const auto hdr = build_header_v3(plan);
        plan.data_offset = hdr.size();
        plan.file_bytes = checked_add(plan.data_offset, relative_offset);
    }
    require(static_cast<uint64_t>(before.st_size) == plan.file_bytes,
            "pack v3 must be an immutable regular file of exact planned size");

    parse_manifest_v3_into_plan(plan, file);

    const auto canonical_header = build_header_v3(plan);
    require(file.data_offset() == canonical_header.size() &&
            read_at(fd.value, canonical_header.size(), 0) == canonical_header,
            "noncanonical or corrupt pack v3 header");

    if (require_full_census) require_full_pack_v3_census(plan);

    bool seal_valid = false;
    if (trust_mode == derived_pack::TrustMode::local_seal && require_full_census) {
        const std::string seal_name = std::filesystem::path(location.name).stem().string() + ".seal";
        const int seal_fd = ::openat(location.directory.value, seal_name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (seal_fd < 0) {
            if (errno == ELOOP) throw FormatError("symlink seal path rejected via O_NOFOLLOW: ELOOP");
        } else {
            Fd seal_guard(seal_fd);
            struct stat sstat{};
            require(::fstat(seal_fd, &sstat) == 0 && S_ISREG(sstat.st_mode) && !(sstat.st_mode & 0222) && sstat.st_nlink == 1,
                    "invalid or writable pack v3 seal file");
            if (sstat.st_size > 0 && sstat.st_size <= 4096) {
                std::string seal_content(static_cast<size_t>(sstat.st_size), '\0');
                read_into(seal_fd, std::as_writable_bytes(std::span(seal_content)), 0);
                while (!seal_content.empty() && (seal_content.back() == '\n' || seal_content.back() == '\r' || seal_content.back() == ' ')) {
                    seal_content.pop_back();
                }
                if (seal_content == derived_pack::hash(file.string_value("tokspur_pack.manifest"))) {
                    seal_valid = true;
                }
            }
        }
    }

    // Build authentication tasks (co-located gate+up pairs form a single atomic verification unit).
    std::vector<size_t> units;
    units.reserve(plan.tensors.size());
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        if (is_coloc_up_kind(plan.tensors[i].kind)) continue;
        units.push_back(i);
    }

    const auto check_padding_after = [&](size_t idx) {
        if (idx + 1 < plan.tensors.size()) {
            const uint64_t end = checked_add(plan.tensors[idx].offset, plan.tensors[idx].bytes);
            const uint64_t next_start = plan.tensors[idx + 1].offset;
            if (next_start > end) {
                const auto pad = read_at(fd.value, static_cast<size_t>(next_start - end),
                                         checked_add(canonical_header.size(), end));
                require(std::all_of(pad.begin(), pad.end(), [](std::byte b) { return b == std::byte(0); }),
                        "nonzero padding between pack v3 tensors");
            }
        }
    };

    std::atomic<bool> source_read{false};
    const auto verify_unit = [&](size_t unit_idx) {
        if (check_cancelled) check_cancelled();
        const size_t i = units[unit_idx];
        const auto& t = plan.tensors[i];
        const auto& packed_meta = model->tensor(t.name);
        require(packed_meta.type == t.type && packed_meta.dimensions == t.dimensions &&
                packed_meta.bytes == t.bytes &&
                packed_meta.offset == checked_add(canonical_header.size(), t.offset),
                "pack v3 tensor directory metadata mismatch");

        if (is_coloc_gate_kind(t.kind)) {
            require(i + 1 < plan.tensors.size() && is_coloc_up_kind(plan.tensors[i + 1].kind),
                    "missing co-located up tensor during pack v3 adoption");
            const auto& t_up = plan.tensors[i + 1];
            const auto& up_meta = model->tensor(t_up.name);
            require(up_meta.type == t_up.type && up_meta.dimensions == t_up.dimensions &&
                    up_meta.bytes == t_up.bytes &&
                    up_meta.offset == checked_add(canonical_header.size(), t_up.offset),
                    "pack v3 co-located up tensor metadata mismatch");

            const bool pack_only_trailing_mtp = !mtp_sidecar && t.name.starts_with("blk.48.") && t_up.name.starts_with("blk.48.");
            AuthenticationSource gate_orig(*source, mtp_sidecar.get(), t.name, !pack_only_trailing_mtp, chunk_bytes);
            AuthenticationSource up_orig(*source, mtp_sidecar.get(), t_up.name, !pack_only_trailing_mtp, chunk_bytes);
            const auto lo_packed = model->tensor_bytes(t.name);
            const auto hi_packed = model->tensor_bytes(t_up.name);

            // Manifest geometry is untrusted until it accounts for every packed byte, so check it before any expert is read.
            require(t.source_dimensions.size() == 3 && t.source_dimensions[2] != 0 &&
                    t.source_dimensions[0] <= UINT32_MAX && t.source_dimensions[1] <= UINT32_MAX && t.source_dimensions[2] <= UINT32_MAX,
                    "pack v3 source dimensions mismatch");
            const auto spec = make_expert_spec(t.source_type,
                static_cast<uint32_t>(t.source_dimensions[0]),
                static_cast<uint32_t>(t.source_dimensions[1]));
            const uint32_t experts = static_cast<uint32_t>(t.source_dimensions[2]);
            const uint32_t half_experts = experts / 2;
            require(checked_mul(uint64_t{2} * half_experts, spec.planar_expert_bytes) == lo_packed.size() &&
                    checked_mul(uint64_t{2} * (experts - half_experts), spec.planar_expert_bytes) == hi_packed.size(),
                    "pack v3 source dimensions mismatch");

            Sha256 gate_src_h, up_src_h, lo_pack_h, hi_pack_h;
            Bytes restored_gate(static_cast<size_t>(spec.source_expert_bytes));
            Bytes restored_up(static_cast<size_t>(spec.source_expert_bytes));

            for (uint32_t e = 0; e < experts; ++e) {
                source->assert_unchanged();
                if (mtp_sidecar) mtp_sidecar->assert_unchanged();
                model->assert_unchanged();

                const auto half_span = (e < half_experts) ? lo_packed : hi_packed;
                const uint32_t local_e = (e < half_experts) ? e : (e - half_experts);
                const auto g_pk = half_span.subspan(static_cast<size_t>((2 * local_e + 0) * spec.planar_expert_bytes),
                                                    static_cast<size_t>(spec.planar_expert_bytes));
                const auto u_pk = half_span.subspan(static_cast<size_t>((2 * local_e + 1) * spec.planar_expert_bytes),
                                                    static_cast<size_t>(spec.planar_expert_bytes));
                unpack_expert_planar_v3(spec, g_pk, restored_gate);
                unpack_expert_planar_v3(spec, u_pk, restored_up);
                if (!pack_only_trailing_mtp) {
                    const auto g_in = gate_orig.read(static_cast<size_t>(e * spec.source_expert_bytes),
                                                        static_cast<size_t>(spec.source_expert_bytes));
                    const auto u_in = up_orig.read(static_cast<size_t>(e * spec.source_expert_bytes),
                                                      static_cast<size_t>(spec.source_expert_bytes));
                    require(std::ranges::equal(g_in, restored_gate) && std::ranges::equal(u_in, restored_up),
                            "pack v3 adoption co-located gate/up inverse byte proof failed");
                    gate_src_h.add(g_in);
                    up_src_h.add(u_in);
                    gate_orig.evict(uint64_t(e) * spec.source_expert_bytes, spec.source_expert_bytes);
                    up_orig.evict(uint64_t(e) * spec.source_expert_bytes, spec.source_expert_bytes);
                }
                if (e < half_experts) {
                    lo_pack_h.add(g_pk);
                    lo_pack_h.add(u_pk);
                } else {
                    hi_pack_h.add(g_pk);
                    hi_pack_h.add(u_pk);
                }
            }
            require((pack_only_trailing_mtp || (gate_src_h.finish() == t.source_sha256 &&
                    up_src_h.finish() == t_up.source_sha256)) &&
                    lo_pack_h.finish() == t.packed_sha256 &&
                    hi_pack_h.finish() == t_up.packed_sha256,
                    "stale source or corrupt co-located tensor digest in pack v3");
            gate_orig.evict();
            up_orig.evict();
            check_padding_after(i);
            check_padding_after(i + 1);
            if (!pack_only_trailing_mtp) source_read.store(true, std::memory_order_relaxed);
            return;
        }

        const bool pack_only_trailing_mtp = !mtp_sidecar && t.name.starts_with("blk.48.");
        AuthenticationSource orig(*source, mtp_sidecar.get(), t.name, !pack_only_trailing_mtp, chunk_bytes);
        const auto packed_bytes = model->tensor_bytes(t.name);
        Sha256 src_h, pack_h;

        if (t.kind == PlanarKind::q51_planar_v3 || t.kind == PlanarKind::q8_planar_v3) {
            // Manifest geometry is untrusted until it accounts for every packed and source byte, so check it before any expert is read.
            require(t.source_dimensions.size() == 3 && t.source_dimensions[2] != 0 &&
                    t.source_dimensions[0] <= UINT32_MAX && t.source_dimensions[1] <= UINT32_MAX && t.source_dimensions[2] <= UINT32_MAX,
                    "pack v3 source dimensions mismatch");
            const auto spec = make_expert_spec(t.source_type,
                static_cast<uint32_t>(t.source_dimensions[0]),
                static_cast<uint32_t>(t.source_dimensions[1]));
            const uint32_t experts = static_cast<uint32_t>(t.source_dimensions[2]);
            require(checked_mul(uint64_t{experts}, spec.planar_expert_bytes) == packed_bytes.size() &&
                    checked_mul(uint64_t{experts}, spec.source_expert_bytes) == t.source_bytes,
                    "pack v3 source dimensions mismatch");
            Bytes restored(static_cast<size_t>(spec.source_expert_bytes));
            for (uint32_t e = 0; e < experts; ++e) {
                source->assert_unchanged();
                if (mtp_sidecar) mtp_sidecar->assert_unchanged();
                model->assert_unchanged();
                const auto exp_pk = packed_bytes.subspan(static_cast<size_t>(e * spec.planar_expert_bytes),
                                                         static_cast<size_t>(spec.planar_expert_bytes));
                unpack_expert_planar_v3(spec, exp_pk, restored);
                if (!pack_only_trailing_mtp) {
                    const auto exp_in = orig.read(static_cast<size_t>(e * spec.source_expert_bytes),
                                                     static_cast<size_t>(spec.source_expert_bytes));
                    require(std::ranges::equal(exp_in, restored), "pack v3 adoption down inverse byte proof failed");
                    src_h.add(exp_in);
                    orig.evict(uint64_t(e) * spec.source_expert_bytes, spec.source_expert_bytes);
                }
                pack_h.add(exp_pk);
            }
        } else if (t.kind == PlanarKind::q8_lanes_v1) {
            const auto layout = lanes::row_layout(t.source_type, static_cast<uint32_t>(t.source_dimensions[0]));
            const uint64_t step = (chunk_bytes / layout.row_bytes) * layout.row_bytes;
            require(step > 0, "chunk_bytes smaller than single Q8 lane row");
            Bytes restored(static_cast<size_t>(step));
            for (uint64_t at = 0; at < t.bytes;) {
                source->assert_unchanged();
                if (mtp_sidecar) mtp_sidecar->assert_unchanged();
                model->assert_unchanged();
                const size_t n = static_cast<size_t>(std::min(step, t.bytes - at));
                const auto pk_chunk = packed_bytes.subspan(static_cast<size_t>(at), n);
                auto inv_chunk = std::span(restored).first(n);
                lanes::unpack_rows(layout, pk_chunk, inv_chunk);
                if (!pack_only_trailing_mtp) {
                    const auto in_chunk = orig.read(static_cast<size_t>(at), n);
                    require(std::ranges::equal(in_chunk, inv_chunk), "pack v3 adoption q8_lanes_v1 inverse proof failed");
                    src_h.add(in_chunk);
                    orig.evict(at, n);
                }
                pack_h.add(pk_chunk);
                at += n;
            }
        } else {
            for (uint64_t at = 0; at < t.bytes;) {
                source->assert_unchanged();
                if (mtp_sidecar) mtp_sidecar->assert_unchanged();
                model->assert_unchanged();
                const size_t n = static_cast<size_t>(std::min(chunk_bytes, t.bytes - at));
                const auto pk_chunk = packed_bytes.subspan(static_cast<size_t>(at), n);
                if (!pack_only_trailing_mtp) {
                    const auto in_chunk = orig.read(static_cast<size_t>(at), n);
                    require(std::ranges::equal(in_chunk, pk_chunk), "pack v3 adoption plain tensor byte proof failed");
                    src_h.add(in_chunk);
                    orig.evict(at, n);
                }
                pack_h.add(pk_chunk);
                at += n;
            }
        }

        require((pack_only_trailing_mtp || src_h.finish() == t.source_sha256) && pack_h.finish() == t.packed_sha256,
                "stale source or corrupt tensor digest in pack v3");
        orig.evict();
        check_padding_after(i);
        if (!pack_only_trailing_mtp) source_read.store(true, std::memory_order_relaxed);
    };

    uint32_t effective_workers = 1;
    if (!seal_valid) {
        effective_workers = std::min(authentication_workers, static_cast<uint32_t>(units.size()));
        if (effective_workers <= 1) {
            for (size_t u = 0; u < units.size(); ++u) verify_unit(u);
        } else {
            std::atomic<size_t> next{0};
            std::vector<std::exception_ptr> failures(units.size());
            const auto work = [&] {
                for (;;) {
                    const size_t u = next.fetch_add(1, std::memory_order_relaxed);
                    if (u >= units.size()) return;
                    try { verify_unit(u); } catch (...) { failures[u] = std::current_exception(); }
                }
            };
            std::vector<std::thread> pool;
            pool.reserve(effective_workers - 1);
            std::exception_ptr launch_failure;
            try { for (uint32_t w = 1; w < effective_workers; ++w) pool.emplace_back(work); }
            catch (...) { launch_failure = std::current_exception(); }
            if (!launch_failure) work();
            for (auto& th : pool) th.join();
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
            "pack v3 file identity changed during adoption");

    const std::string manifest_sha256 = derived_pack::hash(file.string_value("tokspur_pack.manifest"));
    const std::string_view source_read_mode = !source_read.load(std::memory_order_relaxed) ? "not_read" :
#if defined(TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM) && TOKSPUR_PRIVATE_PACK_AUTH_RECLAIM
        "bounded_nocache";
#else
        "mapped";
#endif
    return {std::move(model), std::move(plan), manifest_sha256, effective_workers, source_read_mode};
}

void validate_pack_v3_output_name(const std::filesystem::path& output) {
    std::string name = output.filename().string();
    for (char& c : name) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (!name.ends_with(".gguf") || name == ".gguf" ||
        name == "tokspur-pack-v2.asa.gguf" || name == "tokspur-pack-v2.gguf") {
        throw FormatError("expected a .gguf output distinct from the reserved v2 pack paths");
    }
}

void require_full_pack_v3_census(const PackPlanV3& plan) {
    require(plan.recipe == pack_recipe_v3 && plan.tensors.size() == derived_pack::expected_gpu_tensors &&
            plan.payload_bytes == expected_pack_v3_payload_bytes,
            "pack v3 full census verification failed");
}

namespace {

bool entry_exists(int directory, const std::string& name) {
    struct stat st{};
    if (::fstatat(directory, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0) return true;
    require(errno == ENOENT, "cannot inspect pack v3 output");
    return false;
}

void verify_seal(int directory, const std::string& name, const std::string& digest) {
    Fd seal(::openat(directory, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat st{};
    require(::fstat(seal.value, &st) == 0 && S_ISREG(st.st_mode) && !(st.st_mode & 0222) &&
            st.st_nlink == 1 && (st.st_size == 64 || st.st_size == 65), "invalid pack v3 seal");
    std::string content(static_cast<size_t>(st.st_size), '\0');
    read_into(seal.value, std::as_writable_bytes(std::span(content)), 0);
    require(content == digest || content == digest + "\n", "existing pack v3 seal does not match");
}

void verify_published_name(const Location& location, const AdoptedPackV3& verified) {
    verified.model->assert_unchanged();
    const auto* mapping = verified.model->shards().front().backing_mapping();
    struct stat st{};
    require(mapping && ::fstatat(location.directory.value, location.name.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0 &&
            static_cast<uint64_t>(st.st_dev) == mapping->device() &&
            static_cast<uint64_t>(st.st_ino) == mapping->inode() && S_ISREG(st.st_mode) &&
            !(st.st_mode & 0222) && st.st_nlink == 1, "authenticated pack v3 path changed");
}

} // namespace

PublishedPackV3 build_or_verify_pack_v3(
    PackPlanV3 plan, std::shared_ptr<const ModelFiles> source, std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output, uint32_t experts_per_chunk, uint64_t reserve_bytes,
    uint64_t chunk_bytes, void (*check_cancelled)()
) {
    validate_pack_v3_output_name(output);
    require(source && mtp_sidecar, "verified pack v3 publication requires target and MTP sources");
    const bool full_recipe = plan.recipe == pack_recipe_v3;
    if (full_recipe) require_full_pack_v3_census(plan);
    else require(plan.recipe.starts_with(block_recipe_prefix_v3), "invalid pack v3 recipe");

    Location location(output);
    const auto dir = location.directory.value;
    require(::flock(dir, LOCK_EX | LOCK_NB) == 0, "pack v3 output directory is busy");
    const auto seal_name = output.stem().string() + ".seal";
    const bool existed = entry_exists(dir, location.name);
    const bool seal_exists = entry_exists(dir, seal_name);
    require(existed || !seal_exists, "orphan pack v3 seal exists without its pack");
    double build_seconds = 0;
    if (!existed) {
        const auto start = std::chrono::steady_clock::now();
        plan = build_pack_v3_from_plan(std::move(plan), source, mtp_sidecar, output, experts_per_chunk,
                                       reserve_bytes, chunk_bytes, check_cancelled);
        build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    // An interrupted build may already have published the immutable pack. Both paths
    // prove its complete payload against the sources before trusting or creating a seal.
    auto verified = adopt_pack_v3_with_plan(std::move(plan), source, mtp_sidecar, output, experts_per_chunk,
                                            chunk_bytes, 1, derived_pack::TrustMode::full, full_recipe,
                                            check_cancelled);
    verify_published_name(location, verified);
    if (check_cancelled) check_cancelled();
    if (!seal_exists) {
        TemporaryFile temporary(dir);
        const auto content = verified.manifest_sha256 + "\n";
        write_at(temporary.fd, std::as_bytes(std::span(content)), 0);
        require(::fchmod(temporary.fd, 0444) == 0 && ::fsync(temporary.fd) == 0, "pack v3 seal chmod/fsync failed");
        require(::renameatx_np(dir, temporary.name.c_str(), dir, seal_name.c_str(), RENAME_EXCL) == 0,
                "exclusive pack v3 seal publication refused");
        temporary.published = true;
    }
    verify_seal(dir, seal_name, verified.manifest_sha256);
    verify_published_name(location, verified);
    require(::fsync(dir) == 0, "directory fsync after pack v3 seal failed");
    return {std::move(verified), existed, build_seconds};
}

AdoptedPackV3 adopt_derived_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk,
    uint64_t chunk_bytes,
    uint32_t authentication_workers,
    derived_pack::TrustMode trust_mode,
    bool require_full_census
) {
    require(bool(source), "missing source model files");
    auto plan = plan_derived_pack_v3(*source, mtp_sidecar.get());
    return adopt_pack_v3_with_plan(std::move(plan), std::move(source), std::move(mtp_sidecar),
                                   path, experts_per_chunk, chunk_bytes, authentication_workers,
                                   trust_mode, require_full_census);
}

AdoptedPackV3 adopt_block_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    uint32_t block_id,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk,
    uint64_t chunk_bytes,
    uint32_t authentication_workers
) {
    require(bool(source), "missing source model files");
    auto plan = plan_block_pack_v3(*source, mtp_sidecar.get(), block_id);
    return adopt_pack_v3_with_plan(std::move(plan), std::move(source), std::move(mtp_sidecar),
                                   path, experts_per_chunk, chunk_bytes, authentication_workers,
                                   derived_pack::TrustMode::full, false);
}

} // namespace tokspur::planar_v3
