#include "tokspur/lane_sidecar.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/json.hpp"

#ifndef __APPLE__
#error "The private lane sidecar requires Darwin and CommonCrypto"
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
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/stdio.h>
#include <thread>
#include <unistd.h>

namespace tokspur::lanes {
namespace {
using Bytes = std::vector<std::byte>;
constexpr uint64_t alignment = 32, header_limit = 8'388'608;
void require(bool value, const char* why) { if (!value) throw FormatError(why); }
uint64_t aligned(uint64_t n) { return checked_add(n, alignment - 1) / alignment * alignment; }
struct Fd {
    int value;
    explicit Fd(int n) : value(n) { require(n >= 0, "lane sidecar file open refused"); }
    ~Fd() { ::close(value); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
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
        constexpr char hex[] = "0123456789abcdef"; std::string out;
        for (auto b : bytes) { out += hex[b >> 4]; out += hex[b & 15]; }
        return out;
    }
};
std::string hash(std::string_view text) {
    Hash h; h.add(std::as_bytes(std::span(text.data(), text.size()))); return h.finish();
}
void number(Bytes& b, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) b.push_back(std::byte((value >> (8 * i)) & 255));
}
void string(Bytes& b, std::string_view value) {
    number(b, value.size(), 8);
    for (unsigned char c : value) b.push_back(std::byte(c));
}
void metadata_string(Bytes& b, const char* key, std::string_view value) { string(b, key); number(b, 8, 4); string(b, value); }
void metadata_uint(Bytes& b, const char* key, uint32_t value) { string(b, key); number(b, 4, 4); number(b, value, 4); }
std::vector<uint64_t> offsets(const SidecarPlan& plan) {
    std::vector<uint64_t> result; uint64_t end = 0;
    for (const auto& t : plan.tensors) { end = aligned(end); result.push_back(end); end = checked_add(end, t.bytes); }
    return result;
}
std::string manifest(const SidecarPlan& plan, std::string_view architecture) {
    Json::Array tensors;
    for (const auto& t : plan.tensors) {
        Json::Array dimensions; for (auto d : t.dimensions) dimensions.push_back(Json::integer(d));
        tensors.push_back(Json(Json::Object{{"name", Json(t.name)}, {"type", Json::integer(t.type)},
            {"dimensions", Json(std::move(dimensions))}, {"bytes", Json::integer(t.bytes)},
            {"source_sha256", Json(t.source_sha256.empty() ? std::string(64, '0') : t.source_sha256)},
            {"packed_sha256", Json(t.packed_sha256.empty() ? std::string(64, '0') : t.packed_sha256)}}));
    }
    return Json(Json::Object{{"schema", Json::integer(sidecar_version)}, {"recipe", Json(sidecar_recipe)},
        {"source_architecture", Json(std::string(architecture))}, {"tensors", Json(std::move(tensors))}}).dump(header_limit);
}
Bytes header(const SidecarPlan& plan, std::string_view architecture) {
    Bytes b; number(b, 0x46554747, 4); number(b, 3, 4); number(b, plan.tensors.size(), 8); number(b, 5, 8);
    metadata_string(b, "general.architecture", "tokspur_lanes"); metadata_uint(b, "general.alignment", alignment);
    metadata_uint(b, "tokspur_lanes.version", sidecar_version); metadata_string(b, "tokspur_lanes.recipe", sidecar_recipe);
    metadata_string(b, "tokspur_lanes.manifest", manifest(plan, architecture));
    const auto positions = offsets(plan);
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        const auto& t = plan.tensors[i]; string(b, t.name); number(b, t.dimensions.size(), 4);
        for (auto d : t.dimensions) number(b, d, 8);
        number(b, t.type, 4); number(b, positions[i], 8);
    }
    require(b.size() <= header_limit, "lane sidecar header exceeds bound"); b.resize(aligned(b.size())); return b;
}
void budget(uint64_t bytes) { require(bytes && bytes <= sidecar_max_bytes, "lane sidecar budget must be positive and at most 32 GB"); }
void chunk_bound(uint64_t bytes) { require(bytes >= 32'768 && bytes <= 67'108'864, "lane sidecar chunk must be between 32768 and 67108864 bytes"); }
void write_at(int fd, std::span<const std::byte> bytes, uint64_t offset) {
    for (size_t n = 0; n < bytes.size();) {
        const auto got = ::pwrite(fd, bytes.data() + n, bytes.size() - n, off_t(checked_add(offset, n)));
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "lane sidecar write failed"); n += size_t(got);
    }
}
void read_into(int fd, std::span<std::byte> out, uint64_t offset) {
    for (size_t n = 0; n < out.size();) {
        const auto got = ::pread(fd, out.data() + n, out.size() - n, off_t(checked_add(offset, n)));
        if (got < 0 && errno == EINTR) continue;
        require(got > 0, "lane sidecar read failed"); n += size_t(got);
    }
}
Bytes read_at(int fd, size_t bytes, uint64_t offset) {
    Bytes out(bytes); read_into(fd,out,offset); return out;
}
struct Location {
    std::string name; Fd directory;
    explicit Location(const std::filesystem::path& path)
        : name(path.filename().string()), directory(::open((path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path()).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW)) {
        require(!name.empty() && name != "." && name != ".." && name.find('\0') == std::string::npos, "invalid lane sidecar filename");
    }
};
void absent(const Location& location) {
    struct stat s{};
    const int found = ::fstatat(location.directory.value, location.name.c_str(), &s, AT_SYMLINK_NOFOLLOW);
    require(found < 0 && errno == ENOENT, "lane sidecar output already exists or is inaccessible");
}
struct Temporary {
    int directory, fd = -1; std::string name; bool published = false;
    explicit Temporary(int parent) : directory(parent) {
        for (unsigned tries = 0; tries < 16; ++tries) {
            std::array<unsigned char, 16> random{}; arc4random_buf(random.data(), random.size());
            constexpr char hex[] = "0123456789abcdef"; name = ".tokspur-lanes-";
            for (auto x : random) { name += hex[x >> 4]; name += hex[x & 15]; }
            name += ".tmp";
            fd = ::openat(directory, name.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_RDWR | O_CLOEXEC, 0600);
            if (fd >= 0) return;
            require(errno == EEXIST, "cannot create exclusive lane sidecar temporary");
        }
        throw FormatError("cannot reserve lane sidecar temporary");
    }
    ~Temporary() { if (fd >= 0) ::close(fd); if (!published) ::unlinkat(directory, name.c_str(), 0); }
    Temporary(const Temporary&) = delete;
    Temporary& operator=(const Temporary&) = delete;
};
void object_keys(const Json& value, std::initializer_list<const char*> expected) {
    std::set<std::string> keys;
    for (const auto& [key, ignored] : value.object()) { static_cast<void>(ignored); require(keys.insert(key).second, "duplicate lane manifest field"); }
    std::set<std::string> wanted; for (auto key : expected) wanted.insert(key);
    require(keys == wanted, "unknown or missing lane manifest field");
}
const Json& field(const Json& value, const char* name) { const auto* p = value.find(name); require(p, "missing lane manifest field"); return *p; }
std::string digest_field(const Json& value, const char* name) {
    std::string s(field(value, name).string());
    require(s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }), "invalid lane manifest SHA-256"); return s;
}
void read_manifest(SidecarPlan& plan, const Gguf& file, std::string_view source_architecture) {
    const std::map<std::string, uint32_t> keys{{"general.architecture",8}, {"general.alignment",4}, {"tokspur_lanes.version",4}, {"tokspur_lanes.recipe",8}, {"tokspur_lanes.manifest",8}};
    require(file.metadata().size() == keys.size(), "extra or missing lane GGUF metadata");
    for (const auto& [key, type] : keys) { const auto* m = file.find(key); require(m && m->type == type, "wrong lane GGUF metadata type"); }
    require(file.string_value("general.architecture") == "tokspur_lanes" && file.unsigned_value("general.alignment") == alignment &&
        file.unsigned_value("tokspur_lanes.version") == sidecar_version && file.string_value("tokspur_lanes.recipe") == sidecar_recipe, "unknown lane sidecar layout");
    const auto json = parse_json(file.string_value("tokspur_lanes.manifest"), {header_limit, 200'000, 16});
    object_keys(json, {"schema", "recipe", "source_architecture", "tensors"});
    require(field(json,"schema").unsigned_integer() == sidecar_version && field(json,"recipe").string() == sidecar_recipe &&
        field(json,"source_architecture").string() == source_architecture, "stale or unknown lane manifest source/recipe");
    const auto& entries = field(json,"tensors").array(); require(entries.size() == plan.tensors.size(), "extra or missing lane manifest tensor");
    std::set<std::string> names;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& v = entries[i]; auto& t = plan.tensors[i];
        object_keys(v, {"name","type","dimensions","bytes","source_sha256","packed_sha256"});
        const std::string name(field(v,"name").string()); require(names.insert(name).second && name == t.name, "duplicate, extra or unordered lane manifest tensor");
        require(field(v,"type").unsigned_integer() == t.type && field(v,"bytes").unsigned_integer() == t.bytes, "stale lane manifest tensor type/size");
        const auto& dims = field(v,"dimensions").array(); require(dims.size() == t.dimensions.size(), "lane manifest dimension count mismatch");
        for (size_t j = 0; j < dims.size(); ++j) require(dims[j].unsigned_integer() == t.dimensions[j], "stale lane manifest geometry");
        t.source_sha256 = digest_field(v,"source_sha256"); t.packed_sha256 = digest_field(v,"packed_sha256");
    }
}
uint64_t chunk_size(const SidecarTensor& tensor, uint64_t maximum) {
    const auto row = row_layout(tensor.type, uint32_t(tensor.dimensions[0])).row_bytes;
    require(row <= maximum, "lane chunk cannot hold a complete row"); return maximum / row * row;
}
} // namespace

bool lane_sidecar_eligible(const Tensor& t) {
    if (t.name.empty() || t.name.size() > 1024 || t.name.find("token_embd") != std::string::npos ||
        (t.name.find("ple_") != std::string::npos || t.name.find(".ple.") != std::string::npos || t.name.starts_with("ple.")) || t.dimensions.size() < 2 || !t.dimensions[0] || t.dimensions[0] > 65536 ||
        !t.dimensions[1] || t.dimensions[1] > 1'000'000) return false;
    const auto columns = t.dimensions[0];
    const bool q8 = t.type == 8 && t.dimensions.size() == 2 && columns % 32 == 0 && (columns % 256 == 0 || columns < 512);
    constexpr std::string_view prefix = "blk.", suffix = ".ffn_down_exps.weight";
    const bool named = t.name.starts_with(prefix) && t.name.ends_with(suffix) && t.name.size() > prefix.size() + suffix.size() &&
        std::all_of(t.name.begin() + prefix.size(), t.name.end() - suffix.size(), [](char c) { return c >= '0' && c <= '9'; });
    const bool q51 = t.type == 7 && named && t.dimensions.size() == 3 && columns == 640 && t.dimensions[1] <= 65536 &&
        t.dimensions[2] && t.dimensions[2] <= 4096;
    if (!q8 && !q51) return false;
    try {
        uint64_t bytes = row_layout(t.type, uint32_t(columns)).row_bytes;
        for (size_t i = 1; i < t.dimensions.size(); ++i) bytes = checked_mul(bytes, t.dimensions[i]);
        return bytes == t.bytes;
    } catch (const FormatError&) { return false; }
}
SidecarPlan plan_sidecar(const ModelFiles& source, std::span<const std::string> names, uint64_t byte_budget) {
    budget(byte_budget); source.assert_unchanged(); require(source.architecture() != "tokspur_lanes", "cannot repack a lane sidecar");
    std::set<std::string> selected;
    if (names.empty()) {
        for (const auto& [name, location] : source.tensors()) { static_cast<void>(location); if (lane_sidecar_eligible(source.tensor(name))) selected.insert(name); }
    } else for (const auto& name : names) { require(selected.insert(name).second, "duplicate requested lane tensor"); require(lane_sidecar_eligible(source.tensor(name)), "requested tensor is not eligible for lane layout"); }
    require(!selected.empty() && selected.size() <= 8192, "empty or oversized lane tensor selection");
    SidecarPlan result;
    for (const auto& name : selected) {
        const auto& t = source.tensor(name); result.tensors.push_back({name,t.type,t.dimensions,t.bytes,{},{}});
        result.payload_bytes = checked_add(result.payload_bytes, t.bytes);
    }
    const auto positions = offsets(result); const auto prefix = header(result, source.architecture());
    result.file_bytes = checked_add(prefix.size(), checked_add(positions.back(), result.tensors.back().bytes));
    require(result.file_bytes <= byte_budget && result.file_bytes <= sidecar_max_bytes, "lane sidecar exceeds derived disk budget");
    source.assert_unchanged(); return result;
}
SidecarPlan build_sidecar(std::shared_ptr<const ModelFiles> source, const std::filesystem::path& output,
    std::span<const std::string> names, uint64_t byte_budget, uint64_t chunk_bytes) {
    require(bool(source), "missing original lane source"); chunk_bound(chunk_bytes); auto plan = plan_sidecar(*source,names,byte_budget);
    for (const auto& t : plan.tensors) static_cast<void>(chunk_size(t,chunk_bytes));
    Location location(output); absent(location); Temporary temporary(location.directory.value);
    require(::ftruncate(temporary.fd, off_t(plan.file_bytes)) == 0, "cannot reserve bounded lane sidecar extent");
    const auto initial_header = header(plan,source->architecture()); write_at(temporary.fd,initial_header,0); const auto positions = offsets(plan);
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        auto& t = plan.tensors[i]; const auto layout = row_layout(t.type,uint32_t(t.dimensions[0]));
        const auto original = source->tensor_bytes(t.name); const auto size = chunk_size(t,chunk_bytes); Hash original_hash, packed_hash;
        Bytes packed(size), restored(size);
        for (uint64_t at = 0; at < t.bytes;) {
            source->assert_unchanged(); const auto n = std::min(size,t.bytes-at); const auto input = original.subspan(at,n);
            auto out = std::span(packed).first(n), inverse = std::span(restored).first(n);
            pack_rows(layout,input,out);
            const auto position = checked_add(initial_header.size(),checked_add(positions[i],at));
            write_at(temporary.fd,out,position); read_into(temporary.fd,out,position);
            unpack_rows(layout,out,inverse);
            require(std::ranges::equal(input,inverse), "lane packing stored inverse byte proof failed");
            original_hash.add(input); packed_hash.add(out); at += n;
        }
        t.source_sha256 = original_hash.finish(); t.packed_sha256 = packed_hash.finish(); source->assert_unchanged();
    }
    const auto final_header = header(plan,source->architecture()); require(final_header.size() == initial_header.size(), "lane manifest changed header length");
    write_at(temporary.fd,final_header,0);
    require(read_at(temporary.fd,final_header.size(),0) == final_header, "lane sidecar header readback differs");
    require(::fchmod(temporary.fd,0444) == 0 && ::fsync(temporary.fd) == 0, "lane sidecar seal/fsync failed");
    source->assert_unchanged(); absent(location);
    require(::renameatx_np(location.directory.value,temporary.name.c_str(),location.directory.value,location.name.c_str(),RENAME_EXCL) == 0, "exclusive lane sidecar publication refused");
    temporary.published = true; require(::fsync(location.directory.value) == 0, "lane sidecar directory fsync failed after publication"); return plan;
}
AdoptedSidecar adopt_sidecar(std::shared_ptr<const ModelFiles> source, const std::filesystem::path& path,
    std::span<const std::string> names, uint64_t byte_budget, uint64_t chunk_bytes, uint32_t authentication_workers) {
    require(authentication_workers && authentication_workers <= sidecar_max_authentication_workers, "lane authentication workers must be between 1 and 16");
    require(bool(source), "missing original lane source"); chunk_bound(chunk_bytes); auto plan = plan_sidecar(*source,names,byte_budget);
    Location location(path); Fd fd(::openat(location.directory.value,location.name.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK));
    struct stat before{}; require(::fstat(fd.value,&before) == 0 && S_ISREG(before.st_mode) && !(before.st_mode & 0222) && before.st_nlink == 1 &&
        before.st_size > 0 && uint64_t(before.st_size) == plan.file_bytes, "lane sidecar must be an immutable regular file of the exact planned size");
    // Open the already pinned descriptor, not the caller's pathname a second time.
    auto model = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{"/dev/fd/"+std::to_string(fd.value)});
    require(model->architecture() == "tokspur_lanes" && model->shards().size() == 1 && model->tensors().size() == plan.tensors.size(), "unknown lane sidecar tensor set");
    const auto& file = model->shards()[0]; read_manifest(plan,file,source->architecture());
    const auto canonical = header(plan,source->architecture()); require(file.data_offset() == canonical.size() && read_at(fd.value,canonical.size(),0) == canonical, "noncanonical or corrupt lane sidecar header");
    const auto positions = offsets(plan);
    const auto authenticate = [&](size_t i) {
        const auto& t = plan.tensors[i]; const auto& packed_tensor = model->tensor(t.name);
        require(packed_tensor.type == t.type && packed_tensor.dimensions == t.dimensions && packed_tensor.bytes == t.bytes &&
            packed_tensor.offset == checked_add(canonical.size(),positions[i]), "lane sidecar payload geometry/offset mismatch");
        const auto original = source->tensor_bytes(t.name), packed = model->tensor_bytes(t.name); const auto layout = row_layout(t.type,uint32_t(t.dimensions[0]));
        const auto size = chunk_size(t,chunk_bytes); Bytes restored(size); Hash original_hash, packed_hash;
        for (uint64_t at = 0; at < t.bytes;) {
            source->assert_unchanged(); model->assert_unchanged(); const auto n = std::min(size,t.bytes-at); const auto input = original.subspan(at,n), data = packed.subspan(at,n);
            auto inverse = std::span(restored).first(n); unpack_rows(layout,data,inverse); require(std::ranges::equal(input,inverse), "adopted lane inverse differs from original source bytes");
            original_hash.add(input); packed_hash.add(data); at += n;
        }
        require(original_hash.finish() == t.source_sha256 && packed_hash.finish() == t.packed_sha256, "stale source or corrupt lane tensor digest");
        if (i + 1 < plan.tensors.size()) {
            const auto end = checked_add(positions[i],t.bytes), padding = positions[i+1]-end;
            const auto bytes = read_at(fd.value,size_t(padding),checked_add(canonical.size(),end));
            require(std::all_of(bytes.begin(),bytes.end(),[](std::byte b){return b==std::byte(0);}), "nonzero lane sidecar padding");
        }
    };
    const auto workers = std::min(authentication_workers, uint32_t(plan.tensors.size()));
    if (workers == 1) {
        // Preserve the original serial order and failure boundary by default.
        for (size_t i = 0; i < plan.tensors.size(); ++i) authenticate(i);
    } else {
        // Each job owns its SHA contexts and bounded inverse scratch. The maps,
        // mappings, plan and descriptor stay alive and read-only until all joins.
        std::atomic<size_t> next{0};
        std::vector<std::exception_ptr> failures(plan.tensors.size());
        const auto work = [&] {
            for (;;) {
                const auto i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= plan.tensors.size()) return;
                try { authenticate(i); } catch (...) { failures[i] = std::current_exception(); }
            }
        };
        std::vector<std::thread> pool; pool.reserve(workers - 1);
        std::exception_ptr launch_failure;
        try { for (uint32_t i = 1; i < workers; ++i) pool.emplace_back(work); }
        catch (...) { launch_failure = std::current_exception(); }
        if (!launch_failure) work();
        for (auto& thread : pool) thread.join();
        if (launch_failure) std::rethrow_exception(launch_failure);
        // Completion order cannot change which stable-file failure is reported.
        for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
    }
    source->assert_unchanged(); model->assert_unchanged(); struct stat after{};
    require(::fstat(fd.value,&after) == 0 && before.st_dev == after.st_dev && before.st_ino == after.st_ino && before.st_size == after.st_size &&
        before.st_mode == after.st_mode && before.st_nlink == after.st_nlink, "lane sidecar identity changed during adoption");
    auto manifest_sha256 = hash(file.string_value("tokspur_lanes.manifest"));
    return {std::move(model),std::move(plan),std::move(manifest_sha256),workers};
}
} // namespace tokspur::lanes
