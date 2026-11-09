#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace tokspur {
namespace {

// Encoded block sizes, not in-memory C struct sizes. There is no external engine ABI.
constexpr std::array layouts{
    Encoding{0, "F32", 1, 4}, Encoding{1, "F16", 1, 2},
    Encoding{2, "Q4_0", 32, 18}, Encoding{3, "Q4_1", 32, 20},
    Encoding{6, "Q5_0", 32, 22}, Encoding{7, "Q5_1", 32, 24},
    Encoding{8, "Q8_0", 32, 34}, Encoding{10, "Q2_K", 256, 84},
    Encoding{11, "Q3_K", 256, 110}, Encoding{12, "Q4_K", 256, 144},
    Encoding{13, "Q5_K", 256, 176}, Encoding{14, "Q6_K", 256, 210},
    Encoding{20, "IQ4_NL", 32, 18}, Encoding{23, "IQ4_XS", 256, 136},
    Encoding{24, "I8", 1, 1}, Encoding{25, "I16", 1, 2},
    Encoding{26, "I32", 1, 4}, Encoding{27, "I64", 1, 8},
    Encoding{28, "F64", 1, 8}, Encoding{30, "BF16", 1, 2},
};

constexpr uint64_t header_limit = 256 * 1024 * 1024;
constexpr uint64_t record_limit = 100'000;
constexpr uint64_t array_limit = 16 * 1024 * 1024;

bool valid_utf8(std::span<const std::byte> bytes) {
    size_t i = 0;
    auto u = [&](size_t index) { return std::to_integer<uint32_t>(bytes[index]); };
    while (i < bytes.size()) {
        const auto first = u(i++);
        if (first < 0x80) continue;
        const uint32_t count = first >= 0xf0 ? 3 : (first >= 0xe0 ? 2 : 1);
        if (first < 0xc2 || first > 0xf4 || count > bytes.size() - i) return false;
        uint32_t point = first & ((1u << (6 - count)) - 1);
        for (uint32_t j = 0; j < count; ++j) {
            const auto next = u(i++);
            if ((next & 0xc0) != 0x80) return false;
            point = (point << 6) | (next & 0x3f);
        }
        if ((count == 1 && point < 0x80) || (count == 2 && point < 0x800) ||
            (count == 3 && point < 0x10000) || point > 0x10ffff ||
            (point >= 0xd800 && point <= 0xdfff)) return false;
    }
    return true;
}

class Cursor {
public:
    explicit Cursor(std::span<const std::byte> bytes, uint64_t start = 0)
        : bytes_(bytes), position_(start) {}
    uint64_t position() const { return position_; }
    std::span<const std::byte> take(uint64_t n) {
        const auto end = checked_add(position_, n);
        if (end > bytes_.size() || end > header_limit) throw FormatError("truncated or oversized GGUF header");
        auto result = bytes_.subspan(static_cast<size_t>(position_), static_cast<size_t>(n));
        position_ = end;
        return result;
    }
    uint64_t number(size_t width) {
        const auto view = take(width);
        uint64_t result = 0;
        for (size_t i = 0; i < width; ++i) result |= uint64_t(std::to_integer<uint8_t>(view[i])) << (8 * i);
        return result;
    }
    std::string_view string(uint64_t limit = header_limit) {
        const auto length = number(8);
        if (length > limit) throw FormatError("GGUF string exceeds limit");
        const auto view = take(length);
        if (!valid_utf8(view)) throw FormatError("invalid UTF-8 in GGUF string");
        return {reinterpret_cast<const char*>(view.data()), view.size()};
    }
    Metadata value(uint32_t type, unsigned depth = 0) {
        if (depth > 8) throw FormatError("GGUF array nesting exceeds limit");
        Metadata result{type, position_, 0, 0, 0};
        switch (type) {
            case 0: case 1: take(1); break;
            case 2: case 3: take(2); break;
            case 4: case 5: case 6: take(4); break;
            case 10: case 11: case 12: take(8); break;
            case 7:
                if (number(1) > 1) throw FormatError("invalid GGUF boolean");
                break;
            case 8: string(); break;
            case 9: {
                result.element_type = static_cast<uint32_t>(number(4));
                result.elements = number(8);
                if (result.element_type > 12 || result.elements > array_limit) throw FormatError("invalid GGUF array");
                constexpr std::array<uint8_t, 13> widths{1, 1, 2, 2, 4, 4, 4, 0, 0, 0, 8, 8, 8};
                if (widths[result.element_type]) {
                    take(checked_mul(result.elements, widths[result.element_type]));
                } else {
                    for (uint64_t i = 0; i < result.elements; ++i) value(result.element_type, depth + 1);
                }
                break;
            }
            default: throw FormatError("unknown GGUF metadata type");
        }
        result.end = position_;
        return result;
    }
private:
    std::span<const std::byte> bytes_;
    uint64_t position_;
};

bool valid_key(std::string_view key) {
    if (key.empty() || key.front() == '.' || key.back() == '.') return false;
    char previous = '.';
    for (const char c : key) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.')) return false;
        if (c == '.' && previous == '.') return false;
        previous = c;
    }
    return true;
}

auto modification_time(const struct stat& info) {
#ifdef __APPLE__
    return info.st_mtimespec;
#else
    return info.st_mtim;
#endif
}

auto change_time(const struct stat& info) {
#ifdef __APPLE__
    return info.st_ctimespec;
#else
    return info.st_ctim;
#endif
}

} // namespace

uint64_t checked_add(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b) throw FormatError("GGUF offset overflow");
    return a + b;
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) throw FormatError("GGUF shape overflow");
    return a * b;
}

const Encoding& encoding(uint32_t id) {
    for (const auto& item : layouts) if (item.id == id) return item;
    throw FormatError("unsupported GGUF tensor encoding " + std::to_string(id));
}

MappedFile::MappedFile(const std::filesystem::path& path, ReadOnlyMapMode mode) : path_(path), mapping_mode_(mode) {
    if (mode != ReadOnlyMapMode::private_copy_on_write && mode != ReadOnlyMapMode::shared_file)
        throw FormatError("invalid read-only mapping mode");
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd_ < 0) throw FormatError("cannot open weight file");
    struct stat info{};
    if (fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        uint64_t(info.st_size) > std::numeric_limits<size_t>::max()) {
        close(fd_);
        fd_ = -1;
        throw FormatError("weight file must be a nonempty regular file");
    }
    size_ = static_cast<uint64_t>(info.st_size);
    device_ = static_cast<uint64_t>(info.st_dev);
    inode_ = static_cast<uint64_t>(info.st_ino);
    mtime_seconds_ = modification_time(info).tv_sec;
    mtime_nanos_ = modification_time(info).tv_nsec;
    ctime_seconds_ = change_time(info).tv_sec;
    ctime_nanos_ = change_time(info).tv_nsec;
    if (mode == ReadOnlyMapMode::shared_file && (info.st_mode & 0222)) {
        close(fd_); fd_ = -1;
        throw FormatError("shared read-only mapping requires an immutable file");
    }
    auto* mapped = mmap(nullptr, static_cast<size_t>(size_), PROT_READ,
                        mode == ReadOnlyMapMode::shared_file ? MAP_SHARED : MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        throw FormatError("cannot map weight file");
    }
    data_ = static_cast<const std::byte*>(mapped);
}

MappedFile::~MappedFile() {
    if (data_) munmap(const_cast<std::byte*>(data_), static_cast<size_t>(size_));
    if (fd_ >= 0) close(fd_);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)), data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)), device_(other.device_), inode_(other.inode_),
      mtime_seconds_(other.mtime_seconds_), mtime_nanos_(other.mtime_nanos_),
      ctime_seconds_(other.ctime_seconds_), ctime_nanos_(other.ctime_nanos_), mapping_mode_(other.mapping_mode_) {}

std::span<const std::byte> MappedFile::bytes() const noexcept {
    return {data_, static_cast<size_t>(size_)};
}

void MappedFile::evict_clean_pages(uint64_t offset, uint64_t bytes) const noexcept {
    if (!data_ || !bytes || offset >= size_) return;
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    if (!page || offset > std::numeric_limits<uint64_t>::max() - page) return;
    const uint64_t end = bytes > size_ - offset ? size_ : offset + bytes;
    const uint64_t lo = ((offset + page - 1) / page) * page;
    const uint64_t hi = (end / page) * page;
    if (hi > lo) {
        static_cast<void>(posix_madvise(const_cast<std::byte*>(data_ + lo), static_cast<size_t>(hi - lo), POSIX_MADV_DONTNEED));
    }
}

void MappedFile::assert_unchanged() const {
    struct stat info{};
    if (fstat(fd_, &info) != 0 || static_cast<uint64_t>(info.st_size) != size_ ||
        static_cast<uint64_t>(info.st_dev) != device_ || static_cast<uint64_t>(info.st_ino) != inode_ ||
        modification_time(info).tv_sec != mtime_seconds_ || modification_time(info).tv_nsec != mtime_nanos_ ||
        change_time(info).tv_sec != ctime_seconds_ || change_time(info).tv_nsec != ctime_nanos_) {
        throw FormatError("weight file changed since mapping");
    }
}

Gguf::Gguf(const std::filesystem::path& path, ReadOnlyMapMode mode) {
    mapping_.emplace_back(path, mode);
    data_ = mapping_.front().bytes();
    file_size_ = data_.size();
    parse();
    assert_unchanged();
}

Gguf::Gguf(std::span<const std::byte> bytes) : data_(bytes), file_size_(bytes.size()) { parse(); }

Gguf::Gguf(std::span<const std::byte> header, uint64_t declared_file_size)
    : data_(header), file_size_(declared_file_size), header_only_(true) {
    if (header.size() > declared_file_size) throw FormatError("GGUF header exceeds declared file extent");
    parse();
}

const Metadata* Gguf::find(std::string_view key) const {
    const auto found = metadata_.find(key);
    return found == metadata_.end() ? nullptr : &found->second;
}

uint64_t Gguf::unsigned_value(std::string_view key) const {
    const auto* item = find(key);
    if (!item) throw FormatError("missing GGUF metadata " + std::string(key));
    size_t width = 0;
    bool signed_value = false;
    switch (item->type) {
        case 0: width = 1; break;
        case 1: width = 1; signed_value = true; break;
        case 2: width = 2; break;
        case 3: width = 2; signed_value = true; break;
        case 4: width = 4; break;
        case 5: width = 4; signed_value = true; break;
        case 10: width = 8; break;
        case 11: width = 8; signed_value = true; break;
        default: throw FormatError("GGUF metadata is not an integer");
    }
    Cursor reader(data_, item->offset);
    const auto result = reader.number(width);
    if (signed_value && (result >> (width * 8 - 1))) throw FormatError("negative GGUF count");
    return result;
}

bool Gguf::boolean_value(std::string_view key) const {
    const auto* item = find(key);
    if (!item || item->type != 7) throw FormatError("GGUF metadata is not a boolean");
    assert_unchanged();
    Cursor reader(data_, item->offset);
    const auto value = reader.number(1);
    if (value > 1) throw FormatError("invalid GGUF boolean");
    return value != 0;
}

std::span<const std::byte> Gguf::metadata_bytes(std::string_view key) const {
    const auto* item = find(key);
    if (!item) throw FormatError("missing GGUF metadata " + std::string(key));
    assert_unchanged();
    if (item->end < item->offset || item->end > data_.size()) throw FormatError("GGUF metadata exceeds supplied header bytes");
    return data_.subspan(static_cast<size_t>(item->offset), static_cast<size_t>(item->end - item->offset));
}

double Gguf::floating_value(std::string_view key) const {
    const auto* item = find(key);
    if (!item || (item->type != 6 && item->type != 12)) throw FormatError("GGUF metadata is not floating point");
    assert_unchanged();
    Cursor reader(data_, item->offset);
    const double value = item->type == 6 ? double(std::bit_cast<float>(static_cast<uint32_t>(reader.number(4)))) : std::bit_cast<double>(reader.number(8));
    if (!std::isfinite(value)) throw FormatError("nonfinite GGUF metadata");
    assert_unchanged();
    return value;
}

std::vector<uint64_t> Gguf::unsigned_array(std::string_view key) const {
    const auto* item = find(key);
    if (!item || item->type != 9) throw FormatError("GGUF metadata is not an integer array");
    size_t width = 0;
    bool signed_value = false;
    switch (item->element_type) {
        case 0: width = 1; break;
        case 1: width = 1; signed_value = true; break;
        case 2: width = 2; break;
        case 3: width = 2; signed_value = true; break;
        case 4: width = 4; break;
        case 5: width = 4; signed_value = true; break;
        case 10: width = 8; break;
        case 11: width = 8; signed_value = true; break;
        default: throw FormatError("GGUF array has noninteger elements");
    }
    assert_unchanged();
    Cursor reader(data_, item->offset);
    if (reader.number(4) != item->element_type || reader.number(8) != item->elements) throw FormatError("GGUF array changed after validation");
    std::vector<uint64_t> values;
    values.reserve(static_cast<size_t>(item->elements));
    for (uint64_t i = 0; i < item->elements; ++i) {
        const auto value = reader.number(width);
        if (signed_value && (value >> (width * 8 - 1))) throw FormatError("negative GGUF array count");
        values.push_back(value);
    }
    if (reader.position() != item->end) throw FormatError("GGUF integer array span mismatch");
    assert_unchanged();
    return values;
}

std::string_view Gguf::string_value(std::string_view key) const {
    const auto* item = find(key);
    if (!item || item->type != 8) throw FormatError("GGUF metadata is not a string: " + std::string(key));
    Cursor reader(data_, item->offset);
    return reader.string();
}

std::vector<std::string_view> Gguf::string_array(std::string_view key) const {
    const auto* item = find(key);
    if (!item || item->type != 9 || item->element_type != 8) throw FormatError("GGUF metadata is not a string array");
    assert_unchanged();
    Cursor reader(data_, item->offset);
    if (reader.number(4) != 8 || reader.number(8) != item->elements) throw FormatError("GGUF array changed after validation");
    std::vector<std::string_view> result;
    result.reserve(static_cast<size_t>(item->elements));
    for (uint64_t i = 0; i < item->elements; ++i) result.push_back(reader.string());
    if (reader.position() != item->end) throw FormatError("GGUF string array span mismatch");
    assert_unchanged();
    return result;
}

void Gguf::parse() {
    Cursor reader(data_);
    if (reader.number(4) != 0x46554747 || reader.number(4) != 3) throw FormatError("expected little-endian GGUF v3");
    const auto tensor_count = reader.number(8);
    const auto metadata_count = reader.number(8);
    if (tensor_count > record_limit || metadata_count > record_limit) throw FormatError("GGUF record limit exceeded");
    const auto minimum_records = checked_add(checked_mul(tensor_count, 33), checked_mul(metadata_count, 14));
    if (minimum_records > data_.size() - reader.position()) throw FormatError("GGUF counts exceed available file bytes");
    for (uint64_t i = 0; i < metadata_count; ++i) {
        const std::string key(reader.string(65535));
        if (!valid_key(key)) throw FormatError("invalid GGUF metadata key");
        const auto value = reader.value(static_cast<uint32_t>(reader.number(4)));
        if (!metadata_.emplace(key, value).second) throw FormatError("duplicate GGUF metadata key");
    }
    const auto alignment = find("general.alignment") ? unsigned_value("general.alignment") : 32;
    if (!alignment || alignment > 65536 || (alignment & (alignment - 1))) throw FormatError("unsupported GGUF alignment");
    tensors_.reserve(static_cast<size_t>(tensor_count));
    std::set<std::string, std::less<>> names;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        Tensor tensor;
        tensor.name = reader.string(64);
        if (tensor.name.empty() || !names.insert(tensor.name).second) throw FormatError("empty or duplicate tensor name");
        for (unsigned char c : tensor.name) if (c < 0x21 || c > 0x7e) throw FormatError("non-ASCII tensor name");
        const auto dimensions = reader.number(4);
        if (!dimensions || dimensions > 4) throw FormatError("unsupported GGUF tensor rank");
        tensor.elements = 1;
        for (uint64_t j = 0; j < dimensions; ++j) {
            const auto dimension = reader.number(8);
            if (!dimension) throw FormatError("zero tensor dimension");
            tensor.dimensions.push_back(dimension);
            tensor.elements = checked_mul(tensor.elements, dimension);
        }
        tensor.type = static_cast<uint32_t>(reader.number(4));
        const auto& layout = encoding(tensor.type);
        if (tensor.dimensions.front() % layout.elements_per_block) throw FormatError("quantized tensor row is not block-aligned");
        tensor.bytes = checked_mul(tensor.elements / layout.elements_per_block, layout.bytes_per_block);
        tensor.offset = reader.number(8);
        if (tensor.offset % alignment) throw FormatError("unaligned GGUF tensor offset");
        tensors_.push_back(std::move(tensor));
    }
    data_offset_ = checked_add(reader.position(), alignment - 1) & ~(alignment - 1);
    // Metadata-only shards have no tensor data section requiring alignment padding.
    if (!tensor_count) data_offset_ = std::min(data_offset_, uint64_t(data_.size()));
    if (data_offset_ > data_.size()) throw FormatError("truncated GGUF header padding");
    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    intervals.reserve(tensors_.size());
    for (auto& tensor : tensors_) {
        tensor.offset = checked_add(data_offset_, tensor.offset);
        const auto end = checked_add(tensor.offset, tensor.bytes);
        if (end > file_size_) throw FormatError("truncated GGUF tensor data");
        intervals.emplace_back(tensor.offset, end);
    }
    std::sort(intervals.begin(), intervals.end());
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first < intervals[i - 1].second) throw FormatError("overlapping GGUF tensors");
    }
}

std::span<const std::byte> Gguf::tensor_bytes(const Tensor& tensor) const {
    if (header_only_) throw FormatError("header-only GGUF has no tensor payload access");
    // Verify membership, not just caller-supplied offsets into a large mapped file.
    const auto member = std::find_if(tensors_.begin(), tensors_.end(), [&](const Tensor& item) { return &item == &tensor; });
    if (member == tensors_.end()) throw FormatError("tensor does not belong to this GGUF");
    assert_unchanged();
    return data_.subspan(static_cast<size_t>(tensor.offset), static_cast<size_t>(tensor.bytes));
}

void Gguf::evict_tensor_pages(const Tensor& tensor, uint64_t offset, uint64_t bytes) const noexcept {
    if (mapping_.empty() || header_only_ || offset >= tensor.bytes) return;
    const uint64_t span = bytes ? std::min(bytes, tensor.bytes - offset) : (tensor.bytes - offset);
    if (tensor.offset > mapping_.front().size() || offset > mapping_.front().size() - tensor.offset) return;
    mapping_.front().evict_clean_pages(tensor.offset + offset, span);
}

void Gguf::assert_unchanged() const {
    for (const auto& item : mapping_) item.assert_unchanged();
}

ModelFiles::ModelFiles(const std::vector<std::filesystem::path>& paths, ReadOnlyMapMode mode) {
    if (paths.empty() || paths.size() > 1024) throw FormatError("expected 1 to 1024 GGUF shards");
    shards_.reserve(paths.size());
    for (const auto& path : paths) shards_.emplace_back(path, mode);
    std::set<uint64_t> numbers;
    uint64_t declared_total = 0;
    bool split = false;
    for (size_t s = 0; s < shards_.size(); ++s) {
        const auto& shard = shards_[s];
        const bool has_split = shard.find("split.count") || shard.find("split.no") || shard.find("split.tensors.count");
        if (s == 0) split = has_split;
        if (has_split != split || (!split && paths.size() != 1)) throw FormatError("inconsistent split-file metadata");
        if (split) {
            const auto count = shard.unsigned_value("split.count");
            const auto number = shard.unsigned_value("split.no");
            const auto total = shard.unsigned_value("split.tensors.count");
            if (count != paths.size() || number >= count || !numbers.insert(number).second ||
                (s && declared_total != total)) throw FormatError("incomplete or mismatched GGUF shard set");
            declared_total = total;
        }
        if (shard.find("general.architecture")) {
            const auto architecture = shard.string_value("general.architecture");
            if (!architecture_.empty() && architecture_ != architecture) throw FormatError("shard architecture mismatch");
            architecture_ = architecture;
        }
        for (size_t t = 0; t < shard.tensors().size(); ++t) {
            if (!tensors_.emplace(shard.tensors()[t].name, TensorLocation{s, t}).second) throw FormatError("duplicate tensor across GGUF shards");
        }
    }
    if (architecture_.empty()) throw FormatError("GGUF model has no architecture");
    if (split && declared_total != tensors_.size()) throw FormatError("GGUF split tensor count mismatch");
}

const Tensor& ModelFiles::tensor(std::string_view name) const {
    const auto item = tensors_.find(name);
    if (item == tensors_.end()) throw FormatError("unknown tensor");
    return shards_[item->second.shard].tensors()[item->second.index];
}

std::span<const std::byte> ModelFiles::tensor_bytes(std::string_view name) const {
    const auto item = tensors_.find(name);
    if (item == tensors_.end()) throw FormatError("unknown tensor");
    const auto& shard = shards_[item->second.shard];
    return shard.tensor_bytes(shard.tensors()[item->second.index]);
}

std::optional<TensorBackingFile> ModelFiles::tensor_backing_file(std::string_view name) const {
    const auto item = tensors_.find(name);
    if (item == tensors_.end()) return std::nullopt;
    const auto& shard = shards_[item->second.shard];
    const auto* mapping = shard.backing_mapping();
    if (!mapping) return std::nullopt;
    const auto& t = shard.tensors()[item->second.index];
    return TensorBackingFile{mapping->path(), t.offset, t.bytes, mapping->device(), mapping->inode()};
}

void ModelFiles::evict_tensor_pages(std::string_view name, uint64_t offset, uint64_t bytes) const noexcept {
    const auto item = tensors_.find(name);
    if (item == tensors_.end()) return;
    const auto& shard = shards_[item->second.shard];
    shard.evict_tensor_pages(shard.tensors()[item->second.index], offset, bytes);
}

void ModelFiles::assert_unchanged() const {
    for (const auto& shard : shards_) shard.assert_unchanged();
}

} // namespace tokspur
