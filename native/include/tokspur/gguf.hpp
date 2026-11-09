#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur {

class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Dimensions and offsets are in the on-disk order: the first dimension is contiguous.
struct Encoding {
    uint32_t id;
    const char* name;
    uint32_t elements_per_block;
    uint32_t bytes_per_block;
};

const Encoding& encoding(uint32_t id);
uint64_t checked_add(uint64_t a, uint64_t b);
uint64_t checked_mul(uint64_t a, uint64_t b);

enum class ReadOnlyMapMode { private_copy_on_write, shared_file };

// Holds a read-only descriptor and mapping. No pages are prefetched or made resident.
// Adopted weight files must remain immutable for the lifetime of the mapping.
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path,
                        ReadOnlyMapMode mode = ReadOnlyMapMode::private_copy_on_write);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] uint64_t size() const noexcept { return size_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] uint64_t device() const noexcept { return device_; }
    [[nodiscard]] uint64_t inode() const noexcept { return inode_; }
    [[nodiscard]] ReadOnlyMapMode mapping_mode() const noexcept { return mapping_mode_; }
    void evict_clean_pages(uint64_t offset, uint64_t bytes) const noexcept;
    void assert_unchanged() const;

private:
    std::filesystem::path path_;
    int fd_ = -1;
    const std::byte* data_ = nullptr;
    uint64_t size_ = 0;
    uint64_t device_ = 0;
    uint64_t inode_ = 0;
    int64_t mtime_seconds_ = 0;
    int64_t mtime_nanos_ = 0;
    int64_t ctime_seconds_ = 0;
    int64_t ctime_nanos_ = 0;
    ReadOnlyMapMode mapping_mode_ = ReadOnlyMapMode::private_copy_on_write;
};

struct Metadata {
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t end = 0;
    uint32_t element_type = 0;
    uint64_t elements = 0;
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type = 0;
    uint64_t offset = 0; // Absolute file offset, never relative to tensor_data.
    uint64_t bytes = 0;
    uint64_t elements = 0;
};

class Gguf {
public:
    explicit Gguf(const std::filesystem::path& path,
                  ReadOnlyMapMode mode = ReadOnlyMapMode::private_copy_on_write);
    // Borrowed bytes are useful for model-free tests. Caller must keep them alive.
    explicit Gguf(std::span<const std::byte> bytes);
    // Header-only view: the caller supplies the observed file extent, and keeps
    // these bytes alive. Uses the same bounded parser; tensor_bytes always refuses.
    // This validates a catalogue, never the existence/identity of payload bytes.
    Gguf(std::span<const std::byte> header, uint64_t declared_file_size);
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;
    Gguf(Gguf&&) = default;

    [[nodiscard]] const std::vector<Tensor>& tensors() const { return tensors_; }
    [[nodiscard]] const std::map<std::string, Metadata, std::less<>>& metadata() const {
        return metadata_;
    }
    [[nodiscard]] const Metadata* find(std::string_view key) const;
    [[nodiscard]] uint64_t unsigned_value(std::string_view key) const;
    [[nodiscard]] bool boolean_value(std::string_view key) const;
    [[nodiscard]] double floating_value(std::string_view key) const;
    [[nodiscard]] std::vector<uint64_t> unsigned_array(std::string_view key) const;
    [[nodiscard]] std::string_view string_value(std::string_view key) const;
    [[nodiscard]] std::vector<std::string_view> string_array(std::string_view key) const;
    // Exact typed value bytes, for metadata identity checks without copying
    // vocabulary/merge arrays. The key must belong to this parsed container.
    [[nodiscard]] std::span<const std::byte> metadata_bytes(std::string_view key) const;
    [[nodiscard]] std::span<const std::byte> tensor_bytes(const Tensor& tensor) const;
    [[nodiscard]] const MappedFile* backing_mapping() const noexcept {
        return mapping_.empty() ? nullptr : &mapping_.front();
    }
    void evict_tensor_pages(const Tensor& tensor, uint64_t offset = 0, uint64_t bytes = 0) const noexcept;
    [[nodiscard]] uint64_t data_offset() const noexcept { return data_offset_; }
    [[nodiscard]] uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] bool header_only() const noexcept { return header_only_; }
    void assert_unchanged() const;

private:
    void parse();
    std::vector<MappedFile> mapping_; // Zero or one; makes borrowed test input explicit.
    std::span<const std::byte> data_;
    std::map<std::string, Metadata, std::less<>> metadata_;
    std::vector<Tensor> tensors_;
    uint64_t data_offset_ = 0;
    uint64_t file_size_ = 0;
    bool header_only_ = false;
};

struct TensorLocation {
    size_t shard;
    size_t index;
};

struct TensorBackingFile {
    std::filesystem::path path;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t device = 0;
    uint64_t inode = 0;
};

class ModelFiles {
public:
    explicit ModelFiles(const std::vector<std::filesystem::path>& paths,
                        ReadOnlyMapMode mode = ReadOnlyMapMode::private_copy_on_write);
    [[nodiscard]] const std::vector<Gguf>& shards() const { return shards_; }
    [[nodiscard]] const std::map<std::string, TensorLocation, std::less<>>& tensors() const {
        return tensors_;
    }
    [[nodiscard]] std::span<const std::byte> tensor_bytes(std::string_view name) const;
    [[nodiscard]] const Tensor& tensor(std::string_view name) const;
    [[nodiscard]] std::optional<TensorBackingFile> tensor_backing_file(std::string_view name) const;
    void evict_tensor_pages(std::string_view name, uint64_t offset = 0, uint64_t bytes = 0) const noexcept;
    [[nodiscard]] std::string_view architecture() const { return architecture_; }
    void assert_unchanged() const;

private:
    std::vector<Gguf> shards_;
    std::map<std::string, TensorLocation, std::less<>> tensors_;
    std::string architecture_;
};

} // namespace tokspur
