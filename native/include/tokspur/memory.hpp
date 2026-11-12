#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#ifndef TOKSPUR_PRIVATE_Q8_TILE64_CACHE
#define TOKSPUR_PRIVATE_Q8_TILE64_CACHE 1
#endif
#if TOKSPUR_PRIVATE_Q8_TILE64_CACHE != 0 && TOKSPUR_PRIVATE_Q8_TILE64_CACHE != 1
#error "TOKSPUR_PRIVATE_Q8_TILE64_CACHE must be 0 or 1"
#endif

namespace tokspur {
inline constexpr bool q8_tile64_cache_enabled = TOKSPUR_PRIVATE_Q8_TILE64_CACHE != 0;
// Snapshot inputs: candidate sums describe registrations, not unique source pages.
// Aliases refer to a live owner slot; only distinct owners contribute allocation bytes.
struct Q8Tile64CacheCandidate {
    uint64_t source_bytes = 0, repack_logical_bytes = 0;
    std::optional<size_t> cached_allocation_slot;
};
struct Q8Tile64CacheAllocation {
    size_t slot = 0;
    uint64_t logical_bytes = 0, charged_bytes = 0;
    bool live = false, alias = false;
};
struct Q8Tile64CacheStats {
    bool enabled = true;
    uint64_t registered_candidates = 0, registered_source_bytes = 0, candidate_repack_logical_bytes = 0;
    uint64_t covered_candidates = 0, omitted_candidates = 0, omitted_repack_logical_bytes = 0;
    uint64_t owning_allocations = 0, owned_logical_bytes = 0, owned_charged_bytes = 0, aliases = 0;
};
[[nodiscard]] uint64_t q8_tile64_repack_bytes(uint64_t columns, uint64_t rows);
[[nodiscard]] Q8Tile64CacheStats summarize_q8_tile64_cache(bool enabled,
    std::span<const Q8Tile64CacheCandidate> candidates, std::span<const Q8Tile64CacheAllocation> entries);

enum class HashOrder : uint32_t { newest_first, oldest_first };
enum class NgramBoundary : uint32_t { literal, reset_before, reset_after };
struct NgramPolicy { HashOrder order; NgramBoundary boundary; bool image_as_eos; };
struct NgramSpec {
    uint32_t order, heads, row_width, token_limit, eos, image;
    std::vector<uint64_t> multipliers, offsets, vocabularies;
};
struct NgramHistory {
    std::array<uint32_t, 7> previous{};
    bool operator==(const NgramHistory&) const = default;
};
struct NgramStep { NgramHistory next; std::vector<uint64_t> rows; };
void validate_ngram(const NgramSpec& spec, NgramPolicy policy);
NgramHistory empty_ngram_history(const NgramSpec& spec);
// Raw token IDs and explicit boundary/order hypotheses. No unstated tokenizer
// compression map or checkpoint default is inferred from another implementation.
NgramStep ngram_step(const NgramSpec& spec, NgramPolicy policy, const NgramHistory& history, uint32_t token);
// Allocation-free form; exact row span is validated before any output writes.
// On success returns the same next history as the owning convenience form.
NgramHistory ngram_step(const NgramSpec& spec, NgramPolicy policy, const NgramHistory& history,
                       uint32_t token, std::span<uint64_t> rows);

struct RowCacheStats { uint64_t hits = 0, misses = 0, evictions = 0, failures = 0; };
struct PleGatherScratch {
    std::vector<std::pair<uint64_t, uint32_t>> pairs;
};
struct PreadSource {
    int fd = -1;
    uint64_t file_offset = 0;
    uint64_t byte_length = 0;
    [[nodiscard]] bool valid() const noexcept { return fd >= 0 && byte_length > 0; }
};
class NocacheTensorFile {
public:
    NocacheTensorFile() = default;
    NocacheTensorFile(const std::filesystem::path& path, uint64_t file_offset, uint64_t byte_length,
                      uint64_t expected_device = 0, uint64_t expected_inode = 0);
    ~NocacheTensorFile();
    NocacheTensorFile(const NocacheTensorFile&) = delete;
    NocacheTensorFile& operator=(const NocacheTensorFile&) = delete;
    NocacheTensorFile(NocacheTensorFile&& other) noexcept;
    NocacheTensorFile& operator=(NocacheTensorFile&& other) noexcept;
    [[nodiscard]] PreadSource source() const noexcept { return {fd_, file_offset_, byte_length_}; }
    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0 && byte_length_ > 0; }
private:
    int fd_ = -1;
    uint64_t file_offset_ = 0;
    uint64_t byte_length_ = 0;
};
// Stateless chunk-wide sorted deduplicated row dequantization with coalesced
// read-ahead page hints or F_NOCACHE pread; safe on worker threads without mutating an LRU cache.
void dequantize_rows_sorted_dedup(std::span<const std::byte> source, uint32_t encoding_type, uint32_t row_width,
                                  uint64_t stored_rows, uint64_t logical_rows, std::span<const uint64_t> ids,
                                  std::span<float> destination, PleGatherScratch& scratch, uint32_t max_threads = 1,
                                  PreadSource pread_source = {});
void dequantize_row_at(std::span<const std::byte> source, uint32_t encoding_type, uint32_t row_width,
                       uint64_t stored_rows, uint64_t logical_rows, uint64_t row,
                       std::span<float> destination, PreadSource pread_source = {});
// Fixed slabs + open-addressed index + exact LRU. No unbounded per-entry heap
// allocation and no borrowed row view survives mutation. Source stays immutable.
class EmbeddingRows {
public:
    EmbeddingRows(std::span<const std::byte> encoded, uint32_t encoding, uint32_t row_width,
                   uint64_t stored_rows, uint64_t logical_rows, uint64_t byte_budget,
                   PreadSource pread_source = {}, bool parallel_gather = false);
    EmbeddingRows(const EmbeddingRows&) = delete;
    EmbeddingRows& operator=(const EmbeddingRows&) = delete;
    void copy(uint64_t row, std::span<float> destination);
    [[nodiscard]] bool try_copy(uint64_t row, std::span<float> destination);
    void insert_decoded(uint64_t row, std::span<const float> decoded, bool count_miss = true);
    void record_failure();
    void seed_from_dedup(const PleGatherScratch& scratch, std::span<const float> embeddings);
    void gather(std::span<const uint64_t> rows, std::span<float> destination);
    void gather_chunk_dedup(std::span<const uint64_t> rows, std::span<float> destination, PleGatherScratch& scratch);
    // Release optional slabs and continue decoding from the immutable source without retaining rows.
    uint64_t reclaim();
    [[nodiscard]] uint64_t charged_bytes() const;
    [[nodiscard]] uint32_t capacity() const { return capacity_; }
    [[nodiscard]] RowCacheStats stats() const;
private:
    static constexpr uint32_t none = 0xffffffffu;
    std::span<const std::byte> source_;
    PreadSource pread_source_{};
    bool parallel_gather_ = false;
    uint32_t encoding_, width_, capacity_ = 0, used_ = 0, head_ = none, tail_ = none;
    uint64_t logical_rows_, row_bytes_, budget_;
    std::vector<float> rows_, scratch_;
    std::vector<uint64_t> keys_;
    std::vector<uint32_t> table_, previous_, next_;
    RowCacheStats stats_;
    const std::thread::id thread_ = std::this_thread::get_id();
    uint32_t home(uint64_t row) const;
    uint32_t find(uint64_t row) const;
    void erase(uint64_t row);
    void touch(uint32_t slot);
};
int memory_check_cli(int argc, char** argv);
} // namespace tokspur
