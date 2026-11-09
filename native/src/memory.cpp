#include "tokspur/memory.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace tokspur {
void validate_ngram(const NgramSpec& s, NgramPolicy p) {
    if (s.order < 2 || s.order > 8 || !s.heads || s.heads > 32 || !s.row_width || s.row_width > 65536 ||
        !s.token_limit || s.eos >= s.token_limit || s.image >= s.token_limit || s.multipliers.size() != s.order ||
        s.offsets.size() != (s.order - 1) * s.heads || s.vocabularies.size() != s.offsets.size() ||
        uint32_t(p.order) > 1 || uint32_t(p.boundary) > 2) throw FormatError("invalid n-gram specification/policy");
    uint64_t offset = 0;
    for (size_t i = 0; i < s.offsets.size(); ++i) {
        if (s.offsets[i] != offset || !s.vocabularies[i]) throw FormatError("invalid n-gram row partition");
        offset = checked_add(offset, s.vocabularies[i]);
    }
    for (auto m : s.multipliers) if (!m || !(m & 1)) throw FormatError("n-gram multipliers must be positive odd integers");
}
NgramHistory empty_ngram_history(const NgramSpec& s) {
    validate_ngram(s, {HashOrder::newest_first, NgramBoundary::literal, false});
    NgramHistory out; out.previous.fill(s.eos); return out;
}
NgramStep ngram_step(const NgramSpec& s, NgramPolicy p, const NgramHistory& history, uint32_t token) {
    validate_ngram(s, p);
    NgramStep out{history, std::vector<uint64_t>(s.offsets.size())};
    out.next = ngram_step(s, p, history, token, out.rows); return out;
}
NgramHistory ngram_step(const NgramSpec& s, NgramPolicy p, const NgramHistory& history,
                       uint32_t token, std::span<uint64_t> rows) {
    validate_ngram(s, p);
    if (rows.size() != s.offsets.size()) throw FormatError("n-gram output row span mismatch");
    if (token >= s.token_limit) throw FormatError("n-gram token outside vocabulary");
    for (uint32_t i = 0; i + 1 < s.order; ++i) if (history.previous[i] >= s.token_limit) throw FormatError("invalid n-gram history token");
    const auto current = p.image_as_eos && token == s.image ? s.eos : token;
    const bool boundary = current == s.eos;
    const auto active = boundary && p.boundary == NgramBoundary::reset_before ? empty_ngram_history(s) : history;
    NgramHistory next = active;
    for (uint32_t n = 2; n <= s.order; ++n) {
        uint64_t hash = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const auto age = p.order == HashOrder::newest_first ? i : n - i - 1;
            const auto id = age ? active.previous[age - 1] : current;
            hash ^= s.multipliers[i] * uint64_t(id); // specified modulo-2^64 multiplication
        }
        for (uint32_t h = 0; h < s.heads; ++h) {
            const auto index = (n - 2) * s.heads + h;
            // validate_ngram already bounded every complete row partition.
            rows[index] = s.offsets[index] + hash % s.vocabularies[index];
        }
    }
    if (boundary && p.boundary != NgramBoundary::literal) next = empty_ngram_history(s);
    else {
        for (uint32_t i = s.order - 2; i > 0; --i) next.previous[i] = active.previous[i - 1];
        next.previous[0] = current;
    }
    return next;
}
namespace {
uint64_t table_slots(uint32_t capacity) { return std::bit_ceil(uint64_t(capacity) * 2); }
uint64_t storage(uint32_t n, uint32_t width) {
    return checked_add(sizeof(EmbeddingRows), checked_add(uint64_t(width) * 4,
        checked_add(checked_mul(n, uint64_t(width) * 4 + 16), checked_mul(table_slots(n), 4))));
}
void advise_willneed_bounded(std::span<const std::byte> source, uint64_t first_byte, uint64_t end_byte) noexcept {
    constexpr uintptr_t page_bytes = 16384;
    if (source.size() < (16ull << 20) || first_byte >= end_byte || end_byte > source.size()) return;
    const uintptr_t src_begin = reinterpret_cast<uintptr_t>(source.data());
    const uintptr_t src_end = src_begin + source.size();
    const uintptr_t safe_begin = (src_begin + page_bytes - 1) & ~(page_bytes - 1);
    const uintptr_t safe_end = src_end & ~(page_bytes - 1);
    if (safe_begin >= safe_end) return;
    const uintptr_t req_lo = (src_begin + uintptr_t(first_byte)) & ~(page_bytes - 1);
    const uintptr_t req_hi = (src_begin + uintptr_t(end_byte) + page_bytes - 1) & ~(page_bytes - 1);
    const uintptr_t lo = std::max(req_lo, safe_begin);
    const uintptr_t hi = std::min(req_hi, safe_end);
    if (hi > lo) {
        static_cast<void>(posix_madvise(reinterpret_cast<void*>(lo), size_t(hi - lo), POSIX_MADV_WILLNEED));
    }
}
void pread_exact(PreadSource src, uint64_t rel_offset, std::span<std::byte> buf) {
    if (!src.valid() || buf.empty() || rel_offset > src.byte_length || buf.size() > src.byte_length - rel_offset) {
        throw FormatError("invalid pread tensor slice");
    }
    const uint64_t base_off = checked_add(src.file_offset, rel_offset);
    size_t done = 0;
    while (done < buf.size()) {
        const uint64_t cur_off = checked_add(base_off, done);
        if (cur_off > uint64_t(std::numeric_limits<off_t>::max())) throw FormatError("pread offset overflow");
        const ssize_t n = ::pread(src.fd, buf.data() + done, buf.size() - done, static_cast<off_t>(cur_off));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw FormatError("pread failed on embedding tensor shard");
        }
        if (n == 0) throw FormatError("unexpected EOF during embedding tensor pread");
        done += static_cast<size_t>(n);
    }
}
void decode_one_row(PreadSource pread_src, std::span<const std::byte> mmap_src, uint32_t enc,
                    uint64_t rb, uint64_t row, std::span<float> dst) {
    if (pread_src.valid()) {
        std::array<std::byte, 1024> stack_raw{};
        if (rb <= stack_raw.size()) {
            auto slice = std::span<std::byte>(stack_raw.data(), static_cast<size_t>(rb));
            pread_exact(pread_src, row * rb, slice);
            dequantize(enc, slice, dst);
        } else {
            thread_local std::vector<std::byte> tls_raw;
            tls_raw.resize(static_cast<size_t>(rb));
            pread_exact(pread_src, row * rb, tls_raw);
            dequantize(enc, tls_raw, dst);
        }
    } else {
        dequantize(enc, mmap_src.subspan(row * rb, rb), dst);
    }
}
}
NocacheTensorFile::NocacheTensorFile(const std::filesystem::path& path, uint64_t file_offset, uint64_t byte_length,
                                     uint64_t expected_device, uint64_t expected_inode) {
    if (!byte_length) throw FormatError("nocache tensor file requires positive byte length");
    const int opened = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (opened < 0) throw FormatError("cannot open nocache tensor file");
    struct stat info{};
    if (::fstat(opened, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0) {
        ::close(opened);
        throw FormatError("nocache tensor file must be a regular file");
    }
    if ((expected_device && uint64_t(info.st_dev) != expected_device) ||
        (expected_inode && uint64_t(info.st_ino) != expected_inode)) {
        ::close(opened);
        throw FormatError("nocache tensor file inode mismatch");
    }
    uint64_t end = 0;
    try { end = checked_add(file_offset, byte_length); } catch (...) { ::close(opened); throw; }
    if (end > uint64_t(info.st_size)) {
        ::close(opened);
        throw FormatError("nocache tensor slice exceeds file size");
    }
#ifdef F_NOCACHE
    static_cast<void>(::fcntl(opened, F_NOCACHE, 1));
#endif
    fd_ = opened;
    file_offset_ = file_offset;
    byte_length_ = byte_length;
}
NocacheTensorFile::~NocacheTensorFile() {
    if (fd_ >= 0) ::close(fd_);
}
NocacheTensorFile::NocacheTensorFile(NocacheTensorFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      file_offset_(std::exchange(other.file_offset_, 0)),
      byte_length_(std::exchange(other.byte_length_, 0)) {}
NocacheTensorFile& NocacheTensorFile::operator=(NocacheTensorFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = std::exchange(other.fd_, -1);
        file_offset_ = std::exchange(other.file_offset_, 0);
        byte_length_ = std::exchange(other.byte_length_, 0);
    }
    return *this;
}
void dequantize_row_at(std::span<const std::byte> source, uint32_t encoding_type, uint32_t row_width,
                       uint64_t stored_rows, uint64_t logical_rows, uint64_t row,
                       std::span<float> destination, PreadSource pread_source) {
    const auto& e = encoding(encoding_type);
    if (!can_decode(encoding_type) || !row_width || row_width > 65536 || row_width % e.elements_per_block ||
        !logical_rows || logical_rows > stored_rows) {
        throw FormatError("unsupported stateless single-row gather geometry");
    }
    const uint64_t row_bytes = uint64_t(row_width / e.elements_per_block) * e.bytes_per_block;
    if (checked_mul(row_bytes, stored_rows) != source.size()) {
        throw FormatError("stateless single-row gather table byte size mismatch");
    }
    if (pread_source.valid() && pread_source.byte_length != source.size()) {
        throw FormatError("stateless single-row gather pread byte size mismatch");
    }
    if (row >= logical_rows || destination.size() != row_width) {
        throw FormatError("invalid stateless single-row index or destination");
    }
    decode_one_row(pread_source, source, encoding_type, row_bytes, row, destination);
}
void dequantize_rows_sorted_dedup(std::span<const std::byte> source, uint32_t encoding_type, uint32_t row_width,
                                  uint64_t stored_rows, uint64_t logical_rows, std::span<const uint64_t> ids,
                                  std::span<float> destination, PleGatherScratch& scratch, uint32_t max_threads,
                                  PreadSource pread_source) {
    const auto& e = encoding(encoding_type);
    if (!can_decode(encoding_type) || !row_width || row_width > 65536 || row_width % e.elements_per_block ||
        !logical_rows || logical_rows > stored_rows) {
        throw FormatError("unsupported stateless row gather geometry");
    }
    const uint64_t row_bytes = uint64_t(row_width / e.elements_per_block) * e.bytes_per_block;
    if (checked_mul(row_bytes, stored_rows) != source.size()) {
        throw FormatError("stateless row gather table byte size mismatch");
    }
    if (pread_source.valid() && pread_source.byte_length != source.size()) {
        throw FormatError("stateless row gather pread byte size mismatch");
    }
    if (ids.empty() || ids.size() > 32768 || destination.size() != uint64_t(ids.size()) * row_width) {
        throw FormatError("invalid stateless row gather shape");
    }
    for (const auto row : ids) {
        if (row >= logical_rows) throw FormatError("stateless row gather addresses padding/out-of-range row");
    }
    scratch.pairs.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) scratch.pairs[i] = {ids[i], uint32_t(i)};
    std::sort(scratch.pairs.begin(), scratch.pairs.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });
    const auto decode_range = [&](size_t begin, size_t end) {
        if (begin >= end) return;
        if (pread_source.valid()) {
            constexpr uint64_t coalesce_gap = 4096;
            const size_t io_cap = std::max<size_t>(32768, static_cast<size_t>(row_bytes));
            thread_local std::vector<std::byte> io_buf;
            if (io_buf.size() < io_cap) io_buf.resize(io_cap);
            for (size_t i = begin; i < end;) {
                const uint64_t first_row = scratch.pairs[i].first;
                uint64_t win_lo = first_row * row_bytes;
                uint64_t win_hi = win_lo + row_bytes;
                size_t j = i + 1;
                while (j < end && scratch.pairs[j].first == first_row) ++j;
                while (j < end) {
                    const uint64_t next_row = scratch.pairs[j].first;
                    const uint64_t next_lo = next_row * row_bytes;
                    const uint64_t next_hi = next_lo + row_bytes;
                    if (next_lo > win_hi + coalesce_gap || (next_hi - win_lo) > io_cap) break;
                    win_hi = next_hi;
                    ++j;
                    while (j < end && scratch.pairs[j].first == next_row) ++j;
                }
                const size_t span_bytes = static_cast<size_t>(win_hi - win_lo);
                pread_exact(pread_source, win_lo, std::span<std::byte>(io_buf.data(), span_bytes));
                for (size_t k = i; k < j;) {
                    const uint64_t row = scratch.pairs[k].first;
                    const uint32_t first_slot = scratch.pairs[k].second;
                    auto first_out = destination.subspan(uint64_t(first_slot) * row_width, row_width);
                    const size_t rel = static_cast<size_t>(row * row_bytes - win_lo);
                    dequantize(encoding_type, std::span<const std::byte>(io_buf.data() + rel, static_cast<size_t>(row_bytes)), first_out);
                    ++k;
                    while (k < j && scratch.pairs[k].first == row) {
                        const uint32_t dup_slot = scratch.pairs[k].second;
                        std::memcpy(destination.data() + uint64_t(dup_slot) * row_width, first_out.data(), row_width * sizeof(float));
                        ++k;
                    }
                }
                i = j;
            }
            return;
        }
        if (source.size() >= (16ull << 20)) {
            constexpr uint64_t merge_gap = 65536, max_window = 4ull << 20;
            bool active = false;
            uint64_t win_lo = 0, win_hi = 0;
            for (size_t i = begin; i < end;) {
                const uint64_t row = scratch.pairs[i].first;
                while (i < end && scratch.pairs[i].first == row) ++i;
                const uint64_t lo = row * row_bytes, hi = lo + row_bytes;
                if (!active) { active = true; win_lo = lo; win_hi = hi; }
                else if (lo <= win_hi + merge_gap && (hi - win_lo) <= max_window) { win_hi = std::max(win_hi, hi); }
                else { advise_willneed_bounded(source, win_lo, win_hi); win_lo = lo; win_hi = hi; }
            }
            if (active) advise_willneed_bounded(source, win_lo, win_hi);
        }
        for (size_t i = begin; i < end;) {
            const uint64_t row = scratch.pairs[i].first;
            const uint32_t first_slot = scratch.pairs[i].second;
            auto first_out = destination.subspan(uint64_t(first_slot) * row_width, row_width);
            dequantize(encoding_type, source.subspan(row * row_bytes, row_bytes), first_out);
            ++i;
            while (i < end && scratch.pairs[i].first == row) {
                const uint32_t dup_slot = scratch.pairs[i].second;
                std::memcpy(destination.data() + uint64_t(dup_slot) * row_width, first_out.data(), row_width * sizeof(float));
                ++i;
            }
        }
    };
    const size_t total = scratch.pairs.size();
    const size_t workers = (max_threads > 1 && total >= 512)
        ? std::min<size_t>(std::min<uint32_t>(max_threads, 16u), total / 256)
        : 1;
    if (workers <= 1) {
        decode_range(0, total);
        return;
    }
    std::array<size_t, 17> bounds{};
    bounds[0] = 0;
    bounds[workers] = total;
    for (size_t w = 1; w < workers; ++w) {
        // A repeated row may extend past several nominal partitions, so later boundaries cannot move backwards.
        size_t pos = std::max(bounds[w - 1], (total * w) / workers);
        while (pos > bounds[w - 1] && pos < total && scratch.pairs[pos - 1].first == scratch.pairs[pos].first) {
            ++pos;
        }
        bounds[w] = pos;
    }
    std::array<std::exception_ptr, 16> errors{};
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    try {
        for (size_t w = 1; w < workers; ++w) {
            pool.emplace_back([&, w]() {
                try { decode_range(bounds[w], bounds[w + 1]); }
                catch (...) { errors[w] = std::current_exception(); }
            });
        }
    } catch (...) {
        for (auto& th : pool) th.join();
        throw;
    }
    try { decode_range(bounds[0], bounds[1]); }
    catch (...) { errors[0] = std::current_exception(); }
    for (auto& th : pool) th.join();
    for (size_t w = 0; w < workers; ++w) {
        if (errors[w]) std::rethrow_exception(errors[w]);
    }
}
EmbeddingRows::EmbeddingRows(std::span<const std::byte> source, uint32_t type, uint32_t width, uint64_t stored, uint64_t logical, uint64_t budget, PreadSource pread_source, bool parallel_gather)
    : source_(source), pread_source_(pread_source), parallel_gather_(parallel_gather), encoding_(type), width_(width), logical_rows_(logical), budget_(budget) {
    const auto& e = encoding(type);
    if (!can_decode(type) || !width || width > 65536 || width % e.elements_per_block || !logical || logical > stored ||
        budget > 2ull * 1024 * 1024 * 1024) throw FormatError("unsupported embedding cache geometry/budget");
    row_bytes_ = uint64_t(width / e.elements_per_block) * e.bytes_per_block;
    if (checked_mul(row_bytes_, stored) != source.size()) throw FormatError("embedding table byte size mismatch");
    if (pread_source_.valid() && pread_source_.byte_length != source.size()) throw FormatError("embedding table pread byte size mismatch");
    uint32_t lo = 0, hi = static_cast<uint32_t>(std::min(logical, uint64_t(16'777'216)));
    while (lo < hi) { const auto mid = lo + (hi - lo + 1) / 2; if (storage(mid, width) <= budget) lo = mid; else hi = mid - 1; }
    if (!lo) throw FormatError("embedding cache budget too small for one row");
    capacity_ = lo; rows_.resize(uint64_t(lo) * width); scratch_.resize(width); keys_.resize(lo);
    previous_.resize(lo, none); next_.resize(lo, none); table_.resize(table_slots(lo), none);
    if (charged_bytes() > budget_) throw FormatError("embedding allocator capacities exceed cache budget");
}
uint64_t EmbeddingRows::charged_bytes() const {
    return sizeof(*this) + (rows_.capacity() + scratch_.capacity()) * sizeof(float) + keys_.capacity() * sizeof(uint64_t) +
        (table_.capacity() + previous_.capacity() + next_.capacity()) * sizeof(uint32_t);
}
RowCacheStats EmbeddingRows::stats() const {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache statistics read from another host thread");
    return stats_;
}
uint64_t EmbeddingRows::reclaim() {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache reclaimed from another host thread");
    const auto before = charged_bytes();
    stats_.evictions += used_;
    std::vector<float>().swap(rows_);
    std::vector<uint64_t>().swap(keys_);
    std::vector<uint32_t>().swap(table_); std::vector<uint32_t>().swap(previous_); std::vector<uint32_t>().swap(next_);
    capacity_ = used_ = 0; head_ = tail_ = none;
    return before - charged_bytes();
}
uint32_t EmbeddingRows::home(uint64_t row) const { return uint32_t(((row * 11400714819323198485ull) ^ (row >> 32)) & (table_.size() - 1)); }
uint32_t EmbeddingRows::find(uint64_t row) const {
    auto p = home(row);
    for (size_t probe = 0; probe < table_.size(); ++probe) {
        if (table_[p] == none || keys_[table_[p]] == row) return p;
        p = (p + 1) & (table_.size() - 1);
    }
    throw FormatError("embedding cache index is inconsistent");
}
void EmbeddingRows::erase(uint64_t row) {
    auto hole = find(row);
    if (table_[hole] == none) throw FormatError("missing embedding cache eviction key");
    const auto mask = uint32_t(table_.size() - 1);
    for (uint32_t p = (hole + 1) & mask; table_[p] != none; p = (p + 1) & mask) {
        const auto ideal = home(keys_[table_[p]]);
        if (((hole - ideal) & mask) < ((p - ideal) & mask)) { table_[hole] = table_[p]; hole = p; }
    }
    table_[hole] = none;
}
void EmbeddingRows::touch(uint32_t slot) {
    if (head_ == slot) return;
    const auto before = previous_[slot], after = next_[slot];
    if (before != none) next_[before] = after;
    if (after != none) previous_[after] = before;
    if (tail_ == slot) tail_ = before;
    previous_[slot] = none; next_[slot] = head_;
    if (head_ != none) previous_[head_] = slot;
    head_ = slot; if (tail_ == none) tail_ = slot;
}
void EmbeddingRows::copy(uint64_t row, std::span<float> destination) {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    if (row >= logical_rows_ || destination.size() != width_) throw FormatError("invalid embedding row/destination (padding is not addressable)");
    if (!capacity_) {
        ++stats_.misses;
        try { decode_one_row(pread_source_, source_, encoding_, row_bytes_, row, scratch_); }
        catch (...) { ++stats_.failures; throw; }
        std::memcpy(destination.data(), scratch_.data(), width_ * sizeof(float));
        return;
    }
    auto at = find(row); uint32_t slot = table_[at];
    if (slot == none) {
        ++stats_.misses;
        try { decode_one_row(pread_source_, source_, encoding_, row_bytes_, row, scratch_); }
        catch (...) { ++stats_.failures; throw; }
        if (used_ == capacity_) { slot = tail_; erase(keys_[slot]); ++stats_.evictions; }
        else slot = used_++;
        keys_[slot] = row; std::copy(scratch_.begin(), scratch_.end(), rows_.begin() + uint64_t(slot) * width_);
        at = find(row); table_[at] = slot;
    } else ++stats_.hits;
    touch(slot);
    std::memcpy(destination.data(), rows_.data() + uint64_t(slot) * width_, width_ * sizeof(float));
}
bool EmbeddingRows::try_copy(uint64_t row, std::span<float> destination) {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    if (row >= logical_rows_ || destination.size() != width_) throw FormatError("invalid embedding row/destination (padding is not addressable)");
    if (!capacity_) return false;
    const auto at = find(row);
    const uint32_t slot = table_[at];
    if (slot == none) return false;
    ++stats_.hits;
    touch(slot);
    std::memcpy(destination.data(), rows_.data() + uint64_t(slot) * width_, width_ * sizeof(float));
    return true;
}
void EmbeddingRows::insert_decoded(uint64_t row, std::span<const float> decoded, bool count_miss) {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    if (row >= logical_rows_ || decoded.size() != width_) throw FormatError("invalid decoded embedding row");
    for (const float v : decoded) {
        if (!std::isfinite(v)) throw FormatError("nonfinite decoded embedding row");
    }
    if (!capacity_) { if (count_miss) ++stats_.misses; return; }
    auto at = find(row);
    uint32_t slot = table_[at];
    if (slot == none) {
        if (count_miss) ++stats_.misses;
        if (used_ == capacity_) { slot = tail_; erase(keys_[slot]); ++stats_.evictions; }
        else slot = used_++;
        keys_[slot] = row;
        std::memcpy(rows_.data() + uint64_t(slot) * width_, decoded.data(), width_ * sizeof(float));
        at = find(row);
        table_[at] = slot;
    }
    touch(slot);
}
void EmbeddingRows::record_failure() {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    ++stats_.failures;
}
void EmbeddingRows::seed_from_dedup(const PleGatherScratch& scratch, std::span<const float> embeddings) {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    const size_t total = scratch.pairs.size();
    for (size_t i = 0; i < total;) {
        const uint64_t row = scratch.pairs[i].first;
        const uint32_t first_slot = scratch.pairs[i].second;
        if (row >= logical_rows_ || (uint64_t(first_slot) + 1ull) * width_ > embeddings.size()) {
            throw FormatError("invalid deduplicated embedding seed span");
        }
        ++i;
        while (i < total && scratch.pairs[i].first == row) ++i;
        if (!capacity_) continue;
        const auto at = find(row);
        uint32_t slot = table_[at];
        if (slot == none) {
            if (used_ == capacity_) { slot = tail_; erase(keys_[slot]); ++stats_.evictions; }
            else slot = used_++;
            keys_[slot] = row;
            std::memcpy(rows_.data() + uint64_t(slot) * width_,
                        embeddings.data() + uint64_t(first_slot) * width_,
                        width_ * sizeof(float));
            table_[find(row)] = slot;
        }
        touch(slot);
    }
}
void EmbeddingRows::gather(std::span<const uint64_t> ids, std::span<float> out) {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    if (ids.empty() || ids.size() > 256 || out.size() != uint64_t(ids.size()) * width_) throw FormatError("invalid embedding gather shape");
    // Validate every index before touching the LRU or partial output.
    for (auto row : ids) if (row >= logical_rows_) throw FormatError("embedding gather addresses padding/out-of-range row");
    if (!capacity_ || !parallel_gather_ || ids.size() <= 4 || (!pread_source_.valid() && source_.size() < (16ull << 20))) {
        for (size_t i = 0; i < ids.size(); ++i) copy(ids[i], out.subspan(i * width_, width_));
        return;
    }

    std::array<uint64_t, 256> unique_misses{};
    size_t miss_count = 0;
    for (const uint64_t row : ids) {
        if (table_[find(row)] == none) unique_misses[miss_count++] = row;
    }
    if (miss_count < 4) {
        for (size_t i = 0; i < ids.size(); ++i) copy(ids[i], out.subspan(i * width_, width_));
        return;
    }
    std::sort(unique_misses.begin(), unique_misses.begin() + ptrdiff_t(miss_count));
    miss_count = size_t(std::unique(unique_misses.begin(), unique_misses.begin() + ptrdiff_t(miss_count)) - unique_misses.begin());
    if (miss_count < 4) {
        for (size_t i = 0; i < ids.size(); ++i) copy(ids[i], out.subspan(i * width_, width_));
        return;
    }

    if (!pread_source_.valid()) {
        for (size_t m = 0; m < miss_count; ++m) {
            const uint64_t lo = unique_misses[m] * row_bytes_;
            advise_willneed_bounded(source_, lo, lo + row_bytes_);
        }
    }

    constexpr size_t kWorkers = 11;
    struct MissReaderPool {
        std::mutex call_mu{};
        std::mutex cv_mu{};
        std::condition_variable cv_start{}, cv_done{};
        uint64_t epoch = 0;
        size_t active_workers = 0;
        size_t finished_workers = 0;
        bool stop = false;
        std::span<const std::byte> src{};
        PreadSource pread_src{};
        uint32_t enc = 0, w = 0;
        uint64_t rb = 0;
        const uint64_t* rows_ptr = nullptr;
        size_t count = 0;
        float* dst_ptr = nullptr;
        std::atomic<size_t> next_item{0};
        std::array<std::exception_ptr, kWorkers + 1> errors{};
        std::array<std::thread, kWorkers> threads{};

        MissReaderPool() {
            try {
                for (size_t i = 0; i < kWorkers; ++i) {
                    threads[i] = std::thread([this, i]() {
                        uint64_t seen = 0;
                        for (;;) {
                            size_t my_active = 0;
                            {
                                std::unique_lock<std::mutex> lk(cv_mu);
                                cv_start.wait(lk, [&]() { return stop || epoch != seen; });
                                if (stop) return;
                                seen = epoch;
                                my_active = active_workers;
                            }
                            if (i < my_active) {
                                for (;;) {
                                    const size_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
                                    if (idx >= count) break;
                                    try {
                                        const uint64_t r = rows_ptr[idx];
                                        decode_one_row(pread_src, src, enc, rb, r, std::span<float>(dst_ptr + uint64_t(idx) * w, w));
                                    } catch (...) {
                                        if (!errors[i + 1]) errors[i + 1] = std::current_exception();
                                    }
                                }
                                std::lock_guard<std::mutex> lk(cv_mu);
                                if (++finished_workers == my_active) cv_done.notify_one();
                            }
                        }
                    });
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lk(cv_mu);
                    stop = true;
                }
                cv_start.notify_all();
                for (auto& th : threads) if (th.joinable()) th.join();
                throw;
            }
        }
        ~MissReaderPool() {
            {
                std::lock_guard<std::mutex> lk(cv_mu);
                stop = true;
            }
            cv_start.notify_all();
            for (auto& th : threads) if (th.joinable()) th.join();
        }
    };

    static MissReaderPool pool;
    thread_local std::vector<float> miss_decoded;
    miss_decoded.resize(uint64_t(miss_count) * width_);

    {
        std::lock_guard<std::mutex> call_lk(pool.call_mu);
        const size_t bg_workers = std::min<size_t>(kWorkers, miss_count - 1);
        for (auto& err : pool.errors) err = nullptr;
        pool.src = source_;
        pool.pread_src = pread_source_;
        pool.enc = encoding_;
        pool.w = width_;
        pool.rb = row_bytes_;
        pool.rows_ptr = unique_misses.data();
        pool.count = miss_count;
        pool.dst_ptr = miss_decoded.data();
        pool.next_item.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(pool.cv_mu);
            pool.active_workers = bg_workers;
            pool.finished_workers = 0;
            ++pool.epoch;
        }
        pool.cv_start.notify_all();

        for (;;) {
            const size_t idx = pool.next_item.fetch_add(1, std::memory_order_relaxed);
            if (idx >= miss_count) break;
            try {
                const uint64_t r = unique_misses[idx];
                decode_one_row(pread_source_, source_, encoding_, row_bytes_, r,
                               std::span<float>(miss_decoded.data() + uint64_t(idx) * width_, width_));
            } catch (...) {
                if (!pool.errors[0]) pool.errors[0] = std::current_exception();
            }
        }

        if (bg_workers > 0) {
            std::unique_lock<std::mutex> lk(pool.cv_mu);
            pool.cv_done.wait(lk, [&]() { return pool.finished_workers == bg_workers; });
        }

        for (size_t e = 0; e <= bg_workers; ++e) {
            if (pool.errors[e]) {
                ++stats_.failures;
                std::rethrow_exception(pool.errors[e]);
            }
        }
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        const uint64_t row = ids[i];
        auto at = find(row);
        uint32_t slot = table_[at];
        if (slot == none) {
            const auto* it = std::lower_bound(unique_misses.data(), unique_misses.data() + miss_count, row);
            if (it == unique_misses.data() + miss_count || *it != row) {
                // An initial cache hit may have been evicted by earlier misses in this gather.
                copy(row, out.subspan(i * width_, width_));
                continue;
            }
            ++stats_.misses;
            const size_t m_idx = size_t(it - unique_misses.data());
            const float* decoded_row = miss_decoded.data() + uint64_t(m_idx) * width_;
            if (used_ == capacity_) { slot = tail_; erase(keys_[slot]); ++stats_.evictions; }
            else slot = used_++;
            keys_[slot] = row;
            std::memcpy(rows_.data() + uint64_t(slot) * width_, decoded_row, width_ * sizeof(float));
            at = find(row);
            table_[at] = slot;
        } else {
            ++stats_.hits;
        }
        touch(slot);
        std::memcpy(out.data() + uint64_t(i) * width_, rows_.data() + uint64_t(slot) * width_, width_ * sizeof(float));
    }
}
void EmbeddingRows::gather_chunk_dedup(std::span<const uint64_t> ids, std::span<float> out, PleGatherScratch& scratch) {
    if (thread_ != std::this_thread::get_id()) throw FormatError("embedding cache used from another host thread");
    if (ids.empty() || ids.size() > 32768 || out.size() != uint64_t(ids.size()) * width_) throw FormatError("invalid embedding chunk gather shape");
    for (auto row : ids) if (row >= logical_rows_) throw FormatError("embedding gather addresses padding/out-of-range row");
    scratch.pairs.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) scratch.pairs[i] = {ids[i], uint32_t(i)};
    std::sort(scratch.pairs.begin(), scratch.pairs.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });
    if (!pread_source_.valid() && source_.size() >= (16ull << 20)) {
        constexpr uint64_t merge_gap = 65536, max_window = 4ull << 20;
        bool active = false;
        uint64_t win_lo = 0, win_hi = 0;
        for (size_t i = 0; i < scratch.pairs.size();) {
            const uint64_t row = scratch.pairs[i].first;
            while (i < scratch.pairs.size() && scratch.pairs[i].first == row) ++i;
            if (capacity_ && table_[find(row)] != none) continue;
            const uint64_t lo = row * row_bytes_, hi = lo + row_bytes_;
            if (!active) { active = true; win_lo = lo; win_hi = hi; }
            else if (lo <= win_hi + merge_gap && (hi - win_lo) <= max_window) { win_hi = std::max(win_hi, hi); }
            else { advise_willneed_bounded(source_, win_lo, win_hi); win_lo = lo; win_hi = hi; }
        }
        if (active) advise_willneed_bounded(source_, win_lo, win_hi);
    }
    for (size_t i = 0; i < scratch.pairs.size();) {
        const uint64_t row = scratch.pairs[i].first;
        const uint32_t first_slot = scratch.pairs[i].second;
        auto first_out = out.subspan(uint64_t(first_slot) * width_, width_);
        copy(row, first_out);
        ++i;
        while (i < scratch.pairs.size() && scratch.pairs[i].first == row) {
            const uint32_t dup_slot = scratch.pairs[i].second;
            std::memcpy(out.data() + uint64_t(dup_slot) * width_, first_out.data(), width_ * sizeof(float));
            ++stats_.hits;
            ++i;
        }
    }
}
} // namespace tokspur
