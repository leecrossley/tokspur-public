#include "tokspur/ssd_checkpoint.hpp"

#include <CommonCrypto/CommonCrypto.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace tokspur {

namespace {

void validate_filename(const std::string& name) {
    if (name.empty() || name == "." || name == "..") {
        throw std::invalid_argument("invalid checkpoint filename");
    }
    for (char c : name) {
        if (c == '/' || c == '\0') {
            throw std::invalid_argument("checkpoint filename cannot contain path separators or null bytes");
        }
    }
}

std::array<uint8_t, 32> sha256_buffer(const void* data, size_t length) {
    std::array<uint8_t, 32> digest{};
    CC_SHA256(data, CC_LONG(length), digest.data());
    return digest;
}

constexpr std::array<char, 8> checkpoint_magic_expected{'T', 'O', 'K', 'C', 'K', 'P', 'T', '1'};
constexpr uint32_t max_checkpoint_metadata_bytes = 64u << 20;

bool query_free_bytes(int dir_fd, uint64_t& out_free_bytes) noexcept {
    if (dir_fd < 0) return false;
    struct statvfs vfs{};
    if (::fstatvfs(dir_fd, &vfs) != 0) return false;
    const uint64_t bavail = uint64_t(vfs.f_bavail);
    const uint64_t frsize = uint64_t(vfs.f_frsize);
    if (frsize > 0 && bavail > std::numeric_limits<uint64_t>::max() / frsize) {
        out_free_bytes = std::numeric_limits<uint64_t>::max();
    } else {
        out_free_bytes = bavail * frsize;
    }
    return true;
}

bool has_floor_headroom(uint64_t free_bytes, uint64_t floor_bytes, uint64_t incoming_bytes) noexcept {
    return free_bytes >= floor_bytes && (free_bytes - floor_bytes) >= incoming_bytes;
}

bool has_budget_headroom(uint64_t occupied_bytes, uint64_t budget_bytes, uint64_t incoming_bytes) noexcept {
    return incoming_bytes <= budget_bytes && occupied_bytes <= (budget_bytes - incoming_bytes);
}

uint64_t saturating_add_u64(uint64_t a, uint64_t b) noexcept {
    return (a > std::numeric_limits<uint64_t>::max() - b) ? std::numeric_limits<uint64_t>::max() : (a + b);
}

uint64_t effective_free_after_unlinks(uint64_t initial_free_bytes,
                                      uint64_t fresh_free_bytes,
                                      uint64_t reclaimed_single_link_bytes) noexcept {
    if (fresh_free_bytes < initial_free_bytes) {
        return saturating_add_u64(fresh_free_bytes, reclaimed_single_link_bytes);
    }
    return std::max(fresh_free_bytes, saturating_add_u64(initial_free_bytes, reclaimed_single_link_bytes));
}

bool header_is_valid(const CheckpointFileHeader& header) noexcept {
    if (header.magic != checkpoint_magic_expected || header.version != checkpoint_format_version) {
        return false;
    }
    if (header.chunk_size != uint32_t(checkpoint_chunk_bytes)) {
        return false;
    }
    if (header.total_payload_bytes > maximum_checkpoint_ssd_budget) {
        return false;
    }
    const uint64_t expected_chunks =
        (header.total_payload_bytes + checkpoint_chunk_bytes - 1) / checkpoint_chunk_bytes;
    if (uint64_t(header.chunk_count) != expected_chunks) {
        return false;
    }
    const auto expected_hdr_digest =
        sha256_buffer(&header, sizeof(header) - sizeof(header.header_sha256));
    return header.header_sha256 == expected_hdr_digest;
}

struct Fd {
    int value;
    ~Fd() { if (value >= 0) ::close(value); }
};

bool same_file(const struct stat& a, const struct stat& b) noexcept {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

struct CheckpointLayout {
    CheckpointFileHeader header{};
    uint32_t metadata_bytes = 0;
    off_t metadata_offset = 0;
    uint64_t payload_offset = 0;
};

bool read_checkpoint_layout(int fd, const struct stat& file, CheckpointLayout& layout) noexcept {
    if (!S_ISREG(file.st_mode) || file.st_size < 0 ||
        ::pread(fd, &layout.header, sizeof(layout.header), 0) != ssize_t(sizeof(layout.header)) ||
        !header_is_valid(layout.header)) {
        return false;
    }
    const auto length_offset = off_t(sizeof(layout.header) + uint64_t(layout.header.chunk_count) * 32u);
    if (::pread(fd, &layout.metadata_bytes, sizeof(layout.metadata_bytes), length_offset) != ssize_t(sizeof(layout.metadata_bytes)) ||
        layout.metadata_bytes == 0 || layout.metadata_bytes > max_checkpoint_metadata_bytes) {
        return false;
    }
    layout.metadata_offset = length_offset + off_t(sizeof(layout.metadata_bytes));
    layout.payload_offset = uint64_t(layout.metadata_offset) + layout.metadata_bytes;
    return uint64_t(file.st_size) == layout.payload_offset + layout.header.total_payload_bytes;
}

DIR* open_directory_scan(int dir_fd) noexcept {
    const int fd = ::openat(dir_fd, ".", O_DIRECTORY | O_CLOEXEC | O_RDONLY);
    if (fd < 0) return nullptr;
    DIR* directory = ::fdopendir(fd);
    if (!directory) ::close(fd);
    return directory;
}

bool discard_observed_file(int dir_fd, const std::string& filename, const struct stat& expected,
                           struct stat* removed = nullptr) {
    struct stat named{};
    if (::fstatat(dir_fd, filename.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        !same_file(named, expected) || !S_ISREG(named.st_mode) || named.st_uid != ::geteuid()) {
        return false;
    }
    // Move the entry aside before verifying its identity for unlink.
    const auto staging = ".tokspur-discard-" + std::to_string(::getpid()) + "-" + std::to_string(::arc4random());
    if (::mkdirat(dir_fd, staging.c_str(), 0700) != 0) return false;
    bool discarded = false;
    {
        const Fd stage{::openat(dir_fd, staging.c_str(), O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW | O_RDONLY)};
        struct stat stage_stat{};
        if (stage.value >= 0 && ::fstat(stage.value, &stage_stat) == 0 &&
            stage_stat.st_uid == ::geteuid() && (stage_stat.st_mode & 0777) == 0700 &&
            ::renameatx_np(dir_fd, filename.c_str(), stage.value, "entry", RENAME_EXCL) == 0) {
            struct stat captured{};
            if (::fstatat(stage.value, "entry", &captured, AT_SYMLINK_NOFOLLOW) == 0 &&
                same_file(captured, expected) && S_ISREG(captured.st_mode) && captured.st_uid == ::geteuid() &&
                ::unlinkat(stage.value, "entry", 0) == 0) {
                if (removed) *removed = captured;
                discarded = true;
            }
            if (!discarded) {
                // Keep the captured file if another writer has already filled the name.
                ::renameatx_np(stage.value, "entry", dir_fd, filename.c_str(), RENAME_EXCL);
            }
        }
    }
    ::unlinkat(dir_fd, staging.c_str(), AT_REMOVEDIR);
    return discarded;
}

} // namespace

SsdCheckpointManager::SsdCheckpointManager(const std::filesystem::path& state_dir,
                                           uint64_t budget_bytes,
                                           uint64_t floor_bytes,
                                           std::string eviction_policy)
    : checkpoint_dir_(state_dir / "tokspur-checkpoints-v3"),
      budget_bytes_(budget_bytes),
      floor_bytes_(floor_bytes),
      eviction_policy_(std::move(eviction_policy)) {
    if (eviction_policy_ != "recompute-cost" && eviction_policy_ != "lru") {
        throw std::invalid_argument("checkpoint eviction policy must be recompute-cost or lru");
    }
    if (budget_bytes_ > maximum_checkpoint_ssd_budget) {
        budget_bytes_ = maximum_checkpoint_ssd_budget;
    }
    if (budget_bytes_ > 0) {
        std::error_code path_error;
        const auto parent_status = std::filesystem::symlink_status(state_dir, path_error);
        if (std::filesystem::is_symlink(parent_status)) throw std::runtime_error("checkpoint parent must not be a symlink");
        if (path_error && path_error != std::errc::no_such_file_or_directory) throw std::runtime_error("cannot inspect checkpoint parent");
        open_secure_directory();
        if (!reconcile_directory(true)) {
            ::close(dir_fd_);
            dir_fd_ = -1;
            throw std::runtime_error("cannot scan checkpoint directory");
        }
        int rc = posix_memalign(reinterpret_cast<void**>(&staging_slab_ptr_), 16384, staging_slab_bytes);
        if (rc != 0 || !staging_slab_ptr_) {
            if (dir_fd_ >= 0) {
                ::close(dir_fd_);
                dir_fd_ = -1;
            }
            throw std::runtime_error("failed to allocate 64 MiB staging slab buffer");
        }
        std::memset(staging_slab_ptr_, 0, staging_slab_bytes);
    }
}

SsdCheckpointManager::~SsdCheckpointManager() {
    if (staging_slab_ptr_) {
        free(staging_slab_ptr_);
        staging_slab_ptr_ = nullptr;
    }
    if (dir_fd_ >= 0) {
        ::close(dir_fd_);
        dir_fd_ = -1;
    }
}

void SsdCheckpointManager::open_secure_directory() {
    const std::string dir_str = checkpoint_dir_.string();
    struct stat st{};
    if (::stat(dir_str.c_str(), &st) != 0) {
        if (errno == ENOENT) {
            std::filesystem::create_directories(checkpoint_dir_.parent_path());
            if (::mkdir(dir_str.c_str(), 0700) != 0 && errno != EEXIST) {
                throw std::runtime_error("failed to create checkpoint directory with 0700 permissions: " + dir_str);
            }
        } else {
            throw std::runtime_error("cannot stat checkpoint directory: " + dir_str);
        }
    }

    dir_fd_ = ::open(dir_str.c_str(), O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW | O_RDONLY);
    if (dir_fd_ < 0) {
        throw std::runtime_error("failed to securely open checkpoint directory: " + dir_str);
    }

    if (::fstat(dir_fd_, &st) != 0) {
        ::close(dir_fd_);
        dir_fd_ = -1;
        throw std::runtime_error("failed to fstat checkpoint directory descriptor");
    }

    if (!S_ISDIR(st.st_mode)) {
        ::close(dir_fd_);
        dir_fd_ = -1;
        throw std::runtime_error("checkpoint path is not a directory: " + dir_str);
    }

    if (st.st_uid != ::geteuid()) {
        ::close(dir_fd_); dir_fd_ = -1;
        throw std::runtime_error("checkpoint directory owner does not match effective UID");
    }
    if ((st.st_mode & 0777) != 0700) {
        if (::fchmod(dir_fd_, 0700) != 0 || ::fstat(dir_fd_, &st) != 0 || (st.st_mode & 0777) != 0700) {
            ::close(dir_fd_);
            dir_fd_ = -1;
            throw std::runtime_error("checkpoint directory permissions must be 0700, got: " +
                                     std::to_string(st.st_mode & 0777));
        }
    }

}

bool SsdCheckpointManager::reconcile_directory(bool initialise_epoch) noexcept {
    uint64_t occupied = 0;
    uint64_t newest_epoch = 0;
    if (dir_fd_ < 0) return false;

    DIR* directory = open_directory_scan(dir_fd_);
    if (!directory) return false;

    bool complete = true;
    for (;;) {
        errno = 0;
        const auto* entry = ::readdir(directory);
        if (!entry) {
            complete = errno == 0;
            break;
        }
        const std::string_view name = entry->d_name;
        if (name.size() <= 5 || !name.ends_with(".ckpt")) continue;
        struct stat st{};
        if (::fstatat(dir_fd_, entry->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
            if (errno == ENOENT) continue;
            complete = false;
            break;
        }
        if (!S_ISREG(st.st_mode) || st.st_size < 0) continue;
        occupied = saturating_add_u64(occupied, uint64_t(st.st_size));
        if (initialise_epoch) {
            const Fd fd{::openat(dir_fd_, entry->d_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
            struct stat opened{};
            CheckpointLayout layout{};
            if (fd.value >= 0 && ::fstat(fd.value, &opened) == 0 && same_file(opened, st) &&
                read_checkpoint_layout(fd.value, opened, layout)) {
                newest_epoch = std::max(newest_epoch, layout.header.access_epoch);
            }
        }
    }
    ::closedir(directory);
    if (!complete) return false;
    cached_occupied_bytes_ = occupied;
    if (initialise_epoch) {
        global_access_epoch_.store(saturating_add_u64(newest_epoch, 1), std::memory_order_relaxed);
    }
    return true;
}

bool SsdCheckpointManager::take_access_epoch(uint64_t& value) noexcept {
    auto next = global_access_epoch_.load(std::memory_order_relaxed);
    while (next != std::numeric_limits<uint64_t>::max()) {
        if (global_access_epoch_.compare_exchange_weak(next, next + 1, std::memory_order_relaxed)) {
            value = next;
            return true;
        }
    }
    return false;
}

bool SsdCheckpointManager::enabled() const noexcept {
    return dir_fd_ >= 0 && budget_bytes_ > 0 && staging_slab_ptr_ != nullptr;
}

uint64_t SsdCheckpointManager::budget_bytes() const noexcept {
    return budget_bytes_;
}

uint64_t SsdCheckpointManager::occupied_bytes() const noexcept {
    return cached_occupied_bytes_;
}

uint64_t SsdCheckpointManager::floor_bytes() const noexcept {
    return floor_bytes_;
}

int SsdCheckpointManager::dir_fd() const noexcept {
    return dir_fd_;
}

const std::string& SsdCheckpointManager::eviction_policy() const noexcept {
    return eviction_policy_;
}

bool SsdCheckpointManager::has_checkpoint(const std::string& filename) const noexcept {
    if (!enabled() || filename.empty()) return false;
    for (char c : filename) {
        if (c == '/' || c == '\0') return false;
    }
    const Fd fd{::openat(dir_fd_, filename.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
    struct stat st{};
    CheckpointLayout layout{};
    return fd.value >= 0 && ::fstat(fd.value, &st) == 0 && read_checkpoint_layout(fd.value, st, layout) &&
           layout.header.model_execution_digest == execution_identity_digest_;
}

void SsdCheckpointManager::set_execution_identity(const std::array<uint8_t, 32>& digest) noexcept {
    execution_identity_digest_ = digest;
}

const std::array<uint8_t, 32>& SsdCheckpointManager::execution_identity() const noexcept {
    return execution_identity_digest_;
}

void SsdCheckpointManager::set_free_space_query(FreeSpaceQuery query) {
    free_space_query_ = std::move(query);
}

bool SsdCheckpointManager::query_available_space(uint64_t& out_free_bytes) const {
    if (dir_fd_ < 0) return false;
    if (free_space_query_) {
        return free_space_query_(dir_fd_, out_free_bytes);
    }
    return query_free_bytes(dir_fd_, out_free_bytes);
}

std::span<std::byte> SsdCheckpointManager::staging_slab() noexcept {
    if (!staging_slab_ptr_) return {};
    return {staging_slab_ptr_, staging_slab_bytes};
}

bool SsdCheckpointManager::should_coalesce(double elapsed_seconds, uint32_t newly_committed_tokens) const noexcept {
    return elapsed_seconds >= 5.0 || newly_committed_tokens >= 256;
}

bool SsdCheckpointManager::admit_spill(uint64_t incoming_bytes) {
    if (!enabled() || incoming_bytes > budget_bytes_) return false;

    uint64_t initial_free_bytes = 0;
    if (!query_available_space(initial_free_bytes) || !reconcile_directory()) {
        return false;
    }

    if (!has_budget_headroom(cached_occupied_bytes_, budget_bytes_, incoming_bytes) ||
        !has_floor_headroom(initial_free_bytes, floor_bytes_, incoming_bytes)) {
        const uint64_t effective_free = evict_for_headroom(incoming_bytes, initial_free_bytes);
        if (!reconcile_directory() || !has_budget_headroom(cached_occupied_bytes_, budget_bytes_, incoming_bytes) ||
            !has_floor_headroom(effective_free, floor_bytes_, incoming_bytes)) {
            return false;
        }
    }

    return true;
}

std::vector<SsdCheckpointManager::EntryInfo> SsdCheckpointManager::scan_entries(bool include_incompatible) const {
    std::vector<EntryInfo> entries;
    if (dir_fd_ < 0) return entries;

    const std::unique_ptr<DIR, decltype(&::closedir)> directory(open_directory_scan(dir_fd_), ::closedir);
    if (!directory) return entries;
    uint64_t occupied = 0;
    for (;;) {
        errno = 0;
        const auto* entry = ::readdir(directory.get());
        if (!entry) {
            if (errno != 0) return {};
            break;
        }
        const std::string_view name = entry->d_name;
        if (name.size() <= 5 || !name.ends_with(".ckpt")) continue;
        struct stat st{};
        if (::fstatat(dir_fd_, entry->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
            if (errno == ENOENT) continue;
            return {};
        }
        if (!S_ISREG(st.st_mode) || st.st_size < 0) continue;
        occupied = saturating_add_u64(occupied, uint64_t(st.st_size));

        const Fd fd{::openat(dir_fd_, entry->d_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
        struct stat opened{};
        CheckpointLayout layout{};
        const bool valid = fd.value >= 0 && ::fstat(fd.value, &opened) == 0 && same_file(st, opened) &&
            read_checkpoint_layout(fd.value, opened, layout) &&
            layout.header.model_execution_digest == execution_identity_digest_;
        if (!valid && !include_incompatible) continue;

        EntryInfo info;
        info.filename = name;
        info.file_bytes = uint64_t(st.st_size);
        info.device = uint64_t(st.st_dev);
        info.inode = uint64_t(st.st_ino);
        info.birth_time = st.st_birthtimespec;
        info.priority = -1.0;
        if (valid) {
            info.prefix_tokens = layout.header.prefix_tokens;
            info.access_epoch = layout.header.access_epoch;
            info.is_system_anchor = layout.header.is_system_anchor != 0;
            info.priority = eviction_policy_ == "lru" ? double(info.access_epoch) :
                compute_eviction_priority(info.access_epoch, info.prefix_tokens, info.file_bytes, info.is_system_anchor);
        }
        entries.push_back(std::move(info));
    }
    cached_occupied_bytes_ = occupied;
    return entries;
}

std::vector<SsdCheckpointManager::EntryInfo> SsdCheckpointManager::list_entries() const {
    return scan_entries(false);
}

uint64_t SsdCheckpointManager::evict_for_headroom(uint64_t incoming_bytes, uint64_t initial_free_bytes) {
    if (!enabled() || incoming_bytes > budget_bytes_) return 0;

    uint64_t fresh_free_bytes = initial_free_bytes;
    uint64_t reclaimed_bytes = 0;

    auto entries = scan_entries(true);
    std::sort(entries.begin(), entries.end(), [](const EntryInfo& a, const EntryInfo& b) {
        return a.priority < b.priority;
    });

    for (const auto& entry : entries) {
        const uint64_t effective_free =
            effective_free_after_unlinks(initial_free_bytes, fresh_free_bytes, reclaimed_bytes);
        if (has_budget_headroom(cached_occupied_bytes_, budget_bytes_, incoming_bytes) &&
            has_floor_headroom(effective_free, floor_bytes_, incoming_bytes)) {
            return effective_free;
        }
        struct stat removed{};
        bool discarded = false;
        {
            const Fd candidate{::openat(dir_fd_, entry.filename.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
            struct stat st{};
            if (candidate.value >= 0 && ::fstat(candidate.value, &st) == 0 &&
                uint64_t(st.st_dev) == entry.device && uint64_t(st.st_ino) == entry.inode &&
                st.st_birthtimespec.tv_sec == entry.birth_time.tv_sec &&
                st.st_birthtimespec.tv_nsec == entry.birth_time.tv_nsec) {
                discarded = discard_observed_file(dir_fd_, entry.filename, st, &removed);
            }
        }
        if (!discarded) {
            if (!reconcile_directory()) return 0;
            continue;
        }
        cached_occupied_bytes_ -= std::min(cached_occupied_bytes_, entry.file_bytes);
        if (removed.st_nlink == 1 && removed.st_size > 0) {
            reclaimed_bytes = saturating_add_u64(reclaimed_bytes, uint64_t(removed.st_size));
        }
        if (!query_available_space(fresh_free_bytes)) return 0;
    }

    return effective_free_after_unlinks(initial_free_bytes, fresh_free_bytes, reclaimed_bytes);
}

void SsdCheckpointManager::run_eviction_if_needed(uint64_t incoming_bytes) {
    if (!enabled() || incoming_bytes > budget_bytes_) return;
    uint64_t initial_free_bytes = 0;
    if (!query_available_space(initial_free_bytes) || !reconcile_directory()) return;
    static_cast<void>(evict_for_headroom(incoming_bytes, initial_free_bytes));
    static_cast<void>(reconcile_directory());
}

bool SsdCheckpointManager::remove_checkpoint(const std::string& filename) {
    validate_filename(filename);
    if (dir_fd_ < 0) return false;

    struct stat st{};
    if (::fstatat(dir_fd_, filename.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0) {
        if (::unlinkat(dir_fd_, filename.c_str(), 0) == 0) {
            if (cached_occupied_bytes_ >= uint64_t(st.st_size)) {
                cached_occupied_bytes_ -= uint64_t(st.st_size);
            }
            return true;
        }
    }
    return false;
}

bool SsdCheckpointManager::spill_checkpoint(
    const std::string& filename,
    SlotHandshake& handshake,
    CheckpointFileHeader header,
    const CheckpointMetadata& metadata,
    const std::function<bool(size_t chunk_idx, std::span<std::byte> target)>& read_chunk_fn) {

    validate_filename(filename);
    if (!enabled()) return false;

    // Serialize metadata into an in-memory buffer
    std::vector<uint8_t> meta_buf;
    // 1. SamplingRng: seed (uint64_t), words (uint64_t), engine (sstream)
    {
        uint64_t seed = metadata.rng.seed();
        uint64_t words = metadata.rng.words_consumed();
        std::ostringstream ss;
        ss << metadata.rng.engine();
        std::string eng_str = ss.str();
        uint32_t eng_len = uint32_t(eng_str.size());

        auto append_bytes = [&](const void* ptr, size_t sz) {
            const auto* b = reinterpret_cast<const uint8_t*>(ptr);
            meta_buf.insert(meta_buf.end(), b, b + sz);
        };
        append_bytes(&seed, sizeof(seed));
        append_bytes(&words, sizeof(words));
        append_bytes(&eng_len, sizeof(eng_len));
        append_bytes(eng_str.data(), eng_str.size());

        // 2. StageAControllerState
        append_bytes(&metadata.controller_state.ewma_alpha1, sizeof(metadata.controller_state.ewma_alpha1));
        append_bytes(&metadata.controller_state.current_mode, sizeof(metadata.controller_state.current_mode));
        append_bytes(&metadata.controller_state.consecutive_ordinary_steps, sizeof(metadata.controller_state.consecutive_ordinary_steps));
        append_bytes(metadata.controller_state.controller_policy_digest.data(), 32);

        // 3. MtpHead frontier & anchor carry
        append_bytes(&metadata.mtp_first, sizeof(metadata.mtp_first));
        append_bytes(&metadata.mtp_rows, sizeof(metadata.mtp_rows));
        uint8_t has_anchor = metadata.carried_anchor.has_value() ? 1 : 0;
        append_bytes(&has_anchor, 1);
        if (has_anchor) {
            uint32_t aid = metadata.carried_anchor->id;
            float ascore = metadata.carried_anchor->source_logit;
            uint8_t aacc = metadata.carried_anchor->accepted_proposal ? 1 : 0;
            append_bytes(&aid, sizeof(aid));
            append_bytes(&ascore, sizeof(ascore));
            append_bytes(&aacc, sizeof(aacc));
        }
        append_bytes(&metadata.pending_overflow, sizeof(metadata.pending_overflow));

        // 4. Logits
        uint32_t vocab_size = uint32_t(metadata.logits.size());
        append_bytes(&vocab_size, sizeof(vocab_size));
        if (vocab_size > 0) {
            append_bytes(metadata.logits.data(), vocab_size * sizeof(float));
        }
    }

    if (meta_buf.size() > max_checkpoint_metadata_bytes || header.total_payload_bytes > maximum_checkpoint_ssd_budget) {
        return false;
    }

    const uint32_t meta_size = uint32_t(meta_buf.size());
    const uint64_t total_payload = header.total_payload_bytes;
    const uint32_t chunk_count = uint32_t((total_payload + checkpoint_chunk_bytes - 1) / checkpoint_chunk_bytes);

    const uint64_t total_file_bytes = sizeof(CheckpointFileHeader) +
                                      uint64_t(chunk_count) * 32ull +
                                      sizeof(uint32_t) + meta_size +
                                      total_payload;

    uint64_t access_epoch = 0;
    if (!take_access_epoch(access_epoch) || !admit_spill(total_file_bytes)) {
        return false;
    }

    header.version = checkpoint_format_version;
    header.chunk_size = uint32_t(checkpoint_chunk_bytes);
    header.chunk_count = chunk_count;
    header.access_epoch = access_epoch;
    header.controller_policy_digest = metadata.controller_state.controller_policy_digest;
    header.model_execution_digest = execution_identity_digest_;
    header.metadata_sha256 = sha256_buffer(meta_buf.data(), meta_buf.size());

    const std::string tmp_name = "." + filename + "." + std::to_string(::getpid()) + ".tmp";
    int fd = ::openat(dir_fd_, tmp_name.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        return false;
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0 || (st.st_mode & 0777) != 0600 || st.st_uid != ::geteuid()) {
        ::close(fd);
        ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
        return false;
    }

    std::vector<std::array<uint8_t, 32>> manifest(chunk_count);
    uint64_t payload_offset = sizeof(CheckpointFileHeader) +
                              uint64_t(chunk_count) * 32ull +
                              sizeof(uint32_t) + meta_size;

    // Stream 16 MiB chunks from memory to disk under reader_epoch lock
    for (size_t i = 0; i < chunk_count; ++i) {
        if (handshake.cancel_spill.load(std::memory_order_acquire)) {
            ::close(fd);
            ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
            return false;
        }

        const size_t this_chunk_bytes = std::min<size_t>(checkpoint_chunk_bytes,
                                                          total_payload - i * checkpoint_chunk_bytes);
        auto slab_slice = staging_slab().subspan(0, this_chunk_bytes);

        // Reader epoch handshake
        handshake.reader_epoch.fetch_add(1, std::memory_order_acq_rel);
        bool read_ok = read_chunk_fn(i, slab_slice);
        handshake.reader_epoch.fetch_add(1, std::memory_order_acq_rel);

        if (!read_ok) {
            ::close(fd);
            ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
            return false;
        }

        manifest[i] = sha256_buffer(slab_slice.data(), this_chunk_bytes);

        const off_t target_pos = off_t(payload_offset + i * checkpoint_chunk_bytes);
        ssize_t written = ::pwrite(fd, slab_slice.data(), this_chunk_bytes, target_pos);
        if (written != ssize_t(this_chunk_bytes)) {
            ::close(fd);
            ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
            return false;
        }
    }

    // Compute header SHA-256
    header.header_sha256.fill(0);
    header.header_sha256 = sha256_buffer(&header, sizeof(header) - sizeof(header.header_sha256));

    // Write header, manifest, and metadata
    if (::pwrite(fd, &header, sizeof(header), 0) != sizeof(header)) {
        ::close(fd);
        ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
        return false;
    }

    off_t cur_pos = sizeof(header);
    if (!manifest.empty()) {
        const size_t manifest_bytes = manifest.size() * 32;
        if (::pwrite(fd, manifest.data(), manifest_bytes, cur_pos) != ssize_t(manifest_bytes)) {
            ::close(fd);
            ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
            return false;
        }
        cur_pos += off_t(manifest_bytes);
    }

    if (::pwrite(fd, &meta_size, sizeof(meta_size), cur_pos) != sizeof(meta_size)) {
        ::close(fd);
        ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
        return false;
    }
    cur_pos += sizeof(meta_size);

    if (meta_size > 0) {
        if (::pwrite(fd, meta_buf.data(), meta_size, cur_pos) != ssize_t(meta_size)) {
            ::close(fd);
            ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
            return false;
        }
    }

    // Two-phase commit durability: fsync(fd) + atomic rename + fsync(dir_fd)
    if (::fsync(fd) != 0) {
        ::close(fd);
        ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
        return false;
    }
    ::close(fd);

#ifdef __APPLE__
    int rename_rc = ::renameatx_np(dir_fd_, tmp_name.c_str(), dir_fd_, filename.c_str(), RENAME_EXCL);
    if (rename_rc != 0 && errno == EEXIST) {
        // If file exists, deduct existing file size, unlink existing and retry
        struct stat existing_st{};
        if (::fstatat(dir_fd_, filename.c_str(), &existing_st, AT_SYMLINK_NOFOLLOW) == 0) {
            const uint64_t existing_bytes = uint64_t(existing_st.st_size);
            if (cached_occupied_bytes_ >= existing_bytes) {
                cached_occupied_bytes_ -= existing_bytes;
            } else {
                cached_occupied_bytes_ = 0;
            }
        }
        ::unlinkat(dir_fd_, filename.c_str(), 0);
        rename_rc = ::renameatx_np(dir_fd_, tmp_name.c_str(), dir_fd_, filename.c_str(), RENAME_EXCL);
    }
#else
    int rename_rc = ::renameat(dir_fd_, tmp_name.c_str(), dir_fd_, filename.c_str());
#endif

    if (rename_rc != 0) {
        ::unlinkat(dir_fd_, tmp_name.c_str(), 0);
        return false;
    }

    ::fsync(dir_fd_);
    cached_occupied_bytes_ = saturating_add_u64(cached_occupied_bytes_, total_file_bytes);
    return true;
}

bool SsdCheckpointManager::restore_checkpoint(
    const std::string& filename,
    CheckpointFileHeader& out_header,
    CheckpointMetadata& out_metadata,
    const std::function<bool(size_t chunk_idx, std::span<const std::byte> source)>& write_chunk_fn,
    const std::function<bool(const CheckpointFileHeader&, const CheckpointMetadata&)>& validate_fn) {

    validate_filename(filename);
    if (!enabled()) return false;

    const Fd file{::openat(dir_fd_, filename.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
    const int fd = file.value;
    struct stat st{};
    const auto refuse = [&](bool discard = false) {
        if (discard) discard_observed_file(dir_fd_, filename, st);
        static_cast<void>(reconcile_directory());
        return false;
    };
    if (fd < 0 || ::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        (st.st_mode & 0777) != 0600 || st.st_uid != ::geteuid() || st.st_nlink != 1) {
        return refuse();
    }

    CheckpointLayout layout{};
    if (!read_checkpoint_layout(fd, st, layout) ||
        layout.header.model_execution_digest != execution_identity_digest_) {
        return refuse();
    }
    auto& header = layout.header;
    const uint32_t chunk_count = header.chunk_count;
    const uint64_t payload_offset = layout.payload_offset;
    const uint64_t total_payload = header.total_payload_bytes;
    std::vector<std::array<uint8_t, 32>> manifest(chunk_count);
    const size_t manifest_bytes = size_t(chunk_count) * 32u;
    if (manifest_bytes > 0 &&
        ::pread(fd, manifest.data(), manifest_bytes, sizeof(header)) != ssize_t(manifest_bytes)) {
        return refuse();
    }

    std::vector<uint8_t> meta_buf(layout.metadata_bytes);
    if (::pread(fd, meta_buf.data(), meta_buf.size(), layout.metadata_offset) != ssize_t(meta_buf.size())) {
        return refuse();
    }
    if (sha256_buffer(meta_buf.data(), meta_buf.size()) != header.metadata_sha256) {
        return refuse(true);
    }

    CheckpointMetadata parsed_meta{};
    try {
        size_t offset = 0;
        auto read_bytes = [&](void* ptr, size_t sz) {
            if (sz > meta_buf.size() - offset) throw std::runtime_error("corrupt metadata buffer");
            std::memcpy(ptr, meta_buf.data() + offset, sz);
            offset += sz;
        };

        uint64_t seed = 0, words = 0;
        uint32_t eng_len = 0;
        read_bytes(&seed, sizeof(seed));
        read_bytes(&words, sizeof(words));
        read_bytes(&eng_len, sizeof(eng_len));
        if (eng_len > meta_buf.size() - offset) return refuse();
        std::string eng_str(reinterpret_cast<const char*>(meta_buf.data() + offset), eng_len);
        offset += eng_len;
        std::istringstream ss(eng_str);
        std::mt19937_64 engine;
        ss >> engine;
        if (ss.fail()) return refuse();
        parsed_meta.rng = SamplingRng(seed, words, engine);

        read_bytes(&parsed_meta.controller_state.ewma_alpha1, sizeof(parsed_meta.controller_state.ewma_alpha1));
        read_bytes(&parsed_meta.controller_state.current_mode, sizeof(parsed_meta.controller_state.current_mode));
        read_bytes(&parsed_meta.controller_state.consecutive_ordinary_steps,
                   sizeof(parsed_meta.controller_state.consecutive_ordinary_steps));
        read_bytes(parsed_meta.controller_state.controller_policy_digest.data(), 32);
        if (parsed_meta.controller_state.controller_policy_digest != header.controller_policy_digest) {
            return refuse();
        }

        read_bytes(&parsed_meta.mtp_first, sizeof(parsed_meta.mtp_first));
        read_bytes(&parsed_meta.mtp_rows, sizeof(parsed_meta.mtp_rows));
        uint8_t has_anchor = 0;
        read_bytes(&has_anchor, 1);
        if (has_anchor) {
            uint32_t aid = 0;
            float ascore = 0.0f;
            uint8_t aacc = 0;
            read_bytes(&aid, sizeof(aid));
            read_bytes(&ascore, sizeof(ascore));
            read_bytes(&aacc, sizeof(aacc));
            parsed_meta.carried_anchor = SampledPathToken{aid, ascore, aacc != 0};
        } else {
            parsed_meta.carried_anchor.reset();
        }
        read_bytes(&parsed_meta.pending_overflow, sizeof(parsed_meta.pending_overflow));

        uint32_t vocab_size = 0;
        read_bytes(&vocab_size, sizeof(vocab_size));
        if (uint64_t(vocab_size) * sizeof(float) != meta_buf.size() - offset) {
            return refuse();
        }
        parsed_meta.logits.resize(vocab_size);
        if (vocab_size > 0) {
            read_bytes(parsed_meta.logits.data(), size_t(vocab_size) * sizeof(float));
        }
        if (offset != meta_buf.size()) return refuse();
    } catch (...) {
        return refuse();
    }

    if (validate_fn && !validate_fn(header, parsed_meta)) {
        return refuse();
    }

    // Pass 1: read and SHA-256 verify every chunk on disk before publishing any state.
    const bool fits_in_slab = total_payload <= staging_slab_bytes;
    for (size_t i = 0; i < chunk_count; ++i) {
        const size_t this_chunk_bytes = std::min<size_t>(checkpoint_chunk_bytes,
                                                         total_payload - i * checkpoint_chunk_bytes);
        const size_t slab_offset = fits_in_slab ? (i * checkpoint_chunk_bytes) : 0u;
        auto slab_slice = staging_slab().subspan(slab_offset, this_chunk_bytes);

        const off_t src_pos = off_t(payload_offset + i * checkpoint_chunk_bytes);
        const ssize_t read_bytes_count = ::pread(fd, slab_slice.data(), this_chunk_bytes, src_pos);
        if (read_bytes_count != ssize_t(this_chunk_bytes)) return refuse();

        const auto actual_chunk_sha = sha256_buffer(slab_slice.data(), this_chunk_bytes);
        if (actual_chunk_sha != manifest[i]) return refuse(true);
    }

    uint64_t access_epoch = 0;
    if (!take_access_epoch(access_epoch)) return refuse();
    out_header = header;
    out_metadata = std::move(parsed_meta);

    // Pass 2: publish verified chunks to the caller.
    for (size_t i = 0; i < chunk_count; ++i) {
        const size_t this_chunk_bytes = std::min<size_t>(checkpoint_chunk_bytes,
                                                         total_payload - i * checkpoint_chunk_bytes);
        std::span<const std::byte> verified_slice;
        if (fits_in_slab) {
            verified_slice = staging_slab().subspan(i * checkpoint_chunk_bytes, this_chunk_bytes);
        } else {
            auto slab_slice = staging_slab().subspan(0, this_chunk_bytes);
            const off_t src_pos = off_t(payload_offset + i * checkpoint_chunk_bytes);
            const ssize_t read_bytes_count = ::pread(fd, slab_slice.data(), this_chunk_bytes, src_pos);
            if (read_bytes_count != ssize_t(this_chunk_bytes)) return refuse();
            if (sha256_buffer(slab_slice.data(), this_chunk_bytes) != manifest[i]) {
                return refuse(true);
            }
            verified_slice = slab_slice;
        }

        if (!write_chunk_fn(i, verified_slice)) return refuse();
    }

    struct stat current{};
    if (::fstat(fd, &current) != 0 || current.st_size != st.st_size ||
        (current.st_mode & 0777) != 0600 || current.st_uid != ::geteuid() || current.st_nlink != 1) {
        return refuse();
    }
    header.access_epoch = access_epoch;
    header.header_sha256 = sha256_buffer(&header, sizeof(header) - sizeof(header.header_sha256));
    if (::pwrite(fd, &header, sizeof(header), 0) != sizeof(header) || ::fsync(fd) != 0) {
        return refuse(true);
    }
    out_header = header;
    return true;
}

} // namespace tokspur
