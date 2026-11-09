#include "tokspur/logit_trace.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace tokspur {
namespace {
void require(bool okay, const char* why) { if (!okay) throw FormatError(why); }
void write_all(int fd, const void* bytes, size_t size) {
    auto data = static_cast<const char*>(bytes);
    while (size) {
        const auto count = ::write(fd, data, size);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "private logit trace write failed");
        data += count; size -= size_t(count);
    }
}
int fresh(int directory, const char* name) {
    const int fd = openat(directory, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    require(fd >= 0, "private logit trace refuses existing or unsafe output"); return fd;
}
}

LogitTrace::LogitTrace(const std::filesystem::path& directory, uint32_t rows, uint32_t vocabulary)
    : limit_(rows), vocabulary_(vocabulary) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 && std::endian::native == std::endian::little);
    require(rows && rows <= 1024 && vocabulary && vocabulary <= 248320, "private logit trace shape/row bound");
    require(directory.is_absolute() && std::filesystem::canonical(directory) == directory.lexically_normal() &&
        std::filesystem::is_empty(directory), "private logit trace requires fresh canonical absolute directory");
    try {
        directory_ = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        require(directory_ >= 0, "private logit trace directory open failed");
        struct stat state{};
        require(fstat(directory_, &state) == 0 && S_ISDIR(state.st_mode) && state.st_uid == geteuid() &&
            (state.st_mode & 07777) == 0700, "private logit trace directory must be owner-only0700");
        manifest_ = fresh(directory_, "manifest.jsonl");
    } catch (...) {
        if (manifest_ >= 0) close(manifest_);
        if (directory_ >= 0) close(directory_);
        throw;
    }
}

LogitTrace::~LogitTrace() noexcept {
    if (manifest_ >= 0) close(manifest_);
    if (directory_ >= 0) close(directory_);
}

void LogitTrace::append(uint32_t position, std::span<const float> logits) {
    require(!failed_ && rows_ < limit_, "private logit trace failed or exhausted; no retry");
    require(position && position <= 65536 && logits.size() == vocabulary_ &&
        std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }),
        "private logit trace requires complete finite committed vocabulary");
    const auto filename = "logits-" + std::to_string(rows_) + ".f32";
    int fd = -1;
    try {
        fd = fresh(directory_, filename.c_str());
        write_all(fd, logits.data(), logits.size_bytes());
        const auto closed = close(fd); fd = -1;
        require(closed == 0, "private logit trace row close failed");
        const auto line = Json(Json::Object{{"schema", Json::integer(1)}, {"index", Json::integer(rows_)},
            {"position", Json::integer(position)}, {"vocabulary", Json::integer(vocabulary_)},
            {"bytes", Json::integer(logits.size_bytes())}, {"format", Json("ieee-f32-le")}, {"file", Json(filename)}}).dump() + "\n";
        write_all(manifest_, line.data(), line.size());
        ++rows_;
    } catch (...) {
        failed_ = true;
        if (fd >= 0) close(fd);
        throw; // Preserve partial/orphan files as a failed diagnostic, never overwrite.
    }
}

} // namespace tokspur
