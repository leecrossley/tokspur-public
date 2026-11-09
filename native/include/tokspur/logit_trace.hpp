#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace tokspur {

// Explicit private semantic diagnostic. No model operation or GPU allocation.
// One owner, <=1024 finite IEEE-f32 little-endian vectors, fresh0700 directory.
class LogitTrace {
    int directory_ = -1, manifest_ = -1;
    uint32_t limit_ = 0, vocabulary_ = 0, rows_ = 0;
    bool failed_ = false;
public:
    LogitTrace(const std::filesystem::path& directory, uint32_t rows, uint32_t vocabulary = 248320);
    ~LogitTrace() noexcept;
    LogitTrace(const LogitTrace&) = delete;
    LogitTrace& operator=(const LogitTrace&) = delete;
    void append(uint32_t position, std::span<const float> logits);
    [[nodiscard]] uint32_t rows() const { return rows_; }
    [[nodiscard]] uint32_t limit() const { return limit_; }
    [[nodiscard]] uint64_t bytes() const { return uint64_t(rows_) * vocabulary_ * sizeof(float); }
};

} // namespace tokspur
