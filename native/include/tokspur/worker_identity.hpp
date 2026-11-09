#pragma once
#include "tokspur/gguf.hpp"
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tokspur {
inline constexpr std::string_view scalar_worker_name = "scalar-channel-target";

inline MappedFile execution_identity_artifact(const std::filesystem::path& path) {
    return MappedFile(path, ReadOnlyMapMode::private_copy_on_write);
}

constexpr bool live_worker_candidate(int64_t pid, int64_t self, bool zombie) noexcept {
    return pid > 0 && pid != self && !zombie;
}

// The caller supplies the actual SDK MAXCOMLEN, never a guessed short prefix.
constexpr bool scalar_kernel_worker_name(std::string_view name, size_t maxcomlen) noexcept {
    return maxcomlen && name == scalar_worker_name.substr(0, maxcomlen);
}
} // namespace tokspur
