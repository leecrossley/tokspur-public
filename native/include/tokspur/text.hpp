#pragma once

#include "tokspur/tokenizer.hpp"

#include <filesystem>
#include <memory>

namespace tokspur {
class ModelFiles;

// macOS adapter: only Foundation/Apple system APIs, never tokenizer-engine code.
class NativeText {
public:
    NativeText(const std::filesystem::path& tokenizer_json, std::string_view expected_sha256);
    ~NativeText();
    NativeText(const NativeText&) = delete;
    NativeText& operator=(const NativeText&) = delete;
    [[nodiscard]] std::vector<uint32_t> encode(std::string_view text,
        std::span<const std::pair<size_t, size_t>> literal_ranges = {}) const;
    // Exact represented bytes, including incomplete/invalid UTF-8 and special
    // markers. Response framing must not silently replace bytes before parsing.
    [[nodiscard]] std::string decode_bytes(std::span<const uint32_t> ids, bool skip_special = false) const;
    [[nodiscard]] std::string decode(std::span<const uint32_t> ids, bool skip_special = false) const;
    [[nodiscard]] size_t vocabulary_size() const;
    [[nodiscard]] size_t merge_count() const;
    void verify_gguf(const ModelFiles& model) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int text_cli(int argc, char** argv);
int response_cli(int argc, char** argv);

} // namespace tokspur
