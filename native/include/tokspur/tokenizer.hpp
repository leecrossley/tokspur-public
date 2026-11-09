#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokspur {

// The byte alphabet is format data. Model vocabularies use its visible Unicode
// spelling; the merge engine itself works on the represented bytes.
std::string byte_level_encode(std::string_view bytes);
std::string byte_level_decode(std::string_view spelling);
std::string replace_invalid_utf8(std::string_view bytes);

struct AddedToken {
    uint32_t id;
    std::string content;
    bool special;
};

struct BpeData {
    std::vector<std::string> vocabulary; // Indexed by ID, already decoded to bytes.
    std::vector<std::pair<std::string, std::string>> merges; // Ordered by rank.
    std::vector<AddedToken> added;
};

class TextProcessor {
public:
    virtual ~TextProcessor() = default;
    // Normalize and split one segment that contains no recognized added token.
    virtual std::vector<std::string> pieces(std::string_view segment) const = 0;
};

class ByteBpe {
public:
    explicit ByteBpe(BpeData data);
    [[nodiscard]] std::vector<uint32_t> encode_piece(std::string_view bytes) const;
    [[nodiscard]] std::vector<uint32_t> encode(std::string_view text, const TextProcessor& processor,
        std::span<const std::pair<size_t, size_t>> literal_ranges = {}) const;
    [[nodiscard]] std::string decode_bytes(std::span<const uint32_t> ids, bool skip_special = false) const;
    [[nodiscard]] std::string decode_text(std::span<const uint32_t> ids, bool skip_special = false) const;
    [[nodiscard]] size_t vocabulary_size() const { return vocabulary_.size(); }
    [[nodiscard]] size_t merge_count() const { return merges_.size(); }

private:
    struct Merge { uint32_t rank; uint32_t output; };
    struct MarkerNode {
        std::unordered_map<uint8_t, uint32_t> next;
        int64_t id = -1;
    };
    std::vector<std::string> vocabulary_;
    std::vector<bool> special_;
    std::array<uint32_t, 256> single_{};
    std::unordered_map<uint64_t, Merge> merges_;
    std::vector<MarkerNode> markers_;
};

} // namespace tokspur
