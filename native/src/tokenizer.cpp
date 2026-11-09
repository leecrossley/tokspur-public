#include "tokspur/tokenizer.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>

namespace tokspur {
namespace {

constexpr size_t input_limit = 8 * 1024 * 1024;
constexpr size_t piece_limit = 1024 * 1024;
constexpr size_t token_limit = 1'000'000;
constexpr uint32_t absent = std::numeric_limits<uint32_t>::max();

std::array<uint16_t, 256> alphabet() {
    std::array<uint16_t, 256> result{};
    uint16_t extension = 256;
    for (uint16_t byte = 0; byte < 256; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        result[byte] = visible ? byte : extension++;
    }
    return result;
}

const auto codepoints = alphabet();
const auto reverse_codepoints = [] {
    std::array<int16_t, 324> result{};
    result.fill(-1);
    for (size_t i = 0; i < codepoints.size(); ++i) result[codepoints[i]] = static_cast<int16_t>(i);
    return result;
}();

uint64_t pair_key(uint32_t left, uint32_t right) { return (uint64_t(left) << 32) | right; }

void append_ids(std::vector<uint32_t>& output, const std::vector<uint32_t>& more) {
    if (more.size() > token_limit - output.size()) throw FormatError("tokenizer output limit exceeded");
    output.insert(output.end(), more.begin(), more.end());
}

} // namespace

std::string byte_level_encode(std::string_view bytes) {
    std::string result;
    result.reserve(bytes.size() * 2);
    for (const unsigned char byte : bytes) {
        const auto point = codepoints[byte];
        if (point < 128) result.push_back(static_cast<char>(point));
        else {
            result.push_back(static_cast<char>(0xc0 | (point >> 6)));
            result.push_back(static_cast<char>(0x80 | (point & 63)));
        }
    }
    return result;
}

std::string byte_level_decode(std::string_view spelling) {
    std::string result;
    result.reserve(spelling.size());
    for (size_t i = 0; i < spelling.size();) {
        uint32_t point = static_cast<unsigned char>(spelling[i++]);
        if (point >= 128) {
            if (point < 0xc2 || point > 0xc5 || i == spelling.size()) throw FormatError("invalid byte-alphabet spelling");
            const auto continuation = static_cast<unsigned char>(spelling[i++]);
            if ((continuation & 0xc0) != 0x80) throw FormatError("invalid byte-alphabet continuation");
            point = ((point & 31) << 6) | (continuation & 63);
        }
        if (point >= reverse_codepoints.size() || reverse_codepoints[point] < 0) throw FormatError("character outside byte alphabet");
        result.push_back(static_cast<char>(reverse_codepoints[point]));
    }
    return result;
}

std::string replace_invalid_utf8(std::string_view bytes) {
    std::string result;
    result.reserve(bytes.size());
    for (size_t i = 0; i < bytes.size();) {
        const auto first = static_cast<unsigned char>(bytes[i]);
        if (first < 128) { result.push_back(bytes[i++]); continue; }
        const size_t count = first >= 0xc2 && first <= 0xdf ? 2 :
                             (first >= 0xe0 && first <= 0xef ? 3 : (first >= 0xf0 && first <= 0xf4 ? 4 : 0));
        size_t valid_prefix = 1;
        if (count) {
            while (valid_prefix < count && i + valid_prefix < bytes.size()) {
                const auto next = static_cast<unsigned char>(bytes[i + valid_prefix]);
                const bool continuation = next >= 0x80 && next <= 0xbf;
                const bool boundary = valid_prefix != 1 ||
                    !((first == 0xe0 && next < 0xa0) || (first == 0xed && next > 0x9f) ||
                      (first == 0xf0 && next < 0x90) || (first == 0xf4 && next > 0x8f));
                if (!continuation || !boundary) break;
                ++valid_prefix;
            }
        }
        if (count && valid_prefix == count) result.append(bytes.substr(i, count));
        else result.append("\xef\xbf\xbd");
        i += valid_prefix;
    }
    return result;
}

ByteBpe::ByteBpe(BpeData data) : vocabulary_(std::move(data.vocabulary)), special_(vocabulary_.size(), false), markers_(1) {
    if (vocabulary_.size() < 256 || vocabulary_.size() > token_limit || data.merges.size() > token_limit || data.added.size() > 4096) {
        throw FormatError("unsupported BPE vocabulary size");
    }
    std::set<uint32_t> added_ids;
    for (const auto& marker : data.added) {
        if (marker.id >= vocabulary_.size() || marker.content.empty() || marker.content.size() > 4096 ||
            vocabulary_[marker.id] != marker.content || !added_ids.insert(marker.id).second) throw FormatError("invalid added token");
        special_[marker.id] = marker.special;
        uint32_t node = 0;
        for (const unsigned char c : marker.content) {
            auto found = markers_[node].next.find(c);
            if (found == markers_[node].next.end()) {
                if (markers_.size() >= 65536) throw FormatError("added token trie exceeds limit");
                const auto next = static_cast<uint32_t>(markers_.size());
                markers_[node].next.emplace(c, next);
                markers_.emplace_back();
                node = next;
            } else node = found->second;
        }
        if (markers_[node].id >= 0) throw FormatError("duplicate added token content");
        markers_[node].id = marker.id;
    }
    std::unordered_map<std::string, uint32_t> ids;
    ids.reserve(vocabulary_.size());
    uint64_t total_bytes = 0;
    for (size_t i = 0; i < vocabulary_.size(); ++i) {
        const auto& token = vocabulary_[i];
        total_bytes = checked_add(total_bytes, token.size());
        if (token.empty() || token.size() > 4096 || total_bytes > 64 * 1024 * 1024) throw FormatError("BPE vocabulary byte limit exceeded");
        if (!added_ids.contains(static_cast<uint32_t>(i)) && !ids.emplace(token, static_cast<uint32_t>(i)).second) {
            throw FormatError("duplicate ordinary BPE token");
        }
    }
    for (size_t i = 0; i < 256; ++i) {
        const auto found = ids.find(std::string(1, static_cast<char>(i)));
        if (found == ids.end()) throw FormatError("BPE vocabulary lacks a byte singleton");
        single_[i] = found->second;
    }
    merges_.reserve(data.merges.size());
    for (size_t rank = 0; rank < data.merges.size(); ++rank) {
        const auto& [left, right] = data.merges[rank];
        const auto l = ids.find(left);
        const auto r = ids.find(right);
        if (left.empty() || right.empty() || l == ids.end() || r == ids.end() || left.size() + right.size() > 4096) {
            throw FormatError("invalid BPE merge parts");
        }
        const auto output = ids.find(left + right);
        if (output == ids.end() || !merges_.emplace(pair_key(l->second, r->second), Merge{static_cast<uint32_t>(rank), output->second}).second) {
            throw FormatError("invalid or duplicate BPE merge");
        }
    }
}

std::vector<uint32_t> ByteBpe::encode_piece(std::string_view bytes) const {
    if (bytes.size() > piece_limit) throw FormatError("BPE piece exceeds limit");
    if (bytes.empty()) return {};
    struct Node { uint32_t id; uint32_t previous; uint32_t next; uint32_t version = 0; bool alive = true; };
    struct Candidate { uint32_t rank; uint32_t left; uint32_t right; uint32_t left_version; uint32_t right_version; uint32_t output; };
    struct Later {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.rank != b.rank ? a.rank > b.rank : a.left > b.left;
        }
    };
    std::vector<Node> nodes;
    nodes.reserve(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) {
        nodes.push_back({single_[static_cast<unsigned char>(bytes[i])], i ? static_cast<uint32_t>(i - 1) : absent,
                         i + 1 < bytes.size() ? static_cast<uint32_t>(i + 1) : absent});
    }
    std::priority_queue<Candidate, std::vector<Candidate>, Later> queue;
    const auto consider = [&](uint32_t left) {
        if (left == absent || nodes[left].next == absent) return;
        const auto right = nodes[left].next;
        const auto found = merges_.find(pair_key(nodes[left].id, nodes[right].id));
        if (found == merges_.end()) return;
        queue.push({found->second.rank, left, right, nodes[left].version, nodes[right].version, found->second.output});
    };
    for (uint32_t i = 0; i + 1 < nodes.size(); ++i) consider(i);
    while (!queue.empty()) {
        const auto candidate = queue.top();
        queue.pop();
        auto& left = nodes[candidate.left];
        auto& right = nodes[candidate.right];
        if (!left.alive || !right.alive || left.next != candidate.right ||
            left.version != candidate.left_version || right.version != candidate.right_version) continue;
        left.id = candidate.output;
        left.next = right.next;
        ++left.version;
        right.alive = false;
        ++right.version;
        if (left.next != absent) nodes[left.next].previous = candidate.left;
        consider(left.previous);
        consider(candidate.left);
    }
    std::vector<uint32_t> output;
    for (uint32_t i = 0; i != absent; i = nodes[i].next) {
        if (output.size() >= token_limit) throw FormatError("BPE output exceeds limit");
        output.push_back(nodes[i].id);
    }
    return output;
}

std::vector<uint32_t> ByteBpe::encode(std::string_view text, const TextProcessor& processor,
    std::span<const std::pair<size_t, size_t>> literal_ranges) const {
    if (text.size() > input_limit) throw FormatError("tokenizer input exceeds limit");
    size_t last = 0;
    for (const auto& [start, end] : literal_ranges) {
        if (start < last || end <= start || end > text.size()) throw FormatError("invalid tokenizer literal spans");
        last = end;
    }
    std::vector<uint32_t> output;
    const auto ordinary = [&](std::string_view segment) {
        if (segment.empty()) return;
        size_t normalized_bytes = 0;
        for (const auto& piece : processor.pieces(segment)) {
            normalized_bytes += piece.size();
            if (normalized_bytes > input_limit) throw FormatError("normalized segment exceeds limit");
            append_ids(output, encode_piece(piece));
        }
    };
    size_t plain_start = 0, literal = 0;
    for (size_t i = 0; i < text.size();) {
        uint32_t node = 0;
        int64_t match = -1;
        size_t end = i;
        for (size_t cursor = i; cursor < text.size(); ++cursor) {
            const auto next = markers_[node].next.find(static_cast<unsigned char>(text[cursor]));
            if (next == markers_[node].next.end()) break;
            node = next->second;
            if (markers_[node].id >= 0) { match = markers_[node].id; end = cursor + 1; }
        }
        while (literal < literal_ranges.size() && literal_ranges[literal].second <= i) ++literal;
        if (match < 0 || (literal < literal_ranges.size() && literal_ranges[literal].first < end)) { ++i; continue; }
        ordinary(text.substr(plain_start, i - plain_start));
        if (output.size() >= token_limit) throw FormatError("tokenizer output exceeds limit");
        output.push_back(static_cast<uint32_t>(match));
        i = end;
        plain_start = i;
    }
    ordinary(text.substr(plain_start));
    return output;
}

std::string ByteBpe::decode_bytes(std::span<const uint32_t> ids, bool skip_special) const {
    if (ids.size() > token_limit) throw FormatError("decoder token limit exceeded");
    std::string output;
    for (const auto id : ids) {
        if (id >= vocabulary_.size()) throw FormatError("token ID outside vocabulary");
        if (skip_special && special_[id]) continue;
        if (vocabulary_[id].size() > input_limit - output.size()) throw FormatError("decoded byte limit exceeded");
        output.append(vocabulary_[id]);
    }
    return output;
}

std::string ByteBpe::decode_text(std::span<const uint32_t> ids, bool skip_special) const {
    return replace_invalid_utf8(decode_bytes(ids, skip_special));
}

} // namespace tokspur
