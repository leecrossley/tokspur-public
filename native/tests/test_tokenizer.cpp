#include "tokspur/tokenizer.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <random>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid tokenizer input was accepted");
}

class Identity final : public tokspur::TextProcessor {
public:
    std::vector<std::string> pieces(std::string_view segment) const override { return {std::string(segment)}; }
};

tokspur::BpeData fixture() {
    tokspur::BpeData data;
    for (unsigned i = 0; i < 256; ++i) data.vocabulary.emplace_back(1, static_cast<char>(i));
    for (const auto* word : {"ab", "aa", "aba", "abab", "aaaa", " a", "<X>", "<X>!", "<Y>", "zz"}) data.vocabulary.emplace_back(word);
    data.merges = {{"a", "b"}, {"a", "a"}, {"ab", "a"}, {"ab", "ab"}, {"aa", "aa"}, {" ", "a"}};
    data.added = {{262, "<X>", true}, {263, "<X>!", true}, {264, "<Y>", false}};
    return data;
}

std::vector<uint32_t> slow(const tokspur::BpeData& data, std::string_view input) {
    // Recompute every neighboring pair at every step. Deliberately separate
    // from the production linked nodes, heap, IDs, versions and trie.
    std::vector<std::string> words;
    for (char c : input) words.emplace_back(1, c);
    while (words.size() > 1) {
        size_t best_rank = data.merges.size();
        size_t best_index = words.size();
        for (size_t i = 0; i + 1 < words.size(); ++i) {
            for (size_t rank = 0; rank < best_rank; ++rank) {
                if (data.merges[rank].first == words[i] && data.merges[rank].second == words[i + 1]) {
                    best_rank = rank;
                    best_index = i;
                }
            }
        }
        if (best_index == words.size()) break;
        words[best_index] += words[best_index + 1];
        words.erase(words.begin() + static_cast<ptrdiff_t>(best_index + 1));
    }
    std::vector<uint32_t> ids;
    for (const auto& word : words) {
        const auto found = std::find(data.vocabulary.begin(), data.vocabulary.end(), word);
        if (found == data.vocabulary.end()) throw std::runtime_error("invalid oracle fixture");
        ids.push_back(static_cast<uint32_t>(found - data.vocabulary.begin()));
    }
    return ids;
}

void alphabet() {
    std::string every;
    for (unsigned i = 0; i < 256; ++i) {
        const std::string byte(1, static_cast<char>(i));
        every += byte;
        require(tokspur::byte_level_decode(tokspur::byte_level_encode(byte)) == byte, "byte alphabet round trip");
    }
    require(tokspur::byte_level_decode(tokspur::byte_level_encode(every)) == every, "all-byte alphabet round trip");
    require(tokspur::byte_level_encode(std::string_view(" \n\0", 3)) == "ĠĊĀ", "byte alphabet fixed values");
    require(tokspur::byte_level_decode("ĠHello") == " Hello", "encoded space");
    for (const auto& invalid : {std::string(" "), std::string("🦉"), std::string("\xc0\x80", 2), std::string("\xc4", 1), std::string("\xc4\x20", 2)}) {
        rejected([&] { static_cast<void>(tokspur::byte_level_decode(invalid)); });
    }
}

void merging() {
    const auto data = fixture();
    tokspur::ByteBpe model(data);
    require(model.vocabulary_size() == 266 && model.merge_count() == 6, "BPE geometry");
    require(model.encode_piece("").empty(), "empty BPE piece");
    require(model.encode_piece("abab") == std::vector<uint32_t>({259}), "ranked composition");
    require(model.encode_piece("aaaaa") == std::vector<uint32_t>({260, 97}), "leftmost overlapping merge");
    require(model.encode_piece("zz") == std::vector<uint32_t>({122, 122}), "ignore_merges must stay false");
    Identity identity;
    const auto ids = model.encode("ab<X>!aa<Y>", identity);
    require(ids == std::vector<uint32_t>({256, 263, 257, 264}), "longest added marker");
    require(model.decode_bytes(ids) == "ab<X>!aa<Y>", "added token round trip");
    require(model.decode_bytes(ids, true) == "abaa<Y>", "special and ordinary added tokens differ");
    require(model.encode_piece("<X>") == std::vector<uint32_t>({60, 88, 62}), "piece encoding must not recognize added tokens");
    require(model.encode("<X", identity) == std::vector<uint32_t>({60, 88}), "partial marker");
    require(model.encode("<X><X>", identity) == std::vector<uint32_t>({262, 262}), "adjacent markers");
    std::mt19937 random(0x62706531);
    for (size_t run = 0; run < 5000; ++run) {
        std::string input;
        const size_t count = random() % 129;
        for (size_t i = 0; i < count; ++i) input.push_back(run % 2 ? "ab "[random() % 3] : static_cast<char>(random() & 255));
        const auto actual = model.encode_piece(input);
        require(actual == slow(data, input), "heap BPE differs from exhaustive rank oracle");
        require(model.decode_bytes(actual) == input, "BPE byte round trip");
    }
    rejected([&] { static_cast<void>(model.encode_piece(std::string(1024 * 1024 + 1, 'a'))); });
    rejected([&] { static_cast<void>(model.encode(std::string(8 * 1024 * 1024 + 1, 'a'), identity)); });
    const std::array<uint32_t, 1> invalid{266};
    rejected([&] { static_cast<void>(model.decode_bytes(invalid)); });
}

void invalid_models() {
    auto data = fixture();
    data.vocabulary.clear();
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.vocabulary[1] = data.vocabulary[0];
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.vocabulary[1] = "no-singleton";
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.merges.push_back(data.merges[0]);
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.merges.emplace_back("a", "z");
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.merges.emplace_back("", "a");
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.added[0].id = 0;
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.added.push_back(data.added[0]);
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.vocabulary[263] = "<X>";
    data.added[1].content = "<X>";
    rejected([&] { tokspur::ByteBpe bad(data); });
    data = fixture();
    data.vocabulary[265].resize(4097, 'x');
    rejected([&] { tokspur::ByteBpe bad(data); });
}

void lossy_utf8() {
    const std::string replacement = "\xef\xbf\xbd";
    for (const auto& value : {std::string(""), std::string("café🦉"), std::string("\0abc", 4)}) {
        require(tokspur::replace_invalid_utf8(value) == value, "valid UTF-8 changed");
    }
    require(tokspur::replace_invalid_utf8(std::string("\xc0\x80", 2)) == replacement + replacement, "overlong UTF-8");
    require(tokspur::replace_invalid_utf8(std::string("\xe1\x80", 2) + "A") == replacement + "A", "partial UTF-8 maximal subpart");
    require(tokspur::replace_invalid_utf8(std::string("\xf1\x80\x80", 3)) == replacement, "truncated UTF-8 maximal subpart");
    require(tokspur::replace_invalid_utf8(std::string("\xed\xa0\x80", 3)) == replacement + replacement + replacement, "UTF-8 surrogate");
    require(tokspur::replace_invalid_utf8(std::string("\xf4\x90\x80\x80", 4)) == replacement + replacement + replacement + replacement, "UTF-8 out of range");
}
} // namespace

int main() {
    try {
        alphabet();
        merging();
        invalid_models();
        lossy_utf8();
        std::cout << "Tokenizer core checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Tokenizer core check failed: " << error.what() << '\n';
        return 1;
    }
}
