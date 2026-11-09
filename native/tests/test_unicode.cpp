#include "tokspur/unicode.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <charconv>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid Unicode request accepted");
}

std::string from_hex(std::string_view value) {
    std::istringstream stream{std::string(value)};
    std::string hex;
    std::string output;
    while (stream >> hex) {
        uint32_t point = 0;
        const auto result = std::from_chars(hex.data(), hex.data() + hex.size(), point, 16);
        if (result.ec != std::errc{} || result.ptr != hex.data() + hex.size()) throw std::runtime_error("invalid normalization test scalar");
        output += tokspur::scalar_utf8(point);
    }
    return output;
}

size_t standard_tests(const char* path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open Unicode normalization tests");
    size_t cases = 0;
    std::string line;
    while (std::getline(file, line)) {
        line = line.substr(0, line.find('#'));
        if (line.empty() || line[0] == '@') continue;
        std::array<std::string, 5> columns;
        size_t cursor = 0;
        for (auto& column : columns) {
            const auto end = line.find(';', cursor);
            if (end == std::string::npos) throw std::runtime_error("invalid normalization test row");
            column = from_hex(std::string_view(line).substr(cursor, end - cursor));
            cursor = end + 1;
        }
        require(tokspur::normalize_nfc9(columns[0]) == columns[1], "UAX15 source -> NFC failed");
        require(tokspur::normalize_nfc9(columns[1]) == columns[1], "UAX15 NFC idempotence failed");
        require(tokspur::normalize_nfc9(columns[2]) == columns[1], "UAX15 NFD -> NFC failed");
        require(tokspur::normalize_nfc9(columns[3]) == columns[3], "UAX15 NFKC -> NFC failed");
        require(tokspur::normalize_nfc9(columns[4]) == columns[3], "UAX15 NFKD -> NFC failed");
        ++cases;
    }
    return cases;
}

void basic() {
    require(tokspur::normalize_nfc9("cafe\u0301") == "café", "NFC combining accent");
    require(tokspur::normalize_nfc9("\u212b\u2126") == "ÅΩ", "NFC singleton decompositions");
    require(tokspur::normalize_nfc9("① １") == "① １", "must not do compatibility normalization");
    require(tokspur::normalize_nfc9("\u0340\u0341\u0344") == "\u0300\u0301\u0308\u0301", "NFC excluded composites");
    require(tokspur::normalize_nfc9("\u0eb8\u0eba") == "\u0eb8\u0eba", "Unicode 9 unknown combining class");
    require(tokspur::normalize_nfc9("\U00010f48\U00010f4b") == "\U00010f48\U00010f4b", "do not inherit later Unicode classes");
    require(tokspur::normalize_nfc9("\u1100\u1161\u11a8") == "각", "Hangul composition");
    require(tokspur::normalize_nfc9("\u0301\u0323") == "\u0323\u0301", "leading nonstarter ordering");
    require(tokspur::normalize_nfc9(std::string("\0a\0", 3)) == std::string("\0a\0", 3), "embedded NUL");
    for (uint32_t point = 0; point < 0x110000; ++point) {
        if (point >= 0xd800 && point <= 0xdfff) continue;
        const auto bytes = tokspur::scalar_utf8(point);
        require(tokspur::unicode_scalars(bytes) == std::vector<uint32_t>({point}), "Unicode scalar UTF-8 round trip");
    }
    for (const auto& invalid : {std::string("\xc0\x80", 2), std::string("\xe0\x80\x80", 3), std::string("\xed\xa0\x80", 3),
                                std::string("\xf4\x90\x80\x80", 4), std::string("\xc2", 1), std::string("\x80", 1)}) {
        rejected([&] { static_cast<void>(tokspur::normalize_nfc9(invalid)); });
    }
    rejected([] { static_cast<void>(tokspur::scalar_utf8(0xd800)); });
    rejected([] { static_cast<void>(tokspur::scalar_utf8(0x110000)); });
    std::string long_marks = "a";
    for (size_t i = 0; i < 10000; ++i) long_marks += "\u0301\u0323";
    const auto normalized = tokspur::normalize_nfc9(long_marks);
    require(tokspur::normalize_nfc9(normalized) == normalized, "long combining sequence idempotence");
}
} // namespace

int main(int argc, char** argv) {
    try {
        basic();
        size_t cases = 0;
        if (argc == 2) cases = standard_tests(argv[1]);
        else if (argc != 1) throw std::runtime_error("expected zero or one normalization test path");
        std::cout << "Unicode checks passed: " << checks << ", normative rows: " << cases << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Unicode check failed: " << error.what() << '\n';
        return 1;
    }
}
