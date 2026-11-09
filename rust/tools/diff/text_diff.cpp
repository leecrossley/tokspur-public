// Differential driver for the Rust text port. It links native/src/unicode.cpp, native/src/tokenizer.cpp and native/src/gguf.cpp (tokenizer.cpp calls its checked_add) unchanged, reads the cases that rust/crates/tokspur-text/tests/cpp_differential.rs generates on stdin and writes one result line per case on stdout. text.sh builds and runs it.
//
// Case lines (fields separated by one space):
//   nfc B | scalars B | scalar D | ble B | bld B | riu B | piece B
//   encode PROCESSOR B | decode S I | text S I | sha B C
//   model V M A, then V lines "B", M lines "B B" and A lines "D S B"
// B: bytes, one or more "x<hex>[*<repeat>]" segments joined by '+'. I: IDs, one or more "i<decimal>,...[*<repeat>]" segments joined by '+'. C: "c<decimal>,..." chunk sizes for incremental hashing. S: 0 or 1. Results: "ok <payload>", "err <FormatError::what()>", "nomodel" or "exception <what()>" for any other exception from the code under test.

#include "tokspur/gguf.hpp"
#include "tokspur/tokenizer.hpp"
#include "tokspur/unicode.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr auto npos = std::string_view::npos;

// A malformed case file is a harness error, never a FormatError, so it cannot pass for a refusal by the code under test.
class CaseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<std::string_view> split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    while (true) {
        const auto end = text.find(separator, start);
        parts.push_back(text.substr(start, end == npos ? npos : end - start));
        if (end == npos) return parts;
        start = end + 1;
    }
}

uint64_t decimal(std::string_view text) {
    uint64_t value = 0;
    const auto* last = text.data() + text.size();
    const auto result = std::from_chars(text.data(), last, value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != last) throw CaseError("invalid decimal field");
    return value;
}

unsigned hex_digit(char c) {
    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
    throw CaseError("invalid hexadecimal digit");
}

std::vector<uint64_t> decimals(std::string_view body) {
    std::vector<uint64_t> values;
    if (body.empty()) return values;
    for (const auto part : split(body, ',')) values.push_back(decimal(part));
    return values;
}

// "<prefix><body>[*<repeat>]".
std::pair<std::string_view, uint64_t> repeated(std::string_view field, char prefix) {
    if (field.empty() || field[0] != prefix) throw CaseError("unexpected field kind");
    field.remove_prefix(1);
    uint64_t count = 1;
    if (const auto star = field.find('*'); star != npos) {
        count = decimal(field.substr(star + 1));
        field = field.substr(0, star);
    }
    return {field, count};
}

std::string bytes_field(std::string_view field) {
    std::string output;
    for (const auto segment : split(field, '+')) {
        const auto [body, count] = repeated(segment, 'x');
        if (body.size() % 2) throw CaseError("odd hexadecimal length");
        std::string unit;
        for (size_t i = 0; i < body.size(); i += 2) {
            unit.push_back(static_cast<char>(hex_digit(body[i]) * 16 + hex_digit(body[i + 1])));
        }
        output.reserve(output.size() + unit.size() * count);
        for (uint64_t i = 0; i < count; ++i) output += unit;
    }
    return output;
}

std::vector<uint32_t> ids_field(std::string_view field) {
    std::vector<uint32_t> output;
    for (const auto segment : split(field, '+')) {
        const auto [body, count] = repeated(segment, 'i');
        std::vector<uint32_t> unit;
        for (const auto value : decimals(body)) {
            if (value > std::numeric_limits<uint32_t>::max()) throw CaseError("token ID exceeds uint32_t");
            unit.push_back(static_cast<uint32_t>(value));
        }
        output.reserve(output.size() + unit.size() * count);
        for (uint64_t i = 0; i < count; ++i) output.insert(output.end(), unit.begin(), unit.end());
    }
    return output;
}

bool flag(std::string_view field) {
    if (field == "0") return false;
    if (field == "1") return true;
    throw CaseError("invalid flag");
}

std::string hex(std::string_view bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output = "x";
    output.reserve(1 + bytes.size() * 2);
    for (const unsigned char byte : bytes) {
        output.push_back(digits[byte >> 4]);
        output.push_back(digits[byte & 15]);
    }
    return output;
}

template <typename Values>
std::string ids_text(const Values& ids) {
    std::string output = "i";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) output.push_back(',');
        output += std::to_string(ids[i]);
    }
    return output;
}

std::string digest_text(const std::array<unsigned char, CC_SHA256_DIGEST_LENGTH>& digest) {
    std::string spelled = hex(std::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
    return spelled.substr(1);
}

// Test processors; tests/cpp_differential.rs implements the same ones.
class Processor final : public tokspur::TextProcessor {
public:
    explicit Processor(std::string_view spec) {
        const auto number = [&](std::string_view prefix) { return decimal(spec.substr(prefix.size())); };
        if (spec == "identity") kind_ = Kind::identity;
        else if (spec == "bytes") kind_ = Kind::bytes;
        else if (spec == "split") kind_ = Kind::split;
        else if (spec == "empties") kind_ = Kind::empties;
        else if (spec == "nfc") kind_ = Kind::nfc;
        else if (spec.starts_with("copies")) { kind_ = Kind::copies; count_ = number("copies"); }
        else if (spec.starts_with("grow")) { kind_ = Kind::grow; count_ = number("grow"); }
        else if (spec.starts_with("refuse") && spec.size() == 8) {
            kind_ = Kind::refuse;
            byte_ = static_cast<char>(hex_digit(spec[6]) * 16 + hex_digit(spec[7]));
        } else throw CaseError("unknown processor");
    }

    std::vector<std::string> pieces(std::string_view segment) const override {
        switch (kind_) {
        case Kind::identity: return {std::string(segment)};
        case Kind::bytes: {
            std::vector<std::string> output;
            for (const char c : segment) output.emplace_back(1, c);
            return output;
        }
        case Kind::split: {
            // A new piece starts at every space after the first byte.
            std::vector<std::string> output;
            size_t start = 0;
            for (size_t i = 1; i < segment.size(); ++i) {
                if (segment[i] == ' ') {
                    output.emplace_back(segment.substr(start, i - start));
                    start = i;
                }
            }
            output.emplace_back(segment.substr(start));
            return output;
        }
        case Kind::empties: return {std::string(), std::string(segment), std::string()};
        case Kind::copies: return std::vector<std::string>(count_, std::string(segment));
        case Kind::grow: {
            std::string piece;
            piece.reserve(segment.size() * count_);
            for (uint64_t i = 0; i < count_; ++i) piece += segment;
            return {piece};
        }
        case Kind::refuse:
            if (segment.find(byte_) != npos) throw tokspur::FormatError("text processor refused segment");
            return {std::string(segment)};
        case Kind::nfc: return {tokspur::normalize_nfc9(segment)};
        }
        throw CaseError("unknown processor");
    }

private:
    enum class Kind { identity, bytes, split, empties, copies, grow, refuse, nfc };
    Kind kind_ = Kind::identity;
    uint64_t count_ = 0;
    char byte_ = 0;
};

void want(const std::vector<std::string_view>& fields, size_t count) {
    if (fields.size() != count) throw CaseError("wrong number of fields");
}

std::string sha(std::string_view bytes, std::string_view chunks) {
    if (chunks.empty() || chunks[0] != 'c') throw CaseError("expected chunk sizes");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> once{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), once.data());
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    size_t offset = 0;
    for (const auto size : decimals(chunks.substr(1))) {
        const auto take = static_cast<size_t>(std::min<uint64_t>(size, bytes.size() - offset));
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(take));
        offset += take;
    }
    CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(bytes.size() - offset));
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> incremental{};
    CC_SHA256_Final(incremental.data(), &context);
    return "ok " + digest_text(once) + " " + digest_text(incremental);
}

std::string evaluate(const std::vector<std::string_view>& fields, const tokspur::ByteBpe* model) {
    const auto op = fields[0];
    try {
        if (op == "nfc") { want(fields, 2); return "ok " + hex(tokspur::normalize_nfc9(bytes_field(fields[1]))); }
        if (op == "scalars") { want(fields, 2); return "ok " + ids_text(tokspur::unicode_scalars(bytes_field(fields[1]))); }
        if (op == "scalar") {
            want(fields, 2);
            const auto value = decimal(fields[1]);
            if (value > std::numeric_limits<uint32_t>::max()) throw CaseError("scalar exceeds uint32_t");
            return "ok " + hex(tokspur::scalar_utf8(static_cast<uint32_t>(value)));
        }
        if (op == "ble") { want(fields, 2); return "ok " + hex(tokspur::byte_level_encode(bytes_field(fields[1]))); }
        if (op == "bld") { want(fields, 2); return "ok " + hex(tokspur::byte_level_decode(bytes_field(fields[1]))); }
        if (op == "riu") { want(fields, 2); return "ok " + hex(tokspur::replace_invalid_utf8(bytes_field(fields[1]))); }
        if (op == "sha") { want(fields, 3); return sha(bytes_field(fields[1]), fields[2]); }
        if (op == "piece" || op == "encode" || op == "decode" || op == "text") {
            if (!model) return "nomodel";
            if (op == "piece") { want(fields, 2); return "ok " + ids_text(model->encode_piece(bytes_field(fields[1]))); }
            if (op == "encode") {
                want(fields, 3);
                const Processor processor(fields[1]);
                return "ok " + ids_text(model->encode(bytes_field(fields[2]), processor));
            }
            want(fields, 3);
            const auto skip = flag(fields[1]);
            const auto ids = ids_field(fields[2]);
            if (op == "decode") return "ok " + hex(model->decode_bytes(ids, skip));
            return "ok " + hex(model->decode_text(ids, skip));
        }
        throw CaseError("unknown operation");
    } catch (const tokspur::FormatError& error) {
        return std::string("err ") + error.what();
    } catch (const CaseError&) {
        throw;
    } catch (const std::exception& error) {
        return std::string("exception ") + error.what();
    }
}

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::unique_ptr<tokspur::ByteBpe> model;
    std::string line;
    size_t number = 0;
    const auto next = [&]() -> std::string_view {
        if (!std::getline(std::cin, line)) throw CaseError("truncated model block");
        ++number;
        return line;
    };
    try {
        while (std::getline(std::cin, line)) {
            ++number;
            const auto fields = split(line, ' ');
            if (fields[0] != "model") {
                std::cout << evaluate(fields, model.get()) << '\n';
                continue;
            }
            want(fields, 4);
            const auto vocabulary = decimal(fields[1]);
            const auto merges = decimal(fields[2]);
            const auto added = decimal(fields[3]);
            tokspur::BpeData data;
            for (uint64_t i = 0; i < vocabulary; ++i) data.vocabulary.push_back(bytes_field(next()));
            for (uint64_t i = 0; i < merges; ++i) {
                const auto parts = split(next(), ' ');
                want(parts, 2);
                data.merges.emplace_back(bytes_field(parts[0]), bytes_field(parts[1]));
            }
            for (uint64_t i = 0; i < added; ++i) {
                const auto parts = split(next(), ' ');
                want(parts, 3);
                const auto id = decimal(parts[0]);
                if (id > std::numeric_limits<uint32_t>::max()) throw CaseError("token ID exceeds uint32_t");
                data.added.push_back({static_cast<uint32_t>(id), bytes_field(parts[2]), flag(parts[1])});
            }
            try {
                auto built = std::make_unique<tokspur::ByteBpe>(std::move(data));
                std::cout << "ok " << built->vocabulary_size() << ' ' << built->merge_count() << '\n';
                model = std::move(built);
            } catch (const tokspur::FormatError& error) {
                std::cout << "err " << error.what() << '\n';
            } catch (const std::exception& error) {
                std::cout << "exception " << error.what() << '\n';
            }
        }
    } catch (const CaseError& error) {
        std::cerr << "text-diff: case line " << number << ": " << error.what() << '\n';
        return 2;
    }
    std::cout.flush();
    return std::cout ? 0 : 1;
}
