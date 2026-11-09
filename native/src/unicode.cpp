#include "tokspur/unicode.hpp"
#include "tokspur/gguf.hpp"
#include "../data/unicode9.inc"

#include <algorithm>
#include <iterator>

namespace tokspur {
namespace {

constexpr uint32_t hangul_start = 0xac00;
constexpr uint32_t hangul_count = 19 * 21 * 28;
constexpr size_t work_limit = 8 * 1024 * 1024;

uint8_t combining(uint32_t point) {
    const auto first = std::begin(unicode9_data::combining);
    const auto last = std::end(unicode9_data::combining);
    const auto found = std::lower_bound(first, last, point, [](const auto& range, uint32_t c) { return range.last < c; });
    return found != last && found->first <= point ? found->value : 0;
}

void decompose(uint32_t point, std::vector<uint32_t>& output, unsigned depth = 0) {
    if (depth > 32 || output.size() >= work_limit) throw FormatError("Unicode normalization work limit exceeded");
    if (point >= hangul_start && point < hangul_start + hangul_count) {
        const auto index = point - hangul_start;
        output.push_back(0x1100 + index / (21 * 28));
        output.push_back(0x1161 + (index / 28) % 21);
        if (index % 28) output.push_back(0x11a7 + index % 28);
        return;
    }
    const auto first = std::begin(unicode9_data::decomposition);
    const auto last = std::end(unicode9_data::decomposition);
    const auto found = std::lower_bound(first, last, point, [](const auto& item, uint32_t c) { return item.point < c; });
    if (found == last || found->point != point) { output.push_back(point); return; }
    decompose(found->first, output, depth + 1);
    if (found->second) decompose(found->second, output, depth + 1);
}

uint32_t compose(uint32_t first_point, uint32_t second_point) {
    if (first_point >= 0x1100 && first_point < 0x1100 + 19 && second_point >= 0x1161 && second_point < 0x1161 + 21) {
        return hangul_start + ((first_point - 0x1100) * 21 + second_point - 0x1161) * 28;
    }
    if (first_point >= hangul_start && first_point < hangul_start + hangul_count && (first_point - hangul_start) % 28 == 0 &&
        second_point > 0x11a7 && second_point < 0x11a7 + 28) return first_point + second_point - 0x11a7;
    const uint64_t key = (uint64_t(first_point) << 21) | second_point;
    const auto first = std::begin(unicode9_data::composition);
    const auto last = std::end(unicode9_data::composition);
    const auto found = std::lower_bound(first, last, key, [](const auto& item, uint64_t pair) { return item.pair < pair; });
    return found != last && found->pair == key ? found->point : 0;
}

} // namespace

std::vector<uint32_t> unicode_scalars(std::string_view utf8) {
    if (utf8.size() > work_limit) throw FormatError("Unicode input exceeds limit");
    std::vector<uint32_t> output;
    output.reserve(utf8.size());
    for (size_t i = 0; i < utf8.size();) {
        const auto first = static_cast<unsigned char>(utf8[i++]);
        if (first < 128) { output.push_back(first); continue; }
        const size_t tail = first >= 0xc2 && first <= 0xdf ? 1 :
                            (first >= 0xe0 && first <= 0xef ? 2 : (first >= 0xf0 && first <= 0xf4 ? 3 : 0));
        if (!tail || tail > utf8.size() - i) throw FormatError("invalid UTF-8 input");
        uint32_t point = first & ((1u << (6 - tail)) - 1);
        for (size_t n = 0; n < tail; ++n) {
            const auto next = static_cast<unsigned char>(utf8[i++]);
            if ((next & 0xc0) != 0x80) throw FormatError("invalid UTF-8 continuation");
            point = (point << 6) | (next & 63);
        }
        if ((tail == 1 && point < 0x80) || (tail == 2 && point < 0x800) || (tail == 3 && point < 0x10000) ||
            point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) throw FormatError("non-scalar or overlong UTF-8");
        output.push_back(point);
    }
    return output;
}

std::string scalar_utf8(uint32_t point) {
    if (point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) throw FormatError("invalid Unicode scalar");
    std::string output;
    if (point < 0x80) output.push_back(static_cast<char>(point));
    else {
        const size_t tail = point < 0x800 ? 1 : (point < 0x10000 ? 2 : 3);
        const auto prefix = tail == 1 ? 0xc0 : (tail == 2 ? 0xe0 : 0xf0);
        output.push_back(static_cast<char>(prefix | (point >> (tail * 6))));
        for (size_t remaining = tail; remaining; --remaining) output.push_back(static_cast<char>(0x80 | ((point >> ((remaining - 1) * 6)) & 63)));
    }
    return output;
}

std::string normalize_nfc9(std::string_view utf8) {
    const auto points = unicode_scalars(utf8);
    std::vector<uint32_t> decomposed;
    decomposed.reserve(points.size());
    for (const auto point : points) decompose(point, decomposed);
    // Stable sorting per non-starter run avoids quadratic insertion sorting
    // on adversarial long sequences of combining marks.
    for (size_t cursor = 0; cursor < decomposed.size();) {
        if (combining(decomposed[cursor]) == 0) ++cursor;
        const auto start = cursor;
        while (cursor < decomposed.size() && combining(decomposed[cursor])) ++cursor;
        std::stable_sort(decomposed.begin() + static_cast<ptrdiff_t>(start), decomposed.begin() + static_cast<ptrdiff_t>(cursor),
                         [](uint32_t a, uint32_t b) { return combining(a) < combining(b); });
    }
    std::vector<uint32_t> composed;
    composed.reserve(decomposed.size());
    size_t starter = work_limit;
    uint8_t previous = 0;
    for (const auto point : decomposed) {
        const auto current = combining(point);
        const auto combination = starter != work_limit && (previous < current || previous == 0) ? compose(composed[starter], point) : 0;
        if (combination) composed[starter] = combination;
        else {
            if (!current) starter = composed.size();
            composed.push_back(point);
            previous = current;
        }
    }
    std::string output;
    for (const auto point : composed) {
        const auto bytes = scalar_utf8(point);
        if (bytes.size() > work_limit - output.size()) throw FormatError("normalized UTF-8 exceeds limit");
        output += bytes;
    }
    return output;
}

} // namespace tokspur
