#include "tokspur/attention_math.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

namespace {
size_t checks = 0;
void require(bool b, const char* s) { ++checks; if (!b) throw std::runtime_error(s); }
void reject(const std::function<void()>& f) {
    ++checks; try { f(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid attention reference accepted");
}
}
int main() {
    try {
        using namespace tokspur;
        const QsaShape actual{2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536};
        // Independent boundary table: tokens2048 and2051 need no selection;
        // token2052 starts selection, then each merge depth changes separately.
        for (const auto [position, wanted] : std::array<std::pair<uint32_t, uint64_t>, 10>{{
            {0,18}, {3,21}, {2047,21}, {2050,18}, {2051,25}, {2052,22},
            {4095,25}, {4099,26}, {8195,27}, {65535,29}}})
            require(qsa_dispatches(actual, position) == wanted, "QSA context dispatch boundary");
        reject([&] { static_cast<void>(qsa_dispatches(actual, 65536)); });
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            const auto value = read_bf16(uint16_t(bits));
            if (std::isfinite(value)) require(finite_bf16(value) == bits, "exact BF16 value changed");
        }
        for (uint32_t top = 0x3f00; top < 0x4000; ++top) {
            const auto tie = std::bit_cast<float>((top << 16) | 0x8000u);
            require(finite_bf16(tie) == top + (top & 1), "BF16 halfway tie not nearest even");
            require(finite_bf16(std::nextafter(tie, 0.0f)) == top, "BF16 lower midpoint neighbor");
            require(finite_bf16(std::nextafter(tie, 2.0f)) == top + 1, "BF16 upper midpoint neighbor");
        }
        reject([] { static_cast<void>(finite_bf16(std::numeric_limits<float>::infinity())); });
        reject([] { static_cast<void>(finite_bf16(std::numeric_limits<float>::max())); });
        const std::array<double, 8> packed{1, 2, 3, 4, 5, 6, 7, 8};
        const auto split = split_query_reference(packed, 2, 2, QueryPacking::head_query_gate);
        require(split.query == std::vector<double>({1, 2, 5, 6}) && split.gate == std::vector<double>({3, 4, 7, 8}), "head-local query/gate packing");
        const auto flat = split_query_reference(packed, 2, 2, QueryPacking::all_queries_then_gates);
        require(flat.query == std::vector<double>({1, 2, 3, 4}) && flat.gate == std::vector<double>({5, 6, 7, 8}), "global query/gate packing");
        const auto freq = rotary_frequencies(4, 10000);
        require(freq == std::vector<float>({1, 0.01f}), "rotary frequency formula");
        require(rotary_reference(packed, 1, 8, 0, freq, RotaryPairs::split_half) == std::vector<double>(packed.begin(), packed.end()), "zero-position rotary changed data");
        for (const auto pairs : {RotaryPairs::split_half, RotaryPairs::adjacent}) {
            const auto rotated = rotary_reference(packed, 1, 8, 59000, freq, pairs);
            double before = 0, after = 0;
            for (size_t i = 0; i < 4; ++i) { before += packed[i] * packed[i]; after += rotated[i] * rotated[i]; }
            require(std::abs(before - after) < 1e-10, "rotary norm changed");
            for (size_t i = 4; i < 8; ++i) require(rotated[i] == packed[i], "partial rotary touched nonrotary channels");
        }
        std::array<double, 2> sum{0, 0};
        for (uint32_t n = 0; n < 8; ++n) {
            const auto step = index_accumulate_reference(sum, std::array<double, 2>{double(n), double(n * 2)}, n, 4);
            require(step.complete == (n % 4 == 3), "block completion boundary");
            if (step.complete) require(step.pooled == (n == 3 ? std::vector<double>({1.5, 3}) : std::vector<double>({5.5, 11})), "pooling must precede normalization/position");
            std::copy(step.next_sum.begin(), step.next_sum.end(), sum.begin());
        }
        for (const uint32_t count : {1, 511, 512, 513, 16384, 65536}) {
            std::vector<double> scores(count);
            for (uint32_t i = 0; i < count; ++i) scores[i] = i % 17;
            const auto ids = select_blocks_reference(scores, 512);
            require(ids.size() == std::min(count, 512u), "top-block count");
            for (size_t i = 1; i < ids.size(); ++i) require(ids[i] > ids[i - 1], "selected IDs not canonical/unique");
        }
        const QsaShape s{2, 2, 1, 2, 1, 2, 2, 2, 2, 8};
        std::vector<uint16_t> keys(16), values(16);
        for (uint32_t t = 0; t < 8; ++t) { values[2 * t] = finite_bf16(float(t)); values[2 * t + 1] = finite_bf16(float(2 * t)); }
        const std::array<double, 4> query{0, 0, 0, 0};
        const auto out = sparse_attention_reference(query, keys, values, std::array<uint32_t, 1>{1}, 5, s);
        // Selected block 1 is tokens 2,3; incomplete tail token 4 is always added.
        require(out == std::vector<double>({3, 6, 3, 6}), "uniform sparse attention/tail calculation");
        reject([&] { static_cast<void>(sparse_attention_reference(query, keys, values, std::array<uint32_t, 1>{2}, 5, s)); });
        reject([&] { auto shape = s; shape.token_budget = 3; validate_qsa_shape(shape); });
        std::cout << "Attention math checks passed: " << checks << '\n';
    } catch (const std::exception& e) { std::cerr << "Attention math check failed: " << e.what() << '\n'; return 1; }
}
