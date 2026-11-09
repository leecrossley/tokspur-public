#include "tokspur/attention_math.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>

namespace tokspur {
namespace {
void finite(std::span<const double> values) {
    for (const auto x : values) if (!std::isfinite(x)) throw FormatError("nonfinite attention input/result");
}
}
void validate_qsa_shape(QsaShape s) {
    if (!s.channels || s.channels > 65536 || !s.query_heads || s.query_heads > 64 || !s.kv_heads || s.kv_heads > s.query_heads ||
        s.query_heads % s.kv_heads || !s.head_width || s.head_width > 256 || s.head_width % 2 || !s.index_heads || s.index_heads > 8 ||
        !s.index_width || s.index_width > 256 || s.index_width % 2 || !s.rotary_width || s.rotary_width % 2 ||
        s.rotary_width > s.head_width || s.rotary_width > s.index_width || !s.compression || s.compression > 16 ||
        !s.token_budget || s.token_budget % s.compression || s.token_budget / s.compression > 512 || !s.capacity || s.capacity > 65536) throw FormatError("unsupported sparse-attention shape");
}
void validate_qsa_policy(QsaPolicy p) {
    if (uint32_t(p.packing) > 1 || uint32_t(p.rotary_pairs) > 1 || !std::isfinite(p.rms_epsilon) || p.rms_epsilon <= 0) throw FormatError("invalid sparse-attention policy");
}
uint32_t qsa_block_capacity(QsaShape s) { validate_qsa_shape(s); return (s.capacity + s.compression - 1) / s.compression; }
uint32_t qsa_block_budget(QsaShape s) { validate_qsa_shape(s); return s.token_budget / s.compression; }
uint64_t qsa_dispatches(QsaShape s, uint32_t position) {
    validate_qsa_shape(s); if (position >= s.capacity) throw FormatError("attention cache capacity exceeded");
    const auto blocks = (position + 1) / s.compression;
    uint64_t count = 18 + ((position + 1) % s.compression == 0 ? 3 : 0);
    if (blocks > qsa_block_budget(s)) {
        count += 3; // Score, local sort and canonical IDs.
        for (uint32_t span = 512; span < blocks; span *= 2) ++count;
    }
    return count;
}
uint16_t finite_bf16(float value) {
    if (!std::isfinite(value)) throw FormatError("nonfinite BF16 storage input");
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const auto rounded = uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    if ((rounded & 0x7f80u) == 0x7f80u) throw FormatError("BF16 storage overflow");
    return rounded;
}
float read_bf16(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
std::vector<float> rotary_frequencies(uint32_t width, double base) {
    if (!width || width > 256 || width % 2 || !std::isfinite(base) || base <= 1) throw FormatError("invalid rotary frequency geometry");
    std::vector<float> out(width / 2);
    for (size_t i = 0; i < out.size(); ++i) out[i] = float(std::pow(base, -2.0 * i / width));
    return out;
}
std::vector<double> rotary_reference(std::span<const double> input, uint32_t heads, uint32_t width,
                                      uint32_t position, std::span<const float> freq, RotaryPairs pairs) {
    if (!heads || heads > 64 || !width || width > 256 || input.size() != uint64_t(heads) * width || freq.empty() ||
        freq.size() * 2 > width || uint32_t(pairs) > 1 || position >= 65536) throw FormatError("rotary reference shape mismatch");
    finite(input);
    std::vector<double> out(input.begin(), input.end());
    for (uint32_t h = 0; h < heads; ++h) for (size_t i = 0; i < freq.size(); ++i) {
        if (!std::isfinite(freq[i]) || freq[i] <= 0) throw FormatError("invalid rotary inverse frequency");
        const auto a = h * width + (pairs == RotaryPairs::split_half ? i : 2 * i);
        const auto b = a + (pairs == RotaryPairs::split_half ? freq.size() : 1);
        const float angle = float(position) * freq[i];
        const double c = std::cos(double(angle)), s = std::sin(double(angle));
        out[a] = input[a] * c - input[b] * s; out[b] = input[b] * c + input[a] * s;
    }
    finite(out); return out;
}
QueryGate split_query_reference(std::span<const double> input, uint32_t heads, uint32_t width, QueryPacking packing) {
    if (!heads || heads > 64 || !width || width > 256 || input.size() != 2ull * heads * width || uint32_t(packing) > 1) throw FormatError("query/gate reference shape mismatch");
    finite(input);
    QueryGate out{std::vector<double>(heads * width), std::vector<double>(heads * width)};
    for (uint32_t h = 0; h < heads; ++h) for (uint32_t d = 0; d < width; ++d) {
        const auto q = packing == QueryPacking::head_query_gate ? h * width * 2 + d : h * width + d;
        const auto z = packing == QueryPacking::head_query_gate ? q + width : q + heads * width;
        out.query[h * width + d] = input[q]; out.gate[h * width + d] = input[z];
    }
    return out;
}
IndexStep index_accumulate_reference(std::span<const double> sum, std::span<const double> key, uint32_t position, uint32_t compression) {
    if (sum.empty() || sum.size() > 256 || sum.size() != key.size() || !compression || compression > 16 || position >= 65536) throw FormatError("index pooling reference shape mismatch");
    finite(sum); finite(key);
    IndexStep out{std::vector<double>(sum.size()), std::vector<double>(sum.size()), (position + 1) % compression == 0};
    for (size_t i = 0; i < sum.size(); ++i) {
        const double value = sum[i] + key[i];
        if (!std::isfinite(value)) throw FormatError("index pooling overflow");
        out.next_sum[i] = out.complete ? 0 : value; out.pooled[i] = out.complete ? value / compression : 0;
    }
    return out;
}
std::vector<double> index_scores_reference(std::span<const double> query, std::span<const uint16_t> keys, uint32_t heads, uint32_t width, uint32_t blocks) {
    if (!heads || heads > 8 || !width || width > 256 || query.size() != uint64_t(heads) * width || keys.size() < uint64_t(blocks) * width || blocks > 65536) throw FormatError("index scoring reference shape mismatch");
    finite(query);
    std::vector<double> out(blocks, 0);
    for (uint32_t b = 0; b < blocks; ++b) for (uint32_t h = 0; h < heads; ++h) {
        double score = 0;
        for (uint32_t d = 0; d < width; ++d) {
            const auto k = read_bf16(keys[b * width + d]);
            if (!std::isfinite(k)) throw FormatError("nonfinite index cache data");
            score += query[h * width + d] * k;
        }
        out[b] += std::max(score, 0.0);
    }
    finite(out); return out;
}
std::vector<uint32_t> select_blocks_reference(std::span<const double> scores, uint32_t budget) {
    if (!budget || budget > 512 || scores.size() > 65536) throw FormatError("index selection reference shape mismatch");
    finite(scores);
    std::vector<uint32_t> ids(scores.size()); std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
    ids.resize(std::min(size_t(budget), ids.size())); std::sort(ids.begin(), ids.end());
    return ids;
}
std::vector<double> sparse_attention_reference(std::span<const double> query, std::span<const uint16_t> keys, std::span<const uint16_t> values,
                                                std::span<const uint32_t> selected, uint32_t prefix, QsaShape s) {
    validate_qsa_shape(s);
    const uint64_t kv = uint64_t(s.kv_heads) * s.head_width;
    const auto blocks = prefix / s.compression;
    if (!prefix || prefix > s.capacity || query.size() != uint64_t(s.query_heads) * s.head_width ||
        keys.size() < prefix * kv || values.size() < prefix * kv || selected.size() != std::min(blocks, qsa_block_budget(s))) throw FormatError("sparse attention reference shape mismatch");
    finite(query);
    std::vector<uint32_t> tokens;
    for (size_t i = 0; i < selected.size(); ++i) {
        if (selected[i] >= blocks || (i && selected[i] <= selected[i - 1])) throw FormatError("invalid sorted sparse block index");
        for (uint32_t t = 0; t < s.compression; ++t) tokens.push_back(selected[i] * s.compression + t);
    }
    for (uint32_t t = blocks * s.compression; t < prefix; ++t) tokens.push_back(t);
    if (tokens.empty()) throw FormatError("empty sparse attention set");
    std::vector<double> out(query.size(), 0), logits(tokens.size());
    for (uint32_t h = 0; h < s.query_heads; ++h) {
        const auto kh = h / (s.query_heads / s.kv_heads);
        for (size_t i = 0; i < tokens.size(); ++i) {
            double score = 0;
            for (uint32_t d = 0; d < s.head_width; ++d) {
                const auto k = read_bf16(keys[tokens[i] * kv + kh * s.head_width + d]);
                if (!std::isfinite(k)) throw FormatError("nonfinite KV key data");
                score += query[h * s.head_width + d] * k;
            }
            logits[i] = score / std::sqrt(double(s.head_width));
        }
        finite(logits);
        const auto maximum = *std::max_element(logits.begin(), logits.end());
        double denominator = 0;
        for (auto& score : logits) { score = std::exp(score - maximum); denominator += score; }
        for (size_t i = 0; i < tokens.size(); ++i) for (uint32_t d = 0; d < s.head_width; ++d) {
            const auto value = read_bf16(values[tokens[i] * kv + kh * s.head_width + d]);
            if (!std::isfinite(value)) throw FormatError("nonfinite KV value data");
            out[h * s.head_width + d] += logits[i] / denominator * value;
        }
    }
    finite(out); return out;
}
} // namespace tokspur
