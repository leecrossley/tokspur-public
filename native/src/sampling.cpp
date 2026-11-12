#include "tokspur/carrier_phase.hpp"
#include "tokspur/sampling.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/prepared_sampling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tokspur {
namespace {

void validate_policy(const SamplingPolicy& policy) {
    if (!std::isfinite(policy.temperature) || policy.temperature < 0 ||
        policy.top_k > maximum_sampling_vocabulary ||
        !std::isfinite(policy.top_p) || policy.top_p <= 0 || policy.top_p > 1 ||
        !std::isfinite(policy.min_p) || policy.min_p < 0 || policy.min_p > 1)
        throw FormatError("invalid sampling policy");
}

// Compensated addition makes ranking-cut sums less sensitive to the long
// positive tail. It does not purport to provide exact rational arithmetic.
class Sum {
public:
    void add(double value) {
        const double next = sum_ + value;
        correction_ += sum_ >= value ? (sum_ - next) + value : (value - next) + sum_;
        sum_ = next;
    }
    [[nodiscard]] double value() const { return sum_ + correction_; }
private:
    double sum_ = 0, correction_ = 0;
};

double relative_weight(double value, double maximum, double temperature) {
    // Avoid (maximum-value) overflowing when the two finite logits straddle
    // zero. Dividing its two nonnegative terms first is safe: an overflow then
    // means the exponential is unrepresentably small, not an invalid logit.
    const double gap = maximum >= 0 && value < 0
        ? maximum / temperature - value / temperature
        : (maximum - value) / temperature;
    return std::exp(-gap);
}

struct BuiltDistribution {
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    size_t vocabulary = 0;
#endif
    std::vector<double> probabilities;
    std::vector<uint32_t> support;
    std::vector<double> support_probabilities;
};

#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
[[maybe_unused]] BuiltDistribution build_float_fast(std::span<const float> logits, const SamplingPolicy& policy) {
    validate_policy(policy);
    if (logits.empty() || logits.size() > maximum_sampling_vocabulary)
        throw FormatError("sampling vocabulary limit");
    const auto canon = [](float val) -> float { return val == 0.0f ? 0.0f : val; };
    const auto check_scalar = [](float val) {
        if (!std::isfinite(val)) throw FormatError("nonfinite sampling logit");
        if (std::fpclassify(val) == FP_SUBNORMAL) throw FormatError("subnormal sampling logit");
    };

    const auto scan_block64 = [](const float* ptr, uint32_t& bad_flag, float& block_max) {
#if defined(__aarch64__)
        uint32x4_t bad_vec = vdupq_n_u32(0u);
        float32x4_t max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        const uint32x4_t abs_mask = vdupq_n_u32(0x7fffffffu);
        const uint32x4_t inf_bits = vdupq_n_u32(0x7f800000u);
        const uint32x4_t min_norm = vdupq_n_u32(0x00800000u);
        const uint32x4_t zero_u32 = vdupq_n_u32(0u);
        for (size_t j = 0; j < 64; j += 16) {
            const float32x4_t v0 = vld1q_f32(ptr + j);
            const float32x4_t v1 = vld1q_f32(ptr + j + 4);
            const float32x4_t v2 = vld1q_f32(ptr + j + 8);
            const float32x4_t v3 = vld1q_f32(ptr + j + 12);
            const uint32x4_t a0 = vandq_u32(vreinterpretq_u32_f32(v0), abs_mask);
            const uint32x4_t a1 = vandq_u32(vreinterpretq_u32_f32(v1), abs_mask);
            const uint32x4_t a2 = vandq_u32(vreinterpretq_u32_f32(v2), abs_mask);
            const uint32x4_t a3 = vandq_u32(vreinterpretq_u32_f32(v3), abs_mask);
            const uint32x4_t b0 = vorrq_u32(vcgeq_u32(a0, inf_bits), vandq_u32(vcgtq_u32(a0, zero_u32), vcltq_u32(a0, min_norm)));
            const uint32x4_t b1 = vorrq_u32(vcgeq_u32(a1, inf_bits), vandq_u32(vcgtq_u32(a1, zero_u32), vcltq_u32(a1, min_norm)));
            const uint32x4_t b2 = vorrq_u32(vcgeq_u32(a2, inf_bits), vandq_u32(vcgtq_u32(a2, zero_u32), vcltq_u32(a2, min_norm)));
            const uint32x4_t b3 = vorrq_u32(vcgeq_u32(a3, inf_bits), vandq_u32(vcgtq_u32(a3, zero_u32), vcltq_u32(a3, min_norm)));
            bad_vec = vorrq_u32(bad_vec, vorrq_u32(vorrq_u32(b0, b1), vorrq_u32(b2, b3)));
            max_vec = vmaxq_f32(max_vec, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        }
        bad_flag = vmaxvq_u32(bad_vec);
        block_max = vmaxvq_f32(max_vec);
#else
        bad_flag = 0;
        block_max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < 64; ++j) {
            const float v = ptr[j];
            const uint32_t b = std::bit_cast<uint32_t>(v);
            const uint32_t exp = b & 0x7f800000u;
            bad_flag |= uint32_t(exp == 0x7f800000u) | uint32_t((exp == 0u) & ((b & 0x007fffffu) != 0u));
            block_max = v > block_max ? v : block_max;
        }
#endif
    };

    if (policy.temperature == 0) {
        uint32_t maximum = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        size_t base = 0;
        for (; base + 64 <= logits.size(); base += 64) {
            const float* ptr = logits.data() + base;
            uint32_t bad_flag = 0;
            float block_max = -std::numeric_limits<float>::infinity();
            scan_block64(ptr, bad_flag, block_max);
            if (bad_flag) [[unlikely]] {
                for (size_t j = 0; j < 64; ++j) check_scalar(ptr[j]);
            }
            if (base == 0 || canon(block_max) > best_val) {
                for (size_t j = 0; j < 64; ++j) {
                    const float cv = canon(ptr[j]);
                    if (base + j == 0 || cv > best_val) {
                        best_val = cv;
                        maximum = uint32_t(base + j);
                    }
                }
            }
        }
        for (size_t i = base; i < logits.size(); ++i) {
            check_scalar(logits[i]);
            const float cv = canon(logits[i]);
            if (i == 0 || cv > best_val) {
                best_val = cv;
                maximum = uint32_t(i);
            }
        }
        return BuiltDistribution{logits.size(), {}, {maximum}, {1.0}};
    }

    const auto rank = [&](uint32_t a, uint32_t b) {
        const auto la = canon(logits[a]);
        const auto lb = canon(logits[b]);
        return la > lb || (la == lb && a < b);
    };
    const size_t retained = std::min(size_t(policy.top_k), logits.size());
    std::array<uint32_t, 64> selected;
    for (size_t i = 0; i < retained; ++i) {
        check_scalar(logits[i]);
        selected[i] = uint32_t(i);
    }
    const auto end = selected.begin() + ptrdiff_t(retained);
    std::make_heap(selected.begin(), end, rank);
    float threshold = canon(logits[selected.front()]);

    size_t base = retained;
    for (; base + 64 <= logits.size(); base += 64) {
        const float* ptr = logits.data() + base;
        uint32_t bad_flag = 0;
        float block_max = -std::numeric_limits<float>::infinity();
        scan_block64(ptr, bad_flag, block_max);
        if (bad_flag) [[unlikely]] {
            for (size_t j = 0; j < 64; ++j) check_scalar(ptr[j]);
        }
        if (canon(block_max) > threshold) {
            for (size_t j = 0; j < 64; ++j) {
                if (canon(ptr[j]) > threshold) {
                    std::pop_heap(selected.begin(), end, rank);
                    *(end - 1) = uint32_t(base + j);
                    std::push_heap(selected.begin(), end, rank);
                    threshold = canon(logits[selected.front()]);
                }
            }
        }
    }
    for (size_t id = base; id < logits.size(); ++id) {
        check_scalar(logits[id]);
        if (canon(logits[id]) > threshold) {
            std::pop_heap(selected.begin(), end, rank);
            *(end - 1) = uint32_t(id);
            std::push_heap(selected.begin(), end, rank);
            threshold = canon(logits[selected.front()]);
        }
    }
    std::sort_heap(selected.begin(), end, rank);
    const uint32_t maximum = selected[0];
    const double max_logit = double(canon(logits[maximum]));

    std::array<double, 64> weights{};
    Sum sum;
    size_t positive = 0;
    for (size_t i = 0; i < retained; ++i) {
        const uint32_t id = selected[i];
        const double weight = relative_weight(double(canon(logits[id])), max_logit, policy.temperature);
        if (weight == 0 || weight < policy.min_p) break;
        selected[positive] = id;
        weights[positive] = weight;
        ++positive;
        sum.add(weight);
    }
    if (positive == 0 || !std::isfinite(sum.value()) || sum.value() <= 0)
        throw FormatError("sampling normalization is not representable");

    if (policy.top_p < 1) {
        const double cutoff = policy.top_p * sum.value();
        Sum cumulative;
        size_t prefix = positive;
        for (size_t i = 0; i < positive; ++i) {
            cumulative.add(weights[i]);
            if (cumulative.value() >= cutoff) { prefix = i + 1; break; }
        }
        positive = prefix;
        sum = Sum();
        for (size_t i = 0; i < positive; ++i) sum.add(weights[i]);
    }
    const double denominator = sum.value();
    BuiltDistribution out;
    out.vocabulary = logits.size();
    out.support.reserve(positive);
    out.support_probabilities.reserve(positive);
    for (size_t i = 0; i < positive; ++i) {
        const double prob = weights[i] / denominator;
        if (prob > 0) {
            out.support.push_back(selected[i]);
            out.support_probabilities.push_back(prob);
        }
    }
    return out;
}
#endif

// A finite logit's IEEE bits order like its value once the sign is folded in, so
// an ascending unsigned key is a descending canonical logit. Negative zero takes
// the key of zero, as it does in the ranking comparator.
uint32_t descending_key(float value) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(value == 0.0f ? 0.0f : value);
    return (bits & 0x80000000u) ? bits : bits ^ 0x7fffffffu;
}

float canonical_logit(uint32_t key) noexcept {
    return std::bit_cast<float>((key & 0x80000000u) ? key : key ^ 0x7fffffffu);
}

// Every token as (key << 32 | id), in the ranking comparator's order: descending
// canonical logit, then ascending token ID. That order is total, so any correct
// sort yields it; the stable radix passes start from ascending IDs and
// therefore keep the lowest-ID-first tie break. Callers validate the logits.
std::unique_ptr<uint64_t[]> rank_all(std::span<const float> logits) {
    const size_t count = logits.size();
    auto ranked = std::make_unique_for_overwrite<uint64_t[]>(count);
    auto scratch = std::make_unique_for_overwrite<uint64_t[]>(count);
    constexpr std::array<uint32_t, 3> shifts{0, 11, 22};
    std::array<std::array<uint32_t, 2048>, 3> offsets{};
    for (size_t id = 0; id < count; ++id) {
        const uint32_t key = descending_key(logits[id]);
        ranked[id] = uint64_t(key) << 32 | id;
        for (size_t pass = 0; pass < shifts.size(); ++pass) ++offsets[pass][(key >> shifts[pass]) & 2047u];
    }
    for (size_t pass = 0; pass < shifts.size(); ++pass) {
        auto& bucket = offsets[pass];
        const uint32_t shift = 32 + shifts[pass];
        // A digit every key shares leaves the order unchanged.
        if (bucket[(ranked[0] >> shift) & 2047u] == count) continue;
        uint32_t next = 0;
        for (auto& entry : bucket) next += std::exchange(entry, next);
        for (size_t i = 0; i < count; ++i) scratch[bucket[(ranked[i] >> shift) & 2047u]++] = ranked[i];
        ranked.swap(scratch);
    }
    return ranked;
}

template <typename T>
BuiltDistribution build(std::span<const T> logits, const SamplingPolicy& policy, SamplingRowScan* scan = nullptr) {
    validate_policy(policy);
    if (logits.empty() || logits.size() > maximum_sampling_vocabulary)
        throw FormatError("sampling vocabulary limit");
    uint32_t maximum = 0;
    if constexpr (std::is_same_v<T, float>) { if (scan) maximum = scan->consume(logits); }
    const auto canon = [](T val) -> T { return val == 0.0f ? 0.0f : val; };
    { TOKSPUR_CARRIER_PHASE(numerical);
    for (size_t i = 0; !scan && i < logits.size(); ++i) {
        const auto val = logits[i];
        if (!std::isfinite(val)) throw FormatError("nonfinite sampling logit");
        if (std::fpclassify(val) == FP_SUBNORMAL) throw FormatError("subnormal sampling logit");
        if (canon(val) > canon(logits[maximum])) maximum = uint32_t(i);
    }
    }
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    BuiltDistribution out{logits.size(), std::vector<double>(logits.size(), 0), {}, {}};
#else
    BuiltDistribution out{std::vector<double>(logits.size(), 0), {}, {}};
#endif
    if (policy.temperature == 0) {
        out.probabilities[maximum] = 1;
        out.support.push_back(maximum);
        out.support_probabilities.push_back(1.0);
        return out;
    }

    const auto rank = [&](uint32_t a, uint32_t b) {
        const auto la = canon(logits[a]);
        const auto lb = canon(logits[b]);
        return la > lb || (la == lb && a < b);
    };
    const size_t retained = policy.top_k ? std::min(size_t(policy.top_k), logits.size()) : logits.size();
    if constexpr (std::is_same_v<T, float>) {
        if (retained > 64) {
            // An absent or wide top-k ranks the whole vocabulary, which is what a request without top_k asks for on every row. Rank it by radix and keep the weights beside the order; each weight, sum and quotient is the one the comparison sort below would produce, taken in the same order.
            const auto ranked = rank_all(logits);
            const double best = double(canonical_logit(uint32_t(ranked[0] >> 32)));
            const auto weights = std::make_unique_for_overwrite<double[]>(retained);
            Sum sum;
            size_t positive = 0;
            for (; positive < retained; ++positive) {
                const double weight = relative_weight(double(canonical_logit(uint32_t(ranked[positive] >> 32))), best, policy.temperature);
                if (weight == 0 || weight < policy.min_p) break;
                weights[positive] = weight;
                sum.add(weight);
            }
            if (positive == 0 || !std::isfinite(sum.value()) || sum.value() <= 0)
                throw FormatError("sampling normalization is not representable");
            if (policy.top_p < 1) {
                const double threshold = policy.top_p * sum.value();
                Sum cumulative;
                size_t prefix = positive;
                for (size_t i = 0; i < positive; ++i) {
                    cumulative.add(weights[i]);
                    if (cumulative.value() >= threshold) { prefix = i + 1; break; }
                }
                positive = prefix;
                sum = Sum();
                for (size_t i = 0; i < positive; ++i) sum.add(weights[i]);
            }
            const double denominator = sum.value();
            out.support.reserve(positive);
            out.support_probabilities.reserve(positive);
            for (size_t i = 0; i < positive; ++i) {
                const double probability = weights[i] / denominator;
                if (!(probability > 0)) continue;
                const uint32_t id = uint32_t(ranked[i]);
                out.probabilities[id] = probability;
                out.support.push_back(id);
                out.support_probabilities.push_back(probability);
            }
            return out;
        }
    }
#ifdef TOKSPUR_PRIVATE_BOUNDED_TOPK
    if (policy.top_k && policy.top_k <= 64) {
        // The same heap selection as partial_sort, without materialising a vocabulary-sized index vector. Validation and all mass arithmetic remain unchanged. The worst retained token is the heap root.
        std::array<uint32_t, 64> selected;
        std::iota(selected.begin(), selected.begin() + ptrdiff_t(retained), 0);
        const auto end = selected.begin() + ptrdiff_t(retained);
        std::make_heap(selected.begin(), end, rank);
        for (size_t id = retained; id < logits.size(); ++id) {
            if (!rank(uint32_t(id), selected.front())) continue;
            std::pop_heap(selected.begin(), end, rank);
            *(end - 1) = uint32_t(id);
            std::push_heap(selected.begin(), end, rank);
        }
        std::sort_heap(selected.begin(), end, rank);
        out.support.assign(selected.begin(), end);
    } else
#endif
    {
        out.support.resize(logits.size());
        std::iota(out.support.begin(), out.support.end(), 0);
        // The comparator totally orders distinct token IDs. When every token is
        // retained (no top_k, the OpenAI default), introsort gives the identical
        // order without a full heap sort, so every mode takes it. Leave mass
        // accumulation and normalization in that original order.
        if (retained == out.support.size())
            std::sort(out.support.begin(), out.support.end(), rank);
        else
            std::partial_sort(out.support.begin(), out.support.begin() + ptrdiff_t(retained), out.support.end(), rank);
        out.support.resize(retained);
    }

    Sum sum;
    size_t positive = 0;
    for (const auto id : out.support) {
        const double weight = relative_weight(double(canon(logits[id])), double(canon(logits[maximum])), policy.temperature);
        if (weight == 0 || weight < policy.min_p) break;
        out.probabilities[id] = weight;
        out.support[positive++] = id;
        sum.add(weight);
    }
    out.support.resize(positive);
    // The maximum has weight exactly one, including at subnormal temperature.
    if (out.support.empty() || !std::isfinite(sum.value()) || sum.value() <= 0)
        throw FormatError("sampling normalization is not representable");

    size_t prefix = out.support.size();
    if (policy.top_p < 1) {
        const double threshold = policy.top_p * sum.value();
        Sum cumulative;
        for (size_t i = 0; i < out.support.size(); ++i) {
            cumulative.add(out.probabilities[out.support[i]]);
            if (cumulative.value() >= threshold) { prefix = i + 1; break; }
        }
        for (size_t i = prefix; i < out.support.size(); ++i) out.probabilities[out.support[i]] = 0;
        out.support.resize(prefix);
        sum = Sum();
        for (const auto id : out.support) sum.add(out.probabilities[id]);
    }
    const double denominator = sum.value();
    positive = 0;
    out.support_probabilities.reserve(out.support.size());
    for (const auto id : out.support) {
        out.probabilities[id] /= denominator;
        if (out.probabilities[id] > 0) {
            out.support[positive++] = id;
            out.support_probabilities.push_back(out.probabilities[id]);
        }
    }
    out.support.resize(positive);
    return out;
}

} // namespace

namespace { thread_local bool use_prepared_sampling = false; }
void configure_prepared_sampling(bool enabled) noexcept { use_prepared_sampling = enabled; }
bool prepared_sampling() noexcept { return use_prepared_sampling; }
SamplingDistribution sampling_distribution_after_scan(std::span<const float> logits, SamplingPolicy policy, SamplingRowScan& scan) {
    TOKSPUR_CARRIER_PHASE(sampling);
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) && defined(TOKSPUR_PRIVATE_BOUNDED_TOPK)
    if (policy.temperature == 0 || (policy.top_k > 0 && policy.top_k <= 64)) {
        validate_policy(policy); static_cast<void>(scan.consume(logits));
        return sampling_distribution(logits, policy);
    }
#endif
    auto out = build(logits, policy, &scan);
    return SamplingDistribution(logits.size(), std::move(out.probabilities), std::move(out.support), std::move(out.support_probabilities));
}

SamplingDistribution sampling_distribution(std::span<const float> logits, SamplingPolicy policy) {
    TOKSPUR_CARRIER_PHASE(sampling);
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) && defined(TOKSPUR_PRIVATE_BOUNDED_TOPK)
    if (policy.temperature == 0 || (policy.top_k > 0 && policy.top_k <= 64)) {
        auto out = build_float_fast(logits, policy);
        return SamplingDistribution(out.vocabulary, std::move(out.probabilities),
                                    std::move(out.support), std::move(out.support_probabilities));
    }
#endif
    auto out = build(logits, policy);
    return SamplingDistribution(logits.size(), std::move(out.probabilities), std::move(out.support), std::move(out.support_probabilities));
}

SamplingDistribution sampling_distribution(std::span<const double> logits, SamplingPolicy policy) {
    TOKSPUR_CARRIER_PHASE(sampling);
    auto out = build(logits, policy);
    return SamplingDistribution(logits.size(), std::move(out.probabilities), std::move(out.support), std::move(out.support_probabilities));
}

bool sampling_topk_eligible(SamplingPolicy policy, size_t vocabulary) noexcept {
    if (!std::isfinite(policy.temperature) || policy.temperature < 0 ||
        policy.top_k > maximum_sampling_vocabulary ||
        !std::isfinite(policy.top_p) || policy.top_p <= 0 || policy.top_p > 1 ||
        !std::isfinite(policy.min_p) || policy.min_p < 0 || policy.min_p > 1)
        return false;
    if (vocabulary == 0 || vocabulary > maximum_sampling_vocabulary) return false;
    return policy.temperature == 0.0 ||
           (policy.top_k > 0 && policy.top_k <= gpu_logit_topk_capacity) ||
           vocabulary <= gpu_logit_topk_capacity;
}

GpuLogitTopKRow cpu_logit_topk_row(std::span<const float> logits) {
    if (logits.empty() || logits.size() > maximum_sampling_vocabulary)
        throw FormatError("sampling vocabulary limit");
    GpuLogitTopKRow out{};
    out.vocabulary = static_cast<uint32_t>(logits.size());
    out.first_nonfinite_col = 0xffffffffu;
    out.first_subnormal_col = 0xffffffffu;
    out.logits.fill(-std::numeric_limits<float>::infinity());
    const auto canon = [](float v) noexcept -> float { return v == 0.0f ? 0.0f : v; };
    const auto beats = [&](float va, uint32_t ida, float vb, uint32_t idb) noexcept -> bool {
        const float ca = canon(va);
        const float cb = canon(vb);
        return ca > cb || (ca == cb && ida < idb);
    };

    uint32_t count = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float raw = logits[i];
        const uint32_t col = static_cast<uint32_t>(i);
        const uint32_t bits = std::bit_cast<uint32_t>(raw);
        const uint32_t exp_bits = bits & 0x7f800000u;
        const uint32_t mant_bits = bits & 0x007fffffu;
        if (exp_bits == 0x7f800000u) {
            if (out.first_nonfinite_col == 0xffffffffu) out.first_nonfinite_col = col;
            continue;
        }
        if (exp_bits == 0u && mant_bits != 0u) {
            if (out.first_subnormal_col == 0xffffffffu) out.first_subnormal_col = col;
        }
        if (count == gpu_logit_topk_capacity && !beats(raw, col, out.logits[count - 1], out.ids[count - 1])) {
            continue;
        }
        uint32_t pos = count < gpu_logit_topk_capacity ? count++ : (gpu_logit_topk_capacity - 1);
        while (pos > 0 && beats(raw, col, out.logits[pos - 1], out.ids[pos - 1])) {
            out.logits[pos] = out.logits[pos - 1];
            out.ids[pos] = out.ids[pos - 1];
            --pos;
        }
        out.logits[pos] = raw;
        out.ids[pos] = col;
    }
    out.count = count;
    return out;
}

SamplingDistribution sampling_distribution_from_topk(const GpuLogitTopKRow& row, SamplingPolicy policy) {
    TOKSPUR_CARRIER_PHASE(sampling);
    validate_policy(policy);
    if (row.vocabulary == 0 || row.vocabulary > maximum_sampling_vocabulary)
        throw FormatError("sampling vocabulary limit");
    if ((row.first_nonfinite_col != 0xffffffffu && row.first_nonfinite_col >= row.vocabulary) ||
        (row.first_subnormal_col != 0xffffffffu && row.first_subnormal_col >= row.vocabulary)) {
        throw FormatError("corrupted GPU top-k bad-logit index");
    }
    if (row.first_nonfinite_col != 0xffffffffu || row.first_subnormal_col != 0xffffffffu) {
        if (row.first_nonfinite_col <= row.first_subnormal_col)
            throw FormatError("nonfinite sampling logit");
        throw FormatError("subnormal sampling logit");
    }
    if (!sampling_topk_eligible(policy, row.vocabulary))
        throw FormatError("sampling policy requires full vocabulary logits");

    const uint32_t expected = std::min<uint32_t>(gpu_logit_topk_capacity, row.vocabulary);
    if (row.count != expected)
        throw FormatError("incomplete GPU top-k summary");
    const auto canon = [](float val) noexcept -> float { return val == 0.0f ? 0.0f : val; };
    for (uint32_t k = 0; k < expected; ++k) {
        const uint32_t id = row.ids[k];
        const float val = row.logits[k];
        if (id >= row.vocabulary || !std::isfinite(val) || std::fpclassify(val) == FP_SUBNORMAL)
            throw FormatError("invalid GPU top-k candidate entry");
        if (k > 0) {
            const float prev = canon(row.logits[k - 1]);
            const float cur = canon(val);
            const uint32_t prev_id = row.ids[k - 1];
            if (!(prev > cur || (prev == cur && prev_id < id)))
                throw FormatError("unsorted or duplicate GPU top-k candidate entry");
        }
    }

    if (policy.temperature == 0) {
        return SamplingDistribution(row.vocabulary, {}, {row.ids[0]}, {1.0});
    }

    const size_t retained = policy.top_k ? std::min<size_t>(policy.top_k, row.vocabulary) : size_t(row.vocabulary);
    std::array<uint32_t, gpu_logit_topk_capacity> ids{};
    std::array<double, gpu_logit_topk_capacity> probs{};
    const double max_logit = double(canon(row.logits[0]));

    Sum sum;
    size_t positive = 0;
    for (size_t k = 0; k < retained; ++k) {
        const double weight = relative_weight(double(canon(row.logits[k])), max_logit, policy.temperature);
        if (weight == 0 || weight < policy.min_p) break;
        ids[positive] = row.ids[k];
        probs[positive] = weight;
        ++positive;
        sum.add(weight);
    }
    if (positive == 0 || !std::isfinite(sum.value()) || sum.value() <= 0)
        throw FormatError("sampling normalization is not representable");

    if (policy.top_p < 1) {
        const double threshold = policy.top_p * sum.value();
        Sum cumulative;
        size_t prefix = positive;
        for (size_t i = 0; i < positive; ++i) {
            cumulative.add(probs[i]);
            if (cumulative.value() >= threshold) {
                prefix = i + 1;
                break;
            }
        }
        positive = prefix;
        sum = Sum();
        for (size_t i = 0; i < positive; ++i) sum.add(probs[i]);
    }

    const double denominator = sum.value();
    std::vector<uint32_t> support;
    std::vector<double> support_probs;
    support.reserve(positive);
    support_probs.reserve(positive);
    for (size_t i = 0; i < positive; ++i) {
        const double p = probs[i] / denominator;
        if (p > 0) {
            support.push_back(ids[i]);
            support_probs.push_back(p);
        }
    }
    return SamplingDistribution(row.vocabulary, {}, std::move(support), std::move(support_probs));
}

std::span<const double> SamplingDistribution::probabilities() const {
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    if (vocabulary_ && probabilities_.size() != vocabulary_) {
#else
    if (probabilities_.empty() && vocabulary_ != 0) {
#endif
        probabilities_.assign(vocabulary_, 0.0);
        for (size_t i = 0; i < support_.size(); ++i) {
            probabilities_[support_[i]] = support_probabilities_[i];
        }
    }
    return probabilities_;
}

double SamplingDistribution::probability(uint32_t token) const {
    if (token >= vocabulary_) throw FormatError("sampling token outside vocabulary");
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    if (probabilities_.size() == vocabulary_) return probabilities_[token];
#else
    if (!probabilities_.empty()) return probabilities_[token];
#endif
    for (size_t i = 0; i < support_.size(); ++i) {
        if (support_[i] == token) return support_probabilities_[i];
    }
    return 0.0;
}

double SamplingRng::uniform() {
    if (words_ == std::numeric_limits<uint64_t>::max()) throw FormatError("sampling RNG counter exhausted");
    const uint64_t value = engine_();
    ++words_;
    return double(value >> 11) * 0x1p-53;
}

uint32_t sample_at(const SamplingDistribution& distribution, double unit) {
    if (distribution.vocabulary() == 0 || distribution.support().empty())
        throw FormatError("empty sampling distribution");
    if (!std::isfinite(unit) || unit < 0 || unit >= 1) throw FormatError("invalid sampling unit interval");
    Sum cumulative;
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    const auto ids = distribution.support();
    const auto probs = distribution.support_probabilities();
    for (size_t i = 0; i < ids.size(); ++i) {
        cumulative.add(probs[i]);
        if (unit < cumulative.value()) return ids[i];
    }
    return ids.back();
#else
    for (const auto id : distribution.support()) {
        cumulative.add(distribution.probability(id));
        if (unit < cumulative.value()) return id;
    }
    return distribution.support().back();
#endif
}

uint32_t sample(const SamplingDistribution& distribution, SamplingRng& random) {
    if (distribution.vocabulary() == 0 || distribution.support().empty())
        throw FormatError("empty sampling distribution");
    if (distribution.deterministic()) return distribution.support()[0];
    auto next = random;
    const auto token = sample_at(distribution, next.uniform());
    random = std::move(next);
    return token;
}

} // namespace tokspur
