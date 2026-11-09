#pragma once

#include "tokspur/sampling.hpp"
#include "tokspur/gguf.hpp"
#include <cmath>

namespace tokspur {
// Internal, thread-local selection. A certificate is confined to one immutable
// sampled-path decision, consumed once and destroyed before any publication.
void configure_prepared_sampling(bool enabled) noexcept;
bool prepared_sampling() noexcept;
class SamplingRowScan {
public:
    template<class Poll> static SamplingRowScan inspect(std::span<const float> row, Poll poll) {
        if (row.empty() || row.size() > maximum_sampling_vocabulary) throw FormatError("sampling vocabulary limit");
        SamplingRowScan out; out.data_ = row.data(); out.size_ = row.size();
        for (size_t i = 0; i < row.size(); ++i) {
            if (!(i & 4095)) poll();
            if (!std::isfinite(row[i])) throw FormatError("sampled path nonfinite target logit");
            out.subnormal_ |= std::fpclassify(row[i]) == FP_SUBNORMAL;
            const float value = row[i] == 0 ? 0 : row[i];
            const float maximum = row[out.maximum_] == 0 ? 0 : row[out.maximum_];
            if (value > maximum) out.maximum_ = uint32_t(i);
        }
        return out;
    }
    SamplingRowScan() = default;
    SamplingRowScan(const SamplingRowScan&) = delete;
    SamplingRowScan& operator=(const SamplingRowScan&) = delete;
    SamplingRowScan(SamplingRowScan&& other) noexcept { *this = std::move(other); }
    SamplingRowScan& operator=(SamplingRowScan&& other) noexcept {
        data_ = std::exchange(other.data_, nullptr); size_ = other.size_;
        maximum_ = other.maximum_; subnormal_ = other.subnormal_; return *this;
    }
    uint32_t consume(std::span<const float> row) {
        const auto* expected = std::exchange(data_, nullptr);
        if (!expected || row.data() != expected || row.size() != size_) throw FormatError("expired sampling validation");
        if (subnormal_) throw FormatError("subnormal sampling logit");
        return maximum_;
    }
private:
    const float* data_ = nullptr;
    size_t size_ = 0;
    uint32_t maximum_ = 0;
    bool subnormal_ = false;
};
SamplingDistribution sampling_distribution_after_scan(std::span<const float> row, SamplingPolicy policy, SamplingRowScan& scan);
} // namespace tokspur
