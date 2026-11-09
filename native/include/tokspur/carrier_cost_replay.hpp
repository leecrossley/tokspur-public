#pragma once
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include <array>
#include <charconv>
#include <cmath>
namespace tokspur {
class CarrierCostReplay {
public:
    void configure(const Json& value) {
        if (enabled_ || !value.find("schema") || value.find("schema")->unsigned_integer() != 1 || !value.find("observations_ms")) throw FormatError("invalid or repeated controller timing replay");
        const auto& observations = value.find("observations_ms")->array();
        if (observations.empty() || observations.size() > values_.size()) throw FormatError("controller timing replay bound");
        for (const auto& v : observations) {
            const auto text = v.number_lexeme(); double number = 0;
            const auto result = std::from_chars(text.data(), text.data() + text.size(), number);
            if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !std::isfinite(number) || number <= 0) throw FormatError("invalid controller timing observation");
            values_[size_++] = number;
        }
        enabled_ = true;
    }
    double observation(double live) {
        if (!enabled_) return live;
        if (used_ == size_) throw FormatError("controller timing replay exhausted");
        return values_[used_++];
    }
    bool enabled() const noexcept { return enabled_; }
    bool complete() const noexcept { return !enabled_ || used_ == size_; }
private:
    bool enabled_ = false;
    size_t size_ = 0, used_ = 0;
    std::array<double, 4> values_{};
};
inline thread_local CarrierCostReplay carrier_cost_replay;
inline double controller_observed_decode_ms(double live) { return carrier_cost_replay.observation(live); }
} // namespace tokspur
#else
namespace tokspur { inline double controller_observed_decode_ms(double live) { return live; } }
#endif
