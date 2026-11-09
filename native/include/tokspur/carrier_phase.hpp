#pragma once
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
#include "tokspur/json.hpp"
#include <array>
#include <chrono>
#include <cstdint>

namespace tokspur {
enum class CarrierPhase : uint32_t { other, structural, numerical, sampling, readback, encoding, submission, waiting, publication };
struct CarrierCpuLedger {
    using Clock = std::chrono::steady_clock;
    bool active = false;
    CarrierPhase current = CarrierPhase::other;
    std::array<uint64_t, 9> nanoseconds{}, calls{};
    Clock::time_point changed;
    void transition(CarrierPhase next) noexcept {
        const auto now = Clock::now();
        nanoseconds[uint32_t(current)] += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - changed).count());
        changed = now; current = next;
    }
};
inline thread_local CarrierCpuLedger carrier_cpu_ledger;
inline void carrier_cpu_begin() {
    carrier_cpu_ledger = {}; carrier_cpu_ledger.active = true; carrier_cpu_ledger.changed = CarrierCpuLedger::Clock::now();
}
inline Json carrier_cpu_finish() {
    auto& ledger = carrier_cpu_ledger; ledger.transition(CarrierPhase::other); ledger.active = false;
    constexpr std::array names{"other", "structural", "numerical", "sampling", "readback", "encoding", "submission", "waiting", "native_publication"};
    Json::Object phases;
    for (size_t i = 0; i < names.size(); ++i) phases.emplace_back(names[i], Json(Json::Object{{"nanoseconds", Json::integer(ledger.nanoseconds[i])}, {"calls", Json::integer(ledger.calls[i])}}));
    return Json(std::move(phases));
}
class CarrierCpuPhase {
public:
    explicit CarrierCpuPhase(CarrierPhase phase) : active_(carrier_cpu_ledger.active), previous_(carrier_cpu_ledger.current) {
        if (active_) { carrier_cpu_ledger.transition(phase); ++carrier_cpu_ledger.calls[uint32_t(phase)]; }
    }
    ~CarrierCpuPhase() { if (active_) carrier_cpu_ledger.transition(previous_); }
private:
    bool active_;
    CarrierPhase previous_;
};
} // namespace tokspur
#define TOKSPUR_CARRIER_PHASE(name) ::tokspur::CarrierCpuPhase carrier_cpu_scope(::tokspur::CarrierPhase::name)
#else
#define TOKSPUR_CARRIER_PHASE(name) static_cast<void>(0)
#endif
