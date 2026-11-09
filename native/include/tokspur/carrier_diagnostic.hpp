#pragma once

#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include <cstdint>
#include <string_view>

namespace tokspur {

enum class CarrierDiagnosticMode : uint8_t { off, batch, dispatch, replay };

inline bool carrier_single_action_yield(uint32_t before, uint32_t current) {
    if (current < before || current - before > 1)
        throw FormatError("single-action diagnostic observed an invalid action frontier");
    return current != before;
}

inline uint32_t carrier_ordinary_history_steps(const Json& value, uint32_t rows) {
    if (!value.boolean() || !rows || rows > 7) throw FormatError("ordinary diagnostic history requires one to seven fixed inputs");
    return rows;
}

inline uint32_t carrier_component_prefix(const Json& value, uint32_t rows) {
    const auto* prefix = value.find("accepted_prefix");
    if (value.kind() != Json::Kind::object || value.object().size() != 1 || !prefix || !rows || rows > 8)
        throw FormatError("invalid carrier target-component case");
    const auto accepted = prefix->unsigned_integer();
    if (!accepted || accepted > rows) throw FormatError("carrier component accepted prefix exceeds its width");
    return uint32_t(accepted);
}

inline CarrierDiagnosticMode carrier_diagnostic_mode(std::string_view value) {
    if (value == "off") return CarrierDiagnosticMode::off;
    if (value == "batch") return CarrierDiagnosticMode::batch;
    if (value == "dispatch") return CarrierDiagnosticMode::dispatch;
    if (value == "replay") return CarrierDiagnosticMode::replay;
    throw FormatError("carrier diagnostic mode must be off, batch, dispatch or replay");
}

// Lifetime bounds never reset on a new packet, request, prompt or checkpoint.
class CarrierDiagnosticWindow {
public:
    static constexpr uint32_t maximum_actions = 4, maximum_batches = 8, maximum_dispatch_records = 8192;
    static constexpr uint32_t maximum_dispatches = maximum_dispatch_records;
    void prepare(CarrierDiagnosticMode mode, uint32_t limit, bool record_dispatches = true) {
        if (prepared_ || mode == CarrierDiagnosticMode::off || !limit || limit > maximum_actions ||
            (!record_dispatches && mode != CarrierDiagnosticMode::batch))
            fail("carrier diagnostic requires an immutable lifetime budget of 1..4 actions");
        mode_ = mode; limit_ = limit; prepared_ = true; record_dispatches_ = record_dispatches;
    }
    bool begin(bool generation_v2, bool component = false) {
        if (invalid_ || active_ || (!generation_v2 && !component)) fail("carrier diagnostic requires a non-overlapping v2 generation action");
        if (!prepared_) return false;
        if (actions_ == limit_) fail("carrier diagnostic action budget exhausted");
        ++actions_; batches_ = 0; active_ = true; component_ = component; return true;
    }
    void batch(uint64_t dispatches, uint64_t records) {
        if (invalid_ || !active_ || !dispatches || records > maximum_dispatch_records ||
            records != (record_dispatches_ ? dispatches : 0) || batches_ == maximum_batches)
            fail("carrier diagnostic batch or dispatch budget exhausted");
        ++batches_;
    }
    void batch(uint64_t dispatches) { batch(dispatches, record_dispatches_ ? dispatches : 0); }
    void before_batch() {
        if (invalid_ || !active_ || batches_ == maximum_batches) fail("carrier diagnostic batch budget exhausted before encoding");
    }
    void end() { if (invalid_ || !active_ || !batches_) fail("invalid or empty carrier diagnostic action"); active_ = false; }
    void invalidate() noexcept { invalid_ = true; }
    [[nodiscard]] bool valid() const noexcept { return !invalid_; }
    [[nodiscard]] bool component() const noexcept { return component_; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] CarrierDiagnosticMode mode() const noexcept { return mode_; }
    [[nodiscard]] uint32_t actions() const noexcept { return actions_; }
    [[nodiscard]] uint32_t batches() const noexcept { return batches_; }
    [[nodiscard]] bool records_dispatches() const noexcept { return record_dispatches_; }
private:
    [[noreturn]] void fail(const char* message) { invalid_ = true; throw FormatError(message); }
    CarrierDiagnosticMode mode_ = CarrierDiagnosticMode::off;
    uint32_t limit_ = 0, actions_ = 0, batches_ = 0;
    bool prepared_ = false, active_ = false, invalid_ = false, component_ = false, record_dispatches_ = true;
};

} // namespace tokspur
