#pragma once

// Private host-clock telemetry. This file has no Metal, model or file I/O.
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace tokspur {

inline std::optional<bool> batch_phase_clock_mode(std::string_view value) noexcept {
    if (value == "off") return false;
    if (value == "on") return true;
    return {};
}

struct BatchPhaseClockState {
    bool selected = false, enabled = false, locked = false;
    uint64_t successful_batches = 0, records = 0, failures = 0;

    bool configure(bool next) noexcept {
        if (locked || (selected && enabled != next)) return false;
        selected = true; enabled = next; return true;
    }
    void lock() noexcept { locked = true; }
    static void increment(uint64_t& value) noexcept {
        if (value != std::numeric_limits<uint64_t>::max()) ++value;
    }
    uint64_t completed() noexcept {
        if (!selected) return 0;
        if (successful_batches == std::numeric_limits<uint64_t>::max()) increment(failures);
        else ++successful_batches;
        return successful_batches;
    }
    void emitted(bool written, bool valid) noexcept {
        if (written) {
            if (records == std::numeric_limits<uint64_t>::max()) increment(failures);
            else ++records;
        }
        if (!written || !valid) increment(failures);
    }
};

// CPU stamps: original wall start, submit entry, before endEncoding, after
// endEncoding, before commit, after commit, after wait, original wall end.
// No GPU-clock subtraction or causal interpretation belongs in this record.
struct BatchPhaseClockSample {
    uint64_t batch = 0, dispatches = 0, capture_executes = 0;
    std::array<int64_t, 8> cpu_ns{};
    double reported_wall_ms = 0, reported_gpu_ms = 0;
};
struct BatchPhaseClockIntervals {
    bool valid = false;
    std::array<uint64_t, 7> ns{};
    uint64_t wall_ns = 0;
};

inline BatchPhaseClockIntervals batch_phase_clock_intervals(const BatchPhaseClockSample& sample) noexcept {
    BatchPhaseClockIntervals out;
    if (!sample.batch || !sample.dispatches || sample.capture_executes ||
        !std::isfinite(sample.reported_wall_ms) || sample.reported_wall_ms < 0 ||
        !std::isfinite(sample.reported_gpu_ms) || sample.reported_gpu_ms < 0) return out;
    for (size_t i = 0; i < sample.cpu_ns.size(); ++i)
        if (sample.cpu_ns[i] < 0 || (i && sample.cpu_ns[i] < sample.cpu_ns[i - 1])) return out;
    for (size_t i = 0; i < out.ns.size(); ++i)
        out.ns[i] = uint64_t(sample.cpu_ns[i + 1] - sample.cpu_ns[i]);
    out.wall_ns = uint64_t(sample.cpu_ns.back() - sample.cpu_ns.front());
    const auto milliseconds = double(out.wall_ns) / 1e6;
    if (std::abs(milliseconds - sample.reported_wall_ms) > 1e-6 + milliseconds * 1e-12) return out;
    out.valid = true; return out;
}

// Fixed caller-owned storage, locale-independent JSON, no heap allocation and
// no exception path. Invalid clocks remain an explicit invalid record; a buffer
// refusal returns zero and is a telemetry failure, never a model exception.
inline size_t format_batch_phase_clock(const BatchPhaseClockSample& sample, std::span<char> buffer) noexcept {
    struct Writer {
        std::span<char> output; size_t used = 0; bool okay = true;
        void text(std::string_view value) noexcept {
            if (!okay || value.size() > output.size() - used) { okay = false; return; }
            for (char c : value) output[used++] = c;
        }
        void integer(uint64_t value) noexcept {
            if (!okay) return;
            const auto result = std::to_chars(output.data() + used, output.data() + output.size(), value);
            if (result.ec != std::errc{}) okay = false;
            else used = size_t(result.ptr - output.data());
        }
        void signed_integer(int64_t value) noexcept {
            if (!okay) return;
            const auto result = std::to_chars(output.data() + used, output.data() + output.size(), value);
            if (result.ec != std::errc{}) okay = false;
            else used = size_t(result.ptr - output.data());
        }
        void number(double value) noexcept {
            if (!std::isfinite(value)) { text("null"); return; }
            if (!okay) return;
            const auto result = std::to_chars(output.data() + used, output.data() + output.size(), value);
            if (result.ec != std::errc{}) okay = false;
            else used = size_t(result.ptr - output.data());
        }
    } writer{buffer};
    const auto intervals = batch_phase_clock_intervals(sample);
    writer.text("{\"event\":\"batch_phase_clock\",\"schema\":1,\"clock\":\"steady_clock\",\"batch\":"); writer.integer(sample.batch);
    writer.text(",\"valid\":"); writer.text(intervals.valid ? "true" : "false");
    writer.text(",\"dispatches\":"); writer.integer(sample.dispatches);
    writer.text(",\"capture_executes\":"); writer.integer(sample.capture_executes);
    writer.text(",\"cpu_ns\":[");
    for (size_t i = 0; i < sample.cpu_ns.size(); ++i) { if (i) writer.text(","); writer.signed_integer(sample.cpu_ns[i]); }
    writer.text("],\"interval_ns\":[");
    for (size_t i = 0; i < intervals.ns.size(); ++i) { if (i) writer.text(","); writer.integer(intervals.ns[i]); }
    writer.text("],\"wall_ns\":"); writer.integer(intervals.wall_ns);
    writer.text(",\"reported_wall_ms\":"); writer.number(sample.reported_wall_ms);
    writer.text(",\"reported_gpu_ms\":"); writer.number(sample.reported_gpu_ms);
    writer.text(",\"timings_qualified\":false}\n");
    return writer.okay ? writer.used : 0;
}

} // namespace tokspur
