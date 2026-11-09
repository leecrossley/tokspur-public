#include "tokspur/host_telemetry.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>

namespace tokspur::telemetry {

// ---------------------------------------------------------------------------
// Time

uint64_t ticks_to_nanoseconds(uint64_t ticks, Timebase timebase) {
    if (!timebase.valid()) throw TelemetryError("mach timebase unavailable: ticks cannot be converted");
    // Exact: ticks * numer / denom without intermediate overflow for any realistic timebase.
    const uint64_t whole = ticks / timebase.denom, rest = ticks % timebase.denom;
    if (whole > std::numeric_limits<uint64_t>::max() / timebase.numer) throw TelemetryError("tick conversion overflows 64 bits");
    const uint64_t scaled = whole * timebase.numer;
    const uint64_t fraction = (rest * timebase.numer) / timebase.denom; // rest < denom <= 2^32, numer <= 2^32: no overflow
    if (scaled > std::numeric_limits<uint64_t>::max() - fraction) throw TelemetryError("tick conversion overflows 64 bits");
    return scaled + fraction;
}

double ticks_to_seconds(uint64_t ticks, Timebase timebase) { return double(ticks_to_nanoseconds(ticks, timebase)) / 1e9; }

// ---------------------------------------------------------------------------
// Names

std::string_view status_name(Status status) noexcept {
    switch (status) {
        case Status::observed: return "observed";
        case Status::unavailable: return "unavailable";
        case Status::permission_denied: return "permission_denied";
        case Status::not_found: return "not_found";
        case Status::identity_changed: return "identity_changed";
        case Status::error: return "error";
    }
    return "invalid";
}

std::string_view delta_status_name(DeltaStatus status) noexcept {
    switch (status) {
        case DeltaStatus::valid: return "valid";
        case DeltaStatus::unavailable: return "unavailable";
        case DeltaStatus::bracket_disorder: return "bracket_disorder";
        case DeltaStatus::regression: return "regression";
        case DeltaStatus::identity_changed: return "identity_changed";
        case DeltaStatus::no_elapsed: return "no_elapsed";
        case DeltaStatus::mismatch: return "mismatch";
    }
    return "invalid";
}

// ---------------------------------------------------------------------------
// Identity

bool same_birth(const ProcessIdentity& a, const ProcessIdentity& b) noexcept {
    if (a.pid != b.pid) return false;
    bool compared = false;
    if (a.has_wall_start && b.has_wall_start) {
        if (a.start_wall_seconds != b.start_wall_seconds || a.start_wall_microseconds != b.start_wall_microseconds) return false;
        compared = true;
    }
    if (a.has_abstime_start && b.has_abstime_start) {
        if (a.start_abstime != b.start_abstime) return false;
        compared = true;
    }
    if (a.has_executable_uuid && b.has_executable_uuid) {
        if (a.executable_uuid != b.executable_uuid) return false;
        compared = true;
    }
    return compared;
}

void anchor_identity(std::optional<ProcessIdentity>& anchor, ProcessObservation& observation) {
    if (!observation.identity.observed()) return;
    const auto& identity = observation.identity.value;
    if (!anchor) {
        if (identity.has_wall_start || identity.has_abstime_start || identity.has_executable_uuid) anchor = identity;
        return;
    }
    if (same_birth(*anchor, identity)) return;
    observation.identity.status = Status::identity_changed;
    observation.identity.detail = "pid now names a different process birth than first observed";
    observation.cpu.status = Status::identity_changed;
    observation.cpu.detail = observation.identity.detail;
}

// ---------------------------------------------------------------------------
// Deltas

Elapsed elapsed_between(const Bracket& earlier, const Bracket& later) noexcept {
    Elapsed result;
    if (!earlier.ordered() || !later.ordered()) { result.status = DeltaStatus::bracket_disorder; return result; }
    const uint64_t first = earlier.midpoint(), second = later.midpoint();
    if (second <= first) { result.status = DeltaStatus::no_elapsed; return result; }
    result.status = DeltaStatus::valid;
    result.nominal_ticks = second - first;
    result.minimum_ticks = later.before > earlier.after ? later.before - earlier.after : 0;
    result.maximum_ticks = later.after - earlier.before;
    return result;
}

namespace {
template <class Value>
DeltaStatus combine(bool both_observed, const Elapsed& elapsed, bool regressed) noexcept {
    if (!both_observed) return DeltaStatus::unavailable;
    if (elapsed.status == DeltaStatus::bracket_disorder) return DeltaStatus::bracket_disorder;
    if (regressed) return DeltaStatus::regression;
    if (elapsed.status != DeltaStatus::valid) return elapsed.status;
    return DeltaStatus::valid;
}
} // namespace

TickDelta tick_delta(const Reading<CpuTicks>& earlier, const Reading<CpuTicks>& later) noexcept {
    TickDelta result;
    result.earlier = earlier.value;
    result.later = later.value;
    result.elapsed = elapsed_between(earlier.bracket, later.bracket);
    const bool both = earlier.observed() && later.observed();
    result.changed = both && !(earlier.value == later.value);
    const bool regressed = later.value.user < earlier.value.user || later.value.system < earlier.value.system ||
                           later.value.idle < earlier.value.idle || later.value.nice < earlier.value.nice;
    result.status = combine<CpuTicks>(both, result.elapsed, regressed);
    if (result.status == DeltaStatus::valid) {
        result.user = later.value.user - earlier.value.user;
        result.system = later.value.system - earlier.value.system;
        result.idle = later.value.idle - earlier.value.idle;
        result.nice = later.value.nice - earlier.value.nice;
    }
    return result;
}

size_t CoreTickDeltas::changed_cores() const noexcept { return size_t(std::count_if(cores.begin(), cores.end(), [](const TickDelta& d) { return d.changed; })); }
size_t CoreTickDeltas::regressed_cores() const noexcept {
    return size_t(std::count_if(cores.begin(), cores.end(), [](const TickDelta& d) { return d.status == DeltaStatus::regression; }));
}

CoreTickDeltas core_tick_deltas(const Reading<std::vector<CpuTicks>>& earlier, const Reading<std::vector<CpuTicks>>& later) {
    CoreTickDeltas result;
    if (!earlier.observed() || !later.observed()) { result.status = DeltaStatus::unavailable; return result; }
    if (earlier.value.size() != later.value.size()) { result.status = DeltaStatus::mismatch; return result; }
    result.status = DeltaStatus::valid;
    result.cores.reserve(earlier.value.size());
    for (size_t i = 0; i < earlier.value.size(); ++i) {
        Reading<CpuTicks> a, b;
        a.bracket = earlier.bracket; a.status = Status::observed; a.value = earlier.value[i];
        b.bracket = later.bracket; b.status = Status::observed; b.value = later.value[i];
        result.cores.push_back(tick_delta(a, b));
    }
    return result;
}

CoreTimeDeltas core_time_deltas(const Reading<std::vector<CoreTimes>>& earlier, const Reading<std::vector<CoreTimes>>& later) {
    CoreTimeDeltas result;
    result.elapsed = elapsed_between(earlier.bracket, later.bracket);
    if (!earlier.observed() || !later.observed()) { result.status = DeltaStatus::unavailable; return result; }
    if (earlier.value.size() != later.value.size()) { result.status = DeltaStatus::mismatch; return result; }
    result.status = DeltaStatus::valid;
    result.cores.reserve(earlier.value.size());
    for (size_t i = 0; i < earlier.value.size(); ++i) {
        const auto& a = earlier.value[i];
        const auto& b = later.value[i];
        CoreTimeDelta core;
        core.changed = !(a == b);
        const bool regressed = b.user_abstime < a.user_abstime || b.system_abstime < a.system_abstime || b.idle_abstime < a.idle_abstime;
        core.status = combine<CoreTimes>(true, result.elapsed, regressed);
        if (core.status == DeltaStatus::valid) {
            core.user_abstime = b.user_abstime - a.user_abstime;
            core.system_abstime = b.system_abstime - a.system_abstime;
            core.idle_abstime = b.idle_abstime - a.idle_abstime;
        }
        result.cores.push_back(core);
    }
    return result;
}

ProcessDelta process_delta(const ProcessObservation& earlier, const ProcessObservation& later) noexcept {
    ProcessDelta result;
    result.pid = later.requested_pid;
    result.earlier_status = earlier.cpu.status;
    result.later_status = later.cpu.status;
    result.elapsed = elapsed_between(earlier.cpu.bracket, later.cpu.bracket);
    if (earlier.cpu.status == Status::identity_changed || later.cpu.status == Status::identity_changed ||
        earlier.requested_pid != later.requested_pid) {
        result.status = DeltaStatus::identity_changed;
        return result;
    }
    const bool both = earlier.cpu.observed() && later.cpu.observed();
    if (!both) { result.status = DeltaStatus::unavailable; return result; }
    if (!same_birth(earlier.identity.value, later.identity.value)) { result.status = DeltaStatus::identity_changed; return result; }
    const auto& a = earlier.cpu.value;
    const auto& b = later.cpu.value;
    const bool regressed = b.user_abstime < a.user_abstime || b.system_abstime < a.system_abstime || b.runnable_abstime < a.runnable_abstime;
    result.status = combine<ProcessCpuTimes>(true, result.elapsed, regressed);
    if (result.status == DeltaStatus::valid) {
        result.user_abstime = b.user_abstime - a.user_abstime;
        result.system_abstime = b.system_abstime - a.system_abstime;
        result.runnable_abstime = b.runnable_abstime - a.runnable_abstime;
    }
    return result;
}

SelfDelta self_delta(const Reading<SelfTimes>& earlier, const Reading<SelfTimes>& later) noexcept {
    SelfDelta result;
    result.elapsed = elapsed_between(earlier.bracket, later.bracket);
    const bool both = earlier.observed() && later.observed();
    const auto& a = earlier.value;
    const auto& b = later.value;
    const bool regressed = b.user_microseconds < a.user_microseconds || b.system_microseconds < a.system_microseconds ||
                           b.children_user_microseconds < a.children_user_microseconds || b.children_system_microseconds < a.children_system_microseconds;
    result.status = combine<SelfTimes>(both, result.elapsed, regressed);
    if (result.status == DeltaStatus::valid) {
        result.user_microseconds = b.user_microseconds - a.user_microseconds;
        result.system_microseconds = b.system_microseconds - a.system_microseconds;
        result.children_user_microseconds = b.children_user_microseconds - a.children_user_microseconds;
        result.children_system_microseconds = b.children_system_microseconds - a.children_system_microseconds;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Attribution

namespace {
void note(std::string& reason, std::string_view text) {
    if (!reason.empty()) reason += "; ";
    reason += text;
}
} // namespace

Attribution attribute(const TickDelta& aggregate, const std::vector<ProcessDelta>& processes, const SelfDelta& self,
                      Timebase timebase, uint32_t tick_hz, uint32_t logical_cpus) {
    Attribution result;
    if (!timebase.valid()) note(result.reason, "mach timebase unavailable");
    if (tick_hz == 0) note(result.reason, "scheduler tick rate unavailable");
    if (!result.reason.empty()) return result;
    if (logical_cpus == 0) note(result.reason, "logical processor count unavailable: quantisation bound unknown");
    else result.tick_quantisation_seconds = double(logical_cpus) / double(tick_hz);

    if (aggregate.status != DeltaStatus::valid) {
        note(result.reason, std::string("aggregate delta ") + std::string(delta_status_name(aggregate.status)) + ": residual withheld");
    } else {
        result.elapsed_seconds = ticks_to_seconds(aggregate.elapsed.nominal_ticks, timebase);
        result.elapsed_uncertainty_seconds = ticks_to_seconds(aggregate.elapsed.maximum_ticks - aggregate.elapsed.minimum_ticks, timebase) / 2;
        result.aggregate_busy_seconds = double(aggregate.busy()) / double(tick_hz);
        result.aggregate_total_seconds = double(aggregate.total()) / double(tick_hz);
        if (!aggregate.changed) note(result.reason, "aggregate counter unchanged across the interval (cached/stale read or no tick elapsed): residual withheld");
    }

    for (const auto& process : processes) {
        if (process.status == DeltaStatus::valid) {
            ++result.selected_valid;
            result.selected_process_seconds += ticks_to_seconds(process.user_abstime, timebase) + ticks_to_seconds(process.system_abstime, timebase);
        } else {
            ++result.selected_invalid;
        }
    }
    result.selection_complete = result.selected_invalid == 0;
    if (!result.selection_complete) note(result.reason, std::to_string(result.selected_invalid) + " selected process delta(s) unavailable: selected CPU is a lower bound");

    if (self.status == DeltaStatus::valid) {
        result.collector_self_seconds = double(self.user_microseconds + self.system_microseconds) / 1e6;
        result.collector_children_seconds = double(self.children_user_microseconds + self.children_system_microseconds) / 1e6;
    } else {
        note(result.reason, std::string("collector self delta ") + std::string(delta_status_name(self.status)));
    }

    result.aggregate_usable = aggregate.status == DeltaStatus::valid && aggregate.changed && result.elapsed_seconds.value_or(0) > 0;
    if (result.aggregate_usable) {
        result.residual_cores = (*result.aggregate_busy_seconds - result.selected_process_seconds) / *result.elapsed_seconds;
    }
    return result;
}

// ---------------------------------------------------------------------------
// JSON text

namespace json {
namespace {
bool valid_utf8(std::string_view text) noexcept {
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        size_t length = 0;
        uint32_t minimum = 0, point = 0;
        if (lead < 0x80) { ++i; continue; }
        if (lead >= 0xc2 && lead <= 0xdf) { length = 2; minimum = 0x80; point = lead & 0x1f; }
        else if (lead >= 0xe0 && lead <= 0xef) { length = 3; minimum = 0x800; point = lead & 0x0f; }
        else if (lead >= 0xf0 && lead <= 0xf4) { length = 4; minimum = 0x10000; point = lead & 0x07; }
        else return false;
        if (i + length > text.size()) return false;
        for (size_t k = 1; k < length; ++k) {
            const unsigned char next = static_cast<unsigned char>(text[i + k]);
            if ((next & 0xc0) != 0x80) return false;
            point = (point << 6) | (next & 0x3f);
        }
        if (point < minimum || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
        i += length;
    }
    return true;
}
} // namespace

std::string string(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    const bool utf8 = valid_utf8(text);
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(char(c)); }
        else if (c < 0x20 || (!utf8 && c >= 0x80)) {
            // Control bytes always escape; bytes of an invalid UTF-8 string are
            // escaped individually (lossy by design, never silently dropped).
            out += "\\u00"; out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]);
        } else out.push_back(char(c));
    }
    out.push_back('"');
    return out;
}
std::string number(uint64_t value) { return std::to_string(value); }
std::string number(int64_t value) { return std::to_string(value); }
std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    // std::to_chars is locale-independent and shortest round-trip; snprintf honours LC_NUMERIC.
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof buffer, value);
    if (result.ec != std::errc{}) return "null";
    return std::string(buffer, size_t(result.ptr - buffer));
}
std::string boolean(bool value) { return value ? "true" : "false"; }
std::string null() { return "null"; }
std::string object(const std::vector<Member>& members) {
    std::string out = "{";
    for (size_t i = 0; i < members.size(); ++i) {
        if (i) out.push_back(',');
        out += string(members[i].first);
        out.push_back(':');
        out += members[i].second;
    }
    out.push_back('}');
    return out;
}
std::string array(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) out.push_back(',');
        out += items[i];
    }
    out.push_back(']');
    return out;
}
} // namespace json

namespace {
using json::Member;

std::string bracket_json(const Bracket& bracket) {
    return json::object({{"before", json::number(bracket.before)}, {"after", json::number(bracket.after)}, {"ordered", json::boolean(bracket.ordered())}});
}
std::string seconds_or_null(uint64_t ticks, Timebase timebase) {
    if (!timebase.valid()) return json::null();
    try { return json::number(ticks_to_seconds(ticks, timebase)); } catch (const TelemetryError&) { return json::null(); }
}
template <class Value, class Describe>
std::string reading_json(const Reading<Value>& reading, Describe&& describe_value) {
    return json::object({{"status", json::string(status_name(reading.status))}, {"detail", json::string(reading.detail)},
                         {"bracket_ticks", bracket_json(reading.bracket)}, {"value", reading.observed() ? describe_value(reading.value) : json::null()}});
}
std::string ticks_json(const CpuTicks& ticks) {
    return json::object({{"user", json::number(uint64_t(ticks.user))}, {"system", json::number(uint64_t(ticks.system))},
                         {"idle", json::number(uint64_t(ticks.idle))}, {"nice", json::number(uint64_t(ticks.nice))}});
}
std::string core_times_json(const CoreTimes& times) {
    return json::object({{"user_abstime", json::number(times.user_abstime)}, {"system_abstime", json::number(times.system_abstime)},
                         {"idle_abstime", json::number(times.idle_abstime)}, {"cycles", json::number(times.cycles)},
                         {"instructions", json::number(times.instructions)}, {"energy_nj", json::number(times.energy_nj)}});
}
std::string uuid_json(const ProcessIdentity& identity) {
    if (!identity.has_executable_uuid) return json::null();
    constexpr char hex[] = "0123456789abcdef";
    std::string text;
    for (const auto byte : identity.executable_uuid) { text.push_back(hex[byte >> 4]); text.push_back(hex[byte & 15]); }
    return json::string(text);
}
std::string identity_json(const ProcessIdentity& identity) {
    return json::object({{"pid", json::number(int64_t(identity.pid))}, {"parent_pid", json::number(uint64_t(identity.parent_pid))},
                         {"uid", json::number(uint64_t(identity.uid))}, {"bsd_status", json::number(uint64_t(identity.bsd_status))},
                         {"start_wall_seconds", identity.has_wall_start ? json::number(identity.start_wall_seconds) : json::null()},
                         {"start_wall_microseconds", identity.has_wall_start ? json::number(identity.start_wall_microseconds) : json::null()},
                         {"start_abstime", identity.has_abstime_start ? json::number(identity.start_abstime) : json::null()},
                         {"executable_uuid", uuid_json(identity)}, {"command", json::string(identity.command)}, {"name", json::string(identity.name)}});
}
std::string process_cpu_json(const ProcessCpuTimes& times) {
    return json::object({{"user_abstime", json::number(times.user_abstime)}, {"system_abstime", json::number(times.system_abstime)},
                         {"runnable_abstime", json::number(times.runnable_abstime)}, {"interrupt_wakeups", json::number(times.interrupt_wakeups)},
                         {"package_idle_wakeups", json::number(times.package_idle_wakeups)}});
}
std::string self_json(const SelfTimes& times) {
    return json::object({{"user_microseconds", json::number(times.user_microseconds)}, {"system_microseconds", json::number(times.system_microseconds)},
                         {"children_user_microseconds", json::number(times.children_user_microseconds)},
                         {"children_system_microseconds", json::number(times.children_system_microseconds)},
                         {"voluntary_switches", json::number(times.voluntary_switches)}, {"involuntary_switches", json::number(times.involuntary_switches)}});
}
std::string self_task_json(const SelfTaskTimes& times) {
    return json::object({{"total_user_abstime", json::number(times.total_user_abstime)}, {"total_system_abstime", json::number(times.total_system_abstime)},
                         {"threads_user_abstime", json::number(times.threads_user_abstime)}, {"threads_system_abstime", json::number(times.threads_system_abstime)}});
}
template <class Value, class Describe>
std::string list_json(const std::vector<Value>& values, Describe&& describe_value) {
    std::vector<std::string> items;
    items.reserve(values.size());
    for (const auto& value : values) items.push_back(describe_value(value));
    return json::array(items);
}
} // namespace

std::string describe(const Sample& sample) {
    const auto timebase = json::object({{"numer", json::number(uint64_t(sample.timebase.numer))}, {"denom", json::number(uint64_t(sample.timebase.denom))}});
    return json::object({
        {"sequence", json::number(sample.sequence)}, {"timebase", timebase}, {"tick_hz", json::number(uint64_t(sample.tick_hz))},
        {"logical_cpus", json::number(uint64_t(sample.logical_cpus))},
        {"environment_status", json::string(status_name(sample.environment_status))}, {"environment_detail", json::string(sample.environment_detail)},
        {"span_ticks", bracket_json(sample.span)},
        {"continuous_before", json::number(sample.continuous_before)}, {"continuous_after", json::number(sample.continuous_after)},
        {"realtime_nanoseconds", json::number(sample.realtime_nanoseconds)},
        {"aggregate_scheduler_ticks", reading_json(sample.aggregate, ticks_json)},
        {"core_scheduler_ticks", reading_json(sample.core_ticks, [](const std::vector<CpuTicks>& cores) {
             return json::object({{"processors", json::number(uint64_t(cores.size()))}, {"cores", list_json(cores, ticks_json)}}); })},
        {"core_times", reading_json(sample.core_times, [](const std::vector<CoreTimes>& cores) {
             return json::object({{"processors", json::number(uint64_t(cores.size()))}, {"cores", list_json(cores, core_times_json)}}); })},
        {"processes", list_json(sample.processes, [](const ProcessObservation& process) {
             return json::object({{"requested_pid", json::number(int64_t(process.requested_pid))}, {"identity", reading_json(process.identity, identity_json)},
                                  {"cpu", reading_json(process.cpu, process_cpu_json)}}); })},
        {"self", reading_json(sample.self, self_json)}, {"self_task", reading_json(sample.self_task, self_task_json)}});
}

std::string describe(const Elapsed& elapsed, Timebase timebase) {
    return json::object({{"status", json::string(delta_status_name(elapsed.status))}, {"nominal_ticks", json::number(elapsed.nominal_ticks)},
                         {"minimum_ticks", json::number(elapsed.minimum_ticks)}, {"maximum_ticks", json::number(elapsed.maximum_ticks)},
                         {"nominal_seconds", elapsed.status == DeltaStatus::valid ? seconds_or_null(elapsed.nominal_ticks, timebase) : json::null()}});
}

std::string describe(const TickDelta& delta, Timebase timebase) {
    const bool valid = delta.status == DeltaStatus::valid;
    return json::object({{"status", json::string(delta_status_name(delta.status))}, {"elapsed", describe(delta.elapsed, timebase)},
                         {"earlier_scheduler_ticks", ticks_json(delta.earlier)}, {"later_scheduler_ticks", ticks_json(delta.later)}, {"changed", json::boolean(delta.changed)},
                         {"delta_scheduler_ticks", valid ? ticks_json(CpuTicks{delta.user, delta.system, delta.idle, delta.nice}) : json::null()},
                         {"busy_scheduler_ticks", valid ? json::number(delta.busy()) : json::null()}, {"total_scheduler_ticks", valid ? json::number(delta.total()) : json::null()}});
}

std::string describe(const CoreTickDeltas& deltas, Timebase timebase) {
    return json::object({{"status", json::string(delta_status_name(deltas.status))}, {"processors", json::number(uint64_t(deltas.cores.size()))},
                         {"changed_cores", json::number(uint64_t(deltas.changed_cores()))}, {"regressed_cores", json::number(uint64_t(deltas.regressed_cores()))},
                         {"cores", list_json(deltas.cores, [&](const TickDelta& core) { return describe(core, timebase); })}});
}

std::string describe(const CoreTimeDeltas& deltas, Timebase timebase) {
    return json::object({{"status", json::string(delta_status_name(deltas.status))}, {"elapsed", describe(deltas.elapsed, timebase)},
                         {"processors", json::number(uint64_t(deltas.cores.size()))},
                         {"cores", list_json(deltas.cores, [](const CoreTimeDelta& core) {
                              const bool valid = core.status == DeltaStatus::valid;
                              return json::object({{"status", json::string(delta_status_name(core.status))}, {"changed", json::boolean(core.changed)},
                                                   {"user_abstime", valid ? json::number(core.user_abstime) : json::null()},
                                                   {"system_abstime", valid ? json::number(core.system_abstime) : json::null()},
                                                   {"idle_abstime", valid ? json::number(core.idle_abstime) : json::null()}}); })}});
}

std::string describe(const ProcessDelta& delta, Timebase timebase) {
    const bool valid = delta.status == DeltaStatus::valid;
    return json::object({{"pid", json::number(int64_t(delta.pid))}, {"status", json::string(delta_status_name(delta.status))},
                         {"earlier_status", json::string(status_name(delta.earlier_status))}, {"later_status", json::string(status_name(delta.later_status))},
                         {"elapsed", describe(delta.elapsed, timebase)},
                         {"user_abstime", valid ? json::number(delta.user_abstime) : json::null()}, {"system_abstime", valid ? json::number(delta.system_abstime) : json::null()},
                         {"runnable_abstime", valid ? json::number(delta.runnable_abstime) : json::null()},
                         {"cpu_seconds", valid ? seconds_or_null(delta.user_abstime + delta.system_abstime, timebase) : json::null()}});
}

std::string describe(const SelfDelta& delta, Timebase timebase) {
    const bool valid = delta.status == DeltaStatus::valid;
    return json::object({{"status", json::string(delta_status_name(delta.status))}, {"elapsed", describe(delta.elapsed, timebase)},
                         {"user_microseconds", valid ? json::number(delta.user_microseconds) : json::null()},
                         {"system_microseconds", valid ? json::number(delta.system_microseconds) : json::null()},
                         {"children_user_microseconds", valid ? json::number(delta.children_user_microseconds) : json::null()},
                         {"children_system_microseconds", valid ? json::number(delta.children_system_microseconds) : json::null()}});
}

std::string describe(const Attribution& attribution) {
    const auto seconds = [](const std::optional<double>& value) { return value ? json::number(*value) : json::null(); };
    return json::object({{"aggregate_usable", json::boolean(attribution.aggregate_usable)}, {"reason", json::string(attribution.reason)},
                         {"elapsed_seconds", seconds(attribution.elapsed_seconds)}, {"elapsed_uncertainty_seconds", seconds(attribution.elapsed_uncertainty_seconds)},
                         {"aggregate_busy_seconds", seconds(attribution.aggregate_busy_seconds)}, {"aggregate_total_seconds", seconds(attribution.aggregate_total_seconds)},
                         {"tick_quantisation_seconds", seconds(attribution.tick_quantisation_seconds)},
                         {"selected_process_seconds", json::number(attribution.selected_process_seconds)},
                         {"selected_valid", json::number(uint64_t(attribution.selected_valid))}, {"selected_invalid", json::number(uint64_t(attribution.selected_invalid))},
                         {"selection_complete", json::boolean(attribution.selection_complete)},
                         {"collector_self_seconds", seconds(attribution.collector_self_seconds)}, {"collector_children_seconds", seconds(attribution.collector_children_seconds)},
                         {"residual_cores", attribution.residual_cores ? json::number(*attribution.residual_cores) : json::null()},
                         {"verdict", json::string("none: diagnostic attribution, not a quiet-guard or benchmark result")}});
}

// ---------------------------------------------------------------------------
// Collector

Collector::Collector(Selection selection) : selection_(std::move(selection)) {
    if (selection_.pids.size() > max_selected_processes) throw TelemetryError("too many selected processes (bounded at 64)");
    std::set<int32_t> distinct;
    for (const auto pid : selection_.pids) {
        if (pid < 0) throw TelemetryError("selected pid must be non-negative");
        if (!distinct.insert(pid).second) throw TelemetryError("selected pids must be distinct");
    }
    first_identity_.assign(selection_.pids.size(), std::nullopt);
    const auto environment = platform::environment();
    environment_status_ = environment.status;
    environment_detail_ = environment.detail; // also carries a partial failure under an observed status
    if (environment.status == Status::observed) {
        timebase_ = environment.timebase;
        tick_hz_ = environment.tick_hz;
        logical_cpus_ = environment.logical_cpus;
    }
}

Sample Collector::sample() {
    Sample sample;
    sample.sequence = ++sequence_;
    sample.timebase = timebase_;
    sample.tick_hz = tick_hz_;
    sample.logical_cpus = logical_cpus_;
    sample.environment_status = environment_status_;
    sample.environment_detail = environment_detail_;
    sample.span.before = platform::absolute_now();
    sample.continuous_before = platform::continuous_now();
    sample.realtime_nanoseconds = platform::realtime_nanoseconds();

    if (selection_.aggregate) sample.aggregate = platform::read_aggregate_ticks();
    else sample.aggregate.detail = "not selected";
    if (selection_.core_ticks) sample.core_ticks = platform::read_core_ticks();
    else sample.core_ticks.detail = "not selected";
    if (selection_.core_times) sample.core_times = platform::read_core_times();
    else sample.core_times.detail = "not selected";

    sample.processes.reserve(selection_.pids.size());
    for (size_t i = 0; i < selection_.pids.size(); ++i) {
        const int32_t pid = selection_.pids[i];
        ProcessObservation observation;
        observation.requested_pid = pid;
        observation.identity = platform::read_identity(pid);
        observation.identity.value.pid = pid;
        const auto rusage = platform::read_process_rusage(pid);
        observation.cpu.bracket = rusage.bracket;
        observation.cpu.status = rusage.status;
        observation.cpu.detail = rusage.detail;
        if (rusage.observed()) {
            observation.cpu.value = rusage.value.times;
            auto& identity = observation.identity.value;
            identity.has_abstime_start = true;
            identity.start_abstime = rusage.value.start_abstime;
            identity.executable_uuid = rusage.value.executable_uuid;
            identity.has_executable_uuid = std::any_of(identity.executable_uuid.begin(), identity.executable_uuid.end(), [](uint8_t b) { return b != 0; });
            if (!observation.identity.observed()) {
                // Birth is still established from the rusage read; keep the pidinfo failure visible.
                observation.identity.detail = "proc_pidinfo " + std::string(status_name(observation.identity.status)) +
                                              (observation.identity.detail.empty() ? std::string() : ": " + observation.identity.detail) + "; birth from rusage_info only";
                observation.identity.status = Status::observed;
            }
        }
        anchor_identity(first_identity_[i], observation);
        sample.processes.push_back(std::move(observation));
    }

    sample.self = platform::read_self();
    sample.self_task = platform::read_self_task();
    sample.continuous_after = platform::continuous_now();
    sample.span.after = platform::absolute_now();
    return sample;
}

// ---------------------------------------------------------------------------
// Portable stubs: every kernel reading is explicitly unavailable off Apple platforms.

#if !defined(__APPLE__)
namespace platform {
namespace {
template <class Value>
Reading<Value> unavailable_reading() {
    Reading<Value> reading;
    reading.bracket.before = absolute_now();
    reading.bracket.after = reading.bracket.before;
    reading.status = Status::unavailable;
    reading.detail = "Apple host telemetry is unavailable on this platform";
    return reading;
}
} // namespace
Environment environment() {
    Environment environment;
    environment.status = Status::unavailable;
    environment.detail = "Apple host telemetry is unavailable on this platform";
    return environment;
}
uint64_t absolute_now() noexcept {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
uint64_t continuous_now() noexcept { return absolute_now(); }
int64_t realtime_nanoseconds() noexcept {
    return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
Reading<CpuTicks> read_aggregate_ticks() { return unavailable_reading<CpuTicks>(); }
Reading<std::vector<CpuTicks>> read_core_ticks() { return unavailable_reading<std::vector<CpuTicks>>(); }
Reading<std::vector<CoreTimes>> read_core_times() { return unavailable_reading<std::vector<CoreTimes>>(); }
Reading<ProcessIdentity> read_identity(int32_t) { return unavailable_reading<ProcessIdentity>(); }
Reading<ProcessRusage> read_process_rusage(int32_t) { return unavailable_reading<ProcessRusage>(); }
Reading<SelfTimes> read_self() { return unavailable_reading<SelfTimes>(); }
Reading<SelfTaskTimes> read_self_task() { return unavailable_reading<SelfTaskTimes>(); }
} // namespace platform
#endif

} // namespace tokspur::telemetry
