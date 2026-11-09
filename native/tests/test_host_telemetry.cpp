#include "tokspur/host_telemetry.hpp"
#include "tokspur/json.hpp"

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#if defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

// Model-free checks of the owned host telemetry arithmetic: deltas, repeated
// (cached) counters, clock and counter regressions, identity changes, missing
// permissions and incomplete observations. Nothing here loads a model, touches
// Metal or declares a host quiet. The Apple section performs a few bounded live
// reads of this process, pid 1, a non-existent pid and a reaped child.

namespace {
using namespace tokspur::telemetry;
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template <class Action>
void refuses(Action&& action) {
    ++checks;
    try { action(); } catch (const TelemetryError&) { return; }
    throw std::runtime_error("invalid telemetry input accepted");
}
bool near(double a, double b, double tolerance = 1e-9) { return std::fabs(a - b) <= tolerance; }
bool mentions(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

Reading<CpuTicks> ticks_reading(uint64_t before, uint64_t after, CpuTicks value, Status status = Status::observed) {
    Reading<CpuTicks> reading;
    reading.bracket = Bracket{before, after};
    reading.status = status;
    reading.value = value;
    if (status != Status::observed) reading.detail = "synthetic failure";
    return reading;
}
Reading<std::vector<CpuTicks>> cores_reading(uint64_t before, uint64_t after, std::vector<CpuTicks> cores, Status status = Status::observed) {
    Reading<std::vector<CpuTicks>> reading;
    reading.bracket = Bracket{before, after};
    reading.status = status;
    reading.value = std::move(cores);
    return reading;
}
Reading<std::vector<CoreTimes>> times_reading(uint64_t before, uint64_t after, std::vector<CoreTimes> cores, Status status = Status::observed) {
    Reading<std::vector<CoreTimes>> reading;
    reading.bracket = Bracket{before, after};
    reading.status = status;
    reading.value = std::move(cores);
    return reading;
}
ProcessIdentity identity_of(int32_t pid, uint64_t start_abstime, uint8_t uuid_byte, uint64_t wall_seconds = 0) {
    ProcessIdentity identity;
    identity.pid = pid;
    if (start_abstime) { identity.has_abstime_start = true; identity.start_abstime = start_abstime; }
    if (uuid_byte) { identity.has_executable_uuid = true; identity.executable_uuid[0] = uuid_byte; }
    if (wall_seconds) { identity.has_wall_start = true; identity.start_wall_seconds = wall_seconds; identity.start_wall_microseconds = 42; }
    return identity;
}
ProcessObservation observation(int32_t pid, uint64_t before, uint64_t after, Status cpu_status, ProcessCpuTimes times, uint64_t start_abstime, uint8_t uuid_byte) {
    ProcessObservation result;
    result.requested_pid = pid;
    result.identity.bracket = Bracket{before, after};
    result.identity.status = cpu_status == Status::not_found ? Status::not_found : Status::observed;
    result.identity.value = identity_of(pid, cpu_status == Status::observed ? start_abstime : 0, cpu_status == Status::observed ? uuid_byte : 0, 1700000000);
    result.cpu.bracket = Bracket{before + 2, after + 2};
    result.cpu.status = cpu_status;
    if (cpu_status == Status::observed) result.cpu.value = times;
    else result.cpu.detail = "synthetic failure";
    return result;
}
Reading<SelfTimes> self_reading(uint64_t before, uint64_t after, SelfTimes value, Status status = Status::observed) {
    Reading<SelfTimes> reading;
    reading.bracket = Bracket{before, after};
    reading.status = status;
    reading.value = value;
    return reading;
}

const Timebase apple_silicon{125, 3}; // 24 MHz: the timebase this hardware has reported; tests never assume it at runtime

void conversions() {
    require(ticks_to_nanoseconds(0, apple_silicon) == 0, "zero ticks");
    require(ticks_to_nanoseconds(3, apple_silicon) == 125, "exact 125/3 conversion");
    require(ticks_to_nanoseconds(7, apple_silicon) == 291, "truncating fractional nanosecond");
    require(ticks_to_nanoseconds(24'000'000, apple_silicon) == 1'000'000'000, "24 MHz second");
    require(ticks_to_nanoseconds(123'456'789, Timebase{1, 1}) == 123'456'789, "unit timebase");
    require(ticks_to_nanoseconds(std::numeric_limits<uint64_t>::max(), Timebase{1, 1}) == std::numeric_limits<uint64_t>::max(), "unit timebase maximum without overflow");
    require(near(ticks_to_seconds(24'000'000, apple_silicon), 1.0), "seconds conversion");
    refuses([] { static_cast<void>(ticks_to_nanoseconds(1, Timebase{0, 1})); });
    refuses([] { static_cast<void>(ticks_to_nanoseconds(1, Timebase{1, 0})); });
    refuses([] { static_cast<void>(ticks_to_nanoseconds(std::numeric_limits<uint64_t>::max(), apple_silicon)); });
    refuses([] { static_cast<void>(ticks_to_nanoseconds(std::numeric_limits<uint64_t>::max() / 2, Timebase{3, 1})); });
    require(!Timebase{}.valid() && apple_silicon.valid(), "timebase validity");
}

void brackets() {
    const Bracket ordered{100, 110};
    require(ordered.ordered() && ordered.width() == 10 && ordered.midpoint() == 105, "ordered bracket");
    const Bracket disordered{110, 100};
    require(!disordered.ordered() && disordered.width() == 0 && disordered.midpoint() == 110, "disordered bracket is reported, not repaired");

    auto elapsed = elapsed_between(Bracket{100, 110}, Bracket{200, 210});
    require(elapsed.status == DeltaStatus::valid && elapsed.nominal_ticks == 100 && elapsed.minimum_ticks == 90 && elapsed.maximum_ticks == 110, "elapsed bounds");
    elapsed = elapsed_between(Bracket{100, 110}, Bracket{105, 120});
    require(elapsed.status == DeltaStatus::valid && elapsed.nominal_ticks == 7 && elapsed.minimum_ticks == 0 && elapsed.maximum_ticks == 20, "overlapping brackets have zero minimum elapsed");
    require(elapsed_between(Bracket{100, 110}, Bracket{100, 110}).status == DeltaStatus::no_elapsed, "identical brackets");
    require(elapsed_between(Bracket{200, 210}, Bracket{100, 110}).status == DeltaStatus::no_elapsed, "reversed order");
    require(elapsed_between(Bracket{110, 100}, Bracket{200, 210}).status == DeltaStatus::bracket_disorder, "earlier disordered");
    require(elapsed_between(Bracket{100, 110}, Bracket{210, 200}).status == DeltaStatus::bracket_disorder, "later disordered");
}

void tick_deltas() {
    const auto a = ticks_reading(1000, 1010, CpuTicks{100, 50, 1000, 0});
    const auto b = ticks_reading(2000, 2010, CpuTicks{130, 60, 1900, 0});
    auto delta = tick_delta(a, b);
    require(delta.status == DeltaStatus::valid && delta.changed, "ordinary tick delta");
    require(delta.user == 30 && delta.system == 10 && delta.idle == 900 && delta.nice == 0 && delta.busy() == 40 && delta.total() == 940, "tick delta values");
    require(delta.elapsed.nominal_ticks == 1000 && delta.earlier == a.value && delta.later == b.value, "raw readings retained");

    // Repeated counters: a cached aggregate read is valid but unchanged, never a zero substitute.
    const auto c = ticks_reading(3000, 3010, CpuTicks{130, 60, 1900, 0});
    delta = tick_delta(b, c);
    require(delta.status == DeltaStatus::valid && !delta.changed && delta.busy() == 0 && delta.total() == 0, "repeated counters are flagged unchanged");
    require(delta.elapsed.nominal_ticks == 1000, "repeated counters still have elapsed time");

    // Regression and wrap are reported, with raw values, and never clamped.
    const auto r = ticks_reading(4000, 4010, CpuTicks{129, 60, 1900, 0});
    delta = tick_delta(c, r);
    require(delta.status == DeltaStatus::regression && delta.changed && delta.user == 0 && delta.later.user == 129 && delta.earlier.user == 130, "counter regression");
    const auto wrapped = ticks_reading(5000, 5010, CpuTicks{129, 60, 5, 0});
    require(tick_delta(r, wrapped).status == DeltaStatus::regression, "32-bit wrap reports regression rather than unwrapping");
    require(tick_delta(r, ticks_reading(5000, 5010, CpuTicks{0xffffffffu, 60, 1900, 0})).status == DeltaStatus::valid, "large forward jump is valid");

    // Missing permissions or unavailable API on either side.
    const auto denied = ticks_reading(6000, 6010, CpuTicks{1, 1, 1, 1}, Status::permission_denied);
    delta = tick_delta(r, denied);
    require(delta.status == DeltaStatus::unavailable && !delta.changed && delta.busy() == 0, "permission-denied side yields unavailable delta");
    require(tick_delta(denied, r).status == DeltaStatus::unavailable, "unavailable earlier side");
    require(tick_delta(ticks_reading(1, 2, CpuTicks{}, Status::error), ticks_reading(3, 4, CpuTicks{}, Status::not_found)).status == DeltaStatus::unavailable, "two failures");

    // Clock problems.
    require(tick_delta(r, ticks_reading(7010, 7000, CpuTicks{200, 70, 2000, 0})).status == DeltaStatus::bracket_disorder, "disordered bracket");
    require(tick_delta(r, ticks_reading(4000, 4010, CpuTicks{130, 61, 1901, 0})).status == DeltaStatus::no_elapsed, "counters advanced without elapsed time");
    require(tick_delta(r, ticks_reading(4000, 4010, CpuTicks{100, 61, 1901, 0})).status == DeltaStatus::regression, "regression outranks no_elapsed");
    require(tick_delta(ticks_reading(4010, 4000, CpuTicks{}), ticks_reading(5000, 5010, CpuTicks{}, Status::unavailable)).status == DeltaStatus::unavailable, "unavailable outranks bracket disorder");
}

void core_deltas() {
    const auto a = cores_reading(1000, 1020, {CpuTicks{10, 10, 100, 0}, CpuTicks{20, 20, 200, 0}});
    const auto b = cores_reading(2000, 2020, {CpuTicks{12, 10, 190, 0}, CpuTicks{20, 20, 200, 0}});
    auto deltas = core_tick_deltas(a, b);
    require(deltas.status == DeltaStatus::valid && deltas.cores.size() == 2, "per-core deltas");
    require(deltas.cores[0].changed && deltas.cores[0].user == 2 && deltas.cores[0].idle == 90 && !deltas.cores[1].changed, "per-core values");
    require(deltas.changed_cores() == 1 && deltas.regressed_cores() == 0, "per-core counts");
    require(deltas.cores[0].elapsed.nominal_ticks == 1000, "per-core deltas share the one read bracket");
    require(core_tick_deltas(a, cores_reading(2000, 2020, {CpuTicks{}, CpuTicks{}, CpuTicks{}})).status == DeltaStatus::mismatch, "processor count mismatch");
    require(core_tick_deltas(cores_reading(1, 2, {}, Status::error), b).status == DeltaStatus::unavailable, "unavailable per-core reading");
    deltas = core_tick_deltas(b, cores_reading(3000, 3020, {CpuTicks{11, 10, 190, 0}, CpuTicks{20, 20, 201, 0}}));
    require(deltas.status == DeltaStatus::valid && deltas.cores[0].status == DeltaStatus::regression && deltas.regressed_cores() == 1 && deltas.cores[1].status == DeltaStatus::valid, "one regressed core");

    const auto t0 = times_reading(1000, 1020, {CoreTimes{100, 50, 1000, 0, 0, 0}, CoreTimes{5, 5, 5000, 0, 0, 0}});
    const auto t1 = times_reading(2000, 2020, {CoreTimes{150, 60, 1900, 0, 0, 0}, CoreTimes{5, 5, 5000, 0, 0, 0}});
    auto times = core_time_deltas(t0, t1);
    require(times.status == DeltaStatus::valid && times.elapsed.nominal_ticks == 1000 && times.cores.size() == 2, "per-core time deltas");
    require(times.cores[0].changed && times.cores[0].user_abstime == 50 && times.cores[0].system_abstime == 10 && times.cores[0].idle_abstime == 900, "per-core time values");
    require(!times.cores[1].changed && times.cores[1].status == DeltaStatus::valid, "unchanged core time");
    times = core_time_deltas(t1, times_reading(3000, 3020, {CoreTimes{150, 60, 1899, 0, 0, 0}, CoreTimes{5, 5, 5000, 0, 0, 0}}));
    require(times.cores[0].status == DeltaStatus::regression && times.cores[0].user_abstime == 0, "per-core time regression");
    require(core_time_deltas(t0, times_reading(2000, 2020, {CoreTimes{}})).status == DeltaStatus::mismatch, "per-core time mismatch");
    require(core_time_deltas(t0, times_reading(2000, 2020, {}, Status::unavailable)).status == DeltaStatus::unavailable, "per-core time unavailable");
}

void identities() {
    const auto a = identity_of(42, 777, 1);
    require(same_birth(a, identity_of(42, 777, 1)), "same birth");
    require(!same_birth(a, identity_of(43, 777, 1)), "different pid");
    require(!same_birth(a, identity_of(42, 778, 1)), "different start");
    require(!same_birth(a, identity_of(42, 777, 2)), "different executable uuid");
    require(!same_birth(identity_of(42, 0, 0), identity_of(42, 0, 0)), "nothing comparable is not the same birth");
    require(!same_birth(identity_of(42, 0, 0, 1700000000), identity_of(42, 777, 0)), "disjoint components are not comparable");
    require(same_birth(identity_of(42, 0, 0, 1700000000), identity_of(42, 777, 0, 1700000000)), "shared wall start suffices");
    ProcessIdentity wall_differs = identity_of(42, 777, 1, 1700000000);
    wall_differs.start_wall_microseconds = 43;
    require(!same_birth(identity_of(42, 777, 1, 1700000000), wall_differs), "wall microseconds differ");

    std::optional<ProcessIdentity> anchor;
    auto first = observation(42, 100, 110, Status::observed, ProcessCpuTimes{10, 5, 1, 0, 0}, 777, 1);
    anchor_identity(anchor, first);
    require(anchor.has_value() && anchor->start_abstime == 777 && first.cpu.observed(), "first observation anchors");
    auto same = observation(42, 200, 210, Status::observed, ProcessCpuTimes{20, 6, 1, 0, 0}, 777, 1);
    anchor_identity(anchor, same);
    require(same.cpu.observed() && same.identity.observed(), "same birth stays observed");
    auto replaced = observation(42, 300, 310, Status::observed, ProcessCpuTimes{1, 1, 0, 0, 0}, 999, 1);
    anchor_identity(anchor, replaced);
    require(replaced.identity.status == Status::identity_changed && replaced.cpu.status == Status::identity_changed, "reused pid is marked identity_changed");
    require(anchor->start_abstime == 777, "anchor is not moved to the impostor");
    auto missing = observation(42, 400, 410, Status::not_found, ProcessCpuTimes{}, 0, 0);
    anchor_identity(anchor, missing);
    require(missing.identity.status == Status::not_found && missing.cpu.status == Status::not_found, "unobserved identity is untouched");
    std::optional<ProcessIdentity> empty;
    anchor_identity(empty, missing);
    require(!empty.has_value(), "no anchor from an unobserved identity");
    ProcessObservation componentless;
    componentless.requested_pid = 42;
    componentless.identity.status = Status::observed;
    componentless.identity.value.pid = 42;
    anchor_identity(empty, componentless);
    require(!empty.has_value(), "no anchor without any birth component");
}

void process_deltas() {
    const auto a = observation(42, 1000, 1010, Status::observed, ProcessCpuTimes{1000, 500, 100, 7, 3}, 777, 1);
    const auto b = observation(42, 2000, 2010, Status::observed, ProcessCpuTimes{1300, 600, 150, 9, 3}, 777, 1);
    auto delta = process_delta(a, b);
    require(delta.status == DeltaStatus::valid && delta.pid == 42 && delta.user_abstime == 300 && delta.system_abstime == 100 && delta.runnable_abstime == 50, "process delta");
    require(delta.elapsed.status == DeltaStatus::valid && delta.elapsed.nominal_ticks == 1000, "process elapsed uses the cpu brackets");
    require(delta.earlier_status == Status::observed && delta.later_status == Status::observed, "raw statuses retained");

    const auto impostor = observation(42, 3000, 3010, Status::observed, ProcessCpuTimes{10, 5, 1, 0, 0}, 999, 1);
    delta = process_delta(b, impostor);
    require(delta.status == DeltaStatus::identity_changed && delta.user_abstime == 0, "different birth is identity_changed, not regression");
    auto flagged = impostor;
    flagged.cpu.status = Status::identity_changed;
    require(process_delta(b, flagged).status == DeltaStatus::identity_changed, "collector-flagged identity change propagates");
    require(process_delta(a, observation(43, 2000, 2010, Status::observed, ProcessCpuTimes{}, 777, 1)).status == DeltaStatus::identity_changed, "requested pid mismatch");

    const auto denied = observation(42, 2000, 2010, Status::permission_denied, ProcessCpuTimes{}, 0, 0);
    delta = process_delta(a, denied);
    require(delta.status == DeltaStatus::unavailable && delta.later_status == Status::permission_denied && delta.user_abstime == 0, "missing permission is unavailable, not zero");
    const auto gone = observation(42, 2000, 2010, Status::not_found, ProcessCpuTimes{}, 0, 0);
    delta = process_delta(a, gone);
    require(delta.status == DeltaStatus::unavailable && delta.later_status == Status::not_found, "exited process is unavailable");
    require(process_delta(gone, a).status == DeltaStatus::unavailable, "incomplete earlier observation");

    const auto backwards = observation(42, 2000, 2010, Status::observed, ProcessCpuTimes{999, 500, 100, 7, 3}, 777, 1);
    require(process_delta(a, backwards).status == DeltaStatus::regression, "process counter regression");
    require(process_delta(a, observation(42, 1000, 1010, Status::observed, ProcessCpuTimes{1001, 500, 100, 7, 3}, 777, 1)).status == DeltaStatus::no_elapsed, "process no elapsed");
    require(process_delta(a, observation(42, 2010, 2000, Status::observed, ProcessCpuTimes{1001, 500, 100, 7, 3}, 777, 1)).status == DeltaStatus::bracket_disorder, "process bracket disorder");
}

void self_deltas() {
    const auto a = self_reading(1000, 1010, SelfTimes{100, 50, 10, 5, 1, 1});
    const auto b = self_reading(2000, 2010, SelfTimes{160, 70, 12, 5, 2, 3});
    auto delta = self_delta(a, b);
    require(delta.status == DeltaStatus::valid && delta.user_microseconds == 60 && delta.system_microseconds == 20 && delta.children_user_microseconds == 2 && delta.children_system_microseconds == 0, "self delta");
    require(self_delta(a, self_reading(2000, 2010, SelfTimes{160, 70, 9, 5, 2, 3})).status == DeltaStatus::regression, "children regression");
    require(self_delta(a, self_reading(2000, 2010, SelfTimes{}, Status::error)).status == DeltaStatus::unavailable, "self unavailable");
    require(self_delta(a, self_reading(1000, 1010, SelfTimes{101, 50, 10, 5, 1, 1})).status == DeltaStatus::no_elapsed, "self no elapsed");
}

void attribution() {
    constexpr uint64_t ticks_per_second = 24'000'000; // only for building synthetic values; attribute() receives the timebase explicitly
    const uint32_t tick_hz = 100, cpus = 18;
    TickDelta aggregate;
    aggregate.status = DeltaStatus::valid;
    aggregate.changed = true;
    aggregate.elapsed = Elapsed{DeltaStatus::valid, ticks_per_second * 6 / 10, ticks_per_second * 6 / 10 - 10'000, ticks_per_second * 6 / 10 + 10'000};
    aggregate.user = 100; aggregate.system = 20; aggregate.idle = 960; aggregate.nice = 0; // 1.2 s busy, 10.8 s total

    ProcessDelta process;
    process.pid = 42; process.status = DeltaStatus::valid; process.user_abstime = ticks_per_second / 10; process.system_abstime = 0; // 0.1 s
    SelfDelta self;
    self.status = DeltaStatus::valid; self.user_microseconds = 50'000; self.system_microseconds = 10'000; self.children_user_microseconds = 1'000;

    auto result = attribute(aggregate, {process}, self, apple_silicon, tick_hz, cpus);
    require(result.aggregate_usable && result.reason.empty(), "usable attribution");
    require(near(result.elapsed_seconds.value_or(-1), 0.6) && near(result.elapsed_uncertainty_seconds.value_or(-1), 10'000.0 / ticks_per_second), "elapsed and uncertainty from brackets (nanosecond truncation inside tolerance)");
    require(near(result.aggregate_busy_seconds.value_or(-1), 1.2) && near(result.aggregate_total_seconds.value_or(-1), 10.8), "aggregate seconds");
    require(near(result.tick_quantisation_seconds.value_or(-1), 0.18), "quantisation bound is processors over tick rate");
    require(near(result.selected_process_seconds, 0.1) && result.selected_valid == 1 && result.selected_invalid == 0 && result.selection_complete, "selected process seconds");
    require(near(result.collector_self_seconds.value_or(-1), 0.06) && near(result.collector_children_seconds.value_or(-1), 0.001), "collector overhead reported beside the residual");
    require(result.residual_cores.has_value() && near(*result.residual_cores, (1.2 - 0.1) / 0.6), "residual cores");

    // A negative residual is retained, never clamped to zero.
    ProcessDelta heavy = process;
    heavy.user_abstime = ticks_per_second * 15 / 10; // 1.5 s > 1.2 s aggregate busy
    result = attribute(aggregate, {heavy}, self, apple_silicon, tick_hz, cpus);
    require(result.residual_cores.has_value() && near(*result.residual_cores, (1.2 - 1.5) / 0.6) && *result.residual_cores < 0, "negative residual retained");

    // An unchanged (cached or stale) aggregate withholds the residual instead of reporting zero.
    TickDelta stale = aggregate;
    stale.changed = false; stale.user = stale.system = stale.idle = stale.nice = 0;
    result = attribute(stale, {process}, self, apple_silicon, tick_hz, cpus);
    require(!result.aggregate_usable && !result.residual_cores.has_value() && mentions(result.reason, "unchanged"), "stale aggregate withholds residual");
    require(near(result.elapsed_seconds.value_or(-1), 0.6) && near(result.selected_process_seconds, 0.1), "stale attribution still reports what was measured");

    TickDelta regressed = aggregate;
    regressed.status = DeltaStatus::regression;
    result = attribute(regressed, {process}, self, apple_silicon, tick_hz, cpus);
    require(!result.aggregate_usable && !result.residual_cores.has_value() && mentions(result.reason, "regression"), "aggregate regression withholds residual");
    TickDelta unavailable = aggregate;
    unavailable.status = DeltaStatus::unavailable;
    require(!attribute(unavailable, {process}, self, apple_silicon, tick_hz, cpus).residual_cores.has_value(), "unavailable aggregate withholds residual");

    result = attribute(aggregate, {process}, self, Timebase{}, tick_hz, cpus);
    require(!result.aggregate_usable && mentions(result.reason, "timebase") && !result.residual_cores.has_value(), "invalid timebase refuses attribution");
    result = attribute(aggregate, {process}, self, apple_silicon, 0, cpus);
    require(!result.aggregate_usable && mentions(result.reason, "tick rate"), "missing tick rate refuses attribution");
    result = attribute(aggregate, {process}, self, apple_silicon, tick_hz, 0);
    require(result.aggregate_usable && !result.tick_quantisation_seconds.has_value() && mentions(result.reason, "processor count"), "unknown processor count is stated");
    require(mentions(describe(result), "\"tick_quantisation_seconds\":null"), "unknown quantisation bound describes as null, not zero");

    // Incomplete selections give a lower bound and say so; the residual remains available but flagged.
    ProcessDelta denied;
    denied.pid = 7; denied.status = DeltaStatus::unavailable; denied.later_status = Status::permission_denied;
    result = attribute(aggregate, {process, denied}, self, apple_silicon, tick_hz, cpus);
    require(result.selected_valid == 1 && result.selected_invalid == 1 && !result.selection_complete && mentions(result.reason, "lower bound"), "incomplete selection flagged");
    require(result.aggregate_usable && result.residual_cores.has_value() && near(*result.residual_cores, (1.2 - 0.1) / 0.6), "incomplete selection keeps the raw residual");
    ProcessDelta changed = denied;
    changed.status = DeltaStatus::identity_changed;
    require(!attribute(aggregate, {changed}, self, apple_silicon, tick_hz, cpus).selection_complete, "identity change counts as incomplete");

    SelfDelta no_self;
    no_self.status = DeltaStatus::unavailable;
    result = attribute(aggregate, {process}, no_self, apple_silicon, tick_hz, cpus);
    require(!result.collector_self_seconds.has_value() && mentions(result.reason, "collector self delta") && result.residual_cores.has_value(), "collector overhead unavailable is stated");

    TickDelta no_aggregate;
    no_aggregate.status = DeltaStatus::unavailable;
    result = attribute(no_aggregate, {process}, no_self, apple_silicon, tick_hz, cpus);
    require(!result.elapsed_seconds && !result.aggregate_busy_seconds && !result.aggregate_total_seconds && !result.collector_self_seconds && !result.residual_cores,
            "unavailable deltas leave their seconds unset");
    require(mentions(describe(result), "\"elapsed_seconds\":null") && mentions(describe(result), "\"collector_self_seconds\":null") && !mentions(describe(result), "_seconds\":0,"),
            "unset seconds describe as null, never zero");
}

void json_text() {
    require(json::string("a\"b\\c\n\x01") == "\"a\\\"b\\\\c\\u000a\\u0001\"", "JSON escapes");
    require(json::string("caf\xc3\xa9 \xf0\x9f\xa6\x89") == "\"caf\xc3\xa9 \xf0\x9f\xa6\x89\"", "valid UTF-8 passes through");
    require(json::string(std::string("\xff\x41", 2)) == "\"\\u00ffA\"", "invalid UTF-8 bytes are escaped individually");
    require(json::string(std::string("\xed\xa0\x80", 3)) == "\"\\u00ed\\u00a0\\u0080\"", "encoded surrogate is invalid UTF-8");
    require(json::number(std::numeric_limits<double>::quiet_NaN()) == "null" && json::number(std::numeric_limits<double>::infinity()) == "null", "non-finite numbers are null");
    require(json::number(1.5) == "1.5" && json::number(-0.25) == "-0.25" && json::number(0.0) == "0", "finite numbers");
    require(json::number(std::numeric_limits<uint64_t>::max()) == "18446744073709551615" && json::number(int64_t(-7)) == "-7", "integers");
    require(json::object({{"a", "1"}, {"b", "\"x\""}}) == "{\"a\":1,\"b\":\"x\"}" && json::object({}) == "{}", "objects");
    require(json::array({"1", "2"}) == "[1,2]" && json::array({}) == "[]", "arrays");

    const auto described = describe(Attribution{});
    const auto parsed = tokspur::parse_json(described);
    require(parsed.find("residual_cores")->is_null() && !parsed.find("aggregate_usable")->boolean(), "default attribution describes a withheld residual");
    require(mentions(std::string(parsed.find("verdict")->string()), "not a quiet-guard"), "attribution carries no verdict");

    Sample sample;
    sample.aggregate.status = Status::permission_denied;
    sample.aggregate.detail = "synthetic";
    sample.aggregate.value = CpuTicks{1, 2, 3, 4};
    sample.processes.push_back(observation(42, 1, 2, Status::observed, ProcessCpuTimes{1, 2, 3, 4, 5}, 777, 1));
    sample.processes.push_back(observation(43, 1, 2, Status::not_found, ProcessCpuTimes{}, 0, 0));
    const auto sample_json = tokspur::parse_json(describe(sample));
    const auto* aggregate = sample_json.find("aggregate_scheduler_ticks");
    require(aggregate->find("status")->string() == "permission_denied" && aggregate->find("value")->is_null(), "unobserved readings describe null values, not zeros");
    require(aggregate->find("detail")->string() == "synthetic", "detail is carried");
    const auto& processes = sample_json.find("processes")->array();
    require(processes.size() == 2 && processes[0].find("cpu")->find("value")->find("runnable_abstime")->unsigned_integer() == 3, "process cpu described");
    require(processes[0].find("identity")->find("value")->find("executable_uuid")->string() == "01000000000000000000000000000000", "uuid hex");
    require(processes[1].find("cpu")->find("status")->string() == "not_found" && processes[1].find("cpu")->find("value")->is_null(), "missing process described as not found");
    require(sample_json.find("core_scheduler_ticks")->find("status")->string() == "unavailable", "default per-core reading is unavailable");

    TickDelta delta;
    delta.status = DeltaStatus::regression;
    delta.earlier = CpuTicks{5, 5, 5, 0};
    delta.later = CpuTicks{4, 5, 5, 0};
    delta.changed = true;
    const auto delta_json = tokspur::parse_json(describe(delta, apple_silicon));
    require(delta_json.find("status")->string() == "regression" && delta_json.find("delta_scheduler_ticks")->is_null() &&
            delta_json.find("later_scheduler_ticks")->find("user")->unsigned_integer() == 4, "regression described with raw values and null delta");
    require(tokspur::parse_json(describe(ProcessDelta{}, apple_silicon)).find("cpu_seconds")->is_null(), "unavailable process delta has null seconds");
    require(tokspur::parse_json(describe(SelfDelta{}, apple_silicon)).find("status")->string() == "unavailable", "self delta described");
    require(tokspur::parse_json(describe(CoreTickDeltas{}, apple_silicon)).find("processors")->unsigned_integer() == 0, "core deltas described");
    require(tokspur::parse_json(describe(CoreTimeDeltas{}, apple_silicon)).find("status")->string() == "unavailable", "core time deltas described");
    require(tokspur::parse_json(describe(Elapsed{}, Timebase{})).find("nominal_seconds")->is_null(), "invalid timebase gives null seconds");
}

void collector_selection() {
    refuses([] { Collector collector(Selection{std::vector<int32_t>(max_selected_processes + 1, 0), true, true, true}); });
    refuses([] { Collector collector(Selection{{-1}, true, true, true}); });
    refuses([] { Collector collector(Selection{{5, 5}, true, true, true}); });
    Collector collector(Selection{{}, false, false, false});
    require(collector.selection().pids.empty(), "selection retained");
    auto first = collector.sample();
    auto second = collector.sample();
    require(first.sequence == 1 && second.sequence == 2, "sample sequence");
    require(first.span.ordered() && second.span.ordered() && second.span.before >= first.span.after, "sample spans are ordered and monotonic");
    require(!first.aggregate.observed() && first.aggregate.detail == "not selected" && !first.core_ticks.observed() && !first.core_times.observed(), "unselected sources are explicitly unavailable");
    require(tick_delta(first.aggregate, second.aggregate).status == DeltaStatus::unavailable, "unselected deltas are unavailable");
#if !defined(__APPLE__)
    Collector everything(Selection{{1}, true, true, true});
    const auto sample = everything.sample();
    require(!sample.timebase.valid() && sample.aggregate.status == Status::unavailable && sample.self.status == Status::unavailable, "portable stubs are unavailable, not zero");
    require(sample.processes.size() == 1 && sample.processes[0].cpu.status == Status::unavailable && sample.processes[0].identity.status == Status::unavailable, "portable process reads are unavailable");
    require(!attribute(tick_delta(sample.aggregate, sample.aggregate), {}, SelfDelta{}, sample.timebase, sample.tick_hz, sample.logical_cpus).aggregate_usable, "portable attribution is withheld");
#endif
}

#if defined(__APPLE__)
void live_apple() {
    const int32_t self_pid = int32_t(getpid());
    Collector collector(Selection{{self_pid, 1, 2147483647, 0}, true, true, true});
    const auto first = collector.sample();
    require(first.timebase.valid() && first.tick_hz > 0 && first.logical_cpus > 0, "environment observed");
    require(first.span.ordered() && first.continuous_after >= first.continuous_before && first.realtime_nanoseconds > 0, "sample clocks");

    require(first.aggregate.observed() && first.aggregate.bracket.ordered() && first.aggregate.value.total() > 0, "aggregate ticks observed");
    require(first.core_ticks.observed() && !first.core_ticks.value.empty() && first.core_ticks.value.size() <= max_processors, "per-core ticks observed");
    require(first.core_times.status != Status::identity_changed && (first.core_times.observed() || !first.core_times.detail.empty()), "per-core times have an explicit state");
    if (first.core_times.observed()) require(first.core_times.value.size() == first.core_ticks.value.size(), "per-core time processors match tick processors");

    const auto& self = first.processes[0];
    require(self.identity.observed() && self.identity.value.pid == self_pid && !self.identity.value.command.empty() && self.identity.value.uid == getuid(), "own identity");
    require(self.identity.value.has_wall_start && self.identity.value.has_abstime_start && self.identity.value.start_abstime <= first.span.before, "own birth precedes the sample");
    require(self.cpu.observed() && self.cpu.value.user_abstime + self.cpu.value.system_abstime > 0, "own cpu time observed");

    const auto& launchd = first.processes[1];
    // launchd is always root: its identity and birth must be readable through the short flavour and the
    // process table, and its CPU accounting must be refused to an unprivileged caller.
    require(launchd.identity.observed() && launchd.identity.value.pid == 1 && launchd.identity.value.uid == 0 && launchd.identity.value.has_wall_start,
            "pid 1 identity and birth observed across users");
    if (geteuid() != 0) {
        require(launchd.cpu.status == Status::permission_denied || launchd.cpu.status == Status::error, "other-user cpu is refused explicitly");
        require(launchd.cpu.value.user_abstime == 0 && launchd.cpu.value.system_abstime == 0 && !launchd.cpu.detail.empty(), "refused cpu carries no substitute value");
    }

    const auto& absent = first.processes[2];
    require(absent.identity.status == Status::not_found && absent.cpu.status == Status::not_found, "non-existent pid is not_found");
    require(!absent.identity.detail.empty() && absent.cpu.value.user_abstime == 0, "not_found carries detail and no value");
    const auto& kernel = first.processes[3];
    require(kernel.cpu.observed() ? kernel.identity.value.pid == 0 : !kernel.cpu.detail.empty(), "pid 0 has an explicit state");

    require(first.self.observed() && first.self_task.observed() && first.self_task.value.total_user_abstime + first.self_task.value.total_system_abstime > 0, "collector self accounting");

    const auto second = collector.sample();
    const auto aggregate = tick_delta(first.aggregate, second.aggregate);
    require(aggregate.status == DeltaStatus::valid && aggregate.elapsed.nominal_ticks > 0, "live aggregate delta is valid (changed or cached)");
    const auto cores = core_tick_deltas(first.core_ticks, second.core_ticks);
    require(cores.status == DeltaStatus::valid && cores.regressed_cores() == 0, "live per-core delta");
    const auto own = process_delta(first.processes[0], second.processes[0]);
    require(own.status == DeltaStatus::valid, "live own process delta");
    const auto overhead = self_delta(first.self, second.self);
    require(overhead.status == DeltaStatus::valid, "live self delta");
    const auto result = attribute(aggregate, {own}, overhead, first.timebase, first.tick_hz, first.logical_cpus);
    require(result.residual_cores.has_value() == result.aggregate_usable && result.aggregate_usable == aggregate.changed, "residual only with a changed aggregate");
    require(near(result.tick_quantisation_seconds.value_or(-1), double(first.logical_cpus) / double(first.tick_hz)), "live quantisation bound");
    const auto parsed = tokspur::parse_json(describe(first));
    require(parsed.find("aggregate_scheduler_ticks")->find("status")->string() == "observed" && parsed.find("processes")->array().size() == 4, "live sample describes as JSON");

    // A reaped child becomes not_found; its identity anchor never yields a false match.
    const pid_t child = fork();
    require(child >= 0, "fork");
    if (child == 0) { for (;;) pause(); }
    Collector watcher(Selection{{int32_t(child)}, false, false, false});
    const auto alive = watcher.sample();
    require(alive.processes[0].identity.observed() && alive.processes[0].identity.value.parent_pid == uint32_t(getpid()), "child identity observed");
    require(alive.processes[0].cpu.observed() && alive.processes[0].identity.value.has_abstime_start, "same-user child cpu observed");
    kill(child, SIGKILL);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) { if (errno != EINTR) throw std::runtime_error("waitpid"); }
    const auto gone = watcher.sample();
    require(gone.processes[0].identity.status == Status::not_found && gone.processes[0].cpu.status == Status::not_found, "reaped child is not_found");
    const auto child_delta = process_delta(alive.processes[0], gone.processes[0]);
    require(child_delta.status == DeltaStatus::unavailable && child_delta.later_status == Status::not_found, "reaped child delta is unavailable");
    std::cout << "Live Apple states: core_times=" << status_name(first.core_times.status) << " pid1_cpu=" << status_name(launchd.cpu.status)
              << " pid0_cpu=" << status_name(kernel.cpu.status) << " aggregate_changed_between_adjacent_samples=" << (aggregate.changed ? "true" : "false") << '\n';
}
#endif
} // namespace

int main() {
    try {
        conversions();
        brackets();
        tick_deltas();
        core_deltas();
        identities();
        process_deltas();
        self_deltas();
        attribution();
        json_text();
        collector_selection();
#if defined(__APPLE__)
        live_apple();
#endif
        std::cout << "Owned host telemetry checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned host telemetry check failed: " << error.what() << '\n';
        return 1;
    }
}
