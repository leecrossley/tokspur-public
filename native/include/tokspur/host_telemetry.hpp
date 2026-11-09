#pragma once

// Independently authored, bounded, CPU-only host telemetry.
//
// The collector reads raw aggregate and per-processor CPU state counters and
// the CPU time of explicitly selected processes, bracketing every kernel read
// with two monotonic timestamps. It never initialises Metal, queries a GPU
// device, loads weights, launches inference or signals another process, and it
// never declares a host quiet or a benchmark passing: every derived quantity is
// reported raw, with its uncertainty and its unavailable/error state.
//
// Units (also repeated on each field name in the JSON description):
//   *_ticks           mach_absolute_time() ticks. Convert with `Timebase`:
//                     nanoseconds = ticks * numer / denom. The timebase is read
//                     from the kernel; callers must not assume nanoseconds.
//                     mach_absolute_time does not advance during system sleep.
//   *_continuous      mach_continuous_time() ticks (same timebase, advances
//                     during sleep). Compare with *_ticks to detect sleep.
//   *_scheduler_ticks natural_t (32-bit) scheduler tick counts reported by the
//                     kernel at `tick_hz` per second (100 on macOS, 10 ms). Each
//                     processor's state time is truncated to whole ticks before
//                     aggregation, so an aggregate read can under-report by up
//                     to one tick per processor per state. Counters wrap at 2^32
//                     and are never clamped or unwrapped here.
//   *_abstime         Mach absolute time units carried by process accounting
//                     (rusage_info, task_info); same timebase as *_ticks.
//   *_microseconds    POSIX getrusage timeval fields.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tokspur::telemetry {

class TelemetryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// Time

struct Timebase {
    uint32_t numer = 0, denom = 0;
    [[nodiscard]] bool valid() const noexcept { return numer != 0 && denom != 0; }
};

// Exact integer conversion; throws TelemetryError on an invalid timebase or on
// 64-bit overflow instead of returning a wrapped value.
[[nodiscard]] uint64_t ticks_to_nanoseconds(uint64_t ticks, Timebase timebase);
[[nodiscard]] double ticks_to_seconds(uint64_t ticks, Timebase timebase);

// Two mach_absolute_time() samples taken immediately before and after one kernel
// read (the collector's own SelfTimes reading is the one exception: it brackets
// its RUSAGE_SELF and RUSAGE_CHILDREN calls together). The observation happened
// somewhere inside [before, after]; nothing in this component treats two
// different brackets as simultaneous. A bracket whose `after` precedes `before`
// is reported as disordered and never repaired.
struct Bracket {
    uint64_t before = 0, after = 0;
    [[nodiscard]] bool ordered() const noexcept { return after >= before; }
    [[nodiscard]] uint64_t width() const noexcept { return ordered() ? after - before : 0; }
    [[nodiscard]] uint64_t midpoint() const noexcept { return ordered() ? before + (after - before) / 2 : before; }
};

// ---------------------------------------------------------------------------
// Readings

enum class Status : uint8_t {
    observed,          // the kernel returned the value inside `bracket`
    unavailable,       // API absent/unsupported on this platform or build
    permission_denied, // the kernel refused this caller (EPERM/EACCES or Mach access failure)
    not_found,         // the selected process does not exist (ESRCH)
    identity_changed,  // the pid now names a different process birth than first observed
    error              // any other failure; `detail` carries the kernel/library text
};
[[nodiscard]] std::string_view status_name(Status status) noexcept;

template <class Value>
struct Reading {
    Bracket bracket;
    Status status = Status::unavailable;
    std::string detail; // empty when observed
    Value value{};      // meaningful only when status == observed; never a substitute zero
    [[nodiscard]] bool observed() const noexcept { return status == Status::observed; }
};

// host_statistics(HOST_CPU_LOAD_INFO) / host_processor_info(PROCESSOR_CPU_LOAD_INFO).
struct CpuTicks {
    uint32_t user = 0, system = 0, idle = 0, nice = 0;
    [[nodiscard]] uint64_t busy() const noexcept { return uint64_t(user) + system + nice; }
    [[nodiscard]] uint64_t total() const noexcept { return busy() + idle; }
    [[nodiscard]] bool operator==(const CpuTicks&) const noexcept = default;
};

// host_processor_info(PROCESSOR_CPU_COUNTERS_INFO) reports per-processor Mach absolute time per state. Builds whose SDK lacks the flavour report unavailable. Kernel permission failures remain explicit reading states.
struct CoreTimes {
    uint64_t user_abstime = 0, system_abstime = 0, idle_abstime = 0;
    uint64_t cycles = 0, instructions = 0, energy_nj = 0; // zero when the kernel reports them unsupported
    [[nodiscard]] bool operator==(const CoreTimes&) const noexcept = default;
};

// Process birth identity. `pid` alone is reusable; a process is the same only
// when every component known on both sides agrees (see same_birth).
struct ProcessIdentity {
    int32_t pid = 0;
    uint32_t parent_pid = 0, uid = 0, bsd_status = 0;
    bool has_wall_start = false;      // KERN_PROC_PID p_starttime was read
    uint64_t start_wall_seconds = 0, start_wall_microseconds = 0;
    bool has_abstime_start = false;   // rusage_info ri_proc_start_abstime was read
    uint64_t start_abstime = 0;
    bool has_executable_uuid = false; // rusage_info ri_uuid was read and is non-zero
    std::array<uint8_t, 16> executable_uuid{};
    std::string command;              // proc_bsdshortinfo pbsi_comm, bounded by the kernel
    std::string name;                 // proc_bsdinfo pbi_name where readable (same user or root); may be empty
};
// True only when pids match, no component known on both sides differs and at
// least one birth component is known on both sides.
[[nodiscard]] bool same_birth(const ProcessIdentity& a, const ProcessIdentity& b) noexcept;
struct ProcessObservation;
// Anchors a selected pid to its first observed birth. A later observation that
// names a different birth has both its identity and cpu readings marked
// identity_changed; unobserved identities leave the anchor untouched.
void anchor_identity(std::optional<ProcessIdentity>& anchor, ProcessObservation& observation);

// proc_pid_rusage(RUSAGE_INFO_V4) CPU fields. `runnable_abstime` is exposed raw;
// its support on a given kernel is not assumed.
struct ProcessCpuTimes {
    uint64_t user_abstime = 0, system_abstime = 0, runnable_abstime = 0;
    uint64_t interrupt_wakeups = 0, package_idle_wakeups = 0;
};

struct ProcessObservation {
    int32_t requested_pid = 0;
    Reading<ProcessIdentity> identity; // PROC_PIDT_SHORTBSDINFO plus KERN_PROC_PID: readable for other users' processes
    Reading<ProcessCpuTimes> cpu;      // RUSAGE_INFO_V4: same-user (or root) processes only
};

// The collector process itself, so that telemetry overhead is visible rather
// than silently inside the aggregate.
struct SelfTimes {
    uint64_t user_microseconds = 0, system_microseconds = 0;                   // RUSAGE_SELF
    uint64_t children_user_microseconds = 0, children_system_microseconds = 0; // RUSAGE_CHILDREN: reaped descendants only
    uint64_t voluntary_switches = 0, involuntary_switches = 0;
};
struct SelfTaskTimes { // task_info(TASK_ABSOLUTETIME_INFO), includes terminated threads
    uint64_t total_user_abstime = 0, total_system_abstime = 0, threads_user_abstime = 0, threads_system_abstime = 0;
};

struct Sample {
    uint64_t sequence = 0;
    Timebase timebase;
    uint32_t tick_hz = 0;      // sysconf(_SC_CLK_TCK); 0 when unavailable
    uint32_t logical_cpus = 0; // hw.logicalcpu; 0 when unavailable
    Status environment_status = Status::unavailable; // platform::environment() outcome at construction
    std::string environment_detail;                  // its error or partial-failure text
    Bracket span;              // mach_absolute_time around the whole sample
    uint64_t continuous_before = 0, continuous_after = 0;
    int64_t realtime_nanoseconds = 0; // CLOCK_REALTIME at the start, for correlation only
    Reading<CpuTicks> aggregate;                 // host_statistics: subject to the kernel's per-second cache for non-platform binaries
    Reading<std::vector<CpuTicks>> core_ticks;   // host_processor_info(PROCESSOR_CPU_LOAD_INFO): not cached
    Reading<std::vector<CoreTimes>> core_times;  // host_processor_info(PROCESSOR_CPU_COUNTERS_INFO): availability reported
    std::vector<ProcessObservation> processes;   // one per selected pid, in selection order
    Reading<SelfTimes> self;
    Reading<SelfTaskTimes> self_task;
};

inline constexpr size_t max_selected_processes = 64;
inline constexpr size_t max_processors = 1024;

struct Selection {
    std::vector<int32_t> pids; // distinct, non-negative, at most max_selected_processes
    bool aggregate = true, core_ticks = true, core_times = true;
};

class Collector {
public:
    explicit Collector(Selection selection); // throws TelemetryError on an invalid selection
    [[nodiscard]] Sample sample();            // kernel failures become reading states, never exceptions
    [[nodiscard]] const Selection& selection() const noexcept { return selection_; }
    [[nodiscard]] Timebase timebase() const noexcept { return timebase_; }
private:
    Selection selection_;
    uint64_t sequence_ = 0;
    Timebase timebase_;
    uint32_t tick_hz_ = 0, logical_cpus_ = 0;
    Status environment_status_ = Status::unavailable;
    std::string environment_detail_;
    std::vector<std::optional<ProcessIdentity>> first_identity_; // anchors per selected pid
};

// ---------------------------------------------------------------------------
// Deltas (pure arithmetic, model-free, portable)

enum class DeltaStatus : uint8_t {
    valid,
    unavailable,      // one side was not observed
    bracket_disorder, // a bracket's after precedes its before
    regression,       // a counter moved backwards (or wrapped); raw values retained, no clamp
    identity_changed, // the two observations name different process births
    no_elapsed,       // the later bracket does not follow the earlier one
    mismatch          // incompatible shapes (processor count changed)
};
[[nodiscard]] std::string_view delta_status_name(DeltaStatus status) noexcept;

// Elapsed time between two bracketed reads. `nominal` is the midpoint
// difference; the true elapsed lies within [minimum, maximum].
struct Elapsed {
    DeltaStatus status = DeltaStatus::unavailable;
    uint64_t nominal_ticks = 0, minimum_ticks = 0, maximum_ticks = 0;
};
[[nodiscard]] Elapsed elapsed_between(const Bracket& earlier, const Bracket& later) noexcept;

struct TickDelta {
    DeltaStatus status = DeltaStatus::unavailable;
    Elapsed elapsed;
    CpuTicks earlier, later;                   // raw readings, always retained
    uint32_t user = 0, system = 0, idle = 0, nice = 0; // meaningful only when status == valid
    bool changed = false;                      // any state differs between the two readings
    [[nodiscard]] uint64_t busy() const noexcept { return uint64_t(user) + system + nice; }
    [[nodiscard]] uint64_t total() const noexcept { return busy() + idle; }
};
[[nodiscard]] TickDelta tick_delta(const Reading<CpuTicks>& earlier, const Reading<CpuTicks>& later) noexcept;

struct CoreTickDeltas {
    DeltaStatus status = DeltaStatus::unavailable; // mismatch when processor counts differ
    std::vector<TickDelta> cores;
    [[nodiscard]] size_t changed_cores() const noexcept;
    [[nodiscard]] size_t regressed_cores() const noexcept;
};
[[nodiscard]] CoreTickDeltas core_tick_deltas(const Reading<std::vector<CpuTicks>>& earlier, const Reading<std::vector<CpuTicks>>& later);

struct CoreTimeDelta {
    DeltaStatus status = DeltaStatus::unavailable;
    uint64_t user_abstime = 0, system_abstime = 0, idle_abstime = 0;
    bool changed = false;
};
struct CoreTimeDeltas {
    DeltaStatus status = DeltaStatus::unavailable;
    Elapsed elapsed;
    std::vector<CoreTimeDelta> cores;
};
[[nodiscard]] CoreTimeDeltas core_time_deltas(const Reading<std::vector<CoreTimes>>& earlier, const Reading<std::vector<CoreTimes>>& later);

struct ProcessDelta {
    int32_t pid = 0;
    DeltaStatus status = DeltaStatus::unavailable;
    Status earlier_status = Status::unavailable, later_status = Status::unavailable; // raw cpu reading states
    Elapsed elapsed;
    uint64_t user_abstime = 0, system_abstime = 0, runnable_abstime = 0;
};
[[nodiscard]] ProcessDelta process_delta(const ProcessObservation& earlier, const ProcessObservation& later) noexcept;

struct SelfDelta {
    DeltaStatus status = DeltaStatus::unavailable;
    Elapsed elapsed;
    uint64_t user_microseconds = 0, system_microseconds = 0, children_user_microseconds = 0, children_system_microseconds = 0;
};
[[nodiscard]] SelfDelta self_delta(const Reading<SelfTimes>& earlier, const Reading<SelfTimes>& later) noexcept;

// Raw interval attribution. Nothing here is a verdict: the residual is withheld
// (not zeroed) whenever the aggregate counter did not change or was not validly
// observed, negative values are retained, and collector/child CPU is reported
// beside the residual rather than subtracted from it.
struct Attribution {
    bool aggregate_usable = false;
    std::string reason;            // why the residual is withheld or incomplete
    // Seconds are set only when their source delta was valid; unset values describe as null.
    std::optional<double> elapsed_seconds, elapsed_uncertainty_seconds; // from the aggregate brackets
    std::optional<double> aggregate_busy_seconds, aggregate_total_seconds;
    std::optional<double> tick_quantisation_seconds; // logical_cpus / tick_hz: maximum per-state truncation per read
    double selected_process_seconds = 0;  // user + system over valid process deltas
    size_t selected_valid = 0, selected_invalid = 0;
    std::optional<double> collector_self_seconds, collector_children_seconds;
    bool selection_complete = false;      // every selected process had a valid delta
    std::optional<double> residual_cores; // (aggregate_busy - selected_process) / elapsed, may be negative
};
[[nodiscard]] Attribution attribute(const TickDelta& aggregate, const std::vector<ProcessDelta>& processes, const SelfDelta& self,
                                    Timebase timebase, uint32_t tick_hz, uint32_t logical_cpus);

// ---------------------------------------------------------------------------
// JSON text (RFC 8259 emitter only; values are pre-serialised JSON fragments)

namespace json {
[[nodiscard]] std::string string(std::string_view text);
[[nodiscard]] std::string number(uint64_t value);
[[nodiscard]] std::string number(int64_t value);
[[nodiscard]] std::string number(double value); // "null" for non-finite values, never a substitute zero
[[nodiscard]] std::string boolean(bool value);
[[nodiscard]] std::string null();
using Member = std::pair<std::string, std::string>;
[[nodiscard]] std::string object(const std::vector<Member>& members);
[[nodiscard]] std::string array(const std::vector<std::string>& items);
} // namespace json

[[nodiscard]] std::string describe(const Sample& sample);
[[nodiscard]] std::string describe(const Elapsed& elapsed, Timebase timebase);
[[nodiscard]] std::string describe(const TickDelta& delta, Timebase timebase);
[[nodiscard]] std::string describe(const CoreTickDeltas& deltas, Timebase timebase);
[[nodiscard]] std::string describe(const CoreTimeDeltas& deltas, Timebase timebase);
[[nodiscard]] std::string describe(const ProcessDelta& delta, Timebase timebase);
[[nodiscard]] std::string describe(const SelfDelta& delta, Timebase timebase);
[[nodiscard]] std::string describe(const Attribution& attribution);

// ---------------------------------------------------------------------------
// Platform reads. Implemented by host_telemetry_apple.mm on Apple platforms and
// by portable stubs (every reading unavailable) elsewhere. Not for direct use.

namespace platform {
struct Environment {
    Status status = Status::unavailable;
    std::string detail;
    Timebase timebase;
    uint32_t tick_hz = 0, logical_cpus = 0;
};
struct ProcessRusage {
    ProcessCpuTimes times;
    uint64_t start_abstime = 0;
    std::array<uint8_t, 16> executable_uuid{};
};
[[nodiscard]] Environment environment();
[[nodiscard]] uint64_t absolute_now() noexcept;
[[nodiscard]] uint64_t continuous_now() noexcept;
[[nodiscard]] int64_t realtime_nanoseconds() noexcept;
[[nodiscard]] Reading<CpuTicks> read_aggregate_ticks();
[[nodiscard]] Reading<std::vector<CpuTicks>> read_core_ticks();
[[nodiscard]] Reading<std::vector<CoreTimes>> read_core_times();
[[nodiscard]] Reading<ProcessIdentity> read_identity(int32_t pid);
[[nodiscard]] Reading<ProcessRusage> read_process_rusage(int32_t pid);
[[nodiscard]] Reading<SelfTimes> read_self();
[[nodiscard]] Reading<SelfTaskTimes> read_self_task();
} // namespace platform

} // namespace tokspur::telemetry
