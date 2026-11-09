// tokspur-host-telemetry: standalone CPU-only diagnostic.
//
// Compares the update cadence of the kernel's aggregate CPU counters with the
// per-processor counters and with selected-process accounting, and reports
// interval consistency with every raw value, bracket and state. It never
// initialises Metal, loads weights, launches inference or signals a process,
// and it never declares a host quiet or a benchmark passing.

#include "tokspur/host_telemetry.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace tokspur::telemetry;

constexpr std::string_view scope_text =
    "CPU-only counter cadence and interval-consistency diagnostic; raw values with brackets and states. "
    "It declares nothing quiet or passing and changes no benchmark guard.";

struct Options {
    std::string command;
    double seconds = 5;
    uint32_t poll_milliseconds = 2, interval_milliseconds = 500;
    std::vector<int32_t> pids;
    bool core_times = true, events = false;
};

[[noreturn]] void usage(const char* why) {
    std::cerr << "tokspur-host-telemetry: " << why << "\n"
              << "usage:\n"
              << "  tokspur-host-telemetry sample [--pid P]... [--no-core-times]\n"
              << "  tokspur-host-telemetry cadence [--seconds S<=120] [--poll-milliseconds M>=1] [--events] [--no-core-times]\n"
              << "  tokspur-host-telemetry intervals [--seconds S<=3600] [--interval-milliseconds I>=100] [--pid P]... [--no-core-times]\n";
    std::exit(2);
}

uint64_t parse_unsigned(std::string_view text, uint64_t minimum, uint64_t maximum, const char* what) {
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < minimum || value > maximum) usage(what);
    return value;
}

Options parse(int argc, char** argv) {
    Options options;
    if (argc < 2) usage("command required");
    options.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag = argv[i];
        auto next = [&]() -> std::string_view { if (i + 1 >= argc) usage("option value missing"); return argv[++i]; };
        if (flag == "--seconds") {
            const auto text = next();
            double value = 0;
            const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
            if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !(value > 0) || !std::isfinite(value)) usage("--seconds must be a positive number");
            options.seconds = value;
        } else if (flag == "--poll-milliseconds") options.poll_milliseconds = uint32_t(parse_unsigned(next(), 1, 1000, "--poll-milliseconds must be 1..1000"));
        else if (flag == "--interval-milliseconds") options.interval_milliseconds = uint32_t(parse_unsigned(next(), 100, 60000, "--interval-milliseconds must be 100..60000"));
        else if (flag == "--pid") options.pids.push_back(int32_t(parse_unsigned(next(), 0, 2147483647, "--pid must be a non-negative pid")));
        else if (flag == "--no-core-times") options.core_times = false;
        else if (flag == "--events") options.events = true;
        else usage("unknown option");
    }
    if (options.command == "cadence" && options.seconds > 120) usage("cadence is bounded at 120 seconds");
    if (options.command == "intervals" && options.seconds > 3600) usage("intervals is bounded at 3600 seconds");
    if (options.pids.size() > max_selected_processes) usage("at most 64 selected pids");
    return options;
}

uint64_t seconds_to_ticks(double seconds, Timebase timebase) {
    // ticks = seconds * 1e9 * denom / numer; double is adequate for bounded durations.
    return uint64_t(seconds * 1e9 * double(timebase.denom) / double(timebase.numer));
}

std::string environment_json(const Sample& sample, const Options& options) {
    return json::object({{"tool", json::string("tokspur-host-telemetry")}, {"command", json::string(options.command)}, {"scope", json::string(scope_text)},
                         {"quiet_qualified", json::boolean(false)}, {"performance_qualified", json::boolean(false)},
                         {"timebase", json::object({{"numer", json::number(uint64_t(sample.timebase.numer))}, {"denom", json::number(uint64_t(sample.timebase.denom))}})},
                         {"tick_hz", json::number(uint64_t(sample.tick_hz))}, {"logical_cpus", json::number(uint64_t(sample.logical_cpus))},
                         {"processors_reported", json::number(uint64_t(sample.core_ticks.observed() ? sample.core_ticks.value.size() : 0))},
                         {"core_times_status", json::string(status_name(sample.core_times.status))}, {"core_times_detail", json::string(sample.core_times.detail)},
                         {"aggregate_status", json::string(status_name(sample.aggregate.status))}, {"aggregate_detail", json::string(sample.aggregate.detail)},
                         {"selected_pids", json::number(uint64_t(options.pids.size()))}});
}

struct Stats {
    std::vector<double> values;
    std::string json_text() const {
        if (values.empty()) return json::object({{"count", json::number(uint64_t(0))}});
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        const size_t middle = sorted.size() / 2;
        const double median = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
        double sum = 0;
        for (const auto value : sorted) sum += value;
        return json::object({{"count", json::number(uint64_t(sorted.size()))}, {"minimum", json::number(sorted.front())}, {"median", json::number(median)},
                             {"mean", json::number(sum / double(sorted.size()))}, {"maximum", json::number(sorted.back())}});
    }
};

struct Source {
    explicit Source(std::string label) : name(std::move(label)) {}
    std::string name;
    uint64_t reads = 0, changes = 0, regressions = 0, unavailable = 0;
    uint64_t last_change_ticks = 0;
    bool any_change = false;
    Stats gaps_seconds, jump_scheduler_ticks, changed_cores;
    std::vector<uint64_t> per_window;      // changes per whole continuous second since start
    std::vector<std::string> events;       // optional retained event lines
    void change(uint64_t at_ticks, uint64_t start_ticks, Timebase timebase, double jump, size_t cores, bool keep_event, uint64_t continuous_second) {
        ++changes;
        if (any_change && at_ticks > last_change_ticks) gaps_seconds.values.push_back(ticks_to_seconds(at_ticks - last_change_ticks, timebase));
        any_change = true;
        last_change_ticks = at_ticks;
        jump_scheduler_ticks.values.push_back(jump);
        if (cores) changed_cores.values.push_back(double(cores));
        if (per_window.size() <= continuous_second) per_window.resize(continuous_second + 1, 0);
        ++per_window[continuous_second];
        if (keep_event && events.size() < 200000) {
            events.push_back(json::object({{"source", json::string(name)}, {"at_seconds", json::number(ticks_to_seconds(at_ticks - start_ticks, timebase))},
                                           {"jump", json::number(jump)}, {"changed_cores", json::number(uint64_t(cores))}}));
        }
    }
    std::string json_text(size_t complete_windows) const {
        // Only whole seconds enter the per-window statistics; the trailing partial window is reported apart.
        Stats windows;
        const size_t counted = std::min(per_window.size(), complete_windows);
        for (size_t i = 0; i < counted; ++i) windows.values.push_back(double(per_window[i]));
        uint64_t partial = 0;
        for (size_t i = counted; i < per_window.size(); ++i) partial += per_window[i];
        return json::object({{"reads", json::number(reads)}, {"changes", json::number(changes)}, {"regressions", json::number(regressions)},
                             {"unavailable_reads", json::number(unavailable)},
                             {"unchanged_read_fraction", json::number(reads ? 1.0 - double(changes) / double(reads) : 0.0)},
                             {"gap_between_changes_seconds", gaps_seconds.json_text()}, {"jump_per_change", jump_scheduler_ticks.json_text()},
                             {"changed_cores_per_change", changed_cores.json_text()}, {"changes_per_continuous_second", windows.json_text()},
                             {"changes_in_partial_final_window", json::number(partial)},
                             {"changes_per_continuous_second_list", [&] {
                                  std::vector<std::string> items;
                                  for (size_t i = 0; i < counted; ++i) items.push_back(json::number(per_window[i]));
                                  return json::array(items);
                              }()}});
    }
};

int cadence(const Options& options) {
    Collector collector(Selection{{}, true, true, options.core_times});
    const Sample first = collector.sample();
    if (!first.timebase.valid()) { std::cerr << "host telemetry environment unavailable on this platform\n"; return 2; }
    std::cout << environment_json(first, options) << '\n';
    const uint64_t start = first.span.before, deadline = start + seconds_to_ticks(options.seconds, first.timebase);
    const uint64_t ticks_per_second = seconds_to_ticks(1.0, first.timebase);
    Source aggregate{"aggregate_host_statistics"}, cores{"per_core_host_processor_info"}, times{"per_core_cpu_counters_info"};
    Sample previous = first;
    uint64_t samples = 1;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(options.poll_milliseconds));
        Sample current = collector.sample();
        ++samples;
        const uint64_t second = (current.continuous_before - first.continuous_before) / std::max<uint64_t>(ticks_per_second, 1);

        ++aggregate.reads;
        const auto tick = tick_delta(previous.aggregate, current.aggregate);
        if (tick.status == DeltaStatus::valid) {
            if (tick.changed) aggregate.change(current.aggregate.bracket.midpoint(), start, first.timebase, double(tick.total()), 0, options.events, second);
        } else if (tick.status == DeltaStatus::regression) ++aggregate.regressions;
        else ++aggregate.unavailable;

        ++cores.reads;
        const auto core = core_tick_deltas(previous.core_ticks, current.core_ticks);
        if (core.status == DeltaStatus::valid) {
            uint64_t total = 0;
            for (const auto& d : core.cores) if (d.status == DeltaStatus::valid) total += d.total();
            cores.regressions += core.regressed_cores();
            if (core.changed_cores()) cores.change(current.core_ticks.bracket.midpoint(), start, first.timebase, double(total), core.changed_cores(), options.events, second);
        } else ++cores.unavailable;

        ++times.reads;
        const auto time = core_time_deltas(previous.core_times, current.core_times);
        if (time.status == DeltaStatus::valid) {
            uint64_t total = 0; size_t changed = 0;
            for (const auto& d : time.cores) {
                if (d.status == DeltaStatus::valid) total += d.user_abstime + d.system_abstime + d.idle_abstime;
                else if (d.status == DeltaStatus::regression) ++times.regressions;
                if (d.changed) ++changed;
            }
            if (changed) times.change(current.core_times.bracket.midpoint(), start, first.timebase, ticks_to_seconds(total, first.timebase), changed, options.events, second);
        } else ++times.unavailable;

        previous = std::move(current);
        if (previous.span.after >= deadline) break;
    }
    const auto overhead = self_delta(first.self, previous.self);
    const double wall = ticks_to_seconds(previous.span.after - start, first.timebase);
    const size_t complete_windows = size_t((previous.continuous_after - first.continuous_before) / std::max<uint64_t>(ticks_per_second, 1));
    std::vector<json::Member> summary{
        {"event", json::string("cadence_summary")}, {"scope", json::string(scope_text)}, {"wall_seconds", json::number(wall)},
        {"samples", json::number(samples)}, {"poll_milliseconds", json::number(uint64_t(options.poll_milliseconds))},
        {"sources", json::object({{aggregate.name, aggregate.json_text(complete_windows)}, {cores.name, cores.json_text(complete_windows)}, {times.name, times.json_text(complete_windows)}})},
        {"collector_overhead", describe(overhead, first.timebase)},
        {"collector_overhead_cores", overhead.status == DeltaStatus::valid && wall > 0
                                         ? json::number(double(overhead.user_microseconds + overhead.system_microseconds) / 1e6 / wall) : json::null()},
        {"sleep_ticks_continuous_minus_absolute", json::number(int64_t(previous.continuous_after - first.continuous_before) - int64_t(previous.span.after - first.span.before))},
        {"interpretation_limit", json::string("Change cadence is a property of the kernel interfaces as seen by this unprivileged, non-platform process while it was polling; "
                                              "other processes' host_statistics requests share the kernel's per-second cache budget. This is not a quiet or load measurement.")}};
    if (options.events) {
        std::vector<std::string> all;
        for (const auto* source : {&aggregate, &cores, &times}) all.insert(all.end(), source->events.begin(), source->events.end());
        summary.emplace_back("events", json::array(all));
    }
    std::cout << json::object(summary) << '\n';
    return 0;
}

int intervals(const Options& options) {
    Collector collector(Selection{options.pids, true, true, options.core_times});
    Sample previous = collector.sample();
    if (!previous.timebase.valid()) { std::cerr << "host telemetry environment unavailable on this platform\n"; return 2; }
    std::cout << environment_json(previous, options) << '\n';
    const uint64_t start = previous.span.before, deadline = start + seconds_to_ticks(options.seconds, previous.timebase);
    uint64_t intervals = 0, unchanged = 0, regressions = 0, withheld = 0, negative = 0, identity_changes = 0, process_unavailable = 0, core_mismatch = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(options.interval_milliseconds));
        Sample current = collector.sample();
        ++intervals;
        const auto aggregate = tick_delta(previous.aggregate, current.aggregate);
        const auto cores = core_tick_deltas(previous.core_ticks, current.core_ticks);
        const auto times = core_time_deltas(previous.core_times, current.core_times);
        std::vector<ProcessDelta> processes;
        for (size_t i = 0; i < current.processes.size() && i < previous.processes.size(); ++i) processes.push_back(process_delta(previous.processes[i], current.processes[i]));
        const auto self = self_delta(previous.self, current.self);
        const auto result = attribute(aggregate, processes, self, current.timebase, current.tick_hz, current.logical_cpus);

        if (aggregate.status == DeltaStatus::valid && !aggregate.changed) ++unchanged;
        if (aggregate.status == DeltaStatus::regression) ++regressions;
        if (!result.residual_cores) ++withheld;
        else if (*result.residual_cores < 0) ++negative;
        if (cores.status == DeltaStatus::mismatch) ++core_mismatch;
        uint64_t core_busy = 0, core_total = 0;
        for (const auto& d : cores.cores) if (d.status == DeltaStatus::valid) { core_busy += d.busy(); core_total += d.total(); }
        uint64_t time_busy = 0, time_total = 0; size_t time_changed = 0;
        for (const auto& d : times.cores) {
            if (d.status == DeltaStatus::valid) { time_busy += d.user_abstime + d.system_abstime; time_total += d.user_abstime + d.system_abstime + d.idle_abstime; }
            if (d.changed) ++time_changed;
        }
        std::vector<std::string> process_items;
        for (const auto& p : processes) {
            if (p.status == DeltaStatus::identity_changed) ++identity_changes;
            else if (p.status != DeltaStatus::valid) ++process_unavailable;
            process_items.push_back(describe(p, current.timebase));
        }
        std::cout << json::object({
            {"event", json::string("interval")}, {"index", json::number(intervals)},
            {"aggregate", describe(aggregate, current.timebase)},
            {"per_core_ticks", json::object({{"status", json::string(delta_status_name(cores.status))}, {"processors", json::number(uint64_t(cores.cores.size()))},
                                             {"changed_cores", json::number(uint64_t(cores.changed_cores()))}, {"regressed_cores", json::number(uint64_t(cores.regressed_cores()))},
                                             {"busy_scheduler_ticks", json::number(core_busy)}, {"total_scheduler_ticks", json::number(core_total)},
                                             {"elapsed", describe(elapsed_between(previous.core_ticks.bracket, current.core_ticks.bracket), current.timebase)}})},
            {"per_core_times", json::object({{"status", json::string(delta_status_name(times.status))}, {"processors", json::number(uint64_t(times.cores.size()))},
                                             {"changed_cores", json::number(uint64_t(time_changed))}, {"busy_abstime", json::number(time_busy)}, {"total_abstime", json::number(time_total)},
                                             {"busy_seconds", times.status == DeltaStatus::valid ? json::number(ticks_to_seconds(time_busy, current.timebase)) : json::null()},
                                             {"elapsed", describe(times.elapsed, current.timebase)}})},
            {"processes", json::array(process_items)}, {"collector_self", describe(self, current.timebase)}, {"attribution", describe(result)},
            {"sleep_ticks_continuous_minus_absolute", json::number(int64_t(current.continuous_before - previous.continuous_before) - int64_t(current.span.before - previous.span.before))}})
                  << std::endl;
        previous = std::move(current);
        if (previous.span.after >= deadline) break;
    }
    std::cout << json::object({{"event", json::string("intervals_summary")}, {"scope", json::string(scope_text)}, {"intervals", json::number(intervals)},
                               {"aggregate_unchanged_intervals", json::number(unchanged)}, {"aggregate_regressions", json::number(regressions)},
                               {"residual_withheld_intervals", json::number(withheld)}, {"negative_residual_intervals", json::number(negative)},
                               {"identity_changed_deltas", json::number(identity_changes)}, {"process_deltas_unavailable", json::number(process_unavailable)},
                               {"per_core_processor_count_mismatches", json::number(core_mismatch)},
                               {"verdict", json::string("none: counts describe the measurement, not host quietness or a benchmark result")}})
              << '\n';
    return 0;
}

int sample_once(const Options& options) {
    Collector collector(Selection{options.pids, true, true, options.core_times});
    std::cout << describe(collector.sample()) << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.command == "sample") return sample_once(options);
        if (options.command == "cadence") return cadence(options);
        if (options.command == "intervals") return intervals(options);
        usage("unknown command");
    } catch (const std::exception& error) {
        std::cerr << "tokspur-host-telemetry refused: " << error.what() << '\n';
        return 1;
    }
}
