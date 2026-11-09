// Apple platform reads for the owned host telemetry collector.
//
// Only Mach, libproc and BSD system interfaces declared by the macOS SDK are
// used: host_statistics, host_processor_info, proc_pidinfo, proc_pid_rusage,
// getrusage, task_info, sysctlbyname, sysconf and the Mach clocks. Nothing here
// touches Metal, IOKit, a GPU device, model weights or another process's state.
// Every kernel read is bracketed by mach_absolute_time() immediately before and
// after the call; failures become explicit reading states, never zeros.

#include "tokspur/host_telemetry.hpp"

#include <cerrno>
#include <cstring>
#include <ctime>

#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_error.h>
#include <mach/mach_time.h>
#include <mach/mig_errors.h>
#include <sys/param.h>
#include <sys/proc_info.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>

namespace tokspur::telemetry::platform {
namespace {

std::string kern_text(const char* call, kern_return_t kr) {
    const char* text = mach_error_string(kr);
    return std::string(call) + " kern_return " + std::to_string(kr) + " (" + (text ? text : "unknown") + ")";
}
std::string errno_text(const char* call, int error) {
    return std::string(call) + " errno " + std::to_string(error) + " (" + std::strerror(error) + ")";
}
Status status_for_errno(int error) noexcept {
    switch (error) {
        case ESRCH: return Status::not_found;
        case EPERM: case EACCES: return Status::permission_denied;
        case ENOSYS: case EINVAL: case ENOTSUP: return Status::unavailable;
        default: return Status::error;
    }
}
Status status_for_kern(kern_return_t kr) noexcept {
    switch (kr) {
        case KERN_NO_ACCESS: case KERN_PROTECTION_FAILURE: return Status::permission_denied;
        case KERN_INVALID_ARGUMENT: case MIG_BAD_ID: return Status::unavailable;
        default: return Status::error;
    }
}

// Each host port right obtained with mach_host_self() is released after the read.
struct HostPort {
    mach_port_t port = mach_host_self();
    ~HostPort() { if (port != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), port); }
    HostPort() = default;
    HostPort(const HostPort&) = delete;
    HostPort& operator=(const HostPort&) = delete;
};

// Reply arrays from host_processor_info are owned by the caller's address space.
struct ProcessorReply {
    processor_info_array_t info = nullptr;
    mach_msg_type_number_t count = 0;
    natural_t processors = 0;
    ~ProcessorReply() {
        if (info) vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(info), vm_size_t(count) * sizeof(integer_t));
    }
    ProcessorReply() = default;
    ProcessorReply(const ProcessorReply&) = delete;
    ProcessorReply& operator=(const ProcessorReply&) = delete;
};

template <class Value>
bool processor_reply(Reading<std::vector<Value>>& reading, processor_flavor_t flavor, const char* call, mach_msg_type_number_t per_processor,
                     bool optional_flavor, ProcessorReply& reply) {
    HostPort host_port;
    if (host_port.port == MACH_PORT_NULL) { reading.status = Status::error; reading.detail = "mach_host_self returned no port"; return false; }
    reading.bracket.before = mach_absolute_time();
    const kern_return_t kr = host_processor_info(host_port.port, flavor, &reply.processors, &reply.info, &reply.count);
    reading.bracket.after = mach_absolute_time();
    if (kr != KERN_SUCCESS) {
        reply.info = nullptr; reply.count = 0;
        reading.status = optional_flavor ? (status_for_kern(kr) == Status::permission_denied ? Status::permission_denied : Status::unavailable) : status_for_kern(kr);
        reading.detail = kern_text(call, kr);
        return false;
    }
    if (reply.processors == 0 || reply.processors > max_processors || reply.count != reply.processors * per_processor) {
        reading.status = Status::error;
        reading.detail = std::string(call) + " reply shape unexpected: processors " + std::to_string(reply.processors) + ", count " + std::to_string(reply.count);
        return false;
    }
    return true;
}

uint64_t microseconds(const timeval& value, bool& negative) noexcept {
    if (value.tv_sec < 0 || value.tv_usec < 0) { negative = true; return 0; }
    return uint64_t(value.tv_sec) * 1000000u + uint64_t(value.tv_usec);
}

} // namespace

Environment environment() {
    Environment environment;
    mach_timebase_info_data_t timebase{};
    const kern_return_t kr = mach_timebase_info(&timebase);
    if (kr != KERN_SUCCESS || timebase.numer == 0 || timebase.denom == 0) {
        environment.status = Status::error;
        environment.detail = kern_text("mach_timebase_info", kr);
        return environment;
    }
    environment.timebase = Timebase{timebase.numer, timebase.denom};
    errno = 0;
    const long clock_ticks = sysconf(_SC_CLK_TCK);
    if (clock_ticks <= 0 || clock_ticks > 1000000) {
        environment.status = Status::error;
        environment.detail = errno_text("sysconf(_SC_CLK_TCK)", errno);
        return environment;
    }
    environment.tick_hz = uint32_t(clock_ticks);
    int32_t logical = 0;
    size_t size = sizeof logical;
    if (sysctlbyname("hw.logicalcpu", &logical, &size, nullptr, 0) == 0 && size == sizeof logical && logical > 0 && uint32_t(logical) <= max_processors) {
        environment.logical_cpus = uint32_t(logical);
    } else {
        environment.detail = errno_text("sysctlbyname(hw.logicalcpu)", errno) + "; logical processor count unavailable";
    }
    environment.status = Status::observed;
    return environment;
}

uint64_t absolute_now() noexcept { return mach_absolute_time(); }
uint64_t continuous_now() noexcept { return mach_continuous_time(); }
int64_t realtime_nanoseconds() noexcept {
    timespec now{};
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
    return int64_t(now.tv_sec) * 1000000000 + int64_t(now.tv_nsec);
}

Reading<CpuTicks> read_aggregate_ticks() {
    Reading<CpuTicks> reading;
    HostPort host_port;
    if (host_port.port == MACH_PORT_NULL) { reading.status = Status::error; reading.detail = "mach_host_self returned no port"; return reading; }
    host_cpu_load_info_data_t info{};
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    reading.bracket.before = mach_absolute_time();
    const kern_return_t kr = host_statistics(host_port.port, HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&info), &count);
    reading.bracket.after = mach_absolute_time();
    if (kr != KERN_SUCCESS) { reading.status = status_for_kern(kr); reading.detail = kern_text("host_statistics(HOST_CPU_LOAD_INFO)", kr); return reading; }
    if (count < HOST_CPU_LOAD_INFO_COUNT) { reading.status = Status::error; reading.detail = "host_statistics(HOST_CPU_LOAD_INFO) short reply"; return reading; }
    reading.value = CpuTicks{info.cpu_ticks[CPU_STATE_USER], info.cpu_ticks[CPU_STATE_SYSTEM], info.cpu_ticks[CPU_STATE_IDLE], info.cpu_ticks[CPU_STATE_NICE]};
    reading.status = Status::observed;
    return reading;
}

Reading<std::vector<CpuTicks>> read_core_ticks() {
    Reading<std::vector<CpuTicks>> reading;
    ProcessorReply reply;
    if (!processor_reply(reading, PROCESSOR_CPU_LOAD_INFO, "host_processor_info(PROCESSOR_CPU_LOAD_INFO)", PROCESSOR_CPU_LOAD_INFO_COUNT, false, reply)) return reading;
    reading.value.resize(reply.processors);
    for (natural_t i = 0; i < reply.processors; ++i) {
        processor_cpu_load_info_data_t load{};
        std::memcpy(&load, reply.info + size_t(i) * PROCESSOR_CPU_LOAD_INFO_COUNT, sizeof load);
        reading.value[i] = CpuTicks{load.cpu_ticks[CPU_STATE_USER], load.cpu_ticks[CPU_STATE_SYSTEM], load.cpu_ticks[CPU_STATE_IDLE], load.cpu_ticks[CPU_STATE_NICE]};
    }
    reading.status = Status::observed;
    return reading;
}

Reading<std::vector<CoreTimes>> read_core_times() {
    Reading<std::vector<CoreTimes>> reading;
#if defined(PROCESSOR_CPU_COUNTERS_INFO) && defined(PROCESSOR_CPU_COUNTERS_INFO_COUNT)
    ProcessorReply reply;
    if (!processor_reply(reading, PROCESSOR_CPU_COUNTERS_INFO, "host_processor_info(PROCESSOR_CPU_COUNTERS_INFO)", PROCESSOR_CPU_COUNTERS_INFO_COUNT, true, reply)) return reading;
    reading.value.resize(reply.processors);
    for (natural_t i = 0; i < reply.processors; ++i) {
        processor_cpu_counters_info_data_t counters{};
        std::memcpy(&counters, reply.info + size_t(i) * PROCESSOR_CPU_COUNTERS_INFO_COUNT, sizeof counters);
        reading.value[i] = CoreTimes{counters.user_time_mach, counters.system_time_mach, counters.idle_time_mach, counters.cycles, counters.instructions, counters.energy_nj};
    }
    reading.status = Status::observed;
#else
    reading.status = Status::unavailable;
    reading.detail = "macOS SDK does not provide PROCESSOR_CPU_COUNTERS_INFO";
#endif
    return reading;
}

Reading<ProcessIdentity> read_identity(int32_t pid) {
    // The full PROC_PIDTBSDINFO flavour is refused (EPERM) for other users'
    // processes on this kernel. The short flavour and the KERN_PROC_PID process
    // table entry are readable for every process, so identity and the wall start
    // time come from those; the long name is added only where the full flavour
    // is readable (same user or root).
    Reading<ProcessIdentity> reading;
    reading.value.pid = pid;
    proc_bsdshortinfo info{};
    errno = 0;
    reading.bracket.before = mach_absolute_time();
    const int bytes = proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, &info, int(sizeof info));
    const int error = errno;
    reading.bracket.after = mach_absolute_time();
    if (bytes <= 0) {
        reading.status = error ? status_for_errno(error) : Status::error;
        reading.detail = error ? errno_text("proc_pidinfo(PROC_PIDT_SHORTBSDINFO)", error) : "proc_pidinfo(PROC_PIDT_SHORTBSDINFO) returned no data";
        return reading;
    }
    if (bytes != int(sizeof info)) { reading.status = Status::error; reading.detail = "proc_pidinfo(PROC_PIDT_SHORTBSDINFO) short reply"; return reading; }
    if (info.pbsi_pid != uint32_t(pid)) { reading.status = Status::error; reading.detail = "proc_pidinfo(PROC_PIDT_SHORTBSDINFO) reported a different pid"; return reading; }
    auto& identity = reading.value;
    identity.parent_pid = info.pbsi_ppid;
    identity.uid = info.pbsi_uid;
    identity.bsd_status = info.pbsi_status;
    identity.command.assign(info.pbsi_comm, strnlen(info.pbsi_comm, sizeof info.pbsi_comm));
    // Birth time from the process table; outside the bracket, it is not a counter.
    int selector[] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
    kinfo_proc entry{};
    size_t size = sizeof entry;
    if (sysctl(selector, 4, &entry, &size, nullptr, 0) == 0 && size == sizeof entry && entry.kp_proc.p_pid == pid) {
        identity.has_wall_start = true;
        identity.start_wall_seconds = uint64_t(entry.kp_proc.p_starttime.tv_sec);
        identity.start_wall_microseconds = uint64_t(entry.kp_proc.p_starttime.tv_usec);
    }
    proc_bsdinfo full{};
    if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &full, int(sizeof full)) == int(sizeof full) && full.pbi_pid == uint32_t(pid)) {
        identity.name.assign(full.pbi_name, strnlen(full.pbi_name, sizeof full.pbi_name));
    }
    reading.status = Status::observed;
    return reading;
}

Reading<ProcessRusage> read_process_rusage(int32_t pid) {
    Reading<ProcessRusage> reading;
    rusage_info_v4 info{};
    errno = 0;
    reading.bracket.before = mach_absolute_time();
    const int rc = proc_pid_rusage(pid, RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&info));
    const int error = errno;
    reading.bracket.after = mach_absolute_time();
    if (rc != 0) {
        reading.status = error ? status_for_errno(error) : Status::error;
        reading.detail = error ? errno_text("proc_pid_rusage(RUSAGE_INFO_V4)", error) : "proc_pid_rusage(RUSAGE_INFO_V4) failed without errno";
        return reading;
    }
    auto& value = reading.value;
    value.times.user_abstime = info.ri_user_time;
    value.times.system_abstime = info.ri_system_time;
    value.times.runnable_abstime = info.ri_runnable_time;
    value.times.interrupt_wakeups = info.ri_interrupt_wkups;
    value.times.package_idle_wakeups = info.ri_pkg_idle_wkups;
    value.start_abstime = info.ri_proc_start_abstime;
    std::memcpy(value.executable_uuid.data(), info.ri_uuid, value.executable_uuid.size());
    reading.status = Status::observed;
    return reading;
}

Reading<SelfTimes> read_self() {
    Reading<SelfTimes> reading;
    rusage self{}, children{};
    reading.bracket.before = mach_absolute_time();
    const int self_rc = getrusage(RUSAGE_SELF, &self);
    const int self_error = errno;
    const int children_rc = getrusage(RUSAGE_CHILDREN, &children);
    const int children_error = errno;
    reading.bracket.after = mach_absolute_time();
    if (self_rc != 0) { reading.status = Status::error; reading.detail = errno_text("getrusage(RUSAGE_SELF)", self_error); return reading; }
    if (children_rc != 0) { reading.status = Status::error; reading.detail = errno_text("getrusage(RUSAGE_CHILDREN)", children_error); return reading; }
    bool negative = false;
    auto& value = reading.value;
    value.user_microseconds = microseconds(self.ru_utime, negative);
    value.system_microseconds = microseconds(self.ru_stime, negative);
    value.children_user_microseconds = microseconds(children.ru_utime, negative);
    value.children_system_microseconds = microseconds(children.ru_stime, negative);
    value.voluntary_switches = self.ru_nvcsw < 0 ? 0 : uint64_t(self.ru_nvcsw);
    value.involuntary_switches = self.ru_nivcsw < 0 ? 0 : uint64_t(self.ru_nivcsw);
    if (negative) { reading.status = Status::error; reading.detail = "getrusage returned a negative time"; reading.value = SelfTimes{}; return reading; }
    reading.status = Status::observed;
    return reading;
}

Reading<SelfTaskTimes> read_self_task() {
    Reading<SelfTaskTimes> reading;
    task_absolutetime_info_data_t info{};
    mach_msg_type_number_t count = TASK_ABSOLUTETIME_INFO_COUNT;
    reading.bracket.before = mach_absolute_time();
    const kern_return_t kr = task_info(mach_task_self(), TASK_ABSOLUTETIME_INFO, reinterpret_cast<task_info_t>(&info), &count);
    reading.bracket.after = mach_absolute_time();
    if (kr != KERN_SUCCESS) { reading.status = status_for_kern(kr); reading.detail = kern_text("task_info(TASK_ABSOLUTETIME_INFO)", kr); return reading; }
    if (count < TASK_ABSOLUTETIME_INFO_COUNT) { reading.status = Status::error; reading.detail = "task_info(TASK_ABSOLUTETIME_INFO) short reply"; return reading; }
    reading.value = SelfTaskTimes{info.total_user, info.total_system, info.threads_user, info.threads_system};
    reading.status = Status::observed;
    return reading;
}

} // namespace tokspur::telemetry::platform
