#pragma once
#include <cstdint>
#include <cmath>
#include <limits>
namespace tokspur {
enum class Task4ProfileBoundary : uint32_t { dispatch, encoder_late8 };
inline bool task4_profile_boundary_supported(Task4ProfileBoundary boundary, bool dispatch, bool stage) {
    return boundary == Task4ProfileBoundary::dispatch ? dispatch : boundary == Task4ProfileBoundary::encoder_late8 && stage;
}
inline constexpr uint64_t task4_stderr_limit_bytes = 16ull << 20;
inline constexpr uint64_t task4_stderr_reserve_bytes = 1ull << 20;
inline bool task4_profile_record_fits(uint64_t occupied, uint64_t record) {
    constexpr auto budget = task4_stderr_limit_bytes - task4_stderr_reserve_bytes;
    return record && record <= 8'000'000 && occupied <= budget && record <= budget - occupied;
}
inline bool task4_calibration_span_valid(uint64_t cpu_start, uint64_t cpu_end, uint64_t gpu_start, uint64_t gpu_end,
                                        double elapsed_ns, bool timebase_valid) {
    return cpu_end > cpu_start && gpu_end > gpu_start && std::isfinite(elapsed_ns) && elapsed_ns > 0 && timebase_valid;
}
// A counter failure is unavailable diagnostic data, never a zero-duration kernel.
inline bool task4_counter_ticks_valid(uint64_t start,uint64_t end,uint64_t gpu_start,uint64_t gpu_end){
    constexpr auto error=std::numeric_limits<uint64_t>::max();
    return start && end && start!=error && end!=error && gpu_start && gpu_end>gpu_start && start>=gpu_start && end>=start && end<=gpu_end;
}
}
