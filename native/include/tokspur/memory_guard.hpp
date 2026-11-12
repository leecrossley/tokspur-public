#pragma once

#include <cstdint>

namespace tokspur {

struct MemoryGuardReading {
    uint64_t current;
    uint32_t pressure;
};

inline bool memory_work_admitted(MemoryGuardReading reading, uint64_t limit) {
    return limit && reading.current <= limit && reading.pressure == 1;
}

template<class Read, class Reclaim>
bool reclaim_before_memory_refusal(uint64_t limit, Read&& read, Reclaim&& reclaim) {
    if (memory_work_admitted(read(), limit)) return true;
    reclaim();
    return memory_work_admitted(read(), limit);
}

// Report-only memory plan (MEMORY_PLAN_REPORT=1). Nothing here changes a budget.
// The reserve and the half-of-room rule are provisional parameters, reported with every plan.
inline constexpr uint64_t memory_plan_reserve_bytes = uint64_t(8) << 30;

struct MemoryPlanRoom {
    uint64_t room;                    // limit - footprint - reserve, saturating at zero
    uint64_t would_allow_cache_bytes; // what the half-of-room rule would allow; never applied
};

inline MemoryPlanRoom memory_plan_room(uint64_t limit, uint64_t footprint, uint64_t reserve) {
    const uint64_t used = footprint > UINT64_MAX - reserve ? UINT64_MAX : footprint + reserve;
    const uint64_t room = limit > used ? limit - used : 0;
    return {room, room / 2};
}

} // namespace tokspur
