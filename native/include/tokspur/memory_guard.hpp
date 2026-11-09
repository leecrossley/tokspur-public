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

} // namespace tokspur
