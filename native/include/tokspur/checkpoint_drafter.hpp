#pragma once

#include "tokspur/dflash2_drafter.hpp"
#include "tokspur/mtp_head.hpp"

namespace tokspur {

// The on-disk auxiliary payload fields retain their historical MTP names. The execution digest binds the drafter kind, assets and settings, so payloads cannot cross modes.
struct CheckpointDrafter {
    MtpHead* mtp = nullptr;
    DFlash2Drafter* dflash = nullptr;

    explicit operator bool() const noexcept { return mtp || dflash; }
    uint32_t first_absolute_position() const { return dflash ? dflash->first_absolute_position() : mtp ? mtp->first_absolute_position() : 1; }
    uint32_t cache_rows() const { return dflash ? dflash->cache_rows() : mtp ? mtp->cache_rows() : 0; }
    uint64_t ssd_cache_bytes(uint32_t rows) const { return dflash ? dflash->ssd_cache_bytes(rows) : mtp ? mtp->ssd_cache_bytes(rows) : 0; }
    uint64_t ssd_cache_bytes() const { return ssd_cache_bytes(cache_rows()); }
    bool read_ssd_cache_span(uint64_t offset, std::span<std::byte> out) const {
        return dflash ? dflash->read_ssd_cache_span(offset, out) : mtp ? mtp->read_ssd_cache_span(offset, out) : out.empty();
    }
    bool write_ssd_cache_span(uint32_t first, uint32_t rows, uint64_t offset, std::span<const std::byte> in) {
        return dflash ? dflash->write_ssd_cache_span(first, rows, offset, in) : mtp ? mtp->write_ssd_cache_span(first, rows, offset, in) : in.empty();
    }
    void reset(uint32_t first) {
        if (dflash) {
            if (first) throw FormatError("empty DFlash2 checkpoint has a nonzero frontier");
            dflash->reset();
        } else if (mtp) mtp->reset(first);
    }
    bool matches(uint32_t prefix, uint32_t first, uint32_t rows) const {
        if (dflash) return first <= prefix && rows == prefix - first && rows == std::min(prefix, 2048u);
        return mtp ? first < 65536 && rows <= 65536 - first : rows == 0;
    }
};

} // namespace tokspur
