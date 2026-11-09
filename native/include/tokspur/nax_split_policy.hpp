#pragma once
#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

namespace tokspur {
enum class NaxSplitFamily : uint32_t { q8, q8_t64, bf16 };
struct NaxSplitShape {
    NaxSplitFamily family{};
    uint32_t columns = 0, outputs = 0, tokens = 0, tile = 0;
    bool dual = false;
    auto operator<=>(const NaxSplitShape&) const = default;
};
inline bool valid_nax_split(NaxSplitShape shape, uint32_t split, uint64_t scratch = 3'145'728) {
    if (shape.columns < 32 || shape.columns > 65536 || shape.columns % 32 ||
        !shape.outputs || shape.outputs > 524288 || !shape.tokens || shape.tokens > 8 || !shape.tile) return false;
    if (shape.family == NaxSplitFamily::bf16 && shape.dual) return false;
    if (split == 1) return true;
    const uint64_t tiles = (uint64_t(shape.outputs) + shape.tile - 1) / shape.tile;
    uint64_t units = 0, stride = 0, maximum_grid = 0;
    if (shape.family == NaxSplitFamily::q8) {
        constexpr std::array candidates{20u,16u,12u,10u,8u,6u,5u,4u,3u,2u};
        if (std::find(candidates.begin(), candidates.end(), split) == candidates.end() || shape.columns % 512 || shape.columns < 1024 || tiles > 320) return false;
        units = shape.columns / 512; stride = (uint64_t(shape.outputs) + 15) & ~15ull; maximum_grid = 960;
    } else if (shape.family == NaxSplitFamily::q8_t64) {
        constexpr std::array candidates{16u,12u,10u,8u,6u,5u,4u,3u,2u};
        if (std::find(candidates.begin(), candidates.end(), split) == candidates.end() || shape.columns % 32 || shape.columns / 32 < 8 || tiles >= 160) return false;
        units = shape.columns / 32; stride = (uint64_t(shape.outputs) + (shape.dual ? 31 : 63)) & ~(shape.dual ? 31ull : 63ull); maximum_grid = 320;
        if (units / split < (tiles <= 32 ? 2u : 4u)) return false;
    } else {
        constexpr std::array candidates{10u,8u,5u,4u,2u};
        if (std::find(candidates.begin(), candidates.end(), split) == candidates.end() || shape.dual || shape.columns % 128 || shape.columns < 256 || tiles >= 128) return false;
        units = shape.columns / 128; stride = tiles * 32; maximum_grid = 320;
    }
    return tiles && split <= units && units % split == 0 && tiles * split <= maximum_grid &&
        uint64_t(split) * (shape.dual ? 2 : 1) * 8 * stride * sizeof(float) <= scratch;
}

// Exact shape matches only. Unlisted shapes retain their existing helper's automatic choice.
class NaxSplitOverrides {
public:
    static NaxSplitOverrides parse(const Json& value) {
        NaxSplitOverrides out;
        if (!value.find("schema") || value.find("schema")->unsigned_integer() != 1 || !value.find("entries")) throw FormatError("invalid M8 split override schema");
        const auto& entries = value.find("entries")->array();
        if (entries.empty() || entries.size() > 128 || value.object().size() != 2) throw FormatError("M8 split override entry bound");
        for (const auto& entry : entries) {
            const auto get = [&](const char* key) -> const Json& { const auto* v = entry.find(key); if (!v) throw FormatError("incomplete M8 split override"); return *v; };
            const auto integer = [&](const char* key) { const auto v = get(key).unsigned_integer(); if (v > (1u << 20)) throw FormatError("M8 split shape bound"); return uint32_t(v); };
            if (entry.object().size() != 7) throw FormatError("unknown M8 split override field");
            const auto family = get("family").string();
            if (family != "q8" && family != "q8-t64" && family != "bf16") throw FormatError("unknown M8 split family");
            const NaxSplitShape shape{family == "q8" ? NaxSplitFamily::q8 : family == "q8-t64" ? NaxSplitFamily::q8_t64 : NaxSplitFamily::bf16,
                integer("columns"), integer("outputs"), integer("tokens"), integer("tile"), get("dual").boolean()};
            const auto split = integer("splits");
            if (!valid_nax_split(shape, split) || (shape.family == NaxSplitFamily::q8 && shape.tile != 4 && shape.tile != 8 && shape.tile != 16) ||
                (shape.family == NaxSplitFamily::q8_t64 && shape.tile != (shape.dual ? 32u : 64u)) || (shape.family == NaxSplitFamily::bf16 && shape.tile != 32))
                throw FormatError("M8 split override violates helper candidates or constraints");
            out.entries_.push_back({shape, split});
        }
        std::sort(out.entries_.begin(), out.entries_.end());
        std::string canonical = "target-verification-M8-splits-v1";
        for (size_t i = 0; i < out.entries_.size(); ++i) {
            const auto& [s, split] = out.entries_[i];
            if (i && out.entries_[i - 1].first == s) throw FormatError("duplicate M8 split override shape");
            for (uint32_t n : {uint32_t(s.family), s.columns, s.outputs, s.tokens, s.tile, uint32_t(s.dual), split})
                for (unsigned shift = 0; shift < 32; shift += 8) canonical += char((n >> shift) & 255);
        }
        out.digest_ = derived_pack::hash(canonical); return out;
    }
    uint32_t choose(bool target_verification, NaxSplitShape shape, uint32_t automatic, uint64_t scratch) const {
        if (!target_verification) return automatic;
        for (const auto& [key, split] : entries_) if (key == shape) {
            if (!valid_nax_split(shape, split, scratch)) throw FormatError("M8 split override exceeds live scratch");
            return split;
        }
        return automatic;
    }
    const std::string& digest() const noexcept { return digest_; }
private:
    std::vector<std::pair<NaxSplitShape, uint32_t>> entries_;
    std::string digest_;
};
} // namespace tokspur
