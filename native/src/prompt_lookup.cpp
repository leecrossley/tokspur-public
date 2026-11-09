#include "tokspur/prompt_lookup.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <bit>

namespace tokspur {
namespace {

void require(bool ok, const char* why) {
    if (!ok) throw FormatError(why);
}

void validate(const LookupConfig& c) {
    require(c.match_tokens >= 1 && c.match_tokens <= maximum_match_tokens,
            "prompt lookup match width outside [1, 64]");
    require(c.proposal_tokens >= 1 && c.proposal_tokens <= maximum_proposal_tokens,
            "prompt lookup proposal width outside [1, 16]");
    require(c.history_tokens > c.match_tokens && c.history_tokens <= maximum_history_tokens,
            "prompt lookup history must exceed the match width and fit 65,536 tokens");
    require(c.token_limit >= 2, "prompt lookup token limit must admit at least two token IDs");
    require(c.work_limit >= 1 && c.work_limit <= maximum_lookup_work,
            "prompt lookup work limit outside [1, 2^20]");
    require(c.index_buckets == 0 ||
                (std::has_single_bit(c.index_buckets) && c.index_buckets <= maximum_lookup_buckets),
            "prompt lookup bucket count must be zero or a power of two up to 2^22");
    require(c.memory_budget_bytes >= 1 && c.memory_budget_bytes <= maximum_lookup_budget_bytes,
            "prompt lookup memory budget outside [1 byte, 1 GiB]");
}

uint32_t resolve_buckets(const LookupConfig& c) {
    if (c.index_buckets != 0) return c.index_buckets;
    return uint32_t(std::bit_ceil(uint64_t(c.history_tokens) * 2));
}

} // namespace

uint64_t prompt_lookup_bytes(const LookupConfig& config) {
    validate(config);
    // Bounded by validation: at most 65,536 slots of four words and 2^22 heads.
    return uint64_t(sizeof(PromptLookup)) +
           uint64_t(config.history_tokens) * 4 * sizeof(uint32_t) +
           uint64_t(resolve_buckets(config)) * sizeof(uint32_t);
}

PromptLookup::PromptLookup(LookupConfig config) : config_(config) {
    const uint64_t bytes = prompt_lookup_bytes(config);
    require(bytes <= config.memory_budget_bytes,
            "prompt lookup memory budget cannot hold the configured history and index");
    ring_.assign(config.history_tokens, 0);
    older_.assign(config.history_tokens, none);
    newer_.assign(config.history_tokens, none);
    bucket_.assign(config.history_tokens, none);
    heads_.assign(resolve_buckets(config), none);
    require(charged_bytes() <= config.memory_budget_bytes,
            "prompt lookup allocation exceeded its memory budget");
}

uint64_t PromptLookup::charged_bytes() const {
    owner_check();
    return uint64_t(sizeof(*this)) +
           uint64_t(ring_.capacity() + older_.capacity() + newer_.capacity() + bucket_.capacity() +
                    heads_.capacity()) *
               sizeof(uint32_t);
}

const LookupLedger& PromptLookup::ledger() const {
    owner_check();
    return ledger_;
}

uint32_t PromptLookup::committed_token(uint64_t position) const {
    owner_check();
    require(position >= floor_ && position < length_,
            "prompt lookup position outside the retained committed history");
    return ring_[slot_of(position)];
}

void PromptLookup::owner_check() const {
    require(thread_ == std::this_thread::get_id(), "prompt lookup used from another host thread");
}

void PromptLookup::admit(uint64_t epoch, uint64_t expected_prefix) const {
    owner_check();
    require(epoch == epoch_, "prompt lookup operation carries a stale history epoch");
    require(expected_prefix == length_,
            "prompt lookup operation expects a different committed length");
}

uint32_t PromptLookup::slot_of(uint64_t position) const noexcept {
    return uint32_t(position % config_.history_tokens);
}

uint64_t PromptLookup::position_of(uint32_t slot) const noexcept {
    // Within the window a slot holds exactly one position, floor_ + distance.
    const uint32_t base = slot_of(floor_);
    const uint32_t ahead = slot >= base ? slot - base : slot + (config_.history_tokens - base);
    return floor_ + ahead;
}

uint32_t PromptLookup::bucket_of(uint64_t start) const noexcept {
    // Multiplicative mixing of the exact token IDs, then a mask. The bucket
    // only narrows the candidates; each one is compared token by token.
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < config_.match_tokens; ++i) {
        hash ^= ring_[slot_of(start + i)];
        hash *= 0x100000001b3ull;
    }
    hash ^= hash >> 32;
    hash *= 0x9e3779b97f4a7c15ull;
    hash ^= hash >> 29;
    return uint32_t(hash & uint64_t(heads_.size() - 1));
}

void PromptLookup::link(uint64_t start) {
    const uint32_t slot = slot_of(start);
    require(bucket_[slot] == none, "prompt lookup index already links this gram");
    const uint32_t bucket = bucket_of(start);
    const uint32_t head = heads_[bucket];
    if (head != none) {
        require(newer_[head] == none, "prompt lookup chain head has a newer link");
        newer_[head] = slot;
    }
    older_[slot] = head;
    newer_[slot] = none;
    bucket_[slot] = bucket;
    heads_[bucket] = slot;
}

void PromptLookup::unlink(uint32_t slot) {
    const uint32_t bucket = bucket_[slot];
    require(bucket != none, "prompt lookup index does not link this gram");
    const uint32_t older = older_[slot], newer = newer_[slot];
    if (newer == none) {
        require(heads_[bucket] == slot, "prompt lookup chain head differs from its newest gram");
        heads_[bucket] = older;
    } else {
        require(older_[newer] == slot, "prompt lookup chain newer link is inconsistent");
        older_[newer] = older;
    }
    if (older != none) {
        require(newer_[older] == slot, "prompt lookup chain older link is inconsistent");
        newer_[older] = newer;
    }
    older_[slot] = none;
    newer_[slot] = none;
    bucket_[slot] = none;
}

uint64_t PromptLookup::reset() {
    owner_check();
    uint64_t linked = 0;
    for (const auto bucket : bucket_) {
        if (bucket != none) ++linked;
    }
    std::fill(ring_.begin(), ring_.end(), 0u);
    std::fill(older_.begin(), older_.end(), none);
    std::fill(newer_.begin(), newer_.end(), none);
    std::fill(bucket_.begin(), bucket_.end(), none);
    std::fill(heads_.begin(), heads_.end(), none);
    length_ = 0;
    floor_ = 0;
    ledger_.cleared_grams += linked;
    ++ledger_.resets;
    ++epoch_;
    return epoch_;
}

void PromptLookup::append(uint64_t epoch, uint64_t expected_prefix, std::span<const uint32_t> tokens) {
    admit(epoch, expected_prefix);
    for (const auto token : tokens) {
        require(token < config_.token_limit, "prompt lookup token outside the configured limit");
    }
    const uint32_t history = config_.history_tokens;
    const uint64_t match = config_.match_tokens;
    for (const auto token : tokens) {
        if (length_ - floor_ == history) {
            // The slot about to be reused holds the oldest retained position,
            // whose gram is complete and linked whenever the window is full.
            const uint32_t evicted = slot_of(floor_);
            require(bucket_[evicted] != none, "prompt lookup oldest gram is not linked");
            unlink(evicted);
            ++ledger_.evicted_grams;
            ++floor_;
        }
        const uint32_t slot = slot_of(length_);
        require(bucket_[slot] == none, "prompt lookup reused slot still links a gram");
        ring_[slot] = token;
        ++length_;
        if (length_ >= match && length_ - match >= floor_) {
            link(length_ - match);
            ++ledger_.indexed_grams;
        }
    }
    ledger_.appended_tokens += tokens.size();
}

uint64_t PromptLookup::truncate(uint64_t epoch, uint64_t prefix) {
    owner_check();
    require(epoch == epoch_, "prompt lookup operation carries a stale history epoch");
    require(prefix <= length_, "prompt lookup truncation beyond the committed length");
    const uint64_t match = config_.match_tokens;
    if (length_ >= match) {
        // A gram starting at s covers [s, s + match); it must go when it
        // reaches any position at or after prefix. Every start in the range
        // is linked, newest first, so each unlink finds the chain head.
        const uint64_t newest = length_ - match;
        const uint64_t lowest = std::max(floor_, prefix + 1 > match ? prefix + 1 - match : uint64_t(0));
        for (uint64_t start = newest + 1; start > lowest;) {
            --start;
            unlink(slot_of(start));
            ++ledger_.truncated_grams;
        }
    }
    if (prefix < floor_) floor_ = prefix;
    length_ = prefix;
    ++ledger_.truncations;
    ++epoch_;
    return epoch_;
}

uint64_t PromptLookup::replace(uint64_t epoch, uint64_t prefix, std::span<const uint32_t> tokens) {
    owner_check();
    require(epoch == epoch_, "prompt lookup operation carries a stale history epoch");
    require(prefix <= length_, "prompt lookup replacement beyond the committed length");
    for (const auto token : tokens) {
        require(token < config_.token_limit, "prompt lookup token outside the configured limit");
    }
    const uint64_t next = truncate(epoch, prefix);
    append(next, prefix, tokens);
    return next;
}

LookupProposal PromptLookup::propose(uint64_t epoch, uint64_t expected_prefix) {
    admit(epoch, expected_prefix);
    ++ledger_.queries;
    LookupProposal out;
    out.epoch = epoch_;
    out.base_prefix = length_;
    const uint64_t match = config_.match_tokens;
    const uint32_t width = config_.proposal_tokens;
    if (length_ - floor_ < match + 1) {
        out.outcome = LookupOutcome::short_history;
        ++ledger_.short_history_misses;
        return out;
    }
    // The suffix gram is retained and complete, so it is linked; its chain
    // holds every retained gram with the same bucket, newest first.
    const uint64_t query = length_ - match;
    const uint32_t query_slot = slot_of(query);
    require(bucket_[query_slot] != none, "prompt lookup suffix gram is not linked");
    bool found = false;
    uint64_t best_start = 0, previous = length_;
    uint32_t best_count = 0, examined = 0;
    for (uint32_t slot = heads_[bucket_[query_slot]]; slot != none; slot = older_[slot]) {
        const uint64_t start = position_of(slot);
        require(start < previous && start <= query, "prompt lookup chain order is inconsistent");
        previous = start;
        if (slot == query_slot) continue; // the current suffix never matches itself
        if (examined == config_.work_limit) {
            out.work_limited = true;
            ++ledger_.work_limited_queries;
            break;
        }
        ++examined;
        uint64_t matched = 0;
        while (matched < match && ring_[slot_of(start + matched)] == ring_[slot_of(query + matched)]) {
            ++matched;
        }
        ledger_.token_comparisons += matched < match ? matched + 1 : match;
        if (matched < match) continue;
        // start < query, so at least one committed token follows the occurrence.
        const uint64_t available = length_ - (start + match);
        const uint32_t count = uint32_t(std::min<uint64_t>(available, width));
        if (!found || count > best_count) {
            found = true;
            best_start = start;
            best_count = count;
        }
        if (count == width) break;
    }
    out.candidates_examined = examined;
    ledger_.candidates_examined += examined;
    if (!found) {
        if (out.work_limited) {
            out.outcome = LookupOutcome::work_limited;
        } else {
            out.outcome = LookupOutcome::no_match;
            ++ledger_.no_match_misses;
        }
        return out;
    }
    out.outcome = LookupOutcome::proposed;
    out.source_offset = best_start + match;
    out.count = best_count;
    for (uint32_t i = 0; i < best_count; ++i) {
        out.tokens[i] = ring_[slot_of(out.source_offset + i)];
    }
    ++ledger_.proposals;
    ledger_.proposed_tokens += best_count;
    return out;
}

uint32_t PromptLookup::rebuild() {
    owner_check();
    uint64_t cleared = 0;
    for (const auto bucket : bucket_) {
        if (bucket != none) ++cleared;
    }
    std::fill(older_.begin(), older_.end(), none);
    std::fill(newer_.begin(), newer_.end(), none);
    std::fill(bucket_.begin(), bucket_.end(), none);
    std::fill(heads_.begin(), heads_.end(), none);
    const uint64_t match = config_.match_tokens;
    uint32_t linked = 0;
    for (uint64_t start = floor_; start + match <= length_; ++start) {
        link(start);
        ++linked;
    }
    ledger_.cleared_grams += cleared;
    ledger_.rebuilt_grams += linked;
    ++ledger_.rebuilds;
    return linked;
}

uint32_t PromptLookup::audit() {
    owner_check();
    const uint32_t history = config_.history_tokens;
    const uint64_t match = config_.match_tokens;
    require(floor_ <= length_ && length_ - floor_ <= history,
            "prompt lookup retention window is inconsistent");
    uint32_t linked = 0, tops = 0;
    for (uint32_t slot = 0; slot < history; ++slot) {
        const uint64_t position = position_of(slot);
        const bool complete = position + match <= length_;
        const uint32_t bucket = bucket_[slot];
        if (bucket == none) {
            require(!complete, "prompt lookup index is missing a retained complete gram");
            require(older_[slot] == none && newer_[slot] == none,
                    "prompt lookup unlinked slot keeps chain links");
            continue;
        }
        require(complete, "prompt lookup index links a gram outside the retained window");
        require(bucket < heads_.size() && bucket_of(position) == bucket,
                "prompt lookup index bucket differs from the gram hash");
        const uint32_t older = older_[slot], newer = newer_[slot];
        if (newer == none) {
            require(heads_[bucket] == slot, "prompt lookup chain head is not its newest gram");
            ++tops;
        } else {
            require(newer < history && bucket_[newer] == bucket && older_[newer] == slot &&
                        position_of(newer) > position,
                    "prompt lookup chain newer link is inconsistent");
        }
        if (older != none) {
            require(older < history && bucket_[older] == bucket && newer_[older] == slot &&
                        position_of(older) < position,
                    "prompt lookup chain older link is inconsistent");
        }
        ++linked;
    }
    uint32_t heads = 0;
    for (size_t bucket = 0; bucket < heads_.size(); ++bucket) {
        const uint32_t slot = heads_[bucket];
        if (slot == none) continue;
        require(slot < history && bucket_[slot] == bucket && newer_[slot] == none,
                "prompt lookup chain head is inconsistent");
        ++heads;
    }
    require(heads == tops, "prompt lookup chain heads differ from chain tops");
    const uint64_t expected =
        length_ >= match && length_ - match + 1 > floor_ ? length_ - match + 1 - floor_ : 0;
    require(linked == expected, "prompt lookup linked gram count differs from the retained window");
    ++ledger_.audits;
    return linked;
}

} // namespace tokspur
