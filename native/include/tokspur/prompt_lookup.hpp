#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace tokspur {

// Bounded, model-independent prompt lookup over committed tokens. It finds an
// earlier retained occurrence of the current committed suffix and proposes the
// tokens that followed it. It runs no model, publishes no model state and
// executes nothing. Target verification, acceptance sampling, stop handling
// and composite-state rollback belong to the caller. This is a proposal
// source only and is unrelated to the model's n-gram embedding history.
inline constexpr uint32_t maximum_match_tokens = 64;
inline constexpr uint32_t maximum_proposal_tokens = 16;
inline constexpr uint32_t maximum_history_tokens = 65536;
inline constexpr uint32_t maximum_lookup_buckets = 1u << 22;
inline constexpr uint32_t maximum_lookup_work = 1u << 20;
inline constexpr uint64_t maximum_lookup_budget_bytes = 1ull << 30;

struct LookupConfig {
    // Exact suffix length compared token by token, in [1, maximum_match_tokens].
    uint32_t match_tokens = 16;
    // Longest continuation returned, in [1, maximum_proposal_tokens].
    uint32_t proposal_tokens = 7;
    // Retained committed tokens, in (match_tokens, maximum_history_tokens].
    // Older tokens leave the window oldest first and are never proposed.
    uint32_t history_tokens = maximum_history_tokens;
    // Exclusive bound on accepted token IDs; at least two. Appends carrying a
    // token at or above it are refused before any state changes.
    uint32_t token_limit = 1u << 20;
    // Candidate occurrences examined per query, in [1, maximum_lookup_work].
    // Each examination costs at most match_tokens token comparisons.
    uint32_t work_limit = 128;
    // Zero selects the smallest power of two not below 2 * history_tokens.
    // Otherwise a power of two up to maximum_lookup_buckets. A smaller table
    // only lengthens candidate chains; it never changes a result.
    uint32_t index_buckets = 0;
    // Must cover prompt_lookup_bytes(config) and be at most
    // maximum_lookup_budget_bytes. No successful operation allocates after
    // construction; a refused one allocates only the FormatError it throws.
    uint64_t memory_budget_bytes = 16ull << 20;
};

enum class LookupOutcome : uint32_t {
    proposed,      // count tokens copied from one retained earlier occurrence
    short_history, // fewer than match_tokens + 1 retained tokens: no earlier occurrence can exist
    no_match,      // every candidate was examined and none matched the suffix exactly
    work_limited   // the walk reached work_limit before any exact occurrence was found
};

// Owned fixed-capacity value. It borrows nothing from the history, so it stays
// readable after later appends, edits or resets; its epoch and base prefix say
// which history it extends. tokens[i] equals the committed token at position
// source_offset + i, and the matched occurrence starts at
// source_offset - match_tokens. Unused capacity stays zero.
struct LookupProposal {
    LookupOutcome outcome = LookupOutcome::short_history;
    uint64_t epoch = 0;
    uint64_t base_prefix = 0;
    uint64_t source_offset = 0;
    uint32_t count = 0;
    // Exact comparisons attempted by this query.
    uint32_t candidates_examined = 0;
    // The walk stopped at work_limit. A proposal may then be shorter than the
    // best retained one; a miss then reports LookupOutcome::work_limited.
    bool work_limited = false;
    std::array<uint32_t, maximum_proposal_tokens> tokens{};
    [[nodiscard]] std::span<const uint32_t> proposed() const noexcept {
        return std::span<const uint32_t>(tokens.data(), count);
    }
};

// Maintenance and query costs stay visible here so later comparisons charge
// history upkeep, not only the tokens a proposal saved. Every counter is
// monotonic for the lifetime of the object.
struct LookupLedger {
    uint64_t appended_tokens = 0;     // tokens committed through append or replace
    uint64_t indexed_grams = 0;       // grams linked as they completed: match_tokens token reads each
    uint64_t rebuilt_grams = 0;       // grams linked again by rebuild
    uint64_t evicted_grams = 0;       // grams unlinked because the retention window slid
    uint64_t truncated_grams = 0;     // grams unlinked by truncate or replace
    uint64_t cleared_grams = 0;       // grams dropped wholesale by reset or rebuild
    uint64_t truncations = 0;
    uint64_t resets = 0;
    uint64_t rebuilds = 0;
    uint64_t audits = 0;
    uint64_t queries = 0;             // propose calls that passed admission
    uint64_t proposals = 0;           // queries that returned at least one token
    uint64_t proposed_tokens = 0;
    uint64_t candidates_examined = 0; // exact comparisons attempted
    uint64_t token_comparisons = 0;   // token equality tests inside those comparisons
    uint64_t work_limited_queries = 0;
    uint64_t short_history_misses = 0;
    uint64_t no_match_misses = 0;
    bool operator==(const LookupLedger&) const = default;
};

// Exact bytes a valid configuration charges: the object plus five fixed
// uint32 arrays (token, two chain links and bucket per history slot, one head
// per bucket). Invalid configurations are refused with FormatError.
[[nodiscard]] uint64_t prompt_lookup_bytes(const LookupConfig& config);

// Single-owner history and index. One ring of history_tokens slots holds the
// retained committed tokens; every complete match_tokens-gram that starts
// inside the window is linked, newest first, into the chain of its hash
// bucket. Hashing only narrows candidates: every proposal comes from an exact
// token-by-token comparison, and the current suffix never matches itself.
//
// Selection rule: among the retained earlier occurrences of the suffix, take
// the one with the most available continuation tokens up to proposal_tokens;
// ties go to the most recent occurrence. Chains are walked newest first, so
// the first occurrence with a full-width continuation ends the walk.
//
// Epochs: reset, truncate and replace open a new epoch, because committed
// positions may then carry different tokens. Appends keep the epoch, because
// every earlier position is unchanged. Operations and proposals carry the
// epoch and the committed length they expect; mismatches throw FormatError
// and change nothing. All methods must run on the constructing thread.
class PromptLookup {
public:
    explicit PromptLookup(LookupConfig config);
    PromptLookup(const PromptLookup&) = delete;
    PromptLookup& operator=(const PromptLookup&) = delete;

    // The plain readers also refuse another thread: they read the same
    // non-atomic state as every operation.
    [[nodiscard]] const LookupConfig& config() const { owner_check(); return config_; }
    [[nodiscard]] uint64_t epoch() const { owner_check(); return epoch_; }
    [[nodiscard]] uint64_t committed_length() const { owner_check(); return length_; }
    // Oldest committed position still retained; retained positions are
    // [retained_begin(), committed_length()).
    [[nodiscard]] uint64_t retained_begin() const { owner_check(); return floor_; }
    [[nodiscard]] uint32_t retained_tokens() const { owner_check(); return uint32_t(length_ - floor_); }
    [[nodiscard]] uint32_t index_buckets() const { owner_check(); return uint32_t(heads_.size()); }
    [[nodiscard]] uint64_t charged_bytes() const;
    [[nodiscard]] const LookupLedger& ledger() const;
    [[nodiscard]] uint32_t committed_token(uint64_t position) const;

    // Forgets every token and link, opens a new epoch and returns it.
    uint64_t reset();
    // Commits tokens after the current committed length. Tokens are checked
    // against token_limit before any state changes. Grams are indexed as
    // they complete; the oldest gram is unlinked when the window slides.
    void append(uint64_t epoch, uint64_t expected_prefix, std::span<const uint32_t> tokens);
    // Keeps [0, prefix), unlinks every gram that reached a removed position,
    // opens a new epoch and returns it. A prefix below retained_begin()
    // leaves nothing retained. prefix == committed_length() is a valid
    // branch point that still opens a new epoch.
    uint64_t truncate(uint64_t epoch, uint64_t prefix);
    // truncate followed by append in one operation; returns the new epoch.
    // The tokens are checked before the truncation.
    uint64_t replace(uint64_t epoch, uint64_t prefix, std::span<const uint32_t> tokens);
    // Bounded query with no allocation: at most work_limit exact comparisons
    // of at most match_tokens tokens each. Misses are reported in the result.
    [[nodiscard]] LookupProposal propose(uint64_t epoch, uint64_t expected_prefix);
    // Drops every link and relinks every retained complete gram from the ring,
    // using no memory beyond the fixed arrays. Returns the linked gram count.
    // Content and epoch are unchanged. Cost: history_tokens + index_buckets
    // writes plus match_tokens reads per gram.
    uint32_t rebuild();
    // Verifies every invariant (window bounds, each retained complete gram
    // linked under its hash bucket, mutually consistent newest-first chains,
    // every head the newest gram of its bucket). Throws FormatError on the
    // first violation; returns the linked gram count. Cost: as rebuild.
    uint32_t audit();

private:
    static constexpr uint32_t none = 0xffffffffu;
    LookupConfig config_;
    uint64_t epoch_ = 1, length_ = 0, floor_ = 0;
    std::vector<uint32_t> ring_, older_, newer_, bucket_, heads_;
    LookupLedger ledger_;
    const std::thread::id thread_ = std::this_thread::get_id();
    void owner_check() const;
    void admit(uint64_t epoch, uint64_t expected_prefix) const;
    [[nodiscard]] uint32_t slot_of(uint64_t position) const noexcept;
    [[nodiscard]] uint64_t position_of(uint32_t slot) const noexcept;
    [[nodiscard]] uint32_t bucket_of(uint64_t start) const noexcept;
    void link(uint64_t start);
    void unlink(uint32_t slot);
};

} // namespace tokspur
