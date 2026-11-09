#include "tokspur/prompt_lookup.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace tokspur;
size_t checks = 0;
void require(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid prompt lookup request admitted");
}
using Tokens = std::vector<uint32_t>;

// The work limit equals the history so no oracle comparison is cut short.
LookupConfig configure(uint32_t match, uint32_t width, uint32_t history, uint32_t buckets = 0) {
    LookupConfig config;
    config.match_tokens = match;
    config.proposal_tokens = width;
    config.history_tokens = history;
    config.index_buckets = buckets;
    config.work_limit = history;
    return config;
}

// Independent brute-force reference: a plain vector of every committed token
// with the documented retention and selection rules. No hashing, no ring.
class Oracle {
public:
    Oracle(uint32_t match, uint32_t width, uint32_t history) : match_(match), width_(width), history_(history) {}
    void reset() { tokens_.clear(); floor_ = 0; }
    void append(const Tokens& more) {
        for (const auto token : more) {
            tokens_.push_back(token);
            if (tokens_.size() > history_) floor_ = std::max<uint64_t>(floor_, tokens_.size() - history_);
        }
    }
    void truncate(uint64_t prefix) {
        tokens_.resize(prefix);
        floor_ = std::min<uint64_t>(floor_, prefix);
    }
    [[nodiscard]] uint64_t length() const { return tokens_.size(); }
    [[nodiscard]] uint64_t retained_begin() const { return floor_; }
    [[nodiscard]] const Tokens& tokens() const { return tokens_; }
    [[nodiscard]] uint64_t linked_grams() const {
        return tokens_.size() >= match_ && tokens_.size() - match_ + 1 > floor_ ? tokens_.size() - match_ + 1 - floor_ : 0;
    }
    struct Expectation {
        bool enough_history = false;
        bool proposed = false;
        uint64_t source = 0;
        Tokens continuation;
    };
    [[nodiscard]] Expectation propose() const {
        Expectation out;
        const uint64_t length = tokens_.size();
        if (length - floor_ < uint64_t(match_) + 1) return out;
        out.enough_history = true;
        const uint64_t query = length - match_;
        uint32_t best_count = 0;
        for (uint64_t start = query; start > floor_;) {
            --start;
            if (!std::equal(tokens_.begin() + ptrdiff_t(start), tokens_.begin() + ptrdiff_t(start + match_),
                            tokens_.begin() + ptrdiff_t(query)))
                continue;
            const uint32_t count = uint32_t(std::min<uint64_t>(length - (start + match_), width_));
            if (!out.proposed || count > best_count) {
                out.proposed = true;
                out.source = start + match_;
                best_count = count;
            }
            if (count == width_) break;
        }
        if (out.proposed)
            out.continuation.assign(tokens_.begin() + ptrdiff_t(out.source),
                                    tokens_.begin() + ptrdiff_t(out.source + best_count));
        return out;
    }

private:
    uint32_t match_, width_, history_;
    Tokens tokens_;
    uint64_t floor_ = 0;
};

// Drives the component and the oracle with identical operations and checks
// every observable result, the index audit and the maintenance ledger.
class Harness {
public:
    explicit Harness(const LookupConfig& config)
        : lookup_(config), oracle_(config.match_tokens, config.proposal_tokens, config.history_tokens),
          epoch_(lookup_.epoch()) {}
    [[nodiscard]] PromptLookup& lookup() { return lookup_; }
    [[nodiscard]] const Oracle& oracle() const { return oracle_; }
    [[nodiscard]] uint64_t epoch() const { return epoch_; }
    // Audit through the harness, so the independent ledger replay sees the call.
    auto audit() { ++expected_.audits; return lookup_.audit(); }
    LookupProposal check() { return agree(); }
    LookupProposal append(const Tokens& tokens) {
        account_append(tokens.size());
        lookup_.append(epoch_, lookup_.committed_length(), tokens);
        oracle_.append(tokens);
        return agree();
    }
    LookupProposal truncate(uint64_t prefix) {
        account_truncate(prefix);
        epoch_ = lookup_.truncate(epoch_, prefix);
        oracle_.truncate(prefix);
        return agree();
    }
    LookupProposal replace(uint64_t prefix, const Tokens& tokens) {
        account_truncate(prefix);
        oracle_.truncate(prefix);
        account_append(tokens.size());
        epoch_ = lookup_.replace(epoch_, prefix, tokens);
        oracle_.append(tokens);
        return agree();
    }
    LookupProposal reset() {
        expected_.cleared_grams += oracle_.linked_grams();
        ++expected_.resets;
        epoch_ = lookup_.reset();
        oracle_.reset();
        return agree();
    }
    LookupProposal rebuild() {
        static_cast<void>(agree());
        expected_.cleared_grams += oracle_.linked_grams();
        expected_.rebuilt_grams += oracle_.linked_grams();
        ++expected_.rebuilds;
        require(lookup_.rebuild() == oracle_.linked_grams(), "rebuild linked a different gram count");
        return agree();
    }

private:
    PromptLookup lookup_;
    Oracle oracle_;
    uint64_t epoch_;
    LookupLedger expected_;

    // Replays the documented rules from the oracle's state: a gram links when
    // it completes inside the window; the oldest unlinks when the window slides.
    void account_append(size_t count) {
        const uint64_t match = lookup_.config().match_tokens, history = lookup_.config().history_tokens;
        uint64_t length = oracle_.length(), oldest = oracle_.retained_begin();
        for (size_t i = 0; i < count; ++i) {
            if (length - oldest == history) {
                ++expected_.evicted_grams;
                ++oldest;
            }
            ++length;
            if (length >= match && length - match >= oldest) ++expected_.indexed_grams;
        }
        expected_.appended_tokens += count;
    }
    void account_truncate(uint64_t prefix) {
        const uint64_t match = lookup_.config().match_tokens, oldest = oracle_.retained_begin();
        const uint64_t surviving =
            prefix >= match && prefix - match + 1 > oldest ? prefix - match + 1 - oldest : uint64_t(0);
        expected_.truncated_grams += oracle_.linked_grams() - surviving;
        ++expected_.truncations;
    }
    LookupProposal agree() {
        const auto expected = oracle_.propose();
        const auto actual = lookup_.propose(epoch_, lookup_.committed_length());
        ++expected_.queries;
        const uint32_t match = lookup_.config().match_tokens;
        require(actual.epoch == epoch_ && actual.base_prefix == oracle_.length(),
                "proposal identity differs from the committed history");
        require(!actual.work_limited, "an unlimited walk reported a work limit");
        if (!expected.enough_history) {
            require(actual.outcome == LookupOutcome::short_history && actual.count == 0,
                    "short-history outcome differs from the oracle");
            ++expected_.short_history_misses;
        } else if (!expected.proposed) {
            require(actual.outcome == LookupOutcome::no_match && actual.count == 0,
                    "no-match outcome differs from the oracle");
            ++expected_.no_match_misses;
        } else {
            require(actual.outcome == LookupOutcome::proposed && actual.count == expected.continuation.size(),
                    "proposal outcome or width differs from the oracle");
            require(actual.source_offset == expected.source, "selected occurrence differs from the oracle");
            require(std::equal(actual.proposed().begin(), actual.proposed().end(), expected.continuation.begin()),
                    "proposed continuation differs from the oracle");
            require(actual.source_offset >= lookup_.retained_begin() + match &&
                        actual.source_offset + actual.count <= actual.base_prefix,
                    "proposal source outside the retained history");
            for (uint32_t i = 0; i < actual.count; ++i) {
                require(lookup_.committed_token(actual.source_offset + i) == actual.tokens[i] &&
                            oracle_.tokens()[actual.source_offset + i] == actual.tokens[i],
                        "proposed token is not the committed continuation");
            }
            ++expected_.proposals;
            expected_.proposed_tokens += actual.count;
        }
        for (uint32_t i = actual.count; i < maximum_proposal_tokens; ++i) {
            require(actual.tokens[i] == 0, "unused proposal capacity is not zero");
        }
        require(lookup_.committed_length() == oracle_.length() && lookup_.retained_begin() == oracle_.retained_begin(),
                "retention window differs from the oracle");
        require(lookup_.audit() == oracle_.linked_grams(), "index audit differs from the oracle gram count");
        ++expected_.audits;
        const LookupLedger ledger = lookup_.ledger();
        require(ledger.token_comparisons >= ledger.candidates_examined &&
                    ledger.token_comparisons <= ledger.candidates_examined * match,
                "comparison accounting outside its bounds");
        require(ledger.candidates_examined <= ledger.queries * lookup_.config().work_limit,
                "examined candidates exceed the work limit");
        // Only these two depend on hash collisions; everything else is predicted independently.
        expected_.candidates_examined = ledger.candidates_examined;
        expected_.token_comparisons = ledger.token_comparisons;
        require(ledger == expected_, "maintenance ledger differs from the independent accounting");
        require(ledger.indexed_grams + ledger.rebuilt_grams ==
                    ledger.evicted_grams + ledger.truncated_grams + ledger.cleared_grams + oracle_.linked_grams(),
                "gram conservation failed");
        return actual;
    }
};

bool proposes(const LookupProposal& actual, uint64_t source, const Tokens& continuation) {
    return actual.outcome == LookupOutcome::proposed && actual.source_offset == source &&
           actual.count == continuation.size() &&
           std::equal(continuation.begin(), continuation.end(), actual.proposed().begin());
}

void contract_basics() {
    // Minimum history: two-token match, three-token window.
    Harness minimal(configure(2, 3, 3));
    require(minimal.check().outcome == LookupOutcome::short_history, "empty history must miss");
    require(minimal.append({5}).outcome == LookupOutcome::short_history, "one token must miss");
    require(minimal.append({5}).outcome == LookupOutcome::short_history, "match width alone must miss");
    require(proposes(minimal.append({5}), 2, {5}), "exact minimum match must propose one token");
    require(proposes(minimal.append({5}), 3, {5}), "first eviction keeps the minimum match");
    require(minimal.lookup().retained_begin() == 1 && minimal.lookup().retained_tokens() == 3 &&
                minimal.lookup().ledger().evicted_grams == 1 && minimal.lookup().committed_token(1) == 5,
            "window slid by one token");
    rejected([&] { static_cast<void>(minimal.lookup().committed_token(0)); });
    require(minimal.append({6}).outcome == LookupOutcome::no_match, "changed suffix must miss");
    require(minimal.append({5}).outcome == LookupOutcome::no_match, "rotated suffix must miss");
    require(proposes(minimal.append({6}), 4, {5}) == false && minimal.lookup().retained_begin() == 4,
            "window keeps sliding");

    // The suffix never matches itself, and an overlapping earlier occurrence counts.
    Harness unique(configure(2, 3, 16));
    require(unique.append({1, 2, 3, 4, 5}).outcome == LookupOutcome::no_match, "unique suffix must miss");
    require(unique.append({4}).outcome == LookupOutcome::no_match, "unique two-token suffix must miss");
    require(proposes(unique.append({5}), 5, {4, 5}), "overlapping continuation into the suffix");
    require(unique.append({6, 7}).outcome == LookupOutcome::no_match, "fresh suffix must miss");
}

void selection_rule() {
    // Two full-width occurrences: the most recent wins the tie.
    Harness tie(configure(2, 3, 64));
    require(proposes(tie.append({1, 2, 7, 8, 9, 1, 2, 5, 6, 4, 1, 2}), 7, {5, 6, 4}),
            "most recent full-width occurrence must win the tie");
    require(proposes(tie.truncate(7), 2, {7, 8, 9}), "truncation exposes the older occurrence");
    require(proposes(tie.replace(7, {5, 6, 4, 1, 2}), 7, {5, 6, 4}), "replacement restores the newer occurrence");
    // A longer continuation beats a newer shorter one.
    Harness longer(configure(2, 7, 64));
    require(proposes(longer.append({1, 2, 9, 1, 2, 8, 1, 2}), 2, {9, 1, 2, 8, 1, 2}),
            "longer continuation must beat the newer occurrence");
    // Runs of one token: overlapping occurrences, partial then full width.
    Harness run(configure(3, 7, 64));
    require(proposes(run.append(Tokens(6, 4)), 3, {4, 4, 4}), "short run proposes what is available");
    require(proposes(run.append(Tokens(14, 4)), 13, Tokens(7, 4)), "long run proposes the full width");
    // A periodic sequence: the oldest occurrence has the longest continuation.
    Harness periodic(configure(2, 7, 64));
    require(proposes(periodic.append({1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2}), 2, {3, 1, 2, 3, 1, 2, 3}),
            "periodic history proposes the full period");
}

void collisions() {
    // One bucket makes every retained gram a candidate: four collisions precede
    // the only exact occurrence, and none of them is proposed.
    Harness single(configure(2, 3, 32, 1));
    const auto colliding = single.append({1, 2, 3, 4, 5, 6, 2, 3});
    require(proposes(colliding, 3, {4, 5, 6}) && colliding.candidates_examined == 5,
            "collisions must be rejected by exact comparison");
    Harness hashed(configure(2, 3, 32));
    const auto distinct = hashed.append({1, 2, 3, 4, 5, 6, 2, 3});
    require(proposes(distinct, 3, {4, 5, 6}) && distinct.candidates_examined <= colliding.candidates_examined,
            "bucket count changes work, never the proposal");
}

Tokens block(uint32_t name) {
    return Tokens{10, 20 + name, 30, 31, 32, 40, 41, 42, 43, 50};
}

void code_like_repetition() {
    // Three near-identical declarations, then a fourth with a new name token.
    Harness code(configure(4, 7, 256));
    for (uint32_t name = 1; name <= 3; ++name) static_cast<void>(code.append(block(name)));
    // The shared body also ends every block, so the latest block end already
    // proposes the previous block's recorded continuation: the next block start.
    require(proposes(code.check(), 20, {10, 23, 30, 31, 32, 40, 41}),
            "block end proposes the previous block's continuation");
    // After the new name, the shared body matches the most recent block and the
    // recorded continuation includes that block's following name token.
    require(proposes(code.append({10, 24, 30, 31, 32, 40}), 26, {41, 42, 43, 50, 10, 24, 30}),
            "shared body proposes the most recent recorded continuation");
    static_cast<void>(code.append({41, 42, 43, 50}));
    require(code.append({10, 25}).outcome == LookupOutcome::no_match, "new name token breaks the match");
    require(code.append({30, 31}).outcome == LookupOutcome::no_match, "name still inside the suffix");
    require(code.append({32}).outcome == LookupOutcome::no_match, "name still inside the suffix");
    require(proposes(code.append({40}), 36, {41, 42, 43, 50, 10, 25, 30}),
            "newest block becomes the proposal source");
}

void epochs_and_prefixes() {
    PromptLookup lookup(configure(2, 3, 8));
    require(lookup.epoch() == 1 && lookup.committed_length() == 0 && lookup.retained_begin() == 0 &&
                lookup.retained_tokens() == 0,
            "fresh state");
    const Tokens abc{1, 2, 3};
    rejected([&] { lookup.append(2, 0, abc); });
    rejected([&] { lookup.append(1, 1, abc); });
    rejected([&] { static_cast<void>(lookup.propose(2, 0)); });
    rejected([&] { static_cast<void>(lookup.propose(1, 1)); });
    rejected([&] { static_cast<void>(lookup.truncate(1, 1)); });
    rejected([&] { static_cast<void>(lookup.truncate(2, 0)); });
    rejected([&] { static_cast<void>(lookup.replace(2, 0, abc)); });
    rejected([&] { static_cast<void>(lookup.replace(1, 1, abc)); });
    rejected([&] { static_cast<void>(lookup.committed_token(0)); });
    require(lookup.committed_length() == 0 && lookup.audit() == 0 && lookup.ledger().appended_tokens == 0 &&
                lookup.ledger().queries == 0 && lookup.ledger().truncations == 0,
            "refused operations changed state");
    lookup.append(1, 0, abc);
    lookup.append(1, 3, Tokens{1, 2});
    const auto first = lookup.propose(1, 5);
    require(first.epoch == 1 && first.base_prefix == 5 && proposes(first, 2, {3, 1, 2}),
            "proposal from the original epoch");
    rejected([&] { static_cast<void>(lookup.propose(1, 4)); });
    const auto second = lookup.truncate(1, 4);
    require(second == 2 && lookup.epoch() == 2 && lookup.committed_length() == 4 && lookup.audit() == 3,
            "truncation opens a new epoch");
    rejected([&] { static_cast<void>(lookup.propose(1, 4)); });
    rejected([&] { lookup.append(1, 4, abc); });
    rejected([&] { static_cast<void>(lookup.truncate(1, 2)); });
    require(first.epoch == 1 && proposes(first, 2, {3, 1, 2}), "owned proposal changed after truncation");
    const auto stale = lookup.propose(2, 4);
    require(stale.outcome == LookupOutcome::no_match && stale.count == 0 && stale.epoch == 2 &&
                stale.base_prefix == 4,
            "post-truncation suffix has no earlier occurrence");
    const auto third = lookup.replace(2, 1, Tokens{2, 3, 1, 2});
    require(third == 3 && lookup.epoch() == 3 && lookup.committed_length() == 5 && lookup.audit() == 4,
            "replacement opens a new epoch");
    const auto again = lookup.propose(3, 5);
    require(again.epoch == 3 && proposes(again, 2, {3, 1, 2}), "replacement reproduces the original continuation");
    require(lookup.truncate(3, 5) == 4 && lookup.committed_length() == 5 && lookup.audit() == 4,
            "same-length truncation opens an epoch without removing grams");
    const auto ledger = lookup.ledger();
    require(ledger.truncations == 3 && ledger.truncated_grams == 4 && ledger.indexed_grams == 8 &&
                ledger.appended_tokens == 9 && ledger.queries == 3 && ledger.proposals == 2 &&
                ledger.proposed_tokens == 6 && ledger.no_match_misses == 1 && ledger.evicted_grams == 0,
            "edit accounting");
    const auto fifth = lookup.reset();
    require(fifth == 5 && lookup.epoch() == 5 && lookup.committed_length() == 0 && lookup.retained_begin() == 0 &&
                lookup.audit() == 0 && lookup.ledger().cleared_grams == 4 && lookup.ledger().resets == 1,
            "reset clears and opens a new epoch");
    rejected([&] { static_cast<void>(lookup.propose(4, 0)); });
    require(lookup.propose(5, 0).outcome == LookupOutcome::short_history, "fresh epoch after reset");

    // Token limit: refused appends and replacements change nothing.
    LookupConfig limited = configure(2, 3, 8);
    limited.token_limit = 10;
    PromptLookup small(limited);
    small.append(1, 0, Tokens{9, 9, 9});
    const auto before = small.propose(1, 3);
    require(proposes(before, 2, {9}), "limit-admitted tokens propose");
    rejected([&] { small.append(1, 3, Tokens{1, 10}); });
    rejected([&] { static_cast<void>(small.replace(1, 1, Tokens{10})); });
    require(small.committed_length() == 3 && small.epoch() == 1 && small.audit() == 2 &&
                small.ledger().appended_tokens == 3 && small.ledger().truncations == 0,
            "refused tokens changed the history");
    const auto after = small.propose(1, 3);
    require(proposes(after, 2, {9}) && after.epoch == before.epoch, "refused tokens changed the proposal");
}

void invalid_configuration() {
    const LookupConfig base = configure(16, 7, 65536);
    const auto refuse = [&](auto mutate) {
        LookupConfig bad = base;
        mutate(bad);
        rejected([&] { static_cast<void>(prompt_lookup_bytes(bad)); });
        rejected([&] { static_cast<void>(PromptLookup(bad)); });
    };
    refuse([](LookupConfig& c) { c.match_tokens = 0; });
    refuse([](LookupConfig& c) { c.match_tokens = maximum_match_tokens + 1; });
    refuse([](LookupConfig& c) { c.proposal_tokens = 0; });
    refuse([](LookupConfig& c) { c.proposal_tokens = maximum_proposal_tokens + 1; });
    refuse([](LookupConfig& c) { c.history_tokens = c.match_tokens; });
    refuse([](LookupConfig& c) { c.history_tokens = maximum_history_tokens + 1; });
    refuse([](LookupConfig& c) { c.token_limit = 0; });
    refuse([](LookupConfig& c) { c.token_limit = 1; });
    refuse([](LookupConfig& c) { c.work_limit = 0; });
    refuse([](LookupConfig& c) { c.work_limit = maximum_lookup_work + 1; });
    refuse([](LookupConfig& c) { c.index_buckets = 3; });
    refuse([](LookupConfig& c) { c.index_buckets = maximum_lookup_buckets * 2; });
    refuse([](LookupConfig& c) { c.memory_budget_bytes = 0; });
    refuse([](LookupConfig& c) { c.memory_budget_bytes = maximum_lookup_budget_bytes + 1; });
    // The charge does not depend on the budget; only the constructor compares them.
    LookupConfig short_budget = base;
    short_budget.memory_budget_bytes = prompt_lookup_bytes(base) - 1;
    require(prompt_lookup_bytes(short_budget) == prompt_lookup_bytes(base), "charge changed with the budget");
    rejected([&] { static_cast<void>(PromptLookup(short_budget)); });
    const auto admit = [&](auto mutate) {
        LookupConfig good = base;
        good.memory_budget_bytes = maximum_lookup_budget_bytes;
        mutate(good);
        PromptLookup instance(good);
        require(instance.charged_bytes() == prompt_lookup_bytes(good) && instance.epoch() == 1,
                "valid boundary configuration refused or mischarged");
    };
    admit([](LookupConfig& c) { c.match_tokens = 1; c.history_tokens = 2; });
    admit([](LookupConfig& c) { c.match_tokens = maximum_match_tokens; c.history_tokens = maximum_match_tokens + 1; });
    admit([](LookupConfig& c) { c.proposal_tokens = 1; });
    admit([](LookupConfig& c) { c.proposal_tokens = maximum_proposal_tokens; });
    admit([](LookupConfig& c) { c.token_limit = 2; });
    admit([](LookupConfig& c) { c.work_limit = 1; });
    admit([](LookupConfig& c) { c.work_limit = maximum_lookup_work; });
    admit([](LookupConfig& c) { c.index_buckets = 1; });
    admit([](LookupConfig& c) { c.index_buckets = maximum_lookup_buckets; });
}

void memory_budget() {
    const LookupConfig standard{};
    require(standard.match_tokens == 16 && standard.proposal_tokens == 7 && standard.history_tokens == 65536 &&
                standard.token_limit == (1u << 20) && standard.work_limit == 128 && standard.index_buckets == 0 &&
                standard.memory_budget_bytes == (16ull << 20),
            "documented defaults changed");
    const uint64_t bytes = prompt_lookup_bytes(standard);
    require(bytes == sizeof(PromptLookup) + 16ull * 65536 + 4ull * 131072, "memory formula differs from the documented arrays");
    require(bytes < standard.memory_budget_bytes, "default configuration does not fit its budget");
    PromptLookup instance(standard);
    require(instance.charged_bytes() == bytes && instance.index_buckets() == 131072, "charged bytes or bucket count differ");
    LookupConfig exact = standard;
    exact.memory_budget_bytes = bytes;
    PromptLookup tight(exact);
    require(tight.charged_bytes() == bytes, "exact budget boundary refused");
    exact.memory_budget_bytes = bytes - 1;
    rejected([&] { static_cast<void>(PromptLookup(exact)); });
    LookupConfig wide = standard;
    wide.index_buckets = 1u << 20;
    require(prompt_lookup_bytes(wide) == bytes + 4ull * ((1u << 20) - 131072), "explicit bucket count is not charged");
    require(prompt_lookup_bytes(configure(16, 7, 65536, 1)) == sizeof(PromptLookup) + 16ull * 65536 + 4,
            "single bucket charge differs");
    // Nothing is allocated after construction, whatever the history does.
    Tokens filler(70000);
    for (size_t i = 0; i < filler.size(); ++i) filler[i] = uint32_t(i % 977);
    instance.append(1, 0, filler);
    static_cast<void>(instance.propose(1, 70000));
    static_cast<void>(instance.truncate(1, 1000));
    static_cast<void>(instance.rebuild());
    static_cast<void>(instance.reset());
    require(instance.charged_bytes() == bytes, "operations changed the charged bytes");
}

void work_limits() {
    LookupConfig config = configure(1, 1, 64, 1);
    config.work_limit = 3;
    const Tokens history{9, 1, 2, 3, 4, 9};
    PromptLookup limited(config);
    limited.append(1, 0, history);
    auto result = limited.propose(1, 6);
    require(result.outcome == LookupOutcome::work_limited && result.work_limited && result.count == 0 &&
                result.candidates_examined == 3 && result.base_prefix == 6 && result.epoch == 1,
            "work-limited miss not reported");
    require(limited.ledger().work_limited_queries == 1 && limited.ledger().candidates_examined == 3 &&
                limited.ledger().token_comparisons == 3 && limited.ledger().proposals == 0 &&
                limited.ledger().no_match_misses == 0,
            "work-limited miss accounting");
    config.work_limit = 5;
    PromptLookup enough(config);
    enough.append(1, 0, history);
    result = enough.propose(1, 6);
    require(!result.work_limited && result.candidates_examined == 5 && proposes(result, 1, {1}),
            "five candidates reach the only earlier occurrence");
    require(enough.ledger().token_comparisons == 5 && enough.ledger().work_limited_queries == 0,
            "complete walk accounting");
    // A match found before the limit is kept and flagged; the complete walk
    // prefers the older occurrence with the longer continuation.
    LookupConfig partial = configure(1, 3, 64, 1);
    partial.work_limit = 2;
    const Tokens repeated{7, 1, 2, 3, 7, 5, 7};
    PromptLookup flagged(partial);
    flagged.append(1, 0, repeated);
    result = flagged.propose(1, 7);
    require(result.work_limited && result.candidates_examined == 2 && proposes(result, 5, {5, 7}),
            "flagged partial proposal");
    require(flagged.ledger().work_limited_queries == 1 && flagged.ledger().proposals == 1 &&
                flagged.ledger().proposed_tokens == 2,
            "flagged proposal accounting");
    partial.work_limit = 64;
    PromptLookup complete(partial);
    complete.append(1, 0, repeated);
    result = complete.propose(1, 7);
    require(!result.work_limited && result.candidates_examined == 6 && proposes(result, 1, {1, 2, 3}),
            "complete walk selects the longest continuation");
}

void cross_thread() {
    PromptLookup lookup(configure(2, 3, 8));
    lookup.append(1, 0, Tokens{1, 2, 3});
    unsigned refusals = 0;
    std::thread other([&] {
        try { static_cast<void>(lookup.ledger()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.config()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.epoch()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.committed_length()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.retained_begin()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.retained_tokens()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.index_buckets()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.charged_bytes()); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(lookup.propose(1, 3)); } catch (const FormatError&) { ++refusals; }
        try { lookup.append(1, 3, Tokens{4}); } catch (const FormatError&) { ++refusals; }
    });
    other.join();
    require(refusals == 10 && lookup.committed_length() == 3, "another host thread used the lookup");
}

// Recorded seeds; small alphabets and copied spans create many overlapping
// occurrences, edits, wraps and truncations below the retained window.
void randomized(uint64_t seed, const LookupConfig& config, size_t operations) {
    std::mt19937_64 engine(seed);
    Harness harness(config);
    static_cast<void>(harness.check());
    const uint32_t alphabet = 2 + uint32_t(engine() % 5);
    const auto draw = [&](uint64_t bound) -> uint64_t { return bound ? engine() % bound : 0; };
    for (size_t op = 0; op < operations; ++op) {
        const auto kind = engine() % 20;
        const uint64_t length = harness.oracle().length(), oldest = harness.oracle().retained_begin();
        if (kind < 9) {
            Tokens fresh(1 + draw(12));
            for (auto& token : fresh) token = uint32_t(engine() % alphabet);
            static_cast<void>(harness.append(fresh));
        } else if (kind < 13 && length > oldest) {
            // Code-like repetition: copy a retained earlier span, sometimes with one edit.
            const uint64_t from = oldest + draw(length - oldest);
            const uint64_t count = 1 + draw(std::min<uint64_t>(96, length - from));
            Tokens copy(harness.oracle().tokens().begin() + ptrdiff_t(from),
                        harness.oracle().tokens().begin() + ptrdiff_t(from + count));
            if (engine() % 3 == 0) copy[draw(copy.size())] = uint32_t(engine() % alphabet);
            static_cast<void>(harness.append(copy));
        } else if (kind < 16) {
            static_cast<void>(harness.truncate(draw(length + 1)));
        } else if (kind < 18) {
            Tokens fresh(1 + draw(8));
            for (auto& token : fresh) token = uint32_t(engine() % alphabet);
            static_cast<void>(harness.replace(draw(length + 1), fresh));
        } else if (kind == 18) {
            static_cast<void>(harness.rebuild());
        } else if (engine() % 4 == 0) {
            static_cast<void>(harness.reset());
        } else {
            static_cast<void>(harness.check());
        }
    }
}

void capacity_boundaries() {
    // Full-size history with the default widths: exact fill, first eviction,
    // thousands of evictions, truncation above and below the window start.
    const LookupConfig config = configure(16, 7, maximum_history_tokens);
    Harness harness(config);
    std::mt19937_64 engine(0x6361706163697479ull);
    Tokens stream;
    while (stream.size() < 70000) {
        if (stream.size() > 200 && engine() % 3 == 0) {
            const uint64_t from = engine() % (stream.size() - 100);
            const uint64_t count = 20 + engine() % 60;
            for (uint64_t i = 0; i < count; ++i) {
                const uint32_t token = stream[from + i];
                stream.push_back(token);
            }
        } else {
            const uint64_t count = 1 + engine() % 40;
            for (uint64_t i = 0; i < count; ++i) stream.push_back(uint32_t(engine() % 500));
        }
    }
    const auto slice = [&](uint64_t from, uint64_t to) {
        return Tokens(stream.begin() + ptrdiff_t(from), stream.begin() + ptrdiff_t(to));
    };
    static_cast<void>(harness.append(slice(0, 65535)));
    require(harness.lookup().retained_tokens() == 65535 && harness.lookup().retained_begin() == 0,
            "window one below capacity");
    static_cast<void>(harness.append(slice(65535, 65536)));
    require(harness.lookup().retained_tokens() == 65536 && harness.lookup().retained_begin() == 0 &&
                harness.lookup().ledger().evicted_grams == 0,
            "exact capacity without eviction");
    static_cast<void>(harness.append(slice(65536, 65537)));
    require(harness.lookup().retained_tokens() == 65536 && harness.lookup().retained_begin() == 1 &&
                harness.lookup().ledger().evicted_grams == 1,
            "first eviction");
    static_cast<void>(harness.append(slice(65537, 70000)));
    require(harness.lookup().retained_begin() == 70000 - 65536 && harness.lookup().ledger().evicted_grams == 70000 - 65536,
            "window follows the committed length");
    static_cast<void>(harness.truncate(65000));
    require(harness.lookup().retained_begin() == 70000 - 65536 && harness.lookup().retained_tokens() == 65000 - (70000 - 65536),
            "truncation above the window start keeps it");
    static_cast<void>(harness.append(slice(65000, 66000)));
    static_cast<void>(harness.truncate(4000));
    require(harness.lookup().retained_tokens() == 0 && harness.lookup().retained_begin() == 4000 &&
                harness.audit() == 0,
            "truncation below the window start retains nothing");
    static_cast<void>(harness.append(slice(4000, 4100)));
    static_cast<void>(harness.rebuild());
    static_cast<void>(harness.reset());
    static_cast<void>(harness.append(slice(0, 5000)));
    require(harness.lookup().charged_bytes() == prompt_lookup_bytes(config), "capacity exercise allocated memory");
}
} // namespace

int main() {
    try {
        contract_basics();
        selection_rule();
        collisions();
        code_like_repetition();
        epochs_and_prefixes();
        invalid_configuration();
        memory_budget();
        work_limits();
        cross_thread();
        randomized(0x746f6b7370757200ull, configure(1, 1, 2), 2000);
        randomized(0x6c6f6f6b75700001ull, configure(2, 3, 5), 2000);
        randomized(0x6c6f6f6b75700002ull, configure(3, 7, 16), 2000);
        randomized(0x6c6f6f6b75700003ull, configure(4, 16, 64, 1), 1500);
        randomized(0x6c6f6f6b75700004ull, configure(8, 4, 97, 2), 1500);
        randomized(0x6c6f6f6b75700005ull, configure(16, 7, 48), 1500);
        randomized(0x6c6f6f6b75700006ull, configure(64, 16, 65), 1200);
        randomized(0x6c6f6f6b75700007ull, configure(16, 7, 300), 1200);
        capacity_boundaries();
        std::cout << "Owned prompt lookup checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned prompt lookup check failed: " << error.what() << '\n';
        return 1;
    }
}
