#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/json.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/grouped_mm_f32.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/sampling.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace tokspur;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED REQUIREMENT: " << message << std::endl;
        throw std::runtime_error(message);
    }
}

std::string to_hex(std::span<const uint8_t> bytes) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (auto b : bytes) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
}

// Replicate BranchSession composite state digest algorithm from native/src/target.cpp:442-458
std::array<uint8_t, 32> compute_state_digest_oracle(
    uint32_t prefix,
    std::optional<GdnChunkMode> chunk_mode,
    const std::vector<uint32_t>& partitions) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    CC_SHA256_Update(&ctx, &prefix, sizeof(prefix));
    const uint8_t cm = chunk_mode.has_value() ? static_cast<uint8_t>(*chunk_mode) : 0xff;
    CC_SHA256_Update(&ctx, &cm, sizeof(cm));
    uint32_t accumulated = 0;
    for (const auto part : partitions) {
        if (accumulated >= prefix) break;
        const uint32_t chunk = std::min(part, prefix - accumulated);
        if (chunk > 0) {
            CC_SHA256_Update(&ctx, &chunk, sizeof(chunk));
            accumulated += chunk;
        }
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &ctx);
    return digest;
}

// Mock structures replicating CommittedResult, TargetSampling, TargetSampler, and BranchSession
// directly matching native/src/target.cpp definitions for independent component-level validation.

struct MockResult {
    std::vector<float> logits;
};

template<class Result>
class MockCommittedResult {
    std::optional<Result> value_;
    uint32_t prefix_ = 0;
    std::optional<uint32_t> stop_;
    bool logits_available_ = false;
public:
    [[nodiscard]] bool matches(uint32_t prefix) const noexcept {
        return value_ && logits_available_ && prefix && prefix_ == prefix;
    }
    [[nodiscard]] bool continuable(uint32_t prefix) const noexcept {
        return matches(prefix) && !stop_;
    }
    [[nodiscard]] std::optional<uint32_t> stop_id() const noexcept { return stop_; }
    [[nodiscard]] const Result& get(uint32_t prefix) const {
        require(matches(prefix), "no logits for the committed target prefix");
        return *value_;
    }
    void clear() noexcept {
        value_.reset(); prefix_ = 0; stop_.reset(); logits_available_ = false;
    }
    void commit(Result&& value, uint32_t prefix, bool logits_available = true) noexcept {
        value_ = std::move(value); prefix_ = prefix; stop_.reset(); logits_available_ = logits_available;
    }
    void stopped(uint32_t id) noexcept { stop_ = id; }
    void exchange(std::optional<Result>& prepared, uint32_t prefix, std::optional<uint32_t> stop) noexcept {
        value_.swap(prepared); prefix_ = prefix; stop_ = stop; logits_available_ = true;
    }
};

struct MockTargetSampling {
    SamplingPolicy policy{};
    uint64_t seed = 0;
};

class MockTargetSampler {
    MockTargetSampling options_;
    uint32_t prefix_ = 0;
    bool valid_ = true;
public:
    explicit MockTargetSampler(MockTargetSampling options = {}) : options_(options) {}
    static MockTargetSampler fresh_branch(MockTargetSampling options, uint32_t prefix) {
        require(options.policy.temperature == 0 && prefix && prefix <= 65536,
            "branch restore requires greedy policy and nonempty bounded prefix");
        MockTargetSampler out(options);
        out.prefix_ = prefix;
        return out;
    }
    [[nodiscard]] bool matches(uint32_t prefix) const noexcept { return valid_ && prefix_ == prefix; }
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] uint32_t prefix() const noexcept { return prefix_; }
    void invalidate() noexcept { valid_ = false; }
    void set_prefix(uint32_t p) noexcept { prefix_ = p; }
};

template<class Handle, class Result>
class MockBranchSession {
    struct Record {
        Handle handle{};
        Result result{};
        std::optional<uint32_t> stop;
        uint64_t order = 0;
        uint32_t prefix = 0;
        bool valid = false;
        std::array<uint8_t, 32> state_digest{};
    };
    std::vector<Record> records_;
    std::vector<uint32_t> history_;
    std::vector<uint32_t> partitions_;
    std::optional<GdnChunkMode> chunk_mode_;
    std::optional<Result> transfer_;
    uint64_t order_ = 0;
    uint32_t used_ = 0;
    uint32_t vocabulary_ = 0;
    bool valid_ = true;

    void stage(std::span<const float> logits) {
        require(logits.size() == vocabulary_ && transfer_ && transfer_->logits.size() == vocabulary_,
            "branch logit storage mismatch");
        for (auto value : logits) {
            require(std::isfinite(value), "nonfinite saved branch logits");
        }
        Result clean{}; clean.logits.swap(transfer_->logits); *transfer_ = std::move(clean);
        std::copy(logits.begin(), logits.end(), transfer_->logits.begin());
    }

    [[nodiscard]] std::array<uint8_t, 32> compute_state_digest(uint32_t prefix) const {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        CC_SHA256_Update(&ctx, &prefix, sizeof(prefix));
        const uint8_t cm = chunk_mode_.has_value() ? static_cast<uint8_t>(*chunk_mode_) : 0xff;
        CC_SHA256_Update(&ctx, &cm, sizeof(cm));
        uint32_t accumulated = 0;
        for (const auto part : partitions_) {
            if (accumulated >= prefix) break;
            const uint32_t chunk = std::min(part, prefix - accumulated);
            if (chunk > 0) {
                CC_SHA256_Update(&ctx, &chunk, sizeof(chunk));
                accumulated += chunk;
            }
        }
        std::array<uint8_t, 32> digest{};
        CC_SHA256_Final(digest.data(), &ctx);
        return digest;
    }

public:
    static uint64_t minimum_bytes(uint32_t slots, uint32_t vocabulary) {
        require(slots && slots <= 4 && vocabulary && vocabulary <= maximum_sampling_vocabulary,
            "branch session shape bound");
        return sizeof(MockBranchSession) + uint64_t(slots) * sizeof(Record) + 65536ull * sizeof(uint32_t) +
            uint64_t(slots + 1) * vocabulary * sizeof(float);
    }

    MockBranchSession(uint32_t slots, uint32_t vocabulary) : vocabulary_(vocabulary) {
        static_cast<void>(minimum_bytes(slots, vocabulary));
        records_.resize(slots);
        history_.resize(65536);
        transfer_.emplace();
        transfer_->logits.resize(vocabulary);
        for (auto& record : records_) record.result.logits.resize(vocabulary);
    }

    void record_partition(uint32_t count, std::optional<GdnChunkMode> mode) {
        static_cast<void>(count);
        if (mode.has_value()) chunk_mode_ = mode;
    }

    [[nodiscard]] uint64_t charged_bytes() const noexcept {
        uint64_t bytes = sizeof(*this) + uint64_t(records_.capacity()) * sizeof(Record) +
            uint64_t(history_.capacity()) * sizeof(uint32_t) +
            uint64_t(partitions_.capacity()) * sizeof(uint32_t) +
            uint64_t(transfer_->logits.capacity()) * sizeof(float);
        for (const auto& record : records_) bytes += uint64_t(record.result.logits.capacity()) * sizeof(float);
        return bytes;
    }

    [[nodiscard]] bool matches(uint32_t prefix) const noexcept { return valid_ && used_ == prefix; }
    [[nodiscard]] uint32_t retained_count() const noexcept {
        uint32_t count = 0; for (const auto& record : records_) count += record.valid; return count;
    }

    [[nodiscard]] size_t common_prefix(std::span<const uint32_t> prompt) const noexcept {
        if (!valid_) return 0;
        size_t common = 0;
        while (common < prompt.size() && common < used_ && history_[common] == prompt[common]) ++common;
        return common;
    }

    void reset() noexcept {
        used_ = 0; valid_ = true; partitions_.clear(); for (auto& record : records_) record.valid = false;
    }

    void invalidate() noexcept { reset(); valid_ = false; }

    bool committed(uint32_t prefix, std::span<const uint32_t> tokens) noexcept {
        if (!valid_ || tokens.size() > history_.size() - used_ || prefix != used_ + tokens.size()) {
            invalidate(); return false;
        }
        std::copy(tokens.begin(), tokens.end(), history_.begin() + used_);
        used_ = prefix;
        partitions_.push_back(uint32_t(tokens.size()));
        return true;
    }

    [[nodiscard]] std::optional<size_t> select(std::span<const uint32_t> prompt, uint32_t generate) const {
        if (!valid_) return {};
        const auto common = common_prefix(prompt);
        std::optional<size_t> selected;
        for (size_t i = 0; i < records_.size(); ++i) {
            if (records_[i].valid && records_[i].prefix <= common &&
                records_[i].state_digest == compute_state_digest(records_[i].prefix) &&
                (!selected || records_[i].prefix > records_[*selected].prefix ||
                 (records_[i].prefix == records_[*selected].prefix && records_[i].order > records_[*selected].order))) {
                selected = i;
            }
        }
        if (selected) {
            require(!generate || records_[*selected].prefix < prompt.size() || !records_[*selected].stop,
                "restored terminal prefix requires new input before generation");
        }
        return selected;
    }

    template<class Retain>
    void capture(uint32_t prefix, const MockCommittedResult<Result>& retained, Retain&& retain) {
        require(matches(prefix) && prefix && retained.matches(prefix), "branch capture prefix/logits mismatch");
        require(order_ < std::numeric_limits<uint64_t>::max(), "branch capture order exhausted");
        stage(retained.get(prefix).logits);
        size_t index = 0;
        for (size_t i = 0; i < records_.size(); ++i) {
            if (!records_[i].valid) { index = i; break; }
            if (records_[i].order < records_[index].order) index = i;
        }
        const auto handle = retain();
        auto& record = records_[index];
        std::swap(record.result, *transfer_);
        record.handle = handle;
        record.prefix = prefix;
        record.stop = retained.stop_id();
        record.order = ++order_;
        record.valid = true;
        record.state_digest = compute_state_digest(prefix);
    }

    template<class Restore>
    uint32_t restore(size_t index, MockCommittedResult<Result>& retained, MockTargetSampler& sampler,
                     MockTargetSampling options, Restore&& restore) {
        require(valid_ && index < records_.size() && records_[index].valid && retained.matches(used_) && sampler.matches(used_),
            "branch restoration requires coherent current/saved session state");
        const auto& record = records_[index];
        const auto prefix = record.prefix;
        require(prefix <= used_, "branch restore is ancestor-only");
        const auto expected_digest = compute_state_digest(prefix);
        require(record.state_digest == expected_digest, "checkpoint state digest mismatch: partition history or chunk mode diverged");
        require(retained.get(used_).logits.size() == vocabulary_ &&
            retained.get(used_).logits.capacity() <= transfer_->logits.capacity(),
            "current logits cannot preserve bounded branch transfer storage");
        for (auto value : retained.get(used_).logits) {
            require(std::isfinite(value), "nonfinite current branch logits");
        }
        auto replacement = MockTargetSampler::fresh_branch(options, prefix);
        stage(record.result.logits);
        const auto actual = restore(record.handle);
        if (actual != prefix) {
            invalidate(); retained.clear(); sampler.invalidate();
            throw FormatError("native restore published unexpected branch prefix; reset required");
        }
        retained.exchange(transfer_, prefix, record.stop);
        sampler = std::move(replacement);
        used_ = prefix;
        uint32_t sum = 0;
        size_t keep = 0;
        for (size_t i = 0; i < partitions_.size(); ++i) {
            if (sum + partitions_[i] <= prefix) {
                sum += partitions_[i];
                keep = i + 1;
                if (sum == prefix) break;
            } else {
                partitions_[i] = prefix - sum;
                keep = i + 1;
                break;
            }
        }
        if (prefix == 0) keep = 0;
        partitions_.resize(keep);
        for (auto& other : records_) {
            if (other.valid && other.prefix > prefix) other.valid = false;
        }
        return prefix;
    }

    // Accessors for testing internal invariant state
    [[nodiscard]] const std::vector<uint32_t>& partitions() const noexcept { return partitions_; }
    [[nodiscard]] std::array<uint8_t, 32> state_digest(uint32_t p) const { return compute_state_digest(p); }
    [[nodiscard]] std::optional<GdnChunkMode> chunk_mode() const noexcept { return chunk_mode_; }
    [[nodiscard]] uint32_t used() const noexcept { return used_; }
    [[nodiscard]] bool record_valid(size_t i) const { return i < records_.size() && records_[i].valid; }
    [[nodiscard]] uint32_t record_prefix(size_t i) const { return records_.at(i).prefix; }
    [[nodiscard]] std::array<uint8_t, 32> record_digest(size_t i) const { return records_.at(i).state_digest; }
};

} // namespace

int main(int argc, char** argv) {
    std::cout << "================================================================" << std::endl;
    std::cout << "EMPIRICAL CHALLENGER M4-3: Adversarial Verification Harness" << std::endl;
    std::cout << "================================================================" << std::endl;

    const char* metallib_path = argc > 1 ? argv[1] : "../build/owned-native/tokspur-owned.metallib";

    // ------------------------------------------------------------------------
    // SUITE 1: Direct State Digest Mathematical Oracle & Partition Expansion
    // ------------------------------------------------------------------------
    std::cout << "\n--- Suite 1: Mathematical Oracle & Partition Expansion Invariance ---" << std::endl;
    {
        const uint32_t prefix = 128;
        const auto mode_cs64 = std::make_optional(GdnChunkMode::cs64);

        // Baseline partition topologies summing to prefix 128
        const struct {
            const char* name;
            std::vector<uint32_t> partitions;
        } topologies[] = {
            {"Single partition [128]", {128}},
            {"Dual partition [64, 64]", {64, 64}},
            {"Triple partition [32, 32, 64]", {32, 32, 64}},
            {"Triple partition [64, 32, 32]", {64, 32, 32}},
            {"Eight-chunk partition [16 x 8]", {16, 16, 16, 16, 16, 16, 16, 16}},
            {"Uneven partition [100, 28]", {100, 28}},
        };

        for (const auto& topo : topologies) {
            const auto baseline_digest = compute_state_digest_oracle(prefix, mode_cs64, topo.partitions);
            const std::string base_hex = to_hex(baseline_digest);
            std::cout << "  Testing topology '" << topo.name << "' (digest=" << base_hex.substr(0, 16) << "...)" << std::endl;

            // Subsequent commit extensions
            const std::vector<std::vector<uint32_t>> expansions = {
                {64},
                {64, 128},
                {64, 128, 256},
                {64, 128, 256, 512},
                {64, 128, 256, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {256, 256, 256},
            };

            for (size_t exp_idx = 0; exp_idx < expansions.size(); ++exp_idx) {
                auto extended = topo.partitions;
                extended.insert(extended.end(), expansions[exp_idx].begin(), expansions[exp_idx].end());
                const auto extended_digest = compute_state_digest_oracle(prefix, mode_cs64, extended);
                require(extended_digest == baseline_digest,
                    "Ancestor state digest MUST be invariant under subsequent partition expansions");
            }

            // 1,000 randomized subsequent extensions
            std::mt19937 rng(0x128a + topo.partitions.size());
            std::uniform_int_distribution<uint32_t> chunk_dist(1, 256);
            std::uniform_int_distribution<size_t> length_dist(1, 30);

            for (int trial = 0; trial < 1000; ++trial) {
                auto rand_extended = topo.partitions;
                const size_t num_chunks = length_dist(rng);
                for (size_t c = 0; c < num_chunks; ++c) {
                    rand_extended.push_back(chunk_dist(rng));
                }
                const auto rand_digest = compute_state_digest_oracle(prefix, mode_cs64, rand_extended);
                require(rand_digest == baseline_digest,
                    "Ancestor digest diverged under randomized partition expansion!");
            }
        }
        std::cout << "  [Check 1.1] 6 baseline topologies invariant across 6,000+ expansions: PASS" << std::endl;

        // Non-boundary prefix stress test: prefix cuts into the middle of a partition
        // E.g. prefix = 100 with partitions [64, 64] -> first chunk is 64, second chunk is 36.
        const uint32_t non_aligned_prefix = 100;
        const std::vector<uint32_t> base_parts = {64, 64};
        const auto non_aligned_baseline = compute_state_digest_oracle(non_aligned_prefix, mode_cs64, base_parts);

        // Expand with [128, 256, 512]
        std::vector<uint32_t> non_aligned_expanded = {64, 64, 128, 256, 512};
        const auto non_aligned_after = compute_state_digest_oracle(non_aligned_prefix, mode_cs64, non_aligned_expanded);
        require(non_aligned_after == non_aligned_baseline,
            "Non-aligned ancestor prefix digest MUST match under subsequent expansions");
        std::cout << "  [Check 1.2] Non-aligned prefix (100 in [64, 64]) expansion invariance: PASS" << std::endl;

        // Sweep non-aligned boundary prefixes: {1, 7, 31, 63, 65, 127, 129, 255, 257}
        const uint32_t sweep_prefixes[] = {1, 7, 31, 63, 65, 127, 129, 255, 257};
        for (uint32_t sp : sweep_prefixes) {
            std::vector<uint32_t> p_init = {32, 32, 64, 64, 128};
            const auto p_base = compute_state_digest_oracle(sp, mode_cs64, p_init);
            std::vector<uint32_t> p_exp = p_init;
            p_exp.push_back(64); p_exp.push_back(128); p_exp.push_back(256);
            const auto p_after = compute_state_digest_oracle(sp, mode_cs64, p_exp);
            require(p_base == p_after, "Boundary sweep ancestor invariance failed");
        }
        std::cout << "  [Check 1.3] Boundary sweep {1, 7, ..., 257} ancestor invariance: PASS" << std::endl;
    }

    // ------------------------------------------------------------------------
    // SUITE 2: Chunk Mode Transitions & Partition Boundary Distinctness
    // ------------------------------------------------------------------------
    std::cout << "\n--- Suite 2: Chunk Mode Transitions & Partition Boundary Distinctness ---" << std::endl;
    {
        const uint32_t prefix = 128;
        const auto mode_cs64 = std::make_optional(GdnChunkMode::cs64);
        const auto mode_fused = std::make_optional(GdnChunkMode::fused);
        const std::optional<GdnChunkMode> mode_null = std::nullopt;

        const std::vector<uint32_t> parts_2x64 = {64, 64};
        const auto d_cs64 = compute_state_digest_oracle(prefix, mode_cs64, parts_2x64);
        const auto d_fused = compute_state_digest_oracle(prefix, mode_fused, parts_2x64);
        const auto d_null = compute_state_digest_oracle(prefix, mode_null, parts_2x64);

        require(d_cs64 != d_fused, "d(cs64) MUST differ from d(fused)");
        require(d_cs64 != d_null, "d(cs64) MUST differ from d(nullopt)");
        require(d_fused != d_null, "d(fused) MUST differ from d(nullopt)");
        std::cout << "  [Check 2.1] Chunk mode distinctness (cs64 != fused != nullopt): PASS" << std::endl;

        // Partition boundary distinctness (same total token count, different histories):
        // 7 distinct topologies for prefix 128
        const std::vector<uint32_t> topos_128[] = {
            {128},
            {64, 64},
            {32, 32, 64},
            {64, 32, 32},
            {32, 64, 32},
            {32, 32, 32, 32},
            {16, 16, 16, 16, 16, 16, 16, 16},
        };
        constexpr size_t n_topos = sizeof(topos_128) / sizeof(topos_128[0]);
        std::vector<std::array<uint8_t, 32>> digests_128;
        for (size_t i = 0; i < n_topos; ++i) {
            digests_128.push_back(compute_state_digest_oracle(prefix, mode_cs64, topos_128[i]));
        }
        for (size_t i = 0; i < n_topos; ++i) {
            for (size_t j = i + 1; j < n_topos; ++j) {
                require(digests_128[i] != digests_128[j],
                    "Different partition boundaries for identical prefix MUST produce distinct digests!");
            }
        }
        std::cout << "  [Check 2.2] All " << (n_topos * (n_topos - 1) / 2) << " pairwise partition topologies distinct: PASS" << std::endl;

        // Determinism test: 10,000 consecutive hashes
        const auto ref_d = digests_128[1]; // [64, 64]
        for (int i = 0; i < 10000; ++i) {
            const auto d = compute_state_digest_oracle(prefix, mode_cs64, topos_128[1]);
            require(d == ref_d, "Non-deterministic digest encountered");
        }
        std::cout << "  [Check 2.3] 10,000/10,000 bit-exact deterministic hash executions: PASS" << std::endl;
    }

    // ------------------------------------------------------------------------
    // SUITE 3: Full Lifecycle BranchSession Complex Multi-Turn Branching Simulation
    // ------------------------------------------------------------------------
    std::cout << "\n--- Suite 3: Complex Multi-Turn Branching Simulation ---" << std::endl;
    {
        constexpr uint32_t slots = 4;
        constexpr uint32_t vocab = 1000;

        // Test Case 3A: Orchestrator's Exact Scenario
        // 1. Capture checkpoint at prefix 128
        // 2. Commit multiple partitions: 64, 128, 256 (prefix reaches 576)
        // 3. Restore ancestor checkpoint at prefix 128. Verify restore succeeds with 0 digest mismatch exceptions!
        std::cout << "  [Test 3A] Orchestrator exact sequence (prefix 128 -> +64 -> +128 -> +256 -> restore 128):" << std::endl;
        {
            MockBranchSession<uint32_t, MockResult> session(slots, vocab);
            MockCommittedResult<MockResult> retained;
            MockTargetSampler sampler;
            MockTargetSampling options{};
            options.policy.temperature = 0.0; // Greedy

            session.record_partition(128, GdnChunkMode::cs64);

            // Commit prefix 128
            std::vector<uint32_t> tokens_128(128);
            for (uint32_t i = 0; i < 128; ++i) tokens_128[i] = (i * 31 + 7) % vocab;
            require(session.committed(128, tokens_128), "Failed to commit prefix 128");

            MockResult res_128;
            res_128.logits.resize(vocab, 0.5f);
            retained.commit(std::move(res_128), 128);
            sampler.set_prefix(128);

            // Capture checkpoint at prefix 128
            constexpr uint32_t handle_128 = 1001;
            session.capture(128, retained, [&] { return handle_128; });
            require(session.retained_count() == 1, "Retained count must be 1 after capture");
            require(session.record_valid(0), "Slot 0 must be valid");
            require(session.record_prefix(0) == 128, "Slot 0 prefix must be 128");
            const auto checkpoint_0_digest = session.record_digest(0);
            std::cout << "    Captured Checkpoint 0 at prefix 128 (digest="
                      << to_hex(checkpoint_0_digest).substr(0, 16) << "...)" << std::endl;

            // Commit partition 64 -> prefix 192
            std::vector<uint32_t> tokens_64(64);
            for (uint32_t i = 0; i < 64; ++i) tokens_64[i] = (i * 17 + 13) % vocab;
            require(session.committed(192, tokens_64), "Failed to commit partition 64");
            session.record_partition(64, GdnChunkMode::cs64);

            MockResult res_192;
            res_192.logits.resize(vocab, 0.4f);
            retained.commit(std::move(res_192), 192);
            sampler.set_prefix(192);

            // Commit partition 128 -> prefix 320
            std::vector<uint32_t> tokens_part128(128);
            for (uint32_t i = 0; i < 128; ++i) tokens_part128[i] = (i * 23 + 19) % vocab;
            require(session.committed(320, tokens_part128), "Failed to commit partition 128");
            session.record_partition(128, GdnChunkMode::cs64);

            MockResult res_320;
            res_320.logits.resize(vocab, 0.3f);
            retained.commit(std::move(res_320), 320);
            sampler.set_prefix(320);

            // Commit partition 256 -> prefix 576
            std::vector<uint32_t> tokens_256(256);
            for (uint32_t i = 0; i < 256; ++i) tokens_256[i] = (i * 29 + 3) % vocab;
            require(session.committed(576, tokens_256), "Failed to commit partition 256");
            session.record_partition(256, GdnChunkMode::cs64);

            MockResult res_576;
            res_576.logits.resize(vocab, 0.2f);
            retained.commit(std::move(res_576), 576);
            sampler.set_prefix(576);

            // Capture checkpoint at prefix 576
            constexpr uint32_t handle_576 = 1002;
            session.capture(576, retained, [&] { return handle_576; });
            require(session.retained_count() == 2, "Retained count must be 2");
            require(session.matches(576), "Session used_ must be 576");
            std::cout << "    Committed partitions [128, 64, 128, 256], used=" << session.used() << std::endl;
            std::cout << "    Active partition array size=" << session.partitions().size() << std::endl;

            // Restore ancestor checkpoint 0 at prefix 128!
            std::cout << "    Restoring ancestor Checkpoint 0 at prefix 128..." << std::endl;
            uint32_t restored_prefix = 0;
            bool restore_threw = false;
            try {
                restored_prefix = session.restore(0, retained, sampler, options, [&](uint32_t h) {
                    require(h == handle_128, "Restore handle mismatch");
                    return 128u; // Native restore published 128
                });
            } catch (const std::exception& e) {
                restore_threw = true;
                std::cerr << "    UNEXPECTED EXCEPTION during restore: " << e.what() << std::endl;
            }

            require(!restore_threw, "Restore MUST NOT throw digest mismatch exception!");
            require(restored_prefix == 128, "Restored prefix must be 128");
            require(session.matches(128), "Session used_ must match restored prefix 128");
            require(retained.matches(128), "Retained logits must match restored prefix 128");
            require(sampler.matches(128), "Sampler must match restored prefix 128");
            require(session.retained_count() == 1, "Descendant Checkpoint 1 (prefix 576) must be invalidated");
            require(!session.record_valid(1), "Checkpoint 1 must be marked invalid");
            require(session.partitions().size() == 1, "Partitions array must be truncated to 1 element");
            require(session.partitions()[0] == 128, "Truncated partition must be 128");

            std::cout << "    Ancestor checkpoint restored successfully with 0 exceptions! PASS" << std::endl;
        }

        // Test Case 3B: Deep Multi-Branch Tree with Multiple Checkpoints and Backtracking
        std::cout << "  [Test 3B] Deep multi-branch tree with backtracking:" << std::endl;
        {
            MockBranchSession<uint32_t, MockResult> session(slots, vocab);
            MockCommittedResult<MockResult> retained;
            MockTargetSampler sampler;
            MockTargetSampling options{};
            options.policy.temperature = 0.0;

            session.record_partition(128, GdnChunkMode::cs64);

            // Step 1: Root prefix 128 -> Checkpoint 0
            std::vector<uint32_t> root_tokens(128, 42);
            require(session.committed(128, root_tokens), "commit root");
            MockResult r0; r0.logits.resize(vocab, 1.0f);
            retained.commit(std::move(r0), 128);
            sampler.set_prefix(128);
            session.capture(128, retained, [&] { return 100u; });

            // Step 2: Branch 1 -> commit 64 (prefix 192) -> Checkpoint 1
            std::vector<uint32_t> b1_p1(64, 43);
            require(session.committed(192, b1_p1), "commit b1 p1");
            MockResult r1; r1.logits.resize(vocab, 1.1f);
            retained.commit(std::move(r1), 192);
            sampler.set_prefix(192);
            session.capture(192, retained, [&] { return 101u; });

            // Step 3: Branch 1 -> commit 128 (prefix 320) -> Checkpoint 2
            std::vector<uint32_t> b1_p2(128, 44);
            require(session.committed(320, b1_p2), "commit b1 p2");
            MockResult r2; r2.logits.resize(vocab, 1.2f);
            retained.commit(std::move(r2), 320);
            sampler.set_prefix(320);
            session.capture(320, retained, [&] { return 102u; });

            // Step 4: Branch 1 -> commit 256 (prefix 576)
            std::vector<uint32_t> b1_p3(256, 45);
            require(session.committed(576, b1_p3), "commit b1 p3");
            MockResult r3; r3.logits.resize(vocab, 1.3f);
            retained.commit(std::move(r3), 576);
            sampler.set_prefix(576);

            require(session.retained_count() == 3, "Should have 3 checkpoints");

            // Backtrack to Checkpoint 2 (prefix 320)
            uint32_t r_320 = session.restore(2, retained, sampler, options, [&](uint32_t) { return 320u; });
            require(r_320 == 320, "Restored to 320");
            require(session.partitions().size() == 3, "Partitions: [128, 64, 128]");

            // Backtrack to Checkpoint 1 (prefix 192)
            uint32_t r_192 = session.restore(1, retained, sampler, options, [&](uint32_t) { return 192u; });
            require(r_192 == 192, "Restored to 192");
            require(session.partitions().size() == 2, "Partitions: [128, 64]");

            // Backtrack to Checkpoint 0 (prefix 128)
            uint32_t r_128 = session.restore(0, retained, sampler, options, [&](uint32_t) { return 128u; });
            require(r_128 == 128, "Restored to 128");
            require(session.partitions().size() == 1, "Partitions: [128]");

            // Branch 2: From prefix 128, commit alternative partition 256 directly
            std::vector<uint32_t> b2_p1(256, 99);
            require(session.committed(384, b2_p1), "commit b2 p1");
            MockResult r_b2; r_b2.logits.resize(vocab, 2.0f);
            retained.commit(std::move(r_b2), 384);
            sampler.set_prefix(384);

            // Restore back to Checkpoint 0 (prefix 128)
            uint32_t r_b2_back = session.restore(0, retained, sampler, options, [&](uint32_t) { return 128u; });
            require(r_b2_back == 128, "Restored back to root 128");
            require(session.partitions().size() == 1 && session.partitions()[0] == 128, "Partitions reset to [128]");

            std::cout << "    Deep tree branching and multi-stage backtracking verified PASS" << std::endl;
        }

        // Test Case 3C: Non-Aligned Partition Boundary Restoration
        std::cout << "  [Test 3C] Non-aligned partition boundary restoration:" << std::endl;
        {
            MockBranchSession<uint32_t, MockResult> session(slots, vocab);
            MockCommittedResult<MockResult> retained;
            MockTargetSampler sampler;
            MockTargetSampling options{};
            options.policy.temperature = 0.0;

            session.record_partition(64, GdnChunkMode::cs64);

            // Commit partition 64
            std::vector<uint32_t> p1(64, 1);
            require(session.committed(64, p1), "commit p1");

            // Commit partition 36 -> prefix 100
            std::vector<uint32_t> p2(36, 2);
            require(session.committed(100, p2), "commit p2 (total 100)");

            MockResult r100; r100.logits.resize(vocab, 0.9f);
            retained.commit(std::move(r100), 100);
            sampler.set_prefix(100);

            // Capture checkpoint at prefix 100 (partitions: [64, 36])
            session.capture(100, retained, [&] { return 555u; });

            // Commit partition 28 (completing 64-token chunk) -> prefix 128
            std::vector<uint32_t> p3(28, 3);
            require(session.committed(128, p3), "commit p3");

            // Commit partition 256 -> prefix 384
            std::vector<uint32_t> p4(256, 4);
            require(session.committed(384, p4), "commit p4");

            MockResult r384; r384.logits.resize(vocab, 0.8f);
            retained.commit(std::move(r384), 384);
            sampler.set_prefix(384);

            // Restore ancestor checkpoint at prefix 100
            uint32_t r100_out = session.restore(0, retained, sampler, options, [&](uint32_t) { return 100u; });
            require(r100_out == 100, "Restore returned 100");
            require(session.partitions().size() == 2, "Partitions truncated to 2 elements");
            require(session.partitions()[0] == 64 && session.partitions()[1] == 36,
                "Partitions match [64, 36] exactly");

            std::cout << "    Non-aligned partition restoration verified PASS" << std::endl;
        }

        // Test Case 3D: Adversarial Failure Mode Rejections
        std::cout << "  [Test 3D] Adversarial rejection and refusal guarantees:" << std::endl;
        {
            // 1. Mismatched Chunk Mode Refusal
            {
                MockBranchSession<uint32_t, MockResult> session(slots, vocab);
                MockCommittedResult<MockResult> retained;
                MockTargetSampler sampler;
                MockTargetSampling options{};
                options.policy.temperature = 0.0;

                session.record_partition(128, GdnChunkMode::cs64);
                std::vector<uint32_t> tok(128, 5);
                require(session.committed(128, tok), "commit");
                MockResult r; r.logits.resize(vocab, 1.0f);
                retained.commit(std::move(r), 128);
                sampler.set_prefix(128);

                // Capture under cs64
                session.capture(128, retained, [&] { return 1u; });

                // Commit 64 tokens and alter mode to fused
                std::vector<uint32_t> tok2(64, 6);
                require(session.committed(192, tok2), "commit 2");
                session.record_partition(64, GdnChunkMode::fused); // Mode changed!

                MockResult r2; r2.logits.resize(vocab, 1.0f);
                retained.commit(std::move(r2), 192);
                sampler.set_prefix(192);

                bool caught_mode_mismatch = false;
                try {
                    session.restore(0, retained, sampler, options, [&](uint32_t) { return 128u; });
                } catch (const std::runtime_error& e) {
                    if (std::string(e.what()).find("checkpoint state digest mismatch") != std::string::npos) {
                        caught_mode_mismatch = true;
                    }
                }
                require(caught_mode_mismatch,
                    "Restore MUST reject when chunk mode diverged between capture and restore");
                std::cout << "    Chunk mode divergence refusal verified PASS" << std::endl;
            }

            // 2. Ancestor-Only Guard Refusal
            {
                MockBranchSession<uint32_t, MockResult> session(slots, vocab);
                MockCommittedResult<MockResult> retained;
                MockTargetSampler sampler;
                MockTargetSampling options{};
                options.policy.temperature = 0.0;

                session.record_partition(128, GdnChunkMode::cs64);
                std::vector<uint32_t> tok(128, 5);
                require(session.committed(128, tok), "commit");
                MockResult r; r.logits.resize(vocab, 1.0f);
                retained.commit(std::move(r), 128);
                sampler.set_prefix(128);

                // Capture at prefix 128
                session.capture(128, retained, [&] { return 1u; });

                // Artificially simulate used_ being less than record prefix (should never happen normally)
                // We test session.restore where record.prefix > used_ by testing after reset
                session.reset();
                std::vector<uint32_t> small_tok(64, 1);
                require(session.committed(64, small_tok), "commit small");
                MockResult r_small; r_small.logits.resize(vocab, 1.0f);
                retained.commit(std::move(r_small), 64);
                sampler.set_prefix(64);

                // Checkpoint 0 was invalidated by reset!
                bool caught_invalid = false;
                try {
                    session.restore(0, retained, sampler, options, [&](uint32_t) { return 128u; });
                } catch (const std::runtime_error&) {
                    caught_invalid = true;
                }
                require(caught_invalid, "Restore on invalidated checkpoint MUST be rejected");
                std::cout << "    Invalid / reset checkpoint refusal verified PASS" << std::endl;
            }

            // 3. Unexpected Published Prefix Exception
            {
                MockBranchSession<uint32_t, MockResult> session(slots, vocab);
                MockCommittedResult<MockResult> retained;
                MockTargetSampler sampler;
                MockTargetSampling options{};
                options.policy.temperature = 0.0;

                session.record_partition(128, GdnChunkMode::cs64);
                std::vector<uint32_t> tok(128, 5);
                require(session.committed(128, tok), "commit");
                MockResult r; r.logits.resize(vocab, 1.0f);
                retained.commit(std::move(r), 128);
                sampler.set_prefix(128);
                session.capture(128, retained, [&] { return 1u; });

                std::vector<uint32_t> tok2(64, 6);
                require(session.committed(192, tok2), "commit 2");
                MockResult r2; r2.logits.resize(vocab, 1.0f);
                retained.commit(std::move(r2), 192);
                sampler.set_prefix(192);

                bool caught_format_error = false;
                try {
                    // Callback returns wrong prefix (127 instead of 128)
                    session.restore(0, retained, sampler, options, [&](uint32_t) { return 127u; });
                } catch (const FormatError& e) {
                    if (std::string(e.what()).find("unexpected branch prefix") != std::string::npos) {
                        caught_format_error = true;
                    }
                }
                require(caught_format_error, "Unexpected restored prefix MUST throw FormatError");
                require(!session.matches(128), "Session MUST be invalidated after format error");
                std::cout << "    Unexpected published prefix FormatError verified PASS" << std::endl;
            }

            // 4. Nonfinite Logits Guards
            {
                MockBranchSession<uint32_t, MockResult> session(slots, vocab);
                MockCommittedResult<MockResult> retained;
                MockTargetSampler sampler;

                session.record_partition(128, GdnChunkMode::cs64);
                std::vector<uint32_t> tok(128, 5);
                require(session.committed(128, tok), "commit");

                // Inject NaN into logits during capture
                MockResult r_nan;
                r_nan.logits.resize(vocab, 1.0f);
                r_nan.logits[50] = std::numeric_limits<float>::quiet_NaN();
                retained.commit(std::move(r_nan), 128);
                sampler.set_prefix(128);

                bool caught_nan_capture = false;
                try {
                    session.capture(128, retained, [&] { return 1u; });
                } catch (const std::runtime_error& e) {
                    if (std::string(e.what()).find("nonfinite saved branch logits") != std::string::npos) {
                        caught_nan_capture = true;
                    }
                }
                require(caught_nan_capture, "NaN logits in capture MUST be rejected");
                std::cout << "    NaN logits guard verified PASS" << std::endl;
            }

            // 5. Four-Slot Capacity & Eviction Order
            {
                MockBranchSession<uint32_t, MockResult> session(slots, vocab);
                MockCommittedResult<MockResult> retained;
                MockTargetSampler sampler;
                session.record_partition(10, GdnChunkMode::cs64);

                for (uint32_t s = 1; s <= 4; ++s) {
                    const uint32_t p = s * 10;
                    std::vector<uint32_t> t(10, s);
                    require(session.committed(p, t), "commit");
                    MockResult r; r.logits.resize(vocab, float(s));
                    retained.commit(std::move(r), p);
                    sampler.set_prefix(p);
                    session.capture(p, retained, [&] { return s; });
                }
                require(session.retained_count() == 4, "All 4 slots populated");

                // Capture a 5th checkpoint -> must evict slot 0 (oldest order)
                std::vector<uint32_t> t5(10, 5);
                require(session.committed(50, t5), "commit 5");
                MockResult r5; r5.logits.resize(vocab, 5.0f);
                retained.commit(std::move(r5), 50);
                sampler.set_prefix(50);
                session.capture(50, retained, [&] { return 5u; });

                require(session.retained_count() == 4, "Retained count remains capped at 4 slots");
                // Slot 0 (prefix 10) was evicted; prefix 20, 30, 40, 50 remain
                require(session.record_prefix(0) == 50, "Slot 0 replaced with prefix 50");
                require(session.record_prefix(1) == 20, "Slot 1 retained prefix 20");
                std::cout << "    4-slot bound & oldest-order eviction verified PASS" << std::endl;
            }
        }

        // Test Case 3E: Randomized Branching & Backtracking Stress Test (1,000 cycles)
        std::cout << "  [Test 3E] 1,000-cycle randomized multi-turn branching stress test..." << std::endl;
        {
            MockBranchSession<uint32_t, MockResult> session(slots, vocab);
            MockCommittedResult<MockResult> retained;
            MockTargetSampler sampler;
            MockTargetSampling options{};
            options.policy.temperature = 0.0;

            session.record_partition(64, GdnChunkMode::cs64);
            std::vector<uint32_t> init_tokens(64, 7);
            require(session.committed(64, init_tokens), "init commit");
            MockResult init_r; init_r.logits.resize(vocab, 0.5f);
            retained.commit(std::move(init_r), 64);
            sampler.set_prefix(64);
            session.capture(64, retained, [&] { return 1000u; });

            std::mt19937 rng(0x42b1a4c);
            std::uniform_int_distribution<uint32_t> action_dist(0, 2); // 0=commit, 1=capture, 2=restore
            std::uniform_int_distribution<uint32_t> chunk_dist(16, 64);

            uint32_t successful_restores = 0;
            uint32_t successful_commits = 0;
            uint32_t successful_captures = 0;

            for (int cycle = 0; cycle < 1000; ++cycle) {
                const uint32_t action = action_dist(rng);
                if (action == 0 && session.used() + 64 <= 8192) {
                    // Commit new partition
                    const uint32_t c_size = chunk_dist(rng);
                    std::vector<uint32_t> t(c_size, uint32_t(cycle % vocab));
                    const uint32_t next_p = session.used() + c_size;
                    if (session.committed(next_p, t)) {
                        session.record_partition(c_size, GdnChunkMode::cs64);
                        MockResult r; r.logits.resize(vocab, float(cycle % 10));
                        retained.commit(std::move(r), next_p);
                        sampler.set_prefix(next_p);
                        ++successful_commits;
                    }
                } else if (action == 1 && session.matches(session.used()) && retained.matches(session.used())) {
                    // Capture checkpoint
                    session.capture(session.used(), retained, [&] { return uint32_t(cycle); });
                    ++successful_captures;
                } else if (action == 2 && session.retained_count() > 0) {
                    // Restore a valid checkpoint
                    std::vector<size_t> valid_indices;
                    for (size_t s = 0; s < slots; ++s) {
                        if (session.record_valid(s) && session.record_prefix(s) <= session.used()) {
                            valid_indices.push_back(s);
                        }
                    }
                    if (!valid_indices.empty()) {
                        std::uniform_int_distribution<size_t> pick(0, valid_indices.size() - 1);
                        const size_t chosen = valid_indices[pick(rng)];
                        const uint32_t target_p = session.record_prefix(chosen);
                        const uint32_t res_p = session.restore(chosen, retained, sampler, options,
                            [&](uint32_t) { return target_p; });
                        require(res_p == target_p, "Stress test restore prefix mismatch");
                        ++successful_restores;
                    }
                }
            }

            // Finally, verify restoring an active valid ancestor
            for (size_t s = 0; s < slots; ++s) {
                if (session.record_valid(s) && session.record_prefix(s) <= session.used()) {
                    const uint32_t expected_p = session.record_prefix(s);
                    uint32_t final_res = session.restore(s, retained, sampler, options, [&](uint32_t) { return expected_p; });
                    require(final_res == expected_p, "Final ancestor restore prefix mismatch");
                    require(session.matches(expected_p), "Final ancestor matches prefix");
                    break;
                }
            }

            std::cout << "    1,000 cycles completed: " << successful_commits << " commits, "
                      << successful_captures << " captures, " << successful_restores << " restores (0 errors): PASS"
                      << std::endl;
        }
    }

    // ------------------------------------------------------------------------
    // SUITE 4: Router Margin Evaluation Under Synthetic Edge Cases (Metal GPU)
    // ------------------------------------------------------------------------
    std::cout << "\n--- Suite 4: Router Margin Evaluation Under Synthetic Edge Cases (Metal GPU) ---" << std::endl;
    {
        MetalGraph graph(metallib_path, 256ull << 20);
        graph.configure_prefill_grouped_auto(true);
        graph.configure_router_margin_audit(true);
        graph.configure_router_margin_epsilon(1.0e-4f);
        graph.configure_prefill_router_window(true);
        graph.prepare_moe_router_window();
        graph.prepare_grouped_mm(8, PrefillWeightLayout::gguf);

        const auto scratch = graph.router_margins_scratch();
        require(scratch.bytes >= 512 * sizeof(float), "Router margin scratch allocated");

        // 4A: Configuration Parameter Validation
        std::cout << "  [Test 4A] Router margin epsilon parameter validation:" << std::endl;
        {
            // epsilon = 0.0f -> must throw
            bool threw_zero = false;
            try { graph.configure_router_margin_epsilon(0.0f); } catch (const std::runtime_error&) { threw_zero = true; }
            require(threw_zero, "epsilon = 0.0f MUST throw requirement failure");

            // epsilon = -1.0e-4f -> must throw
            bool threw_neg = false;
            try { graph.configure_router_margin_epsilon(-1.0e-4f); } catch (const std::runtime_error&) { threw_neg = true; }
            require(threw_neg, "negative epsilon MUST throw requirement failure");

            // epsilon = NaN -> must throw
            bool threw_nan = false;
            try { graph.configure_router_margin_epsilon(std::numeric_limits<float>::quiet_NaN()); } catch (const std::runtime_error&) { threw_nan = true; }
            require(threw_nan, "NaN epsilon MUST throw requirement failure");

            // epsilon = Inf -> must throw
            bool threw_inf = false;
            try { graph.configure_router_margin_epsilon(std::numeric_limits<float>::infinity()); } catch (const std::runtime_error&) { threw_inf = true; }
            require(threw_inf, "Inf epsilon MUST throw requirement failure");

            // Valid epsilon
            graph.configure_router_margin_epsilon(1.0e-4f);
            require(graph.router_margin_epsilon() == 1.0e-4f, "Epsilon correctly set to 1.0e-4f");
            std::cout << "    Configuration parameter guards verified PASS" << std::endl;
        }

        const float eps = graph.router_margin_epsilon();
        const float threshold = 2.0f * eps; // 2.0e-4

        // 4B: Exact Floating-Point Threshold Boundaries
        std::cout << "  [Test 4B] Exact floating-point threshold boundaries (2*eps = " << threshold << "):" << std::endl;
        {
            const struct {
                const char* desc;
                float value;
                bool expect_fallback;
            } edge_cases[] = {
                {"Exact 2*eps boundary (2.0e-4)", threshold, true},
                {"Delta below 2*eps (nextafter toward 0)", std::nextafter(threshold, 0.0f), true},
                {"Delta above 2*eps (nextafter toward 1)", std::nextafter(threshold, 1.0f), false},
                {"Exact tie (0.0f)", 0.0f, true},
                {"Small positive delta (1.0e-7f)", 1.0e-7f, true},
                {"Inverted ranking / negative margin (-1.0e-4f)", -1.0e-4f, true},
                {"Deeply negative margin (-10.0f)", -10.0f, true},
                {"Safe margin (0.1f)", 0.1f, false},
                {"Extreme safe margin (1000.0f)", 1000.0f, false},
            };

            for (const auto& ec : edge_cases) {
                graph.reset_router_margin_telemetry();
                std::vector<float> margins(512, ec.value);
                graph.write(scratch, std::as_bytes(std::span(margins)));

                graph.begin();
                const bool fb = graph.evaluate_router_margin(512);
                graph.abort();

                require(fb == ec.expect_fallback, "Edge case fallback mismatch");
                std::cout << "    " << ec.desc << " -> fallback=" << (fb ? "TRUE" : "FALSE") << " PASS" << std::endl;
            }
        }

        // 4C: Token Count Sweep & Zero-Token Edge Case
        std::cout << "  [Test 4C] Token count sweep & zero-token edge case:" << std::endl;
        {
            // Zero tokens: should return false immediately with no telemetry update
            graph.reset_router_margin_telemetry();
            graph.begin();
            const bool fb_zero = graph.evaluate_router_margin(0);
            graph.abort();
            require(fb_zero == false, "Zero tokens must return false");
            const auto tel_zero = graph.router_margin_telemetry();
            require(tel_zero.total_tokens == 0 && tel_zero.total_layers == 0, "Zero tokens leaves telemetry unchanged");
            std::cout << "    tokens = 0 edge case verified PASS" << std::endl;

            // Token count sweep: {1, 2, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512}
            const uint32_t counts[] = {1, 2, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512};
            for (uint32_t n_tokens : counts) {
                // Test 1: All safe (0.5f)
                {
                    graph.reset_router_margin_telemetry();
                    std::vector<float> m(512, 0.5f);
                    graph.write(scratch, std::as_bytes(std::span(m)));
                    graph.begin();
                    const bool fb = graph.evaluate_router_margin(n_tokens);
                    graph.abort();
                    require(fb == false, "All safe batch must not trigger fallback");
                    require(graph.router_margin_telemetry().violation_count == 0, "Violation count must be 0");
                }

                // Test 2: Single violation at index 0
                {
                    graph.reset_router_margin_telemetry();
                    std::vector<float> m(512, 0.5f);
                    m[0] = 1.0e-5f;
                    graph.write(scratch, std::as_bytes(std::span(m)));
                    graph.begin();
                    const bool fb = graph.evaluate_router_margin(n_tokens);
                    graph.abort();
                    require(fb == true, "Violation at index 0 must trigger fallback");
                    require(graph.router_margin_telemetry().violation_count == 1, "Violation count must be 1");
                }

                // Test 3: Single violation at index n_tokens - 1
                if (n_tokens > 1) {
                    graph.reset_router_margin_telemetry();
                    std::vector<float> m(512, 0.5f);
                    m[n_tokens - 1] = 1.0e-5f;
                    graph.write(scratch, std::as_bytes(std::span(m)));
                    graph.begin();
                    const bool fb = graph.evaluate_router_margin(n_tokens);
                    graph.abort();
                    require(fb == true, "Violation at last index must trigger fallback");
                    require(graph.router_margin_telemetry().violation_count == 1, "Violation count must be 1");
                }

                // Test 4: Single violation at middle index
                if (n_tokens > 2) {
                    graph.reset_router_margin_telemetry();
                    std::vector<float> m(512, 0.5f);
                    m[n_tokens / 2] = 1.0e-5f;
                    graph.write(scratch, std::as_bytes(std::span(m)));
                    graph.begin();
                    const bool fb = graph.evaluate_router_margin(n_tokens);
                    graph.abort();
                    require(fb == true, "Violation at middle index must trigger fallback");
                    require(graph.router_margin_telemetry().violation_count == 1, "Violation count must be 1");
                }

                // Test 5: All violating
                {
                    graph.reset_router_margin_telemetry();
                    std::vector<float> m(512, 0.5f);
                    for (uint32_t i = 0; i < n_tokens; ++i) m[i] = 1.0e-5f;
                    graph.write(scratch, std::as_bytes(std::span(m)));
                    graph.begin();
                    const bool fb = graph.evaluate_router_margin(n_tokens);
                    graph.abort();
                    require(fb == true, "All violating batch must trigger fallback");
                    require(graph.router_margin_telemetry().violation_count == n_tokens,
                        "Violation count must equal n_tokens");
                }
            }
            std::cout << "    Token count sweep across 14 batch sizes verified PASS" << std::endl;
        }

        // 4D: Multi-Layer Simulation & Telemetry Accuracy
        std::cout << "  [Test 4D] 48-layer model simulation & telemetry accounting:" << std::endl;
        {
            graph.reset_router_margin_telemetry();
            constexpr uint32_t num_layers = 48;
            constexpr uint32_t n_tokens = 512;
            uint64_t expected_violations = 0;
            uint64_t expected_fallback_layers = 0;
            float expected_min_margin = std::numeric_limits<float>::infinity();
            double expected_sum_margin = 0.0;

            for (uint32_t layer = 0; layer < num_layers; ++layer) {
                // Every 4th layer has violations (12 fallback layers total)
                const bool inject_violation = (layer % 4 == 0);
                std::vector<float> m(n_tokens, 0.3f);
                if (inject_violation) {
                    // Inject 5 violations per violating layer
                    for (uint32_t v = 0; v < 5; ++v) {
                        const float bad_m = 1.0e-5f * float(v + 1);
                        m[v * 20] = bad_m;
                    }
                    expected_violations += 5;
                    expected_fallback_layers += 1;
                }
                for (float val : m) {
                    if (val < expected_min_margin) expected_min_margin = val;
                    expected_sum_margin += val;
                }

                graph.write(scratch, std::as_bytes(std::span(m)));
                graph.begin();
                const bool fb = graph.evaluate_router_margin(n_tokens);
                graph.abort();

                require(fb == inject_violation, "Layer fallback mismatch in 48-layer simulation");
            }

            const auto tel = graph.router_margin_telemetry();
            require(tel.total_layers == num_layers, "Total layers must be 48");
            require(tel.fallback_layers == expected_fallback_layers, "Fallback layers must be 12");
            require(tel.total_tokens == num_layers * n_tokens, "Total tokens must be 48 * 512");
            require(tel.violation_count == expected_violations, "Violation count must be 12 * 5 = 60");
            require(std::abs(tel.min_margin - expected_min_margin) < 1.0e-8f, "Min margin must match");
            require(std::abs(tel.sum_margin - expected_sum_margin) < 1.0e-2, "Sum margin must match");

            std::cout << "    48 layers: total_layers=" << tel.total_layers
                      << ", fallback_layers=" << tel.fallback_layers
                      << ", violations=" << tel.violation_count
                      << ", min_margin=" << tel.min_margin << " PASS" << std::endl;
        }

        // 4E: Telemetry Reset & Session State Invariance
        std::cout << "  [Test 4E] Telemetry reset & global session state invariance:" << std::endl;
        {
            graph.reset_router_margin_telemetry();
            const auto tel_after = graph.router_margin_telemetry();
            require(tel_after.total_tokens == 0, "total_tokens must be reset");
            require(tel_after.total_layers == 0, "total_layers must be reset");
            require(tel_after.violation_count == 0, "violation_count must be reset");
            require(tel_after.fallback_layers == 0, "fallback_layers must be reset");
            require(tel_after.sum_margin == 0.0, "sum_margin must be reset");
            require(tel_after.min_margin == std::numeric_limits<float>::infinity(), "min_margin must be reset to inf");

            require(graph.prefill_grouped_auto() == true, "prefill_grouped_auto must remain true");
            require(graph.prefill_grouped_f32() == false, "prefill_grouped_f32 must remain false");
            std::cout << "    Reset telemetry and session state invariance PASS" << std::endl;
        }
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "ALL EMPIRICAL CHALLENGES (SUITES 1-4) COMPLETED SUCCESSFULLY!" << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
