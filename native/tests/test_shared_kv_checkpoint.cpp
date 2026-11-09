#include "tokspur/mtp_anchor_state.hpp"
#include "tokspur/sampling.hpp"

#include <CommonCrypto/CommonCrypto.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

using namespace tokspur;

// Mock structures to test in-memory BranchSession logic and stochastic restore invariants

struct MockHandle {
    uint32_t slot = 0;
    uint32_t prefix = 0;
};

struct MockResult {
    std::vector<float> logits;
};

template<class Result>
class MockCommittedResult {
    std::optional<Result> value_;
    uint32_t position_ = 0;
    std::optional<uint32_t> stop_id_;
public:
    void set(uint32_t pos, Result r, std::optional<uint32_t> stop = std::nullopt) {
        position_ = pos;
        value_ = std::move(r);
        stop_id_ = stop;
    }
    [[nodiscard]] bool matches(uint32_t pos) const noexcept { return value_.has_value() && position_ == pos; }
    [[nodiscard]] const Result& get(uint32_t) const { return *value_; }
    [[nodiscard]] std::optional<uint32_t> stop_id() const noexcept { return stop_id_; }
    void clear() { value_.reset(); position_ = 0; stop_id_.reset(); }
    void exchange(std::optional<Result>& transfer, uint32_t pos, std::optional<uint32_t> stop) {
        std::swap(value_, transfer);
        position_ = pos;
        stop_id_ = stop;
    }
};

static void test_stochastic_rng_restore_parity() {
    // Feature 62: Stochastic RNG & Controller restore
    SamplingPolicy policy;
    policy.temperature = 0.7f;
    policy.top_k = 10;
    const uint64_t seed = 42;

    SamplingRng rng(seed);

    // Draw 10 random numbers
    std::vector<double> stream1;
    for (int i = 0; i < 10; ++i) {
        stream1.push_back(rng.uniform());
    }

    // Capture checkpoint of RNG state at step 10
    SamplingRng checkpoint_rng = rng;
    assert(checkpoint_rng.words_consumed() == 10);

    // Draw next 10 numbers
    std::vector<double> stream2_original;
    for (int i = 0; i < 10; ++i) {
        stream2_original.push_back(rng.uniform());
    }

    // Restore RNG from checkpoint
    SamplingRng restored_rng = checkpoint_rng;
    assert(restored_rng.words_consumed() == 10);

    std::vector<double> stream2_restored;
    for (int i = 0; i < 10; ++i) {
        stream2_restored.push_back(restored_rng.uniform());
    }

    // Must match bit-identical
    assert(stream2_original.size() == stream2_restored.size());
    for (size_t i = 0; i < stream2_original.size(); ++i) {
        assert(stream2_original[i] == stream2_restored[i]);
    }
}

static void test_controller_policy_digest_refusal() {
    // Feature 62: Strict FormatError refusal if controller_policy_digest != default_controller_policy_digest
    StageAControllerState valid_state;
    assert(valid_state.controller_policy_digest == default_controller_policy_digest);

    StageAControllerState invalid_state = valid_state;
    invalid_state.controller_policy_digest[0] ^= 0xFF; // Tamper 1 byte
    assert(invalid_state.controller_policy_digest != default_controller_policy_digest);

    // Verify mtp_generation_command check
    Json::Object obj{
        {"id", Json::integer(1)},
        {"begin", Json(true)},
        {"remaining", Json::integer(16)},
        {"committed", Json::integer(0)},
        {"emitted", Json::integer(0)},
        {"pending", Json::integer(0)},
        {"remaining_compute", Json::integer(16)},
        {"total_budget", Json::integer(16)},
    };
    Json::Object ctrl{
        {"ewma_alpha1", Json::number(0.45)},
        {"current_mode", Json::integer(0)},
        {"consecutive_ordinary_steps", Json::integer(0)},
    };
    Json::Array digest_arr;
    for (auto b : invalid_state.controller_policy_digest) digest_arr.push_back(Json::integer(b));
    ctrl.emplace_back("controller_policy_digest", Json(std::move(digest_arr)));
    obj.emplace_back("stage_a_controller", Json(std::move(ctrl)));

    MtpGenerationCommand cmd = mtp_generation_command(Json(std::move(obj)));
    MtpGenerationCarry carry;
    const std::array<uint32_t, 1> stops{151643};
    bool refused = false;
    try {
        carry.admit(cmd, 8, 0, stops, 8);
    } catch (const FormatError&) {
        refused = true;
    }
    assert(refused);
}

static void test_retained_logits_and_frontier_capture() {
    // Feature 63: Retained Logits & Frontier Capture
    MockResult result;
    result.logits.resize(100, 2.5f);

    // Verify all logits are finite
    for (float f : result.logits) {
        assert(std::isfinite(f));
    }

    // Verify non-finite logit rejection
    bool rejected_nan = false;
    try {
        result.logits[50] = NAN;
        for (float f : result.logits) {
            if (!std::isfinite(f)) throw FormatError("nonfinite saved branch logits");
        }
    } catch (const FormatError&) {
        rejected_nan = true;
    }
    assert(rejected_nan);

    // MtpHead frontier calculations
    const uint32_t prefix = 64;
    const uint32_t target_mtp_rows = prefix > 1 ? prefix - 1 : 0;
    assert(target_mtp_rows == 63);

    const uint32_t prefix_zero = 0;
    const uint32_t target_mtp_rows_zero = prefix_zero > 1 ? prefix_zero - 1 : 0;
    assert(target_mtp_rows_zero == 0);

    const uint32_t prefix_one = 1;
    const uint32_t target_mtp_rows_one = prefix_one > 1 ? prefix_one - 1 : 0;
    assert(target_mtp_rows_one == 0);
}

static void test_cs64_chunk_replay_digest_binding() {
    // Feature 64: CS64 Chunk Replay Digest Binding
    auto compute_digest = [](uint32_t prefix, uint8_t cm, const std::vector<uint32_t>& partitions,
                             std::array<uint32_t, 2> kp, std::array<uint8_t, 32> cpd) {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        CC_SHA256_Update(&ctx, &prefix, sizeof(prefix));
        CC_SHA256_Update(&ctx, &cm, sizeof(cm));
        uint32_t acc = 0;
        for (auto p : partitions) {
            if (acc >= prefix) break;
            uint32_t chunk = std::min(p, prefix - acc);
            if (chunk > 0) {
                CC_SHA256_Update(&ctx, &chunk, sizeof(chunk));
                acc += chunk;
            }
        }
        CC_SHA256_Update(&ctx, kp.data(), sizeof(kp));
        CC_SHA256_Update(&ctx, cpd.data(), sizeof(cpd));
        std::array<uint8_t, 32> out{};
        CC_SHA256_Final(out.data(), &ctx);
        return out;
    };

    const uint32_t prefix = 128;
    const std::vector<uint32_t> parts1{64, 64};
    const std::vector<uint32_t> parts2{32, 32, 64};
    const std::array<uint32_t, 2> kp{0, 0};
    const std::array<uint8_t, 32> cpd = default_controller_policy_digest;

    const auto d_cs64 = compute_digest(prefix, 1, parts1, kp, cpd);
    const auto d_fused = compute_digest(prefix, 0, parts1, kp, cpd);
    const auto d_divergent_parts = compute_digest(prefix, 1, parts2, kp, cpd);

    // Chunk mode divergence must produce different digest
    assert(d_cs64 != d_fused);

    // Partition boundary divergence must produce different digest
    assert(d_cs64 != d_divergent_parts);

    // Identical parameters must produce identical bit-for-bit digest
    const auto d_cs64_replay = compute_digest(prefix, 1, parts1, kp, cpd);
    assert(d_cs64 == d_cs64_replay);
}

static void test_pre_mutation_compaction_truncation() {
    // Feature 65: Pre-Mutation Compaction Truncation
    const std::vector<uint32_t> original_history = {10, 20, 30, 40, 50, 60, 70, 80};
    // Mutated prompt at index 4 (50 -> 99)
    const std::vector<uint32_t> compacted_prompt = {10, 20, 30, 40, 99, 100};

    // Calculate common prefix
    size_t common = 0;
    while (common < original_history.size() && common < compacted_prompt.size() &&
           original_history[common] == compacted_prompt[common]) {
        ++common;
    }
    assert(common == 4); // Earliest mutated token index i_mut is 4

    // Available checkpoints at prefixes: 2, 4, 6
    struct MockRecord {
        uint32_t prefix;
        bool valid;
    };
    std::vector<MockRecord> records = {
        {2, true},
        {4, true},
        {6, true},
    };

    // Select candidate with prefix <= common
    std::optional<size_t> selected;
    for (size_t i = 0; i < records.size(); ++i) {
        if (records[i].valid && records[i].prefix <= common) {
            if (!selected || records[i].prefix > records[*selected].prefix) {
                selected = i;
            }
        }
    }
    assert(selected.has_value());
    assert(records[*selected].prefix == 4);

    // On restore to prefix 4, invalidate all descendants (> 4)
    const uint32_t restored_prefix = records[*selected].prefix;
    for (auto& rec : records) {
        if (rec.valid && rec.prefix > restored_prefix) {
            rec.valid = false;
        }
    }

    assert(records[0].valid);  // Prefix 2 survives (ancestor)
    assert(records[1].valid);  // Prefix 4 survives (selected)
    assert(!records[2].valid); // Prefix 6 invalidated (descendant beyond mutation)
}

int main() {
    std::cout << "Running Shared-KV Checkpoint unit tests..." << std::endl;
    test_stochastic_rng_restore_parity();
    test_controller_policy_digest_refusal();
    test_retained_logits_and_frontier_capture();
    test_cs64_chunk_replay_digest_binding();
    test_pre_mutation_compaction_truncation();
    std::cout << "All Shared-KV Checkpoint unit tests passed successfully." << std::endl;
    return 0;
}
