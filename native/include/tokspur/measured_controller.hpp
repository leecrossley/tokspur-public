#pragma once

#include "tokspur/json.hpp"
#include "tokspur/sampling.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur {

enum class MeasuredSampling : uint8_t { greedy, full, topk, coding };
MeasuredSampling measured_sampling(const SamplingPolicy& policy);
std::string_view measured_sampling_name(MeasuredSampling sampling);
enum class MeasuredDrafter : uint8_t { mtp, dflash2 };
using AcceptanceRates = std::array<double, 7>;

struct MeasuredCosts {
    double ordinary_ms = 0;
    std::array<double, 8> draft_ms{};
    std::array<std::array<double, 8>, 8> remaining_ms{};
    AcceptanceRates conditional{};
    bool calibrated = false;
};
struct MeasuredCostAnchor {
    uint32_t context = 0;
    MeasuredCosts costs;
};
struct MeasuredCostMode {
    MeasuredSampling sampling = MeasuredSampling::greedy;
    std::vector<MeasuredCostAnchor> anchors;
};

// Immutable after loading. Execution identity and an exactness receipt are
// mandatory, but a parsed receipt alone does not establish model qualification.
class MeasuredCostProfile {
public:
    static MeasuredCostProfile parse(const Json& value);
    [[nodiscard]] MeasuredCosts costs(uint32_t context, MeasuredSampling sampling) const;
    [[nodiscard]] bool supports_sampling(MeasuredSampling sampling) const noexcept;
    void bind(std::string_view execution_digest, MeasuredDrafter drafter, bool prepared_validation = false,
              bool shared_pack_mapping = false) const;
    void admit_exactness(const Json& receipt, std::string_view sha256) const;
    [[nodiscard]] uint32_t maximum_rows() const noexcept { return maximum_rows_; }
    [[nodiscard]] MeasuredDrafter drafter() const noexcept { return drafter_; }
    [[nodiscard]] const std::string& policy_digest() const noexcept { return policy_digest_; }
    [[nodiscard]] const std::string& execution_digest() const noexcept { return execution_digest_; }
private:
    uint32_t maximum_rows_ = 0;
    bool prepared_validation_ = false;
    bool shared_pack_mapping_ = false;
    MeasuredDrafter drafter_ = MeasuredDrafter::mtp;
    std::string execution_digest_, proof_digest_, policy_digest_;
    std::vector<MeasuredCostMode> modes_;
};

struct MeasuredChoice {
    uint32_t rows = 1;
    double lambda = 0, surplus = 0;
    bool calibrated = false, converged = false, probe = false;
};

struct MeasuredObservation {
    uint64_t generation = 0, actions = 0;
    AcceptanceRates acceptance{};
    double cost_multiplier = 1, rate = 0, last_decode_ms = 0;
    uint32_t history_size = 0, since_deep_probe = 0;
    bool converged = true, calibrated = true;
    bool operator==(const MeasuredObservation&) const = default;
    [[nodiscard]] Json json() const;
    static MeasuredObservation parse(const Json& value);
};

// A DONE carries every timing input used by its decoded actions, including
// several actions in one output packet. These observations are not persisted.
void validate_measured_timing_packet(const Json& ready, const Json& previous, const Json& done,
    bool begin, uint64_t generation, uint32_t actions);

class MeasuredController {
public:
    static constexpr uint32_t pool_capacity = 32, maximum_iterations = 20, probe_interval = 16;
    static constexpr double relative_tolerance = 1e-6, acceptance_beta = 0.125, timing_decay = 0.97;
    explicit MeasuredController(const MeasuredCostProfile& profile) : profile_(profile) {}
    void set_ordinary_allowed(bool enabled) noexcept { ordinary_allowed_ = enabled; last_solve_context_ = 0; }
    void begin_generation(uint64_t id, uint32_t context, MeasuredSampling sampling);
    void begin_packet() noexcept { packet_size_ = 0; }
    [[nodiscard]] Json timing_packet() const;
    [[nodiscard]] MeasuredChoice before_draft(uint32_t context, uint32_t maximum_rows);
    [[nodiscard]] MeasuredChoice after_draft(uint32_t context, uint32_t maximum_rows,
        const AcceptanceRates& probabilities, uint32_t paid_draft_rows = 0);
    [[nodiscard]] MeasuredChoice after_draft_causal(uint32_t context, uint32_t maximum_rows,
        const AcceptanceRates& probabilities, uint32_t paid_draft_rows);
    void observe(uint32_t context, uint32_t rows, uint32_t accepted, double decode_ms,
        const AcceptanceRates& probabilities, bool labels = true, uint32_t drafted_rows = 0, bool stopped = false);
    void restore_acceptance(const AcceptanceRates& rates);
    [[nodiscard]] const AcceptanceRates& acceptance() const noexcept { return acceptance_; }
    [[nodiscard]] double cost_multiplier() const noexcept { return drift_; }
    [[nodiscard]] uint32_t history_size() const noexcept { return pool_size_; }
    [[nodiscard]] uint32_t actions_since_deep_probe() const noexcept { return since_probe_; }
    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] bool speculative_mode() const noexcept { return speculative_; }
    [[nodiscard]] MeasuredObservation observation() const noexcept;
    static double reward(uint32_t rows, const AcceptanceRates& probabilities);
    static double expected_cost(uint32_t rows, const AcceptanceRates& probabilities,
        const MeasuredCosts& costs, bool draft_paid);
private:
    MeasuredChoice choose(uint32_t context, uint32_t maximum_rows,
        const AcceptanceRates* probabilities, uint32_t paid_draft_rows = 0);
    double rate(const MeasuredCosts& costs, bool& converged, uint32_t context);
    void condition(const MeasuredCosts& costs);
    const MeasuredCostProfile& profile_;
    MeasuredSampling sampling_ = MeasuredSampling::greedy;
    AcceptanceRates acceptance_{}, anchor_prior_{};
    std::array<AcceptanceRates, pool_capacity> pool_{};
    std::array<double, 64> packet_ms_{};
    uint32_t packet_size_ = 0;
    uint32_t pool_size_ = 0, next_pool_ = 0, since_probe_ = 0;
    uint64_t generation_ = 0, actions_ = 0;
    double last_decode_ms_ = 0;
    double drift_ = 1, last_lambda_ = 0;
    bool speculative_ = true, ordinary_allowed_ = true;
    bool converged_ = true, calibrated_ = true;
    uint32_t last_solve_context_ = 0;
    uint64_t last_solve_actions_ = 0;
    bool last_solve_converged_ = false;
};

} // namespace tokspur
