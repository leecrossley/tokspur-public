#include "tokspur/measured_controller.hpp"
#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <set>

namespace tokspur {
namespace {
void need(bool value, const char* message) { if (!value) throw FormatError(message); }
const Json& field(const Json& v, std::string_view key) {
    const auto* found = v.find(key); need(found != nullptr, "incomplete measured cost profile"); return *found;
}
void keys(const Json& v, std::initializer_list<std::string_view> allowed) {
    need(v.kind() == Json::Kind::object, "measured cost profile object required");
    for (const auto& item : v.object())
        need(std::find(allowed.begin(), allowed.end(), item.first) != allowed.end(), "unknown measured cost profile field");
}
double real(const Json& v, bool positive = true) {
    const auto s = v.number_lexeme(); double out = 0;
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), out);
    need(parsed.ec == std::errc{} && parsed.ptr == s.data() + s.size() && std::isfinite(out)
        && (positive ? out > 0 : out >= 0), "invalid measured cost or probability"); return out;
}
bool hash(std::string_view s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void valid_rates(const AcceptanceRates& values, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i)
        need(std::isfinite(values[i]) && values[i] >= 0 && values[i] <= 1, "invalid measured acceptance estimate");
}
double logit(double p) {
    p = std::clamp(p, 1e-6, 1.0 - 1e-6); return std::log(p / (1.0 - p));
}
double logistic(double v) { return v >= 0 ? 1.0 / (1.0 + std::exp(-v)) : std::exp(v) / (1.0 + std::exp(v)); }
void u64(std::string& out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.push_back(char((value >> (i * 8)) & 255));
}
void number(std::string& out, double value) { u64(out, std::bit_cast<uint64_t>(value)); }
MeasuredCosts interpolate(const MeasuredCosts& a, const MeasuredCosts& b, double t, uint32_t rows) {
    MeasuredCosts out; out.calibrated = true;
    const auto mix = [t](double x, double y) { return x + t * (y - x); };
    out.ordinary_ms = mix(a.ordinary_ms, b.ordinary_ms);
    for (uint32_t m = 0; m < rows; ++m) {
        out.draft_ms[m] = mix(a.draft_ms[m], b.draft_ms[m]);
        for (uint32_t k = 0; k <= m; ++k) out.remaining_ms[m][k] = mix(a.remaining_ms[m][k], b.remaining_ms[m][k]);
        if (m + 1 < rows) out.conditional[m] = mix(a.conditional[m], b.conditional[m]);
    }
    return out;
}
}

MeasuredSampling measured_sampling(const SamplingPolicy& policy) {
    if (policy.temperature == 0) return MeasuredSampling::greedy;
    need(policy.temperature == 1 && policy.min_p == 0,
        "sampling policy is not calibrated by measured profile schema 1");
    if (policy.top_k == 20 && policy.top_p == 0.95) return MeasuredSampling::coding;
    need(policy.top_p == 1 && (policy.top_k == 0 || policy.top_k == 20),
        "sampling policy is not calibrated by measured profile schema 1");
    return policy.top_k == 20 ? MeasuredSampling::topk : MeasuredSampling::full;
}
std::string_view measured_sampling_name(MeasuredSampling sampling) {
    switch (sampling) {
        case MeasuredSampling::greedy: return "greedy";
        case MeasuredSampling::full: return "full";
        case MeasuredSampling::topk: return "topk";
        case MeasuredSampling::coding: return "coding";
    }
    throw FormatError("unknown measured sampling mode");
}

MeasuredCostProfile MeasuredCostProfile::parse(const Json& v) {
    keys(v, {"schema", "drafter", "maximum_rows", "execution_digest", "exactness_receipt", "prepared_validation", "shared_pack_mapping", "modes"});
    need(field(v, "schema").unsigned_integer() == 1, "unsupported measured cost profile schema");
    MeasuredCostProfile out;
    const auto kind = field(v, "drafter").string();
    need(kind == "mtp" || kind == "dflash2", "unknown measured drafter");
    out.drafter_ = kind == "mtp" ? MeasuredDrafter::mtp : MeasuredDrafter::dflash2;
    const auto rows = field(v, "maximum_rows").unsigned_integer();
    need(rows >= 2 && rows <= (out.drafter_ == MeasuredDrafter::mtp ? 4u : 5u), "measured serving row bound exceeded");
    out.maximum_rows_ = uint32_t(rows);
    if (const auto* prepared = v.find("prepared_validation")) out.prepared_validation_ = prepared->boolean();
    if (const auto* shared = v.find("shared_pack_mapping")) out.shared_pack_mapping_ = shared->boolean();
    out.execution_digest_ = field(v, "execution_digest").string();
    out.proof_digest_ = field(v, "exactness_receipt").string();
    need(hash(out.execution_digest_) && hash(out.proof_digest_), "measured profile requires execution and exactness digests");
    const auto& modes = field(v, "modes").array(); need(!modes.empty() && modes.size() <= 4, "measured sampling mode bound");
    std::set<MeasuredSampling> seen;
    for (const auto& mode : modes) {
        keys(mode, {"sampling", "anchors"}); MeasuredCostMode next;
        const auto name = field(mode, "sampling").string();
        need(name == "greedy" || name == "full" || name == "topk" || name == "coding", "unknown measured sampling mode");
        next.sampling = name == "greedy" ? MeasuredSampling::greedy : name == "full" ? MeasuredSampling::full : name == "topk" ? MeasuredSampling::topk : MeasuredSampling::coding;
        need(seen.insert(next.sampling).second, "duplicate measured sampling mode");
        const auto& anchors = field(mode, "anchors").array(); need(!anchors.empty() && anchors.size() <= 8, "measured context anchor bound");
        for (const auto& anchor : anchors) {
            keys(anchor, {"context", "ordinary_ms", "draft_ms", "remaining_ms", "conditional_acceptance"});
            MeasuredCostAnchor a; const auto context = field(anchor, "context").unsigned_integer();
            need(context && context <= 65536 && (next.anchors.empty() || context > next.anchors.back().context), "measured context anchors must increase");
            a.context = uint32_t(context); a.costs.calibrated = true; a.costs.ordinary_ms = real(field(anchor, "ordinary_ms"));
            const auto& draft = field(anchor, "draft_ms").array();
            const auto& remaining = field(anchor, "remaining_ms").array();
            const auto& acceptance = field(anchor, "conditional_acceptance").array();
            need(draft.size() == rows && remaining.size() == rows && acceptance.size() + 1 == rows, "incomplete measured row costs");
            for (size_t m = 0; m < rows; ++m) {
                a.costs.draft_ms[m] = real(draft[m], false);
                need(m || a.costs.draft_ms[m] == 0, "ordinary action cannot contain drafting");
                const auto& costs = remaining[m].array(); need(costs.size() == m + 1, "incomplete measured accepted-prefix costs");
                for (size_t k = 0; k <= m; ++k) a.costs.remaining_ms[m][k] = real(costs[k]);
                if (m + 1 < rows) a.costs.conditional[m] = real(acceptance[m], false);
            }
            valid_rates(a.costs.conditional, uint32_t(rows - 1));
            if (out.drafter_ == MeasuredDrafter::dflash2) for (size_t m = 2; m < rows; ++m)
                need(a.costs.draft_ms[m] == a.costs.draft_ms[1], "DFlash2 verification widths must include the same complete draft cost");
            next.anchors.push_back(a);
        }
        out.modes_.push_back(std::move(next));
    }
    std::sort(out.modes_.begin(), out.modes_.end(), [](const auto& a, const auto& b) { return a.sampling < b.sampling; });
    // Hash the effective in-memory values, not the spelling/order of the input JSON.
    std::string bytes("tokspur-measured-controller-v4-causal-stopping-prefix-context-packet-tail");
    bytes += out.execution_digest_; u64(bytes, uint64_t(out.drafter_)); u64(bytes, rows); u64(bytes, out.prepared_validation_);
    for (const auto n : {MeasuredController::pool_capacity, MeasuredController::maximum_iterations, MeasuredController::probe_interval}) u64(bytes, n);
    for (double n : {MeasuredController::relative_tolerance, MeasuredController::acceptance_beta, MeasuredController::timing_decay, 0.04}) number(bytes, n);
    for (const auto& mode : out.modes_) {
        u64(bytes, uint64_t(mode.sampling)); u64(bytes, mode.anchors.size());
        for (const auto& a : mode.anchors) {
            u64(bytes, a.context); number(bytes, a.costs.ordinary_ms);
            for (uint32_t m = 0; m < rows; ++m) {
                number(bytes, a.costs.draft_ms[m]);
                for (uint32_t k = 0; k <= m; ++k) number(bytes, a.costs.remaining_ms[m][k]);
                if (m + 1 < rows) number(bytes, a.costs.conditional[m]);
            }
        }
    }
    // Preserve existing private-mapping policy digests. Mapping affects measured
    // costs, but neither represented operands nor model/checkpoint compatibility.
    if (out.shared_pack_mapping_) bytes += ";immutable-pack-mapping=shared-v1";
    out.policy_digest_ = derived_pack::hash(bytes); return out;
}

Json MeasuredObservation::json() const {
    Json::Array p; for (double v : acceptance) p.push_back(Json::number(v));
    return Json(Json::Object{{"generation", Json::integer(generation)}, {"actions", Json::integer(actions)}, {"last_decode_ms", Json::number(last_decode_ms)}, {"acceptance", Json(std::move(p))},
        {"cost_multiplier", Json::number(cost_multiplier)}, {"rate", Json::number(rate)},
        {"history_size", Json::integer(history_size)}, {"since_deep_probe", Json::integer(since_deep_probe)},
        {"converged", Json(converged)}, {"calibrated", Json(calibrated)}});
}
MeasuredObservation MeasuredObservation::parse(const Json& v) {
    keys(v, {"generation", "actions", "last_decode_ms", "acceptance", "cost_multiplier", "rate", "history_size", "since_deep_probe", "converged", "calibrated"});
    MeasuredObservation out; out.generation = field(v, "generation").unsigned_integer();
    need(out.generation != 0, "invalid measured generation");
    out.actions = field(v, "actions").unsigned_integer(); out.last_decode_ms = real(field(v, "last_decode_ms"), false);
    need(out.actions <= 32768 && ((out.actions == 0) == (out.last_decode_ms == 0)), "incoherent measured action observation");
    const auto& a = field(v, "acceptance").array(); need(a.size() == 7, "invalid measured acceptance shape");
    for (size_t i = 0; i < a.size(); ++i) out.acceptance[i] = real(a[i], false);
    valid_rates(out.acceptance, 7); out.cost_multiplier = real(field(v, "cost_multiplier")); out.rate = real(field(v, "rate"), false);
    const auto pool = field(v, "history_size").unsigned_integer(), probe = field(v, "since_deep_probe").unsigned_integer();
    need(pool && pool <= 32 && probe <= 16, "measured observation bounds exceeded");
    out.history_size = uint32_t(pool); out.since_deep_probe = uint32_t(probe);
    out.converged = field(v, "converged").boolean(); out.calibrated = field(v, "calibrated").boolean(); return out;
}
MeasuredObservation MeasuredController::observation() const noexcept {
    return {generation_, actions_, acceptance_, drift_, last_lambda_, last_decode_ms_, pool_size_, since_probe_, converged_, calibrated_};
}
Json MeasuredController::timing_packet() const {
    Json::Array times; for (uint32_t i = 0; i < packet_size_; ++i) times.push_back(Json::number(packet_ms_[i]));
    return Json(Json::Object{{"first_action", Json::integer(packet_size_ ? actions_ - packet_size_ + 1 : 0)},
        {"milliseconds", Json(std::move(times))}});
}
void validate_measured_timing_packet(const Json& ready, const Json& previous, const Json& done,
    bool begin, uint64_t generation, uint32_t actions) {
    const auto* measured = ready.find("native_measured_controller");
    const auto* packet = done.find("native_measured_decode_observations");
    if (!measured) {
        need(!packet && !done.find("native_measured_timing_schema"), "unsolicited measured timing packet"); return;
    }
    need(measured->boolean() && field(ready, "native_measured_timing_schema").unsigned_integer() == 1 &&
        field(done, "native_measured_timing_schema").unsigned_integer() == 1 && packet, "measured timing negotiation differs");
    keys(*packet, {"first_action", "milliseconds"});
    const auto first = field(*packet, "first_action").unsigned_integer();
    const auto& times = field(*packet, "milliseconds").array();
    need(actions <= 64 && times.size() == actions, "measured timing packet omitted or duplicated an action");
    if (!actions) { need(!first, "empty measured timing packet has an action"); return; }
    const auto state = [](const Json& value) { return MeasuredObservation::parse(field(field(field(value,
        "native_mtp_generation"), "stage_a_controller"), "measured")); };
    uint64_t before = 0;
    if (!begin) { const auto old = state(previous); need(old.generation == generation, "measured timing previous generation differs"); before = old.actions; }
    const auto current = state(done);
    need(first == before + 1 && current.generation == generation && current.actions == before + actions,
        "measured timing action frontier differs");
    for (const auto& time : times) static_cast<void>(real(time));
    need(real(times.back()) == current.last_decode_ms, "measured timing final observation differs");
}

void MeasuredCostProfile::admit_exactness(const Json& receipt, std::string_view sha256) const {
    need(sha256 == proof_digest_, "exact controller receipt hash differs");
    need(field(receipt, "schema").unsigned_integer() == 3 && field(receipt, "runtime_base").string() == "tokspur-execution-v3" &&
        field(receipt, "execution_digest").string() == execution_digest_, "exact controller receipt identity differs");
    const auto& cases = field(receipt, "comparisons").array();
    need(!cases.empty() && cases.size() <= 2048, "exact controller receipt case bound");
    for (const auto& mode : modes_) for (const auto& anchor : mode.anchors) {
        const auto name = measured_sampling_name(mode.sampling);
        for (uint32_t rows = 2; rows <= maximum_rows_; ++rows) for (uint32_t accepted = 1; accepted <= rows; ++accepted) {
            bool found = false;
            for (const auto& item : cases) {
                if (field(item, "context").unsigned_integer() != anchor.context || field(item, "rows").unsigned_integer() != rows ||
                    field(item, "accepted_prefix").unsigned_integer() != accepted || field(item, "sampling").string() != name) continue;
                need(!found, "duplicate exact controller receipt case"); found = true;
                const auto& captures = field(item, "capture_ids");
                const auto first_capture = field(captures, "ordinary").string(), second_capture = field(captures, "verification").string();
                need(first_capture.size() == 32 && second_capture.size() == 32 && first_capture != second_capture, "exact controller captures are not independent");
                const auto& identity = field(item, "source_identity");
                for (const auto key : {"worker_sha256", "library_sha256", "capture_sha256"}) need(hash(field(identity, key).string()), "exact controller capture build is missing");
                need(field(identity, "source_revision").string().size() == 40, "exact controller source revision is missing");
                const auto& sizes = field(item, "artifact_sizes");
                for (const auto arm : {"ordinary", "verification"}) {
                    const auto& values = field(sizes, arm);
                    need(field(values, "complete_logits").unsigned_integer() == 248320ull * 4, "exact controller logits are incomplete");
                    for (const auto name : {"initial_state", "accepted_state"}) need(field(values, name).unsigned_integer() >= 256 && field(values, name).unsigned_integer() % 4 == 0, "exact controller state is incomplete");
                    need(field(values, "sampling_distribution").unsigned_integer() >= 20 && field(values, "routes").unsigned_integer() >= rows * 4, "exact controller distribution or routes are incomplete");
                }
                const auto& ordinary = field(item, "ordinary"); const auto& verification = field(item, "verification");
                for (const auto key : {"initial_state", "complete_logits", "accepted_state", "sampling_distribution", "routes"}) {
                    const auto a = field(ordinary, key).string(), b = field(verification, key).string();
                    need(hash(a) && a == b, "exact controller admission failed: byte or route mismatch");
                }
            }
            need(found, "exact controller admission lacks a width, accepted prefix, context or sampling case");
        }
    }
}

void MeasuredCostProfile::bind(std::string_view execution, MeasuredDrafter kind, bool prepared, bool shared) const {
    need(execution == execution_digest_ && kind == drafter_ && prepared == prepared_validation_, "measured profile execution or drafter identity differs");
    need(shared == shared_pack_mapping_, "measured profile pack mapping differs");
}
bool MeasuredCostProfile::supports_sampling(MeasuredSampling sampling) const noexcept {
    return std::any_of(modes_.begin(), modes_.end(), [sampling](const auto& mode) { return mode.sampling == sampling; });
}
MeasuredCosts MeasuredCostProfile::costs(uint32_t context, MeasuredSampling sampling) const {
    need(context && context <= 65536, "measured context outside supported model");
    const auto mode = std::find_if(modes_.begin(), modes_.end(), [sampling](const auto& m) { return m.sampling == sampling; });
    need(mode != modes_.end(), "sampling mode is not calibrated by measured profile");
    const auto& anchors = mode->anchors;
    if (context <= anchors.front().context) { auto c = anchors.front().costs; c.calibrated = context == anchors.front().context; return c; }
    if (context >= anchors.back().context) { auto c = anchors.back().costs; c.calibrated = context == anchors.back().context; return c; }
    const auto b = std::lower_bound(anchors.begin(), anchors.end(), context, [](const auto& a, uint32_t c) { return a.context < c; });
    const auto& a = *(b - 1);
    return interpolate(a.costs, b->costs, double(context - a.context) / double(b->context - a.context), maximum_rows_);
}

double MeasuredController::reward(uint32_t rows, const AcceptanceRates& p) {
    need(rows && rows <= 8, "invalid measured action width"); valid_rates(p, rows - 1);
    double result = 1, survival = 1;
    for (uint32_t i = 0; i + 1 < rows; ++i) { survival *= p[i]; result += survival; }
    return result;
}
double MeasuredController::expected_cost(uint32_t rows, const AcceptanceRates& p, const MeasuredCosts& c, bool draft_paid) {
    need(rows && rows <= 8, "invalid measured action width"); valid_rates(p, rows - 1);
    if (rows == 1) return c.ordinary_ms;
    double result = draft_paid ? 0 : c.draft_ms[rows - 1], survival = 1;
    for (uint32_t a = 0; a + 1 < rows; ++a) {
        result += survival * (1 - p[a]) * c.remaining_ms[rows - 1][a]; survival *= p[a];
    }
    return result + survival * c.remaining_ms[rows - 1][rows - 1];
}
void MeasuredController::begin_generation(uint64_t id, uint32_t context, MeasuredSampling sampling) {
    need(id && id != generation_, "measured controller requires a fresh generation identity");
    const auto c = profile_.costs(context, sampling);
    begin_packet(); generation_ = id; actions_ = 0; last_decode_ms_ = 0; sampling_ = sampling; acceptance_ = anchor_prior_ = c.conditional;
    pool_ = {}; pool_size_ = 1; next_pool_ = 1; since_probe_ = 0;
    last_solve_context_ = 0;
    drift_ = 1; last_lambda_ = 0; speculative_ = true; converged_ = true; calibrated_ = c.calibrated;
}
void MeasuredController::restore_acceptance(const AcceptanceRates& rates) {
    valid_rates(rates, profile_.maximum_rows() - 1); acceptance_ = rates; last_solve_context_ = 0;
}
void MeasuredController::condition(const MeasuredCosts& costs) {
    for (uint32_t i = 0; i + 1 < profile_.maximum_rows(); ++i) {
        if (costs.conditional[i] == anchor_prior_[i]) continue;
        acceptance_[i] = acceptance_[i] == anchor_prior_[i] ? costs.conditional[i]
            : logistic(logit(costs.conditional[i]) + logit(acceptance_[i]) - logit(anchor_prior_[i]));
    }
    anchor_prior_ = costs.conditional;
}
double MeasuredController::rate(const MeasuredCosts& c, bool& converged, uint32_t context) {
    if (last_solve_context_ == context && last_solve_actions_ == actions_) { converged = last_solve_converged_; return last_lambda_; }
    last_solve_context_ = context; last_solve_actions_ = actions_;
    const uint32_t rows = profile_.maximum_rows();
    double lambda = last_lambda_ > 0 ? last_lambda_ * drift_ : 1.0 / c.ordinary_ms;
    for (uint32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        double rewards = 0, costs = 0;
        for (uint32_t i = 0; i < pool_size_; ++i) {
            AcceptanceRates p{};
            for (uint32_t k = 0; k + 1 < rows; ++k) p[k] = pool_[i][k] == 0 ? acceptance_[k] : logistic(logit(acceptance_[k]) + pool_[i][k]);
            uint32_t best = 1; double best_r = 1, best_c = c.ordinary_ms + (!ordinary_allowed_ && profile_.drafter() == MeasuredDrafter::dflash2 ? c.draft_ms[1] : 0), best_s = best_r - lambda * best_c;
            for (uint32_t m = 2; m <= rows; ++m) {
                const double r = reward(m, p), cost = expected_cost(m, p, c, false), surplus = r - lambda * cost;
                if (surplus > best_s) { best = m; best_r = r; best_c = cost; best_s = surplus; }
            }
            static_cast<void>(best); rewards += best_r; costs += best_c;
        }
        const double next = rewards / costs;
        need(std::isfinite(next) && next > 0, "invalid measured rate estimate");
        if (std::abs(next - lambda) <= relative_tolerance * std::max(next, lambda)) {
            converged = last_solve_converged_ = true; last_lambda_ = next / drift_; return last_lambda_;
        }
        lambda = next;
    }
    converged = last_solve_converged_ = false;
    if (!(last_lambda_ > 0)) last_lambda_ = 1.0 / (c.ordinary_ms * drift_);
    return last_lambda_;
}
MeasuredChoice MeasuredController::choose(uint32_t context, uint32_t maximum_rows, const AcceptanceRates* probabilities, uint32_t paid_draft_rows) {
    need(generation_ && maximum_rows && maximum_rows <= profile_.maximum_rows(), "measured action outside configured generation");
    const auto c = profile_.costs(context, sampling_); condition(c); MeasuredChoice out;
    out.calibrated = c.calibrated; out.lambda = rate(c, out.converged, context);
    calibrated_ = out.calibrated; converged_ = out.converged;
    const auto& p = probabilities ? *probabilities : acceptance_; valid_rates(p, maximum_rows - 1);
    const auto remaining = [&](uint32_t m) {
        double cost = expected_cost(m, p, c, probabilities != nullptr);
        if (probabilities && paid_draft_rows) {
            need(paid_draft_rows <= profile_.maximum_rows(), "invalid paid drafting width");
            cost += std::max(0.0, c.draft_ms[m - 1] - c.draft_ms[paid_draft_rows - 1]);
        }
        return cost;
    };
    out.surplus = probabilities || maximum_rows == 1 ? 1 - out.lambda * drift_ * c.ordinary_ms
        : reward(2, p) - out.lambda * drift_ * expected_cost(2, p, c, false);
    out.rows = probabilities || maximum_rows == 1 ? 1u : 2u;
    for (uint32_t m = 2; m <= maximum_rows; ++m) {
        const double s = reward(m, p) - out.lambda * drift_ * remaining(m);
        if (s > out.surplus) { out.rows = m; out.surplus = s; }
    }
    if (maximum_rows == profile_.maximum_rows() && since_probe_ >= probe_interval) {
        out.rows = maximum_rows; out.probe = true;
    }
    return out;
}
MeasuredChoice MeasuredController::before_draft(uint32_t context, uint32_t rows) {
    auto out = choose(context, rows, nullptr);
    const auto c = profile_.costs(context, sampling_);
    const double ordinary = 1 - out.lambda * drift_ * c.ordinary_ms;
    if (!out.probe) {
        const double gain = out.surplus - ordinary;
        if (speculative_ && gain < 0) speculative_ = false;
        else if (!speculative_ && gain > 0.04 * out.lambda * drift_ * c.ordinary_ms) speculative_ = true;
        if (!speculative_) out.rows = 1;
    }
    return out;
}
MeasuredChoice MeasuredController::after_draft(uint32_t context, uint32_t rows, const AcceptanceRates& p, uint32_t paid_draft_rows) {
    return choose(context, rows, &p, paid_draft_rows);
}
MeasuredChoice MeasuredController::after_draft_causal(uint32_t context, uint32_t maximum_rows,
    const AcceptanceRates& probabilities, uint32_t paid_draft_rows) {
    need(generation_ && maximum_rows && maximum_rows <= profile_.maximum_rows() && paid_draft_rows && paid_draft_rows <= profile_.maximum_rows(), "invalid causal measured action");
    valid_rates(probabilities, maximum_rows - 1);
    const auto c = profile_.costs(context, sampling_); condition(c);
    MeasuredChoice out; out.calibrated = c.calibrated; out.lambda = rate(c, out.converged, context);
    calibrated_ = out.calibrated; converged_ = out.converged;
    const auto surplus = [&](uint32_t rows, const AcceptanceRates& p) {
        const double incremental_draft = rows == 1 ? 0 : std::max(0.0, c.draft_ms[rows - 1] - c.draft_ms[paid_draft_rows - 1]);
        return reward(rows, p) - out.lambda * drift_ * (expected_cost(rows, p, c, true) + incremental_draft);
    };
    if (maximum_rows == profile_.maximum_rows() && since_probe_ >= probe_interval) {
        out.rows = maximum_rows; out.probe = true; out.surplus = surplus(maximum_rows, probabilities); return out;
    }
    auto forecast = acceptance_;
    // Confidence i is known before drawing proposal i. Later confidences can reveal discarded proposals and must not decide whether i is included.
    for (uint32_t included = 0; included + 1 < maximum_rows; ++included) {
        forecast[included] = probabilities[included];
        uint32_t best = included + 1; double best_surplus = surplus(best, forecast);
        for (uint32_t rows = best + 1; rows <= maximum_rows; ++rows) {
            const double value = surplus(rows, forecast);
            if (value > best_surplus) { best = rows; best_surplus = value; }
        }
        out.rows = included + 1; out.surplus = surplus(out.rows, forecast);
        if (best == out.rows) return out;
    }
    out.rows = maximum_rows; out.surplus = surplus(maximum_rows, probabilities); return out;
}

void MeasuredController::observe(uint32_t context, uint32_t rows, uint32_t accepted, double milliseconds,
    const AcceptanceRates& probabilities, bool labels, uint32_t drafted_rows, bool stopped) {
    need(generation_ && rows && rows <= profile_.maximum_rows() && accepted < rows,
        "invalid measured controller outcome");
    need(std::isfinite(milliseconds) && milliseconds > 0, "invalid measured decode observation");
    need(packet_size_ < packet_ms_.size(), "measured packet observation bound exceeded");
    valid_rates(probabilities, rows - 1);
    const auto c = profile_.costs(context, sampling_); condition(c); calibrated_ = c.calibrated;
    if (!drafted_rows) drafted_rows = rows;
    need(drafted_rows >= rows && drafted_rows <= profile_.maximum_rows(), "invalid actual drafting width");
    const double predicted = c.draft_ms[drafted_rows - 1] + (rows == 1 ? c.ordinary_ms : c.remaining_ms[rows - 1][accepted]);
    // One common factor preserves the measured width cliffs. No per-width line clamp.
    const double ratio = milliseconds / predicted;
    need(std::isfinite(ratio) && ratio > 0, "invalid measured timing drift");
    drift_ = timing_decay * drift_ + (1 - timing_decay) * ratio;
    ++actions_; last_decode_ms_ = milliseconds;
    packet_ms_[packet_size_++] = milliseconds;
    since_probe_ = rows == profile_.maximum_rows() ? 0 : std::min(since_probe_ + 1, probe_interval);
    if (rows >= 2) {
        auto& feature = pool_[next_pool_];
        for (uint32_t i = 0; i + 1 < profile_.maximum_rows(); ++i)
            feature[i] = i + 1 < rows ? logit(probabilities[i]) - logit(acceptance_[i]) : 0;
        next_pool_ = (next_pool_ + 1) % pool_capacity; pool_size_ = std::min(pool_size_ + 1, pool_capacity);
    }
    if (labels && rows >= 2) for (uint32_t i = 0; i < rows - 1 && (i < accepted || (i == accepted && !stopped)); ++i)
        acceptance_[i] = (1 - acceptance_beta) * acceptance_[i] + acceptance_beta * double(i < accepted);
}

} // namespace tokspur
