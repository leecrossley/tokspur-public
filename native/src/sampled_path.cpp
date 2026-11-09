#include "tokspur/carrier_phase.hpp"
#include "tokspur/sampled_path.hpp"
#include "tokspur/sampled_path_detail.hpp"
#include "tokspur/prepared_sampling.hpp"

#include <algorithm>
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
#include <bit>
#endif
#include <cmath>
#include <limits>
#include <utility>
#if defined(TOKSPUR_PRIVATE_M5_CPU_CANDIDATES) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tokspur {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw FormatError(message);
}

// Same compensated positive sum as sampling.cpp's inverse CDF. This is existing
// Tokspur arithmetic, not imported upstream rejection-sampler code.
class Cumulative {
    double sum_ = 0, correction_ = 0;
public:
    void add(double value) {
        const double next = sum_ + value;
        correction_ += sum_ >= value ? (sum_ - next) + value : (value - next) + sum_;
        sum_ = next;
    }
    [[nodiscard]] double value() const { return sum_ + correction_; }
};

uint64_t cdf_end(double cumulative) {
    require(std::isfinite(cumulative) && cumulative >= 0, "sampled path invalid cumulative mass");
    if (cumulative >= 1) return sampled_path_grid_size;
    // u=j*2^-53 is below c precisely for j < ceil(c*2^53).
    return uint64_t(std::ceil(std::ldexp(cumulative, 53)));
}

bool stop(uint32_t id, std::span<const uint32_t> ids) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

template<class Next, class Poll>
uint32_t draw_mass(std::span<const uint64_t> masses, Next&& next, Poll&& poll) {
    uint64_t total = 0;
    uint32_t positives = 0, last = 0;
    for (uint32_t id = 0; id < masses.size(); ++id) {
        require(masses[id] <= sampled_path_grid_size - total, "sampled path mass overflow");
        total += masses[id];
        if (masses[id]) { ++positives; last = id; }
    }
    require(total && positives, "sampled path collapsed residual; no fallback");
    if (positives == 1) return last;
    const uint64_t ticket = sampled_path_detail::uniform_below(total, next, poll);
    uint64_t cumulative = 0;
    for (uint32_t id = 0; id < masses.size(); ++id) {
        cumulative += masses[id];
        if (ticket < cumulative) return id;
    }
    throw FormatError("sampled path integer CDF exhausted; no fallback");
}

} // namespace

SampledSparseLaw SampledSparseLaw::from_masses(uint32_t vocabulary, std::span<const uint32_t> ids,
    std::span<const double> probabilities, SampledPathGrid grid) {
    require(grid == SampledPathGrid::native_u53_inverse_cdf_v1, "sampled path unsupported proposal grid");
    require(vocabulary && vocabulary <= maximum_sampling_vocabulary && !ids.empty()
        && ids.size() <= sampled_path_maximum_support && ids.size() == probabilities.size(),
        "sampled path sparse law shape");
    Cumulative total;
    uint32_t last_positive = 0;
    for (uint32_t i = 0; i < ids.size(); ++i) {
        require(ids[i] < vocabulary && std::find(ids.begin(), ids.begin() + i, ids[i]) == ids.begin() + i,
            "sampled path duplicate or out-of-range proposal ID");
        const double value = probabilities[i];
        require(std::isfinite(value) && value >= 0 && value <= 1, "sampled path nonfinite/negative proposal mass");
        total.add(value);
        if (value > 0) last_positive = i;
    }
    require(std::abs(total.value() - 1) <= 32 * std::numeric_limits<double>::epsilon(),
        "sampled path proposal masses are not normalized");
    SampledSparseLaw out;
    out.vocabulary_ = vocabulary;
    Cumulative cumulative;
    uint64_t previous = 0;
    for (uint32_t i = 0; i < ids.size(); ++i) {
        if (!probabilities[i]) continue;
        cumulative.add(probabilities[i]);
        const uint64_t end = i == last_positive ? sampled_path_grid_size : cdf_end(cumulative.value());
        require(end >= previous && end <= sampled_path_grid_size, "sampled path nonmonotonic proposal CDF");
        out.ids_[out.size_] = ids[i]; out.units_[out.size_] = end - previous;
        ++out.size_; previous = end;
    }
    require(out.size_ && previous == sampled_path_grid_size, "sampled path empty proposal law");
    return out;
}

SampledSparseLaw SampledSparseLaw::from_grid_units(uint32_t vocabulary, std::span<const uint32_t> ids,
    std::span<const uint64_t> grid_units, SampledPathGrid grid) {
    require(grid == SampledPathGrid::native_u53_inverse_cdf_v1, "sampled path unsupported proposal grid");
    require(vocabulary && vocabulary <= maximum_sampling_vocabulary && !ids.empty()
        && ids.size() <= sampled_path_maximum_support && ids.size() == grid_units.size(),
        "sampled path sparse law shape");
    uint64_t total = 0;
    for (uint32_t i = 0; i < ids.size(); ++i) {
        require(ids[i] < vocabulary && std::find(ids.begin(), ids.begin() + i, ids[i]) == ids.begin() + i,
            "sampled path duplicate or out-of-range proposal ID");
        require(grid_units[i] <= sampled_path_grid_size - total, "sampled path proposal mass overflow");
        total += grid_units[i];
    }
    require(total == sampled_path_grid_size, "sampled path proposal grid units do not sum to grid size");
    SampledSparseLaw out;
    out.vocabulary_ = vocabulary;
    for (uint32_t i = 0; i < ids.size(); ++i) {
        if (!grid_units[i]) continue;
        out.ids_[out.size_] = ids[i];
        out.units_[out.size_] = grid_units[i];
        ++out.size_;
    }
    require(out.size_, "sampled path empty proposal law");
    return out;
}


uint64_t SampledSparseLaw::units(uint32_t token) const noexcept {
    for (uint32_t i = 0; i < size_; ++i) if (ids_[i] == token) return units_[i];
    return 0;
}

uint32_t SampledSparseLaw::sample_at(uint64_t word) const {
    require(size_ && word < sampled_path_grid_size, "sampled path invalid proposal grid draw");
    uint64_t cumulative = 0;
    for (uint32_t i = 0; i < size_; ++i) {
        cumulative += units_[i];
        if (word < cumulative) return ids_[i];
    }
    throw FormatError("sampled path proposal CDF exhausted");
}

bool SampledSparseLaw::deterministic() const noexcept {
    uint32_t positive = 0;
    for (uint32_t i = 0; i < size_; ++i) positive += units_[i] != 0;
    return positive == 1;
}

namespace sampled_path_detail {

std::vector<uint64_t> target_grid_units(const SamplingDistribution& distribution) {
    require(distribution.vocabulary() && distribution.vocabulary() <= maximum_sampling_vocabulary
        && !distribution.support().empty(), "sampled path moved-from/invalid target distribution");
    std::vector<uint64_t> masses(distribution.vocabulary(), 0);
    Cumulative cumulative;
    uint64_t previous = 0;
    const auto support = distribution.support();
    for (size_t i = 0; i < support.size(); ++i) {
        const auto id = support[i];
        require(id < masses.size() && distribution.probability(id) > 0, "sampled path target support");
        cumulative.add(distribution.probability(id));
        const uint64_t end = i + 1 == support.size() ? sampled_path_grid_size : cdf_end(cumulative.value());
        require(end >= previous && end <= sampled_path_grid_size, "sampled path nonmonotonic target CDF");
        masses[id] = end - previous; previous = end;
    }
    require(previous == sampled_path_grid_size, "sampled path target grid did not close");
    return masses;
}

std::vector<SparseTokenMass> target_sparse_grid_units(const SamplingDistribution& distribution) {
    TOKSPUR_CARRIER_PHASE(sampling);
    require(distribution.vocabulary() && distribution.vocabulary() <= maximum_sampling_vocabulary
        && !distribution.support().empty(), "sampled path moved-from/invalid target distribution");
    const auto support = distribution.support();
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    const auto probs = distribution.support_probabilities();
    require(probs.size() == support.size(), "sampled path target support probabilities");
#endif
    std::vector<SparseTokenMass> masses;
    const bool dense = prepared_sampling() && support.size() == distribution.vocabulary();
    if (dense) masses.resize(support.size());
    else masses.reserve(support.size());
    Cumulative cumulative;
    uint64_t previous = 0;
    for (size_t i = 0; i < support.size(); ++i) {
        const auto id = support[i];
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
        const double prob = probs[i];
        require(id < distribution.vocabulary() && prob > 0, "sampled path target support");
        cumulative.add(prob);
#else
        require(id < distribution.vocabulary() && distribution.probability(id) > 0, "sampled path target support");
        cumulative.add(distribution.probability(id));
#endif
        const uint64_t end = i + 1 == support.size() ? sampled_path_grid_size : cdf_end(cumulative.value());
        require(end >= previous && end <= sampled_path_grid_size, "sampled path nonmonotonic target CDF");
        // Keep the CDF arithmetic in its original support order. A complete
        // support contains every token once, so place each resulting integer
        // mass directly in the final token-ID order instead of sorting it.
        if (dense) masses[id] = {id, end - previous};
        else masses.push_back({id, end - previous});
        previous = end;
    }
    require(previous == sampled_path_grid_size, "sampled path target grid did not close");
    if (!dense) std::sort(masses.begin(), masses.end(), [](const SparseTokenMass& a, const SparseTokenMass& b) {
        return a.token_id < b.token_id;
    });
    return masses;
}

float lookup_topk_logit(const GpuLogitTopKRow& row, uint32_t id) {
    const uint32_t limit = std::min<uint32_t>(gpu_logit_topk_capacity, row.count);
    for (uint32_t k = 0; k < limit; ++k) {
        if (row.ids[k] == id) return row.logits[k];
    }
    throw FormatError("sampled path emitted token missing from GPU top-k support");
}

template <typename TargetRowSpan, typename ValidateRowFn, typename BuildDistFn, typename SourceLogitFn>
SampledPathDecision decide_sampled_path_core(
    SampledPathIdentity proposals_identity, std::span<const SampledProposalRow> proposals,
    SampledPathIdentity target_identity, TargetRowSpan target_rows, SamplingPolicy policy,
    uint32_t remaining, std::span<const uint32_t> stops, const SamplingRng& after_proposals,
    uint64_t proposal_begin_words, SampledPathCancellation cancellation,
    ValidateRowFn&& validate_row, BuildDistFn&& build_dist, SourceLogitFn&& source_logit) {
    const auto poll = [&] {
        require(!cancellation.requested || !cancellation.requested(cancellation.context),
            "sampled path cancelled; no publication");
    };
    poll();
    const auto identity = proposals_identity;
    require(identity == target_identity && identity.owner && identity.epoch && identity.policy_epoch
        && identity.prefix && identity.vocabulary && identity.vocabulary <= maximum_sampling_vocabulary
        && identity.last_token < identity.vocabulary, "sampled path stale/mismatched transaction identity");
    require(!proposals.empty() && proposals.size() <= sampled_path_maximum_proposals
        && target_rows.size() == proposals.size() + 1 && remaining
        && identity.prefix <= std::numeric_limits<uint32_t>::max() - target_rows.size()
        && proposal_begin_words <= after_proposals.words_consumed(), "sampled path bounded round shape");
    require(stops.size() <= sampled_path_maximum_stops, "sampled path stop set bound");
    for (size_t i = 0; i < stops.size(); ++i)
        require(stops[i] < identity.vocabulary && std::find(stops.begin(), stops.begin() + ptrdiff_t(i), stops[i]) == stops.begin() + ptrdiff_t(i),
            "sampled path duplicate/out-of-range stop ID");
    uint64_t proposal_words = 0;
    for (size_t i = 0; i < proposals.size(); ++i) {
        poll(); const auto& row = proposals[i];
        require(row.position == identity.prefix + i && row.predecessor == (i ? proposals[i-1].selected : identity.last_token)
            && row.actual_q.vocabulary() == identity.vocabulary && row.actual_q.size()
            && row.selected < identity.vocabulary && row.actual_q.units(row.selected),
            "sampled path proposal conditional identity or zero selected mass");
        proposal_words += !row.actual_q.deterministic();
    }
    require(after_proposals.words_consumed() - proposal_begin_words == proposal_words,
        "sampled path actual proposal RNG word accounting");
    for (size_t i = 0; i < target_rows.size(); ++i) {
        poll();
        validate_row(i, target_rows[i], identity, poll);
    }
    SampledPathDecision out(after_proposals);
    out.identity = identity; out.target_policy = policy;
    out.proposal_words = after_proposals.words_consumed() - proposal_begin_words;
    out.active_proposals = uint32_t(std::min<size_t>(proposals.size(), remaining - 1));
    const auto next = [&] { return uint64_t(std::ldexp(out.random_after.uniform(), 53)); };
    const auto emit = [&](uint32_t id, uint32_t row, bool accepted) {
        require(out.count < out.tokens.size() && out.count < remaining, "sampled path output budget overflow");
        out.tokens[out.count++] = {id, source_logit(target_rows[row], id), accepted};
        out.stopped = stop(id, stops); out.output_budget_exhausted = out.count == remaining;
    };
#ifdef TOKSPUR_PRIVATE_SPARSE_RESIDUAL
    for (uint32_t i = 0;; ++i) {
        poll();
        auto masses = target_sparse_grid_units(build_dist(target_rows[i], policy));
        poll();
        const auto lookup_p = [&](uint32_t token) -> uint64_t {
            auto it = std::lower_bound(masses.begin(), masses.end(), token,
                [](const SparseTokenMass& item, uint32_t val) {
                    return item.token_id < val;
                });
            if (it != masses.end() && it->token_id == token) return it->mass;
            return 0;
        };
        if (i == out.active_proposals) {
            const auto before = out.random_after.words_consumed();
            const uint32_t id = draw_mass_sparse(masses, next, poll);
            out.terminal_words += out.random_after.words_consumed() - before;
            emit(id, i, false); out.terminal = SampledPathTerminal::target_bonus;
            break;
        }
        const auto& proposal = proposals[i];
        const uint64_t p = lookup_p(proposal.selected), q = proposal.actual_q.units(proposal.selected);
        bool accepted = p >= q;
        if (p && p < q) {
            const auto before = out.random_after.words_consumed();
            accepted = uniform_below(q, next, poll) < p;
            out.acceptance_words += out.random_after.words_consumed() - before;
        }
        if (accepted) {
            emit(proposal.selected, i, true); ++out.accepted;
            if (out.stopped) { out.terminal = SampledPathTerminal::accepted_stop; break; }
            continue;
        }
        const auto ids = proposal.actual_q.identifiers();
        const auto units = proposal.actual_q.mass_units();
        for (size_t bin = 0; bin < ids.size(); ++bin) {
            const auto id = ids[bin];
            const auto q_mass = units[bin];
            auto it = std::lower_bound(masses.begin(), masses.end(), id,
                [](const SparseTokenMass& item, uint32_t val) {
                    return item.token_id < val;
                });
            if (it != masses.end() && it->token_id == id) {
                it->mass = it->mass > q_mass ? it->mass - q_mass : 0;
            }
        }
        const auto before = out.random_after.words_consumed();
        const uint32_t id = draw_mass_sparse(masses, next, poll);
        out.terminal_words += out.random_after.words_consumed() - before;
        emit(id, i, false); out.terminal = SampledPathTerminal::correction;
        break;
    }
#else
    for (uint32_t i = 0;; ++i) {
        poll();
        auto masses = target_grid_units(build_dist(target_rows[i], policy));
        poll();
        if (i == out.active_proposals) {
            const auto before = out.random_after.words_consumed();
            const uint32_t id = draw_mass(masses, next, poll);
            out.terminal_words += out.random_after.words_consumed() - before;
            emit(id, i, false); out.terminal = SampledPathTerminal::target_bonus;
            break;
        }
        const auto& proposal = proposals[i];
        const uint64_t p = masses[proposal.selected], q = proposal.actual_q.units(proposal.selected);
        bool accepted = p >= q;
        if (p && p < q) {
            const auto before = out.random_after.words_consumed();
            accepted = uniform_below(q, next, poll) < p;
            out.acceptance_words += out.random_after.words_consumed() - before;
        }
        if (accepted) {
            emit(proposal.selected, i, true); ++out.accepted;
            if (out.stopped) { out.terminal = SampledPathTerminal::accepted_stop; break; }
            continue;
        }
        for (uint32_t id = 0; id < identity.vocabulary; ++id) {
            const auto q_mass = proposal.actual_q.units(id);
            masses[id] = masses[id] > q_mass ? masses[id] - q_mass : 0;
        }
        const auto before = out.random_after.words_consumed();
        const uint32_t id = draw_mass(masses, next, poll);
        out.terminal_words += out.random_after.words_consumed() - before;
        emit(id, i, false); out.terminal = SampledPathTerminal::correction;
        break;
    }
#endif
    poll();
    require(out.count && out.random_after.words_consumed() - after_proposals.words_consumed()
        == out.acceptance_words + out.terminal_words, "sampled path entropy accounting");
    return out;
}

} // namespace sampled_path_detail

SampledPathDecision decide_sampled_path(SampledPathIdentity proposals_identity,
    std::span<const SampledProposalRow> proposals, SampledPathIdentity target_identity,
    std::span<const SampledTargetRow> target_rows, SamplingPolicy policy, uint32_t remaining,
    std::span<const uint32_t> stops, const SamplingRng& after_proposals, uint64_t proposal_begin_words,
    SampledPathCancellation cancellation) {
    std::array<SamplingRowScan, 8> scans;
    const bool prepared = prepared_sampling();
    return sampled_path_detail::decide_sampled_path_core(
        proposals_identity, proposals, target_identity, target_rows, policy, remaining,
        stops, after_proposals, proposal_begin_words, cancellation,
        [&](size_t i, const SampledTargetRow& row, const SampledPathIdentity& identity, const auto& poll) {
            require(row.position == identity.prefix + i && row.logits.size() == identity.vocabulary,
                "sampled path complete aligned target rows");
            TOKSPUR_CARRIER_PHASE(numerical);
            if (prepared) { scans[i] = SamplingRowScan::inspect(row.logits, poll); return; }
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
            for (size_t base = 0; base < row.logits.size(); base += 4096) {
                poll();
                const size_t end = std::min(base + 4096, row.logits.size());
                uint32_t bad = 0;
                size_t at = base;
#if defined(__aarch64__)
                uint32x4_t bad_vec = vdupq_n_u32(0u);
                const uint32x4_t exp_mask = vdupq_n_u32(0x7f800000u);
                for (; at + 16 <= end; at += 16) {
                    const float* fp = row.logits.data() + at;
                    const uint32x4_t c0 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp)), exp_mask), exp_mask);
                    const uint32x4_t c1 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp + 4)), exp_mask), exp_mask);
                    const uint32x4_t c2 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp + 8)), exp_mask), exp_mask);
                    const uint32x4_t c3 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp + 12)), exp_mask), exp_mask);
                    bad_vec = vorrq_u32(bad_vec, vorrq_u32(vorrq_u32(c0, c1), vorrq_u32(c2, c3)));
                }
                bad |= vmaxvq_u32(bad_vec);
#endif
                for (; at < end; ++at) {
                    bad |= uint32_t((std::bit_cast<uint32_t>(row.logits[at]) & 0x7f800000u) == 0x7f800000u);
                }
                require(!bad, "sampled path nonfinite target logit");
            }
#else
            for (size_t at = 0; at < row.logits.size(); ++at) {
                if (!(at & 4095)) poll();
                require(std::isfinite(row.logits[at]), "sampled path nonfinite target logit");
            }
#endif
        },
        [&](const SampledTargetRow& row, SamplingPolicy p) {
            if (prepared) return sampling_distribution_after_scan(row.logits, p, scans[&row - target_rows.data()]);
            return sampling_distribution(row.logits, p);
        },
        [](const SampledTargetRow& row, uint32_t id) {
            return row.logits[id];
        });
}

SampledPathDecision decide_sampled_path_topk(SampledPathIdentity proposals_identity,
    std::span<const SampledProposalRow> proposals, SampledPathIdentity target_identity,
    std::span<const SampledTargetTopKRow> target_rows, SamplingPolicy policy, uint32_t remaining,
    std::span<const uint32_t> stops, const SamplingRng& after_proposals, uint64_t proposal_begin_words,
    SampledPathCancellation cancellation) {
    return sampled_path_detail::decide_sampled_path_core(
        proposals_identity, proposals, target_identity, target_rows, policy, remaining,
        stops, after_proposals, proposal_begin_words, cancellation,
        [](size_t i, const SampledTargetTopKRow& row, const SampledPathIdentity& identity, const auto&) {
            require(row.position == identity.prefix + i && row.topk.vocabulary == identity.vocabulary,
                "sampled path complete aligned target rows");
            require(row.topk.first_nonfinite_col == 0xffffffffu || row.topk.first_nonfinite_col < identity.vocabulary,
                "sampled path corrupted GPU top-k nonfinite index");
            require(row.topk.first_nonfinite_col == 0xffffffffu, "sampled path nonfinite target logit");
        },
        [](const SampledTargetTopKRow& row, SamplingPolicy p) {
            return sampling_distribution_from_topk(row.topk, p);
        },
        [](const SampledTargetTopKRow& row, uint32_t id) {
            return sampled_path_detail::lookup_topk_logit(row.topk, id);
        });
}

} // namespace tokspur
