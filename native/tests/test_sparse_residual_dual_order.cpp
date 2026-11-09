#include "tokspur/sampled_path.hpp"
#include "tokspur/sampled_path_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {
using tokspur::sampled_path_grid_size;
using tokspur::sampled_path_detail::SparseTokenMass;
using tokspur::sampled_path_detail::draw_mass_sparse;

template<class Next, class Poll>
uint32_t reference_draw_mass(std::span<const uint64_t> masses, Next&& next, Poll&& poll) {
    uint64_t total = 0;
    uint32_t positives = 0, last = 0;
    for (uint32_t id = 0; id < masses.size(); ++id) {
        total += masses[id];
        if (masses[id]) { ++positives; last = id; }
    }
    if (positives == 1) return last;
    const uint64_t ticket = tokspur::sampled_path_detail::uniform_below(total, next, poll);
    uint64_t cumulative = 0;
    for (uint32_t id = 0; id < masses.size(); ++id) {
        cumulative += masses[id];
        if (ticket < cumulative) return id;
    }
    throw std::runtime_error("dense CDF exhausted");
}

} // namespace

int main() {
    constexpr uint32_t trials = 10000;
    constexpr uint32_t vocab = 248320;
    std::mt19937_64 rng(0x535041525345ULL);

    std::cout << "Running 10,000 paired trials comparing dense draw_mass vs draw_mass_sparse...\n";

    for (uint32_t trial = 0; trial < trials; ++trial) {
        const uint32_t num_active = 1 + (rng() % 64);
        std::vector<uint32_t> active_ids(num_active);
        for (uint32_t i = 0; i < num_active; ++i) {
            active_ids[i] = uint32_t(rng() % vocab);
        }
        std::sort(active_ids.begin(), active_ids.end());
        active_ids.erase(std::unique(active_ids.begin(), active_ids.end()), active_ids.end());

        std::vector<uint64_t> dense_masses(vocab, 0);
        std::vector<SparseTokenMass> sparse_masses;
        sparse_masses.reserve(active_ids.size());

        uint64_t remaining_mass = sampled_path_grid_size;
        for (size_t i = 0; i < active_ids.size(); ++i) {
            uint64_t m = (i + 1 == active_ids.size()) ? remaining_mass : (rng() % (remaining_mass / (active_ids.size() - i) + 1));
            if (m == 0 && remaining_mass > 0) m = 1;
            remaining_mass -= m;
            dense_masses[active_ids[i]] = m;
            sparse_masses.push_back({active_ids[i], m});
        }

        tokspur::SamplingRng rng_dense(trial + 1);
        tokspur::SamplingRng rng_sparse(trial + 1);

        const auto poll = [] {};
        const auto next_dense = [&] { return uint64_t(std::ldexp(rng_dense.uniform(), 53)); };
        const auto next_sparse = [&] { return uint64_t(std::ldexp(rng_sparse.uniform(), 53)); };

        const uint32_t dense_token = reference_draw_mass(dense_masses, next_dense, poll);
        const uint32_t sparse_token = draw_mass_sparse(sparse_masses, next_sparse, poll);

        if (dense_token != sparse_token) {
            std::cerr << "Mismatch at trial " << trial << ": dense=" << dense_token << " sparse=" << sparse_token << '\n';
            return 1;
        }
        if (rng_dense.words_consumed() != rng_sparse.words_consumed()) {
            std::cerr << "Entropy word mismatch at trial " << trial << ": dense=" << rng_dense.words_consumed()
                      << " sparse=" << rng_sparse.words_consumed() << '\n';
            return 1;
        }
    }

    std::cout << "All 10,000 paired sparse sampling trials passed with 100% bit-exact equivalence and 0 mismatches.\n";
    return 0;
}
