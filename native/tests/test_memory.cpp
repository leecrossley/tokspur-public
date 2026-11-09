#include "tokspur/memory.hpp"
#include "tokspur/memory_guard.hpp"
#include "tokspur/operators.hpp"
#include "tokspur/ple.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <random>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
size_t checks = 0;
void require(bool b, const char* m) { ++checks; if (!b) throw std::runtime_error(m); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const tokspur::FormatError&) { return; } throw std::runtime_error("invalid memory operation accepted"); }
}
int main() {
    try {
        using namespace tokspur;
        const NgramSpec spec{3, 2, 4, 20, 16, 15, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
        auto history = empty_ngram_history(spec); history.previous[0] = 9; history.previous[1] = 8;
        const auto before = history;
        const auto step = ngram_step(spec, {HashOrder::newest_first, NgramBoundary::literal, false}, history, 10);
        require(step.rows == std::vector<uint64_t>({2, 14, 29, 42}), "hand-calculated multiplicative-XOR rows differ");
        require(history.previous == before.previous && step.next.previous[0] == 10 && step.next.previous[1] == 9, "n-gram current/next ownership");
        for (auto order : {HashOrder::newest_first, HashOrder::oldest_first}) for (auto boundary : {NgramBoundary::literal, NgramBoundary::reset_before, NgramBoundary::reset_after}) {
            auto h = empty_ngram_history(spec);
            for (uint32_t t : {1, 2, 3, 16, 4, 5, 15, 6}) {
                const auto result = ngram_step(spec, {order, boundary, true}, h, t);
                std::array<uint64_t, 4> prepared_rows{};
                const auto prepared_next = ngram_step(spec, {order, boundary, true}, h, t, prepared_rows);
                require(std::equal(prepared_rows.begin(), prepared_rows.end(), result.rows.begin()) &&
                    prepared_next.previous == result.next.previous, "allocation-free ngram rows/history differ");
                for (size_t i = 0; i < result.rows.size(); ++i) require(result.rows[i] >= spec.offsets[i] && result.rows[i] - spec.offsets[i] < spec.vocabularies[i], "hash crossed a partition");
                if ((t == 16 || t == 15) && boundary != NgramBoundary::literal) require(result.next.previous == empty_ngram_history(spec).previous, "boundary did not clear history");
                h = result.next;
            }
        }
        reject([&] { static_cast<void>(ngram_step(spec, {HashOrder(2), NgramBoundary::literal, false}, history, 1)); });
        reject([&] { static_cast<void>(ngram_step(spec, {HashOrder::newest_first, NgramBoundary::literal, false}, history, 20)); });
        auto invalid = spec; invalid.offsets[1] = 8;
        reject([&] { validate_ngram(invalid, {HashOrder::newest_first, NgramBoundary::literal, false}); });
        std::array<uint64_t, 4> untouched{101, 102, 103, 104}; const auto sentinels = untouched;
        const NgramPolicy literal{HashOrder::newest_first, NgramBoundary::literal, false};
        reject([&] { static_cast<void>(ngram_step(spec, literal, history, 1, std::span(untouched).first(3))); });
        reject([&] { static_cast<void>(ngram_step(spec, literal, history, 20, untouched)); });
        reject([&] { static_cast<void>(ngram_step(invalid, literal, history, 1, untouched)); });
        auto bad_history = history; bad_history.previous[1] = 20;
        reject([&] { static_cast<void>(ngram_step(spec, literal, bad_history, 1, untouched)); });
        reject([&] { static_cast<void>(ngram_step(spec, {HashOrder(2), NgramBoundary::literal, false}, history, 1, untouched)); });
        require(untouched == sentinels && history.previous == before.previous, "refused prepared ngram changed rows/history");
        constexpr uint32_t width = 8, logical = 257, stored = 264;
        std::vector<float> data(stored * width);
        for (uint32_t r = 0; r < stored; ++r) for (uint32_t c = 0; c < width; ++c) data[r * width + c] = float(r * 16 + c) / 32;
        data[256 * width] = std::numeric_limits<float>::infinity();
        EmbeddingRows cache(std::as_bytes(std::span(data)), 0, width, stored, logical, 1024);
        require(cache.capacity() > 1 && cache.capacity() < logical && cache.charged_bytes() <= 1024, "cache budget/capacity");
        std::list<uint64_t> oracle; RowCacheStats stats; std::mt19937 rng(0x6c727531); std::array<float, width> output{};
        for (size_t i = 0; i < 50000; ++i) {
            const uint64_t row = i < 64 ? (i % 8) * 32 : rng() % 256;
            const auto found = std::find(oracle.begin(), oracle.end(), row);
            if (found == oracle.end()) { ++stats.misses; if (oracle.size() == cache.capacity()) { oracle.pop_back(); ++stats.evictions; } }
            else { ++stats.hits; oracle.erase(found); }
            oracle.push_front(row); cache.copy(row, output);
            for (uint32_t c = 0; c < width; ++c) require(output[c] == data[row * width + c], "LRU/hash-index row content mismatch");
            const auto actual = cache.stats();
            require(actual.hits == stats.hits && actual.misses == stats.misses && actual.evictions == stats.evictions, "cache policy differs from independent list oracle");
        }
        const auto previous = output;
        reject([&] { cache.copy(logical, output); }); require(previous == output, "invalid row changed destination");
        reject([&] { cache.copy(256, output); }); require(previous == output && cache.stats().failures == 1, "bad source row changed destination or wasn't tracked");
        cache.copy(oracle.front(), output); require(cache.stats().hits == stats.hits + 1, "failed source decode evicted a valid row");
        bool thread_rejected = false;
        std::thread thread([&] { try { static_cast<void>(cache.stats()); } catch (const FormatError&) { thread_rejected = true; } });
        thread.join(); require(thread_rejected, "cross-thread cache access accepted");
        reject([&] { EmbeddingRows bad(std::as_bytes(std::span(data)), 0, width, stored, logical, 1); });
        {
            const auto tmp_path = std::filesystem::temp_directory_path() / ("tokspur_test_memory_pread_" + std::to_string(::getpid()) + ".bin");
            struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove(p, ec); } } cleanup{tmp_path};
            constexpr uint64_t prefix_pad = 192;
            const auto bytes_span = std::as_bytes(std::span(data));
            {
                std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                std::array<char, prefix_pad> hdr{};
                out.write(hdr.data(), hdr.size());
                out.write(reinterpret_cast<const char*>(bytes_span.data()), static_cast<std::streamsize>(bytes_span.size()));
                require(bool(out), "temporary pread fixture write failed");
            }
            struct stat st{};
            require(::stat(tmp_path.c_str(), &st) == 0, "temporary pread fixture stat failed");
            reject([&] { NocacheTensorFile bad_ino(tmp_path, prefix_pad, bytes_span.size(), uint64_t(st.st_dev), uint64_t(st.st_ino) + 1); });
            NocacheTensorFile ncf(tmp_path, prefix_pad, bytes_span.size(), uint64_t(st.st_dev), uint64_t(st.st_ino));
            require(ncf.valid(), "nocache tensor file should be valid");
            std::vector<uint64_t> batch_ids(640);
            for (size_t i = 0; i < batch_ids.size(); ++i) batch_ids[i] = uint64_t((i * 19 + (i % 7)) % 256);
            std::vector<float> ref_dedup(batch_ids.size() * width), got_dedup(batch_ids.size() * width);
            PleGatherScratch s1, s2;
            dequantize_rows_sorted_dedup(bytes_span, 0, width, stored, logical, batch_ids, ref_dedup, s1, 4u);
            dequantize_rows_sorted_dedup(bytes_span, 0, width, stored, logical, batch_ids, got_dedup, s2, 4u, ncf.source());
            require(ref_dedup == got_dedup, "pread stateless deduplicated gather differs from in-memory reference");

            // Duplicate runs cross several nominal worker partitions and must never share output slots between workers.
            for (const uint32_t worker_count : {4u, 12u, 16u}) {
                std::vector<uint64_t> repeated_ids(32768, 7);
                for (const bool mixed : {false, true}) {
                    if (mixed) for (size_t i = 24000; i < repeated_ids.size(); ++i) repeated_ids[i] = 8 + (i % 3);
                    std::vector<float> serial_rows(repeated_ids.size() * width), parallel_rows(serial_rows.size());
                    PleGatherScratch serial_scratch, parallel_scratch;
                    dequantize_rows_sorted_dedup(bytes_span, 0, width, stored, logical, repeated_ids, serial_rows, serial_scratch);
                    for (const PreadSource source : {PreadSource{}, ncf.source()}) {
                        std::fill(parallel_rows.begin(), parallel_rows.end(), -1.0f);
                        dequantize_rows_sorted_dedup(bytes_span, 0, width, stored, logical, repeated_ids, parallel_rows, parallel_scratch, worker_count, source);
                        require(parallel_rows == serial_rows, "duplicate-heavy parallel gather differs from serial rows");
                    }
                }
            }

            // A row cached at entry can be evicted before its later occurrence in the same gather.
            for (const uint64_t warm_row : {2ull, 255ull}) {
                EmbeddingRows eviction_cache(bytes_span, 0, width, stored, logical, 1024, ncf.source(), true);
                EmbeddingRows eviction_oracle(bytes_span, 0, width, stored, logical, 1024);
                std::array<float, width> warm_output{};
                eviction_cache.copy(warm_row, warm_output);
                eviction_oracle.copy(warm_row, warm_output);
                std::vector<uint64_t> eviction_ids;
                for (uint64_t row = 0; eviction_ids.size() < eviction_cache.capacity() + 4ull; ++row) {
                    if (row != warm_row) eviction_ids.push_back(row);
                }
                eviction_ids.push_back(warm_row);
                eviction_ids.push_back(warm_row);
                require(eviction_ids.size() <= 256 && eviction_ids[eviction_ids.size() - 3] < 255, "eviction fixture exceeds gather geometry");
                std::vector<float> eviction_got(eviction_ids.size() * width), eviction_expected(eviction_got.size());
                eviction_cache.gather(eviction_ids, eviction_got);
                eviction_oracle.gather(eviction_ids, eviction_expected);
                require(eviction_got == eviction_expected, "gather lost a cached row after earlier miss insertion evicted it");
                const auto actual = eviction_cache.stats(), expected = eviction_oracle.stats();
                require(actual.hits == expected.hits && actual.misses == expected.misses && actual.evictions == expected.evictions && actual.failures == expected.failures, "eviction gather cache statistics differ from serial oracle");
                require(eviction_cache.charged_bytes() <= 1024, "eviction gather exceeded its row cache budget");
            }

            EmbeddingRows pread_cache(bytes_span, 0, width, stored, logical, 2048, ncf.source(), true);
            EmbeddingRows default_pread_cache(bytes_span, 0, width, stored, logical, 2048, ncf.source());
            EmbeddingRows mem_cache(bytes_span, 0, width, stored, logical, 2048);
            std::array<uint64_t, 64> gather_ids{};
            for (size_t i = 0; i < gather_ids.size(); ++i) gather_ids[i] = uint64_t((i * 13 + 5) % 256);
            std::vector<float> got_gather(gather_ids.size() * width), ref_gather(gather_ids.size() * width), default_gather(gather_ids.size() * width);
            pread_cache.gather(gather_ids, got_gather);
            default_pread_cache.gather(gather_ids, default_gather);
            mem_cache.gather(gather_ids, ref_gather);
            require(got_gather == ref_gather, "pread MissReaderPool gather differs from in-memory reference");
            require(default_gather == ref_gather, "default pread gather differs from serial in-memory reference");
            const auto ps = pread_cache.stats(), ms = mem_cache.stats();
            require(ps.hits == ms.hits && ps.misses == ms.misses && ps.evictions == ms.evictions, "pread cache stats differ");
            const auto ds = default_pread_cache.stats();
            require(ds.hits == ms.hits && ds.misses == ms.misses && ds.evictions == ms.evictions && ds.failures == ms.failures, "default pread cache stats differ");
            require(pread_cache.charged_bytes() <= 2048 && default_pread_cache.charged_bytes() <= 2048 && mem_cache.charged_bytes() <= 2048, "gather mode exceeded its cache budget");

            EmbeddingRows seeded_cache(bytes_span, 0, width, stored, logical, 4096, ncf.source());
            seeded_cache.seed_from_dedup(s2, got_dedup);
            const uint64_t mru_seeded_row = s2.pairs.back().first;
            std::array<float, width> probe_row{}, single_row{};
            require(seeded_cache.try_copy(mru_seeded_row, probe_row), "seeded dedup row should hit in cache");
            dequantize_row_at(bytes_span, 0, width, stored, logical, mru_seeded_row, single_row, ncf.source());
            require(probe_row == single_row, "try_copy / dequantize_row_at mismatch");
            EmbeddingRows insert_cache(bytes_span, 0, width, stored, logical, 2048, ncf.source());
            require(!insert_cache.try_copy(7, probe_row), "cold cache should report miss via try_copy");
            dequantize_row_at(bytes_span, 0, width, stored, logical, 7, single_row, ncf.source());
            insert_cache.insert_decoded(7, single_row, true);
            require(insert_cache.try_copy(7, probe_row) && probe_row == single_row, "insert_decoded row mismatch");
            require(insert_cache.stats().misses == 1 && insert_cache.stats().hits == 1, "insert_decoded / try_copy stats mismatch");
        }
        {
            EmbeddingRows reclaimed(std::as_bytes(std::span(data)), 0, width, stored, logical, 1024, {}, true);
            reclaimed.copy(4, output);
            const auto before_bytes = reclaimed.charged_bytes();
            const auto freed = reclaimed.reclaim();
            require(freed > 0 && reclaimed.charged_bytes() + freed == before_bytes && reclaimed.capacity() == 0, "reclaim retained row slabs or charged capacity");
            require(reclaimed.reclaim() == 0, "repeated reclaim is not idempotent");
            require(!reclaimed.try_copy(4, output), "reclaimed row remained indexed");
            reclaimed.copy(4, output);
            for (size_t c = 0; c < width; ++c) require(output[c] == data[4 * width + c], "uncached row differs after reclaim");
            const auto old_output = output;
            reject([&] { reclaimed.copy(256, output); });
            require(output == old_output, "uncached failed decode changed output");
            reclaimed.insert_decoded(4, output);
            require(!reclaimed.try_copy(4, output) && reclaimed.capacity() == 0, "insert repopulated a reclaimed cache");
            const std::array<uint64_t, 6> ids{4, 5, 4, 7, 8, 5};
            std::array<float, ids.size() * width> gathered{}, deduped{};
            reclaimed.gather(ids, gathered);
            PleGatherScratch scratch;
            reclaimed.gather_chunk_dedup(ids, deduped, scratch);
            require(gathered == deduped, "reclaimed ordinary and deduplicated gathers differ");
            for (size_t i = 0; i < ids.size(); ++i) for (size_t c = 0; c < width; ++c)
                require(gathered[i * width + c] == data[ids[i] * width + c], "reclaimed gather differs from source");
            reclaimed.seed_from_dedup(scratch, gathered);
            require(reclaimed.charged_bytes() + freed == before_bytes, "reclaimed gather allocated cache storage");
            bool refused = false;
            std::thread t([&] { try { static_cast<void>(reclaimed.reclaim()); } catch (const FormatError&) { refused = true; } });
            t.join(); require(refused, "cross-thread reclamation accepted");
        }
        {
            size_t reads = 0, reclaims = 0;
            auto check = [&](MemoryGuardReading first, MemoryGuardReading second) {
                reads = reclaims = 0;
                return reclaim_before_memory_refusal(100, [&]() { return reads++ ? second : first; }, [&]() { ++reclaims; });
            };
            require(check({100, 1}, {0, 0}) && reads == 1 && reclaims == 0, "healthy memory guard reclaimed or reread");
            require(check({101, 1}, {90, 1}) && reads == 2 && reclaims == 1, "guard did not remeasure after footprint reclamation");
            require(check({80, 2}, {80, 1}) && reads == 2 && reclaims == 1, "guard did not recover after pressure reclamation");
            require(!check({101, 1}, {101, 1}) && reads == 2 && reclaims == 1, "guard admitted persistent footprint excess");
            require(!check({80, 2}, {80, 4}) && reads == 2 && reclaims == 1, "guard admitted persistent host pressure");
            require(!check({80, 0}, {80, 0}), "guard admitted unavailable pressure data");
            reject([&] { static_cast<void>(reclaim_before_memory_refusal(100, []() -> MemoryGuardReading { throw FormatError("sensor failure"); }, []() {})); });
        }
        // Dilation=3, kernel=4: taps are t-9,t-6,t-3,t, not four adjacent tokens.
        std::vector<double> hist(9, 0); const std::array<double, 4> weights{1, 10, 100, 1000};
        std::vector<double> sequence;
        for (size_t t = 0; t < 20; ++t) {
            sequence.push_back(double(t + 1)); const auto conv = causal_depthwise_reference(hist, std::array<double, 1>{sequence.back()}, weights, 4, 3);
            double expected = 0;
            for (size_t tap = 0; tap < 4; ++tap) if (t >= (3 - tap) * 3) expected += sequence[t - (3 - tap) * 3] * weights[tap];
            require(conv.output[0] == expected, "dilated history uses incorrect time taps"); hist = conv.history;
        }
        std::cout << "Memory lookup/cache math checks passed: " << checks << '\n';
    } catch (const std::exception& e) { std::cerr << "Memory math check failed: " << e.what() << '\n'; return 1; }
}
