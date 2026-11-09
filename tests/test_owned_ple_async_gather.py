"""Model-free coverage of the actual async PLE gatherer under controlled wakes."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r"""
#include "tokspur/memory.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
namespace tokspur { namespace {
void require(bool ok, const char* message) { if (!ok) throw FormatError(message); }
thread_local size_t fixture_worker_id = 8;
class ControlledConditionVariable;
ControlledConditionVariable* requests = nullptr;

class ControlledThread {
    std::thread thread_;
public:
    static inline int fail_after = -1, attempts = 0, joined = 0;
    static inline std::atomic<int> live{0};
    ControlledThread() = default;
    template<class Function> explicit ControlledThread(Function&& function) {
        if (attempts++ == fail_after) throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        thread_ = std::thread([function = std::forward<Function>(function)]() mutable {
            ++live;
            function();
            --live;
        });
    }
    ControlledThread(ControlledThread&&) = default;
    ControlledThread& operator=(ControlledThread&&) = default;
    ~ControlledThread() {
        if (thread_.joinable()) {
            std::cerr << "thread constructor unwound with an unjoined worker\n";
            std::_Exit(3);
        }
    }
    bool joinable() const { return thread_.joinable(); }
    void join() { thread_.join(); ++joined; }
};

// Per-worker condition variables make notify_one choose worker 1, a permitted choice which is ineligible for a one-row job.
class ControlledConditionVariable {
    std::array<std::condition_variable, 8> waiters_;
    std::array<size_t, 8> calls_{};
    std::mutex state_mutex_;
    std::condition_variable changed_;
    uint32_t waiting_ = 0;
public:
    ControlledConditionVariable() { requests = this; }
    template<class Predicate> void wait(std::unique_lock<std::mutex>& lock, Predicate predicate) {
        const size_t id = fixture_worker_id;
        require(id < 8, "request waiter is not a worker");
        {
            std::lock_guard state(state_mutex_);
            ++calls_[id];
        }
        while (!predicate()) {
            {
                std::lock_guard state(state_mutex_);
                waiting_ |= 1u << id;
                changed_.notify_all();
            }
            waiters_[id].wait(lock);
            {
                std::lock_guard state(state_mutex_);
                waiting_ &= ~(1u << id);
            }
        }
    }
    void notify_one() {
        std::lock_guard state(state_mutex_);
        require((waiting_ & 2u) != 0, "controlled worker 1 is not waiting");
        waiters_[1].notify_one();
    }
    void notify_all() { for (auto& waiter : waiters_) waiter.notify_all(); }
    size_t worker_zero_calls() {
        std::lock_guard state(state_mutex_);
        return calls_[0];
    }
    void await_idle(size_t worker_zero_calls) {
        std::unique_lock state(state_mutex_);
        if (!changed_.wait_for(state, std::chrono::seconds(5), [&] {
            return waiting_ == 255 && calls_[0] >= worker_zero_calls;
        })) {
            std::cerr << "eligible worker 0 did not complete the PLE job\n";
            std::_Exit(2);
        }
    }
};
"""

PREFILL_PRELUDE = r"""
void run_prefill_policy() {
    constexpr uint64_t rows = 16 * 128;
    constexpr size_t width = 160;
    NgramSpec spec{3, 8, width, 128, 127, 126, {3, 5, 7}, {}, {}};
    for (size_t i = 0; i < 16; ++i) { spec.offsets.push_back(i * 128); spec.vocabularies.push_back(128); }
    struct { NgramPolicy ngram; } effective_policy{{HashOrder::newest_first, NgramBoundary::literal, false}};
    struct { uint64_t ple_storage_rows, ple_addressable_rows; } layout{rows, rows};
    const auto worker_entry_history = empty_ngram_history(spec);
    std::vector<float> data(rows * width);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(i) / 256;
    const auto ple_bytes = std::as_bytes(std::span(data));
    constexpr uint32_t ple_type = 0;
    constexpr uint64_t ple_row_bytes = width * sizeof(float), ple_logical = rows;
    NocacheTensorFile ple_nocache_file;
    for (const bool ple_async_gather_enabled : {false, true}) {
        for (const size_t count : {2u, 63u, 64u, 129u, 2048u, 2051u}) {
            std::vector<uint32_t> tokens(count);
            for (size_t i = 0; i < count; ++i) tokens[i] = (i * 19) % 125;
            std::vector<float> alternate_embeddings(count * 2560);
            std::exception_ptr worker_error;
            size_t candidate_calls = 0;
            const auto traced_dedup = [&](auto bytes, auto type, auto row_width, auto stored, auto logical,
                auto ids, auto output, auto& scratch, auto threads, auto pread) {
                ++candidate_calls;
                require(threads == 16 && !pread.valid(), "candidate worker geometry or default I/O changed");
                dequantize_rows_sorted_dedup(bytes, type, row_width, stored, logical, ids, output, scratch, threads, pread);
            };
"""

PREFILL_CHECKS = r"""
            worker_task();
            if (worker_error) std::rethrow_exception(worker_error);
            require(candidate_calls == (ple_async_gather_enabled ? (count + 2047) / 2048 : 0),
                "new stateless prefill scheduling escaped its opt-in");
            auto history = worker_entry_history;
            for (size_t token = 0; token < count; ++token) {
                const auto step = ngram_step(spec, effective_policy.ngram, history, tokens[token]);
                history = step.next;
                for (size_t row = 0; row < 16; ++row) {
                    const auto begin = alternate_embeddings.begin() + token * 2560 + row * width;
                    require(std::equal(begin, begin + width, data.begin() + step.rows[row] * width),
                        "prefill row output differs from the serial reference");
                }
            }
        }
    }
    std::cout << "12 default/candidate prefill cases passed\n";
}
"""

CHECKS = r"""
void run() {
    constexpr uint64_t rows = 16 * 128;
    constexpr size_t width = 160;
    NgramSpec spec{3, 8, width, 128, 127, 126, {3, 5, 7}, {}, {}};
    for (size_t i = 0; i < 16; ++i) { spec.offsets.push_back(i * 128); spec.vocabularies.push_back(128); }
    const NgramPolicy policy{HashOrder::newest_first, NgramBoundary::literal, false};
    const auto entry = empty_ngram_history(spec);
    std::vector<float> data(rows * width);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(i) / 256;
    const auto bytes = std::as_bytes(std::span(data));
    EmbeddingRows cache(bytes, 0, width, rows, rows, 2ull << 20);
    PleAsyncGatherer gatherer(cache, spec, policy, bytes, 0, rows, rows);
    requests->await_idle(1);
    std::array<float, 2560> output{};
    std::array<float, width> warm{};
    size_t completed = 0, cancelled = 0;
    for (uint32_t token = 0; token < 32; ++token) {
        const auto expected = ngram_step(spec, policy, entry, token);
        for (size_t row = 0; row < 15; ++row) cache.copy(expected.rows[row], warm);
        require(!cache.try_copy(expected.rows[15], warm), "fixture must have exactly one missing row");
        const auto before = requests->worker_zero_calls();
        gatherer.start_tokens(entry, std::span(&token, 1), output);
        requests->await_idle(before + 1);
        if (token % 2 == 0) {
            gatherer.cancel_join_nothrow();
            ++cancelled;
            require(!cache.try_copy(expected.rows[15], warm), "cancelled gather published its missing row");
            gatherer.start_tokens(entry, std::span(&token, 1), output);
            requests->await_idle(before + 2);
        }
        require(gatherer.finish() == expected.next, "gather changed the n-gram exit history");
        ++completed;
        for (size_t row = 0; row < 16; ++row) {
            require(std::equal(output.begin() + row * width, output.begin() + (row + 1) * width,
                data.begin() + expected.rows[row] * width), "gather returned the wrong decoded row");
        }
        require(cache.try_copy(expected.rows[15], warm), "finished gather did not publish its missing row");
        const auto cached_calls = requests->worker_zero_calls();
        gatherer.start_tokens(entry, std::span(&token, 1), output);
        require(gatherer.finish() == expected.next, "cached gather changed history");
        require(requests->worker_zero_calls() == cached_calls, "cache-only gather launched a worker");
    }
    const uint32_t cold_token = 64;
    const auto cold = ngram_step(spec, policy, entry, cold_token);
    for (auto row : cold.rows) require(!cache.try_copy(row, warm), "multi-worker fixture row was cached");
    const auto before = requests->worker_zero_calls();
    gatherer.start_tokens(entry, std::span(&cold_token, 1), output);
    requests->await_idle(before + 1);
    require(gatherer.finish() == cold.next, "multi-worker gather changed history");
    for (size_t row = 0; row < 16; ++row) {
        require(std::equal(output.begin() + row * width, output.begin() + (row + 1) * width,
            data.begin() + cold.rows[row] * width), "multi-worker gather returned the wrong row");
    }
    std::cout << completed << " one-miss gathers, " << cancelled << " cancellations, 32 cached gathers and 1 multi-worker gather passed\n";
}
void run_creation_failures() {
    constexpr size_t width = 160;
    NgramSpec spec{3, 8, width, 128, 127, 126, {3, 5, 7}, {}, {}};
    for (size_t i = 0; i < 16; ++i) { spec.offsets.push_back(i * 128); spec.vocabularies.push_back(128); }
    const NgramPolicy policy{HashOrder::newest_first, NgramBoundary::literal, false};
    std::vector<float> data(2048 * width);
    const auto bytes = std::as_bytes(std::span(data));
    EmbeddingRows cache(bytes, 0, width, 2048, 2048, 2ull << 20);
    require(ControlledThread::live == 0 && ControlledThread::joined == 8, "normal destruction left workers running");
    for (const int successful_threads : {0, 1, 4, 7}) {
        ControlledThread::attempts = 0;
        ControlledThread::fail_after = successful_threads;
        const auto joined_before = ControlledThread::joined;
        bool rejected = false;
        try { PleAsyncGatherer failed(cache, spec, policy, bytes, 0, 2048, 2048); }
        catch (const std::system_error& error) {
            require(error.code() == std::errc::resource_unavailable_try_again, "constructor replaced the creation failure");
            rejected = true;
        }
        require(rejected && ControlledThread::attempts == successful_threads + 1, "constructor failure injection was not reached");
        require(ControlledThread::live == 0 && ControlledThread::joined - joined_before == successful_threads,
            "constructor failure left a created thread running or unjoined");
    }
    ControlledThread::fail_after = -1;
    std::cout << "4 partial thread-creation failures drained and rethrown\n";
}
} }
int main() {
    try { tokspur::run(); tokspur::run_creation_failures(); tokspur::run_prefill_policy(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
"""


def extracted_unit(source: str) -> str:
    gatherer = source.split("class PleAsyncGatherer {", 1)[1].split(
        "\nstruct PleAsyncScopeGuard {", 1
    )[0]
    gatherer = "class PleAsyncGatherer {" + gatherer
    declaration = "std::condition_variable cv_req_, cv_done_;"
    worker = "void worker_loop(size_t worker_id) {"
    assert gatherer.count(declaration) == gatherer.count(worker) == 1
    assert gatherer.count("std::thread") == 2
    gatherer = gatherer.replace(
        declaration,
        "ControlledConditionVariable cv_req_; std::condition_variable cv_done_;",
    ).replace(worker, worker + "\n        fixture_worker_id = worker_id;")
    gatherer = gatherer.replace("std::thread", "ControlledThread")
    worker_task = (
        source.split("const auto advance_known =", 1)[1]
        .split("            auto worker_task = [&]() {", 1)[1]
        .split("\n            ScopedJoinThread worker_thread", 1)[0]
    )
    worker_task = "            auto worker_task = [&]() {" + worker_task
    worker_task = worker_task.replace("dequantize_rows_sorted_dedup(", "traced_dedup(")
    return PRELUDE + gatherer + PREFILL_PRELUDE + worker_task + PREFILL_CHECKS + CHECKS


def test_async_ple_gather_wakes_eligible_worker_and_preserves_publication(tmp_path):
    source = tmp_path / "ple_async.cpp"
    executable = tmp_path / "ple_async"
    source.write_text(extracted_unit((ROOT / "native/src/target.cpp").read_text()))
    compiled = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-pthread",
            "-I" + str(ROOT / "native/include"),
            str(source),
            *[
                str(ROOT / f"native/src/{name}.cpp")
                for name in ("memory", "numeric", "gguf")
            ],
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(executable)], check=True, timeout=60)
