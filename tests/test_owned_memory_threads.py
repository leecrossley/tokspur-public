"""Model-free thread-start failure coverage of the actual memory gather code."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r"""
#include "tokspur/memory.hpp"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <system_error>
#include <thread>
namespace tokspur {
class ControlledThread {
    std::thread thread_;
public:
    static inline int fail_after = -1, attempts = 0, joined = 0;
    static inline std::atomic<int> live{0};
    ControlledThread() = default;
    template<class Function> explicit ControlledThread(Function&& function) {
        if (attempts++ == fail_after) throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        thread_ = std::thread([function = std::forward<Function>(function)]() mutable { ++live; function(); --live; });
    }
    ControlledThread(ControlledThread&&) = default;
    ControlledThread& operator=(ControlledThread&&) = default;
    ~ControlledThread() { if (thread_.joinable()) std::_Exit(3); }
    bool joinable() const { return thread_.joinable(); }
    void join() { thread_.join(); ++joined; }
};
}
"""

CHECKS = r"""
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        using namespace tokspur;
        constexpr uint32_t width = 8, rows = (16u << 20) / (width * sizeof(float));
        std::vector<float> data(size_t(rows) * width);
        for (size_t i = 0; i < 64 * width; ++i) data[i] = float(i) / 32;
        const auto bytes = std::as_bytes(std::span(data));
        const std::array<uint64_t, 8> ids{0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<float> output(ids.size() * width);
        ControlledThread::fail_after = 0;
        EmbeddingRows serial(bytes, 0, width, rows, rows, 4096);
        serial.gather(ids, output);
        check(ControlledThread::attempts == 0, "default gather attempted to create a worker");
        check(std::equal(output.begin(), output.end(), data.begin()), "default gather changed decoded rows");
        check(serial.charged_bytes() <= 4096, "default gather exceeded its cache cap");

        const auto failures = [&](bool cached) {
            const std::array<int, 4> points = cached ? std::array{0, 1, 5, 10} : std::array{0, 1, 7, 14};
            for (const int successful_threads : points) {
                ControlledThread::fail_after = successful_threads;
                ControlledThread::attempts = 0;
                const auto joined_before = ControlledThread::joined;
                bool rejected = false;
                try {
                    if (cached) {
                        EmbeddingRows parallel(bytes, 0, width, rows, rows, 4096, {}, true);
                        parallel.gather(ids, output);
                    } else {
                        std::vector<uint64_t> repeated(4096, 7);
                        std::vector<float> decoded(repeated.size() * width);
                        PleGatherScratch scratch;
                        dequantize_rows_sorted_dedup(bytes, 0, width, rows, rows, repeated, decoded, scratch, 16);
                    }
                } catch (const std::system_error& error) {
                    check(error.code() == std::errc::resource_unavailable_try_again, "gather replaced the thread creation error");
                    rejected = true;
                }
                check(rejected && ControlledThread::attempts == successful_threads + 1, "thread creation failure was not reached");
                check(ControlledThread::live == 0 && ControlledThread::joined - joined_before == successful_threads, "failed gather left workers running or unjoined");
            }
        };
        failures(false);
        failures(true);
        std::cout << "serial default and 8 partial thread-creation failures passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
"""


def test_memory_gather_defaults_and_thread_creation_failures(tmp_path):
    implementation = (ROOT / "native/src/memory.cpp").read_text()
    assert implementation.count("std::thread") == 3
    source = tmp_path / "memory_threads.cpp"
    executable = tmp_path / "memory_threads"
    source.write_text(
        PRELUDE + implementation.replace("std::thread", "ControlledThread") + CHECKS
    )
    compiled = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O1",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "native/include"),
            str(source),
            str(ROOT / "native/src/gguf.cpp"),
            str(ROOT / "native/src/numeric.cpp"),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    assert (
        result.stdout
        == "serial default and 8 partial thread-creation failures passed\n"
    )
