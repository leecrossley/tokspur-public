#include "tokspur/worker_identity.hpp"
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef __APPLE__
#include <sys/param.h>
static_assert(MAXCOMLEN == 16, "update the Python Darwin MAXCOMLEN binding");
#endif

int main() {
    using namespace tokspur;
    unsigned checks = 0;
    const auto check = [&](bool value) { ++checks; if (!value) throw std::runtime_error("worker identity check failed"); };
    try {
        check(scalar_kernel_worker_name(scalar_worker_name, scalar_worker_name.size()));
        for (size_t limit : {size_t(1), size_t(15), size_t(16), scalar_worker_name.size(), size_t(32)}) {
            const auto expected = std::string(scalar_worker_name.substr(0, limit));
            check(scalar_kernel_worker_name(expected, limit));
            check(!scalar_kernel_worker_name(expected + "x", limit));
            check(!scalar_kernel_worker_name("x" + expected, limit));
            check(!scalar_kernel_worker_name(expected.substr(0, expected.size() - 1), limit));
        }
        check(!scalar_kernel_worker_name("", 0));
        check(!scalar_kernel_worker_name("unrelated-worker", 16));
        check(live_worker_candidate(7, 8, false));
        check(!live_worker_candidate(7, 7, false));
        check(!live_worker_candidate(7, 8, true));
        check(!live_worker_candidate(0, 8, false));
        check(!live_worker_candidate(-1, 8, false));
#ifdef __APPLE__
        check(scalar_kernel_worker_name(scalar_worker_name.substr(0, MAXCOMLEN), MAXCOMLEN));
#endif
        std::cout << "worker identity CPU checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
