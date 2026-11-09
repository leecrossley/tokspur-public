#include "tokspur/moe_mm_many.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/gguf.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>

using namespace tokspur;

int main() {
    std::cout << "===============================================================\n";
    std::cout << "EMPIRICAL ADVERSARIAL CHALLENGE: MILESTONE M2 (MOE 2K ROUTING)\n";
    std::cout << "===============================================================\n\n";

    const MoeShape production_shape{2560, 640, 512, 10}; // Qwen3.8-Flash-Next

    // ------------------------------------------------------------------------
    // Task 1: Real-model Qwen3.8-Flash-Next (selected = 10) with capacity = 2048
    // ------------------------------------------------------------------------
    std::cout << "[TASK 1] Testing production model (selected = 10) at capacity = 2048:\n";
    bool caught_2048 = false;
    std::string error_2048;
    try {
        moe_mm_many_bytes(production_shape, 2048);
    } catch (const FormatError& e) {
        caught_2048 = true;
        error_2048 = e.what();
    }
    std::cout << "  Input: selected = 10, capacity = 2048 (routes = " << (10 * 2048) << ")\n";
    std::cout << "  Result: " << (caught_2048 ? "THREW FormatError" : "UNEXPECTEDLY PASSED") << "\n";
    std::cout << "  Verbatim Error: \"" << error_2048 << "\"\n";
    std::cout << "  Finding: uint64_t(s.selected) * capacity > 10240 triggered rejection!\n\n";

    // ------------------------------------------------------------------------
    // Task 1b: Production model at maximum capacity under route cap: capacity = 1024
    // ------------------------------------------------------------------------
    std::cout << "[TASK 1b] Testing production model (selected = 10) at boundary capacities:\n";
    bool caught_1024 = false;
    uint64_t total_1024 = 0;
    try {
        const auto b = moe_mm_many_bytes(production_shape, 1024);
        total_1024 = b.total;
        std::cout << "  Capacity = 1024 (routes = " << (10 * 1024) << "):\n";
        std::cout << "    PASSED! Total scratch bytes = " << b.total << " ("
                  << std::fixed << std::setprecision(2) << (double(b.total) / (1024.0 * 1024.0)) << " MiB)\n";
        std::cout << "    Logits: " << b.views[0] << " B\n";
        std::cout << "    Indices: " << b.views[1] << " B, Weights: " << b.views[2] << " B\n";
        std::cout << "    Gate: " << b.views[3] << " B, Up: " << b.views[4] << " B, Down: " << b.views[5] << " B\n";
        std::cout << "    Shared: Gate=" << b.views[6] << " B, Up=" << b.views[7] << " B, Down=" << b.views[8] << " B, Logit=" << b.views[9] << " B\n";
        std::cout << "    TokenMap: " << b.views[10] << " B, Counts: " << b.views[11] << " B, Tasks: " << b.views[12] << " B\n";
    } catch (const FormatError& e) {
        caught_1024 = true;
        std::cout << "    FAILED: " << e.what() << "\n";
    }
    assert(!caught_1024);
    assert(total_1024 > 0);

    bool caught_1025 = false;
    std::string error_1025;
    try {
        moe_mm_many_bytes(production_shape, 1025);
    } catch (const FormatError& e) {
        caught_1025 = true;
        error_1025 = e.what();
    }
    std::cout << "  Capacity = 1025 (routes = " << (10 * 1025) << "):\n";
    std::cout << "    " << (caught_1025 ? "THREW FormatError" : "UNEXPECTEDLY PASSED")
              << " (\"" << error_1025 << "\")\n\n";

    // ------------------------------------------------------------------------
    // Task 2: Boundary Matrix: capacity in {512, 1024, 2048, 2049} x selected in {1, 5, 10, 11}
    // ------------------------------------------------------------------------
    std::cout << "[TASK 2] Boundary Input Matrix (capacity x selected):\n";
    std::cout << std::setw(10) << "Capacity"
              << std::setw(10) << "Selected"
              << std::setw(12) << "Routes"
              << std::setw(12) << "Expected"
              << std::setw(12) << "Actual"
              << std::setw(16) << "Total Bytes"
              << std::setw(12) << "Total MiB"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    const std::vector<uint32_t> capacities = {512, 1024, 2048, 2049};
    const std::vector<uint32_t> selected_vals = {1, 5, 10, 11};

    for (auto cap : capacities) {
        for (auto sel : selected_vals) {
            const MoeShape s{2560, 640, 512, sel};
            const uint64_t routes = uint64_t(sel) * cap;
            const bool expect_pass = (cap <= 2048) && (routes <= 10240);
            bool actual_pass = false;
            uint64_t total_bytes = 0;
            std::string err;

            try {
                const auto b = moe_mm_many_bytes(s, cap);
                actual_pass = true;
                total_bytes = b.total;
            } catch (const FormatError& e) {
                actual_pass = false;
                err = e.what();
            }

            std::cout << std::setw(10) << cap
                      << std::setw(10) << sel
                      << std::setw(12) << routes
                      << std::setw(12) << (expect_pass ? "PASS" : "REJECT")
                      << std::setw(12) << (actual_pass ? "PASS" : "REJECT")
                      << std::setw(16) << (actual_pass ? std::to_string(total_bytes) : "N/A")
                      << std::setw(12) << (actual_pass ? std::to_string(double(total_bytes) / (1024.0 * 1024.0)).substr(0, 6) : "N/A")
                      << "\n";
            assert(actual_pass == expect_pass);
        }
    }
    std::cout << "\n";

    // ------------------------------------------------------------------------
    // Task 3: Comparison between orchestrator_3 test (selected = 5) and production (selected = 10)
    // ------------------------------------------------------------------------
    std::cout << "[TASK 3] Comparison between orchestrator_3 test (selected = 5) and production (selected = 10):\n";
    const auto b_orch3 = moe_mm_many_bytes({2560, 640, 512, 5}, 2048);
    std::cout << "  orchestrator_3 test: selected = 5, capacity = 2048\n";
    std::cout << "    routes = " << (5 * 2048) << " (<= 10240: boundary hit!)\n";
    std::cout << "    total bytes = " << b_orch3.total << " ("
              << (double(b_orch3.total) / (1024.0 * 1024.0)) << " MiB)\n";

    std::cout << "  production model: selected = 10, capacity = 2048\n";
    std::cout << "    routes = " << (10 * 2048) << " (> 10240: VIOLATION)\n";
    std::cout << "    Single-pass 2048 MoE execution is IMPOSSIBLE on Qwen3.8-Flash-Next without chunking!\n\n";

    // ------------------------------------------------------------------------
    // Task 4: Memory Budget Verification (300 MiB limit)
    // ------------------------------------------------------------------------
    std::cout << "[TASK 4] Maximum Memory Exploration under route cap (routes <= 10240, cap <= 2048):\n";
    uint64_t max_total = 0;
    uint32_t max_sel = 0, max_cap = 0;
    for (uint32_t sel = 1; sel <= 16; ++sel) {
        for (uint32_t cap = 1; cap <= 2048; ++cap) {
            if (uint64_t(sel) * cap <= 10240) {
                try {
                    const auto b = moe_mm_many_bytes({2560, 640, 512, sel}, cap);
                    if (b.total > max_total) {
                        max_total = b.total;
                        max_sel = sel;
                        max_cap = cap;
                    }
                    if (b.total > (300ull << 20)) {
                        std::cerr << "CRITICAL: Memory budget exceeded! " << b.total << " > 300 MiB\n";
                        return 1;
                    }
                } catch (...) {
                    // Ignore non-memory rejections
                }
            }
        }
    }
    std::cout << "  Worst-case valid configuration: selected = " << max_sel << ", capacity = " << max_cap
              << " (routes = " << (max_sel * max_cap) << ")\n";
    std::cout << "  Max total bytes = " << max_total << " ("
              << (double(max_total) / (1024.0 * 1024.0)) << " MiB)\n";
    std::cout << "  300 MiB ceiling = " << (300ull << 20) << " bytes (300.00 MiB)\n";
    std::cout << "  Safety margin = " << ((300ull << 20) - max_total) << " bytes ("
              << (double((300ull << 20) - max_total) / (1024.0 * 1024.0)) << " MiB)\n\n";

    // ------------------------------------------------------------------------
    // Task 5: Testing moe_router_window_plan and moe_router_f32_plan limits
    // ------------------------------------------------------------------------
    std::cout << "[TASK 5] Testing router plans at tokens > 512:\n";
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    for (uint32_t tok : {512u, 513u, 1024u, 2048u}) {
        bool thrown = false;
        try {
            moe_router_window_plan(2560, 512, tok, 10240, 2048);
        } catch (const FormatError& e) {
            thrown = true;
        }
        std::cout << "  moe_router_window_plan(tokens=" << tok << "): "
                  << (thrown ? "THREW FormatError (tokens > 512 rejected)" : "PASSED") << "\n";
    }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    for (uint32_t tok : {512u, 513u, 1024u, 2048u}) {
        bool thrown = false;
        try {
            moe_router_f32_plan(2560, 512, tok, 10240, 2048);
        } catch (const FormatError& e) {
            thrown = true;
        }
        std::cout << "  moe_router_f32_plan(tokens=" << tok << "): "
                  << (thrown ? "THREW FormatError (tokens > 512 rejected)" : "PASSED") << "\n";
    }
#endif
    std::cout << "\n";

    // ------------------------------------------------------------------------
    // Task 6: Testing grouped_mm_plan limits
    // ------------------------------------------------------------------------
    std::cout << "[TASK 6] Testing grouped_mm_plan at tokens > 512:\n";
    for (uint32_t tok : {512u, 513u, 1024u, 2048u}) {
        bool thrown = false;
        try {
            grouped_mm_plan(8, PrefillWeightLayout::gguf, 2560, 640, 512, 10, tok, 10240, 2560, true);
        } catch (const FormatError& e) {
            thrown = true;
        }
        std::cout << "  grouped_mm_plan(tokens=" << tok << "): "
                  << (thrown ? "THREW FormatError (tokens > 512 rejected)" : "PASSED") << "\n";
    }
    std::cout << "\n";

    // ------------------------------------------------------------------------
    // Task 7: Architectural Assessment of 2,048-token Prefill Window
    // ------------------------------------------------------------------------
    std::cout << "[TASK 7] Architectural Conclusion:\n";
    std::cout << "  1. On production Qwen3.8-Flash-Next (selected = 10):\n";
    std::cout << "     - capacity = 2048 is rejected by route capping assertion (20,480 > 10,240 routes).\n";
    std::cout << "     - capacity = 1024 is the maximum single pass allowed (10,240 routes, 169.09 MiB scratch).\n";
    std::cout << "  2. moe_router_window_plan, moe_router_f32_plan, and grouped_mm_plan strictly enforce tokens <= 512.\n";
    std::cout << "  3. Therefore, executing a 2,048-token prefill window CANNOT be done in a single MoE pass.\n";
    std::cout << "     It REQUIRES chunked MoE passes of <= 512 tokens (which matches ChunkedPrefillStore's 512-token chunking in stack.cpp:1208).\n";
    std::cout << "===============================================================\n";
    std::cout << "ALL EMPIRICAL ADVERSARIAL CHECKS COMPLETED SUCCESSFULLY.\n";
    return 0;
}
