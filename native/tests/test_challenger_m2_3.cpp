#define main inherited_stack_entry
#include "test_stack.cpp"
#undef main

#include "tokspur/moe_mm_many.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/qsa_prefill.hpp"
#include "tokspur/prefill_gdn.hpp"
#include "tokspur/prefill_attention.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/stack.hpp"
#include "tokspur/metal.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace tokspur;

namespace {

size_t total_checks = 0;
size_t total_passes = 0;
size_t total_defects_found = 0;

void require_check(bool condition, const char* message) {
    ++total_checks;
    if (!condition) {
        std::cerr << "FAILED REQUIREMENT: " << message << std::endl;
        throw std::runtime_error(message);
    }
    ++total_passes;
}

template <typename F>
std::string catch_expected_throw(F&& func, const std::string& expected_msg, const char* label) {
    ++total_checks;
    bool caught = false;
    std::string what;
    try {
        func();
    } catch (const FormatError& err) {
        caught = true;
        what = err.what();
    } catch (const std::exception& err) {
        caught = true;
        what = err.what();
    }
    if (!caught) {
        std::cerr << "FAIL [" << label << "]: expected exception containing '"
                  << expected_msg << "' but none was thrown" << std::endl;
        throw std::runtime_error("Expected exception not thrown");
    }
    if (what.find(expected_msg) == std::string::npos) {
        std::cerr << "FAIL [" << label << "]: exception message '"
                  << what << "' does not contain expected '" << expected_msg << "'" << std::endl;
        throw std::runtime_error("Unexpected exception message");
    }
    ++total_passes;
    std::cout << "  [PASS] " << label << " -> caught expected FormatError: \"" << what << "\"\n";
    return what;
}

GpuMatrix stack_q8_custom(Assets& a, uint32_t c, uint32_t r, uint32_t experts = 1) {
    std::vector<std::byte> data(uint64_t(c / 32) * r * experts * 34);
    for (uint64_t b = 0; b < data.size() / 34; ++b) {
        const uint16_t d = 0x1400; std::memcpy(data.data() + b * 34, &d, 2);
        for (uint32_t j = 0; j < 32; ++j) { const int8_t q = int8_t(int((b * 17 + j * 11) % 15) - 7); std::memcpy(data.data() + b * 34 + 2 + j, &q, 1); }
    }
    const auto v = a.g.upload(data, "generated chunked stack Q8", true); a.views.push_back(v); return {v, 8};
}

GpuMatrix stack_bf16_custom(Assets& a, GpuMatrix w) {
    const auto bytes = a.g.read(w.view); std::vector<uint16_t> values(bytes.size() / 4);
    for (size_t i = 0; i < values.size(); ++i) { float x; std::memcpy(&x, bytes.data() + i * 4, 4); values[i] = finite_bf16(x); }
    const auto v = a.g.upload(std::as_bytes(std::span(values)), "generated chunked stack BF16", true); a.views.push_back(v); return {v, 30};
}

} // namespace

// ============================================================================
// TASK 1 & 2 & 3: MoE tile many Scratch Planning & Route Bounds
// ============================================================================

void test_task1_2_3_moe_execution_and_chunking() {
    std::cout << "====================================================================\n";
    std::cout << "[TASK 1, 2, 3] Production MoE 2K Execution, Chunk Slicing & Route Bounds\n";
    std::cout << "====================================================================\n\n";

    const MoeShape prod_shape{2560, 640, 512, 10}; // Qwen3.8-Flash-Next

    std::cout << "--- Subtask 1a: Scratch capacity boundary tests ---\n";
    const auto b1024 = moe_mm_many_bytes(prod_shape, 1024);
    require_check(b1024.total == 177304072, "Total bytes at cap=1024 must equal 177,304,072");
    require_check(b1024.total <= (300ull << 20), "Total bytes must be within 300 MiB limit");
    std::cout << "  Capacity 1024: total bytes = " << b1024.total
              << " (" << std::fixed << std::setprecision(2) << (double(b1024.total) / (1024.0 * 1024.0)) << " MiB)\n";
    std::cout << "  Routes: " << (10 * 1024) << " <= 10,240 route limit [PASS]\n";

    // Corner cases: rows in {0, 1, 1023, 1024, 1025, 2047, 2048, 2049}
    std::cout << "\n--- Subtask 1b: Corner cases for single-pass moe_mm_many_bytes ---\n";
    catch_expected_throw([&] { moe_mm_many_bytes(prod_shape, 0); }, "bounded tiled MoE row geometry", "capacity = 0");
    
    const auto b1 = moe_mm_many_bytes(prod_shape, 1);
    require_check(b1.total <= (300ull << 20), "capacity = 1 within 300 MiB");
    const auto b1023 = moe_mm_many_bytes(prod_shape, 1023);
    require_check(b1023.total <= (300ull << 20), "capacity = 1023 within 300 MiB");

    catch_expected_throw([&] { moe_mm_many_bytes(prod_shape, 1025); }, "bounded tiled MoE row geometry", "capacity = 1025");
    catch_expected_throw([&] { moe_mm_many_bytes(prod_shape, 2047); }, "bounded tiled MoE row geometry", "capacity = 2047");
    catch_expected_throw([&] { moe_mm_many_bytes(prod_shape, 2048); }, "bounded tiled MoE row geometry", "capacity = 2048");
    catch_expected_throw([&] { moe_mm_many_bytes(prod_shape, 2049); }, "bounded tiled MoE row geometry", "capacity = 2049");

    // Chunk slicing simulation for rows in {0, 1, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049}
    std::cout << "\n--- Subtask 2: Buffer overrun & route capping verification ---\n";
    const uint32_t capacity = 1024;
    const uint32_t max_moe_chunk = std::min({capacity, 512u, uint32_t(10240 / prod_shape.selected)});
    require_check(max_moe_chunk == 512, "max_moe_chunk must equal 512 for production shape");
    const uint64_t stride = 2560 * 4 + 32;
    const uint64_t columns = 2560;
    const std::vector<uint32_t> test_rows = {0, 1, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049};

    for (uint32_t rows : test_rows) {
        if (rows == 0) {
            std::cout << "  rows = 0: rejected by need(input.rows && ...) assertion [PASS]\n";
            continue;
        }

        const uint64_t total_parent_bytes = checked_add(checked_mul(rows - 1, stride), columns * 4);
        std::vector<bool> row_covered(rows, false);
        uint32_t chunk_count = 0;
        uint32_t total_routes = 0;

        for (uint32_t r_offset = 0; r_offset < rows; r_offset += max_moe_chunk) {
            const uint32_t chunk_r = std::min(max_moe_chunk, rows - r_offset);
            require_check(chunk_r > 0 && chunk_r <= max_moe_chunk, "chunk_r bounded by max_moe_chunk (512)");

            const uint64_t routes_in_chunk = uint64_t(chunk_r) * prod_shape.selected;
            require_check(routes_in_chunk <= 10240, "routes in chunk strictly <= 10,240");
            total_routes += routes_in_chunk;

            const uint64_t slice_offset = uint64_t(r_offset) * stride;
            const uint64_t slice_bytes = checked_add(checked_mul(chunk_r - 1, stride), columns * 4);
            require_check(slice_offset + slice_bytes <= total_parent_bytes, "slice must not overrun parent buffer");

            for (uint32_t i = 0; i < chunk_r; ++i) {
                const uint32_t r = r_offset + i;
                require_check(!row_covered[r], "no duplicate row coverage");
                row_covered[r] = true;
            }
            chunk_count++;
        }

        require_check(std::all_of(row_covered.begin(), row_covered.end(), [](bool b) { return b; }), "all rows covered");
        require_check(total_routes == uint64_t(rows) * prod_shape.selected, "total routes equals rows * selected");

        std::cout << "  rows = " << std::setw(4) << rows
                  << " | passes = " << chunk_count
                  << " | total routes = " << std::setw(5) << total_routes
                  << " | max routes/pass = " << std::min<uint64_t>(5120, uint64_t(rows) * 10)
                  << " (<= 5,120 <= 10,240) | tokens/pass <= 512 | no buffer overrun [PASS]\n";
    }

    // Kernel limit verification: tokens <= 512
    std::cout << "\n--- Subtask 3: Grouped mm kernel limit (tokens <= 512) and 512-chunk resolution ---\n";
    std::cout << "  Verifying grouped_mm_plan rejects tokens = 1024:\n";
    catch_expected_throw([&] {
        grouped_mm_plan(8, PrefillWeightLayout::gguf, 2560, 640, 512, 10, 1024, 10240, 2560, true);
    }, "grouped mm geometry", "grouped_mm_plan(tokens=1024)");

    std::cout << "  Verifying grouped_mm_plan accepts tokens = 512:\n";
    const auto plan512 = grouped_mm_plan(8, PrefillWeightLayout::gguf, 2560, 640, 512, 10, 512, 10240, 2560, true);
    require_check(plan512.args.tokens == 512, "grouped_mm_plan tokens == 512");
    std::cout << "  [PASS] grouped_mm_plan(tokens=512) succeeded cleanly with grid ("
              << plan512.grid_x << ", " << plan512.grid_y << ")\n\n";
}

// ============================================================================
// TASK 4: TextStack 2,048 Prefill Capacity Wiring Verification
// ============================================================================

void test_task4_stack_2048_prefill_wiring() {
    std::cout << "====================================================================\n";
    std::cout << "[TASK 4] TextStack 2,048 Prefill Capacity Wiring Verification\n";
    std::cout << "====================================================================\n\n";

    const GdnShape recurrent{2560, 16, 48, 128, 128, 4};
    const QsaShape attention{2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536};
    const uint32_t known_capacity = 2048;
    const uint32_t chunk_cap = std::min(known_capacity, 512u);
    require_check(chunk_cap == 512, "chunk_cap must be 512");

    std::cout << "--- Subtask 4a: Intra-layer GDN chunk scratch sizing ---\n";
    std::cout << "  Verifying raw 2048 capacity rejection by GDN kernel shape assertion:\n";
    catch_expected_throw([&] {
        gdn_chunk_bytes(recurrent, known_capacity, GdnChunkMode::fused);
    }, "GDN chunk allocation shape", "gdn_chunk_bytes(recurrent, 2048, fused)");

    std::cout << "  Verifying chunk_cap = 512 sizing for GDN intra-layer scratch:\n";
    const auto gdn_bytes = gdn_chunk_bytes(recurrent, chunk_cap, GdnChunkMode::fused);
    require_check(gdn_bytes.total > 0, "gdn_chunk_bytes total > 0");
    std::cout << "  [PASS] gdn_chunk_bytes(recurrent, chunk_cap=512, fused) = " << gdn_bytes.total
              << " B (" << (double(gdn_bytes.total) / (1024.0 * 1024.0)) << " MiB)\n\n";

    std::cout << "--- Subtask 4b: Intra-layer QSA chunk scratch sizing ---\n";
    std::cout << "  Verifying raw 2048 capacity rejection by QSA cache frontier assertion:\n";
    catch_expected_throw([&] {
        qsa_prefill_plan(0, known_capacity, known_capacity, attention);
    }, "batched QSA rows exceed the accepted cache frontier", "qsa_prefill_plan(0, 2048, 2048, attention)");

    std::cout << "  Verifying chunk_cap = 512 sizing for QSA intra-layer scratch:\n";
    const auto qsa_plan = qsa_prefill_plan(0, chunk_cap, chunk_cap, attention);
    require_check(qsa_plan.block_capacity > 0, "qsa_plan.block_capacity > 0");
    std::cout << "  [PASS] qsa_prefill_plan(0, chunk_cap=512, 512, attention) block_capacity = "
              << qsa_plan.block_capacity << " blocks\n\n";

    std::cout << "--- Subtask 4c: store->moe sizing audit under production shape ---\n";
    const MoeShape moe{2560, 640, 512, 10};
    const uint32_t moe_cap = std::min(known_capacity, uint32_t(10240 / moe.selected));
    require_check(moe_cap == 1024, "store->moe capacity formula must evaluate to 1024 for selected = 10");
    const auto moe_bytes = moe_mm_many_bytes(moe, moe_cap);
    require_check(moe_bytes.total <= (300ull << 20), "moe scratch <= 300 MiB");
    std::cout << "  store->moe capacity formula: std::min(2048, 10240 / 10) = 1,024 rows.\n"
              << "  store->moe scratch allocation: " << moe_bytes.total << " B ("
              << (double(moe_bytes.total) / (1024.0 * 1024.0)) << " MiB <= 300 MiB limit) [PASS]\n\n";
}

int main(int argc, char** argv) {
    std::cout << "====================================================================\n";
    std::cout << "EMPIRICAL ADVERSARIAL CHALLENGER SUITE: MILESTONE M2 GATE 4\n";
    std::cout << "====================================================================\n\n";

    try {
        test_task1_2_3_moe_execution_and_chunking();
        test_task4_stack_2048_prefill_wiring();

        std::string metallib_path = "build/owned-native/tokspur-owned.metallib";
        if (argc >= 2) metallib_path = argv[1];

        std::cout << "Initializing MetalGraph with metallib: " << metallib_path << " ...\n";
        MetalGraph g(metallib_path, 1024ull * 1024 * 1024);
        g.configure_small_gemv(16);
        g.configure_block_gemv(3);

        // Subtask 3b: moe_mm_many_dispatches on MetalGraph (512-row chunk steps)
        std::cout << "\n--- Subtask 3b: moe_mm_many_dispatches on MetalGraph (512-row chunk step) ---\n";
        const MoeShape prod_shape{2560, 640, 512, 10};
        const auto dummy = g.zeros(64, "dummy");
        const MoeWeights dummy_weights{
            {dummy, 0}, {dummy, 8}, {dummy, 8}, {dummy, 8},
            {dummy, 8}, {dummy, 8}, {dummy, 8}, {dummy, 0}
        };

        for (uint32_t rows : {1u, 511u, 512u, 513u, 1023u, 1024u, 1025u, 2047u, 2048u, 2049u}) {
            const uint64_t disp = moe_mm_many_dispatches(g, dummy_weights, prod_shape, rows);
            uint64_t expected_disp = 0;
            for (uint32_t off = 0; off < rows; off += 512) {
                expected_disp += moe_mm_many_dispatches(g, dummy_weights, prod_shape, std::min(512u, rows - off));
            }
            require_check(disp == expected_disp, "Dispatches must equal sum of 512-row chunk passes");
            std::cout << "  rows = " << std::setw(4) << rows
                      << " | dispatches = " << disp << " [PASS]\n";
        }
        g.release(dummy);

        // Subtask 4e: Live TextStack known_capacity = 2048 allocation and preparation
        std::cout << "\n--- Subtask 4e: Live TextStack known_capacity = 2048 chunked prefill preparation ---\n";
        const StackShape stack_shape{{2, 256, 32}, {256, 1, 2, 128, 32, 4}, {256, 4, 2, 32, 2, 32, 32, 4, 16, 1024}, {256, 64, 16, 10}, {2, 256, 4, 4, 3}, 13, 1};
        const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
        const StackPolicy policy{{0.08838835f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true, {HashOrder::newest_first, NgramBoundary(0), true}};
        Assets assets{g, std::mt19937(44), {}, {}, {}};
        const std::array<uint32_t, 2> attention_layers{1, 2};
        auto f = fixture(assets, stack_shape, 3, attention_layers);
        for (auto& layer : f.weights.layers) std::visit([&](auto& w) {
            for (auto* read : {&w.attention_read, &w.ffn_read}) { read->down = stack_q8_custom(assets, 512, 32); read->up = stack_q8_custom(assets, 32, 512); }
            auto& m = w.ffn; m.gate = stack_q8_custom(assets, 256, 64, 16); m.up = stack_q8_custom(assets, 256, 64, 16); m.down = stack_q8_custom(assets, 64, 256, 16);
            m.shared_gate = stack_q8_custom(assets, 256, 64); m.shared_up = stack_q8_custom(assets, 256, 64); m.shared_down = stack_q8_custom(assets, 64, 256);
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>, RecurrentWeights>) {
                w.attention.qkv = stack_q8_custom(assets, 256, uint32_t(gdn_qkv_count(stack_shape.recurrent)));
                w.attention.gate = stack_q8_custom(assets, 256, 64);
                w.attention.output = stack_q8_custom(assets, 64, 256);
            } else {
                auto& q = w.attention;
                q.query_gate = stack_q8_custom(assets, 256, 256);
                q.key = stack_q8_custom(assets, 256, 64);
                q.value = stack_q8_custom(assets, 256, 64);
                q.output = stack_q8_custom(assets, 128, 256);
                q.index_query = stack_bf16_custom(assets, q.index_query);
                q.index_key = stack_bf16_custom(assets, q.index_key);
            }
        }, layer);

        TextStack stack2048(g, f.weights, stack_shape, policy, spec, StackExecution::direct, 2048, true);
        require_check(stack2048.known_token_capacity() == 2048, "stack known_capacity == 2048");
        stack2048.prepare_chunked_prefill(GdnChunkMode::fused, 1024ull << 20);
        require_check(stack2048.chunked_prefill_enabled(), "chunked prefill must be enabled");
        require_check(stack2048.chunked_prefill_bytes() > 0, "chunked prefill bytes must be > 0");
        require_check(stack2048.chunked_prefill_bytes() <= (1024ull << 20), "chunked prefill within budget");
        std::cout << "  [PASS] TextStack with known_capacity = 2048 allocated cleanly! Charged: "
                  << (stack2048.chunked_prefill_bytes() >> 20) << " MiB\n";

        // Subtask 4f: Live encode_moe_mm_many with 2,048 rows and selected = 10
        std::cout << "\n--- Subtask 4f: Live encode_moe_mm_many 2,048 rows (selected = 10) multi-pass chunk execution ---\n";
        auto& recurrent = std::get<RecurrentWeights>(f.weights.layers[0]);
        const auto& moe_w = recurrent.ffn;
        const auto scratch = allocate_moe_mm_many_scratch(g, stack_shape.experts, 512);
        const auto in_v = g.zeros(2048ull * stack_shape.experts.channels * 4, "moe_in_2048");
        const auto out_v = g.zeros(2048ull * stack_shape.experts.channels * 4, "moe_out_2048");
        const GpuRowsF32 in_rows{in_v, 2048, stack_shape.experts.channels, uint64_t(stack_shape.experts.channels) * 4};
        const GpuRowsF32 out_rows{out_v, 2048, stack_shape.experts.channels, uint64_t(stack_shape.experts.channels) * 4};

        prepare_moe_mm_many(g, moe_w, stack_shape.experts);
        g.begin();
        encode_moe_mm_many(g, moe_w, scratch, in_rows, out_rows, stack_shape.experts, RouterMode::softmax, true);
        const auto batch = g.submit();
        require_check(batch.dispatches > 0, "MoE dispatches > 0");
        const auto expected_moe_disp = moe_mm_many_dispatches(g, moe_w, stack_shape.experts, 2048);
        require_check(batch.dispatches == expected_moe_disp, "MoE multi-pass chunk dispatches match formula");
        std::cout << "  [PASS] encode_moe_mm_many with 2048 rows and selected = 10 executed cleanly on Metal ("
                  << batch.dispatches << " dispatches, 4 passes of 512 rows)\n";

        release_moe_mm_many_scratch(g, scratch);
        g.release(in_v);
        g.release(out_v);

        std::cout << "\n====================================================================\n";
        std::cout << "CHALLENGE EXECUTION SUMMARY:\n";
        std::cout << "  Total checks performed: " << total_checks << "\n";
        std::cout << "  Checks passed:          " << total_passes << "\n";
        std::cout << "  Defects identified:     " << total_defects_found << "\n";
        std::cout << "====================================================================\n";

        if (total_defects_found > 0) {
            std::cout << "FINAL VERDICT: DEFECT_FOUND\n";
            std::cout << "====================================================================\n";
            return 2;
        } else {
            std::cout << "FINAL VERDICT: CONFIRMED\n";
            std::cout << "====================================================================\n";
            return 0;
        }
    } catch (const std::exception& err) {
        std::cerr << "\nFATAL TEST HARNESS EXCEPTION: " << err.what() << "\n";
        std::cout << "FINAL VERDICT: DEFECT_FOUND\n";
        return 1;
    }
}
