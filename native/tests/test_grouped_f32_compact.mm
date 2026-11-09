// SPDX-License-Identifier: Apache-2.0
#include "grouped_f32_test_support.hpp"
using namespace grouped_test;

int main(int argc, char** argv) { @autoreleasepool { try {
    if (argc == 5 && std::string_view(argv[1]) == "--library-selection") { library_selection(argv+2); return 0; }
    need(argc >= 3 && argc <= 4, "usage: test-grouped_f32_compact control.metallib candidate.metallib [--validate|--stress|--smoke|--timing]");
    const std::string_view mode = argc == 4 ? argv[3] : "";
    need(mode.empty() || mode == "--validate" || mode == "--stress" || mode == "--smoke" || mode == "--timing", "unknown mode");
    Runner runner(argv+1, mode == "--validate");
    if (mode == "--timing") {
        for (uint32_t tokens : {32u,128u,512u}) {
            timing(runner, {12,2560,640,16,10,tokens,true});
            timing(runner, {7,640,2560,16,10,tokens,false,tokspur::PrefillWeightLayout::q51_lanes_v1});
        }
        std::cout << "PASS grouped timing checks=" << checks << " dispatches=" << dispatches << " byte_exact=true component_only=true\n";
        return 0;
    }
    const bool stress = mode == "--stress";
    known_cases(runner,stress,mode == "--smoke");
    context = stress ? "full-range grouped stress" : "bounded grouped comparison";
    std::cout << "checks=" << checks << " dispatches=" << dispatches << " cases=" << cases << " pipelines=" << runner.cache.size()
              << " worst_scaled=" << std::setprecision(12) << worst[0] << ',' << worst[1]
              << " numerical_failures=" << numerical_failures[0] << ',' << numerical_failures[1]
              << " byte_exact=true component_only=true\n";
    need(numerical_failures[0] == numerical_failures[1], "numerical failure counts differ");
    need(numerical_failures[0] == 0 && numerical_failures[1] == 0, "double-reference stress failures");
    std::cout << "PASS grouped FP32 compact\n"; return 0;
} catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; } } }
