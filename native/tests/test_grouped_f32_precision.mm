// SPDX-License-Identifier: Apache-2.0
#include "grouped_f32_test_support.hpp"
using namespace grouped_test;

namespace {
void precision_library_selection(char** paths) {
    auto device = MTLCreateSystemDefaultDevice(); need(device != nil, "Metal unavailable");
    for (unsigned i = 0; i < 4; ++i) {
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:paths[i]]] error:&error];
        need(library != nil, "precision selection library missing");
        need([library.functionNames containsObject:@"graph_grouped_mm_f32"], "retained tile missing");
        need(bool([library.functionNames containsObject:@"graph_grouped_mm_f32_compact"]) == (i == 2), "compact exclusion differs");
        need(bool([library.functionNames containsObject:@"graph_grouped_mm_f32_compensated"]) == (i == 3), "compensated exclusion differs");
    }
    std::cout << "PASS grouped precision selection checks=" << checks << '\n';
}
void summary(std::string_view phase, const Runner& runner) {
    std::cout << phase << " checks=" << checks << " dispatches=" << dispatches << " cases=" << cases << " pipelines=" << runner.cache.size()
              << " worst_scaled=" << std::setprecision(12) << worst[0] << ',' << worst[1] << ',' << worst[2]
              << " numerical_failures=" << numerical_failures[0] << ',' << numerical_failures[1] << ',' << numerical_failures[2]
              << " repaired=" << repaired << " changed=" << changed << " exact_control_compact=true component_only=true\n";
}
}

int main(int argc, char** argv) { @autoreleasepool { try {
    if (argc == 6 && std::string_view(argv[1]) == "--library-selection") { precision_library_selection(argv+2); return 0; }
    need(argc >= 4 && argc <= 5, "usage: test-grouped_f32_precision control.metallib compact.metallib compensated.metallib [--validate|--timing|--smoke]");
    const std::string_view mode = argc == 5 ? argv[4] : "";
    need(mode.empty() || mode == "--validate" || mode == "--timing" || mode == "--smoke", "unknown precision mode");
    Runner runner(argv+1,mode == "--validate",3);
    if (mode == "--timing") {
        for (uint32_t tokens : {32u,128u,512u}) {
            timing(runner,{12,2560,640,16,10,tokens,true});
            timing(runner,{7,640,2560,16,10,tokens,false,tokspur::PrefillWeightLayout::q51_lanes_v1});
        }
        std::cout << "PASS grouped precision timing checks=" << checks << " dispatches=" << dispatches << " repeat_exact=true component_only=true\n";
        return 0;
    }
    if (mode == "--smoke") {
        exercise(runner,{13,256,65,7,2,33,true},false,true,true);
        exercise(runner,{8,65536,3,2,2,3,false},true,true,true);
        summary("smoke",runner);
    } else {
        known_cases(runner,true);
        need(cases == 166, "known fixture coverage changed");
        summary("known",runner);
        for (uint32_t type : {7u,8u,12u,13u}) for (unsigned pattern = 1; pattern <= 4; ++pattern) {
            exercise(runner,{type,type < 12 ? 32u : 256u,33,7,3,17,pattern%2 == 0},false,true,pattern == 1,pattern);
            exercise(runner,{type,2560,17,7,2,33,pattern%2 == 0},false,true,false,pattern);
            exercise(runner,{type,65536,3,3,2,3,pattern%2 == 0},false,true,false,pattern);
        }
        for (unsigned pattern = 1; pattern <= 4; ++pattern)
            exercise(runner,{7,640,33,7,3,33,pattern%2 == 0,tokspur::PrefillWeightLayout::q51_lanes_v1},false,true,false,pattern);
        summary("expanded",runner);
    }
    context = "grouped precision qualification";
    need(numerical_failures[0] == numerical_failures[1], "baseline numerical counts differ");
    need(numerical_failures[2] == 0 && repaired == numerical_failures[0], "compensation leaves reference failures");
    std::cout << "PASS grouped FP32 precision\n"; return 0;
} catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; } } }
