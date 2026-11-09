// SPDX-License-Identifier: Apache-2.0
#include "grouped_f32_test_support.hpp"
using namespace grouped_test;

namespace {
void reuse_library_selection(char** paths, unsigned count) {
    auto device = MTLCreateSystemDefaultDevice(); need(device != nil, "Metal unavailable");
    const unsigned production_count = count - 3;
    for (unsigned i = 0; i < count; ++i) {
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:paths[i]]] error:&error];
        need(library != nil, "reuse selection library missing");
        need([library.functionNames containsObject:@"graph_grouped_mm_f32"], "retained tile missing");
        need(bool([library.functionNames containsObject:@"graph_grouped_mm_f32_compact"]) == (i == production_count), "compact exclusion differs");
        need(bool([library.functionNames containsObject:@"graph_grouped_mm_f32_compensated"]) == (i == production_count + 1), "compensated exclusion differs");
        need(bool([library.functionNames containsObject:@"graph_grouped_mm_f32_reuse"]) == (i == production_count + 2), "reuse exclusion differs");
    }
    std::cout << "PASS grouped reuse selection checks=" << checks << '\n';
}
void summary(std::string_view phase, const Runner& runner) {
    std::cout << phase << " checks=" << checks << " dispatches=" << dispatches << " cases=" << cases << " pipelines=" << runner.cache.size();
    for (unsigned arm = 0; arm < 4; ++arm)
        std::cout << " arm=" << arm << " worst_scaled=" << std::setprecision(12) << worst[arm] << " numerical_failures=" << numerical_failures[arm];
    std::cout << " repaired=" << repaired << " changed=" << changed << " exact_compensated_reuse=true component_only=true\n";
}
}

int main(int argc, char** argv) { @autoreleasepool { try {
    if ((argc == 6 || argc == 7) && std::string_view(argv[1]) == "--library-selection") { reuse_library_selection(argv+2, unsigned(argc-2)); return 0; }
    need(argc >= 5 && argc <= 6, "usage: test-grouped_f32_reuse control.metallib compact.metallib compensated.metallib reuse.metallib [--validate|--timing|--timing-small|--smoke]");
    const std::string_view mode = argc == 6 ? argv[5] : "";
    need(mode.empty() || mode == "--validate" || mode == "--timing" || mode == "--timing-small" || mode == "--smoke", "unknown reuse mode");
    Runner runner(argv+1,mode == "--validate",4);
    if (mode == "--timing") {
        for (uint32_t tokens : {32u,128u,512u}) {
            timing(runner,{12,2560,640,16,10,tokens,true});
            timing(runner,{7,640,2560,16,10,tokens,false,tokspur::PrefillWeightLayout::q51_lanes_v1});
        }
        std::cout << "PASS grouped reuse timing checks=" << checks << " dispatches=" << dispatches << " exact_compensated_reuse=true component_only=true\n";
        return 0;
    }
    if (mode == "--timing-small") {
        for (uint32_t tokens : {1u,3u,17u}) {
            timing(runner,{12,2560,640,16,10,tokens,true});
            timing(runner,{7,640,2560,16,10,tokens,false,tokspur::PrefillWeightLayout::q51_lanes_v1});
        }
        timing(runner,{8,65536,3,2,2,3,false});
        std::cout << "PASS grouped reuse small timing checks=" << checks << " dispatches=" << dispatches << " exact_compensated_reuse=true component_only=true\n";
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
        need(cases == 218, "expanded fixture coverage changed");
        summary("expanded",runner);
        for (uint32_t type : {7u,8u,12u,13u}) for (uint32_t rows : {2u,3u,7u,8u,9u}) for (uint32_t tokens : {3u,4u,5u})
            exercise(runner,{type,256,rows,3,2,tokens,tokens%2 == 0},true,true,rows == 9);
        for (uint32_t rows : {2u,3u,7u,8u,9u}) for (uint32_t tokens : {3u,4u,5u})
            exercise(runner,{7,640,rows,3,2,tokens,tokens%2 == 0,tokspur::PrefillWeightLayout::q51_lanes_v1},true,true,rows == 9);
        need(cases == 293, "tile boundary fixture coverage changed");
        summary("boundaries",runner);
    }
    context = "grouped reuse qualification";
    need(numerical_failures[0] == numerical_failures[1], "baseline numerical counts differ");
    need(numerical_failures[2] == 0 && numerical_failures[3] == 0 && repaired == numerical_failures[0], "reuse leaves reference failures");
    std::cout << "PASS grouped FP32 reuse\n"; return 0;
} catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; } } }
