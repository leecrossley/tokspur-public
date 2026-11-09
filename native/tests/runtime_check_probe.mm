#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <initializer_list>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <array>
#include <vector>

void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    @autoreleasepool {
    try {
        need(argc == 2 || argc == 3, "probe requires its separate metallib and optional serving arithmetic reference");
        auto device = MTLCreateSystemDefaultDevice(); need(device != nil, "Metal device missing");
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        need(library != nil && library.functionNames.count == 4, "probe library is not isolated");
        id<MTLLibrary> serving = nil;
        if (argc == 3) {
            serving = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];
            need(serving != nil && [serving newFunctionWithName:@"probe_unchecked"] == nil, "invalid serving arithmetic reference");
        }
        auto queue = [device newCommandQueue]; need(queue != nil, "probe queue failed");
        constexpr uint32_t count = 1u << 20, dispatches = 32;
        auto input = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
        auto output = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
        auto flag = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        need(input && output && flag, "probe buffer allocation failed");
        auto* values = static_cast<float*>(input.contents);
        for (uint32_t i = 0; i < count; ++i) values[i] = float(int(i % 257) - 128) / 128;
        // Both arms are admitted only after an independent complete CPU finite scan.
        for (uint32_t i = 0; i < count; ++i) need(std::isfinite(values[i]), "unchecked probe refused invalid input");
        for (const bool checked : {true, false, false, true}) {
            auto function = [library newFunctionWithName:checked ? @"probe_checked" : @"probe_unchecked"];
            auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            need(pipeline != nil, "probe pipeline failed");
            *static_cast<uint32_t*>(flag.contents) = 0;
            auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
            need(command && encoder, "probe command allocation failed");
            [encoder setComputePipelineState:pipeline]; [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:1]; [encoder setBuffer:flag offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            for (uint32_t i = 0; i < dispatches; ++i) {
                [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            need(command.status == MTLCommandBufferStatusCompleted && !*static_cast<uint32_t*>(flag.contents), "probe submission or finite check failed");
            need(std::memcmp(input.contents, output.contents, count * sizeof(float)) == 0, "probe output differs");
            const double milliseconds = (command.GPUEndTime - command.GPUStartTime) * 1000;
            need(std::isfinite(milliseconds) && milliseconds > 0, "invalid probe duration");
            need(std::printf("{\"probe_only\":true,\"checked\":%s,\"dispatches\":%u,\"gpu_ms\":%.17g,\"serving_qualification\":false}\n", checked ? "true" : "false", dispatches, milliseconds) > 0, "probe output failed");
        }
        auto gains = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
        need(gains != nil, "probe gain allocation failed");
        auto* gain_values = static_cast<float*>(gains.contents);
        for (uint32_t i = 0; i < count; ++i) gain_values[i] = .75f + float(i % 97) / 194;
        // Independent magnitude bounds prove finite square sums, a positive
        // denominator, inverse and products for every declared generated shape.
        for (uint32_t i = 0; i < count; ++i)
            need(std::isfinite(values[i]) && std::abs(values[i]) <= 1 &&
                std::isfinite(gain_values[i]) && std::abs(gain_values[i]) <= 2, "unchecked RMS probe finite-domain refusal");
        for (uint32_t width : {128u, 2560u, 10240u}) for (uint32_t groups : {1u, 4u, 32u}) for (uint32_t shared : {0u, 1u}) {
            const std::array<uint32_t, 4> shape{width * groups, width, shared, 0}; const float epsilon = 1e-6f;
            need(shape[0] <= count && shape[1] && shape[0] % shape[1] == 0, "RMS probe structural domain refusal");
            std::vector<float> reference(shape[0]);
            const auto run = [&](id<MTLLibrary> source, NSString* name, uint32_t iterations) {
                NSError* run_error = nil;
                auto function = [source newFunctionWithName:name];
                auto pipeline = [device newComputePipelineStateWithFunction:function error:&run_error];
                need(pipeline != nil, "RMS probe pipeline failed");
                *static_cast<uint32_t*>(flag.contents) = 0;
                auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
                need(command && encoder, "RMS probe command allocation failed");
                [encoder setComputePipelineState:pipeline]; [encoder setBuffer:input offset:0 atIndex:0];
                [encoder setBuffer:gains offset:0 atIndex:1]; [encoder setBuffer:output offset:0 atIndex:2];
                [encoder setBytes:shape.data() length:sizeof(shape) atIndex:3];
                [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4]; [encoder setBuffer:flag offset:0 atIndex:7];
                for (uint32_t i = 0; i < iterations; ++i) {
                    [encoder dispatchThreadgroups:MTLSizeMake((groups + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
                need(command.status == MTLCommandBufferStatusCompleted && !*static_cast<uint32_t*>(flag.contents), "RMS probe submission or finite check failed");
                const double ms = (command.GPUEndTime - command.GPUStartTime) * 1000;
                need(std::isfinite(ms) && ms > 0, "RMS probe timing failed"); return ms;
            };
            run(serving ? serving : library, serving ? @"graph_rms" : @"probe_rms_checked", 1);
            std::memcpy(reference.data(), output.contents, shape[0] * sizeof(float));
            for (const float value : reference) need(std::isfinite(value), "independent checked RMS output is invalid");
            for (const bool checked : {true, false, false, true}) {
                const double ms = run(library, checked ? @"probe_rms_checked" : @"probe_rms_unchecked", dispatches);
                need(std::memcmp(reference.data(), output.contents, shape[0] * sizeof(float)) == 0, "RMS probe arithmetic differs from reference");
                need(std::printf("{\"probe_only\":true,\"family\":\"graph_rms\",\"width\":%u,\"groups\":%u,\"shared_gains\":%u,\"checked\":%s,\"dispatches\":%u,\"gpu_ms\":%.17g,\"serving_reference\":%s,\"serving_qualification\":false}\n", width, groups, shared, checked ? "true" : "false", dispatches, ms, serving ? "true" : "false") > 0, "RMS probe output failed");
            }
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    }
}
