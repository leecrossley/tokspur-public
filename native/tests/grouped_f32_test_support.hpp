// SPDX-License-Identifier: Apache-2.0
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "grouped_mm_fixture.hpp"
#include "moe_projection_reference.hpp"
#include "tokspur/expert_lanes.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string_view>
#include <tuple>

namespace grouped_test {
uint64_t checks = 0, dispatches = 0, cases = 0;
std::array<uint64_t, 4> numerical_failures{};
uint64_t changed = 0, repaired = 0;
std::array<double, 4> worst{};
std::string context;
constexpr uint32_t poison = 0x5a5a5a5a;
void need(bool ok, const char* reason) { ++checks; if (!ok) throw std::runtime_error(context + ": " + reason); }
struct Buffer {
    id<MTLBuffer> object;
    size_t size;
    Buffer(id<MTLDevice> device, size_t bytes) : object([device newBufferWithLength:bytes + 32 options:MTLResourceStorageModeShared]), size(bytes) {
        need(object != nil, "buffer allocation"); std::memset(object.contents, 0x5a, bytes + 32);
    }
    void* data() const { return static_cast<std::byte*>(object.contents) + 16; }
    void write(const void* source) { std::memcpy(data(), source, size); }
    void reset() { std::memset(data(), 0x5a, size); }
    std::vector<std::byte> bytes() const { const auto* p = static_cast<const std::byte*>(data()); return {p, p + size}; }
    void guards() const {
        const auto* p = static_cast<const uint8_t*>(object.contents);
        for (unsigned i = 0; i < 16; ++i) need(p[i] == 0x5a && p[size + 16 + i] == 0x5a, "buffer canary changed");
    }
};
struct Binding { Buffer* buffer; unsigned index; };
struct Runner {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLLibrary> libraries[4];
    Buffer flag{device, 4};
    bool validation;
    unsigned arms;
    double gpu_ms = 0;
    std::map<std::tuple<unsigned, uint32_t, uint32_t>, id<MTLComputePipelineState>> cache;
    Runner(char** paths, bool validate, unsigned count = 2) : validation(validate), arms(count) {
        need(device != nil && queue != nil, "Metal unavailable");
        std::cout << "device=" << device.name.UTF8String << " max_threadgroup_bytes=" << device.maxThreadgroupMemoryLength << '\n';
        for (unsigned i = 0; i < arms; ++i) {
            NSError* error = nil;
            libraries[i] = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:paths[i]]] error:&error];
            need(libraries[i] != nil, "library missing");
        }
        need(![libraries[0].functionNames containsObject:@"graph_grouped_mm_f32_compact"], "control library contains compact experiment");
        need([libraries[1].functionNames containsObject:@"graph_grouped_mm_f32_compact"], "candidate library lacks compact experiment");
        if (arms >= 3) {
            need([libraries[2].functionNames containsObject:@"graph_grouped_mm_f32_compensated"], "precision library lacks compensated experiment");
            need(![libraries[2].functionNames containsObject:@"graph_grouped_mm_f32_compact"], "precision library includes compact experiment");
        }
        if (arms == 4) {
            need([libraries[3].functionNames containsObject:@"graph_grouped_mm_f32_reuse"], "reuse library lacks candidate");
            need(![libraries[3].functionNames containsObject:@"graph_grouped_mm_f32_compact"], "reuse library includes compact experiment");
            need(![libraries[3].functionNames containsObject:@"graph_grouped_mm_f32_compensated"], "reuse library includes scalar compensation");
        }
    }
    id<MTLComputePipelineState> pipeline(unsigned arm, uint32_t type, uint32_t layout) {
        const auto key = std::make_tuple(arm, type, layout);
        if (cache.contains(key)) return cache.at(key);
        auto values = [MTLFunctionConstantValues new];
        [values setConstantValue:&type type:MTLDataTypeUInt atIndex:0];
        [values setConstantValue:&layout type:MTLDataTypeUInt atIndex:1];
        NSError* error = nil;
        NSString* name = arm == 0 ? @"graph_grouped_mm_f32" : arm == 1 ? @"graph_grouped_mm_f32_compact" : arm == 2 ? @"graph_grouped_mm_f32_compensated" : arm == 3 ? @"graph_grouped_mm_f32_reuse" : arm == 4 ? @"graph_grouped_mm_f32_decode" : @"graph_grouped_mm_f32_decode16";
        auto function = [libraries[arm < 4 ? arm : 0] newFunctionWithName:name constantValues:values error:&error];
        need(function != nil, "function missing");
        auto descriptor = [MTLComputePipelineDescriptor new]; descriptor.computeFunction = function;
        if (validation) descriptor.shaderValidation = MTLShaderValidationEnabled;
        auto state = [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!state) std::cerr << error.localizedDescription.UTF8String << '\n';
        need(state != nil && state.threadExecutionWidth == 32 && state.maxTotalThreadsPerThreadgroup >= 128, "pipeline geometry");
        const uint32_t staged = arm == 0 ? 12288 : arm == 1 ? 6144 : 0;
        need(state.staticThreadgroupMemoryLength + staged <= device.maxThreadgroupMemoryLength, "threadgroup memory limit");
        if (validation) need(state.shaderValidation == MTLShaderValidationEnabled, "shader validation disabled");
        std::cout << "arm=" << arm << " type=" << type << " layout=" << layout << " static_bytes=" << state.staticThreadgroupMemoryLength
                  << " staged_bytes=" << staged << " shader_validation=" << state.shaderValidation << '\n';
        cache[key] = state; return state;
    }
    uint32_t run(unsigned arm, uint32_t type, uint32_t layout, const tokspur::GroupedMmArgs& args, std::initializer_list<Binding> bindings, bool oversized = true) {
        *static_cast<uint32_t*>(flag.data()) = 0;
        auto state = pipeline(arm, type, layout);
        auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
        need(command != nil && encoder != nil, "command allocation");
        [encoder setComputePipelineState:state];
        for (const auto& b : bindings) [encoder setBuffer:b.buffer->object offset:16 atIndex:b.index];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder setBuffer:flag.object offset:16 atIndex:7];
        if (arm < 2) [encoder setThreadgroupMemoryLength:arm ? 6144 : 12288 atIndex:0];
        const uint32_t rows = arm == 1 ? 32 : arm == 3 ? 8 : 64;
        const uint32_t panels = arm == 1 ? 2 : arm == 3 ? (std::min(32u,args.tokens)+3)/4 : 1;
        [encoder dispatchThreadgroups:MTLSizeMake((args.rows + rows - 1) / rows + oversized, args.maximum_tasks + oversized, panels + (oversized && (arm == 1 || arm == 3))) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted]; ++dispatches;
        if (command.status != MTLCommandBufferStatusCompleted) std::cerr << command.error.localizedDescription.UTF8String << '\n';
        need(command.status == MTLCommandBufferStatusCompleted, "command failed");
        gpu_ms = (command.GPUEndTime-command.GPUStartTime)*1000;
        for (const auto& b : bindings) b.buffer->guards(); flag.guards();
        return *static_cast<const uint32_t*>(flag.data());
    }
    void decoders(uint32_t type, uint32_t layout, Buffer& weights, uint32_t columns, uint32_t rows, uint32_t row_bytes, std::span<const float> expected) {
        Buffer output(device, expected.size_bytes());
        const std::array<uint32_t,4> shape{columns,rows,row_bytes,0};
        for (unsigned arm : {4u,5u}) {
            auto state = pipeline(arm,type,layout);
            *static_cast<uint32_t*>(flag.data()) = 0;
            auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
            need(command != nil && encoder != nil, "decoder command allocation");
            [encoder setComputePipelineState:state]; [encoder setBuffer:weights.object offset:16 atIndex:0];
            [encoder setBuffer:output.object offset:16 atIndex:1]; [encoder setBytes:shape.data() length:16 atIndex:2];
            [encoder setBuffer:flag.object offset:16 atIndex:7];
            const size_t elements = expected.size()/(arm == 4 ? 1 : 16);
            [encoder dispatchThreads:MTLSizeMake(elements+32,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted]; ++dispatches;
            need(command.status == MTLCommandBufferStatusCompleted && *static_cast<uint32_t*>(flag.data()) == 0, "decoder command failed");
            need(std::memcmp(output.data(),expected.data(),expected.size_bytes()) == 0, "GPU decoder differs from independent represented weights");
            output.guards(); weights.guards(); flag.guards();
        }
    }
};

// Stable expert lists are built from flat route IDs. The oracle never uses the GPU task builder or tile ordering.
struct Routes {
    std::vector<uint32_t> ids, map, counts, tasks;
    Routes(const tokspur::GroupedMmArgs& a, bool skew) : ids(a.tokens*a.selected), map(a.experts*a.tokens, poison), counts(a.experts), tasks(2+2*a.maximum_tasks, poison) {
        for (uint32_t token = 0; token < a.tokens; ++token) for (uint32_t slot = 0; slot < a.selected; ++slot)
            ids[token*a.selected+slot] = (a.experts-1 - (skew ? slot : (token*17+slot)%a.experts));
        std::vector<std::vector<uint32_t>> grouped(a.experts);
        for (uint32_t rid = 0; rid < ids.size(); ++rid) grouped[ids[rid]].push_back(rid);
        tasks[0] = 0; tasks[1] = uint32_t(ids.size());
        for (uint32_t expert = 0; expert < a.experts; ++expert) {
            counts[expert] = uint32_t(grouped[expert].size());
            need(counts[expert] <= a.tokens, "duplicate expert in token");
            std::copy(grouped[expert].begin(), grouped[expert].end(), map.begin()+size_t(expert)*a.tokens);
            for (uint32_t start = 0; start < counts[expert]; start += 32) {
                const auto task = tasks[0]++;
                need(task < a.maximum_tasks, "task capacity");
                tasks[2+2*task] = expert; tasks[3+2*task] = start;
            }
        }
        for (uint32_t i = 0; i < tasks[0]/2; ++i) {
            std::swap(tasks[2+2*i], tasks[2+2*(tasks[0]-1-i)]);
            std::swap(tasks[3+2*i], tasks[3+2*(tasks[0]-1-i)]);
        }
    }
};

void exercise(Runner& runner, grouped_fixture::Shape shape, bool skew, bool stress, bool faults, unsigned pattern = 0) {
    const auto [type, columns, rows, experts, selected, tokens, broadcast, layout] = shape;
    context = "type=" + std::to_string(type) + " columns=" + std::to_string(columns) + " rows=" + std::to_string(rows) + " tokens=" + std::to_string(tokens);
    const uint64_t xs = (uint64_t(columns)+4)*4, ys = ((uint64_t(rows)+3)/4*4+4)*4;
    auto plan = tokspur::grouped_mm_plan(type, layout, columns, rows, experts, selected, tokens, xs, ys, broadcast);
    Routes routes(plan.args, skew);
    auto raw = grouped_fixture::weights(shape), payload = raw;
    if (layout == tokspur::PrefillWeightLayout::q51_lanes_v1) tokspur::lanes::pack_rows(tokspur::lanes::row_layout(type, columns), raw, payload);
    std::vector<float> represented(size_t(experts)*rows*columns), existing(represented.size());
    tokspur::dequantize(type, raw, existing);
    for (size_t i = 0; i < represented.size(); ++i) {
        represented[i] = moe_projection_reference::represented(type, raw.data()+(i/columns)*plan.args.row_bytes, uint32_t(i%columns));
        need(represented[i] == existing[i], "independent represented decoder differs");
    }
    const size_t input_rows = broadcast ? tokens : tokens*selected;
    std::vector<float> input(input_rows*xs/4, NAN);
    uint32_t seed = 81;
    for (size_t row = 0; row < input_rows; ++row) for (uint32_t c = 0; c < columns; ++c) {
        seed = seed*1664525u+1013904223u;
        input[row*xs/4+c] = float(int(seed%511)-255)/256/(stress ? 1 : 64);
        if (pattern == 1) input[row*xs/4+c] *= 16;
        if (pattern == 2) input[row*xs/4+c] = c%64 < 32 ? 1 : -1;
        if (pattern == 3) input[row*xs/4+c] = 0;
        if (pattern == 4) input[row*xs/4+c] *= c%32 == 0 ? 1024 : 1.0f/1024;
    }
    std::vector<double> expected(size_t(tokens)*selected*rows);
    for (size_t rid = 0; rid < routes.ids.size(); ++rid) for (uint32_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (uint32_t c = 0; c < columns; ++c) sum += double(represented[(size_t(routes.ids[rid])*rows+row)*columns+c])*input[(broadcast ? rid/selected : rid)*xs/4+c];
        expected[rid*rows+row] = sum;
    }
    Buffer weights(runner.device, payload.size()), x(runner.device, input.size()*4), y(runner.device, size_t(tokens)*selected*ys+16);
    Buffer ids(runner.device, routes.ids.size()*4), map(runner.device, routes.map.size()*4), counts(runner.device, routes.counts.size()*4), tasks(runner.device, routes.tasks.size()*4);
    weights.write(payload.data()); x.write(input.data()); ids.write(routes.ids.data()); map.write(routes.map.data()); counts.write(routes.counts.data()); tasks.write(routes.tasks.data());
    if (runner.arms >= 3) runner.decoders(type,uint32_t(layout),weights,columns,rows*experts,plan.args.row_bytes,represented);
    const auto immutable = std::array{weights.bytes(), x.bytes(), ids.bytes(), map.bytes(), counts.bytes(), tasks.bytes()};
    std::vector<std::byte> baseline, precision_baseline;
    std::vector<bool> baseline_failures(expected.size());
    for (uint32_t swizzle : {0u, 2u}) for (unsigned arm = 0; arm < runner.arms; ++arm) {
        auto args = plan.args; args.broadcast |= swizzle;
        const auto run = [&](uint32_t encoding = 0, uint32_t storage = UINT32_MAX) {
            return runner.run(arm, encoding ? encoding : type, storage == UINT32_MAX ? uint32_t(layout) : storage, args,
                {{&weights, 0}, {&x, 1}, {&y, 2}, {&ids, 4}, {&map, 6}, {&tasks, 8}, {&counts, 9}});
        };
        y.reset(); need(run() == 0, "valid tile reported error");
        const auto result = y.bytes();
        if (!swizzle && !arm) baseline = result;
        else if (arm < 2) need(result == baseline, "compact or swizzled output differs from retained bytes");
        else if (!swizzle && arm == 2) precision_baseline = result;
        else need(result == precision_baseline, "precision candidate or swizzle output differs");
        const auto* values = static_cast<const float*>(y.data());
        for (size_t rid = 0; rid < routes.ids.size(); ++rid) for (uint32_t row = 0; row < ys/4; ++row) {
            if (row >= rows) { uint32_t bits; std::memcpy(&bits, values+rid*ys/4+row, 4); need(bits == poison, "output padding changed"); continue; }
            const double reference = expected[rid*rows+row], actual = values[rid*ys/4+row];
            need(std::isfinite(actual) && std::isfinite(reference), "nonfinite reference or output");
            const double error = std::abs(actual-reference)/std::max(1.0,std::abs(reference));
            worst[arm] = std::max(worst[arm],error);
            if (!swizzle && error > 5e-5) {
                ++numerical_failures[arm];
                if (numerical_failures[arm] <= 20) std::cerr << context << " arm=" << arm << " rid=" << rid << " row=" << row << " error=" << error << '\n';
            }
            if (!swizzle && !arm) baseline_failures[rid*rows+row] = error > 5e-5;
            if (arm == 2 && !swizzle) {
                changed += std::memcmp(result.data()+rid*ys+row*4,baseline.data()+rid*ys+row*4,4) != 0;
                repaired += baseline_failures[rid*rows+row] && error <= 5e-5;
            }
            if (arm >= 2 || (!stress && runner.arms == 2)) need(error <= 5e-5, "represented double error exceeds 5e-5");
        }
        for (size_t i = y.size-16; i < y.size; ++i) need(static_cast<const uint8_t*>(y.data())[i] == 0x5a, "output tail changed");
        y.reset(); need(run() == 0 && y.bytes() == result, "repeat differs");
        if (faults) {
            for (unsigned fault = 0; fault < 8; ++fault) {
                auto bad_tasks = routes.tasks, bad_map = routes.map, bad_counts = routes.counts;
                const uint32_t expert = bad_tasks[2], at = bad_tasks[3];
                if (fault == 0) bad_tasks[0] = args.maximum_tasks+1;
                if (fault == 1) ++bad_tasks[1];
                if (fault == 2) bad_tasks[2] = experts;
                if (fault == 3) bad_tasks[3] = tokens;
                if (fault == 4) bad_counts[expert] = tokens+1;
                if (fault == 5) bad_counts[expert] = 0;
                if (fault == 6) bad_map[size_t(expert)*tokens+at] = tokens*selected;
                if (fault == 7) bad_map[size_t(expert)*tokens+at] = (bad_map[size_t(expert)*tokens+at]+1)%(tokens*selected);
                tasks.write(bad_tasks.data()); map.write(bad_map.data()); counts.write(bad_counts.data());
                need(run() & 4, "invalid metadata not reported");
            }
            tasks.write(routes.tasks.data()); map.write(routes.map.data()); counts.write(routes.counts.data());
            need(run() == 0 && y.bytes() == result, "metadata recovery differs");
            for (float invalid : {INFINITY, NAN}) {
                auto bad = input; bad[0] = invalid; x.write(bad.data()); need(run() & 1, "nonfinite input not reported");
            }
            x.write(input.data()); need(run() == 0 && y.bytes() == result, "input recovery differs");
            for (uint8_t high : {0x7cu, 0x7eu}) {
                auto bad = payload; const auto offset = size_t(routes.ids[0])*plan.args.expert_stride_bytes;
                bad[offset] = std::byte{0}; bad[offset+1] = std::byte{high}; weights.write(bad.data()); need(run() & 1, "nonfinite weight not reported");
            }
            weights.write(payload.data()); need(run() == 0 && y.bytes() == result, "weight recovery differs");
            auto empty = routes.tasks; empty[0] = 0; tasks.write(empty.data()); y.reset(); const auto untouched = y.bytes();
            need(run() == 0 && y.bytes() == untouched, "empty tasks wrote output");
            tasks.write(routes.tasks.data());
            need(run(9) & 4, "unsupported encoding accepted");
            need(run(0,1) & 4, "unsupported layout accepted");
            need(run() == 0 && y.bytes() == result, "empty or invalid layout recovery differs");
        }
    }
    need(immutable == std::array{weights.bytes(), x.bytes(), ids.bytes(), map.bytes(), counts.bytes(), tasks.bytes()}, "input or routing storage changed");
    ++cases;
}

void timing(Runner& runner, grouped_fixture::Shape shape) {
    const auto [type, columns, rows, experts, selected, tokens, broadcast, layout] = shape;
    context = "timing type=" + std::to_string(type) + " tokens=" + std::to_string(tokens);
    const uint64_t output_stride = ((uint64_t(rows)+3)/4)*16;
    const auto plan = tokspur::grouped_mm_plan(type, layout, columns, rows, experts, selected, tokens, uint64_t(columns)*4, output_stride, broadcast);
    Routes routes(plan.args, false);
    auto raw = grouped_fixture::weights(shape), payload = raw;
    if (layout == tokspur::PrefillWeightLayout::q51_lanes_v1) tokspur::lanes::pack_rows(tokspur::lanes::row_layout(type, columns), raw, payload);
    std::vector<float> input(plan.input_bytes/4);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i*17)%511)-255)/16384;
    Buffer weights(runner.device, payload.size()), x(runner.device, input.size()*4), y(runner.device, size_t(tokens)*selected*output_stride);
    Buffer ids(runner.device, routes.ids.size()*4), map(runner.device, routes.map.size()*4), counts(runner.device, routes.counts.size()*4), tasks(runner.device, routes.tasks.size()*4);
    weights.write(payload.data()); x.write(input.data()); ids.write(routes.ids.data()); map.write(routes.map.data()); counts.write(routes.counts.data()); tasks.write(routes.tasks.data());
    const auto immutable = std::array{weights.bytes(), x.bytes(), ids.bytes(), map.bytes(), counts.bytes(), tasks.bytes()};
    std::vector<std::byte> baseline, precision_baseline;
    for (uint32_t swizzle : {0u, 2u}) {
        std::array<std::vector<double>, 4> samples;
        auto args = plan.args; args.broadcast |= swizzle;
        for (unsigned round = 0; round < 14; ++round) for (unsigned side = 0; side < runner.arms; ++side) {
            const unsigned arm = round%2 ? runner.arms-1-side : side;
            y.reset();
            need(runner.run(arm, type, uint32_t(layout), args, {{&weights, 0}, {&x, 1}, {&y, 2}, {&ids, 4}, {&map, 6}, {&tasks, 8}, {&counts, 9}}, false) == 0, "timed tile reported error");
            const auto result = y.bytes();
            if (baseline.empty()) baseline = result;
            else if (arm < 2) need(result == baseline, "timed output bytes differ");
            else if (precision_baseline.empty()) precision_baseline = result;
            else need(result == precision_baseline, "timed precision candidate or repeat differs");
            for (size_t rid = 0; rid < routes.ids.size(); ++rid) for (uint32_t row = rows; row < output_stride/4; ++row) {
                uint32_t bits;
                std::memcpy(&bits,result.data()+rid*output_stride+row*4,4);
                need(bits == poison,"timed output padding changed");
            }
            need(std::isfinite(runner.gpu_ms) && runner.gpu_ms > 0, "GPU timestamps unavailable");
            if (round >= 3) samples[arm].push_back(runner.gpu_ms);
        }
        for (unsigned arm = 0; arm < runner.arms; ++arm) {
            std::sort(samples[arm].begin(), samples[arm].end());
            std::cout << "timing type=" << type << " columns=" << columns << " rows=" << rows << " tokens=" << tokens
                      << " layout=" << uint32_t(layout) << " swizzle=" << (swizzle != 0) << " arm=" << arm << " samples=" << samples[arm].size()
                      << " min_gpu_ms=" << samples[arm].front() << " median_gpu_ms=" << samples[arm][samples[arm].size()/2] << " max_gpu_ms=" << samples[arm].back() << '\n';
        }
    }
    need(immutable == std::array{weights.bytes(), x.bytes(), ids.bytes(), map.bytes(), counts.bytes(), tasks.bytes()}, "timed input storage changed");
}

void library_selection(char** paths) {
    auto device = MTLCreateSystemDefaultDevice(); need(device != nil, "Metal unavailable");
    for (unsigned i = 0; i < 3; ++i) {
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:paths[i]]] error:&error];
        need(library != nil, "selection library missing");
        for (NSString* name in @[@"graph_grouped_mm_f32", @"graph_grouped_mm_f32_decode", @"graph_grouped_mm_f32_decode16"])
            need([library.functionNames containsObject:name], "retained grouped function missing");
        need(bool([library.functionNames containsObject:@"graph_grouped_mm_f32_compact"]) == (i == 2), "compact production exclusion differs");
    }
    std::cout << "PASS grouped library selection checks=" << checks << '\n';
}

void known_cases(Runner& runner, bool stress, bool smoke = false) {
    const std::string_view mode = smoke ? "--smoke" : "";
    for (uint32_t type : {7u, 8u, 12u, 13u}) {
        if (mode == "--smoke" && type != 8) continue;
        exercise(runner, {type,256,65,7,2,33,true}, false, stress, true);
        if (mode == "--smoke") continue;
        const std::array<uint32_t, 15> tokens{1,15,16,17,31,32,33,63,64,65,255,256,257,511,512};
        const std::array<uint32_t, 13> rows{1,15,16,17,31,32,33,63,64,65,127,128,129};
        for (size_t i = 0; i < tokens.size(); ++i) for (bool broadcast : {false,true}) {
            const uint32_t columns = type >= 12 ? 256 : i%3 == 0 ? 32 : i%3 == 1 ? 64 : 96;
            exercise(runner, {type,columns,rows[i%rows.size()],7,2,tokens[i],broadcast}, i%2 == 0, stress, false);
        }
        for (uint32_t columns : {512u,768u,2560u,32768u,65536u})
            exercise(runner, {type,columns,3,2,2,3,false}, true, stress, false);
        exercise(runner, {type,256,1,512,16,512,true}, false, stress, false);
        exercise(runner, {type,256,32768,1,1,1,true}, false, stress, false);
    }
    if (mode != "--smoke") for (uint32_t tokens : {1u,16u,17u,32u,33u,65u,512u}) for (bool broadcast : {false,true})
        exercise(runner, {7,640,65,7,2,tokens,broadcast,tokspur::PrefillWeightLayout::q51_lanes_v1}, tokens%2 == 0, stress, tokens == 33);
}

}
