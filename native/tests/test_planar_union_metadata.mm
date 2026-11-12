#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t poison = 0xbaadf00d;
using Metadata = std::array<uint32_t, 2320>;
struct Shape {
    uint32_t experts, selected, tokens, hidden, channels;
    uint32_t reserved[3]{};
};

#if defined(TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS) && TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS
constexpr uint32_t gate_tile = 4, down_tile = 8;
#else
constexpr uint32_t gate_tile = 8, down_tile = 16;
#endif

// Independent row-major reference. New records appear on first use, including
// a later use of an expert whose previous record reached capacity.
Metadata reference(Shape s, const std::vector<uint32_t>& ids, bool valid) {
    Metadata meta;
    meta.fill(poison);
    std::fill_n(meta.begin(), 8, 0u);
    if (!valid) return meta;
    uint32_t records = 0;
    const uint32_t capacity = s.tokens <= 4 ? 4 : 8;
    for (uint32_t flat = 0; flat < s.tokens * s.selected; ++flat) {
        uint32_t slot = 0;
        while (slot < records && (meta[8 + slot] != ids[flat] || meta[136 + slot] == capacity)) ++slot;
        if (slot == records) {
            ++records;
            meta[8 + slot] = ids[flat];
            meta[136 + slot] = 0;
            std::fill_n(meta.begin() + 264 + 8 * slot, 8, flat / s.selected);
            std::fill_n(meta.begin() + 1288 + 8 * slot, 8, flat);
        }
        const uint32_t offset = 8 * slot + meta[136 + slot]++;
        meta[264 + offset] = flat / s.selected;
        meta[1288 + offset] = flat;
    }
    const std::array<uint32_t, 8> header{
        (s.hidden + gate_tile - 1) / gate_tile, records, 1,
        (s.channels + down_tile - 1) / down_tile, records, 1, records, 0
    };
    std::copy(header.begin(), header.end(), meta.begin());
    return meta;
}

struct Fixture {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLComputePipelineState> kernel;
    size_t checks = 0;

    explicit Fixture(const char* path) {
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
        auto function = [library newFunctionWithName:@"planar_moe_build_union_m8"];
        if (!function || !queue) throw std::runtime_error("planar union kernel or queue unavailable");
        kernel = [device newComputePipelineStateWithFunction:function error:&error];
        if (!kernel || kernel.threadExecutionWidth != 32) throw std::runtime_error("expected SIMD32 union kernel");
    }

    void check(Shape s, const std::vector<uint32_t>& ids, bool valid = true) {
        @autoreleasepool {
            auto input = [device newBufferWithBytes:ids.data() length:ids.size() * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
            auto meta = [device newBufferWithLength:sizeof(Metadata) options:MTLResourceStorageModeShared];
            const uint32_t initial_error = 8;
            auto flag = [device newBufferWithBytes:&initial_error length:sizeof(initial_error)
                                     options:MTLResourceStorageModeShared];
            if (!input || !meta || !flag) throw std::runtime_error("planar union test buffers unavailable");
            auto* words = static_cast<uint32_t*>(meta.contents);
            std::fill_n(words, Metadata{}.size(), poison);
            auto command = [queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:kernel];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:meta offset:0 atIndex:1];
            [encoder setBytes:&s length:sizeof(s) atIndex:2];
            [encoder setBuffer:flag offset:0 atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            auto expected = reference(s, ids, valid);
            if (command.status != MTLCommandBufferStatusCompleted ||
                *static_cast<uint32_t*>(flag.contents) != (valid ? initial_error : initial_error | 4u) ||
                !std::equal(expected.begin(), expected.end(), words)) {
                std::cerr << "case " << checks << " failed for " << s.tokens << " rows, " << s.selected << " selected\n";
                throw std::runtime_error("ordered planar union metadata or sticky error differs");
            }
            ++checks;
        }
    }
};
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    @autoreleasepool {
        try {
            Fixture fixture(argv[1]);
            std::mt19937 rng(1729);
            for (uint32_t tokens = 1; tokens <= 8; ++tokens) {
                for (uint32_t selected = 1; selected <= 16; ++selected) {
                    Shape s{512, selected, tokens, 639, 2559};
                    std::vector<uint32_t> ids(tokens * selected, 511);
                    fixture.check(s, ids); // Overflow from repeated occurrences within one token.
                    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<uint32_t>((511 - i) % 512);
                    fixture.check(s, ids); // Unique records across SIMD block boundaries.
                    for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % 3 ? static_cast<uint32_t>(rng() % 7) : 511;
                    fixture.check(s, ids); // Interleaved new, partial and full records.
                    ids.back() = s.experts;
                    fixture.check(s, ids, false);
                }
            }
            fixture.check({3, 1, 1, 640, 2560}, {0}, false);
            fixture.check({2, 3, 1, 640, 2560}, {0, 1, 0}, false);
            std::cout << fixture.checks << " ordered planar union metadata cases passed\n";
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
}
