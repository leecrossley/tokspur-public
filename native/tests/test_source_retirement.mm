// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc != 3) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 77;
        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            std::fputs("cannot inspect the compiled Metal library\n", stderr);
            return 1;
        }
        NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
        if (!data) return 2;
        NSDictionary* manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        NSArray* retired = manifest[@"retired_shader_symbols"];
        if (![retired isKindOfClass:[NSArray class]] || !retired.count) return 2;
        NSSet* functions = [NSSet setWithArray:library.functionNames];
        for (NSString* name in retired) {
            if ([functions containsObject:name]) {
                std::fputs("retired shader remains in the compiled library\n", stderr);
                return 1;
            }
        }
        for (NSString* name in @[@"graph_quantized_gemv", @"dense_lanes_q8", @"graph_q8_three_rows", @"graph_lanes_q8_three_rows", @"gdn_chunk_f32", @"mtp_concat", @"mtp_dense_attention"]) {
            if (![functions containsObject:name]) {
                std::fputs("baseline or native MTP shader is missing\n", stderr);
                return 1;
            }
        }
        std::puts("retired shaders absent, baseline and native MTP shaders present");
    }
}
