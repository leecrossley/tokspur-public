#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tokspur/metal.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

std::runtime_error failure(const char* operation, NSError* error) {
    return std::runtime_error(std::string(operation) + (error ? ": " + std::string(error.localizedDescription.UTF8String) : " failed"));
}
} // namespace

uint32_t host_thermal_state() { return uint32_t([NSProcessInfo processInfo].thermalState); }

struct MetalGemv::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
};

MetalGemv::MetalGemv(const std::filesystem::path& owned_library) : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) throw std::runtime_error("no Metal device");
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) throw std::runtime_error("cannot create Metal command queue");
        NSError* error = nil;
        NSString* path = [[NSString alloc] initWithUTF8String:owned_library.c_str()];
        impl_->library = [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (!impl_->library) throw failure("loading owned Metal library", error);
    }
}

MetalGemv::~MetalGemv() = default;

std::string MetalGemv::device_name() const { return impl_->device.name.UTF8String; }

MetalResult MetalGemv::run(uint32_t type, std::span<const std::byte> matrix, uint32_t columns,
                          uint32_t rows, std::span<const float> input, uint32_t repetitions) {
    @autoreleasepool {
        const auto start = Clock::now();
        if (!can_decode(type) || !columns || !rows || columns != input.size() || rows > 16384 ||
            columns > 65536 || !repetitions || repetitions > 1000) throw FormatError("unsupported Metal GEMV request");
        const auto& layout = encoding(type);
        const auto row_bytes = checked_mul(columns / layout.elements_per_block, layout.bytes_per_block);
        if (columns % layout.elements_per_block || checked_mul(row_bytes, rows) != matrix.size() ||
            matrix.size() > 256 * 1024 * 1024) throw FormatError("Metal GEMV buffer size mismatch");
        for (float x : input) if (!std::isfinite(x)) throw FormatError("nonfinite GEMV input");
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        [constants setConstantValue:&type type:MTLDataTypeUInt atIndex:0];
        NSError* error = nil;
        id<MTLFunction> function = [impl_->library newFunctionWithName:@"quantized_gemv" constantValues:constants error:&error];
        if (!function) throw failure("specializing owned GEMV", error);
        id<MTLComputePipelineState> pipeline = [impl_->device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) throw failure("building owned GEMV pipeline", error);
        if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 128) {
            throw std::runtime_error("owned GEMV requires 32-wide SIMD and 128-thread groups");
        }
        id<MTLBuffer> weights = [impl_->device newBufferWithBytes:matrix.data() length:matrix.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> x = [impl_->device newBufferWithBytes:input.data() length:input.size_bytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> y = [impl_->device newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeShared];
        if (!weights || !x || !y) throw std::runtime_error("Metal buffer allocation refused");
        MetalResult result;
        result.buffer_bytes = weights.length + x.length + y.length;
        result.setup_ms = elapsed(start);
        const std::array<uint32_t, 4> shape{columns, rows, static_cast<uint32_t>(row_bytes), 0};
        std::vector<double> submissions;
        std::vector<double> executions;
        for (uint32_t iteration = 0; iteration <= repetitions; ++iteration) {
            const auto submit = Clock::now();
            id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
            if (!command) throw std::runtime_error("cannot create Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("cannot create Metal encoder");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weights offset:0 atIndex:0];
            [encoder setBuffer:x offset:0 atIndex:1];
            [encoder setBuffer:y offset:0 atIndex:2];
            [encoder setBytes:shape.data() length:sizeof(shape) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) throw failure("owned GEMV execution", command.error);
            const auto duration = elapsed(submit);
            if (!iteration) result.first_submission_ms = duration;
            else {
                submissions.push_back(duration);
                const auto gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000;
                if (!std::isfinite(gpu_ms) || gpu_ms <= 0) throw std::runtime_error("invalid Metal GPU timing");
                executions.push_back(gpu_ms);
            }
        }
        const auto* output = static_cast<const float*>(y.contents);
        result.output.assign(output, output + rows);
        for (float y_value : result.output) if (!std::isfinite(y_value)) throw FormatError("nonfinite Metal result");
        result.median_submission_ms = median(submissions);
        result.median_gpu_ms = median(executions);
        return result;
    }
}

} // namespace tokspur
