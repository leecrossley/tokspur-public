#pragma once
#include <cstdint>

namespace tokspur {
enum class Q8ThreeLayout : uint32_t { gguf=0, q8_lanes_v1=1 };
// Shared verbatim with MSL. Exactly three active rows, with byte strides.
struct Q8ThreeArgs {
    uint32_t columns,rows,row_bytes,reserved;
    uint64_t input_stride_bytes,output_stride_bytes;
};
static_assert(sizeof(Q8ThreeArgs)==32);
struct Q8ThreePlan {
    Q8ThreeArgs args{};
    uint64_t matrix_bytes=0,input_bytes=0,output_bytes=0;
    uint32_t groups=0,threads=128;
    const char* kernel=nullptr;
};
// Pure bounds only. Registration and Graph ownership remain with the caller.
Q8ThreePlan q8_three_rows_plan(Q8ThreeLayout,uint32_t columns,uint32_t rows,
    uint64_t input_stride_bytes,uint64_t output_stride_bytes);
const char* q8_three_rows_kernel(Q8ThreeLayout);
const char* q8_three_rows_msg_kernel(Q8ThreeLayout);
}
