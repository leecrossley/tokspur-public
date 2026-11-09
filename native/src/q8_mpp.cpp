#include "tokspur/q8_mpp.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
const char* q8_mpp_kernel(Q8MppLayout layout) {
    if (layout == Q8MppLayout::single) return "graph_q8_mpp";
    if (layout == Q8MppLayout::independent_four) return "graph_q8_mpp_quad";
    if (layout == Q8MppLayout::single_wide) return "graph_q8_mpp_wide";
    throw FormatError("unsupported Q8 TensorOps SIMD layout");
}
uint32_t q8_mpp_simdgroups(Q8MppLayout layout) {
    static_cast<void>(q8_mpp_kernel(layout));
    return layout == Q8MppLayout::independent_four ? 4 : 1;
}
Q8MppBytes q8_mpp_bytes(uint32_t columns, uint32_t rows, uint32_t batches) {
    if (!columns || columns > 65536 || columns % 32 || !rows || rows > 32768 || !batches || batches > 2048)
        throw FormatError("unsupported Q8 TensorOps geometry");
    const Q8MppBytes bytes{checked_mul(checked_mul(columns / 32, rows), 34),
        checked_mul(checked_mul(columns, batches), 4), checked_mul(checked_mul(rows, batches), 4)};
    for (auto count : {bytes.matrix, bytes.input, bytes.output})
        if (count > 128ull * 1024 * 1024) throw FormatError("Q8 TensorOps view exceeds private bound");
    return bytes;
}
}
