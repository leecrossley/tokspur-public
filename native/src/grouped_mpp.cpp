#include "tokspur/grouped_mpp.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
const char* grouped_mpp_kernel(uint32_t type, uint32_t tile, uint32_t staging_k) {
    if (type != 7 && type != 8 && type != 12 && type != 13)
        throw FormatError("unsupported grouped TensorOps encoding");
#ifdef TOKSPUR_PRIVATE_GROUPED_ORDERED
    if (staging_k == 1 && tile == 1) return "graph_grouped_ordered1";
    if (staging_k == 1 && tile == 4) return "graph_grouped_ordered4";
    if (staging_k == 1 && tile == 8) return "graph_grouped_ordered8";
    if (staging_k == 1 && tile == 16) return "graph_grouped_ordered16";
#endif
    if (staging_k == 32 && tile == 32) return "graph_grouped_staged32";
    if (staging_k == 128 && tile == 32) return "graph_grouped_staged128";
    if (!staging_k && tile == 16) return "graph_grouped_mpp16";
    if (!staging_k && tile == 32) return "graph_grouped_mpp32";
    throw FormatError("unsupported grouped TensorOps tile");
}
GroupedMppBytes grouped_mpp_bytes(uint32_t type, GroupedMppShape s, bool broadcast) {
    static_cast<void>(grouped_mpp_kernel(type,s.tile,s.staging_k));
    const auto& e = encoding(type);
    if (!s.columns || s.columns > 65536 || s.columns % e.elements_per_block ||
        !s.rows || s.rows > 32768 || !s.experts || s.experts > 512 ||
        !s.selected || s.selected > 16 || s.selected > s.experts || !s.batches || s.batches > 2048)
        throw FormatError("unsupported grouped TensorOps geometry");
    const auto routes = checked_mul(s.batches,s.selected);
    const auto maximum_tasks = uint32_t(routes / s.tile + s.experts);
    const auto row = uint32_t(checked_mul(s.columns / e.elements_per_block,e.bytes_per_block));
    GroupedMppBytes result{
        checked_mul(checked_mul(row,s.rows),s.experts),checked_mul(routes,4),
        checked_mul(checked_mul(s.columns,broadcast ? s.batches : routes),4),
        checked_mul(checked_mul(s.rows,routes),4),checked_mul(checked_mul(s.experts,s.batches),4),
        checked_mul(s.experts,4),checked_mul(2ull + 2ull*maximum_tasks,4),maximum_tasks,row};
    if (result.matrix > (2ull << 30)) throw FormatError("grouped TensorOps matrix bound exceeded");
    for (auto size : {result.input,result.output,result.token_map,result.counts,result.tasks})
        if (size > (256ull << 20)) throw FormatError("grouped TensorOps workspace bound exceeded");
    return result;
}
}
