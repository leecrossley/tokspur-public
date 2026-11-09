#include "tokspur/moe_mm_many.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
MoeRouterF32Plan moe_router_f32_plan(uint32_t columns, uint32_t experts, uint32_t tokens, uint64_t input_stride_bytes, uint64_t output_stride_bytes) {
    if (columns!=2560 || experts!=512 || !tokens || tokens>512 || input_stride_bytes<uint64_t(columns)*4 || output_stride_bytes<uint64_t(experts)*4 || input_stride_bytes%16 || output_stride_bytes%16)
        throw FormatError("bounded raw FP32 router batch geometry/stride");
    const auto input=checked_add(checked_mul(tokens-1,input_stride_bytes),uint64_t(columns)*4);
    const auto output=checked_add(checked_mul(tokens-1,output_stride_bytes),uint64_t(experts)*4);
    if (input>(256ull<<20) || output>(256ull<<20)) throw FormatError("bounded raw FP32 router batch span");
    return {{columns,experts,tokens,0,input_stride_bytes,output_stride_bytes},uint64_t(columns)*experts*4,input,output,(experts+31)/32,(tokens+31)/32};
}
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
MoeRouterWindowPlan moe_router_window_plan(uint32_t columns, uint32_t experts, uint32_t tokens, uint64_t input_stride_bytes, uint64_t output_stride_bytes, bool allow_nax_2k) {
    const uint32_t max_tokens = allow_nax_2k ? 2048u : 512u;
    if (columns != 2560 || experts != 512 || !tokens || tokens > max_tokens || input_stride_bytes < uint64_t(columns) * 4 ||
        output_stride_bytes < uint64_t(experts) * 4 || input_stride_bytes % 16 || output_stride_bytes % 16)
        throw FormatError("bounded exact FP32 router window geometry/stride");
    const auto input = checked_add(checked_mul(tokens - 1, input_stride_bytes), uint64_t(columns) * 4);
    const auto output = checked_add(checked_mul(tokens - 1, output_stride_bytes), uint64_t(experts) * 4);
    if (input > (256ull << 20) || output > (256ull << 20)) throw FormatError("bounded exact FP32 router window span");
    return {{columns, experts, tokens, 0, input_stride_bytes, output_stride_bytes}, uint64_t(columns) * experts * 4,
        input, output, experts / 4, (tokens + moe_router_window_rows - 1) / moe_router_window_rows};
}
#endif
MoeMmManyBytes moe_mm_many_bytes(MoeShape s, uint32_t capacity, bool allow_nax_2k) {
    const uint64_t max_routes = allow_nax_2k ? 20480ull : 10240ull;
    const uint64_t max_total_bytes = allow_nax_2k ? (360ull << 20) : (300ull << 20);
    if (!s.channels || s.channels > 2560 || s.channels % 32 || !s.hidden || s.hidden > 640 || s.hidden % 32 ||
        !s.experts || s.experts > 512 || !s.selected || s.selected > 16 || s.selected > s.experts || !capacity || capacity > 2048 ||
        uint64_t(s.selected) * capacity > max_routes)
        throw FormatError("bounded tiled MoE row geometry");
    const std::array<uint64_t, 10> widths{s.experts,s.selected,s.selected,uint64_t(s.hidden)*s.selected,uint64_t(s.hidden)*s.selected,
        uint64_t(s.channels)*s.selected,s.hidden,s.hidden,s.channels,1};
    MoeMmManyBytes b;
    for (size_t i = 0; i < widths.size(); ++i) b.views[i] = checked_mul(checked_mul(widths[i],capacity),4);
    if (capacity <= 512) {
        const auto meta = grouped_mm_plan(8,PrefillWeightLayout::gguf,s.channels,s.hidden,s.experts,s.selected,capacity,uint64_t(s.channels)*4,uint64_t(s.hidden)*4,true);
        b.views[10]=meta.token_map_bytes;b.views[11]=meta.counts_bytes;b.views[12]=meta.tasks_bytes;
    } else {
        const auto routes = checked_mul(capacity, s.selected);
        const auto maximum_tasks = routes / 32 + s.experts;
        b.views[10] = checked_mul(checked_mul(s.experts, capacity), 4);
        b.views[11] = checked_mul(s.experts, 4);
        b.views[12] = checked_mul(checked_add(2, checked_mul(maximum_tasks, 2)), 4);
    }
    for (auto n : b.views) { if (n > (256ull<<20)) throw FormatError("tiled MoE workspace bound"); b.total=checked_add(b.total,n); }
    if (b.total > max_total_bytes || b.views[3]/4 > 16777216 || b.views[6]/4 > 16777216) throw FormatError("tiled MoE total/activation bound");
    return b;
}
std::array<GpuView,13> MoeMmManyScratch::views() const {
    return {rows.logits,rows.indices,rows.weights,rows.gate,rows.up,rows.down,rows.shared_gate,rows.shared_up,rows.shared_down,rows.shared_logit,token_map,counts,tasks};
}
}
