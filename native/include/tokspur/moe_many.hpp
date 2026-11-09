#pragma once
#include "tokspur/moe.hpp"

namespace tokspur {
struct MoeManyScratch {
    MoeScratch rows;
    GpuView token_map, counts, tasks;
    uint32_t capacity = 0;
    [[nodiscard]] std::array<GpuView,13> views() const;
};
// Explicit known-input FFN prototype. No allocation/read/submit/publication in
// encode. Router and mixture equations/order are unchanged; routed matmuls are
// O-tier. Shared Q8 projections retain the original multi-input arithmetic.
MoeManyScratch allocate_moe_many_scratch(MetalGraph&,MoeShape,uint32_t capacity);
void release_moe_many_scratch(MetalGraph&,const MoeManyScratch&);
void prepare_moe_many(MetalGraph&,const MoeWeights&);
uint64_t moe_many_dispatches(const MetalGraph&,const MoeWeights&,MoeShape,uint32_t steps);
void encode_moe_many(MetalGraph&,const MoeWeights&,const MoeManyScratch&,GpuView input,GpuView output,
    MoeShape,RouterMode,bool normalize_selected,uint32_t steps);
}
