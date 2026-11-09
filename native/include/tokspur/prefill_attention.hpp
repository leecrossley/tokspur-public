#pragma once
#include "tokspur/prefill_residual.hpp"
#include "tokspur/qsa_prefill.hpp"
#include "tokspur/moe_mm_many.hpp"

namespace tokspur {
struct AttentionPrefillLayouts {
    PrefillWeightLayout query_gate=PrefillWeightLayout::gguf,key=PrefillWeightLayout::gguf,value=PrefillWeightLayout::gguf,output=PrefillWeightLayout::gguf;
    MoeMmLayouts moe;
};
// Query each complete registered matrix. Layouts are explicit inputs to preparation and encoding, never inferred from an engine-wide flag.
AttentionPrefillLayouts attention_prefill_layouts(const MetalGraph&,const AttentionLayerWeights&);
void prepare_attention_prefill(MetalGraph&,const AttentionLayerWeights&,AttentionLayerShape,AttentionPrefillLayouts);
uint64_t attention_prefill_dispatches(const MetalGraph&,const AttentionLayerWeights&,AttentionLayerShape,uint32_t base,uint32_t rows);
// Borrowed shared layer, QSA and MoE scratch. The outer owner allocates/prepares while idle and owns submission, drain, physical cache rollback and all-layer publication.
void encode_attention_prefill(MetalGraph&,const AttentionLayerWeights&,const PrefillLayerScratch&,const QsaPrefillScratch&,
    const MoeMmManyScratch&,const QsaCache&,GpuView accepted_sum,GpuView candidate_sum,GpuRowsF32 streams,GpuRowsF32 output,
    uint32_t base,AttentionLayerShape,QsaPolicy,RouterMode,bool normalize_selected,AttentionPrefillLayouts,GpuView work_sum={});
} // namespace tokspur
