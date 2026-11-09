#include "tokspur/prefill_attention.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>

#if !defined(TOKSPUR_PRIVATE_QSA_PREFILL) || !defined(TOKSPUR_PRIVATE_GDN_CHUNK) || !defined(TOKSPUR_PRIVATE_GROUPED_MM)
#error "attention prefill requires the explicit QSA, shared RMS and layout-aware projection build"
#endif
namespace tokspur {
namespace {
void need(bool okay,const char* why){if(!okay)throw FormatError(why);}
bool prefill_nax_2k(const MetalGraph& g) {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    return g.prefill_dense_nax() && g.prefill_grouped_nax();
#else
    static_cast<void>(g); return false;
#endif
}
void geometry(const MetalGraph& g,AttentionLayerShape s,uint32_t base,uint32_t rows){
    validate_residual_shape(s.residual);need(rows && rows<=2048,"attention prefill row bound");
    const bool allow_2k = g.prefill_dense_nax();
    const uint32_t slice_cap = prefill_nax_2k(g) ? 2048u : 512u;
    for(uint32_t off=0;off<rows;off+=slice_cap){const uint32_t c=std::min(slice_cap,rows-off);static_cast<void>(qsa_prefill_plan(base+off,c,c,s.attention,allow_2k));}
    const bool grouped_nax =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        g.prefill_grouped_nax();
#else
        false;
#endif
    static_cast<void>(moe_mm_many_bytes(s.moe,std::min(rows,grouped_nax?2048u:512u),grouped_nax));
    need(s.residual.channels==s.attention.channels && s.moe.channels==s.attention.channels,"attention prefill channel geometry");
}
std::array<PrefillWeightLayout,10> layouts(AttentionPrefillLayouts l){return {l.query_gate,l.key,l.value,l.output,
    l.moe.gate,l.moe.up,l.moe.down,l.moe.shared_gate,l.moe.shared_up,l.moe.shared_down};}
void matrix(GpuMatrix w,uint32_t columns,uint32_t rows,uint32_t type){
    need(w.encoding==type,"attention prefill projection encoding");const auto& e=encoding(type);
    need(columns%e.elements_per_block==0 && w.view.bytes==checked_mul(checked_mul(columns/e.elements_per_block,e.bytes_per_block),rows),"attention prefill projection extent");
}
void weights(const MetalGraph& g,const AttentionLayerWeights& w,AttentionLayerShape shape,AttentionPrefillLayouts l){
    need(layouts(attention_prefill_layouts(g,w))==layouts(l),"attention prefill registered layout mismatch");
    const auto s=shape.attention;const uint32_t q=s.query_heads*s.head_width,k=s.kv_heads*s.head_width,iq=s.index_heads*s.index_width;
    matrix(w.attention.query_gate,s.channels,q*2,8);matrix(w.attention.key,s.channels,k,8);matrix(w.attention.value,s.channels,k,8);matrix(w.attention.output,q,s.channels,8);
    matrix(w.attention.index_query,s.channels,iq,30);matrix(w.attention.index_key,s.channels,s.index_width,30);
    need(g.prefill_weight_layout(w.attention.index_query.view)==PrefillWeightLayout::gguf && g.prefill_weight_layout(w.attention.index_key.view)==PrefillWeightLayout::gguf,"attention prefill BF16 indexer storage");
}
void row_extent(GpuRowsF32 r,uint32_t n,uint32_t columns){
    need(r.rows==n && r.columns==columns && r.stride_bytes%4==0 && r.stride_bytes>=uint64_t(columns)*4 &&
        r.view.bytes>=checked_add(checked_mul(n-1,r.stride_bytes),uint64_t(columns)*4),"attention prefill borrowed row extent");
}
GpuRowsF32 rows(GpuView v,uint32_t n,uint32_t width){return {v,n,width,uint64_t(width)*4};}
std::array<GpuView,27> weight_views(const AttentionLayerWeights& w){
    const auto& a=w.attention;const auto& m=w.ffn;
    return {w.attention_read.gains,w.attention_read.down.view,w.attention_read.up.view,w.attention_read.inject.view,
        w.ffn_read.gains,w.ffn_read.down.view,w.ffn_read.up.view,w.ffn_read.inject.view,
        a.query_gate.view,a.key.view,a.value.view,a.output.view,a.index_query.view,a.index_key.view,
        a.query_gains,a.key_gains,a.index_query_gains,a.index_key_gains,a.frequencies,
        m.router.view,m.gate.view,m.up.view,m.down.view,m.shared_gate.view,m.shared_up.view,m.shared_down.view,m.shared_router.view};
}
void abort(MetalGraph& g){try{g.abort();}catch(const FormatError&) {}}
}
AttentionPrefillLayouts attention_prefill_layouts(const MetalGraph& g,const AttentionLayerWeights& w){
    const auto& a=w.attention;const auto& m=w.ffn;
    return {g.prefill_weight_layout(a.query_gate.view),g.prefill_weight_layout(a.key.view),g.prefill_weight_layout(a.value.view),g.prefill_weight_layout(a.output.view),
        {g.prefill_weight_layout(m.gate.view),g.prefill_weight_layout(m.up.view),g.prefill_weight_layout(m.down.view),
            g.prefill_weight_layout(m.shared_gate.view),g.prefill_weight_layout(m.shared_up.view),g.prefill_weight_layout(m.shared_down.view)}};
}
void prepare_attention_prefill(MetalGraph& g,const AttentionLayerWeights& w,AttentionLayerShape shape,AttentionPrefillLayouts l){
    geometry(g,shape,0,1);weights(g,w,shape,l);
    prepare_prefill_residual(g,w.attention_read,shape.residual);prepare_prefill_residual(g,w.ffn_read,shape.residual);
    g.prepare_qsa_prefill();for(auto layout:{l.query_gate,l.key,l.value,l.output})g.prepare_dense_mm_q8(layout);
    prepare_moe_mm_many(g,w.ffn,shape.moe,l.moe);
}
uint64_t attention_prefill_dispatches(const MetalGraph& g,const AttentionLayerWeights& w,AttentionLayerShape shape,uint32_t base,uint32_t n){
    geometry(g,shape,base,n);
    const auto l=attention_prefill_layouts(g,w);
    const uint32_t slice_cap = prefill_nax_2k(g) ? 2048u : 512u;
    const bool allow_2k = g.prefill_dense_nax(), nax_qsa = g.nax_qsa_attention_supported(shape.attention), fused_prep = g.qsa_prefill_fused_enabled();
    const uint64_t extra_q8 = (g.dense_mm_q8_dispatches(shape.attention.channels) - 1) +
        (g.dense_mm_q8_dispatches(shape.attention.query_heads * shape.attention.head_width) - 1);
    uint64_t qsa_total = 0;
    for (uint32_t offset = 0; offset < n; offset += slice_cap) {
        const uint32_t chunk = std::min(slice_cap, n - offset);
        const uint64_t qkv_saved = g.can_dense_mm_q8_multiseg({l.query_gate,l.key,l.value}, chunk) ? 2ull : 0ull;
        qsa_total = checked_add(qsa_total, qsa_prefill_dispatches(base + offset, chunk, shape.attention, allow_2k, nax_qsa, fused_prep) + extra_q8 - qkv_saved);
    }
    const uint64_t hc_saved = can_fuse_prefill_hc_consumer(g, shape.residual, n) ? 1ull : 0ull;
    return qsa_total+prefill_residual_dispatches(g,w.attention_read,shape.residual,n)+
        prefill_residual_dispatches(g,w.ffn_read,shape.residual,n)+2ull*prefill_residual_write_dispatches(g,shape.residual,n)-hc_saved+
        moe_mm_many_dispatches(g,w.ffn,shape.moe,n);
}
void encode_attention_prefill(MetalGraph& g,const AttentionLayerWeights& w,const PrefillLayerScratch& layer,const QsaPrefillScratch& qsa,
    const MoeMmManyScratch& moe,const QsaCache& cache,GpuView accepted,GpuView candidate,GpuRowsF32 input,GpuRowsF32 output,
    uint32_t base,AttentionLayerShape shape,QsaPolicy policy,RouterMode mode,bool normalise,AttentionPrefillLayouts l,GpuView work_sum){
    try{
        const auto n=input.rows;geometry(g,shape,base,n);validate_qsa_policy(policy);need(uint32_t(mode)<=1,"attention prefill router mode");
        const uint32_t slice_cap = prefill_nax_2k(g) ? 2048u : 512u;
        need(n<=layer.residual.capacity && qsa.capacity==std::min(layer.residual.capacity,slice_cap) && moe.capacity > 0,"attention prefill shared scratch capacity");
        const uint32_t num_slices=(n+slice_cap-1)/slice_cap;
        need(num_slices==1 || work_sum.owner,"multi-slice attention prefill requires temporary index sum workspace");
        const auto width=shape.residual.branches*shape.residual.channels,c=shape.residual.channels;
        row_extent(input,n,width);row_extent(output,n,width);weights(g,w,shape,l);
        need(layer.intermediate_streams.bytes==uint64_t(layer.residual.capacity)*width*4 && layer.attention_output.bytes==uint64_t(layer.residual.capacity)*c*4 &&
            layer.ffn_output.bytes==layer.attention_output.bytes,"attention prefill shared output capacity");
        const auto protected_views=weight_views(w);std::array<GpuView,32> protected_all{input.view,accepted,cache.keys,cache.values,cache.index_keys};
        std::copy(protected_views.begin(),protected_views.end(),protected_all.begin()+5);
        const auto lv=layer.views();const auto qv=qsa.views();const auto mv=moe.views();
        std::array<GpuView,50> destinations{output.view,candidate};auto at=std::copy(lv.begin(),lv.end(),destinations.begin()+2);
        at=std::copy(qv.begin(),qv.end(),at);at=std::copy(mv.begin(),mv.end(),at);
        if(num_slices>1)*at++=work_sum;
        g.validate_disjoint(protected_all,std::span(destinations).first(size_t(at-destinations.begin())));
        const auto all_weights=weight_views(w);g.validate_disjoint(std::span(all_weights).first(27),cache.views());
        const std::array cache_protected{input.view,accepted};g.validate_disjoint(cache_protected,cache.views());
        const auto intermediate=rows(layer.intermediate_streams,n,width),attention_output=rows(layer.attention_output,n,c),ffn_output=rows(layer.ffn_output,n,c);
        encode_prefill_residual(g,w.attention_read,layer.residual,input,shape.residual,policy.rms_epsilon);
        const auto x_all=prefill_residual_input(layer.residual,shape.residual,n);
        for(uint32_t idx=0;idx<num_slices;++idx){
            const uint32_t offset=idx*slice_cap,chunk=std::min(slice_cap,n-offset),rem=num_slices-1-idx;
            const GpuView in_sum=idx==0?accepted:((rem%2==1)?candidate:work_sum);
            const GpuView out_sum=(rem%2==1)?work_sum:candidate;
            const auto x=rows(g.slice(x_all.view,uint64_t(offset)*c*4,uint64_t(chunk)*c*4),chunk,c);
            const auto attn_out_slice=rows(g.slice(layer.attention_output,uint64_t(offset)*c*4,uint64_t(chunk)*c*4),chunk,c);
            const auto p=qsa_prefill_projection_rows(qsa,chunk);
            if (g.can_dense_mm_q8_multiseg({l.query_gate,l.key,l.value},chunk)) {
                g.dense_mm_q8_multiseg3(l.query_gate,w.attention.query_gate.view,l.key,w.attention.key.view,l.value,w.attention.value.view,x,p.packed,p.key,p.value);
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","qsa_qkv_multiseg3",{chunk,x.columns,p.packed.columns+p.key.columns+p.value.columns,8});
            } else {
                g.dense_mm_q8(l.query_gate,w.attention.query_gate.view,x,p.packed);
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","qsa_query_gate",{chunk,x.columns,p.packed.columns,8});
                g.dense_mm_q8(l.key,w.attention.key.view,x,p.key,g.can_reuse_nax_dense_bf16(x));
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","qsa_key",{chunk,x.columns,p.key.columns,8});
                g.dense_mm_q8(l.value,w.attention.value.view,x,p.value,g.can_reuse_nax_dense_bf16(x));
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","qsa_value",{chunk,x.columns,p.value.columns,8});
            }
            encode_qsa_prefill_indexers(g,w.attention,qsa,x,shape.attention);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"qsa","bf16_indexers",{chunk,x.columns,shape.attention.index_heads,shape.attention.index_width});
            const auto gated=encode_qsa_prefill_projected(g,w.attention,qsa,cache,in_sum,out_sum,base+offset,chunk,shape.attention,policy);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"qsa","core_including_norm_rotary_append_index_selection_attention_gate",{chunk,base+offset,shape.attention.query_heads,shape.attention.head_width});
            g.dense_mm_q8(l.output,w.attention.output.view,gated,attn_out_slice);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","qsa_output",{chunk,gated.columns,attn_out_slice.columns,8});
        }
        encode_prefill_residual_write_and_read(g,w.ffn_read,layer.residual,input,attention_output,intermediate,shape.residual,policy.rms_epsilon);
        encode_moe_mm_many(g,w.ffn,moe,prefill_residual_input(layer.residual,shape.residual,n),ffn_output,shape.moe,mode,normalise,l.moe);
        encode_prefill_residual_write(g,layer.residual,intermediate,ffn_output,output,shape.residual);
    }catch(...){abort(g);throw;}
}
} // namespace tokspur
