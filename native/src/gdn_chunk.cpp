#include "tokspur/gdn_chunk.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <cmath>

namespace tokspur {
namespace {
void need(bool x,const char* why){if(!x)throw FormatError(why);}
std::array<uint64_t,18> counts(GdnShape s,uint32_t n,GdnChunkMode mode){
    const auto raw=gdn_qkv_count(s),q=uint64_t(s.value_heads)*128,v=uint64_t(s.value_heads)*s.value_width,h=uint64_t(s.value_heads);
    std::array<uint64_t,18> c{raw*n,raw*n,q*n,q*n,v*n,v*n,h*n,h*n,h*n,h*n,v*n,v*n};
    if(mode==GdnChunkMode::cs64){c[12]=h*64;c[13]=c[14]=h*64*64;c[15]=64*v;c[16]=c[17]=gdn_state_count(s);}return c;
}
GdnChunkParams params(GdnShape s,GdnPolicy policy,uint32_t rows){GdnChunkParams p{};p.rows=rows;p.columns=s.channels;p.heads=s.value_heads;p.key_heads=s.key_heads;p.key_width=s.key_width;p.value_width=s.value_width;p.query_scale=policy.query_scale;p.l2_epsilon=policy.l2_epsilon;p.rms_epsilon=policy.rms_epsilon;p.l2_mode=uint32_t(policy.l2_mode);p.input_stride=uint64_t(s.value_heads)*128*4;p.output_stride=p.auxiliary_stride=uint64_t(s.value_heads)*s.value_width*4;p.gate_stride=uint64_t(s.value_heads)*4;return p;}
void emit(MetalGraph& g,GdnChunkOp op,GdnChunkParams p,std::initializer_list<std::pair<uint32_t,GpuView>> views){GdnChunkDispatch d{};d.op=op;d.params=p;for(const auto&[index,v]:views)d.buffers.at(index)=v;validate_gdn_chunk_dispatch(d,g.prefill_dense_nax());g.gdn_chunk(d);}
void exact(GpuView v,uint64_t n){need(v.owner && v.bytes==n,"GDN chunk buffer geometry");}
void abort(MetalGraph& g){try{g.abort();}catch(const FormatError&) {}}
}
std::array<GpuView,18> GdnChunkScratch::views()const{return {projected,convolved,query,key,value,z,alpha,beta_logits,log_decay,beta,raw,gated,prefix,lower,attention,residual,state_a,state_b};}
GdnChunkScratch allocate_gdn_chunk_scratch(MetalGraph& g,GdnShape s,uint32_t capacity,GdnChunkMode mode,GpuView arena){
    const auto budget=gdn_chunk_bytes(s,capacity,mode,g.prefill_dense_nax());const auto n=counts(s,capacity,mode);std::array<GpuView,18> views{};
    if(arena.owner){
        need(arena.bytes>=budget.total,"GDN chunk scratch arena size");
        uint64_t offset=0;
        for(size_t i=0;i<n.size();++i){if(n[i]){const auto b=checked_mul(n[i],4);views[i]=g.slice(arena,offset,b);offset=checked_add(offset,b);}}
        need(offset==budget.total,"GDN chunk scratch accounting");
        return {capacity,views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9],views[10],views[11],views[12],views[13],views[14],views[15],views[16],views[17],true};
    }
    uint64_t bytes=0;
    try{for(size_t i=0;i<n.size();++i)if(n[i]){const auto b=checked_mul(n[i],4);bytes=checked_add(bytes,b);views[i]=g.zeros(b,"GDN committed chunk scratch");}need(bytes==budget.total,"GDN chunk scratch accounting");}catch(...){for(auto v:views)if(v.owner)g.release(v);throw;}
    return {capacity,views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9],views[10],views[11],views[12],views[13],views[14],views[15],views[16],views[17],false};
}
void release_gdn_chunk_scratch(MetalGraph& g,const GdnChunkScratch& s){if(s.suballocated)return;for(auto v:s.views())if(v.owner)g.release(v);}
void encode_gdn_prefill_chunk(MetalGraph& g,const GdnWeights& w,const GdnChunkScratch& s,const GdnState& accepted,const GdnState& candidate,GpuRowsF32 input,GpuRowsF32 output,GdnShape shape,GdnPolicy policy,GdnChunkMode mode,GdnChunkProjection q8,bool legacy_order,bool allow_multiseg){
    try{
        need(!legacy_order || mode==GdnChunkMode::fused,"old-order recurrence requires fused chunk scheduling");
#ifndef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        need(!legacy_order,"old-order recurrence is not compiled");
#endif
        const bool allow_2k = g.prefill_dense_nax();
        validate_gdn_policy(policy);static_cast<void>(gdn_chunk_bytes(shape,s.capacity,mode,allow_2k));need(q8 && input.rows==output.rows && input.rows<=s.capacity && input.columns==shape.channels && output.columns==shape.channels,"GDN chunk rows/projector");
        static_cast<void>(gdn_chunk_row_extent(input,allow_2k));static_cast<void>(gdn_chunk_row_extent(output,allow_2k));
        const auto n=counts(shape,s.capacity,mode);const auto views=s.views();for(size_t i=0;i<n.size();++i){if(n[i])exact(views[i],checked_mul(n[i],4));else need(!views[i].owner && !views[i].bytes,"unexpected GDN CS64 scratch");}
        exact(accepted.matrix,gdn_state_count(shape)*4);exact(candidate.matrix,accepted.matrix.bytes);exact(accepted.convolution,gdn_qkv_count(shape)*12);exact(candidate.convolution,accepted.convolution.bytes);
        exact(w.convolution,gdn_qkv_count(shape)*16);exact(w.negative_rates,uint64_t(shape.value_heads)*4);exact(w.dt_bias,w.negative_rates.bytes);exact(w.gains,uint64_t(shape.value_width)*4);need(w.alpha.encoding==0 && w.beta.encoding==0,"GDN alpha/beta must remain F32");
        const std::array read{input.view,accepted.matrix,accepted.convolution,w.qkv.view,w.gate.view,w.alpha.view,w.beta.view,w.output.view,w.convolution,w.negative_rates,w.dt_bias,w.gains};
        std::array<GpuView,21> write{output.view,candidate.matrix,candidate.convolution};std::copy(views.begin(),views.end(),write.begin()+3);std::vector<GpuView> active;for(auto v:write)if(v.owner)active.push_back(v);g.validate_disjoint(read,active);
        const uint32_t rows=input.rows,raw=uint32_t(gdn_qkv_count(shape)),v=shape.value_heads*shape.value_width;const auto row=[&](GpuView view,uint32_t width){return GpuRowsF32{view,rows,width,uint64_t(width)*4};};

        TOKSPUR_OPERATOR_CHECKPOINT(g,"unclassified","before_gdn_projection",{rows,shape.channels,raw,v});
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
        if (allow_multiseg && w.qkv.encoding==8 && w.gate.encoding==8 && input.stride_bytes==uint64_t(shape.channels)*4 &&
            g.can_dense_mm_q8_multiseg({g.prefill_weight_layout(w.qkv.view),g.prefill_weight_layout(w.gate.view)},rows)) {
            g.dense_mm_q8_multiseg2(g.prefill_weight_layout(w.qkv.view),w.qkv.view,
                g.prefill_weight_layout(w.gate.view),w.gate.view,input,row(s.projected,raw),row(s.z,v));
            TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","gdn_qkv_gate_multiseg2",{rows,shape.channels,raw+v,8});
        } else
#else
        static_cast<void>(allow_multiseg);
#endif
        {
            q8(g,w.qkv,input,row(s.projected,raw));
            TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","gdn_qkv",{rows,shape.channels,raw,8});
            q8(g,w.gate,input,row(s.z,v));
            TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","gdn_gate",{rows,shape.channels,v,8});
        }
        const bool compact_qk=(mode==GdnChunkMode::fused && !legacy_order);
        const uint32_t qk_heads=compact_qk?shape.key_heads:shape.value_heads;
        auto p=params(shape,policy,rows);p.columns=shape.channels;p.outputs=shape.value_heads;p.input_stride=input.stride_bytes;
        emit(g,GdnChunkOp::projection_gates_f32,p,{{0,w.alpha.view},{1,w.beta.view},{2,input.view},{3,w.negative_rates},{4,w.dt_bias},{5,s.log_decay},{6,s.beta}});
        p=params(shape,policy,rows);p.mode=compact_qk?1u:0u;p.input_stride=uint64_t(raw)*4;p.output_stride=uint64_t(qk_heads)*128*4;
        emit(g,GdnChunkOp::conv_silu_prepare,p,{{0,accepted.convolution},{1,s.projected},{2,w.convolution},{3,candidate.convolution},{4,s.query},{5,s.key},{6,s.value}});

        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","f32_projections_convolution_activation_prepare_gates",{rows,shape.channels,shape.value_heads,shape.value_width});
        p=params(shape,policy,rows);
        if(mode==GdnChunkMode::fused){
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
            if(legacy_order){GdnChunkDispatch d{};d.op=GdnChunkOp::fused;d.params=p;d.buffers[0]=accepted.matrix;d.buffers[1]=s.query;d.buffers[2]=s.key;d.buffers[3]=s.value;d.buffers[4]=s.log_decay;d.buffers[5]=s.beta;d.buffers[6]=candidate.matrix;d.buffers[8]=s.raw;g.gdn_legacy_batch(d);}else
#endif
            {p.mode=1u;p.input_stride=uint64_t(shape.key_heads)*128*4;emit(g,GdnChunkOp::fused,p,{{0,accepted.matrix},{1,s.query},{2,s.key},{3,s.value},{4,s.log_decay},{5,s.beta},{6,candidate.matrix},{8,s.raw}});}
        }
        else {auto current=accepted.matrix;uint32_t chunk=0;for(uint32_t offset=0;offset<rows;offset+=64,++chunk){p.offset=offset;p.count=std::min(64u,rows-offset);const auto next=offset+p.count==rows?candidate.matrix:(chunk%2?s.state_b:s.state_a);
            emit(g,GdnChunkOp::prefix64,p,{{0,s.log_decay},{1,s.prefix}});emit(g,GdnChunkOp::coefficients64,p,{{0,s.query},{1,s.key},{2,s.prefix},{3,s.beta},{4,s.lower},{5,s.attention}});
            emit(g,GdnChunkOp::solve64,p,{{0,current},{1,s.key},{2,s.value},{3,s.prefix},{4,s.beta},{5,s.lower},{6,s.residual}});emit(g,GdnChunkOp::emit64,p,{{0,current},{1,s.query},{2,s.key},{3,s.prefix},{4,s.attention},{5,s.residual},{6,next},{8,s.raw}});current=next;}
        }

        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","core",{rows,shape.value_heads,shape.key_width,shape.value_width});
        p=params(shape,policy,rows);p.input_stride=p.auxiliary_stride=p.output_stride=uint64_t(v)*4;emit(g,GdnChunkOp::rms_gate,p,{{0,s.raw},{1,w.gains},{2,s.z},{3,s.gated}});

        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","rms_gate",{rows,v,0,0});
        q8(g,w.output,row(s.gated,v),output);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","gdn_output",{rows,v,shape.channels,8});
    }catch(...){abort(g);throw;}
}
} // namespace tokspur
