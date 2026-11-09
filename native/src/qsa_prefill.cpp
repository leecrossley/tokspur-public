#include "tokspur/qsa_prefill.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <bit>

#if !defined(TOKSPUR_PRIVATE_QSA_PREFILL) || !defined(TOKSPUR_PRIVATE_GDN_CHUNK)
#error "QSA prefill requires its private dispatch and the shared row RMS component"
#endif
namespace tokspur {
namespace {
void need(bool value,const char* why) {if(!value)throw FormatError(why);}
GpuRowsF32 rows(GpuView view,uint32_t n,uint32_t width) {return {view,n,width,uint64_t(width)*4};}
GpuRowsF32 active(GpuRowsF32 r,uint32_t n,bool allow_nax_2k=false) {r.rows=n;validate_qsa_prefill_rows(r,n,r.columns,allow_nax_2k);return r;}
void op(MetalGraph& g,QsaPrefillOp operation,std::initializer_list<GpuView> buffers,QsaPrefillParams p) {
    g.qsa_prefill_dispatch(operation,std::span<const GpuView>(buffers.begin(),buffers.size()),p);
}
GpuView stripe(MetalGraph& g,GpuView view,uint32_t first,uint32_t n,uint32_t width) {return g.slice(view,uint64_t(first)*width*4,uint64_t(n)*width*4);}
void rope(MetalGraph& g,GpuView input,GpuView frequencies,GpuView output,uint32_t n,uint32_t heads,uint32_t width,
    uint32_t pairs,uint32_t base,uint32_t step,RotaryPairs mode) {
    op(g,QsaPrefillOp::rope_rows,{input,frequencies,output},{n,heads,width,pairs,base,step,heads*width,heads*width,uint32_t(mode)});
}
void store(MetalGraph& g,GpuView input,GpuView output,uint32_t n,uint32_t width) {
    op(g,QsaPrefillOp::store_bf16,{input,output},{n,width,width,width});
}
void abort(MetalGraph& g) {try{g.abort();}catch(const FormatError&) {}}
}
std::array<GpuView,25> QsaPrefillScratch::views() const {
    return {projected.packed.view,projected.key.view,projected.value.view,projected.index_query.view,projected.index_key.view,
        query,gate,normalized_query,normalized_key,rotary_query,rotary_key,normalized_index_query,rotary_index_query,
        pools,normalized_pools,rotary_pools,scores,score_a,id_a,score_b,id_b,selected_ids,partials,attention,gated};
}
QsaPrefillScratch allocate_qsa_prefill_scratch(MetalGraph& g,QsaShape s,uint32_t capacity,GpuView arena) {
    const auto p=qsa_prefill_plan(0,capacity,capacity,s,g.prefill_dense_nax());const uint64_t q=s.query_heads*s.head_width,k=s.kv_heads*s.head_width,iq=s.index_heads*s.index_width;
    const uint64_t pool_rows=(capacity+3)/4,stripe_n=std::min(capacity,32u),tiles=(s.token_budget+3+127)/128;
    const std::array<uint64_t,25> count{capacity*q*2,capacity*k,capacity*k,capacity*iq,uint64_t(capacity)*s.index_width,
        capacity*q,capacity*q,capacity*q,capacity*k,capacity*q,capacity*k,capacity*iq,capacity*iq,
        pool_rows*s.index_width,pool_rows*s.index_width,pool_rows*s.index_width,stripe_n*p.block_capacity,
        stripe_n*p.selection_stride,stripe_n*p.selection_stride,stripe_n*p.selection_stride,stripe_n*p.selection_stride,
        uint64_t(capacity)*p.block_budget,stripe_n*s.query_heads*tiles*(s.head_width+2),capacity*q,capacity*q};
    std::array<GpuView,25> v{};
    if (arena.owner) {
        uint64_t total_needed = 0;
        for (auto c : count) total_needed = checked_add(total_needed, checked_mul(c, 4));
        need(arena.bytes >= total_needed, "QSA prefill scratch arena size");
        uint64_t offset = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            const auto b = checked_mul(count[i], 4);
            v[i] = g.slice(arena, offset, b);
            offset = checked_add(offset, b);
        }
        return {capacity,{rows(v[0],capacity,uint32_t(q*2)),rows(v[1],capacity,uint32_t(k)),rows(v[2],capacity,uint32_t(k)),
            rows(v[3],capacity,uint32_t(iq)),rows(v[4],capacity,s.index_width)},v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],
            v[13],v[14],v[15],v[16],v[17],v[18],v[19],v[20],v[21],v[22],v[23],v[24],true};
    }
    size_t made=0;
    try{for(;made<v.size();++made)v[made]=g.zeros(count[made]*4,"causal QSA prefill workspace");}
    catch(...){for(size_t i=0;i<made;++i)g.release(v[i]);throw;}
    return {capacity,{rows(v[0],capacity,uint32_t(q*2)),rows(v[1],capacity,uint32_t(k)),rows(v[2],capacity,uint32_t(k)),
        rows(v[3],capacity,uint32_t(iq)),rows(v[4],capacity,s.index_width)},v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],
        v[13],v[14],v[15],v[16],v[17],v[18],v[19],v[20],v[21],v[22],v[23],v[24],false};
}
void release_qsa_prefill_scratch(MetalGraph& g,const QsaPrefillScratch& s) {if(s.suballocated)return;for(const auto v:s.views())g.release(v);}
QsaPrefillProjections qsa_prefill_projection_rows(const QsaPrefillScratch& s,uint32_t n) {
    need(n&&n<=s.capacity,"QSA projection rows exceed workspace capacity");
    const bool allow_2k = s.capacity > 512;
    return {active(s.projected.packed,n,allow_2k),active(s.projected.key,n,allow_2k),active(s.projected.value,n,allow_2k),active(s.projected.index_query,n,allow_2k),active(s.projected.index_key,n,allow_2k)};
}
void encode_qsa_prefill_indexers(MetalGraph& g,const QsaWeights& w,const QsaPrefillScratch& s,GpuRowsF32 input,QsaShape shape) {
    try {
        const bool allow_2k = g.prefill_dense_nax();
        static_cast<void>(qsa_prefill_plan(0,input.rows,s.capacity,shape,allow_2k));validate_qsa_prefill_rows(input,input.rows,shape.channels,allow_2k);
        const auto projected=qsa_prefill_projection_rows(s,input.rows);
        need(w.index_query.encoding==30 && w.index_key.encoding==30,"QSA prefill indexers require original BF16 weights");
        need(w.index_query.view.bytes==uint64_t(shape.channels)*shape.index_heads*shape.index_width*2 &&
            w.index_key.view.bytes==uint64_t(shape.channels)*shape.index_width*2,"QSA BF16 indexer matrix extent mismatch");
        const std::array reads{input.view,w.index_query.view,w.index_key.view};const std::array writes{projected.index_query.view,projected.index_key.view};
        g.validate_disjoint(reads,writes);
        const uint32_t iq = shape.index_heads * shape.index_width;
        if (g.prefill_dense_nax() && g.can_reuse_nax_dense_bf16(input) &&
            (shape.channels % 64u == 0u) && (iq % 64u == 0u) && (shape.index_width % 64u == 0u) &&
            (projected.index_query.stride_bytes % 4u == 0u) && (projected.index_key.stride_bytes % 4u == 0u)) {
            g.nax_dense_bf16(w.index_query.view, input, projected.index_query, true);
            g.nax_dense_bf16(w.index_key.view, input, projected.index_key, true);
        } else {
            const uint32_t swizzle = g.prefill_swizzle_enabled() ? 8u : 0u;
            op(g,QsaPrefillOp::bf16_project,{w.index_query.view,input.view,projected.index_query.view},
                {input.rows,shape.channels,iq,uint32_t(input.stride_bytes/4),uint32_t(projected.index_query.stride_bytes/4),swizzle});
            op(g,QsaPrefillOp::bf16_project,{w.index_key.view,input.view,projected.index_key.view},
                {input.rows,shape.channels,shape.index_width,uint32_t(input.stride_bytes/4),uint32_t(projected.index_key.stride_bytes/4),swizzle});
        }
    }catch(...){abort(g);throw;}
}
GpuRowsF32 encode_qsa_prefill_projected(MetalGraph& g,const QsaWeights& w,const QsaPrefillScratch& s,const QsaCache& cache,
    GpuView accepted,GpuView candidate,uint32_t base,uint32_t n,QsaShape shape,QsaPolicy policy) {
    try {
        const bool allow_2k = g.prefill_dense_nax();
        const auto plan=qsa_prefill_plan(base,n,s.capacity,shape,allow_2k);validate_qsa_policy(policy);
        const auto p=qsa_prefill_projection_rows(s,n);const uint32_t q=shape.query_heads*shape.head_width,k=shape.kv_heads*shape.head_width,iq=shape.index_heads*shape.index_width;
        validate_qsa_prefill_rows(p.packed,n,q*2,allow_2k);validate_qsa_prefill_rows(p.key,n,k,allow_2k);validate_qsa_prefill_rows(p.value,n,k,allow_2k);
        validate_qsa_prefill_rows(p.index_query,n,iq,allow_2k);validate_qsa_prefill_rows(p.index_key,n,shape.index_width,allow_2k);
        need(cache.keys.bytes==uint64_t(shape.capacity)*k*2 && cache.values.bytes==cache.keys.bytes &&
            cache.index_keys.bytes==uint64_t(plan.block_capacity)*shape.index_width*2 && accepted.bytes==uint64_t(shape.index_width)*4 &&
            candidate.bytes==accepted.bytes && w.query_gains.bytes==uint64_t(shape.head_width)*4 && w.key_gains.bytes==w.query_gains.bytes &&
            w.index_query_gains.bytes==accepted.bytes && w.index_key_gains.bytes==accepted.bytes &&
            w.frequencies.bytes==uint64_t(shape.rotary_width)*2,"QSA prefill cache/norm/partial-sum extent mismatch");
        const auto sv=s.views();const std::array protected_reads{accepted,w.query_gains,w.key_gains,w.index_query_gains,w.index_key_gains,w.frequencies};
        g.validate_disjoint(protected_reads,sv);g.validate_disjoint(protected_reads,cache.views());
        const std::array next{candidate};g.validate_disjoint(protected_reads,next);g.validate_disjoint(cache.views(),sv);g.validate_disjoint(cache.views(),next);
        // This also rejects overlapping private outputs before any dispatch is encoded.
        g.validate_disjoint(sv,next);
        if (g.qsa_prefill_fused_enabled()) {
            const uint32_t eps_bits = std::bit_cast<uint32_t>(policy.rms_epsilon);
            op(g,QsaPrefillOp::prep_query,{p.packed.view,w.query_gains,w.frequencies,s.rotary_query,s.gate},
                {n,shape.query_heads,shape.head_width,shape.rotary_width/2,base,uint32_t(policy.packing),uint32_t(p.packed.stride_bytes/4),q,uint32_t(policy.rotary_pairs),eps_bits});
            op(g,QsaPrefillOp::prep_kv,{p.key.view,p.value.view,w.key_gains,w.frequencies,
                g.slice(cache.keys,plan.key_append_offset,plan.key_append_bytes),g.slice(cache.values,plan.key_append_offset,plan.key_append_bytes)},
                {n,shape.kv_heads,shape.head_width,shape.rotary_width/2,base,uint32_t(p.key.stride_bytes/4),uint32_t(p.value.stride_bytes/4),k,uint32_t(policy.rotary_pairs),eps_bits});
            op(g,QsaPrefillOp::prep_index_query,{p.index_query.view,w.index_query_gains,w.frequencies,s.rotary_index_query},
                {n,shape.index_heads,shape.index_width,shape.rotary_width/2,base,uint32_t(p.index_query.stride_bytes/4),iq,uint32_t(policy.rotary_pairs),eps_bits});
            if(plan.pool_count) {
                op(g,QsaPrefillOp::prep_index_pool,{p.index_key.view,accepted,w.index_key_gains,w.frequencies,
                    g.slice(cache.index_keys,plan.index_append_offset,plan.index_append_bytes),candidate},
                    {n,shape.index_width,base,uint32_t(p.index_key.stride_bytes/4),plan.pool_count,shape.rotary_width/2,uint32_t(policy.rotary_pairs),eps_bits});
            } else {
                op(g,QsaPrefillOp::index_pool,{p.index_key.view,accepted,s.pools,candidate},
                    {n,shape.index_width,base,uint32_t(p.index_key.stride_bytes/4),shape.index_width,plan.pool_count,4});
            }
        } else {
            op(g,QsaPrefillOp::split_gate,{p.packed.view,s.query,s.gate},
                {n,shape.query_heads,shape.head_width,uint32_t(policy.packing),uint32_t(p.packed.stride_bytes/4),q,q});
            g.rms_rows(rows(s.query,n,q),w.query_gains,rows(s.normalized_query,n,q),shape.head_width,policy.rms_epsilon,true);
            g.rms_rows(p.key,w.key_gains,rows(s.normalized_key,n,k),shape.head_width,policy.rms_epsilon,true);
            g.rms_rows(p.index_query,w.index_query_gains,rows(s.normalized_index_query,n,iq),shape.index_width,policy.rms_epsilon,true);
            rope(g,s.normalized_query,w.frequencies,s.rotary_query,n,shape.query_heads,shape.head_width,shape.rotary_width/2,base,1,policy.rotary_pairs);
            rope(g,s.normalized_key,w.frequencies,s.rotary_key,n,shape.kv_heads,shape.head_width,shape.rotary_width/2,base,1,policy.rotary_pairs);
            rope(g,s.normalized_index_query,w.frequencies,s.rotary_index_query,n,shape.index_heads,shape.index_width,shape.rotary_width/2,base,1,policy.rotary_pairs);
            store(g,s.rotary_key,g.slice(cache.keys,plan.key_append_offset,plan.key_append_bytes),n,k);
            op(g,QsaPrefillOp::store_bf16,{p.value.view,g.slice(cache.values,plan.key_append_offset,plan.key_append_bytes)},
                {n,k,uint32_t(p.value.stride_bytes/4),k});
            op(g,QsaPrefillOp::index_pool,{p.index_key.view,accepted,s.pools,candidate},
                {n,shape.index_width,base,uint32_t(p.index_key.stride_bytes/4),shape.index_width,plan.pool_count,4});
            if(plan.pool_count) {
                g.rms_rows(rows(s.pools,plan.pool_count,shape.index_width),w.index_key_gains,rows(s.normalized_pools,plan.pool_count,shape.index_width),shape.index_width,policy.rms_epsilon,true);
                rope(g,s.normalized_pools,w.frequencies,s.rotary_pools,plan.pool_count,1,shape.index_width,shape.rotary_width/2,plan.first_pool*4,4,policy.rotary_pairs);
                store(g,s.rotary_pools,g.slice(cache.index_keys,plan.index_append_offset,plan.index_append_bytes),plan.pool_count,shape.index_width);
            }
        }
        const bool use_nax_qsa = g.nax_qsa_attention_supported(shape);
        if (use_nax_qsa) {
            if ((base + n) / 4u > plan.block_budget) {
                g.nax_qsa_select_blocks(g.slice(s.rotary_index_query,0,uint64_t(n)*iq*4),cache.index_keys,
                    g.slice(s.selected_ids,0,uint64_t(n)*plan.block_budget*4),base,n,shape,plan.block_budget);
            }
            g.nax_qsa_attention(g.slice(s.rotary_query,0,uint64_t(n)*q*4),cache.keys,cache.values,
                g.slice(s.selected_ids,0,uint64_t(n)*plan.block_budget*4),
                g.slice(s.attention,0,uint64_t(n)*q*4),base,n,shape,plan.block_budget);
        } else {
            for(uint32_t first=0;first<n;first+=32) {
                const uint32_t sn=std::min(32u,n-first),sb=base+first,max_blocks=(sb+sn)/4;
                const auto index_query=stripe(g,s.rotary_index_query,first,sn,iq),query=stripe(g,s.rotary_query,first,sn,q);
                const auto ids=stripe(g,s.selected_ids,first,sn,plan.block_budget);
                GpuView ranked=s.id_a;
                if(max_blocks>plan.block_budget) {
                    op(g,QsaPrefillOp::index_scores,{index_query,cache.index_keys,s.scores},
                        {sn,sb,shape.index_heads,shape.index_width,plan.block_capacity,iq,plan.block_budget});
                    op(g,QsaPrefillOp::select_sort,{s.scores,s.score_a,s.id_a},{sn,sb,plan.block_capacity,plan.selection_stride,plan.block_budget});
                    GpuView in_scores=s.score_a,in_ids=s.id_a,out_scores=s.score_b,out_ids=s.id_b;
                    for(uint32_t span=512;span<max_blocks;span*=2) {
                        op(g,QsaPrefillOp::select_merge,{in_scores,in_ids,out_scores,out_ids},{sn,sb,plan.selection_stride,plan.block_budget,span});
                        std::swap(in_scores,out_scores);std::swap(in_ids,out_ids);
                    }
                    ranked=in_ids;
                }
                op(g,QsaPrefillOp::select_ids,{ranked,ids},{sn,sb,plan.selection_stride,plan.block_budget,plan.block_budget});
                op(g,QsaPrefillOp::attention_partial128,{query,cache.keys,cache.values,ids,s.partials},
                    {sn,sb,shape.query_heads,shape.kv_heads,shape.head_width,4,plan.block_budget,q,plan.block_budget,plan.attention_tiles});
                op(g,QsaPrefillOp::attention_reduce,{s.partials,stripe(g,s.attention,first,sn,q)},
                    {sn,sb,shape.query_heads,shape.head_width,4,plan.block_budget,plan.attention_tiles,q});
            }
        }
        // The existing multiply implements sigmoid(gate) and retains its finite/error contract.
        g.multiply(g.slice(s.attention,0,uint64_t(n)*q*4),g.slice(s.gate,0,uint64_t(n)*q*4),g.slice(s.gated,0,uint64_t(n)*q*4),true);
        return rows(s.gated,n,q);
    }catch(...){abort(g);throw;}
}
} // namespace tokspur
