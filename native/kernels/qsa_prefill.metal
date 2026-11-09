// Bounded Tokspur causal QSA operators. Selection and BF16 boundaries retain attention.metal semantics.
// The FP32 staged TensorOps projection follows Tokspur grouped_staged32, adapted from llama.cpp 6050b467 (MIT, ggml authors 2023-2026). See ../third_party/llama-cpp/LICENSE.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct QsaPrefillParams { uint v[16]; };
float qpf_bf16(ushort x) { return as_type<float>(uint(x)<<16); }
void qpf_write(device float* out,ulong at,float value,device atomic_uint* error) {
    if(!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    out[at]=value;
}
void qpf_bf16_write(device ushort* out,ulong at,float value,device atomic_uint* error) {
    const uint bits=as_type<uint>(value);const ushort rounded=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
    if(!isfinite(value) || (rounded&0x7f80u)==0x7f80u) {atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
    out[at]=rounded;
}
bool qpf_before(float a,uint ai,float b,uint bi) {return a>b || (a==b && ai<bi);}
inline uint2 qpf_swizzle_2d(uint2 group,uint num_gx,uint num_gy,uint swizzle) {
    if(swizzle<=1 || num_gx<=1 || num_gy<=1)return group;
    const uint linear=group.y*num_gx+group.x,band_tiles=num_gx*swizzle,band_idx=linear/band_tiles;
    const uint first_y=band_idx*swizzle,band_h=min(swizzle,num_gy-first_y),in_band=linear-band_idx*band_tiles;
    return uint2(in_band/band_h,first_y+(in_band%band_h));
}

kernel void qsa_prefill_bf16_project(device const ushort* weights [[buffer(0)]],device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint2 raw_block [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    const uint2 block=qpf_swizzle_2d(raw_block,(p.v[2]+31)/32,(p.v[0]+31)/32,p.v[5]);
    threadgroup float l[1024],r[1024];
    auto left=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(l,dextents<int32_t,2>(32,32));
    auto right=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(r,dextents<int32_t,2>(32,32));
    constexpr auto descriptor=matmul2d_descriptor(32,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto result=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(auto it=result.begin();it!=result.end();++it)*it=0;
    for(uint at=0;at<p.v[1];at+=32) {
        for(uint i=tid;i<1024;i+=128) {
            const uint row=block.x*32+i/32,col=at+i%32;
            const float a=row<p.v[2] && col<p.v[1]?qpf_bf16(weights[ulong(row)*p.v[1]+col]):0;
            const uint token=block.y*32+i%32,channel=at+i/32;
            const float b=token<p.v[0] && channel<p.v[1]?input[ulong(token)*p.v[3]+channel]:0;
            if(!isfinite(a)||!isfinite(b))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            l[i]=a;r[i]=b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,result);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=result.begin();it!=result.end();++it) {
        if(!result.is_valid_element(it))continue;
        const auto c=result.get_multidimensional_index(it);const uint row=block.x*32+c[1],token=block.y*32+c[0];
        if(c[0]<32 && c[1]<32 && row<p.v[2] && token<p.v[0])qpf_write(output,ulong(token)*p.v[4]+row,*it,error);
    }
}
kernel void qsa_prefill_bf16_project_nax(device const ushort* weights [[buffer(0)]],device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint2 raw_block [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    const uint2 block=qpf_swizzle_2d(raw_block,(p.v[2]+31)/32,(p.v[0]+31)/32,p.v[5]);
    threadgroup float l[32*33],r[32*33];
    auto left=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(l,dextents<int32_t,2>(32,32),array<int32_t,2>{1,33});
    auto right=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(r,dextents<int32_t,2>(32,32),array<int32_t,2>{1,33});
    constexpr auto descriptor=matmul2d_descriptor(32,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto result=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(auto it=result.begin();it!=result.end();++it)*it=0;
    uint bad=0;
    for(uint at=0;at<p.v[1];at+=32) {
        for(uint i=tid;i<1024;i+=128) {
            const uint row_local=i/32,k_local=i%32;
            const uint row=block.x*32+row_local,col=at+k_local;
            const float a=row<p.v[2] && col<p.v[1]?qpf_bf16(weights[ulong(row)*p.v[1]+col]):0;
            const uint token=block.y*32+row_local;
            const float b=token<p.v[0] && col<p.v[1]?input[ulong(token)*p.v[3]+col]:0;
            if(!isfinite(a)||!isfinite(b))bad=1u;
            l[row_local*33+k_local]=a;r[k_local*33+row_local]=b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,result);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(bad)atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    for(auto it=result.begin();it!=result.end();++it) {
        if(!result.is_valid_element(it))continue;
        const auto c=result.get_multidimensional_index(it);const uint row=block.x*32+c[1],token=block.y*32+c[0];
        if(c[0]<32 && c[1]<32 && row<p.v[2] && token<p.v[0])qpf_write(output,ulong(token)*p.v[4]+row,*it,error);
    }
}
kernel void qsa_prefill_split_gate(device const float* packed [[buffer(0)]],device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    const uint width=p.v[1]*p.v[2];if(i>=p.v[0]*width)return;
    const uint row=i/width,c=i%width,q=p.v[3]?c:(c/p.v[2])*2*p.v[2]+c%p.v[2],z=q+(p.v[3]?width:p.v[2]);
    qpf_write(query,ulong(row)*p.v[5]+c,packed[ulong(row)*p.v[4]+q],error);
    qpf_write(gate,ulong(row)*p.v[6]+c,packed[ulong(row)*p.v[4]+z],error);
}
kernel void qsa_prefill_rope_rows(device const float* input [[buffer(0)]],device const float* freq [[buffer(1)]],
    device float* output [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    const uint width=p.v[1]*p.v[2];if(i>=p.v[0]*width)return;
    const uint row=i/width,c=i%width,d=c%p.v[2],head=c-d;
    if(d<p.v[3]) {
        if(!isfinite(freq[d]) || freq[d]<=0){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const uint a=head+(p.v[8]?2*d:d),b=a+(p.v[8]?1:p.v[3]);
        const float angle=float(p.v[4]+row*p.v[5])*freq[d],cosine=cos(angle),sine=sin(angle);
        const float x=input[ulong(row)*p.v[6]+a],y=input[ulong(row)*p.v[6]+b];
        // Legacy qsa_rotary contracts x*cos + (-y)*sin and y*cos + x*sin. Explicit fma keeps those cache-boundary bits under this file's contraction-off build.
        const float rotated_a=fma(x,cosine,(-y)*sine),rotated_b=fma(y,cosine,x*sine);
        qpf_write(output,ulong(row)*p.v[7]+a,rotated_a,error);qpf_write(output,ulong(row)*p.v[7]+b,rotated_b,error);
    } else if(d>=2*p.v[3])qpf_write(output,ulong(row)*p.v[7]+c,input[ulong(row)*p.v[6]+c],error);
}
kernel void qsa_prefill_store_bf16(device const float* input [[buffer(0)]],device ushort* output [[buffer(1)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.v[0]*p.v[1])return;const uint row=i/p.v[1],c=i%p.v[1];
    qpf_bf16_write(output,ulong(row)*p.v[3]+c,input[ulong(row)*p.v[2]+c],error);
}
kernel void qsa_prefill_index_pool(device const float* raw [[buffer(0)]],device const float* accepted [[buffer(1)]],
    device float* pooled [[buffer(2)]],device float* next [[buffer(3)]],constant QsaPrefillParams& p [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    if(i>=(p.v[5]+1)*p.v[1])return;const uint group=i/p.v[1],c=i%p.v[1],first=p.v[2]/4;
    if(group==p.v[5] && (p.v[2]+p.v[0])%4==0){qpf_write(next,c,0,error);return;}
    const uint start=max(p.v[2],(first+group)*4),end=min(p.v[2]+p.v[0],(first+group+1)*4);
    float sum=group==0?accepted[c]:0;
    for(uint token=start;token<end;++token)sum=sum+raw[ulong(token-p.v[2])*p.v[3]+c];
    if(group<p.v[5])qpf_write(pooled,ulong(group)*p.v[4]+c,sum/4.0f,error);else qpf_write(next,c,sum,error);
}
kernel void qsa_prefill_index_scores(device const float* query [[buffer(0)]],device const ushort* keys [[buffer(1)]],
    device float* output [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    const uint b=group.x*4+sg,blocks=(p.v[1]+group.y+1)/4;if(b>=blocks || blocks<=p.v[6])return;
    float result=0;
    for(uint h=0;h<p.v[2];++h) {
        float dot=0;for(uint d=lane;d<p.v[3];d+=32)dot+=query[ulong(group.y)*p.v[5]+h*p.v[3]+d]*qpf_bf16(keys[ulong(b)*p.v[3]+d]);
        dot=simd_sum(dot);if(!isfinite(dot))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);result+=max(dot,0.0f);
    }
    if(lane==0)qpf_write(output,ulong(group.y)*p.v[4]+b,result,error);
}
kernel void qsa_prefill_select_sort(device const float* input [[buffer(0)]],device float* scores [[buffer(1)]],
    device uint* ids [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    const uint blocks=(p.v[1]+group.y+1)/4;if(blocks<=p.v[4] || group.x*512>=blocks)return;
    threadgroup float ls[512];threadgroup uint li[512];
    for(uint i=tid;i<512;i+=128){const uint id=group.x*512+i;const float score=id<blocks?input[ulong(group.y)*p.v[2]+id]:-INFINITY;
        if(id<blocks && (!isfinite(score)||score<0))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        ls[i]=score;li[i]=id<blocks?id:0xffffffffu;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=2;k<=512;k*=2)for(uint stride=k/2;stride;stride/=2){for(uint i=tid;i<512;i+=128){const uint j=i^stride;
        if(j>i){const bool before=qpf_before(ls[i],li[i],ls[j],li[j]);if(((i&k)==0&&!before)||((i&k)!=0&&before)){
            const float a=ls[i];const uint id=li[i];ls[i]=ls[j];li[i]=li[j];ls[j]=a;li[j]=id;}}}threadgroup_barrier(mem_flags::mem_threadgroup);}
    const uint count=min(p.v[4],min(512u,blocks-group.x*512));
    for(uint i=tid;i<count;i+=128){const ulong at=ulong(group.y)*p.v[3]+group.x*p.v[4]+i;scores[at]=ls[i];ids[at]=li[i];}
}
kernel void qsa_prefill_select_merge(device const float* scores [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* out_scores [[buffer(2)]],device uint* out_ids [[buffer(3)]],constant QsaPrefillParams& p [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    const uint blocks=(p.v[1]+group.y+1)/4;if(blocks<=p.v[3])return;
    const uint start=group.x*p.v[4]*2,left=min(p.v[3],blocks>start?blocks-start:0u),right=min(p.v[3],blocks>start+p.v[4]?blocks-start-p.v[4]:0u);
    const ulong base=ulong(group.y)*p.v[2]+group.x*p.v[3]*2;
    for(uint item=tid;item<left+right;item+=128){const bool r=item>=left;const uint own=r?item-left:item;
        const ulong at=base+(r?p.v[3]:0)+own,other=base+(r?0:p.v[3]);const uint count=r?left:right;const float score=scores[at];const uint id=ids[at];
        if(id>=blocks || !isfinite(score)){atomic_fetch_or_explicit(error,4u,memory_order_relaxed);continue;}
        uint low=0,high=count;while(low<high){const uint mid=(low+high)/2;if(qpf_before(scores[other+mid],ids[other+mid],score,id))low=mid+1;else high=mid;}
        const uint rank=own+low;if(rank<p.v[3]){const ulong out=ulong(group.y)*p.v[2]+group.x*p.v[3]+rank;out_scores[out]=score;out_ids[out]=id;}}
}
kernel void qsa_prefill_select_ids(device const uint* input [[buffer(0)]],device uint* output [[buffer(1)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    const uint blocks=(p.v[1]+group.y+1)/4,count=min(blocks,p.v[4]);threadgroup uint ids[512];
    for(uint i=tid;i<512;i+=128){const uint id=i<count?(blocks<=p.v[4]?i:input[ulong(group.y)*p.v[2]+i]):0xffffffffu;
        ids[i]=id;if(i<count && id>=blocks)atomic_fetch_or_explicit(error,4u,memory_order_relaxed);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=2;k<=512;k*=2)for(uint stride=k/2;stride;stride/=2){for(uint i=tid;i<512;i+=128){const uint j=i^stride;
        if(j>i&&(((i&k)==0&&ids[i]>ids[j])||((i&k)!=0&&ids[i]<ids[j]))){const uint id=ids[i];ids[i]=ids[j];ids[j]=id;}}threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(uint i=tid;i<count;i+=128){if(i&&ids[i]==ids[i-1])atomic_fetch_or_explicit(error,4u,memory_order_relaxed);output[ulong(group.y)*p.v[3]+i]=ids[i];}
}
// A SIMD owns each 128-key partial. Its online softmax retains the existing per-token arithmetic; only the deterministic partial merge reassociates sums.
kernel void qsa_prefill_attention_partial128(device const float* query [[buffer(0)]],device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]],device const uint* ids [[buffer(3)]],device float* partials [[buffer(4)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
    const uint prefix=p.v[1]+group.y+1,blocks=prefix/4,selected=min(blocks,p.v[6]),tokens=selected*4+prefix%4;
    const uint head=group.x,key_head=head/(p.v[2]/p.v[3]),start=group.z*128,end=min(start+128,tokens),width=p.v[3]*p.v[4];
    const ulong out=((ulong(group.y)*p.v[2]+head)*p.v[9]+group.z)*(p.v[4]+2);
    float maximum=-INFINITY,denominator=0,sum[8];for(uint c=0;c<8;++c)sum[c]=0;
    for(uint item=start;item<end;++item){uint token;
        if(item<selected*4){const uint slot=item/4,id=blocks<=p.v[6]?slot:ids[ulong(group.y)*p.v[8]+slot];
            if(id>=blocks || (blocks>p.v[6]&&slot&&id<=ids[ulong(group.y)*p.v[8]+slot-1])){atomic_fetch_or_explicit(error,4u,memory_order_relaxed);return;}
            token=id*4+item%4;
        }else token=blocks*4+item-selected*4;
        const ulong at=ulong(token)*width+key_head*p.v[4];float dot=0;
        for(uint d=lane;d<p.v[4];d+=32)dot+=query[ulong(group.y)*p.v[7]+head*p.v[4]+d]*qpf_bf16(keys[at+d]);
        const float score=simd_sum(dot)*rsqrt(float(p.v[4]));if(!isfinite(score)){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const float nm=max(maximum,score),a=exp(maximum-nm),b=exp(score-nm);denominator=denominator*a+b;
        for(uint c=0,d=lane;d<p.v[4];++c,d+=32)sum[c]=sum[c]*a+b*qpf_bf16(values[at+d]);maximum=nm;
    }
    if(lane==0){partials[out]=start<tokens?maximum:0;partials[out+1]=denominator;}
    for(uint c=0,d=lane;d<p.v[4];++c,d+=32)qpf_write(partials,out+2+d,sum[c],error);
}
kernel void qsa_prefill_attention_reduce(device const float* partials [[buffer(0)]],device float* output [[buffer(1)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
    const uint prefix=p.v[1]+group.y+1,tokens=min(prefix/4,p.v[5])*4+prefix%4,tiles=(tokens+127)/128;
    const ulong base=(ulong(group.y)*p.v[2]+group.x)*p.v[6]*(p.v[3]+2);float maximum=-INFINITY,denominator=0,sum[8];for(uint c=0;c<8;++c)sum[c]=0;
    for(uint tile=0;tile<tiles;++tile){const ulong at=base+ulong(tile)*(p.v[3]+2);const float pm=partials[at],pd=partials[at+1];
        if(!isfinite(pm)||!isfinite(pd)||pd<=0){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const float nm=max(maximum,pm),a=exp(maximum-nm),b=exp(pm-nm);denominator=denominator*a+pd*b;
        for(uint c=0,d=lane;d<p.v[3];++c,d+=32)sum[c]=sum[c]*a+partials[at+2+d]*b;maximum=nm;
    }
    for(uint c=0,d=lane;d<p.v[3];++c,d+=32)qpf_write(output,ulong(group.y)*p.v[7]+group.x*p.v[3]+d,sum[c]/denominator,error);
}
kernel void qsa_prefill_prep_query(device const float* packed [[buffer(0)]],device const float* gains [[buffer(1)]],
    device const float* freq [[buffer(2)]],device float* rotary_query [[buffer(3)]],device float* gate [[buffer(4)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float norm_sh[4][256];
    const uint id=group*4+sg,H=p.v[1],D=p.v[2],width=H*D;if(id>=p.v[0]*H)return;
    const uint row=id/H,h=id%H;float sum=0;
    for(uint d=lane;d<D;d+=32){const uint c=h*D+d,q=p.v[5]?c:h*2*D+d,z=q+(p.v[5]?width:D);
        const float qv=packed[ulong(row)*p.v[6]+q],zv=packed[ulong(row)*p.v[6]+z];
        qpf_write(gate,ulong(row)*p.v[7]+c,zv,error);
        if(!isfinite(qv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        sum=fma(qv,qv,sum);}
    sum=simd_sum(sum);const float denom=sum/float(D)+as_type<float>(p.v[9]);
    if(!isfinite(sum)||!isfinite(denom)){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
    const float inv=rsqrt(denom);
    for(uint d=lane;d<D;d+=32){const uint c=h*D+d,q=p.v[5]?c:h*2*D+d;
        const float g=gains[d];if(!isfinite(g))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        const float nv=(packed[ulong(row)*p.v[6]+q]*inv)*g;
        if(!isfinite(nv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        norm_sh[sg][d]=nv;}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<p.v[3];d+=32){if(!isfinite(freq[d])||freq[d]<=0){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const uint a=p.v[8]?2*d:d,b=a+(p.v[8]?1:p.v[3]);
        const float angle=float(p.v[4]+row)*freq[d],cosine=cos(angle),sine=sin(angle),x=norm_sh[sg][a],y=norm_sh[sg][b];
        qpf_write(rotary_query,ulong(row)*p.v[7]+h*D+a,fma(x,cosine,(-y)*sine),error);
        qpf_write(rotary_query,ulong(row)*p.v[7]+h*D+b,fma(y,cosine,x*sine),error);}
    for(uint d=2*p.v[3]+lane;d<D;d+=32)qpf_write(rotary_query,ulong(row)*p.v[7]+h*D+d,norm_sh[sg][d],error);
}
kernel void qsa_prefill_prep_kv(device const float* key_in [[buffer(0)]],device const float* val_in [[buffer(1)]],
    device const float* gains [[buffer(2)]],device const float* freq [[buffer(3)]],
    device ushort* cache_keys [[buffer(4)]],device ushort* cache_values [[buffer(5)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float norm_sh[4][256];
    const uint id=group*4+sg,H=p.v[1],D=p.v[2];if(id>=p.v[0]*H)return;
    const uint row=id/H,h=id%H;float sum=0;
    for(uint d=lane;d<D;d+=32){const float kv=key_in[ulong(row)*p.v[5]+h*D+d];
        if(!isfinite(kv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        sum=fma(kv,kv,sum);
        qpf_bf16_write(cache_values,ulong(row)*p.v[7]+h*D+d,val_in[ulong(row)*p.v[6]+h*D+d],error);}
    sum=simd_sum(sum);const float denom=sum/float(D)+as_type<float>(p.v[9]);
    if(!isfinite(sum)||!isfinite(denom)){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
    const float inv=rsqrt(denom);
    for(uint d=lane;d<D;d+=32){const float g=gains[d];if(!isfinite(g))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        const float nv=(key_in[ulong(row)*p.v[5]+h*D+d]*inv)*g;
        if(!isfinite(nv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        norm_sh[sg][d]=nv;}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<p.v[3];d+=32){if(!isfinite(freq[d])||freq[d]<=0){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const uint a=p.v[8]?2*d:d,b=a+(p.v[8]?1:p.v[3]);
        const float angle=float(p.v[4]+row)*freq[d],cosine=cos(angle),sine=sin(angle),x=norm_sh[sg][a],y=norm_sh[sg][b];
        qpf_bf16_write(cache_keys,ulong(row)*p.v[7]+h*D+a,fma(x,cosine,(-y)*sine),error);
        qpf_bf16_write(cache_keys,ulong(row)*p.v[7]+h*D+b,fma(y,cosine,x*sine),error);}
    for(uint d=2*p.v[3]+lane;d<D;d+=32)qpf_bf16_write(cache_keys,ulong(row)*p.v[7]+h*D+d,norm_sh[sg][d],error);
}
kernel void qsa_prefill_prep_index_query(device const float* input [[buffer(0)]],device const float* gains [[buffer(1)]],
    device const float* freq [[buffer(2)]],device float* output [[buffer(3)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float norm_sh[4][256];
    const uint id=group*4+sg,H=p.v[1],D=p.v[2];if(id>=p.v[0]*H)return;
    const uint row=id/H,h=id%H;float sum=0;
    for(uint d=lane;d<D;d+=32){const float v=input[ulong(row)*p.v[5]+h*D+d];
        if(!isfinite(v))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        sum=fma(v,v,sum);}
    sum=simd_sum(sum);const float denom=sum/float(D)+as_type<float>(p.v[8]);
    if(!isfinite(sum)||!isfinite(denom)){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
    const float inv=rsqrt(denom);
    for(uint d=lane;d<D;d+=32){const float g=gains[d];if(!isfinite(g))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        const float nv=(input[ulong(row)*p.v[5]+h*D+d]*inv)*g;
        if(!isfinite(nv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        norm_sh[sg][d]=nv;}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<p.v[3];d+=32){if(!isfinite(freq[d])||freq[d]<=0){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const uint a=p.v[7]?2*d:d,b=a+(p.v[7]?1:p.v[3]);
        const float angle=float(p.v[4]+row)*freq[d],cosine=cos(angle),sine=sin(angle),x=norm_sh[sg][a],y=norm_sh[sg][b];
        qpf_write(output,ulong(row)*p.v[6]+h*D+a,fma(x,cosine,(-y)*sine),error);
        qpf_write(output,ulong(row)*p.v[6]+h*D+b,fma(y,cosine,x*sine),error);}
    for(uint d=2*p.v[3]+lane;d<D;d+=32)qpf_write(output,ulong(row)*p.v[6]+h*D+d,norm_sh[sg][d],error);
}
kernel void qsa_prefill_prep_index_pool(device const float* raw [[buffer(0)]],device const float* accepted [[buffer(1)]],
    device const float* gains [[buffer(2)]],device const float* freq [[buffer(3)]],
    device ushort* cache_index_keys [[buffer(4)]],device float* next [[buffer(5)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float norm_sh[4][256];
    const uint grp=group*4+sg,D=p.v[1],first=p.v[2]/4;if(grp>p.v[4])return;
    if(grp==p.v[4] && (p.v[2]+p.v[0])%4==0){for(uint d=lane;d<D;d+=32)qpf_write(next,d,0.0f,error);return;}
    const uint start=max(p.v[2],(first+grp)*4),end=min(p.v[2]+p.v[0],(first+grp+1)*4);
    if(grp==p.v[4]){
        for(uint d=lane;d<D;d+=32){float acc=grp==0?accepted[d]:0.0f;
            for(uint tok=start;tok<end;++tok)acc=acc+raw[ulong(tok-p.v[2])*p.v[3]+d];
            qpf_write(next,d,acc,error);}
        return;
    }
    float sum_sq=0;
    for(uint d=lane;d<D;d+=32){float acc=grp==0?accepted[d]:0.0f;
            for(uint tok=start;tok<end;++tok)acc=acc+raw[ulong(tok-p.v[2])*p.v[3]+d];
            const float pv=acc/4.0f;if(!isfinite(pv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            norm_sh[sg][d]=pv;sum_sq=fma(pv,pv,sum_sq);}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    sum_sq=simd_sum(sum_sq);const float denom=sum_sq/float(D)+as_type<float>(p.v[7]);
    if(!isfinite(sum_sq)||!isfinite(denom)){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
    const float inv=rsqrt(denom);
    for(uint d=lane;d<D;d+=32){const float g=gains[d];if(!isfinite(g))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        const float nv=(norm_sh[sg][d]*inv)*g;if(!isfinite(nv))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        norm_sh[sg][d]=nv;}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<p.v[5];d+=32){if(!isfinite(freq[d])||freq[d]<=0){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const uint a=p.v[6]?2*d:d,b=a+(p.v[6]?1:p.v[5]);
        const float angle=float(first*4+grp*4)*freq[d],cosine=cos(angle),sine=sin(angle),x=norm_sh[sg][a],y=norm_sh[sg][b];
        qpf_bf16_write(cache_index_keys,ulong(grp)*D+a,fma(x,cosine,(-y)*sine),error);
        qpf_bf16_write(cache_index_keys,ulong(grp)*D+b,fma(y,cosine,x*sine),error);}
    for(uint d=2*p.v[5]+lane;d<D;d+=32)qpf_bf16_write(cache_index_keys,ulong(grp)*D+d,norm_sh[sg][d],error);
}
kernel void nax_qsa_index_scores(device const float* query [[buffer(0)]],device const ushort* keys [[buffer(1)]],
    device float* output [[buffer(2)]],constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    const uint b_base=group.x*32u,tok_base=group.y*8u;
    if(tok_base>=p.v[0])return;
    const uint end_tok=min(p.v[0],tok_base+8u);
    const uint tile_max_blocks=(p.v[1]+end_tok)/4u;
    if(tile_max_blocks<=p.v[6] || b_base>=tile_max_blocks)return;
    threadgroup float l[32*33],r[32*33];
    auto left=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(l,dextents<int32_t,2>(32,32),array<int32_t,2>{1,33});
    auto right=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(r,dextents<int32_t,2>(32,32),array<int32_t,2>{1,33});
    constexpr auto descriptor=matmul2d_descriptor(32,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto result=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(auto it=result.begin();it!=result.end();++it)*it=0;
    uint bad=0;
    for(uint at=0u;at<128u;at+=32u) {
        for(uint i=tid;i<1024u;i+=128u) {
            const uint row_local=i/32u,k_local=i%32u;
            const uint b_col=b_base+row_local;
            const float kv=b_col<tile_max_blocks?qpf_bf16(keys[ulong(b_col)*128ul+at+k_local]):0.0f;
            const uint tok=tok_base+(row_local>>2u),h=row_local&3u;
            const float qv=tok<p.v[0]?query[ulong(tok)*p.v[5]+h*128u+at+k_local]:0.0f;
            if(!isfinite(kv)||!isfinite(qv))bad=1u;
            l[row_local*33u+k_local]=kv;
            r[k_local*33u+row_local]=qv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,result);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=result.begin();it!=result.end();++it) {
        if(!result.is_valid_element(it))continue;
        const auto c=result.get_multidimensional_index(it);
        if(c[0]<32 && c[1]<32)l[uint(c[0])*33u+uint(c[1])]=float(*it);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tok_local=uint(tid)>>4u,kb_pair=(uint(tid)&15u)*2u;
    const uint tok=tok_base+tok_local;
    if(tok<p.v[0]) {
        const uint blocks=(p.v[1]+tok+1u)/4u;
        if(blocks>p.v[6]) {
            for(ushort sub=0;sub<2;++sub) {
                const uint kb=kb_pair+sub,b=b_base+kb;
                if(b<blocks) {
                    float sum_pos=0.0f;
                    for(ushort h=0;h<4;++h) {
                        const float dot=l[(tok_local*4u+h)*33u+kb];
                        if(!isfinite(dot))bad=1u;
                        sum_pos+=max(dot,0.0f);
                    }
                    qpf_write(output,ulong(tok)*p.v[4]+b,sum_pos,error);
                }
            }
        }
    }
    if(bad)atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
}
kernel void qsa_prefill_verify_index_pool(device const float* raw [[buffer(0)]],device const float* accepted [[buffer(1)]],
    device float* pooled [[buffer(2)]],device float* prefix_sums [[buffer(3)]],device float* next [[buffer(4)]],
    constant QsaPrefillParams& p [[buffer(6)]],device atomic_uint* error [[buffer(7)]],uint c [[thread_position_in_grid]]) {
    if(c>=p.v[1])return;
    float acc=accepted[c];uint pool_idx=0;
    for(uint t=0;t<p.v[0];++t) {
        const float value=acc+raw[ulong(t)*p.v[3]+c];
        if(!isfinite(value)){atomic_fetch_or_explicit(error,1u,memory_order_relaxed);return;}
        const bool complete=((p.v[2]+t+1u)%4u==0u);
        acc=complete?0.0f:value;
        if(t+1u<p.v[0])prefix_sums[ulong(t)*p.v[1]+c]=acc;
        else next[c]=acc;
        if(complete){pooled[ulong(pool_idx)*p.v[4]+c]=value/4.0f;++pool_idx;}
    }
}
