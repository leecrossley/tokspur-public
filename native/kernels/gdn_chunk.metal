/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <metal_stdlib>
using namespace metal;

// The CS64 algebra follows delta-net-base.cpp at llama.cpp6050b467. No external graph/runtime is used. Native persistent state stays [head,key,value].
// The fused recurrence follows the bounded Gated Delta Net equations. All staging, state, projections and accumulators here are FP32.
struct GdnChunkParams {
    uint rows, columns, outputs, heads;
    uint key_heads, key_width, value_width, mode;
    uint offset, count, l2_mode, reserved;
    ulong input_stride, output_stride, auxiliary_stride, gate_stride;
    float query_scale, l2_epsilon, rms_epsilon, reserved_float;
};
static_assert(sizeof(GdnChunkParams)==96, "GDN chunk ABI");
inline float sigmoid_gdn(float x){if(x>=0)return 1.0f/(1.0f+exp(-x));const float z=exp(x);return z/(1.0f+z);}
inline float softplus_gdn(float x){const float z=exp(-abs(x)),s=1.0f+z;return max(x,0.0f)+(s==1.0f?z:log(s)*(z/(s-1.0f)));}
inline void bad_gdn(device atomic_uint* e){atomic_fetch_or_explicit(e,1u,memory_order_relaxed);}
inline uint nonfin_gdn(float x){return (as_type<uint>(x)&0x7f800000u)==0x7f800000u?1u:0u;}
inline void put_gdn(device float* p,ulong i,float x,device atomic_uint* e){if(!isfinite(x))bad_gdn(e);p[i]=x;}
inline float checked_gdn(float x,device atomic_uint* e){if(!isfinite(x))bad_gdn(e);return x;}
inline device const float* grow(device const float* p,uint row,ulong stride){return (device const float*)((device const char*)p+ulong(row)*stride);}
inline device float* grow(device float* p,uint row,ulong stride){return (device float*)((device char*)p+ulong(row)*stride);}

// The default projection retains four input rows per SIMD group and the NAX candidate handles eight.
kernel void gdn_chunk_f32(device const float* w[[buffer(0)]],device const float* x[[buffer(1)]],device float* y[[buffer(2)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint2 g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint out=g.x*4+sg,first=g.y*4;if(out>=p.outputs)return;float sums[4]={0,0,0,0};
    for(uint k=lane;k<p.columns;k+=32){const float weight=checked_gdn(w[ulong(out)*p.columns+k],e);for(uint j=0;j<4;++j)if(first+j<p.rows)sums[j]+=weight*checked_gdn(grow(x,first+j,p.input_stride)[k],e);}
    for(uint j=0;j<4;++j){const float sum=simd_sum(sums[j]);if(!lane && first+j<p.rows)put_gdn(grow(y,first+j,p.output_stride),out,sum,e);}
}
kernel void gdn_chunk_f32_nax(device const float* w[[buffer(0)]],device const float* x[[buffer(1)]],device float* y[[buffer(2)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint2 g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint out=g.x*4+sg,first=g.y*8;if(out>=p.outputs)return;float sums[8]={0,0,0,0,0,0,0,0};uint bad=0u;
    device const float* wr=w+ulong(out)*p.columns;
    if(first+7<p.rows){
        device const float* x0=grow(x,first,p.input_stride),*x1=grow(x,first+1,p.input_stride),*x2=grow(x,first+2,p.input_stride),*x3=grow(x,first+3,p.input_stride);
        device const float* x4=grow(x,first+4,p.input_stride),*x5=grow(x,first+5,p.input_stride),*x6=grow(x,first+6,p.input_stride),*x7=grow(x,first+7,p.input_stride);
        for(uint k=lane;k<p.columns;k+=32){
            const float weight=wr[k],v0=x0[k],v1=x1[k],v2=x2[k],v3=x3[k],v4=x4[k],v5=x5[k],v6=x6[k],v7=x7[k];
            bad|=nonfin_gdn(weight)|nonfin_gdn(v0)|nonfin_gdn(v1)|nonfin_gdn(v2)|nonfin_gdn(v3)|nonfin_gdn(v4)|nonfin_gdn(v5)|nonfin_gdn(v6)|nonfin_gdn(v7);
            sums[0]+=weight*v0;sums[1]+=weight*v1;sums[2]+=weight*v2;sums[3]+=weight*v3;
            sums[4]+=weight*v4;sums[5]+=weight*v5;sums[6]+=weight*v6;sums[7]+=weight*v7;
        }
    }else{
        const uint rem=p.rows>first?p.rows-first:0u;
        for(uint k=lane;k<p.columns;k+=32){
            const float weight=wr[k];bad|=nonfin_gdn(weight);
            #pragma unroll
            for(uint j=0;j<8;++j)if(j<rem){const float vj=grow(x,first+j,p.input_stride)[k];bad|=nonfin_gdn(vj);sums[j]+=weight*vj;}
        }
    }
    #pragma unroll
    for(uint j=0;j<8;++j){const float sum=simd_sum(sums[j]);if(!lane && first+j<p.rows){bad|=nonfin_gdn(sum);grow(y,first+j,p.output_stride)[out]=sum;}}
    if(bad)bad_gdn(e);
}
kernel void gdn_chunk_conv(device const float* history[[buffer(0)]],device const float* x[[buffer(1)]],device const float* w[[buffer(2)]],device float* next[[buffer(3)]],device float* y[[buffer(4)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint id[[thread_position_in_grid]]){
    if(id>=p.rows*p.columns)return;const uint t=id/p.columns,c=id%p.columns;float a[4];
    for(uint j=0;j<4;++j){const uint time=t+j;a[j]=checked_gdn(time<3?history[ulong(time)*p.columns+c]:grow(x,time-3,p.input_stride)[c],e);}
    float result=a[3]*checked_gdn(w[4*c+3],e);for(uint j=0;j<3;++j)result+=a[j]*checked_gdn(w[4*c+j],e);
    put_gdn(grow(y,t,p.output_stride),c,result,e);
    if(!t)for(uint j=0;j<3;++j){const uint time=p.rows+j;put_gdn(next,ulong(j)*p.columns+c,checked_gdn(time<3?history[ulong(time)*p.columns+c]:grow(x,time-3,p.input_stride)[c],e),e);}
}
kernel void gdn_chunk_prepare(device const float* x[[buffer(0)]],device float* q[[buffer(1)]],device float* k[[buffer(2)]],device float* v[[buffer(3)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint id=g*4+sg;if(id>=p.rows*p.heads)return;const uint t=id/p.heads,h=id%p.heads,block=p.key_heads*128,base=(h%p.key_heads)*128;const auto row=grow(x,t,p.input_stride);float qs=0,ks=0;
    for(uint j=lane;j<128;j+=32){const float qv=checked_gdn(row[base+j],e),kv=checked_gdn(row[block+base+j],e);qs+=qv*qv;ks+=kv*kv;}
    qs=simd_sum(qs);ks=simd_sum(ks);const float qd=p.l2_mode?max(qs,p.l2_epsilon):qs+p.l2_epsilon,kd=p.l2_mode?max(ks,p.l2_epsilon):ks+p.l2_epsilon;
    if(!isfinite(qs)||!isfinite(ks)||!isfinite(qd)||!isfinite(kd)){bad_gdn(e);return;}const float qi=rsqrt(qd),ki=rsqrt(kd);
    for(uint j=lane;j<128;j+=32){put_gdn(grow(q,t,p.output_stride),h*128+j,(row[base+j]*qi)*p.query_scale,e);put_gdn(grow(k,t,p.output_stride),h*128+j,row[block+base+j]*ki,e);}
    for(uint j=lane;j<p.value_width;j+=32)put_gdn(grow(v,t,p.auxiliary_stride),h*p.value_width+j,checked_gdn(row[2*block+h*p.value_width+j],e),e);
}
kernel void gdn_chunk_gates(device const float* alpha[[buffer(0)]],device const float* beta_logits[[buffer(1)]],device const float* rate[[buffer(2)]],device const float* bias[[buffer(3)]],device float* logg[[buffer(4)]],device float* beta[[buffer(5)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint id[[thread_position_in_grid]]){
    if(id>=p.rows*p.heads)return;const uint t=id/p.heads,h=id%p.heads;const float a=checked_gdn(grow(alpha,t,p.gate_stride)[h],e),b=checked_gdn(grow(beta_logits,t,p.gate_stride)[h],e),r=checked_gdn(rate[h],e),d=checked_gdn(bias[h],e),sum=a+d;
    if(r>0||!isfinite(sum)){bad_gdn(e);return;}put_gdn(grow(logg,t,p.gate_stride),h,r*softplus_gdn(sum),e);put_gdn(grow(beta,t,p.gate_stride),h,sigmoid_gdn(b),e);
}
// Four value columns per threadgroup, one SIMD group per column. Token order is serial inside this fused kernel. There is no per-token state tape.
kernel void gdn_chunk_fused(device const float* accepted[[buffer(0)]],device const float* q[[buffer(1)]],device const float* k[[buffer(2)]],device const float* v[[buffer(3)]],device const float* logg[[buffer(4)]],device const float* beta[[buffer(5)]],device float* next[[buffer(6)]],device float* y[[buffer(8)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint id=g*4+sg;if(id>=p.heads*p.value_width)return;if(p.mode>1||((p.mode!=0)&&(!p.key_heads||(p.heads%p.key_heads)!=0))){bad_gdn(e);return;}
    const uint h=id/p.value_width,col=id%p.value_width,qk_h=p.mode?(h%p.key_heads):h;float state[4];
    for(uint j=0;j<4;++j)state[j]=checked_gdn(accepted[(ulong(h)*128+lane*4+j)*p.value_width+col],e);
    for(uint t=0;t<p.rows;++t){const float exponent=checked_gdn(grow(logg,t,p.gate_stride)[h],e),b=checked_gdn(grow(beta,t,p.gate_stride)[h],e);if(exponent>0||b<0||b>1){bad_gdn(e);return;}const float a=exp(exponent);float prediction=0;
        for(uint j=0;j<4;++j){const uint at=qk_h*128+lane*4+j;state[j]*=a;prediction+=state[j]*checked_gdn(grow(k,t,p.input_stride)[at],e);}
        prediction=simd_sum(prediction);const float delta=(checked_gdn(grow(v,t,p.auxiliary_stride)[id],e)-prediction)*b;float out=0;
        for(uint j=0;j<4;++j){const uint at=qk_h*128+lane*4+j;state[j]+=grow(k,t,p.input_stride)[at]*delta;checked_gdn(state[j],e);out+=state[j]*checked_gdn(grow(q,t,p.input_stride)[at],e);}out=simd_sum(out);if(!lane)put_gdn(grow(y,t,p.output_stride),id,out,e);
    }
    for(uint j=0;j<4;++j)put_gdn(next,(ulong(h)*128+lane*4+j)*p.value_width+col,state[j],e);
}

inline float gdn_conv4_silu_channel(device const float* history,device const float* x,device const float* w,device float* next,uint rows,uint raw_cols,ulong in_stride,uint t,uint c,device atomic_uint* e){
    float a[4];
    for(uint j=0;j<4;++j){const uint time=t+j;a[j]=checked_gdn(time<3?history[ulong(time)*raw_cols+c]:grow(x,time-3,in_stride)[c],e);}
    float res=a[3]*checked_gdn(w[4*c+3],e);
    for(uint j=0;j<3;++j)res+=a[j]*checked_gdn(w[4*c+j],e);
    checked_gdn(res,e);
    if(!t)for(uint j=0;j<3;++j){const uint time=rows+j;put_gdn(next,ulong(j)*raw_cols+c,checked_gdn(time<3?history[ulong(time)*raw_cols+c]:grow(x,time-3,in_stride)[c],e),e);}
    if(!isfinite(res)){bad_gdn(e);return res;}
    return checked_gdn(res*sigmoid_gdn(res),e);
}

kernel void gdn_chunk_proj_gates_f32(device const float* w_alpha[[buffer(0)]],device const float* w_beta[[buffer(1)]],device const float* x[[buffer(2)]],device const float* rate[[buffer(3)]],device const float* bias[[buffer(4)]],device float* logg[[buffer(5)]],device float* beta[[buffer(6)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint h=g.x*4+sg,first=g.y*4;
    threadgroup float sx[4][256];
    float asums[4]={0,0,0,0},bsums[4]={0,0,0,0};
    for(uint k0=0;k0<p.columns;k0+=256){
        const uint cnt=min(256u,p.columns-k0);
        for(uint j=0;j<4;++j){
            if(first+j<p.rows){
                device const float* x_row=grow(x,first+j,p.input_stride)+k0;
                if(tid<cnt)sx[j][tid]=checked_gdn(x_row[tid],e);
                if(uint(tid)+128u<cnt)sx[j][uint(tid)+128u]=checked_gdn(x_row[uint(tid)+128u],e);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(h<p.heads){
            for(uint kk=lane;kk<cnt;kk+=32){
                const float wa=checked_gdn(w_alpha[ulong(h)*p.columns+k0+kk],e);
                const float wb=checked_gdn(w_beta[ulong(h)*p.columns+k0+kk],e);
                for(uint j=0;j<4;++j)if(first+j<p.rows){asums[j]+=wa*sx[j][kk];bsums[j]+=wb*sx[j][kk];}
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(h>=p.heads)return;
    const float r=checked_gdn(rate[h],e),d=checked_gdn(bias[h],e);
    for(uint j=0;j<4;++j){
        const float asum=simd_sum(asums[j]),bsum=simd_sum(bsums[j]);
        if(!lane && first+j<p.rows){
            const float a=checked_gdn(asum,e),b=checked_gdn(bsum,e),sum=a+d;
            if(r>0||!isfinite(sum)){bad_gdn(e);}else{
                put_gdn(grow(logg,first+j,p.gate_stride),h,r*softplus_gdn(sum),e);
                put_gdn(grow(beta,first+j,p.gate_stride),h,sigmoid_gdn(b),e);
            }
        }
    }
}

kernel void gdn_chunk_conv_silu_prepare(device const float* history[[buffer(0)]],device const float* x[[buffer(1)]],device const float* w[[buffer(2)]],device float* next[[buffer(3)]],device float* q[[buffer(4)]],device float* k[[buffer(5)]],device float* v[[buffer(6)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint id=g*4+sg;if(id>=p.rows*p.key_heads)return;
    if(!p.key_heads||(p.heads%p.key_heads)!=0||p.mode>1){bad_gdn(e);return;}
    const uint t=id/p.key_heads,kh=id%p.key_heads,ratio=p.heads/p.key_heads,block=p.key_heads*128,base=kh*128;
    const uint raw_cols=2*block+p.heads*p.value_width;
    float qv[4],kv[4],qs=0,ks=0;
    for(uint i=0;i<4;++i){
        const uint j=lane+i*32;
        qv[i]=gdn_conv4_silu_channel(history,x,w,next,p.rows,raw_cols,p.input_stride,t,base+j,e);
        kv[i]=gdn_conv4_silu_channel(history,x,w,next,p.rows,raw_cols,p.input_stride,t,block+base+j,e);
        qs+=qv[i]*qv[i];ks+=kv[i]*kv[i];
    }
    qs=simd_sum(qs);ks=simd_sum(ks);
    const float qd=p.l2_mode?max(qs,p.l2_epsilon):qs+p.l2_epsilon,kd=p.l2_mode?max(ks,p.l2_epsilon):ks+p.l2_epsilon;
    if(!isfinite(qs)||!isfinite(ks)||!isfinite(qd)||!isfinite(kd)){bad_gdn(e);}else{
        const float qi=rsqrt(qd),ki=rsqrt(kd);
        if(p.mode){
            for(uint i=0;i<4;++i){
                const uint j=lane+i*32;
                put_gdn(grow(q,t,p.output_stride),kh*128+j,(qv[i]*qi)*p.query_scale,e);
                put_gdn(grow(k,t,p.output_stride),kh*128+j,kv[i]*ki,e);
            }
        }else{
            for(uint r=0;r<ratio;++r){
                const uint h=r*p.key_heads+kh;
                for(uint i=0;i<4;++i){
                    const uint j=lane+i*32;
                    put_gdn(grow(q,t,p.output_stride),h*128+j,(qv[i]*qi)*p.query_scale,e);
                    put_gdn(grow(k,t,p.output_stride),h*128+j,kv[i]*ki,e);
                }
            }
        }
    }
    for(uint r=0;r<ratio;++r){
        const uint h=r*p.key_heads+kh;
        for(uint j=lane;j<p.value_width;j+=32){
            const float vv=gdn_conv4_silu_channel(history,x,w,next,p.rows,raw_cols,p.input_stride,t,2*block+h*p.value_width+j,e);
            put_gdn(grow(v,t,p.auxiliary_stride),h*p.value_width+j,vv,e);
        }
    }
}
kernel void gdn_chunk_rms_gate(device const float* x[[buffer(0)]],device const float* gains[[buffer(1)]],device const float* z[[buffer(2)]],device float* y[[buffer(3)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint id=g*4+sg;if(id>=p.rows*p.heads)return;const uint t=id/p.heads,base=(id%p.heads)*p.value_width;
    if(p.value_width==128u){
        device const float* xr=grow(x,t,p.input_stride)+base;
        device const float* zr=grow(z,t,p.auxiliary_stride)+base;
        device float* yr=grow(y,t,p.output_stride);
        float rv[4],sum=0;
        #pragma unroll
        for(uint k=0;k<4;++k){const float a=checked_gdn(xr[lane+k*32],e);rv[k]=a;sum+=a*a;}
        sum=simd_sum(sum);const float denominator=sum/128.0f+p.rms_epsilon;
        if(!isfinite(sum)||!isfinite(denominator)){bad_gdn(e);return;}
        const float inv=rsqrt(denominator);
        #pragma unroll
        for(uint k=0;k<4;++k){const uint j=lane+k*32;const float gain=checked_gdn(gains[j],e),gate=checked_gdn(zr[j],e);put_gdn(yr,base+j,((rv[k]*inv)*gain)*sigmoid_gdn(gate),e);}
        return;
    }
    float sum=0;for(uint j=lane;j<p.value_width;j+=32){const float a=checked_gdn(grow(x,t,p.input_stride)[base+j],e);sum+=a*a;}sum=simd_sum(sum);const float denominator=sum/float(p.value_width)+p.rms_epsilon;if(!isfinite(sum)||!isfinite(denominator)){bad_gdn(e);return;}const float inv=rsqrt(denominator);
    for(uint j=lane;j<p.value_width;j+=32){const float a=grow(x,t,p.input_stride)[base+j],gain=checked_gdn(gains[j],e),gate=checked_gdn(grow(z,t,p.auxiliary_stride)[base+j],e);put_gdn(grow(y,t,p.output_stride),base+j,((a*inv)*gain)*sigmoid_gdn(gate),e);}
}
kernel void gdn_chunk_rms_gate_nax(device const float* x[[buffer(0)]],device const float* gains[[buffer(1)]],device const float* z[[buffer(2)]],device float* y[[buffer(3)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint id=g*4+sg;if(id>=p.rows*p.heads)return;const uint t=id/p.heads,base=(id%p.heads)*p.value_width;
    device const float* xr=grow(x,t,p.input_stride)+base,*zr=grow(z,t,p.auxiliary_stride)+base;device float* yr=grow(y,t,p.output_stride)+base;
    if(p.value_width==128u){
        const float av[4]={xr[lane],xr[lane+32u],xr[lane+64u],xr[lane+96u]};
        uint bad=nonfin_gdn(av[0])|nonfin_gdn(av[1])|nonfin_gdn(av[2])|nonfin_gdn(av[3]);
        float sum=0.0f;
        #pragma unroll
        for(uint k=0u;k<4u;++k)sum+=av[k]*av[k];
        sum=simd_sum(sum);const float denominator=sum/128.0f+p.rms_epsilon;
        if(simd_or(bad)||nonfin_gdn(sum)||nonfin_gdn(denominator)){bad_gdn(e);return;}
        const float inv=rsqrt(denominator);
        #pragma unroll
        for(uint k=0u;k<4u;++k){
            const uint j=uint(lane)+k*32u;
            const float gain=gains[j],gate=zr[j];
            bad|=nonfin_gdn(gain)|nonfin_gdn(gate);
            const float out=((av[k]*inv)*gain)*sigmoid_gdn(gate);
            bad|=nonfin_gdn(out);yr[j]=out;
        }
        if(bad)bad_gdn(e);
        return;
    }
    uint bad=0u;float sum=0;for(uint j=lane;j<p.value_width;j+=32){const float a=xr[j];bad|=nonfin_gdn(a);sum+=a*a;}sum=simd_sum(sum);const float denominator=sum/float(p.value_width)+p.rms_epsilon;
    if(bad||nonfin_gdn(sum)||nonfin_gdn(denominator)){bad_gdn(e);return;}const float inv=rsqrt(denominator);
    for(uint j=lane;j<p.value_width;j+=32){const float a=xr[j],gain=gains[j],gate=zr[j];bad|=nonfin_gdn(gain)|nonfin_gdn(gate);const float out=((a*inv)*gain)*sigmoid_gdn(gate);bad|=nonfin_gdn(out);yr[j]=out;}
    if(bad)bad_gdn(e);
}
kernel void gdn_chunk_rms_rows(device const float* x[[buffer(0)]],device const float* gains[[buffer(1)]],device float* y[[buffer(2)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint groups=p.columns/p.value_width,id=g*4+sg;if(id>=p.rows*groups)return;const uint t=id/groups,base=(id%groups)*p.value_width;
    if(p.value_width==128u){
        device const float* xr=grow(x,t,p.input_stride)+base;
        device float* yr=grow(y,t,p.output_stride);
        const uint gbase=p.mode?base:0u;
        float rv[4],sum=0;
        #pragma unroll
        for(uint k=0;k<4;++k){const float a=checked_gdn(xr[lane+k*32],e);rv[k]=a;sum+=a*a;}
        sum=simd_sum(sum);const float denominator=sum/128.0f+p.rms_epsilon;
        if(!isfinite(sum)||!isfinite(denominator)){bad_gdn(e);return;}
        const float inv=rsqrt(denominator);
        #pragma unroll
        for(uint k=0;k<4;++k){const uint j=lane+k*32;put_gdn(yr,base+j,(rv[k]*inv)*checked_gdn(gains[gbase+j],e),e);}
        return;
    }
    if(p.value_width==256u){
        device const float* xr=grow(x,t,p.input_stride)+base;
        device float* yr=grow(y,t,p.output_stride);
        const uint gbase=p.mode?base:0u;
        float rv[8],sum=0;
        #pragma unroll
        for(uint k=0;k<8;++k){const float a=checked_gdn(xr[lane+k*32],e);rv[k]=a;sum+=a*a;}
        sum=simd_sum(sum);const float denominator=sum/256.0f+p.rms_epsilon;
        if(!isfinite(sum)||!isfinite(denominator)){bad_gdn(e);return;}
        const float inv=rsqrt(denominator);
        #pragma unroll
        for(uint k=0;k<8;++k){const uint j=lane+k*32;put_gdn(yr,base+j,(rv[k]*inv)*checked_gdn(gains[gbase+j],e),e);}
        return;
    }
    float sum=0;for(uint j=lane;j<p.value_width;j+=32){const float a=checked_gdn(grow(x,t,p.input_stride)[base+j],e);sum+=a*a;}sum=simd_sum(sum);const float denominator=sum/float(p.value_width)+p.rms_epsilon;if(!isfinite(sum)||!isfinite(denominator)){bad_gdn(e);return;}const float inv=rsqrt(denominator);
    for(uint j=lane;j<p.value_width;j+=32)put_gdn(grow(y,t,p.output_stride),base+j,(grow(x,t,p.input_stride)[base+j]*inv)*checked_gdn(gains[(p.mode?base:0)+j],e),e);
}
kernel void gdn_chunk_rms_rows_nax(device const float* x[[buffer(0)]],device const float* gains[[buffer(1)]],device float* y[[buffer(2)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint groups=p.columns/p.value_width,id=g*4+sg;if(id>=p.rows*groups)return;const uint t=id/groups,base=(id%groups)*p.value_width;
    device const float* xr=grow(x,t,p.input_stride)+base,*gr=gains+(p.mode?base:0);device float* yr=grow(y,t,p.output_stride)+base;
    uint bad=0u;float sum=0;
    if(p.value_width==128u){
        const float av[4]={xr[lane],xr[lane+32u],xr[lane+64u],xr[lane+96u]};
        bad=nonfin_gdn(av[0])|nonfin_gdn(av[1])|nonfin_gdn(av[2])|nonfin_gdn(av[3]);
        #pragma unroll
        for(uint k=0u;k<4u;++k)sum+=av[k]*av[k];
        sum=simd_sum(sum);const float denominator=sum/128.0f+p.rms_epsilon;
        if(simd_or(bad)||nonfin_gdn(sum)||nonfin_gdn(denominator)){bad_gdn(e);return;}
        const float inv=rsqrt(denominator);
        #pragma unroll
        for(uint k=0u;k<4u;++k){
            const uint j=uint(lane)+k*32u;
            const float gn=gr[j];bad|=nonfin_gdn(gn);
            const float out=(av[k]*inv)*gn;bad|=nonfin_gdn(out);yr[j]=out;
        }
    }else if(p.value_width==256u){
        float av[8];
        #pragma unroll
        for(uint k=0u;k<8u;++k){const float a=xr[uint(lane)+k*32u];av[k]=a;bad|=nonfin_gdn(a);sum+=a*a;}
        sum=simd_sum(sum);const float denominator=sum/256.0f+p.rms_epsilon;
        if(simd_or(bad)||nonfin_gdn(sum)||nonfin_gdn(denominator)){bad_gdn(e);return;}
        const float inv=rsqrt(denominator);
        #pragma unroll
        for(uint k=0u;k<8u;++k){
            const uint j=uint(lane)+k*32u;
            const float gn=gr[j];bad|=nonfin_gdn(gn);
            const float out=(av[k]*inv)*gn;bad|=nonfin_gdn(out);yr[j]=out;
        }
    }else{
        for(uint j=lane;j<p.value_width;j+=32){const float a=xr[j];bad|=nonfin_gdn(a);sum+=a*a;}sum=simd_sum(sum);const float denominator=sum/float(p.value_width)+p.rms_epsilon;
        if(bad||nonfin_gdn(sum)||nonfin_gdn(denominator)){bad_gdn(e);return;}const float inv=rsqrt(denominator);
        for(uint j=lane;j<p.value_width;j+=32){const float gn=gr[j];bad|=nonfin_gdn(gn);const float out=(xr[j]*inv)*gn;bad|=nonfin_gdn(out);yr[j]=out;}
    }
    if(bad)bad_gdn(e);
}
// One (row, group) per 128-thread group so small-batch and wide RMS use all GPU cores, cache inputs in registers, and preserve the exact 32-lane graph_rms FMA reduction tree.
kernel void gdn_chunk_rms_rows_fast(device const float* x[[buffer(0)]],device const float* gains[[buffer(1)]],device float* y[[buffer(2)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    threadgroup float v_tile[128][32];
    threadgroup float shared_inv;
    threadgroup uint shared_abort;
    threadgroup uint sg_bad[4];
    if(!tid){shared_inv=0.0f;shared_abort=0u;}
    const uint groups=p.columns/p.value_width;
    if(g>=p.rows*groups)return;
    const uint t=g/groups,base=(g%groups)*p.value_width;
    device const float* xr=grow(x,t,p.input_stride)+base,*gr=gains+(p.mode?base:0);
    device float* yr=grow(y,t,p.output_stride)+base;
    if((p.value_width&127u)!=0u||p.value_width>4096u){
        if(sg==0u){
            uint bad=0u;float sum=0.0f;
            for(uint j=lane;j<p.value_width;j+=32u){const float a=xr[j];bad|=nonfin_gdn(a);sum+=a*a;}
            sum=simd_sum(sum);const float denominator=sum/float(p.value_width)+p.rms_epsilon;
            if(simd_or(bad)||nonfin_gdn(sum)||nonfin_gdn(denominator)){bad_gdn(e);return;}
            const float inv=rsqrt(denominator);
            for(uint j=lane;j<p.value_width;j+=32u){const float gn=gr[j];bad|=nonfin_gdn(gn);const float out=(xr[j]*inv)*gn;bad|=nonfin_gdn(out);yr[j]=out;}
            if(bad)bad_gdn(e);
        }
        return;
    }
    const uint chunks128=p.value_width>>7u;
    uint bad=0u;
    #pragma unroll 4
    for(uint k=0u;k<chunks128;++k){
        const uint idx=k*128u+uint(tid);
        const float v=xr[idx];
        bad|=nonfin_gdn(v);
        v_tile[k*4u+uint(sg)][lane]=v;
    }
    const uint sbad=simd_or(bad);
    if(!lane)sg_bad[sg]=sbad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0u){
        const uint steps32=chunks128*4u;
        float sum=0.0f;
        #pragma unroll 4
        for(uint m=0u;m<steps32;++m){const float vm=v_tile[m][lane];sum+=vm*vm;}
        sum=simd_sum(sum);
        if(!lane){
            const uint all_bad=sg_bad[0]|sg_bad[1]|sg_bad[2]|sg_bad[3];
            const float denom=sum/float(p.value_width)+p.rms_epsilon;
            const uint abort_flag=(all_bad||nonfin_gdn(sum)||nonfin_gdn(denom))?1u:0u;
            shared_abort=abort_flag;
            shared_inv=abort_flag?0.0f:rsqrt(denom);
            if(abort_flag)bad_gdn(e);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(shared_abort)return;
    const float inv=shared_inv;
    uint wbad=0u;
    #pragma unroll 4
    for(uint k=0u;k<chunks128;++k){
        const uint idx=k*128u+uint(tid);
        const float rv=v_tile[k*4u+uint(sg)][lane];
        const float gn=gr[idx];
        wbad|=nonfin_gdn(gn);
        const float out=(rv*inv)*gn;
        wbad|=nonfin_gdn(out);
        yr[idx]=out;
    }
    if(wbad)bad_gdn(e);
}
kernel void gdn_chunk_prefix64(device const float* logg[[buffer(0)]],device float* prefix[[buffer(1)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint h[[thread_position_in_grid]]){
    if(h>=p.heads)return;float sum=0;for(uint i=0;i<p.count;++i){const float g=checked_gdn(grow(logg,p.offset+i,p.gate_stride)[h],e);if(g>0){bad_gdn(e);return;}sum+=g;put_gdn(prefix,h*64+i,sum,e);}
}
kernel void gdn_chunk_coeff64(device const float* q[[buffer(0)]],device const float* k[[buffer(1)]],device const float* prefix[[buffer(2)]],device const float* beta[[buffer(3)]],device float* lower[[buffer(4)]],device float* attention[[buffer(5)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint g[[threadgroup_position_in_grid]],ushort sg[[simdgroup_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]]){
    const uint id=g*4+sg;if(id>=p.heads*p.count*p.count)return;const uint h=id/(p.count*p.count),i=(id/p.count)%p.count,j=id%p.count,at=(h*64+i)*64+j;if(j>i){if(!lane){lower[at]=0;attention[at]=0;}return;}float kk=0,qk=0;
    for(uint a=lane;a<128;a+=32){const float kj=checked_gdn(grow(k,p.offset+j,p.input_stride)[h*128+a],e);kk+=checked_gdn(grow(k,p.offset+i,p.input_stride)[h*128+a],e)*kj;qk+=checked_gdn(grow(q,p.offset+i,p.input_stride)[h*128+a],e)*kj;}kk=simd_sum(kk);qk=simd_sum(qk);
    if(!lane){const float b=checked_gdn(grow(beta,p.offset+i,p.gate_stride)[h],e),difference=checked_gdn(prefix[h*64+i]-prefix[h*64+j],e);if(b<0||b>1){bad_gdn(e);return;}const float decay=exp(difference);put_gdn(lower,at,j<i?(b*kk)*decay:0,e);put_gdn(attention,at,qk*decay,e);}
}
// Direct forward substitution is the same unit-lower triangular system as the pinned CS64 builder, without materialising a full inverse. Each thread owns one value column.
kernel void gdn_chunk_solve64(device const float* state[[buffer(0)]],device const float* k[[buffer(1)]],device const float* v[[buffer(2)]],device const float* prefix[[buffer(3)]],device const float* beta[[buffer(4)]],device const float* lower[[buffer(5)]],device float* residual[[buffer(6)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint id[[thread_position_in_grid]]){
    if(id>=p.heads*p.value_width)return;const uint h=id/p.value_width,c=id%p.value_width;float r[64];
    for(uint i=0;i<p.count;++i){float predicted=0;for(uint a=0;a<128;++a)predicted+=checked_gdn(state[(ulong(h)*128+a)*p.value_width+c],e)*checked_gdn(grow(k,p.offset+i,p.input_stride)[h*128+a],e);const float b=checked_gdn(grow(beta,p.offset+i,p.gate_stride)[h],e),gv=checked_gdn(prefix[h*64+i],e);if(b<0||b>1||gv>0){bad_gdn(e);return;}float x=b*(checked_gdn(grow(v,p.offset+i,p.auxiliary_stride)[id],e)-exp(gv)*predicted);for(uint j=0;j<i;++j)x-=lower[(h*64+i)*64+j]*r[j];r[i]=x;put_gdn(residual,ulong(i)*p.heads*p.value_width+id,x,e);}
}
kernel void gdn_chunk_emit64(device const float* state[[buffer(0)]],device const float* q[[buffer(1)]],device const float* k[[buffer(2)]],device const float* prefix[[buffer(3)]],device const float* attention[[buffer(4)]],device const float* residual[[buffer(5)]],device float* next[[buffer(6)]],device float* y[[buffer(8)]],constant GdnChunkParams& p[[buffer(14)]],device atomic_uint* e[[buffer(7)]],uint id[[thread_position_in_grid]]){
    if(id>=p.heads*p.value_width)return;const uint h=id/p.value_width,c=id%p.value_width;const uint width=p.heads*p.value_width;
#ifndef TOKSPUR_TEST_GDN_CS64_REFERENCE
    // The 128-thread dispatch owns one complete head at value_width 128.
    // Its decay factors depend only on (head,token), not on key/value columns.
    // Preserve all FP32 products and sequential sums; share only the factors.
    if (p.value_width == 128u) {
        threadgroup float prefix_exp[64], tail_exp[64];
        const float last = checked_gdn(prefix[h * 64 + p.count - 1], e);
        const float last_exp = exp(last);
        if (c < p.count) {
            prefix_exp[c] = exp(checked_gdn(prefix[h * 64 + c], e));
            tail_exp[c] = exp(checked_gdn(last - prefix[h * 64 + c], e));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < p.count; ++i) {
            float x = 0;
            for (uint a = 0; a < 128; ++a)
                x += checked_gdn(state[(ulong(h) * 128 + a) * p.value_width + c], e)
                    * checked_gdn(grow(q, p.offset + i, p.input_stride)[h * 128 + a], e);
            x *= prefix_exp[i];
            for (uint j = 0; j <= i; ++j)
                x += attention[(h * 64 + i) * 64 + j] * residual[ulong(j) * width + id];
            put_gdn(grow(y, p.offset + i, p.output_stride), id, x, e);
        }
        for (uint a = 0; a < 128; ++a) {
            float x = last_exp * checked_gdn(state[(ulong(h) * 128 + a) * p.value_width + c], e);
            for (uint j = 0; j < p.count; ++j)
                x += (tail_exp[j] * checked_gdn(grow(k, p.offset + j, p.input_stride)[h * 128 + a], e))
                    * residual[ulong(j) * width + id];
            put_gdn(next, (ulong(h) * 128 + a) * p.value_width + c, x, e);
        }
        return;
    }
#endif
    for(uint i=0;i<p.count;++i){float x=0;for(uint a=0;a<128;++a)x+=checked_gdn(state[(ulong(h)*128+a)*p.value_width+c],e)*checked_gdn(grow(q,p.offset+i,p.input_stride)[h*128+a],e);x*=exp(checked_gdn(prefix[h*64+i],e));for(uint j=0;j<=i;++j)x+=attention[(h*64+i)*64+j]*residual[ulong(j)*width+id];put_gdn(grow(y,p.offset+i,p.output_stride),id,x,e);}
    const float last=checked_gdn(prefix[h*64+p.count-1],e);
    for(uint a=0;a<128;++a){float x=exp(last)*checked_gdn(state[(ulong(h)*128+a)*p.value_width+c],e);for(uint j=0;j<p.count;++j)x+=(exp(checked_gdn(last-prefix[h*64+j],e))*checked_gdn(grow(k,p.offset+j,p.input_stride)[h*128+a],e))*residual[ulong(j)*width+id];put_gdn(next,(ulong(h)*128+a)*p.value_width+c,x,e);}
}
