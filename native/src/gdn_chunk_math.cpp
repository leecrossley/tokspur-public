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
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tokspur {
namespace {
void need(bool x, const char* why) { if (!x) throw FormatError(why); }
uint32_t ceildiv(uint64_t n, uint32_t d) { need(n && n <= UINT32_MAX, "GDN chunk grid bound"); return uint32_t((n+d-1)/d); }
uint64_t extent(uint32_t n, uint32_t columns, uint64_t stride, bool allow_nax_2k = false) {
    const uint32_t max_rows = allow_nax_2k ? 2048u : 512u;
    need(n && n <= max_rows && columns && columns <= 65536 && stride % 4 == 0 && stride >= uint64_t(columns)*4, "GDN chunk row geometry");
    return checked_add(checked_mul(n-1,stride),uint64_t(columns)*4);
}
bool overlap(GpuView a, GpuView b) {
    return a.owner == b.owner && a.slot == b.slot && a.offset < checked_add(b.offset,b.bytes) && b.offset < checked_add(a.offset,a.bytes);
}
void finite(std::span<const double> x) { for (double v:x) need(std::isfinite(v),"nonfinite GDN chunk reference"); }
}
uint64_t gdn_chunk_row_extent(GpuRowsF32 x, bool allow_nax_2k) { const auto n=extent(x.rows,x.columns,x.stride_bytes,allow_nax_2k); need(x.view.owner && x.view.offset%4==0 && x.view.bytes>=n,"GDN chunk row view extent"); return n; }
GdnChunkSpec gdn_chunk_spec(const GdnChunkDispatch& d) {
    const auto& p=d.params;
    switch(d.op) {
    case GdnChunkOp::projection_f32:return {"gdn_chunk_f32",3,4,{ceildiv(p.outputs,4),ceildiv(p.rows,4),1}};
    case GdnChunkOp::convolution:return {"gdn_chunk_conv",7,24,{ceildiv(uint64_t(p.rows)*p.columns,128),1,1}};
    case GdnChunkOp::prepare:return {"gdn_chunk_prepare",1,14,{ceildiv(uint64_t(p.rows)*p.heads,4),1,1}};
    case GdnChunkOp::gates:return {"gdn_chunk_gates",15,48,{ceildiv(uint64_t(p.rows)*p.heads,128),1,1}};
    case GdnChunkOp::fused:return {"gdn_chunk_fused",63,320,{ceildiv(uint64_t(p.heads)*p.value_width,4),1,1}};
    case GdnChunkOp::rms_gate:return {"gdn_chunk_rms_gate",7,8,{ceildiv(uint64_t(p.rows)*p.heads,4),1,1}};
    case GdnChunkOp::rms_rows:need(p.value_width!=0,"zero GDN RMS width");return {"gdn_chunk_rms_rows",3,4,{ceildiv(uint64_t(p.rows)*p.columns/p.value_width,4),1,1}};
    case GdnChunkOp::prefix64:return {"gdn_chunk_prefix64",1,2,{ceildiv(p.heads,128),1,1}};
    case GdnChunkOp::coefficients64:return {"gdn_chunk_coeff64",15,48,{ceildiv(uint64_t(p.heads)*p.count*p.count,4),1,1}};
    case GdnChunkOp::solve64:return {"gdn_chunk_solve64",63,64,{ceildiv(uint64_t(p.heads)*p.value_width,128),1,1}};
    case GdnChunkOp::emit64:return {"gdn_chunk_emit64",63,320,{ceildiv(uint64_t(p.heads)*p.value_width,128),1,1}};
    case GdnChunkOp::projection_gates_f32:return {"gdn_chunk_proj_gates_f32",31,96,{ceildiv(p.heads,4),ceildiv(p.rows,4),1}};
    case GdnChunkOp::conv_silu_prepare:return {"gdn_chunk_conv_silu_prepare",7,120,{ceildiv(uint64_t(p.rows)*p.key_heads,4),1,1}};
    }
    throw FormatError("unknown GDN chunk dispatch");
}
void validate_gdn_chunk_dispatch(const GdnChunkDispatch& d, bool allow_nax_2k) {
    const auto& p=d.params;
    const uint32_t max_rows = allow_nax_2k ? 2048u : 512u;
    need(p.rows && p.rows<=max_rows && !p.reserved && p.reserved_float==0,"GDN chunk row count/reserved fields");
    const auto spec=gdn_chunk_spec(d);
    const auto minimum=[&](uint32_t b,uint64_t n){const auto v=d.buffers[b];need(v.owner && v.offset%4==0 && v.bytes>=n,"GDN chunk buffer extent"); checked_add(v.offset,v.bytes);};
    const auto exact=[&](uint32_t b,uint64_t n){minimum(b,n);need(d.buffers[b].bytes==n,"GDN chunk exact buffer extent");};
    const auto rows=[&](uint32_t b,uint32_t columns,uint64_t stride){minimum(b,extent(p.rows,columns,stride,allow_nax_2k));};
    if(d.op==GdnChunkOp::projection_f32){
        need(p.columns && p.columns<=65536 && p.outputs && p.outputs<=65536,"GDN F32 projection geometry");
        exact(0,checked_mul(checked_mul(p.columns,p.outputs),4));rows(1,p.columns,p.input_stride);rows(2,p.outputs,p.output_stride);
    } else if(d.op==GdnChunkOp::convolution){
        need(p.columns && p.columns<=65536,"GDN convolution channels");exact(0,uint64_t(p.columns)*12);rows(1,p.columns,p.input_stride);exact(2,uint64_t(p.columns)*16);exact(3,uint64_t(p.columns)*12);rows(4,p.columns,p.output_stride);
    } else if(d.op==GdnChunkOp::rms_rows){
        need(p.value_width && p.value_width<=65536 && p.columns && p.columns%p.value_width==0 && p.mode<=1 && std::isfinite(p.rms_epsilon) && p.rms_epsilon>0,"GDN row RMS geometry");
        rows(0,p.columns,p.input_stride);exact(1,uint64_t(p.mode?p.columns:p.value_width)*4);rows(2,p.columns,p.output_stride);
    } else if(d.op==GdnChunkOp::projection_gates_f32){
        need(p.columns && p.columns<=65536 && p.heads && p.heads<=48,"GDN F32 gate projection geometry");
        const uint64_t wb=checked_mul(checked_mul(p.columns,p.heads),4);
        exact(0,wb);exact(1,wb);rows(2,p.columns,p.input_stride);exact(3,uint64_t(p.heads)*4);exact(4,uint64_t(p.heads)*4);rows(5,p.heads,p.gate_stride);rows(6,p.heads,p.gate_stride);
    } else {
        need(p.heads && p.heads<=48 && p.key_width==128 && p.value_width && p.value_width<=128,"GDN chunk core geometry");
        const uint32_t qk=p.heads*128,v=p.heads*p.value_width;
        const uint64_t state=uint64_t(qk)*p.value_width*4, prefix=uint64_t(p.heads)*64*4, matrix=uint64_t(p.heads)*64*64*4, residual=uint64_t(64)*v*4;
        switch(d.op){
        case GdnChunkOp::prepare:
            need(p.key_heads && p.heads%p.key_heads==0 && p.key_heads<=p.heads && p.l2_mode<=1 && std::isfinite(p.query_scale) && p.query_scale>0 && std::isfinite(p.l2_epsilon) && p.l2_epsilon>0,"GDN chunk preparation policy");
            rows(0,2*p.key_heads*128+v,p.input_stride);rows(1,qk,p.output_stride);rows(2,qk,p.output_stride);rows(3,v,p.auxiliary_stride);break;
        case GdnChunkOp::conv_silu_prepare:{
            need(p.key_heads && p.heads%p.key_heads==0 && p.key_heads<=p.heads && p.mode<=1 && p.l2_mode<=1 && std::isfinite(p.query_scale) && p.query_scale>0 && std::isfinite(p.l2_epsilon) && p.l2_epsilon>0,"GDN chunk fused conv-prepare policy");
            const uint32_t raw_cols=2*p.key_heads*128+v,qk_out=(p.mode?p.key_heads:p.heads)*128;
            exact(0,uint64_t(raw_cols)*12);rows(1,raw_cols,p.input_stride);exact(2,uint64_t(raw_cols)*16);exact(3,uint64_t(raw_cols)*12);rows(4,qk_out,p.output_stride);rows(5,qk_out,p.output_stride);rows(6,v,p.auxiliary_stride);break;}
        case GdnChunkOp::gates:
            rows(0,p.heads,p.gate_stride);rows(1,p.heads,p.gate_stride);exact(2,uint64_t(p.heads)*4);exact(3,uint64_t(p.heads)*4);rows(4,p.heads,p.gate_stride);rows(5,p.heads,p.gate_stride);break;
        case GdnChunkOp::fused:{
            need(p.mode<=1 && (!p.mode || (p.key_heads && p.heads%p.key_heads==0 && p.key_heads<=p.heads)),"GDN chunk fused QK mode");
            const uint32_t qk_in=(p.mode?p.key_heads:p.heads)*128;
            exact(0,state);rows(1,qk_in,p.input_stride);rows(2,qk_in,p.input_stride);rows(3,v,p.auxiliary_stride);rows(4,p.heads,p.gate_stride);rows(5,p.heads,p.gate_stride);exact(6,state);rows(8,v,p.output_stride);break;}
        case GdnChunkOp::rms_gate:
            need(std::isfinite(p.rms_epsilon) && p.rms_epsilon>0,"GDN chunk RMS epsilon");rows(0,v,p.input_stride);exact(1,uint64_t(p.value_width)*4);rows(2,v,p.auxiliary_stride);rows(3,v,p.output_stride);break;
        case GdnChunkOp::prefix64:case GdnChunkOp::coefficients64:case GdnChunkOp::solve64:case GdnChunkOp::emit64:
            need(p.count && p.count<=64 && p.offset<p.rows && p.count<=p.rows-p.offset,"GDN CS64 active microchunk");
            if(d.op==GdnChunkOp::prefix64){rows(0,p.heads,p.gate_stride);exact(1,prefix);}
            if(d.op==GdnChunkOp::coefficients64){rows(0,qk,p.input_stride);rows(1,qk,p.input_stride);exact(2,prefix);rows(3,p.heads,p.gate_stride);exact(4,matrix);exact(5,matrix);}
            if(d.op==GdnChunkOp::solve64){exact(0,state);rows(1,qk,p.input_stride);rows(2,v,p.auxiliary_stride);exact(3,prefix);rows(4,p.heads,p.gate_stride);exact(5,matrix);exact(6,residual);}
            if(d.op==GdnChunkOp::emit64){exact(0,state);rows(1,qk,p.input_stride);rows(2,qk,p.input_stride);exact(3,prefix);exact(4,matrix);exact(5,residual);exact(6,state);rows(8,v,p.output_stride);}break;
        default:throw FormatError("invalid GDN chunk operation");
        }
    }
    const auto used=spec.reads|spec.writes;
    for(uint32_t i=0;i<d.buffers.size();++i){
        if(!(used&(1u<<i)))need(!d.buffers[i].owner && !d.buffers[i].bytes,"unused GDN chunk binding");
        if(spec.writes&(1u<<i))for(uint32_t j=0;j<d.buffers.size();++j)if(j!=i && (used&(1u<<j)))need(!overlap(d.buffers[i],d.buffers[j]),"GDN chunk overlapping live buffers");
    }
}
GdnChunkBytes gdn_chunk_bytes(GdnShape s,uint32_t n,GdnChunkMode mode,bool allow_nax_2k){
    const uint32_t max_rows = allow_nax_2k ? 2048u : 512u;
    validate_gdn_shape(s);need(n && n<=max_rows && s.key_width==128 && s.value_heads<=48 && s.value_width<=128 && s.kernel==4 && uint32_t(mode)<=1,"GDN chunk allocation shape");
    const auto raw=gdn_qkv_count(s),v=uint64_t(s.value_heads)*s.value_width,qk=uint64_t(s.value_heads)*128;
    const uint64_t rows=checked_mul(checked_mul(2*raw+2*qk+4*v+4*s.value_heads,n),4);
    const uint64_t recurrence=mode==GdnChunkMode::cs64?checked_mul(uint64_t(s.value_heads)*64+2ull*s.value_heads*64*64+64*v+2*gdn_state_count(s),4):0;
    return {rows,recurrence,checked_add(rows,recurrence)};
}
GdnChunkReference gdn_chunk_reference(std::span<const double> initial,std::span<const double> q,std::span<const double> k,std::span<const double> v,std::span<const double> logg,std::span<const double> beta,uint32_t heads,uint32_t dk,uint32_t dv,uint32_t rows,uint32_t cs){
    need(heads && heads<=48 && dk && dk<=128 && dv && dv<=128 && rows && rows<=512 && cs && cs<=64,"GDN chunk reference geometry");
    const size_t state_size=size_t(heads)*dk*dv,qstride=size_t(heads)*dk,vstride=size_t(heads)*dv;
    need(initial.size()==state_size && q.size()==rows*qstride && k.size()==q.size() && v.size()==rows*vstride && logg.size()==size_t(rows)*heads && beta.size()==logg.size(),"GDN chunk reference extent");
    finite(initial);finite(q);finite(k);finite(v);finite(logg);finite(beta);
    for(size_t i=0;i<logg.size();++i)need(logg[i]<=0 && beta[i]>=0 && beta[i]<=1,"GDN chunk reference gate");
    GdnChunkReference out{std::vector<double>(v.size()),std::vector<double>(initial.begin(),initial.end())};
    for(uint32_t start=0;start<rows;start+=cs){const auto n=std::min(cs,rows-start);
        for(uint32_t h=0;h<heads;++h){
            std::vector<double> g(n),r(size_t(n)*dv),lower(size_t(n)*n),attention(size_t(n)*n);
            double cumulative=0;for(uint32_t i=0;i<n;++i){cumulative+=logg[size_t(start+i)*heads+h];g[i]=cumulative;}
            for(uint32_t i=0;i<n;++i)for(uint32_t j=0;j<=i;++j){double kk=0,qk=0;for(uint32_t a=0;a<dk;++a){const auto kj=k[size_t(start+j)*qstride+h*dk+a];kk+=k[size_t(start+i)*qstride+h*dk+a]*kj;qk+=q[size_t(start+i)*qstride+h*dk+a]*kj;}const auto decay=std::exp(g[i]-g[j]);lower[size_t(i)*n+j]=j<i?beta[size_t(start+i)*heads+h]*kk*decay:0;attention[size_t(i)*n+j]=qk*decay;}
            for(uint32_t i=0;i<n;++i)for(uint32_t b=0;b<dv;++b){double predicted=0;for(uint32_t a=0;a<dk;++a)predicted+=k[size_t(start+i)*qstride+h*dk+a]*out.state[(size_t(h)*dk+a)*dv+b];double x=beta[size_t(start+i)*heads+h]*(v[size_t(start+i)*vstride+h*dv+b]-std::exp(g[i])*predicted);for(uint32_t j=0;j<i;++j)x-=lower[size_t(i)*n+j]*r[size_t(j)*dv+b];r[size_t(i)*dv+b]=x;}
            for(uint32_t i=0;i<n;++i)for(uint32_t b=0;b<dv;++b){double x=0;for(uint32_t a=0;a<dk;++a)x+=q[size_t(start+i)*qstride+h*dk+a]*out.state[(size_t(h)*dk+a)*dv+b];x*=std::exp(g[i]);for(uint32_t j=0;j<=i;++j)x+=attention[size_t(i)*n+j]*r[size_t(j)*dv+b];out.output[size_t(start+i)*vstride+h*dv+b]=x;}
            for(uint32_t a=0;a<dk;++a)for(uint32_t b=0;b<dv;++b){auto& x=out.state[(size_t(h)*dk+a)*dv+b];x*=std::exp(g[n-1]);for(uint32_t j=0;j<n;++j)x+=std::exp(g[n-1]-g[j])*k[size_t(start+j)*qstride+h*dk+a]*r[size_t(j)*dv+b];}
        }
    }
    finite(out.output);finite(out.state);return out;
}
} // namespace tokspur
