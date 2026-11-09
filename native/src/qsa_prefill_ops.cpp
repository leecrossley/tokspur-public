#include "tokspur/qsa_prefill_ops.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <limits>

namespace tokspur {
namespace {
void need(bool okay, const char* why) { if (!okay) throw FormatError(why); }
uint32_t div_up(uint32_t n, uint32_t d) { return (n + d - 1) / d; }
void tail_zero(const QsaPrefillParams& p, uint32_t from) {
    for (uint32_t i=from;i<p.size();++i) need(p[i]==0,"nonzero reserved QSA prefill parameter");
}
void rows(uint32_t n, uint32_t width, uint32_t stride, bool allow_nax_2k = false) {
    const uint32_t max_rows = allow_nax_2k ? 2048u : 512u;
    need(n && n<=max_rows && width && width<=12288 && stride>=width && stride<=1048576,
        "invalid QSA prefill row geometry");
}
uint32_t blocks(const QsaPrefillParams& p) { return (p[1]+p[0])/4; }
void stripe(const QsaPrefillParams& p, bool allow_nax_2k = false) {
    const uint32_t max_rows = allow_nax_2k ? 2048u : 32u;
    need(p[0] && p[0]<=max_rows && p[1]<65536 && p[0]<=65536-p[1],"invalid QSA prefill stripe");
}
uint64_t extent(uint32_t n,uint32_t width,uint32_t stride,uint32_t bytes=4) {
    return (uint64_t(n-1)*stride+width)*bytes;
}
}
const char* qsa_prefill_kernel_name(QsaPrefillOp op) {
    switch(op) {
#define QSA_NAME(OP) case QsaPrefillOp::OP: return "qsa_prefill_" #OP;
    QSA_NAME(bf16_project) QSA_NAME(split_gate) QSA_NAME(rope_rows) QSA_NAME(store_bf16)
    QSA_NAME(index_pool) QSA_NAME(index_scores) QSA_NAME(select_sort) QSA_NAME(select_merge)
    QSA_NAME(select_ids) QSA_NAME(attention_partial128) QSA_NAME(attention_reduce)
    QSA_NAME(prep_query) QSA_NAME(prep_kv) QSA_NAME(prep_index_query) QSA_NAME(prep_index_pool)
    QSA_NAME(verify_index_pool)
#undef QSA_NAME
    }
    throw FormatError("unknown QSA prefill operation");
}
QsaPrefillDispatchSpec qsa_prefill_spec(QsaPrefillOp op,const QsaPrefillParams& p,bool allow_nax_2k) {
    QsaPrefillDispatchSpec out{qsa_prefill_kernel_name(op),0,0,{1,1,1},{128,1,1}};
    switch(op) {
    case QsaPrefillOp::bf16_project:
        rows(p[0],p[1],p[3],allow_nax_2k); rows(p[0],p[2],p[4],allow_nax_2k);
        need(p[1]<=2560 && p[2]<=512 && (p[5]==0 || p[5]==8),"unsupported QSA BF16 matrix"); tail_zero(p,6);
        out.buffers=3; out.writable_mask=4; out.groups={div_up(p[2],32),div_up(p[0],32),1}; break;
    case QsaPrefillOp::split_gate:
        need(p[1] && p[1]<=24 && p[2] && p[2]<=256 && p[3]<=1,"invalid QSA gate geometry");
        rows(p[0],2*p[1]*p[2],p[4],allow_nax_2k); rows(p[0],p[1]*p[2],p[5],allow_nax_2k); rows(p[0],p[1]*p[2],p[6],allow_nax_2k); tail_zero(p,7);
        out.buffers=3;out.writable_mask=6;out.groups={div_up(p[0]*p[1]*p[2],128),1,1};break;
    case QsaPrefillOp::rope_rows:
        need(p[1] && p[1]<=24 && p[2] && p[2]<=256 && p[3] && 2*p[3]<=p[2] && p[8]<=1 &&
            (p[5]==1 || p[5]==4) && p[4]<65536 && p[4]+uint64_t(p[0]-1)*p[5]<65536,
            "invalid QSA rotary geometry");
        rows(p[0],p[1]*p[2],p[6],allow_nax_2k); rows(p[0],p[1]*p[2],p[7],allow_nax_2k);tail_zero(p,9);
        out.buffers=3;out.writable_mask=4;out.groups={div_up(p[0]*p[1]*p[2],128),1,1};break;
    case QsaPrefillOp::store_bf16:
        rows(p[0],p[1],p[2],allow_nax_2k); rows(p[0],p[1],p[3],allow_nax_2k);tail_zero(p,4);
        out.buffers=2;out.writable_mask=2;out.groups={div_up(p[0]*p[1],128),1,1};break;
    case QsaPrefillOp::index_pool:
        rows(p[0],p[1],p[3],allow_nax_2k);need(p[1]<=256 && p[2]<65536 && p[0]<=65536-p[2] && p[6]==4 &&
            p[5]==(p[2]+p[0])/4-p[2]/4 && p[4]>=p[1] && p[4]<=1048576,"invalid QSA pool geometry");tail_zero(p,7);
        out.buffers=4;out.writable_mask=12;out.groups={div_up((p[5]+1)*p[1],128),1,1};break;
    case QsaPrefillOp::index_scores:
        stripe(p, allow_nax_2k);need(p[2] && p[2]<=8 && p[3] && p[3]<=256 && p[4]>=blocks(p) && p[4]<=16384 &&
            p[6] && p[6]<=512 && blocks(p)>p[6],"invalid QSA index score geometry");
        rows(p[0],p[2]*p[3],p[5],allow_nax_2k);tail_zero(p,7);
        out.buffers=3;out.writable_mask=4;out.groups={div_up(blocks(p),4),p[0],1};break;
    case QsaPrefillOp::select_sort:
        stripe(p, allow_nax_2k);need(p[4] && p[4]<=512 && p[2]>=blocks(p) && p[2]<=16384 && blocks(p)>p[4] &&
            p[3]>=div_up(p[2],512)*p[4] && p[3]<=16384,"invalid QSA local selection geometry");tail_zero(p,5);
        out.buffers=3;out.writable_mask=6;out.groups={div_up(blocks(p),512),p[0],1};break;
    case QsaPrefillOp::select_merge:
        stripe(p, allow_nax_2k);need(p[3] && p[3]<=512 && p[2]>=div_up(blocks(p),512)*p[3] && p[2]<=16384 &&
            p[4]>=512 && p[4]<16384 && (p[4]&(p[4]-1))==0 && blocks(p)>p[4],"invalid QSA selection merge geometry");tail_zero(p,5);
        out.buffers=4;out.writable_mask=12;out.groups={div_up(blocks(p),2*p[4]),p[0],1};break;
    case QsaPrefillOp::select_ids:
        stripe(p, allow_nax_2k);need(p[4] && p[4]<=512 && p[2]>=p[4] && p[2]<=16384 && p[3]>=p[4] && p[3]<=16384,
            "invalid QSA selected ID geometry");tail_zero(p,5);
        out.buffers=2;out.writable_mask=2;out.groups={1,p[0],1};break;
    case QsaPrefillOp::attention_partial128: {
        stripe(p);need(p[2] && p[2]<=24 && p[3] && p[2]%p[3]==0 && p[4] && p[4]<=256 && p[5]==4 &&
            p[6] && p[6]<=512 && p[8]>=p[6] && p[8]<=16384 && p[9] && p[9]<=17 &&
            p[9]>=div_up(std::min(p[1]+p[0],p[6]*4+3),128),"invalid QSA partial attention geometry");
        rows(p[0],p[2]*p[4],p[7]);tail_zero(p,10);
        out.buffers=5;out.writable_mask=16;out.groups={p[2],p[0],p[9]};out.threads={32,1,1};break;
    }
    case QsaPrefillOp::attention_reduce:
        stripe(p);need(p[2] && p[2]<=24 && p[3] && p[3]<=256 && p[4]==4 && p[5] && p[5]<=512 &&
            p[6] && p[6]<=17 && p[6]>=div_up(std::min(p[1]+p[0],p[5]*4+3),128),"invalid QSA attention reduction geometry");
        rows(p[0],p[2]*p[3],p[7]);tail_zero(p,8);
        out.buffers=2;out.writable_mask=2;out.groups={p[2],p[0],1};out.threads={32,1,1};break;
    case QsaPrefillOp::prep_query:
        need(p[1] && p[1]<=24 && p[2] && p[2]<=256 && p[3] && 2*p[3]<=p[2] && p[4]<65536 &&
            p[0]<=65536-p[4] && p[5]<=1 && p[8]<=1,"invalid QSA prep_query geometry");
        rows(p[0],2*p[1]*p[2],p[6],allow_nax_2k); rows(p[0],p[1]*p[2],p[7],allow_nax_2k); tail_zero(p,10);
        out.buffers=5;out.writable_mask=24;out.groups={div_up(p[0]*p[1],4),1,1};break;
    case QsaPrefillOp::prep_kv:
        need(p[1] && p[1]<=24 && p[2] && p[2]<=256 && p[3] && 2*p[3]<=p[2] && p[4]<65536 &&
            p[0]<=65536-p[4] && p[8]<=1,"invalid QSA prep_kv geometry");
        rows(p[0],p[1]*p[2],p[5],allow_nax_2k); rows(p[0],p[1]*p[2],p[6],allow_nax_2k); rows(p[0],p[1]*p[2],p[7],allow_nax_2k); tail_zero(p,10);
        out.buffers=6;out.writable_mask=48;out.groups={div_up(p[0]*p[1],4),1,1};break;
    case QsaPrefillOp::prep_index_query:
        need(p[1] && p[1]<=8 && p[2] && p[2]<=256 && p[3] && 2*p[3]<=p[2] && p[4]<65536 &&
            p[0]<=65536-p[4] && p[7]<=1,"invalid QSA prep_index_query geometry");
        rows(p[0],p[1]*p[2],p[5],allow_nax_2k); rows(p[0],p[1]*p[2],p[6],allow_nax_2k); tail_zero(p,9);
        out.buffers=4;out.writable_mask=8;out.groups={div_up(p[0]*p[1],4),1,1};break;
    case QsaPrefillOp::prep_index_pool:
        rows(p[0],p[1],p[3],allow_nax_2k);
        need(p[1]<=256 && p[2]<65536 && p[0]<=65536-p[2] && p[4]>0 && p[4]==(p[2]+p[0])/4-p[2]/4 &&
            p[5] && 2*p[5]<=p[1] && p[6]<=1,"invalid QSA prep_index_pool geometry");
        tail_zero(p,8);
        out.buffers=6;out.writable_mask=48;out.groups={div_up(p[4]+1,4),1,1};break;
    case QsaPrefillOp::verify_index_pool:
        rows(p[0],p[1],p[3],allow_nax_2k);
        need(p[0]>=2 && p[0]<=8 && p[1]<=256 && p[2]<65536 && p[0]<=65536-p[2] && p[6]==4 &&
            p[5]==(p[2]+p[0])/4-p[2]/4 && p[4]>=p[1] && p[4]<=1048576,"invalid QSA verify_index_pool geometry");
        tail_zero(p,7);
        out.buffers=5;out.writable_mask=28;out.groups={div_up(p[1],128),1,1};break;
    }
    return out;
}
void validate_qsa_prefill_dispatch(QsaPrefillOp op,std::span<const GpuView> b,const QsaPrefillParams& p,bool allow_nax_2k) {
    const auto spec=qsa_prefill_spec(op,p,allow_nax_2k);need(b.size()==spec.buffers,"QSA prefill buffer arity mismatch");
    std::array<uint64_t,6> size{};std::array<uint32_t,6> align{4,4,4,4,4,4};
    switch(op) {
    case QsaPrefillOp::bf16_project:size={uint64_t(p[1])*p[2]*2,extent(p[0],p[1],p[3]),extent(p[0],p[2],p[4])};align[0]=2;break;
    case QsaPrefillOp::split_gate:size={extent(p[0],p[1]*p[2]*2,p[4]),extent(p[0],p[1]*p[2],p[5]),extent(p[0],p[1]*p[2],p[6])};break;
    case QsaPrefillOp::rope_rows:size={extent(p[0],p[1]*p[2],p[6]),uint64_t(p[3])*4,extent(p[0],p[1]*p[2],p[7])};break;
    case QsaPrefillOp::store_bf16:size={extent(p[0],p[1],p[2]),extent(p[0],p[1],p[3],2)};align[1]=2;break;
    case QsaPrefillOp::index_pool:size={extent(p[0],p[1],p[3]),uint64_t(p[1])*4,p[5]?extent(p[5],p[1],p[4]):4,uint64_t(p[1])*4};break;
    case QsaPrefillOp::index_scores:size={extent(p[0],p[2]*p[3],p[5]),uint64_t(blocks(p))*p[3]*2,uint64_t(p[0])*p[4]*4};align[1]=2;break;
    case QsaPrefillOp::select_sort:size={uint64_t(p[0])*p[2]*4,uint64_t(p[0])*p[3]*4,uint64_t(p[0])*p[3]*4};break;
    case QsaPrefillOp::select_merge:size={uint64_t(p[0])*p[2]*4,uint64_t(p[0])*p[2]*4,uint64_t(p[0])*p[2]*4,uint64_t(p[0])*p[2]*4};break;
    case QsaPrefillOp::select_ids:size={uint64_t(p[0])*p[2]*4,uint64_t(p[0])*p[3]*4};break;
    case QsaPrefillOp::attention_partial128:
        size={extent(p[0],p[2]*p[4],p[7]),uint64_t(p[1]+p[0])*p[3]*p[4]*2,uint64_t(p[1]+p[0])*p[3]*p[4]*2,
            uint64_t(p[0])*p[8]*4,uint64_t(p[0])*p[2]*p[9]*(p[4]+2)*4};align[1]=align[2]=2;break;
    case QsaPrefillOp::attention_reduce:size={uint64_t(p[0])*p[2]*p[6]*(p[3]+2)*4,extent(p[0],p[2]*p[3],p[7])};break;
    case QsaPrefillOp::prep_query:
        size={extent(p[0],2*p[1]*p[2],p[6]),uint64_t(p[2])*4,uint64_t(p[3])*4,extent(p[0],p[1]*p[2],p[7]),extent(p[0],p[1]*p[2],p[7])};break;
    case QsaPrefillOp::prep_kv:
        size={extent(p[0],p[1]*p[2],p[5]),extent(p[0],p[1]*p[2],p[6]),uint64_t(p[2])*4,uint64_t(p[3])*4,
            extent(p[0],p[1]*p[2],p[7],2),extent(p[0],p[1]*p[2],p[7],2)};align[4]=align[5]=2;break;
    case QsaPrefillOp::prep_index_query:
        size={extent(p[0],p[1]*p[2],p[5]),uint64_t(p[2])*4,uint64_t(p[3])*4,extent(p[0],p[1]*p[2],p[6])};break;
    case QsaPrefillOp::prep_index_pool:
        size={extent(p[0],p[1],p[3]),uint64_t(p[1])*4,uint64_t(p[1])*4,uint64_t(p[5])*4,extent(p[4],p[1],p[1],2),uint64_t(p[1])*4};align[4]=2;break;
    case QsaPrefillOp::verify_index_pool:
        size={extent(p[0],p[1],p[3]),uint64_t(p[1])*4,p[5]?extent(p[5],p[1],p[4]):4,uint64_t(p[0]-1)*p[1]*4,uint64_t(p[1])*4};break;
    }
    for(size_t i=0;i<b.size();++i) {
        need(b[i].owner && b[i].bytes>=size[i] && b[i].offset%align[i]==0 && b[i].offset<=std::numeric_limits<uint64_t>::max()-b[i].bytes,
            "QSA prefill buffer is absent, misaligned or truncated");
        need(b[i].owner==b[0].owner,"QSA prefill buffers have different owners");
        for(size_t j=0;j<i;++j) if((spec.writable_mask&((1u<<i)|(1u<<j))) && b[i].slot==b[j].slot)
            need(b[i].offset>=b[j].offset+b[j].bytes || b[j].offset>=b[i].offset+b[i].bytes,"QSA prefill writable buffers overlap");
    }
}
} // namespace tokspur
