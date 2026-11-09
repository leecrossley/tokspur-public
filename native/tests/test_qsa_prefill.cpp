#include "tokspur/qsa_prefill.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using namespace tokspur;
uint64_t checks=0;
void need(bool b,const char* why) {++checks;if(!b)throw std::runtime_error(why);}
template<class T> std::vector<T> read(MetalGraph& g,GpuView v) {
    const auto bytes=g.read(v);std::vector<T> out(bytes.size()/sizeof(T));std::memcpy(out.data(),bytes.data(),bytes.size());return out;
}
struct Fixture {
    MetalGraph& g;std::mt19937 random{443};std::vector<GpuView> owned;
    explicit Fixture(MetalGraph& graph):g(graph) {}
    ~Fixture(){try{g.abort();}catch(const FormatError&){}for(auto v:owned)g.release(v);}
    std::vector<float> data(size_t n) {std::vector<float> v(n);for(auto& x:v)x=float(int(random()%31)-15)*0.03125f;return v;}
    GpuView floats(std::span<const float> x,bool immutable=true) {auto v=g.upload(std::as_bytes(x),"generated QSA data",immutable);owned.push_back(v);return v;}
    GpuView zero(uint64_t n) {auto v=g.zeros(n,"generated QSA output");owned.push_back(v);return v;}
    void fill(GpuView v) {const auto x=data(v.bytes/4);g.write(v,std::as_bytes(std::span(x)));}
};
void scalar_projected(MetalGraph& g,const QsaWeights& w,const QsaScratch& s,const QsaCache& c,GpuView sum,GpuView next,
    QsaPrefillProjections raw,uint32_t row,GpuView out,uint32_t position,QsaShape shape,QsaPolicy p) {
    const uint32_t q=shape.query_heads*shape.head_width,k=shape.kv_heads*shape.head_width,iq=shape.index_heads*shape.index_width;
    const auto input=[&](GpuRowsF32 source){return g.slice(source.view,uint64_t(row)*source.stride_bytes,source.columns*4ull);};
    g.split_query_gate(input(raw.packed),s.query,s.gate,shape.query_heads,shape.head_width,p.packing);
    g.rms(s.query,w.query_gains,s.normalized_query,shape.head_width,p.rms_epsilon);g.rms(input(raw.key),w.key_gains,s.normalized_key,shape.head_width,p.rms_epsilon);
    g.rms(input(raw.index_query),w.index_query_gains,s.normalized_index_query,shape.index_width,p.rms_epsilon);
    g.rotary(s.normalized_query,w.frequencies,s.rotary_query,shape.query_heads,shape.head_width,position,p.rotary_pairs);
    g.rotary(s.normalized_key,w.frequencies,s.rotary_key,shape.kv_heads,shape.head_width,position,p.rotary_pairs);
    g.rotary(s.normalized_index_query,w.frequencies,s.rotary_index_query,shape.index_heads,shape.index_width,position,p.rotary_pairs);
    g.store_bf16(s.rotary_key,g.slice(c.keys,uint64_t(position)*k*2,k*2));g.store_bf16(input(raw.value),g.slice(c.values,uint64_t(position)*k*2,k*2));
    g.index_accumulate(sum,input(raw.index_key),next,s.pooled,position,4);const uint32_t blocks=(position+1)/4;
    if((position+1)%4==0){g.rms(s.pooled,w.index_key_gains,s.normalized_index_key,shape.index_width,p.rms_epsilon);
        g.rotary(s.normalized_index_key,w.frequencies,s.rotary_index_key,1,shape.index_width,position+1-4,p.rotary_pairs);
        g.store_bf16(s.rotary_index_key,g.slice(c.index_keys,uint64_t(blocks-1)*shape.index_width*2,shape.index_width*2));}
    if(blocks>shape.token_budget/4){g.index_scores(s.rotary_index_query,c.index_keys,s.scores,shape.index_heads,shape.index_width,blocks);
        g.select_blocks(s.scores,s.selected_ids,s.selection,blocks,shape.token_budget/4);}
    g.sparse_attention(s.rotary_query,c.keys,c.values,s.selected_ids,s.attention,position+1,shape);g.multiply(s.attention,s.gate,out,true);
    static_cast<void>(q);static_cast<void>(iq);
}
void connected(MetalGraph& g,QsaShape shape,uint32_t base,uint32_t n,QsaPolicy policy,bool ties) {
    std::cerr<<"QSA case base="<<base<<" rows="<<n<<" heads="<<shape.query_heads<<" kv_heads="<<shape.kv_heads<<" width="<<shape.head_width<<" index_heads="<<shape.index_heads<<" index_width="<<shape.index_width<<" budget="<<shape.token_budget<<" packing="<<uint32_t(policy.packing)<<" rotary="<<uint32_t(policy.rotary_pairs)<<" ties="<<ties<<'\n';
    Fixture f{g};const uint32_t q=shape.query_heads*shape.head_width,iw=shape.index_width,k=shape.kv_heads*shape.head_width;
    const auto plan=qsa_prefill_plan(base,n,n,shape);
    std::vector<float> serial_norm_key(uint64_t(n)*k),serial_rope_key(serial_norm_key.size()),serial_pool(uint64_t(plan.pool_count)*iw),serial_norm_pool(serial_pool.size()),serial_rope_pool(serial_pool.size());
    const auto scratch=allocate_qsa_prefill_scratch(g,shape,n);const auto single=allocate_qsa_scratch(g,shape);
    const auto c=allocate_qsa_cache(g,shape),serial=allocate_qsa_cache(g,shape);
    const auto accepted=f.floats(base%4?f.data(iw):std::vector<float>(iw),false),candidate=f.zero(iw*4),a=f.zero(iw*4),b=f.zero(iw*4);
    const auto raw=qsa_prefill_projection_rows(scratch,n);
    for(auto v:{raw.packed.view,raw.key.view,raw.value.view,raw.index_query.view,raw.index_key.view})f.fill(v);
    if(ties){const std::vector<float> zero(raw.index_query.view.bytes/4);g.write(raw.index_query.view,std::as_bytes(std::span(zero)));}
    QsaWeights w{};w.query_gains=f.floats(std::vector<float>(shape.head_width,1));w.key_gains=w.query_gains;
    w.index_query_gains=f.floats(std::vector<float>(iw,1));w.index_key_gains=w.index_query_gains;
    w.frequencies=f.floats(rotary_frequencies(shape.rotary_width,10000));
    for(size_t cache_index=0;cache_index<3;++cache_index){const auto v=c.views()[cache_index];std::vector<uint16_t> x(v.bytes/2);
        for(auto& e:x)e=finite_bf16(float(int(f.random()%31)-15)*0.03125f);
        const uint64_t future=cache_index==2?uint64_t((base+n)/4)*iw:uint64_t(base+n)*shape.kv_heads*shape.head_width;
        std::fill(x.begin()+future,x.end(),uint16_t(0x7f80));g.write(v,std::as_bytes(std::span(x)));}
    const auto oldk=g.read(c.keys),oldv=g.read(c.values),oldi=g.read(c.index_keys),oldsum=g.read(accepted);
    g.write(serial.keys,oldk);g.write(serial.values,oldv);g.write(serial.index_keys,oldi);g.write(a,oldsum);
    const auto reference=f.zero(uint64_t(n)*q*4);GpuView current=a,next=b;
    for(uint32_t row=0;row<n;++row){g.begin();scalar_projected(g,w,single,serial,current,next,raw,row,g.slice(reference,uint64_t(row)*q*4,q*4),base+row,shape,policy);
        static_cast<void>(g.submit());std::swap(current,next);
        const auto nk=read<float>(g,single.normalized_key),rk=read<float>(g,single.rotary_key);
        std::copy(nk.begin(),nk.end(),serial_norm_key.begin()+uint64_t(row)*k);std::copy(rk.begin(),rk.end(),serial_rope_key.begin()+uint64_t(row)*k);
        if((base+row+1)%4==0){const auto pool=(base+row+1)/4-1-plan.first_pool;
            for(const auto& [source,dest]:std::array{std::pair{single.pooled,&serial_pool},std::pair{single.normalized_index_key,&serial_norm_pool},std::pair{single.rotary_index_key,&serial_rope_pool}}){
                const auto values=read<float>(g,source);std::copy(values.begin(),values.end(),dest->begin()+uint64_t(pool)*iw);}}
    }
    const auto allocations=g.memory().allocations;
    g.begin();const auto output=encode_qsa_prefill_projected(g,w,scratch,c,accepted,candidate,base,n,shape,policy);const auto batch=g.submit();
    need(batch.dispatches==qsa_prefill_projected_dispatches(base,n,shape),"QSA projected dispatch count mismatch");
    need(g.memory().allocations==allocations,"QSA allocated inside batch");
    need(g.read(accepted)==oldsum,"QSA changed accepted partial sum");
    const auto stage_difference=[&](const char* name,GpuView actual,std::span<const float> expected,uint32_t row,uint32_t width,uint32_t focus){
        const auto got=read<float>(g,g.slice(actual,uint64_t(row)*width*4,uint64_t(width)*4));
        std::cerr<<"stage_focus="<<name<<" row="<<row<<" channel="<<focus<<" expected="<<expected[focus]<<" actual="<<got[focus]<<" expected_bits="<<std::hex<<std::bit_cast<uint32_t>(expected[focus])<<" actual_bits="<<std::bit_cast<uint32_t>(got[focus])<<std::dec<<'\n';
        for(uint32_t channel=0;channel<width;++channel)if(std::bit_cast<uint32_t>(got[channel])!=std::bit_cast<uint32_t>(expected[channel])){
            std::cerr<<"stage="<<name<<" row="<<row<<" channel="<<channel<<" expected="<<expected[channel]<<" actual="<<got[channel]<<" expected_bits="<<std::hex<<std::bit_cast<uint32_t>(expected[channel])<<" actual_bits="<<std::bit_cast<uint32_t>(got[channel])<<std::dec<<'\n';return;}
        std::cerr<<"stage="<<name<<" row="<<row<<" exact\n";
    };
    for(size_t cache_index=0;cache_index<3;++cache_index){const auto actual=read<uint16_t>(g,c.views()[cache_index]),expected=read<uint16_t>(g,serial.views()[cache_index]);
        if(actual!=expected){const auto at=size_t(std::mismatch(actual.begin(),actual.end(),expected.begin()).first-actual.begin());
            const uint32_t width=cache_index==2?iw:k,global_row=uint32_t(at/width),channel=uint32_t(at%width);
            std::cerr<<"cache="<<(cache_index==0?"key":cache_index==1?"value":"pooled_index")<<" element="<<at<<" global_row="<<global_row<<" channel="<<channel<<" expected_bf16="<<std::hex<<expected[at]<<" actual_bf16="<<actual[at]<<std::dec<<" expected_value="<<read_bf16(expected[at])<<" actual_value="<<read_bf16(actual[at])<<'\n';
            if(cache_index==0 && global_row>=base && global_row<base+n){const auto row=global_row-base;
                stage_difference("normalized_key",scratch.normalized_key,std::span(serial_norm_key).subspan(uint64_t(row)*k,k),row,k,channel);
                stage_difference("rotary_key",scratch.rotary_key,std::span(serial_rope_key).subspan(uint64_t(row)*k,k),row,k,channel);}
            if(cache_index==2 && global_row>=plan.first_pool && global_row<plan.first_pool+plan.pool_count){const auto row=global_row-plan.first_pool;
                stage_difference("raw_pool",scratch.pools,std::span(serial_pool).subspan(uint64_t(row)*iw,iw),row,iw,channel);
                stage_difference("normalized_pool",scratch.normalized_pools,std::span(serial_norm_pool).subspan(uint64_t(row)*iw,iw),row,iw,channel);
                stage_difference("rotary_pool",scratch.rotary_pools,std::span(serial_rope_pool).subspan(uint64_t(row)*iw,iw),row,iw,channel);}
            need(false,"same raw projections changed BF16 cache bytes");}}

    need(g.read(candidate)==g.read(current),"same raw projections changed chronological FP32 partial sum");
    const auto expected=read<float>(g,reference),actual=read<float>(g,g.slice(output.view,0,uint64_t(n)*q*4));
    for(size_t i=0;i<actual.size();++i)need(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<=3e-5f*(1+std::abs(expected[i])),"tiled attention differs from serial generated control");
    const auto outside=[&](GpuView v,const std::vector<std::byte>& before,uint64_t start,uint64_t size){const auto now=g.read(v);
        need(std::equal(before.begin(),before.begin()+start,now.begin()),"QSA modified published cache prefix");
        need(std::equal(before.begin()+start+size,before.end(),now.begin()+start+size),"QSA wrote outside declared append span");};
    outside(c.keys,oldk,plan.key_append_offset,plan.key_append_bytes);outside(c.values,oldv,plan.key_append_offset,plan.key_append_bytes);
    outside(c.index_keys,oldi,plan.index_append_offset,plan.index_append_bytes);
    const auto completed=g.read(c.keys);g.begin();static_cast<void>(encode_qsa_prefill_projected(g,w,scratch,c,accepted,candidate,base,n,shape,policy));g.abort();
    need(g.read(c.keys)==completed,"aborted unsubmitted QSA executed writes");
    // An owning caller can restore all physical tail bytes after drain. Logical clipping is not asserted as rollback.
    g.write(c.keys,oldk);g.write(c.values,oldv);g.write(c.index_keys,oldi);need(g.read(c.keys)==oldk&&g.read(c.values)==oldv&&g.read(c.index_keys)==oldi,"full physical tail restoration failed");
    release_qsa_cache(g,c);release_qsa_cache(g,serial);release_qsa_scratch(g,single);release_qsa_prefill_scratch(g,scratch);
}
void indexers(MetalGraph& g) {
    Fixture f{g};const QsaShape shape{37,4,2,8,2,8,4,4,16,4096};const uint32_t n=33;
    auto scratch=allocate_qsa_prefill_scratch(g,shape,n);QsaWeights w{};
    const auto matrix=[&](uint32_t rows){std::vector<uint16_t> data(uint64_t(rows)*shape.channels);for(auto& x:data)x=finite_bf16(float(int(f.random()%31)-15)*0.03125f);
        auto v=g.upload(std::as_bytes(std::span(data)),"generated original BF16",true);f.owned.push_back(v);return GpuMatrix{v,30};};
    w.index_query=matrix(16);w.index_key=matrix(8);const auto x=f.data(n*43);const auto input=f.floats(x),reference=f.zero(n*24*4);
    g.begin();encode_qsa_prefill_indexers(g,w,scratch,{input,n,37,43*4},shape);need(g.submit().dispatches==2,"QSA indexer dispatch count mismatch");
    for(uint32_t row=0;row<n;++row){g.begin();g.gemv(30,w.index_query.view,g.slice(input,uint64_t(row)*43*4,37*4),g.slice(reference,uint64_t(row)*24*4,16*4),37,16);
        g.gemv(30,w.index_key.view,g.slice(input,uint64_t(row)*43*4,37*4),g.slice(reference,(uint64_t(row)*24+16)*4,8*4),37,8);static_cast<void>(g.submit());}
    const auto gotq=read<float>(g,scratch.projected.index_query.view),gotk=read<float>(g,scratch.projected.index_key.view),expected=read<float>(g,reference);
    for(uint32_t r=0;r<n;++r){for(uint32_t c=0;c<16;++c)need(std::abs(gotq[r*16+c]-expected[r*24+c])<=2e-5f*(1+std::abs(expected[r*24+c])),"BF16 query projector generated control");
        for(uint32_t c=0;c<8;++c)need(std::abs(gotk[r*8+c]-expected[r*24+16+c])<=2e-5f*(1+std::abs(expected[r*24+16+c])),"BF16 key projector generated control");}
    release_qsa_prefill_scratch(g,scratch);
}
void invalid_ids(MetalGraph& g) {
    Fixture f{g};const auto query=f.floats(std::vector<float>(8)),keys=f.zero(16*8*2),values=f.zero(16*8*2),partials=f.zero(10*4);
    for(const auto bad: {std::array<uint32_t,2>{0,0},std::array<uint32_t,2>{1,0},std::array<uint32_t,2>{0,4}}) {
        const auto ids=g.upload(std::as_bytes(std::span(bad)),"bad selected ids",true);f.owned.push_back(ids);g.begin();
        const std::array buffers{query,keys,values,ids,partials};g.qsa_prefill_dispatch(QsaPrefillOp::attention_partial128,buffers,{1,15,1,1,8,4,2,8,2,1});
        bool rejected=false;try{static_cast<void>(g.submit());}catch(const FormatError&){rejected=true;}need(rejected,"duplicate/permuted/future selected ids admitted");
    }
}
}
int main(int argc,char** argv){try{if(argc!=2)throw std::runtime_error("usage: test-qsa_prefill metallib");MetalGraph g(argv[1],512ull<<20);g.prepare_qsa_prefill();g.prepare_gdn_chunk();
    indexers(g);g.configure_prefill_dense_f32(true);g.configure_prefill_dense_nax(true);g.prepare_qsa_prefill();indexers(g);
    g.configure_prefill_dense_nax(false);g.configure_prefill_dense_f32(false);invalid_ids(g);
    const QsaShape small{16,4,2,8,2,8,4,4,16,8192};
    for(auto packing:{QueryPacking::head_query_gate,QueryPacking::all_queries_then_gates})for(auto rotary:{RotaryPairs::split_half,RotaryPairs::adjacent})
        for(uint32_t base:{0u,1u,2u,3u,15u,16u,17u,18u,19u,20u,2047u,4095u})connected(g,small,base,33,{packing,rotary,1e-6f},true);
    auto wide=small;wide.token_budget=2048;
    for(uint32_t base:{2047u,2048u,2049u,2050u,2051u,4095u})connected(g,wide,base,8,{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},false);
    connected(g,wide,2047,512,{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},true);
    connected(g,{16,24,2,256,4,128,64,4,2048,4096},2047,8,{QueryPacking::head_query_gate,RotaryPairs::split_half,1e-6f},false);
    std::cout<<"Generated QSA operator checks passed: "<<checks<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
