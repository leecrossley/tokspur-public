#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/gdn_chunk.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace tokspur;
void need(bool value, const char* why) { if (!value) throw FormatError(why); }
struct Assets {
    MetalGraph& g;
    std::vector<GpuView> owned;
    ~Assets() { for (auto view : owned) g.release(view); }
    GpuView upload(std::span<const float> values, bool immutable = true) { auto view=g.upload(std::as_bytes(values),"generated HC row fixture",immutable);owned.push_back(view);return view; }
};
struct Guarded {
    GpuView storage, view;
    std::vector<std::byte> initial;
};
Guarded guarded(Assets& a, uint64_t count) {
    std::vector<float> values(count+32,12345.75f);auto storage=a.upload(values,false);
    return {storage,a.g.slice(storage,64,count*4),a.g.read(storage)};
}
void guard(MetalGraph& g, const Guarded& view) {
    const auto now=g.read(view.storage);
    need(std::equal(now.begin(),now.begin()+64,view.initial.begin())&&std::equal(now.end()-64,now.end(),view.initial.end()-64),"HC row output guard changed");
}
std::vector<float> floats(MetalGraph& g, GpuView view) {
    const auto bytes=g.read(view);std::vector<float> out(bytes.size()/4);std::memcpy(out.data(),bytes.data(),bytes.size());return out;
}
double sigmoid(double x) { return x>=0 ? 1/(1+std::exp(-x)) : std::exp(x)/(1+std::exp(x)); }
void reference_check(std::span<const float> actual, std::span<const double> expected) {
    need(actual.size()==expected.size(),"HC row independent reference extent");double error=0,signal=0;
    for(size_t i=0;i<actual.size();++i){need(std::isfinite(actual[i])&&std::isfinite(expected[i]),"HC row nonfinite reference/output");const double d=actual[i]-expected[i];error+=d*d;signal+=expected[i]*expected[i];}
    need(error/std::max(signal,1e-30)<=1e-7,"HC row independent FP64 NMSE gate");
}
template<class F> void refuses(MetalGraph& g,F&& operation) {
    bool rejected=false;try{operation();}catch(const FormatError&){rejected=true;}try{g.abort();}catch(const FormatError&){}need(rejected,"HC row invalid operation accepted");
}
enum class SharedCase { projection, rms, rms_gate, hc_inject, hc_write_rms, router };
std::vector<std::vector<std::byte>> shared_case(MetalGraph& g, SharedCase kind, uint32_t n, uint32_t width, int poison = -1) {
    Assets a{g,{}};
    std::vector<std::vector<float>> inputs;
    std::vector<uint64_t> output_sizes;
    const auto values = [](uint64_t count, uint32_t salt) {
        std::vector<float> result(count);
        for (size_t i = 0; i < result.size(); ++i) result[i] = float(int((i * 29 + salt * 17) % 131) - 65) / 97.0f;
        return result;
    };
    const uint32_t columns = width * 2, stride = columns + 3;
    if (kind == SharedCase::projection) {
        inputs = {values(37 * 9, 1), values(uint64_t(n + 1) * 43, 2)};
        output_sizes = {uint64_t(n + 1) * 15};
    } else if (kind == SharedCase::rms || kind == SharedCase::rms_gate) {
        inputs = {values(uint64_t(n + 1) * stride, 3), values(kind == SharedCase::rms ? columns : width, 4)};
        if (kind == SharedCase::rms_gate) inputs.push_back(values(uint64_t(n + 1) * stride, 5));
        output_sizes = {uint64_t(n + 1) * stride};
    } else if (kind == SharedCase::hc_inject) {
        inputs = {values(4 * 10240, 6), values(uint64_t(n) * 10240, 7)};
        output_sizes = {uint64_t(n) * 4 * 16, uint64_t(n) * 4};
    } else if (kind == SharedCase::hc_write_rms) {
        inputs = {values(uint64_t(n) * 10240, 8), values(uint64_t(n) * 2560, 9), values(uint64_t(n) * 4, 10), values(10240, 11)};
        output_sizes = {uint64_t(n) * 10240, uint64_t(n) * 10240};
    } else {
        inputs = {values(2560 * 512, 12), values(uint64_t(n) * 2560, 13)};
        output_sizes = {uint64_t(n) * 512};
    }
    if (poison >= 0) inputs.at(size_t(poison))[0] = std::numeric_limits<float>::infinity();
    std::vector<GpuView> uploaded;
    for (const auto& input : inputs) uploaded.push_back(a.upload(input));
    std::vector<Guarded> outputs;
    for (const auto count : output_sizes) outputs.push_back(guarded(a, count));
    const auto baseline = g.memory();
    const auto run = [&] {
        g.begin();
        if (kind == SharedCase::projection || kind == SharedCase::rms_gate) {
            GdnChunkDispatch d{};
            d.params.rows = n;
            d.buffers[0] = uploaded[0];
            d.buffers[1] = uploaded[1];
            if (kind == SharedCase::projection) {
                d.op = GdnChunkOp::projection_f32;
                d.params.columns = 37; d.params.outputs = 9;
                d.params.input_stride = 43 * 4; d.params.output_stride = 15 * 4;
                d.buffers[2] = outputs[0].view;
            } else {
                d.op = GdnChunkOp::rms_gate;
                d.params.heads = 2; d.params.key_width = 128; d.params.value_width = width;
                d.params.input_stride = d.params.auxiliary_stride = d.params.output_stride = uint64_t(stride) * 4;
                d.params.rms_epsilon = 1e-6f;
                d.buffers[2] = uploaded[2]; d.buffers[3] = outputs[0].view;
            }
            g.gdn_chunk(d);
        } else if (kind == SharedCase::rms) {
            g.rms_rows({uploaded[0], n, columns, uint64_t(stride) * 4}, uploaded[1],
                {outputs[0].view, n, columns, uint64_t(stride) * 4}, width, 1e-6f, false);
        } else if (kind == SharedCase::hc_inject) {
            g.hc_inject_rows(uploaded[0], uploaded[1], outputs[0].view, outputs[1].view, n);
        } else if (kind == SharedCase::hc_write_rms) {
            g.hc_write_rms_rows(uploaded[0], uploaded[1], uploaded[2], uploaded[3], outputs[0].view, outputs[1].view, n, 1e-6f);
        } else {
            g.moe_router_f32_window(uploaded[0], {uploaded[1], n, 2560, 2560 * 4}, {outputs[0].view, n, 512, 512 * 4});
        }
        const auto batch = g.submit();
        need(batch.dispatches == (kind == SharedCase::hc_inject ? 2u : 1u), "shared kernel dispatch count changed");
    };
    if (poison >= 0) refuses(g, run); else run();
    need(g.memory().allocations == baseline.allocations && g.memory().live_bytes == baseline.live_bytes, "shared kernel encode allocated storage");
    std::vector<std::vector<std::byte>> result;
    for (const auto& output : outputs) { guard(g, output); if (poison < 0) result.push_back(g.read(output.view)); }
    for (size_t i = 0; i < inputs.size(); ++i) need(g.read(uploaded[i]) == std::vector<std::byte>(std::as_bytes(std::span(inputs[i])).begin(), std::as_bytes(std::span(inputs[i])).end()), "shared kernel mutated input");
    return result;
}
void shared_kernels(const char* library) {
    MetalGraph baseline(library, 128ull << 20), candidate(library, 128ull << 20);
    candidate.configure_prefill_dense_f32(true); candidate.configure_prefill_dense_nax(true);
    for (auto* graph : {&baseline, &candidate}) {
        graph->configure_small_gemv(16); graph->configure_block_gemv(3);
        graph->prepare_hc_rows(); graph->prepare_gdn_chunk();
    }
    const auto check = [&](SharedCase kind, uint32_t n, uint32_t width, int input_count) {
        const auto expected = shared_case(baseline, kind, n, width);
        need(expected == shared_case(candidate, kind, n, width), "NAX shared kernel differs from preserved main output bytes");
        for (int input = 0; input < input_count; ++input) {
            static_cast<void>(shared_case(baseline, kind, n, width, input));
            static_cast<void>(shared_case(candidate, kind, n, width, input));
        }
        need(expected == shared_case(baseline, kind, n, width) && expected == shared_case(candidate, kind, n, width), "shared kernels failed recovery after nonfinite input");
    };
    for (const uint32_t n : {1u, 4u, 5u, 8u, 9u, 17u}) check(SharedCase::projection, n, 0, 2);
    for (const uint32_t width : {3u, 128u, 256u, 512u, 2560u}) {
        check(SharedCase::rms, 3, width, 2);
        check(SharedCase::rms, 193, width, 0);
    }
    check(SharedCase::rms_gate, 3, 7, 3); check(SharedCase::rms_gate, 3, 128, 3);
    for (const uint32_t n : {2u, 3u, 17u}) {
        check(SharedCase::hc_inject, n, 0, 2); check(SharedCase::hc_write_rms, n, 0, 4);
    }
    MetalGraph router_default(library, 128ull << 20), router_nax(library, 128ull << 20);
    for (auto* graph : {&router_default, &router_nax}) {
        graph->configure_prefill_router_window(true); graph->prepare_moe_router_window();
    }
    router_nax.configure_verify_nax(true); router_nax.prepare_verify_nax({});
    for (const uint32_t n : {2u, 3u, 8u}) {
        const auto expected = shared_case(router_default, SharedCase::router, n, 0);
        need(expected == shared_case(router_nax, SharedCase::router, n, 0), "inactive NAX scope changed default router");
        router_nax.set_verify_nax_scope(true);
        need(expected == shared_case(router_nax, SharedCase::router, n, 0), "NAX small-row router differs from main bytes");
        for (const int input : {0, 1}) {
            static_cast<void>(shared_case(router_default, SharedCase::router, n, 0, input));
            static_cast<void>(shared_case(router_nax, SharedCase::router, n, 0, input));
        }
        need(expected == shared_case(router_nax, SharedCase::router, n, 0), "NAX small-row router failed recovery");
        router_nax.set_verify_nax_scope(false);
    }
    std::cout << "shared default/NAX kernels: exact output bytes, input guards, nonfinite refusal and recovery pass; no model or speed qualification\n";
}
void test_case(MetalGraph& g,uint32_t n) {
    constexpr uint32_t width=10240,channels=2560,branches=4;Assets a{g,{}};
    std::vector<float> weight(width*branches),normalized(uint64_t(n)*width),scores(normalized.size()),streams(normalized.size()),values(uint64_t(n)*channels);
    for(size_t i=0;i<weight.size();++i)weight[i]=float(int((i*13)%37)-18)/1021;
    for(size_t i=0;i<normalized.size();++i){normalized[i]=float(int((i*29)%131)-65)/997;scores[i]=float(int((i*17)%127)-63)/32;streams[i]=float(int((i*7)%107)-53)/809;}
    for(size_t i=0;i<values.size();++i)values[i]=float(int((i*19)%113)-56)/991;
    const auto w=a.upload(weight,false),x=a.upload(normalized),s=a.upload(scores),r=a.upload(streams),v=a.upload(values);
    const std::array inputs{w,x,s,r,v};std::array<std::vector<std::byte>,5> originals;for(size_t i=0;i<inputs.size();++i)originals[i]=g.read(inputs[i]);
    auto partial=guarded(a,uint64_t(n)*branches*16);
    std::array<Guarded,4> old{guarded(a,n*branches),guarded(a,n*channels),guarded(a,n*branches),guarded(a,uint64_t(n)*width)};
    std::array<Guarded,4> batch{guarded(a,n*branches),guarded(a,n*channels),guarded(a,n*branches),guarded(a,uint64_t(n)*width)};
    const auto baseline_allocations=g.memory().allocations;
    const auto row=[&](GpuView view,uint32_t token,uint32_t columns){return g.slice(view,uint64_t(token)*columns*4,uint64_t(columns)*4);};
    const auto run=[&](bool selected){
        g.begin();
        if(selected){g.hc_inject_rows(w,x,partial.view,batch[0].view,n);g.hc_read_rows(x,s,batch[0].view,batch[1].view,batch[2].view,n);g.hc_write_rows(r,v,batch[2].view,batch[3].view,n);}
        else for(uint32_t t=0;t<n;++t){g.gemv(0,w,row(x,t,width),row(old[0].view,t,branches),width,branches);g.residual_read(row(x,t,width),row(s,t,width),row(old[0].view,t,branches),row(old[1].view,t,channels),row(old[2].view,t,branches),branches,channels);g.residual_write(row(r,t,width),row(v,t,channels),row(old[2].view,t,branches),row(old[3].view,t,width));}
        const auto result=g.submit();need(result.dispatches==(selected?4u:n*4u),"HC row dispatch count differs");need(g.memory().allocations==baseline_allocations,"HC row encode allocated graph storage");return result;
    };
    static_cast<void>(run(false));static_cast<void>(run(true));
    for(size_t i=0;i<old.size();++i)need(g.read(old[i].view)==g.read(batch[i].view),"HC batching changed exact represented FP32 output bytes");
    std::vector<double> inject(uint64_t(n)*branches),read(uint64_t(n)*channels),scales(inject.size()),write(uint64_t(n)*width);
    for(uint32_t t=0;t<n;++t){
        for(uint32_t b=0;b<branches;++b){double sum=0;for(uint32_t c=0;c<width;++c)sum+=double(weight[uint64_t(b)*width+c])*normalized[uint64_t(t)*width+c];inject[t*branches+b]=sum;scales[t*branches+b]=2*sigmoid(sum/branches);}
        for(uint32_t c=0;c<channels;++c){double sum=0;for(uint32_t b=0;b<branches;++b){const auto at=(uint64_t(t)*branches+b)*channels+c;sum+=double(normalized[at])*sigmoid(scores[at]);}read[uint64_t(t)*channels+c]=sum/branches;}
        for(uint32_t b=0;b<branches;++b)for(uint32_t c=0;c<channels;++c){const auto at=(uint64_t(t)*branches+b)*channels+c;write[at]=double(streams[at])+scales[t*branches+b]*values[uint64_t(t)*channels+c];}
    }
    for(size_t i=0;i<4;++i){const std::array<std::span<const double>,4> expected{inject,read,scales,write};reference_check(floats(g,batch[i].view),expected[i]);}
    if(n==512){
        for(uint32_t pair=0;pair<8;++pair)for(uint32_t order=0;order<2;++order){const bool selected=(pair+order)%2;const auto z=run(selected);std::cout<<"{\"event\":\"hc_rows_trial\",\"tokens\":"<<n<<",\"arm\":\""<<(selected?"batched-four":"legacy-per-token")<<"\",\"phase\":\""<<(pair<2?"warmup":"measured")<<"\",\"pair\":"<<pair<<",\"order\":"<<order<<",\"dispatches\":"<<z.dispatches<<",\"batch_gpu_ms\":"<<z.gpu_ms<<",\"batch_wall_ms\":"<<z.wall_ms<<"}\n";}
    }
    for(const auto& view:old)guard(g,view);for(const auto& view:batch)guard(g,view);guard(g,partial);
    for(size_t i=0;i<inputs.size();++i)need(g.read(inputs[i])==originals[i],"HC row input or represented weight bytes changed");
    refuses(g,[&]{g.begin();g.hc_inject_rows(w,x,partial.view,batch[0].view,1);});
    refuses(g,[&]{g.begin();g.hc_inject_rows(w,x,partial.view,batch[0].view,513);});
    refuses(g,[&]{g.begin();g.hc_read_rows(x,s,batch[0].view,x,batch[2].view,n);});
    refuses(g,[&]{g.begin();g.hc_write_rows(r,v,batch[2].view,r,n);});
    auto invalid=weight;invalid[17]=std::numeric_limits<float>::infinity();g.write(w,std::as_bytes(std::span(invalid)));
    refuses(g,[&]{static_cast<void>(run(false));});refuses(g,[&]{static_cast<void>(run(true));});g.write(w,std::as_bytes(std::span(weight)));
    static_cast<void>(run(false));static_cast<void>(run(true));
    for(size_t i=0;i<old.size();++i){need(g.read(old[i].view)==g.read(batch[i].view),"HC rows did not recover after refused nonfinite work");guard(g,old[i]);guard(g,batch[i]);}guard(g,partial);
    std::cout<<"{\"event\":\"hc_rows_case\",\"tokens\":"<<n<<",\"byte_exact\":true,\"fp64_nmse_limit\":1e-7,\"guards_and_inputs_unchanged\":true,\"nonfinite_refused_both\":true,\"no_encode_allocations\":true,\"graph_peak_bytes\":"<<g.memory().peak_bytes<<",\"performance_qualified\":false}\n";
}
}
int main(int argc,char** argv){try{
    need(argc==2||(argc==3&&(std::string(argv[2])=="serial"||std::string(argv[2])=="shared-kernels")),"usage: test-hc_rows OWNED_METALLIB [serial|shared-kernels]");std::cout<<std::setprecision(17);
    if(argc==3&&std::string(argv[2])=="shared-kernels"){shared_kernels(argv[1]);return 0;}
    MetalGraph g(argv[1],512ull<<20);g.configure_small_gemv(16);g.configure_block_gemv(3);g.configure_concurrent_dispatch(argc==2);g.prepare_hc_rows();
    std::cout<<"{\"event\":\"hc_rows_settings\",\"concurrent\":"<<(argc==2?"true":"false")<<",\"split_columns\":16,\"branches\":4,\"channels\":2560,\"warmup_pairs\":2,\"measured_pairs\":6,\"scope\":\"generated HC injection and residual read/write only, no down/up, normalisation, target state or serving\",\"performance_qualified\":false}\n";
    const auto initial=g.memory().live_bytes;for(uint32_t n:{2u,3u,17u,32u,511u,512u}){test_case(g,n);need(g.memory().live_bytes==initial,"HC row fixture leaked graph storage");}
    if(argc==2)shared_kernels(argv[1]);
    std::cout<<"{\"event\":\"hc_rows_complete\",\"passed\":true,\"model_loaded\":false,\"performance_qualified\":false}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
