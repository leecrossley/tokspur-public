// SPDX-License-Identifier: Apache-2.0
// Byte admission against the unchanged generic CS64 path, compiled separately.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "tokspur/gdn_chunk.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

void need(bool b, const char* why) { if (!b) throw std::runtime_error(why); }
struct Buffer {
    id<MTLBuffer> obj;
    size_t count;
    Buffer(id<MTLDevice> d, size_t n) : obj([d newBufferWithLength:n*4+32 options:MTLResourceStorageModeShared]), count(n) {
        need(obj != nil, "buffer allocation"); std::memset(obj.contents, 0x5a, n*4+32);
    }
    float* data() const { return reinterpret_cast<float*>(static_cast<char*>(obj.contents)+16); }
    void reset() { std::memset(data(), 0x5a, count*4); }
    void check() const {
        auto p = static_cast<unsigned char*>(obj.contents);
        for (size_t i=0; i<16; ++i) need(p[i]==0x5a && p[count*4+16+i]==0x5a, "buffer canary");
    }
    bool equal(const Buffer& b) const { return count==b.count && std::memcmp(data(),b.data(),count*4)==0; }
};
struct Runner {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::array<id<MTLLibrary>,2> libs;
    std::map<std::string,id<MTLComputePipelineState>> pipelines;
    Buffer error{device,1};
    bool validate;
    Runner(char** paths, bool v) : validate(v) {
        need(device != nil && queue != nil, "Metal unavailable");
        std::cout << "device=" << device.name.UTF8String << " validation=" << validate << '\n';
        for(size_t i=0;i<2;++i){NSError* e=nil;libs[i]=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:paths[i]]] error:&e];
            if(!libs[i]) std::cerr << e.localizedDescription.UTF8String << '\n';need(libs[i]!=nil,"library load");}
    }
    id<MTLComputePipelineState> pipeline(const std::string& name, unsigned arm) {
        const auto key = std::to_string(arm) + name;
        if(pipelines.contains(key))return pipelines.at(key);
        auto f=[libs[arm] newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        need(f!=nil,"kernel missing");
        auto desc=[MTLComputePipelineDescriptor new];desc.computeFunction=f;
        if(validate)desc.shaderValidation=MTLShaderValidationEnabled;
        NSError* e=nil;auto p=[device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&e];
        if(!p)std::cerr<<e.localizedDescription.UTF8String<<'\n';need(p!=nil,"pipeline creation");
        need(p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=128,"CS64 pipeline thread geometry");
        need(arm ? p.staticThreadgroupMemoryLength>=512 : p.staticThreadgroupMemoryLength==0,"reference/reuse library selection");
        if(validate)need(p.shaderValidation==MTLShaderValidationEnabled,"shader validation unavailable");
        else need(p.shaderValidation!=MTLShaderValidationEnabled,"timing requires shader validation off");
        std::cout<<"pipeline arm="<<arm<<" static_bytes="<<p.staticThreadgroupMemoryLength<<" shader_validation="<<p.shaderValidation<<'\n';
        pipelines[key]=p;return p;
    }
    template<class Encode> double run(unsigned arm,size_t groups,unsigned repeats,Encode encode){
        if(!validate)need(NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateNominal,"timing thermal guard");
        auto p=pipeline("gdn_chunk_emit64",arm);*reinterpret_cast<uint32_t*>(error.data())=0;
        auto cb=[queue commandBuffer];auto ce=[cb computeCommandEncoder];need(cb!=nil&&ce!=nil,"command creation");
        [ce setComputePipelineState:p];[ce setBuffer:error.obj offset:16 atIndex:7];encode(ce);
        for(unsigned n=0;n<repeats;++n)[ce dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [ce endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)std::cerr<<cb.error.localizedDescription.UTF8String<<'\n';
        need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");error.check();
        if(!validate)need(NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateNominal,"timing thermal guard");
        return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
    }
    uint32_t flags()const{return *reinterpret_cast<uint32_t*>(error.data());}
};
void fill(Buffer& b, unsigned seed, float scale=.1f){std::mt19937 rng(seed);for(size_t i=0;i<b.count;++i)b.data()[i]=float(int(rng()%2001)-1000)*scale/1000.0f;}
template<class F> void timing(const std::string& label,F run){
    for(unsigned arm:{0,1,1,0})run(arm,8);
    std::vector<double> a,b,ratios;
    for(unsigned round=0;round<12;++round){std::array<double,2> s{};for(unsigned arm:{0,1,1,0})s[arm]+=run(arm,32)/2;
        a.push_back(s[0]);b.push_back(s[1]);ratios.push_back(s[0]/s[1]);
        std::cout<<"PAIR "<<label<<" round="<<round<<" control_us="<<s[0]<<" reuse_us="<<s[1]<<" speedup="<<s[0]/s[1]<<'\n';}
    auto median=[](std::vector<double>x){std::sort(x.begin(),x.end());const auto n=x.size();return n%2?x[n/2]:(x[n/2-1]+x[n/2])/2;};
    std::cout<<label<<" control_us="<<median(a)<<" reuse_us="<<median(b)<<" speedup="<<median(ratios)
        <<" paired_min="<<*std::min_element(ratios.begin(),ratios.end())<<" paired_max="<<*std::max_element(ratios.begin(),ratios.end())<<'\n';
}
void emit(Runner& r,uint32_t heads,uint32_t width,uint32_t count,uint32_t offset,bool timed){
    tokspur::GdnChunkParams p{};p.rows=offset+count+1;p.heads=heads;p.key_width=128;p.value_width=width;p.offset=offset;p.count=count;
    p.input_stride=(heads*128+3)*4;p.output_stride=(heads*width+5)*4;
    Buffer state(r.device,heads*128*width),q(r.device,p.rows*p.input_stride/4),k(r.device,q.count),prefix(r.device,heads*64),attention(r.device,heads*64*64),residual(r.device,64*heads*width),next0(r.device,state.count),next1(r.device,state.count),y0(r.device,p.rows*p.output_stride/4),y1(r.device,y0.count);
    fill(state,44);fill(q,45);fill(k,46);fill(attention,47,.01f);fill(residual,48,.01f);
    std::fill_n(prefix.data(),prefix.count,NAN);
    for(uint32_t h=0;h<heads;++h)for(uint32_t i=0;i<count;++i)prefix.data()[h*64+i]=-float(i+1)/97;
    auto run=[&](unsigned arm,unsigned n){return r.run(arm,(heads*width+127)/128,n,[&](id<MTLComputeCommandEncoder> ce){
        std::array<Buffer*,9>b{&state,&q,&k,&prefix,&attention,&residual,arm?&next1:&next0,nullptr,arm?&y1:&y0};
        for(unsigned i=0;i<b.size();++i)if(b[i])[ce setBuffer:b[i]->obj offset:16 atIndex:i];[ce setBytes:&p length:sizeof(p) atIndex:14];});};
    for(unsigned seed=0;seed<4;++seed){fill(state,seed+44);fill(q,seed+45);fill(k,seed+46);
        if(seed==3)for(uint32_t h=0;h<heads;++h)for(uint32_t i=0;i<count;++i)prefix.data()[h*64+i]=-1000.0f-float(i);
        const std::vector<float> accepted(state.data(),state.data()+state.count);
        next0.reset();next1.reset();y0.reset();y1.reset();run(0,1);const auto f0=r.flags();run(1,1);
        need(f0==0&&r.flags()==0,"emit finite flag");need(next0.equal(next1)&&y0.equal(y1),"emit FP32 bits or inactive rows");
        need(std::memcmp(accepted.data(),state.data(),state.count*4)==0,"emit changed accepted state");
        const auto ys = reinterpret_cast<const uint32_t*>(y1.data());
        const uint32_t stride = p.output_stride/4;
        for (uint32_t row=0;row<p.rows;++row) for (uint32_t c=0;c<stride;++c)
            if (row<offset || row>=offset+count || c>=heads*width)
                need(ys[row*stride+c]==0x5a5a5a5a,"emit touched inactive output or row padding");
    }
    if(timed){for(uint32_t h=0;h<heads;++h)for(uint32_t i=0;i<count;++i)prefix.data()[h*64+i]=-float(i+1)/97;
        timing("emit heads="+std::to_string(heads)+" width="+std::to_string(width)+" count="+std::to_string(count),run);}
    for(auto* source:{&state,&q,&k,&prefix,&attention,&residual})for(float bad:{INFINITY,-INFINITY,NAN}){
        const size_t ix=source==&q||source==&k?offset*p.input_stride/4:0;const float old=source->data()[ix];source->data()[ix]=bad;
        run(0,1);const auto f0=r.flags();run(1,1);need(f0&&r.flags()==f0,"emit nonfinite refusal");source->data()[ix]=old;}
    for(auto* b:{&state,&q,&k,&prefix,&attention,&residual,&next0,&next1,&y0,&y1})b->check();
    std::cout<<"PASS emit heads="<<heads<<" width="<<width<<" count="<<count<<" offset="<<offset<<" seeds=4 nonfinite_cases=18 byte_exact=true\n";
}
int main(int argc,char**argv){@autoreleasepool{try{
    need(argc==4,"usage: test-gdn_cs64_reuse reference.metallib production.metallib --validate|--timing");
    const std::string mode=argv[3];need(mode=="--validate"||mode=="--timing","unknown mode");
    Runner r(argv+1,mode=="--validate");
    for(uint32_t count:{1u,2u,3u,7u,8u,31u,32u,63u,64u}) {
        emit(r,3,7,count,2,false);
        emit(r,48,128,count,0,mode=="--timing"&&(count==1||count==8||count==32||count==64));
    }
    emit(r,1,128,64,3,false);
    for(uint32_t width:{1u,127u})emit(r,3,width,3,1,false);
    std::cout<<"PASS all model-free intermediate reuse screens\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}}
