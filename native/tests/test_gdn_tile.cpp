#include "tokspur/gdn_tile.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

namespace {
using namespace tokspur;
size_t checks=0;
void require(bool ok,const char* why) { ++checks; if(!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch(const FormatError&) { return; } throw std::runtime_error("invalid GDN tile reference admitted"); }
void geometry() {
    const auto c=gdn_conv_tile_bytes(10240,{32,32});const auto d=gdn_delta_tile_bytes(48,128,{32,32});
    require(c.history==122880 && c.weights==163840 && c.projected==1310720 && c.output==1310720,"actual convolution geometry differs");
    require(d.state==3145728 && d.query_key==786432 && d.value_output==786432 && d.gates==6144,"actual delta geometry differs");
    for(StateTile t:std::array<StateTile,7>{{{1,1},{3,1},{64,1},{2,0},{2,3},{32,33},{0,0}}}) {
        reject([&]{ validate_state_tile(t); });
        reject([&]{ static_cast<void>(gdn_conv_tile_bytes(10240,t)); });
        reject([&]{ static_cast<void>(gdn_delta_tile_bytes(48,128,t)); });
    }
    for(uint32_t bad:{0,65537}) reject([&]{static_cast<void>(gdn_conv_tile_bytes(bad,{2,1}));});
    for(uint32_t bad:{0,49}) reject([&]{static_cast<void>(gdn_delta_tile_bytes(bad,128,{2,1}));});
    for(uint32_t bad:{0,129}) reject([&]{static_cast<void>(gdn_delta_tile_bytes(48,bad,{2,1}));});
}
void convolution() {
    for(uint32_t channels:{1,3,31,10240}) for(uint32_t capacity:{2,4,8,16,32}) {
        const auto b=gdn_conv_tile_bytes(channels,{capacity,capacity});
        std::vector<double> history(b.history/4),x(b.projected/4),w(b.weights/4);
        for(size_t i=0;i<history.size();++i) history[i]=double(int(i%17)-8)/32;
        for(size_t i=0;i<x.size();++i) x[i]=double(int(i%31)-15)/64;
        for(uint32_t channel=0;channel<channels;++channel) {
            w[channel*4]=0.125;w[channel*4+1]=-0.25;w[channel*4+2]=0.5;w[channel*4+3]=1;
        }
        const auto accepted=history,input=x,weights=w;
        for(uint32_t steps=1;steps<=capacity;++steps) {
            const auto got=gdn_convolution_tile_reference(history,x,w,channels,{capacity,steps},64*1024*1024);
            const auto source=[&](uint32_t time,uint32_t channel) {
                return time<3?history[uint64_t(time)*channels+channel]:x[uint64_t(time-3)*channels+channel];
            };
            require(got.output.size()==uint64_t(steps)*channels,"convolution valid-prefix output length");
            for(uint32_t step=0;step<steps;++step) for(uint32_t channel=0;channel<channels;++channel) {
                const double expected=source(step+3,channel)+source(step,channel)*0.125-source(step+1,channel)*0.25+source(step+2,channel)*0.5;
                require(got.output[uint64_t(step)*channels+channel]==expected,"independent finite convolution tile reference");
            }
            for(uint32_t time=0;time<3;++time) for(uint32_t channel=0;channel<channels;++channel)
                require(got.history[uint64_t(time)*channels+channel]==source(steps+time,channel),"convolution prefix history selection");
            require(history==accepted && x==input && w==weights,"convolution reference changed accepted inputs");
        }
    }
    // Capacity padding is not part of the processed prefix.
    std::vector<double> history(3,0),x{1,std::numeric_limits<double>::quiet_NaN()},w{0,0,0,1};
    require(gdn_convolution_tile_reference(history,x,w,1,{2,1},32).output==std::vector<double>{1},"unused convolution padding was read");
    reject([&]{static_cast<void>(gdn_convolution_tile_reference(history,x,w,1,{2,2},40));});
    reject([&]{static_cast<void>(gdn_convolution_tile_reference(history,x,w,1,{2,1},31));});
    reject([&]{static_cast<void>(gdn_convolution_tile_reference(std::span(history).first(2),x,w,1,{2,1},32));});
}
void delta() {
    constexpr uint32_t width=3,heads=1;
    for(uint32_t capacity:{2,4,8,16,32}) for(uint32_t mode:{0,1}) {
        const auto b=gdn_delta_tile_bytes(heads,width,{capacity,capacity});
        std::vector<double> state(b.state/4),q(b.query_key/4),key(q.size()),value(b.value_output/4),a(b.gates/4),beta(a.size());
        for(size_t i=0;i<state.size();++i) state[i]=double(int(i%17)-8)/64;
        for(uint32_t step=0;step<capacity;++step) {
            q[step*128]=1;key[step*128]=1;a[step]=mode?1:0.5;beta[step]=mode?1:0;
            for(uint32_t v=0;v<width;++v) value[step*width+v]=double(int((step*3+v)%17)-8)/32;
        }
        const auto accepted=state;
        for(uint32_t steps=1;steps<=capacity;++steps) {
            const auto got=gdn_delta_tile_reference(state,q,key,value,a,beta,heads,width,{capacity,steps},64*1024*1024);
            for(uint32_t step=0;step<steps;++step) for(uint32_t v=0;v<width;++v) {
                const auto expected=mode?value[step*width+v]:std::ldexp(state[v],-int(step+1));
                require(got.output[step*width+v]==expected,"independently calculable delta trajectory differs");
            }
            for(uint32_t k=0;k<128;++k) for(uint32_t v=0;v<width;++v) {
                const auto expected=mode?(k?state[k*width+v]:value[(steps-1)*width+v]):std::ldexp(state[k*width+v],-int(steps));
                require(got.state[k*width+v]==expected,"independent prefix-k delta state differs");
            }
            require(state==accepted,"delta reference changed accepted state");
        }
        const auto maximum=uint64_t(state.size()+value.size())*8;
        auto retained=gdn_delta_tile_reference(state,q,key,value,a,beta,heads,width,{capacity,capacity},maximum);
        const auto before=retained;
        const auto current=value.back();value.back()=std::numeric_limits<double>::quiet_NaN();
        reject([&]{retained=gdn_delta_tile_reference(state,q,key,value,a,beta,heads,width,{capacity,capacity},maximum);});
        require(retained.state==before.state && retained.output==before.output && state==accepted,"late delta refusal changed retained/accepted result");
        value.back()=current;
        reject([&]{static_cast<void>(gdn_delta_tile_reference(state,q,key,value,a,beta,heads,width,{capacity,capacity},maximum-1));});
    }
    std::vector<double> state(128),q(256),key(256),value{1,std::numeric_limits<double>::quiet_NaN()},a{0,std::numeric_limits<double>::quiet_NaN()},beta{1,std::numeric_limits<double>::quiet_NaN()};
    q[0]=key[0]=1;
    require(gdn_delta_tile_reference(state,q,key,value,a,beta,1,1,{2,1},1032).output==std::vector<double>{1},"unused delta padding was read");
    reject([&]{static_cast<void>(gdn_delta_tile_reference(state,q,key,value,a,beta,1,1,{2,2},1040));});
    a[0]=1.01;
    reject([&]{static_cast<void>(gdn_delta_tile_reference(state,q,key,value,a,beta,1,1,{2,1},1032));});
}
}
int main() {
    try {geometry();convolution();delta();std::cout<<"Owned GDN tile CPU checks passed: "<<checks<<'\n';}
    catch(const std::exception& error){std::cerr<<"Owned GDN tile CPU failed: "<<error.what()<<'\n';return 1;}
}
