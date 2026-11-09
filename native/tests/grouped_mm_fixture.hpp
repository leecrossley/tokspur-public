#pragma once
#include "tokspur/grouped_mm.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include <array>
#include <vector>

namespace grouped_fixture {
struct Shape { uint32_t type,k,r,e,s,n; bool broadcast; tokspur::PrefillWeightLayout layout=tokspur::PrefillWeightLayout::gguf; };
inline std::vector<std::byte> weights(Shape f){
    const auto& q=tokspur::encoding(f.type);std::vector<std::byte> out(uint64_t(f.k/q.elements_per_block)*q.bytes_per_block*f.r*f.e);uint32_t rng=44;
    if(out.size()>(32ull<<20))throw tokspur::FormatError("generated Graph fixture weight budget");
    constexpr std::array<uint16_t,8> ds{0x0401,0x0568,0x068e,0x0ac2,0x2401,0x2803,0xa401,0x1801};
    for(size_t at=0;at<out.size();at+=q.bytes_per_block){
        for(uint32_t i=0;i<q.bytes_per_block;++i){rng=rng*1664525u+1013904223u;out[at+i]=std::byte(rng>>24);}
        auto d=ds[(at/q.bytes_per_block)%ds.size()];out[at]=std::byte(d&255);out[at+1]=std::byte(d>>8);
        if(f.type!=8){out[at+2]=std::byte{0x03};out[at+3]=std::byte{0x1c};}
    }return out;
}
inline std::vector<float> represented(Shape f,std::span<const std::byte> raw){std::vector<float> out(uint64_t(f.k)*f.r*f.e);tokspur::dequantize(f.type,raw,out);return out;}
inline float staged(float x){return float(_Float16(x));}
}
