#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/q8_many.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
using namespace tokspur;
size_t checks = 0, batches = 0; double worst = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid Q8-many GPU request admitted"); }
struct Arena {
    MetalGraph& g; std::vector<GpuView> all;
    GpuView bytes(std::span<const std::byte> data, bool immutable = true) {
        std::vector<std::byte> padded(data.size()+8,std::byte{0x6d}); std::copy(data.begin(),data.end(),padded.begin()+4);
        const auto v = g.upload(padded,"Q8-many generated fixture",immutable); all.push_back(v); return g.slice(v,4,data.size());
    }
    GpuView floats(std::span<const float> data, bool immutable = true) { return bytes(std::as_bytes(data),immutable); }
    GpuView zeros(size_t count) { return floats(std::vector<float>(count),false); }
    void guards() const {
        for (auto v : all) { const auto bytes = g.read(v); for (size_t i = 0; i < 4; ++i)
            require(bytes[i] == std::byte{0x6d} && bytes[bytes.size()-1-i] == std::byte{0x6d},"Q8-many wrote beyond an admitted view"); }
    }
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (auto v : all) g.release(v); }
};
void submit(MetalGraph& g, uint64_t dispatches, GraphMemory before) {
    const auto b = g.submit(); ++batches; const auto after = g.memory();
    require(b.dispatches == dispatches && after.allocations == before.allocations && after.live_bytes == before.live_bytes &&
        after.imported_backing_bytes == before.imported_backing_bytes,"Q8-many dispatch/allocation invariant failed");
}
void compare(std::span<const std::byte> bytes, std::span<const double> reference) {
    require(bytes.size() == reference.size()*4,"Q8-many result geometry differs");
    for (size_t i = 0; i < reference.size(); ++i) {
        float value; std::memcpy(&value,bytes.data()+i*4,4);
        const auto error = std::abs(double(value)-reference[i]) / std::max(1.0,std::abs(reference[i])); worst = std::max(worst,error);
        require(std::isfinite(value) && error <= 5e-5,"Q8-many double reference error");
    }
}
void exercise(MetalGraph& g, uint32_t c, uint32_t r, const std::vector<std::byte>& weights, const std::vector<float>& input, uint32_t width) {
    Arena a{g,{}}; const auto w = a.bytes(weights), x = a.floats(input,false), out = a.zeros(uint64_t(r)*width);
    const auto baseline = a.zeros(uint64_t(r)*width); const auto before = g.memory(); const auto policy = g.kernel_policy();
    const auto reference = reference_q8_many(weights,input,{c,r,width},64*1024*1024);
    g.configure_block_gemv(1); g.begin();
    for (uint32_t t = 0; t < width; ++t) g.gemv(8,w,g.slice(x,uint64_t(t)*c*4,uint64_t(c)*4),g.slice(baseline,uint64_t(t)*r*4,uint64_t(r)*4),c,r);
    submit(g,width,before); const auto expected = g.read(baseline); compare(expected,reference);
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        g.begin(); g.q8_matvec_many(w,x,out,c,r,width); submit(g,1,before);
        require(g.read(out) == expected,"Q8-many changed individual FP32 result bits"); compare(g.read(out),reference);
    }
    require(g.read(w) == weights && g.read(x) == std::vector<std::byte>(std::as_bytes(std::span(input)).begin(),std::as_bytes(std::span(input)).end()),"Q8-many changed input weights/vectors");
    g.configure_block_gemv(policy[1]); require(g.kernel_policy() == policy,"Q8-many changed execution policy");
    auto bad = input; bad.back() = INFINITY; g.write(x,std::as_bytes(std::span(bad)));
    g.begin(); g.q8_matvec_many(w,x,out,c,r,width); reject([&] { static_cast<void>(g.submit()); });
    g.write(x,std::as_bytes(std::span(input))); g.begin(); g.q8_matvec_many(w,x,out,c,r,width); submit(g,1,before);
    require(g.read(out) == expected,"Q8-many valid reuse failed after nonfinite input");
    a.guards();
}
void admission(MetalGraph& g) {
    Arena a{g,{}}; constexpr uint32_t c=32,r=32,n=8;
    const auto matrix = a.bytes(std::vector<std::byte>(r*34),false), input_backing = a.zeros(c*n+1), out = a.zeros(r*n);
    const auto input = g.slice(input_backing,0,c*n*4); const auto before = g.memory();
    g.begin(); reject([&] { g.prepare_q8_matvec_many(8); }); g.abort();
    for (uint32_t width : {0,3,16}) { g.begin(); reject([&] { g.q8_matvec_many(matrix,input,out,c,r,width); }); g.abort(); }
    g.begin(); reject([&] { g.q8_matvec_many(matrix,input,out,c-1,r,n); }); g.abort();
    g.begin(); reject([&] { g.q8_matvec_many(matrix,input,out,c,r+1,n); }); g.abort();
    g.begin(); reject([&] { g.q8_matvec_many(g.slice(matrix,0,matrix.bytes-1),input,out,c,r,n); }); g.abort();
    g.begin(); reject([&] { g.q8_matvec_many(matrix,input,input,c,r,n); }); g.abort();
    g.begin(); reject([&] { g.q8_matvec_many(matrix,input,g.slice(input_backing,4,r*n*4),c,r,n); }); g.abort();
    g.begin(); reject([&] { g.q8_matvec_many(matrix,input,g.slice(matrix,4,r*n*4),c,r,n); }); g.abort();
    auto unaligned = matrix; unaligned.offset += 2;
    g.begin(); reject([&] { g.q8_matvec_many(unaligned,input,out,c,r,n); }); g.abort();
    const auto immutable = a.floats(std::vector<float>(r*n));
    g.begin(); reject([&] { g.q8_matvec_many(matrix,input,immutable,c,r,n); }); g.abort();
    reject([&] { static_cast<void>(g.zeros(64*1024*1024,"must exceed existing GPU budget")); });
    g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); submit(g,1,g.memory());
    compare(g.read(out),std::vector<double>(r*n,0));
    require(g.memory().allocations == before.allocations+1,"Q8-many refusal leaked an allocation");
    // All nonfinite half classes, in the final block after valid earlier rows.
    for (uint16_t bits : {uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e01),uint16_t(0xfe01)}) {
        std::vector<std::byte> bad(r*34); bad[bad.size()-34] = std::byte(bits&255); bad[bad.size()-33] = std::byte(bits>>8);
        g.write(matrix,bad); g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); reject([&] { static_cast<void>(g.submit()); });
        g.write(matrix,std::vector<std::byte>(r*34)); g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); submit(g,1,g.memory());
        compare(g.read(out),std::vector<double>(r*n,0));
    }
    std::vector<float> bad_input(c*n,0);
    for (float value : {INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
        bad_input.back() = value; g.write(input,std::as_bytes(std::span(bad_input)));
        g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); reject([&] { static_cast<void>(g.submit()); });
        bad_input.back() = 0; g.write(input,std::as_bytes(std::span(bad_input)));
        g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); submit(g,1,g.memory()); compare(g.read(out),std::vector<double>(r*n,0));
    }
    std::vector<std::byte> huge(r*34,std::byte{127});
    for (size_t at=0;at<huge.size();at+=34) { huge[at]=std::byte{0xff}; huge[at+1]=std::byte{0x7b}; }
    std::fill(bad_input.begin(),bad_input.end(),std::numeric_limits<float>::max());
    g.write(matrix,huge); g.write(input,std::as_bytes(std::span(bad_input)));
    g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); reject([&] { static_cast<void>(g.submit()); });
    g.write(matrix,std::vector<std::byte>(r*34)); std::fill(bad_input.begin(),bad_input.end(),0); g.write(input,std::as_bytes(std::span(bad_input)));
    g.begin(); g.q8_matvec_many(matrix,input,out,c,r,n); submit(g,1,g.memory()); compare(g.read(out),std::vector<double>(r*n,0)); a.guards();
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph g(argv[1],64*1024*1024); const auto initial = g.memory().live_bytes;
        { Arena a{g,{}}; const auto w=a.bytes(std::vector<std::byte>(34)),x=a.zeros(32),y=a.zeros(1);
          g.begin(); reject([&] { g.q8_matvec_many(w,x,y,32,1,1); }); g.abort(); }
        for (uint32_t n : {1,2,4,8}) g.prepare_q8_matvec_many(n);
        std::mt19937 random(0x51384d4e);
        for (const auto shape : std::array<std::array<uint32_t,2>,7>{{{32,1},{64,3},{96,5},{256,7},{2560,48},{2560,513},{6144,9}}}) {
            const auto c=shape[0],r=shape[1]; std::vector<std::byte> w(uint64_t(c/32)*r*34);
            for (auto& v:w) v=std::byte(random()&255);
            for (size_t at=0;at<w.size();at+=34) { w[at]=std::byte{0}; w[at+1]=std::byte{0x18}; }
            for (uint32_t n : {1,2,4,8}) {
                std::vector<float> x(uint64_t(c)*n); for(auto& v:x) v=float(int(random()%2049)-1024)/4096;
                exercise(g,c,r,w,x,n);
            }
        }
        // Every finite positive/negative half scale, including signed zero and
        // subnormals, and row tails. Int8 extremes exercise scale-product range.
        std::vector<std::byte> w(63488*34); uint32_t row=0;
        for (uint32_t bits=0;bits<65536;++bits) if ((bits&0x7c00)!=0x7c00) {
            w[row*34]=std::byte(bits&255); w[row*34+1]=std::byte(bits>>8);
            for (uint32_t lane=0;lane<32;++lane) w[row*34+2+lane]=std::byte(lane?127:128); ++row;
        }
        for (uint32_t n : {1,2,4,8}) exercise(g,32,row,w,std::vector<float>(32*n,0.03125f),n);
        admission(g); require(g.memory().live_bytes==initial,"Q8-many fixture leaked GPU buffers");
        std::cout << "Owned Q8-many Metal checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "Owned Q8-many Metal failed: " << e.what() << '\n'; return 1; }
}
