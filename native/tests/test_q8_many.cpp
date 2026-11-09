#include "tokspur/q8_many.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

namespace {
using namespace tokspur;
size_t checks = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw std::runtime_error("invalid Q8-many CPU request admitted"); }
void geometry() {
    for (uint32_t n : {1,2,4,8}) {
        const auto b = q8_many_bytes({2560,10240,n});
        require(b.matrix == 27852800 && b.input == uint64_t(n) * 10240 && b.output == uint64_t(n) * 40960,"real Q8-many geometry differs");
    }
#ifndef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    for (const Q8ManyShape s : std::array<Q8ManyShape,10>{{{0,1,1},{31,1,1},{33,1,1},{65568,1,1},{32,0,1},{32,1000001,1},{32,1,0},{32,1,3},{32,1,16},{UINT32_MAX,UINT32_MAX,8}}})
        reject([&] { static_cast<void>(q8_many_bytes(s)); });
#else
    for (const Q8ManyShape s : std::array<Q8ManyShape,10>{{{0,1,1},{31,1,1},{33,1,1},{65568,1,1},{32,0,1},{32,1000001,1},{32,1,0},{32,1,5},{32,1,16},{UINT32_MAX,UINT32_MAX,8}}})
        reject([&] { static_cast<void>(q8_many_bytes(s)); });
#endif
}
void reference() {
    for (uint32_t c : {32,96,320,2560}) for (uint32_t r : {1,3,7}) for (uint32_t n : {1,2,4,8}) {
        const Q8ManyShape s{c,r,n}; const auto b = q8_many_bytes(s);
        std::vector<std::byte> w(b.matrix); std::vector<float> x(b.input / 4);
        for (uint32_t row = 0; row < r; ++row) for (uint32_t block = 0; block < c / 32; ++block) {
            const auto at = (uint64_t(row) * (c / 32) + block) * 34;
            w[at] = std::byte{0}; w[at+1] = std::byte{0x34}; // exactly 1/4
            for (uint32_t lane = 0; lane < 32; ++lane) w[at+2+lane] = std::byte((row * 11 + block * 7 + lane * 3) & 255);
        }
        for (uint32_t t = 0; t < n; ++t) for (uint32_t column = 0; column < c; ++column)
            x[uint64_t(t) * c + column] = float(int((t * 19 + column * 13) % 129) - 64) / 1024;
        const auto original = w; const auto input = x;
        const auto got = reference_q8_many(w,x,s,b.output * 2);
        require(got.size() == uint64_t(n) * r,"Q8-many CPU result size differs");
        for (uint32_t t = 0; t < n; ++t) for (uint32_t row = 0; row < r; ++row) {
            double expected = 0;
            for (uint32_t column = 0; column < c; ++column) {
                const auto code = (row * 11 + column / 32 * 7 + column % 32 * 3) & 255;
                const auto signed_code = code < 128 ? int(code) : int(code) - 256;
                expected += (0.25 * signed_code) * x[uint64_t(t) * c + column];
            }
            require(got[uint64_t(t) * r + row] == expected,"independent Q8-many double reference differs");
        }
        require(w == original && x == input,"Q8-many CPU reference mutated inputs");
        reject([&] { static_cast<void>(reference_q8_many(w,x,s,b.output * 2 - 1)); });
        reject([&] { static_cast<void>(reference_q8_many(std::span(w).first(w.size()-1),x,s,b.output * 2)); });
        reject([&] { static_cast<void>(reference_q8_many(w,std::span(x).first(x.size()-1),s,b.output * 2)); });
    }
    std::vector<std::byte> w(34); w[1] = std::byte{0x3c}; std::vector<float> x(32,1);
    for (float bad : {INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
        x.back() = bad; reject([&] { static_cast<void>(reference_q8_many(w,x,{32,1,1},8)); });
    }
    x.back() = 1;
    for (uint16_t bits : {uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e01),uint16_t(0xfe01)}) {
        w[0] = std::byte(bits & 255); w[1] = std::byte(bits >> 8);
        reject([&] { static_cast<void>(reference_q8_many(w,x,{32,1,1},8)); });
    }
    w[0] = std::byte{0}; w[1] = std::byte{0x3c};
    require(reference_q8_many(w,x,{32,1,1},8) == std::vector<double>{0},"Q8-many CPU valid reuse failed");
}
} // namespace
int main() {
    try { geometry(); reference(); std::cout << "Owned Q8-many CPU checks passed: " << checks << '\n'; }
    catch (const std::exception& e) { std::cerr << "Owned Q8-many CPU failed: " << e.what() << '\n'; return 1; }
}
