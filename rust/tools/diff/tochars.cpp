#include <charconv>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
// Usage: tochars N  -> prints "<hex bits> <to_chars text>" for N random doubles plus fixed cases.
int main(int argc, char** argv) {
    const double fixed[] = {0.0,-0.0,1.0,0.7,100.0,1e-5,1e-4,1e-3,0.123,1e5,1e4,123456.0,1.5e-300,0.1+0.2,5e-324,1e100,1e16,1.2345678901234567e16,9007199254740993.0,1e23,123456789012345680.0};
    auto emit = [](double v) { char buf[64]; auto r = std::to_chars(buf, buf + 64, v); uint64_t bits; std::memcpy(&bits, &v, 8); printf("%016llx %.*s\n", (unsigned long long)bits, int(r.ptr - buf), buf); };
    for (double v : fixed) emit(v);
    const long n = argc > 1 ? std::atol(argv[1]) : 0;
    std::mt19937_64 random(0x746f6368617273ull);
    for (long i = 0; i < n; ++i) {
        uint64_t bits = random(); double v; std::memcpy(&v, &bits, 8);
        if (v != v || v - v != 0) continue;
        emit(v);
        double small = double(random() % 100000001) / 1000000.0; emit(small);   // sampling-like values in [0,100]
        double p = double(random() % 1000001) / 1000000.0; emit(p);
    }
}
