#include <charconv>
#include <cstdio>
#include <cstring>
#include <cmath>
int main() {
    const char* cases[] = {"1e-400","4.9e-324","5e-324","2.4703282292062327e-324","2.4703282292062328e-324","2.47032822920623272e-324","1e-320","2.2250738585072011e-308","1e309","1.7976931348623157e308","1.7976931348623158e308","1.7976931348623159e308","-0","0e5","0e-999999","1e999999","-1e-400","0.1","123456789012345678901234567890","1E2","1e+2","00","-","1e","inf","nan","+1"};
    for (auto* c : cases) {
        double v = -12345.0; auto r = std::from_chars(c, c + std::strlen(c), v);
        printf("%-28s ec=%d consumed=%zu value=%.17g\n", c, int(r.ec), size_t(r.ptr - c), v);
    }
}
