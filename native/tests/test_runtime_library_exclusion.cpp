#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include <iostream>
#include <stdexcept>
#include <string_view>

#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
#error "library exclusion must exercise the production graph"
#endif

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("expected production, diagnostic and probe libraries");
        { tokspur::MetalGraph production(argv[1], 64ull << 20); }
        for (int i = 2; i < 4; ++i) {
            bool refused = false;
            try { tokspur::MetalGraph forbidden(argv[i], 64ull << 20); }
            catch (const tokspur::FormatError& error) {
                const std::string_view expected = i == 2 ? "diagnostic metallib is refused by serving graphs"
                    : "probe-only metallib is refused by serving graphs";
                refused = error.what() == expected;
            }
            if (!refused) throw std::runtime_error("production graph did not reject the isolated library");
        }
        std::cout << "production graph accepted its library and refused diagnostic and unchecked libraries\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
