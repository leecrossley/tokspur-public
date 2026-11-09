#include "tokspur/nax_split_policy.hpp"
#include <iostream>
#include <stdexcept>
using namespace tokspur;
void check(bool b) { if (!b) throw std::runtime_error("M8 split policy check failed"); }
int main() {
    try {
        NaxSplitShape q{NaxSplitFamily::q8, 2560, 512, 5, 4, false};
        check(valid_nax_split(q, 5)); check(valid_nax_split(q, 1));
        check(!valid_nax_split(q, 2)); check(!valid_nax_split(q, 4)); check(!valid_nax_split(q, 32));
        check(!valid_nax_split(q, 5, 1));
        for (uint32_t columns : {1u, 31u, 33u, 65568u}) {
            auto invalid = q; invalid.columns = columns; check(!valid_nax_split(invalid, 1));
        }
        auto too_wide = q; too_wide.outputs = 524289; check(!valid_nax_split(too_wide, 1));
        NaxSplitShape t{NaxSplitFamily::q8_t64, 2560, 1024, 5, 64, false};
        check(valid_nax_split(t, 10)); check(!valid_nax_split(t, 20));
        NaxSplitShape b{NaxSplitFamily::bf16, 2560, 1024, 5, 32, false};
        check(valid_nax_split(b, 5)); check(!valid_nax_split(b, 3));
        b.dual = true; check(!valid_nax_split(b, 1));
        auto overrides = NaxSplitOverrides::parse(parse_json(R"({"schema":1,"entries":[{"family":"q8","columns":2560,"outputs":512,"tokens":5,"tile":4,"dual":false,"splits":1}]})"));
        check(overrides.choose(true, q, 5, 0) == 1);
        check(overrides.choose(false, q, 5, 0) == 5); // Drafter/priming roles cannot acquire the override.
        ++q.tokens; check(overrides.choose(true, q, 5, 0) == 5);
        std::cout << "M8 helper candidates, shapes, scratch and target role passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
