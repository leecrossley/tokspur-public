#include "tokspur/speculative_controls.hpp"

#include <array>
#include <iostream>

namespace {
using namespace tokspur;
unsigned tap_checks = 0;
void check(bool result, const char* why) { ++tap_checks; if (!result) throw FormatError(why); }
template<class F> void refuse(F&& action) {
    bool rejected = false;
    try { action(); } catch (const FormatError&) { rejected = true; }
    check(rejected, "invalid DFlash tap plan admitted");
}

void tap_budgets() {
    struct Case { uint32_t capacity; uint64_t known_bytes, budget; };
    // Five target layers of width 2560, plus the separately resident eight-row verification bank.
    const std::array cases{
        Case{0, 409600, 8ull << 20}, Case{8, 409600, 8ull << 20},
        Case{512, 26214400, 48ull << 20}, Case{1024, 52428800, 64ull << 20},
        Case{2048, 104857600, 160ull << 20}};
    for (const auto& test : cases) {
        const auto budget = dflash2_tap_budget(test.capacity);
        const auto plan = dflash2_tap_plan(test.capacity, 5, 2560, budget);
        check(budget == test.budget, "DFlash tap startup budget changed");
        check(plan.known_capacity == (test.capacity ? test.capacity : 8), "DFlash tap ordinary capacity differs");
        check(plan.known_bytes == test.known_bytes && plan.verify_bytes == 409600, "DFlash tap bank extents differ from target geometry");
        const auto exact = test.known_bytes + 409600;
        check(dflash2_tap_plan(test.capacity, 5, 2560, exact).known_bytes == test.known_bytes, "exact tap bank budget refused");
        refuse([&] { static_cast<void>(dflash2_tap_plan(test.capacity, 5, 2560, exact - 1)); });
    }
    // This was the capacity-1024 startup budget before the repair. It cannot hold both banks.
    refuse([] { static_cast<void>(dflash2_tap_plan(1024, 5, 2560, 48ull << 20)); });
    for (const uint32_t capacity : {1u, 7u, 9u, 513u, 1023u, 1025u, 2049u, UINT32_MAX})
        refuse([&] { static_cast<void>(dflash2_tap_plan(capacity, 5, 2560, 160ull << 20)); });
    for (const uint32_t slots : {0u, 17u, UINT32_MAX})
        refuse([&] { static_cast<void>(dflash2_tap_plan(8, slots, 2560, 8ull << 20)); });
    for (const uint32_t width : {0u, 3u, UINT32_MAX - 3})
        refuse([&] { static_cast<void>(dflash2_tap_plan(2048, 16, width, maximum_dflash2_tap_budget_bytes)); });
    for (const uint64_t budget : {uint64_t(0), maximum_dflash2_tap_budget_bytes + 1, UINT64_MAX})
        refuse([&] { static_cast<void>(dflash2_tap_plan(8, 5, 2560, budget)); });
    const auto small = dflash2_tap_plan(8, 16, 32, 32768);
    check(small.known_bytes == 16384 && small.verify_bytes == 16384, "generated small tap geometry differs");
}
}

int main() {
    using namespace tokspur;
    try {
        if (parse_dflash2_precision("q8_0") != DFlash2Precision::q8_0 || parse_dflash2_precision("bf16") != DFlash2Precision::bf16 || parse_native_speculative_depth("1") != 1 || parse_native_speculative_depth("2") != 2 || parse_native_speculative_depth("3") != 3 || parse_native_speculative_depth("7") != 7) throw FormatError("supported speculative control changed");
        if (dflash2_default_max_drafts != 7 || dflash2_width4_max_drafts != 3 || dflash2_default_vocab_slice != 32768) throw FormatError("DFlash2 speculative constants changed");
        for (uint32_t k = 1; k <= 7; ++k) {
            const char digit[2] = {char('0' + k), '\0'};
            if (parse_dflash2_max_drafts(digit) != k) throw FormatError("parse_dflash2_max_drafts mismatch");
        }
        if (parse_dflash2_vocab_slice("0") != 0 || parse_dflash2_vocab_slice("full") != 0 || parse_dflash2_vocab_slice("256") != 256 || parse_dflash2_vocab_slice("32768") != 32768 || parse_dflash2_vocab_slice("65536") != 65536) throw FormatError("parse_dflash2_vocab_slice mismatch");
        unsigned refused = 0;
        for (const auto value : {"fp16", "", "Q8_0", "BF16"}) {
            try { static_cast<void>(parse_dflash2_precision(value)); } catch (const FormatError&) { ++refused; }
        }
        for (const auto value : {"0", "4", "8", "", "02", "03"}) {
            try { static_cast<void>(parse_native_speculative_depth(value)); } catch (const FormatError&) { ++refused; }
        }
        for (const auto value : {"0", "8", "", "03", "-1"}) {
            try { static_cast<void>(parse_dflash2_max_drafts(value)); } catch (const FormatError&) { ++refused; }
        }
        for (const auto value : {"", "00", "32", "224", "32769", "032768", "2097152"}) {
            try { static_cast<void>(parse_dflash2_vocab_slice(value)); } catch (const FormatError&) { ++refused; }
        }
        if (refused != 22) throw FormatError("unsupported speculative control admitted");
        tap_budgets();
        std::cout << "speculative controls: supported modes and 22 unsupported-mode refusals passed; " << tap_checks << " DFlash tap planning checks, no GPU execution\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
