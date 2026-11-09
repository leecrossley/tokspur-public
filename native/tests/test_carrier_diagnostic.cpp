#include "tokspur/carrier_diagnostic.hpp"
#include "tokspur/json.hpp"
#include "tokspur/mtp_anchor_state.hpp"
#include <bit>
#include <charconv>
#include <iostream>
#include <stdexcept>

template<class F> void refuses(F action) {
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("diagnostic admitted an invalid transition");
}
int main() {
    using namespace tokspur;
    for (uint32_t before = 0; before < 4; ++before) {
        if (carrier_single_action_yield(before, before) || !carrier_single_action_yield(before, before + 1)) return 1;
        refuses([&] { carrier_single_action_yield(before, before + 2); });
    }
    refuses([] { carrier_single_action_yield(2, 1); });
    {
        CarrierDiagnosticWindow totals;
        totals.prepare(CarrierDiagnosticMode::batch, 1, false);
        if (totals.records_dispatches()) return 1;
        totals.begin(true);
        for (uint32_t i = 0; i < 8; ++i) totals.batch(16384);
        totals.end();
        refuses([&] { totals.begin(true); });
        for (auto mode : {CarrierDiagnosticMode::dispatch, CarrierDiagnosticMode::replay}) {
            CarrierDiagnosticWindow invalid;
            refuses([&] { invalid.prepare(mode, 1, false); });
        }
        CarrierDiagnosticWindow invalid;
        invalid.prepare(CarrierDiagnosticMode::batch, 1, false); invalid.begin(true);
        refuses([&] { invalid.batch(16384, 1); });
    }
    {
        const std::array<uint32_t, 2> ids{8, 9}; const std::array<float, 2> scores{.2f, .3f};
        MtpGenerationCarry carry;
        carry.admit({.id=1, .remaining=16, .begin=true, .remaining_compute=16, .total_budget=16, .v2=true}, 8, 100, {});
        carry.commit_round(2, ids, scores, 100, 102, 2, false);
        refuses([&] { static_cast<void>(carry.finish(2, 102, false)); });
        refuses([&] { static_cast<void>(carry.finish_diagnostic_action(1, 102, false)); });
        refuses([&] { static_cast<void>(carry.finish_diagnostic_action(2, 103, false)); });
        if (carry.finish_diagnostic_action(2, 102, false) || !carry.active() || carry.remaining() != 14) return 1;
        refuses([&] { static_cast<void>(carry.finish_diagnostic_action(2, 102, false)); });
        static_cast<void>(carry.abort({.id=1, .abort=true}));
        MtpGenerationCarry old;
        old.admit({.id=1, .remaining=16, .begin=true}, 8, 100, {});
        old.commit_round(2, ids, scores, 100, 102, 2, false);
        refuses([&] { static_cast<void>(old.finish_diagnostic_action(2, 102, false)); });
        MtpGenerationCarry overflow;
        overflow.admit({.id=1, .remaining=16, .begin=true, .remaining_compute=16, .total_budget=16, .v2=true}, 8, 100, {});
        overflow.commit_round(2, ids, scores, 100, 102, 1, false);
        refuses([&] { static_cast<void>(overflow.finish_diagnostic_action(1, 102, false)); });
    }
    for (auto limit : {0u, 5u}) { CarrierDiagnosticWindow w; refuses([&] { w.prepare(CarrierDiagnosticMode::batch, limit); }); }
    for (auto bad : {0u, 8193u}) {
        CarrierDiagnosticWindow w; w.prepare(CarrierDiagnosticMode::batch, 4); w.begin(true);
        refuses([&] { w.batch(bad); }); if (w.valid()) return 1;
        refuses([&] { w.batch(1); }); refuses([&] { w.end(); });
    }
    { CarrierDiagnosticWindow w; w.prepare(CarrierDiagnosticMode::batch, 4); refuses([&] { w.begin(false); }); if (w.valid()) return 1; }
    CarrierDiagnosticWindow w; w.prepare(CarrierDiagnosticMode::batch, 4);
    for (unsigned a = 0; a < 4; ++a) {
        if (!w.begin(true)) return 1;
        for (unsigned b = 0; b < 8; ++b) w.batch(8192);
        w.end();
    }
    refuses([&] { w.begin(true); }); if (w.valid() || w.actions() != 4) return 1;
    { CarrierDiagnosticWindow exhausted; exhausted.prepare(CarrierDiagnosticMode::batch, 1); exhausted.begin(true);
      for (unsigned b = 0; b < 8; ++b) { exhausted.before_batch(); exhausted.batch(1); } refuses([&] { exhausted.before_batch(); }); }
    refuses([] { carrier_diagnostic_mode("enabled"); });
    for (uint32_t rows = 1; rows <= 7; ++rows) if (carrier_ordinary_history_steps(Json(true), rows) != rows) return 1;
    refuses([] { carrier_ordinary_history_steps(Json(true), 0); });
    refuses([] { carrier_ordinary_history_steps(Json(true), 8); });
    refuses([] { carrier_ordinary_history_steps(Json(false), 1); });
    for (uint32_t rows = 1; rows <= 8; ++rows) for (uint32_t accepted = 1; accepted <= rows; ++accepted)
        if (carrier_component_prefix(Json(Json::Object{{"accepted_prefix", Json::integer(accepted)}}), rows) != accepted) return 1;
    for (const auto input : {"{}", "{\"accepted_prefix\":0}", "{\"accepted_prefix\":3}", "{\"accepted_prefix\":1,\"force_decision\":true}"})
        refuses([&] { carrier_component_prefix(parse_json(input), 2); });
    // The native JSON encoder preserves the binary64 observations used by replay.
    for (const uint64_t bits : {0x3fd5555555555555ull, 0x40b123456789abcdull, 0x0010000000000000ull}) {
        const double value = std::bit_cast<double>(bits);
        const auto parsed = parse_json(Json::number(value).dump()); const auto lexeme = parsed.number_lexeme(); double decoded = 0;
        const auto result = std::from_chars(lexeme.data(), lexeme.data() + lexeme.size(), decoded);
        if (result.ec != std::errc{} || std::bit_cast<uint64_t>(decoded) != bits) return 1;
    }
    std::cout << "carrier diagnostic bounds, poisoning, v2 and timing round-trip passed\n";
}
