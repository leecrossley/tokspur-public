#pragma once
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace tokspur {
using RawDecodeRouteCounts = std::array<uint64_t, 2>;
inline constexpr std::array<std::string_view, 2> raw_decode_route_names{"q4_k", "q5_1"};
inline constexpr std::array<std::string_view, 9> raw_decode_identity_names{
    "native_raw_decode_schema", "native_raw_decode_version", "native_raw_decode_mode", "native_raw_decode_layout",
    "native_raw_decode_q4_kernel", "native_raw_decode_q51_kernel", "native_raw_decode_threads", "native_raw_decode_fp_contract",
    "native_raw_decode_counter_scope"};
inline const Json& raw_decode_field(const Json& value, std::string_view name) {
    const auto* field = value.find(name);
    if (!field) throw FormatError("missing raw decode metadata: " + std::string(name));
    return *field;
}
inline Json raw_decode_routes_json(const RawDecodeRouteCounts& routes) {
    return Json(Json::Object{{"q4_k", Json::integer(routes[0])}, {"q5_1", Json::integer(routes[1])}});
}
inline RawDecodeRouteCounts raw_decode_routes_value(const Json& value) {
    if (value.object().size() != 2) throw FormatError("raw decode route shape differs");
    return {raw_decode_field(value, "q4_k").unsigned_integer(), raw_decode_field(value, "q5_1").unsigned_integer()};
}
inline RawDecodeRouteCounts raw_decode_routes_delta(const RawDecodeRouteCounts& after, const RawDecodeRouteCounts& before) {
    RawDecodeRouteCounts result{};
    for (size_t i = 0; i < result.size(); ++i) {
        if (after[i] < before[i]) throw FormatError("raw decode completed routes decreased");
        result[i] = after[i] - before[i];
    }
    return result;
}
inline Json::Object raw_decode_identity_fields(bool enabled) {
    return {{"native_raw_decode_schema", Json::integer(1)}, {"native_raw_decode_version", Json("raw-routed-v1")},
        {"native_raw_decode_mode", Json(enabled ? "on" : "off")}, {"native_raw_decode_layout", Json(enabled ? "raw-gguf" : "disabled")},
        {"native_raw_decode_q4_kernel", Json(enabled ? "raw_decode_q4_k" : "disabled")},
        {"native_raw_decode_q51_kernel", Json(enabled ? "raw_decode_q5_1" : "disabled")},
        {"native_raw_decode_threads", Json::integer(enabled ? 64 : 0)}, {"native_raw_decode_fp_contract", Json(enabled ? "off" : "native-existing")},
        {"native_raw_decode_counter_scope", Json("successful-completed-physical-dispatches")}};
}
inline Json::Object raw_decode_metadata_fields(bool enabled, const RawDecodeRouteCounts& completed, const RawDecodeRouteCounts& request) {
    auto fields = raw_decode_identity_fields(enabled);
    fields.emplace_back("native_raw_decode_routes", raw_decode_routes_json(completed));
    fields.emplace_back("native_raw_decode_request_routes", raw_decode_routes_json(request));
    return fields;
}
inline std::optional<bool> raw_decode_selection(const Json& value, bool startup = false) {
    size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("native_raw_decode_")) {
        static_cast<void>(ignored);
        if (std::find(raw_decode_identity_names.begin(), raw_decode_identity_names.end(), name) == raw_decode_identity_names.end() &&
            name != "native_raw_decode_routes" && name != "native_raw_decode_request_routes") throw FormatError("unknown raw decode metadata");
        ++count;
    }
    if (!count) return std::nullopt;
#ifndef TOKSPUR_PRIVATE_RAW_DECODE
    static_cast<void>(startup);
    throw FormatError("raw decode metadata requires the private build");
#else
    if (count != 11) throw FormatError("incomplete raw decode metadata");
    const auto& mode = raw_decode_field(value, "native_raw_decode_mode").string();
    if (mode != "on" && mode != "off") throw FormatError("unknown raw decode mode");
    const bool enabled = mode == "on";
    for (const auto& [name, expected] : raw_decode_identity_fields(enabled))
        if (raw_decode_field(value, name).dump() != expected.dump()) throw FormatError("raw decode immutable identity differs");
    const auto routes = raw_decode_routes_value(raw_decode_field(value, "native_raw_decode_routes"));
    const auto request = raw_decode_routes_value(raw_decode_field(value, "native_raw_decode_request_routes"));
    for (size_t i = 0; i < routes.size(); ++i) if ((startup && (routes[i] || request[i])) || (!enabled && (routes[i] || request[i])) || request[i] > routes[i])
        throw FormatError("raw decode startup/off counts or request bounds differ");
    return enabled;
#endif
}
inline void validate_raw_decode_ready(const Json& value, std::optional<bool> expected) {
    if (raw_decode_selection(value, true) != expected) throw FormatError("raw decode metadata differs from startup consent");
}
inline void validate_raw_decode_previous(const Json& ready, const Json& previous) {
    const auto selected = raw_decode_selection(ready, true);
    const auto& event = raw_decode_field(previous, "event").string();
    if ((event != "ready" && event != "done") || raw_decode_selection(previous, event == "ready") != selected)
        throw FormatError("raw decode previous ancestry differs");
}
inline void validate_raw_decode_done(const Json& ready, const Json& previous, const Json& done, bool cold_resync = false) {
    validate_raw_decode_previous(ready, previous);
    const auto selected = raw_decode_selection(ready, true);
    if (raw_decode_selection(done) != selected) throw FormatError("raw decode DONE selection differs");
    if (!selected.has_value()) return;
    const auto before = raw_decode_routes_value(raw_decode_field(previous, "native_raw_decode_routes"));
    const auto after = raw_decode_routes_value(raw_decode_field(done, "native_raw_decode_routes"));
    const auto request = raw_decode_routes_value(raw_decode_field(done, "native_raw_decode_request_routes"));
    for (size_t i = 0; i < after.size(); ++i) if (after[i] < before[i] || (cold_resync ? request[i] > after[i] - before[i] : request[i] != after[i] - before[i]))
        throw FormatError("raw decode completed routes rewound or per-packet delta differs");
}
} // namespace tokspur
