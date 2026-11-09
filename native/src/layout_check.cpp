#include "tokspur/target.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/q8_layout.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t derived_budget = 32 * 1024 * 1024, gpu_budget = 64 * 1024 * 1024;
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
Json real(double v) {
    require(std::isfinite(v), "nonfinite layout diagnostic"); std::array<char,64> b{};
    const auto result = std::to_chars(b.data(), b.data() + b.size(), v); require(result.ec == std::errc{}, "layout number encoding");
    return parse_json({b.data(), size_t(result.ptr - b.data())});
}
double elapsed(Clock::time_point from) { return std::chrono::duration<double,std::milli>(Clock::now() - from).count(); }
const Json& field(const Json& object, std::string_view key) { const auto* p = object.find(key); require(p, "missing layout request field"); return *p; }
struct Probe { uint32_t c, r; GpuView weights{}, scales{}, values{}, f32{}, input{}, output{}; std::vector<std::byte> ordinary; };
void encode(MetalGraph& g, const Probe& p, std::string_view mode) {
    if (mode == "block") g.gemv(8, p.weights, p.input, p.output, p.c, p.r);
    else if (mode == "soa") g.q8_soa_gemv(p.scales, p.values, p.input, p.output, p.c, p.r);
    else if (mode == "f32") g.typed_f32_gemv(p.f32, p.input, p.output, p.c, p.r);
    else throw FormatError("unsupported layout mode");
}
bool line(std::string& text) {
    text.clear(); bool any = false, too_long = false; char c;
    while (std::cin.get(c)) { any = true; if (c == '\n') break; if (text.size() < 4096) text.push_back(c); else too_long = true; }
    require(!too_long, "layout request byte bound"); return any;
}
} // namespace
int layout_check_cli(int argc, char** argv) {
    std::vector<std::filesystem::path> paths; for (int i = 2; i < argc; ++i) paths.emplace_back(argv[i]);
    auto source = std::make_shared<ModelFiles>(paths); const auto started = Clock::now();
    MetalGraph g(std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib", gpu_budget);
    std::vector<std::string> names{"blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight"};
    for (uint32_t i = 1; i <= 6; ++i) names.push_back("blk." + std::to_string(i) + ".hc_attn_down.weight");
    std::vector<TensorSlice> slices;
    for (const auto& name : names) {
        const auto& t = source->tensor(name);
        const bool up = name == names[1];
        require(t.type == 8 && t.dimensions.size() == 2 && t.dimensions[0] == (up ? 320 : 10240) &&
            t.dimensions[1] == (up ? 10240 : 320), "layout probe requires declared Q8 projection geometry");
        slices.push_back({name, 0, t.bytes});
    }
    const auto imported = g.import_weights(source, slices);
    std::array<Probe,2> probes{}; uint64_t derived_bytes = 0; Json::Array conversions;
    for (uint32_t i = 0; i < 2; ++i) {
        const auto& t = source->tensor(names[i]); auto& p = probes[i]; p.c = uint32_t(t.dimensions[0]); p.r = uint32_t(t.dimensions[1]); p.weights = imported.tensors[i];
        const auto bytes = source->tensor_bytes(names[i]); const auto conversion_start = Clock::now();
        const auto soa = q8_soa(bytes, p.c, p.r, derived_budget - derived_bytes); derived_bytes += soa.scales.size() * 2 + soa.values.size();
        const auto expanded = q8_expand_f32(bytes, p.c, p.r, derived_budget - derived_bytes); derived_bytes += expanded.size() * 4;
        require(q8_reconstruct(soa, derived_budget) == std::vector<std::byte>(bytes.begin(), bytes.end()), "adopted SoA reconstruction differs");
        const auto conversion_ms = elapsed(conversion_start); const auto upload_start = Clock::now();
        p.scales = g.upload(std::as_bytes(std::span(soa.scales)), "derived half scales");
        p.values = g.upload(std::as_bytes(std::span(soa.values)), "derived signed bytes");
        p.f32 = g.upload(std::as_bytes(std::span(expanded)), "lossless expanded FP32");
        std::vector<float> x(p.c); for (uint32_t c = 0; c < p.c; ++c) x[c] = float(int((c * 17 + 3) % 257) - 128) / 4096;
        p.input = g.upload(std::as_bytes(std::span(x)), "layout input"); p.output = g.zeros(p.r * 4, "layout output");
        conversions.push_back(Json(Json::Object{{"name",Json(names[i])},{"columns",Json::integer(p.c)},
            {"rows",Json::integer(p.r)},{"source_bytes",Json::integer(bytes.size())},
            {"derived_payload_bytes",Json::integer(soa.scales.size() * 2 + soa.values.size() + expanded.size() * 4)},
            {"cpu_conversion_validation_ms",real(conversion_ms)},{"gpu_upload_setup_ms",real(elapsed(upload_start))}}));
        // All validation is outside measured requests, with fixed references.
        const auto reference = reference_gemv(8, bytes, p.c, p.r, x);
        for (uint32_t variant = 0; variant < 4; ++variant) {
            g.configure_block_gemv(variant == 0 ? 0 : 1); const auto allocations = g.memory().allocations;
            g.begin(); encode(g, p, variant < 2 ? "block" : variant == 2 ? "soa" : "f32"); const auto batch = g.submit();
            require(batch.dispatches == 1 && g.memory().allocations == allocations, "layout validation dispatch/allocation mismatch");
            const auto result = g.read(p.output); if (!variant) p.ordinary = result; else require(result == p.ordinary, "adopted lossless layout differs from ordinary GPU bits");
            for (uint32_t row = 0; row < p.r; ++row) {
                float value; std::memcpy(&value, result.data() + row * 4, 4);
                require(std::isfinite(value) && std::abs(value - reference[row]) / std::max(1.0,std::abs(reference[row])) <= 5e-5, "layout adopted scalar reference differs");
            }
        }
    }
    g.configure_block_gemv(1);
    const auto churn_output = g.zeros(320 * 4, "fixed rotation output"); source->assert_unchanged();
    const auto memory = g.memory();
    std::cout << Json(Json::Object{{"event",Json("ready")},{"scope",Json("two bounded projection layouts, not complete inference or a runtime speed qualification")},
        {"derived_payload_bytes",Json::integer(derived_bytes)},{"derived_payload_limit_bytes",Json::integer(derived_budget)},
        {"gpu_buffer_bytes",Json::integer(memory.live_bytes)},{"gpu_buffer_limit_bytes",Json::integer(gpu_budget)},
        {"imported_original_backing_bytes",Json::integer(memory.imported_backing_bytes)}, {"original_matrices",Json::integer(names.size())},
        {"derived_weight_files_written",Json(false)},{"all_validation_outputs_exact",Json(true)},{"setup_ms",real(elapsed(started))},
        {"conversions",Json(std::move(conversions))},{"rotation_scope",Json("six other original projections before each probe; not guaranteed LLC-cold or representative of all 194 matrices")}}).dump() << std::endl;
    std::string request;
    while (true) {
        try {
            if (!line(request)) break;
            const auto data = parse_json(request,{4096,64,4}); std::set<std::string> keys;
            for (const auto& [key,value] : data.object()) { static_cast<void>(value); require(key == "orientation" || key == "mode" || key == "rotating" || key == "cycles", "unknown layout request field"); keys.insert(key); }
            require(keys.size() == 4, "layout request fields required");
            const auto orientation = field(data,"orientation").string(), mode = field(data,"mode").string();
            require(orientation == "down" || orientation == "up", "unknown layout orientation");
            require(mode == "block" || mode == "soa" || mode == "f32", "unknown layout mode");
            const auto rotating = field(data,"rotating").boolean(); const auto cycles = field(data,"cycles").unsigned_integer();
            require(cycles && cycles <= 4096, "layout cycle bound"); const auto& p = probes[orientation == "down" ? 0 : 1];
            source->assert_unchanged(); const auto allocations = g.memory().allocations; const auto begin = Clock::now();
            try {
                g.begin();
                for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
                    if (rotating) for (uint32_t i = 2; i < imported.tensors.size(); ++i)
                        g.gemv(8, imported.tensors[i], probes[0].input, churn_output, 10240, 320);
                    encode(g, p, mode);
                }
                const auto batch = g.submit(); const auto wall_ms = elapsed(begin);
                require(batch.dispatches == cycles * (rotating ? 7 : 1) && g.memory().allocations == allocations, "layout measured dispatch/allocation invariant");
                require(g.read(p.output) == p.ordinary, "layout measured output drift"); source->assert_unchanged();
                std::cout << Json(Json::Object{{"event",Json("done")},{"orientation",Json(std::string(orientation))},{"mode",Json(std::string(mode))},
                    {"rotating",Json(rotating)},{"cycles",Json::integer(cycles)},{"dispatches",Json::integer(batch.dispatches)},
                    {"gpu_ms_per_cycle",real(batch.gpu_ms / cycles)},{"batch_wall_ms_per_cycle",real(batch.wall_ms / cycles)},
                    {"encode_submit_wall_ms_per_cycle",real(wall_ms / cycles)},{"outputs_exact",Json(true)}}).dump() << std::endl;
            } catch (...) { try { g.abort(); } catch (const FormatError&) {} throw; }
        } catch (const FormatError& e) { std::cout << Json(Json::Object{{"event",Json("error")},{"message",Json(e.what())}}).dump() << std::endl; if (!std::cin) break; }
    }
    return 0;
}
} // namespace tokspur
