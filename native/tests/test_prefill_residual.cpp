#include "tokspur/prefill_residual.hpp"
#include "tokspur/gguf.hpp"
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
#include "tokspur/grouped_mm.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/json.hpp"
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
#include "tokspur/metal.hpp"
#endif
#include <array>
#include <chrono>
#include <iomanip>
#include <sstream>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace tokspur;
uint64_t checks = 0;
double worst = 0;
void need(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& action) { ++checks; try { action(); } catch (const FormatError&) { return; } throw std::runtime_error("missing residual refusal"); }
struct Arena {
    MetalGraph& g; std::vector<GpuView> owned;
    GpuView floats(const std::vector<float>& x, bool immutable = true) { auto v = g.upload(std::as_bytes(std::span(x)), "generated residual FP32", immutable); owned.push_back(v); return v; }
    GpuView zero(uint64_t bytes) { auto v = g.zeros(bytes, "generated residual output"); owned.push_back(v); return v; }
    GpuMatrix q8(uint32_t c, uint32_t r) {
        std::vector<std::byte> b(uint64_t(c / 32) * r * 34);
        for (uint64_t i = 0; i < b.size() / 34; ++i) {
            const uint16_t scale = 0x1400; std::memcpy(b.data() + i * 34, &scale, 2);
            for (uint32_t j = 0; j < 32; ++j) { const int8_t q = int8_t(int((i * 17 + j * 11) % 15) - 7); std::memcpy(b.data() + i * 34 + 2 + j, &q, 1); }
        }
        const auto v = g.upload(b, "generated residual Q8", true); owned.push_back(v); return {v, 8};
    }
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (auto v : owned) g.release(v); }
};
std::vector<float> read(MetalGraph& g, GpuView v) { const auto b = g.read(v); std::vector<float> x(b.size() / 4); std::memcpy(x.data(), b.data(), b.size()); return x; }
void close(float x, float y) { need(std::isfinite(x) && std::isfinite(y), "nonfinite residual value"); const double e = std::abs(double(x) - y) / (1 + std::abs(double(y))); worst = std::max(worst, e); need(e <= 0.0005, "prefill residual component error"); }
void exercise(MetalGraph& g, uint32_t n, ResidualShape shape) {
    const auto initial = g.memory().live_bytes; const auto width = shape.branches * shape.channels; const auto routes = g.hc_exact_routes();
    {
        Arena a{g, {}}; std::vector<float> gains(width), inject(uint64_t(width) * shape.branches);
        for (size_t i = 0; i < gains.size(); ++i) gains[i] = 0.9f + float(i % 11) / 100;
        for (size_t i = 0; i < inject.size(); ++i) inject[i] = float(int(i % 17) - 8) / 32768;
        const ResidualWeights w{a.floats(gains), a.q8(width, shape.rank), a.q8(shape.rank, width), {a.floats(inject), 0}};
        prepare_prefill_residual(g, w, shape); auto candidate = allocate_prefill_residual_scratch(g, shape, 512); const auto serial = allocate_residual_scratch(g, shape);
        const uint64_t stride = uint64_t(width) * 4 + 16, out_stride = uint64_t(width) * 4 + 32;
        std::vector<float> x(stride / 4 * n, -7654.25f), destination(out_stride / 4 * n, -7654.25f);
        for (uint32_t t = 0; t < n; ++t) for (uint32_t j = 0; j < width; ++j) x[t * stride / 4 + j] = float(int((uint64_t(t) * 13 + j * 7) % 131) - 65) / 997;
        const auto xv = a.floats(x), output = a.floats(destination, false), values = a.zero(uint64_t(n) * shape.channels * 4);
        const GpuRowsF32 input{xv, n, width, stride}, out{output, n, width, out_stride};
        const auto allocations = g.memory().allocations; g.begin(); encode_prefill_residual(g, w, candidate, input, shape, 1e-6f);
        const auto batch = g.submit(); need(batch.dispatches == prefill_residual_dispatches(g, w, shape, n), "prefill residual dispatch count");
        need(g.memory().allocations == allocations, "prefill residual encoded an allocation");
        const auto actual = read(g, candidate.rows.input), scales = read(g, candidate.rows.scales), down = read(g, candidate.rows.bottleneck);
        for (uint32_t t = 0; t < n; ++t) {
            g.begin(); encode_residual_read(g, w, serial, g.slice(xv, t * stride, uint64_t(width) * 4), shape, 1e-6f); static_cast<void>(g.submit());
            const auto expected = read(g, serial.input), expected_scales = read(g, serial.scales), expected_down = read(g, serial.bottleneck);
            for (uint32_t j = 0; j < shape.channels; ++j) close(actual[uint64_t(t) * shape.channels + j], expected[j]);
            for (uint32_t j = 0; j < shape.branches; ++j) close(scales[uint64_t(t) * shape.branches + j], expected_scales[j]);
            for (uint32_t j = 0; j < shape.rank; ++j) need(down[uint64_t(t) * shape.rank + j] == expected_down[j], "HC down or its activation changed arithmetic");
        }
        g.begin(); encode_prefill_residual_write(g, candidate, input, {values, n, shape.channels, uint64_t(shape.channels) * 4}, out, shape);
        need(g.submit().dispatches == n, "prefill residual write dispatch count"); const auto copied = read(g, output);
        for (uint32_t t = 0; t < n; ++t) for (uint32_t j = 0; j < out_stride / 4; ++j)
            need(copied[t * out_stride / 4 + j] == (j < width ? x[t * stride / 4 + j] : -7654.25f), "residual write or strided padding changed");
        if (width == 10240 && shape.rank == 320) {
            const auto completed = g.hc_exact_routes(); need(completed[0] - routes[0] == 2 * n, "exact HC down route was shadowed");
            for (size_t i = 1; i < routes.size(); ++i) need(completed[i] == routes[i], "prefill residual changed exact HC many policy");
        }
        g.begin(); reject([&] { auto bad = input; bad.columns--; encode_prefill_residual(g, w, candidate, bad, shape, 1e-6f); });
        g.begin(); reject([&] { encode_prefill_residual_write(g, candidate, input, {values, n, shape.channels, uint64_t(shape.channels) * 4}, input, shape); });
        need(g.read(xv) == std::vector<std::byte>(std::as_bytes(std::span(x)).begin(), std::as_bytes(std::span(x)).end()), "immutable residual input changed");
        release_residual_scratch(g, serial); release_prefill_residual_scratch(g, candidate);
    }
    need(g.memory().live_bytes == initial, "residual component leaked buffers");
}

#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
struct HcDownResult {
    std::vector<float> input, scales, down, normalized, raw_down;
    std::vector<std::byte> down_weights;
    uint64_t dispatches = 0;
};
HcDownResult hc_down_case(const char* library, uint32_t n, ResidualShape shape, uint32_t down_type, bool enabled, bool staged = false) {
    MetalGraph g(library, 512ull << 20); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    need(!g.prefill_hc_down_f32(), "HC down batching default changed");
    if (enabled) reject([&] { g.configure_prefill_hc_down_f32(true); });
    g.configure_prefill_dense_f32(true);
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    if (staged) { g.configure_prefill_grouped_f32(true); g.configure_prefill_dense_staged_f32(true); }
#else
    need(!staged, "staged HC down component is not compiled");
#endif
    g.configure_prefill_hc_down_f32(enabled);
    need(g.prefill_hc_down_f32() == enabled, "HC down selector projection");
    if (enabled) reject([&] { g.configure_prefill_dense_f32(false); });
    const auto initial = g.memory().live_bytes; const auto width = shape.branches * shape.channels;
    const bool selected = enabled && (staged ? n >= 2 && n <= 512 : n == 512) && shape.branches == 4 && shape.channels == 2560 && shape.rank == 320 && down_type == 8;
    HcDownResult result;
    {
        Arena a{g, {}}; std::vector<float> gains(width, 1), inject(uint64_t(width) * shape.branches);
        for (size_t i = 0; i < inject.size(); ++i) inject[i] = float(int(i % 17) - 8) / 32768;
        GpuMatrix down;
        if (down_type == 8) down = a.q8(width, shape.rank);
        else {
            need(down_type == 0, "generated non-Q8 down fixture encoding");
            std::vector<float> values(uint64_t(width) * shape.rank);
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 15) - 7) / 1024;
            down = {a.floats(values), 0};
        }
        const ResidualWeights w{a.floats(gains), down, a.q8(shape.rank, width), {a.floats(inject), 0}};
        result.down_weights = g.read(w.down.view);
        prepare_prefill_residual(g, w, shape);
        reject([&] { g.configure_prefill_hc_down_f32(!enabled); });
        auto scratch = allocate_prefill_residual_scratch(g, shape, 512);
        const uint64_t stride = uint64_t(width) * 4 + 16;
        std::vector<float> values(stride / 4 * n, -7654.25f);
        for (uint32_t t = 0; t < n; ++t) for (uint32_t j = 0; j < width; ++j) values[t * stride / 4 + j] = float(int((uint64_t(t) * 13 + j * 7) % 131) - 65) / 997;
        const auto input = a.floats(values);
        const auto raw_down = selected ? a.zero(uint64_t(n) * shape.rank * 4) : GpuView{};
        const auto allocations = g.memory().allocations; const auto routes = g.hc_exact_routes();
        const auto encoded_before = g.prefill_hc_down_f32_encoded_calls();
        g.begin(); encode_prefill_residual(g, w, scratch, {input, n, width, stride}, shape, 1e-6f);
        result.dispatches = g.submit().dispatches;
        need(g.prefill_hc_down_f32_encoded_calls() - encoded_before == uint64_t(selected), "HC down encoded call count differs from actual selected wrapper");
        need(result.dispatches == prefill_residual_dispatches(g, w, shape, n), "HC down candidate dispatch predictor mismatch");
        need(g.memory().allocations == allocations, "HC down candidate allocated while encoding");
        const auto after = g.hc_exact_routes();
        const uint64_t expected_exact = width == 10240 && shape.rank == 320 && down_type == 8 && !selected ? n : 0;
        need(after[0] - routes[0] == expected_exact, "HC down candidate/fallback exact route count");
        for (size_t i = 1; i < routes.size(); ++i) need(after[i] == routes[i], "HC down candidate changed many-row exact routes");
        result.input = read(g, scratch.rows.input); result.scales = read(g, scratch.rows.scales);
        result.down = read(g, scratch.rows.bottleneck); result.normalized = read(g, scratch.rows.normalized);
        for (size_t i = uint64_t(n) * shape.channels; i < result.input.size(); ++i) need(result.input[i] == 0, "HC down input scratch tail changed");
        for (size_t i = uint64_t(n) * shape.branches; i < result.scales.size(); ++i) need(result.scales[i] == 0, "HC down scale scratch tail changed");
        for (size_t i = uint64_t(n) * shape.rank; i < result.down.size(); ++i) need(result.down[i] == 0, "HC down bottleneck scratch tail changed");
        if (selected) {
            g.begin(); g.dense_mm_q8(PrefillWeightLayout::gguf, w.down.view,
                {scratch.rows.normalized, n, width, uint64_t(width) * 4}, {raw_down, n, shape.rank, uint64_t(shape.rank) * 4});
            need(g.submit().dispatches == 1, "HC down independent projection dispatch count"); result.raw_down = read(g, raw_down);
            need(g.prefill_hc_down_f32_encoded_calls() == encoded_before + 1, "generic dense reference inflated HC down counter");
            g.begin(); g.dense_hc_down_f32(w.down.view, {scratch.rows.normalized, n, width, uint64_t(width) * 4}, {raw_down, n, shape.rank, uint64_t(shape.rank) * 4});
            need(g.prefill_hc_down_f32_encoded_calls() == encoded_before + 2, "actual encoded HC down wrapper was not counted"); g.abort();
            need(g.prefill_hc_down_f32_encoded_calls() == encoded_before + 2, "HC down encoded lifetime rewound on abort");
        }
        need(g.read(w.down.view) == result.down_weights && g.read(input) == std::vector<std::byte>(std::as_bytes(std::span(values)).begin(), std::as_bytes(std::span(values)).end()), "HC down candidate changed represented weights or input");
        release_prefill_residual_scratch(g, scratch);
    }
    need(g.memory().live_bytes == initial, "HC down candidate leaked buffers"); return result;
}
void hc_down_pair(const char* library, uint32_t n, ResidualShape shape, uint32_t type, bool staged = false) {
    const auto ordinary = hc_down_case(library, n, shape, type, false, staged), candidate = hc_down_case(library, n, shape, type, true, staged);
    const bool selected = (staged ? n >= 2 && n <= 512 : n == 512) && shape.branches == 4 && shape.channels == 2560 && shape.rank == 320 && type == 8;
    if (!selected) {
        need(ordinary.dispatches == candidate.dispatches && ordinary.input == candidate.input && ordinary.scales == candidate.scales && ordinary.down == candidate.down && ordinary.normalized == candidate.normalized,
            "HC down singleton, tail, non-Q8 or geometry fallback changed"); return;
    }
    need(ordinary.dispatches == candidate.dispatches + n - 1, "HC down full chunk did not replace per-token dispatches");
    double square = 0, signal = 0;
    for (size_t i = 0; i < candidate.down.size(); ++i) {
        need(std::isfinite(candidate.down[i]) && std::isfinite(ordinary.down[i]), "nonfinite experimental HC down activation");
        const double d = double(candidate.down[i]) - ordinary.down[i]; square += d * d; signal += double(ordinary.down[i]) * ordinary.down[i];
    }
    need((signal ? square / signal : square) <= 1e-7, "experimental HC down activated FP32 NMSE screen");
    for (size_t i = 0; i < candidate.input.size(); ++i) close(candidate.input[i], ordinary.input[i]);
    for (size_t i = 0; i < candidate.scales.size(); ++i) close(candidate.scales[i], ordinary.scales[i]);
    double reference_square = 0, reference_signal = 0;
    const uint32_t width = shape.branches * shape.channels;
    for (uint32_t t : {0u, 1u, 15u, 16u, 17u, n / 2, n - 1}) if (t < n) for (uint32_t r : {0u, 1u, 15u, 16u, 17u, shape.rank / 2, shape.rank - 1}) {
        const uint64_t row_bytes = uint64_t(width / 32) * 34;
        const auto expected = reference_gemv(8, std::span(candidate.down_weights).subspan(uint64_t(r) * row_bytes, row_bytes), width, 1,
            std::span(candidate.normalized).subspan(uint64_t(t) * width, width))[0];
        const auto actual = candidate.raw_down[uint64_t(t) * shape.rank + r]; need(std::isfinite(actual), "nonfinite experimental HC down projection");
        const double d = double(actual) - expected; reference_square += d * d; reference_signal += expected * expected;
    }
    need((reference_signal ? reference_square / reference_signal : reference_square) <= 1e-7, "experimental HC down sampled independent FP64 NMSE screen");
    std::cout << "Experimental HC down n=" << n << " kernel=" << (staged ? "graph_dense_q8_staged_f32" : "graph_q8_mpp_wide") << " ordinary_dispatches=" << ordinary.dispatches << " candidate_dispatches=" << candidate.dispatches
        << " activated_nmse=" << (signal ? square / signal : square) << " sampled_fp64_nmse=" << (reference_signal ? reference_square / reference_signal : reference_square)
        << " no model/state/fidelity or latency admission\n";
}
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
struct HcCostTrial {
    uint32_t arm = 0, cycle = 0, order = 0;
    bool measured = false;
    uint64_t encode_ns = 0, submit_wait_ns = 0, whole_ns = 0, barriers = 0;
    HcExactRoutes successful_routes{};
    BatchResult batch;
};
const char* hc_cost_arm(uint32_t arm) { static constexpr std::array names{"scalar-hc-exact", "many-hc-exact", "dense-f32-single-wide", "dense-staged-f32-4sg", "dense-staged-half-operands-4sg"}; return names.at(arm); }
Json hc_cost_number(double value) { if (!std::isfinite(value)) return Json{}; std::ostringstream out; out << std::setprecision(17) << value; return parse_json(out.str()); }
std::string hc_cost_raw_number(double value) { std::ostringstream out; out << std::setprecision(17) << value; return out.str(); }
Json hc_cost_routes(HcExactRoutes routes) { Json::Array out; for (auto value : routes) out.push_back(Json::integer(value)); return Json(std::move(out)); }
void hc_cost_emit(Json::Object fields) {
    static uint64_t bytes = 0; const auto line = Json(std::move(fields)).dump(4096); bytes += line.size() + 1;
    need(bytes <= (1u << 20), "HC down cost output exceeded its fixed 1 MiB bound"); std::cout << line << '\n';
}
double hc_cost_median(const std::vector<HcCostTrial>& trials, uint32_t arm, const std::function<double(const HcCostTrial&)>& metric) {
    std::vector<double> values; for (const auto& t : trials) if (t.measured && t.arm == arm) values.push_back(metric(t));
    need(values.size() == 10, "HC down cost measured repetition count"); std::sort(values.begin(), values.end()); return (values[4] + values[5]) / 2;
}
// Actual retained chunked-prefill dense Q8 shapes. K/R are input/output columns, independent of active token rows.
struct DenseShapeCost { uint32_t columns, rows; const char* uses; };
constexpr std::array<DenseShapeCost, 8> dense_shape_costs{{
    {320, 10240, "HC up"}, {2560, 640, "shared FFN gate/up"}, {640, 2560, "shared FFN down"},
    {2560, 12288, "QSA packed query/gate"}, {2560, 512, "QSA key/value"},
    {6144, 2560, "QSA/GDN output"}, {2560, 10240, "GDN QKV"}, {2560, 6144, "GDN gate"}
}};
const char* dense_shape_arm(uint32_t arm) {
    static constexpr std::array names{"dense-old-mpp-single-wide", "dense-staged-fp32-4sg", "dense-explicit-half-operands-4sg"}; return names.at(arm);
}
double dense_shape_median(const std::vector<HcCostTrial>& trials, uint32_t arm, const std::function<double(const HcCostTrial&)>& metric) {
    std::vector<double> values; for (const auto& t : trials) if (t.measured && t.arm == arm) values.push_back(metric(t));
    need(values.size() == 6, "dense shape measured repetition count"); std::sort(values.begin(), values.end()); return (values[2] + values[3]) / 2;
}
void dense_q8_shape_cost_screen(const char* library) {
    constexpr uint32_t maximum_n = 512, guard = 32, warmup_cycles = 2, measured_cycles = 6;
    constexpr float sentinel = -7654.25f;
    constexpr std::array<std::array<uint32_t, 3>, 6> orders{{{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}}};
    need(host_thermal_state() == 0, "dense shape cost screen requires a coordinated nominal entry");
    MetalGraph g(library, 512ull << 20); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    g.configure_wide_q8_exact(false); g.configure_wide_q8_many(false); g.configure_concurrent_dispatch(true); g.configure_prefill_dense_f32(true);
    g.prepare_q8_mpp(Q8MppLayout::single_wide); g.prepare_dense_q8_staged(false); g.prepare_dense_q8_staged(true);
    need(!g.prefill_hc_down_f32() && g.prefill_hc_down_f32_encoded_calls() == 0, "dense shape cost must omit the experimental HC wrapper");
    const auto initial = g.memory().live_bytes; Json::Array shapes;
    for (const auto& shape : dense_shape_costs) shapes.push_back(Json(Json::Object{{"columns", Json::integer(shape.columns)}, {"rows", Json::integer(shape.rows)}, {"uses", Json(shape.uses)}}));
    hc_cost_emit({{"schema", Json::integer(1)}, {"event", Json("dense_q8_shape_cost_settings")}, {"shapes", Json(std::move(shapes))}, {"token_rows", Json(Json::Array{Json::integer(17), Json::integer(363), Json::integer(512)})}, {"input_profiles", Json(Json::Array{Json("mixed"), Json("cancellation-pairs-with-residual")})}, {"weight_layout", Json("raw-gguf-q8-0-generated")}, {"dense_precision", Json("f32")}, {"old_kernel", Json("graph_q8_mpp_wide")}, {"old_layout", Json("single_wide")}, {"old_simdgroups", Json::integer(1)}, {"staged_fp32_kernel", Json("graph_dense_q8_staged_f32")}, {"explicit_half_kernel", Json("graph_dense_mm_q8")}, {"staged_simdgroups", Json::integer(4)}, {"staged_fp32_threadgroup_bytes", Json::integer(12288)}, {"explicit_half_threadgroup_bytes", Json::integer(8192)}, {"kernel_identity_scope", Json("forced direct Graph entry points; expected one physical dispatch per arm, no compiled load/spill claim")}, {"warmup_cycles", Json::integer(warmup_cycles)}, {"measured_cycles", Json::integer(measured_cycles)}, {"balanced_order", Json("all six permutations; each arm twice in each position")}, {"full_fp64_reference", Json("every output checked; repeated raw weight rows verified bytewise before caching their independent FP64 dots")}, {"fp32_nmse_limit", hc_cost_number(1e-7)}, {"half_quality_admitted", Json(false)}, {"timing_scope", Json("prepared direct projection; begin/encode and submit/wait API walls separate; BatchResult command-buffer GPU duration, raw timestamps/calibration unavailable")}, {"thermal_before", Json::integer(host_thermal_state())}, {"performance_qualified", Json(false)}});
    uint32_t case_count = 0;
    for (uint32_t shape_index = 0; shape_index < dense_shape_costs.size(); ++shape_index) {
        const auto shape = dense_shape_costs[shape_index]; const auto c = shape.columns, rows = shape.rows;
        {
            Arena a{g, {}}; const auto matrix = a.q8(c, rows); const auto weight_bytes = g.read(matrix.view); const uint64_t row_bytes = uint64_t(c / 32) * 34;
            // The generated Q8 rows repeat in at most15 classes. Verify every raw row instead of assuming this property or sampling output rows.
            uint32_t classes = 1; while (classes <= 15 && (uint64_t(classes) * (c / 32) * 17) % 15) ++classes;
            need(classes <= 15 && classes <= rows, "dense shape generated reference row class bound");
            for (uint32_t row = 0; row < rows; ++row) need(!std::memcmp(weight_bytes.data() + uint64_t(row) * row_bytes, weight_bytes.data() + uint64_t(row % classes) * row_bytes, row_bytes), "dense shape cached FP64 row class bytes differ");
            for (uint32_t row = 0; row < classes; ++row) for (uint32_t block = 0; block < c / 32; ++block) {
                const auto* quant = weight_bytes.data() + (uint64_t(row) * (c / 32) + block) * 34 + 2;
                need(!std::memcmp(quant, quant + 15, 15), "dense shape cancellation pairs require equal represented quant bytes");
            }
            std::vector<float> values(uint64_t(maximum_n) * c); const auto input = a.floats(values, false);
            const std::vector<float> blank(uint64_t(maximum_n) * rows + guard, sentinel);
            const std::array outputs{a.floats(blank, false), a.floats(blank, false), a.floats(blank, false)};
            const auto prepared = g.memory();
            for (uint32_t input_profile = 0; input_profile < 2; ++input_profile) {
                for (uint32_t t = 0; t < maximum_n; ++t) {
                    for (uint32_t j = 0; j < c; ++j) values[uint64_t(t) * c + j] = float(int((uint64_t(t) * 13 + j * 7) % 131) - 65) / 997;
                    if (input_profile) for (uint32_t block = 0; block < c / 32; ++block) {
                        const uint64_t base = uint64_t(t) * c + uint64_t(block) * 32;
                        for (uint32_t j = 0; j < 15; ++j) { const float v = float(int((uint64_t(t) * 13 + block * 7 + j * 5) % 131) - 65) / 997; values[base + j] = v; values[base + j + 15] = -v; }
                        // Equal represented Q8 weights at j and j+15 cancel. Keep a finite residual signal rather than normalising a deliberately all-zero reference.
                        values[base + 30] = float(int((t + block) % 17) - 8) / 97; values[base + 31] = -values[base + 30] * (1.0f + 1.0f / 1024);
                    }
                }
                const std::vector<std::byte> input_bytes(std::as_bytes(std::span(values)).begin(), std::as_bytes(std::span(values)).end()); g.write(input, input_bytes);
                std::vector<double> reference(uint64_t(maximum_n) * classes);
                for (uint32_t t = 0; t < maximum_n; ++t) for (uint32_t row = 0; row < classes; ++row)
                    reference[uint64_t(t) * classes + row] = reference_gemv(8, std::span(weight_bytes).subspan(uint64_t(row) * row_bytes, row_bytes), c, 1, std::span(values).subspan(uint64_t(t) * c, c))[0];
                for (uint32_t n : {17u, 363u, 512u}) {
                    const auto thermal_before = host_thermal_state(); for (auto output : outputs) g.write(output, std::as_bytes(std::span(blank)));
                    const auto active_input = g.slice(input, 0, uint64_t(n) * c * 4);
                    std::array<GpuView, 3> active_outputs; for (uint32_t arm = 0; arm < 3; ++arm) active_outputs[arm] = g.slice(outputs[arm], 0, uint64_t(n) * rows * 4);
                    auto launch = [&](uint32_t arm, bool measured, uint32_t cycle, uint32_t order) {
                        const auto start = std::chrono::steady_clock::now(); g.begin();
                        if (arm == 0) g.q8_mpp(matrix.view, active_input, active_outputs[arm], c, rows, n, Q8MppLayout::single_wide);
                        else g.dense_q8_staged(matrix.view, active_input, active_outputs[arm], c, rows, n, arm == 2);
                        const auto encoded = std::chrono::steady_clock::now(); const auto batch = g.submit(); const auto end = std::chrono::steady_clock::now();
                        const auto ns = [](auto from, auto to) { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()); };
                        HcCostTrial trial{arm, cycle, order, measured, ns(start, encoded), ns(encoded, end), ns(start, end), g.concurrent_barriers(), {}, batch};
                        need(batch.dispatches == 1 && batch.capture_executes == 0, "dense shape arm must encode exactly one direct physical dispatch");
                        need(g.prefill_hc_down_f32_encoded_calls() == 0 && g.hc_exact_routes() == HcExactRoutes{}, "dense shape arm changed HC wrapper or exact route counters");
                        need(g.memory().allocations == prepared.allocations && g.memory().live_bytes == prepared.live_bytes, "dense shape screen allocated during encode or submit");
                        need(std::isfinite(batch.wall_ms) && batch.wall_ms > 0 && trial.encode_ns && trial.submit_wait_ns && trial.whole_ns == trial.encode_ns + trial.submit_wait_ns, "dense shape host clock invalid"); return trial;
                    };
                    auto validate = [&] {
                        const std::array actual{read(g, outputs[0]), read(g, outputs[1]), read(g, outputs[2])};
                        std::array<double, 3> square{}, maximum{}; double signal = 0, staged_old_square = 0, half_old_square = 0, old_signal = 0;
                        for (uint32_t arm = 0; arm < 3; ++arm) need(actual[arm].size() == blank.size(), "dense shape output allocation extent");
                        for (uint64_t at = 0; at < blank.size(); ++at) {
                            for (uint32_t arm = 0; arm < 3; ++arm) need(std::isfinite(actual[arm][at]), "dense shape nonfinite output");
                            if (at >= uint64_t(n) * rows) { for (uint32_t arm = 0; arm < 3; ++arm) need(actual[arm][at] == sentinel, "dense shape inactive row or output guard changed"); continue; }
                            const auto t = at / rows, row = at % rows; const double dot = reference[t * classes + row % classes]; need(std::isfinite(dot), "dense shape nonfinite FP64 reference"); signal += dot * dot;
                            for (uint32_t arm = 0; arm < 3; ++arm) { const double delta = double(actual[arm][at]) - dot; square[arm] += delta * delta; maximum[arm] = std::max(maximum[arm], std::abs(delta)); }
                            const double staged_delta = double(actual[1][at]) - actual[0][at], half_delta = double(actual[2][at]) - actual[0][at]; staged_old_square += staged_delta * staged_delta; half_old_square += half_delta * half_delta; old_signal += double(actual[0][at]) * actual[0][at];
                        }
                        std::array<double, 3> nmse{}; for (uint32_t arm = 0; arm < 3; ++arm) nmse[arm] = signal ? square[arm] / signal : square[arm];
                        const double staged_old = old_signal ? staged_old_square / old_signal : staged_old_square, half_old = old_signal ? half_old_square / old_signal : half_old_square;
                        need(nmse[0] <= 1e-7 && nmse[1] <= 1e-7 && staged_old <= 1e-7, "dense shape unchanged full-output FP32 NMSE screen failed");
                        need(g.read(matrix.view) == weight_bytes && g.read(input) == input_bytes, "dense shape weights or input bytes changed");
                        return std::array{nmse[0], nmse[1], nmse[2], staged_old, half_old, maximum[0], maximum[1], maximum[2]};
                    };
                    for (uint32_t arm = 0; arm < 3; ++arm) static_cast<void>(launch(arm, false, 0, arm)); static_cast<void>(validate());
                    std::vector<HcCostTrial> trials; trials.reserve((warmup_cycles + measured_cycles) * 3);
                    for (uint32_t phase = 0; phase < 2; ++phase) for (uint32_t cycle = 0; cycle < (phase ? measured_cycles : warmup_cycles); ++cycle)
                        for (uint32_t order = 0; order < 3; ++order) trials.push_back(launch(orders[cycle][order], phase != 0, cycle, order));
                    const auto validated = validate();
                    for (const auto& t : trials) hc_cost_emit({{"schema", Json::integer(1)}, {"event", Json("dense_q8_shape_cost_trial")}, {"shape", Json::integer(shape_index)}, {"n", Json::integer(n)}, {"input_profile", Json::integer(input_profile)}, {"arm", Json(dense_shape_arm(t.arm))}, {"phase", Json(t.measured ? "measured" : "warmup")}, {"cycle", Json::integer(t.cycle)}, {"order", Json::integer(t.order)}, {"encode_api_wall_ns", Json::integer(t.encode_ns)}, {"submit_wait_api_wall_ns", Json::integer(t.submit_wait_ns)}, {"whole_api_wall_ns", Json::integer(t.whole_ns)}, {"batch_wall_ms", hc_cost_number(t.batch.wall_ms)}, {"batch_gpu_ms", std::isfinite(t.batch.gpu_ms) && t.batch.gpu_ms > 0 ? hc_cost_number(t.batch.gpu_ms) : Json{}}, {"batch_gpu_ms_raw_lexeme", Json(hc_cost_raw_number(t.batch.gpu_ms))}, {"gpu_positive", Json(std::isfinite(t.batch.gpu_ms) && t.batch.gpu_ms > 0)}, {"dispatches", Json::integer(t.batch.dispatches)}, {"concurrent_barriers", Json::integer(t.barriers)}});
                    Json::Array arms;
                    for (uint32_t arm = 0; arm < 3; ++arm) {
                        const bool gpu_observed = std::all_of(trials.begin(), trials.end(), [&](const auto& t) { return !t.measured || t.arm != arm || (std::isfinite(t.batch.gpu_ms) && t.batch.gpu_ms > 0); });
                        arms.push_back(Json(Json::Object{{"arm", Json(dense_shape_arm(arm))}, {"kernel", Json(arm == 0 ? "graph_q8_mpp_wide" : arm == 1 ? "graph_dense_q8_staged_f32" : "graph_dense_mm_q8")}, {"threads_per_group", Json::integer(arm == 0 ? 32 : 128)}, {"grid_x", Json::integer((rows + (arm == 0 ? 31 : 63)) / (arm == 0 ? 32 : 64))}, {"grid_y", Json::integer((n + 31) / 32)}, {"full_output_fp64_nmse", hc_cost_number(validated[arm])}, {"maximum_abs_fp64_error", hc_cost_number(validated[5 + arm])}, {"median_encode_api_wall_ns", hc_cost_number(dense_shape_median(trials, arm, [](const auto& t) { return double(t.encode_ns); }))}, {"median_submit_wait_api_wall_ns", hc_cost_number(dense_shape_median(trials, arm, [](const auto& t) { return double(t.submit_wait_ns); }))}, {"median_whole_api_wall_ns", hc_cost_number(dense_shape_median(trials, arm, [](const auto& t) { return double(t.whole_ns); }))}, {"median_batch_wall_ms", hc_cost_number(dense_shape_median(trials, arm, [](const auto& t) { return t.batch.wall_ms; }))}, {"all_gpu_durations_positive", Json(gpu_observed)}, {"median_batch_gpu_ms", gpu_observed ? hc_cost_number(dense_shape_median(trials, arm, [](const auto& t) { return t.batch.gpu_ms; })) : Json{}}, {"quality_admitted", Json(false)}}));
                    }
                    hc_cost_emit({{"schema", Json::integer(1)}, {"event", Json("dense_q8_shape_cost_case")}, {"shape", Json::integer(shape_index)}, {"columns", Json::integer(c)}, {"rows", Json::integer(rows)}, {"n", Json::integer(n)}, {"input_profile", Json::integer(input_profile)}, {"fp64_outputs_checked", Json::integer(uint64_t(n) * rows)}, {"byte_verified_reference_row_classes", Json::integer(classes)}, {"full_fp64_nmse_old_mpp", hc_cost_number(validated[0])}, {"full_fp64_nmse_staged_fp32", hc_cost_number(validated[1])}, {"full_fp64_nmse_explicit_half", hc_cost_number(validated[2])}, {"staged_fp32_vs_old_full_nmse", hc_cost_number(validated[3])}, {"explicit_half_vs_old_full_nmse", hc_cost_number(validated[4])}, {"explicit_half_within_fp32_screen", Json(validated[2] <= 1e-7 && validated[4] <= 1e-7)}, {"explicit_half_quality_admitted", Json(false)}, {"arms", Json(std::move(arms))}, {"thermal_before", Json::integer(thermal_before)}, {"thermal_after", Json::integer(host_thermal_state())}, {"fp32_nmse_screen_passed", Json(true)}, {"performance_qualified", Json(false)}}); ++case_count;
                }
            }
        }
        need(g.memory().live_bytes == initial, "dense shape screen leaked generated buffers");
    }
    need(case_count == 48, "dense shape screen case inventory changed");
    hc_cost_emit({{"schema", Json::integer(1)}, {"event", Json("dense_q8_shape_cost_complete")}, {"cases", Json::integer(case_count)}, {"checks", Json::integer(checks)}, {"passed", Json(true)}, {"fp32_thresholds_unchanged", Json(true)}, {"half_quality_admitted", Json(false)}, {"thermal_after", Json::integer(host_thermal_state())}, {"performance_qualified", Json(false)}, {"scope", Json("generated dense shape component only; no model load, state, logits, fidelity, serving or release qualification")}});
}
void hc_down_cost_screen(const char* library) {
    constexpr uint32_t c = 10240, r = 320, maximum_n = 512, guard = 32, warmup_cycles = 2, measured_cycles = 10;
    constexpr float sentinel = -7654.25f;
    constexpr std::array<std::array<uint32_t, 5>, 10> orders{{{0, 1, 2, 3, 4}, {1, 2, 3, 4, 0}, {2, 3, 4, 0, 1}, {3, 4, 0, 1, 2}, {4, 0, 1, 2, 3}, {0, 4, 3, 2, 1}, {1, 0, 4, 3, 2}, {2, 1, 0, 4, 3}, {3, 2, 1, 0, 4}, {4, 3, 2, 1, 0}}};
    MetalGraph g(library, 512ull << 20); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    g.configure_wide_q8_exact(false); g.configure_wide_q8_many(false); g.configure_concurrent_dispatch(true); g.configure_prefill_dense_f32(true);
    need(g.concurrent_dispatch() && g.prefill_dense_f32() && !g.prefill_hc_down_f32(), "HC down cost settings or wrapper omission changed");
    need(!g.wide_q8_exact_enabled() && !g.wide_q8_many_enabled() && g.hc_exact_mode() == HcExactMode::combined && !g.hc_q8_policy().splits, "HC down cost scalar route differs from retained settings");
    need(g.gemv_dispatches(8, c, r) == 1, "HC down cost scalar physical dispatch predictor changed");
    for (uint32_t width : {1u, 2u, 4u, 8u}) g.prepare_q8_matvec_many(width);
    g.prepare_dense_mm_q8(PrefillWeightLayout::gguf); g.prepare_dense_q8_staged(false); g.prepare_dense_q8_staged(true); const auto initial = g.memory().live_bytes;
    hc_cost_emit({{"schema", Json::integer(3)}, {"event", Json("hc_down_cost_settings")}, {"columns", Json::integer(c)}, {"rows", Json::integer(r)}, {"maximum_n", Json::integer(maximum_n)}, {"small_gemv_splits", Json::integer(16)}, {"block_gemv_formats", Json::integer(3)}, {"hc_exact", Json("combined")}, {"wide_q8_exact", Json(false)}, {"wide_q8_many", Json(false)}, {"concurrent_dispatch", Json(true)}, {"dense_precision", Json("f32-reassociated")}, {"scalar_kernel", Json("graph_hc_q8_exact_r1_u16")}, {"many_kernels", Json(Json::Array{Json("graph_hc_exact_many_1"), Json("graph_hc_exact_many_2"), Json("graph_hc_exact_many_4"), Json("graph_hc_exact_many_8")})}, {"dense_kernel", Json("graph_q8_mpp_wide")}, {"dense_layout", Json("single_wide")}, {"staged_fp32_kernel", Json("graph_dense_q8_staged_f32")}, {"staged_half_comparison_kernel", Json("graph_dense_mm_q8")}, {"staged_fp32_operands", Json("float threadgroup operands and float cooperative accumulators")}, {"staged_half_comparison_operands", Json("half threadgroup operands and float cooperative accumulators; independent approximate comparison")}, {"staged_threads", Json::integer(128)}, {"staged_simdgroups", Json::integer(4)}, {"staged_fp32_threadgroup_bytes", Json::integer(12288)}, {"staged_half_threadgroup_bytes", Json::integer(8192)}, {"weight_layout", Json("raw-gguf-q8-0")}, {"wrapper_enabled", Json(false)}, {"warmup_cycles_per_n", Json::integer(warmup_cycles)}, {"measured_cycles_per_n", Json::integer(measured_cycles)}, {"measured_order", Json("5 cyclic forward and 5 cyclic reverse orders; each arm twice in every position")}, {"route_bucket_order", Json("scalar,many1,many2,many4,many8")}, {"timing_scope", Json("prepared direct projection only; one command buffer per arm; begin plus encode and submit plus wait measured separately")}, {"gpu_clock_scope", Json("BatchResult command-buffer GPUEndTime minus GPUStartTime; raw timestamps and calibration unavailable")}, {"partial_scope", Json("n below 512 exercises prospective direct primitives; current serving wrapper still requires 512")}, {"performance_qualified", Json(false)}});
    {
        Arena a{g, {}}; const auto matrix = a.q8(c, r); const auto weight_bytes = g.read(matrix.view);
        std::vector<float> values(uint64_t(maximum_n) * c);
        for (uint32_t t = 0; t < maximum_n; ++t) for (uint32_t j = 0; j < c; ++j) values[uint64_t(t) * c + j] = float(int((uint64_t(t) * 13 + j * 7) % 131) - 65) / 997;
        const auto input = a.floats(values); const std::vector<std::byte> input_bytes(std::as_bytes(std::span(values)).begin(), std::as_bytes(std::span(values)).end());
        const std::vector<float> blank(uint64_t(maximum_n) * r + guard, sentinel);
        const std::array outputs{a.floats(blank, false), a.floats(blank, false), a.floats(blank, false), a.floats(blank, false), a.floats(blank, false)};
        std::array<GpuView, maximum_n> input_rows, output_rows;
        for (uint32_t t = 0; t < maximum_n; ++t) { input_rows[t] = g.slice(input, uint64_t(t) * c * 4, uint64_t(c) * 4); output_rows[t] = g.slice(outputs[0], uint64_t(t) * r * 4, uint64_t(r) * 4); }
        const auto prepared = g.memory();
        for (uint32_t n : {2u, 3u, 16u, 32u, 37u, 64u, 128u, 256u, 363u, 511u, 512u}) {
            const GpuRowsF32 x{g.slice(input, 0, uint64_t(n) * c * 4), n, c, uint64_t(c) * 4};
            std::array<GpuView, 5> active_outputs; for (uint32_t arm = 0; arm < 5; ++arm) active_outputs[arm] = g.slice(outputs[arm], 0, uint64_t(n) * r * 4);
            const GpuRowsF32 y{active_outputs[2], n, r, uint64_t(r) * 4};
            struct Chunk { GpuView input, output; uint32_t width; }; std::vector<Chunk> chunks; HcExactRoutes many_routes{};
            for (uint32_t at = 0; at < n;) {
                uint32_t width = 8; while (width > n - at) width >>= 1;
                chunks.push_back({g.slice(input, uint64_t(at) * c * 4, uint64_t(width) * c * 4), g.slice(outputs[1], uint64_t(at) * r * 4, uint64_t(width) * r * 4), width});
                ++many_routes[width == 1 ? 1 : width == 2 ? 2 : width == 4 ? 3 : 4]; at += width;
            }
            need(n != 512 || (chunks.size() == 64 && many_routes == HcExactRoutes{0, 0, 0, 0, 64}), "HC down full512 many arm must select exactly 64 r8 calls");
            auto launch = [&](uint32_t arm, bool measured, uint32_t cycle, uint32_t order) {
                const auto routes = g.hc_exact_routes(); const auto encoded = g.prefill_hc_down_f32_encoded_calls();
                const auto start = std::chrono::steady_clock::now(); g.begin();
                if (arm >= 3) g.dense_q8_staged(matrix.view, x.view, active_outputs[arm], c, r, n, arm == 4);
                else if (arm == 2) g.dense_mm_q8(PrefillWeightLayout::gguf, matrix.view, x, y);
                else if (arm == 1) for (const auto& chunk : chunks) g.q8_matvec_many(matrix.view, chunk.input, chunk.output, c, r, chunk.width);
                else for (uint32_t t = 0; t < n; ++t) g.gemv(8, matrix.view, input_rows[t], output_rows[t], c, r);
                const auto encoded_end = std::chrono::steady_clock::now(); const auto batch = g.submit(); const auto end = std::chrono::steady_clock::now();
                const auto after = g.hc_exact_routes(); const auto ns = [](auto from, auto to) { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()); };
                HcExactRoutes delta{}, expected{}; for (size_t i = 0; i < delta.size(); ++i) delta[i] = after[i] - routes[i];
                if (arm == 0) expected[0] = n; else if (arm == 1) expected = many_routes;
                HcCostTrial trial{arm, cycle, order, measured, ns(start, encoded_end), ns(encoded_end, end), ns(start, end), g.concurrent_barriers(), delta, batch};
                need(batch.dispatches == (arm >= 2 ? 1 : arm == 1 ? chunks.size() : n) && batch.capture_executes == 0, "HC down cost physical dispatch or direct batch mismatch");
                need(delta == expected, "HC down cost actual successful scalar/many route buckets differ");
                need(g.prefill_hc_down_f32_encoded_calls() == encoded && encoded == 0, "HC down direct primitive inflated selected-wrapper counter");
                need(g.memory().allocations == prepared.allocations && g.memory().live_bytes == prepared.live_bytes, "HC down cost allocated while encoding or submitting");
                need(std::isfinite(batch.wall_ms) && batch.wall_ms > 0 && trial.encode_ns > 0 && trial.submit_wait_ns > 0 && trial.whole_ns == trial.encode_ns + trial.submit_wait_ns, "HC down cost host wall invalid");
                return trial;
            };
            auto validate = [&] {
                const auto expected = read(g, outputs[0]), exact = read(g, outputs[1]), actual = read(g, outputs[2]), staged = read(g, outputs[3]), half = read(g, outputs[4]); double square = 0, signal = 0, maximum = 0, staged_square = 0, half_square = 0;
                need(expected.size() == blank.size() && exact.size() == blank.size() && actual.size() == blank.size() && staged.size() == blank.size() && half.size() == blank.size(), "HC down cost output allocation shape");
                for (size_t i = 0; i < actual.size(); ++i) {
                    need(std::isfinite(actual[i]) && std::isfinite(exact[i]) && std::isfinite(expected[i]) && std::isfinite(staged[i]) && std::isfinite(half[i]), "HC down cost full output finite screen");
                    if (i >= uint64_t(n) * r) { need(actual[i] == sentinel && exact[i] == sentinel && expected[i] == sentinel && staged[i] == sentinel && half[i] == sentinel, "HC down cost changed inactive rows or output guards"); continue; }
                    const double delta = double(actual[i]) - expected[i], staged_delta = double(staged[i]) - expected[i], half_delta = double(half[i]) - expected[i]; square += delta * delta; staged_square += staged_delta * staged_delta; half_square += half_delta * half_delta; signal += double(expected[i]) * expected[i]; maximum = std::max(maximum, std::abs(delta));
                }
                need(std::memcmp(expected.data(), exact.data(), expected.size() * sizeof(float)) == 0, "HC down exact-many output differs from scalar bytes");
                const double nmse = signal ? square / signal : square, staged_nmse = signal ? staged_square / signal : staged_square, half_nmse = signal ? half_square / signal : half_square; need(nmse <= 1e-7 && staged_nmse <= 1e-7, "HC down cost full FP32 NMSE screen");
                double dense_square = 0, scalar_square = 0, staged_reference_square = 0, half_reference_square = 0, reference_signal = 0;
                for (uint32_t t : {0u, 1u, 7u, 8u, 15u, 16u, n / 2, n - 1}) if (t < n) for (uint32_t row : {0u, 1u, 15u, 16u, 17u, r / 2, r - 1}) {
                    const auto dot = reference_gemv(8, std::span(weight_bytes).subspan(uint64_t(row) * c / 32 * 34, uint64_t(c) / 32 * 34), c, 1, std::span(values).subspan(uint64_t(t) * c, c))[0];
                    const double dense_delta = double(actual[uint64_t(t) * r + row]) - dot, scalar_delta = double(expected[uint64_t(t) * r + row]) - dot;
                    const double staged_delta = double(staged[uint64_t(t) * r + row]) - dot, half_delta = double(half[uint64_t(t) * r + row]) - dot;
                    dense_square += dense_delta * dense_delta; scalar_square += scalar_delta * scalar_delta; staged_reference_square += staged_delta * staged_delta; half_reference_square += half_delta * half_delta; reference_signal += dot * dot;
                }
                const double dense_nmse = reference_signal ? dense_square / reference_signal : dense_square, scalar_nmse = reference_signal ? scalar_square / reference_signal : scalar_square;
                const double staged_fp64_nmse = reference_signal ? staged_reference_square / reference_signal : staged_reference_square, half_fp64_nmse = reference_signal ? half_reference_square / reference_signal : half_reference_square;
                need(dense_nmse <= 1e-7 && scalar_nmse <= 1e-7 && staged_fp64_nmse <= 1e-7, "HC down cost sampled independent FP64 NMSE screen");
                need(g.read(matrix.view) == weight_bytes && g.read(input) == input_bytes, "HC down cost changed represented weights or immutable input");
                return std::array{nmse, maximum, dense_nmse, scalar_nmse, staged_nmse, staged_fp64_nmse, half_nmse, half_fp64_nmse};
            };
            for (uint32_t arm = 0; arm < 5; ++arm) static_cast<void>(launch(arm, false, 0, arm)); static_cast<void>(validate());
            std::vector<HcCostTrial> trials; trials.reserve((warmup_cycles + measured_cycles) * 5);
            for (uint32_t phase = 0; phase < 2; ++phase) for (uint32_t cycle = 0; cycle < (phase ? measured_cycles : warmup_cycles); ++cycle)
                for (uint32_t order = 0; order < 5; ++order) trials.push_back(launch(orders[cycle][order], phase != 0, cycle, order));
            const auto validated = validate();
            for (const auto& t : trials) hc_cost_emit({{"schema", Json::integer(3)}, {"event", Json("hc_down_cost_trial")}, {"n", Json::integer(n)}, {"arm", Json(hc_cost_arm(t.arm))}, {"phase", Json(t.measured ? "measured" : "warmup")}, {"cycle", Json::integer(t.cycle)}, {"order_in_cycle", Json::integer(t.order)}, {"encode_api_wall_ns", Json::integer(t.encode_ns)}, {"submit_wait_api_wall_ns", Json::integer(t.submit_wait_ns)}, {"whole_batch_api_wall_ns", Json::integer(t.whole_ns)}, {"batch_wall_ms", hc_cost_number(t.batch.wall_ms)}, {"batch_gpu_ms", std::isfinite(t.batch.gpu_ms) && t.batch.gpu_ms > 0 ? hc_cost_number(t.batch.gpu_ms) : Json{}}, {"batch_gpu_ms_raw_lexeme", Json(hc_cost_raw_number(t.batch.gpu_ms))}, {"gpu_duration_reported_positive", Json(std::isfinite(t.batch.gpu_ms) && t.batch.gpu_ms > 0)}, {"dispatches", Json::integer(t.batch.dispatches)}, {"successful_hc_routes", hc_cost_routes(t.successful_routes)}, {"concurrent_barriers", Json::integer(t.barriers)}, {"wrapper_encoded_calls", Json::integer(0)}});
            const auto encode = [](const HcCostTrial& t) { return double(t.encode_ns); }; const auto whole = [](const HcCostTrial& t) { return double(t.whole_ns); }; const auto gpu = [](const HcCostTrial& t) { return t.batch.gpu_ms; }; const auto wall = [](const HcCostTrial& t) { return t.batch.wall_ms; };
            auto gpu_observed = [&](uint32_t arm) { return std::all_of(trials.begin(), trials.end(), [&](const auto& t) { return !t.measured || t.arm != arm || (std::isfinite(t.batch.gpu_ms) && t.batch.gpu_ms > 0); }); };
            Json::Array summaries;
            for (uint32_t arm = 0; arm < 5; ++arm) {
                const double arm_encode = hc_cost_median(trials, arm, encode), arm_whole = hc_cost_median(trials, arm, whole);
                const bool observed = gpu_observed(arm), comparable = observed && gpu_observed(0); const double arm_gpu = observed ? hc_cost_median(trials, arm, gpu) : 0;
                summaries.push_back(Json(Json::Object{{"arm", Json(hc_cost_arm(arm))}, {"median_encode_api_wall_ns", hc_cost_number(arm_encode)}, {"median_submit_wait_api_wall_ns", hc_cost_number(hc_cost_median(trials, arm, [](const auto& t) { return double(t.submit_wait_ns); }))}, {"median_whole_batch_api_wall_ns", hc_cost_number(arm_whole)}, {"median_batch_wall_ms", hc_cost_number(hc_cost_median(trials, arm, wall))}, {"gpu_duration_all_measured_positive", Json(observed)}, {"median_batch_gpu_ms", observed ? hc_cost_number(arm_gpu) : Json{}}, {"lower_encode_wall_than_scalar", Json(arm_encode < hc_cost_median(trials, 0, encode))}, {"lower_whole_wall_than_scalar", Json(arm_whole < hc_cost_median(trials, 0, whole))}, {"lower_gpu_duration_than_scalar", comparable ? Json(arm_gpu < hc_cost_median(trials, 0, gpu)) : Json{}}}));
            }
            hc_cost_emit({{"schema", Json::integer(3)}, {"event", Json("hc_down_cost_case")}, {"n", Json::integer(n)}, {"prospective_partial_primitive", Json(n < 512)}, {"warmup_cycles", Json::integer(warmup_cycles)}, {"measured_cycles", Json::integer(measured_cycles)}, {"ordinary_dispatches", Json::integer(n)}, {"many_dispatches", Json::integer(chunks.size())}, {"many_successful_hc_routes", hc_cost_routes(many_routes)}, {"dense_dispatches", Json::integer(1)}, {"exact_many_byte_exact", Json(true)}, {"dense_full_output_nmse", hc_cost_number(validated[0])}, {"dense_maximum_abs_error", hc_cost_number(validated[1])}, {"dense_sampled_fp64_nmse", hc_cost_number(validated[2])}, {"scalar_sampled_fp64_nmse", hc_cost_number(validated[3])}, {"staged_fp32_full_output_nmse", hc_cost_number(validated[4])}, {"staged_fp32_sampled_fp64_nmse", hc_cost_number(validated[5])}, {"half_comparison_full_output_nmse", hc_cost_number(validated[6])}, {"half_comparison_sampled_fp64_nmse", hc_cost_number(validated[7])}, {"half_comparison_within_fp32_screen", Json(validated[6] <= 1e-7 && validated[7] <= 1e-7)}, {"half_comparison_quality_admitted", Json(false)}, {"arms", Json(std::move(summaries))}, {"exact_and_fp32_correctness_screens_passed", Json(true)}, {"performance_qualified", Json(false)}});
        }
    }
    need(g.memory().live_bytes == initial, "HC down cost leaked prepared test buffers");
    hc_cost_emit({{"schema", Json::integer(3)}, {"event", Json("hc_down_cost_complete")}, {"cases", Json::integer(11)}, {"checks", Json::integer(checks)}, {"passed", Json(true)}, {"half_comparison_quality_admitted", Json(false)}, {"fp32_thresholds_unchanged", Json(true)}, {"performance_qualified", Json(false)}, {"scope", Json("generated raw-Q8 projection cost screen only; no model, state, logits, fidelity or serving route admission; nominal host evidence supplied separately")}});
}
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
struct HcDownWindowResult {
    std::vector<float> input, scales, down, normalized;
    std::vector<std::byte> down_weights;
    uint64_t dispatches = 0;
};
HcDownWindowResult hc_down_window_case(const char* library, uint32_t n, ResidualShape shape, uint32_t down_type, bool enabled) {
    MetalGraph g(library, 512ull << 20); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    g.configure_prefill_dense_f32(true); g.configure_prefill_grouped_f32(true); g.configure_prefill_dense_staged_f32(true); g.configure_prefill_hc_rows(true);
    need(!g.prefill_hc_down_window() && g.hc_down_window_encoded() == 0, "HC down window default changed");
    g.configure_prefill_hc_down_window(enabled);
    need(g.prefill_hc_down_window() == enabled, "HC down window selector projection");
    reject([&] { g.configure_prefill_hc_down_window(!enabled); });
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    if (enabled) reject([&] { g.configure_prefill_hc_down_f32(true); });
#endif
    const auto initial = g.memory().live_bytes; const auto width = shape.branches * shape.channels;
    const bool selected = enabled && n >= 2 && n <= 512 && shape.branches == 4 && shape.channels == 2560 && shape.rank == 320 && down_type == 8;
    HcDownWindowResult result;
    {
        Arena a{g, {}}; std::vector<float> gains(width, 1), inject(uint64_t(width) * shape.branches);
        for (size_t i = 0; i < gains.size(); ++i) gains[i] = 0.9f + float(i % 11) / 100;
        for (size_t i = 0; i < inject.size(); ++i) inject[i] = float(int(i % 17) - 8) / 32768;
        GpuMatrix down;
        if (down_type == 8) down = a.q8(width, shape.rank);
        else {
            need(down_type == 0, "generated non-Q8 down fixture encoding");
            std::vector<float> values(uint64_t(width) * shape.rank);
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 15) - 7) / 1024;
            down = {a.floats(values), 0};
        }
        const ResidualWeights w{a.floats(gains), down, a.q8(shape.rank, width), {a.floats(inject), 0}};
        result.down_weights = g.read(w.down.view);
        prepare_prefill_residual(g, w, shape);
        auto scratch = allocate_prefill_residual_scratch(g, shape, 512);
        const uint64_t stride = uint64_t(width) * 4 + 16;
        std::vector<float> values(stride / 4 * n, -7654.25f);
        for (uint32_t t = 0; t < n; ++t) for (uint32_t j = 0; j < width; ++j) values[t * stride / 4 + j] = float(int((uint64_t(t) * 13 + j * 7) % 131) - 65) / 997;
        const auto input = a.floats(values);
        const auto allocations = g.memory().allocations; const auto routes = g.hc_exact_routes();
        const auto encoded_before = g.hc_down_window_encoded();
        g.begin(); encode_prefill_residual(g, w, scratch, {input, n, width, stride}, shape, 1e-6f);
        result.dispatches = g.submit().dispatches;
        need(g.hc_down_window_encoded() - encoded_before == uint64_t(selected), "HC down window encoded call count differs");
        need(result.dispatches == prefill_residual_dispatches(g, w, shape, n), "HC down window dispatch predictor mismatch");
        need(g.memory().allocations == allocations, "HC down window allocated while encoding");
        const auto after = g.hc_exact_routes();
        const uint64_t expected_exact = width == 10240 && shape.rank == 320 && down_type == 8 && !selected ? n : 0;
        need(after[0] - routes[0] == expected_exact, "HC down window/fallback exact route count");
        for (size_t i = 1; i < routes.size(); ++i) need(after[i] == routes[i], "HC down window changed many-row exact routes");
        result.input = read(g, scratch.rows.input); result.scales = read(g, scratch.rows.scales);
        result.down = read(g, scratch.rows.bottleneck); result.normalized = read(g, scratch.rows.normalized);
        for (size_t i = uint64_t(n) * shape.channels; i < result.input.size(); ++i) need(result.input[i] == 0, "HC down window input scratch tail changed");
        for (size_t i = uint64_t(n) * shape.branches; i < result.scales.size(); ++i) need(result.scales[i] == 0, "HC down window scale scratch tail changed");
        for (size_t i = uint64_t(n) * shape.rank; i < result.down.size(); ++i) need(result.down[i] == 0, "HC down window bottleneck scratch tail changed");
        if (selected) {
            const auto raw_down = a.zero(uint64_t(n) * shape.rank * 4);
            g.begin(); g.hc_down_q8_window(w.down.view, {scratch.rows.normalized, n, width, uint64_t(width) * 4}, {raw_down, n, shape.rank, uint64_t(shape.rank) * 4});
            need(g.hc_down_window_encoded() == encoded_before + 2, "direct HC down window call was not counted"); g.abort();
            need(g.hc_down_window_encoded() == encoded_before + 2, "HC down window encoded lifetime rewound on abort");
        }
        need(g.read(w.down.view) == result.down_weights && g.read(input) == std::vector<std::byte>(std::as_bytes(std::span(values)).begin(), std::as_bytes(std::span(values)).end()), "HC down window changed represented weights or input");
        release_prefill_residual_scratch(g, scratch);
    }
    need(g.memory().live_bytes == initial, "HC down window leaked buffers"); return result;
}
void hc_down_window_controls(const char* library) {
    for (uint32_t n : {1u, 2u, 3u, 15u, 16u, 17u, 31u, 32u, 33u, 332u, 490u, 511u, 512u}) {
        const auto ordinary = hc_down_window_case(library, n, {4, 2560, 320}, 8, false), candidate = hc_down_window_case(library, n, {4, 2560, 320}, 8, true);
        if (n >= 2) need(ordinary.dispatches == candidate.dispatches + n - 1, "HC down window did not replace per-token dispatches");
        else need(ordinary.dispatches == candidate.dispatches, "HC down window singleton dispatch count changed");
        need(ordinary.down == candidate.down && ordinary.input == candidate.input && ordinary.scales == candidate.scales && ordinary.normalized == candidate.normalized,
            "HC down window was not byte-identical to scalar exact HC down");
    }
    for (const auto& [n, shape, type] : std::array<std::tuple<uint32_t, ResidualShape, uint32_t>, 2>{{{37u, {4, 256, 32}, 8u}, {37u, {4, 2560, 320}, 0u}}}) {
        const auto ordinary = hc_down_window_case(library, n, shape, type, false), candidate = hc_down_window_case(library, n, shape, type, true);
        need(ordinary.dispatches == candidate.dispatches && ordinary.down == candidate.down && ordinary.input == candidate.input && ordinary.scales == candidate.scales,
            "HC down window unsupported geometry or non-Q8 fallback changed");
    }
    // Verify padded 4-byte and 16-byte strides, unaligned view slice offsets, guard sentinels and indirect-execution refusal.
    MetalGraph g(library, 512ull << 20); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    g.configure_prefill_hc_down_window(true); g.prepare_hc_down_window();
    reject([&] { g.begin_indirect_compatible(); });
    {
        Arena a{g, {}}; const auto weights = a.q8(10240, 320);
        for (uint64_t pad_bytes : {0ull, 4ull, 16ull}) {
            const uint32_t n = 33; const uint64_t in_stride = 10240ull * 4 + pad_bytes, out_stride = 320ull * 4 + pad_bytes;
            std::vector<float> x(4 + in_stride / 4 * n, -7654.25f), ref_init(out_stride / 4 * n + 8, -7654.25f);
            for (uint32_t t = 0; t < n; ++t) for (uint32_t j = 0; j < 10240; ++j) x[1 + t * (in_stride / 4) + j] = float(int((uint64_t(t) * 19 + j * 11) % 199) - 99) / 503.0f;
            const auto xv_full = a.floats(x), xv = g.slice(xv_full, 4, in_stride * n);
            const auto out_ref = a.floats(ref_init, false), out_win = a.floats(ref_init, false);
            g.begin();
            for (uint32_t t = 0; t < n; ++t) g.gemv(8, weights.view, g.slice(xv, t * in_stride, 10240ull * 4), g.slice(out_ref, t * out_stride, 320ull * 4), 10240, 320);
            static_cast<void>(g.submit());
            g.begin(); g.hc_down_q8_window(weights.view, {xv, n, 10240, in_stride}, {out_win, n, 320, out_stride});
            need(g.submit().dispatches == 1, "HC down window strided call dispatch count");
            need(read(g, out_ref) == read(g, out_win), "HC down window padded/unaligned stride bytes or guard sentinels differ");
        }
    }
    std::cout << "Exact HC down window byte-equality, singleton/geometry fallback and padded-stride controls pass; no model or latency admission\n";
}
#endif
}
int main(int argc, char** argv) { try {
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    if (argc == 3 && std::string(argv[2]) == "--hc-down-window") { hc_down_window_controls(argv[1]); return 0; }
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_STAGED
    if (argc == 3 && std::string(argv[2]) == "--hc-down-staged-rows") {
        for (uint32_t n : {1u, 2u, 3u, 18u, 37u, 332u, 490u, 511u, 512u}) hc_down_pair(argv[1], n, {4, 2560, 320}, 8, true);
        hc_down_pair(argv[1], 37, {4, 256, 32}, 8, true); hc_down_pair(argv[1], 37, {4, 2560, 320}, 0, true);
        std::cout << "Staged HC down row composition gates pass; existing FP32 thresholds, singleton and unsupported-geometry fallback retained; no model or latency admission\n"; return 0;
    }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    if (argc == 3 && std::string(argv[2]) == "--dense-q8-shape-cost-staged") { dense_q8_shape_cost_screen(argv[1]); return 0; }
    if (argc == 3 && std::string(argv[2]) == "--hc-down-cost-staged") { hc_down_cost_screen(argv[1]); return 0; }
#endif
    if (argc != 2 && !(argc == 3 && std::string(argv[2]) == "--hc-down-f32")) throw std::runtime_error("usage: test-prefill_residual OWNED_METALLIB [--hc-down-f32] (staged component also accepts --hc-down-cost-staged)");
#else
    if (argc != 2) throw std::runtime_error("usage: test-prefill_residual OWNED_METALLIB");
#endif
    MetalGraph g(argv[1], 512ull << 20); g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    for (uint32_t n : {1, 3, 512}) exercise(g, n, {4, 256, 32}); exercise(g, 3, {4, 2560, 320});
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    if (argc == 3) {
        hc_down_pair(argv[1], 512, {4, 2560, 320}, 8);
        hc_down_pair(argv[1], 1, {4, 2560, 320}, 8);
        hc_down_pair(argv[1], 3, {4, 2560, 320}, 8);
        hc_down_pair(argv[1], 512, {4, 256, 32}, 8);
        hc_down_pair(argv[1], 512, {4, 2560, 320}, 0);
    }
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    hc_down_window_controls(argv[1]);
#endif
    std::cout << "Generated residual checks " << checks << ", maximum scaled error " << worst << " pass; no model or admission\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
