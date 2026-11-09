#include "tokspur/ffn_check.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"
#include "tokspur/moe.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/operators.hpp"
#include "tokspur/routing.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
uint32_t integer(std::string_view input, uint32_t limit) {
    uint32_t value = 0;
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (result.ec != std::errc{} || result.ptr != input.data() + input.size() || value > limit) throw FormatError("invalid bounded FFN check integer");
    return value;
}
Json number(double value) {
    if (!std::isfinite(value)) throw FormatError("nonfinite FFN diagnostic");
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (result.ec != std::errc{}) throw FormatError("cannot encode FFN diagnostic");
    return parse_json({buffer.data(), static_cast<size_t>(result.ptr - buffer.data())});
}
template<class T> std::vector<T> read(MetalGraph& gpu, GpuView view) {
    const auto bytes = gpu.read(view);
    if (bytes.size() % sizeof(T)) throw FormatError("invalid FFN readback size");
    std::vector<T> output(bytes.size() / sizeof(T));
    std::memcpy(output.data(), bytes.data(), bytes.size());
    return output;
}
Json indices(std::span<const uint32_t> ids) {
    Json::Array values;
    for (const auto id : ids) values.push_back(Json::integer(id));
    return Json(std::move(values));
}
double error(std::span<const float> actual, std::span<const double> expected) {
    if (actual.size() != expected.size()) throw FormatError("FFN reference size mismatch");
    double maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw FormatError("nonfinite FFN reference/result");
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i])));
    }
    return maximum;
}
}

int ffn_check_cli(int argc, char** argv) {
    uint32_t layer = 0, repeats = 2;
    std::vector<uint32_t> tokens;
    std::optional<RouterMode> mode;
    std::optional<bool> normalized;
    auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    std::vector<std::filesystem::path> paths;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (!option.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        if (++i == argc) throw FormatError("missing FFN check option value");
        const std::string_view value(argv[i]);
        if (option == "--layer") layer = integer(value, 47);
        else if (option == "--token") { if (tokens.size() == 16) throw FormatError("too many FFN check tokens"); tokens.push_back(integer(value, 248319)); }
        else if (option == "--repeats") { repeats = integer(value, 16); if (!repeats) throw FormatError("FFN repeats must be positive"); }
        else if (option == "--library") library = value;
        else if (option == "--router-mode" && !mode) {
            if (value == "softmax") mode = RouterMode::softmax;
            else if (value == "sigmoid") mode = RouterMode::sigmoid;
            else throw FormatError("unsupported FFN router mode");
        } else if (option == "--normalize-selected" && !normalized) {
            if (value == "true") normalized = true;
            else if (value == "false") normalized = false;
            else throw FormatError("expected explicit FFN normalization boolean");
        } else throw FormatError("unknown or duplicate FFN check option");
    }
    if (!mode || !normalized) throw FormatError("FFN check requires explicit --router-mode softmax|sigmoid --normalize-selected true|false; checkpoint semantics are not yet qualified");
    if (tokens.empty()) tokens = {248045, 0, 12000};
    ModelFiles files(paths);
    static_cast<void>(inspect_flash_next(files));
    constexpr MoeShape shape{2560, 640, 512, 10};
    const auto prefix = "blk." + std::to_string(layer) + ".";
    const auto setup = Clock::now();
    MetalGraph gpu(library, uint64_t(4) * 1024 * 1024 * 1024);
    const auto load = [&](const char* suffix) {
        const auto name = prefix + suffix;
        const auto& tensor = files.tensor(name);
        return GpuMatrix{gpu.upload(files.tensor_bytes(name), name), tensor.type};
    };
    const MoeWeights weights{load("ffn_gate_inp.weight"), load("ffn_gate_exps.weight"), load("ffn_up_exps.weight"), load("ffn_down_exps.weight"),
        load("ffn_gate_shexp.weight"), load("ffn_up_shexp.weight"), load("ffn_down_shexp.weight"), load("ffn_gate_inp_shexp.weight")};
    const auto scratch = allocate_moe_scratch(gpu, shape);
    const auto input = gpu.zeros(shape.channels * 4, "FFN check input"), output = gpu.zeros(shape.channels * 4, "FFN check output");
    const auto setup_ms = elapsed(setup);
    const auto allocations = gpu.memory().allocations;
    const auto project = [&](const char* suffix, std::span<const double> x, std::optional<uint32_t> expert = std::nullopt) {
        const auto name = prefix + suffix;
        const auto& tensor = files.tensor(name);
        auto data = files.tensor_bytes(name);
        if (expert) {
            const auto bytes = tensor.bytes / shape.experts;
            data = data.subspan(*expert * bytes, bytes);
        }
        const auto rows = tensor.dimensions.size() > 1 ? static_cast<uint32_t>(tensor.dimensions[1]) : 1u;
        return reference_gemv_doubleinput(tensor.type, data, static_cast<uint32_t>(tensor.dimensions[0]), rows, x);
    };
    bool passed = true;
    Json::Array cases;
    for (const auto token : tokens) {
        const auto reference_start = Clock::now();
        std::vector<float> x(shape.channels);
        const auto& embedding = files.tensor("token_embd.weight");
        const auto row_bytes = embedding.bytes / embedding.dimensions[1];
        dequantize(embedding.type, files.tensor_bytes(embedding.name).subspan(token * row_bytes, row_bytes), x);
        const std::vector<double> precise_input(x.begin(), x.end());
        const auto logits = project("ffn_gate_inp.weight", precise_input);
        const auto routes = route_reference(logits, shape.selected, *mode, *normalized);
        std::vector<double> activated, routed;
        for (const auto expert : routes.indices) {
            auto gate = project("ffn_gate_exps.weight", precise_input, expert);
            const auto up = project("ffn_up_exps.weight", precise_input, expert);
            for (size_t i = 0; i < gate.size(); ++i) gate[i] = stable_silu(gate[i]) * up[i];
            activated.insert(activated.end(), gate.begin(), gate.end());
            const auto down = project("ffn_down_exps.weight", gate, expert);
            routed.insert(routed.end(), down.begin(), down.end());
        }
        auto shared_gate = project("ffn_gate_shexp.weight", precise_input);
        const auto shared_up = project("ffn_up_shexp.weight", precise_input);
        for (size_t i = 0; i < shared_gate.size(); ++i) shared_gate[i] = stable_silu(shared_gate[i]) * shared_up[i];
        const auto shared_down = project("ffn_down_shexp.weight", shared_gate);
        const auto expected = mixture_reference(routed, routes.weights, shared_down, project("ffn_gate_inp_shexp.weight", precise_input)[0]);
        const auto cpu_ms = elapsed(reference_start);
        gpu.write(input, std::as_bytes(std::span(x)));
        Json::Array executions;
        for (uint32_t run = 0; run < repeats; ++run) {
            gpu.begin();
            encode_moe(gpu, weights, scratch, input, output, shape, *mode, *normalized);
            const auto batch = gpu.submit();
            if (batch.dispatches != 14 || gpu.memory().allocations != allocations) throw FormatError("connected FFN resource/dispatch invariant failed");
            const auto actual_routes = read<uint32_t>(gpu, scratch.indices);
            const std::array<double, 7> errors{error(read<float>(gpu, scratch.logits), logits), error(read<float>(gpu, scratch.weights), routes.weights),
                error(read<float>(gpu, scratch.gate), activated), error(read<float>(gpu, scratch.down), routed),
                error(read<float>(gpu, scratch.shared_gate), shared_gate), error(read<float>(gpu, scratch.shared_down), shared_down), error(read<float>(gpu, output), expected)};
            const auto maximum = *std::max_element(errors.begin(), errors.end());
            const bool valid = actual_routes == routes.indices && maximum <= 5e-5;
            passed = passed && valid;
            Json::Array stage_errors;
            for (const auto value : errors) stage_errors.push_back(number(value));
            executions.push_back(Json(Json::Object{{"pass", Json(valid)}, {"route_ids", indices(actual_routes)}, {"max_scaled_error", number(maximum)},
                {"stage_scaled_errors", Json(std::move(stage_errors))}, {"dispatches", Json::integer(batch.dispatches)}, {"wall_ms", number(batch.wall_ms)}, {"gpu_ms", number(batch.gpu_ms)}}));
        }
        cases.push_back(Json(Json::Object{{"token_fixture", Json::integer(token)}, {"cpu_reference_ms", number(cpu_ms)}, {"runs", Json(std::move(executions))}}));
    }
    files.assert_unchanged();
    std::cout << Json(Json::Object{{"schema", Json::integer(1)}, {"scope", Json("connected real-weight FFN subgraph; embedding rows are synthetic hidden-state probes, not actual layer/prefix outputs")},
        {"layer", Json::integer(layer)}, {"router_mode", Json(*mode == RouterMode::softmax ? "softmax" : "sigmoid")}, {"normalize_selected", Json(*normalized)},
        {"mode_semantics_qualified", Json(false)}, {"timings_qualified", Json(false)}, {"weight_setup_ms", number(setup_ms)},
        {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)}, {"allocations", Json::integer(allocations)},
        {"stage_order", Json("router_logits,route_weights,expert_activated,expert_outputs,shared_activated,shared_output,final")}, {"cases", Json(std::move(cases))}, {"pass", Json(passed)}}).dump() << '\n';
    return passed ? 0 : 1;
}

} // namespace tokspur
