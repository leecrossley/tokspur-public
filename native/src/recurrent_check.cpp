#include "tokspur/recurrent_check.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
uint32_t integer(std::string_view input, uint32_t limit) {
    uint32_t value = 0;
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (result.ec != std::errc{} || result.ptr != input.data() + input.size() || value > limit) throw FormatError("invalid bounded recurrent check integer");
    return value;
}
Json number(double value) {
    if (!std::isfinite(value)) throw FormatError("nonfinite recurrent diagnostic");
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (result.ec != std::errc{}) throw FormatError("cannot encode recurrent diagnostic");
    return parse_json({buffer.data(), static_cast<size_t>(result.ptr - buffer.data())});
}
}
int recurrent_check_cli(int argc, char** argv) {
    uint32_t layer = 0, repeats = 1;
    std::vector<uint32_t> tokens;
    std::optional<RouterMode> mode;
    std::optional<bool> normalized, scaled_query;
    std::optional<L2Mode> l2;
    auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    std::vector<std::filesystem::path> paths;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (!option.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        if (++i == argc) throw FormatError("missing recurrent check option value");
        const std::string_view value(argv[i]);
        if (option == "--layer") layer = integer(value, 47);
        else if (option == "--token") { if (tokens.size() == 16) throw FormatError("too many recurrent check tokens"); tokens.push_back(integer(value, 248319)); }
        else if (option == "--repeats") { repeats = integer(value, 16); if (!repeats) throw FormatError("recurrent repeats must be positive"); }
        else if (option == "--library") library = value;
        else if (option == "--router-mode" && !mode) {
            if (value == "softmax") mode = RouterMode::softmax;
            else if (value == "sigmoid") mode = RouterMode::sigmoid;
            else throw FormatError("unsupported recurrent router mode");
        } else if (option == "--normalize-selected" && !normalized) {
            if (value == "true") normalized = true;
            else if (value == "false") normalized = false;
            else throw FormatError("expected explicit recurrent normalization boolean");
        } else if (option == "--query-scale" && !scaled_query) {
            if (value == "unit") scaled_query = false;
            else if (value == "inverse-sqrt-key-width") scaled_query = true;
            else throw FormatError("unsupported recurrent query scale hypothesis");
        } else if (option == "--l2-mode" && !l2) {
            if (value == "add") l2 = L2Mode::add_squared_epsilon;
            else if (value == "floor") l2 = L2Mode::floor_squared_epsilon;
            else throw FormatError("unsupported recurrent L2 mode hypothesis");
        } else throw FormatError("unknown or duplicate recurrent check option");
    }
    if (!mode || !normalized || !scaled_query || !l2) throw FormatError("recurrent check requires explicit router, normalization, query-scale and L2 hypotheses; checkpoint execution semantics remain unqualified");
    if (layer % 4 == 3) throw FormatError("sparse attention is not an ordinary recurrent layer");
    if (layer == 1) throw FormatError("PLE layer requires n-gram memory; refusing to silently omit it");
    if (tokens.empty()) tokens = {248045, 0, 12000};
    ModelFiles files(paths);
    static_cast<void>(inspect_flash_next(files));
    constexpr RecurrentShape shape{{4, 2560, 320}, {2560, 16, 48, 128, 128, 4}, {2560, 640, 512, 10}};
    const GdnPolicy policy{*scaled_query ? 1 / std::sqrt(128.0f) : 1, 1e-6f, *l2, 1e-6f};
    const auto prefix = "blk." + std::to_string(layer) + ".";
    const auto setup = Clock::now();
    MetalGraph gpu(library, uint64_t(4) * 1024 * 1024 * 1024);
    std::map<std::string, GpuMatrix> uploaded;
    const auto load = [&](const std::string& suffix) {
        if (!uploaded.contains(suffix)) {
            const auto name = prefix + suffix;
            const auto& t = files.tensor(name);
            uploaded.emplace(suffix, GpuMatrix{gpu.upload(files.tensor_bytes(name), name), t.type});
        }
        return uploaded.at(suffix);
    };
    const auto reference_matrix = [&](const std::string& suffix) {
        const auto name = prefix + suffix;
        return ReferenceMatrix{files.tensor_bytes(name), files.tensor(name).type};
    };
    std::map<std::string, std::vector<double>> scalar_data;
    const auto scalar = [&](const std::string& suffix) -> std::span<const double> {
        if (!scalar_data.contains(suffix)) {
            const auto name = prefix + suffix;
            const auto& t = files.tensor(name);
            if (t.type != 0 || t.elements > 65536) throw FormatError("expected bounded F32 checkpoint parameter");
            std::vector<float> decoded(t.elements);
            dequantize(t.type, files.tensor_bytes(name), decoded);
            scalar_data.emplace(suffix, std::vector<double>(decoded.begin(), decoded.end()));
        }
        return scalar_data.at(suffix);
    };
    const auto residual = [&](const std::string& kind) {
        return ResidualWeights{load(kind + "_norm.weight").view, load(kind + "_down.weight"), load(kind + "_up.weight"), load(kind + "_inject.weight")};
    };
    const auto reference_residual = [&](const std::string& kind) {
        return ReferenceResidualWeights{scalar(kind + "_norm.weight"), reference_matrix(kind + "_down.weight"), reference_matrix(kind + "_up.weight"), reference_matrix(kind + "_inject.weight")};
    };
    const GdnWeights attention{load("attn_qkv.weight"), load("attn_gate.weight"), load("ssm_alpha.weight"), load("ssm_beta.weight"), load("ssm_out.weight"),
        load("ssm_conv1d.weight").view, load("ssm_a").view, load("ssm_dt.bias").view, load("ssm_norm.weight").view};
    const ReferenceGdnWeights reference_attention{reference_matrix("attn_qkv.weight"), reference_matrix("attn_gate.weight"), reference_matrix("ssm_alpha.weight"),
        reference_matrix("ssm_beta.weight"), reference_matrix("ssm_out.weight"), scalar("ssm_conv1d.weight"), scalar("ssm_a"), scalar("ssm_dt.bias"), scalar("ssm_norm.weight")};
    const MoeWeights ffn{load("ffn_gate_inp.weight"), load("ffn_gate_exps.weight"), load("ffn_up_exps.weight"), load("ffn_down_exps.weight"),
        load("ffn_gate_shexp.weight"), load("ffn_up_shexp.weight"), load("ffn_down_shexp.weight"), load("ffn_gate_inp_shexp.weight")};
    const ReferenceMoeWeights reference_ffn{reference_matrix("ffn_gate_inp.weight"), reference_matrix("ffn_gate_exps.weight"), reference_matrix("ffn_up_exps.weight"),
        reference_matrix("ffn_down_exps.weight"), reference_matrix("ffn_gate_shexp.weight"), reference_matrix("ffn_up_shexp.weight"), reference_matrix("ffn_down_shexp.weight"), reference_matrix("ffn_gate_inp_shexp.weight")};
    const RecurrentWeights weights{residual("hc_attn"), residual("hc_ffn"), attention, ffn};
    const ReferenceLayerWeights reference_weights{reference_residual("hc_attn"), reference_residual("hc_ffn"), reference_attention, reference_ffn};
    const auto scratch = allocate_recurrent_scratch(gpu, shape);
    auto current = allocate_gdn_state(gpu, shape.gdn), next = allocate_gdn_state(gpu, shape.gdn);
    const auto input = gpu.zeros(4 * 2560 * 4, "recurrent check input"), output = gpu.zeros(4 * 2560 * 4, "recurrent check output");
    const auto setup_ms = elapsed(setup);
    const auto allocations = gpu.memory().allocations;
    Json::Array runs;
    bool passed = true;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        std::vector<double> cpu_state(gdn_state_count(shape.gdn), 0), cpu_history(gdn_qkv_count(shape.gdn) * 3, 0);
        const std::vector<float> state_zero(cpu_state.size()), history_zero(cpu_history.size());
        gpu.write(current.matrix, std::as_bytes(std::span(state_zero))); gpu.write(current.convolution, std::as_bytes(std::span(history_zero)));
        for (size_t step = 0; step < tokens.size(); ++step) {
            const auto start = Clock::now();
            std::vector<float> x(2560), streams(10240);
            const auto& embedding = files.tensor("token_embd.weight");
            const auto row_bytes = embedding.bytes / embedding.dimensions[1];
            dequantize(embedding.type, files.tensor_bytes(embedding.name).subspan(tokens[step] * row_bytes, row_bytes), x);
            for (size_t branch = 0; branch < 4; ++branch) std::copy(x.begin(), x.end(), streams.begin() + branch * x.size());
            const auto expected = recurrent_reference(reference_weights, std::vector<double>(streams.begin(), streams.end()), cpu_state, cpu_history, shape, policy, *mode, *normalized);
            const auto cpu_ms = elapsed(start);
            gpu.write(input, std::as_bytes(std::span(streams)));
            const auto before_state = gpu.read(current.matrix), before_history = gpu.read(current.convolution);
            gpu.begin(); encode_recurrent_layer(gpu, weights, scratch, current, next, input, output, shape, policy, *mode, *normalized);
            const auto batch = gpu.submit();
            if (gpu.memory().allocations != allocations || batch.dispatches != 40) throw FormatError("recurrent resource/dispatch invariant failed");
            const bool immutable = gpu.read(current.matrix) == before_state && gpu.read(current.convolution) == before_history;
            const auto comparison = compare_recurrent_layer(gpu, scratch, next, output, expected);
            const auto valid = comparison.passed && immutable;
            passed = passed && valid;
            Json::Object errors;
            for (const auto& [name, value] : comparison.scaled_errors) errors.emplace_back(name, number(value));
            runs.push_back(Json(Json::Object{{"pass", Json(valid)}, {"repeat", Json::integer(repeat)}, {"step", Json::integer(step)}, {"token_fixture", Json::integer(tokens[step])},
                {"expert_ids_match", Json(comparison.routes_match)}, {"accepted_state_unchanged", Json(immutable)}, {"compared_values", Json::integer(comparison.compared_values)},
                {"max_scaled_error", number(comparison.maximum)}, {"stage_scaled_errors", Json(std::move(errors))}, {"dispatches", Json::integer(batch.dispatches)},
                {"cpu_reference_ms", number(cpu_ms)}, {"wall_ms", number(batch.wall_ms)}, {"gpu_ms", number(batch.gpu_ms)}}));
            if (!valid) break; // Never publish a failed candidate, even in a diagnostic.
            cpu_state = expected.attention.delta.state; cpu_history = expected.attention.convolution.history;
            std::swap(current, next);
        }
        if (!passed) break;
    }
    files.assert_unchanged();
    std::cout << Json(Json::Object{{"schema", Json::integer(1)}, {"scope", Json("ordinary connected recurrent layer with synthetic embedding-row branch inputs; not actual stacked model-prefix execution")},
        {"layer", Json::integer(layer)}, {"router_mode", Json(*mode == RouterMode::softmax ? "softmax" : "sigmoid")}, {"normalize_selected", Json(*normalized)},
        {"query_scale", Json(*scaled_query ? "inverse-sqrt-key-width" : "unit")}, {"l2_mode", Json(*l2 == L2Mode::add_squared_epsilon ? "add" : "floor")},
        {"mode_semantics_qualified", Json(false)}, {"timings_qualified", Json(false)}, {"weight_setup_ms", number(setup_ms)},
        {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)}, {"allocations", Json::integer(allocations)}, {"runs", Json(std::move(runs))}, {"pass", Json(passed)}}).dump() << '\n';
    return passed ? 0 : 1;
}
} // namespace tokspur
