#include "tokspur/attention_check.hpp"
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
double elapsed(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
uint32_t integer(std::string_view x, uint32_t maximum) {
    uint32_t value = 0; const auto r = std::from_chars(x.data(), x.data() + x.size(), value);
    if (r.ec != std::errc{} || r.ptr != x.data() + x.size() || value > maximum) throw FormatError("invalid bounded attention check integer");
    return value;
}
Json number(double x) {
    if (!std::isfinite(x)) throw FormatError("nonfinite attention diagnostic");
    std::array<char, 64> b{}; const auto r = std::to_chars(b.data(), b.data() + b.size(), x);
    if (r.ec != std::errc{}) throw FormatError("attention diagnostic number error");
    return parse_json({b.data(), static_cast<size_t>(r.ptr - b.data())});
}
float fixture(uint64_t i, uint32_t seed) { return float(int((i * 37 + uint64_t(seed) * 19) % 4093) - 2046) / 8192; }
std::vector<std::byte> prefix_bytes(MetalGraph& g, GpuView view, uint64_t count) { return count ? g.read(g.slice(view, 0, count)) : std::vector<std::byte>{}; }
}
int attention_check_cli(int argc, char** argv) {
    uint32_t layer = 3, repeats = 1;
    std::vector<uint32_t> positions, tokens;
    std::optional<RouterMode> mode; std::optional<bool> normalize; std::optional<QueryPacking> packing; std::optional<RotaryPairs> rotary;
    auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    std::vector<std::filesystem::path> paths;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (!option.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        if (++i == argc) throw FormatError("missing attention check option value");
        const std::string_view value(argv[i]);
        if (option == "--layer") layer = integer(value, 47);
        else if (option == "--position") { if (positions.size() == 16) throw FormatError("too many attention check positions"); positions.push_back(integer(value, 65535)); }
        else if (option == "--token") { if (tokens.size() == 16) throw FormatError("too many attention check tokens"); tokens.push_back(integer(value, 248319)); }
        else if (option == "--repeats") { repeats = integer(value, 16); if (!repeats) throw FormatError("attention repeats must be positive"); }
        else if (option == "--library") library = value;
        else if (option == "--router-mode" && !mode) {
            if (value == "softmax") mode = RouterMode::softmax; else if (value == "sigmoid") mode = RouterMode::sigmoid; else throw FormatError("unsupported attention router mode");
        } else if (option == "--normalize-selected" && !normalize) {
            if (value == "true") normalize = true; else if (value == "false") normalize = false; else throw FormatError("expected attention normalization boolean");
        } else if (option == "--query-packing" && !packing) {
            if (value == "head") packing = QueryPacking::head_query_gate; else if (value == "flat") packing = QueryPacking::all_queries_then_gates; else throw FormatError("unsupported attention query packing hypothesis");
        } else if (option == "--rotary-pairs" && !rotary) {
            if (value == "split") rotary = RotaryPairs::split_half; else if (value == "adjacent") rotary = RotaryPairs::adjacent; else throw FormatError("unsupported attention rotary hypothesis");
        } else throw FormatError("unknown or duplicate attention check option");
    }
    if (!mode || !normalize || !packing || !rotary) throw FormatError("attention check requires explicit router, normalization, query-packing and rotary hypotheses");
    if (layer % 4 != 3) throw FormatError("selected layer is not a sparse-attention layer");
    if (positions.empty()) positions = {0, 3, 7, 2047, 2051, 2052, 59000, 65535};
    if (tokens.empty()) tokens = {248045, 0, 12000};
    ModelFiles files(paths); const auto layout = inspect_flash_next(files);
    constexpr AttentionLayerShape shape{{4, 2560, 320}, {2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536}, {2560, 640, 512, 10}};
    const QsaPolicy policy{*packing, *rotary, float(layout.rms_epsilon)};
    const auto prefix = "blk." + std::to_string(layer) + ".";
    const auto start = Clock::now();
    MetalGraph gpu(library, uint64_t(4) * 1024 * 1024 * 1024);
    std::map<std::string, GpuMatrix> uploaded; std::map<std::string, std::vector<double>> scalars;
    const auto load = [&](const std::string& suffix) {
        if (!uploaded.contains(suffix)) { const auto name = prefix + suffix; uploaded.emplace(suffix, GpuMatrix{gpu.upload(files.tensor_bytes(name), name), files.tensor(name).type}); }
        return uploaded.at(suffix);
    };
    const auto matrix = [&](const std::string& suffix) { const auto name = prefix + suffix; return ReferenceMatrix{files.tensor_bytes(name), files.tensor(name).type}; };
    const auto scalar = [&](const std::string& suffix) -> std::span<const double> {
        if (!scalars.contains(suffix)) {
            const auto name = prefix + suffix; const auto& t = files.tensor(name);
            if (t.type != 0 || t.elements > 65536) throw FormatError("expected bounded F32 attention parameter");
            std::vector<float> data(t.elements); dequantize(t.type, files.tensor_bytes(name), data); scalars.emplace(suffix, std::vector<double>(data.begin(), data.end()));
        }
        return scalars.at(suffix);
    };
    const auto residual = [&](const std::string& kind) { return ResidualWeights{load(kind + "_norm.weight").view, load(kind + "_down.weight"), load(kind + "_up.weight"), load(kind + "_inject.weight")}; };
    const auto rr = [&](const std::string& kind) { return ReferenceResidualWeights{scalar(kind + "_norm.weight"), matrix(kind + "_down.weight"), matrix(kind + "_up.weight"), matrix(kind + "_inject.weight")}; };
    const auto frequencies = rotary_frequencies(64, layout.rope_base);
    const QsaWeights attention{load("attn_q.weight"), load("attn_k.weight"), load("attn_v.weight"), load("attn_output.weight"), load("indexer.q_proj.weight"), load("indexer.k_proj.weight"),
        load("attn_q_norm.weight").view, load("attn_k_norm.weight").view, load("indexer.q_norm.weight").view, load("indexer.k_norm.weight").view, gpu.upload(std::as_bytes(std::span(frequencies)), "owned rotary frequencies")};
    const ReferenceQsaWeights ra{matrix("attn_q.weight"), matrix("attn_k.weight"), matrix("attn_v.weight"), matrix("attn_output.weight"), matrix("indexer.q_proj.weight"), matrix("indexer.k_proj.weight"),
        scalar("attn_q_norm.weight"), scalar("attn_k_norm.weight"), scalar("indexer.q_norm.weight"), scalar("indexer.k_norm.weight"), frequencies};
    const MoeWeights moe{load("ffn_gate_inp.weight"), load("ffn_gate_exps.weight"), load("ffn_up_exps.weight"), load("ffn_down_exps.weight"), load("ffn_gate_shexp.weight"), load("ffn_up_shexp.weight"), load("ffn_down_shexp.weight"), load("ffn_gate_inp_shexp.weight")};
    const ReferenceMoeWeights rm{matrix("ffn_gate_inp.weight"), matrix("ffn_gate_exps.weight"), matrix("ffn_up_exps.weight"), matrix("ffn_down_exps.weight"), matrix("ffn_gate_shexp.weight"), matrix("ffn_up_shexp.weight"), matrix("ffn_down_shexp.weight"), matrix("ffn_gate_inp_shexp.weight")};
    const AttentionLayerWeights weights{residual("hc_attn"), residual("hc_ffn"), attention, moe};
    const ReferenceAttentionLayerWeights reference{rr("hc_attn"), rr("hc_ffn"), ra, rm};
    const auto scratch = allocate_attention_layer_scratch(gpu, shape); const auto cache = allocate_qsa_cache(gpu, shape.attention);
    const auto sum = gpu.zeros(128 * 4, "current index sum"), next_sum = gpu.zeros(128 * 4, "candidate index sum"), input = gpu.zeros(10240 * 4, "attention input"), output = gpu.zeros(10240 * 4, "attention output");
    const auto setup_ms = elapsed(start); const auto allocations = gpu.memory().allocations;
    Json::Array runs; bool passed = true;
    for (uint32_t repeat = 0; repeat < repeats && passed; ++repeat) for (size_t pi = 0; pi < positions.size(); ++pi) {
        const auto position = positions[pi], token = tokens[pi % tokens.size()];
        ReferenceQsaCache cpu_cache{std::vector<uint16_t>(cache.keys.bytes / 2), std::vector<uint16_t>(cache.values.bytes / 2), std::vector<uint16_t>(cache.index_keys.bytes / 2)};
        for (size_t i = 0; i < uint64_t(position) * 512; ++i) { cpu_cache.keys[i] = finite_bf16(fixture(i, 1)); cpu_cache.values[i] = finite_bf16(fixture(i, 2)); }
        for (size_t i = 0; i < uint64_t(position / 4) * 128; ++i) cpu_cache.index_keys[i] = finite_bf16(fixture(i, 3));
        gpu.write(cache.keys, std::as_bytes(std::span(cpu_cache.keys))); gpu.write(cache.values, std::as_bytes(std::span(cpu_cache.values))); gpu.write(cache.index_keys, std::as_bytes(std::span(cpu_cache.index_keys)));
        std::vector<float> psum(128), x(2560), streams(10240);
        for (size_t i = 0; i < psum.size(); ++i) psum[i] = position % 4 ? fixture(i, 4) * float(position % 4) : 0;
        gpu.write(sum, std::as_bytes(std::span(psum)));
        const auto& emb = files.tensor("token_embd.weight"); const auto bytes = emb.bytes / emb.dimensions[1];
        dequantize(emb.type, files.tensor_bytes(emb.name).subspan(token * bytes, bytes), x);
        for (size_t branch = 0; branch < 4; ++branch) std::copy(x.begin(), x.end(), streams.begin() + branch * x.size());
        gpu.write(input, std::as_bytes(std::span(streams)));
        const auto cpu_start = Clock::now();
        const auto expected = attention_layer_reference(reference, cpu_cache, std::vector<double>(psum.begin(), psum.end()), std::vector<double>(streams.begin(), streams.end()), position, shape, policy, *mode, *normalize);
        const auto cpu_ms = elapsed(cpu_start);
        const auto before_k = prefix_bytes(gpu, cache.keys, uint64_t(position) * 1024), before_v = prefix_bytes(gpu, cache.values, uint64_t(position) * 1024),
            before_index = prefix_bytes(gpu, cache.index_keys, uint64_t(position / 4) * 256), before_sum = gpu.read(sum);
        gpu.begin(); encode_attention_layer(gpu, weights, scratch, cache, sum, next_sum, input, output, position, shape, policy, *mode, *normalize);
        const auto batch = gpu.submit();
        if (batch.dispatches != 28 + qsa_dispatches(shape.attention, position) || gpu.memory().allocations != allocations) throw FormatError("attention batch resource invariant failed");
        const bool unchanged = before_k == prefix_bytes(gpu, cache.keys, before_k.size()) && before_v == prefix_bytes(gpu, cache.values, before_v.size()) &&
            before_index == prefix_bytes(gpu, cache.index_keys, before_index.size()) && before_sum == gpu.read(sum);
        const auto comparison = compare_attention_layer(gpu, scratch, cache, next_sum, output, position, shape, expected);
        const bool valid = comparison.passed && unchanged; passed = passed && valid;
        Json::Object errors; for (const auto& [name, value] : comparison.scaled_errors) errors.emplace_back(name, number(value));
        runs.push_back(Json(Json::Object{{"pass", Json(valid)}, {"position", Json::integer(position)}, {"token_fixture", Json::integer(token)}, {"repeat", Json::integer(repeat)},
            {"accepted_state_unchanged", Json(unchanged)}, {"expert_ids_match", Json(comparison.routes_match)}, {"selected_blocks_match", Json(comparison.selected_blocks_match)},
            {"bf16_storage_matches", Json(comparison.storage_matches)}, {"bf16_reference_boundary_differences", Json::integer(comparison.bf16_reference_boundary_differences)},
            {"compared_values", Json::integer(comparison.compared_values)}, {"max_scaled_error", number(comparison.maximum)}, {"stage_scaled_errors", Json(std::move(errors))},
            {"dispatches", Json::integer(batch.dispatches)}, {"cpu_reference_ms", number(cpu_ms)}, {"wall_ms", number(batch.wall_ms)}, {"gpu_ms", number(batch.gpu_ms)}}));
        if (!passed) break;
    }
    files.assert_unchanged();
    std::cout << Json(Json::Object{{"schema", Json::integer(1)}, {"scope", Json("connected sparse-attention layer with synthetic embedding-row branches and primed BF16 cache history; not actual stacked model prefill")},
        {"layer", Json::integer(layer)}, {"router_mode", Json(*mode == RouterMode::softmax ? "softmax" : "sigmoid")}, {"normalize_selected", Json(*normalize)},
        {"query_packing", Json(*packing == QueryPacking::head_query_gate ? "head" : "flat")}, {"rotary_pairs", Json(*rotary == RotaryPairs::split_half ? "split" : "adjacent")},
        {"mode_semantics_qualified", Json(false)}, {"timings_qualified", Json(false)}, {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)},
        {"weight_setup_ms", number(setup_ms)}, {"runs", Json(std::move(runs))}, {"pass", Json(passed)}}).dump() << '\n';
    return passed ? 0 : 1;
}
} // namespace tokspur
