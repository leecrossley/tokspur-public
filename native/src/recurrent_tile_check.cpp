#include "tokspur/recurrent_tile.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <charconv>
#include <cmath>
#include <deque>
#include <iostream>
#include <optional>
#include <set>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
uint32_t integer(std::string_view text, uint32_t maximum) {
    uint32_t v = 0; const auto r = std::from_chars(text.data(), text.data() + text.size(), v);
    require(r.ec == std::errc{} && r.ptr == text.data() + text.size() && v <= maximum, "invalid recurrent tile integer"); return v;
}
Json real(double value) {
    require(std::isfinite(value), "nonfinite recurrent tile diagnostic"); std::array<char, 64> buffer{};
    const auto r = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value); require(r.ec == std::errc{}, "recurrent tile number encoding");
    return parse_json({buffer.data(), size_t(r.ptr - buffer.data())});
}
}
int recurrent_tile_check_cli(int argc, char** argv) {
    uint32_t layer = 0, capacity = 8, steps = 8, group = 32, blocks = 1, splits = 0;
    bool reuse_residual = false;
    std::optional<RouterMode> mode; std::optional<bool> normalize, scaled; std::optional<L2Mode> l2;
    std::set<std::string_view> seen; std::vector<std::filesystem::path> paths;
    auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    for (int i = 2; i < argc; ++i) {
        const std::string_view key(argv[i]); if (!key.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        require(seen.insert(key).second && ++i < argc, "duplicate or missing recurrent tile option"); const std::string_view value(argv[i]);
        if (key == "--layer") layer = integer(value, 2);
        else if (key == "--capacity") capacity = integer(value, 32);
        else if (key == "--steps") steps = integer(value, 32);
        else if (key == "--delta-group") group = integer(value, 128);
        else if (key == "--block-formats") blocks = integer(value, 3);
        else if (key == "--small-gemv-splits") { splits = integer(value, 32); require(splits == 0 || splits == 1 || splits == 4 || splits == 8 || splits == 16 || splits == 32, "unsupported recurrent tile split count"); }
        else if (key == "--reuse-residual-projections") { require(value == "true" || value == "false", "expected residual reuse boolean"); reuse_residual = value == "true"; }
        else if (key == "--library") library = value;
        else if (key == "--router-mode") { require(value == "softmax" || value == "sigmoid", "unsupported recurrent tile router"); mode = value == "softmax" ? RouterMode::softmax : RouterMode::sigmoid; }
        else if (key == "--normalize-selected") { require(value == "true" || value == "false", "expected recurrent tile normalization boolean"); normalize = value == "true"; }
        else if (key == "--query-scale") { require(value == "unit" || value == "inverse-sqrt-key-width", "unsupported recurrent tile query policy"); scaled = value != "unit"; }
        else if (key == "--l2-mode") { require(value == "add" || value == "floor", "unsupported recurrent tile L2 mode"); l2 = value == "add" ? L2Mode::add_squared_epsilon : L2Mode::floor_squared_epsilon; }
        else throw FormatError("unknown recurrent tile option");
    }
    require(paths.size() == 4 && (layer == 0 || layer == 2) && mode && normalize && scaled && l2,
        "recurrent tile check requires four existing shards, layer0/2 and explicit router/normalization/query/L2 policies");
    const StateTile tile{capacity, steps}; validate_state_tile(tile); require(group == 32 || group == 64 || group == 128, "unsupported recurrent tile group");
    require(blocks == 1 || blocks == 3, "recurrent tile block formats must be1 or3");
    for (size_t i = 0; i < paths.size(); ++i) for (size_t j = 0; j < i; ++j) require(!std::filesystem::equivalent(paths[i], paths[j]), "duplicate recurrent tile shard backing");
    auto source = std::make_shared<ModelFiles>(paths); static_cast<void>(inspect_flash_next(*source)); const auto prefix = "blk." + std::to_string(layer) + ".";
    const std::array<std::string, 25> names{"hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
        "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight", "attn_qkv.weight", "attn_gate.weight",
        "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight", "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight",
        "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"};
    std::vector<TensorSlice> slices; uint64_t payload = 0;
    for (const auto& name : names) { const auto& t = source->tensor(prefix + name); slices.push_back({t.name, 0, t.bytes}); payload = checked_add(payload, t.bytes); }
    require(payload == (layer == 0 ? 1'660'099'456ull : 2'131'958'656ull), "recurrent tile25 tensor payload differs");
    constexpr uint64_t budget = 4ull * 1024 * 1024 * 1024;
    MetalGraph g(library, budget); const auto imported = g.import_weights(source, slices);
    g.configure_small_gemv(splits); g.configure_block_gemv(blocks); for (uint32_t width : {1, 2, 4, 8}) g.prepare_q8_matvec_many(width); g.prepare_gdn_tiles(group);
    const auto matrix = [&](size_t i) { return GpuMatrix{imported.tensors[i], source->tensor(prefix + names[i]).type}; };
    const auto reference_matrix = [&](size_t i) { return ReferenceMatrix{source->tensor_bytes(prefix + names[i]), source->tensor(prefix + names[i]).type}; };
    std::deque<std::vector<double>> scalars;
    const auto scalar = [&](size_t i) -> std::span<const double> {
        const auto& t = source->tensor(prefix + names[i]); require(t.type == 0 && t.elements <= 65536, "expected bounded recurrent F32 parameter");
        std::vector<float> v(t.elements); dequantize(0, source->tensor_bytes(t.name), v); scalars.emplace_back(v.begin(), v.end()); return scalars.back();
    };
    const auto residual = [&](size_t i) { return ResidualWeights{imported.tensors[i], matrix(i + 1), matrix(i + 2), matrix(i + 3)}; };
    const auto reference_residual = [&](size_t i) { return ReferenceResidualWeights{scalar(i), reference_matrix(i + 1), reference_matrix(i + 2), reference_matrix(i + 3)}; };
    const GdnWeights core{matrix(8), matrix(9), matrix(10), matrix(11), matrix(12), imported.tensors[13], imported.tensors[14], imported.tensors[15], imported.tensors[16]};
    const ReferenceGdnWeights reference_core{reference_matrix(8), reference_matrix(9), reference_matrix(10), reference_matrix(11), reference_matrix(12), scalar(13), scalar(14), scalar(15), scalar(16)};
    const MoeWeights moe{matrix(17), matrix(18), matrix(19), matrix(20), matrix(21), matrix(22), matrix(23), matrix(24)};
    const ReferenceMoeWeights reference_moe{reference_matrix(17), reference_matrix(18), reference_matrix(19), reference_matrix(20), reference_matrix(21), reference_matrix(22), reference_matrix(23), reference_matrix(24)};
    const RecurrentWeights weights{residual(0), residual(4), core, moe};
    Json::Array residual_encodings; uint32_t reused_projections = 0;
    for (const auto& r : {weights.attention_read, weights.ffn_read}) for (const auto& m : {r.down, r.up}) {
        residual_encodings.push_back(Json::integer(m.encoding));
        if (reuse_residual && m.encoding == 8) ++reused_projections;
    }
    const ReferenceLayerWeights reference{reference_residual(0), reference_residual(4), reference_core, reference_moe};
    constexpr RecurrentShape shape{{4, 2560, 320}, {2560, 16, 48, 128, 128, 4}, {2560, 640, 512, 10}};
    const GdnPolicy policy{*scaled ? 1 / std::sqrt(128.0f) : 1, 1e-6f, *l2, 1e-6f};
    std::vector<float> inputs(uint64_t(capacity) * 4 * 2560), state(gdn_state_count(shape.gdn)), history(gdn_qkv_count(shape.gdn) * 3);
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(int((i * 17 + 13) % 257) - 128) / 4096;
    for (size_t i = 0; i < state.size(); ++i) state[i] = float(int((i * 7 + 3) % 127) - 63) / 16384;
    for (size_t i = 0; i < history.size(); ++i) history[i] = float(int((i * 11 + 5) % 127) - 63) / 4096;
    const auto retained = g.memory().live_bytes;
    const auto result = check_recurrent_tile(g, weights, reference, shape, policy, *mode, *normalize, tile, inputs, state, history, false, reuse_residual);
    source->assert_unchanged(); const auto memory = g.memory(); require(memory.live_bytes == retained && memory.peak_bytes <= budget, "recurrent tile diagnostic budget/leak invariant");
    std::cout << Json(Json::Object{{"schema", Json::integer(1)}, {"pass", Json(true)}, {"scope", Json("one ordinary recurrent layer, deterministic synthetic known branch inputs; not actual stacked model-prefix execution or speed qualification")},
        {"layer", Json::integer(layer)}, {"capacity", Json::integer(capacity)}, {"steps", Json::integer(steps)}, {"delta_group", Json::integer(group)},
        {"block_gemv_formats", Json::integer(blocks)},
        {"small_gemv_column_splits", Json::integer(splits)}, {"reuse_residual_projections", Json(reuse_residual)},
        {"residual_projection_encodings", Json(std::move(residual_encodings))}, {"reused_residual_projection_count", Json::integer(reused_projections)},
        {"router_mode", Json(*mode == RouterMode::softmax ? "softmax" : "sigmoid")}, {"normalize_selected", Json(*normalize)},
        {"query_scale", Json(*scaled ? "inverse-sqrt-key-width" : "unit")}, {"l2_mode", Json(*l2 == L2Mode::add_squared_epsilon ? "add" : "floor")},
        {"checks", Json::integer(result.checks)}, {"batches", Json::integer(result.batches)}, {"compared_values", Json::integer(result.compared_values)},
        {"maximum_scaled_error", real(result.maximum_scaled_error)}, {"reference_scaled_error_limit", real(5e-5)},
        {"exact_scratch_stages", Json::integer(32)}, {"double_stage_mapping", Json("29 token-local stages plus2 final-state stages; expert IDs separate")},
        {"serial_dispatches", Json::integer(result.serial_dispatches)}, {"tile_dispatches", Json::integer(result.tile_dispatches)},
        {"all_stage_bits_exact", Json(true)}, {"expert_ids_exact", Json(true)}, {"accepted_state_unchanged", Json(true)}, {"prefix_replay_exact", Json(true)}, {"tape_outputs_unchanged", Json(true)},
        {"output_sha256", Json(result.output_sha256)}, {"state_sha256", Json(result.state_sha256)}, {"history_sha256", Json(result.history_sha256)},
        {"imported_original_payload_bytes", Json::integer(payload)}, {"imported_original_backing_bytes", Json::integer(memory.imported_backing_bytes)},
        {"gpu_buffer_peak_bytes", Json::integer(memory.peak_bytes)}, {"gpu_buffer_limit_bytes", Json::integer(budget)}, {"profiling_enabled", Json(false)},
        {"full_model_loaded", Json(false)}, {"derived_weight_files_written", Json(false)}, {"timings_qualified", Json(false)}}).dump() << '\n';
    return 0;
}
} // namespace tokspur
