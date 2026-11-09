#include "tokspur/gdn_core_compare.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <array>
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
    require(r.ec == std::errc{} && r.ptr == text.data() + text.size() && v <= maximum, "invalid bounded GDN core check integer"); return v;
}
Json real(double value) {
    require(std::isfinite(value), "nonfinite GDN core diagnostic"); std::array<char, 64> bytes{};
    const auto r = std::to_chars(bytes.data(), bytes.data() + bytes.size(), value); require(r.ec == std::errc{}, "GDN core number encoding");
    return parse_json({bytes.data(), size_t(r.ptr - bytes.data())});
}
}
int gdn_core_tile_check_cli(int argc, char** argv) {
    uint32_t layer = 0, capacity = 8, steps = 8, group = 32;
    std::optional<bool> scaled; std::optional<L2Mode> l2; std::set<std::string_view> seen;
    auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    std::vector<std::filesystem::path> paths;
    for (int i = 2; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (!key.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        require(seen.insert(key).second && ++i < argc, "duplicate or missing GDN core check option"); const std::string_view value(argv[i]);
        if (key == "--layer") layer = integer(value, 2);
        else if (key == "--capacity") capacity = integer(value, 32);
        else if (key == "--steps") steps = integer(value, 32);
        else if (key == "--delta-group") group = integer(value, 128);
        else if (key == "--library") library = value;
        else if (key == "--query-scale") {
            require(value == "unit" || value == "inverse-sqrt-key-width", "unsupported GDN core query policy"); scaled = value != "unit";
        } else if (key == "--l2-mode") {
            require(value == "add" || value == "floor", "unsupported GDN core L2 policy"); l2 = value == "add" ? L2Mode::add_squared_epsilon : L2Mode::floor_squared_epsilon;
        } else throw FormatError("unknown GDN core check option");
    }
    require(paths.size() == 4 && (layer == 0 || layer == 2) && scaled && l2, "GDN core check requires four existing shards, layer0/2 and explicit query/L2 policies");
    const StateTile tile{capacity, steps}; validate_state_tile(tile);
    require(group == 32 || group == 64 || group == 128, "unsupported explicit GDN core group");
    for (size_t i = 0; i < paths.size(); ++i) for (size_t j = 0; j < i; ++j)
        require(!std::filesystem::equivalent(paths[i], paths[j]), "duplicate GDN core shard backing file");
    auto source = std::make_shared<ModelFiles>(paths); static_cast<void>(inspect_flash_next(*source));
    const auto prefix = "blk." + std::to_string(layer) + ".";
    const std::array<std::string, 9> suffix{"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight",
        "ssm_out.weight", "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight"};
    std::vector<TensorSlice> slices; uint64_t payload = 0;
    for (const auto& name : suffix) { const auto& t = source->tensor(prefix + name); slices.push_back({t.name, 0, t.bytes}); payload = checked_add(payload, t.bytes); }
    require(payload == 62'423'936, "GDN core nine-tensor payload mismatch");
    constexpr uint64_t budget = 256 * 1024 * 1024;
    MetalGraph g(library, budget); const auto imported = g.import_weights(source, slices);
    g.configure_block_gemv(1); for (uint32_t width : {1, 2, 4, 8}) g.prepare_q8_matvec_many(width); g.prepare_gdn_tiles(group);
    const auto matrix = [&](size_t i) { return GpuMatrix{imported.tensors[i], source->tensor(prefix + suffix[i]).type}; };
    const GdnWeights weights{matrix(0), matrix(1), matrix(2), matrix(3), matrix(4), imported.tensors[5], imported.tensors[6], imported.tensors[7], imported.tensors[8]};
    const auto reference_matrix = [&](size_t i) { return ReferenceMatrix{source->tensor_bytes(prefix + suffix[i]), source->tensor(prefix + suffix[i]).type}; };
    std::deque<std::vector<double>> scalars;
    const auto scalar = [&](size_t i) -> std::span<const double> {
        const auto& t = source->tensor(prefix + suffix[i]); require(t.type == 0 && t.elements <= 65536, "expected bounded F32 GDN parameter");
        std::vector<float> decoded(t.elements); dequantize(0, source->tensor_bytes(t.name), decoded); scalars.emplace_back(decoded.begin(), decoded.end()); return scalars.back();
    };
    const ReferenceGdnWeights reference{reference_matrix(0), reference_matrix(1), reference_matrix(2), reference_matrix(3), reference_matrix(4),
        scalar(5), scalar(6), scalar(7), scalar(8)};
    constexpr GdnShape shape{2560, 16, 48, 128, 128, 4};
    const GdnPolicy policy{*scaled ? 1 / std::sqrt(128.0f) : 1, 1e-6f, *l2, 1e-6f};
    std::vector<float> inputs(uint64_t(capacity) * shape.channels), state(gdn_state_count(shape)), history(gdn_qkv_count(shape) * 3);
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(int((i * 17 + 13) % 257) - 128) / 4096;
    for (size_t i = 0; i < state.size(); ++i) state[i] = float(int((i * 7 + 3) % 127) - 63) / 16384;
    for (size_t i = 0; i < history.size(); ++i) history[i] = float(int((i * 11 + 5) % 127) - 63) / 4096;
    const auto retained = g.memory().live_bytes;
    const auto result = check_gdn_core_tile(g, weights, reference, shape, policy, tile, inputs, state, history, false);
    source->assert_unchanged(); const auto memory = g.memory(); require(memory.live_bytes == retained && memory.peak_bytes <= budget, "GDN diagnostic budget/leak invariant");
    std::cout << Json(Json::Object{{"schema", Json::integer(1)}, {"pass", Json(true)}, {"scope", Json("bounded connected GDN attention core with synthetic known inputs; not residual/FFN, actual model-prefix execution or speed qualification")},
        {"layer", Json::integer(layer)}, {"capacity", Json::integer(capacity)}, {"steps", Json::integer(steps)}, {"delta_group", Json::integer(group)},
        {"query_scale", Json(*scaled ? "inverse-sqrt-key-width" : "unit")}, {"l2_mode", Json(*l2 == L2Mode::add_squared_epsilon ? "add" : "floor")},
        {"checks", Json::integer(result.checks)}, {"batches", Json::integer(result.batches)}, {"compared_values", Json::integer(result.compared_values)},
        {"maximum_scaled_error", real(result.maximum_scaled_error)}, {"reference_scaled_error_limit", real(5e-5)},
        {"serial_dispatches", Json::integer(result.serial_dispatches)}, {"tile_dispatches", Json::integer(result.tile_dispatches)},
        {"all_stage_bits_exact", Json(true)}, {"accepted_state_unchanged", Json(true)}, {"prefix_replay_exact", Json(true)}, {"tape_unchanged", Json(true)},
        {"output_sha256", Json(result.output_sha256)}, {"state_sha256", Json(result.state_sha256)}, {"history_sha256", Json(result.history_sha256)},
        {"imported_original_payload_bytes", Json::integer(payload)}, {"imported_original_backing_bytes", Json::integer(memory.imported_backing_bytes)},
        {"gpu_buffer_peak_bytes", Json::integer(memory.peak_bytes)}, {"gpu_buffer_limit_bytes", Json::integer(budget)}, {"profiling_enabled", Json(false)},
        {"full_model_loaded", Json(false)}, {"derived_weight_files_written", Json(false)}, {"timings_qualified", Json(false)}}).dump() << '\n';
    return 0;
}
} // namespace tokspur
