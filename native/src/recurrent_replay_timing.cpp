#include "tokspur/recurrent_replay_timing.hpp"
#include "tokspur/recurrent_check.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <set>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<std::byte>;
constexpr uint32_t cycle_count = 4, request_limit = 256, physical_dispatches = 45;
constexpr uint64_t budget = 4ull * 1024 * 1024 * 1024, stream_width = 4 * 2560;
constexpr uint32_t freshness_views_per_bank = 35;
constexpr uint64_t freshness_bytes_per_bank = 3'890'292, freshness_payload_limit = 8 * 1024 * 1024;
constexpr double reference_limit = 5e-5;
constexpr RecurrentShape shape{{4, 2560, 320}, {2560, 16, 48, 128, 128, 4}, {2560, 640, 512, 10}};
constexpr std::array<RecurrentReplayTimingMode, 3> modes{RecurrentReplayTimingMode::direct,
    RecurrentReplayTimingMode::compatible, RecurrentReplayTimingMode::capture};
constexpr std::array<const char*, 25> tensor_names{
    "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
    "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ffn_gate_inp.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"};
constexpr const char* timing_scope = "four independent N1 begin/encode/submit cycles; setup/host-encode/submit-wait clocks and command creation charged; validation/readback/hash excluded";
constexpr const char* checksum_framing = "tag-length u64le, UTF-8 tag, payload-length u64le, raw payload; scratch_0..31 then output/matrix/history";
constexpr const char* input_scope = "deterministic synthetic known branch input and immutable recurrent base; not stacked model-prefix outputs";
constexpr const char* allocation_scope = "owned Metal buffer and capture-resource ledger only; excludes opaque host/driver allocations";
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
double milliseconds(Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); }
uint32_t integer(std::string_view text) {
    uint32_t value = 0; const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size(), "invalid recurrent replay timing integer"); return value;
}
Json real(double value) {
    require(std::isfinite(value) && value >= 0, "invalid recurrent replay timing number"); std::array<char, 64> data{};
    const auto result = std::to_chars(data.data(), data.data() + data.size(), value);
    require(result.ec == std::errc{}, "recurrent replay timing number encoding failed");
    return parse_json({data.data(), size_t(result.ptr - data.data())});
}
const char* mode_name(RecurrentReplayTimingMode mode) {
    switch (mode) {
        case RecurrentReplayTimingMode::direct: return "direct";
        case RecurrentReplayTimingMode::compatible: return "compatible";
        case RecurrentReplayTimingMode::capture: return "capture";
    }
    throw FormatError("invalid recurrent replay timing mode");
}
size_t mode_index(RecurrentReplayTimingMode mode) {
    for (size_t i = 0; i < modes.size(); ++i) if (mode == modes[i]) return i;
    throw FormatError("invalid recurrent replay timing mode index");
}
void emit(Json value) { std::cout << value.dump(64 * 1024 - 1) << std::endl; require(bool(std::cout), "recurrent replay timing output failure"); }
std::string line() {
    std::string result; result.reserve(256); char c;
    while (std::cin.get(c)) {
        if (c == '\n') return result;
        require(result.size() < 4096, "recurrent replay timing request byte bound"); result.push_back(c);
    }
    throw FormatError(result.empty() ? "recurrent replay timing EOF before explicit close" : "recurrent replay timing unterminated request");
}
struct Digest {
    CC_SHA256_CTX context{};
    Digest() { CC_SHA256_Init(&context); }
    void update(std::span<const std::byte> bytes) {
        while (!bytes.empty()) {
            const auto count = std::min<size_t>(bytes.size(), 1024 * 1024);
            CC_SHA256_Update(&context, bytes.data(), static_cast<CC_LONG>(count)); bytes = bytes.subspan(count);
        }
    }
    void length(uint64_t value) {
        std::array<std::byte, 8> bytes{};
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte((value >> (8 * i)) & 255);
        update(bytes);
    }
    void part(std::string_view tag, std::span<const std::byte> bytes) {
        length(tag.size()); update(std::as_bytes(std::span(tag.data(), tag.size()))); length(bytes.size()); update(bytes);
    }
    std::string finish() {
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{}; CC_SHA256_Final(bytes.data(), &context);
        constexpr char digits[] = "0123456789abcdef"; std::string result; result.reserve(64);
        for (const auto byte : bytes) { result.push_back(digits[byte >> 4]); result.push_back(digits[byte & 15]); } return result;
    }
};
std::string checksum(std::span<const std::byte> bytes) { Digest digest; digest.update(bytes); return digest.finish(); }
struct Bank { RecurrentScratch scratch; GdnState candidate; GpuView output; GraphCapture replay{}; };
Bank allocate_bank(MetalGraph& graph) {
    return {allocate_recurrent_scratch(graph, shape), allocate_gdn_state(graph, shape.gdn),
        graph.zeros(stream_width * 4, "recurrent replay timing output"), {}};
}
void encode(MetalGraph& graph, const RecurrentWeights& weights, const Bank& bank, GpuView input, GdnState base, GdnPolicy policy) {
    encode_recurrent_layer(graph, weights, bank.scratch, base, bank.candidate, input, bank.output,
        shape, policy, RouterMode::softmax, true);
}
void begin(MetalGraph& graph, RecurrentReplayTimingMode mode) {
    if (mode == RecurrentReplayTimingMode::compatible) graph.begin_indirect_compatible(); else graph.begin();
}
void encode_mode(MetalGraph& graph, const RecurrentWeights& weights, const Bank& bank, GpuView input,
    GdnState base, GdnPolicy policy, RecurrentReplayTimingMode mode) {
    if (mode == RecurrentReplayTimingMode::capture) graph.execute_capture(bank.replay);
    else encode(graph, weights, bank, input, base, policy);
}
void check_batch(const BatchResult& result, RecurrentReplayTimingMode mode) {
    require(result.dispatches == physical_dispatches && result.capture_executes == (mode == RecurrentReplayTimingMode::capture ? 1 : 0) &&
        std::isfinite(result.wall_ms) && result.wall_ms > 0 && std::isfinite(result.gpu_ms) && result.gpu_ms > 0,
        "recurrent replay timing batch dispatch/time/capture invariant");
}
void same_ledger(const GraphMemory& before, const GraphMemory& after) {
    require(before.budget == budget && after.budget == budget && before.live_bytes == after.live_bytes && before.peak_bytes == after.peak_bytes &&
        before.allocations == after.allocations && before.max_buffer_bytes == after.max_buffer_bytes &&
        before.recommended_working_set_bytes == after.recommended_working_set_bytes && before.imported_backing_bytes == after.imported_backing_bytes &&
        before.profile_counter_payload_bytes == 0 && after.profile_counter_payload_bytes == 0 &&
        before.live_captures == cycle_count && after.live_captures == cycle_count &&
        before.capture_resource_bytes > 0 && before.capture_resource_bytes == after.capture_resource_bytes &&
        after.live_bytes <= budget && after.peak_bytes <= budget, "recurrent replay timing resource ledger changed");
}
void add_ledger(Json::Object& object, const GraphMemory& memory, uint64_t payload) {
    object.emplace_back("gpu_buffer_bytes", Json::integer(memory.live_bytes));
    object.emplace_back("gpu_buffer_peak_bytes", Json::integer(memory.peak_bytes));
    object.emplace_back("gpu_buffer_limit_bytes", Json::integer(memory.budget));
    object.emplace_back("gpu_allocation_count", Json::integer(memory.allocations));
    object.emplace_back("imported_original_payload_bytes", Json::integer(payload));
    object.emplace_back("imported_original_backing_bytes", Json::integer(memory.imported_backing_bytes));
    object.emplace_back("capture_resource_bytes", Json::integer(memory.capture_resource_bytes));
    object.emplace_back("live_captures", Json::integer(memory.live_captures));
}
struct Snapshot { std::array<Bytes, 32> scratch; Bytes output, matrix, history; };
std::array<std::span<const std::byte>, freshness_views_per_bank> parts(const Snapshot& value) {
    std::array<std::span<const std::byte>, freshness_views_per_bank> result;
    for (size_t i = 0; i < value.scratch.size(); ++i) result[i] = value.scratch[i];
    result[32] = value.output; result[33] = value.matrix; result[34] = value.history; return result;
}
std::array<GpuView, freshness_views_per_bank> written_views(const Bank& bank) {
    // All 32 scratch allocations have exactly their promised output extents.
    // RecurrentScratch::views excludes internal GEMV scratch and MTL padding.
    std::array<GpuView, freshness_views_per_bank> result; const auto scratch = bank.scratch.views();
    std::copy(scratch.begin(), scratch.end(), result.begin());
    result[32] = bank.output; result[33] = bank.candidate.matrix; result[34] = bank.candidate.convolution; return result;
}
Snapshot poison_payload(const Snapshot& expected) {
    uint64_t bytes = 0;
    for (const auto part : parts(expected)) bytes = checked_add(bytes, part.size());
    require(bytes == freshness_bytes_per_bank && bytes <= freshness_payload_limit, "recurrent replay freshness payload geometry/bound differs");
    Snapshot result;
    for (size_t i = 0; i < expected.scratch.size(); ++i) result.scratch[i] = recurrent_replay_poison_bytes(expected.scratch[i]);
    result.output = recurrent_replay_poison_bytes(expected.output); result.matrix = recurrent_replay_poison_bytes(expected.matrix);
    result.history = recurrent_replay_poison_bytes(expected.history); return result;
}
struct FreshnessReset { uint32_t banks = 0, views = 0; uint64_t bytes = 0; };
FreshnessReset poison_banks(MetalGraph& graph, const std::array<Bank, cycle_count>& banks, const Snapshot& payload) {
    FreshnessReset result; const auto values = parts(payload);
    for (const auto& bank : banks) {
        const auto views = written_views(bank);
        for (size_t i = 0; i < views.size(); ++i) {
            require(views[i].bytes == values[i].size(), "recurrent replay freshness write extent differs");
            graph.write(views[i], values[i]); ++result.views; result.bytes += values[i].size();
        }
        ++result.banks;
    }
    require(result.banks == cycle_count && result.views == cycle_count * freshness_views_per_bank &&
        result.bytes == cycle_count * freshness_bytes_per_bank, "recurrent replay freshness reset incomplete");
    return result;
}
void add_freshness(Json::Object& object) {
    object.emplace_back("freshness_condition", Json(std::string(recurrent_replay_freshness_condition)));
    object.emplace_back("freshness_reset_outside_timing", Json(true)); object.emplace_back("freshness_payload_preallocated", Json(true));
    object.emplace_back("freshness_views_per_bank", Json::integer(freshness_views_per_bank));
    object.emplace_back("freshness_bytes_per_bank", Json::integer(freshness_bytes_per_bank));
    object.emplace_back("freshness_banks_per_request", Json::integer(cycle_count));
    object.emplace_back("freshness_views_per_request", Json::integer(cycle_count * freshness_views_per_bank));
    object.emplace_back("freshness_bytes_per_request", Json::integer(cycle_count * freshness_bytes_per_bank));
}
Snapshot snapshot(MetalGraph& graph, const Bank& bank) {
    Snapshot result; const auto views = bank.scratch.views();
    for (size_t i = 0; i < views.size(); ++i) result.scratch[i] = graph.read(views[i]);
    result.output = graph.read(bank.output); result.matrix = graph.read(bank.candidate.matrix); result.history = graph.read(bank.candidate.convolution);
    return result;
}
void exact(const Snapshot& actual, const Snapshot& expected) {
    for (size_t i = 0; i < actual.scratch.size(); ++i) if (actual.scratch[i] != expected.scratch[i])
        throw FormatError("recurrent replay timing scratch bits differ at stage " + std::to_string(i));
    require(actual.output == expected.output && actual.matrix == expected.matrix && actual.history == expected.history,
        "recurrent replay timing output/state/history bits differ");
}
struct Hashes { std::string bank, scratch, output, matrix, history; };
Hashes hashes(const Snapshot& value) {
    Digest whole, scratch;
    for (size_t i = 0; i < value.scratch.size(); ++i) {
        const auto tag = "scratch_" + std::to_string(i); whole.part(tag, value.scratch[i]); scratch.part(tag, value.scratch[i]);
    }
    whole.part("output", value.output); whole.part("matrix", value.matrix); whole.part("history", value.history);
    return {whole.finish(), scratch.finish(), checksum(value.output), checksum(value.matrix), checksum(value.history)};
}
void add_hashes(Json::Object& object, const Hashes& value) {
    object.emplace_back("bank_sha256", Json(value.bank)); object.emplace_back("scratch_sha256", Json(value.scratch));
    object.emplace_back("output_sha256", Json(value.output)); object.emplace_back("matrix_sha256", Json(value.matrix)); object.emplace_back("history_sha256", Json(value.history));
}
struct ExtraReference { std::vector<double> up, shared_up, shared_logit; };
ExtraReference extra_reference(const ReferenceMoeWeights& weights, const ReferenceLayer& reference) {
    const auto& input = reference.ffn_read.read.input; const auto& ids = reference.ffn.routes.indices;
    require(input.size() == shape.moe.channels && ids.size() == shape.moe.selected && weights.up.bytes.size() % shape.moe.experts == 0,
        "recurrent replay timing extra reference shape");
    const auto stride = weights.up.bytes.size() / shape.moe.experts; ExtraReference result;
    result.up.reserve(uint64_t(shape.moe.selected) * shape.moe.hidden);
    for (const auto id : ids) {
        require(id < shape.moe.experts, "recurrent replay timing extra reference expert ID");
        const auto values = reference_gemv_doubleinput(weights.up.encoding, weights.up.bytes.subspan(uint64_t(id) * stride, stride),
            shape.moe.channels, shape.moe.hidden, input);
        result.up.insert(result.up.end(), values.begin(), values.end());
    }
    result.shared_up = reference_gemv_doubleinput(weights.shared_up.encoding, weights.shared_up.bytes, shape.moe.channels, shape.moe.hidden, input);
    result.shared_logit = reference_gemv_doubleinput(weights.shared_router.encoding, weights.shared_router.bytes, shape.moe.channels, 1, input);
    return result;
}
double numeric(MetalGraph& graph, const Bank& bank, const ReferenceLayer& reference, const ExtraReference& extra) {
    const auto comparison = compare_recurrent_layer(graph, bank.scratch, bank.candidate, bank.output, reference);
    require(comparison.routes_match && comparison.passed && comparison.scaled_errors.size() == 31,
        "recurrent replay timing ordinary31 reference failed");
    double maximum = comparison.maximum;
    const auto close = [&](GpuView view, std::span<const double> expected) {
        const auto bytes = graph.read(view); require(bytes.size() == expected.size() * sizeof(float), "recurrent replay timing extra reference byte count");
        std::vector<float> actual(expected.size()); std::memcpy(actual.data(), bytes.data(), bytes.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "recurrent replay timing nonfinite extra reference");
            const auto error = std::abs(double(actual[i]) - expected[i]) / std::max(1.0, std::abs(expected[i]));
            require(error <= reference_limit, "recurrent replay timing extra double reference exceeds5e-5"); maximum = std::max(maximum, error);
        }
    };
    close(bank.scratch.moe.up, extra.up); close(bank.scratch.moe.shared_up, extra.shared_up); close(bank.scratch.moe.shared_logit, extra.shared_logit);
    require(std::isfinite(maximum) && maximum <= reference_limit, "recurrent replay timing numeric qualification failed"); return maximum;
}
Json capability(GpuView view) {
    return Json(Json::Object{{"owner", Json::integer(view.owner)}, {"slot", Json::integer(view.slot)},
        {"offset", Json::integer(view.offset)}, {"bytes", Json::integer(view.bytes)}});
}
Json capture_binding(const Bank& bank, uint32_t index, GpuView input, GdnState base) {
    return Json(Json::Object{{"bank", Json::integer(index)}, {"capture_owner", Json::integer(bank.replay.owner)}, {"capture_slot", Json::integer(bank.replay.slot)},
        {"base_matrix", capability(base.matrix)}, {"base_history", capability(base.convolution)},
        {"candidate_matrix", capability(bank.candidate.matrix)}, {"candidate_history", capability(bank.candidate.convolution)},
        {"input", capability(input)}, {"output", capability(bank.output)}});
}
struct Timing { BatchResult batch{}; double setup = 0, encode = 0, submit_wait = 0; };
} // namespace

int recurrent_replay_timing_cli(int argc, char** argv) {
    uint64_t completed = 0;
    try {
        uint32_t layer = 0; std::set<std::string_view> seen; std::vector<std::filesystem::path> paths;
        auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
        for (int i = 2; i < argc; ++i) {
            const std::string_view key(argv[i]);
            if (!key.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
            require(seen.insert(key).second && ++i < argc, "duplicate/missing recurrent replay timing option"); const std::string_view value(argv[i]);
            if (key == "--layer") layer = integer(value);
            else if (key == "--library") library = value;
            else throw FormatError("unknown recurrent replay timing option");
        }
        require(seen.contains("--layer") && (layer == 0 || layer == 2) && paths.size() == 4,
            "recurrent replay timing requires explicit layer0/2 and four existing shards");
        for (size_t i = 0; i < paths.size(); ++i) for (size_t j = 0; j < i; ++j)
            require(!std::filesystem::equivalent(paths[i], paths[j]), "duplicate recurrent replay timing shard backing");
        const auto setup_started = Clock::now(); auto source = std::make_shared<ModelFiles>(paths);
        static_cast<void>(inspect_flash_next(*source)); const auto prefix = "blk." + std::to_string(layer) + ".";
        std::vector<TensorSlice> slices; Json::Array weight_pins; uint64_t payload = 0;
        for (const auto* suffix : tensor_names) {
            const auto& tensor = source->tensor(prefix + suffix); slices.push_back({tensor.name, 0, tensor.bytes}); payload = checked_add(payload, tensor.bytes);
            Json::Array dimensions; for (const auto value : tensor.dimensions) dimensions.push_back(Json::integer(value));
            weight_pins.emplace_back(Json::Object{{"name", Json(tensor.name)}, {"type", Json::integer(tensor.type)}, {"shape", Json(std::move(dimensions))},
                {"bytes", Json::integer(tensor.bytes)}, {"sha256", Json(checksum(source->tensor_bytes(tensor.name)))}});
        }
        require(payload == (layer == 0 ? 1'660'099'456ull : 2'131'958'656ull), "recurrent replay timing25 tensor payload differs");
        MetalGraph graph(library, budget); const auto imported = graph.import_weights(source, slices);
        graph.configure_small_gemv(16); graph.configure_block_gemv(3);
        const auto matrix = [&](size_t i) { return GpuMatrix{imported.tensors[i], source->tensor(prefix + tensor_names[i]).type}; };
        const auto reference_matrix = [&](size_t i) { return ReferenceMatrix{source->tensor_bytes(prefix + tensor_names[i]), source->tensor(prefix + tensor_names[i]).type}; };
        std::deque<std::vector<double>> scalars;
        const auto scalar = [&](size_t i) -> std::span<const double> {
            const auto& tensor = source->tensor(prefix + tensor_names[i]); require(tensor.type == 0 && tensor.elements <= 65536, "expected bounded recurrent replay timing F32 parameter");
            std::vector<float> values(tensor.elements); dequantize(0, source->tensor_bytes(tensor.name), values);
            scalars.emplace_back(values.begin(), values.end()); return scalars.back();
        };
        const auto residual = [&](size_t i) { return ResidualWeights{imported.tensors[i], matrix(i + 1), matrix(i + 2), matrix(i + 3)}; };
        const auto reference_residual = [&](size_t i) { return ReferenceResidualWeights{scalar(i), reference_matrix(i + 1), reference_matrix(i + 2), reference_matrix(i + 3)}; };
        const GdnWeights core{matrix(8), matrix(9), matrix(10), matrix(11), matrix(12), imported.tensors[13], imported.tensors[14], imported.tensors[15], imported.tensors[16]};
        const ReferenceGdnWeights reference_core{reference_matrix(8), reference_matrix(9), reference_matrix(10), reference_matrix(11), reference_matrix(12), scalar(13), scalar(14), scalar(15), scalar(16)};
        const MoeWeights moe{matrix(17), matrix(18), matrix(19), matrix(20), matrix(21), matrix(22), matrix(23), matrix(24)};
        const ReferenceMoeWeights reference_moe{reference_matrix(17), reference_matrix(18), reference_matrix(19), reference_matrix(20), reference_matrix(21), reference_matrix(22), reference_matrix(23), reference_matrix(24)};
        const RecurrentWeights weights{residual(0), residual(4), core, moe};
        const ReferenceLayerWeights references{reference_residual(0), reference_residual(4), reference_core, reference_moe};
        const GdnPolicy policy{1 / std::sqrt(128.0f), 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
        std::vector<float> inputs(stream_width), state(gdn_state_count(shape.gdn)), history(gdn_qkv_count(shape.gdn) * 3);
        for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(int((i * 17 + 13) % 257) - 128) / 4096;
        for (size_t i = 0; i < state.size(); ++i) state[i] = float(int((i * 7 + 3) % 127) - 63) / 16384;
        for (size_t i = 0; i < history.size(); ++i) history[i] = float(int((i * 11 + 5) % 127) - 63) / 4096;
        const auto input = graph.upload(std::as_bytes(std::span(inputs)), "recurrent replay immutable input");
        const GdnState base{graph.upload(std::as_bytes(std::span(state)), "recurrent replay immutable base matrix"),
            graph.upload(std::as_bytes(std::span(history)), "recurrent replay immutable base history")};
        const auto saved_input = graph.read(input), saved_matrix = graph.read(base.matrix), saved_history = graph.read(base.convolution);
        const auto immutable_unchanged = [&] {
            require(graph.read(input) == saved_input && graph.read(base.matrix) == saved_matrix && graph.read(base.convolution) == saved_history,
                "recurrent replay timing changed immutable input or accepted base");
        };
        std::array<std::array<Bank, cycle_count>, modes.size()> banks{};
        for (auto& mode_banks : banks) for (auto& bank : mode_banks) bank = allocate_bank(graph);
        Json::Array capture_bindings;
        for (uint32_t i = 0; i < cycle_count; ++i) {
            auto& bank = banks[2][i]; graph.begin_capture(physical_dispatches);
            encode(graph, weights, bank, input, base, policy); bank.replay = graph.finish_capture();
            capture_bindings.push_back(capture_binding(bank, i, input, base));
        }
        immutable_unchanged(); const auto fixed_memory = graph.memory();
        require(graph.kernel_policy() == std::array<uint32_t, 2>{16, 3} && fixed_memory.live_captures == cycle_count &&
            fixed_memory.capture_resource_bytes > 0 && fixed_memory.profile_counter_payload_bytes == 0 && fixed_memory.imported_backing_bytes >= payload,
            "recurrent replay timing preparation policy/resources differ");
        same_ledger(fixed_memory, fixed_memory);
        // All scalar reference work and all pipeline/bank first use precede ready.
        const auto reference = recurrent_reference(references, std::vector<double>(inputs.begin(), inputs.end()),
            std::vector<double>(state.begin(), state.end()), std::vector<double>(history.begin(), history.end()), shape, policy, RouterMode::softmax, true);
        const auto extra = extra_reference(reference_moe, reference);
        Snapshot expected; Hashes expected_hashes; Json::Array validations; double reference_error = 0;
        for (size_t mi = 0; mi < modes.size(); ++mi) for (uint32_t i = 0; i < cycle_count; ++i) {
            const auto& bank = banks[mi][i]; begin(graph, modes[mi]); encode_mode(graph, weights, bank, input, base, policy, modes[mi]);
            check_batch(graph.submit(), modes[mi]); same_ledger(fixed_memory, graph.memory()); immutable_unchanged();
            const auto actual = snapshot(graph, bank); const auto error = numeric(graph, bank, reference, extra);
            if (mi == 0 && i == 0) { expected = actual; expected_hashes = hashes(expected); } else exact(actual, expected);
            reference_error = std::max(reference_error, error); const auto h = hashes(actual);
            require(h.bank == expected_hashes.bank, "recurrent replay timing validated bank digest differs");
            Json::Object item{{"mode", Json(mode_name(modes[mi]))}, {"bank", Json::integer(i)}, {"outputs_exact", Json(true)},
                {"route_ids_exact", Json(true)}, {"reference_max_scaled_error", real(error)}}; add_hashes(item, h); validations.emplace_back(std::move(item));
        }
        // Precompute a bounded complement once. Per-request writes below must
        // not allocate payloads, change captures, or leave old expected results.
        const auto poison = poison_payload(expected);
        source->assert_unchanged(); immutable_unchanged(); same_ledger(fixed_memory, graph.memory()); const auto ready_memory = graph.memory();
        Json::Object ready{{"event", Json("ready")}, {"schema", Json::integer(1)}, {"layer", Json::integer(layer)}, {"tokens", Json::integer(1)},
            {"cycles", Json::integer(cycle_count)}, {"banks_per_mode", Json::integer(cycle_count)}, {"total_banks", Json::integer(12)},
            {"modes", Json(Json::Array{Json("direct"), Json("compatible"), Json("capture")})}, {"small_gemv_column_splits", Json::integer(16)},
            {"effective_block_gemv_formats", Json::integer(3)}, {"dispatches_per_cycle", Json::integer(physical_dispatches)},
            {"router_mode", Json("softmax")}, {"normalize_selected", Json(true)}, {"query_scale", Json("inverse-sqrt-key-width")},
            {"query_scale_numeric", real(policy.query_scale)}, {"l2_mode", Json("add")}, {"l2_epsilon", real(policy.l2_epsilon)},
            {"rms_epsilon", real(policy.rms_epsilon)}, {"profiling_enabled", Json(false)},
            {"all_validation_outputs_exact", Json(true)}, {"all_routes_match", Json(true)}, {"numeric_reference_stages", Json::integer(34)},
            {"exact_scratch_views", Json::integer(32)}, {"reference_scaled_error_limit", real(reference_limit)}, {"reference_max_scaled_error", real(reference_error)},
            {"validations", Json(std::move(validations))}, {"capture_bindings", Json(std::move(capture_bindings))}, {"weight_pins", Json(std::move(weight_pins))},
            {"input_sha256", Json(checksum(saved_input))}, {"base_matrix_sha256", Json(checksum(saved_matrix))}, {"base_history_sha256", Json(checksum(saved_history))},
            {"full_model_loaded", Json(false)}, {"derived_weight_files_written", Json(false)}, {"timing_scope", Json(timing_scope)},
            {"checksum_framing", Json(checksum_framing)}, {"input_scope", Json(input_scope)}, {"gpu_allocation_scope", Json(allocation_scope)},
            {"setup_ms", real(milliseconds(Clock::now() - setup_started))}};
        add_freshness(ready); add_ledger(ready, ready_memory, payload); emit(Json(std::move(ready)));
        while (true) {
            const auto request = parse_recurrent_replay_timing_request(line()); source->assert_unchanged();
            same_ledger(ready_memory, graph.memory());
            require(graph.kernel_policy() == std::array<uint32_t, 2>{16, 3}, "recurrent replay timing kernel policy changed");
            if (request.close) {
                immutable_unchanged(); Json::Object end{{"event", Json("end")}, {"schema", Json::integer(1)}, {"layer", Json::integer(layer)},
                    {"requests_completed", Json::integer(completed)}, {"cycles_completed", Json::integer(completed * cycle_count)},
                    {"profiling_enabled", Json(false)}, {"pass", Json(true)}};
                add_ledger(end, graph.memory(), payload); emit(Json(std::move(end))); return 0;
            }
            require(completed < request_limit, "recurrent replay timing request count bound");
            const auto& selected = banks[mode_index(request.mode)]; immutable_unchanged();
            const auto before = graph.memory(); std::array<Timing, cycle_count> measured{};
            // A fresh condition, not comparable by pooling historical cohorts:
            // overwrite only logical mutable output bytes, while idle and
            // outside every measured clock. Captures and input/base stay fixed.
            const auto reset = poison_banks(graph, selected, poison);
            same_ledger(before, graph.memory());
            // Entire measured region: four independent begin/encode/submit
            // cycles, four distinct candidates, same immutable base. No owned
            // output read/reference/hash/upload/allocation/policy work occurs.
            // Fixed clock instrumentation and ordinary completion/error checks
            // remain charged equally to A/B/C, including command creation.
            const auto started = Clock::now();
            for (uint32_t i = 0; i < cycle_count; ++i) {
                const auto setup_start = Clock::now(); begin(graph, request.mode); const auto encode_start = Clock::now();
                encode_mode(graph, weights, selected[i], input, base, policy, request.mode); const auto submit_start = Clock::now();
                measured[i].batch = graph.submit(); const auto submit_end = Clock::now();
                measured[i].setup = milliseconds(encode_start - setup_start);
                measured[i].encode = milliseconds(submit_start - encode_start);
                measured[i].submit_wait = milliseconds(submit_end - submit_start);
            }
            const auto finished = Clock::now();
            const auto after = graph.memory(); same_ledger(before, after); same_ledger(ready_memory, after);
            require(graph.kernel_policy() == std::array<uint32_t, 2>{16, 3}, "recurrent replay timing policy changed during cycles");
            double setup_ms = 0, encode_ms = 0, submit_ms = 0, batch_ms = 0, gpu_ms = 0; uint64_t dispatch_count = 0, capture_count = 0;
            Json::Array bank_results;
            for (uint32_t i = 0; i < cycle_count; ++i) {
                const auto& timing = measured[i]; check_batch(timing.batch, request.mode);
                require(std::isfinite(timing.setup) && timing.setup >= 0 && std::isfinite(timing.encode) && timing.encode > 0 &&
                    std::isfinite(timing.submit_wait) && timing.submit_wait > 0, "recurrent replay timing host clock invariant");
                setup_ms += timing.setup; encode_ms += timing.encode; submit_ms += timing.submit_wait;
                batch_ms += timing.batch.wall_ms; gpu_ms += timing.batch.gpu_ms; dispatch_count += timing.batch.dispatches; capture_count += timing.batch.capture_executes;
                const auto actual = snapshot(graph, selected[i]); exact(actual, expected); const auto h = hashes(actual);
                require(h.bank == expected_hashes.bank, "recurrent replay timing postcheck bank digest differs");
                Json::Object item{{"bank", Json::integer(i)}, {"outputs_exact", Json(true)}, {"route_ids_exact", Json(true)}};
                add_hashes(item, h); bank_results.emplace_back(std::move(item));
            }
            source->assert_unchanged(); immutable_unchanged(); same_ledger(ready_memory, graph.memory()); ++completed;
            Json::Object result{{"event", Json("done")}, {"schema", Json::integer(1)}, {"request_index", Json::integer(completed)},
                {"layer", Json::integer(layer)}, {"tokens", Json::integer(1)}, {"mode", Json(mode_name(request.mode))}, {"cycles", Json::integer(cycle_count)},
                {"banks_per_mode", Json::integer(cycle_count)}, {"small_gemv_column_splits", Json::integer(16)}, {"effective_block_gemv_formats", Json::integer(3)},
                {"profiling_enabled", Json(false)}, {"dispatches", Json::integer(dispatch_count)}, {"dispatches_per_cycle", Json::integer(physical_dispatches)},
                {"capture_executes", Json::integer(capture_count)}, {"command_setup_wall_ms_per_cycle", real(setup_ms / cycle_count)},
                {"host_encode_wall_ms_per_cycle", real(encode_ms / cycle_count)}, {"submit_wait_wall_ms_per_cycle", real(submit_ms / cycle_count)},
                {"encode_submit_wall_ms_per_cycle", real(milliseconds(finished - started) / cycle_count)},
                {"batch_wall_ms_per_cycle", real(batch_ms / cycle_count)}, {"gpu_ms_per_cycle", real(gpu_ms / cycle_count)},
                {"outputs_exact", Json(true)}, {"route_ids_exact", Json(true)}, {"immutable_inputs_unchanged", Json(true)},
                {"reference_max_scaled_error", real(reference_error)}, {"gpu_buffer_allocations_during_request", Json::integer(0)}, {"banks", Json(std::move(bank_results))}};
            add_freshness(result); result.emplace_back("freshness_reset_banks", Json::integer(reset.banks));
            result.emplace_back("freshness_reset_views", Json::integer(reset.views)); result.emplace_back("freshness_reset_bytes", Json::integer(reset.bytes));
            add_ledger(result, after, payload); emit(Json(std::move(result)));
        }
    } catch (const std::exception& error) {
        const std::string message(error.what());
        try { emit(Json(Json::Object{{"event", Json("error")}, {"schema", Json::integer(1)}, {"requests_completed", Json::integer(completed)},
            {"message", Json(message.substr(0, 2048))}})); } catch (const std::exception&) {}
        std::cerr << "Owned recurrent replay timing worker failed: " << error.what() << '\n'; return 1;
    }
}
} // namespace tokspur
