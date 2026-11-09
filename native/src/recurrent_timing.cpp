#include "tokspur/recurrent_timing.hpp"
#include "tokspur/recurrent_tile.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <set>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<std::byte>;
constexpr uint32_t capacity = 32, cycle_count = 4, request_limit = 256;
constexpr uint64_t budget = 4ull * 1024 * 1024 * 1024;
constexpr float sentinel = -12345.25f;
constexpr RecurrentShape shape{{4, 2560, 320}, {2560, 16, 48, 128, 128, 4}, {2560, 640, 512, 10}};
constexpr uint64_t stream_width = 4 * 2560;
constexpr std::array<uint32_t, 4> token_counts{2, 8, 17, 32};
constexpr std::array<uint32_t, 2> block_masks{1, 3};
constexpr std::array<const char*, 25> tensor_names{
    "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
    "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ffn_gate_inp.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"};
constexpr const char* timing_scope = "four independent begin/encode/submit cycles; command creation charged; validation/readback/hash excluded";
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
double milliseconds(Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); }
uint32_t integer(std::string_view text) {
    uint32_t v = 0; const auto r = std::from_chars(text.data(), text.data() + text.size(), v);
    require(r.ec == std::errc{} && r.ptr == text.data() + text.size(), "invalid recurrent timing integer"); return v;
}
Json real(double value) {
    require(std::isfinite(value) && value >= 0, "nonfinite/negative recurrent timing number"); std::array<char, 64> buffer{};
    const auto r = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    require(r.ec == std::errc{}, "recurrent timing number encoding");
    return parse_json({buffer.data(), size_t(r.ptr - buffer.data())});
}
void emit(Json value) { std::cout << value.dump(64 * 1024 - 1) << std::endl; require(bool(std::cout), "recurrent timing output failure"); }
std::string line() {
    std::string result; result.reserve(256); char c;
    while (std::cin.get(c)) {
        if (c == '\n') return result;
        require(result.size() < 4096, "recurrent timing request byte bound"); result.push_back(c);
    }
    throw FormatError(result.empty() ? "recurrent timing EOF before explicit close" : "recurrent timing unterminated request");
}
std::string hex_digest(const std::array<unsigned char, CC_SHA256_DIGEST_LENGTH>& digest) {
    constexpr char digits[] = "0123456789abcdef"; std::string result; result.reserve(64);
    for (const auto byte : digest) { result.push_back(digits[byte >> 4]); result.push_back(digits[byte & 15]); } return result;
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
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte((value >> (i * 8)) & 255);
        update(bytes);
    }
    // Every framed part is tag-length(u64le), tag UTF-8, payload-length(u64le),
    // payload bytes. Scratch parts are scratch_0 through scratch_31 in order.
    void part(std::string_view tag, std::span<const std::byte> bytes) {
        length(tag.size()); update(std::as_bytes(std::span(tag.data(), tag.size()))); length(bytes.size()); update(bytes);
    }
    std::string finish() { std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256_Final(digest.data(), &context); return hex_digest(digest); }
};
std::string checksum(std::span<const std::byte> bytes) { Digest digest; digest.update(bytes); return digest.finish(); }
uint64_t dispatches(uint32_t tokens, RecurrentTimingMode mode) {
    if (mode == RecurrentTimingMode::serial) return uint64_t(tokens) * 40;
    uint32_t chunks = tokens / 8; for (auto tail = tokens % 8; tail; tail >>= 1) chunks += tail & 1;
    return uint64_t(tokens) * 35 + chunks * 3 + 2;
}
struct Step {
    RecurrentScratch scratch;
    GdnState current, next;
    GpuView input, output;
};
struct Bank {
    RecurrentScratch scratch;
    GpuView output;
    std::array<GdnState, 2> states{};
    std::array<Step, capacity> steps{};
};
Bank allocate_bank(MetalGraph& g, GpuView input, GdnState base, bool serial) {
    Bank result{}; result.scratch = allocate_recurrent_tile_scratch(g, shape, capacity);
    result.output = g.zeros(uint64_t(capacity) * stream_width * 4, "recurrent timing output");
    result.states[0] = allocate_gdn_state(g, shape.gdn);
    if (serial) result.states[1] = allocate_gdn_state(g, shape.gdn);
    if (serial) for (uint32_t i = 0; i < capacity; ++i) {
        result.steps[i] = {recurrent_tile_token(g, result.scratch, shape, capacity, i), i ? result.states[(i - 1) % 2] : base,
            result.states[i % 2], g.slice(input, uint64_t(i) * stream_width * 4, stream_width * 4),
            g.slice(result.output, uint64_t(i) * stream_width * 4, stream_width * 4)};
    }
    return result;
}
void clear_bank(MetalGraph& g, const Bank& bank) {
    for (auto view : bank.scratch.views()) { const std::vector<float> values(view.bytes / 4, sentinel); g.write(view, std::as_bytes(std::span(values))); }
    const std::vector<float> output(bank.output.bytes / 4, sentinel); g.write(bank.output, std::as_bytes(std::span(output)));
}
BatchResult execute(MetalGraph& g, const RecurrentWeights& weights, const Bank& bank, GpuView input,
    GdnState base, GdnPolicy policy, uint32_t tokens, RecurrentTimingMode mode) {
    g.begin();
    if (mode == RecurrentTimingMode::serial) {
        for (uint32_t i = 0; i < tokens; ++i) {
            const auto& step = bank.steps[i];
            encode_recurrent_layer(g, weights, step.scratch, step.current, step.next, step.input, step.output,
                shape, policy, RouterMode::softmax, true);
        }
    } else encode_recurrent_layer_tile(g, weights, bank.scratch, base, bank.states[0], input, bank.output,
        shape, policy, RouterMode::softmax, true, {capacity, tokens});
    return g.submit();
}
struct Snapshot {
    std::array<Bytes, 32> scratch;
    Bytes output, matrix, history, auxiliary_matrix, auxiliary_history;
};
Snapshot capture(MetalGraph& g, const Bank& bank, uint32_t tokens, RecurrentTimingMode mode) {
    Snapshot result; const auto views = bank.scratch.views();
    for (size_t i = 0; i < views.size(); ++i) result.scratch[i] = g.read(g.slice(views[i], 0, views[i].bytes / capacity * tokens));
    result.output = g.read(g.slice(bank.output, 0, stream_width * tokens * 4));
    const auto final = bank.states[mode == RecurrentTimingMode::serial ? (tokens - 1) % 2 : 0];
    result.matrix = g.read(final.matrix); result.history = g.read(final.convolution);
    if (mode == RecurrentTimingMode::serial) {
        const auto auxiliary = bank.states[tokens % 2];
        result.auxiliary_matrix = g.read(auxiliary.matrix); result.auxiliary_history = g.read(auxiliary.convolution);
    }
    return result;
}
void compare(const Snapshot& actual, const Snapshot& expected, bool auxiliary) {
    for (size_t i = 0; i < actual.scratch.size(); ++i) if (actual.scratch[i] != expected.scratch[i])
        throw FormatError("recurrent timing scratch bits differ at stage " + std::to_string(i));
    require(actual.output == expected.output && actual.matrix == expected.matrix && actual.history == expected.history,
        "recurrent timing output/final state bits differ");
    if (auxiliary) require(actual.auxiliary_matrix == expected.auxiliary_matrix && actual.auxiliary_history == expected.auxiliary_history,
        "recurrent timing auxiliary chronological state bits differ");
}
struct Hashes { std::string bank, scratch, output, matrix, history, auxiliary; };
Hashes hashes(const Snapshot& snapshot) {
    Digest whole, scratch, auxiliary;
    for (size_t i = 0; i < snapshot.scratch.size(); ++i) {
        const auto tag = "scratch_" + std::to_string(i); whole.part(tag, snapshot.scratch[i]); scratch.part(tag, snapshot.scratch[i]);
    }
    whole.part("output", snapshot.output); whole.part("matrix", snapshot.matrix); whole.part("history", snapshot.history);
    auxiliary.part("matrix", snapshot.auxiliary_matrix); auxiliary.part("history", snapshot.auxiliary_history);
    return {whole.finish(), scratch.finish(), checksum(snapshot.output), checksum(snapshot.matrix), checksum(snapshot.history), auxiliary.finish()};
}
std::string tails(MetalGraph& g, const Bank& bank, uint32_t tokens) {
    Digest digest; const auto views = bank.scratch.views();
    for (size_t i = 0; i < views.size(); ++i) {
        const auto offset = views[i].bytes / capacity * tokens;
        const auto bytes = offset < views[i].bytes ? g.read(g.slice(views[i], offset, views[i].bytes - offset)) : Bytes{};
        digest.part("scratch_" + std::to_string(i), bytes);
    }
    const auto offset = stream_width * tokens * 4;
    const auto bytes = offset < bank.output.bytes ? g.read(g.slice(bank.output, offset, bank.output.bytes - offset)) : Bytes{};
    digest.part("output", bytes); return digest.finish();
}
void same_ledger(const GraphMemory& before, const GraphMemory& after) {
    require(before.live_bytes == after.live_bytes && before.allocations == after.allocations && before.peak_bytes == after.peak_bytes &&
        before.budget == after.budget && before.imported_backing_bytes == after.imported_backing_bytes &&
        before.capture_resource_bytes == 0 && before.live_captures == 0 && after.capture_resource_bytes == 0 && after.live_captures == 0 &&
        after.profile_counter_payload_bytes == 0 && after.live_bytes <= budget && after.peak_bytes <= budget,
        "recurrent timing GPU buffer ledger changed");
}
Json common_hashes(const Hashes& h, uint32_t tokens, uint32_t mask, double error) {
    return Json(Json::Object{{"tokens", Json::integer(tokens)}, {"block_mask", Json::integer(mask)}, {"outputs_exact", Json(true)},
        {"reference_max_scaled_error", real(error)}, {"bank_sha256", Json(h.bank)}, {"scratch_sha256", Json(h.scratch)},
        {"output_sha256", Json(h.output)}, {"matrix_sha256", Json(h.matrix)}, {"history_sha256", Json(h.history)},
        {"serial_auxiliary_state_sha256", Json(h.auxiliary)}, {"serial_dispatches", Json::integer(dispatches(tokens, RecurrentTimingMode::serial))},
        {"tile_dispatches", Json::integer(dispatches(tokens, RecurrentTimingMode::tile))}});
}
struct Condition { Snapshot expected; Hashes expected_hashes; double reference_error = 0; };
} // namespace

int recurrent_timing_cli(int argc, char** argv) {
    uint64_t completed = 0;
    try {
        uint32_t layer = 0, group = 32; std::set<std::string_view> seen; std::vector<std::filesystem::path> paths;
        auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
        for (int i = 2; i < argc; ++i) {
            const std::string_view key(argv[i]);
            if (!key.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
            require(seen.insert(key).second && ++i < argc, "duplicate or missing recurrent timing option"); const std::string_view value(argv[i]);
            if (key == "--layer") layer = integer(value);
            else if (key == "--delta-group") group = integer(value);
            else if (key == "--library") library = value;
            else throw FormatError("unknown recurrent timing option");
        }
        require(seen.contains("--layer") && paths.size() == 4 && (layer == 0 || layer == 2) && (group == 32 || group == 64 || group == 128),
            "recurrent timing requires four existing shards, explicit layer0/2 and supported delta group");
        for (size_t i = 0; i < paths.size(); ++i) for (size_t j = 0; j < i; ++j)
            require(!std::filesystem::equivalent(paths[i], paths[j]), "duplicate recurrent timing shard backing");
        const auto setup_started = Clock::now(); auto source = std::make_shared<ModelFiles>(paths);
        static_cast<void>(inspect_flash_next(*source)); const auto prefix = "blk." + std::to_string(layer) + ".";
        std::vector<TensorSlice> slices; uint64_t payload = 0; Json::Array weight_pins;
        for (const auto* name : tensor_names) {
            const auto& tensor = source->tensor(prefix + name); slices.push_back({tensor.name, 0, tensor.bytes}); payload = checked_add(payload, tensor.bytes);
            Json::Array dimensions; for (auto dimension : tensor.dimensions) dimensions.push_back(Json::integer(dimension));
            weight_pins.emplace_back(Json::Object{{"name", Json(tensor.name)}, {"type", Json::integer(tensor.type)}, {"shape", Json(std::move(dimensions))},
                {"bytes", Json::integer(tensor.bytes)}, {"sha256", Json(checksum(source->tensor_bytes(tensor.name)))}});
        }
        require(payload == (layer == 0 ? 1'660'099'456ull : 2'131'958'656ull), "recurrent timing25 tensor payload differs");
        MetalGraph g(library, budget); const auto imported = g.import_weights(source, slices);
        g.configure_small_gemv(0); g.configure_block_gemv(1);
        for (const auto width : {1, 2, 4, 8}) g.prepare_q8_matvec_many(width); g.prepare_gdn_tiles(group);
        const auto matrix = [&](size_t i) { return GpuMatrix{imported.tensors[i], source->tensor(prefix + tensor_names[i]).type}; };
        const auto reference_matrix = [&](size_t i) { return ReferenceMatrix{source->tensor_bytes(prefix + tensor_names[i]), source->tensor(prefix + tensor_names[i]).type}; };
        std::deque<std::vector<double>> scalars;
        const auto scalar = [&](size_t i) -> std::span<const double> {
            const auto& tensor = source->tensor(prefix + tensor_names[i]); require(tensor.type == 0 && tensor.elements <= 65536, "expected bounded recurrent timing F32 parameter");
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
        const ReferenceLayerWeights reference{reference_residual(0), reference_residual(4), reference_core, reference_moe};
        const GdnPolicy policy{1 / std::sqrt(128.0f), 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
        std::vector<float> inputs(uint64_t(capacity) * stream_width), state(gdn_state_count(shape.gdn)), history(gdn_qkv_count(shape.gdn) * 3);
        for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(int((i * 17 + 13) % 257) - 128) / 4096;
        for (size_t i = 0; i < state.size(); ++i) state[i] = float(int((i * 7 + 3) % 127) - 63) / 16384;
        for (size_t i = 0; i < history.size(); ++i) history[i] = float(int((i * 11 + 5) % 127) - 63) / 4096;
        const auto input = g.upload(std::as_bytes(std::span(inputs)), "recurrent timing immutable input");
        const GdnState base{g.upload(std::as_bytes(std::span(state)), "recurrent timing immutable matrix"),
            g.upload(std::as_bytes(std::span(history)), "recurrent timing immutable history")};
        const auto saved_input = g.read(input), saved_matrix = g.read(base.matrix), saved_history = g.read(base.convolution);
        std::array<Bank, cycle_count> serial_banks{}, tile_banks{};
        for (auto& bank : serial_banks) bank = allocate_bank(g, input, base, true);
        for (auto& bank : tile_banks) bank = allocate_bank(g, input, base, false);
        const auto immutable_unchanged = [&] {
            require(g.read(input) == saved_input && g.read(base.matrix) == saved_matrix && g.read(base.convolution) == saved_history,
                "recurrent timing changed immutable input/accepted state");
        };
        std::array<std::array<Condition, token_counts.size()>, block_masks.size()> conditions; Json::Array validations;
        // Independent double/exact/replay proofs, pipeline compilation, uploads,
        // scratch initialization and expected-byte collection all precede ready.
        for (size_t mi = 0; mi < block_masks.size(); ++mi) {
            const auto mask = block_masks[mi]; g.configure_block_gemv(mask);
            for (size_t ni = 0; ni < token_counts.size(); ++ni) {
                const auto tokens = token_counts[ni]; const auto before = g.memory();
                const auto proof = check_recurrent_tile(g, weights, reference, shape, policy, RouterMode::softmax, true,
                    {capacity, tokens}, inputs, state, history, false);
                require(g.memory().live_bytes == before.live_bytes && proof.maximum_scaled_error <= 5e-5 &&
                    proof.serial_dispatches == dispatches(tokens, RecurrentTimingMode::serial) && proof.tile_dispatches == dispatches(tokens, RecurrentTimingMode::tile),
                    "recurrent timing independent validation proof differs");
                clear_bank(g, serial_banks[0]); clear_bank(g, tile_banks[0]); const auto ledger = g.memory();
                const auto serial_batch = execute(g, weights, serial_banks[0], input, base, policy, tokens, RecurrentTimingMode::serial);
                const auto ordinary = capture(g, serial_banks[0], tokens, RecurrentTimingMode::serial);
                const auto tile_batch = execute(g, weights, tile_banks[0], input, base, policy, tokens, RecurrentTimingMode::tile);
                const auto tiled = capture(g, tile_banks[0], tokens, RecurrentTimingMode::tile);
                compare(tiled, ordinary, false); same_ledger(ledger, g.memory()); immutable_unchanged();
                require(serial_batch.capture_executes == 0 && tile_batch.capture_executes == 0 &&
                    serial_batch.dispatches == proof.serial_dispatches && tile_batch.dispatches == proof.tile_dispatches &&
                    checksum(g.read(tile_banks[0].output)) == proof.output_sha256 && checksum(tiled.matrix) == proof.state_sha256 &&
                    checksum(tiled.history) == proof.history_sha256, "recurrent timing preallocated bank differs from independent proof");
                auto& condition = conditions[mi][ni]; condition.expected = ordinary; condition.expected_hashes = hashes(ordinary);
                condition.reference_error = proof.maximum_scaled_error;
                validations.push_back(common_hashes(condition.expected_hashes, tokens, mask, condition.reference_error));
            }
        }
        source->assert_unchanged(); immutable_unchanged(); const auto ready_memory = g.memory();
        require(ready_memory.budget == budget && ready_memory.live_bytes <= budget && ready_memory.peak_bytes <= budget &&
            ready_memory.imported_backing_bytes >= payload && ready_memory.profile_counter_payload_bytes == 0,
            "recurrent timing ready memory invariant");
        require(ready_memory.capture_resource_bytes == 0 && ready_memory.live_captures == 0, "recurrent timing must not prepare captures");
        emit(Json(Json::Object{{"event", Json("ready")}, {"schema", Json::integer(1)}, {"layer", Json::integer(layer)},
            {"capacity", Json::integer(capacity)}, {"tokens", Json(Json::Array{Json::integer(2), Json::integer(8), Json::integer(17), Json::integer(32)})},
            {"block_masks", Json(Json::Array{Json::integer(1), Json::integer(3)})}, {"cycles", Json::integer(cycle_count)}, {"banks_per_mode", Json::integer(cycle_count)},
            {"delta_group", Json::integer(group)}, {"small_gemv_column_splits", Json::integer(0)}, {"router_mode", Json("softmax")},
            {"normalize_selected", Json(true)}, {"query_scale", Json("inverse-sqrt-key-width")}, {"l2_mode", Json("add")},
            {"profiling_enabled", Json(false)}, {"gpu_buffer_bytes", Json::integer(ready_memory.live_bytes)},
            {"gpu_buffer_peak_bytes", Json::integer(ready_memory.peak_bytes)}, {"gpu_buffer_limit_bytes", Json::integer(budget)},
            {"gpu_allocation_count", Json::integer(ready_memory.allocations)}, {"imported_original_payload_bytes", Json::integer(payload)},
            {"imported_original_backing_bytes", Json::integer(ready_memory.imported_backing_bytes)}, {"reference_scaled_error_limit", real(5e-5)},
            {"full_model_loaded", Json(false)}, {"derived_weight_files_written", Json(false)}, {"validations", Json(std::move(validations))},
            {"weight_pins", Json(std::move(weight_pins))}, {"timing_scope", Json(timing_scope)},
            {"gpu_allocation_scope", Json("owned Metal buffer ledger only; excludes opaque host/driver allocations")},
            {"input_scope", Json("deterministic synthetic known branch inputs, not stacked model-prefix outputs")},
            {"checksum_framing", Json("tag-length u64le, UTF-8 tag, payload-length u64le, raw payload; scratch_0..31 then output/matrix/history")},
            {"setup_ms", real(milliseconds(Clock::now() - setup_started))}}));
        while (true) {
            const auto request = parse_recurrent_timing_request(line()); source->assert_unchanged();
            if (request.close) {
                same_ledger(ready_memory, g.memory()); immutable_unchanged();
                emit(Json(Json::Object{{"event", Json("end")}, {"schema", Json::integer(1)}, {"layer", Json::integer(layer)},
                    {"requests_completed", Json::integer(completed)}, {"cycles_completed", Json::integer(completed * cycle_count)},
                    {"profiling_enabled", Json(false)}, {"gpu_buffer_bytes", Json::integer(g.memory().live_bytes)},
                    {"gpu_allocation_count", Json::integer(g.memory().allocations)},
                    {"imported_original_backing_bytes", Json::integer(g.memory().imported_backing_bytes)}, {"pass", Json(true)}})); return 0;
            }
            require(completed < request_limit, "recurrent timing request count bound");
            const auto mi = size_t(std::find(block_masks.begin(), block_masks.end(), request.block_mask) - block_masks.begin());
            const auto ni = size_t(std::find(token_counts.begin(), token_counts.end(), request.tokens) - token_counts.begin());
            const auto& condition = conditions[mi][ni]; auto& banks = request.mode == RecurrentTimingMode::serial ? serial_banks : tile_banks;
            g.configure_block_gemv(request.block_mask); require(g.kernel_policy() == std::array<uint32_t, 2>{0, request.block_mask}, "recurrent timing kernel policy differs");
            same_ledger(ready_memory, g.memory()); std::array<std::string, cycle_count> tail_before;
            for (size_t i = 0; i < banks.size(); ++i) tail_before[i] = tails(g, banks[i], request.tokens);
            const auto before = g.memory(); std::array<BatchResult, cycle_count> measured{};
            // Measured interval: four independent one-batch cycles. No owned
            // readback/reference/hash/upload/policy change or GPU-buffer allocation.
            // Graph command/encoder creation, completion/error checks stay charged.
            const auto started = Clock::now();
            for (size_t i = 0; i < banks.size(); ++i)
                measured[i] = execute(g, weights, banks[i], input, base, policy, request.tokens, request.mode);
            const auto finished = Clock::now();
            const auto after = g.memory(); same_ledger(before, after); same_ledger(ready_memory, after);
            require(g.kernel_policy() == std::array<uint32_t, 2>{0, request.block_mask}, "recurrent timing policy changed during execution");
            double gpu_ms = 0, batch_ms = 0; uint64_t total_dispatches = 0; const auto per_cycle = dispatches(request.tokens, request.mode);
            Json::Array bank_results;
            for (size_t i = 0; i < banks.size(); ++i) {
                require(measured[i].capture_executes == 0 && measured[i].dispatches == per_cycle && std::isfinite(measured[i].gpu_ms) && measured[i].gpu_ms > 0 &&
                    std::isfinite(measured[i].wall_ms) && measured[i].wall_ms > 0, "recurrent timing measured dispatch/time invariant");
                total_dispatches += measured[i].dispatches; gpu_ms += measured[i].gpu_ms; batch_ms += measured[i].wall_ms;
                const auto actual = capture(g, banks[i], request.tokens, request.mode);
                compare(actual, condition.expected, request.mode == RecurrentTimingMode::serial); const auto h = hashes(actual);
                const auto tail_after = tails(g, banks[i], request.tokens); require(tail_before[i] == tail_after, "recurrent timing modified unused tail bytes");
                bank_results.emplace_back(Json::Object{{"bank", Json::integer(i)}, {"bank_sha256", Json(h.bank)}, {"scratch_sha256", Json(h.scratch)},
                    {"output_sha256", Json(h.output)}, {"matrix_sha256", Json(h.matrix)}, {"history_sha256", Json(h.history)},
                    {"auxiliary_state_sha256", request.mode == RecurrentTimingMode::serial ? Json(h.auxiliary) : Json()},
                    {"tail_before_sha256", Json(tail_before[i])}, {"tail_after_sha256", Json(tail_after)}, {"tails_unchanged", Json(true)}, {"outputs_exact", Json(true)}});
            }
            source->assert_unchanged(); immutable_unchanged(); ++completed;
            emit(Json(Json::Object{{"event", Json("done")}, {"schema", Json::integer(1)}, {"request_index", Json::integer(completed)},
                {"layer", Json::integer(layer)}, {"tokens", Json::integer(request.tokens)}, {"mode", Json(request.mode == RecurrentTimingMode::serial ? "serial" : "tile")},
                {"block_mask", Json::integer(request.block_mask)}, {"cycles", Json::integer(cycle_count)}, {"capacity", Json::integer(capacity)},
                {"delta_group", Json::integer(group)}, {"small_gemv_column_splits", Json::integer(0)}, {"effective_block_gemv_formats", Json::integer(request.block_mask)},
                {"profiling_enabled", Json(false)}, {"dispatches", Json::integer(total_dispatches)}, {"dispatches_per_cycle", Json::integer(per_cycle)},
                {"gpu_ms_per_cycle", real(gpu_ms / cycle_count)}, {"batch_wall_ms_per_cycle", real(batch_ms / cycle_count)},
                {"encode_submit_wall_ms_per_cycle", real(milliseconds(finished - started) / cycle_count)},
                {"outputs_exact", Json(true)}, {"reference_max_scaled_error", real(condition.reference_error)},
                {"gpu_buffer_bytes", Json::integer(after.live_bytes)}, {"gpu_buffer_peak_bytes", Json::integer(after.peak_bytes)},
                {"gpu_buffer_limit_bytes", Json::integer(budget)}, {"gpu_allocation_count", Json::integer(after.allocations)},
                {"imported_original_payload_bytes", Json::integer(payload)}, {"imported_original_backing_bytes", Json::integer(after.imported_backing_bytes)},
                {"gpu_buffer_allocations_during_request", Json::integer(0)}, {"banks", Json(std::move(bank_results))}}));
        }
    } catch (const std::exception& error) {
        const std::string message(error.what());
        try { emit(Json(Json::Object{{"event", Json("error")}, {"schema", Json::integer(1)}, {"requests_completed", Json::integer(completed)},
            {"message", Json(message.substr(0, 2048))}})); } catch (const std::exception&) {}
        std::cerr << "Owned recurrent timing worker failed: " << error.what() << '\n'; return 1;
    }
}
} // namespace tokspur
