#include "tokspur/q8_many.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"

#include <CommonCrypto/CommonDigest.h>

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
constexpr uint64_t gpu_budget = 64 * 1024 * 1024;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
double elapsed(Clock::time_point from) { return std::chrono::duration<double, std::milli>(Clock::now() - from).count(); }
Json real(double value) {
    require(std::isfinite(value), "nonfinite Q8-many diagnostic"); std::array<char,64> text{};
    const auto r = std::to_chars(text.data(), text.data() + text.size(), value);
    require(r.ec == std::errc{}, "Q8-many number encoding"); return parse_json({text.data(), size_t(r.ptr - text.data())});
}
const Json& field(const Json& data, std::string_view key) {
    const auto* value = data.find(key); require(value, "missing Q8-many request field"); return *value;
}
bool line(std::string& text) {
    text.clear(); bool any = false, too_long = false; char c;
    while (std::cin.get(c)) { any = true; if (c == '\n') break; if (text.size() < 4096) text.push_back(c); else too_long = true; }
    require(!too_long, "Q8-many request byte bound"); return any;
}
std::string checksum(std::span<const std::byte> bytes) {
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest.data());
    constexpr char hex[] = "0123456789abcdef"; std::string out; out.reserve(digest.size() * 2);
    for (auto value : digest) { out.push_back(hex[value >> 4]); out.push_back(hex[value & 15]); } return out;
}
struct Projection { uint32_t columns, rows; GpuView weights, input, output; };
void ordinary(MetalGraph& g, const Projection& p, uint32_t width) {
    for (uint32_t token = 0; token < width; ++token)
        g.gemv(8, p.weights, g.slice(p.input, uint64_t(token) * p.columns * 4, uint64_t(p.columns) * 4),
            g.slice(p.output, uint64_t(token) * p.rows * 4, uint64_t(p.rows) * 4), p.columns, p.rows);
}
void many(MetalGraph& g, const Projection& p, uint32_t width) {
    g.q8_matvec_many(p.weights, g.slice(p.input, 0, uint64_t(width) * p.columns * 4),
        g.slice(p.output, 0, uint64_t(width) * p.rows * 4), p.columns, p.rows, width);
}
} // namespace
int q8_many_check_cli(int argc, char** argv) {
    std::vector<std::filesystem::path> paths;
    for (int i = 2; i < argc; ++i) paths.emplace_back(argv[i]);
    require(paths.size() == 4, "Q8-many check requires the existing four-shard target");
    for (size_t i = 0; i < paths.size(); ++i) for (size_t j = 0; j < i; ++j)
        require(!std::filesystem::equivalent(paths[i], paths[j]), "duplicate Q8-many shard backing file");
    const auto started = Clock::now(); auto source = std::make_shared<ModelFiles>(paths);
    static_cast<void>(inspect_flash_next(*source)); // Metadata/geometry only, never full-model upload.
    const std::array<std::string,3> names{"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight"};
    const std::array<std::array<uint32_t,2>,3> shapes{{{2560,10240},{2560,6144},{6144,2560}}};
    std::vector<TensorSlice> slices; uint64_t payload = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const auto& tensor = source->tensor(names[i]);
        require(tensor.type == 8 && tensor.dimensions == std::vector<uint64_t>{shapes[i][0],shapes[i][1]}, "Q8-many exact named projection geometry mismatch");
        slices.push_back({names[i],0,tensor.bytes});
        payload = checked_add(payload,tensor.bytes);
    }
    MetalGraph g(std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib", gpu_budget);
    const auto imported = g.import_weights(source,slices); g.configure_block_gemv(1);
    for (const uint32_t width : {1,2,4,8}) g.prepare_q8_matvec_many(width);
    std::array<Projection,3> p{}; std::array<std::vector<float>,3> inputs;
    for (size_t i = 0; i < p.size(); ++i) {
        const auto c = shapes[i][0], r = shapes[i][1]; inputs[i].resize(uint64_t(c) * 8);
        for (uint32_t token = 0; token < 8; ++token) for (uint32_t column = 0; column < c; ++column)
            inputs[i][uint64_t(token) * c + column] = float(int((column * 17 + token * 37 + i * 13 + 3) % 257) - 128) / 4096;
        p[i] = {c,r,imported.tensors[i],g.upload(std::as_bytes(std::span(inputs[i])),"Q8-many fixed inputs"),g.zeros(uint64_t(r) * 8 * 4,"Q8-many outputs")};
    }
    std::array<std::array<std::vector<std::byte>,4>,3> saved; std::array<std::array<double,4>,3> errors{};
    const std::array<uint32_t,4> widths{1,2,4,8}; Json::Array validation, rotation_validation;
    const auto allocations = g.memory().allocations;
    for (size_t i = 0; i < p.size(); ++i) {
        // One double reference of all eight independent inputs also covers
        // every smaller width's leading rows. Validation is outside timings.
        const auto reference = reference_q8_many(source->tensor_bytes(names[i]), inputs[i], {p[i].columns,p[i].rows,8}, 1024 * 1024);
        for (size_t wi = 0; wi < widths.size(); ++wi) {
            const auto width = widths[wi]; g.begin(); ordinary(g,p[i],width); const auto baseline = g.submit();
            require(baseline.dispatches == width && g.memory().allocations == allocations,"Q8-many validation baseline dispatch/allocation mismatch");
            const auto out = g.slice(p[i].output,0,uint64_t(width) * p[i].rows * 4); const auto expected = g.read(out);
            double worst = 0;
            for (size_t element = 0; element < expected.size() / 4; ++element) {
                float value; std::memcpy(&value,expected.data() + element * 4,4);
                const auto error = std::abs(double(value) - reference[element]) / std::max(1.0,std::abs(reference[element])); worst = std::max(worst,error);
                require(std::isfinite(value) && error <= 5e-5,"Q8-many scalar reference differs");
            }
            g.begin(); many(g,p[i],width); const auto candidate = g.submit();
            require(candidate.dispatches == 1 && g.memory().allocations == allocations,"Q8-many validation candidate dispatch/allocation mismatch");
            require(g.read(out) == expected,"Q8-many real-weight output bits differ from block baseline");
            saved[i][wi] = expected; errors[i][wi] = worst;
            if (!i) {
                validation.push_back(Json(Json::Object{{"width",Json::integer(width)},{"output_sha256",Json(checksum(expected))},
                    {"reference_max_scaled_error",real(worst)},{"outputs_exact",Json(true)}}));
            } else rotation_validation.push_back(Json(Json::Object{{"tensor",Json(names[i])},{"width",Json::integer(width)},
                {"output_sha256",Json(checksum(expected))},{"reference_max_scaled_error",real(worst)},{"outputs_exact",Json(true)}}));
        }
    }
    source->assert_unchanged(); const auto memory = g.memory();
    require(memory.live_bytes <= gpu_budget,"Q8-many buffer ceiling exceeded");
    std::cout << Json(Json::Object{{"event",Json("ready")},{"scope",Json("bounded Q8-many projection only, not complete inference or speed qualification")},
        {"columns",Json::integer(2560)},{"rows",Json::integer(10240)},
        {"widths",Json(Json::Array{Json::integer(1),Json::integer(2),Json::integer(4),Json::integer(8)})},
        {"gpu_buffer_bytes",Json::integer(memory.live_bytes)},{"gpu_buffer_limit_bytes",Json::integer(gpu_budget)},
        {"gpu_allocation_count",Json::integer(memory.allocations)},{"imported_original_backing_bytes",Json::integer(memory.imported_backing_bytes)},
        {"imported_original_payload_bytes",Json::integer(payload)},
        {"original_matrices",Json::integer(3)},{"derived_weight_files_written",Json(false)},{"full_model_loaded",Json(false)},
        {"all_validation_outputs_exact",Json(true)},{"profiling_enabled",Json(false)},{"setup_ms",real(elapsed(started))},
        {"reference_scaled_error_limit",real(5e-5)},{"validations",Json(std::move(validation))},
        {"rotation_validations",Json(std::move(rotation_validation))},
        {"rotation_scope",Json("two unchanged ordinary block projections per input before each probe; not guaranteed cache-cold or full-model representative")}}).dump() << std::endl;
    std::string request;
    while (true) {
        try {
            if (!line(request)) break;
            const auto data = parse_json(request,{4096,64,4}); std::set<std::string> keys;
            for (const auto& [key,value] : data.object()) { static_cast<void>(value); require(key == "width" || key == "mode" || key == "rotating" || key == "cycles","unknown Q8-many request field"); keys.insert(key); }
            require(keys.size() == 4,"all Q8-many request fields required"); const auto width64 = field(data,"width").unsigned_integer();
            require(width64 == 1 || width64 == 2 || width64 == 4 || width64 == 8,"unsupported Q8-many request width"); const auto width = uint32_t(width64);
            const auto mode = field(data,"mode").string(); require(mode == "block" || mode == "many","unsupported Q8-many request mode");
            const auto rotating = field(data,"rotating").boolean(); const auto cycles = field(data,"cycles").unsigned_integer(); require(cycles && cycles <= 4096,"Q8-many cycle bound");
            source->assert_unchanged(); const auto before = g.memory(); const auto begin = Clock::now();
            try {
                g.begin();
                for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
                    if (rotating) { ordinary(g,p[1],width); ordinary(g,p[2],width); }
                    if (mode == "block") ordinary(g,p[0],width); else many(g,p[0],width);
                }
                const auto batch = g.submit(); const auto wall = elapsed(begin); const auto after = g.memory();
                require(batch.dispatches == cycles * ((rotating ? 2 * width : 0) + (mode == "block" ? width : 1)) &&
                    after.allocations == before.allocations && after.live_bytes == before.live_bytes && after.imported_backing_bytes == before.imported_backing_bytes,"Q8-many measured dispatch/allocation invariant");
                const auto wi = size_t(std::find(widths.begin(),widths.end(),width) - widths.begin());
                const auto output = g.read(g.slice(p[0].output,0,uint64_t(width) * p[0].rows * 4));
                require(output == saved[0][wi],"Q8-many measured output drift"); Json::Array rotation_outputs;
                if (rotating) for (size_t i=1;i<p.size();++i) {
                    const auto bytes = g.read(g.slice(p[i].output,0,uint64_t(width) * p[i].rows * 4));
                    require(bytes == saved[i][wi],"Q8-many unchanged rotation output drift");
                    rotation_outputs.push_back(Json(Json::Object{{"tensor",Json(names[i])},{"output_sha256",Json(checksum(bytes))},
                        {"reference_max_scaled_error",real(errors[i][wi])},{"outputs_exact",Json(true)}}));
                }
                source->assert_unchanged();
                std::cout << Json(Json::Object{{"event",Json("done")},{"width",Json::integer(width)},{"mode",Json(std::string(mode))},{"rotating",Json(rotating)},
                    {"cycles",Json::integer(cycles)},{"dispatches",Json::integer(batch.dispatches)},{"gpu_ms_per_cycle",real(batch.gpu_ms / cycles)},
                    {"batch_wall_ms_per_cycle",real(batch.wall_ms / cycles)},{"encode_submit_wall_ms_per_cycle",real(wall / cycles)},
                    {"outputs_exact",Json(true)},{"profiling_enabled",Json(false)},
                    {"output_sha256",Json(checksum(output))},{"reference_max_scaled_error",real(errors[0][wi])},
                    {"rotation_outputs",Json(std::move(rotation_outputs))},
                    {"gpu_buffer_bytes",Json::integer(after.live_bytes)},{"gpu_allocation_count",Json::integer(after.allocations)},
                    {"imported_original_backing_bytes",Json::integer(after.imported_backing_bytes)},
                    {"gpu_buffer_allocations_during_request",Json::integer(0)}}).dump() << std::endl;
            } catch (...) { try { g.abort(); } catch (const FormatError&) {} throw; }
        } catch (const FormatError& error) {
            std::cout << Json(Json::Object{{"event",Json("error")},{"message",Json(error.what())}}).dump() << std::endl;
            if (!std::cin) break;
        }
    }
    return 0;
}
} // namespace tokspur
