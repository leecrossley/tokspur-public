#include "tokspur/memory.hpp"
#include "tokspur/ple.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>

namespace tokspur {
namespace {
Json number(double x) {
    if (!std::isfinite(x)) throw FormatError("nonfinite memory diagnostic");
    std::array<char, 64> bytes{}; const auto r = std::to_chars(bytes.data(), bytes.data() + bytes.size(), x);
    if (r.ec != std::errc{}) throw FormatError("memory diagnostic encoding failed");
    return parse_json({bytes.data(), size_t(r.ptr - bytes.data())});
}
uint32_t integer(std::string_view x, uint32_t max) {
    uint32_t value = 0; const auto r = std::from_chars(x.data(), x.data() + x.size(), value);
    if (r.ec != std::errc{} || r.ptr != x.data() + x.size() || value > max) throw FormatError("invalid bounded memory check integer");
    return value;
}
std::vector<float> read(MetalGraph& g, GpuView v) {
    const auto data = g.read(v); std::vector<float> out(data.size() / 4); std::memcpy(out.data(), data.data(), data.size()); return out;
}
double difference(std::span<const float> actual, std::span<const double> expected) {
    if (actual.size() != expected.size()) throw FormatError("memory diagnostic size mismatch");
    double maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw FormatError("nonfinite memory result/reference");
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i])));
    }
    return maximum;
}
Json floats(std::span<const float> x) { Json::Array out; for (auto v : x) out.push_back(number(v)); return Json(std::move(out)); }
Json rows(std::span<const uint64_t> ids) { Json::Array out; for (auto id : ids) out.push_back(Json::integer(id)); return Json(std::move(out)); }
}
int memory_check_cli(int argc, char** argv) {
    std::optional<HashOrder> order; std::optional<NgramBoundary> boundary; uint32_t dilation = 0, repeats = 2;
    bool full_reference = false; std::vector<uint32_t> tokens; std::vector<std::filesystem::path> paths;
    auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]); if (!option.starts_with("--")) { paths.emplace_back(argv[i]); continue; }
        if (++i == argc) throw FormatError("missing memory check option value"); const std::string_view value(argv[i]);
        if (option == "--hash-order" && !order) {
            if (value == "newest") order = HashOrder::newest_first; else if (value == "oldest") order = HashOrder::oldest_first; else throw FormatError("invalid hash order");
        } else if (option == "--boundary" && !boundary) {
            if (value == "literal") boundary = NgramBoundary::literal; else if (value == "before") boundary = NgramBoundary::reset_before;
            else if (value == "after") boundary = NgramBoundary::reset_after; else throw FormatError("invalid n-gram boundary policy");
        } else if (option == "--dilation" && !dilation) { dilation = integer(value, 3); if (dilation != 1 && dilation != 3) throw FormatError("expected explicit dilation 1 or 3"); }
        else if (option == "--token") { if (tokens.size() >= 32) throw FormatError("too many memory probe tokens"); tokens.push_back(integer(value, 248319)); }
        else if (option == "--repeats") { repeats = integer(value, 8); if (!repeats) throw FormatError("memory repeats must be positive"); }
        else if (option == "--full-output-reference") { if (value == "true") full_reference = true; else if (value == "false") full_reference = false; else throw FormatError("expected full-output-reference boolean"); }
        else if (option == "--library") library = value;
        else throw FormatError("unknown or duplicate memory check option");
    }
    if (!order || !boundary || !dilation) throw FormatError("memory check requires explicit hash-order, boundary and dilation hypotheses");
    if (tokens.empty()) tokens = {248045, 0, 12000, 248044, 248056, 42, 7, 8, 9, 10, 11, 248046};
    ModelFiles files(paths); const auto layout = inspect_flash_next(files);
    const NgramSpec spec{3, 8, 160, 248320, 248044, 248056, layout.ple_multipliers, layout.ple_offsets, layout.ple_vocabularies};
    const NgramPolicy policy{*order, *boundary, true};
    const auto& table = files.tensor("per_layer_token_embd.weight");
    EmbeddingRows cache(files.tensor_bytes(table.name), table.type, 160, layout.ple_storage_rows, layout.ple_addressable_rows, 16 * 1024 * 1024);
    MetalGraph gpu(library, uint64_t(4) * 1024 * 1024 * 1024);
    const auto load = [&](const std::string& name) { return GpuMatrix{gpu.upload(files.tensor_bytes(name), name), files.tensor(name).type}; };
    const auto matrix = [&](const std::string& name) { return ReferenceMatrix{files.tensor_bytes(name), files.tensor(name).type}; };
    std::map<std::string, std::vector<double>> parameters;
    const auto scalar = [&](const std::string& name) -> std::span<const double> {
        if (!parameters.contains(name)) { const auto& t = files.tensor(name); if (t.type != 0 || t.elements > 65536) throw FormatError("expected bounded memory parameter");
            std::vector<float> data(t.elements); dequantize(t.type, files.tensor_bytes(name), data); parameters.emplace(name, std::vector<double>(data.begin(), data.end())); }
        return parameters.at(name);
    };
    const std::string prefix = "blk.1.";
    const PleWeights pw{load(prefix + "ple_key.weight"), load(prefix + "ple_value.weight"), load(prefix + "ple_norm_key.weight").view,
        load(prefix + "ple_norm_query.weight").view, load(prefix + "ple_norm_conv.weight").view, load(prefix + "ple_conv1d.weight").view};
    const ReferencePleWeights pr{matrix(prefix + "ple_key.weight"), matrix(prefix + "ple_value.weight"), scalar(prefix + "ple_norm_key.weight"),
        scalar(prefix + "ple_norm_query.weight"), scalar(prefix + "ple_norm_conv.weight"), scalar(prefix + "ple_conv1d.weight")};
    const MixerWeights mw{load("output_hc_norm.weight").view, load("output_hc_down.weight"), load("output_hc_up.weight")};
    const auto head = load("output.weight");
    const PleShape shape{4, 2560, 2560, 4, dilation, PleGateTransform::signed_root}; const ResidualShape mixshape{4, 2560, 320};
    const auto ps = allocate_ple_scratch(gpu, shape); const auto ms = allocate_mixer_scratch(gpu, mixshape);
    auto history = gpu.zeros(uint64_t(10240) * 3 * dilation * 4, "current PLE history"), next = gpu.zeros(uint64_t(10240) * 3 * dilation * 4, "candidate PLE history");
    const auto embedding = gpu.zeros(2560 * 4, "lookup embedding"), input = gpu.zeros(10240 * 4, "memory input"), output = gpu.zeros(10240 * 4, "memory output"), logits = gpu.zeros(248320 * 4, "full vocabulary logits");
    const auto allocations = gpu.memory().allocations; Json::Array runs; bool passed = true;
    for (uint32_t repeat = 0; repeat < repeats && passed; ++repeat) {
        auto ngram = empty_ngram_history(spec); std::vector<double> cpu_history(history.bytes / 4, 0); const std::vector<float> zero(cpu_history.size());
        gpu.write(history, std::as_bytes(std::span(zero)));
        for (size_t step = 0; step < tokens.size(); ++step) {
            const auto token = tokens[step]; const auto ids = ngram_step(spec, policy, ngram, token);
            std::vector<float> e(2560), x(2560), streams(10240); cache.gather(ids.rows, e);
            const auto& emb = files.tensor("token_embd.weight"); const auto bytes = emb.bytes / emb.dimensions[1];
            dequantize(emb.type, files.tensor_bytes(emb.name).subspan(token * bytes, bytes), x);
            for (size_t b = 0; b < 4; ++b) std::copy(x.begin(), x.end(), streams.begin() + b * 2560);
            const auto expected = ple_reference(pr, std::vector<double>(e.begin(), e.end()), std::vector<double>(streams.begin(), streams.end()), cpu_history, shape, layout.rms_epsilon);
            const auto mixed = mixer_reference(scalar("output_hc_norm.weight"), matrix("output_hc_down.weight"), matrix("output_hc_up.weight"), expected.output, mixshape, layout.rms_epsilon);
            gpu.write(embedding, std::as_bytes(std::span(e))); gpu.write(input, std::as_bytes(std::span(streams)));
            const auto accepted = gpu.read(history);
            gpu.begin(); encode_ple(gpu, pw, ps, embedding, input, history, next, output, shape, float(layout.rms_epsilon));
            encode_mixer(gpu, mw, ms, output, mixshape, float(layout.rms_epsilon)); gpu.gemv(head.encoding, head.view, ms.mixed, logits, 2560, 248320);
            const auto batch = gpu.submit();
            if (batch.dispatches != 14 || gpu.memory().allocations != allocations) throw FormatError("memory batch resource invariant failed");
            Json::Object errors; double maximum = 0;
            const auto check = [&](const char* name, GpuView view, std::span<const double> reference) { const auto error = difference(read(gpu, view), reference); maximum = std::max(maximum, error); errors.emplace_back(name, number(error)); };
            check("key", ps.key, expected.key); check("value", ps.value, expected.value); check("normalized_key", ps.normalized_key, expected.normalized_key);
            check("normalized_query", ps.normalized_query, expected.normalized_query); check("gates", ps.gates, expected.gates); check("gated", ps.gated, expected.gated);
            check("normalized_gated", ps.normalized_gated, expected.normalized_gated); check("convolved", ps.convolved, expected.convolution.output); check("next_history", next, expected.convolution.history);
            check("memory_output", output, expected.output); check("mixer_normalized", ms.normalized, mixed.normalized); check("mixer_bottleneck", ms.bottleneck, mixed.bottleneck);
            check("mixer_scores", ms.scores, mixed.scores); check("mixed", ms.mixed, mixed.mixed);
            const bool all_rows = full_reference && repeat == 0 && step == 0; uint32_t compared = 0; double logit_error = 0;
            const auto row_bytes = files.tensor("output.weight").bytes / 248320;
            const auto logit_check = [&](uint32_t first, uint32_t count) {
                const auto reference = reference_gemv_doubleinput(head.encoding, files.tensor_bytes("output.weight").subspan(uint64_t(first) * row_bytes, uint64_t(count) * row_bytes), 2560, count, mixed.mixed);
                logit_error = std::max(logit_error, difference(read(gpu, gpu.slice(logits, uint64_t(first) * 4, uint64_t(count) * 4)), reference)); compared += count;
            };
            if (all_rows) logit_check(0, 248320); else for (uint32_t first : {0, 12000, 248000, 248192}) logit_check(first, 128);
            errors.emplace_back("output_logits", number(logit_error)); maximum = std::max(maximum, logit_error);
            const bool unchanged = gpu.read(history) == accepted; const bool valid = unchanged && maximum <= 5e-5; passed = passed && valid;
            runs.push_back(Json(Json::Object{{"pass", Json(valid)}, {"repeat", Json::integer(repeat)}, {"step", Json::integer(step)}, {"token_fixture", Json::integer(token)},
                {"lookup_rows", rows(ids.rows)}, {"lookup_embedding", floats(e)}, {"compared_output_rows", Json::integer(compared)}, {"full_output_reference", Json(all_rows)},
                {"accepted_history_unchanged", Json(unchanged)}, {"max_scaled_error", number(maximum)}, {"stage_scaled_errors", Json(std::move(errors))}, {"dispatches", Json::integer(batch.dispatches)}}));
            if (!valid) break;
            cpu_history = expected.convolution.history; ngram = ids.next; std::swap(history, next);
        }
    }
    const auto stats = cache.stats(); files.assert_unchanged();
    std::cout << Json(Json::Object{{"schema", Json::integer(1)}, {"ple_gate_transform", Json("signed-root")}, {"scope", Json("real lookup + PLE + final mixer/full vocabulary projection using synthetic branch inputs; intervening model layers are not executed")},
        {"hash_order", Json(*order == HashOrder::newest_first ? "newest" : "oldest")}, {"boundary", Json(*boundary == NgramBoundary::literal ? "literal" : *boundary == NgramBoundary::reset_before ? "before" : "after")}, {"image_as_eos", Json(true)},
        {"dilation", Json::integer(dilation)}, {"mode_semantics_qualified", Json(false)}, {"timings_qualified", Json(false)}, {"tracked_metal_buffer_bytes", Json::integer(gpu.memory().live_bytes)},
        {"row_cache_charged_bytes", Json::integer(cache.charged_bytes())}, {"row_cache_capacity", Json::integer(cache.capacity())}, {"cache_hits", Json::integer(stats.hits)}, {"cache_misses", Json::integer(stats.misses)},
        {"runs", Json(std::move(runs))}, {"pass", Json(passed)}}).dump() << '\n';
    return passed ? 0 : 1;
}
} // namespace tokspur
