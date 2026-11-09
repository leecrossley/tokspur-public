#include "tokspur/target.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>

namespace tokspur {
int mapping_check_cli(int argc, char** argv) {
    size_t checks = 0;
    const auto require = [&](bool value, const char* why) { ++checks; if (!value) throw FormatError(why); };
    const auto reject = [&](const std::function<void()>& f) { ++checks; try { f(); } catch (const FormatError&) { return; } throw FormatError("unsafe mapped-weight import accepted"); };
    std::vector<std::filesystem::path> paths; for (int i = 2; i < argc; ++i) paths.emplace_back(argv[i]);
    auto source = std::make_shared<ModelFiles>(paths); std::weak_ptr<const ModelFiles> weak = source;
    const auto library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    MetalGraph gpu(library, 32 * 1024 * 1024); const auto base = gpu.memory().live_bytes;
    std::vector<TensorSlice> slices;
    struct Probe { uint32_t type, columns, rows; std::span<const std::byte> bytes; };
    std::vector<Probe> probes;
    const auto add = [&](std::string name, uint64_t first, uint32_t rows) {
        const auto& t = source->tensor(name); const auto columns = uint32_t(t.dimensions[0]);
        const auto row_bytes = t.bytes / (t.elements / columns);
        slices.push_back({name, first * row_bytes, uint64_t(rows) * row_bytes});
        probes.push_back({t.type, columns, rows, source->tensor_bytes(name).subspan(first * row_bytes, uint64_t(rows) * row_bytes)});
    };
    add("blk.0.ffn_gate_inp.weight", 0, 32); add("blk.3.indexer.q_proj.weight", 0, 32);
    add("blk.0.attn_qkv.weight", 0, 32); add("blk.0.ffn_gate_exps.weight", 0, 32);
    add("blk.0.ffn_gate_exps.weight", 16, 32); // deliberate overlapping request, one backing region
    add("blk.2.ffn_gate_exps.weight", 0, 32); add("blk.0.ffn_down_exps.weight", 0, 32);
    add("per_layer_token_embd.weight", 0, 32); add("per_layer_token_embd.weight", 320001444, 2);
    reject([&] { static_cast<void>(gpu.import_weights({}, slices)); });
    reject([&] { static_cast<void>(gpu.import_weights(source, {})); });
    for (const auto& bad : std::vector<TensorSlice>{{"missing", 0, 16}, {slices[0].name, 1, 16},
        {slices[0].name, source->tensor(slices[0].name).bytes, 16}, {slices[0].name, 0, source->tensor(slices[0].name).bytes + 4}, {"per_layer_token_embd.weight", 0, 0}}) {
        reject([&] { static_cast<void>(gpu.import_weights(source, std::span(&bad, 1))); });
    }
    require(gpu.memory().live_bytes == base && gpu.memory().imported_backing_bytes == 0, "refused imports changed live accounting");
    const auto imported = gpu.import_weights(source, slices);
    require(imported.tensors.size() == probes.size() && imported.regions.size() < imported.tensors.size(), "mapped import did not merge overlapping ranges");
    require(imported.tensors[3].slot == imported.tensors[4].slot, "overlapping tensor views have distinct backing allocations");
    uint64_t backing = 0; for (auto v : imported.regions) backing += v.bytes;
    require(backing == gpu.memory().imported_backing_bytes && backing < 4 * 1024 * 1024, "mapped region backing count/bound differs");
    reject([&] { static_cast<void>(gpu.import_weights(source, slices)); });
    source.reset(); require(!weak.expired(), "Metal import did not retain mapped source lifetime");
    double worst = 0;
    std::vector<GpuView> inputs, outputs; inputs.reserve(probes.size()); outputs.reserve(probes.size());
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto actual_bytes = gpu.read(imported.tensors[i]);
        require(std::equal(actual_bytes.begin(), actual_bytes.end(), probes[i].bytes.begin(), probes[i].bytes.end()), "Metal mapped tensor bytes differ from original file");
        reject([&] { gpu.write(imported.tensors[i], probes[i].bytes); });
        std::vector<float> x(probes[i].columns); for (size_t c = 0; c < x.size(); ++c) x[c] = float(int((c * 17 + 3) % 257) - 128) / 128;
        inputs.push_back(gpu.upload(std::as_bytes(std::span(x)), "mapping check input")); outputs.push_back(gpu.zeros(probes[i].rows * 4, "mapping check output"));
    }
    const auto allocations = gpu.memory().allocations;
    std::vector<std::vector<std::byte>> ordinary;
    for (uint32_t mode = 0; mode < 4; ++mode) {
        gpu.configure_block_gemv(mode); gpu.begin();
        for (size_t i = 0; i < probes.size(); ++i) gpu.gemv(probes[i].type, imported.tensors[i], inputs[i], outputs[i], probes[i].columns, probes[i].rows);
        const auto batch = gpu.submit(); require(batch.dispatches == probes.size() && gpu.memory().allocations == allocations, "mapped batch allocated or lost dispatches");
        for (size_t i = 0; i < probes.size(); ++i) {
            if (!mode) ordinary.push_back(gpu.read(outputs[i]));
            else require(gpu.read(outputs[i]) == ordinary[i], "mapped block kernel changed ordinary output bits");
        }
    }
    gpu.configure_block_gemv(0);
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto input = gpu.read(inputs[i]), output = gpu.read(outputs[i]); std::vector<float> x(probes[i].columns), y(probes[i].rows);
        std::memcpy(x.data(), input.data(), input.size()); std::memcpy(y.data(), output.data(), output.size());
        const auto reference = reference_gemv(probes[i].type, probes[i].bytes, probes[i].columns, probes[i].rows, x);
        for (size_t r = 0; r < y.size(); ++r) { const auto error = std::abs(y[r] - reference[r]) / std::max(1.0, std::abs(reference[r])); worst = std::max(worst, error); require(std::isfinite(y[r]) && error <= 5e-5, "mapped GPU operator differs from scalar reference"); }
        gpu.release(inputs[i]); gpu.release(outputs[i]);
    }
    weak.lock()->assert_unchanged();
    gpu.begin(); reject([&] { gpu.release(imported.regions[0]); }); gpu.abort();
    for (auto v : imported.regions) gpu.release(v);
    require(weak.expired(), "released imports still retain mapped source");
    require(gpu.memory().live_bytes == base && gpu.memory().imported_backing_bytes == 0, "mapped imports leaked backing accounting/resources");
    std::cout << "Owned mapped-weight checks passed: " << checks << ", tensor slices: " << probes.size() << ", regions: " << imported.regions.size()
              << ", accounted backing bytes: " << backing << ", max scaled error: " << worst << '\n';
    return 0;
}
} // namespace tokspur
