#include "grouped_mm_fixture.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/verify_nax.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace tokspur;
using namespace tokspur::planar_v3;

namespace {
using Bytes = std::vector<std::byte>;
constexpr uint32_t channels = 256, hidden = 64, experts = 16, selected = 2, max_rows = 8;
constexpr uint64_t graph_budget = 32ull << 20;
uint64_t checks = 0;
void need(bool condition, const char* message) { ++checks; if (!condition) throw FormatError(message); }

const char* counter_boundary(ProfileMode mode) {
    switch (mode) {
        case ProfileMode::dispatch_boundary: return "dispatch";
        case ProfileMode::encoder_boundary: return "encoder";
        default: throw FormatError("fixture counter boundary is disabled or unknown");
    }
}

template<class F> void refuses(F&& operation, std::string_view expected, const char* label) {
    bool caught = false;
    try { operation(); } catch (const FormatError& e) { caught = std::string_view(e.what()).find(expected) != std::string_view::npos; }
    need(caught, label);
    std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"refusal\",\"case\":\"" << label << "\",\"passed\":1}\n";
}

struct Env {
    std::string name;
    std::optional<std::string> previous;
    explicit Env(const char* key) : name(key) { if (const char* value = std::getenv(key)) previous = value; }
    void set(const char* value) { need(value ? ::setenv(name.c_str(), value, 1) == 0 : ::unsetenv(name.c_str()) == 0, "environment update failed"); }
    ~Env() { if (previous) ::setenv(name.c_str(), previous->c_str(), 1); else ::unsetenv(name.c_str()); }
};

struct Packed {
    uint32_t gu, down;
    Bytes lo, hi, dn;
};

// Reuse the bounded quantised generator and co-located packing convention from test_planar_pack_v3.cpp.
Packed packed(uint32_t gu_type, uint32_t down_type) {
    auto gate = grouped_fixture::weights({gu_type, channels, hidden, experts, 1, 1, true});
    auto up = gate;
    const uint32_t block_bytes = encoding(gu_type).bytes_per_block;
    const uint32_t code_offset = gu_type == 12 ? 16 : (gu_type == 13 ? 48 : 2);
    for (size_t at = 0; at < up.size(); at += block_bytes)
        for (uint32_t i = code_offset; i < block_bytes; ++i) up[at + i] ^= std::byte{0x5a};
    const auto down = grouped_fixture::weights({down_type, hidden, channels, experts, 1, 1, true});
    const auto gs = make_expert_spec(gu_type, channels, hidden), ds = make_expert_spec(down_type, hidden, channels);
    Packed result{gu_type, down_type, Bytes(experts * gs.planar_expert_bytes), Bytes(experts * gs.planar_expert_bytes), Bytes(experts * ds.planar_expert_bytes)};
    for (uint32_t e = 0; e < experts; ++e) {
        auto& pair = e < experts / 2 ? result.lo : result.hi;
        const uint64_t pair_offset = uint64_t(e % (experts / 2)) * 2 * gs.planar_expert_bytes;
        pack_expert_planar_v3(gs, std::span(gate).subspan(e * gs.source_expert_bytes, gs.source_expert_bytes), std::span(pair).subspan(pair_offset, gs.planar_expert_bytes));
        pack_expert_planar_v3(gs, std::span(up).subspan(e * gs.source_expert_bytes, gs.source_expert_bytes), std::span(pair).subspan(pair_offset + gs.planar_expert_bytes, gs.planar_expert_bytes));
        pack_expert_planar_v3(ds, std::span(down).subspan(e * ds.source_expert_bytes, ds.source_expert_bytes), std::span(result.dn).subspan(e * ds.planar_expert_bytes, ds.planar_expert_bytes));
    }
    return result;
}

struct Layer { GpuView lo{}, hi{}, dn{}, ids{}, x{}, h{}, y{}; };
struct Snapshot {
    std::array<Bytes, 4> outputs;
    uint64_t dispatches = 0;
    bool operator==(const Snapshot&) const = default;
};

struct Fixture {
    MetalGraph graph;
    ProfileMode profile_mode = ProfileMode::disabled;
    std::array<Layer, 3> layers{};
    uint64_t startup_allocations = 0, startup_bytes = 0;
    Fixture(const char* library, const std::array<Packed, 3>& weights, bool profile = true) : graph(library, graph_budget) {
        graph.configure_verify_nax(true);
        graph.configure_verify_nax_m8(true);
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& w = weights[i];
            auto& l = layers[i];
            l.lo = graph.upload(w.lo, "private diagnostic gate pair low", true);
            l.hi = graph.upload(w.hi, "private diagnostic gate pair high", true);
            l.dn = graph.upload(w.dn, "private diagnostic down", true);
            const auto gate_kind = w.gu == 12 ? PlanarKind::q4k_coloc_gate_v3 : (w.gu == 13 ? PlanarKind::q5k_coloc_gate_v3 : PlanarKind::q8_coloc_gate_v3);
            const auto up_kind = w.gu == 12 ? PlanarKind::q4k_coloc_up_v3 : (w.gu == 13 ? PlanarKind::q5k_coloc_up_v3 : PlanarKind::q8_coloc_up_v3);
            graph.register_planar_matrix(l.lo, {w.gu, uint32_t(gate_kind), channels, hidden, experts});
            graph.register_planar_matrix(l.hi, {w.gu, uint32_t(up_kind), channels, hidden, experts});
            graph.register_planar_matrix(l.dn, {w.down, uint32_t(w.down == 7 ? PlanarKind::q51_planar_v3 : PlanarKind::q8_planar_v3), hidden, channels, experts});
            l.ids = graph.zeros(max_rows * selected * sizeof(uint32_t), "private diagnostic route scratch");
            l.x = graph.zeros(max_rows * channels * sizeof(float), "private diagnostic input scratch");
            l.h = graph.zeros(max_rows * selected * hidden * sizeof(float), "private diagnostic hidden scratch");
            l.y = graph.zeros(max_rows * selected * channels * sizeof(float), "private diagnostic output scratch");
        }
        for (const auto& w : weights) graph.prepare_planar_moe_v3(w.gu, w.down, false);
        graph.prepare_verify_nax({});
        if (profile) {
            profile_mode = graph.configure_profiling(64);
            need(profile_mode == ProfileMode::dispatch_boundary || profile_mode == ProfileMode::encoder_boundary, "fixture requires per-dispatch counters");
        }
        graph.set_verify_nax_scope(true);
        startup_allocations = graph.memory().allocations;
        startup_bytes = graph.memory().live_bytes;
        need(startup_bytes < graph_budget && graph.planar_matrix_count() == 9, "bounded fixture registration or memory mismatch");
    }
    ~Fixture() { try { graph.abort(); } catch (const FormatError&) {} }
    GpuRowsF32 input(size_t layer, uint32_t rows) { return {graph.slice(layers[layer].x, 0, uint64_t(rows) * channels * 4), rows, channels, channels * 4}; }
    GpuRowsF32 intermediate(size_t layer, uint32_t rows) { return {graph.slice(layers[layer].h, 0, uint64_t(rows) * selected * hidden * 4), rows * selected, hidden, hidden * 4}; }
    GpuRowsF32 output(size_t layer, uint32_t rows) { return {graph.slice(layers[layer].y, 0, uint64_t(rows) * selected * channels * 4), rows * selected, channels, channels * 4}; }
    void encode(size_t layer, uint32_t rows) {
        const auto& l = layers[layer];
        graph.planar_moe_routed_m8(l.lo, l.hi, l.dn, graph.slice(l.ids, 0, rows * selected * 4), input(layer, rows), intermediate(layer, rows), output(layer, rows), experts, selected);
    }
};

std::vector<uint32_t> routes(uint32_t rows, uint32_t pattern, size_t layer) {
    std::vector<uint32_t> result(rows * selected);
    for (uint32_t t = 0; t < rows; ++t) for (uint32_t s = 0; s < selected; ++s) {
        if (pattern < 2) result[t * selected + s] = (layer == pattern) ? s : (t * selected + s);
        else result[t * selected + s] = layer == 0 ? 0 : (s == 0 ? 0 : t + 1);
    }
    return result;
}

std::array<uint32_t, 8> expected_histogram(std::span<const uint32_t> ids, uint32_t rows) {
    std::array<uint32_t, experts> frequencies{};
    for (const auto id : ids) { need(id < experts, "fixture route index is invalid"); ++frequencies[id]; }
    const uint32_t capacity = rows <= 4 ? 4 : 8;
    std::array<uint32_t, 8> histogram{};
    for (const auto count : frequencies) {
        histogram[capacity - 1] += count / capacity;
        if (count % capacity) ++histogram[count % capacity - 1];
    }
    return histogram;
}

void write_inputs(Fixture& f, uint32_t rows, uint32_t pattern) {
    for (size_t layer = 0; layer < 2; ++layer) {
        std::vector<float> x(uint64_t(rows) * channels);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int((i * 37 + layer * 17 + pattern * 13) % 211) - 105) / 107.0f;
        const auto ids = routes(rows, pattern, layer);
        f.graph.write(f.input(layer, rows).view, std::as_bytes(std::span(x)));
        f.graph.write(f.graph.slice(f.layers[layer].ids, 0, ids.size() * sizeof(uint32_t)), std::as_bytes(std::span(ids)));
    }
}

void marker(uint32_t arm, uint32_t case_id, uint32_t rows, uint32_t pattern, bool diagnostic, const char* action) {
    std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"batch_begin\",\"arm\":" << arm << ",\"case\":" << case_id << ",\"action\":\"" << action << "\",\"diagnostic_expected\":" << diagnostic << ",\"tokens\":" << rows << ",\"registered_gate_matrices\":3,\"observed_calls\":2,\"expected_histograms\":[";
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto h = expected_histogram(routes(rows, pattern, layer), rows);
        std::cerr << (layer ? ",[" : "[");
        for (size_t n = 0; n < h.size(); ++n) std::cerr << (n ? "," : "") << h[n];
        std::cerr << "]";
    }
    std::cerr << "]}" << std::endl;
}

Snapshot complete_batch(Fixture& f, uint32_t arm, uint32_t case_id, uint32_t rows, uint32_t pattern, bool diagnostic, const char* action) {
    write_inputs(f, rows, pattern);
    marker(arm, case_id, rows, pattern, diagnostic, action);
    const auto before = f.graph.memory();
    need(host_thermal_state() == 0, "model-free diagnostic fixture requires nominal thermal state");
    f.graph.begin();
    f.encode(0, rows);
    f.encode(1, rows);
    // This host pause stabilises short counter calibration. No latency result is reported by this correctness fixture.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto batch = f.graph.submit();
    const auto& profile = f.graph.last_profile();
    need(host_thermal_state() == 0, "model-free diagnostic fixture left nominal thermal state");
    need(batch.dispatches == 6 && !batch.capture_executes, "two-layer routed dispatch count differs");
    need(profile.mode == f.profile_mode && profile.kernels.size() == batch.dispatches, "fixture counter boundary or dispatch attribution changed");
    need(f.graph.memory().allocations == before.allocations && f.graph.memory().live_bytes == before.live_bytes, "diagnostic encoding allocated scratch");
    Snapshot snapshot;
    snapshot.dispatches = batch.dispatches;
    for (size_t layer = 0; layer < 2; ++layer) {
        snapshot.outputs[layer * 2] = f.graph.read(f.intermediate(layer, rows).view);
        snapshot.outputs[layer * 2 + 1] = f.graph.read(f.output(layer, rows).view);
    }
    for (const auto& bytes : snapshot.outputs) {
        bool nonzero = false;
        for (size_t i = 0; i < bytes.size(); i += sizeof(float)) {
            float value = 0;
            std::memcpy(&value, bytes.data() + i, sizeof(value));
            need(std::isfinite(value), "nonfinite diagnostic output");
            nonzero |= value != 0;
        }
        need(nonzero, "diagnostic output has no signal");
    }
    std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"batch_end\",\"arm\":" << arm << ",\"case\":" << case_id << ",\"dispatches\":" << batch.dispatches << ",\"counter_boundary\":\"" << counter_boundary(profile.mode) << "\",\"allocation_delta\":0,\"passed\":1}" << std::endl;
    return snapshot;
}

struct ArmResult { std::vector<Snapshot> snapshots; uint64_t allocations = 0, bytes = 0; ProfileMode profile_mode = ProfileMode::disabled; };
ArmResult run_arm(const char* library, const std::array<Packed, 3>& weights, Env& selector, uint32_t arm) {
    selector.set(arm == 0 ? nullptr : (arm == 1 ? "1" : "0"));
    Fixture f(library, weights);
    ArmResult result{{}, f.startup_allocations, f.startup_bytes, f.profile_mode};
    // The selector is captured at construction. Later environment changes must not enable this explicit zero arm.
    if (arm == 2) selector.set("1");
    if (arm == 1) {
        refuses([&] { f.graph.configure_concurrent_dispatch(true); }, "requires unprofiled", "concurrent_profile");
        const auto extra = f.graph.upload(weights[0].lo, "late diagnostic registration", true);
        refuses([&] { f.graph.register_planar_matrix(extra, {12, uint32_t(PlanarKind::q4k_coloc_gate_v3), channels, hidden, experts}); }, "all matrix registration before preparation", "late_registration");
        f.graph.release(extra);
    }
    uint32_t case_id = 0;
    for (const auto rows : {2u, 3u, 4u, 8u}) for (uint32_t pattern = 0; pattern < 3; ++pattern)
        result.snapshots.push_back(complete_batch(f, arm, case_id++, rows, pattern, arm == 1, "ordinary"));
    if (arm == 1) {
        write_inputs(f, 3, 2);
        std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"abort_begin\",\"case\":\"explicit_abort\"}" << std::endl;
        f.graph.begin(); f.encode(0, 3); f.graph.abort();
        std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"abort_end\",\"case\":\"explicit_abort\"}" << std::endl;
        need(complete_batch(f, arm, 12, 3, 2, true, "after_abort") == result.snapshots[5], "abort retry changed outputs");
        f.graph.begin(); f.encode(0, 3);
        refuses([&] { f.encode(0, 3); }, "repeated gate matrices", "repeated_gate");
        f.graph.abort();
        need(complete_batch(f, arm, 13, 3, 2, true, "after_repeated_gate") == result.snapshots[5], "refusal retry changed outputs");
        f.graph.begin();
        const auto& l = f.layers[0];
        refuses([&] { f.graph.planar_moe_gate_up_swiglu_m8(l.lo, l.hi, f.graph.slice(l.ids, 0, 3 * selected * 4), f.input(0, 3), f.intermediate(0, 3), experts, selected); }, "complete routed operator", "standalone_gate_up");
        f.graph.abort();
        need(complete_batch(f, arm, 14, 3, 2, true, "after_standalone_refusal") == result.snapshots[5], "standalone refusal retry changed outputs");
        f.graph.set_verify_nax_scope(false);
        need(complete_batch(f, arm, 15, 2, 0, false, "outside_verify_scope") == result.snapshots[0], "ordinary metadata restoration changed outputs");
        f.graph.set_verify_nax_scope(true);
        need(complete_batch(f, arm, 16, 2, 0, true, "after_outside_scope") == result.snapshots[0], "verify scope restoration changed outputs");
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test-planar_union_profile <tokspur-owned.metallib>\n"; return 2; }
    Env selector("TOKSPUR_DUMP_PLANAR_UNION_PROFILE"), known_profile("TOKSPUR_ALLOW_KNOWN_PROFILE");
    try {
        known_profile.set("1");
        for (const char* value : {"", "yes", "2"}) {
            selector.set(value);
            refuses([&] { MetalGraph invalid(argv[1], graph_budget); }, "selector must be 0 or 1", "invalid_selector");
        }
        selector.set(nullptr);
        {
            MetalGraph graph(argv[1], graph_budget);
            MoeWeights router{};
            router.router = {graph.zeros(2560ull * 512 * sizeof(float), "retired router selection weights"), 0};
            router.shared_router = {graph.zeros(2560ull * sizeof(float), "retired shared router selection weights"), 0};
            need(!verify_nax_router_window_supported(graph, router, {2560, 64, 512, 8}),
                "NAX must retain per-row routing after router-window removal");
        }
        const std::array weights{packed(12, 7), packed(13, 8), packed(8, 8)};
        const auto absent = run_arm(argv[1], weights, selector, 0);
        const auto enabled = run_arm(argv[1], weights, selector, 1);
        const auto zero = run_arm(argv[1], weights, selector, 2);
        need(absent.snapshots == enabled.snapshots && absent.snapshots == zero.snapshots, "selector changed output bytes or dispatch counts");
        need(absent.profile_mode == enabled.profile_mode && absent.profile_mode == zero.profile_mode, "selector changed the counter boundary");
        need(absent.allocations == zero.allocations && absent.bytes == zero.bytes, "disabled selector changed graph allocation");
        need(enabled.allocations == absent.allocations + 1 && enabled.bytes >= absent.bytes + 3 * 9280, "diagnostic bank allocation differs");
        selector.set("1");
        {
            Fixture missing_profile(argv[1], weights, false);
            refuses([&] { missing_profile.graph.begin(); }, "direct dispatch profiling", "missing_profile_scope");
        }
        std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"summary\",\"passed\":1,\"checks\":" << checks << ",\"counter_boundary\":\"" << counter_boundary(enabled.profile_mode) << "\",\"ordinary_cases_per_arm\":12,\"enabled_dumps_expected\":16,\"bank_logical_bytes\":27840,\"bank_charged_bytes\":" << enabled.bytes - absent.bytes << ",\"graph_budget_bytes\":" << graph_budget << "}" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "UNION_DIAGNOSTIC_TEST {\"event\":\"summary\",\"passed\":0,\"checks\":" << checks << "}\n";
        std::cerr << "diagnostic fixture failure: " << e.what() << '\n';
        return 1;
    }
}
