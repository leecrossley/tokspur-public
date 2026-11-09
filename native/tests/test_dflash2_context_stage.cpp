#include "tokspur/dflash2_drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using namespace tokspur;

namespace {
void check(bool ok, const char* why) { if (!ok) throw FormatError(why); }
template<class F> void rejects(F&& action) {
    try { action(); } catch (const FormatError&) { return; }
    throw FormatError("invalid pending context admitted");
}
uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return uint16_t(bits >> 16);
}
float bf16_to_f32(uint16_t raw) {
    const uint32_t bits = uint32_t(raw) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
void check_dflash2_bf16_kernel(const char* metallib) {
    MetalGraph graph(metallib, 8ull << 20);
    graph.prepare_dflash2();
    const auto run_dflash2 = [&](GpuView w, GpuRowsF32 in, GpuRowsF32 out, bool swiglu) {
        struct ScopeGuard {
            MetalGraph& g;
            ~ScopeGuard() { try { g.set_dflash2_scope(false); } catch (...) {} }
        };
        graph.begin();
        ScopeGuard guard{graph};
        graph.set_dflash2_scope(true);
        graph.dflash2_dense_bf16(w, in, out, swiglu);
        static_cast<void>(graph.submit());
    };
    constexpr uint32_t columns = 64, rows = 37;
    std::vector<uint16_t> w_bf16(size_t(rows) * columns);
    for (size_t i = 0; i < w_bf16.size(); ++i) {
        w_bf16[i] = f32_to_bf16(float(int(i % 29) - 14) * 0.0625f);
    }
    const auto w_gpu = graph.upload(std::as_bytes(std::span(w_bf16)), "dflash2 bf16 test weights");
    for (uint32_t tokens : {1u, 8u, 17u}) {
        std::vector<float> x_host(size_t(tokens) * columns), gate_host(size_t(tokens) * rows), out_host(size_t(tokens) * rows);
        for (size_t i = 0; i < x_host.size(); ++i) x_host[i] = float(int((i * 7 + 3) % 31) - 15) * 0.125f;
        for (size_t i = 0; i < gate_host.size(); ++i) gate_host[i] = float(int((i * 5 + 1) % 23) - 11) * 0.25f;
        const auto x_gpu = graph.upload(std::as_bytes(std::span(x_host)), "dflash2 bf16 test input");
        const auto out_gpu = graph.zeros(out_host.size() * sizeof(float), "dflash2 bf16 test output");
        const auto x_rows = GpuRowsF32{x_gpu, tokens, columns, uint64_t(columns) * sizeof(float)};
        const auto out_rows = GpuRowsF32{out_gpu, tokens, rows, uint64_t(rows) * sizeof(float)};
        run_dflash2(w_gpu, x_rows, out_rows, false);
        graph.read(out_gpu, std::as_writable_bytes(std::span(out_host)));
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t r = 0; r < rows; ++r) {
                float ref = 0.0f;
                for (uint32_t c = 0; c < columns; ++c) {
                    ref += bf16_to_f32(w_bf16[size_t(r) * columns + c]) * x_host[size_t(t) * columns + c];
                }
                check(std::fabs(out_host[size_t(t) * rows + r] - ref) <= 1e-5f, "dflash2_dense_bf16 linear mismatch");
            }
        }
        graph.write(out_gpu, std::as_bytes(std::span(gate_host)));
        run_dflash2(w_gpu, x_rows, out_rows, true);
        graph.read(out_gpu, std::as_writable_bytes(std::span(out_host)));
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t r = 0; r < rows; ++r) {
                float up = 0.0f;
                for (uint32_t c = 0; c < columns; ++c) {
                    up += bf16_to_f32(w_bf16[size_t(r) * columns + c]) * x_host[size_t(t) * columns + c];
                }
                const float g = gate_host[size_t(t) * rows + r];
                const float sig = g >= 0.0f ? 1.0f / (1.0f + std::exp(-g)) : std::exp(g) / (1.0f + std::exp(g));
                const float ref = (g * sig) * up;
                check(std::fabs(out_host[size_t(t) * rows + r] - ref) <= 1e-4f, "dflash2_dense_bf16 swiglu mismatch");
            }
        }
        graph.release(out_gpu);
        graph.release(x_gpu);
    }
    // Non-finite input, BF16 weight and SwiGLU gate guards must all trip sticky error.
    {
        std::vector<float> bad_x(columns, 1.0f);
        bad_x[7] = std::numeric_limits<float>::quiet_NaN();
        const auto x_gpu = graph.upload(std::as_bytes(std::span(bad_x)), "dflash2 bad input", false);
        const auto out_gpu = graph.zeros(rows * sizeof(float), "dflash2 bad output");
        const auto x_rows = GpuRowsF32{x_gpu, 1u, columns, uint64_t(columns) * sizeof(float)};
        const auto out_rows = GpuRowsF32{out_gpu, 1u, rows, uint64_t(rows) * sizeof(float)};
        rejects([&] { run_dflash2(w_gpu, x_rows, out_rows, false); });
        std::fill(bad_x.begin(), bad_x.end(), 1.0f);
        graph.write(x_gpu, std::as_bytes(std::span(bad_x)));
        std::vector<float> bad_gate(rows, 0.5f);
        bad_gate[3] = std::numeric_limits<float>::infinity();
        graph.write(out_gpu, std::as_bytes(std::span(bad_gate)));
        rejects([&] { run_dflash2(w_gpu, x_rows, out_rows, true); });
        std::vector<uint16_t> bad_w = w_bf16;
        bad_w[19] = 0x7f80u;
        const auto bad_w_gpu = graph.upload(std::as_bytes(std::span(bad_w)), "dflash2 bad weight");
        std::fill(bad_gate.begin(), bad_gate.end(), 0.0f);
        graph.write(out_gpu, std::as_bytes(std::span(bad_gate)));
        rejects([&] { run_dflash2(bad_w_gpu, x_rows, out_rows, false); });
        graph.release(bad_w_gpu);
        graph.release(out_gpu);
        graph.release(x_gpu);
    }
    graph.release(w_gpu);
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: test-dflash2_context_stage METALLIB");
        check_dflash2_bf16_kernel(argv[1]);
        MetalGraph graph(argv[1], 1ull << 20);
        const auto initial_bytes = graph.memory().live_bytes;
        constexpr uint32_t width = 32;
        const auto source = graph.zeros(8 * width * sizeof(float), "generated target tap");
        const auto owned = graph.zeros(source.bytes, "generated owned pending context");
        DFlash2ContextStage stage;
        stage.prepare(owned, width);
        rejects([&] { stage.prepare(owned, width); });
        const auto allocations = graph.memory().allocations;
        std::array<float, 8 * width> original{}, replacement{}, actual{};
        for (size_t i = 0; i < original.size(); ++i) { original[i] = float(i + 1); replacement[i] = -float(i + 1); }
        graph.write(source, std::as_bytes(std::span(original)));
        stage.stage(graph, source, 3, 100);
        // An ordinary target step reuses its tap before pending draft priming runs.
        graph.write(source, std::as_bytes(std::span(replacement)));
        graph.read(stage.storage(), std::as_writable_bytes(std::span(actual)));
        check(std::equal(original.begin(), original.begin() + 3 * width, actual.begin()), "target tap overwrite corrupted pending context");
        rejects([&] { stage.stage(graph, source, 1, 103); });
        stage.truncate(103);
        check(stage.rows() == 3 && stage.start() == 100, "same-frontier truncate lost pending context");
        stage.truncate(102);
        check(stage.rows() == 2 && stage.start() == 100, "partial truncate lost surviving pending prefix");
        stage.truncate(100);
        check(!stage.rows(), "truncate retained discarded rows");
        rejects([&] { stage.stage(graph, source, 9, 100); });
        rejects([&] { stage.stage(graph, source, 1, UINT32_MAX); });
        replacement[0] = std::numeric_limits<float>::quiet_NaN();
        graph.write(source, std::as_bytes(std::span(replacement)));
        rejects([&] { stage.stage(graph, source, 1, 100); });
        check(!stage.rows(), "nonfinite refusal published pending context");
        graph.write(source, std::as_bytes(std::span(original)));
        stage.stage(graph, source, 8, 0);
        check(stage.rows() == 8 && !stage.start(), "full pending block failed");
        stage.clear();
        check(!stage.rows() && graph.memory().allocations == allocations, "pending reset changed allocation or retained rows");
        graph.release(owned); graph.release(source);
        check(graph.memory().live_bytes == initial_bytes, "pending fixture leaked GPU storage");
        std::cout << "DFlash2 pending context: owned copy, tap overwrite, truncate, reset and refusal passed; generated buffers only\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
