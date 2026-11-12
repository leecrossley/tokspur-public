#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/operators.hpp"
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#include "tokspur/grouped_mm.hpp"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

namespace {
size_t checks = 0, batches = 0;
double worst = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid graph operation was accepted");
}
std::vector<double> real(std::span<const float> input) { return {input.begin(), input.end()}; }
std::vector<float> random_values(size_t count, std::mt19937& rng, float scale = 0.125f) {
    std::vector<float> values(count);
    for (auto& value : values) value = (int(rng() % 2001) - 1000) / 1000.0f * scale;
    return values;
}
std::vector<float> read(tokspur::MetalGraph& gpu, tokspur::GpuView view) {
    const auto bytes = gpu.read(view);
    require(bytes.size() % 4 == 0, "nonfloat readback");
    std::vector<float> values(bytes.size() / 4);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}
void close(std::span<const float> actual, std::span<const double> expected, double tolerance = 5e-5) {
    require(actual.size() == expected.size(), "graph result size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i]));
        worst = std::max(worst, error);
        require(std::isfinite(actual[i]) && error <= tolerance, "graph numerical tolerance exceeded");
    }
}

struct Arena {
    tokspur::MetalGraph& gpu;
    std::vector<tokspur::GpuView> buffers;
    tokspur::GpuView zero(size_t floats) {
        auto view = gpu.zeros(floats * 4, "test scratch"); buffers.push_back(view); return view;
    }
    tokspur::GpuView upload(std::span<const float> values, bool immutable = true) {
        auto view = gpu.upload(std::as_bytes(values), "test input", immutable); buffers.push_back(view); return view;
    }
    tokspur::GpuView bytes(std::span<const std::byte> values) {
        auto view = gpu.upload(values, "test encoded weight"); buffers.push_back(view); return view;
    }
    ~Arena() {
        try { gpu.abort(); } catch (const tokspur::FormatError&) {}
        for (const auto view : buffers) gpu.release(view);
    }
};

void done(tokspur::MetalGraph& gpu, uint64_t dispatches, uint64_t allocations) {
    const auto result = gpu.submit();
    require(result.dispatches == dispatches, "wrong number of encoded dispatches");
    require(std::isfinite(result.wall_ms) && result.wall_ms > 0 && std::isfinite(result.gpu_ms) && result.gpu_ms > 0, "invalid batch timing");
    require(gpu.memory().allocations == allocations, "hidden allocation during graph execution");
    ++batches;
}

void format_cases(tokspur::MetalGraph& gpu) {
    for (uint32_t type : {0, 1, 7, 8, 12, 13, 14, 20, 30}) {
        Arena arena{gpu, {}};
        const auto& layout = tokspur::encoding(type);
        const auto columns = layout.elements_per_block * 3;
        constexpr uint32_t rows = 7;
        std::vector<std::byte> bytes(3 * rows * layout.bytes_per_block);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte((i * 37 + 19) & 255);
        const auto put = [&](size_t offset, uint64_t value, size_t width) {
            for (size_t i = 0; i < width; ++i) bytes.at(offset + i) = std::byte((value >> (i * 8)) & 255);
        };
        for (size_t at = 0; at < bytes.size(); at += layout.bytes_per_block) {
            if (type == 0) put(at, std::bit_cast<uint32_t>(0.125f), 4);
            else if (type == 30) put(at, 0x3e00, 2);
            else {
                put(at + (type == 14 ? 208 : 0), 0x2400, 2);
                if (type == 7 || type == 12 || type == 13) put(at + 2, 0x2000, 2);
            }
        }
        std::vector<float> input(columns);
        for (size_t i = 0; i < input.size(); ++i) input[i] = (int(i % 31) - 15) / 16.0f;
        const auto expected = tokspur::reference_gemv(type, bytes, columns, rows, input);
        const auto weights = arena.bytes(bytes), x = arena.upload(input), output = arena.zero(rows);
        const auto allocations = gpu.memory().allocations;
        for (size_t repeat = 0; repeat < 3; ++repeat) {
            gpu.begin(); gpu.gemv(type, weights, x, output, columns, rows);
            done(gpu, 1, allocations);
            close(read(gpu, output), expected);
        }
    }
}

void residual_cases(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    for (const auto& shape : std::array<std::array<uint32_t, 3>, 4>{{{1, 7, 3}, {4, 9, 5}, {3, 257, 16}, {4, 2560, 320}}}) {
        const auto branches = shape[0], channels = shape[1], rank = shape[2], width = branches * channels;
        Arena arena{gpu, {}};
        auto streams = random_values(width, rng, 1.0f), gains = random_values(width, rng, 0.1f);
        for (auto& value : gains) value += 1;
        const auto down = random_values(width * rank, rng, 0.01f), up = random_values(rank * width, rng, 0.01f);
        const auto inject = random_values(width * branches, rng, 0.01f);
        constexpr float epsilon = 1e-6f;
        const auto reference = tokspur::gated_residual_reference(real(streams), real(gains), real(down), real(up), real(inject), branches, channels, rank, epsilon);
        const auto stream = arena.upload(streams), norm_gain = arena.upload(gains), w_down = arena.upload(down);
        const auto w_up = arena.upload(up), w_inject = arena.upload(inject);
        const auto normalized = arena.zero(width), bottleneck = arena.zero(rank), scores = arena.zero(width);
        const auto inject_logits = arena.zero(branches), scales = arena.zero(branches), output = arena.zero(channels), written = arena.zero(width);
        const auto allocations = gpu.memory().allocations;
        gpu.begin();
        gpu.rms(stream, norm_gain, normalized, channels, epsilon);
        gpu.gemv(0, w_down, normalized, bottleneck, width, rank);
        gpu.activate(bottleneck, bottleneck, tokspur::Activation::silu, 1.0f / branches);
        gpu.gemv(0, w_up, bottleneck, scores, rank, width);
        gpu.gemv(0, w_inject, normalized, inject_logits, width, branches);
        gpu.residual_read(normalized, scores, inject_logits, output, scales, branches, channels);
        gpu.residual_write(stream, output, scales, written);
        done(gpu, 7, allocations);
        close(read(gpu, output), reference.input);
        close(read(gpu, scales), reference.write_scales);
        close(read(gpu, written), tokspur::residual_write_reference(real(streams), reference.input, reference.write_scales));
        close(read(gpu, stream), real(streams), 0);
    }
}

void delta_cases(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    for (const auto& shape : std::array<std::array<uint32_t, 3>, 4>{{{1, 1, 1}, {3, 7, 9}, {4, 32, 64}, {48, 128, 128}}}) {
        const auto heads = shape[0], k = shape[1], v = shape[2];
        Arena arena{gpu, {}};
        const auto state = random_values(heads * k * v, rng), q = random_values(heads * k, rng);
        const auto key = random_values(heads * k, rng), value = random_values(heads * v, rng);
        auto decay = random_values(heads, rng, 0.5f), beta = random_values(heads, rng, 0.5f);
        for (auto& x : decay) x += 0.5f;
        for (auto& x : beta) x += 0.5f;
        const auto expected = tokspur::gated_delta_reference(real(state), real(q), real(key), real(value), real(decay), real(beta), heads, k, v);
        const auto s = arena.upload(state), query = arena.upload(q), keys = arena.upload(key), values = arena.upload(value);
        const auto a = arena.upload(decay), b = arena.upload(beta), next = arena.zero(state.size()), output = arena.zero(value.size());
        const auto normalized = arena.zero(value.size()), gated = arena.zero(value.size());
        const auto gains = random_values(v, rng);
        const auto gain = arena.upload(gains);
        const auto allocations = gpu.memory().allocations;
        gpu.begin();
        gpu.delta(s, query, keys, values, a, b, next, output, heads, k, v);
        gpu.rms(output, gain, normalized, v, 1e-6f);
        gpu.multiply(normalized, values, gated, true);
        done(gpu, 3, allocations);
        close(read(gpu, next), expected.state);
        close(read(gpu, output), expected.output);
        close(read(gpu, s), real(state), 0);
        std::vector<double> broadcast_gains(value.size());
        for (size_t i = 0; i < broadcast_gains.size(); ++i) broadcast_gains[i] = gains[i % v];
        auto expected_gated = tokspur::group_rms_reference(expected.output, broadcast_gains, v, 1e-6f);
        for (size_t i = 0; i < expected_gated.size(); ++i) expected_gated[i] *= tokspur::stable_sigmoid(value[i]);
        close(read(gpu, gated), expected_gated);
    }
}

void convolution_cases(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    for (const auto channels : {1u, 31u, 128u, 10240u}) {
        Arena arena{gpu, {}};
        constexpr uint32_t kernel = 4;
        auto history = random_values(channels * (kernel - 1), rng);
        const auto weights = random_values(channels * kernel, rng);
        const auto w = arena.upload(weights);
        auto h = arena.upload(history, false), next = arena.zero(history.size());
        auto input = random_values(channels, rng);
        const auto x = arena.upload(input, false), output = arena.zero(channels), active = arena.zero(channels);
        const auto allocations = gpu.memory().allocations;
        for (size_t step = 0; step < 10; ++step) {
            input = random_values(channels, rng);
            gpu.write(x, std::as_bytes(std::span(input)));
            const auto expected = tokspur::causal_depthwise_reference(real(history), real(input), real(weights), kernel);
            gpu.begin();
            gpu.convolution(h, x, w, next, output, kernel);
            gpu.activate(output, active, tokspur::Activation::silu);
            done(gpu, 2, allocations);
            close(read(gpu, next), expected.history, 0);
            close(read(gpu, output), expected.output);
            close(read(gpu, h), real(history), 0);
            auto activated = expected.output;
            for (auto& value : activated) value = tokspur::stable_silu(value);
            close(read(gpu, active), activated);
            history.assign(expected.history.begin(), expected.history.end());
            std::swap(h, next);
        }
    }
}

void safety_cases(tokspur::MetalGraph& gpu, const std::filesystem::path& library) {
    Arena arena{gpu, {}};
    const std::array<float, 4> values{1, 2, 3, 4};
    const auto immutable = arena.upload(values), mutable_input = arena.upload(values, false), output = arena.zero(4);
    const auto overlapping = arena.zero(8), scalar_output = arena.zero(1);
    rejected([&] { gpu.write(immutable, std::as_bytes(std::span(values))); });
    rejected([&] { static_cast<void>(gpu.zeros(0, "empty")); });
    rejected([&] { static_cast<void>(gpu.zeros(gpu.memory().budget, "over budget")); });
    rejected([&] { static_cast<void>(gpu.slice(output, 2, 4)); });
    rejected([&] { static_cast<void>(gpu.slice(output, uint64_t(-4), 4)); });
    const auto released = gpu.zeros(16, "released");
    gpu.release(released);
    rejected([&] { static_cast<void>(gpu.read(released)); });
    rejected([&] { gpu.release(released); });
    auto foreign = output; foreign.owner += 1;
    rejected([&] { static_cast<void>(gpu.read(foreign)); });
    std::array<float, 4> prepared_readback{};
    const auto before_readback = gpu.memory();
    gpu.read(immutable, std::as_writable_bytes(std::span(prepared_readback)));
    require(prepared_readback == values && gpu.memory().allocations == before_readback.allocations &&
        gpu.memory().live_bytes == before_readback.live_bytes, "prepared readback changed values/ledger");
    rejected([&] { gpu.read(foreign, std::as_writable_bytes(std::span(prepared_readback))); });
    rejected([&] { gpu.read(released, std::as_writable_bytes(std::span(prepared_readback))); });
    rejected([&] { gpu.read(immutable, std::as_writable_bytes(std::span(prepared_readback).first(3))); });
    require(prepared_readback == values, "refused readback changed destination");
    const auto copy_a = arena.zero(4), copy_b = arena.zero(4);
    const std::array<tokspur::GraphCopy, 1> valid_copy{{{immutable, copy_b}}};
    const auto before_copy = gpu.memory(); gpu.copy(valid_copy);
    close(read(gpu, copy_b), real(values), 0);
    require(gpu.memory().allocations == before_copy.allocations && gpu.memory().live_bytes == before_copy.live_bytes,
        "idle copy allocated payload storage");
    const std::array<tokspur::GraphCopy, 2> late_immutable{{{immutable, copy_a}, {immutable, immutable}}};
    const std::array<tokspur::GraphCopy, 2> later_source{{{immutable, copy_a}, {copy_a, copy_b}}};
    const std::array<tokspur::GraphCopy, 2> earlier_source{{{copy_a, copy_b}, {immutable, copy_a}}};
    const std::array<tokspur::GraphCopy, 2> duplicate_destination{{{immutable, copy_a}, {mutable_input, copy_a}}};
    const std::array<tokspur::GraphCopy, 2> late_foreign{{{immutable, copy_a}, {foreign, copy_b}}};
    const std::array<tokspur::GraphCopy, 2> late_length{{{immutable, copy_a}, {gpu.slice(immutable, 0, 4), copy_b}}};
    rejected([&] { gpu.copy(late_immutable); });
    rejected([&] { gpu.copy(later_source); });
    rejected([&] { gpu.copy(earlier_source); });
    rejected([&] { gpu.copy(duplicate_destination); });
    rejected([&] { gpu.copy(late_foreign); });
    rejected([&] { gpu.copy(late_length); });
    rejected([&] { gpu.copy({}); });
    const std::array<tokspur::GraphCopy, 1> wrong_span{{{gpu.slice(immutable, 0, 4), copy_a}}};
    const std::array<tokspur::GraphCopy, 1> released_copy{{{released, copy_a}}};
    rejected([&] { gpu.copy(wrong_span); }); rejected([&] { gpu.copy(released_copy); });
    close(read(gpu, copy_a), std::vector<double>(4, 0), 0);
    close(read(gpu, copy_b), real(values), 0);
    const std::array<tokspur::GraphCopy, 1> seed_slices{{{immutable, gpu.slice(overlapping, 0, 16)}}};
    gpu.copy(seed_slices);
    const std::array<tokspur::GraphCopy, 2> disjoint_slices{{
        {gpu.slice(overlapping, 0, 8), gpu.slice(overlapping, 16, 8)},
        {gpu.slice(overlapping, 8, 8), gpu.slice(overlapping, 24, 8)}}};
    gpu.copy(disjoint_slices); close(read(gpu, gpu.slice(overlapping, 16, 16)), real(values), 0);
    bool cross_thread_rejected = false;
    std::thread worker([&] {
        try { static_cast<void>(gpu.memory()); } catch (const tokspur::FormatError&) { cross_thread_rejected = true; }
    });
    worker.join(); require(cross_thread_rejected, "cross-thread ownership failure");
    rejected([&] { gpu.activate(immutable, output, tokspur::Activation::identity); });
    gpu.begin();
    rejected([&] { gpu.begin(); });
    rejected([&] { static_cast<void>(gpu.submit()); });
    rejected([&] { static_cast<void>(gpu.read(output)); });
    rejected([&] { gpu.read(output, std::as_writable_bytes(std::span(prepared_readback))); });
    rejected([&] { gpu.write(mutable_input, std::as_bytes(std::span(values))); });
    rejected([&] { gpu.copy(valid_copy); });
    rejected([&] { gpu.release(output); });
    rejected([&] { static_cast<void>(gpu.zeros(4, "during batch")); });
    rejected([&] { gpu.activate(immutable, immutable, tokspur::Activation::identity); });
    rejected([&] { gpu.activate(immutable, foreign, tokspur::Activation::identity); });
    rejected([&] { gpu.activate(immutable, output, static_cast<tokspur::Activation>(99)); });
    rejected([&] { gpu.activate(gpu.slice(overlapping, 0, 16), gpu.slice(overlapping, 4, 16), tokspur::Activation::identity); });
    rejected([&] { gpu.rms(immutable, immutable, output, 3, 1e-6f); });
    rejected([&] { gpu.rms(immutable, immutable, output, 4, 0); });
    rejected([&] { gpu.delta(immutable, immutable, immutable, immutable, immutable, immutable, immutable, output, 0, 1, 1); });
    gpu.activate(immutable, output, tokspur::Activation::sigmoid);
    gpu.abort();
    close(read(gpu, output), std::vector<double>(4, 0), 0);
    gpu.begin(); gpu.activate(immutable, output, tokspur::Activation::sigmoid);
    done(gpu, 1, gpu.memory().allocations);
    std::vector<double> expected;
    for (const auto value : values) expected.push_back(tokspur::stable_sigmoid(value));
    close(read(gpu, output), expected);
    const std::array<float, 4> extreme{-1000, -90, 90, 1000};
    gpu.write(mutable_input, std::as_bytes(std::span(extreme)));
    for (const auto activation : {tokspur::Activation::sigmoid, tokspur::Activation::softplus}) {
        gpu.begin(); gpu.activate(mutable_input, output, activation);
        done(gpu, 1, gpu.memory().allocations);
        expected.clear();
        for (const auto value : extreme) expected.push_back(activation == tokspur::Activation::sigmoid ? tokspur::stable_sigmoid(value) : tokspur::stable_softplus(value));
        close(read(gpu, output), expected);
    }
    const std::array<float, 4> nonfinite{0, 1, std::numeric_limits<float>::infinity(), 2};
    gpu.write(mutable_input, std::as_bytes(std::span(nonfinite)));
    gpu.begin(); gpu.activate(mutable_input, output, tokspur::Activation::identity);
    rejected([&] { static_cast<void>(gpu.submit()); });
    gpu.begin(); gpu.activate(mutable_input, output, tokspur::Activation::sigmoid);
    rejected([&] { static_cast<void>(gpu.submit()); });
    const std::array<float, 4> overflow{1e30f, -1e30f, 1e30f, -1e30f};
    gpu.write(mutable_input, std::as_bytes(std::span(overflow)));
    gpu.begin(); gpu.rms(mutable_input, immutable, output, 4, 1e-6f);
    rejected([&] { static_cast<void>(gpu.submit()); });
    gpu.write(mutable_input, std::as_bytes(std::span(values)));
    gpu.begin(); gpu.activate(mutable_input, output, tokspur::Activation::identity);
    done(gpu, 1, gpu.memory().allocations);
    close(read(gpu, output), real(values), 0);
    gpu.begin();
    const auto one = gpu.slice(immutable, 0, 4), invalid_beta = gpu.slice(immutable, 4, 4);
    gpu.delta(one, one, one, one, one, invalid_beta, gpu.slice(output, 0, 4), scalar_output, 1, 1, 1);
    rejected([&] { static_cast<void>(gpu.submit()); });
    close(read(gpu, immutable), real(values), 0);
    const auto nonfinite_weights = arena.upload(nonfinite);
    gpu.begin(); gpu.gemv(0, nonfinite_weights, immutable, scalar_output, 4, 1);
    rejected([&] { static_cast<void>(gpu.submit()); });
    gpu.begin(); gpu.activate(immutable, output, tokspur::Activation::identity);
    done(gpu, 1, gpu.memory().allocations);
    close(read(gpu, output), real(values), 0);
    rejected([&] { tokspur::MetalGraph unsafe(library, 1); });
    rejected([&] { tokspur::MetalGraph unsafe(library, uint64_t(-1)); });
}
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
struct OperatorTestSink {
    uint64_t owner = 0, physical_count = 0, owners = 0, callbacks = 0, empty_finals = 0;
    tokspur::BatchResult accumulated{};
    bool fail_next = false;
    static bool emit(void* context, const tokspur::OperatorCheckpointRecord& record, const tokspur::BatchResult* physical, const tokspur::BatchResult& total) noexcept {
        try {
            auto& self = *static_cast<OperatorTestSink*>(context);
            if (self.fail_next) { self.fail_next = false; return false; }
            if (self.owner != record.owner_batch) { self.owner = record.owner_batch; self.accumulated = {}; }
            if (physical) { ++self.physical_count; self.accumulated.dispatches += physical->dispatches; self.accumulated.wall_ms += physical->wall_ms; self.accumulated.gpu_ms += physical->gpu_ms; }
            require(total.dispatches == self.accumulated.dispatches && total.wall_ms == self.accumulated.wall_ms && total.gpu_ms == self.accumulated.gpu_ms,
                "operator physical/owner timing and dispatch rollup differs");
            if (record.owner_final) { ++self.owners; self.empty_finals += physical == nullptr; }
            ++self.callbacks; return true;
        } catch (...) { return false; }
    }
};
void operator_checkpoint_controls(const char* library) {
    using namespace tokspur;
    {
        MetalGraph ordinary(library, 64ull << 20); ordinary.begin(); rejected([&] { static_cast<void>(ordinary.submit()); }); ordinary.abort();
        MetalGraph explicit_off(library, 64ull << 20); explicit_off.configure_operator_checkpoints(nullptr, nullptr, 0);
        explicit_off.begin(); rejected([&] { static_cast<void>(explicit_off.submit()); }); explicit_off.abort();
    }
    {
        MetalGraph gpu(library, 64ull << 20); OperatorTestSink sink;
        rejected([&] { gpu.configure_operator_checkpoints(nullptr, &sink, 16); });
        rejected([&] { gpu.configure_operator_checkpoints(OperatorTestSink::emit, nullptr, 16); });
        rejected([&] { gpu.configure_operator_checkpoints(OperatorTestSink::emit, &sink, 0); });
        rejected([&] { gpu.configure_operator_checkpoints(OperatorTestSink::emit, &sink, 16385); });
        gpu.configure_operator_checkpoints(OperatorTestSink::emit, &sink, 64);
        Arena arena{gpu,{}}; const std::array<float,4> values{1,2,3,4}; const auto x=arena.upload(values), y=arena.zero(4), z=arena.zero(4); const auto allocations=gpu.memory().allocations;
        gpu.begin(); gpu.activate(x,y,Activation::identity); gpu.operator_checkpoint("control","first",{4,0,0,0});
        gpu.activate(y,z,Activation::identity); gpu.operator_checkpoint("control","second",{4,0,0,0}); const auto total=gpu.submit();
        require(total.dispatches==2 && sink.physical_count==2 && sink.owners==1 && sink.empty_finals==1 && sink.callbacks==3,"two-segment empty-final accounting");
        close(read(gpu,z),real(values),0); require(gpu.memory().allocations==allocations,"operator controls allocated while encoding");
        rejected([&] { gpu.configure_operator_checkpoints(nullptr,nullptr,0); });
        gpu.begin(); rejected([&] { static_cast<void>(gpu.submit()); }); gpu.abort();
        gpu.begin(); gpu.activate(x,y,Activation::identity); gpu.operator_checkpoint("control","aborted",{}); gpu.abort();
        gpu.begin(); gpu.activate(x,z,Activation::identity); require(gpu.submit().dispatches==1,"abort leaked an old owner rollup");
        sink.fail_next=true; gpu.begin(); gpu.activate(x,y,Activation::identity); rejected([&] { gpu.operator_checkpoint("control","sink_failure",{}); });
        gpu.begin(); gpu.activate(x,z,Activation::identity); require(gpu.submit().dispatches==1,"sink failure leaked an old owner rollup");
        close(read(gpu,z),real(values),0);
    }
#if defined(TOKSPUR_PRIVATE_HC_EXACT) && defined(TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32)
    {
        MetalGraph gpu(library, 64ull << 20); OperatorTestSink sink; gpu.configure_small_gemv(16); gpu.configure_block_gemv(3); gpu.configure_hc_exact(HcExactMode::combined);
        gpu.configure_prefill_dense_f32(true); gpu.configure_prefill_hc_down_f32(true); gpu.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        gpu.configure_operator_checkpoints(OperatorTestSink::emit,&sink,64); Arena arena{gpu,{}};
        std::vector<std::byte> weights(3481600); for(size_t at=0;at<weights.size();at+=34){const uint16_t scale=0x1400;std::memcpy(weights.data()+at,&scale,2);for(size_t j=0;j<32;++j)weights[at+2+j]=std::byte(j%7);}
        std::vector<float> values(512*10240,0.125f); const auto w=arena.bytes(weights),x=arena.upload(values),y=arena.zero(512*320);
        const auto one_x=gpu.slice(x,0,10240*4),one_y=gpu.slice(y,0,320*4); const auto routes=gpu.hc_exact_routes(); const auto encoded=gpu.prefill_hc_down_f32_encoded_calls();
        gpu.begin(); gpu.gemv(8,w,one_x,one_y,10240,320); gpu.operator_checkpoint("control","hold_success_routes",{});
        gpu.dense_hc_down_f32(w,{x,512,10240,10240*4},{y,512,320,320*4}); gpu.abort();
        require(gpu.hc_exact_routes()==routes,"aborted owner published held successful routes");
        require(gpu.prefill_hc_down_f32_encoded_calls()==encoded+1,"abort rewound the encoded HC lifetime");
        gpu.begin(); gpu.gemv(8,w,one_x,one_y,10240,320); gpu.operator_checkpoint("control","hold_before_owner_sink_failure",{}); sink.fail_next=true;
        rejected([&] { static_cast<void>(gpu.submit()); }); require(gpu.hc_exact_routes()==routes,"failed owner sink published held routes");
        gpu.begin(); gpu.gemv(8,w,one_x,one_y,10240,320); require(gpu.submit().dispatches==1,"route owner recovery leaked old dispatches");
        auto expected=routes; ++expected[0]; require(gpu.hc_exact_routes()==expected,"successful owner did not publish exactly its current routes");
        sink.fail_next=true; gpu.begin(); gpu.dense_hc_down_f32(w,{x,512,10240,10240*4},{y,512,320,320*4}); rejected([&] { gpu.operator_checkpoint("control","encoded_sink_failure",{}); });
        require(gpu.prefill_hc_down_f32_encoded_calls()==encoded+2 && gpu.hc_exact_routes()==expected,"sink failure rewound encoded lifetime or published rejected routes");
        require(gpu.read(w)==weights && gpu.read(x)==std::vector<std::byte>(std::as_bytes(std::span(values)).begin(),std::as_bytes(std::span(values)).end()),"operator controls changed immutable input/weights");
    }
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    {
        MetalGraph policy(library,64ull<<20);
        rejected([&]{policy.configure_prefill_dense_staged_f32(true);});
        policy.configure_prefill_dense_f32(true);rejected([&]{policy.configure_prefill_dense_staged_f32(true);});
        policy.configure_prefill_grouped_f32(true);policy.configure_prefill_hc_down_f32(true);
        rejected([&]{policy.configure_prefill_dense_staged_f32(true);});policy.configure_prefill_hc_down_f32(false);
        policy.configure_prefill_dense_staged_f32(true);
        rejected([&]{policy.configure_prefill_dense_f32(false);});rejected([&]{policy.configure_prefill_grouped_f32(false);});rejected([&]{policy.configure_prefill_hc_down_f32(true);});
    }
    // Omission, explicit OFF and explicit ON compare the wrapper with the retained direct old-MPP entry point. No callback is needed for this independent route control.
    for(unsigned mode=0;mode<3;++mode){
        MetalGraph gpu(library,64ull<<20);gpu.configure_prefill_dense_f32(true);gpu.configure_prefill_grouped_f32(true);
        if(mode)gpu.configure_prefill_dense_staged_f32(mode==2);
        require(gpu.prefill_dense_staged_f32()==(mode==2)&&std::string(gpu.prefill_dense_kernel())==(mode==2?"graph_dense_q8_staged_f32":"graph_q8_mpp_wide"),"operator staged dense identity/default differs");
        constexpr uint32_t c=320,r=64,n=512;Arena arena{gpu,{}};
        std::vector<std::byte> weights(uint64_t(c/32)*r*34);for(uint64_t block=0;block<weights.size()/34;++block){const uint16_t scale=0x1400;std::memcpy(weights.data()+block*34,&scale,2);for(uint32_t j=0;j<32;++j){const int8_t q=int8_t(int((block*17+j*11)%15)-7);std::memcpy(weights.data()+block*34+2+j,&q,1);}}
        std::vector<float> values(uint64_t(c)*n);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%131)-65)/997;
        const auto w=arena.bytes(weights),x=arena.upload(values),old=arena.zero(uint64_t(r)*n),candidate=arena.zero(uint64_t(r)*n);const auto allocated=gpu.memory();
        gpu.prepare_dense_mm_q8(PrefillWeightLayout::gguf);gpu.prepare_q8_mpp(Q8MppLayout::single_wide);
        rejected([&]{gpu.configure_prefill_dense_staged_f32(mode!=2);});
        gpu.begin();gpu.q8_mpp(w,x,old,c,r,n,Q8MppLayout::single_wide);require(gpu.submit().dispatches==1,"operator old dense route dispatch count");
        gpu.begin();gpu.dense_mm_q8(PrefillWeightLayout::gguf,w,{x,n,c,c*4},{candidate,n,r,r*4});require(gpu.submit().dispatches==1,"operator selected dense route dispatch count");
        const auto expected=gpu.read(old);require(gpu.read(candidate)==expected,"operator selected dense route changed exact generated output");
        gpu.begin();gpu.dense_mm_q8(PrefillWeightLayout::gguf,w,{x,n,c,c*4},{candidate,n,r,r*4});gpu.abort();require(gpu.read(candidate)==expected,"operator staged abort changed output");
        gpu.begin();gpu.dense_mm_q8(PrefillWeightLayout::gguf,w,{x,n,c,c*4},{candidate,n,r,r*4});static_cast<void>(gpu.submit());require(gpu.read(candidate)==expected,"operator staged valid reuse changed output");
        require(gpu.memory().allocations==allocated.allocations&&gpu.memory().live_bytes==allocated.live_bytes&&gpu.prefill_hc_down_f32_encoded_calls()==0,"operator staged route allocated or inflated HC counter");
        require(gpu.read(w)==weights,"operator staged route changed weights");
    }
#endif
    std::cout<<"Generated operator checkpoint controls "<<checks<<" pass; Metal required, no model or timing admission\n";
}
#endif

#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
// gemv_rows is gemv() per row to the byte in either dispatch mode. Its staged form keeps independent rows in one barrier scope.
void row_gemv_cases(const std::filesystem::path& library, std::mt19937& rng) {
    constexpr uint32_t columns = 1024, count = 300, splits = 16;
    constexpr uint64_t stride = uint64_t(columns) * 4 + 32;
    const auto values = random_values(count * stride / 4, rng, 1.0f), matrix_values = random_values(size_t(columns) * 80, rng);
    for (const bool concurrent : {false, true}) {
        tokspur::MetalGraph gpu(library, 64 * 1024 * 1024);
        gpu.configure_small_gemv(splits); gpu.configure_concurrent_dispatch(concurrent);
        Arena arena{gpu, {}};
        const auto input = arena.upload(values);
        // One and three outputs share partial-sum regions freely, 64 leaves two regions, and 80 is an ordinary single-stage GEMV.
        for (const uint32_t rows : {1u, 3u, 64u, 80u}) {
            const auto matrix = arena.upload(std::span(matrix_values).first(size_t(columns) * rows));
            const auto expected = arena.zero(size_t(count) * rows), actual = arena.zero(size_t(count) * rows);
            const auto allocations = gpu.memory().allocations;
            const uint64_t dispatches = uint64_t(count) * gpu.gemv_dispatches(0, columns, rows);
            gpu.begin();
            for (uint32_t t = 0; t < count; ++t)
                gpu.gemv(0, matrix, gpu.slice(input, t * stride, uint64_t(columns) * 4), gpu.slice(expected, uint64_t(t) * rows * 4, uint64_t(rows) * 4), columns, rows);
            done(gpu, dispatches, allocations);
            const auto row_barriers = gpu.concurrent_barriers();
            gpu.begin();
            gpu.gemv_rows(0, matrix, {input, count, columns, stride}, {actual, count, rows, uint64_t(rows) * 4});
            done(gpu, dispatches, allocations);
            require(gpu.read(actual) == gpu.read(expected), "row GEMV differs from per-row GEMV");
            if (!concurrent) continue;
            if (rows > 64) { require(row_barriers == 0 && gpu.concurrent_barriers() == 0, "independent single-stage rows took a barrier"); continue; }
            // Per row: one barrier before each reduce and one before each reuse of the shared partial sums. Staged: the same two per group of regions.
            const uint64_t regions = 64 * 32 / (uint64_t(rows) * splits), groups = (count + regions - 1) / regions;
            require(row_barriers == 2 * uint64_t(count) - 1 && gpu.concurrent_barriers() == 2 * groups - 1, "staged row GEMV barrier count");
        }
        // Outputs written into the padding of the input rows: staging would let one row's result precede another row's read, so the per-row order stays.
        const auto rows_and_padding = arena.upload(values, false), expected = arena.zero(count);
        const auto matrix = arena.upload(std::span(matrix_values).first(columns));
        const auto allocations = gpu.memory().allocations;
        gpu.begin();
        for (uint32_t t = 0; t < count; ++t) gpu.gemv(0, matrix, gpu.slice(input, t * stride, uint64_t(columns) * 4), gpu.slice(expected, uint64_t(t) * 4, 4), columns, 1);
        done(gpu, 2 * count, allocations);
        gpu.begin();
        gpu.gemv_rows(0, matrix, {rows_and_padding, count, columns, stride}, {gpu.slice(rows_and_padding, uint64_t(columns) * 4, (count - 1) * stride + 4), count, 1, stride});
        done(gpu, 2 * count, allocations);
        require(!concurrent || gpu.concurrent_barriers() == 2 * uint64_t(count) - 1, "aliased row GEMV left the per-row order");
        const auto padded = gpu.read(rows_and_padding), reference = gpu.read(expected);
        for (uint32_t t = 0; t < count; ++t)
            require(std::memcmp(padded.data() + t * stride + uint64_t(columns) * 4, reference.data() + uint64_t(t) * 4, 4) == 0, "aliased row GEMV result");
        gpu.begin();
        rejected([&] { gpu.gemv_rows(0, matrix, {input, count, columns, stride - 36}, {expected, count, 1, 4}); });
        rejected([&] { gpu.gemv_rows(0, matrix, {input, count + 1, columns, stride}, {expected, count + 1, 1, 4}); });
        rejected([&] { gpu.gemv_rows(0, matrix, {input, count, columns, stride}, {expected, count - 1, 1, 4}); });
        gpu.abort();
    }
}

// A barrier exactly where a dispatch touches bytes an earlier one in the same scope wrote, or writes bytes one read: never for bytes that merely lie between them.
void concurrent_hazard_cases(const std::filesystem::path& library, std::mt19937& rng) {
    constexpr uint32_t width = 64;
    const auto a = random_values(width, rng, 1.0f), gate = random_values(width, rng, 1.0f), rows = random_values(width * 3, rng, 1.0f);
    std::vector<std::byte> serial;
    for (const bool concurrent : {false, true}) {
        tokspur::MetalGraph gpu(library, 16 * 1024 * 1024);
        gpu.configure_concurrent_dispatch(concurrent);
        Arena arena{gpu, {}};
        const auto x = arena.upload(a), g = arena.upload(gate), b = arena.upload(rows, false), c = arena.zero(width * 2);
        const auto row = [&](tokspur::GpuView view, uint32_t at) { return gpu.slice(view, uint64_t(at) * width * 4, uint64_t(width) * 4); };
        const auto allocations = gpu.memory().allocations;
        // Each batch touches rows 0 and 2 of one buffer, then makes one more access. Only the last three accesses conflict.
        struct Access { bool write; uint32_t at; };
        const auto batch = [&](std::array<Access, 3> accesses, uint64_t barriers, const char* why) {
            gpu.begin();
            uint32_t reads = 0;
            for (const auto access : accesses) {
                if (access.write) gpu.multiply(x, g, row(b, access.at)); else gpu.multiply(row(b, access.at), g, row(c, reads++));
            }
            done(gpu, 3, allocations);
            require(gpu.concurrent_barriers() == (concurrent ? barriers : 3u), why);
        };
        batch({{{true, 0}, {true, 2}, {false, 1}}}, 0, "read between two written rows took a barrier");
        batch({{{false, 0}, {false, 2}, {true, 1}}}, 0, "write between two read rows took a barrier");
        batch({{{true, 0}, {true, 2}, {false, 2}}}, 1, "read of a written row took no barrier");
        batch({{{false, 0}, {false, 2}, {true, 2}}}, 1, "write of a read row took no barrier");
        batch({{{true, 0}, {true, 2}, {true, 0}}}, 1, "rewrite of a written row took no barrier");
        auto bytes = gpu.read(b); const auto more = gpu.read(c); bytes.insert(bytes.end(), more.begin(), more.end());
        if (!concurrent) serial = bytes; else require(bytes == serial, "concurrent hazard scope changed results");
    }
}
#endif

#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
// Only whole registered matrices and bytes outside every matrix may be bound from packed weight storage.
void packed_storage_cases(const std::filesystem::path& library) {
    tokspur::MetalGraph gpu(library, 16 * 1024 * 1024);
    Arena arena{gpu, {}};
    constexpr uint64_t lane = 4 * 34; // Q8, 32 columns, four rows
    const auto storage = arena.bytes(std::vector<std::byte>(1024));
    for (const uint64_t at : {640u, 64u, 320u}) gpu.register_lane_matrix(gpu.slice(storage, at, lane), {8, 32, 4, 1});
    require(gpu.lane_matrix_count() == 3, "lane fixture registration");
    const auto bound = [&](uint64_t offset, uint64_t bytes) {
        const auto view = gpu.slice(storage, offset, bytes), gate = arena.zero(bytes / 4), output = arena.zero(bytes / 4);
        gpu.begin(); gpu.multiply(view, gate, output); static_cast<void>(gpu.submit()); ++batches; ++checks;
    };
    const auto refused = [&](uint64_t offset, uint64_t bytes) {
        const auto view = gpu.slice(storage, offset, bytes), gate = arena.zero(bytes / 4), output = arena.zero(bytes / 4);
        gpu.begin(); rejected([&] { gpu.multiply(view, gate, output); }); gpu.abort();
    };
    bound(0, 64); bound(200, 120); bound(456, 184); bound(776, 248); // before, between and after the matrices
    bound(200, 4); bound(316, 4);                                    // touching a matrix end and a matrix start
    refused(196, 8);   // across the end of the first matrix
    refused(316, 8);   // across the start of the second
    refused(324, 8);   // inside the second
    refused(636, 144); // around the whole third
    refused(64, 8);    // the first bytes of the first
    refused(768, 8);   // the last bytes of the third
    refused(64, lane); // a whole matrix, but not through a layout-aware consumer
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
        if (argc == 3 && std::string(argv[2]) == "--operator-checkpoints") { operator_checkpoint_controls(argv[1]); return 0; }
#endif
        if (argc != 2) throw std::runtime_error("expected owned metallib");
        tokspur::MetalGraph gpu(argv[1], 256 * 1024 * 1024);
        const auto base = gpu.memory().live_bytes;
        std::mt19937 rng(0x67726170);
        format_cases(gpu);
        residual_cases(gpu, rng);
        delta_cases(gpu, rng);
        convolution_cases(gpu, rng);
        safety_cases(gpu, argv[1]);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        row_gemv_cases(argv[1], rng);
        concurrent_hazard_cases(argv[1], rng);
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        packed_storage_cases(argv[1]);
#endif
        require(gpu.memory().live_bytes == base, "graph allocation leak");
        std::cout << "Persistent Metal graph checks passed: " << checks << ", batches: " << batches
                  << ", max scaled error: " << worst << ", peak allocated bytes: " << gpu.memory().peak_bytes
                  << ", device: " << gpu.device_name() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Persistent Metal graph check failed: " << error.what() << '\n';
        return 1;
    }
}
