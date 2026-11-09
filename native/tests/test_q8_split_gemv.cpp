#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/q8_split.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>

namespace {
using namespace tokspur;
constexpr std::array<uint32_t, 6> splits{1, 2, 4, 8, 16, 32};
constexpr float partial_sentinel = 321.25f;
size_t checks = 0, batches = 0;
double worst = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid Q8-split request admitted");
}
void same_resources(GraphMemory a, GraphMemory b) {
    require(a.live_bytes == b.live_bytes && a.allocations == b.allocations &&
        a.imported_backing_bytes == b.imported_backing_bytes &&
        a.profile_counter_payload_bytes == b.profile_counter_payload_bytes &&
        a.capture_resource_bytes == b.capture_resource_bytes && a.live_captures == b.live_captures,
        "Q8-split unexpectedly changed the resource ledger");
}
struct Arena {
    MetalGraph& graph;
    std::vector<GpuView> all;
    GpuView bytes(std::span<const std::byte> data, bool immutable = true) {
        std::vector<std::byte> padded(data.size() + 8, std::byte{0x6d});
        std::copy(data.begin(), data.end(), padded.begin() + 4);
        const auto full = graph.upload(padded, "Q8-split generated fixture", immutable);
        all.push_back(full);
        return graph.slice(full, 4, data.size());
    }
    GpuView floats(std::span<const float> values, bool immutable = true) { return bytes(std::as_bytes(values), immutable); }
    GpuView fill(size_t count, float value = 0) { return floats(std::vector<float>(count, value), false); }
    void guards() const {
        for (auto full : all) {
            const auto data = graph.read(full);
            for (size_t i = 0; i < 4; ++i)
                require(data[i] == std::byte{0x6d} && data[data.size() - 1 - i] == std::byte{0x6d},
                    "Q8-split wrote outside an admitted view");
        }
    }
    ~Arena() { try { graph.abort(); } catch (const FormatError&) {} for (auto full : all) graph.release(full); }
};
std::vector<float> read_floats(MetalGraph& graph, GpuView view) {
    const auto data = graph.read(view);
    std::vector<float> values(data.size() / 4);
    std::memcpy(values.data(), data.data(), data.size());
    return values;
}
void scratch_tail(MetalGraph& graph, GpuView maximum, uint64_t active_bytes) {
    const auto values = read_floats(graph, maximum);
    for (size_t i = active_bytes / 4; i < values.size(); ++i)
        require(values[i] == partial_sentinel, "Q8-split wrote outside the active scratch prefix");
}
void submit(MetalGraph& graph, uint64_t dispatches, GraphMemory before) {
    const auto result = graph.submit(); ++batches;
    require(result.dispatches == dispatches, "Q8-split physical dispatch count differs");
    same_resources(before, graph.memory());
}
void close(double actual, double expected) {
    const double error = std::abs(actual - expected) / std::max(1.0, std::abs(expected));
    worst = std::max(worst, error);
    require(std::isfinite(actual) && std::isfinite(expected) && error <= 5e-5,
        "Q8-split double reference error exceeds limit");
}
void scale(std::vector<std::byte>& matrix, size_t block, uint16_t bits) {
    matrix[block * 34] = std::byte(bits & 255); matrix[block * 34 + 1] = std::byte(bits >> 8);
}
// Scalar double accumulation is independent of the GPU's SIMD reductions.
std::vector<double> partial_reference(std::span<const std::byte> matrix, std::span<const float> input,
                                      uint32_t columns, uint32_t rows, uint32_t count) {
    const uint32_t blocks = columns / 32;
    std::vector<double> values(uint64_t(rows) * count);
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t part = 0; part < count; ++part) {
        double sum = 0;
        for (uint32_t block = blocks * part / count; block < blocks * (part + 1) / count; ++block) {
            const auto at = (uint64_t(row) * blocks + block) * 34;
            const auto bits = uint16_t(std::to_integer<uint8_t>(matrix[at])) |
                uint16_t(std::to_integer<uint8_t>(matrix[at + 1])) << 8;
            const double d = binary16(bits);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const int byte = std::to_integer<uint8_t>(matrix[at + 2 + lane]);
                sum += d * (byte < 128 ? byte : byte - 256) * double(input[block * 32 + lane]);
            }
        }
        values[uint64_t(row) * count + part] = sum;
    }
    return values;
}
void exercise(MetalGraph& graph, uint32_t columns, uint32_t rows,
              const std::vector<std::byte>& matrix, const std::vector<float>& input) {
    Arena arena{graph, {}};
    const auto weights = arena.bytes(matrix), x = arena.floats(input, false);
    const auto output = arena.fill(rows, -123.5f), baseline = arena.fill(rows);
    const auto policy = graph.kernel_policy();
    const auto reference = reference_gemv(8, matrix, columns, rows, input);
    graph.configure_block_gemv(1);
    const auto baseline_before = graph.memory();
    graph.begin(); graph.gemv(8, weights, x, baseline, columns, rows);
    submit(graph, 1, baseline_before);
    const auto exact = graph.read(baseline);
    graph.configure_block_gemv(policy[1]);
    const auto maximum_bytes = q8_split_bytes({columns, rows, 32}).partials;
    const auto prior = graph.memory();
    const auto maximum = arena.fill(maximum_bytes / 4, partial_sentinel);
    const auto fixed = graph.memory();
    require(fixed.allocations == prior.allocations + 1 && fixed.live_bytes >= prior.live_bytes + maximum_bytes,
        "caller-owned maximum Q8-split scratch was not charged exactly once");
    const std::vector<float> sentinel(maximum_bytes / 4, partial_sentinel);
    for (auto count : splits) {
        const auto bytes = q8_split_bytes({columns, rows, count});
        require(bytes.matrix == uint64_t(columns / 32) * rows * 34 && bytes.input == uint64_t(columns) * 4 &&
            bytes.output == uint64_t(rows) * 4 && bytes.partials == uint64_t(rows) * count * 4,
            "Q8-split logical geometry byte counts differ");
        const auto partials = graph.slice(maximum, 0, bytes.partials);
        require(partials.owner == maximum.owner && partials.slot == maximum.slot && partials.offset == maximum.offset,
            "Q8-split did not reuse the one maximum scratch allocation");
        graph.write(maximum, std::as_bytes(std::span(sentinel)));
        same_resources(fixed, graph.memory());
        const auto expected = partial_reference(matrix, input, columns, rows, count);
        std::vector<std::byte> first_output, first_partials;
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            const auto before = graph.memory();
            graph.begin(); graph.q8_matvec_split(weights, x, output, partials, columns, rows, count);
            submit(graph, 2, before);
            const auto values = read_floats(graph, output), parts = read_floats(graph, partials);
            for (uint32_t row = 0; row < rows; ++row) {
                close(values[row], reference[row]);
                volatile float reduced = parts[uint64_t(row) * count];
                for (uint32_t part = 0; part < count; ++part) {
                    close(parts[uint64_t(row) * count + part], expected[uint64_t(row) * count + part]);
                    if (columns / 32 * part / count == columns / 32 * (part + 1) / count)
                        require(parts[uint64_t(row) * count + part] == 0, "empty Q8-split partition was not written as zero");
                    if (part) reduced = reduced + parts[uint64_t(row) * count + part];
                }
                require(std::bit_cast<uint32_t>(float(reduced)) == std::bit_cast<uint32_t>(values[row]),
                    "Q8-split reduction did not use ascending partial order");
            }
            const auto result = graph.read(output), scratch = graph.read(partials);
            scratch_tail(graph, maximum, bytes.partials);
            if (count == 1) require(result == exact && scratch == exact, "split one differs from fast block-Q8 bits");
            if (!repeat) { first_output = result; first_partials = scratch; }
            else require(result == first_output && scratch == first_partials, "Q8-split repeat changed result bits");
        }
        same_resources(fixed, graph.memory());
    }
    require(graph.read(weights) == matrix && graph.read(x) ==
        std::vector<std::byte>(std::as_bytes(std::span(input)).begin(), std::as_bytes(std::span(input)).end()),
        "Q8-split changed weights or inputs");
    require(graph.kernel_policy() == policy, "Q8-split changed execution policy");
    arena.guards();
}
void joint_maximum(const char* library) {
    // The unchanged Q8 matrix alone is 68 MiB: use a separate, explicit bounded
    // budget rather than accidentally asking the ordinary 64-MiB fixture to fit.
    MetalGraph graph(library, 128 * 1024 * 1024);
    const auto initial = graph.memory();
    graph.prepare_q8_matvec_split();
    same_resources(initial, graph.memory());
    constexpr uint32_t columns = 65536, rows = 1024;
    const auto bytes = q8_split_bytes({columns, rows, 32});
    std::vector<std::byte> matrix(bytes.matrix, std::byte{1});
    // Each row has its own exactly represented positive half scale, q=1 and
    // x=2^-16. Full sums equal that row's scale; every equal partition is scale/S.
    // FP32 lane sums need at most 22 significant bits, so this geometry oracle
    // is exact without seven extra 67-million-element scalar reference passes.
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t block = 0; block < columns / 32; ++block)
        scale(matrix, uint64_t(row) * (columns / 32) + block, uint16_t(0x3c00 + row));
    const std::vector<float> input(columns, 0x1p-16f), sentinel(bytes.partials / 4, partial_sentinel);
    { Arena arena{graph, {}};
      const auto weights = arena.bytes(matrix), x = arena.floats(input);
      const auto output = arena.fill(rows), baseline = arena.fill(rows);
      const auto before_scratch = graph.memory();
      const auto maximum = arena.fill(bytes.partials / 4, partial_sentinel);
      const auto fixed = graph.memory();
      require(fixed.allocations == before_scratch.allocations + 1 &&
          fixed.live_bytes >= before_scratch.live_bytes + bytes.partials,
          "joint-maximum scratch was not charged exactly once");
      const auto policy = graph.kernel_policy();
      graph.configure_block_gemv(1); graph.begin(); graph.gemv(8, weights, x, baseline, columns, rows);
      submit(graph, 1, fixed); const auto exact = graph.read(baseline);
      graph.configure_block_gemv(policy[1]);
      for (auto count : splits) {
          const auto active = q8_split_bytes({columns, rows, count}).partials;
          const auto partials = graph.slice(maximum, 0, active);
          require(partials.owner == maximum.owner && partials.slot == maximum.slot && partials.offset == maximum.offset,
              "joint-maximum split replaced the prepared scratch allocation");
          graph.write(maximum, std::as_bytes(std::span(sentinel)));
          graph.begin(); graph.q8_matvec_split(weights, x, output, partials, columns, rows, count);
          submit(graph, 2, fixed);
          const auto values = read_floats(graph, output), parts = read_floats(graph, partials);
          for (uint32_t row = 0; row < rows; ++row) {
              const float expected = binary16(uint16_t(0x3c00 + row)), part = expected / float(count);
              require(std::bit_cast<uint32_t>(values[row]) == std::bit_cast<uint32_t>(expected),
                  "joint-maximum output differs from exact analytic row sum");
              for (uint32_t split = 0; split < count; ++split)
                  require(std::bit_cast<uint32_t>(parts[uint64_t(row) * count + split]) == std::bit_cast<uint32_t>(part),
                      "joint-maximum partial differs from exact analytic partition");
          }
          if (count == 1) require(graph.read(output) == exact && graph.read(partials) == exact,
              "joint-maximum split one differs from fast block-Q8 bits");
          scratch_tail(graph, maximum, active); same_resources(fixed, graph.memory());
      }
      require(graph.kernel_policy() == policy, "joint-maximum split mutated kernel policy");
      arena.guards(); }
    require(graph.memory().live_bytes == initial.live_bytes && graph.memory().capture_resource_bytes == initial.capture_resource_bytes,
        "joint-maximum fixture leaked GPU resources");
}
void admission(MetalGraph& graph) {
    Arena arena{graph, {}};
    constexpr uint32_t columns = 32, rows = 32;
    const auto matrix = arena.bytes(std::vector<std::byte>(34 * rows), false), input = arena.fill(columns);
    const auto output = arena.fill(rows, -3), partials = arena.fill(rows, 7);
    const std::array valid{matrix, input, output, partials};
    const auto out_before = graph.read(output), partials_before = graph.read(partials);
    auto untouched = [&] { require(graph.read(output) == out_before && graph.read(partials) == partials_before,
        "Q8-split encode refusal published output or scratch"); };
    auto call = [&](std::array<GpuView, 4> views, uint32_t c = 32, uint32_t r = 32, uint32_t s = 1) {
        graph.q8_matvec_split(views[0], views[1], views[2], views[3], c, r, s);
    };
    auto refused = [&](const std::function<void()>& action) {
        const auto before = graph.memory(); graph.begin(); reject(action); graph.abort();
        untouched(); same_resources(before, graph.memory());
    };
    reject([&] { call(valid); });
    refused([&] { graph.prepare_q8_matvec_split(); });
    for (auto shape : std::array<Q8SplitShape, 10>{{{0,32,1},{31,32,1},{33,32,1},{65568,32,1},
        {32,0,1},{32,1025,1},{32,32,0},{32,32,3},{32,32,64},{UINT32_MAX,32,1}}})
        refused([&] { call(valid, shape.columns, shape.rows, shape.splits); });
    for (size_t i = 0; i < valid.size(); ++i) {
        auto bad = valid; bad[i].bytes -= i ? 4 : 1;
        refused([&] { call(bad); });
        bad = valid; bad[i].bytes += 4;
        refused([&] { call(bad); });
        bad = valid; ++bad[i].owner;
        refused([&] { call(bad); });
        bad = valid; bad[i].offset += 2;
        refused([&] { call(bad); });
        bad = valid; bad[i].offset += 8192;
        refused([&] { call(bad); });
    }
    // All write/read pairs and the write/write pair, including shifted overlap.
    for (size_t destination : {size_t(2), size_t(3)}) for (size_t source : {size_t(0), size_t(1)}) {
        auto bad = valid; bad[destination] = graph.slice(valid[source], 0, rows * 4);
        refused([&] { call(bad); });
    }
    auto bad = valid; bad[3] = output; refused([&] { call(bad); });
    const auto shared = arena.fill(rows * 2 + 1);
    bad = valid; bad[2] = graph.slice(shared, 0, rows * 4); bad[3] = graph.slice(shared, 4, rows * 4);
    refused([&] { call(bad); });
    const auto immutable = arena.floats(std::vector<float>(rows));
    for (size_t destination : {size_t(2), size_t(3)}) {
        bad = valid; bad[destination] = immutable; refused([&] { call(bad); });
    }
    const auto stale = graph.zeros(columns * 4, "released Q8-split fixture"); graph.release(stale);
    bad = valid; bad[1] = stale; refused([&] { call(bad); });
    bool thread_refused = false;
    std::thread prepare_thread([&] { try { graph.prepare_q8_matvec_split(); } catch (const FormatError&) { thread_refused = true; } });
    prepare_thread.join(); require(thread_refused, "Q8-split preparation admitted a foreign host thread");
    graph.begin(); thread_refused = false;
    std::thread encode_thread([&] { try { call(valid); } catch (const FormatError&) { thread_refused = true; } });
    encode_thread.join(); graph.abort(); require(thread_refused, "Q8-split encode admitted a foreign host thread"); untouched();
    graph.begin_indirect_compatible(); reject([&] { call(valid); }); graph.abort(); untouched();
    const auto before_capture = graph.memory();
    graph.begin_capture(2); reject([&] { call(valid); }); reject([&] { graph.prepare_q8_matvec_split(); }); graph.abort();
    untouched(); require(graph.memory().live_bytes == before_capture.live_bytes && graph.memory().capture_resource_bytes == 0,
        "refused Q8-split capture leaked resources");
    const auto before_budget = graph.memory();
    reject([&] { static_cast<void>(graph.zeros(before_budget.budget, "over-budget Q8-split fixture")); });
    same_resources(before_budget, graph.memory());
    static_cast<void>(graph.configure_profiling(1));
    refused([&] { call(valid); });
    static_cast<void>(graph.configure_profiling(2));
    const auto profile_before = graph.memory();
    graph.begin(); graph.activate(input, graph.slice(shared, rows * 4, columns * 4), Activation::identity);
    reject([&] { call(valid); }); graph.abort(); untouched(); same_resources(profile_before, graph.memory());
    const auto profile_valid_before = graph.memory();
    graph.begin(); call(valid); submit(graph, 2, profile_valid_before);
    static_cast<void>(graph.configure_profiling(0));
    // Adjacent non-overlapping destinations in one allocation are valid.
    auto disjoint = valid; disjoint[2] = graph.slice(shared, 0, rows * 4); disjoint[3] = graph.slice(shared, rows * 4, rows * 4);
    const auto disjoint_before = graph.memory();
    graph.begin(); call(disjoint); submit(graph, 2, disjoint_before);
    for (auto view : {disjoint[2], disjoint[3]}) for (auto value : read_floats(graph, view)) require(value == 0, "disjoint slice result differs");
    // Protected reads may overlap each other; only writes must be disjoint.
    auto shared_reads = valid; shared_reads[1] = graph.slice(matrix, 0, columns * 4);
    const auto reads_before = graph.memory();
    graph.begin(); call(shared_reads); submit(graph, 2, reads_before);
    for (auto view : {output, partials}) for (auto value : read_floats(graph, view)) require(value == 0, "read/read overlap result differs");
    arena.guards();
}
void nonfinite(MetalGraph& graph) {
    Arena arena{graph, {}};
    constexpr uint32_t columns = 1024, rows = 5, count = 32;
    std::vector<std::byte> weights(uint64_t(columns / 32) * rows * 34);
    std::vector<float> values(columns);
    const auto matrix = arena.bytes(weights, false), input = arena.floats(values, false);
    const auto output = arena.fill(rows), partials = arena.fill(rows * count);
    auto run = [&] { graph.begin(); graph.q8_matvec_split(matrix, input, output, partials, columns, rows, count); };
    auto recover = [&] {
        std::fill(weights.begin(), weights.end(), std::byte{0}); std::fill(values.begin(), values.end(), 0);
        graph.write(matrix, weights); graph.write(input, std::as_bytes(std::span(values)));
        const auto before = graph.memory(); run(); submit(graph, 2, before);
        for (auto view : {output, partials}) for (auto value : read_floats(graph, view)) require(value == 0, "Q8-split valid reuse failed after GPU refusal");
    };
    for (uint16_t bits : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e01), uint16_t(0xfe01)}) {
        scale(weights, weights.size() / 34 - 1, bits); graph.write(matrix, weights);
        const auto before = graph.memory(); run(); reject([&] { static_cast<void>(graph.submit()); });
        same_resources(before, graph.memory()); recover();
    }
    for (float value : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        values.back() = value; graph.write(input, std::as_bytes(std::span(values)));
        run(); reject([&] { static_cast<void>(graph.submit()); }); recover();
    }
    // First overflows within a partial; second has finite per-block partials
    // (about 2.66e38 each) but overflows the ascending final reduction.
    for (float value : {std::numeric_limits<float>::max(), 1e30f}) {
        std::fill(weights.begin(), weights.end(), std::byte{127});
        for (size_t block = 0; block < weights.size() / 34; ++block) scale(weights, block, 0x7bff);
        std::fill(values.begin(), values.end(), value);
        graph.write(matrix, weights); graph.write(input, std::as_bytes(std::span(values)));
        run(); reject([&] { static_cast<void>(graph.submit()); });
        if (value == 1e30f) for (auto partial : read_floats(graph, partials))
            require(std::isfinite(partial) && partial > 1e38f, "final-reduction overflow fixture had a nonfinite partial");
        recover();
    }
    arena.guards();
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph graph(argv[1], 64 * 1024 * 1024);
        const auto initial = graph.memory();
        { Arena arena{graph, {}};
          const auto w = arena.bytes(std::vector<std::byte>(34)), x = arena.fill(32), y = arena.fill(1, 9), p = arena.fill(1, 7);
          const auto before = graph.memory();
          graph.begin(); reject([&] { graph.q8_matvec_split(w, x, y, p, 32, 1, 1); }); graph.abort();
          require(read_floats(graph, y)[0] == 9 && read_floats(graph, p)[0] == 7, "unprepared Q8-split published data");
          same_resources(before, graph.memory()); arena.guards(); }
        graph.configure_small_gemv(1); graph.configure_block_gemv(3);
        const auto policy = graph.kernel_policy();
        const auto before = graph.memory();
        graph.prepare_q8_matvec_split(); graph.prepare_q8_matvec_split();
        same_resources(before, graph.memory()); require(graph.kernel_policy() == policy, "Q8-split preparation mutated policy");
        std::mt19937 random(0x51385350);
        for (auto shape : std::array<std::array<uint32_t, 2>, 8>{{{32,1},{64,3},{96,5},{10240,319},{10240,320},{10240,321},{96,1024},{65536,5}}}) {
            const auto columns = shape[0], rows = shape[1];
            const auto bytes = q8_split_bytes({columns, rows, 32});
            std::vector<std::byte> weights(bytes.matrix); std::vector<float> input(columns);
            for (auto& value : weights) value = std::byte(random() & 255);
            for (size_t block = 0; block < weights.size() / 34; ++block)
                scale(weights, block, uint16_t(0x1000 + (random() % 8) * 0x400 + ((random() & 1) ? 0x8000 : 0)));
            for (auto& value : input) value = float(int(random() % 2049) - 1024) / 4096;
            exercise(graph, columns, rows, weights, input);
        }
        const std::array<uint16_t, 14> finite{0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3800,0xb800,0x3c00,0xbc00,0x7bff,0xfbff};
        std::vector<std::byte> weights(finite.size() * 34, std::byte{127});
        for (size_t row = 0; row < finite.size(); ++row) { scale(weights, row, finite[row]); weights[row * 34 + 2] = std::byte{128}; }
        exercise(graph, 32, finite.size(), weights, std::vector<float>(32, 0.03125f));
        admission(graph); nonfinite(graph);
        require(graph.memory().live_bytes == initial.live_bytes && graph.memory().profile_counter_payload_bytes == 0 &&
            graph.memory().capture_resource_bytes == 0, "Q8-split fixtures leaked GPU resources");
        joint_maximum(argv[1]);
        std::cout << "Owned Q8-split Metal checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& error) { std::cerr << "Owned Q8-split Metal failed: " << error.what() << '\n'; return 1; }
}
