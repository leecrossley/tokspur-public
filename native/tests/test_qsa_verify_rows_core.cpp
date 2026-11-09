#include "tokspur/graph.hpp"
#include "tokspur/qsa_verify_rows.hpp"

// Owned fixture adaptation from 4f56424. Scalar chronological-channel APIs are
// the differential control; the independent double guard is unchanged.

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
using namespace tokspur;
constexpr QsaShape actual{2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536};
constexpr QsaShape small{32, 3, 1, 34, 2, 32, 16, 4, 16, 8192};
constexpr float sentinel = 321.25f;
size_t checks = 0, batches = 0;
double worst = 0;
void prepare_scalar(MetalGraph& g) {
    g.configure_small_gemv(16); g.configure_block_gemv(3); g.configure_hc_exact(HcExactMode::combined);
    g.configure_qsa_stage_v8(true); g.configure_qsa_channel_chronological(true);
}
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid QSA verifier rows operation admitted");
}
void same_memory(GraphMemory a, GraphMemory b) {
    require(a.live_bytes == b.live_bytes && a.peak_bytes == b.peak_bytes && a.allocations == b.allocations &&
        a.imported_backing_bytes == b.imported_backing_bytes && a.profile_counter_payload_bytes == b.profile_counter_payload_bytes &&
        a.capture_resource_bytes == b.capture_resource_bytes && a.live_captures == b.live_captures &&
        a.capture_binding_aliases == b.capture_binding_aliases &&
        a.capture_binding_alias_resource_bytes == b.capture_binding_alias_resource_bytes,
        "QSA verifier rows changed the prepared allocation ledger");
}
template<class T> std::vector<T> read(MetalGraph& graph, GpuView view) {
    const auto bytes = graph.read(view); std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size()); return out;
}
bool same_bits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }
struct Arena {
    MetalGraph& g; std::vector<GpuView> full;
    GpuView bytes(std::span<const std::byte> data, bool immutable = false) {
        std::vector<std::byte> padded(data.size() + 8, std::byte{0x69});
        std::copy(data.begin(), data.end(), padded.begin() + 4);
        const auto allocation = g.upload(padded, "generated guarded QSA verifier rows fixture", immutable);
        full.push_back(allocation); return g.slice(allocation, 4, data.size());
    }
    template<class T> GpuView array(std::span<const T> data, bool immutable = false) {
        return bytes(std::as_bytes(data), immutable);
    }
    GpuView floats(size_t count, float value = 0) { return array<float>(std::vector<float>(count, value)); }
    void guards() {
        for (const auto view : full) for (const auto at : {uint64_t(0), view.bytes - 4})
            require(g.read(g.slice(view, at, 4)) == std::vector<std::byte>(4, std::byte{0x69}), "QSA verifier rows canary changed");
    }
    ~Arena() { try { g.abort(); } catch (const FormatError&) {} for (const auto view : full) g.release(view); }
};
struct Fixture {
    MetalGraph& g; Arena arena; QsaShape shape;
    uint32_t base, rows, cache_rows, vector_width, id_width;
    std::vector<float> queries, initial_sum, index_keys;
    std::vector<uint16_t> keys, values;
    std::vector<uint32_t> ids;
    std::vector<std::vector<uint32_t>> selected;
    GpuView q, k, v, id, output, ordinary, serial, scores, serial_scores, factors, factor_storage;
    GpuView sum, raw_index_keys, sums, pooled;
    std::vector<std::byte> fixed_sums, fixed_pooled;
    Fixture(MetalGraph& graph, QsaShape s, uint32_t prefix, uint32_t count, uint32_t pattern = 0)
        : g(graph), arena{graph, {}}, shape(s), base(prefix), rows(count),
          cache_rows(std::min(s.capacity, prefix + count + 4)), vector_width(s.query_heads * s.head_width), id_width(qsa_block_budget(s)) {
        static_cast<void>(qsa_verify_rows_plan(base, rows, shape));
        queries.resize(8ull * vector_width, std::numeric_limits<float>::quiet_NaN());
        keys.resize(uint64_t(cache_rows) * s.kv_heads * s.head_width, uint16_t(0x7fc1)); values = keys;
        ids.resize(8ull * id_width, 0xffffffffu); selected.resize(rows);
        for (uint32_t row = 0; row < rows; ++row) for (uint32_t d = 0; d < vector_width; ++d) {
            float x = float(int((d * 19 + row * 23 + 7) % 101) - 50) / 512;
            if (pattern == 1) x = 0;
            if (pattern == 2 || pattern == 3) x = 1;
            if (pattern == 4) x = (d / 32) % 2 ? -1.0f : 1.0f;
            if (pattern == 5) x = d % 2 ? -0.0f : std::numeric_limits<float>::denorm_min();
            queries[uint64_t(row) * vector_width + d] = x;
        }
        const uint32_t kv_width = s.kv_heads * s.head_width;
        for (uint32_t token = 0; token < base + rows; ++token) for (uint32_t d = 0; d < kv_width; ++d) {
            const auto at = uint64_t(token) * kv_width + d;
            float key = float(int((at * 13 + 11) % 127) - 63) / 256;
            if (pattern == 1) key = d % 2 ? -0.0f : 0.0f;
            if (pattern == 2) key = float(token) / 4096;
            if (pattern == 3) key = -float(token) / 4096;
            if (pattern == 4) key = token % 2 ? 128.0f : -128.0f;
            keys[at] = finite_bf16(key);
            values[at] = finite_bf16(at % 37 ? float(int((at * 23 + 3) % 257) - 128) / 512 : -0.0f);
        }
        q = arena.array<float>(queries); k = arena.array<uint16_t>(keys); v = arena.array<uint16_t>(values);
        id = arena.array<uint32_t>(ids); output = arena.floats(8ull * vector_width, sentinel);
        ordinary = arena.floats(8ull * vector_width, sentinel); serial = arena.floats(8ull * vector_width, sentinel);
        // Both comparison schedules use this SAME fixed scratch ledger. Serial
        // scores use packed per-head rows; verifier-row uses fixed2051-item head rows.
        scores = arena.floats(maximum_qsa_verify_rows_scores / 4, sentinel);
        serial_scores = arena.floats(maximum_qsa_verify_rows_scores / 4, sentinel);
        // Arena adds a four-byte outer canary. The extra inner float on each
        // side puts this float2 capability at offset8, with adjacent canaries.
        factor_storage = arena.floats(maximum_qsa_verify_rows_factors / 4 + 2, sentinel);
        factors = g.slice(factor_storage, 4, maximum_qsa_verify_rows_factors);
        prepare_index_tape(); prepare_selection();
    }
    GpuView row(GpuView view, uint32_t at, uint64_t bytes) { return g.slice(view, uint64_t(at) * bytes, bytes); }
    void submit(uint64_t count, GraphMemory memory) {
        const auto reply = g.submit(); ++batches;
        require(reply.dispatches == count && !reply.capture_executes, "QSA verifier rows physical dispatch count differs");
        same_memory(memory, g.memory());
    }
    void prepare_index_tape() {
        // Reuse the existing ordered compression primitive; this component does
        // NOT batch or replace it. Each candidate row retains its own sum tape.
        initial_sum.resize(shape.index_width); index_keys.resize(8ull * shape.index_width, std::numeric_limits<float>::quiet_NaN());
        for (size_t d = 0; d < initial_sum.size(); ++d) initial_sum[d] = float(base % 4) * float(int(d % 17) - 8) / 128;
        for (uint32_t r = 0; r < rows; ++r) for (uint32_t d = 0; d < shape.index_width; ++d)
            index_keys[uint64_t(r) * shape.index_width + d] = float(int((r * 13 + d * 7) % 31) - 15) / 128;
        sum = arena.array<float>(initial_sum); raw_index_keys = arena.array<float>(index_keys);
        sums = arena.floats(8ull * shape.index_width, sentinel); pooled = arena.floats(8ull * shape.index_width, sentinel);
        const auto fixed = g.memory(); g.begin();
        for (uint32_t r = 0; r < rows; ++r) g.index_accumulate(r ? row(sums, r - 1, shape.index_width * 4ull) : sum,
            row(raw_index_keys, r, shape.index_width * 4ull), row(sums, r, shape.index_width * 4ull),
            row(pooled, r, shape.index_width * 4ull), base + r, 4);
        submit(rows, fixed);
        const auto actual_sums = read<float>(g, sums), actual_pool = read<float>(g, pooled);
        auto expected = initial_sum;
        for (uint32_t r = 0; r < 8; ++r) for (uint32_t d = 0; d < shape.index_width; ++d) {
            const auto at = uint64_t(r) * shape.index_width + d;
            float next = sentinel, pool = sentinel;
            if (r < rows) {
                const float value = expected[d] + index_keys[at];
                const bool complete = (base + r + 1) % 4 == 0;
                next = complete ? 0 : value; pool = complete ? value / 4 : 0; expected[d] = next;
            }
            require(same_bits(actual_sums[at], next) && same_bits(actual_pool[at], pool), "row compression intermediate/sum tape differs");
        }
        fixed_sums = g.read(sums); fixed_pooled = g.read(pooled);
    }
    void prepare_selection() {
        // Existing canonical selection from independent generated index scores,
        // including exact ties. Actual index-query projection is out of scope.
        for (uint32_t r = 0; r < rows; ++r) {
            const auto blocks = (base + r + 1) / 4;
            std::vector<double> reference_scores(blocks);
            for (uint32_t b = 0; b < blocks; ++b) reference_scores[b] = double((b * 97 + r * 31) % 29);
            selected[r] = select_blocks_reference(reference_scores, id_width);
            if (blocks <= id_width) continue; // The kernels derive chronological IDs; poisoned unused ID bytes must be ignored.
            const std::vector<float> input(reference_scores.begin(), reference_scores.end());
            const auto input_view = arena.array<float>(input);
            const auto scratch_count = ((blocks + 511) / 512) * id_width;
            const SelectionScratch work{arena.floats(scratch_count), arena.floats(scratch_count),
                arena.floats(scratch_count), arena.floats(scratch_count)};
            const auto fixed = g.memory(); g.begin();
            g.select_blocks(input_view, row(id, r, id_width * 4ull), work, blocks, id_width);
            uint64_t dispatches = 2; for (uint32_t span = 512; span < blocks; span *= 2) ++dispatches;
            submit(dispatches, fixed);
            const auto chosen = read<uint32_t>(g, row(id, r, id_width * 4ull));
            require(chosen == selected[r], "row-specific canonical selected IDs differ");
            std::copy(chosen.begin(), chosen.end(), ids.begin() + uint64_t(r) * id_width);
        }
    }
    void core(GpuView query = {}, GpuView key = {}, GpuView value = {}, GpuView indices = {}, GpuView out = {}, GpuView scratch = {},
              uint32_t count = 0) {
        g.sparse_attention_rows_chronological(query.owner ? query : q, key.owner ? key : k, value.owner ? value : v,
            indices.owner ? indices : id, out.owner ? out : output, scratch.owner ? scratch : scores, factors,
            base, count ? count : rows, shape);
    }
    void references() {
        const auto fixed = g.memory(); g.begin();
        for (uint32_t r = 0; r < rows; ++r) g.sparse_attention(row(q, r, vector_width * 4ull), k, v,
            row(id, r, id_width * 4ull), row(ordinary, r, vector_width * 4ull), base + r + 1, shape);
        submit(3ull * rows, fixed); g.begin();
        for (uint32_t r = 0; r < rows; ++r) g.sparse_attention_stage_v8(row(q, r, vector_width * 4ull), k, v,
            row(id, r, id_width * 4ull), row(serial, r, vector_width * 4ull),
            g.slice(serial_scores, r * qsa_verify_rows_row_scores * 4, qsa_stage_v8_bytes(base + r + 1, shape).scores), base + r + 1, shape);
        submit(3ull * rows, fixed);
        require(g.read(ordinary) == g.read(serial), "frozen serial staged oracle differs from ordinary attention");
    }
    void check_sources() {
        const auto factor_guards = read<float>(g, factor_storage);
        require(same_bits(factor_guards.front(), sentinel) && same_bits(factor_guards.back(), sentinel),
            "aligned factor capability escaped its immediate canaries");
        require(read<float>(g, q).size() == queries.size() && g.read(q) == std::vector<std::byte>(std::as_bytes(std::span(queries)).begin(),
            std::as_bytes(std::span(queries)).end()), "query row input changed");
        require(read<uint16_t>(g, k) == keys && read<uint16_t>(g, v) == values && read<uint32_t>(g, id) == ids,
            "QSA verifier rows changed read-only KV/selected IDs");
        const auto raw = std::as_bytes(std::span(index_keys));
        require(g.read(sums) == fixed_sums && g.read(pooled) == fixed_pooled && read<float>(g, sum) == initial_sum &&
            g.read(raw_index_keys) == std::vector<std::byte>(raw.begin(), raw.end()),
            "QSA verifier rows changed prepared chronological sum/pooled state");
        arena.guards();
    }
    void check_factors(uint32_t active_rows) {
        const auto values = read<float>(g, factors); const auto plan = qsa_verify_rows_plan(base, active_rows, shape);
        for (uint32_t r = 0; r < 8; ++r) for (uint32_t h = 0; h < 24; ++h) for (uint32_t item = 0; item < qsa_verify_rows_stride; ++item) {
            const auto at = (r * qsa_verify_rows_row_scores + uint64_t(h) * qsa_verify_rows_stride + item) * 2;
            const bool active = r < active_rows && h < shape.query_heads && item < plan.attended[r];
            if (active) require(std::isfinite(values[at]) && std::isfinite(values[at + 1]) &&
                values[at] >= 0 && values[at] <= 1 && values[at + 1] >= 0 && values[at + 1] <= 1,
                "active chronological factors must be finite exp scales");
            else require(same_bits(values[at], sentinel) && same_bits(values[at + 1], sentinel),
                "row factors wrote an inactive row/head/item");
        }
    }
    void check_outputs() {
        check_factors(rows);
        require(g.read(output) == g.read(ordinary), "QSA verifier rows output bytes differ from row-wise ordinary/staged attention");
        const auto actual_scores = read<float>(g, scores), expected_scores = read<float>(g, serial_scores);
        const auto plan = qsa_verify_rows_plan(base, rows, shape); const auto results = read<float>(g, output);
        for (uint32_t r = 0; r < 8; ++r) for (uint32_t h = 0; h < 24; ++h) for (uint32_t item = 0; item < qsa_verify_rows_stride; ++item) {
            const auto at = r * qsa_verify_rows_row_scores + uint64_t(h) * qsa_verify_rows_stride + item;
            const bool active = r < rows && h < shape.query_heads && item < plan.attended[r];
            const float expected = active ? expected_scores[r * qsa_verify_rows_row_scores + uint64_t(h) * plan.attended[r] + item] : sentinel;
            require(same_bits(actual_scores[at], expected), "QSA verifier rows row score or inactive scratch canary differs");
        }
        for (uint32_t r = 0; r < rows; ++r) {
            const auto start = queries.begin() + uint64_t(r) * vector_width;
            const auto reference = sparse_attention_reference(std::vector<double>(start, start + vector_width), keys, values,
                selected[r], base + r + 1, shape);
            for (uint32_t d = 0; d < vector_width; ++d) {
                const auto value = results[uint64_t(r) * vector_width + d];
                const auto error = std::abs(double(value) - reference[d]) / std::max(1.0, std::abs(reference[d]));
                worst = std::max(worst, error);
                require(std::isfinite(value) && std::isfinite(reference[d]) && error <= 5e-5, "QSA verifier rows independent double guard exceeded");
            }
        }
        for (size_t i = uint64_t(rows) * vector_width; i < results.size(); ++i)
            require(same_bits(results[i], sentinel), "QSA verifier rows wrote an inactive output row");
        check_sources();
    }
    void run() {
        const auto fixed = g.memory(); const auto before_routes = g.qsa_verify_rows_routes();
        const auto legacy_stage = g.qsa_stage_v8_calls(), legacy_channel = g.qsa_channel_calls();
        g.begin(); core(); submit(3, fixed);
        require(g.qsa_stage_v8_calls() == legacy_stage && g.qsa_channel_calls() == legacy_channel,
            "row attention changed legacy scalar-call or physical-pass metadata");
        auto expected = before_routes; const QsaVerifyRowsRoutes delta{1, rows, 1, 1, 1};
        for (size_t i = 0; i < expected.size(); ++i) expected[i] += delta[i];
        require(g.qsa_verify_rows_routes() == expected, "logical rows were confused with physical score/value passes");
        check_outputs();
    }
};
void exercise(MetalGraph& graph, QsaShape shape, uint32_t base, uint32_t rows, uint32_t pattern = 0) {
    Fixture f(graph, shape, base, rows, pattern); f.references(); f.run();
    const auto old_output = graph.read(f.output), old_scores = graph.read(f.scores), old_factors = graph.read(f.factors); const auto before = graph.qsa_verify_rows_routes();
    const auto fixed = graph.memory(); graph.begin(); f.core(); graph.abort();
    require(graph.qsa_verify_rows_routes() == before && graph.read(f.output) == old_output && graph.read(f.scores) == old_scores && graph.read(f.factors) == old_factors,
        "discarded unsubmitted tile published counters/writes"); same_memory(fixed, graph.memory());
    f.run(); // Same live cache/tapes reusable after abort; no implicit publication.
}
void partial_frontiers(MetalGraph& graph) {
    Fixture f(graph, actual, 2047, 8); f.references();
    const auto baseline = graph.read(f.ordinary); const auto packed_scores = read<float>(graph, f.serial_scores);
    const std::vector<float> empty_output(8ull * f.vector_width, sentinel), empty_scores(maximum_qsa_verify_rows_scores / 4, sentinel);
    for (uint32_t keep = 1; keep <= 8; ++keep) {
        auto query = f.queries; auto keys = f.keys, values = f.values; auto ids = f.ids;
        // Same prepared allocations/tapes; future data may already exist, but
        // must remain invisible to EVERY admitted prefix of this component.
        std::fill(query.begin() + uint64_t(keep) * f.vector_width, query.end(), std::numeric_limits<float>::quiet_NaN());
        const auto future = uint64_t(f.base + keep) * f.shape.kv_heads * f.shape.head_width;
        std::fill(keys.begin() + future, keys.end(), uint16_t(0x7fc1)); std::fill(values.begin() + future, values.end(), uint16_t(0x7f80));
        std::fill(ids.begin() + uint64_t(keep) * f.id_width, ids.end(), 0xffffffffu);
        graph.write(f.q, std::as_bytes(std::span(query))); graph.write(f.k, std::as_bytes(std::span(keys)));
        graph.write(f.v, std::as_bytes(std::span(values))); graph.write(f.id, std::as_bytes(std::span(ids)));
        graph.write(f.output, std::as_bytes(std::span(empty_output))); graph.write(f.scores, std::as_bytes(std::span(empty_scores)));
        const std::vector<float> empty_factors(maximum_qsa_verify_rows_factors / 4, sentinel);
        graph.write(f.factors, std::as_bytes(std::span(empty_factors)));
        const auto fixed = graph.memory(); const auto routes = graph.qsa_verify_rows_routes(); graph.begin(); f.core({}, {}, {}, {}, {}, {}, keep); f.submit(3, fixed);
        auto expected_output = std::vector<std::byte>(std::as_bytes(std::span(empty_output)).begin(), std::as_bytes(std::span(empty_output)).end());
        std::copy(baseline.begin(), baseline.begin() + uint64_t(keep) * f.vector_width * 4, expected_output.begin());
        require(graph.read(f.output) == expected_output, "partial frontier consumed future KV/query/ID rows");
        f.check_factors(keep);
        const auto actual_scores = read<float>(graph, f.scores); const auto plan = qsa_verify_rows_plan(f.base, keep, f.shape);
        for (uint32_t r = 0; r < 8; ++r) for (uint32_t h = 0; h < 24; ++h) for (uint32_t t = 0; t < qsa_verify_rows_stride; ++t) {
            const bool active = r < keep && t < plan.attended[r];
            const auto expected = active ? packed_scores[r * qsa_verify_rows_row_scores + uint64_t(h) * plan.attended[r] + t] : sentinel;
            require(same_bits(actual_scores[r * qsa_verify_rows_row_scores + h * qsa_verify_rows_stride + t], expected),
                "partial frontier score/unused-row canary differs");
        }
        auto expected_routes = routes; const QsaVerifyRowsRoutes delta{1, keep, 1, 1, 1};
        for (size_t i = 0; i < delta.size(); ++i) expected_routes[i] += delta[i];
        require(graph.qsa_verify_rows_routes() == expected_routes && graph.read(f.sums) == f.fixed_sums && graph.read(f.pooled) == f.fixed_pooled,
            "partial frontier changed state tape or physical/logical accounting"); f.arena.guards();
        // Simulate caller refusal/rollback by restoring bytes before reusing
        // the full component. No state is committed by this attention-only API.
        graph.write(f.q, std::as_bytes(std::span(f.queries))); graph.write(f.k, std::as_bytes(std::span(f.keys)));
        graph.write(f.v, std::as_bytes(std::span(f.values))); graph.write(f.id, std::as_bytes(std::span(f.ids)));
    }
    f.run();
    const auto before = graph.qsa_verify_rows_routes(); const auto memory = graph.memory();
    graph.begin(); f.core(); f.core(); f.submit(6, memory);
    auto expected = before; const QsaVerifyRowsRoutes delta{2, 16, 2, 2, 2};
    for (size_t i = 0; i < delta.size(); ++i) expected[i] += delta[i];
    require(graph.qsa_verify_rows_routes() == expected, "two component calls collapsed physical-pass/row counters"); f.check_outputs();
}
void host_refusals(const char* library) {
    MetalGraph graph(library, 64ull << 20); prepare_scalar(graph); graph.prepare_qsa_verify_rows();
    const auto prepared = graph.memory(); graph.prepare_qsa_verify_rows(); same_memory(prepared, graph.memory());
    Fixture f(graph, small, 63, 8); f.references();
    const auto rejected = [&](const std::function<void()>& action) {
        const auto old_output = graph.read(f.output), old_scores = graph.read(f.scores); const auto routes = graph.qsa_verify_rows_routes();
        const auto fixed = graph.memory(); graph.begin(); reject(action); graph.abort();
        require(graph.read(f.output) == old_output && graph.read(f.scores) == old_scores && graph.qsa_verify_rows_routes() == routes,
            "host refusal changed component outputs/counters"); same_memory(fixed, graph.memory()); f.check_sources();
    };
    for (uint32_t n : {0u, 9u, UINT32_MAX}) rejected([&] { graph.sparse_attention_rows_chronological(f.q, f.k, f.v, f.id, f.output, f.scores, f.factors, f.base, n, small); });
    for (uint32_t base : {8192u, UINT32_MAX}) rejected([&] { graph.sparse_attention_rows_chronological(f.q, f.k, f.v, f.id, f.output, f.scores, f.factors, base, 8, small); });
    rejected([&] { graph.sparse_attention_rows_chronological(f.q, f.k, f.v, f.id, f.output, f.scores, f.factors, 8190, 3, small); });
    rejected([&] { f.core(graph.slice(f.q, 0, f.q.bytes - 4)); });
    rejected([&] { f.core({}, graph.slice(f.k, 0, f.k.bytes - 4)); });
    rejected([&] { f.core({}, {}, {}, graph.slice(f.id, 0, f.id.bytes - 4)); });
    rejected([&] { f.core({}, {}, {}, {}, graph.slice(f.output, 0, f.output.bytes - 4)); });
    rejected([&] { f.core({}, {}, {}, {}, {}, graph.slice(f.scores, 0, f.scores.bytes - 4)); });
    rejected([&] { graph.sparse_attention_rows_chronological(f.q, f.k, f.v, f.id, f.output, f.scores,
        graph.slice(f.factors, 0, f.factors.bytes - 4), f.base, f.rows, small); });
    const auto offset4_factors = graph.slice(f.factor_storage, 0, f.factors.bytes);
    require(offset4_factors.offset % 8 == 4, "negative factor fixture must be offset4");
    rejected([&] { graph.sparse_attention_rows_chronological(f.q, f.k, f.v, f.id, f.output, f.scores,
        offset4_factors, f.base, f.rows, small); });
    rejected([&] { graph.sparse_attention_rows_chronological(f.q, f.k, f.v, f.id, f.output,
        graph.slice(f.factors, 0, f.scores.bytes), f.factors, f.base, f.rows, small); });
    rejected([&] { f.core({}, {}, {}, {}, f.q); });
    rejected([&] { f.core({}, {}, {}, {}, graph.slice(f.scores, 0, f.output.bytes)); });
    const auto immutable = f.arena.array<float>(std::vector<float>(8ull * f.vector_width), true);
    rejected([&] { f.core({}, {}, {}, {}, immutable); });
    const auto immutable_scores = f.arena.array<float>(std::vector<float>(maximum_qsa_verify_rows_scores / 4), true);
    rejected([&] { f.core({}, {}, {}, {}, {}, immutable_scores); });
    const auto released = graph.zeros(f.q.bytes, "released verifier-row query"); graph.release(released);
    rejected([&] { f.core(released); });
    MetalGraph foreign(library, 1ull << 20); const auto other = foreign.zeros(f.q.bytes, "foreign verifier-row query");
    rejected([&] { f.core(other); }); foreign.release(other);
    auto unaligned = f.q; unaligned.offset += 2; rejected([&] { f.core(unaligned); });
    reject([&] { f.core(); }); reject([&] { graph.prepare_qsa_verify_rows(); });
    graph.begin(); reject([&] { graph.prepare_qsa_verify_rows(); }); graph.abort();
    // A foreign-thread submit/abort must not clear the owner's unpublished rows.
    const auto before = graph.qsa_verify_rows_routes(); const auto fixed = graph.memory(); graph.begin(); f.core();
    uint32_t refusals = 0;
    std::thread wrong([&] {
        try { graph.abort(); } catch (const FormatError&) { ++refusals; }
        try { static_cast<void>(graph.submit()); } catch (const FormatError&) { ++refusals; }
    }); wrong.join(); require(refusals == 2, "foreign thread consumed the owner component draft");
    f.submit(3, fixed); auto expected = before; const QsaVerifyRowsRoutes delta{1, 8, 1, 1, 1};
    for (size_t i = 0; i < expected.size(); ++i) expected[i] += delta[i];
    require(graph.qsa_verify_rows_routes() == expected, "foreign refusal cleared successful owner counters"); f.check_outputs();
    // Component has no capture/replay or profiler contract in this first slice.
    reject([&] { graph.begin_indirect_compatible(); });
    reject([&] { graph.begin_capture(3); });
    reject([&] { static_cast<void>(graph.configure_profiling(3)); });
    f.run();
}
void device_refusals(MetalGraph& graph) {
    Fixture f(graph, actual, 2051, 4); f.references();
    const auto failed = [&](const std::function<void()>& poison) {
        poison(); const auto routes = graph.qsa_verify_rows_routes(); const auto fixed = graph.memory();
        graph.begin(); f.core(); reject([&] { static_cast<void>(graph.submit()); }); ++batches;
        require(graph.qsa_verify_rows_routes() == routes, "failed GPU component published physical/logical counters"); same_memory(fixed, graph.memory());
        graph.write(f.q, std::as_bytes(std::span(f.queries))); graph.write(f.k, std::as_bytes(std::span(f.keys)));
        graph.write(f.v, std::as_bytes(std::span(f.values))); graph.write(f.id, std::as_bytes(std::span(f.ids)));
        // Scratch/output are speculative after a failed submit. Clear caller's
        // inactive canaries, then rerun; accepted inputs/state are never changed.
        const std::vector<float> clean_scores(maximum_qsa_verify_rows_scores / 4, sentinel), clean_output(8ull * f.vector_width, sentinel);
        graph.write(f.scores, std::as_bytes(std::span(clean_scores))); graph.write(f.output, std::as_bytes(std::span(clean_output)));
        const std::vector<float> clean_factors(maximum_qsa_verify_rows_factors / 4, sentinel);
        graph.write(f.factors, std::as_bytes(std::span(clean_factors)));
        f.run();
    };
    for (uint32_t r = 0; r < f.rows; ++r) {
        failed([&] { auto q = f.queries; q[uint64_t(r) * f.vector_width] = std::numeric_limits<float>::quiet_NaN(); graph.write(f.q, std::as_bytes(std::span(q))); });
        failed([&] { auto ids = f.ids; ids[uint64_t(r) * f.id_width] = (f.base + r + 1) / 4; graph.write(f.id, std::as_bytes(std::span(ids))); });
        failed([&] { auto ids = f.ids; ids[uint64_t(r) * f.id_width + 1] = ids[uint64_t(r) * f.id_width]; graph.write(f.id, std::as_bytes(std::span(ids))); });
    }
    const auto first_token = uint64_t(f.selected[0][0]) * 4 * f.shape.kv_heads * f.shape.head_width;
    failed([&] { auto v = f.values; v[first_token] = 0x7fc1; graph.write(f.v, std::as_bytes(std::span(v))); });
    failed([&] { auto k = f.keys; k[first_token] = 0x7f80; graph.write(f.k, std::as_bytes(std::span(k))); });
    // First-pass active scores must overwrite poison; inactive score slots are
    // deliberately not touched (and are checked against their own old bytes).
    auto poison = read<float>(graph, f.scores); const auto plan = qsa_verify_rows_plan(f.base, f.rows, f.shape);
    for (uint32_t r = 0; r < f.rows; ++r) for (uint32_t h = 0; h < f.shape.query_heads; ++h)
        for (uint32_t t = 0; t < plan.attended[r]; ++t) poison[r * qsa_verify_rows_row_scores + h * qsa_verify_rows_stride + t] = std::numeric_limits<float>::quiet_NaN();
    graph.write(f.scores, std::as_bytes(std::span(poison))); f.run();
}
void preparation_refusals(const char* full, const char* partial) {
    MetalGraph missing(partial, 32ull << 20); prepare_scalar(missing); const auto old = missing.memory();
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        bool refused = false; try { missing.prepare_qsa_verify_rows(); } catch (const std::exception&) { refused = true; }
        require(refused, "missing verifier-row second PSO published preparation"); same_memory(old, missing.memory());
    }
    Fixture f(missing, small, 0, 1); const auto before = missing.memory();
    const auto old_output = missing.read(f.output), old_scores = missing.read(f.scores);
    missing.begin(); reject([&] { f.core(); }); missing.abort(); same_memory(before, missing.memory());
    require(missing.qsa_verify_rows_routes() == QsaVerifyRowsRoutes{}, "partial preparation published route counters");
    require(missing.read(f.output) == old_output && missing.read(f.scores) == old_scores, "partial PSO preparation executed writes");
    f.check_sources();
    MetalGraph late(full, 2ull << 20); prepare_scalar(late); late.begin(); late.abort(); reject([&] { late.prepare_qsa_verify_rows(); });
    MetalGraph noalloc(full, 2ull << 20); prepare_scalar(noalloc); const auto tiny = noalloc.memory();
    noalloc.prepare_qsa_verify_rows(); same_memory(tiny, noalloc.memory());
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2 || argc == 3, "expected full library and optional missing-row-PSO library");
        require(maximum_qsa_verify_rows_scores == 1'575'168 && maximum_qsa_verify_rows_factors == 3'150'336,
            "fixed verifier-row score/factor bounds changed");
        for (uint32_t n = 1; n <= 8; ++n) {
            const auto plan = qsa_verify_rows_plan(2047, n, actual);
            require(plan.base_prefix == 2047 && plan.rows == n && plan.vector_row_bytes == 24576 && plan.id_row_bytes == 2048,
                "bounded verifier-row host plan changed");
            for (uint32_t r = 0; r < 8; ++r) require(plan.attended[r] == (r < n ? qsa_stage_v8_bytes(2048 + r, actual).attended_tokens : 0),
                "row-specific host attended plan differs");
        }
        for (uint32_t bad : {0u, 9u, UINT32_MAX}) reject([&] { static_cast<void>(qsa_verify_rows_plan(0, bad, actual)); });
        reject([&] { static_cast<void>(qsa_verify_rows_plan(UINT32_MAX, 8, actual)); });
        for (const auto field : {&QsaShape::channels, &QsaShape::query_heads, &QsaShape::kv_heads, &QsaShape::head_width, &QsaShape::index_heads,
            &QsaShape::index_width, &QsaShape::rotary_width, &QsaShape::compression, &QsaShape::token_budget, &QsaShape::capacity}) {
            auto bad = actual; ++(bad.*field); reject([&] { static_cast<void>(qsa_verify_rows_plan(0, 8, bad)); });
        }
        MetalGraph graph(argv[1], 512ull << 20); prepare_scalar(graph); const auto initial = graph.memory();
        graph.prepare_qsa_verify_rows(); same_memory(initial, graph.memory());
        for (uint32_t n = 1; n <= 8; ++n) {
            exercise(graph, small, 0, n); exercise(graph, small, 62, n);
            // N1..8 partial rows cross prefix2047..2054 and the exact512-block
            // threshold; the next full tile adds prefix2055 and later tails.
            exercise(graph, actual, 2046, n);
        }
        exercise(graph, actual, 2047, 8); exercise(graph, actual, 2051, 8);
        exercise(graph, actual, 2054, 8);
        exercise(graph, actual, 4999, 3); exercise(graph, actual, 8191, 7);
        for (uint32_t pattern = 1; pattern <= 5; ++pattern) exercise(graph, actual, 2047, 8, pattern);
        exercise(graph, actual, 65528, 8);
        partial_frontiers(graph); device_refusals(graph); host_refusals(argv[1]);
        if (argc == 3) preparation_refusals(argv[1], argv[2]);
        require(graph.memory().live_bytes == initial.live_bytes, "QSA verifier rows component leaked a caller allocation");
        std::cout << "Owned QSA verifier rows component checks passed: " << checks << ", batches: " << batches
            << ", worst double scaled error: " << worst << ", fixed logical scores: " << maximum_qsa_verify_rows_scores << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "Owned QSA verifier rows component check failed: " << error.what() << '\n'; return 1; }
}
