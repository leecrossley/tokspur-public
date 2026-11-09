#include "tokspur/dflash2_checkpoint.hpp"
#include "tokspur/derived_pack.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace tokspur;

namespace {
unsigned checks = 0;
void check(bool value, const char* why) { ++checks; if (!value) throw std::runtime_error(why); }
uint16_t sample(uint32_t layer, uint32_t value, uint32_t head, uint32_t channel, uint32_t position) {
    return uint16_t(31 * position + 7919 * layer + 1237 * value + 71 * head + 13 * channel);
}

constexpr bool bounded_work =
#if defined(TOKSPUR_PRIVATE_DFLASH2_CACHE_WORK) && TOKSPUR_PRIVATE_DFLASH2_CACHE_WORK
    true;
#else
    false;
#endif

void intermediate_prefill_checkpoint(uint32_t total_rows, uint32_t chunk_rows, bool shared_prefix_cache, uint32_t packet_rows = 65536) {
    constexpr uint32_t window = 2048;
    std::array<std::vector<uint16_t>, 2> ring;
    for (auto& plane : ring) plane.assign(window, 0xdead);
    uint32_t skipped = 0, captures = 0;
    const auto write_rows = [&](auto& destination, uint32_t first, uint32_t end) {
        for (uint32_t value = 0; value < 2; ++value) for (uint32_t position = first; position < end; ++position)
            destination[value][position % window] = sample(0, value, 0, 0, position);
    };
    const auto check_live = [&](const auto& actual, uint32_t prefix) {
        for (uint32_t value = 0; value < 2; ++value) for (uint32_t position = prefix - std::min(prefix, window); position < prefix; ++position)
            check(actual[value][position % window] == sample(0, value, 0, 0, position), "prefill checkpoint retained stale or missing ring rows");
    };
    for (uint32_t base = 0; base < total_rows;) {
        const uint32_t packet_end = std::min(total_rows, (base / packet_rows + 1) * packet_rows);
        const uint32_t rows = std::min(chunk_rows, packet_end - base);
        const uint64_t remaining = packet_end - base - rows;
        const auto effective_remaining = dflash2_prefill_remaining_after(remaining, shared_prefix_cache, base + rows, 2048);
        const auto plan = dflash2_context_commit_plan(rows, effective_remaining, window);
        skipped += plan.keep_from;
        write_rows(ring, base + plan.keep_from, base + rows);
        base += rows; // The logical frontier advances even when the original optimisation skipped writes.
        if (base == packet_end) check_live(ring, base);
        if (!shared_prefix_cache || base > 2048 || base % 512) continue;
        ++captures;
        const DFlash2CheckpointLayout layout{base - std::min(base, window), std::min(base, window), window, 1, 1, 1};
        std::vector<std::byte> checkpoint(size_t(layout.bytes()));
        check(layout.visit(0, checkpoint.size(), [&](uint32_t, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
            std::memcpy(checkpoint.data() + request, std::as_bytes(std::span(ring[value])).data() + physical, size_t(bytes));
            return true;
        }), "intermediate prefill checkpoint capture failed");
        std::vector<uint16_t> expected;
        for (uint32_t value = 0; value < 2; ++value) for (uint32_t position = layout.first; position < base; ++position)
            expected.push_back(sample(0, value, 0, 0, position));
        const auto expected_bytes = std::as_bytes(std::span(expected));
        check(std::equal(checkpoint.begin(), checkpoint.end(), expected_bytes.begin(), expected_bytes.end()), "intermediate checkpoint differs from independently materialised prefix");
        std::array<std::vector<uint16_t>, 2> restored;
        for (auto& plane : restored) plane.assign(window, 0xbeef);
        check(layout.visit(0, checkpoint.size(), [&](uint32_t, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
            std::memcpy(std::as_writable_bytes(std::span(restored[value])).data() + physical, checkpoint.data() + request, size_t(bytes));
            return true;
        }), "intermediate prefill checkpoint restore failed");
        check_live(restored, base);
        write_rows(restored, base, base + 8);
        check_live(restored, base + 8);
    }
    check_live(ring, total_rows);
    std::vector<bool> needed(total_rows, shared_prefix_cache && !bounded_work);
    if (shared_prefix_cache) std::fill_n(needed.begin(), std::min(total_rows, 2048u), true);
    for (uint32_t begin = 0; begin < total_rows; begin += packet_rows) {
        const uint32_t end = std::min(total_rows, begin + packet_rows);
        std::fill(needed.begin() + end - std::min(end - begin, window), needed.begin() + end, true);
    }
    check(captures == (shared_prefix_cache ? 2048 / chunk_rows : 0), "intermediate anchor coverage changed");
    check(skipped == uint32_t(std::count(needed.begin(), needed.end(), false)), "prefill retained unnecessary rows or skipped required context");
    if (total_rows == 32768 && packet_rows == 8192 && shared_prefix_cache)
        check(total_rows - skipped == (bounded_work ? 10240u : 32768u), "32K prefill context projection count differs");
}

void checkpoint_identity() {
    check(dflash2_checkpoint_identity_required(true), "SSD checkpoint skipped its full weight identity");
    check(dflash2_checkpoint_identity_required(false) == !bounded_work, "uncached identity policy differs");
    unsigned calls = 0;
    const auto hash = [&](std::string_view data) { ++calls; return derived_pack::hash(data); };
    const std::string config = derived_pack::hash("generated config"), weights = "generated immutable weights";
    check(dflash2_checkpoint_identity(false, config, weights, hash).empty() && calls == 0, "disabled identity read or hashed payload");
    const auto expected = derived_pack::hash("dflash2-context-v1:" + config + ":" + derived_pack::hash(weights));
    const auto identity = dflash2_checkpoint_identity(true, config, weights, hash);
    check(calls == 2 && identity == expected, "persistent checkpoint identity changed format or omitted weight bytes");
    check(dflash2_checkpoint_identity(true, config, weights + "!", hash) != identity, "changed weights reused checkpoint identity");
    check(dflash2_checkpoint_identity(true, derived_pack::hash("changed config"), weights, hash) != identity, "changed configuration reused checkpoint identity");
}

void roundtrip(uint32_t prefix) {
    constexpr uint32_t window = 2048, heads = 2, width = 3, layers = 2;
    const uint32_t rows = std::min(prefix, window), first = prefix - rows;
    const DFlash2CheckpointLayout layout{first, rows, window, layers, heads, width};
    std::array<std::vector<uint16_t>, 2 * layers> original, restored;
    for (auto& plane : original) plane.assign(window * heads * width, 0xdead);
    for (auto& plane : restored) plane.assign(window * heads * width, 0xbeef);
    std::vector<uint16_t> expected;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t head = 0; head < heads; ++head) for (uint32_t position = first; position < prefix; ++position) for (uint32_t channel = 0; channel < width; ++channel) {
            const auto value = sample(layer, 0, head, channel, position);
            original[2 * layer][(head * window + position % window) * width + channel] = value;
            expected.push_back(value);
        }
        for (uint32_t head = 0; head < heads; ++head) for (uint32_t channel = 0; channel < width; ++channel) for (uint32_t position = first; position < prefix; ++position) {
            const auto value = sample(layer, 1, head, channel, position);
            original[2 * layer + 1][(head * width + channel) * window + position % window] = value;
            expected.push_back(value);
        }
    }
    std::vector<std::byte> payload(size_t(layout.bytes()));
    for (uint64_t offset = 0; offset < payload.size();) {
        const uint64_t count = std::min<uint64_t>(7919, payload.size() - offset);
        check(layout.visit(offset, count, [&](uint32_t layer, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
            const auto plane = std::as_bytes(std::span(original[2 * layer + value]));
            check(physical + bytes <= plane.size(), "capture escaped ring");
            std::memcpy(payload.data() + offset + request, plane.data() + physical, size_t(bytes));
            return true;
        }), "capture refused valid ring");
        offset += count;
    }
    const auto reference = std::as_bytes(std::span(expected));
    check(reference.size() == payload.size() && std::equal(reference.begin(), reference.end(), payload.begin()), "checkpoint differs from independent chronological oracle");
    for (uint64_t offset = 0; offset < payload.size();) {
        const uint64_t count = std::min<uint64_t>(4051, payload.size() - offset);
        check(layout.visit(offset, count, [&](uint32_t layer, bool value, uint64_t physical, uint64_t request, uint64_t bytes) {
            auto plane = std::as_writable_bytes(std::span(restored[2 * layer + value]));
            check(physical + bytes <= plane.size(), "restore escaped ring");
            std::memcpy(plane.data() + physical, payload.data() + offset + request, size_t(bytes));
            return true;
        }), "restore refused valid ring");
        offset += count;
    }
    for (uint32_t layer = 0; layer < layers; ++layer) for (uint32_t head = 0; head < heads; ++head) for (uint32_t slot = 0; slot < window; ++slot) for (uint32_t channel = 0; channel < width; ++channel) {
        const auto k = (head * window + slot) * width + channel, v = (head * width + channel) * window + slot;
        const bool live = rows == window || slot < rows;
        check(restored[2 * layer][k] == (live ? original[2 * layer][k] : 0xbeef), "key restoration changed live values or unused slots");
        check(restored[2 * layer + 1][v] == (live ? original[2 * layer + 1][v] : 0xbeef), "value restoration changed live values or unused slots");
    }
    unsigned copies = 0;
    const auto refuse = [&](auto...) { ++copies; return false; };
    check(!layout.visit(layout.bytes() + 1, 0, refuse) && !copies, "out-of-bounds empty span accepted");
    check(!layout.visit(layout.bytes(), 1, refuse) && !copies, "overflowing span reached storage");
    if (rows) check(!layout.visit(0, 1, refuse) && copies == 1, "storage failure was ignored");
}
}

int main() {
    try {
        checkpoint_identity();
        for (const auto& [total, chunk] : {std::pair{6144u, 2048u}, std::pair{4096u, 512u}, std::pair{2500u, 512u}, std::pair{2049u, 512u}, std::pair{2048u, 512u}}) {
            intermediate_prefill_checkpoint(total, chunk, false);
            intermediate_prefill_checkpoint(total, chunk, true);
        }
        intermediate_prefill_checkpoint(32768, 2048, false, 8192);
        intermediate_prefill_checkpoint(32768, 2048, true, 8192);
        check(!dflash2_context_commit_plan(8, 0, 2048).discard_pending, "short commit discarded preceding pending context");
        check(dflash2_context_commit_plan(2048, 0, 2048).discard_pending, "complete replacement retained obsolete pending context");
        for (uint32_t prefix : {0u, 1u, 512u, 2047u, 2048u, 2049u, 4095u, 4096u, 65536u}) roundtrip(prefix);
        for (const auto layout : {DFlash2CheckpointLayout{1, 1}, DFlash2CheckpointLayout{65536, 1}, DFlash2CheckpointLayout{0, 2049}, DFlash2CheckpointLayout{0, 0, 0}}) {
            check(!layout.valid(), "invalid checkpoint geometry accepted");
            check(!layout.visit(0, 0, [](auto...) { throw std::runtime_error("invalid layout touched storage"); return true; }), "invalid layout visited");
        }
        std::cout << "DFlash2 checkpoint layout checks=" << checks << "; generated ring geometry, no model qualification\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
