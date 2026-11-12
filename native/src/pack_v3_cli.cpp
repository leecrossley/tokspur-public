#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/json.hpp"

#include <charconv>
#include <csignal>
#include <iostream>
#include <set>

namespace tokspur::planar_v3 {
namespace {
volatile std::sig_atomic_t interrupted = 0;

void handle_interrupt(int) { interrupted = 1; }

void check_cancelled() {
    if (interrupted) throw FormatError("pack v3 build interrupted");
}

uint64_t byte_count(std::string_view text) {
    uint64_t value = 0;
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) throw FormatError("invalid pack v3 byte count");
    return value;
}

std::filesystem::path absolute_path(std::string_view text) {
    std::filesystem::path path(text);
    if (!path.is_absolute() || path != path.lexically_normal()) {
        throw FormatError("pack v3 paths must be absolute and normalized");
    }
    return path;
}

} // namespace

int pack_v3_cli(int argc, char** argv) {
    const bool build = std::string_view(argv[1]) == "pack-v3-build";
    std::filesystem::path output, mtp_path;
    std::vector<std::filesystem::path> targets;
    std::set<std::string_view> seen;
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes;
    uint64_t chunk_bytes = 16ULL * 1024 * 1024;
    constexpr uint32_t experts_per_chunk = 8;
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (++i >= argc) throw FormatError("missing value for pack v3 option");
        if (flag != "--gguf" && !seen.insert(flag).second) throw FormatError("duplicate pack v3 option");
        const std::string_view value(argv[i]);
        if (flag == "--output") output = absolute_path(value);
        else if (flag == "--mtp") mtp_path = absolute_path(value);
        else if (flag == "--gguf") targets.push_back(absolute_path(value));
        else if (flag == "--reserve-bytes") reserve_bytes = byte_count(value);
        else if (flag == "--chunk-bytes") chunk_bytes = byte_count(value);
        else throw FormatError("unknown pack v3 option");
    }
    if (output.empty() || mtp_path.empty() || targets.empty()) {
        throw FormatError("pack v3 requires --output, --mtp, and at least one --gguf");
    }
    validate_pack_v3_output_name(output);
    if (chunk_bytes < 32'768 || chunk_bytes > 67'108'864) throw FormatError("invalid chunk_bytes bound");
    // SIGHUP too: a closed terminal must unwind and remove the multi-GB temporary, not kill the build mid-write. An inherited ignore (nohup) stays in place.
    const auto keep_hangup = [] {
        const auto previous = std::signal(SIGHUP, handle_interrupt);
        return previous != SIG_ERR && (previous != SIG_IGN || std::signal(SIGHUP, SIG_IGN) != SIG_ERR);
    };
    if (build && (std::signal(SIGINT, handle_interrupt) == SIG_ERR || std::signal(SIGTERM, handle_interrupt) == SIG_ERR ||
                  !keep_hangup())) {
        throw FormatError("cannot install pack v3 interruption handler");
    }
    const auto source = std::make_shared<ModelFiles>(targets);
    const auto mtp = std::make_shared<ModelFiles>(std::vector<std::filesystem::path>{mtp_path});
    auto plan = plan_derived_pack_v3(*source, mtp.get());
    require_full_pack_v3_census(plan);
    const auto budget = measure_pack_v3_budget(plan, output.parent_path(), {}, experts_per_chunk, reserve_bytes, chunk_bytes);
    Json::Object receipt{
        {"recipe", Json(plan.recipe)}, {"version", Json::integer(plan.version)},
        {"tensor_count", Json::integer(plan.tensors.size())}, {"payload_bytes", Json::integer(plan.payload_bytes)},
        {"file_bytes", Json::integer(plan.file_bytes)}
    };
    if (build) {
        check_cancelled();
        if (!budget.ram_ceiling_satisfied) throw FormatError("pack v3 streaming buffer exceeds RAM ceiling");
        const auto result = build_or_verify_pack_v3(std::move(plan), source, mtp, output, experts_per_chunk,
                                                    reserve_bytes, chunk_bytes, check_cancelled);
        receipt.emplace_back("status", Json(result.existed ? "verified-existing" : "built"));
        receipt.emplace_back("manifest_sha256", Json(result.verified.manifest_sha256));
    } else {
        receipt.emplace_back("available_bytes", Json::integer(budget.filesystem_available_bytes));
        receipt.emplace_back("reserve_bytes", Json::integer(budget.required_reserve_bytes));
        receipt.emplace_back("disk_ok", Json(budget.reserve_satisfied));
        receipt.emplace_back("streaming_buffer_bytes", Json::integer(budget.streaming_ring_buffer_bytes));
        receipt.emplace_back("streaming_ceiling_bytes", Json::integer(budget.streaming_ram_ceiling_bytes));
        receipt.emplace_back("ram_ok", Json(budget.ram_ceiling_satisfied));
    }
    std::cout << Json(std::move(receipt)).dump() << '\n';
    return 0;
}
} // namespace tokspur::planar_v3
