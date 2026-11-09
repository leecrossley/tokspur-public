#pragma once
// Private diagnostic input only; not part of the target request protocol.
#include "tokspur/json.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace tokspur::verification_cost {
struct Arm { uint32_t rows = 0, accepted = 0; std::vector<uint32_t> proposals; };
struct Prefix {
    std::vector<uint32_t> input, continuation;
    std::vector<std::string> logit_hashes;
    std::vector<Arm> order;
};
struct Manifest { uint64_t seed = 0; std::string sealed_source_sha256; std::vector<Prefix> prefixes; uint64_t bytes = 0; };
inline void require(bool ok, const char* message) { if (!ok) throw FormatError(message); }
inline const Json& field(const Json& value, std::string_view name) {
    const auto* out = value.find(name); require(out, "missing verification cost manifest field"); return *out;
}
inline void keys(const Json& value, std::initializer_list<std::string_view> allowed) {
    std::set<std::string_view> seen;
    for (const auto& [name, unused] : value.object()) {
        static_cast<void>(unused);
        require(std::find(allowed.begin(), allowed.end(), name) != allowed.end() && seen.insert(name).second,
            "unknown or duplicate verification cost manifest field");
    }
    require(seen.size() == allowed.size(), "missing verification cost manifest field");
}
inline bool hash(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
inline std::vector<uint32_t> ids(const Json& value, size_t count) {
    require(value.array().size() == count, "verification cost token count differs");
    std::vector<uint32_t> out; out.reserve(count);
    for (const auto& id : value.array()) {
        const auto n = id.unsigned_integer(); require(n < 248320, "verification cost token outside vocabulary");
        out.push_back(uint32_t(n));
    }
    return out;
}
inline Manifest parse(const Json& value) {
    keys(value, {"schema", "seed", "sealed_source_sha256", "prefixes"});
    const auto schema = field(value, "schema").unsigned_integer();
    require(schema == 1 || schema == 2, "unsupported verification cost manifest schema");
    Manifest out; out.seed = field(value, "seed").unsigned_integer(); out.sealed_source_sha256 = field(value, "sealed_source_sha256").string();
    require(hash(out.sealed_source_sha256), "invalid sealed verification cost source hash");
    const auto& prefixes = field(value, "prefixes").array(); require(prefixes.size() == 2, "verification cost requires two frozen prefixes");
    for (size_t index = 0; index < prefixes.size(); ++index) {
        const auto& item = prefixes[index]; keys(item, {"input_ids", "continuation_ids", "logit_sha256", "order"});
        Prefix p; p.input = ids(field(item, "input_ids"), index ? 3072 : 1536);
        // Nine already-sealed greedy IDs: eight candidate rows and one continuation witness.
        p.continuation = ids(field(item, "continuation_ids"), 9);
        const auto& hashes = field(item, "logit_sha256").array(); require(hashes.size() == 10, "verification cost requires base plus nine sealed logit hashes");
        for (const auto& h : hashes) { p.logit_hashes.emplace_back(h.string()); require(hash(p.logit_hashes.back()), "invalid sealed logit hash"); }
        const auto& order = field(item, "order").array(); require(order.size() >= 6 && order.size() <= 18, "verification cost arm count bound");
        // Schema1 retains its original admission. Schema2 is a prospective
        // N3 screen: the six old shapes plus all three N3 prefixes, twice each.
        require(schema != 2 || order.size() == 18, "verification cost N3 requires eighteen arms");
        std::set<std::pair<uint32_t, uint32_t>> coverage;
        std::multiset<std::pair<uint32_t, uint32_t>> repetitions;
        for (const auto& entry : order) {
            keys(entry, {"rows", "accepted", "proposal_ids"}); Arm arm;
            const auto n = field(entry, "rows").unsigned_integer(), k = field(entry, "accepted").unsigned_integer();
            require((n == 1 || n == 2 || n == 8 || (schema == 2 && n == 3)) && k <= n && (n != 1 || k == 1), "invalid verification cost arm shape");
            arm.rows = uint32_t(n); arm.accepted = uint32_t(k); arm.proposals = ids(field(entry, "proposal_ids"), arm.rows);
            for (uint32_t i = 0; i < arm.accepted; ++i)
                require(arm.proposals[i] == p.continuation[i], "accepted proposal differs from sealed continuation");
            if (arm.accepted < arm.rows) require(arm.proposals[arm.accepted] != p.continuation[arm.accepted], "partial arm lacks its explicit mismatch");
            coverage.emplace(arm.rows, arm.accepted); repetitions.emplace(arm.rows, arm.accepted); p.order.push_back(std::move(arm));
        }
        for (const auto required : {std::pair{1u, 1u}, {2u, 2u}, {8u, 8u}, {2u, 1u}, {8u, 1u}, {8u, 7u}})
            require(coverage.contains(required), "verification cost arm coverage incomplete");
        if (schema == 2)
            for (const auto required : {std::pair{1u, 1u}, {2u, 2u}, {8u, 8u}, {2u, 1u}, {8u, 1u}, {8u, 7u}, {3u, 1u}, {3u, 2u}, {3u, 3u}})
                require(repetitions.count(required) == 2, "verification cost N3 requires two of every registered shape");
        out.prefixes.push_back(std::move(p));
    }
    return out;
}
inline Manifest load(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path), "verification cost manifest must be a regular file");
    const auto bytes = std::filesystem::file_size(path); require(bytes && bytes <= 512000, "verification cost manifest byte bound");
    std::ifstream file(path, std::ios::binary); require(bool(file), "cannot open verification cost manifest");
    std::string text(size_t(bytes), '\0'); file.read(text.data(), std::streamsize(text.size()));
    require(file.gcount() == std::streamsize(text.size()) && file.peek() == std::ifstream::traits_type::eof(), "verification cost manifest changed while reading");
    auto out = parse(parse_json(text, {512000, 20000, 8})); out.bytes = bytes; return out;
}
#ifdef TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST
// Separate schema; old checkpoint/greedy COST manifests retain their admission.
// Teacher-forced rows measure target work, not drafter acceptance or sampled law.
inline constexpr uint32_t sampled_current_rows = 12, sampled_reference_rows = 24;
struct SampledManifest {
    uint64_t seed = 0, bytes = 0;
    std::string sealed_source_sha256;
    std::vector<uint32_t> input, teacher;
};
inline SampledManifest parse_sampled_current(const Json& value) {
    keys(value, {"schema", "scope", "seed", "sealed_source_sha256", "input_ids", "teacher_ids"});
    require(field(value, "schema").unsigned_integer() == 3 &&
        field(value, "scope").string() == "sampled-current-carrier-cost", "explicit current sampled COST schema required");
    SampledManifest out; out.seed = field(value, "seed").unsigned_integer();
    out.sealed_source_sha256 = field(value, "sealed_source_sha256").string();
    require(hash(out.sealed_source_sha256), "invalid sampled COST retained source hash");
    out.input = ids(field(value, "input_ids"), 3072);
    out.teacher = ids(field(value, "teacher_ids"), sampled_reference_rows);
    return out;
}
inline SampledManifest load_sampled_current(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path), "sampled COST manifest must be a regular file");
    const auto bytes = std::filesystem::file_size(path);
    require(bytes && bytes <= 512000, "sampled COST manifest byte bound");
    std::ifstream file(path, std::ios::binary); require(bool(file), "cannot open sampled COST manifest");
    std::string text(size_t(bytes), '\0'); file.read(text.data(), std::streamsize(text.size()));
    require(file.gcount() == std::streamsize(text.size()) && file.peek() == std::ifstream::traits_type::eof(),
        "sampled COST manifest changed while reading");
    auto out = parse_sampled_current(parse_json(text, {512000, 20000, 8})); out.bytes = bytes; return out;
}
#endif
}
