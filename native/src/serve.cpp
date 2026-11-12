#include "tokspur/text_assets.hpp"
#include "tokspur/serve.hpp"
#include "tokspur/http.hpp"
#include "tokspur/owned_worker.hpp"
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#include "tokspur/mtp_wire.hpp"
#endif
#include "tokspur/text.hpp"
#include "tokspur/gguf.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

namespace tokspur {
namespace {
constexpr std::string_view tokenizer_pin = text_assets::tokenizer_sha256;
constexpr std::string_view template_pin = text_assets::template_sha256;
std::atomic<bool> stopping{false};
static_assert(std::atomic<bool>::is_always_lock_free, "shutdown flag must be signal-safe lock-free storage");
void stop_signal(int) { stopping.store(true, std::memory_order_relaxed); }
void require(bool value, const char* why) { if (!value) throw FormatError(why); }
std::string_view pack_startup_message(const Json& ready, bool requested_pack) {
    const auto* schema = ready.find("derived_pack_schema");
    const auto* recipe = ready.find("derived_pack_recipe");
    if (!requested_pack && !schema && !recipe) return "tokspur: loaded raw GGUF weights\n";
    if (requested_pack && schema && schema->kind() == Json::Kind::number &&
        recipe && recipe->kind() == Json::Kind::string) {
        if (schema->number_lexeme() == "2" && recipe->string() == "tokspur-pack-v2")
            return "tokspur: loaded Pack v2 (tokspur-pack-v2)\n";
        if (schema->number_lexeme() == "3" && recipe->string() == "tokspur-pack-v3")
            return "tokspur: loaded Pack v3 (tokspur-pack-v3)\n";
    }
    return "tokspur: weight pack identity unavailable\n";
}

bool json_content_type(std::string_view input) {
    const auto trim = [](std::string_view value) {
        const auto first = value.find_first_not_of(" \t\r\n"), last = value.find_last_not_of(" \t\r\n");
        return first == std::string_view::npos ? std::string_view{} : value.substr(first, last - first + 1);
    };
    const auto lower = [](std::string_view value) { std::string result(value); for (auto& c : result) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return result; };
    const auto separator = input.find(';');
    if (lower(trim(input.substr(0, separator))) != "application/json") return false;
    if (separator == std::string_view::npos) return true;
    const auto parameter = input.substr(separator + 1);
    const auto equals = parameter.find('=');
    if (equals == std::string_view::npos || lower(trim(parameter.substr(0, equals))) != "charset") return false;
    auto value = trim(parameter.substr(equals + 1));
    if (value.starts_with('"') || value.ends_with('"')) {
        if (value.size() < 2 || !value.starts_with('"') || !value.ends_with('"')) return false;
        value = value.substr(1, value.size() - 2);
    }
    return lower(value) == "utf-8";
}
using Clock = std::chrono::steady_clock;
uint64_t random_word() { uint64_t result; arc4random_buf(&result, sizeof(result)); return result; }
std::string random_hex(size_t bytes) {
    require(bytes <= 64, "random ID bound"); std::array<unsigned char, 64> data{}; arc4random_buf(data.data(), bytes);
    constexpr char hex[] = "0123456789abcdef"; std::string result;
    for (size_t i = 0; i < bytes; ++i) { result += hex[data[i] >> 4]; result += hex[data[i] & 15]; }
    return result;
}
bool equal_secret(std::string_view left, std::string_view right) {
    size_t difference = left.size() ^ right.size();
    for (size_t i = 0; i < right.size(); ++i) difference |= static_cast<unsigned char>(right[i]) ^ (i < left.size() ? static_cast<unsigned char>(left[i]) : 0U);
    return difference == 0;
}
struct Fd {
    int value = -1;
    bool owns_flock = false;
    explicit Fd(int fd = -1) : value(fd) {}
    int release_lock() {
        if (!owns_flock) return 0;
        int result;
        do { result = ::flock(value, LOCK_UN); } while (result < 0 && errno == EINTR);
        if (result == 0) owns_flock = false;
        return result;
    }
    ~Fd() {
        if (value < 0) return;
        if (release_lock() < 0) std::cerr << "tokspur-native failed to release private state lock: errno " << errno << '\n';
        ::close(value);
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
};
class PrivateState {
    Fd directory_, lock_;
    std::string key_;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    Fd prefix_trace_;
    uint64_t prefix_trace_bytes_ = 0;
    bool prefix_trace_failed_ = false;
#endif
public:
    explicit PrivateState(const std::filesystem::path& path) {
        require(path.is_absolute(), "state directory must be absolute");
        if (::mkdir(path.c_str(), 0700) != 0) require(errno == EEXIST, "cannot create private state directory");
        directory_.value = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        struct stat state{}; require(directory_.value >= 0 && ::fstat(directory_.value, &state) == 0 && S_ISDIR(state.st_mode) && state.st_uid == geteuid() && (state.st_mode & 0777) == 0700,
            "state directory must be owned private mode0700, not a symlink");
        lock_.value = ::openat(directory_.value, "server.lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
        struct stat lock{}; require(lock_.value >= 0 && ::fstat(lock_.value, &lock) == 0 && S_ISREG(lock.st_mode) && lock.st_uid == geteuid() && lock.st_nlink == 1 && (lock.st_mode & 0777) == 0600,
            "invalid private server lock");
        require(::flock(lock_.value, LOCK_EX | LOCK_NB) == 0, "another server owns this state directory");
        lock_.owns_flock = true;
        Fd created(::openat(directory_.value, "api-key", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (created.value >= 0) {
            const auto data = random_hex(32) + "\n"; size_t offset = 0;
            while (offset < data.size()) {
                const auto n = ::write(created.value, data.data() + offset, data.size() - offset);
                if (n < 0 && errno == EINTR) continue;
                require(n > 0, "cannot initialize API key"); offset += size_t(n);
            }
            require(::fsync(created.value) == 0 && ::fsync(directory_.value) == 0, "cannot persist API key");
        } else require(errno == EEXIST, "cannot create private API key");
        Fd input(::openat(directory_.value, "api-key", O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
        struct stat info{}; require(input.value >= 0 && ::fstat(input.value, &info) == 0 && S_ISREG(info.st_mode) && info.st_uid == geteuid() && info.st_nlink == 1 && (info.st_mode & 0777) == 0600 && info.st_size == 65,
            "API key must be an owned mode0600 single-link 65-byte file");
        std::array<char, 65> bytes{}; size_t offset = 0;
        while (offset < bytes.size()) {
            const auto n = ::read(input.value, bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot read API key"); offset += size_t(n);
        }
        key_.assign(bytes.data(), 64);
        require(bytes[64] == '\n' && key_.find_first_not_of("0123456789abcdef") == std::string::npos, "malformed API key");
    }
    [[nodiscard]] const std::string& key() const { return key_; }
    void release_lock() { require(lock_.release_lock() == 0, "cannot release private state lock"); }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    void enable_prefix_trace() {
        prefix_trace_.value = ::openat(directory_.value, "prefix-trace.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        struct stat info{}; require(prefix_trace_.value >= 0 && ::fstat(prefix_trace_.value, &info) == 0 && S_ISREG(info.st_mode) && info.st_uid == geteuid() && info.st_nlink == 1 && (info.st_mode & 0777) == 0600,
            "private prefix trace requires a new owned mode0600 file");
        require(::fsync(directory_.value) == 0, "cannot persist private prefix trace entry");
    }
    void prefix_trace(const Json& record) {
        require(prefix_trace_.value >= 0 && !prefix_trace_failed_, "private prefix trace is unavailable");
        try {
            const auto bytes = record.dump(1024 * 1024) + '\n';
            require(bytes.size() <= (16ull << 20) - prefix_trace_bytes_, "private prefix trace exceeds16MiB bound");
            size_t offset = 0;
            while (offset < bytes.size()) {
                const auto written = ::write(prefix_trace_.value, bytes.data() + offset, bytes.size() - offset);
                if (written < 0 && errno == EINTR) continue;
                require(written > 0, "cannot write private prefix trace"); offset += size_t(written); prefix_trace_bytes_ += uint64_t(written);
            }
            require(::fsync(prefix_trace_.value) == 0, "cannot persist private prefix trace");
        } catch (...) { prefix_trace_failed_ = true; throw; }
    }
#endif
};
std::string template_data(const std::filesystem::path& path) {
    MappedFile file(path); require(file.size() && file.size() <= 65536, "template byte bound");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256(file.bytes().data(), CC_LONG(file.size()), digest.data());
    constexpr char digits[] = "0123456789abcdef"; std::string hash;
    for (const auto byte : digest) { hash += digits[byte >> 4]; hash += digits[byte & 15]; }
    require(hash == template_pin, "template pin mismatch");
    const auto bytes = file.bytes(); std::string result(reinterpret_cast<const char*>(bytes.data()), bytes.size()); file.assert_unchanged(); return result;
}
Json error_json(std::string_view message, std::string_view type) {
    return Json(Json::Object{{"error", Json(Json::Object{{"message", Json(std::string(message.substr(0, 2048)))}, {"type", Json(std::string(type))}, {"code", Json(std::string(type))}})}});
}
struct Options {
    OwnedWorkerOptions worker;
    std::filesystem::path tokenizer, model_template, state;
    uint16_t port = 18081;
    uint32_t request_seconds = 3600;
    std::string alias = "tokspur";
    ToolEnumValidation tool_enums = ToolEnumValidation::strict;
    ToolArgValidation tool_args = ToolArgValidation::strict;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    bool prefix_trace = false;
    std::optional<uint32_t> mtp_first_output_packet;
#endif
};
uint32_t integer(std::string_view text, uint32_t low, uint32_t high) {
    uint32_t value; const auto r = std::from_chars(text.data(), text.data() + text.size(), value);
    require(r.ec == std::errc{} && r.ptr == text.data() + text.size() && value >= low && value <= high, "server option outside bound"); return value;
}
Options options(int argc, char** argv) {
    Options out; std::set<std::string_view> seen; bool allow = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (!option.starts_with("--")) { out.worker.shards.emplace_back(argv[i]); continue; }
        require(seen.insert(option).second && ++i < argc, "duplicate server option or missing value"); const std::string_view value(argv[i]);
        if (option == "--worker") out.worker.executable = value;
        else if (option == "--derived-pack") {
            require(!value.empty(), "derived pack path cannot be empty"); out.worker.derived_pack = value;
        }
        else if (option == "--derived-pack-auth-workers") out.worker.derived_pack_auth_workers = uint32_t(integer(value, 1, 16));
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        else if (option == "--native-weight-lanes") {
            require(!value.empty(), "native lane-layout path cannot be empty"); out.worker.weight_lanes = value;
        }
        else if (option == "--native-weight-lanes-auth-workers") out.worker.weight_lanes_auth_workers = uint32_t(integer(value, 1, 16));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        else if (option == "--native-fast-math") {
            require(value == "staged" || value == "channel", "private native-fast-math must be staged or channel");
            out.worker.fast_channel = value == "channel";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
        else if (option == "--native-scalar-channel") {
            require(value == "on" || value == "off", "private native-scalar-channel must be off or on");
            out.worker.scalar_channel = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        else if (option == "--native-sampled-prefill-input8") {
            require(value == "on" || value == "off", "private sampled input8 must be off or on");
            out.worker.sampled_prefill = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        else if (option == "--native-raw-decode") {
            require(value == "on" || value == "off", "native raw routed decode requires on or off"); out.worker.raw_decode = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        else if(option=="--native-q8-three-rows"){
            require(value=="on"||value=="off","native three-row Q8 must be on or off");out.worker.q8_three_rows=value=="on";
        }
#endif
        else if (option == "--private-prefix-trace") {
            require(value == "on" || value == "off", "private prefix trace must be off or on"); out.prefix_trace = value == "on";
        }
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        else if (option == "--native-prefill-input512") {
            require(value == "on" || value == "off", "native input512 must be off or on");
            out.worker.prefill_input512 = value == "on";
        }
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        else if (option == "--native-gdn-old-order-batch") {
            require(value == "on" || value == "off", "native old-order GDN must be off or on"); out.worker.gdn_old_order_batch = value == "on";
        }
#endif
        else if (option == "--native-prefill-hc-rows") {
            require(value == "on" || value == "off", "native HC rows must be off or on"); out.worker.prefill_hc_rows = value == "on";
        }
        else if (option == "--native-prefill-router-window") {
            require(value == "on" || value == "off", "native router window must be off or on"); out.worker.prefill_router_window = value == "on";
        }
        else if (option == "--native-prefill-hc-down-window") {
            require(value == "on" || value == "off", "native HC down window must be off or on"); out.worker.prefill_hc_down_window = value == "on";
        }
        else if (option == "--native-qsa-sparse") {
            require(value == "on" || value == "off", "native sparse QSA must be off or on"); out.worker.qsa_sparse = value == "on";
        }
        else if (option == "--native-gdn-ported") {
            require(value == "on" || value == "off", "native ported GDN must be off or on"); out.worker.gdn_ported = value == "on";
        }
        else if (option == "--native-gdn-prefill-ported") {
            require(value == "on" || value == "off", "native prefill-only ported GDN must be off or on"); out.worker.gdn_prefill_ported = value == "on";
        }
        else if (option == "--native-prefill-hc-down-f32") {
            require(value == "on" || value == "off", "native HC down batching must be off or on"); out.worker.prefill_hc_down_f32 = value == "on";
        }
        else if (option == "--native-prefill-final-logits") {
            require(value == "on" || value == "off", "native final-input logits must be off or on");
            out.worker.prefill_final_logits = value == "on";
        }
        else if (option == "--native-prefill-dense") {
            require(value == "f16" || value == "f32"
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
                || value == "f32-staged" || value == "nax"
#endif
                , "native prefill dense must select a compiled f16/f32/nax backend");
            out.worker.prefill_dense = std::string(value);
        }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        else if (option == "--native-prefill-grouped") {
            require(value == "serial" || value == "f32" || value == "auto" || value == "nax", "native prefill grouped must be serial, f32, auto or nax");
            out.worker.prefill_grouped = std::string(value);
        }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        else if (option == "--native-mtp-output-packet") {
            require(value == "8" || value == "64", "native MTP output packet must be8 or64");
            out.worker.mtp_output_packet = value == "8" ? 8 : 64;
        }
        else if (option == "--native-mtp-first-output-packet" || option == "--first-output-packet") {
            out.mtp_first_output_packet = integer(value, 1, 64);
        }
#endif
        else if (option == "--native-concurrent-dispatch") {
            require(value == "on" || value == "off", "native concurrent dispatch must be off or on");
            out.worker.concurrent_dispatch = value == "on";
        }
#endif
        else if (option == "--native-mtp-sidecar") {
            require(!value.empty(), "native MTP sidecar cannot be empty"); out.worker.mtp_sidecar = value;
        }
        else if (option == "--drafter" || option == "--native-drafter") {
            require(value == "mtp" || value == "dflash2", "drafter must be mtp or dflash2");
            out.worker.drafter = std::string(value);
        }
        else if (option == "--dflash2-dir" || option == "--native-dflash2-dir") {
            require(!value.empty(), "DFlash2 directory cannot be empty"); out.worker.dflash2_dir = value;
        }
        else if (option == "--dflash2-precision" || option == "--native-dflash2-precision") {
            require(value == "q8_0" || value == "bf16", "DFlash2 precision must be q8_0 or bf16");
            out.worker.dflash2_precision = std::string(value);
        }
        else if (option == "--dflash2-max-drafts" || option == "--native-dflash2-max-drafts") {
            out.worker.dflash2_max_drafts = integer(value, 1, 7);
        }
        else if (option == "--dflash2-vocab-slice" || option == "--native-dflash2-vocab-slice") {
            const auto slice = integer(value, 0, 248319);
            require(slice == 0 || (slice >= 256 && slice % 32 == 0), "DFlash2 vocab slice must be 0 or a multiple of 32 in [256, 248319]");
            out.worker.dflash2_vocab_slice = slice;
        }
        else if (option == "--native-prepared-validation") {
            require(value == "off" || value == "on", "native prepared validation must be off or on"); out.worker.prepared_validation = value == "on";
        }
        else if (option == "--native-verify-cost-profile") { require(!value.empty(), "empty measured cost profile"); out.worker.verify_cost_profile = std::filesystem::path(value); }
        else if (option == "--native-verify-m8-splits") { require(!value.empty(), "empty M8 split policy"); out.worker.verify_m8_splits = std::filesystem::path(value); }
        else if (option == "--native-dflash2-draft-policy") {
            require(value == "always" || value == "auto", "native DFlash2 draft policy must be always or auto"); out.worker.dflash2_draft_policy = std::string(value);
        }
        else if (option == "--native-dflash2-defer-context" || option == "--native-verify-nax-m8" || option == "--verify-nax-m8") {
            require(value == "off" || value == "on", "native boolean control must be off or on");
            if (option == "--native-dflash2-defer-context") out.worker.dflash2_defer_context = value == "on";
            else out.worker.verify_nax_m8 = value == "on";
        }
        else if (option == "--native-dflash2-verify-rows") {
            require(value == "all" || value == "auto", "native DFlash2 verify rows must be all or auto"); out.worker.dflash2_verify_rows = std::string(value);
        }
        else if (option == "--logit-topk" || option == "--native-logit-topk") {
            require(value == "on" || value == "off", "native logit top-k must be off or on");
            out.worker.logit_topk = value == "on";
        }
        else if (option == "--ple-async-gather" || option == "--native-ple-async-gather") {
            require(value == "on" || value == "off", "native PLE async gather must be off or on");
            out.worker.ple_async_gather = value == "on";
        }
        else if (option == "--mtp-fast-round" || option == "--native-mtp-fast-round") {
            require(value == "on" || value == "off", "native MTP fast round must be off or on");
            out.worker.mtp_fast_round = value == "on";
        }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        else if (option == "--native-mtp-depth" || option == "--mtp-depth") {
            const auto depth = integer(value, 1, 7);
            require(depth == 1 || depth == 2 || depth == 3 || depth == 7, "native MTP depth must be 1, 2, 3 or 7");
            out.worker.mtp_depth = depth;
        }
#endif
        else if (option == "--native-mtp-request-carry") {
            require(value == "on" || value == "off", "native MTP request carry must be off or on"); out.worker.mtp_request_carry = value == "on";
        }
        else if (option == "--native-mtp-draft-attention") {
            require(value == "serial" || value == "tiled128" || value == "tiled256" || value == "split16", "unsupported native MTP draft-attention mode");
            out.worker.mtp_draft_attention = value;
        }
        else if (option == "--native-mtp-draft-slice") {
            require(value == "on" || value == "off", "native MTP draft slice must be off or on");
            out.worker.mtp_draft_slice = value == "on";
        }
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
        else if (option == "--native-verify-nax" || option == "--verify-nax") {
            require(value == "on" || value == "off", "native verify NAX must be off or on");
            out.worker.verify_nax = value == "on";
        }
        else if (option == "--native-verify-hc-gdn-fused" || option == "--verify-hc-gdn-fused") {
            require(value == "on" || value == "off", "native verify HC+GDN fused must be off or on");
            out.worker.verify_hc_gdn_fused = value == "on";
        }
#endif
        else if (option == "--known-token-capacity" || option == "--native-known-token-capacity") {
            const auto cap = integer(value, 0, 2048);
            require(cap == 0 || cap == 8 || cap == 512 || cap == 2048, "known token capacity must be 0, 8, 512 or 2048");
            out.worker.known_token_capacity = cap;
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        else if (option == "--native-verify8") {
            require(value == "off" || value == "on", "private native-verify8 must be off or on"); out.worker.verify8 = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        else if (option == "--native-branch8") {
            require(value == "off" || value == "on", "private native-branch8 must be off or on"); out.worker.branch8 = value == "on";
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        else if (option == "--native-qsa-stage-v8") {
            require(value == "off" || value == "on", "private native-qsa-stage-v8 must be off or on");
            out.worker.stage8_present = true; out.worker.stage8_enabled = value == "on";
        }
#endif
        else if (option == "--stack-execution") {
            require(value == "direct" || value == "recurrent-replay", "unsupported serving startup execution mode");
            out.worker.stack_execution = value;
        }
        else if (option == "--policy") out.worker.policy = value;
        else if (option == "--library") out.worker.library = value;
        else if (option == "--checkpoint-slots") out.worker.checkpoint_slots = integer(value, 0, 4);
        else if (option == "--checkpoint-budget-bytes") out.worker.checkpoint_budget_bytes = integer(value, 0, 512u << 20);
        else if (option == "--checkpoint-dir") out.worker.checkpoint_ssd_dir = std::filesystem::path(value);
        else if (option == "--checkpoint-cap-gib") {
            const double gib = std::stod(std::string(value));
            require(gib >= 0.0 && gib <= 32.0, "checkpoint-cap-gib must be between 0 and 32 GiB");
            out.worker.checkpoint_ssd = (gib > 0.0);
            out.worker.checkpoint_ssd_budget_bytes = static_cast<uint64_t>(gib * (1024.0 * 1024.0 * 1024.0));
        }
        else if (option == "--checkpoint-floor-gib") {
            const double gib = std::stod(std::string(value));
            require(gib >= 0.0, "checkpoint-floor-gib must be non-negative");
            out.worker.checkpoint_ssd_floor_bytes = static_cast<uint64_t>(gib * (1024.0 * 1024.0 * 1024.0));
        }
        else if (option == "--checkpoint-policy") {
            require(value == "recompute-cost" || value == "lru", "checkpoint policy requires recompute-cost or lru");
            out.worker.checkpoint_policy = std::string(value);
        }
        else if (option == "--overlap-prefill") {
            require(value == "true" || value == "false" || value == "on" || value == "off", "overlap prefill requires boolean");
            out.worker.overlap_prefill = (value == "true" || value == "on");
        }
        else if (option == "--native-swiglu-fused" || option == "--swiglu-fused") {
            require(value == "on" || value == "off", "native SwiGLU fusion must be off or on");
            out.worker.swiglu_fused = (value == "on");
        }
        else if (option == "--native-q8-gemv-rg4" || option == "--q8-gemv-rg4") {
            require(value == "off" || value == "on" || value == "wide", "native 4-row/SG Q8 GEMV must be off, on or wide");
            out.worker.q8_gemv_rg4 = std::string(value);
        }
        else if (option == "--native-q8-three-msg" || option == "--q8-three-msg") {
            require(value == "off" || value == "on" || value == "wide", "native multi-SG three-row Q8 must be off, on or wide");
            out.worker.q8_three_msg = std::string(value);
        }
        else if (option == "--native-moe-three-union" || option == "--moe-three-union") {
            require(value == "on" || value == "off", "native unioned three-row MoE must be off or on");
            out.worker.moe_three_union = (value == "on");
        }
        else if (option == "--native-mtp-lookup" || option == "--mtp-lookup") {
            require(value == "off" || value == "bypass" || value == "hybrid", "native MTP lookup must be off, bypass or hybrid");
            out.worker.mtp_lookup = std::string(value);
        }
        else if (option == "--native-mtp-lookup-match" || option == "--mtp-lookup-match") {
            out.worker.mtp_lookup_match = integer(value, 2, 16);
        }
        else if (option == "--native-prefill-m64" || option == "--prefill-m64") {
            require(value == "off" || value == "on" || value == "short", "native prefill M64 must be off, on or short");
            out.worker.prefill_m64 = std::string(value);
        }
        else if (option == "--native-prefill-multiseg" || option == "--prefill-multiseg") {
            require(value == "off" || value == "on", "native prefill multiseg must be off or on");
            out.worker.prefill_multiseg = (value == "on");
        }
        else if (option == "--native-prefill-swizzle" || option == "--prefill-swizzle") {
            require(value == "off" || value == "on", "native prefill swizzle must be off or on");
            out.worker.prefill_swizzle = (value == "on");
        }
        else if (option == "--native-prefill-router-fast" || option == "--prefill-router-fast") {
            require(value == "off" || value == "on", "native prefill router fast must be off or on");
            out.worker.prefill_router_fast = (value == "on");
        }
        else if (option == "--native-qsa-prefill-fused" || option == "--qsa-prefill-fused") {
            require(value == "off" || value == "on", "native fused QSA prefill must be off or on");
            out.worker.qsa_prefill_fused = (value == "on");
        }
        else if (option == "--native-hc-consumer-fused" || option == "--hc-consumer-fused") {
            require(value == "off" || value == "on", "native fused HC consumer must be off or on");
            out.worker.hc_consumer_fused = (value == "on");
        }
        else if (option == "--native-ple-overlap" || option == "--ple-overlap") {
            require(value == "off" || value == "on", "native PLE overlap must be off or on");
            out.worker.ple_overlap = (value == "on");
        }
        else if (option == "--shared-prefix-cache" || option == "--native-shared-prefix-cache") {
            require(value == "off" || value == "on", "shared prefix cache must be off or on");
            out.worker.shared_prefix_cache = (value == "on");
        }
        else if (option == "--tokenizer") out.tokenizer = value;
        else if (option == "--template") out.model_template = value;
        else if (option == "--state-dir") out.state = value;
        else if (option == "--model-alias") { require(!value.empty() && value.size() <= 64 && value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") == std::string_view::npos, "invalid model alias"); out.alias = value; }
        else if (option == "--tool-enum-validation") out.tool_enums = parse_tool_enum_validation(value);
        else if (option == "--tool-arg-validation") out.tool_args = parse_tool_arg_validation(value);
        else if (option == "--port") out.port = uint16_t(integer(value, 0, 65535));
        else if (option == "--request-timeout-seconds") out.request_seconds = integer(value, 1, 86400);
        else if (option == "--startup-timeout-seconds") out.worker.startup_timeout = std::chrono::seconds(integer(value, 1, 3600));
        else if (option == "--step-timeout-seconds") out.worker.step_timeout = std::chrono::seconds(integer(value, 1, 300));
        else if (option == "--allow-full-model") { require(value == "true", "server requires explicit full-model consent"); allow = true; }
        else throw FormatError("unknown server option");
    }
    require(allow && out.tokenizer.is_absolute() && out.model_template.is_absolute() && !out.state.empty(), "server requires consent, absolute tokenizer/template/state and owned worker assets");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(out.worker.weight_lanes.empty() || (owned_weight_lanes_transport(out.worker) && out.worker.weight_lanes.is_absolute()),
        "native lane layout requires explicit corrected scalar-channel consent and an absolute path");
    require(!out.worker.weight_lanes_auth_workers || !out.worker.weight_lanes.empty(),
        "native lane authentication workers require an explicit sidecar");
#endif
    require(bool(out.worker.checkpoint_slots) == bool(out.worker.checkpoint_budget_bytes) &&
        (!out.worker.checkpoint_slots || out.worker.stack_execution == "direct"), "HTTP broker requires paired bounded direct checkpoint preparation");
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
    const bool scalar_transport = scalar_channel_transport(out.worker);
    require(!out.worker.scalar_channel || scalar_transport,
        "private scalar-channel requires explicit channel/Stage-ON direct execution without V8/branch/history");
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    require(!out.worker.sampled_prefill || scalar_transport, "sampled input8 requires explicit scalar-channel consent");
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    require(out.worker.mtp_sidecar.empty() || out.worker.dflash2_dir.empty(), "cannot combine MTP sidecar and DFlash2 drafter");
    require(!out.worker.drafter || *out.worker.drafter == "mtp" || *out.worker.drafter == "dflash2", "drafter must be mtp or dflash2");
    require(out.worker.drafter != std::optional<std::string>("dflash2") || !out.worker.dflash2_dir.empty(), "dflash2 drafter requires --dflash2-dir");
    require(out.worker.drafter != std::optional<std::string>("mtp") || out.worker.dflash2_dir.empty(), "mtp drafter refuses --dflash2-dir");
    require(!out.worker.dflash2_precision || !out.worker.dflash2_dir.empty(), "dflash2 precision requires --dflash2-dir");
    if (!out.worker.dflash2_dir.empty()) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        if (!out.worker.mtp_depth) out.worker.mtp_depth = 7;
#endif
        if (!out.worker.mtp_draft_attention) out.worker.mtp_draft_attention = "split16";
        out.worker.mtp_request_carry = true;
    }
    const bool has_spec = !out.worker.mtp_sidecar.empty() || !out.worker.dflash2_dir.empty();
    require(!has_spec || (scalar_transport && !out.worker.sampled_prefill &&
        (out.worker.mtp_sidecar.empty() || out.worker.mtp_sidecar.is_absolute()) &&
        (out.worker.dflash2_dir.empty() || out.worker.dflash2_dir.is_absolute())),
        "native MTP/DFlash2 requires its distinct explicit fixed channel consent and absolute asset path");
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    require(!out.worker.prefill_input512 || owned_concurrent_transport(out.worker), "input512 requires explicit fixed depth-two MTP scalar transport");
    require(!out.worker.prefill_final_logits || out.worker.prefill_input512, "native final-input logits require input512");
    const bool dense_staged_or_nax = (out.worker.prefill_dense == "f32-staged" || out.worker.prefill_dense == "nax");
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    const bool grouped_ok = (out.worker.prefill_grouped == "f32" || out.worker.prefill_grouped == "auto" || out.worker.prefill_grouped == "nax");
    require(!out.worker.prefill_hc_rows.has_value() || owned_concurrent_transport(out.worker), "HC rows require fixed MTP transport");
    require(!out.worker.prefill_hc_rows.value_or(false) || (out.worker.prefill_input512 && dense_staged_or_nax && grouped_ok && !out.worker.prefill_hc_down_f32.value_or(false)), "HC rows require staged FP32 or NAX input512 and scalar HC down");
#else
    require(!out.worker.prefill_hc_rows.has_value(), "HC rows selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    require(!out.worker.prefill_router_window.has_value() || owned_concurrent_transport(out.worker), "router window require fixed MTP transport");
    require(!out.worker.prefill_router_window.value_or(false) || (out.worker.prefill_input512 && dense_staged_or_nax && grouped_ok && !out.worker.prefill_hc_down_f32.value_or(false)), "router window requires staged FP32 or NAX input512 and scalar HC down");
#else
    require(!out.worker.prefill_router_window.has_value(), "router window selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    require(!out.worker.prefill_hc_down_window.has_value() || owned_concurrent_transport(out.worker), "HC down window requires fixed MTP transport");
    require(!out.worker.prefill_hc_down_window.value_or(false) || (out.worker.prefill_input512 && dense_staged_or_nax && grouped_ok && out.worker.prefill_hc_rows.value_or(false) && out.worker.prefill_router_window.value_or(false) && !out.worker.prefill_hc_down_f32.value_or(false)), "HC down window requires staged FP32 or NAX input512, exact HC rows, exact router window and reassociating HC down off");
#else
    require(!out.worker.prefill_hc_down_window.has_value(), "HC down window selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    require(!out.worker.qsa_sparse.has_value() || owned_concurrent_transport(out.worker), "sparse QSA requires fixed MTP transport");
    require(!out.worker.qsa_sparse.value_or(false) || (out.worker.prefill_input512 && dense_staged_or_nax && grouped_ok && out.worker.prefill_hc_rows.value_or(false) && out.worker.prefill_router_window.value_or(false) && out.worker.prefill_hc_down_window.value_or(false) && !out.worker.prefill_hc_down_f32.value_or(false)), "sparse QSA requires staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
#else
    require(!out.worker.qsa_sparse.has_value(), "sparse QSA selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    require(!out.worker.gdn_ported.has_value() || owned_concurrent_transport(out.worker), "ported GDN requires fixed MTP transport");
    require(!out.worker.gdn_ported.value_or(false) || (out.worker.prefill_input512 && dense_staged_or_nax && grouped_ok && out.worker.prefill_hc_rows.value_or(false) && out.worker.prefill_router_window.value_or(false) && out.worker.prefill_hc_down_window.value_or(false) && !out.worker.prefill_hc_down_f32.value_or(false)), "ported GDN requires staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
    require(!out.worker.gdn_prefill_ported.has_value() || owned_concurrent_transport(out.worker), "prefill-only ported GDN requires fixed MTP transport");
    require(!out.worker.gdn_prefill_ported.value_or(false) || (!out.worker.gdn_ported.value_or(false) && out.worker.prefill_input512 && dense_staged_or_nax && grouped_ok && out.worker.prefill_hc_rows.value_or(false) && out.worker.prefill_router_window.value_or(false) && out.worker.prefill_hc_down_window.value_or(false) && !out.worker.prefill_hc_down_f32.value_or(false)), "prefill-only ported GDN requires --native-gdn-ported off, staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
#else
    require(!out.worker.gdn_ported.has_value(), "ported GDN selection is not compiled");
    require(!out.worker.gdn_prefill_ported.has_value(), "prefill-only ported GDN selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    require(!out.worker.prefill_hc_down_f32.has_value() || owned_concurrent_transport(out.worker), "native HC down selection requires fixed MTP transport");
    require(!out.worker.prefill_hc_down_f32.value_or(false) || (out.worker.prefill_input512 && out.worker.prefill_dense == "f32" && (out.worker.prefill_grouped == "f32" || out.worker.prefill_grouped == "auto")), "native HC down batching requires explicit f32/f32 input512");
#else
    require(!out.worker.prefill_hc_down_f32.has_value(), "native HC down batching selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    require(!out.worker.gdn_old_order_batch || (out.worker.prefill_input512 && (out.worker.prefill_dense == "f32" || out.worker.prefill_dense == "f32-staged") && (out.worker.prefill_grouped == "f32" || out.worker.prefill_grouped == "auto")),
        "native old-order GDN requires explicit f32/f32 input512");
#endif
    if (dense_staged_or_nax) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
        require(out.worker.prefill_input512 && (out.worker.prefill_grouped == "f32" || out.worker.prefill_grouped == "auto" || out.worker.prefill_grouped == "nax") && !out.worker.prefill_hc_down_f32.value_or(false),
            "staged/NAX dense backend requires input512 compatible grouped mode and HC down off");
#else
        throw FormatError("staged dense backend is not compiled");
#endif
    }
    require(!out.worker.prefill_dense || out.worker.prefill_input512, "native prefill dense requires input512");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    require(!out.worker.prefill_grouped || out.worker.prefill_input512, "native prefill grouped requires input512");
    require(out.worker.prefill_grouped != "nax" || (dense_staged_or_nax && !out.worker.prefill_hc_down_f32.value_or(false)), "NAX grouped prefill requires staged/NAX dense and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    require(!out.worker.prefill_input512 || out.worker.weight_lanes.empty(), "input512 dense projections require original weight views");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    require(!out.worker.mtp_depth || (has_spec && (!out.worker.dflash2_dir.empty() ?
        (*out.worker.mtp_depth == 7 && out.worker.mtp_request_carry && out.worker.mtp_draft_attention == std::optional<std::string>("split16")) :
        (*out.worker.mtp_depth == 1 || (out.worker.mtp_request_carry && out.worker.mtp_draft_attention == std::optional<std::string>("tiled128"))))),
        "explicit speculative HTTP depth requires matching scoped carry/attention");
#endif
    require(!out.worker.mtp_request_carry || has_spec, "native MTP request carry requires its sidecar or DFlash2 directory");
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
    require(!out.worker.concurrent_dispatch.has_value() || owned_concurrent_transport(out.worker),
        "concurrent HTTP requires explicit fixed depth-two/seven carry scalar transport");
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    require(!out.worker.mtp_output_packet || owned_concurrent_transport(out.worker),
        "output-packet selection requires explicit fixed depth-two/seven carry scalar transport");
    require(!out.mtp_first_output_packet || owned_concurrent_transport(out.worker),
        "first-output-packet selection requires explicit fixed depth-two/seven carry scalar transport");
    require(!out.mtp_first_output_packet || *out.mtp_first_output_packet <= out.worker.mtp_output_packet.value_or(8),
        "first-output-packet cannot exceed native MTP output packet");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
    require(!out.worker.q8_three_rows.has_value()||owned_concurrent_transport(out.worker),"three-row Q8 HTTP requires the current fixed MTP transport");
#else
    require(!out.worker.q8_three_rows.has_value(),"three-row Q8 HTTP requires the current MTP transport build");
#endif
#endif
    require(!out.worker.mtp_draft_attention || has_spec, "native MTP draft-attention requires its sidecar or DFlash2 directory");
#endif
#elif defined(TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8) || defined(TOKSPUR_PRIVATE_HTTP_FAST_MATH)
    constexpr bool scalar_transport = false;
#endif
    const bool branch_transport =
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        out.worker.branch8 && out.worker.verify8 && out.worker.checkpoint_slots && out.worker.checkpoint_budget_bytes;
    require(!out.worker.branch8 || branch_transport, "private native-branch8 requires explicit V8 and prepared history");
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    require(!out.worker.stage8_present || ((branch_transport || scalar_transport) && out.worker.stack_execution == "direct"),
        "private staged HTTP requires explicit branch8/V8/direct/history consent");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    require(!out.worker.fast_channel.has_value() || ((branch_transport || scalar_transport) && out.worker.stage8_present && out.worker.stage8_enabled),
        "private fast HTTP math requires explicit staged-ON branch8/V8/history consent");
#endif
#else
        false;
#endif
    require(!out.worker.verify8 || (out.worker.stack_execution == "direct" && (branch_transport || (!out.worker.checkpoint_slots && !out.worker.checkpoint_budget_bytes))),
        "private native-verify8 requires direct execution without checkpoints");
#endif
    // A differently named parent is intentional: the existing target's worker
    // exclusion must continue to reject every other tokspur-native model child.
    require(std::filesystem::path(argv[0]).filename() == "tokspur-server", "owned HTTP parent must be named tokspur-server");
    return out;
}
struct Job {
    http::Connection connection;
    Json request;
    Clock::time_point deadline;
};
} // namespace

int server_cli(int argc, char** argv) {
    auto config = options(argc, argv);
    require(std::signal(SIGINT, stop_signal) != SIG_ERR && std::signal(SIGTERM, stop_signal) != SIG_ERR && std::signal(SIGHUP, stop_signal) != SIG_ERR,
        "cannot arm native server shutdown handlers");
    config.worker.startup_cancelled = [] { return stopping.load(std::memory_order_relaxed); };
    require(!stopping.load(), "startup cancelled before asset preparation");
    PrivateState state(config.state);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (config.prefix_trace) state.enable_prefix_trace();
#endif
    http::Listener listener(config.port, 4); // Port and state are reserved before loading a model.
    NativeText tokenizer(config.tokenizer, tokenizer_pin);
    ChatRenderer renderer(extract_flash_next_literals(template_data(config.model_template)));
    tokenizer.verify_gguf(ModelFiles(config.worker.shards));
    const auto marker = [&](std::string_view text) {
        const auto ids = tokenizer.encode(text); require(ids.size() == 1 && tokenizer.decode_bytes(ids) == text, "unsupported terminal token identity"); return ids[0];
    };
    const ResponseAdmission admission{uint32_t(tokenizer.vocabulary_size()), marker("<|im_end|>"), marker("<|endoftext|>"), marker("<|im_start|>")};
    // Constructor arms its independent guardian before exec/model loading.
    require(!stopping.load(), "startup cancelled before model launch");
    OwnedWorker worker(config.worker);
    validate_worker_ready(worker.ready(), config.worker.stack_execution, config.worker.checkpoint_slots, config.worker.checkpoint_budget_bytes);
    std::cerr << pack_startup_message(worker.ready(), !config.worker.derived_pack.empty());
    ServeSession session({[&](const Json& input) { return renderer.render(input); }, [&](const RenderedChat& input) { return tokenizer.encode(input.prompt, input.literal_ranges); },
        [&](std::span<const uint32_t> ids) { return tokenizer.decode_bytes(ids); }, admission}, config.alias, config.worker.stack_execution,
        config.worker.checkpoint_slots ? worker.ready() : Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , config.worker.verify8, config.worker.verify8 ? worker.ready() : Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , config.worker.branch8
#endif
        , config.tool_enums
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        , config.worker.sampled_prefill
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , (config.worker.mtp_sidecar.empty() && config.worker.dflash2_dir.empty()) ? Json() : worker.ready()
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        , config.worker.weight_lanes.empty() ? Json() : worker.ready()
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , config.mtp_first_output_packet
#endif
        , config.tool_args
        );
    std::atomic<bool> shutdown{false}, available{true}; std::mutex mutex; std::condition_variable condition;
    std::optional<Job> pending;
    const auto authorization = "Bearer " + state.key();
    const auto run_job = [&](Job& job) {
        bool streaming = false, target_started = false;
        const auto cancelled = [&] { return shutdown.load() || stopping || Clock::now() >= job.deadline || job.connection.disconnected(); };
        try {
            require(!cancelled(), "queued request cancelled or expired");
            const auto result = session.complete(job.request, "chatcmpl-" + random_hex(16),
                [&](const Json& request, const auto& on_token) { target_started = true; return worker.request(request, job.deadline, cancelled, on_token); },
                {[&](const Json& event) {
                    if (!streaming) { job.connection.begin_sse(job.deadline); streaming = true; }
                    job.connection.send_sse_data(event.dump(1024 * 1024), job.deadline);
                }, cancelled, random_word
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                , config.prefix_trace ? std::function<void(const Json&)>([&](const Json& record) { state.prefix_trace(record); }) : std::function<void(const Json&)>()
#endif
                });
            if (streaming) { job.connection.send_sse_data("[DONE]", job.deadline); job.connection.finish_sse(job.deadline); }
            else job.connection.send_json(200, result.dump(8 * 1024 * 1024), job.deadline);
        } catch (const std::exception& error) {
            if (target_started) session.invalidate();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (target_started && config.worker.mtp_request_carry) {
                try { worker.abort_mtp_generation(Clock::now() + config.worker.drain_timeout); }
                catch (...) { /* An ambiguous cleanup invalidates/drains through OwnedWorker::request. */ }
            }
#endif
            available.store(worker.alive());
            if (!job.connection.disconnected() && !shutdown.load() && !stopping) {
                const auto deadline = Clock::now() + std::chrono::seconds(2);
                const auto payload = error_json(error.what(), target_started ? "generation_failed_no_retry" : "invalid_request_error").dump(8192);
                try {
                    if (streaming) { job.connection.send_sse_data(payload, deadline); job.connection.finish_sse(deadline); }
                    else job.connection.send_json(Clock::now() >= job.deadline ? 408 : target_started ? 502 : 400, payload, deadline);
                } catch (...) { /* Broken clients cannot waive worker draining/cleanup. */ }
            }
        }
    };
    std::thread generation([&] {
        while (true) {
            std::optional<Job> job;
            {
                std::unique_lock lock(mutex); condition.wait_for(lock, std::chrono::milliseconds(100), [&] { return shutdown.load() || pending.has_value(); });
                if (shutdown.load()) { pending.reset(); break; }
                if (!pending) { const bool live = worker.alive(); available.store(live); if (!live) session.invalidate(); continue; }
                job = std::move(pending); pending.reset();
            }
            run_job(*job);
        }
    });
    std::exception_ptr failure;
    try {
        Json::Object ready{{"event", Json("ready")}, {"endpoint", Json("http://127.0.0.1:" + std::to_string(listener.port()) + "/v1")},
            {"model", Json(config.alias)}, {"one_active_generation", Json(true)}, {"waiting_requests", Json::integer(1)},
            {"stack_execution", Json(config.worker.stack_execution)},
            {"tool_enum_validation", Json(std::string(tool_enum_validation_name(config.tool_enums)))},
            {"tool_enum_validation_policy", Json(std::string(tool_enum_validation_policy(config.tool_enums)))},
            {"checkpoint_slots", Json::integer(config.worker.checkpoint_slots)},
            {"checkpoint_combined_budget_bytes", Json::integer(config.worker.checkpoint_budget_bytes)},
            {"external_engine", Json(false)}, {"tools_executed", Json(false)}, {"performance_qualified", Json(false)}};
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (!config.worker.weight_lanes.empty()) for (const auto name : owned_weight_lane_fields)
            ready.emplace_back(std::string(name), *worker.ready().find(name)); // Validated before startup publication.
        if (config.worker.weight_lanes_auth_workers) for (const auto name : owned_lane_auth_fields)
            ready.emplace_back(std::string(name), *worker.ready().find(name));
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        ready.emplace_back("native_verify8", Json(config.worker.verify8));
        if (config.worker.verify8) {
            for (const auto name : {"verification_max_tokens", "verification_head_many", "lookup_proposal_tokens", "known_token_capacity", "known_residual_reuse",
                "hc_exact_mode", "wide_q8_exact_enabled", "wide_q8_many_enabled", "lookup_verification_budget_bytes", "lookup_verification_charged_bytes"})
                ready.emplace_back(name, *worker.ready().find(name));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        ready.emplace_back("native_branch8", Json(config.worker.branch8));
        if (config.worker.branch8) {
            ready.emplace_back("broker_protocol", *worker.ready().find("broker_protocol"));
            ready.emplace_back("broker_batch_width", *worker.ready().find("broker_batch_width"));
            ready.emplace_back("checkpoint_capture_stride", Json::integer(512));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        if (config.worker.stage8_present) {
            for (const auto name : {"qsa_stage_v8_prepared", "qsa_stage_v8_enabled", "qsa_stage_v8_scratch_logical_bytes", "qsa_stage_v8_scratch_charged_bytes",
                "qsa_stage_v8_calls", "qsa_stage_v8_physical_dispatches", "qsa_stage_v8_extra_dispatches"})
                ready.emplace_back(name, *worker.ready().find(name));
        }
#endif
        if (config.worker.stack_execution == "recurrent-replay") {
            for (const auto name : {"replay_prepared", "recurrent_replay_captures", "mapped_weight_region_limit_bytes"})
                ready.emplace_back(name, *worker.ready().find(name)); // Validated before readiness publication.
        }
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        if (config.worker.fast_channel.has_value()) {
            ready.emplace_back("native_fast_math", Json(*config.worker.fast_channel ? "channel" : "staged"));
            for (const auto name : {"ple_gate_transform", "ple_gate_transform_schema", "qsa_channel_chronological_mode",
                "qsa_channel_chronological_schema", "qsa_channel_factor_logical_bytes", "qsa_channel_factor_charged_bytes",
                "qsa_channel_calls", "qsa_channel_extra_dispatches"}) ready.emplace_back(name, *worker.ready().find(name));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
        ready.emplace_back("native_scalar_channel", Json(config.worker.scalar_channel));
        if (config.worker.scalar_channel) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!config.worker.mtp_sidecar.empty() || !config.worker.dflash2_dir.empty()) {
                ready.emplace_back("native_scalar_channel_protocol", *worker.ready().find("native_mtp_protocol"));
                ready.emplace_back("native_scalar_channel_input_width", Json::integer(mtp_input_packet(worker.ready())));
                ready.emplace_back("native_scalar_channel_output_width", Json::integer(8));
                ready.emplace_back("native_sampled_mtp", Json(true));
                for (const auto& [name, value] : worker.ready().object())
                    if (name == "native_drafter" || name.starts_with("native_mtp_") || name.starts_with("native_dflash2_")) ready.emplace_back(name, value);
                if (config.mtp_first_output_packet) ready.emplace_back("native_mtp_first_output_packet", Json::integer(*config.mtp_first_output_packet));
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
                if(config.worker.q8_three_rows.has_value())for(const auto&[name,value]:worker.ready().object())
                    if(name.starts_with("q8_three_rows_"))ready.emplace_back(name,value);
#endif
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
                if (config.worker.raw_decode.has_value()) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_raw_decode_")) ready.emplace_back(name, value);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
                if (config.worker.prefill_hc_rows.value_or(false)) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_prefill_hc_rows_")) ready.emplace_back(name, value);
                if (config.worker.prefill_router_window.value_or(false)) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_prefill_router_window_")) ready.emplace_back(name, value);
                if (config.worker.prefill_hc_down_window.value_or(false)) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_prefill_hc_down_window_")) ready.emplace_back(name, value);
                if (config.worker.qsa_sparse.value_or(false)) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_qsa_sparse_")) ready.emplace_back(name, value);
                if (config.worker.gdn_ported.value_or(false)) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_gdn_ported_")) ready.emplace_back(name, value);
                if (config.worker.prefill_input512) ready.emplace_back("native_prefill_dense", *worker.ready().find("native_prefill_dense"));
                if (const auto* kernel = worker.ready().find("native_prefill_dense_kernel")) ready.emplace_back("native_prefill_dense_kernel", *kernel);
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
                if (config.worker.prefill_hc_down_f32.value_or(false)) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_prefill_hc_down_f32_")) ready.emplace_back(name, value);
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
                if (config.worker.gdn_old_order_batch) for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("native_gdn_old_order_")) ready.emplace_back(name, value);
#endif
                if (config.worker.prefill_final_logits) {
                    ready.emplace_back("native_prefill_output_policy", *worker.ready().find("native_prefill_output_policy"));
                    ready.emplace_back("logits_available", *worker.ready().find("logits_available"));
                }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                if (config.worker.prefill_grouped) for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel"})
                    ready.emplace_back(name, *worker.ready().find(name));
                if (const auto* vn = worker.ready().find("native_verify_nax")) ready.emplace_back("native_verify_nax", *vn);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
                for (const auto& [name, value] : worker.ready().object())
                    if (name.starts_with("concurrent_")) ready.emplace_back(name, value);
#endif
            } else {
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            if (config.worker.sampled_prefill) {
                ready.emplace_back("native_scalar_channel_protocol", Json("sampled-prefill-input8-v1"));
                ready.emplace_back("native_scalar_channel_input_width", Json::integer(8));
                ready.emplace_back("native_scalar_channel_output_width", Json::integer(1));
                ready.emplace_back("native_sampled_prefill_input8", Json(true));
                for (const auto name : {"sampled_prefill_protocol", "known_token_capacity", "known_residual_reuse",
                    "known_qsa_rows_schema", "known_qsa_rows_mode", "known_qsa_rows_logical_bytes", "known_qsa_rows_charged_bytes",
                    "known_qsa_rows_allocations", "known_qsa_rows_physical_tiles", "known_qsa_rows_logical_rows", "known_qsa_rows_http_admitted"})
                    ready.emplace_back(name, *worker.ready().find(name));
            } else {
#endif
            ready.emplace_back("native_scalar_channel_protocol", Json("scalar-channel-v1"));
            ready.emplace_back("native_scalar_channel_input_width", Json::integer(1));
            ready.emplace_back("native_scalar_channel_output_width", Json::integer(1));
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            }
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            }
#endif
        }
#endif
        if (config.tool_args != ToolArgValidation::strict) {
            ready.emplace_back("tool_arg_validation", Json(std::string(tool_arg_validation_name(config.tool_args))));
            ready.emplace_back("tool_arg_validation_policy", Json(std::string(tool_arg_validation_policy(config.tool_args))));
        }
        std::cout << Json(std::move(ready)).dump() << std::endl;
        while (!stopping) {
            try {
                http::Limits limits; limits.body_bytes = 1024 * 1024;
                auto connection = listener.accept(Clock::now() + std::chrono::milliseconds(100), limits);
                const auto arrival = Clock::now();
                try {
                    const auto request = connection.read_request(arrival + std::chrono::seconds(5));
                    const auto answer = [&](int code, Json body) { connection.send_json(code, body.dump(8192), Clock::now() + std::chrono::seconds(2)); };
                    if (!equal_secret(request.header("authorization"), authorization)) { answer(401, error_json("invalid API credentials", "authentication_error")); continue; }
                    if (!request.header("origin").empty()) { answer(403, error_json("cross-origin browser access is not enabled; use the trusted local client proxy", "origin_refused")); continue; }
                    if (request.method == "GET" && request.path == "/health") {
                        answer(available.load() ? 200 : 503, Json(Json::Object{{"status", Json(available.load() ? "ready" : "worker_failed")}, {"model", Json(config.alias)}})); continue;
                    }
                    if (request.method == "GET" && request.path == "/v1/models") {
                        answer(200, Json(Json::Object{{"object", Json("list")}, {"data", Json(Json::Array{Json(Json::Object{{"id", Json(config.alias)}, {"object", Json("model")}, {"created", Json::integer(0)}, {"owned_by", Json("tokspur")}})})}})); continue;
                    }
                    if (request.method != "POST" || request.path != "/v1/chat/completions") { answer(404, error_json("unknown endpoint", "not_found")); continue; }
                    const auto content_type = request.header("content-type");
                    if (!json_content_type(content_type)) { answer(415, error_json("application/json required", "unsupported_media_type")); continue; }
                    if (!available.load()) { answer(503, error_json("owned worker is unavailable; no automatic restart or generation retry", "worker_unavailable")); continue; }
                    auto input = parse_json(request.body, {1024 * 1024, 150000, 32});
                    static_cast<void>(admit_chat(input, config.alias, config.tool_enums, config.tool_args)); // No mutation on invalid admission.
                    std::lock_guard lock(mutex);
                    if (pending) { answer(429, error_json("one waiting request already admitted", "queue_full")); continue; }
                    pending.emplace(Job{std::move(connection), std::move(input), arrival + std::chrono::seconds(config.request_seconds)}); condition.notify_one();
                } catch (const http::Error& error) {
                    if (!error.disconnected()) try { connection.send_json(error.status() ? error.status() : 400, error_json("invalid or timed-out HTTP request", "http_error").dump(), Clock::now() + std::chrono::seconds(1)); } catch (...) {}
                } catch (const std::exception&) {
                    try { connection.send_json(400, error_json("invalid chat request", "invalid_request_error").dump(), Clock::now() + std::chrono::seconds(1)); } catch (...) {}
                }
            } catch (const http::Error& error) { if (error.status() != 408) throw; }
        }
    } catch (...) { failure = std::current_exception(); }
    shutdown.store(true); condition.notify_one(); generation.join();
    worker.close(); // A clean exit is not reported unless the entire owned group drains.
    if (failure) std::rethrow_exception(failure);
    state.release_lock();
    std::cout << Json(Json::Object{{"event", Json("end")}, {"owned_worker_exit_verified", Json(true)}, {"tools_executed", Json(false)}}).dump() << std::endl;
    return 0;
}
} // namespace tokspur
