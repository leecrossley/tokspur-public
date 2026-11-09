#include "tokspur/owned_worker.hpp"
#include "tokspur/raw_decode_wire.hpp"
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#include "tokspur/mtp_wire.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#include "tokspur/serve_verify8.hpp"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>

#include <fcntl.h>
#ifdef __APPLE__
#include <libproc.h>
#endif
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tokspur {
namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t line_limit = 1024 * 1024, stderr_limit = 64 * 1024;
constexpr int poll_slice_ms = 25;
enum : int { armed_notice = 1, launched_notice = 2, exit_notice = 3, failure_notice = 4 };
struct Notice { int kind, value; };
struct Channels { int control[2]{-1,-1}, status[2]{-1,-1}, input[2]{-1,-1}, output[2]{-1,-1}, error[2]{-1,-1}, gate[2]{-1,-1}; };
void require(bool value, const char* message) { if (!value) throw WorkerError(message); }
const Json& worker_field(const Json& value, std::string_view name) {
    const auto* field = value.find(name); require(field, "missing owned broker field"); return *field;
}
uint64_t worker_integer(const Json& value, std::string_view name) { return worker_field(value, name).unsigned_integer(); }
double worker_number(const Json& value) {
    const auto text = value.number_lexeme(); double number = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), number);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(number), "invalid owned broker numeric field");
    return number;
}
void worker_keys(const Json& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& [name, ignored] : value.object()) {
        static_cast<void>(ignored);
        require(std::find(allowed.begin(), allowed.end(), name) != allowed.end(), "unsupported owned broker command field");
    }
}
void drop(int& fd) noexcept { if (fd >= 0) { close(fd); fd = -1; } }
void drop_channels(Channels& c) noexcept {
    for (auto* pair : {c.control, c.status, c.input, c.output, c.error, c.gate}) { drop(pair[0]); drop(pair[1]); }
}
void descriptor(int& fd) {
    if (fd < 3) { const int moved = fcntl(fd, F_DUPFD_CLOEXEC, 3); drop(fd); fd = moved; }
    require(fd >= 3 && fcntl(fd, F_SETFD, FD_CLOEXEC) == 0, "cannot secure worker pipe descriptor");
}
void pipe_pair(int (&fds)[2]) {
    require(pipe(fds) == 0, "cannot create worker pipe"); descriptor(fds[0]); descriptor(fds[1]);
}
void socket_pair(int (&fds)[2]) {
    require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "cannot create worker control stream");
    descriptor(fds[0]); descriptor(fds[1]);
#ifdef SO_NOSIGPIPE
    const int yes = 1;
    require(setsockopt(fds[0], SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) == 0 &&
        setsockopt(fds[1], SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) == 0, "cannot suppress local control-stream SIGPIPE");
#endif
}
ssize_t send_bytes(int fd, const void* data, size_t size) {
#ifdef MSG_NOSIGNAL
    return send(fd, data, size, MSG_NOSIGNAL);
#else
    return send(fd, data, size, 0);
#endif
}
void nonblocking(int fd) { const auto flags = fcntl(fd, F_GETFL); require(flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0, "cannot configure bounded worker pipe"); }

// Everything below in the forked guardian/exec child uses precomputed data and
// fixed-size storage, no C++ allocation, locks, streams, Foundation or callbacks.
// Thus a multithreaded HTTP parent does not run heap/Objective-C work after fork.
bool child_notice(int fd, int kind, int value) noexcept {
    const Notice n{kind, value}; const char* bytes = reinterpret_cast<const char*>(&n); size_t sent = 0;
    while (sent < sizeof(n)) { const auto count = write(fd, bytes + sent, sizeof(n) - sent); if (count > 0) sent += size_t(count); else if (count < 0 && errno == EINTR) continue; else return false; }
    return true;
}
bool child_byte(int fd, char expected) noexcept {
    char byte = 0; ssize_t n; do { n = read(fd, &byte, 1); } while (n < 0 && errno == EINTR);
    return n == 1 && byte == expected;
}
void child_close_except(int ceiling, const int* kept, size_t count) noexcept {
#ifdef __APPLE__
    static_cast<void>(ceiling);
    proc_fdinfo descriptors[256];
    for (;;) {
        // Apple's proc_pidinfo wrapper makes one __proc_info call without allocation or locks.
        const int bytes = proc_pidinfo(getpid(), PROC_PIDLISTFDS, 0, descriptors, sizeof(descriptors));
        if (bytes <= 0 || size_t(bytes) > sizeof(descriptors) || size_t(bytes) % sizeof(proc_fdinfo)) _exit(119);
        const size_t entries = size_t(bytes) / sizeof(proc_fdinfo); size_t closed = 0;
        for (size_t n = 0; n < entries; ++n) {
            if (descriptors[n].proc_fd < 0) _exit(119);
            bool keep = false; for (size_t i = 0; i < count; ++i) if (descriptors[n].proc_fd == kept[i]) keep = true;
            if (!keep) { close(descriptors[n].proc_fd); ++closed; }
        }
        if (entries < std::size(descriptors)) return;
        if (!closed) _exit(119);
    }
#else
    for (int fd = 0; fd < ceiling; ++fd) {
        bool keep = false; for (size_t i = 0; i < count; ++i) if (fd == kept[i]) keep = true;
        if (!keep) close(fd);
    }
#endif
}

bool exited_without_reaping(pid_t pid) noexcept {
    siginfo_t info{};
    return waitid(P_PID, id_t(pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == pid;
}
int reap_child(pid_t pid) noexcept {
    int status = 0; pid_t result;
    do { result = waitpid(pid, &status, 0); } while (result < 0 && errno == EINTR);
    return result == pid ? status : -1;
}
[[noreturn]] void guardian(Channels c, int ceiling, pid_t parent, char* const* argv, char* const* environment) noexcept {
    struct sigaction action{}; sigemptyset(&action.sa_mask); action.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &action, nullptr); action.sa_handler = SIG_DFL; sigaction(SIGCHLD, &action, nullptr);
    const int keep[]{c.control[1], c.status[1], c.input[1], c.output[1], c.error[1], c.gate[0], c.gate[1]};
    child_close_except(ceiling, keep, sizeof(keep) / sizeof(*keep));
    if (getppid() != parent || setsid() < 0 || !child_notice(c.status[1], armed_notice, int(getpid())) || !child_byte(c.control[1], 'L')) _exit(120);
    const pid_t worker = fork();
    if (worker < 0) { child_notice(c.status[1], failure_notice, errno); _exit(121); }
    if (!worker) {
        if (dup2(c.input[1], STDIN_FILENO) < 0 || dup2(c.output[1], STDOUT_FILENO) < 0 || dup2(c.error[1], STDERR_FILENO) < 0) _exit(122);
        const int child_keep[]{STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO, c.gate[0]};
        child_close_except(ceiling, child_keep, sizeof(child_keep) / sizeof(*child_keep));
        if (!child_byte(c.gate[0], 'E')) _exit(123);
        close(c.gate[0]); action.sa_handler = SIG_DFL; sigaction(SIGPIPE, &action, nullptr);
        execve(argv[0], argv, environment); _exit(127);
    }
    close(c.input[1]); close(c.output[1]); close(c.error[1]); close(c.gate[0]);
    bool stopping = setpgid(worker, worker) != 0 || getpgid(worker) != worker;
    if (!stopping) stopping = !child_notice(c.status[1], launched_notice, int(worker)) || !child_byte(c.control[1], 'E');
    if (!stopping) { const char permit = 'E'; stopping = write(c.gate[1], &permit, 1) != 1; }
    close(c.gate[1]);
    // Parent death is EOF on its sole lifetime endpoint. Target exec cannot
    // happen before ARMED and LAUNCHED were observed and acknowledged above.
    // Keep the worker PID unreaped until its group is killed: no PID/group-ID
    // reuse window can redirect descendant cleanup to an unrelated process.
    unsigned stop_ticks = 0;
    for (;;) {
        if (exited_without_reaping(worker)) {
            kill(-worker, SIGKILL); const int status = reap_child(worker);
            child_notice(c.status[1], exit_notice, status); _exit(status < 0 ? 124 : 0);
        }
        if (stopping) {
            if (!stop_ticks) { kill(-worker, SIGTERM); kill(worker, SIGTERM); }
            if (++stop_ticks >= 10) { kill(-worker, SIGKILL); kill(worker, SIGKILL); }
        }
        struct pollfd watch{stopping ? -1 : c.control[1], POLLIN, 0};
        const int result = poll(&watch, 1, poll_slice_ms);
        if (result < 0 && errno != EINTR) stopping = true;
        if (result > 0 && watch.revents) { char ignored; const auto n = read(c.control[1], &ignored, 1); if (n != -1 || errno != EINTR) stopping = true; }
    }
}
std::filesystem::path fixed_file(const std::filesystem::path& path) {
    require(path.is_absolute() && path.string().find('\0') == std::string::npos, "worker assets require absolute fixed paths");
    const auto result = std::filesystem::canonical(path); struct stat state{};
    require(stat(result.c_str(), &state) == 0 && S_ISREG(state.st_mode), "worker asset must be a regular file");
    return result;
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
std::filesystem::path fixed_dir(const std::filesystem::path& path) {
    require(path.is_absolute() && path.string().find('\0') == std::string::npos, "worker assets require absolute fixed paths");
    const auto result = std::filesystem::canonical(path); struct stat state{};
    require(stat(result.c_str(), &state) == 0 && S_ISDIR(state.st_mode), "worker asset must be a directory");
    return result;
}
#endif
bool event_is(const Json& value, std::string_view name) {
    const auto* event = value.find("event"); return event && event->kind() == Json::Kind::string && event->string() == name;
}
} // namespace

#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
std::optional<uint32_t> owned_lane_auth_workers_requested(const Json& value) {
    if (!value.find(owned_lane_auth_fields[0])) return std::nullopt;
    const auto workers = worker_integer(value, owned_lane_auth_fields[0]);
    require(workers && workers <= 16, "invalid owned lane authentication request");
    return uint32_t(workers);
}
void validate_owned_weight_lanes_ready(const Json& value, bool selected, std::optional<uint32_t> workers) {
    size_t count = 0, authentication = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("weight_lanes_")) {
        static_cast<void>(ignored);
        require(std::find(owned_weight_lane_fields.begin(), owned_weight_lane_fields.end(), name) != owned_weight_lane_fields.end(),
            "unknown owned lane-layout metadata");
        ++count;
    }
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("lane_auth_")) {
        static_cast<void>(ignored);
        require(std::find(owned_lane_auth_fields.begin(), owned_lane_auth_fields.end(), name) != owned_lane_auth_fields.end(),
            "unknown owned lane authentication metadata");
        ++authentication;
    }
    require(count == (selected ? owned_weight_lane_fields.size() : 0), "owned lane-layout presence differs from startup consent");
    require(!workers || (selected && *workers && *workers <= 16), "lane authentication requires selected sidecar and bounded workers");
    require(authentication == (workers ? owned_lane_auth_fields.size() : 0), "owned lane authentication presence differs from startup consent");
    if (!selected) return;
    const auto digest = worker_field(value, "weight_lanes_manifest_sha256").string();
    require(digest.size() == 64 && digest.find_first_not_of("0123456789abcdef") == std::string_view::npos,
        "invalid owned lane-layout manifest digest");
    const auto matrices = worker_integer(value, "weight_lanes_tensor_count");
    const auto file_bytes = worker_integer(value, "weight_lanes_file_bytes");
    const auto payload_bytes = worker_integer(value, "weight_lanes_original_payload_bytes_skipped");
    require(worker_integer(value, "weight_lanes_schema") == 1 && matrices && matrices <= 1024 &&
        payload_bytes && payload_bytes < file_bytes && file_bytes <= 32'000'000'000,
        "invalid owned lane-layout schema or byte geometry");
    if (workers) require(owned_lane_auth_workers_requested(value) == workers &&
        worker_integer(value, owned_lane_auth_fields[1]) == std::min<uint64_t>(*workers, matrices),
        "owned lane authentication worker count differs");
}
void validate_owned_weight_lanes_done(const Json& ready, const Json& done) {
    const bool selected = ready.find("weight_lanes_schema");
    const auto workers = owned_lane_auth_workers_requested(ready);
    validate_owned_weight_lanes_ready(ready, selected, workers);
    validate_owned_weight_lanes_ready(done, selected, workers);
    if (selected) for (const auto name : owned_weight_lane_fields)
        require(worker_field(done, name).dump() == worker_field(ready, name).dump(), "owned lane-layout immutable identity drift");
    if (workers) for (const auto name : owned_lane_auth_fields)
        require(worker_field(done, name).dump() == worker_field(ready, name).dump(), "owned lane authentication immutable identity drift");
}
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
bool owned_concurrent_transport(const OwnedWorkerOptions& options) {
    const bool mtp2_ok = !options.mtp_sidecar.empty() && options.dflash2_dir.empty() &&
        (options.mtp_depth == std::optional<uint32_t>(2) || options.mtp_depth == std::optional<uint32_t>(3)) && options.mtp_request_carry &&
        options.mtp_draft_attention == std::optional<std::string>("tiled128");
    const bool dflash2_ok = !options.dflash2_dir.empty() && options.mtp_sidecar.empty() &&
        options.mtp_depth == std::optional<uint32_t>(7) && options.mtp_request_carry &&
        options.mtp_draft_attention == std::optional<std::string>("split16");
    return scalar_channel_transport(options) && !options.sampled_prefill && (mtp2_ok || dflash2_ok);
}
namespace {
void concurrent_metadata(const Json& value, std::optional<bool> selected, bool startup) {
    constexpr std::array<std::string_view, 4> names{"concurrent_dispatch_schema", "concurrent_dispatch_mode", "concurrent_dispatch_barriers", "concurrent_state_proof"};
    size_t count = 0;
    for (const auto& [name, ignored] : value.object()) if (name.starts_with("concurrent_")) {
        static_cast<void>(ignored);
        require(std::find(names.begin(), names.end(), name) != names.end(), "unknown concurrent HTTP metadata"); ++count;
    }
    require(count == (selected.has_value() ? names.size() : 0), "concurrent HTTP metadata differs from startup consent");
    if (!selected.has_value()) return;
    require(worker_integer(value, names[0]) == 1 && worker_field(value, names[1]).string() == (*selected ? "on" : "off") &&
        !worker_field(value, names[3]).boolean(), "concurrent HTTP mode/schema/proof differs");
    const auto barriers = worker_integer(value, names[2]);
    require(!startup || !barriers, "concurrent HTTP READY already executed a batch");
}
}
void validate_owned_concurrent_ready(const Json& value, std::optional<bool> selected) { concurrent_metadata(value, selected, true); }
void validate_owned_concurrent_done(const Json& ready, const Json& done) {
    const auto* mode = ready.find("concurrent_dispatch_mode");
    const std::optional<bool> selected = mode ? std::optional<bool>(mode->string() == "on") : std::nullopt;
    concurrent_metadata(ready, selected, true); concurrent_metadata(done, selected, false);
}
#endif

#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
namespace {
constexpr std::array<std::string_view,5> q8_route_names{"raw","packed","hc_raw","wide_qkv_raw","wide_output_raw"};
void q8_three_metadata(const Json& value,std::optional<bool> selected,bool startup) {
    constexpr std::array<std::string_view,6> names{"q8_three_rows_schema","q8_three_rows_mode","q8_three_rows_routes","q8_three_rows_padding_rows","q8_three_rows_hc_route","q8_three_rows_request_routes"};
    size_t count=0;for(const auto&[name,ignored]:value.object())if(name.starts_with("q8_three_rows_")){static_cast<void>(ignored);require(std::find(names.begin(),names.end(),name)!=names.end(),"unknown three-row Q8 metadata");++count;}
    if(!count){require(!selected.has_value(),"selected three-row Q8 metadata missing");return;}
    require(count==(startup?5u:6u),"incomplete three-row Q8 metadata");
    require(worker_integer(value,names[0])==1 && worker_field(value,names[1]).string()==(selected.value_or(false)?"on":"off") &&
        worker_integer(value,names[3])==0 && worker_field(value,names[4]).string()=="generic-exact-fp32","three-row Q8 selected identity differs");
    const auto& routes=worker_field(value,names[2]);require(routes.object().size()==q8_route_names.size(),"three-row Q8 route shape");
    for(const auto name:q8_route_names){const auto n=worker_integer(routes,name);require(!startup||!n,"three-row Q8 READY already executed work");require(selected.value_or(false)||!n,"off three-row Q8 reported executed work");}
    if(!startup){const auto& delta=worker_field(value,names[5]);require(delta.object().size()==q8_route_names.size(),"three-row Q8 request route shape");for(const auto name:q8_route_names)static_cast<void>(worker_integer(delta,name));}
}
}
void validate_owned_q8_three_ready(const Json& value,std::optional<bool> selected){q8_three_metadata(value,selected,true);}
void validate_owned_q8_three_previous(const Json& ready,const Json& previous){
    const auto* mode=ready.find("q8_three_rows_mode");const std::optional<bool> selected=mode?std::optional<bool>(mode->string()=="on"):std::nullopt;
    q8_three_metadata(ready,selected,true);
    const auto event=worker_field(previous,"event").string();require(event=="ready"||event=="done","invalid previous N3 ancestry event");
    q8_three_metadata(previous,selected,event=="ready");
    require(bool(mode)==bool(previous.find("q8_three_rows_mode")),"previous N3 metadata appeared or disappeared");
    if(mode)for(const auto name:{"q8_three_rows_schema","q8_three_rows_mode","q8_three_rows_padding_rows","q8_three_rows_hc_route"})
        require(worker_field(previous,name).dump()==worker_field(ready,name).dump(),"previous N3 immutable identity drifted");
}
void validate_owned_q8_three_done(const Json& ready,const Json& previous,const Json& done,bool cold_resync){
    validate_owned_q8_three_previous(ready,previous);
    const auto* mode=ready.find("q8_three_rows_mode");const std::optional<bool> selected=mode?std::optional<bool>(mode->string()=="on"):std::nullopt;
    q8_three_metadata(ready,selected,true);q8_three_metadata(done,selected,false);
    require(bool(mode)==bool(done.find("q8_three_rows_mode")),"three-row Q8 metadata appeared or disappeared");
    if(mode)for(const auto name:q8_route_names){const auto before=worker_integer(worker_field(previous,"q8_three_rows_routes"),name),after=worker_integer(worker_field(done,"q8_three_rows_routes"),name),delta=worker_integer(worker_field(done,"q8_three_rows_request_routes"),name);
        require(after>=before && (cold_resync?delta<=after-before:delta==after-before),"three-row Q8 routes rewound or per-packet delta differs from admitted ancestry");}
}
#endif

struct OwnedWorker::Impl {
    OwnedWorkerOptions options;
    Channels channels;
    pid_t guardian_pid = -1, worker_pid = -1;
    bool armed = false, worker_exited = false, guardian_reaped = false, out_eof = false, err_eof = false, status_eof = false;
    bool failed = false, closed = false, needs_reset = false, starting = true;
    int worker_status = -1, guardian_status = -1;
    size_t stderr_bytes = 0;
    std::string pending, notice_bytes;
    Json ready_value;
    uint64_t broker_epoch = 0, broker_command = 0, broker_lease = 0;
    uint32_t broker_position = 0;
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
    Json scalar_previous;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    Json verify_previous, verify_sampling;
    uint64_t verify_seed = 0;
    uint32_t verify_position = 0;
#endif
    mutable std::mutex mutex;
    explicit Impl(OwnedWorkerOptions value) : options(std::move(value)) {}
    ~Impl() { try { shutdown(); } catch (...) {} drop_channels(channels); }
    void broker_storage(const Json& value) const {
        require(worker_integer(value, "checkpoint_slots") == options.checkpoint_slots &&
            worker_integer(value, "checkpoint_combined_budget_bytes") == options.checkpoint_budget_bytes,
            "owned broker checkpoint configuration changed");
        uint64_t total = 0;
        for (const auto name : {"checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"}) {
            const auto bytes = worker_integer(value, name);
            require(bytes && bytes <= options.checkpoint_budget_bytes - total, "owned broker storage exceeds budget"); total += bytes;
            if (broker_epoch) require(bytes == worker_integer(ready_value, name), "owned broker prepared storage changed");
        }
    }
    void broker_identity(const Json& value, uint64_t command, uint64_t lease) const {
        require(worker_integer(value, "broker_epoch") == broker_epoch && worker_integer(value, "command_id") == command &&
            worker_integer(value, "lease_id") == lease, "stale owned broker reply identity");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (options.branch8) require(worker_field(value, "broker_protocol").string() == "branch8-v1" &&
            worker_integer(value, "broker_batch_width") == 8, "owned branch8 wire identity changed");
#endif
    }
    void start() {
        require(options.stack_execution == "direct" || options.stack_execution == "recurrent-replay", "unsupported owned startup execution mode");
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
        require(!options.raw_decode.has_value() || owned_concurrent_transport(options), "raw routed decode HTTP requires the current fixed MTP transport");
#else
        require(!options.raw_decode.has_value(), "raw routed decode HTTP requires the current MTP transport build");
#endif
#endif
        require(options.checkpoint_slots <= 4 && options.checkpoint_budget_bytes <= (512ull << 20) &&
            bool(options.checkpoint_slots) == bool(options.checkpoint_budget_bytes) &&
            (!options.checkpoint_slots || options.stack_execution == "direct"), "owned broker requires paired bounded direct preparation");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        require(options.weight_lanes.empty() || owned_weight_lanes_transport(options),
            "lane-layout HTTP requires explicit corrected scalar-channel startup consent");
        require(!options.weight_lanes_auth_workers || (!options.weight_lanes.empty() &&
            *options.weight_lanes_auth_workers && *options.weight_lanes_auth_workers <= 16),
            "lane authentication requires explicit sidecar and worker count 1..16");
        if (!options.weight_lanes.empty()) options.weight_lanes = fixed_file(options.weight_lanes);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
        require(!options.concurrent_dispatch.has_value() || owned_concurrent_transport(options),
            "concurrent HTTP requires explicit fixed depth-two carry/tiled128 scalar transport");
#endif
#if defined(TOKSPUR_PRIVATE_Q8_THREE_ROWS) && defined(TOKSPUR_PRIVATE_NATIVE_MTP)
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
        require(!options.q8_three_rows.has_value() || owned_concurrent_transport(options), "three-row Q8 HTTP requires the current fixed MTP transport");
#else
        require(!options.q8_three_rows.has_value(), "three-row Q8 HTTP requires the current MTP transport build");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        require(!options.prefill_input512 || owned_concurrent_transport(options), "input512 requires explicit fixed depth-two MTP scalar transport");
        require(!options.prefill_final_logits || options.prefill_input512, "final-input logits require explicit input512");
        const bool dense_staged_or_nax = (options.prefill_dense == "f32-staged" || options.prefill_dense == "nax");
        const bool grouped_ok = (options.prefill_grouped == "f32" || options.prefill_grouped == "auto" || options.prefill_grouped == "nax");
#ifdef TOKSPUR_PRIVATE_HTTP_HC_ROWS
    require(!options.prefill_hc_rows.has_value() || owned_concurrent_transport(options), "HC rows require fixed MTP transport");
    require(!options.prefill_hc_rows.value_or(false) || (options.prefill_input512 && dense_staged_or_nax && grouped_ok && !options.prefill_hc_down_f32.value_or(false)), "HC rows require staged FP32 or NAX input512 and scalar HC down");
#else
    require(!options.prefill_hc_rows.has_value(), "HC rows selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_ROUTER_WINDOW
    require(!options.prefill_router_window.has_value() || owned_concurrent_transport(options), "router window requires fixed MTP transport");
    require(!options.prefill_router_window.value_or(false) || (options.prefill_input512 && dense_staged_or_nax && grouped_ok && !options.prefill_hc_down_f32.value_or(false)), "router window requires staged FP32 or NAX input512 and scalar HC down");
#else
    require(!options.prefill_router_window.has_value(), "router window selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_HC_DOWN_WINDOW
    require(!options.prefill_hc_down_window.has_value() || owned_concurrent_transport(options), "HC down window requires fixed MTP transport");
    require(!options.prefill_hc_down_window.value_or(false) || (options.prefill_input512 && dense_staged_or_nax && grouped_ok && options.prefill_hc_rows.value_or(false) && options.prefill_router_window.value_or(false) && !options.prefill_hc_down_f32.value_or(false)), "HC down window requires staged FP32 or NAX input512, exact HC rows, exact router window and reassociating HC down off");
#else
    require(!options.prefill_hc_down_window.has_value(), "HC down window selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_SPARSE
    require(!options.qsa_sparse.has_value() || owned_concurrent_transport(options), "sparse QSA requires fixed MTP transport");
    require(!options.qsa_sparse.value_or(false) || (options.prefill_input512 && dense_staged_or_nax && grouped_ok && options.prefill_hc_rows.value_or(false) && options.prefill_router_window.value_or(false) && options.prefill_hc_down_window.value_or(false) && !options.prefill_hc_down_f32.value_or(false)), "sparse QSA requires staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
#else
    require(!options.qsa_sparse.has_value(), "sparse QSA selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_GDN_PORTED
    require(!options.gdn_ported.has_value() || owned_concurrent_transport(options), "ported GDN requires fixed MTP transport");
    require(!options.gdn_ported.value_or(false) || (options.prefill_input512 && dense_staged_or_nax && grouped_ok && options.prefill_hc_rows.value_or(false) && options.prefill_router_window.value_or(false) && options.prefill_hc_down_window.value_or(false) && !options.prefill_hc_down_f32.value_or(false)), "ported GDN requires staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off");
#else
    require(!options.gdn_ported.has_value(), "ported GDN selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        require(!options.prefill_hc_down_f32.has_value() || owned_concurrent_transport(options), "HC down batching selection requires the fixed MTP transport");
        require(!options.prefill_hc_down_f32.value_or(false) || (options.prefill_input512 && options.prefill_dense == "f32" && options.prefill_grouped == "f32"), "HC down batching requires explicit f32/f32 input512");
#else
        require(!options.prefill_hc_down_f32.has_value(), "HC down batching selection is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        require(!options.gdn_old_order_batch || (options.prefill_input512 && (options.prefill_dense == "f32" || options.prefill_dense == "f32-staged") && options.prefill_grouped == "f32"),
            "old-order GDN requires explicit f32/f32 input512");
#endif
        require(!options.prefill_dense || (options.prefill_input512 && (*options.prefill_dense == "f16" || *options.prefill_dense == "f32" || dense_staged_or_nax)),
            "native prefill dense requires input512 and f16, f32 or nax");
        if (dense_staged_or_nax) {
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
            require(options.prefill_input512 && grouped_ok && !options.prefill_hc_down_f32.value_or(false),
                "staged/NAX dense backend requires input512 compatible grouped mode and HC down off");
#else
            throw FormatError("staged dense backend is not compiled");
#endif
        }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        require(!options.prefill_grouped || (options.prefill_input512 && (*options.prefill_grouped == "serial" || grouped_ok)),
            "native prefill grouped requires input512 and serial, f32, auto or nax");
        require(options.prefill_grouped != "nax" || (dense_staged_or_nax && !options.prefill_hc_down_f32.value_or(false)),
            "NAX grouped prefill requires staged/NAX dense and HC down off");
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        require(!options.prefill_input512 || options.weight_lanes.empty(), "input512 dense projections require original weight views");
#endif
#endif
        options.executable = fixed_file(options.executable);
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
        const bool scalar_transport = scalar_channel_transport(options);
        require(!options.scalar_channel || scalar_transport,
            "scalar-channel worker requires fixed direct channel/Stage-ON policy without V8/branch/history");
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        require(!options.sampled_prefill || scalar_transport,
            "sampled input8 requires its explicit sampled scalar channel policy");
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        require(options.mtp_sidecar.empty() || options.dflash2_dir.empty(), "cannot combine MTP sidecar and DFlash2 drafter");
        require(!options.drafter || *options.drafter == "mtp" || *options.drafter == "dflash2", "drafter must be mtp or dflash2");
        require(options.drafter != std::optional<std::string>("dflash2") || !options.dflash2_dir.empty(), "dflash2 drafter requires --dflash2-dir");
        require(options.drafter != std::optional<std::string>("mtp") || options.dflash2_dir.empty(), "mtp drafter refuses --dflash2-dir");
        require(!options.dflash2_precision || (!options.dflash2_dir.empty() && (*options.dflash2_precision == "q8_0" || *options.dflash2_precision == "bf16")),
            "dflash2 precision requires --dflash2-dir and q8_0 or bf16");
        const bool has_spec = !options.mtp_sidecar.empty() || !options.dflash2_dir.empty();
        require(!has_spec || (scalar_transport && !options.sampled_prefill),
            "native MTP requires its distinct fixed scalar-channel startup consent");
        require(!options.mtp_draft_attention || (has_spec &&
            (!options.dflash2_dir.empty() ? *options.mtp_draft_attention == "split16" :
            (*options.mtp_draft_attention == "serial" || *options.mtp_draft_attention == "tiled128" || *options.mtp_draft_attention == "tiled256"))),
            "MTP draft-attention requires explicit MTP and a supported mode");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        require(!options.mtp_depth || (!options.dflash2_dir.empty() ? (*options.mtp_depth == 7 && options.mtp_request_carry &&
            options.mtp_draft_attention == std::optional<std::string>("split16")) :
            (*options.mtp_depth == 1 || ((*options.mtp_depth == 2 || *options.mtp_depth == 3) && options.mtp_request_carry &&
            options.mtp_draft_attention == std::optional<std::string>("tiled128")))), "depth-two/three/seven HTTP requires matching scoped carry/attention");
        require(!options.mtp_depth || has_spec, "explicit MTP depth requires a sidecar or DFlash2 directory");
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        require(!options.mtp_output_packet || ((*options.mtp_output_packet == 8 || *options.mtp_output_packet == 64) && owned_concurrent_transport(options)),
            "MTP output-packet selection requires explicit fixed depth-two/seven scoped transport");
#endif
#endif
        require(!options.mtp_request_carry || has_spec, "MTP generation lifetime requires a sidecar or DFlash2 directory");
        require(!options.mtp_fast_round.value_or(false) || !options.mtp_sidecar.empty(), "native MTP fast round requires --mtp-sidecar");
        if (!options.mtp_sidecar.empty()) options.mtp_sidecar = fixed_file(options.mtp_sidecar);
        if (!options.dflash2_dir.empty()) options.dflash2_dir = fixed_dir(options.dflash2_dir);
#endif
#else
        constexpr bool scalar_transport = false;
#endif
        const bool branch_transport =
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            options.branch8 && options.verify8 && options.checkpoint_slots && options.checkpoint_budget_bytes;
        require(!options.branch8 || (branch_transport && options.stack_execution == "direct"), "branch8 worker requires explicit V8/history/direct consent");
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        require((!options.stage8_enabled || options.stage8_present) && (!options.stage8_present || ((branch_transport || scalar_transport) && options.stack_execution == "direct")),
            "staged HTTP worker requires selector presence and explicit branch8/V8/direct/history consent");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        require(!options.fast_channel.has_value() || ((branch_transport || scalar_transport) && options.stage8_present && options.stage8_enabled),
            "fast HTTP math requires explicit staged-ON branch8/V8/history consent");
#endif
#else
            false;
#endif
        require(!options.verify8 || (options.stack_execution == "direct" && (branch_transport || (!options.checkpoint_slots && !options.checkpoint_budget_bytes))),
            "private HTTP V8 requires fixed direct execution without checkpoints");
        require(options.executable.filename() == (scalar_transport ? "scalar-channel-target" : branch_transport ? "branch8-target" : options.verify8 ? "verify8-target" : "tokspur-native") && access(options.executable.c_str(), X_OK) == 0,
            "owned worker executable does not match the explicit private startup mode");
#else
        require(options.executable.filename() == "tokspur-native" && access(options.executable.c_str(), X_OK) == 0, "only an explicit owned tokspur-native executable is admitted");
#endif
        options.policy = fixed_file(options.policy); options.library = fixed_file(options.library);
        require(options.shards.size() == 4, "owned target requires four fixed shards");
        std::set<std::filesystem::path> distinct;
        for (auto& shard : options.shards) { shard = fixed_file(shard); require(distinct.emplace(shard).second, "owned target shard paths must be distinct"); }
        for (const auto timeout : {options.startup_timeout, options.step_timeout, options.drain_timeout})
            require(timeout.count() > 0 && timeout <= std::chrono::hours(24), "invalid worker deadline bound");
        struct rlimit limit{};
        require(getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur <= 1'048'576, "cannot bound inherited worker descriptors");
        std::vector<std::string> words{options.executable.string(), "target", "--policy", options.policy.string(), "--library", options.library.string(),
            "--allow-full-model", "true", "--small-gemv-splits", "16", "--block-gemv-formats", "3", "--stack-execution", options.stack_execution};
        if (!options.derived_pack.empty()) words.insert(words.end(), {"--derived-pack", options.derived_pack.string()});
        if (options.derived_pack_auth_workers) words.insert(words.end(), {"--derived-pack-auth-workers", std::to_string(*options.derived_pack_auth_workers)});
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        if (!options.weight_lanes.empty()) words.insert(words.end(), {"--weight-lanes", options.weight_lanes.string()});
        if (options.weight_lanes_auth_workers) words.insert(words.end(), {"--weight-lanes-auth-workers", std::to_string(*options.weight_lanes_auth_workers)});
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
        if (options.scalar_channel) words.insert(words.end(), {"--hc-exact", "combined", "--wide-q8-exact", "off", "--wide-q8-many", "off"});
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        if (options.sampled_prefill) words.insert(words.end(), {"--sampled-prefill-input8", "on",
            "--known-token-capacity", "8", "--reuse-residual-projections", "true", "--known-qsa-rows", "q8"});
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (!options.mtp_sidecar.empty() || !options.dflash2_dir.empty()) {
            const auto cap_str = std::to_string(options.known_token_capacity.value_or(options.prefill_input512 ? 512 : 8));
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            if (options.prefill_input512) words.insert(words.end(), {"--known-token-capacity", cap_str, "--reuse-residual-projections", "true",
                "--native-prefill-chunks", "fused", "--native-prefill-dense", options.prefill_dense.value_or("f16")});
            else
#endif
            words.insert(words.end(), {"--known-token-capacity", cap_str, "--reuse-residual-projections", "true", "--known-qsa-rows", "q8"});
            if (!options.dflash2_dir.empty()) {
                words.insert(words.end(), {"--drafter", "dflash2", "--dflash2-dir", options.dflash2_dir.string(), "--mtp-lookahead-anchor", "on"});
                if (options.dflash2_precision) words.insert(words.end(), {"--dflash2-precision", *options.dflash2_precision});
                if (options.dflash2_max_drafts) words.insert(words.end(), {"--dflash2-max-drafts", std::to_string(*options.dflash2_max_drafts)});
                if (options.dflash2_vocab_slice) words.insert(words.end(), {"--dflash2-vocab-slice", std::to_string(*options.dflash2_vocab_slice)});
            } else {
                words.insert(words.end(), {"--mtp-sidecar", options.mtp_sidecar.string(), "--mtp-lookahead-anchor", "on"});
            }
            if (options.logit_topk.has_value()) words.insert(words.end(), {"--native-logit-topk", *options.logit_topk ? "on" : "off"});
            if (options.ple_async_gather.has_value()) words.insert(words.end(), {"--native-ple-async-gather", *options.ple_async_gather ? "on" : "off"});
            if (options.mtp_fast_round.has_value()) words.insert(words.end(), {"--native-mtp-fast-round", *options.mtp_fast_round ? "on" : "off"});
        }
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (options.raw_decode.has_value()) words.insert(words.end(), {"--raw-decode", *options.raw_decode ? "on" : "off"});
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        if (options.prefill_final_logits) words.insert(words.end(), {"--native-prefill-final-logits", "on"});
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
        if (options.prefill_hc_rows.has_value()) words.insert(words.end(), {"--native-prefill-hc-rows", *options.prefill_hc_rows ? "on" : "off"});
        if (options.prefill_router_window.has_value()) words.insert(words.end(), {"--native-prefill-router-window", *options.prefill_router_window ? "on" : "off"});
        if (options.prefill_hc_down_window.has_value()) words.insert(words.end(), {"--native-prefill-hc-down-window", *options.prefill_hc_down_window ? "on" : "off"});
        if (options.qsa_sparse.has_value()) words.insert(words.end(), {"--native-qsa-sparse", *options.qsa_sparse ? "on" : "off"});
        if (options.gdn_ported.has_value()) words.insert(words.end(), {"--native-gdn-ported", *options.gdn_ported ? "on" : "off"});
        if (options.gdn_prefill_ported.has_value()) words.insert(words.end(), {"--native-gdn-prefill-ported", *options.gdn_prefill_ported ? "on" : "off"});
        if (options.prefill_hc_down_f32.has_value()) words.insert(words.end(), {"--native-prefill-hc-down-f32", *options.prefill_hc_down_f32 ? "on" : "off"});
#endif
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
        if (options.gdn_old_order_batch) words.insert(words.end(), {"--native-gdn-old-order-batch", "on"});
#endif
#endif
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
        if (options.prefill_grouped) words.insert(words.end(), {"--native-prefill-grouped", *options.prefill_grouped});
        if (options.verify_nax.has_value()) words.insert(words.end(), {"--verify-nax", *options.verify_nax ? "on" : "off"});
        if (options.verify_hc_gdn_fused.has_value()) words.insert(words.end(), {"--verify-hc-gdn-fused", *options.verify_hc_gdn_fused ? "on" : "off"});
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        if (options.mtp_depth) words.insert(words.end(), {"--mtp-depth", std::to_string(*options.mtp_depth)});
#endif
        if (options.prepared_validation) words.insert(words.end(), {"--native-prepared-validation", "on"});
        if (options.verify_cost_profile) words.insert(words.end(), {"--native-verify-cost-profile", options.verify_cost_profile->string()});
        if (options.verify_m8_splits) words.insert(words.end(), {"--native-verify-m8-splits", options.verify_m8_splits->string()});
        if (options.dflash2_draft_policy == "auto") words.insert(words.end(), {"--native-dflash2-draft-policy", "auto"});
        if (options.dflash2_defer_context) words.insert(words.end(), {"--native-dflash2-defer-context", *options.dflash2_defer_context ? "on" : "off"});
        if (options.dflash2_verify_rows) words.insert(words.end(), {"--native-dflash2-verify-rows", *options.dflash2_verify_rows});
        if (options.verify_nax_m8) words.insert(words.end(), {"--verify-nax-m8", *options.verify_nax_m8 ? "on" : "off"});
        if (options.mtp_request_carry) words.insert(words.end(), {"--mtp-request-carry", "on"});
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        if (options.mtp_output_packet) words.insert(words.end(), {"--mtp-output-packet", std::to_string(*options.mtp_output_packet)});
#endif
        if (options.mtp_draft_attention) words.insert(words.end(), {"--mtp-draft-attention", *options.mtp_draft_attention});
        if (options.mtp_draft_slice.has_value()) words.insert(words.end(), {"--native-mtp-draft-slice", *options.mtp_draft_slice ? "on" : "off"});
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        if (options.verify8) words.insert(words.end(), {"--known-token-capacity", "8", "--reuse-residual-projections", "true",
            "--prompt-lookup", "true", "--verification-budget-bytes", "134217728", "--allow-prompt-lookup-experiments",
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            options.stage8_present ? "false" :
#endif
            "true",
            "--verification-max-tokens", "8", "--hc-exact", "combined", "--wide-q8-exact", "off", "--wide-q8-many", "off"});
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (options.branch8) words.insert(words.end(), {"--verification8-branch-reuse", "true", "--worker-broker8", "true"});
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        if (options.stage8_present) words.insert(words.end(), {"--qsa-stage-v8", options.stage8_enabled ? "on" : "off"});
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        if (options.fast_channel.has_value()) words.insert(words.end(), {"--ple-gate-transform", "signed-root",
            "--qsa-channel-chronological", *options.fast_channel ? "on" : "off"});
#endif
        if (options.checkpoint_slots) {
            words.insert(words.end(), {"--checkpoint-slots", std::to_string(options.checkpoint_slots), "--checkpoint-budget-bytes",
                std::to_string(options.checkpoint_budget_bytes), "--worker-broker", "true"});
        }
        if (options.checkpoint_ssd) {
            words.insert(words.end(), {"--checkpoint-ssd", "true",
                "--checkpoint-ssd-budget-bytes", std::to_string(options.checkpoint_ssd_budget_bytes),
                "--checkpoint-ssd-floor-bytes", std::to_string(options.checkpoint_ssd_floor_bytes),
                "--checkpoint-ssd-dir", options.checkpoint_ssd_dir.string(),
                "--checkpoint-policy", options.checkpoint_policy});
        }
        if (options.overlap_prefill) {
            words.insert(words.end(), {"--overlap-prefill", "true"});
        }
        if (options.swiglu_fused.has_value()) {
            words.insert(words.end(), {"--swiglu-fused", *options.swiglu_fused ? "on" : "off"});
        }
        if (options.q8_gemv_rg4.has_value()) {
            words.insert(words.end(), {"--q8-gemv-rg4", *options.q8_gemv_rg4});
        }
        if (options.q8_three_msg.has_value()) {
            words.insert(words.end(), {"--q8-three-msg", *options.q8_three_msg});
        }
        if (options.moe_three_union.has_value()) {
            words.insert(words.end(), {"--moe-three-union", *options.moe_three_union ? "on" : "off"});
        }
        if (options.mtp_lookup.has_value()) {
            words.insert(words.end(), {"--mtp-lookup", *options.mtp_lookup});
        }
        if (options.mtp_lookup_match.has_value()) {
            words.insert(words.end(), {"--mtp-lookup-match", std::to_string(*options.mtp_lookup_match)});
        }
        if (options.prefill_m64.has_value()) {
            words.insert(words.end(), {"--prefill-m64", *options.prefill_m64});
        }
        if (options.prefill_multiseg.has_value()) {
            words.insert(words.end(), {"--prefill-multiseg", *options.prefill_multiseg ? "on" : "off"});
        }
        if (options.prefill_swizzle.has_value()) {
            words.insert(words.end(), {"--prefill-swizzle", *options.prefill_swizzle ? "on" : "off"});
        }
        if (options.prefill_router_fast.has_value()) {
            words.insert(words.end(), {"--native-prefill-router-fast", *options.prefill_router_fast ? "on" : "off"});
        }
        if (options.qsa_prefill_fused.has_value()) {
            words.insert(words.end(), {"--qsa-prefill-fused", *options.qsa_prefill_fused ? "on" : "off"});
        }
        if (options.hc_consumer_fused.has_value()) {
            words.insert(words.end(), {"--hc-consumer-fused", *options.hc_consumer_fused ? "on" : "off"});
        }
        if (options.ple_overlap.has_value()) {
            words.insert(words.end(), {"--ple-overlap", *options.ple_overlap ? "on" : "off"});
        }
        if (options.shared_prefix_cache.has_value()) {
            words.insert(words.end(), {"--shared-prefix-cache", *options.shared_prefix_cache ? "on" : "off"});
        }
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
        if (options.concurrent_dispatch.has_value()) words.insert(words.end(), {"--concurrent-dispatch", *options.concurrent_dispatch ? "on" : "off"});
#endif
#if defined(TOKSPUR_PRIVATE_Q8_THREE_ROWS) && defined(TOKSPUR_PRIVATE_NATIVE_MTP)
        if(options.q8_three_rows.has_value())words.insert(words.end(),{"--q8-three-rows",*options.q8_three_rows?"on":"off"});
#endif
        for (const auto& shard : options.shards) words.push_back(shard.string());
        std::vector<char*> arguments; for (auto& word : words) arguments.push_back(word.data()); arguments.push_back(nullptr);
        // No loader/preload, Python, remote-model or runtime-selection variables.
        char path_env[] = "PATH=/usr/bin:/bin:/usr/sbin:/sbin", lang_env[] = "LANG=C", locale_env[] = "LC_ALL=C";
        char* environment[]{path_env, lang_env, locale_env, nullptr};
        socket_pair(channels.control); pipe_pair(channels.status); socket_pair(channels.input);
        pipe_pair(channels.output); pipe_pair(channels.error); pipe_pair(channels.gate);
        const auto parent = getpid(); guardian_pid = fork();
        require(guardian_pid >= 0, "cannot launch owned worker guardian");
        if (!guardian_pid) guardian(channels, int(limit.rlim_cur), parent, arguments.data(), environment);
        drop(channels.control[1]); drop(channels.status[1]); drop(channels.input[1]); drop(channels.output[1]); drop(channels.error[1]); drop(channels.gate[0]); drop(channels.gate[1]);
        for (const int fd : {channels.control[0], channels.status[0], channels.input[0], channels.output[0], channels.error[0]}) nonblocking(fd);
        const auto deadline = Clock::now() + options.startup_timeout;
        while (!armed) { require(Clock::now() < deadline && !status_eof, "owned guardian did not arm before the startup deadline"); pump(deadline); }
        send_control('L', deadline);
        while (worker_pid < 0) { require(Clock::now() < deadline && !status_eof, "owned guardian did not identify its worker"); pump(deadline); }
        send_control('E', deadline);
        ready_value = next(deadline);
        require(ready_value.kind() == Json::Kind::object && event_is(ready_value, "ready"), "owned target did not emit a single ready event");
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        validate_raw_decode_ready(ready_value, options.raw_decode);
#else
        validate_raw_decode_ready(ready_value, std::nullopt);
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        validate_owned_weight_lanes_ready(ready_value, !options.weight_lanes.empty(), options.weight_lanes_auth_workers);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
        validate_owned_concurrent_ready(ready_value, options.concurrent_dispatch);
#endif
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
        validate_mtp_output_packet_selection(ready_value, options.mtp_output_packet);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        validate_owned_q8_three_ready(ready_value,options.q8_three_rows);
#else
        validate_owned_q8_three_ready(ready_value,std::nullopt);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        validate_mtp_input_packet_selection(ready_value, options.prefill_input512);
        validate_mtp_final_input_selection(ready_value, options.prefill_final_logits);
        validate_mtp_hc_rows_selection(ready_value, options.prefill_hc_rows.value_or(false));
        validate_mtp_router_window_selection(ready_value, options.prefill_router_window.value_or(false));
        validate_mtp_hc_down_window_selection(ready_value, options.prefill_hc_down_window.value_or(false));
        validate_mtp_qsa_sparse_selection(ready_value, options.qsa_sparse.value_or(false));
        validate_mtp_gdn_ported_selection(ready_value, options.gdn_ported.value_or(false));
        if (options.gdn_prefill_ported.has_value()) {
            require(worker_field(ready_value, "native_prefill_gdn_tb12").boolean() == *options.gdn_prefill_ported,
                "owned prefill GDN tb12 readiness does not match requested option");
        }
        validate_mtp_hc_down_f32_selection(ready_value,
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
            options.prefill_hc_down_f32.value_or(false)
#else
            false
#endif
            );
        validate_mtp_gdn_old_order_selection(ready_value,
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
            options.gdn_old_order_batch
#else
            false
#endif
            );
        validate_mtp_prefill_selection(ready_value, options.prefill_dense
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            , options.prefill_grouped
#endif
        );
#else
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        validate_mtp_input_packet_selection(ready_value, false);
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        if (options.verify8) {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (options.branch8) { validate_branch8_ready(ready_value); verify_previous = ready_value; }
            else
#endif
            validate_verify8_ready(ready_value);
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        const auto* staged = ready_value.find("qsa_stage_v8_prepared");
        require(bool(staged) == options.stage8_present && (!staged || (staged->boolean() &&
            worker_field(ready_value, "qsa_stage_v8_enabled").boolean() == options.stage8_enabled)),
            "owned staged-QSA readiness does not match the exact requested startup arm");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
        require(verify8_fast_math(ready_value) == options.fast_channel,
            "owned fast-math readiness does not match the exact requested startup arm");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
        if (options.scalar_channel) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!options.mtp_sidecar.empty() || !options.dflash2_dir.empty()) {
                validate_mtp_ready(ready_value,
                    options.mtp_draft_attention.value_or(!options.dflash2_dir.empty() ? "split16" : "serial"), options.mtp_request_carry
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                    , options.mtp_depth.value_or(!options.dflash2_dir.empty() ? 7u : 1u)
#endif
                    );
                if (options.dflash2_precision)
                    require(worker_field(ready_value, "native_dflash2_precision").string() == *options.dflash2_precision,
                        "owned DFlash2 precision readiness differs");
                if (options.dflash2_max_drafts)
                    require(worker_integer(ready_value, "native_dflash2_max_drafts") == *options.dflash2_max_drafts,
                        "owned DFlash2 max drafts readiness differs");
                if (options.dflash2_vocab_slice)
                    require(worker_integer(ready_value, "native_dflash2_vocab_slice") == *options.dflash2_vocab_slice,
                        "owned DFlash2 vocab slice readiness differs");
                if (options.prepared_validation) require(worker_field(ready_value, "native_prepared_validation").boolean(), "prepared validation startup differs");
                if (options.verify_cost_profile) require(worker_field(ready_value, "native_measured_controller").boolean() && worker_integer(ready_value, "native_measured_timing_schema") == 1 && worker_field(ready_value, "native_controller_policy_digest").array().size() == 32, "measured controller startup differs");
                if (options.dflash2_draft_policy == "auto") require(worker_field(ready_value, "native_dflash2_draft_policy").string() == "auto", "drafting policy startup differs");
                if (options.verify_m8_splits) require(worker_field(ready_value, "native_target_m8_split_digest").string().size() == 64, "M8 split policy startup differs");
                if (options.dflash2_defer_context) require(worker_field(ready_value, "native_dflash2_defer_context").boolean() == *options.dflash2_defer_context, "DFlash2 context startup differs");
                if (options.dflash2_verify_rows) require(worker_field(ready_value, "native_dflash2_verify_rows").string() == *options.dflash2_verify_rows, "DFlash2 rows startup differs");
                if (options.logit_topk.has_value())
                    require(worker_field(ready_value, "native_logit_topk").boolean() == *options.logit_topk,
                        "owned logit top-k readiness differs");
                if (options.ple_async_gather.has_value())
                    require(worker_field(ready_value, "native_ple_async_gather").boolean() == *options.ple_async_gather,
                        "owned PLE async gather readiness differs");
                if (options.mtp_fast_round.has_value())
                    require(worker_field(ready_value, "native_mtp_fast_round").boolean() == *options.mtp_fast_round,
                        "owned MTP fast round readiness differs");
                if (options.prefill_router_fast.has_value())
                    require(worker_field(ready_value, "native_prefill_router_fast").boolean() == *options.prefill_router_fast,
                        "owned prefill router fast readiness differs");
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
                if (options.verify_hc_gdn_fused.has_value())
                    require(worker_field(ready_value, "native_verify_hc_gdn_fused").boolean() == *options.verify_hc_gdn_fused,
                        "owned verify HC+GDN fused readiness differs");
#endif
            } else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            if (options.sampled_prefill) validate_sampled_prefill_ready(ready_value); else
#endif
            validate_scalar_channel_ready(ready_value);
            scalar_previous = ready_value;
        }
#endif
        if (options.checkpoint_slots) {
            require(worker_field(ready_value, "checkpoint_broker").boolean(), "owned target did not prepare broker controls");
            broker_storage(ready_value); broker_epoch = worker_integer(ready_value, "broker_epoch");
            require(broker_epoch, "owned broker epoch is invalid");
        } else if (const auto* broker = ready_value.find("checkpoint_broker")) require(!broker->boolean(), "unexpected owned broker preparation");
        idle_check();
        starting = false;
    }
    void status_input() {
        std::array<char, 128> data{};
        for (;;) {
            const auto n = read(channels.status[0], data.data(), data.size());
            if (!n) { status_eof = true; break; }
            if (n < 0) { if (errno == EINTR) continue; if (errno == EAGAIN || errno == EWOULDBLOCK) break; throw WorkerError("worker guardian status pipe failed"); }
            notice_bytes.append(data.data(), size_t(n)); require(notice_bytes.size() <= sizeof(Notice) * 4, "worker guardian status framing overflow");
            while (notice_bytes.size() >= sizeof(Notice)) {
                Notice note{}; std::memcpy(&note, notice_bytes.data(), sizeof(note)); notice_bytes.erase(0, sizeof(note));
                if (note.kind == armed_notice && !armed && note.value == guardian_pid) armed = true;
                else if (note.kind == launched_notice && armed && worker_pid < 0 && note.value > 1) worker_pid = note.value;
                else if (note.kind == exit_notice && armed && !worker_exited) { worker_exited = true; worker_status = note.value; }
                else throw WorkerError("worker guardian state transition refused");
            }
        }
        require(!status_eof || notice_bytes.empty(), "truncated worker guardian status");
    }
    void read_stream(int fd, bool& eof, bool standard_output, bool discard = false) {
        std::array<char, 4096> data{};
        for (unsigned reads = 0; reads < 16; ++reads) {
            const auto n = read(fd, data.data(), data.size());
            if (!n) { eof = true; break; }
            if (n < 0) { if (errno == EINTR) continue; if (errno == EAGAIN || errno == EWOULDBLOCK) break; throw WorkerError("owned worker pipe read failed"); }
            if (standard_output && !discard) {
                pending.append(data.data(), size_t(n));
                const auto newline = pending.find('\n');
                require((newline == std::string::npos ? pending.size() : newline) <= line_limit && pending.size() <= line_limit + 65536,
                    "owned worker output framing limit exceeded");
            } else if (!standard_output) stderr_bytes = std::min(stderr_limit, stderr_bytes + size_t(n));
        }
    }
    void pump(WorkerDeadline deadline, bool discard = false, int wait_ms = poll_slice_ms) {
        if (!discard && starting && options.startup_cancelled && options.startup_cancelled())
            throw WorkerError("owned startup cancelled before readiness; no automatic restart");
        if (Clock::now() >= deadline) return;
        std::array<struct pollfd, 3> watches{{{out_eof ? -1 : channels.output[0], POLLIN, 0},
            {err_eof ? -1 : channels.error[0], POLLIN, 0}, {status_eof ? -1 : channels.status[0], POLLIN, 0}}};
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        const auto result = poll(watches.data(), watches.size(), int(std::min<int64_t>(std::max<int64_t>(1, remaining), wait_ms)));
        if (result < 0 && errno != EINTR) throw WorkerError("owned worker poll failed");
        for (size_t i = 0; i < watches.size(); ++i) if (watches[i].revents) {
            require(!(watches[i].revents & POLLNVAL), "owned worker descriptor became invalid");
            if (i == 0 && !out_eof) read_stream(channels.output[0], out_eof, true, discard);
            if (i == 1 && !err_eof) read_stream(channels.error[0], err_eof, false);
            if (i == 2 && !status_eof) status_input();
        }
    }
    void send_control(char byte, WorkerDeadline deadline) {
        for (;;) { require(Clock::now() < deadline, "owned guardian handshake deadline expired"); const auto n = send_bytes(channels.control[0], &byte, 1); if (n == 1) return;
            if (n < 0 && errno == EINTR) continue;
            require(n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK), "owned guardian control channel failed"); pump(deadline); }
    }
    Json next(WorkerDeadline deadline) {
        for (;;) {
            require(Clock::now() < deadline, "owned worker response deadline expired");
            const auto newline = pending.find('\n');
            if (newline != std::string::npos) {
                require(newline && newline <= line_limit, "owned worker emitted an empty/oversized line");
                const auto line = pending.substr(0, newline); pending.erase(0, newline + 1);
                try { return parse_json(line, {line_limit, 300000, 12}); }
                catch (...) { throw WorkerError("owned worker emitted malformed JSON"); }
            }
            require(!out_eof && !worker_exited && !status_eof, "owned worker ended before a complete response");
            pump(deadline);
        }
    }
    void idle_check() {
        pump(Clock::now() + options.step_timeout, false, 0);
        require(!failed && !closed && !worker_exited && !out_eof && !status_eof && pending.empty(), "owned worker is not drained and ready for another command");
    }
    void shutdown() {
        if (closed) return;
        if (guardian_pid <= 0) { closed = true; drop_channels(channels); return; }
        drop(channels.input[0]); pending.clear();
        auto deadline = Clock::now() + options.drain_timeout;
        bool forced = false;
        for (;;) {
            if (worker_exited && out_eof && err_eof && status_eof && notice_bytes.empty()) break;
            if (worker_pid > 0 && status_eof && !worker_exited) {
                // If the guardian itself crashes, keep its PID unreaped while
                // checking the worker's exact owned session/group. A recycled
                // PID in another session is never a permissible signal target.
                siginfo_t state{};
                if (waitid(P_PID, id_t(guardian_pid), &state, WEXITED | WNOHANG | WNOWAIT) == 0 && state.si_pid == guardian_pid) {
                    if (getsid(worker_pid) == guardian_pid && getpgid(worker_pid) == worker_pid) { kill(-worker_pid, SIGKILL); kill(worker_pid, SIGKILL); }
                    if (out_eof && err_eof && kill(worker_pid, 0) < 0 && errno == ESRCH) {
                        // No exact guardian-to-worker wait record exists. Stop
                        // the child, reap our guardian, but never claim verified
                        // normal closure for this exceptional path.
                        guardian_status = reap_child(guardian_pid); guardian_reaped = true;
                        throw WorkerError("guardian crashed; worker stopped but exact child wait evidence is missing");
                    }
                }
            }
            if (worker_pid < 0 && status_eof) {
                // No exec acknowledgement was sent. A failed guardian cannot
                // have started the target; close its gate/lifetime endpoint and
                // verify the exact guardian wait rather than inventing a ready.
                drop(channels.control[0]);
                const auto result = waitpid(guardian_pid, &guardian_status, WNOHANG);
                if (result == guardian_pid) { guardian_reaped = true; closed = true; drop_channels(channels); return; }
                require(result == 0 || (result < 0 && errno == EINTR), "unlaunched guardian waitpid failed");
            }
            if (Clock::now() >= deadline) {
                if (forced) throw WorkerError("owned worker exit/EOF drain could not be verified; guardian remains responsible");
                drop(channels.control[0]); forced = true; deadline = Clock::now() + std::chrono::milliseconds(1500);
            }
            pump(deadline, true);
        }
        drop(channels.control[0]);
        if (!guardian_reaped && guardian_pid > 0) {
            for (;;) {
                const auto result = waitpid(guardian_pid, &guardian_status, WNOHANG);
                if (result == guardian_pid) { guardian_reaped = true; break; }
                if (result < 0 && errno == EINTR) continue;
                require(result == 0 && Clock::now() < deadline, "exact owned guardian waitpid could not be verified");
                struct pollfd pause{-1, 0, 0}; poll(&pause, 0, poll_slice_ms);
            }
        }
        require(guardian_reaped && WIFEXITED(guardian_status) && WEXITSTATUS(guardian_status) == 0 && worker_status != -1,
            "owned guardian/worker exit was not verified");
        closed = true; drop_channels(channels);
    }
    void invalidate() noexcept {
        failed = true; needs_reset = true; drop(channels.control[0]);
        try { shutdown(); } catch (...) {}
    }
};

OwnedWorker::OwnedWorker(OwnedWorkerOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
    try { impl_->start(); } catch (const WorkerError&) { impl_->invalidate(); throw; }
    catch (...) { impl_->invalidate(); throw WorkerError("owned worker startup configuration failed"); }
}
OwnedWorker::~OwnedWorker() = default;
const Json& OwnedWorker::ready() const { return impl_->ready_value; }
bool OwnedWorker::alive() const {
    const auto& p = *impl_; std::unique_lock lock(p.mutex, std::try_to_lock);
    if (!lock || p.failed || p.closed || p.worker_exited || p.out_eof || p.status_eof) return false;
    struct pollfd status{p.channels.status[0], POLLIN, 0};
    return poll(&status, 1, 0) == 0;
}
Json OwnedWorker::request(const Json& body, WorkerDeadline deadline, const std::function<bool()>& cancelled,
                          const std::function<void(const Json&)>& on_token) {
    auto& p = *impl_; std::unique_lock lock(p.mutex, std::try_to_lock); require(bool(lock), "owned worker already has an active command");
    return request_locked(body, deadline, cancelled, on_token);
}
Json OwnedWorker::request_locked(const Json& body, WorkerDeadline deadline, const std::function<bool()>& cancelled,
                                 const std::function<void(const Json&)>& on_token) {
    auto& p = *impl_;
    // Validate admission before a single request byte is written.
    require(!p.failed && !p.closed, "owned worker context is unavailable; no automatic restart");
    require(body.kind() == Json::Kind::object, "owned worker command must be an object");
    const auto* tokens = body.find("tokens");
    uint64_t output_count = 0; size_t input_count = 0;
    bool reset = false, begin = false, allow_restore = false, control = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    bool mtp_abort = false;
#endif
    std::string operation;
    try {
        if (const auto* op = body.find("op")) {
            require(p.broker_epoch, "owned broker controls require explicit startup preparation");
            operation = op->string(); control = true; begin = operation == "branch_begin";
            require(begin || operation == "branch_capture", "unknown owned broker operation");
            if (begin) {
                worker_keys(body, {"op", "prompt_ids", "generation_requested", "allow_restore", "sampling", "seed"});
                const auto& prompt = worker_field(body, "prompt_ids").array();
                require(!prompt.empty() && prompt.size() <= 65536, "owned broker future prompt bound");
                for (const auto& token : prompt) require(token.unsigned_integer() < 248320, "owned broker future token bound");
                static_cast<void>(worker_field(body, "generation_requested").boolean());
                allow_restore = worker_field(body, "allow_restore").boolean();
                require(!allow_restore || p.broker_lease, "owned broker reuse requires an established lease");
                const auto& policy = worker_field(body, "sampling");
                worker_keys(policy, {"temperature", "top_k", "top_p", "min_p"});
                const auto top_p = worker_number(worker_field(policy, "top_p")), min_p = worker_number(worker_field(policy, "min_p"));
                require(worker_number(worker_field(policy, "temperature")) == 0 && worker_integer(policy, "top_k") <= 248320 &&
                    top_p > 0 && top_p <= 1 && min_p >= 0 && min_p <= 1, "owned broker requires bounded greedy sampling");
                static_cast<void>(worker_integer(body, "seed"));
            } else worker_keys(body, {"op"});
        } else {
            const auto* generate = body.find("generate"); const auto* clear = body.find("reset");
            require(tokens && generate && clear && tokens->kind() == Json::Kind::array && generate->kind() == Json::Kind::number && clear->kind() == Json::Kind::boolean,
                "owned worker command shape refused");
            output_count = generate->unsigned_integer(); input_count = tokens->array().size(); reset = clear->boolean();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (const auto* generation = body.find("mtp_generation")) {
                require(p.options.mtp_request_carry, "MTP generation command requires startup consent");
                mtp_abort = mtp_generation_command(*generation).abort;
            }
#endif
            for (const auto& token : tokens->array()) require(token.unsigned_integer() < 248320, "owned input token is outside vocabulary");
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (!p.options.mtp_sidecar.empty() || !p.options.dflash2_dir.empty()) require((input_count >= 1 && input_count <= mtp_input_packet(p.ready_value) && output_count == 0) ||
                (input_count == 0 && output_count >= 1 && output_count <= mtp_output_packet(p.ready_value)) ||
                (mtp_abort && !input_count && !output_count && !reset), "MTP commands exceed the negotiated input or output bound");
            else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            if (p.options.sampled_prefill) require((input_count >= 1 && input_count <= 8 && output_count == 0) ||
                (input_count == 0 && output_count == 1), "sampled input8 commands require1..8 known inputs OR1 output");
            else
#endif
            if (p.options.verify8) require(input_count <= 8 && output_count <= 8 && bool(input_count) != bool(output_count),
                "private HTTP V8 commands require <=8 input OR output tokens");
            else
#endif
            require(input_count <= 1 && output_count <= 1 && input_count + output_count <= 1,
                "owned worker commands are bounded to one model step");
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
            require(!p.options.scalar_channel ||
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                !p.options.mtp_sidecar.empty() || !p.options.dflash2_dir.empty() ||
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
                p.options.sampled_prefill ||
#endif
                input_count + output_count == 1,
                "scalar-channel HTTP requires exactly one input or output step");
            if (p.options.scalar_channel) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (p.options.mtp_request_carry) worker_keys(body, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed", "mtp_generation", "prefill_final"}); else
#endif
                worker_keys(body, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed"});
            }
#endif
            if (p.broker_epoch) {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (p.options.branch8) worker_keys(body, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed", "prompt_lookup"});
                else
#endif
                worker_keys(body, {"tokens", "generate", "reset", "stop_ids", "sampling", "seed"});
            }
        }
        require(!p.needs_reset || reset || (begin && !allow_restore)
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            || mtp_abort
#endif
            , "cancelled/ambiguous context requires an explicit cold reset");
        require(!p.broker_epoch || begin || reset || p.broker_lease, "owned broker requires an initial reset or begin");
    } catch (const WorkerError&) { throw; }
    catch (...) { throw WorkerError("owned worker encoding refused before submission"); }
    bool before_cancelled = false;
    try { before_cancelled = cancelled && cancelled(); }
    catch (...) { throw WorkerError("owned cancellation probe failed before submission"); }
    if (before_cancelled) { p.needs_reset = true; throw WorkerCancelled("owned request cancelled before submission"); }
    deadline = std::min(deadline, Clock::now() + p.options.step_timeout);
    require(Clock::now() < deadline, "owned request deadline expired before submission");
    uint64_t command = p.broker_command, lease = p.broker_lease; const auto before = p.broker_position;
    Json wire = body;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::optional<MtpWireReply> mtp_reply;
    if (!p.options.mtp_sidecar.empty() || !p.options.dflash2_dir.empty()) {
        try { mtp_reply.emplace(p.ready_value, p.scalar_previous, body); }
        catch (...) { throw WorkerError("MTP command refused before submission"); }
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
    std::optional<Json> scalar_token;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    std::optional<Verify8Reply> verify;
    Json verify_policy; uint64_t verify_seed = 0;
    if (p.options.verify8 && !control) {
        verify_policy = reset ? worker_field(body, "sampling") : p.verify_sampling;
        verify_seed = reset ? worker_integer(body, "seed") : p.verify_seed;
        verify.emplace(p.ready_value, p.verify_previous, body, p.verify_position, verify_policy, verify_seed, false
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            , p.options.branch8
#endif
            );
    }
#endif
    if (p.broker_epoch) {
        require(command < std::numeric_limits<uint64_t>::max(), "owned broker command space exhausted"); ++command;
        if (begin || reset) { require(lease < std::numeric_limits<uint64_t>::max(), "owned broker lease space exhausted"); ++lease; }
        auto fields = body.object();
        for (const auto& field : Json::Object{{"broker_epoch", Json::integer(p.broker_epoch)}, {"command_id", Json::integer(command)},
            {"lease_id", Json::integer(lease)}, {"expected_prefix", Json::integer(before)}}) fields.push_back(field);
        wire = Json(std::move(fields));
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (p.options.branch8) {
            auto fields = wire.object(); fields.emplace_back("broker_protocol", Json("branch8-v1")); fields.emplace_back("broker_batch_width", Json::integer(8));
            wire = Json(std::move(fields));
        }
#endif
    }
    const auto bytes = wire.dump(line_limit - 1) + '\n';
    bool cancellation = false, timed_out = false, consumer_failure = false; auto drain_deadline = deadline;
    try {
        p.idle_check(); size_t sent = 0;
        while (sent < bytes.size()) {
            require(Clock::now() < deadline, "owned request submission deadline expired");
            const auto n = send_bytes(p.channels.input[0], bytes.data() + sent, bytes.size() - sent);
            if (n > 0) sent += size_t(n);
            else if (n < 0 && errno == EINTR) continue;
            else { require(n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK), "owned request pipe failed; submission is ambiguous"); p.pump(deadline); }
        }
        if (p.broker_epoch) { p.broker_command = command; p.broker_lease = lease; }
        size_t token_events = 0; std::optional<uint32_t> observed;
        for (;;) {
            if (!cancellation && !timed_out && !consumer_failure) {
                try { cancellation = cancelled && cancelled(); } catch (...) { consumer_failure = true; }
                timed_out = Clock::now() >= deadline;
                if (cancellation || timed_out || consumer_failure) { p.needs_reset = true; drain_deadline = Clock::now() + p.options.drain_timeout; }
            }
            const bool draining = cancellation || timed_out || consumer_failure;
            const auto newline = p.pending.find('\n');
            if (newline == std::string::npos) {
                require(!p.out_eof && !p.worker_exited && !p.status_eof, "owned target ended during a command");
                require(!draining || Clock::now() < drain_deadline, "owned cancelled/deadline command did not drain in time");
                p.pump(draining ? drain_deadline : deadline); continue;
            }
            const auto value = p.next(draining ? drain_deadline : deadline);
            require(value.kind() == Json::Kind::object, "owned target emitted a nonobject event");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (mtp_reply && !event_is(value, "done")) { mtp_reply->event(value); continue; }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
            if (verify && !event_is(value, "done")) {
                if (p.broker_epoch) p.broker_identity(value, command, lease);
                verify->event(value); continue; // No token/round callback before complete admission.
            }
#endif
            if (event_is(value, "token")) {
                require(!control && ++token_events <= 1, "owned target emitted unsolicited/excess token events");
                if (p.broker_epoch) {
                    p.broker_identity(value, command, lease);
                    const auto id = worker_integer(value, "id");
                    require(output_count == 1 && id < 248320 && worker_integer(value, "committed_prefix") == (reset ? 0 : before) + 1,
                        "owned broker token publication mismatch"); observed = uint32_t(id);
                }
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
                if (p.options.scalar_channel) scalar_token = value; // Release only after math/state admission below.
                else
#endif
                if (!draining && on_token) {
                    try { on_token(value); } catch (...) { consumer_failure = true; p.needs_reset = true; drain_deadline = Clock::now() + p.options.drain_timeout; }
                }
            } else if (event_is(value, control ? "control_done" : "done")) {
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
                validate_owned_weight_lanes_done(p.ready_value, value);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
                validate_owned_concurrent_done(p.ready_value, value);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
                if(!control)validate_owned_q8_three_done(p.ready_value,
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
                    p.options.scalar_channel?p.scalar_previous:p.ready_value,
#else
                    p.ready_value,
#endif
                    value);
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
                if (p.options.scalar_channel) {
                    require(!control, "scalar-channel transport has no broker controls");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    if (mtp_reply) mtp_reply->finish(value); else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
                    if (p.options.sampled_prefill) validate_sampled_prefill_done(p.ready_value, p.scalar_previous, body, value,
                        scalar_token ? &*scalar_token : nullptr);
                    else
#endif
                    validate_scalar_channel_done(p.ready_value, p.scalar_previous, body, value,
                        scalar_token ? &*scalar_token : nullptr);
                    p.scalar_previous = value;
                }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
                if (verify) {
                    verify->finish(value);
                    p.verify_previous = value; p.verify_sampling = verify_policy; p.verify_seed = verify_seed;
                    p.verify_position = uint32_t(worker_integer(value, "committed_prefix"));
                }
#endif
                if (p.broker_epoch) {
                    p.broker_identity(value, command, lease); p.broker_storage(value);
                    const auto position = worker_integer(value, "committed_prefix");
                    require(position <= 65536 && worker_integer(value, "checkpoint_retained") <= p.options.checkpoint_slots,
                        "owned broker prefix/retention bound");
                    const auto& generated = worker_field(value, "generated_ids").array();
                    if (control) {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                        if (p.options.branch8) {
                            validate_branch8_control(p.ready_value, p.verify_previous, body, value);
                            p.verify_previous = value; p.verify_position = uint32_t(position);
                            if (begin) { p.verify_sampling = worker_field(body, "sampling"); p.verify_seed = worker_integer(body, "seed"); }
                        }
#endif
                        require(worker_field(value, "op").string() == operation && worker_integer(value, "model_steps") == 0 &&
                            generated.empty() && !token_events, "owned control executed model work");
                        const auto outcome = worker_field(value, "outcome").string();
                        const auto common = worker_integer(value, "common_prefix");
                        if (begin) {
                            const auto count = worker_field(body, "prompt_ids").array().size();
                            require(common <= std::min<uint64_t>(before, count) && (allow_restore || !common) && position <= common &&
                                ((outcome == "cold_miss" && !position) || (outcome == "restored" && position && allow_restore)),
                                "owned broker restore selection mismatch");
                            const auto& sampling = worker_field(value, "sampling");
                            require(worker_integer(sampling, "seed") == worker_integer(body, "seed"), "owned broker seed changed");
                            for (const auto& [name, parameter] : worker_field(body, "sampling").object())
                                require(worker_number(worker_field(sampling, name)) == worker_number(parameter), "owned broker sampling policy changed");
                        } else require(outcome == "captured" && position == before && common == before, "owned broker capture prefix mismatch");
                        const bool logits = worker_field(value, "logits_available").boolean();
                        const auto& stop = worker_field(value, "stop_id");
                        require(logits == bool(position) && (stop.is_null() || (logits && stop.unsigned_integer() < 248320)) &&
                            worker_field(value, "continuation_allowed").boolean() == (logits && stop.is_null()),
                            "owned broker terminal/logit metadata mismatch");
                    } else {
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
                        if (verify) require(position == p.verify_position, "owned branch8 verified prefix mismatch");
                        else
#endif
                        require(position == (reset ? 0 : before) + input_count + output_count && generated.size() == output_count &&
                            token_events == output_count && (!observed || generated[0].unsigned_integer() == *observed),
                            "owned broker step publication mismatch");
                    }
                    const auto& sampling = worker_field(value, "sampling");
                    require(worker_field(sampling, "valid").boolean() && worker_integer(sampling, "entropy_prefix") == position,
                        "owned broker entropy publication mismatch");
                    p.broker_position = uint32_t(position);
                }
                p.idle_check();
                if (consumer_failure) throw WorkerError("owned token/cancellation validation callback failed after drain");
                if (cancellation) throw WorkerCancelled("owned request cancelled after its current step drained; reset required");
                if (timed_out) throw WorkerError("owned request deadline expired after drain; reset required");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (mtp_reply && on_token) for (const auto& event : mtp_reply->events()) {
                    if (cancelled && cancelled()) { p.needs_reset = true; throw WorkerCancelled("MTP cancelled after admitted drain; reset required"); }
                    p.needs_reset = true;
                    on_token(event);
                }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
                if (p.options.scalar_channel && scalar_token && on_token) {
                    if (cancelled && cancelled()) { p.needs_reset = true; throw WorkerCancelled("scalar-channel cancelled after admitted drain; reset required"); }
                    // The model/RNG already advanced, but the callback has not
                    // published it successfully. WorkerCancelled must retain
                    // this dirty state through its dedicated catch below.
                    p.needs_reset = true;
                    on_token(*scalar_token);
                }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
                if (verify && on_token) for (const auto& event : verify->events()) {
                    if (cancelled && cancelled()) { p.needs_reset = true; throw WorkerCancelled("private V8 cancelled after admitted drain; reset required"); }
                    on_token(event); // Callback failures invalidate, not a second drain of an already completed command.
                }
#endif
                p.needs_reset = false; return value;
            } else throw WorkerError("owned target rejected or corrupted a command; context invalidated");
        }
    } catch (const WorkerCancelled&) { throw; }
    catch (...) { p.invalidate(); throw WorkerError("owned command failed or became ambiguous; worker invalidated without retry"); }
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
void OwnedWorker::abort_mtp_generation(WorkerDeadline deadline) {
    auto& p = *impl_;
    std::unique_lock lock(p.mutex, std::try_to_lock); require(bool(lock), "cannot abort an active owned command reentrantly");
    if (!p.options.mtp_request_carry || p.failed || p.closed) return;
    const auto* generation = p.scalar_previous.find("native_mtp_generation");
    if (!generation || !worker_field(*generation, "active").boolean()) return;
    const auto id = worker_integer(*generation, "id");
    try {
        static_cast<void>(request_locked(Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(0)},
            {"reset", Json(false)}, {"mtp_generation", Json(Json::Object{{"id", Json::integer(id)}, {"abort", Json(true)}})}}),
            deadline, {}, {}));
    } catch (...) { p.invalidate(); throw; }
    // The HTTP response failed even though this cleanup command was admitted.
    // Its next model request must still cold reset the target and entropy.
    p.needs_reset = true;
}
#endif
void OwnedWorker::close() { auto& p = *impl_; std::unique_lock lock(p.mutex, std::try_to_lock); require(bool(lock), "cannot close an active owned command reentrantly"); p.shutdown(); }

} // namespace tokspur
