#include "tokspur/owned_worker.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace tokspur;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
size_t checks = 0;
void check(bool value, const char* why) { ++checks; if (!value) throw std::runtime_error(why); }
template<class F> void refuses(F&& action) { ++checks; try { action(); } catch (const WorkerError&) { return; } throw std::runtime_error("unsafe worker operation accepted"); }
template<class F> void cancellation(F&& action) { ++checks; try { action(); } catch (const WorkerCancelled&) { return; } throw std::runtime_error("cancelled worker request did not report cancellation"); }
void pause_ms(int milliseconds) { struct pollfd empty{-1, 0, 0}; while (poll(&empty, 0, milliseconds) < 0 && errno == EINTR) {} }
void write_all(int fd, std::string_view bytes) {
    while (!bytes.empty()) { const auto n = write(fd, bytes.data(), bytes.size()); if (n > 0) bytes.remove_prefix(size_t(n)); else if (n < 0 && errno == EINTR) continue; else _exit(90); }
}
Json broker_storage(Json value) {
    auto fields = value.object();
    fields.emplace_back("checkpoint_slots", Json::integer(2)); fields.emplace_back("checkpoint_combined_budget_bytes", Json::integer(1 << 20));
    fields.emplace_back("checkpoint_host_bytes", Json::integer(512)); fields.emplace_back("checkpoint_store_host_bytes", Json::integer(256));
    fields.emplace_back("checkpoint_gpu_bytes", Json::integer(1024)); return Json(std::move(fields));
}
Json ready_fixture(std::string_view execution = "direct", bool broker = false) {
    Json::Object ready{{"event", Json("ready")}, {"pid", Json::integer(getpid())}, {"guardian_pid", Json::integer(getppid())},
        {"group", Json::integer(getpgrp())}, {"session", Json::integer(getsid(0))}, {"small_gemv_column_splits", Json::integer(16)},
        {"block_gemv_formats", Json::integer(3)}, {"stack_execution", Json(std::string(execution))}, {"maximum_generated_tokens", Json::integer(4096)}};
    if (execution == "recurrent-replay") {
        ready.emplace_back("replay_prepared", Json(true));
        ready.emplace_back("recurrent_replay_captures", Json::integer(24));
        ready.emplace_back("mapped_weight_region_limit_bytes", Json::integer(uint64_t(4) * 1024 * 1024 * 1024));
    }
    if (broker) { ready.emplace_back("checkpoint_broker", Json(true)); ready.emplace_back("broker_epoch", Json::integer(77)); return broker_storage(Json(std::move(ready))); }
    return Json(std::move(ready));
}
void emit(const Json& value, bool fragment) {
    const auto line = value.dump() + '\n';
    if (!fragment) write_all(STDOUT_FILENO, line);
    else for (size_t at = 0; at < line.size(); at += 3) { write_all(STDOUT_FILENO, std::string_view(line).substr(at, 3)); pause_ms(1); }
}
int fake_worker(int argc, char** argv) {
    std::filesystem::path policy;
    std::string_view execution; size_t selections = 0; bool broker = false;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--policy") policy = argv[i + 1];
        if (std::string_view(argv[i]) == "--stack-execution") { execution = argv[i + 1]; ++selections; }
        if (std::string_view(argv[i]) == "--worker-broker") broker = std::string_view(argv[i + 1]) == "true";
    }
    if (policy.empty() || selections != 1 || (execution != "direct" && execution != "recurrent-replay")) return 91;
    const auto mode = policy.filename().string(); const bool fragment = mode == "fragment";
    { std::ofstream pid(policy.parent_path() / "worker.pid"); pid << getpid() << '\n' << getppid() << '\n'; }
    if (mode == "stall-ready") for (;;) pause();
    if (mode == "eof-ready") return 0;
    if (mode == "bad-ready") { write_all(STDOUT_FILENO, "{invalid}\n"); return 0; }
    if (mode == "huge-ready") { write_all(STDOUT_FILENO, std::string(1024 * 1024 + 1, 'x')); for (;;) pause(); }
    emit(ready_fixture(execution, broker), fragment);
    uint32_t prefix = 0;
    uint64_t seed = 0; Json sampling(Json::Object{{"temperature", Json::integer(0)}, {"top_k", Json::integer(0)}, {"top_p", Json::integer(1)}, {"min_p", Json::integer(0)}});
    std::string line;
    while (std::getline(std::cin, line)) {
        { std::ofstream count(policy.parent_path() / "commands", std::ios::app); count << '1' << '\n'; }
        if (mode == "eof-command") return 0;
        if (mode == "partial-eof") { write_all(STDOUT_FILENO, "{\"event\":\"done\""); return 0; }
        if (mode == "bad-command") { write_all(STDOUT_FILENO, "{invalid}\n"); continue; }
        if (mode == "stderr-stall") for (unsigned i = 0; i < 64; ++i) write_all(STDERR_FILENO, std::string(4096, 'e'));
        if (mode == "stall" || mode == "stderr-stall") for (;;) pause();
        const auto body = parse_json(line);
        const auto sampled = [&] {
            auto fields = sampling.object(); fields.emplace_back("valid", Json(true)); fields.emplace_back("entropy_prefix", Json::integer(prefix));
            fields.emplace_back("seed", Json::integer(seed)); return Json(std::move(fields));
        };
        const auto stamp = [&](Json value) {
            if (!broker) return value;
            auto fields = value.object();
            for (const auto* name : {"broker_epoch", "command_id", "lease_id"}) {
                auto n = body.find(name)->unsigned_integer();
                if ((mode == "broker-stale-epoch" && std::string_view(name) == "broker_epoch") ||
                    (mode == "broker-stale-lease" && std::string_view(name) == "lease_id")) ++n;
                fields.emplace_back(name, Json::integer(n));
            }
            return broker_storage(Json(std::move(fields)));
        };
        if (const auto* op = body.find("op")) {
            const bool begin = op->string() == "branch_begin";
            if (begin) { prefix = 0; sampling = *body.find("sampling"); seed = body.find("seed")->unsigned_integer(); }
            if (mode == "broker-slow-control") pause_ms(80);
            if (mode == "broker-control-token") emit(stamp(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(42)}, {"committed_prefix", Json::integer(prefix + 1)}})), false);
            emit(stamp(Json(Json::Object{{"event", Json("control_done")}, {"op", *op}, {"outcome", Json(begin ? "cold_miss" : "captured")},
                {"model_steps", Json::integer(0)}, {"generated_ids", Json(Json::Array{})}, {"common_prefix", Json::integer(prefix)},
                {"committed_prefix", Json::integer(prefix)}, {"sampling", sampled()}, {"stop_id", Json()}, {"logits_available", Json(bool(prefix))},
                {"continuation_allowed", Json(bool(prefix))}, {"checkpoint_retained", Json::integer(begin ? 0 : 1)}})), false);
            continue;
        }
        if (body.find("reset")->boolean()) prefix = 0;
        if (broker && body.find("sampling")) { sampling = *body.find("sampling"); seed = body.find("seed")->unsigned_integer(); }
        prefix += uint32_t(body.find("tokens")->array().size());
        Json::Array generated;
        if (body.find("generate")->unsigned_integer()) {
            ++prefix; generated.push_back(Json::integer(42));
            emit(stamp(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(42)}, {"committed_prefix", Json::integer(prefix)}, {"text", Json("café 🦉")}})), fragment);
            if (mode == "slow-terminal") pause_ms(80);
        }
        auto fields = Json::Object{{"event", Json("done")}, {"generated_ids", Json(std::move(generated))}, {"committed_prefix", Json::integer(prefix)},
            {"stack_execution", Json(std::string(execution))}, {"sampling", broker ? sampled() : Json(Json::Object{{"valid", Json(true)}, {"entropy_prefix", Json::integer(prefix)}})}};
        if (broker) fields.emplace_back("checkpoint_retained", Json::integer(0));
        const auto done = stamp(Json(std::move(fields)));
        if (mode == "extra") write_all(STDOUT_FILENO, done.dump() + "\n" + ready_fixture(execution).dump() + "\n");
        else emit(done, fragment);
    }
    return 0;
}
struct Fixture {
    std::filesystem::path directory;
    OwnedWorkerOptions options;
    Fixture(const std::filesystem::path& executable, std::string_view mode) {
        auto pattern = (std::filesystem::temp_directory_path() / "tokspur-owned-worker-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end()); writable.push_back('\0');
        const auto* made = mkdtemp(writable.data()); if (!made) throw std::runtime_error("cannot create worker fixture directory"); directory = made;
        options.executable = directory / "tokspur-native";
        std::error_code linked; std::filesystem::create_hard_link(executable, options.executable, linked);
        if (linked) std::filesystem::copy_file(executable, options.executable);
        options.policy = directory / mode; options.library = directory / "owned.metallib";
        for (const auto& file : {options.policy, options.library}) { std::ofstream out(file); out << "owned CPU fixture\n"; }
        for (unsigned i = 0; i < 4; ++i) { const auto path = directory / ("shard-" + std::to_string(i)); { std::ofstream out(path); out << i; } options.shards.push_back(path); }
        options.startup_timeout = 8s; options.step_timeout = 3s; options.drain_timeout = 300ms;
    }
    ~Fixture() { std::error_code ignored; std::filesystem::remove_all(directory, ignored); }
    size_t commands() const { std::ifstream input(directory / "commands"); size_t count = 0; std::string line; while (std::getline(input, line)) ++count; return count; }
};
Json request(bool reset, bool generate = false) {
    return Json(Json::Object{{"tokens", Json(generate ? Json::Array{} : Json::Array{Json::integer(1)})},
        {"generate", Json::integer(generate ? 1 : 0)}, {"reset", Json(reset)}});
}
void ordinary(const std::filesystem::path& executable, std::string_view mode, std::string_view execution = "direct") {
    Fixture fixture(executable, mode);
    fixture.options.stack_execution = execution;
    OwnedWorker worker(fixture.options);
    const auto& ready = worker.ready();
    check(ready.find("stack_execution")->string() == execution, "fixed startup execution choice was not passed to target argv");
    check(ready.find("pid")->unsigned_integer() == ready.find("group")->unsigned_integer(), "target did not own its process group before ready");
    check(ready.find("guardian_pid")->unsigned_integer() == ready.find("session")->unsigned_integer(), "guardian was not an armed isolated session before target ready");
    check(worker.alive(), "ready worker not available");
    const auto one = worker.request(request(true), Clock::now() + 3s, [] { return false; });
    check(one.find("stack_execution")->string() == execution, "persistent child changed execution choice");
    check(one.find("committed_prefix")->unsigned_integer() == 1, "persistent input prefix differs");
    size_t tokens = 0;
    const auto two = worker.request(request(false, true), Clock::now() + 3s, [] { return false; }, [&](const Json& token) {
        ++tokens; check(token.find("text")->string() == "café 🦉", "fragmented UTF-8 worker token changed");
    });
    check(tokens == 1 && two.find("committed_prefix")->unsigned_integer() == 2 && fixture.commands() == 2, "persistent worker command/token accounting differs");
    worker.close(); check(!worker.alive(), "closed worker remained admissible"); worker.close();
}
void refusal_cases(const std::filesystem::path& executable) {
    for (const auto mode : {"eof-command", "partial-eof", "bad-command", "extra"}) {
        Fixture fixture(executable, mode); OwnedWorker worker(fixture.options);
        refuses([&] { static_cast<void>(worker.request(request(true), Clock::now() + 3s, [] { return false; })); });
        check(!worker.alive() && fixture.commands() == 1, "protocol failure retained admission or replayed a command");
        refuses([&] { static_cast<void>(worker.request(request(true), Clock::now() + 3s, [] { return false; })); });
        check(fixture.commands() == 1, "invalidated worker implicitly retried"); worker.close();
    }
    for (const auto mode : {"bad-ready", "eof-ready", "huge-ready"}) {
        Fixture fixture(executable, mode);
        refuses([&] { OwnedWorker invalid(fixture.options); });
        check(fixture.commands() == 0, "unsafe startup submitted inference commands");
    }
    for (const auto mode : {"stall", "stderr-stall"}) {
        Fixture fixture(executable, mode); OwnedWorker worker(fixture.options); const auto started = Clock::now();
        refuses([&] { static_cast<void>(worker.request(request(true), Clock::now() + 80ms, [] { return false; })); });
        check(Clock::now() - started < 5s && !worker.alive() && fixture.commands() == 1, "stalled worker exceeded cleanup bound or retried"); worker.close();
    }
}
void cancellation_cases(const std::filesystem::path& executable) {
    Fixture fixture(executable, "slow-terminal"); OwnedWorker worker(fixture.options);
    cancellation([&] { static_cast<void>(worker.request(request(true), Clock::now() + 3s, [] { return true; })); });
    check(fixture.commands() == 0 && worker.alive(), "pre-submission cancellation sent command bytes");
    static_cast<void>(worker.request(request(true), Clock::now() + 3s, [] { return false; }));
    std::atomic<bool> stopped{false};
    cancellation([&] {
        static_cast<void>(worker.request(request(false, true), Clock::now() + 3s, [&] { return stopped.load(); }, [&](const Json&) { stopped = true; }));
    });
    check(worker.alive() && fixture.commands() == 2, "cancelled single step did not drain or was replayed");
    refuses([&] { static_cast<void>(worker.request(request(false), Clock::now() + 3s, [] { return false; })); });
    check(fixture.commands() == 2, "cancelled context appended without explicit reset");
    static_cast<void>(worker.request(request(true), Clock::now() + 3s, [] { return false; }));
    refuses([&] {
        static_cast<void>(worker.request(request(false, true), Clock::now() + 3s, [] { return false; }, [](const Json&) { throw std::runtime_error("consumer disconnected"); }));
    });
    check(!worker.alive() && fixture.commands() == 4, "failed token callback was hidden as a reusable cancellation"); worker.close();
}
void bounded_admission(const std::filesystem::path& executable) {
    Fixture fixture(executable, "normal"); OwnedWorker worker(fixture.options);
    refuses([&] { static_cast<void>(worker.request(Json(Json::Object{{"tokens", Json(Json::Array{Json::integer(1), Json::integer(2)})},
        {"generate", Json::integer(0)}, {"reset", Json(true)}}), Clock::now() + 3s, [] { return false; })); });
    refuses([&] { static_cast<void>(worker.request(Json(Json::Object{{"tokens", Json(Json::Array{Json::integer(1)})},
        {"generate", Json::integer(1)}, {"reset", Json(true)}}), Clock::now() + 3s, [] { return false; })); });
    check(fixture.commands() == 0 && worker.alive(), "multi-step admission wrote bytes or damaged persistent child");
    static_cast<void>(worker.request(request(true), Clock::now() + 3s, [] { return false; })); worker.close();
    auto bad = fixture.options; bad.shards[3] = bad.shards[0]; refuses([&] { OwnedWorker invalid(bad); });
    bad = fixture.options; bad.executable = "/bin/sh"; refuses([&] { OwnedWorker invalid(bad); });
    for (const auto* execution : {"", "capture", "indirect-compatible", "unknown"}) {
        Fixture invalid_mode(executable, "normal"); invalid_mode.options.stack_execution = execution;
        refuses([&] { OwnedWorker invalid(invalid_mode.options); });
        check(!std::filesystem::exists(invalid_mode.directory / "worker.pid") && invalid_mode.commands() == 0,
            "unsupported execution choice launched a child or submitted work");
    }
    Fixture loading(executable, "stall-ready"); const auto started = Clock::now();
    loading.options.startup_cancelled = [&] { return Clock::now() - started > 80ms; };
    refuses([&] { OwnedWorker invalid(loading.options); });
    check(Clock::now() - started < 5s && loading.commands() == 0, "startup cancellation did not bound readiness/cleanup");
}
void parent_death(const std::filesystem::path& executable, bool before_ready) {
    Fixture fixture(executable, before_ready ? "stall-ready" : "normal");
    int status_pipe[2]; check(pipe(status_pipe) == 0, "cannot create parent-death fixture channel");
    const auto controller = fork(); check(controller >= 0, "cannot fork parent-death fixture controller");
    if (!controller) {
        close(status_pipe[0]);
        try {
            OwnedWorker worker(fixture.options);
            write_all(status_pipe[1], "R");
            for (;;) pause();
        } catch (...) { _exit(80); }
    }
    struct ControllerGuard {
        pid_t pid;
        ~ControllerGuard() {
            if (pid > 0) { kill(pid, SIGKILL); int status = 0; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} }
        }
    } controller_guard{controller};
    close(status_pipe[1]);
    pid_t target = -1; const auto deadline = Clock::now() + 8s;
    if (!before_ready) { struct pollfd watch{status_pipe[0], POLLIN, 0}; check(poll(&watch, 1, 8000) > 0, "controller did not reach ready"); char ready = 0; check(read(status_pipe[0], &ready, 1) == 1 && ready == 'R', "controller ready framing differs"); }
    while (Clock::now() < deadline && target <= 1) { std::ifstream pid(fixture.directory / "worker.pid"); pid >> target; if (target <= 1) pause_ms(10); }
    check(target > 1, "worker never crossed the pre-exec armed gate");
    check(kill(controller, SIGKILL) == 0, "cannot terminate exact fixture controller");
    int status = 0; check(waitpid(controller, &status, 0) == controller && WIFSIGNALED(status), "controller death was not verified"); controller_guard.pid = -1; close(status_pipe[0]);
    bool gone = false;
    for (unsigned i = 0; i < 300; ++i) { if (kill(target, 0) < 0 && errno == ESRCH) { gone = true; break; } pause_ms(10); }
    check(gone, "armed guardian did not reap target after parent death");
}
Json begin_control(bool restore = false) {
    return Json(Json::Object{{"op", Json("branch_begin")}, {"prompt_ids", Json(Json::Array{Json::integer(1), Json::integer(2)})},
        {"generation_requested", Json(true)}, {"allow_restore", Json(restore)}, {"seed", Json::integer(7)},
        {"sampling", Json(Json::Object{{"temperature", Json::integer(0)}, {"top_k", Json::integer(0)}, {"top_p", Json::integer(1)}, {"min_p", Json::integer(0)}})}});
}
void broker_cases(const std::filesystem::path& executable) {
    const auto prepared = [&](Fixture& fixture) { fixture.options.checkpoint_slots = 2; fixture.options.checkpoint_budget_bytes = 1 << 20; };
    const Json capture(Json::Object{{"op", Json("branch_capture")}});
    {
        Fixture fixture(executable, "broker-normal"); prepared(fixture); OwnedWorker worker(fixture.options);
        refuses([&] { static_cast<void>(worker.request(begin_control(true), Clock::now() + 3s, [] { return false; })); });
        for (const auto body : {R"({"tokens":[1,2],"generate":0,"reset":true})",
            R"({"tokens":[1],"generate":1,"reset":true})",
            R"({"op":"branch_capture","tokens":[1]})"})
            refuses([&] { static_cast<void>(worker.request(parse_json(body), Clock::now() + 3s, [] { return false; })); });
        check(fixture.commands() == 0 && worker.alive(), "unleased reuse submitted control bytes");
        const auto began = worker.request(begin_control(), Clock::now() + 3s, [] { return false; });
        check(began.find("event")->string() == "control_done" && began.find("committed_prefix")->unsigned_integer() == 0 &&
            began.find("generated_ids")->array().empty(), "zero-step cold begin performed model work");
        static_cast<void>(worker.request(request(false), Clock::now() + 3s, [] { return false; }));
        static_cast<void>(worker.request(capture, Clock::now() + 3s, [] { return false; }));
        const auto before = fixture.commands();
        cancellation([&] { static_cast<void>(worker.request(begin_control(), Clock::now() + 3s, [] { return true; })); });
        refuses([&] { static_cast<void>(worker.request(capture, Clock::now() + 3s, [] { return false; })); });
        refuses([&] { static_cast<void>(worker.request(begin_control(true), Clock::now() + 3s, [] { return false; })); });
        check(fixture.commands() == before && worker.alive(), "pre-control cancellation bypassed cold-reset requirement");
        static_cast<void>(worker.request(begin_control(), Clock::now() + 3s, [] { return false; }));
        worker.close();
    }
    {
        Fixture fixture(executable, "broker-slow-control"); prepared(fixture); OwnedWorker worker(fixture.options);
        const auto start = Clock::now();
        cancellation([&] { static_cast<void>(worker.request(begin_control(), Clock::now() + 3s, [&] { return Clock::now() - start > 10ms; })); });
        const auto count = fixture.commands();
        refuses([&] { static_cast<void>(worker.request(request(false), Clock::now() + 3s, [] { return false; })); });
        check(fixture.commands() == count && worker.alive(), "drained control cancellation left an append lease reusable");
        const auto reset = worker.request(begin_control(), Clock::now() + 3s, [] { return false; });
        check(reset.find("lease_id")->unsigned_integer() == 2, "control recovery reused the cancelled lease"); worker.close();
    }
    for (const auto mode : {"broker-stale-epoch", "broker-stale-lease", "broker-control-token"}) {
        Fixture fixture(executable, mode); prepared(fixture); OwnedWorker worker(fixture.options);
        refuses([&] { static_cast<void>(worker.request(begin_control(), Clock::now() + 3s, [] { return false; })); });
        check(!worker.alive(), "ambiguous control reply retained worker admission"); worker.close();
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "target") return fake_worker(argc, argv);
    try {
        const auto executable = std::filesystem::canonical(argv[0]);
        ordinary(executable, "normal"); ordinary(executable, "fragment");
        ordinary(executable, "normal", "recurrent-replay"); ordinary(executable, "fragment", "recurrent-replay");
        refusal_cases(executable); cancellation_cases(executable); bounded_admission(executable);
        parent_death(executable, false); parent_death(executable, true);
        broker_cases(executable);
        std::cout << "Owned worker CPU fixture checks passed: " << checks << '\n';
    } catch (const std::exception& error) { std::cerr << "Owned worker check failed: " << error.what() << '\n'; return 1; }
}
