#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tokspur::http {
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;

// Deliberately narrow HTTP/1.1 service: GET/POST, one request per connection,
// mandatory Host, canonical Content-Length for POST, no transfer-coded requests,
// Expect/upgrade, duplicate fields, pipelining or persistent connections.
// Auth, JSON, route policy and admission belong to the owned caller.
struct Limits {
    size_t header_bytes = 32768;
    size_t header_line_bytes = 8192;
    size_t header_count = 64;
    size_t target_bytes = 4096;
    size_t body_bytes = 8 * 1024 * 1024;
    size_t event_bytes = 1024 * 1024;
};

class Error : public std::runtime_error {
public:
    Error(int status, std::string message, bool disconnected = false);
    [[nodiscard]] int status() const noexcept { return status_; }
    [[nodiscard]] bool disconnected() const noexcept { return disconnected_; }
private:
    int status_;
    bool disconnected_;
};

struct Request {
    std::string method, path;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body; // Unmodified bytes; the caller validates JSON/UTF-8.
    [[nodiscard]] std::string_view header(std::string_view name) const noexcept;
};

class Connection {
public:
    // Adopts fd even when constructor validation/setup fails. Intended for an
    // accepted socket or a model-free socketpair. Sets nonblocking+CLOEXEC.
    explicit Connection(int owned_fd, Limits limits = {});
    ~Connection();
    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    [[nodiscard]] int native_handle() const noexcept { return fd_; }
    [[nodiscard]] Request read_request(Deadline deadline);
    void send_json(int status, std::string_view body, Deadline deadline);
    void begin_sse(Deadline deadline);
    // Each call writes one HTTP chunk carrying a complete SSE data event.
    // LF data is split into data fields. CR/NUL are refused to prevent injection;
    // callers normally pass a complete owned-JSON serialization or [DONE].
    void send_sse_data(std::string_view data, Deadline deadline);
    void finish_sse(Deadline deadline); // Terminal HTTP zero chunk, not [DONE].
    // Nonconsuming advisory check. No readable data is not a disconnect.
    // EOF, socket failure, or unexpected bytes after the admitted request cancel
    // the connection. This service does not support request-side half-close.
    [[nodiscard]] bool disconnected() noexcept;
    void close() noexcept;
private:
    int fd_ = -1;
    Limits limits_{};
    bool read_started_ = false, request_complete_ = false, poisoned_ = false;
    enum class Response { idle, json, sse, finished } response_ = Response::idle;
    void write_all(std::string_view data, Deadline deadline);
};

class Listener {
public:
    // IPv4 127.0.0.1 only. Port zero is allowed for explicit tests. No port reuse
    // or eviction of an existing listener. No inference or background threads.
    explicit Listener(uint16_t port, int backlog = 8);
    ~Listener();
    Listener(Listener&& other) noexcept;
    Listener& operator=(Listener&& other) noexcept;
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    [[nodiscard]] uint16_t port() const noexcept { return port_; }
    [[nodiscard]] int native_handle() const noexcept { return fd_; }
    // An idle accept deadline throws Error(408,...,false), so a caller can use
    // short bounded polls for shutdown. All accepted fds are owned by Connection.
    [[nodiscard]] Connection accept(Deadline deadline, Limits limits = {});
    void close() noexcept;
private:
    int fd_ = -1;
    uint16_t port_ = 0;
};
} // namespace tokspur::http
