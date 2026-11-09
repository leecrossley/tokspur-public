// Original bounded HTTP/1.1 and SSE transport. Inputs are RFC 9112 framing,
// WHATWG event-stream semantics, and platform socket/poll API declarations.
// No network framework, external server or inference implementation is used.
#include "tokspur/http.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstring>
#include <limits>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tokspur::http {
namespace {
[[noreturn]] void system_error(const char* operation, bool disconnected = false) {
    throw Error(0, std::string(operation) + ": " + std::strerror(errno), disconnected);
}
void validate_limits(Limits l) {
    if (l.header_bytes < 16 || l.header_bytes > 1024 * 1024 || !l.header_line_bytes ||
        l.header_line_bytes > l.header_bytes || !l.header_count || l.header_count > 256 ||
        !l.target_bytes || l.target_bytes > 8192 || l.body_bytes > 64 * 1024 * 1024 ||
        l.event_bytes < 8 || l.event_bytes > 8 * 1024 * 1024)
        throw Error(500, "Invalid bounded HTTP transport limits");
}
void prepare_fd(int fd) {
    if (fd < 0) throw Error(0, "Invalid adopted HTTP socket");
    int type = 0; socklen_t type_bytes = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_bytes) < 0) system_error("Inspect HTTP socket type");
    if (type_bytes != sizeof(type) || type != SOCK_STREAM) throw Error(0, "HTTP transport requires a stream socket");
    const auto flags = fcntl(fd, F_GETFL, 0), descriptor = fcntl(fd, F_GETFD, 0);
    if (flags < 0 || descriptor < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0 ||
        fcntl(fd, F_SETFD, descriptor | FD_CLOEXEC) < 0) system_error("Configure HTTP socket");
#ifdef SO_NOSIGPIPE
    const int yes = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) < 0) system_error("Protect HTTP socket from SIGPIPE");
#endif
}
void wait_fd(int fd, short events, Deadline deadline) {
    for (;;) {
        const auto now = Clock::now();
        if (now >= deadline) throw Error(408, "HTTP transport deadline exceeded");
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const auto millis = std::min<int64_t>(INT_MAX, remaining.count() + 1);
        pollfd item{fd, events, 0};
        const auto result = poll(&item, 1, static_cast<int>(millis));
        if (result < 0) { if (errno == EINTR) continue; system_error("Poll HTTP socket", true); }
        if (!result) continue;
        if (item.revents & POLLNVAL) throw Error(0, "HTTP socket is invalid", true);
        // POLLIN can coexist with HUP while already-received bytes remain.
        if (item.revents & events) return;
        if (item.revents & (POLLERR | POLLHUP)) throw Error(0, "HTTP peer disconnected", true);
    }
}
char lower(char c) { return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c; }
bool token_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
        std::string_view("!#$%&'*+-.^_`|~").find(char(c)) != std::string_view::npos;
}
std::string lower_text(std::string_view value) {
    std::string result(value); for (auto& c : result) c = lower(c); return result;
}
std::string_view trimmed(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.remove_suffix(1);
    return value;
}
void crlf_only(std::string_view value) {
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\n' && (!i || value[i - 1] != '\r')) throw Error(400, "Bare LF in HTTP headers");
        if (value[i] == '\r' && i + 1 < value.size() && value[i + 1] != '\n') throw Error(400, "Bare CR in HTTP headers");
    }
}
std::string_view reason(int status) {
    switch (status) {
        case 200: return "OK"; case 201: return "Created"; case 202: return "Accepted";
        case 400: return "Bad Request"; case 401: return "Unauthorized"; case 403: return "Forbidden";
        case 404: return "Not Found"; case 405: return "Method Not Allowed"; case 408: return "Request Timeout";
        case 409: return "Conflict"; case 411: return "Length Required"; case 413: return "Content Too Large";
        case 415: return "Unsupported Media Type"; case 417: return "Expectation Failed";
        case 422: return "Unprocessable Content"; case 429: return "Too Many Requests";
        case 431: return "Request Header Fields Too Large"; case 500: return "Internal Server Error";
        case 501: return "Not Implemented"; case 502: return "Bad Gateway"; case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout"; case 505: return "HTTP Version Not Supported";
        default: throw Error(500, "Unregistered HTTP response status");
    }
}
size_t content_length(std::string_view value, size_t maximum) {
    if (value.empty() || (value.size() > 1 && value.front() == '0')) throw Error(400, "Noncanonical Content-Length");
    size_t result = 0;
    for (const auto c : value) {
        if (c < '0' || c > '9') throw Error(400, "Invalid Content-Length");
        const auto digit = size_t(c - '0');
        if (result > (std::numeric_limits<size_t>::max() - digit) / 10) throw Error(413, "Content-Length overflows");
        result = result * 10 + digit;
    }
    if (result > maximum) throw Error(413, "HTTP body limit exceeded");
    return result;
}
Request parse_headers(std::string_view head, Limits limits, size_t& body_length) {
    crlf_only(head);
    std::vector<std::string_view> lines;
    for (size_t start = 0;;) {
        const auto end = head.find("\r\n", start);
        const auto line = head.substr(start, end == std::string_view::npos ? end : end - start);
        if (line.size() > limits.header_line_bytes) throw Error(431, "HTTP header line limit exceeded");
        lines.push_back(line);
        if (lines.size() > limits.header_count + 1) throw Error(431, "HTTP header count limit exceeded");
        if (end == std::string_view::npos) break;
        start = end + 2;
    }
    if (lines.empty()) throw Error(400, "Missing HTTP request line");
    const auto first = lines[0]; const auto first_space = first.find(' ');
    if (first_space == std::string_view::npos || !first_space) throw Error(400, "Invalid HTTP request line");
    const auto second_space = first.find(' ', first_space + 1);
    if (second_space == std::string_view::npos || second_space == first_space + 1 ||
        first.find(' ', second_space + 1) != std::string_view::npos) throw Error(400, "Ambiguous HTTP request line");
    const auto method = first.substr(0, first_space), path = first.substr(first_space + 1, second_space - first_space - 1);
    for (const unsigned char c : method) if (!token_char(c)) throw Error(400, "Invalid HTTP method");
    if (method != "GET" && method != "POST") throw Error(405, "Only GET and POST are supported");
    if (first.substr(second_space + 1) != "HTTP/1.1") throw Error(505, "Only HTTP/1.1 is supported");
    if (path.empty() || path.front() != '/' || path.size() > limits.target_bytes) throw Error(400, "Bounded origin-form HTTP target required");
    for (const unsigned char c : path) if (c < 0x21 || c > 0x7e || c == '#' || c == '\\') throw Error(400, "Invalid HTTP target byte");
    Request result; result.method = method; result.path = path;
    bool length_present = false;
    for (size_t i = 1; i < lines.size(); ++i) {
        const auto line = lines[i]; const auto colon = line.find(':');
        if (line.empty() || line.front() == ' ' || line.front() == '\t' || colon == std::string_view::npos || !colon)
            throw Error(400, "Invalid or folded HTTP field");
        const auto name = line.substr(0, colon);
        for (const unsigned char c : name) if (!token_char(c)) throw Error(400, "Invalid HTTP field name");
        const auto value = trimmed(line.substr(colon + 1));
        for (const unsigned char c : value) if ((c < 0x20 && c != '\t') || c > 0x7e) throw Error(400, "Invalid HTTP field value");
        const auto normalized = lower_text(name);
        for (const auto& field : result.headers) if (field.first == normalized) throw Error(400, "Duplicate HTTP field");
        if (normalized == "transfer-encoding" || normalized == "trailer" || normalized == "upgrade" || normalized == "proxy-connection")
            throw Error(400, "Unsupported ambiguous HTTP framing/upgrade field");
        if (normalized == "expect") throw Error(417, "HTTP expectations are unsupported");
        if (normalized == "connection" && lower_text(value) != "close" && lower_text(value) != "keep-alive")
            throw Error(400, "Unsupported HTTP connection option");
        if (normalized == "content-length") { length_present = true; body_length = content_length(value, limits.body_bytes); }
        result.headers.emplace_back(normalized, value);
    }
    const auto host = result.header("host");
    if (host.empty()) throw Error(400, "Exactly one nonempty Host field required");
    for (const unsigned char c : host) if (c <= 0x20 || c == ',' || c == '/' || c == '\\' || c == '@') throw Error(400, "Invalid Host field");
    if (method == "POST" && !length_present) throw Error(411, "POST requires Content-Length");
    if (method == "GET" && body_length) throw Error(400, "GET request bodies are unsupported");
    return result;
}
std::string chunk_size(size_t count) {
    std::array<char, 2 * sizeof(size_t)> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), count, 16);
    if (converted.ec != std::errc{}) throw Error(500, "Cannot encode bounded SSE chunk size");
    return std::string(buffer.data(), converted.ptr);
}
} // namespace

Error::Error(int status, std::string message, bool disconnected)
    : std::runtime_error(std::move(message)), status_(status), disconnected_(disconnected) {}
std::string_view Request::header(std::string_view name) const noexcept {
    for (const auto& item : headers) {
        if (item.first.size() != name.size()) continue;
        bool same = true;
        for (size_t i = 0; i < name.size(); ++i) if (item.first[i] != lower(name[i])) { same = false; break; }
        if (same) return item.second;
    }
    return {};
}
Connection::Connection(int owned_fd, Limits limits) : fd_(owned_fd), limits_(limits) {
    try { validate_limits(limits); prepare_fd(fd_); } catch (...) { close(); throw; }
}
Connection::~Connection() { close(); }
Connection::Connection(Connection&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), limits_(other.limits_), read_started_(other.read_started_),
      request_complete_(other.request_complete_), poisoned_(other.poisoned_), response_(other.response_) {}
Connection& Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        close(); fd_ = std::exchange(other.fd_, -1); limits_ = other.limits_; read_started_ = other.read_started_;
        request_complete_ = other.request_complete_; poisoned_ = other.poisoned_; response_ = other.response_;
    }
    return *this;
}
void Connection::close() noexcept { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }
bool Connection::disconnected() noexcept {
    if (fd_ < 0 || poisoned_) return true;
    pollfd item{fd_, POLLIN, 0};
    const auto ready = poll(&item, 1, 0);
    if (ready < 0) return errno != EINTR;
    if (!ready) return false;
    if (item.revents & POLLNVAL) return true;
    if (item.revents & POLLIN) {
        char byte = 0;
        const auto count = recv(fd_, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        if (count > 0) { if (request_complete_) poisoned_ = true; return poisoned_; }
        if (!count) return true;
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
    }
    return (item.revents & (POLLERR | POLLHUP)) != 0;
}
Request Connection::read_request(Deadline deadline) {
    if (fd_ < 0 || read_started_ || response_ != Response::idle) throw Error(400, "HTTP connection accepts exactly one request");
    read_started_ = true;
    std::array<char, 8192> buffer{};
    std::string wire;
    size_t end = std::string::npos;
    while (end == std::string::npos) {
        wait_fd(fd_, POLLIN, deadline);
        const auto count = recv(fd_, buffer.data(), buffer.size(), 0);
        if (count < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue; system_error("Read HTTP headers", true); }
        if (!count) throw Error(wire.empty() ? 0 : 400, "HTTP peer closed before complete headers", true);
        wire.append(buffer.data(), size_t(count));
        end = wire.find("\r\n\r\n");
        if (end == std::string::npos) {
            crlf_only(wire);
            if (wire.size() >= limits_.header_bytes) throw Error(431, "HTTP header byte limit exceeded");
        } else if (end + 4 > limits_.header_bytes) throw Error(431, "HTTP header byte limit exceeded");
    }
    size_t length = 0;
    auto result = parse_headers(std::string_view(wire).substr(0, end), limits_, length);
    result.body = wire.substr(end + 4);
    if (result.body.size() > length) throw Error(400, "Unexpected bytes after HTTP request body");
    result.body.reserve(length);
    while (result.body.size() < length) {
        wait_fd(fd_, POLLIN, deadline);
        const auto count = recv(fd_, buffer.data(), std::min(buffer.size(), length - result.body.size()), 0);
        if (count < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue; system_error("Read HTTP body", true); }
        if (!count) throw Error(400, "HTTP peer closed before complete body", true);
        result.body.append(buffer.data(), size_t(count));
    }
    if (Clock::now() >= deadline) throw Error(408, "HTTP request deadline exceeded");
    char extra = 0;
    const auto tail = recv(fd_, &extra, 1, MSG_PEEK | MSG_DONTWAIT);
    if (tail > 0) { poisoned_ = true; throw Error(400, "HTTP pipelining/trailing bytes refused"); }
    if (tail < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) system_error("Inspect HTTP trailing bytes", true);
    request_complete_ = true;
    return result;
}
void Connection::write_all(std::string_view data, Deadline deadline) {
    while (!data.empty()) {
        if (fd_ < 0 || disconnected()) throw Error(0, "HTTP peer disconnected or sent unexpected bytes", true);
        wait_fd(fd_, POLLOUT, deadline);
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const auto count = send(fd_, data.data(), std::min<size_t>(65536, data.size()), flags);
        if (count < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue; system_error("Write HTTP response", true); }
        if (!count) throw Error(0, "HTTP response write made no progress", true);
        data.remove_prefix(size_t(count));
    }
}
void Connection::send_json(int status, std::string_view body, Deadline deadline) {
    if (response_ != Response::idle) throw Error(500, "HTTP response already started");
    if (body.size() > limits_.body_bytes) throw Error(500, "JSON response exceeds configured body limit");
    const auto message = reason(status);
    const auto extra = status == 405 ? "Allow: GET, POST\r\n" : status == 401 ? "WWW-Authenticate: Bearer realm=\"tokspur\"\r\n" : (status == 429 || status == 503) ? "Retry-After: 5\r\n" : "";
    const auto header = "HTTP/1.1 " + std::to_string(status) + " " + std::string(message) +
        "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: " + std::to_string(body.size()) +
        "\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n" + extra + "\r\n";
    response_ = Response::json;
    write_all(header, deadline); write_all(body, deadline); response_ = Response::finished;
}
void Connection::begin_sse(Deadline deadline) {
    if (response_ != Response::idle) throw Error(500, "HTTP response already started");
    response_ = Response::sse;
    write_all("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n", deadline);
}
void Connection::send_sse_data(std::string_view data, Deadline deadline) {
    if (response_ != Response::sse) throw Error(500, "SSE response is not active");
    if (data.size() > limits_.event_bytes || data.find('\0') != std::string_view::npos || data.find('\r') != std::string_view::npos)
        throw Error(500, "Unsafe or excessive SSE data");
    std::string event;
    for (size_t start = 0;;) {
        const auto end = data.find('\n', start);
        const auto line = data.substr(start, end == std::string_view::npos ? end : end - start);
        if (event.size() > limits_.event_bytes - 8 || line.size() > limits_.event_bytes - 8 - event.size())
            throw Error(500, "SSE framed event limit exceeded");
        event += "data: "; event.append(line); event += '\n';
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    event += '\n';
    const auto prefix = chunk_size(event.size()) + "\r\n";
    write_all(prefix, deadline); write_all(event, deadline); write_all("\r\n", deadline);
}
void Connection::finish_sse(Deadline deadline) {
    if (response_ != Response::sse) throw Error(500, "SSE response is not active");
    write_all("0\r\n\r\n", deadline); response_ = Response::finished;
}

Listener::Listener(uint16_t port, int backlog) {
    if (backlog < 1 || backlog > 128) throw Error(500, "Bounded HTTP listener backlog required");
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) system_error("Create loopback HTTP socket");
    try {
        prepare_fd(fd_);
        const int reuse = 1;
        if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) system_error("Allow loopback HTTP address reuse");
        sockaddr_in address{}; address.sin_family = AF_INET;
#ifdef __APPLE__
        address.sin_len = sizeof(address);
#endif
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
        if (bind(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) system_error("Bind loopback HTTP socket");
        if (listen(fd_, backlog) < 0) system_error("Listen on loopback HTTP socket");
        socklen_t bytes = sizeof(address);
        if (getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &bytes) < 0 || bytes != sizeof(address)) system_error("Inspect loopback HTTP address");
        if (address.sin_family != AF_INET || address.sin_addr.s_addr != htonl(INADDR_LOOPBACK)) throw Error(0, "HTTP listener is not loopback-only");
        port_ = ntohs(address.sin_port);
    } catch (...) { close(); throw; }
}
Listener::~Listener() { close(); }
Listener::Listener(Listener&& other) noexcept : fd_(std::exchange(other.fd_, -1)), port_(std::exchange(other.port_, 0)) {}
Listener& Listener::operator=(Listener&& other) noexcept {
    if (this != &other) { close(); fd_ = std::exchange(other.fd_, -1); port_ = std::exchange(other.port_, 0); }
    return *this;
}
void Listener::close() noexcept { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } port_ = 0; }
Connection Listener::accept(Deadline deadline, Limits limits) {
    validate_limits(limits);
    if (fd_ < 0) throw Error(0, "HTTP listener is closed");
    for (;;) {
        wait_fd(fd_, POLLIN, deadline);
        sockaddr_in peer{}; socklen_t bytes = sizeof(peer);
        const auto accepted = ::accept(fd_, reinterpret_cast<sockaddr*>(&peer), &bytes);
        if (accepted < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue; system_error("Accept loopback HTTP connection"); }
        if (bytes != sizeof(peer) || peer.sin_family != AF_INET || (ntohl(peer.sin_addr.s_addr) >> 24) != 127) {
            ::close(accepted); throw Error(0, "Refused non-loopback HTTP peer");
        }
        return Connection(accepted, limits);
    }
}
} // namespace tokspur::http
