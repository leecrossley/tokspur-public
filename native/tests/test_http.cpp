#include "tokspur/http.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

// Original generated wire/socket fixtures only. No model, authentication secret,
// external server or engine is launched. All waits have short bounded deadlines.
namespace {
using namespace tokspur::http;
size_t checks = 0;
void need(bool condition, const char* message) {
    ++checks; if (!condition) throw std::runtime_error(message);
}
Deadline soon() { return Clock::now() + std::chrono::seconds(1); }
void rejects(const std::function<void()>& operation, int status, bool disconnected = false) {
    ++checks;
    try { operation(); }
    catch (const Error& error) {
        need(error.status() == status, "HTTP rejection status differs");
        if (disconnected) need(error.disconnected(), "HTTP disconnect was not identified");
        return;
    }
    throw std::runtime_error("Invalid HTTP operation accepted");
}
struct Pair {
    int server = -1, peer = -1;
    Pair() {
        std::array<int, 2> fds{};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds.data()) < 0) throw std::runtime_error("Cannot create HTTP test socketpair");
        server = fds[0]; peer = fds[1];
#ifdef SO_NOSIGPIPE
        const int yes = 1;
        if (setsockopt(peer, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) < 0) {
            ::close(server); ::close(peer); server = peer = -1;
            throw std::runtime_error("Cannot protect test socket SIGPIPE");
        }
#endif
    }
    ~Pair() { if (server >= 0) ::close(server); if (peer >= 0) ::close(peer); }
    int take() { return std::exchange(server, -1); }
    void close_peer() { if (peer >= 0) { ::close(peer); peer = -1; } }
};
void put(int fd, std::string_view data) {
    while (!data.empty()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const auto count = send(fd, data.data(), data.size(), flags);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Test HTTP peer write failed");
        data.remove_prefix(size_t(count));
    }
}
std::string get_all(int fd) {
    std::string result;
    const auto deadline = soon();
    std::array<char, 4096> bytes{};
    for (;;) {
        need(Clock::now() < deadline, "Test peer read did not reach bounded EOF");
        pollfd item{fd, POLLIN, 0};
        const auto ready = poll(&item, 1, 20);
        if (ready < 0 && errno == EINTR) continue;
        need(ready >= 0, "Test peer poll failed");
        if (!ready) continue;
        const auto count = recv(fd, bytes.data(), bytes.size(), MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        need(count >= 0, "Test peer response read failed");
        if (!count) return result;
        need(result.size() + size_t(count) <= 1024 * 1024, "Test HTTP response exceeded fixture bound");
        result.append(bytes.data(), size_t(count));
    }
}
std::string post(std::string_view body) {
    return "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:18081\r\nContent-Type: application/json\r\nAuthorization: Bearer fixture-only\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
}
std::string_view get_request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";

void fragmented_bytes() {
    const std::string utf8 = "{\"text\":\"\xf0\x9f\xa7\xa0\xe9\x9b\xaa\"}";
    const auto wire = post(utf8);
    for (size_t split = 0; split <= wire.size(); ++split) {
        Pair pair; Connection connection(pair.take());
        std::exception_ptr sender_error;
        std::jthread sender([&] {
            try { put(pair.peer, std::string_view(wire).substr(0, split)); std::this_thread::yield(); put(pair.peer, std::string_view(wire).substr(split)); }
            catch (...) { sender_error = std::current_exception(); }
        });
        const auto request = connection.read_request(soon()); sender.join();
        if (sender_error) std::rethrow_exception(sender_error);
        need(request.method == "POST" && request.path == "/v1/chat/completions" && request.body == utf8,
            "Fragmented HTTP request changed raw UTF-8 body bytes");
        need(request.header("Authorization") == "Bearer fixture-only" && request.header("CONTENT-TYPE") == "application/json" &&
            request.header("missing").empty(), "HTTP field normalization/lookup differs");
        need(!connection.disconnected(), "No-readable-data incorrectly treated as disconnect");
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, 400);
    }
    Pair pair; Connection connection(pair.take());
    const std::string binary = std::string("a\0b", 3) + "\xf0\x9f\xa7\xa0";
    put(pair.peer, post(binary));
    need(connection.read_request(soon()).body == binary, "Transport changed raw body NUL/UTF-8 before caller JSON validation");
}

void malformed_requests() {
    const std::vector<std::pair<std::string, int>> cases{
        {"GET / HTTP/1.1\nHost: a\n\n", 400},
        {"GET / HTTP/1.1\rHost: a\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost:\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nhOsT: a\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost : a\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\n X-Test: value\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a b\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nX: a\r\nX: b\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\n\r\n", 411},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0,0\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: +1\r\n\r\nx", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: -1\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 00\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 184467440737095516160\r\n\r\n", 413},
        {"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: identity\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nExpect: 100-continue\r\n\r\n", 417},
        {"GET / HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nConnection: content-length\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nProxy-Connection: keep-alive\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nTrailer: X-Foo\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nx", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n\r\nx", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nxy", 400},
        {std::string(get_request) + std::string(get_request), 400},
        {"GET  / HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {"GET / HTTP/1.0\r\nHost: a\r\n\r\n", 505},
        {"PUT / HTTP/1.1\r\nHost: a\r\n\r\n", 405},
        {"GET http://localhost/ HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {"GET /#fragment HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {"GET /\\other HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {std::string("GET / HTTP/1.1\r\nHost: a\r\nX: a") + std::string("\0", 1) + "b\r\n\r\n", 400}
    };
    for (const auto& [wire, status] : cases) {
        Pair pair; Connection connection(pair.take()); put(pair.peer, wire);
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, status);
    }
    for (const auto wire : {std::string_view("GET / HTTP/1.1\r\nHost: a\r\n"),
        std::string_view("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\n\r\nx")}) {
        Pair pair; Connection connection(pair.take()); put(pair.peer, wire); shutdown(pair.peer, SHUT_WR);
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, 400, true);
    }
    Pair pair; Connection connection(pair.take()); pair.close_peer();
    rejects([&] { static_cast<void>(connection.read_request(soon())); }, 0, true);
}

void limits_and_deadlines() {
    for (int variant = 0; variant < 5; ++variant) {
        Limits limits; std::string wire; int status = 431;
        if (variant == 0) { limits.header_bytes = 64; limits.header_line_bytes = 64; wire = "GET / HTTP/1.1\r\nHost: a\r\nX: " + std::string(70, 'x'); }
        if (variant == 1) { limits.header_line_bytes = 16; wire = "GET / HTTP/1.1\r\nHost: a\r\nX: " + std::string(17, 'x') + "\r\n\r\n"; }
        if (variant == 2) { limits.header_count = 1; wire = "GET / HTTP/1.1\r\nHost: a\r\nX: x\r\n\r\n"; }
        if (variant == 3) { limits.target_bytes = 2; wire = "GET /long HTTP/1.1\r\nHost: a\r\n\r\n"; status = 400; }
        if (variant == 4) { limits.body_bytes = 2; wire = post("123"); status = 413; }
        Pair pair; Connection connection(pair.take(), limits); put(pair.peer, wire);
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, status);
    }
    {
        Pair pair; Connection connection(pair.take());
        rejects([&] { static_cast<void>(connection.read_request(Clock::now())); }, 408);
    }
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, "GET / HTTP/1.1\r\n");
        rejects([&] { static_cast<void>(connection.read_request(Clock::now() + std::chrono::milliseconds(2))); }, 408);
    }
    {
        Pair pair; Limits invalid; invalid.event_bytes = 0; const auto fd = pair.take();
        rejects([&] { Connection adopted(fd, invalid); }, 500);
        need(fcntl(fd, F_GETFD) < 0 && errno == EBADF, "Failed HTTP constructor leaked adopted fd");
    }
    {
        Pair pair;
        const int small = 1024;
        need(setsockopt(pair.server, SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)) == 0, "Cannot bound backpressure fixture socket");
        Connection connection(pair.take());
        connection.begin_sse(soon());
        rejects([&] { connection.send_sse_data(std::string(512 * 1024, 'x'), Clock::now() + std::chrono::milliseconds(3)); }, 408);
    }
}

std::vector<std::string> decode_chunks(std::string_view body) {
    std::vector<std::string> chunks;
    for (;;) {
        const auto end = body.find("\r\n"); need(end != std::string_view::npos && end > 0, "Missing HTTP chunk size");
        size_t count = 0;
        const auto parsed = std::from_chars(body.data(), body.data() + end, count, 16);
        need(parsed.ec == std::errc{} && parsed.ptr == body.data() + end, "Invalid HTTP hex chunk size");
        body.remove_prefix(end + 2);
        if (!count) { need(body == "\r\n", "Invalid terminal HTTP zero chunk/trailer"); return chunks; }
        need(count <= body.size() && body.size() - count >= 2 && body.substr(count, 2) == "\r\n", "HTTP chunk byte length/framing differs");
        chunks.emplace_back(body.substr(0, count)); body.remove_prefix(count + 2);
        need(chunks.size() <= 8, "Bounded test chunk count exceeded");
    }
}
void response_wire() {
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, get_request); static_cast<void>(connection.read_request(soon()));
        const std::string body = "{\"text\":\"\xe9\x9b\xaa\"}";
        connection.send_json(200, body, soon());
        rejects([&] { connection.send_json(200, "{}", soon()); }, 500);
        rejects([&] { connection.begin_sse(soon()); }, 500);
        connection.close(); const auto raw = get_all(pair.peer); const auto end = raw.find("\r\n\r\n");
        need(raw.starts_with("HTTP/1.1 200 OK\r\n") && end != std::string::npos && raw.substr(end + 4) == body, "JSON HTTP status/body changed");
        need(raw.find("Content-Length: " + std::to_string(body.size()) + "\r\n") != std::string::npos &&
            raw.find("Connection: close\r\n") != std::string::npos && raw.find("Access-Control-") == std::string::npos &&
            raw.find("Transfer-Encoding:") == std::string::npos, "JSON HTTP response framing or private policy differs");
    }
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, get_request); static_cast<void>(connection.read_request(soon()));
        connection.begin_sse(soon());
        const std::string data = "\xf0\x9f\xa7\xa0\nsecond\n";
        connection.send_sse_data(data, soon()); connection.send_sse_data("", soon()); connection.send_sse_data("[DONE]", soon());
        rejects([&] { connection.send_sse_data("bad\rline", soon()); }, 500);
        rejects([&] { connection.send_sse_data(std::string("a\0b", 3), soon()); }, 500);
        connection.finish_sse(soon());
        rejects([&] { connection.finish_sse(soon()); }, 500);
        rejects([&] { connection.send_sse_data("x", soon()); }, 500);
        connection.close(); const auto raw = get_all(pair.peer); const auto end = raw.find("\r\n\r\n");
        need(end != std::string::npos && raw.find("Transfer-Encoding: chunked\r\n") < end &&
            raw.find("Content-Length:") == std::string::npos && raw.find("Content-Type: text/event-stream; charset=utf-8\r\n") < end,
            "SSE response must use valid chunked framing and UTF-8 media type");
        const auto chunks = decode_chunks(std::string_view(raw).substr(end + 4));
        need(chunks == std::vector<std::string>{"data: \xf0\x9f\xa7\xa0\ndata: second\ndata: \n\n", "data: \n\n", "data: [DONE]\n\n"},
            "SSE UTF-8/multiline/empty/terminal data framing differs");
    }
    {
        Pair pair; Limits limits; limits.event_bytes = 8; Connection connection(pair.take(), limits);
        connection.begin_sse(soon()); connection.send_sse_data("", soon());
        rejects([&] { connection.send_sse_data("x", soon()); }, 500);
        rejects([&] { connection.send_sse_data("\n", soon()); }, 500);
        connection.finish_sse(soon());
    }
    for (const auto status : {401, 405}) {
        Pair pair; Connection connection(pair.take()); connection.send_json(status, "{}", soon()); connection.close();
        const auto response = get_all(pair.peer);
        need(response.find(status == 401 ? "WWW-Authenticate: Bearer realm=\"tokspur\"\r\n" : "Allow: GET, POST\r\n") != std::string::npos,
            "Required authentication/method response metadata missing");
    }
    {
        Pair pair; Connection connection(pair.take()); connection.send_json(502, "{}", soon()); connection.close();
        need(get_all(pair.peer).starts_with("HTTP/1.1 502 Bad Gateway\r\n"), "Owned worker failure status could not be written");
    }
}

void ownership_disconnect_listener() {
    {
        Pair pair; Connection first(pair.take()); const auto fd = first.native_handle();
        need((fcntl(fd, F_GETFL) & O_NONBLOCK) && (fcntl(fd, F_GETFD) & FD_CLOEXEC), "HTTP socket flags are not nonblocking+CLOEXEC");
        Connection moved(std::move(first)); need(first.native_handle() == -1 && moved.native_handle() == fd, "HTTP move constructor duplicated ownership");
        Pair replacement; Connection second(replacement.take()); const auto old = second.native_handle(); second = std::move(moved);
        need(moved.native_handle() == -1 && second.native_handle() == fd && fcntl(old, F_GETFD) < 0, "HTTP move assignment leaked previous fd");
        second.close(); need(fcntl(fd, F_GETFD) < 0 && second.disconnected(), "HTTP close failed to drain fd ownership");
    }
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, get_request);
        need(!connection.disconnected(), "Unread initial HTTP request misclassified as disconnect");
        static_cast<void>(connection.read_request(soon())); put(pair.peer, "x");
        need(connection.disconnected(), "Delayed HTTP pipeline byte did not cancel connection");
        char value = 0;
        need(recv(connection.native_handle(), &value, 1, MSG_PEEK | MSG_DONTWAIT) == 1 && value == 'x', "HTTP disconnect check consumed unexpected input");
        rejects([&] { connection.send_json(200, "{}", soon()); }, 0, true);
    }
    {
        Pair pair; Connection connection(pair.take()); pair.close_peer(); need(connection.disconnected(), "Peer close not detected");
        rejects([&] { connection.send_json(200, "{}", soon()); }, 0, true); // Must not deliver SIGPIPE.
    }
    Listener listener(0); need(listener.port() != 0, "Loopback test listener failed to adopt ephemeral port");
    sockaddr_in address{}; socklen_t bytes = sizeof(address);
    need(getsockname(listener.native_handle(), reinterpret_cast<sockaddr*>(&address), &bytes) == 0 &&
        address.sin_family == AF_INET && address.sin_addr.s_addr == htonl(INADDR_LOOPBACK), "Listener escaped IPv4 loopback");
    rejects([&] { Listener occupied(listener.port()); }, 0);
    rejects([&] { static_cast<void>(listener.accept(Clock::now())); }, 408);
    const auto client = socket(AF_INET, SOCK_STREAM, 0); need(client >= 0, "Cannot create loopback test client");
    struct Client { int fd; ~Client() { if (fd >= 0) ::close(fd); } } owned_client{client};
    need(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect to owned loopback listener");
    auto connection = listener.accept(soon()); put(client, get_request);
    need(connection.read_request(soon()).path == "/health", "Accepted loopback request differs");
    connection.send_json(200, "{}", soon()); connection.close(); need(get_all(client).ends_with("\r\n\r\n{}"), "Loopback HTTP exchange failed");
    Listener moved(std::move(listener)); need(listener.native_handle() == -1 && moved.port() != 0, "Listener move lost socket/port ownership");
    const auto port = moved.port(); moved.close(); rejects([&] { static_cast<void>(moved.accept(soon())); }, 0);
    Listener restarted(port); need(restarted.port() == port, "Closed listener could not restart while its connection was in TIME_WAIT");
}
} // namespace
int main() {
    try {
        fragmented_bytes(); malformed_requests(); limits_and_deadlines(); response_wire(); ownership_disconnect_listener();
        std::cout << "Owned CPU HTTP/SSE checks passed: " << checks << "; model-free socket/wire fixtures, not native inference or release qualification\n";
    } catch (const std::exception& error) { std::cerr << "Owned HTTP/SSE check failed: " << error.what() << '\n'; return 1; }
}
