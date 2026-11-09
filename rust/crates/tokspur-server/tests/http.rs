//! Port of `native/tests/test_http.cpp`. Harnessless and single-threaded (apart from the sender threads), like the C++, because it checks that closed descriptor numbers were not leaked.
//!
//! The C++ move-constructor/assignment checks map to Rust moves: the moved value is statically inaccessible, so only the observable effect (the previous descriptor is closed on reassignment) is checked.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::io::Write;
use std::net::{Shutdown, TcpStream};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokspur_server::http::{Connection, Deadline, HttpError, Limits, Listener};
use tokspur_sys::fd as sysfd;

static CHECKS: AtomicUsize = AtomicUsize::new(0);

fn need(condition: bool, message: &str) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    assert!(condition, "{message}");
}

fn soon() -> Deadline {
    Instant::now() + Duration::from_secs(1)
}

fn rejects<T>(result: Result<T, HttpError>, status: i32, disconnected: bool) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    match result {
        Err(error) => {
            need(
                error.status() == status,
                &format!(
                    "HTTP rejection status differs: {} ({error})",
                    error.status()
                ),
            );
            if disconnected {
                need(error.disconnected(), "HTTP disconnect was not identified");
            }
        }
        Ok(_) => panic!("Invalid HTTP operation accepted"),
    }
}

struct Pair {
    server: Option<OwnedFd>,
    peer: Option<UnixStream>,
}

impl Pair {
    fn new() -> Self {
        let (server, peer) = UnixStream::pair().expect("Cannot create HTTP test socketpair");
        sysfd::set_nosigpipe(peer.as_raw_fd()).expect("Cannot protect test socket SIGPIPE");
        Self {
            server: Some(OwnedFd::from(server)),
            peer: Some(peer),
        }
    }

    fn take(&mut self) -> OwnedFd {
        self.server.take().unwrap()
    }

    fn server_fd(&self) -> i32 {
        self.server.as_ref().unwrap().as_raw_fd()
    }

    fn peer(&self) -> &UnixStream {
        self.peer.as_ref().unwrap()
    }

    fn close_peer(&mut self) {
        self.peer = None;
    }
}

fn put(mut stream: &UnixStream, data: &[u8]) {
    stream.write_all(data).expect("Test HTTP peer write failed");
}

fn get_all_fd(fd: i32) -> Vec<u8> {
    let mut result = Vec::new();
    let deadline = soon();
    let mut bytes = [0u8; 4096];
    loop {
        need(
            Instant::now() < deadline,
            "Test peer read did not reach bounded EOF",
        );
        let mut item = [tokspur_sys::PollFd {
            fd,
            events: tokspur_sys::POLLIN,
            revents: 0,
        }];
        let ready = match sysfd::poll(&mut item, 20) {
            Err(error) if error.raw_os_error() == Some(tokspur_sys::EINTR) => continue,
            other => other,
        };
        need(ready.is_ok(), "Test peer poll failed");
        if ready.unwrap() == 0 {
            continue;
        }
        let count = match sysfd::recv_raw(fd, &mut bytes, sysfd::MSG_DONTWAIT) {
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(tokspur_sys::EINTR | tokspur_sys::EAGAIN)
                ) =>
            {
                continue;
            }
            other => other,
        };
        need(count.is_ok(), "Test peer response read failed");
        let count = count.unwrap();
        if count == 0 {
            return result;
        }
        need(
            result.len() + count <= 1024 * 1024,
            "Test HTTP response exceeded fixture bound",
        );
        result.extend_from_slice(&bytes[..count]);
    }
}

fn post(body: &[u8]) -> Vec<u8> {
    let mut wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:18081\r\nContent-Type: application/json\r\nAuthorization: Bearer fixture-only\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    wire.extend_from_slice(body);
    wire
}

const GET_REQUEST: &[u8] = b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";

fn contains(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn fragmented_bytes() {
    let utf8 = b"{\"text\":\"\xf0\x9f\xa7\xa0\xe9\x9b\xaa\"}".to_vec();
    let wire = post(&utf8);
    for split in 0..=wire.len() {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        let peer = pair.peer.take().unwrap();
        let sent = wire.clone();
        let sender = std::thread::spawn(move || {
            put(&peer, &sent[..split]);
            std::thread::yield_now();
            put(&peer, &sent[split..]);
            peer
        });
        let request = connection.read_request(soon()).unwrap();
        pair.peer = Some(sender.join().unwrap());
        need(
            request.method == "POST"
                && request.path == "/v1/chat/completions"
                && request.body == utf8,
            "Fragmented HTTP request changed raw UTF-8 body bytes",
        );
        need(
            request.header("Authorization") == "Bearer fixture-only"
                && request.header("CONTENT-TYPE") == "application/json"
                && request.header("missing").is_empty(),
            "HTTP field normalization/lookup differs",
        );
        need(
            !connection.disconnected(),
            "No-readable-data incorrectly treated as disconnect",
        );
        rejects(connection.read_request(soon()), 400, false);
    }
    let mut pair = Pair::new();
    let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
    let mut binary = b"a\0b".to_vec();
    binary.extend_from_slice(b"\xf0\x9f\xa7\xa0");
    put(pair.peer(), &post(&binary));
    need(
        connection.read_request(soon()).unwrap().body == binary,
        "Transport changed raw body NUL/UTF-8 before caller JSON validation",
    );
}

fn malformed_requests() {
    let mut twice = GET_REQUEST.to_vec();
    twice.extend_from_slice(GET_REQUEST);
    let mut nul = b"GET / HTTP/1.1\r\nHost: a\r\nX: a".to_vec();
    nul.push(0);
    nul.extend_from_slice(b"b\r\n\r\n");
    let cases: Vec<(Vec<u8>, i32)> = vec![
        (b"GET / HTTP/1.1\nHost: a\n\n".to_vec(), 400),
        (b"GET / HTTP/1.1\rHost: a\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost:\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\nhOsT: a\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost : a\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\n X-Test: value\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a b\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\nX: a\r\nX: b\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\n\r\n".to_vec(), 411),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0,0\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: +1\r\n\r\nx".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: -1\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 00\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 184467440737095516160\r\n\r\n".to_vec(), 413),
        (b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: identity\r\n\r\n".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nExpect: 100-continue\r\n\r\n".to_vec(), 417),
        (b"GET / HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\nConnection: content-length\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\nProxy-Connection: keep-alive\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\nTrailer: X-Foo\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nx".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n\r\nx".to_vec(), 400),
        (b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nxy".to_vec(), 400),
        (twice, 400),
        (b"GET  / HTTP/1.1\r\nHost: a\r\n\r\n".to_vec(), 400),
        (b"GET / HTTP/1.0\r\nHost: a\r\n\r\n".to_vec(), 505),
        (b"PUT / HTTP/1.1\r\nHost: a\r\n\r\n".to_vec(), 405),
        (b"GET http://localhost/ HTTP/1.1\r\nHost: a\r\n\r\n".to_vec(), 400),
        (b"GET /#fragment HTTP/1.1\r\nHost: a\r\n\r\n".to_vec(), 400),
        (b"GET /\\other HTTP/1.1\r\nHost: a\r\n\r\n".to_vec(), 400),
        (nul, 400),
    ];
    for (wire, status) in &cases {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        put(pair.peer(), wire);
        rejects(connection.read_request(soon()), *status, false);
    }
    for wire in [
        b"GET / HTTP/1.1\r\nHost: a\r\n".as_slice(),
        b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\n\r\nx".as_slice(),
    ] {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        put(pair.peer(), wire);
        pair.peer().shutdown(Shutdown::Write).unwrap();
        rejects(connection.read_request(soon()), 400, true);
    }
    let mut pair = Pair::new();
    let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
    pair.close_peer();
    rejects(connection.read_request(soon()), 0, true);
}

fn limits_and_deadlines() {
    for variant in 0..5 {
        let mut limits = Limits::default();
        let mut status = 431;
        let wire: Vec<u8> = match variant {
            0 => {
                limits.header_bytes = 64;
                limits.header_line_bytes = 64;
                format!("GET / HTTP/1.1\r\nHost: a\r\nX: {}", "x".repeat(70)).into_bytes()
            }
            1 => {
                limits.header_line_bytes = 16;
                format!("GET / HTTP/1.1\r\nHost: a\r\nX: {}\r\n\r\n", "x".repeat(17)).into_bytes()
            }
            2 => {
                limits.header_count = 1;
                b"GET / HTTP/1.1\r\nHost: a\r\nX: x\r\n\r\n".to_vec()
            }
            3 => {
                limits.target_bytes = 2;
                status = 400;
                b"GET /long HTTP/1.1\r\nHost: a\r\n\r\n".to_vec()
            }
            _ => {
                limits.body_bytes = 2;
                status = 413;
                post(b"123")
            }
        };
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), limits).unwrap();
        put(pair.peer(), &wire);
        rejects(connection.read_request(soon()), status, false);
    }
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        rejects(connection.read_request(Instant::now()), 408, false);
    }
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        put(pair.peer(), b"GET / HTTP/1.1\r\n");
        rejects(
            connection.read_request(Instant::now() + Duration::from_millis(2)),
            408,
            false,
        );
    }
    {
        let mut pair = Pair::new();
        let invalid = Limits {
            event_bytes: 0,
            ..Limits::default()
        };
        let fd = pair.take();
        let raw = fd.as_raw_fd();
        rejects(Connection::new(fd, invalid), 500, false);
        need(
            !sysfd::is_open(raw),
            "Failed HTTP constructor leaked adopted fd",
        );
    }
    {
        let mut pair = Pair::new();
        need(
            sysfd::set_send_buffer(pair.server_fd(), 1024).is_ok(),
            "Cannot bound backpressure fixture socket",
        );
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        connection.begin_sse(soon()).unwrap();
        rejects(
            connection.send_sse_data(
                &vec![b'x'; 512 * 1024],
                Instant::now() + Duration::from_millis(3),
            ),
            408,
            false,
        );
    }
}

fn decode_chunks(mut body: &[u8]) -> Vec<Vec<u8>> {
    let mut chunks = Vec::new();
    loop {
        let end = contains(body, b"\r\n");
        need(end.is_some_and(|end| end > 0), "Missing HTTP chunk size");
        let end = end.unwrap();
        let count = std::str::from_utf8(&body[..end])
            .ok()
            .and_then(|hex| usize::from_str_radix(hex, 16).ok());
        need(count.is_some(), "Invalid HTTP hex chunk size");
        let count = count.unwrap();
        body = &body[end + 2..];
        if count == 0 {
            need(body == b"\r\n", "Invalid terminal HTTP zero chunk/trailer");
            return chunks;
        }
        need(
            count <= body.len() && body.len() - count >= 2 && &body[count..count + 2] == b"\r\n",
            "HTTP chunk byte length/framing differs",
        );
        chunks.push(body[..count].to_vec());
        body = &body[count + 2..];
        need(chunks.len() <= 8, "Bounded test chunk count exceeded");
    }
}

fn response_wire() {
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        put(pair.peer(), GET_REQUEST);
        connection.read_request(soon()).unwrap();
        let body = b"{\"text\":\"\xe9\x9b\xaa\"}";
        connection.send_json(200, body, soon()).unwrap();
        rejects(connection.send_json(200, b"{}", soon()), 500, false);
        rejects(connection.begin_sse(soon()), 500, false);
        connection.close();
        let raw = get_all_fd(pair.peer().as_raw_fd());
        let end = contains(&raw, b"\r\n\r\n");
        need(
            raw.starts_with(b"HTTP/1.1 200 OK\r\n")
                && end.is_some_and(|end| &raw[end + 4..] == body),
            "JSON HTTP status/body changed",
        );
        need(
            contains(
                &raw,
                format!("Content-Length: {}\r\n", body.len()).as_bytes(),
            )
            .is_some()
                && contains(&raw, b"Connection: close\r\n").is_some()
                && contains(&raw, b"Access-Control-").is_none()
                && contains(&raw, b"Transfer-Encoding:").is_none(),
            "JSON HTTP response framing or private policy differs",
        );
    }
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        put(pair.peer(), GET_REQUEST);
        connection.read_request(soon()).unwrap();
        connection.begin_sse(soon()).unwrap();
        connection
            .send_sse_data(b"\xf0\x9f\xa7\xa0\nsecond\n", soon())
            .unwrap();
        connection.send_sse_data(b"", soon()).unwrap();
        connection.send_sse_data(b"[DONE]", soon()).unwrap();
        rejects(connection.send_sse_data(b"bad\rline", soon()), 500, false);
        rejects(connection.send_sse_data(b"a\0b", soon()), 500, false);
        connection.finish_sse(soon()).unwrap();
        rejects(connection.finish_sse(soon()), 500, false);
        rejects(connection.send_sse_data(b"x", soon()), 500, false);
        connection.close();
        let raw = get_all_fd(pair.peer().as_raw_fd());
        let end = contains(&raw, b"\r\n\r\n");
        need(
            end.is_some_and(|end| {
                contains(&raw, b"Transfer-Encoding: chunked\r\n").is_some_and(|at| at < end)
                    && contains(&raw, b"Content-Length:").is_none()
                    && contains(&raw, b"Content-Type: text/event-stream; charset=utf-8\r\n")
                        .is_some_and(|at| at < end)
            }),
            "SSE response must use valid chunked framing and UTF-8 media type",
        );
        let chunks = decode_chunks(&raw[end.unwrap() + 4..]);
        need(
            chunks
                == vec![
                    b"data: \xf0\x9f\xa7\xa0\ndata: second\ndata: \n\n".to_vec(),
                    b"data: \n\n".to_vec(),
                    b"data: [DONE]\n\n".to_vec(),
                ],
            "SSE UTF-8/multiline/empty/terminal data framing differs",
        );
    }
    {
        let mut pair = Pair::new();
        let limits = Limits {
            event_bytes: 8,
            ..Limits::default()
        };
        let mut connection = Connection::new(pair.take(), limits).unwrap();
        connection.begin_sse(soon()).unwrap();
        connection.send_sse_data(b"", soon()).unwrap();
        rejects(connection.send_sse_data(b"x", soon()), 500, false);
        rejects(connection.send_sse_data(b"\n", soon()), 500, false);
        connection.finish_sse(soon()).unwrap();
    }
    for status in [401, 405] {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        connection.send_json(status, b"{}", soon()).unwrap();
        connection.close();
        let response = get_all_fd(pair.peer().as_raw_fd());
        let expected: &[u8] = if status == 401 {
            b"WWW-Authenticate: Bearer realm=\"tokspur\"\r\n"
        } else {
            b"Allow: GET, POST\r\n"
        };
        need(
            contains(&response, expected).is_some(),
            "Required authentication/method response metadata missing",
        );
    }
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        connection.send_json(502, b"{}", soon()).unwrap();
        connection.close();
        need(
            get_all_fd(pair.peer().as_raw_fd()).starts_with(b"HTTP/1.1 502 Bad Gateway\r\n"),
            "Owned worker failure status could not be written",
        );
    }
}

fn ownership_disconnect_listener() {
    {
        let mut pair = Pair::new();
        let first = Connection::new(pair.take(), Limits::default()).unwrap();
        let fd = first.native_handle();
        let (nonblocking, cloexec) = sysfd::nonblocking_cloexec(fd);
        need(
            nonblocking && cloexec,
            "HTTP socket flags are not nonblocking+CLOEXEC",
        );
        let moved = first;
        need(
            moved.native_handle() == fd,
            "HTTP move constructor duplicated ownership",
        );
        let mut replacement = Pair::new();
        let mut second = Connection::new(replacement.take(), Limits::default()).unwrap();
        let old = second.native_handle();
        second = moved;
        need(
            second.native_handle() == fd && !sysfd::is_open(old),
            "HTTP move assignment leaked previous fd",
        );
        second.close();
        need(
            !sysfd::is_open(fd) && second.disconnected(),
            "HTTP close failed to drain fd ownership",
        );
    }
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        put(pair.peer(), GET_REQUEST);
        need(
            !connection.disconnected(),
            "Unread initial HTTP request misclassified as disconnect",
        );
        connection.read_request(soon()).unwrap();
        put(pair.peer(), b"x");
        need(
            connection.disconnected(),
            "Delayed HTTP pipeline byte did not cancel connection",
        );
        let mut value = [0u8; 1];
        let peeked = sysfd::recv_raw(
            connection.native_handle(),
            &mut value,
            sysfd::MSG_PEEK | sysfd::MSG_DONTWAIT,
        );
        need(
            peeked.is_ok_and(|n| n == 1) && value[0] == b'x',
            "HTTP disconnect check consumed unexpected input",
        );
        rejects(connection.send_json(200, b"{}", soon()), 0, true);
    }
    {
        let mut pair = Pair::new();
        let mut connection = Connection::new(pair.take(), Limits::default()).unwrap();
        pair.close_peer();
        need(connection.disconnected(), "Peer close not detected");
        rejects(connection.send_json(200, b"{}", soon()), 0, true); // Must not deliver SIGPIPE.
    }
    let listener = Listener::new(0, 8).unwrap();
    need(
        listener.port() != 0,
        "Loopback test listener failed to adopt ephemeral port",
    );
    let address = listener.local_address();
    need(
        address.is_ok_and(|a| a.family == 2 && a.address == 0x7f00_0001),
        "Listener escaped IPv4 loopback",
    );
    rejects(Listener::new(listener.port(), 8), 0, false);
    rejects(
        listener.accept(Instant::now(), Limits::default()),
        408,
        false,
    );
    let client = TcpStream::connect(("127.0.0.1", listener.port()));
    need(client.is_ok(), "Cannot create loopback test client");
    let mut client = client.unwrap();
    need(true, "Cannot connect to owned loopback listener");
    let mut connection = listener.accept(soon(), Limits::default()).unwrap();
    client.write_all(GET_REQUEST).unwrap();
    need(
        connection.read_request(soon()).unwrap().path == "/health",
        "Accepted loopback request differs",
    );
    connection.send_json(200, b"{}", soon()).unwrap();
    connection.close();
    need(
        get_all_fd(client.as_raw_fd()).ends_with(b"\r\n\r\n{}"),
        "Loopback HTTP exchange failed",
    );
    let mut moved = listener;
    need(
        moved.port() != 0,
        "Listener move lost socket/port ownership",
    );
    let port = moved.port();
    moved.close();
    rejects(moved.accept(soon(), Limits::default()), 0, false);
    let restarted = Listener::new(port, 8);
    need(
        restarted.is_ok_and(|listener| listener.port() == port),
        "Closed listener could not restart while its connection was in TIME_WAIT",
    );
}

fn main() {
    if std::env::args().any(|arg| arg == "--list") {
        println!("http: test");
        return;
    }
    let section = |name: &str| {
        if std::env::var_os("TOKSPUR_HTTP_SECTIONS").is_some() {
            println!("{name} {}", CHECKS.load(Ordering::Relaxed));
        }
    };
    fragmented_bytes();
    section("s1");
    malformed_requests();
    section("s2");
    limits_and_deadlines();
    section("s3");
    response_wire();
    section("s4");
    ownership_disconnect_listener();
    section("s5");
    println!(
        "Owned CPU HTTP/SSE checks passed: {}; model-free socket/wire fixtures, not native inference or release qualification",
        CHECKS.load(Ordering::Relaxed)
    );
}
