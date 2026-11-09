//! HTTP request framing target: `tokspur-server`'s `Connection::read_request` over a real socket pair, with the stream written whole and in chunks.
//!
//! Input layout: a flags byte, a separator byte, five limit bytes when bit 0 of the flags asks for small limits, then the request stream, split into writes on the separator (see [`crate::split`]).
//!
//! The whole stream fits the socket buffer, so the "whole" run writes it and shuts the write side before reading starts: the reader sees every byte in its first receive, and the outcome is deterministic. The "split" run writes one chunk at a time from a second thread and writes the next only once the reader has taken all of it (`poll` on a duplicate of the reader's socket finds nothing to read), so the reader meets the chunk boundaries. Only a chunk boundary exactly at the end of a complete request races with the reader's final trailing-byte check. Both ends stay open until the reader returns, and the peer is protected from `SIGPIPE` (a `#![no_main]` binary does not ignore it).

#![forbid(unsafe_code)]

use std::io::{ErrorKind, Write};
use std::net::Shutdown;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tokspur_server::http::{Connection, HttpError, HttpFailure, Limits, Request};
use tokspur_sys::fd as sysfd;
use tokspur_sys::{POLLIN, PollFd};

use crate::encode::escape;
use crate::split;

/// The largest stream a run writes: a whole stream must fit the AF_UNIX socket buffer (8 KiB on macOS) so the whole run cannot block.
pub const MAXIMUM_STREAM: usize = 4096;
/// Far beyond any run; reaching it means the reader hung.
const DEADLINE: Duration = Duration::from_secs(20);

/// Every refusal `read_request` and `Connection::new` can return, with its status, from http.cpp. EOF before the first byte reports status 0.
const REFUSALS: [(i32, &str); 34] = [
    (400, "Bare LF in HTTP headers"),
    (400, "Bare CR in HTTP headers"),
    (431, "HTTP header line limit exceeded"),
    (431, "HTTP header count limit exceeded"),
    (400, "Invalid HTTP request line"),
    (400, "Ambiguous HTTP request line"),
    (400, "Invalid HTTP method"),
    (405, "Only GET and POST are supported"),
    (505, "Only HTTP/1.1 is supported"),
    (400, "Bounded origin-form HTTP target required"),
    (400, "Invalid HTTP target byte"),
    (400, "Invalid or folded HTTP field"),
    (400, "Invalid HTTP field name"),
    (400, "Invalid HTTP field value"),
    (400, "Duplicate HTTP field"),
    (400, "Unsupported ambiguous HTTP framing/upgrade field"),
    (417, "HTTP expectations are unsupported"),
    (400, "Unsupported HTTP connection option"),
    (400, "Noncanonical Content-Length"),
    (400, "Invalid Content-Length"),
    (413, "Content-Length overflows"),
    (413, "HTTP body limit exceeded"),
    (400, "Exactly one nonempty Host field required"),
    (400, "Invalid Host field"),
    (411, "POST requires Content-Length"),
    (400, "GET request bodies are unsupported"),
    (400, "HTTP peer closed before complete headers"),
    (0, "HTTP peer closed before complete headers"),
    (431, "HTTP header byte limit exceeded"),
    (400, "Unexpected bytes after HTTP request body"),
    (400, "HTTP peer closed before complete body"),
    (400, "HTTP pipelining/trailing bytes refused"),
    (400, "HTTP connection accepts exactly one request"),
    (500, "Invalid bounded HTTP transport limits"),
];

const TRAILING: &str = "Unexpected bytes after HTTP request body";

/// Small limits from five bytes; some are invalid, so `Connection::new`'s refusal is exercised too.
fn small_limits(bytes: [u8; 5]) -> Limits {
    Limits {
        header_bytes: 16 + usize::from(bytes[0]) * 16,
        header_line_bytes: 1 + usize::from(bytes[1]) * 8,
        header_count: usize::from(bytes[2] % 10),
        target_bytes: usize::from(bytes[3]) * 4,
        body_bytes: usize::from(bytes[4]) * 8,
        event_bytes: Limits::default().event_bytes,
    }
}

/// One decoded input: limits, the chunks and their concatenation.
pub struct Case<'a> {
    pub limits: Limits,
    pub chunks: Vec<&'a [u8]>,
    pub stream: Vec<u8>,
}

/// Decodes a fuzz input; `None` when it is too short or too long.
pub fn case(data: &[u8]) -> Option<Case<'_>> {
    let (&flags, rest) = data.split_first()?;
    let (&separator, mut rest) = rest.split_first()?;
    let limits = if flags & 1 == 0 {
        Limits::default()
    } else {
        let (bytes, tail) = rest.split_first_chunk::<5>()?;
        rest = tail;
        small_limits(*bytes)
    };
    let chunks = split::chunks(rest, separator);
    let stream = chunks.concat();
    (stream.len() <= MAXIMUM_STREAM).then_some(Case {
        limits,
        chunks,
        stream,
    })
}

/// What one run produced.
pub enum Outcome {
    Accepted(Request),
    Refused(HttpError),
}

impl Outcome {
    fn describe(&self) -> String {
        match self {
            Self::Accepted(request) => format!(
                "accepted {} {} with {} fields and {} body bytes",
                request.method,
                request.path,
                request.headers.len(),
                request.body.len()
            ),
            Self::Refused(error) => format!("refused {} {error}", error.status()),
        }
    }
}

fn socket_pair() -> (UnixStream, UnixStream) {
    let (server, peer) = match UnixStream::pair() {
        Ok(pair) => pair,
        Err(error) => panic!("cannot create a socket pair: {error}"),
    };
    if let Err(error) = sysfd::set_nosigpipe(peer.as_raw_fd()) {
        panic!("cannot protect the peer from SIGPIPE: {error}");
    }
    (server, peer)
}

fn read(connection: Result<Connection, HttpError>) -> Outcome {
    let mut connection = match connection {
        Ok(connection) => connection,
        Err(error) => return Outcome::Refused(error),
    };
    match connection.read_request(Instant::now() + DEADLINE) {
        Ok(request) => Outcome::Accepted(request),
        Err(error) => Outcome::Refused(error),
    }
}

/// Writes the whole stream, shuts the write side, then reads.
pub fn run_whole(stream: &[u8], limits: Limits) -> Outcome {
    let (server, mut peer) = socket_pair();
    if let Err(error) = peer.write_all(stream) {
        panic!("cannot write the whole stream: {error}");
    }
    if let Err(error) = peer.shutdown(Shutdown::Write) {
        panic!("cannot shut the peer's write side: {error}");
    }
    read(Connection::new(OwnedFd::from(server), limits))
}

/// The writer of a split run: one chunk at a time, each only once the reader has taken every earlier byte or has finished.
///
/// It watches the reader's socket with `poll`, never with a peeking `recv`: on macOS a `MSG_DONTWAIT` receive fails with `EWOULDBLOCK` while another thread holds the socket's receive lock, so a concurrent peek here could make the reader's own trailing-byte peek miss bytes that are present.
fn write_chunks(peer: &mut UnixStream, probe: &UnixStream, chunks: &[Vec<u8>], done: &AtomicBool) {
    for chunk in chunks.iter().filter(|chunk| !chunk.is_empty()) {
        if done.load(Ordering::Acquire) {
            break;
        }
        if let Err(error) = peer.write_all(chunk) {
            panic!("cannot write a chunk: {error}");
        }
        while !done.load(Ordering::Acquire) {
            let mut readable = [PollFd {
                fd: probe.as_raw_fd(),
                events: POLLIN,
                revents: 0,
            }];
            match sysfd::poll(&mut readable, 0) {
                Ok(0) => break,
                Ok(_) => thread::yield_now(),
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => panic!("cannot poll the reader's socket: {error}"),
            }
        }
    }
    if let Err(error) = peer.shutdown(Shutdown::Write) {
        panic!("cannot shut the peer's write side: {error}");
    }
}

/// Writes the chunks one by one from a second thread while reading.
pub fn run_split(chunks: &[&[u8]], limits: Limits) -> Outcome {
    let (server, mut peer) = socket_pair();
    let probe = match server.try_clone() {
        Ok(probe) => probe,
        Err(error) => panic!("cannot duplicate the reader's socket: {error}"),
    };
    let connection = match Connection::new(OwnedFd::from(server), limits) {
        Ok(connection) => connection,
        Err(error) => return Outcome::Refused(error),
    };
    let done = Arc::new(AtomicBool::new(false));
    let owned: Vec<Vec<u8>> = chunks.iter().map(|chunk| chunk.to_vec()).collect();
    let writer = {
        let done = Arc::clone(&done);
        // The writer hands both sockets back, so neither closes before the reader has returned.
        thread::spawn(move || {
            write_chunks(&mut peer, &probe, &owned, &done);
            (peer, probe)
        })
    };
    let outcome = read(Ok(connection));
    done.store(true, Ordering::Release);
    let Ok(sockets) = writer.join() else {
        panic!("the split-run writer panicked");
    };
    drop(sockets);
    outcome
}

fn known(error: &HttpError, input: &[u8]) {
    let message = match error.failure() {
        HttpFailure::Refused(message) => *message,
        HttpFailure::System { operation, errno } => {
            panic!(
                "system failure {operation} ({errno}) for {}",
                escape(input, 512)
            )
        }
    };
    assert!(
        REFUSALS.contains(&(error.status(), message)),
        "unknown HTTP refusal {} {message:?} for {}",
        error.status(),
        escape(input, 512)
    );
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn token(c: u8) -> bool {
    c.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&c)
}

fn trim(mut value: &[u8]) -> &[u8] {
    while let [b' ' | b'\t', rest @ ..] = value {
        value = rest;
    }
    while let [rest @ .., b' ' | b'\t'] = value {
        value = rest;
    }
    value
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Re-derives, independently of the parser, what an accepted request must be and that every framing rule holds for it. Returns where the request ends in the stream.
fn verify_accepted(request: &Request, stream: &[u8], limits: Limits) -> usize {
    let input = || escape(stream, 512);
    let Some(end) = find(stream, b"\r\n\r\n") else {
        panic!("accepted a request without a blank line: {}", input());
    };
    let head = &stream[..end];
    assert!(
        end + 4 <= limits.header_bytes,
        "header byte limit: {}",
        input()
    );
    let lines: Vec<&[u8]> = head.split(|&c| c == b'\n').collect();
    let mut stripped = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let line = if i + 1 < lines.len() {
            let Some(line) = line.strip_suffix(b"\r") else {
                panic!("bare LF accepted: {}", input());
            };
            line
        } else {
            line
        };
        assert!(!line.contains(&b'\r'), "bare CR accepted: {}", input());
        assert!(
            line.len() <= limits.header_line_bytes,
            "line limit: {}",
            input()
        );
        stripped.push(line);
    }
    assert!(
        stripped.len() <= limits.header_count + 1,
        "count limit: {}",
        input()
    );

    let parts: Vec<&[u8]> = stripped[0].split(|&c| c == b' ').collect();
    assert!(parts.len() == 3, "request line: {}", input());
    let (method, path) = (parts[0], parts[1]);
    assert!(
        (method == b"GET" || method == b"POST") && parts[2] == b"HTTP/1.1",
        "method or version: {}",
        input()
    );
    assert!(
        path.first() == Some(&b'/')
            && path.len() <= limits.target_bytes
            && path
                .iter()
                .all(|&c| (0x21..=0x7e).contains(&c) && c != b'#' && c != b'\\'),
        "target: {}",
        input()
    );

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut length = None;
    for line in &stripped[1..] {
        let Some(colon) = line.iter().position(|&c| c == b':') else {
            panic!("field without colon accepted: {}", input());
        };
        let name = line[..colon].to_ascii_lowercase();
        let value = trim(&line[colon + 1..]);
        assert!(
            !name.is_empty() && name.iter().all(|&c| token(c)),
            "field name: {}",
            input()
        );
        assert!(
            value
                .iter()
                .all(|&c| (c >= 0x20 || c == b'\t') && c <= 0x7e),
            "field value: {}",
            input()
        );
        let name = text(&name);
        assert!(
            headers.iter().all(|(field, _)| *field != name),
            "duplicate field: {}",
            input()
        );
        assert!(
            !matches!(
                name.as_str(),
                "transfer-encoding" | "trailer" | "upgrade" | "proxy-connection" | "expect"
            ),
            "refused field accepted: {}",
            input()
        );
        if name == "connection" {
            let option = value.to_ascii_lowercase();
            assert!(
                option == b"close" || option == b"keep-alive",
                "connection option: {}",
                input()
            );
        }
        if name == "content-length" {
            assert!(
                !value.is_empty()
                    && value.iter().all(u8::is_ascii_digit)
                    && (value.len() == 1 || value[0] != b'0'),
                "content length: {}",
                input()
            );
            let parsed = text(value).parse::<usize>().ok();
            assert!(
                parsed.is_some_and(|parsed| parsed <= limits.body_bytes),
                "content length: {}",
                input()
            );
            length = parsed;
        }
        headers.push((name, text(value)));
    }
    let host = headers
        .iter()
        .find(|(field, _)| field == "host")
        .map(|(_, value)| value.as_bytes());
    assert!(
        host.is_some_and(|host| {
            !host.is_empty() && !host.iter().any(|&c| c <= 0x20 || b",/\\@".contains(&c))
        }),
        "host: {}",
        input()
    );
    assert!(
        method != b"POST" || length.is_some(),
        "POST without length: {}",
        input()
    );
    let length = length.unwrap_or(0);
    assert!(method != b"GET" || length == 0, "GET body: {}", input());
    let request_end = end + 4 + length;
    assert!(
        request_end <= stream.len(),
        "body beyond the stream: {}",
        input()
    );
    assert!(
        request.method.as_bytes() == method
            && request.path.as_bytes() == path
            && request.headers == headers
            && request.body == stream[end + 4..request_end],
        "accepted request differs from the stream: {}",
        input()
    );
    request_end
}

fn same_request(a: &Request, b: &Request) -> bool {
    a.method == b.method && a.path == b.path && a.headers == b.headers && a.body == b.body
}

/// `read_request` on an arbitrary stream, written whole and in chunks. Every refusal is a known status and message. Every accepted request matches an independent reading of the stream and satisfies every framing rule. A request accepted whole is accepted identically when split; a request accepted only when split must be the whole-stream refusal of trailing bytes, with a chunk boundary exactly where the request ends.
pub fn check(data: &[u8]) {
    let Some(case) = case(data) else {
        return;
    };
    let whole = run_whole(&case.stream, case.limits);
    let split = run_split(&case.chunks, case.limits);
    let mut ends = Vec::new();
    for outcome in [&whole, &split] {
        match outcome {
            Outcome::Refused(error) => known(error, &case.stream),
            Outcome::Accepted(request) => {
                ends.push(verify_accepted(request, &case.stream, case.limits));
            }
        }
    }
    let mismatch = || {
        format!(
            "whole run {}, split run {}: {}",
            whole.describe(),
            split.describe(),
            escape(data, 512)
        )
    };
    match (&whole, &split) {
        (Outcome::Accepted(a), Outcome::Accepted(b)) => {
            assert!(same_request(a, b), "{}", mismatch());
        }
        (Outcome::Accepted(_), Outcome::Refused(_)) => panic!("{}", mismatch()),
        (Outcome::Refused(error), Outcome::Accepted(_)) => {
            let end = ends[0];
            assert!(
                error.failure() == &HttpFailure::Refused(TRAILING)
                    && end < case.stream.len()
                    && split::boundaries(&case.chunks).contains(&end),
                "{}",
                mismatch()
            );
        }
        (Outcome::Refused(_), Outcome::Refused(_)) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(flags: u8, separator: u8, stream: &[u8]) -> Vec<u8> {
        let mut data = vec![flags, separator];
        data.extend_from_slice(stream);
        data
    }

    const STREAMS: [&[u8]; 9] = [
        b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:18081\r\nContent-Length: 2\r\n\r\n{}",
        b"GET / HTTP/1.1\r\nHost: a\r\n|\r\n",
        b"GE|T / HTTP/1.1\r|\nHost: a\r\n\r|\n",
        b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nx|y",
        b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nxy",
        b"GET / HTTP/1.1\nHost: a\n\n",
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n",
        b"",
    ];

    #[test]
    fn properties_hold_for_the_http_test_streams() {
        for stream in STREAMS {
            check(&input(0, b'|', stream));
            let mut small = input(1, b'|', &[]);
            small.extend_from_slice(&[2, 4, 3, 2, 1]);
            small.extend_from_slice(stream);
            check(&small);
        }
    }

    #[test]
    fn a_boundary_at_the_request_end_may_race_the_trailing_check() {
        let data = input(0, b'|', STREAMS[4]);
        let case = case(&data).unwrap();
        assert!(matches!(
            run_whole(&case.stream, case.limits),
            Outcome::Refused(error) if error.to_string() == TRAILING
        ));
        // Accepted, or refused as pipelining, depending on the race.
        match run_split(&case.chunks, case.limits) {
            Outcome::Accepted(request) => assert_eq!(request.body, b"x"),
            Outcome::Refused(error) => {
                assert_eq!(error.to_string(), "HTTP pipelining/trailing bytes refused");
            }
        }
    }
}
