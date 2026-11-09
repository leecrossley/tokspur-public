//! Bounded HTTP/1.1 and SSE transport, ported one to one from `native/src/http.cpp`.
//!
//! Deliberately narrow: GET/POST, one request per connection, mandatory Host, canonical Content-Length for POST, no transfer-coded requests, Expect/upgrade, duplicate fields, pipelining or persistent connections. Auth, JSON, route policy and admission belong to the caller.

use std::fmt;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::time::Instant;

use tokspur_sys::fd as sysfd;
use tokspur_sys::net::{self, Ipv4Address};
use tokspur_sys::{EAGAIN, EINTR, POLLERR, POLLHUP, POLLIN, POLLNVAL, POLLOUT, PollFd};

pub type Deadline = Instant;

/// `http::Limits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub header_bytes: usize,
    pub header_line_bytes: usize,
    pub header_count: usize,
    pub target_bytes: usize,
    pub body_bytes: usize,
    pub event_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            header_bytes: 32768,
            header_line_bytes: 8192,
            header_count: 64,
            target_bytes: 4096,
            body_bytes: 8 * 1024 * 1024,
            event_bytes: 1024 * 1024,
        }
    }
}

/// What failed, as the C++ message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpFailure {
    /// A framing, limit or state refusal with a fixed message.
    Refused(&'static str),
    /// `system_error(operation)`: `"{operation}: {strerror(errno)}"`.
    System { operation: &'static str, errno: i32 },
}

/// `http::Error`: an HTTP status (0 when none applies), the failure, and whether the peer is gone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpError {
    status: i32,
    failure: HttpFailure,
    disconnected: bool,
}

impl HttpError {
    fn new(status: i32, message: &'static str, disconnected: bool) -> Self {
        Self {
            status,
            failure: HttpFailure::Refused(message),
            disconnected,
        }
    }

    fn system(operation: &'static str, disconnected: bool) -> Self {
        Self {
            status: 0,
            failure: HttpFailure::System {
                operation,
                errno: tokspur_sys::errno(),
            },
            disconnected,
        }
    }

    fn system_from(operation: &'static str, error: &std::io::Error, disconnected: bool) -> Self {
        Self {
            status: 0,
            failure: HttpFailure::System {
                operation,
                errno: error.raw_os_error().unwrap_or(0),
            },
            disconnected,
        }
    }

    pub fn status(&self) -> i32 {
        self.status
    }

    pub fn disconnected(&self) -> bool {
        self.disconnected
    }

    pub fn failure(&self) -> &HttpFailure {
        &self.failure
    }
}

impl fmt::Display for HttpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            HttpFailure::Refused(message) => formatter.write_str(message),
            HttpFailure::System { operation, errno } => {
                write!(formatter, "{operation}: {}", tokspur_sys::strerror(*errno))
            }
        }
    }
}

impl std::error::Error for HttpError {}

type Result<T> = std::result::Result<T, HttpError>;

fn validate_limits(l: Limits) -> Result<()> {
    if l.header_bytes < 16
        || l.header_bytes > 1024 * 1024
        || l.header_line_bytes == 0
        || l.header_line_bytes > l.header_bytes
        || l.header_count == 0
        || l.header_count > 256
        || l.target_bytes == 0
        || l.target_bytes > 8192
        || l.body_bytes > 64 * 1024 * 1024
        || l.event_bytes < 8
        || l.event_bytes > 8 * 1024 * 1024
    {
        return Err(HttpError::new(
            500,
            "Invalid bounded HTTP transport limits",
            false,
        ));
    }
    Ok(())
}

fn prepare_fd(fd: &OwnedFd) -> Result<()> {
    let (kind, length) = sysfd::socket_type(fd.as_fd())
        .map_err(|e| HttpError::system_from("Inspect HTTP socket type", &e, false))?;
    if length != size_of::<i32>() || kind != net::SOCK_STREAM {
        return Err(HttpError::new(
            0,
            "HTTP transport requires a stream socket",
            false,
        ));
    }
    let configure = || -> std::io::Result<()> {
        let flags = sysfd::status_flags(fd.as_fd())?;
        let descriptor = sysfd::descriptor_flags(fd.as_fd())?;
        sysfd::set_status_nonblocking(fd.as_fd(), flags)?;
        sysfd::set_descriptor_cloexec(fd.as_fd(), descriptor)
    };
    configure().map_err(|e| HttpError::system_from("Configure HTTP socket", &e, false))?;
    sysfd::set_nosigpipe(fd.as_raw_fd())
        .map_err(|e| HttpError::system_from("Protect HTTP socket from SIGPIPE", &e, false))
}

fn wait_fd(fd: &OwnedFd, events: i16, deadline: Deadline) -> Result<()> {
    loop {
        if tokspur_sys::stop_requested() {
            return Err(HttpError::new(
                0,
                "HTTP operation cancelled by shutdown",
                true,
            ));
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(HttpError::new(
                408,
                "HTTP transport deadline exceeded",
                false,
            ));
        }
        // duration_cast<milliseconds>(deadline - now).count() + 1, capped at INT_MAX.
        let millis = ((deadline - now).as_millis() + 1).min(100) as i32;
        let mut item = [PollFd {
            fd: fd.as_raw_fd(),
            events,
            revents: 0,
        }];
        match sysfd::poll(&mut item, millis) {
            Err(error) => {
                if error.raw_os_error() == Some(EINTR) {
                    continue;
                }
                return Err(HttpError::system_from("Poll HTTP socket", &error, true));
            }
            Ok(0) => continue,
            Ok(_) => {}
        }
        let revents = item[0].revents;
        if revents & POLLNVAL != 0 {
            return Err(HttpError::new(0, "HTTP socket is invalid", true));
        }
        // POLLIN can coexist with HUP while already-received bytes remain.
        if revents & events != 0 {
            return Ok(());
        }
        if revents & (POLLERR | POLLHUP) != 0 {
            return Err(HttpError::new(0, "HTTP peer disconnected", true));
        }
    }
}

fn lower(c: u8) -> u8 {
    c.to_ascii_lowercase()
}

fn token_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&c)
}

fn lower_text(value: &[u8]) -> Vec<u8> {
    value.iter().copied().map(lower).collect()
}

fn trimmed(mut value: &[u8]) -> &[u8] {
    while let [b' ' | b'\t', rest @ ..] = value {
        value = rest;
    }
    while let [rest @ .., b' ' | b'\t'] = value {
        value = rest;
    }
    value
}

fn crlf_only(value: &[u8]) -> Result<()> {
    for i in 0..value.len() {
        if value[i] == b'\n' && (i == 0 || value[i - 1] != b'\r') {
            return Err(HttpError::new(400, "Bare LF in HTTP headers", false));
        }
        if value[i] == b'\r' && i + 1 < value.len() && value[i + 1] != b'\n' {
            return Err(HttpError::new(400, "Bare CR in HTTP headers", false));
        }
    }
    Ok(())
}

fn reason(status: i32) -> Result<&'static str> {
    Ok(match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        417 => "Expectation Failed",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        _ => {
            return Err(HttpError::new(
                500,
                "Unregistered HTTP response status",
                false,
            ));
        }
    })
}

fn content_length(value: &[u8], maximum: usize) -> Result<usize> {
    if value.is_empty() || (value.len() > 1 && value[0] == b'0') {
        return Err(HttpError::new(400, "Noncanonical Content-Length", false));
    }
    let mut result: usize = 0;
    for &c in value {
        if !c.is_ascii_digit() {
            return Err(HttpError::new(400, "Invalid Content-Length", false));
        }
        let digit = usize::from(c - b'0');
        if result > (usize::MAX - digit) / 10 {
            return Err(HttpError::new(413, "Content-Length overflows", false));
        }
        result = result * 10 + digit;
    }
    if result > maximum {
        return Err(HttpError::new(413, "HTTP body limit exceeded", false));
    }
    Ok(result)
}

fn find(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if start > haystack.len() {
        return None;
    }
    haystack[start..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|at| at + start)
}

/// `http::Request`. Method, path and field values are validated ASCII.
#[derive(Clone, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    /// Unmodified bytes; the caller validates JSON/UTF-8.
    pub body: Vec<u8>,
}

impl fmt::Debug for Request {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Headers carry credentials and the body carries prompts.
        formatter
            .debug_struct("Request")
            .field("method", &self.method)
            .field("path", &self.path)
            .field("headers", &self.headers.len())
            .field("body_bytes", &self.body.len())
            .finish()
    }
}

impl Request {
    /// The value of the named field (case-insensitive), or empty.
    pub fn header(&self, name: &str) -> &str {
        for (field, value) in &self.headers {
            if field.len() != name.len() {
                continue;
            }
            if field.bytes().zip(name.bytes()).all(|(a, b)| a == lower(b)) {
                return value;
            }
        }
        ""
    }
}

fn ascii(bytes: &[u8]) -> String {
    bytes.iter().copied().map(char::from).collect()
}

fn parse_headers(head: &[u8], limits: Limits, body_length: &mut usize) -> Result<Request> {
    crlf_only(head)?;
    let mut lines: Vec<&[u8]> = Vec::new();
    let mut start = 0;
    loop {
        let end = find(head, b"\r\n", start);
        let line = &head[start..end.unwrap_or(head.len())];
        if line.len() > limits.header_line_bytes {
            return Err(HttpError::new(
                431,
                "HTTP header line limit exceeded",
                false,
            ));
        }
        lines.push(line);
        if lines.len() > limits.header_count + 1 {
            return Err(HttpError::new(
                431,
                "HTTP header count limit exceeded",
                false,
            ));
        }
        match end {
            None => break,
            Some(end) => start = end + 2,
        }
    }
    let first = lines[0];
    let first_space = match first.iter().position(|&c| c == b' ') {
        None | Some(0) => return Err(HttpError::new(400, "Invalid HTTP request line", false)),
        Some(at) => at,
    };
    let second_space = find(first, b" ", first_space + 1);
    let second_space = match second_space {
        Some(at) if at != first_space + 1 && find(first, b" ", at + 1).is_none() => at,
        _ => return Err(HttpError::new(400, "Ambiguous HTTP request line", false)),
    };
    let method = &first[..first_space];
    let path = &first[first_space + 1..second_space];
    if !method.iter().all(|&c| token_char(c)) {
        return Err(HttpError::new(400, "Invalid HTTP method", false));
    }
    if method != b"GET" && method != b"POST" {
        return Err(HttpError::new(
            405,
            "Only GET and POST are supported",
            false,
        ));
    }
    if &first[second_space + 1..] != b"HTTP/1.1" {
        return Err(HttpError::new(505, "Only HTTP/1.1 is supported", false));
    }
    if path.is_empty() || path[0] != b'/' || path.len() > limits.target_bytes {
        return Err(HttpError::new(
            400,
            "Bounded origin-form HTTP target required",
            false,
        ));
    }
    if path
        .iter()
        .any(|&c| !(0x21..=0x7e).contains(&c) || c == b'#' || c == b'\\')
    {
        return Err(HttpError::new(400, "Invalid HTTP target byte", false));
    }
    let mut result = Request {
        method: ascii(method),
        path: ascii(path),
        ..Request::default()
    };
    let mut length_present = false;
    for line in &lines[1..] {
        let colon = line.iter().position(|&c| c == b':');
        let colon = match colon {
            Some(at) if !line.is_empty() && line[0] != b' ' && line[0] != b'\t' && at != 0 => at,
            _ => return Err(HttpError::new(400, "Invalid or folded HTTP field", false)),
        };
        let name = &line[..colon];
        if !name.iter().all(|&c| token_char(c)) {
            return Err(HttpError::new(400, "Invalid HTTP field name", false));
        }
        let value = trimmed(&line[colon + 1..]);
        if value.iter().any(|&c| (c < 0x20 && c != b'\t') || c > 0x7e) {
            return Err(HttpError::new(400, "Invalid HTTP field value", false));
        }
        let normalized = ascii(&lower_text(name));
        if result.headers.iter().any(|(field, _)| *field == normalized) {
            return Err(HttpError::new(400, "Duplicate HTTP field", false));
        }
        if matches!(
            normalized.as_str(),
            "transfer-encoding" | "trailer" | "upgrade" | "proxy-connection"
        ) {
            return Err(HttpError::new(
                400,
                "Unsupported ambiguous HTTP framing/upgrade field",
                false,
            ));
        }
        if normalized == "expect" {
            return Err(HttpError::new(
                417,
                "HTTP expectations are unsupported",
                false,
            ));
        }
        if normalized == "connection"
            && lower_text(value) != b"close"
            && lower_text(value) != b"keep-alive"
        {
            return Err(HttpError::new(
                400,
                "Unsupported HTTP connection option",
                false,
            ));
        }
        if normalized == "content-length" {
            length_present = true;
            *body_length = content_length(value, limits.body_bytes)?;
        }
        result.headers.push((normalized, ascii(value)));
    }
    let host = result.header("host");
    if host.is_empty() {
        return Err(HttpError::new(
            400,
            "Exactly one nonempty Host field required",
            false,
        ));
    }
    if host
        .bytes()
        .any(|c| c <= 0x20 || c == b',' || c == b'/' || c == b'\\' || c == b'@')
    {
        return Err(HttpError::new(400, "Invalid Host field", false));
    }
    if result.method == "POST" && !length_present {
        return Err(HttpError::new(411, "POST requires Content-Length", false));
    }
    if result.method == "GET" && *body_length != 0 {
        return Err(HttpError::new(
            400,
            "GET request bodies are unsupported",
            false,
        ));
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Response {
    Idle,
    Json,
    Sse,
    Finished,
}

/// One accepted connection; owns its descriptor.
#[derive(Debug)]
pub struct Connection {
    fd: Option<OwnedFd>,
    limits: Limits,
    read_started: bool,
    request_complete: bool,
    poisoned: bool,
    response: Response,
}

impl Connection {
    /// Adopts `fd` even when validation/setup fails (it is closed then). Intended for an accepted socket or a model-free socketpair. Sets nonblocking and close-on-exec.
    pub fn new(fd: OwnedFd, limits: Limits) -> Result<Self> {
        validate_limits(limits)?;
        prepare_fd(&fd)?;
        Ok(Self {
            fd: Some(fd),
            limits,
            read_started: false,
            request_complete: false,
            poisoned: false,
            response: Response::Idle,
        })
    }

    pub fn close(&mut self) {
        self.fd = None;
    }

    /// The owned descriptor number, or -1 once closed (`native_handle`).
    pub fn native_handle(&self) -> i32 {
        self.fd.as_ref().map_or(-1, AsRawFd::as_raw_fd)
    }

    /// Nonconsuming advisory check. EOF, socket failure, or unexpected bytes after the admitted request cancel the connection.
    pub fn disconnected(&mut self) -> bool {
        let Some(fd) = self.fd.as_ref() else {
            return true;
        };
        if self.poisoned {
            return true;
        }
        let mut item = [PollFd {
            fd: fd.as_raw_fd(),
            events: POLLIN,
            revents: 0,
        }];
        let ready = match sysfd::poll(&mut item, 0) {
            Err(error) => return error.raw_os_error() != Some(EINTR),
            Ok(ready) => ready,
        };
        if ready == 0 {
            return false;
        }
        let revents = item[0].revents;
        if revents & POLLNVAL != 0 {
            return true;
        }
        if revents & POLLIN != 0 {
            let mut byte = [0u8; 1];
            match sysfd::recv(fd.as_fd(), &mut byte, sysfd::MSG_PEEK | sysfd::MSG_DONTWAIT) {
                Ok(0) => return true,
                Ok(_) => {
                    if self.request_complete {
                        self.poisoned = true;
                    }
                    return self.poisoned;
                }
                Err(error) => {
                    if !matches!(error.raw_os_error(), Some(EAGAIN | EINTR)) {
                        return true;
                    }
                }
            }
        }
        revents & (POLLERR | POLLHUP) != 0
    }

    pub fn read_request(&mut self, deadline: Deadline) -> Result<Request> {
        let Some(fd) = self.fd.as_ref() else {
            return Err(HttpError::new(
                400,
                "HTTP connection accepts exactly one request",
                false,
            ));
        };
        if self.read_started || self.response != Response::Idle {
            return Err(HttpError::new(
                400,
                "HTTP connection accepts exactly one request",
                false,
            ));
        }
        self.read_started = true;
        let mut buffer = [0u8; 8192];
        let mut wire: Vec<u8> = Vec::new();
        let mut end = None;
        while end.is_none() {
            wait_fd(fd, POLLIN, deadline)?;
            let count = match sysfd::recv(fd.as_fd(), &mut buffer, 0) {
                Ok(count) => count,
                Err(error) => {
                    if matches!(error.raw_os_error(), Some(EAGAIN | EINTR)) {
                        continue;
                    }
                    return Err(HttpError::system_from("Read HTTP headers", &error, true));
                }
            };
            if count == 0 {
                let status = if wire.is_empty() { 0 } else { 400 };
                return Err(HttpError::new(
                    status,
                    "HTTP peer closed before complete headers",
                    true,
                ));
            }
            wire.extend_from_slice(&buffer[..count]);
            end = find(&wire, b"\r\n\r\n", 0);
            match end {
                None => {
                    crlf_only(&wire)?;
                    if wire.len() >= self.limits.header_bytes {
                        return Err(HttpError::new(
                            431,
                            "HTTP header byte limit exceeded",
                            false,
                        ));
                    }
                }
                Some(at) if at + 4 > self.limits.header_bytes => {
                    return Err(HttpError::new(
                        431,
                        "HTTP header byte limit exceeded",
                        false,
                    ));
                }
                Some(_) => {}
            }
        }
        let end = end.unwrap_or(0);
        let mut length = 0;
        let mut result = parse_headers(&wire[..end], self.limits, &mut length)?;
        result.body = wire[end + 4..].to_vec();
        if result.body.len() > length {
            return Err(HttpError::new(
                400,
                "Unexpected bytes after HTTP request body",
                false,
            ));
        }
        result.body.reserve(length);
        while result.body.len() < length {
            wait_fd(fd, POLLIN, deadline)?;
            let want = buffer.len().min(length - result.body.len());
            let count = match sysfd::recv(fd.as_fd(), &mut buffer[..want], 0) {
                Ok(count) => count,
                Err(error) => {
                    if matches!(error.raw_os_error(), Some(EAGAIN | EINTR)) {
                        continue;
                    }
                    return Err(HttpError::system_from("Read HTTP body", &error, true));
                }
            };
            if count == 0 {
                return Err(HttpError::new(
                    400,
                    "HTTP peer closed before complete body",
                    true,
                ));
            }
            result.body.extend_from_slice(&buffer[..count]);
        }
        if Instant::now() >= deadline {
            return Err(HttpError::new(408, "HTTP request deadline exceeded", false));
        }
        let mut extra = [0u8; 1];
        match sysfd::recv(
            fd.as_fd(),
            &mut extra,
            sysfd::MSG_PEEK | sysfd::MSG_DONTWAIT,
        ) {
            Ok(count) if count > 0 => {
                self.poisoned = true;
                return Err(HttpError::new(
                    400,
                    "HTTP pipelining/trailing bytes refused",
                    false,
                ));
            }
            Ok(_) => {}
            Err(error) => {
                if !matches!(error.raw_os_error(), Some(EAGAIN | EINTR)) {
                    return Err(HttpError::system_from(
                        "Inspect HTTP trailing bytes",
                        &error,
                        true,
                    ));
                }
            }
        }
        self.request_complete = true;
        Ok(result)
    }

    fn write_all(&mut self, mut data: &[u8], deadline: Deadline) -> Result<()> {
        while !data.is_empty() {
            if self.fd.is_none() || self.disconnected() {
                return Err(HttpError::new(
                    0,
                    "HTTP peer disconnected or sent unexpected bytes",
                    true,
                ));
            }
            let Some(fd) = self.fd.as_ref() else {
                return Err(HttpError::new(
                    0,
                    "HTTP peer disconnected or sent unexpected bytes",
                    true,
                ));
            };
            wait_fd(fd, POLLOUT, deadline)?;
            let chunk = &data[..data.len().min(65536)];
            let count = match sysfd::send(fd.as_fd(), chunk, 0) {
                Ok(count) => count,
                Err(error) => {
                    if matches!(error.raw_os_error(), Some(EAGAIN | EINTR)) {
                        continue;
                    }
                    return Err(HttpError::system_from("Write HTTP response", &error, true));
                }
            };
            if count == 0 {
                return Err(HttpError::new(
                    0,
                    "HTTP response write made no progress",
                    true,
                ));
            }
            data = &data[count..];
        }
        Ok(())
    }

    pub fn send_json(&mut self, status: i32, body: &[u8], deadline: Deadline) -> Result<()> {
        if self.response != Response::Idle {
            return Err(HttpError::new(500, "HTTP response already started", false));
        }
        if body.len() > self.limits.body_bytes {
            return Err(HttpError::new(
                500,
                "JSON response exceeds configured body limit",
                false,
            ));
        }
        let message = reason(status)?;
        let extra = match status {
            405 => "Allow: GET, POST\r\n",
            401 => "WWW-Authenticate: Bearer realm=\"tokspur\"\r\n",
            429 | 503 => "Retry-After: 5\r\n",
            _ => "",
        };
        let header = format!(
            "HTTP/1.1 {status} {message}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n{extra}\r\n",
            body.len()
        );
        self.response = Response::Json;
        self.write_all(header.as_bytes(), deadline)?;
        self.write_all(body, deadline)?;
        self.response = Response::Finished;
        Ok(())
    }

    pub fn begin_sse(&mut self, deadline: Deadline) -> Result<()> {
        if self.response != Response::Idle {
            return Err(HttpError::new(500, "HTTP response already started", false));
        }
        self.response = Response::Sse;
        self.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
            deadline,
        )
    }

    /// One HTTP chunk carrying one complete SSE data event. LF splits data fields; CR and NUL are refused to prevent injection.
    pub fn send_sse_data(&mut self, data: &[u8], deadline: Deadline) -> Result<()> {
        if self.response != Response::Sse {
            return Err(HttpError::new(500, "SSE response is not active", false));
        }
        if data.len() > self.limits.event_bytes || data.contains(&0) || data.contains(&b'\r') {
            return Err(HttpError::new(500, "Unsafe or excessive SSE data", false));
        }
        let mut event: Vec<u8> = Vec::new();
        let mut start = 0;
        loop {
            let end = find(data, b"\n", start);
            let line = &data[start..end.unwrap_or(data.len())];
            if event.len() > self.limits.event_bytes - 8
                || line.len() > self.limits.event_bytes - 8 - event.len()
            {
                return Err(HttpError::new(
                    500,
                    "SSE framed event limit exceeded",
                    false,
                ));
            }
            event.extend_from_slice(b"data: ");
            event.extend_from_slice(line);
            event.push(b'\n');
            match end {
                None => break,
                Some(end) => start = end + 1,
            }
        }
        event.push(b'\n');
        let prefix = format!("{:x}\r\n", event.len());
        self.write_all(prefix.as_bytes(), deadline)?;
        self.write_all(&event, deadline)?;
        self.write_all(b"\r\n", deadline)
    }

    /// Terminal HTTP zero chunk (not `[DONE]`).
    pub fn finish_sse(&mut self, deadline: Deadline) -> Result<()> {
        if self.response != Response::Sse {
            return Err(HttpError::new(500, "SSE response is not active", false));
        }
        self.write_all(b"0\r\n\r\n", deadline)?;
        self.response = Response::Finished;
        Ok(())
    }
}

/// IPv4 127.0.0.1 only. Port zero is allowed for explicit tests. Closed connections may remain in TIME_WAIT across restart, but an active listener retains exclusive ownership.
#[derive(Debug)]
pub struct Listener {
    fd: Option<OwnedFd>,
    port: u16,
}

impl Listener {
    pub fn new(port: u16, backlog: i32) -> Result<Self> {
        if !(1..=128).contains(&backlog) {
            return Err(HttpError::new(
                500,
                "Bounded HTTP listener backlog required",
                false,
            ));
        }
        let fd = net::tcp_socket()
            .map_err(|e| HttpError::system_from("Create loopback HTTP socket", &e, false))?;
        prepare_fd(&fd)?;
        net::set_reuseaddr(fd.as_fd())
            .map_err(|e| HttpError::system_from("Allow loopback HTTP address reuse", &e, false))?;
        net::bind_loopback(fd.as_fd(), port)
            .map_err(|e| HttpError::system_from("Bind loopback HTTP socket", &e, false))?;
        net::listen(fd.as_fd(), backlog)
            .map_err(|e| HttpError::system_from("Listen on loopback HTTP socket", &e, false))?;
        let address = match net::local_address(fd.as_fd()) {
            Ok(address) if address.length == Ipv4Address::SOCKADDR_IN_BYTES => address,
            Ok(_) => return Err(HttpError::system("Inspect loopback HTTP address", false)),
            Err(e) => {
                return Err(HttpError::system_from(
                    "Inspect loopback HTTP address",
                    &e,
                    false,
                ));
            }
        };
        if address.family != Ipv4Address::FAMILY_INET || address.address != Ipv4Address::LOOPBACK {
            return Err(HttpError::new(
                0,
                "HTTP listener is not loopback-only",
                false,
            ));
        }
        Ok(Self {
            fd: Some(fd),
            port: address.port,
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// The listening descriptor number, or -1 once closed.
    pub fn native_handle(&self) -> i32 {
        self.fd.as_ref().map_or(-1, AsRawFd::as_raw_fd)
    }

    /// `getsockname` on the listening socket.
    pub fn local_address(&self) -> std::io::Result<Ipv4Address> {
        match self.fd.as_ref() {
            Some(fd) => net::local_address(fd.as_fd()),
            None => Err(std::io::Error::from_raw_os_error(9)),
        }
    }

    pub fn close(&mut self) {
        self.fd = None;
        self.port = 0;
    }

    /// An idle accept deadline returns status 408 (not disconnected), so a caller can poll briefly for shutdown.
    pub fn accept(&self, deadline: Deadline, limits: Limits) -> Result<Connection> {
        validate_limits(limits)?;
        let Some(fd) = self.fd.as_ref() else {
            return Err(HttpError::new(0, "HTTP listener is closed", false));
        };
        loop {
            wait_fd(fd, POLLIN, deadline)?;
            let (accepted, peer) = match net::accept(fd.as_fd()) {
                Ok(pair) => pair,
                Err(error) => {
                    if matches!(error.raw_os_error(), Some(EAGAIN | EINTR)) {
                        continue;
                    }
                    return Err(HttpError::system_from(
                        "Accept loopback HTTP connection",
                        &error,
                        false,
                    ));
                }
            };
            if peer.length != Ipv4Address::SOCKADDR_IN_BYTES
                || peer.family != Ipv4Address::FAMILY_INET
                || (peer.address >> 24) != 127
            {
                drop(accepted);
                return Err(HttpError::new(0, "Refused non-loopback HTTP peer", false));
            }
            match Connection::new(accepted, limits) {
                Ok(connection) => return Ok(connection),
                // A reset peer can refuse socket adoption. Its descriptor is dropped and the listener keeps the original deadline.
                Err(_) => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_debug_counts_headers_and_body_without_their_bytes() {
        // Arrange
        let request = Request {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: vec![("authorization".to_owned(), "Bearer private-key".to_owned())],
            body: br#"{"messages":"private prompt"}"#.to_vec(),
        };

        // Act
        let rendered = format!("{request:?}");

        // Assert
        assert_eq!(
            rendered,
            r#"Request { method: "POST", path: "/v1/chat/completions", headers: 1, body_bytes: 29 }"#
        );
        assert!(!rendered.contains("private"));
    }
}
