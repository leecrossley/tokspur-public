//! The model-free accumulated-response CLI, ported one to one from `native/src/response_cli.cpp` and the `main` of `native/tests/response_cli_main.cpp`, over generic streams.
//!
//! The pinned tokenizer is loaded through [`ResponseTokenizer`]; the production implementation is `NativeText` (`native/src/text.mm`), which this crate does not port. Protocol lines are written explicitly with [`std::io::Write`].
//!
//! Stream failures follow the C++ iostream state rules: a read error ends input as end-of-file does (`get` fails and `!std::cin` holds), and a write error sets a sticky bad state that drops later output without changing the exit status (`std::cout` with `badbit`). An interrupted read is retried.

use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::{BufRead, ErrorKind, Write};
use std::path::{Path, PathBuf};

use tokspur_json::{Json, JsonError, JsonLimits, parse_json_limited};

use crate::response_protocol::{
    RESPONSE_MAXIMUM_IDS, RESPONSE_OUTPUT_BYTES, RESPONSE_REQUEST_BYTES, ResponseAdmission,
    ResponseDecoder, ResponseProtocolError, native_response_error, parse_native_response,
};

/// `tokenizer_pin`: the only tokenizer JSON the response CLI accepts.
pub const TOKENIZER_PIN: &str = crate::assets::TOKENIZER_SHA256;

/// What the CLI needs from `NativeText`. `Error` is its `FormatError`.
pub trait ResponseTokenizer {
    type Error: std::error::Error;

    /// `NativeText::encode`.
    fn encode(&self, text: &[u8]) -> Result<Vec<u32>, Self::Error>;

    /// `NativeText::decode_bytes` without skipping special tokens.
    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, Self::Error>;

    /// `NativeText::vocabulary_size`.
    fn vocabulary_size(&self) -> usize;
}

/// An error that escapes `response_cli`; `main` prints it to stderr and exits 1. `Display` is the exact C++ `what()` text.
#[derive(Clone, PartialEq, Eq)]
pub enum ResponseCliError<E> {
    /// The command-line options were refused.
    Options(&'static str),
    /// Loading the pinned tokenizer failed (`NativeText::NativeText`).
    Load(E),
    /// The tokenizer failed while deriving the admission markers.
    Tokenize(E),
    /// A marker is not one exact pinned tokenizer token.
    Admit(&'static str),
    /// Dumping the ready line or an error envelope failed.
    Envelope(JsonError),
}

impl<E> fmt::Debug for ResponseCliError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options(reason) => formatter.debug_tuple("Options").field(reason).finish(),
            Self::Load(_) => formatter.write_str("Load([redacted])"),
            Self::Tokenize(_) => formatter.write_str("Tokenize([redacted])"),
            Self::Admit(reason) => formatter.debug_tuple("Admit").field(reason).finish(),
            Self::Envelope(error) => formatter.debug_tuple("Envelope").field(error).finish(),
        }
    }
}

impl<E: fmt::Display> fmt::Display for ResponseCliError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options(reason) | Self::Admit(reason) => formatter.write_str(reason),
            Self::Load(error) | Self::Tokenize(error) => error.fmt(formatter),
            Self::Envelope(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ResponseCliError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Options(_) | Self::Admit(_) => None,
            Self::Load(error) | Self::Tokenize(error) => Some(error),
            Self::Envelope(error) => Some(error),
        }
    }
}

/// A `FormatError` raised while serving one request line; it becomes an error envelope and the loop continues.
enum LineError<E> {
    Read(&'static str),
    Parse(JsonError),
    Respond(ResponseProtocolError<E>),
    Dump(JsonError),
}

impl<E: fmt::Display> fmt::Display for LineError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(reason) => formatter.write_str(reason),
            Self::Parse(error) | Self::Dump(error) => error.fmt(formatter),
            Self::Respond(error) => error.fmt(formatter),
        }
    }
}

/// `std::cin` read through `std::istream::get`.
struct Input<R> {
    reader: R,
    /// `!std::cin`: a `get` has failed at end of input or on a read error.
    failed: bool,
}

impl<R: BufRead> Input<R> {
    /// `read_line`: one `\n`-terminated line, keeping at most [`RESPONSE_REQUEST_BYTES`] bytes and draining the rest. Returns whether any byte was read.
    fn read_line(&mut self, line: &mut Vec<u8>) -> Result<bool, &'static str> {
        line.clear();
        let mut any = false;
        let mut oversized = false;
        loop {
            let available = match self.reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(_) => {
                    // `get` fails on a stream error as it does at end of input.
                    self.failed = true;
                    break;
                }
            };
            if available.is_empty() {
                self.failed = true;
                break;
            }
            any = true;
            let newline = available.iter().position(|&c| c == b'\n');
            let body = &available[..newline.unwrap_or(available.len())];
            let room = RESPONSE_REQUEST_BYTES - line.len();
            line.extend_from_slice(&body[..body.len().min(room)]);
            oversized |= body.len() > room;
            let used = newline.map_or(available.len(), |at| at + 1);
            self.reader.consume(used);
            if newline.is_some() {
                break;
            }
        }
        if oversized {
            return Err("response request exceeds 1-MiB bound");
        }
        Ok(any)
    }
}

/// `std::cout` written with `<< ... << std::endl`.
struct Output<W> {
    writer: W,
    /// `badbit`: a write or flush failed; later output is dropped.
    bad: bool,
}

impl<W: Write> Output<W> {
    fn line(&mut self, bytes: &[u8]) {
        if self.bad {
            return;
        }
        let written = self
            .writer
            .write_all(bytes)
            .and_then(|()| self.writer.write_all(b"\n"));
        if written.and_then(|()| self.writer.flush()).is_err() {
            self.bad = true;
        }
    }
}

fn require<E>(ok: bool, message: &'static str) -> Result<(), ResponseCliError<E>> {
    if ok {
        Ok(())
    } else {
        Err(ResponseCliError::Options(message))
    }
}

fn token<T: ResponseTokenizer>(
    tokenizer: &T,
    marker: &[u8],
) -> Result<u32, ResponseCliError<T::Error>> {
    let ids = tokenizer
        .encode(marker)
        .map_err(ResponseCliError::Tokenize)?;
    if let [id] = ids.as_slice()
        && tokenizer
            .decode_bytes(&ids)
            .map_err(ResponseCliError::Tokenize)?
            == marker
    {
        return Ok(*id);
    }
    Err(ResponseCliError::Admit(
        "response marker is not one exact pinned tokenizer token",
    ))
}

/// `response_cli(argc, argv)`. `args` is the whole `argv`: options start at index 2, after the program and its subcommand. `load` is `NativeText`'s constructor. Returns the exit status, or the error that escapes to `main`.
pub fn response_cli<T, L, R, W>(
    args: &[OsString],
    load: L,
    input: R,
    output: W,
) -> Result<i32, ResponseCliError<T::Error>>
where
    T: ResponseTokenizer,
    L: FnOnce(&Path, &str) -> Result<T, T::Error>,
    R: BufRead,
    W: Write,
{
    let mut tokenizer_path = PathBuf::new();
    let mut pin = OsString::new();
    let mut seen: BTreeSet<&OsStr> = BTreeSet::new();
    let mut i = 2;
    while i < args.len() {
        let option = args[i].as_os_str();
        let inserted = seen.insert(option);
        if inserted {
            i += 1;
        }
        require(
            inserted && i < args.len(),
            "duplicate response option or missing value",
        )?;
        if option == "--tokenizer" {
            tokenizer_path = PathBuf::from(&args[i]);
        } else if option == "--sha256" {
            pin.clone_from(&args[i]);
        } else {
            return Err(ResponseCliError::Options("unknown response option"));
        }
        i += 1;
    }
    require(
        !tokenizer_path.as_os_str().is_empty() && pin == TOKENIZER_PIN,
        "response requires the supported pinned tokenizer and explicit SHA256",
    )?;
    let tokenizer = load(&tokenizer_path, TOKENIZER_PIN).map_err(ResponseCliError::Load)?;
    // A braced initializer: evaluated left to right. The C++ narrows the vocabulary size with `uint32_t(...)`.
    let admission = ResponseAdmission {
        vocabulary: tokenizer.vocabulary_size() as u32,
        im_end: token(&tokenizer, b"<|im_end|>")?,
        end_of_text: token(&tokenizer, b"<|endoftext|>")?,
        im_start: token(&tokenizer, b"<|im_start|>")?,
    };
    let mut output = Output {
        writer: output,
        bad: false,
    };
    let ready = Json::new_object([
        ("event", Json::from("ready")),
        (
            "scope",
            Json::from("model-free accumulated-response parser; no tools are executed"),
        ),
        ("tokenizer_sha256", Json::from(TOKENIZER_PIN)),
        (
            "vocabulary_size",
            Json::integer(u64::from(admission.vocabulary)),
        ),
        ("im_end_id", Json::integer(u64::from(admission.im_end))),
        (
            "end_of_text_id",
            Json::integer(u64::from(admission.end_of_text)),
        ),
        ("im_start_id", Json::integer(u64::from(admission.im_start))),
        ("maximum_ids", Json::integer(RESPONSE_MAXIMUM_IDS as u64)),
        (
            "maximum_request_bytes",
            Json::integer(RESPONSE_REQUEST_BYTES as u64),
        ),
        ("tools_executed", Json::Bool(false)),
    ]);
    output.line(&ready.dump().map_err(ResponseCliError::Envelope)?);
    let mut input = Input {
        reader: input,
        failed: false,
    };
    let mut line = Vec::new();
    let mut failed = false;
    let decode = |ids: &[u32]| tokenizer.decode_bytes(ids);
    loop {
        match serve_line(&mut input, &mut line, admission, &decode) {
            Ok(None) => break,
            Ok(Some(response)) => output.line(&response),
            Err(error) => {
                failed = true;
                let envelope = native_response_error(error.to_string())
                    .dump_limited(8192)
                    .map_err(ResponseCliError::Envelope)?;
                output.line(&envelope);
                if input.failed {
                    break;
                }
            }
        }
    }
    Ok(if failed { 1 } else { 0 })
}

/// The `try` block of the request loop: `None` at end of input.
fn serve_line<R: BufRead, E: fmt::Display>(
    input: &mut Input<R>,
    line: &mut Vec<u8>,
    admission: ResponseAdmission,
    decode: &ResponseDecoder<'_, E>,
) -> Result<Option<Vec<u8>>, LineError<E>> {
    if !input.read_line(line).map_err(LineError::Read)? {
        return Ok(None);
    }
    let request = parse_json_limited(line, JsonLimits::new(RESPONSE_REQUEST_BYTES, 150_000, 32))
        .map_err(LineError::Parse)?;
    let response = parse_native_response(&request, admission, Some(decode), RESPONSE_MAXIMUM_IDS)
        .map_err(LineError::Respond)?;
    response
        .dump_limited(RESPONSE_OUTPUT_BYTES)
        .map(Some)
        .map_err(LineError::Dump)
}

/// `main` of `native/tests/response_cli_main.cpp`: `text` and `chat` go to `text_cli` (`native/src/text.mm`, supplied by the caller), everything else to [`response_cli`]. An escaping error is written to `error_output` with a newline and the status is 1.
pub fn response_cli_main<T, L, C, CE, R, W, X>(
    args: &[OsString],
    text_cli: C,
    load: L,
    input: R,
    output: W,
    mut error_output: X,
) -> i32
where
    T: ResponseTokenizer,
    L: FnOnce(&Path, &str) -> Result<T, T::Error>,
    C: FnOnce(&[OsString]) -> Result<i32, CE>,
    CE: fmt::Display,
    R: BufRead,
    W: Write,
    X: Write,
{
    let text = args.len() >= 2 && (args[1] == "text" || args[1] == "chat");
    let message = if text {
        match text_cli(args) {
            Ok(status) => return status,
            Err(error) => error.to_string(),
        }
    } else {
        match response_cli(args, load, input, output) {
            Ok(status) => return status,
            Err(error) => error.to_string(),
        }
    };
    // `std::cerr << what() << '\n'`. A failed write cannot be reported anywhere and, as with `std::cerr` in `badbit`, leaves the status at 1.
    let reported = error_output
        .write_all(message.as_bytes())
        .and_then(|()| error_output.write_all(b"\n"))
        .and_then(|()| error_output.flush());
    match reported {
        Ok(()) | Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use tokspur_json::UnicodeError;

    #[test]
    fn errors_display_the_exact_cpp_messages_and_retain_sources() {
        let cases: [(ResponseCliError<UnicodeError>, &str, bool); 5] = [
            (
                ResponseCliError::Options("unknown response option"),
                "unknown response option",
                false,
            ),
            (
                ResponseCliError::Load(UnicodeError::Decode("invalid UTF-8 input")),
                "invalid UTF-8 input",
                true,
            ),
            (
                ResponseCliError::Tokenize(UnicodeError::Encode),
                "invalid Unicode scalar",
                true,
            ),
            (
                ResponseCliError::Admit("response marker is not one exact pinned tokenizer token"),
                "response marker is not one exact pinned tokenizer token",
                false,
            ),
            (
                ResponseCliError::Envelope(JsonError::Dump("JSON output byte limit exceeded")),
                "JSON output byte limit exceeded",
                true,
            ),
        ];
        for (error, message, has_source) in cases {
            assert_eq!(error.to_string(), message);
            assert_eq!(error.source().is_some(), has_source);
            let debug = format!("{error:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn read_line_keeps_carriage_returns_and_reports_end_of_input() {
        let mut input = Input {
            reader: &b"a\r\n\nlast"[..],
            failed: false,
        };
        let mut line = Vec::new();
        assert_eq!(input.read_line(&mut line), Ok(true));
        assert_eq!(line, b"a\r");
        assert_eq!(input.read_line(&mut line), Ok(true));
        assert!(line.is_empty());
        assert!(!input.failed);
        assert_eq!(input.read_line(&mut line), Ok(true));
        assert_eq!(line, b"last");
        assert!(input.failed);
        assert_eq!(input.read_line(&mut line), Ok(false));
    }

    #[test]
    fn read_line_drains_an_oversized_line_before_refusing_it() {
        let mut bytes = vec![b'x'; RESPONSE_REQUEST_BYTES + 1];
        bytes.extend_from_slice(b"\nnext\n");
        let mut input = Input {
            reader: bytes.as_slice(),
            failed: false,
        };
        let mut line = Vec::new();
        assert_eq!(
            input.read_line(&mut line),
            Err("response request exceeds 1-MiB bound")
        );
        assert_eq!(line.len(), RESPONSE_REQUEST_BYTES);
        assert_eq!(input.read_line(&mut line), Ok(true));
        assert_eq!(line, b"next");
    }

    #[test]
    fn output_drops_everything_after_the_first_failed_write() {
        struct Failing(usize);
        impl Write for Failing {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 += 1;
                if self.0 == 1 {
                    return Err(std::io::Error::other("closed"));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut output = Output {
            writer: Failing(0),
            bad: false,
        };
        output.line(b"first");
        assert!(output.bad);
        output.line(b"second");
        assert_eq!(output.writer.0, 1);
    }
}
