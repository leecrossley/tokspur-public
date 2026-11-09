//! `tokspur-server`, ported from `native/server_main.cpp`.

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

fn main() -> ExitCode {
    // Rust ignores SIGPIPE before main; the C++ server keeps the default.
    if !tokspur_sys::default_sigpipe() {
        return ExitCode::from(1);
    }
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    match tokspur_server::serve::server_cli(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let line = format!("tokspur-server refused startup/shutdown: {error}\n");
            if std::io::stderr().write_all(line.as_bytes()).is_err() {
                // Nothing else can report it; the exit status still does.
            }
            let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
            while let Some(cause) = source {
                if let Some(worker) = cause.downcast_ref::<tokspur_wire::worker::WorkerError>()
                    && let Some(bytes) = worker.startup_output()
                {
                    let mut private = String::from("worker startup diagnostic:\n");
                    for &byte in bytes {
                        if byte == b'\n' || byte == b'\t' || (32..127).contains(&byte) {
                            private.push(char::from(byte));
                        } else {
                            use std::fmt::Write as _;
                            if write!(&mut private, "\\x{byte:02x}").is_err() {
                                break;
                            }
                        }
                    }
                    private.push('\n');
                    if std::io::stderr().write_all(private.as_bytes()).is_err() {
                        break;
                    }
                    break;
                }
                source = cause.source();
            }
            ExitCode::from(1)
        }
    }
}
