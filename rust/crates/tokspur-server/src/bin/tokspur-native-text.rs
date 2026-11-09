//! The `text`, `chat` and `response` subcommands of `tokspur-native` (`native/src/main.cpp`), for running the owned tokenizer, chat and round-trip checks against the Rust port. Escaping errors print `tokspur-native: <what>` and exit 1, as the C++ `main` does.

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

use tokspur_server::native_text::{NativeText, text_cli};
use tokspur_text::response_cli::response_cli;

fn main() -> ExitCode {
    if !tokspur_sys::default_sigpipe() {
        return ExitCode::from(1);
    }
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let command = args.get(1).and_then(|arg| arg.to_str()).unwrap_or("");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let outcome: Result<i32, String> = match command {
        "response" => response_cli(&args, NativeText::new, stdin.lock(), stdout.lock())
            .map_err(|error| error.to_string()),
        "text" | "chat" => text_cli(&args, &mut stdin.lock(), &mut stdout.lock())
            .map_err(|error| error.to_string()),
        _ => Err("tokspur-native-text supports only text, chat and response".to_owned()),
    };
    match outcome {
        Ok(status) => ExitCode::from(u8::try_from(status).unwrap_or(1)),
        Err(message) => {
            let line = format!("tokspur-native: {message}\n");
            if std::io::stderr().write_all(line.as_bytes()).is_err() {
                // Nothing else can report it; the exit status still does.
            }
            ExitCode::from(1)
        }
    }
}
