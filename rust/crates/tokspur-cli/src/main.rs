//! Entry point for the standalone `tokspur` CLI binary.

#![forbid(unsafe_code)]

use std::io::{self, Write};
use std::process::ExitCode;

use tokspur_cli::{error::CliError, execute, parse_args};

fn main() -> ExitCode {
    let args = std::env::args_os()
        .skip(1)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| CliError::Usage("command-line arguments must be UTF-8".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>();
    let mut stdout = io::stdout().lock();
    match args
        .and_then(parse_args)
        .and_then(|parsed| execute(parsed, &mut stdout))
    {
        Ok(0) => ExitCode::SUCCESS,
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(error) => {
            let mut stderr = io::stderr().lock();
            let _ignored = writeln!(stderr, "tokspur: {error}");
            ExitCode::from(u8::try_from(error.exit_code()).unwrap_or(1))
        }
    }
}
