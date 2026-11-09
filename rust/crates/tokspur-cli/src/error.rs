//! Error types for `tokspur-cli` with unbroken `Error::source()` chains and redacting `Debug` implementations.

use std::fmt;
use std::io;

use tokspur_json::JsonError;
use tokspur_pack::PackError;

/// Top-level error returned by `tokspur` CLI operations.
pub enum CliError {
    /// Invalid command-line arguments or options (exits with code `2`).
    Usage(String),
    /// `.tokspur/state/serve.lock` is held by another process (exits with code `1`).
    StateLocked,
    /// State, adoption, port, host, build or supervision refusal (exits with code `1`).
    Validation(String),
    /// Filesystem or OS operation failed (exits with code `1`).
    Io { context: String, source: io::Error },
    /// JSON parsing or serialisation failed (exits with code `1`).
    Json {
        context: &'static str,
        source: JsonError,
    },
    /// Derived pack planning or construction failed (exits with code `1`).
    Pack(PackError),
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::StateLocked
            | Self::Validation(_)
            | Self::Io { .. }
            | Self::Json { .. }
            | Self::Pack(_) => 1,
        }
    }
}

impl fmt::Debug for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(_) => formatter.write_str("Usage([redacted])"),
            Self::StateLocked => formatter.write_str("StateLocked"),
            Self::Validation(_) => formatter.write_str("Validation([redacted])"),
            Self::Io { .. } => formatter
                .debug_struct("Io")
                .field("context", &"[redacted]")
                .field("source", &"[redacted]")
                .finish(),
            Self::Json { context, source } => formatter
                .debug_struct("Json")
                .field("context", context)
                .field("source", source)
                .finish(),
            Self::Pack(error) => formatter.debug_tuple("Pack").field(error).finish(),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(msg) | Self::Validation(msg) => formatter.write_str(msg),
            Self::StateLocked => formatter.write_str("Another tokspur operation owns this state"),
            Self::Io { context, source } => write!(formatter, "{context}: {source}"),
            Self::Json { context, source } => write!(formatter, "{context}: {source}"),
            Self::Pack(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Pack(error) => Some(error),
            Self::Usage(_) | Self::StateLocked | Self::Validation(_) => None,
        }
    }
}

impl From<PackError> for CliError {
    fn from(error: PackError) -> Self {
        Self::Pack(error)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn cli_error_retains_source_and_redacts_debug() {
        let io_err = CliError::Io {
            context: "cannot open /Users/secret/path".to_owned(),
            source: io::Error::new(io::ErrorKind::PermissionDenied, "/Users/secret/path"),
        };
        assert!(io_err.source().is_some());
        let debug_str = format!("{io_err:?}");
        assert!(!debug_str.contains("secret"));
        assert!(debug_str.contains("[redacted]"));

        let val_err = CliError::Validation("/Users/secret/token=abc".to_owned());
        assert!(val_err.source().is_none());
        assert!(!format!("{val_err:?}").contains("secret"));
    }
}
