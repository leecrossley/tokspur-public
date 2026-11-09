//! `MappedFile` without `mmap`, which safe Rust cannot call.
//!
//! The C++ maps the whole file read-only and lets parsing fault in the pages it touches. This port opens the file with the same flags and checks, snapshots the same `fstat` identity from the open descriptor, and reads with positioned reads (`pread`) only when bytes are needed: the whole file for small assets ([`MappedFile::bytes`]), and only the header region for GGUF shards ([`Header`]). [`MappedFile::assert_unchanged`] compares a fresh `fstat` of the same descriptor with the snapshot, as the C++ does.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::Path;

use crate::{GgufError, HEADER_LIMIT, TRUNCATED_HEADER};

/// `O_NONBLOCK` from `<sys/fcntl.h>`, written out because no `libc` crate is admitted. The C++ opens with `O_RDONLY | O_CLOEXEC | O_NONBLOCK`; std adds `O_CLOEXEC` itself. `O_NONBLOCK` keeps `open` of a FIFO from blocking before `fstat` refuses it, and does not affect reads of a regular file.
#[cfg(target_vendor = "apple")]
const O_NONBLOCK: i32 = 0x0004;
/// `O_NONBLOCK` from the generic Linux `<asm-generic/fcntl.h>` (x86-64, AArch64).
#[cfg(target_os = "linux")]
const O_NONBLOCK: i32 = 0o4000;
#[cfg(not(any(target_vendor = "apple", target_os = "linux")))]
compile_error!("tokspur-gguf needs this platform's O_NONBLOCK value");

/// Most bytes one header read may fetch beyond what the parser asked for.
const READ_AHEAD_LIMIT: u64 = 4 * 1024 * 1024;

/// `MappedFile`: a read-only descriptor and the identity snapshot that [`assert_unchanged`](Self::assert_unchanged) compares. Nothing is read until a caller asks for bytes. Adopted weight files must stay immutable for the lifetime of this value, as they must for the C++ mapping.
///
/// The C++ also refuses with "cannot map weight file" when `mmap` fails. This port has no mapping step: it reports that message when a positioned read of a byte the C++ would touch fails (where the C++ would fault instead), and "weight file changed since mapping" when the file has shrunk under it.
pub struct MappedFile {
    file: File,
    size: u64,
    device: u64,
    inode: u64,
    mtime_seconds: i64,
    mtime_nanos: i64,
    ctime_seconds: i64,
    ctime_nanos: i64,
}

impl fmt::Debug for MappedFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MappedFile")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl MappedFile {
    /// `MappedFile(path)`. Follows symlinks: the C++ opens without `O_NOFOLLOW`, and the adopted shard sets are symlinks.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(O_NONBLOCK)
            .open(path)
            .map_err(GgufError::Open)?;
        let info = file
            .metadata()
            .map_err(|error| GgufError::Inspect(Some(error)))?;
        // `st_size <= 0 || st_size > SIZE_MAX`; std reports the signed `st_size` reinterpreted as `u64`.
        if !info.file_type().is_file()
            || info.size() == 0
            || i64::try_from(info.size()).is_err()
            || usize::try_from(info.size()).is_err()
        {
            return Err(GgufError::Inspect(None));
        }
        Ok(Self {
            file,
            size: info.size(),
            device: info.dev(),
            inode: info.ino(),
            mtime_seconds: info.mtime(),
            mtime_nanos: info.mtime_nsec(),
            ctime_seconds: info.ctime(),
            ctime_nanos: info.ctime_nsec(),
        })
    }

    /// `size()`, from the `fstat` snapshot.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// `bytes()`: the whole file, read now. For small assets (tokenizer JSON, chat template) whose callers bound [`size`](Self::size) first, as the C++ callers do; each call reads the file again.
    pub fn bytes(&self) -> Result<Vec<u8>, GgufError> {
        let mut buffer = Vec::new();
        self.read_into(0, self.size, &mut buffer)?;
        Ok(buffer)
    }

    /// `assert_unchanged()`: a fresh `fstat` of the same descriptor must match the snapshot's size, device, inode and both timestamps to the nanosecond.
    pub fn assert_unchanged(&self) -> Result<(), GgufError> {
        let info = self
            .file
            .metadata()
            .map_err(|error| GgufError::Unchanged(Some(error)))?;
        if info.size() != self.size
            || info.dev() != self.device
            || info.ino() != self.inode
            || info.mtime() != self.mtime_seconds
            || info.mtime_nsec() != self.mtime_nanos
            || info.ctime() != self.ctime_seconds
            || info.ctime_nsec() != self.ctime_nanos
        {
            return Err(GgufError::Unchanged(None));
        }
        Ok(())
    }

    /// Appends the `length` bytes at `offset` to `buffer`. On failure `buffer` keeps its previous contents.
    pub(crate) fn read_into(
        &self,
        offset: u64,
        length: u64,
        buffer: &mut Vec<u8>,
    ) -> Result<(), GgufError> {
        let length = usize::try_from(length)
            .map_err(|error| GgufError::Map(io::Error::new(io::ErrorKind::InvalidInput, error)))?;
        buffer
            .try_reserve_exact(length)
            .map_err(|error| GgufError::Map(io::Error::new(io::ErrorKind::OutOfMemory, error)))?;
        let start = buffer.len();
        buffer.resize(start + length, 0);
        if let Err(error) = self.file.read_exact_at(&mut buffer[start..], offset) {
            buffer.truncate(start);
            return Err(self.read_failure(error));
        }
        Ok(())
    }

    /// The C++ would fault on a page it cannot read. A short read means the file shrank, which `assert_unchanged` reports; anything else is a failure to make the bytes available.
    fn read_failure(&self, error: io::Error) -> GgufError {
        if error.kind() == io::ErrorKind::UnexpectedEof
            && let Err(changed) = self.assert_unchanged()
        {
            return changed;
        }
        GgufError::Map(error)
    }
}

/// A GGUF file's header region, grown with positioned reads as the parser advances.
///
/// The cursor checks every read against the whole file's size and the header limit before asking for bytes, so validation sees exactly the bytes the C++ mapping would. Read-ahead is bounded by `floor`, a lower bound on the bytes a successful parse must still request, so reading a valid file stops exactly at the end of its tensor infos and never touches tensor data.
pub(crate) struct Header {
    file: MappedFile,
    bytes: Vec<u8>,
    floor: u64,
}

impl Header {
    pub(crate) fn new(file: MappedFile) -> Self {
        Self {
            file,
            bytes: Vec::new(),
            floor: 0,
        }
    }

    pub(crate) fn size(&self) -> u64 {
        self.file.size()
    }

    /// The prefix of the file read so far.
    pub(crate) fn loaded(&self) -> &[u8] {
        &self.bytes
    }

    /// Raises the read-ahead bound: a successful parse requests at least `end` bytes.
    pub(crate) fn floor(&mut self, end: u64) {
        self.floor = self.floor.max(end);
    }

    /// Bytes `[start, end)`, read from the file through `end` first if needed. The cursor has checked `end` against the file size and the header limit.
    pub(crate) fn bytes(&mut self, start: u64, end: u64) -> Result<&[u8], GgufError> {
        let loaded = self.bytes.len() as u64;
        if end > loaded {
            let ahead = self
                .floor
                .min(self.file.size())
                .min(HEADER_LIMIT)
                .min(loaded.saturating_add(READ_AHEAD_LIMIT));
            // Read-ahead is only a hint. If it fails, read exactly the bytes requested, so only a byte the C++ parser would touch can fail the parse.
            let read_ahead = ahead > end
                && self
                    .file
                    .read_into(loaded, ahead - loaded, &mut self.bytes)
                    .is_ok();
            if !read_ahead {
                self.file.read_into(loaded, end - loaded, &mut self.bytes)?;
            }
        }
        // Both bounds are at most the header limit, so the casts are exact.
        self.bytes
            .get(start as usize..end as usize)
            .ok_or(GgufError::Header(TRUNCATED_HEADER))
    }

    pub(crate) fn into_parts(self) -> (MappedFile, Vec<u8>) {
        (self.file, self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let name = format!(
                "tokspur-gguf-unit-{}-{}-{label}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let root = std::env::temp_dir().join(name);
            std::fs::create_dir(&root).unwrap();
            Self(root)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0) {
                panic!("cannot remove {}: {error}", self.0.display());
            }
        }
    }

    #[test]
    fn nonblocking_flag_is_the_platform_value() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(O_NONBLOCK, 4);
        #[cfg(target_os = "linux")]
        assert_eq!(O_NONBLOCK, 2048);
    }

    #[test]
    fn bytes_reads_the_whole_file_each_call() {
        let scratch = Scratch::new("bytes");
        let path = scratch.write("asset", b"template text");
        let file = MappedFile::new(&path).unwrap();
        assert_eq!(file.size(), 13);
        assert_eq!(file.bytes().unwrap(), b"template text");
        assert_eq!(file.bytes().unwrap(), b"template text");
        file.assert_unchanged().unwrap();
    }

    #[test]
    fn short_read_after_truncation_reports_a_changed_file() {
        let scratch = Scratch::new("shrink");
        let path = scratch.write("asset", &[7u8; 64]);
        let file = MappedFile::new(&path).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(8)
            .unwrap();
        let error = file.bytes().unwrap_err();
        assert!(matches!(error, GgufError::Unchanged(None)));
        assert_eq!(error.to_string(), "weight file changed since mapping");
    }

    #[test]
    fn failed_read_leaves_the_buffer_unchanged() {
        let scratch = Scratch::new("buffer");
        let path = scratch.write("asset", &[1u8; 16]);
        let file = MappedFile::new(&path).unwrap();
        let mut buffer = vec![9u8; 3];
        assert!(file.read_into(8, 32, &mut buffer).is_err());
        assert_eq!(buffer, [9, 9, 9]);
        file.read_into(8, 8, &mut buffer).unwrap();
        assert_eq!(buffer, [9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn header_reads_ahead_only_to_the_floor() {
        let scratch = Scratch::new("floor");
        let bytes: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let path = scratch.write("header", &bytes);
        let mut header = Header::new(MappedFile::new(&path).unwrap());
        assert_eq!(header.bytes(0, 4).unwrap(), [0, 1, 2, 3]);
        assert_eq!(header.loaded().len(), 4);
        header.floor(100);
        assert_eq!(header.bytes(4, 6).unwrap(), [4, 5]);
        assert_eq!(header.loaded().len(), 100);
        assert_eq!(header.bytes(10, 20).unwrap(), &bytes[10..20]);
        assert_eq!(header.loaded().len(), 100);
        header.floor(50);
        assert_eq!(header.bytes(100, 120).unwrap(), &bytes[100..120]);
        assert_eq!(header.loaded().len(), 120);
    }

    #[test]
    fn header_read_ahead_never_passes_the_file_size() {
        let scratch = Scratch::new("clamp");
        let path = scratch.write("header", &[5u8; 64]);
        let mut header = Header::new(MappedFile::new(&path).unwrap());
        header.floor(1 << 40);
        assert_eq!(header.bytes(0, 1).unwrap(), [5]);
        assert_eq!(header.loaded().len(), 64);
        let (file, bytes) = header.into_parts();
        assert_eq!(file.size(), 64);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn debug_prints_only_the_size() {
        let scratch = Scratch::new("debug");
        let path = scratch.write("asset", b"secret template");
        let file = MappedFile::new(&path).unwrap();
        assert_eq!(format!("{file:?}"), "MappedFile { size: 15, .. }");
    }

    #[test]
    fn appended_byte_is_detected() {
        let scratch = Scratch::new("append");
        let path = scratch.write("asset", b"abc");
        let file = MappedFile::new(&path).unwrap();
        file.assert_unchanged().unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        assert_eq!(
            file.assert_unchanged().unwrap_err().to_string(),
            "weight file changed since mapping"
        );
    }
}
