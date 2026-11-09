//! Descriptor creation and raw I/O used by the owned worker and HTTP layer. Results carry the OS `errno` in `std::io::Error` so callers can branch on `EINTR`/`EAGAIN` exactly as the C++ does.

use std::ffi::{CStr, c_int};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};

use crate::ffi;

fn check(result: c_int) -> io::Result<c_int> {
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result)
    }
}

fn adopt(fd: c_int) -> OwnedFd {
    // SAFETY: callers pass a descriptor a successful libSystem call just returned, which nothing else owns.
    unsafe { OwnedFd::from_raw_fd(fd) }
}

/// Why a worker pipe/stream pair could not be created and secured.
pub enum PairError {
    /// `pipe`/`socketpair` failed.
    Create(io::Error),
    /// Moving a descriptor above 2 or setting `FD_CLOEXEC` failed.
    Secure(Option<io::Error>),
    /// `SO_NOSIGPIPE` could not be set on a stream.
    NoSigpipe(io::Error),
}

impl std::fmt::Debug for PairError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(_) => formatter.write_str("Create([redacted])"),
            Self::Secure(Some(_)) => formatter.write_str("Secure(Some([redacted]))"),
            Self::Secure(None) => formatter.write_str("Secure(None)"),
            Self::NoSigpipe(_) => formatter.write_str("NoSigpipe([redacted])"),
        }
    }
}

impl std::fmt::Display for PairError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Create(_) => "cannot create worker descriptor pair",
            Self::Secure(_) => "cannot secure worker pipe descriptor",
            Self::NoSigpipe(_) => "cannot suppress local control-stream SIGPIPE",
        })
    }
}

impl std::error::Error for PairError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Create(error) | Self::NoSigpipe(error) => Some(error),
            Self::Secure(error) => error
                .as_ref()
                .map(|error| error as &(dyn std::error::Error + 'static)),
        }
    }
}

/// `descriptor()` in owned_worker.cpp: moves a descriptor below 3 up with `F_DUPFD_CLOEXEC`, then requires it to be at least 3 with `FD_CLOEXEC`.
fn secure(fd: OwnedFd) -> Result<OwnedFd, PairError> {
    let fd = if fd.as_raw_fd() < 3 {
        // SAFETY: fcntl on a descriptor this function owns.
        let moved = unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_DUPFD_CLOEXEC, 3 as c_int) };
        let error = if moved < 0 {
            Some(io::Error::last_os_error())
        } else {
            None
        };
        drop(fd);
        if let Some(error) = error {
            return Err(PairError::Secure(Some(error)));
        }
        adopt(moved)
    } else {
        fd
    };
    // SAFETY: fcntl on a descriptor this function owns.
    let flagged = unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_SETFD, ffi::FD_CLOEXEC) };
    if flagged != 0 {
        return Err(PairError::Secure(Some(io::Error::last_os_error())));
    }
    if fd.as_raw_fd() >= 3 {
        Ok(fd)
    } else {
        Err(PairError::Secure(None))
    }
}

/// `pipe_pair`: a pipe whose ends are both secured. Returns (read, write).
pub fn pipe_pair() -> Result<(OwnedFd, OwnedFd), PairError> {
    let mut fds = [-1 as c_int; 2];
    // SAFETY: `fds` has room for the two descriptors pipe writes.
    if unsafe { ffi::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(PairError::Create(io::Error::last_os_error()));
    }
    let (read, write) = (adopt(fds[0]), adopt(fds[1]));
    Ok((secure(read)?, secure(write)?))
}

/// `socket_pair`: an AF_UNIX stream pair, both ends secured and protected from `SIGPIPE`. Returns (parent end, child end).
pub fn socket_pair() -> Result<(OwnedFd, OwnedFd), PairError> {
    let mut fds = [-1 as c_int; 2];
    // SAFETY: `fds` has room for the two descriptors socketpair writes.
    if unsafe { ffi::socketpair(ffi::AF_UNIX, ffi::SOCK_STREAM, 0, fds.as_mut_ptr()) } != 0 {
        return Err(PairError::Create(io::Error::last_os_error()));
    }
    let (first, second) = (adopt(fds[0]), adopt(fds[1]));
    let (first, second) = (secure(first)?, secure(second)?);
    set_nosigpipe(first.as_raw_fd()).map_err(PairError::NoSigpipe)?;
    set_nosigpipe(second.as_raw_fd()).map_err(PairError::NoSigpipe)?;
    Ok((first, second))
}

/// `setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, 1)`.
pub fn set_nosigpipe(fd: RawFd) -> io::Result<()> {
    let yes: c_int = 1;
    // SAFETY: the option value points to one readable int of the given size.
    check(unsafe {
        ffi::setsockopt(
            fd,
            ffi::SOL_SOCKET,
            ffi::SO_NOSIGPIPE,
            std::ptr::from_ref(&yes).cast(),
            size_of::<c_int>() as ffi::SocklenT,
        )
    })
    .map(drop)
}

/// `getsockopt(fd, SOL_SOCKET, SO_TYPE)`: the socket type and the length the kernel wrote.
pub fn socket_type(fd: BorrowedFd<'_>) -> io::Result<(c_int, usize)> {
    let mut kind: c_int = 0;
    let mut length = size_of::<c_int>() as ffi::SocklenT;
    // SAFETY: value and length point to writable storage of the stated size.
    check(unsafe {
        ffi::getsockopt(
            fd.as_raw_fd(),
            ffi::SOL_SOCKET,
            ffi::SO_TYPE,
            std::ptr::from_mut(&mut kind).cast(),
            &mut length,
        )
    })?;
    Ok((kind, length as usize))
}

/// `fcntl(fd, F_GETFL)`.
pub fn status_flags(fd: BorrowedFd<'_>) -> io::Result<c_int> {
    // SAFETY: F_GETFL takes no argument and reads descriptor state.
    check(unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_GETFL, 0 as c_int) })
}

/// `fcntl(fd, F_SETFL, flags | O_NONBLOCK)`.
pub fn set_status_nonblocking(fd: BorrowedFd<'_>, flags: c_int) -> io::Result<()> {
    // SAFETY: F_SETFL with an int flag word.
    check(unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_SETFL, flags | ffi::O_NONBLOCK) }).map(drop)
}

/// `fcntl(fd, F_GETFD)`.
pub fn descriptor_flags(fd: BorrowedFd<'_>) -> io::Result<c_int> {
    // SAFETY: F_GETFD takes no argument and reads descriptor state.
    check(unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_GETFD, 0 as c_int) })
}

/// `fcntl(fd, F_SETFD, flags | FD_CLOEXEC)`.
pub fn set_descriptor_cloexec(fd: BorrowedFd<'_>, flags: c_int) -> io::Result<()> {
    // SAFETY: F_SETFD with an int flag word.
    check(unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_SETFD, flags | ffi::FD_CLOEXEC) }).map(drop)
}

/// `nonblocking(fd)` in owned_worker.cpp: get then set `O_NONBLOCK`.
pub fn make_nonblocking(fd: BorrowedFd<'_>) -> io::Result<()> {
    let flags = status_flags(fd)?;
    set_status_nonblocking(fd, flags)
}

/// One `read(2)`.
pub fn read(fd: BorrowedFd<'_>, buffer: &mut [u8]) -> io::Result<usize> {
    // SAFETY: the pointer and length describe one writable slice.
    let count = unsafe { ffi::read(fd.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) };
    usize::try_from(count).map_err(|_| io::Error::last_os_error())
}

/// One `send(2)`.
pub fn send(fd: BorrowedFd<'_>, data: &[u8], flags: c_int) -> io::Result<usize> {
    // SAFETY: the pointer and length describe one readable slice.
    let count = unsafe { ffi::send(fd.as_raw_fd(), data.as_ptr().cast(), data.len(), flags) };
    usize::try_from(count).map_err(|_| io::Error::last_os_error())
}

/// One `recv(2)`.
pub fn recv(fd: BorrowedFd<'_>, buffer: &mut [u8], flags: c_int) -> io::Result<usize> {
    // SAFETY: the pointer and length describe one writable slice.
    let count = unsafe {
        ffi::recv(
            fd.as_raw_fd(),
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            flags,
        )
    };
    usize::try_from(count).map_err(|_| io::Error::last_os_error())
}

pub use ffi::{MSG_DONTWAIT, MSG_PEEK};

/// One `poll(2)` over `fds`, returning the ready count.
pub fn poll(fds: &mut [ffi::PollFd], timeout_ms: c_int) -> io::Result<usize> {
    // SAFETY: the pointer and count describe one writable slice of pollfd.
    let ready = unsafe { ffi::poll(fds.as_mut_ptr(), fds.len() as ffi::NfdsT, timeout_ms) };
    usize::try_from(ready).map_err(|_| io::Error::last_os_error())
}

/// `poll(nullptr-equivalent, 0, ms)`: the C++ uses an empty poll as a bounded sleep that a signal can interrupt.
pub fn pause(timeout_ms: c_int) {
    let mut none: [ffi::PollFd; 0] = [];
    // SAFETY: a zero-length poll reads no descriptors.
    unsafe { ffi::poll(none.as_mut_ptr(), 0, timeout_ms) };
}

/// `openat(dir, name, flags, mode)`, adopting the result as a `File`.
pub fn openat(dir: BorrowedFd<'_>, name: &CStr, flags: c_int, mode: u32) -> io::Result<File> {
    // SAFETY: `name` is NUL-terminated; mode is passed as the promoted unsigned int the variadic prototype expects.
    let fd = check(unsafe { ffi::openat(dir.as_raw_fd(), name.as_ptr(), flags, mode as c_int) })?;
    Ok(File::from(adopt(fd)))
}

pub use ffi::{
    AT_FDCWD, AT_REMOVEDIR, AT_SYMLINK_NOFOLLOW, F_NOCACHE, LOCK_EX, LOCK_NB, LOCK_UN, O_CLOEXEC,
    O_CREAT, O_DIRECTORY, O_EXCL, O_NOFOLLOW, O_RDONLY, O_RDWR, O_WRONLY, RENAME_EXCL, S_IFDIR,
    S_IFLNK, S_IFMT, S_IFREG,
};

/// Filesystem capacity and unprivileged availability reported by `fstatvfs(2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilesystemSpace {
    pub available_bytes: u64,
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub fragment_bytes: u64,
}

/// File metadata reported by `fstat(2)` / `fstatat(2)` on macOS `arm64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStat {
    pub dev: i32,
    pub ino: u64,
    pub mode: u16,
    pub nlink: u16,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
}

impl FileStat {
    fn from_raw(raw: &ffi::Stat) -> io::Result<Self> {
        let size = u64::try_from(raw.st_size).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "negative file size in stat")
        })?;
        let mtime_ns = i128::from(raw.st_mtimespec.tv_sec) * 1_000_000_000
            + i128::from(raw.st_mtimespec.tv_nsec);
        let ctime_ns = i128::from(raw.st_ctimespec.tv_sec) * 1_000_000_000
            + i128::from(raw.st_ctimespec.tv_nsec);
        Ok(Self {
            dev: raw.st_dev,
            ino: raw.st_ino,
            mode: raw.st_mode,
            nlink: raw.st_nlink,
            uid: raw.st_uid,
            gid: raw.st_gid,
            size,
            mtime_ns,
            ctime_ns,
        })
    }

    pub const fn is_regular(&self) -> bool {
        (self.mode & S_IFMT) == S_IFREG
    }

    pub const fn is_dir(&self) -> bool {
        (self.mode & S_IFMT) == S_IFDIR
    }

    pub const fn is_symlink(&self) -> bool {
        (self.mode & S_IFMT) == S_IFLNK
    }

    pub const fn permissions(&self) -> u16 {
        self.mode & 0o7777
    }
}

fn zeroed_stat() -> ffi::Stat {
    ffi::Stat {
        st_dev: 0,
        st_mode: 0,
        st_nlink: 0,
        st_ino: 0,
        st_uid: 0,
        st_gid: 0,
        st_rdev: 0,
        st_atimespec: ffi::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        st_mtimespec: ffi::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        st_ctimespec: ffi::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        st_birthtimespec: ffi::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        st_size: 0,
        st_blocks: 0,
        st_blksize: 0,
        st_flags: 0,
        st_gen: 0,
        st_lspare: 0,
        st_qspare: [0; 2],
    }
}

/// `flock(fd, LOCK_EX | LOCK_NB)`, shared with Python `fcntl.flock(..., LOCK_EX | LOCK_NB)` on `serve.lock`.
pub fn flock_exclusive_nonblocking(fd: BorrowedFd<'_>) -> io::Result<()> {
    loop {
        // SAFETY: flock operates only on the valid borrowed descriptor.
        let rc = unsafe { ffi::flock(fd.as_raw_fd(), ffi::LOCK_EX | ffi::LOCK_NB) };
        if rc == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(ffi::EINTR) {
            continue;
        }
        return Err(err);
    }
}

/// `flock(fd, LOCK_UN)`.
pub fn flock_unlock(fd: BorrowedFd<'_>) -> io::Result<()> {
    // SAFETY: flock operates only on the valid borrowed descriptor.
    check(unsafe { ffi::flock(fd.as_raw_fd(), ffi::LOCK_UN) }).map(drop)
}

/// `fcntl(fd, F_NOCACHE, enabled)`.
pub fn set_nocache(fd: BorrowedFd<'_>, enabled: bool) -> io::Result<()> {
    let flag: c_int = c_int::from(enabled);
    // SAFETY: F_NOCACHE takes one int flag word on a valid descriptor.
    check(unsafe { ffi::fcntl(fd.as_raw_fd(), ffi::F_NOCACHE, flag) }).map(drop)
}

/// `fstatvfs(fd)`: queries available, free and total bytes on `fd`'s filesystem.
pub fn filesystem_space(fd: BorrowedFd<'_>) -> io::Result<FilesystemSpace> {
    let mut raw = ffi::Statvfs {
        f_bsize: 0,
        f_frsize: 0,
        f_blocks: 0,
        f_bfree: 0,
        f_bavail: 0,
        f_files: 0,
        f_ffree: 0,
        f_favail: 0,
        f_fsid: 0,
        f_flag: 0,
        f_namemax: 0,
    };
    // SAFETY: `raw` is writable storage matching Darwin `struct statvfs`.
    check(unsafe { ffi::fstatvfs(fd.as_raw_fd(), &mut raw) })?;
    let unit = if raw.f_frsize != 0 {
        raw.f_frsize
    } else {
        raw.f_bsize
    };
    Ok(FilesystemSpace {
        available_bytes: u64::from(raw.f_bavail).saturating_mul(unit),
        free_bytes: u64::from(raw.f_bfree).saturating_mul(unit),
        total_bytes: u64::from(raw.f_blocks).saturating_mul(unit),
        fragment_bytes: unit,
    })
}

/// `fstat(fd)`.
pub fn fstat(fd: BorrowedFd<'_>) -> io::Result<FileStat> {
    let mut raw = zeroed_stat();
    // SAFETY: `raw` is writable storage matching Darwin `struct stat`.
    check(unsafe { ffi::fstat(fd.as_raw_fd(), &mut raw) })?;
    FileStat::from_raw(&raw)
}

/// Reads the file's volume UUID, with None for filesystems that do not supply one.
pub fn file_volume_uuid(fd: BorrowedFd<'_>) -> io::Result<Option<[u8; 16]>> {
    let mut attrs = ffi::AttrList {
        bitmapcount: ffi::ATTR_BIT_MAP_COUNT,
        reserved: 0,
        commonattr: 0,
        volattr: ffi::ATTR_VOL_INFO | ffi::ATTR_VOL_UUID,
        dirattr: 0,
        fileattr: 0,
        forkattr: 0,
    };
    let mut reply = ffi::VolumeUuidReply {
        length: 0,
        uuid: [0; 16],
    };
    // SAFETY: the borrowed descriptor remains open, the request matches Darwin attrlist and the reply has space for the length and requested UUID.
    let status = unsafe {
        ffi::fgetattrlist(
            fd.as_raw_fd(),
            &mut attrs,
            std::ptr::from_mut(&mut reply).cast(),
            std::mem::size_of_val(&reply),
            0,
        )
    };
    if status != 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Unsupported {
            return Ok(None);
        }
        return Err(error);
    }
    if reply.length as usize != std::mem::size_of_val(&reply) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid volume attribute length",
        ));
    }
    Ok((reply.uuid != [0; 16]).then_some(reply.uuid))
}

/// `fstatat(dir, name, ..., AT_SYMLINK_NOFOLLOW)`.
pub fn lstat_at(dir: BorrowedFd<'_>, name: &CStr) -> io::Result<FileStat> {
    let mut raw = zeroed_stat();
    // SAFETY: `name` is NUL-terminated and `raw` is writable storage matching Darwin `struct stat`.
    check(unsafe {
        ffi::fstatat(
            dir.as_raw_fd(),
            name.as_ptr(),
            &mut raw,
            ffi::AT_SYMLINK_NOFOLLOW,
        )
    })?;
    FileStat::from_raw(&raw)
}

/// `fstatat(AT_FDCWD, path, ..., AT_SYMLINK_NOFOLLOW)`.
pub fn lstat_path(path: &CStr) -> io::Result<FileStat> {
    let mut raw = zeroed_stat();
    // SAFETY: `path` is NUL-terminated and `raw` is writable storage matching Darwin `struct stat`.
    check(unsafe {
        ffi::fstatat(
            ffi::AT_FDCWD,
            path.as_ptr(),
            &mut raw,
            ffi::AT_SYMLINK_NOFOLLOW,
        )
    })?;
    FileStat::from_raw(&raw)
}

/// Positioned read of exact `buffer.len()` bytes starting at `offset`, retrying on `EINTR` and partial reads.
pub fn pread_exact(fd: BorrowedFd<'_>, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buffer.is_empty() {
        let pos = ffi::OffT::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "pread offset overflow"))?;
        // SAFETY: `buffer` is writable for `buffer.len()` bytes and `fd` is borrowed for the call.
        let count = unsafe {
            ffi::pread(
                fd.as_raw_fd(),
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                pos,
            )
        };
        if count < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(ffi::EINTR) {
                continue;
            }
            return Err(err);
        }
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positioned read",
            ));
        }
        let read_bytes = count as usize;
        offset = offset
            .checked_add(read_bytes as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pread offset overflow"))?;
        buffer = &mut buffer[read_bytes..];
    }
    Ok(())
}

/// Positioned write of all `data` bytes starting at `offset`, retrying on `EINTR` and partial writes.
pub fn pwrite_all(fd: BorrowedFd<'_>, mut data: &[u8], mut offset: u64) -> io::Result<()> {
    while !data.is_empty() {
        let pos = ffi::OffT::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "pwrite offset overflow"))?;
        // SAFETY: `data` is readable for `data.len()` bytes and `fd` is borrowed for the call.
        let count = unsafe { ffi::pwrite(fd.as_raw_fd(), data.as_ptr().cast(), data.len(), pos) };
        if count < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(ffi::EINTR) {
                continue;
            }
            return Err(err);
        }
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "zero-byte positioned write",
            ));
        }
        let written = count as usize;
        offset = offset
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pwrite offset overflow"))?;
        data = &data[written..];
    }
    Ok(())
}

/// `ftruncate(fd, length)`.
pub fn ftruncate(fd: BorrowedFd<'_>, length: u64) -> io::Result<()> {
    let raw = ffi::OffT::try_from(length)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ftruncate length overflow"))?;
    // SAFETY: ftruncate has no memory preconditions.
    check(unsafe { ffi::ftruncate(fd.as_raw_fd(), raw) }).map(drop)
}

/// `fchmod(fd, mode)`.
pub fn fchmod(fd: BorrowedFd<'_>, mode: u16) -> io::Result<()> {
    // SAFETY: fchmod has no memory preconditions.
    check(unsafe { ffi::fchmod(fd.as_raw_fd(), mode) }).map(drop)
}

/// `unlinkat(dir, name, flag)`.
pub fn unlink_at(dir: BorrowedFd<'_>, name: &CStr, remove_dir: bool) -> io::Result<()> {
    let flag = if remove_dir { ffi::AT_REMOVEDIR } else { 0 };
    // SAFETY: `name` is NUL-terminated and `dir` is a valid borrowed descriptor.
    check(unsafe { ffi::unlinkat(dir.as_raw_fd(), name.as_ptr(), flag) }).map(drop)
}

/// `renameatx_np(from_dir, from, to_dir, to, RENAME_EXCL)`: refuses with `AlreadyExists` (`EEXIST`) if `to` exists.
pub fn rename_exclusive_at(
    from_dir: BorrowedFd<'_>,
    from: &CStr,
    to_dir: BorrowedFd<'_>,
    to: &CStr,
) -> io::Result<()> {
    // SAFETY: `from` and `to` are NUL-terminated and both directory descriptors are borrowed for the call; `RENAME_EXCL` (`0x4`) prevents replacing an existing destination.
    check(unsafe {
        ffi::renameatx_np(
            from_dir.as_raw_fd(),
            from.as_ptr(),
            to_dir.as_raw_fd(),
            to.as_ptr(),
            ffi::RENAME_EXCL,
        )
    })
    .map(drop)
}

/// `fsync(2)`. std's `File::sync_all` uses `F_FULLFSYNC` on macOS, which the C++ does not, and which directories may refuse.
pub fn fsync(fd: BorrowedFd<'_>) -> io::Result<()> {
    // SAFETY: fsync has no memory preconditions.
    check(unsafe { ffi::fsync(fd.as_raw_fd()) }).map(drop)
}

/// Whether a descriptor number is open (`fcntl(fd, F_GETFD) >= 0`), for leak checks in single-threaded tests.
pub fn is_open(fd: RawFd) -> bool {
    // SAFETY: F_GETFD only queries the descriptor table; an unused number fails with EBADF.
    unsafe { ffi::fcntl(fd, ffi::F_GETFD, 0 as c_int) >= 0 }
}

/// `(fcntl(F_GETFL) & O_NONBLOCK != 0, fcntl(F_GETFD) & FD_CLOEXEC != 0)`.
pub fn nonblocking_cloexec(fd: RawFd) -> (bool, bool) {
    // SAFETY: both commands only query descriptor state.
    let (status, descriptor) = unsafe {
        (
            ffi::fcntl(fd, ffi::F_GETFL, 0 as c_int),
            ffi::fcntl(fd, ffi::F_GETFD, 0 as c_int),
        )
    };
    (
        status >= 0 && status & ffi::O_NONBLOCK != 0,
        descriptor >= 0 && descriptor & ffi::FD_CLOEXEC != 0,
    )
}

/// One `recv(2)` on a descriptor number (test peers and peeks).
pub fn recv_raw(fd: RawFd, buffer: &mut [u8], flags: c_int) -> io::Result<usize> {
    // SAFETY: the pointer and length describe one writable slice; an invalid descriptor fails with EBADF.
    let count = unsafe { ffi::recv(fd, buffer.as_mut_ptr().cast(), buffer.len(), flags) };
    usize::try_from(count).map_err(|_| io::Error::last_os_error())
}

/// `setsockopt(fd, SOL_SOCKET, SO_SNDBUF, bytes)`.
pub fn set_send_buffer(fd: RawFd, bytes: c_int) -> io::Result<()> {
    // SAFETY: the option value points to one readable int of the given size.
    check(unsafe {
        ffi::setsockopt(
            fd,
            ffi::SOL_SOCKET,
            ffi::SO_SNDBUF,
            std::ptr::from_ref(&bytes).cast(),
            size_of::<c_int>() as ffi::SocklenT,
        )
    })
    .map(drop)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn pair_error_retains_source_and_redacts_debug() {
        let err = PairError::Create(io::Error::other("/private/socket/secret"));
        assert!(err.source().is_some());
        let dbg = format!("{err:?}");
        assert!(!dbg.contains("secret"), "unexpected secret in Debug: {dbg}");
        assert_eq!(dbg, "Create([redacted])");
    }
}
