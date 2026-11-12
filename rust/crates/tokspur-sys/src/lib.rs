//! The only runtime crate allowed `unsafe`.
//!
//! It declares, by hand, the Apple libSystem, libobjc and CoreFoundation symbols the C++ serving layer and Rust CLI/pack tooling call directly, and wraps each in a typed safe function. No third-party code is involved. Every `unsafe` block carries a `SAFETY` comment; every other runtime crate forbids `unsafe`. The dev-only differential fuzz oracle has its own confined exception, documented in `rust/crates/tokspur-fuzz/Cargo.toml`. macOS on arm64 only, like the C++ server.

#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("tokspur-sys declares macOS arm64 layouts only");

pub mod digest;
pub mod fd;
mod ffi;
pub mod foundation;
pub mod host;
pub mod net;
pub mod process;

pub use ffi::{
    EADDRINUSE, EAGAIN, ECONNREFUSED, EEXIST, EINPROGRESS, EINTR, ESRCH, POLLERR, POLLHUP, POLLIN,
    POLLNVAL, POLLOUT, PollFd, SIGINT, SIGKILL, SIGTERM, errno,
};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The exact libc `strerror` text for `errnum` (`std::strerror` in C++).
pub fn strerror(errnum: i32) -> String {
    let mut buffer = [0 as std::ffi::c_char; 256];
    // SAFETY: the buffer is writable for its full length; strerror_r writes a NUL-terminated message (truncated if needed) and returns a status.
    let status = unsafe { ffi::strerror_r(errnum, buffer.as_mut_ptr(), buffer.len()) };
    if status != 0 && status != ffi::ERANGE {
        return format!("Unknown error: {errnum}");
    }
    // SAFETY: strerror_r NUL-terminates within the buffer on success/ERANGE.
    let text = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) };
    text.to_string_lossy().into_owned()
}

/// Fills `buffer` from the system CSPRNG (`arc4random_buf`).
pub fn random_bytes(buffer: &mut [u8]) {
    // SAFETY: the pointer and length describe one writable slice.
    unsafe { ffi::arc4random_buf(buffer.as_mut_ptr().cast(), buffer.len()) }
}

/// `geteuid()`.
pub fn effective_uid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { ffi::geteuid() }
}

static STOPPING: AtomicBool = AtomicBool::new(false);

extern "C" fn stop_signal(_signal: std::ffi::c_int) {
    STOPPING.store(true, Ordering::Relaxed);
}

/// `std::signal(SIGINT/SIGTERM/SIGHUP, stop_signal)` in that order, stopping at the first `SIG_ERR` as the C++ `&&` chain does. Returns whether all three were installed. The handler only stores a lock-free atomic.
pub fn install_stop_handlers() -> bool {
    let handler = stop_signal as extern "C" fn(std::ffi::c_int) as usize;
    [ffi::SIGINT, ffi::SIGTERM, ffi::SIGHUP]
        .into_iter()
        .all(|signal| {
            // SAFETY: the handler is async-signal-safe (one relaxed atomic store) and lives for the whole program.
            unsafe { ffi::signal(signal, handler) != ffi::SIG_ERR }
        })
}

/// Requests the same orderly shutdown as a stop signal.
pub fn request_stop() {
    STOPPING.store(true, Ordering::Relaxed);
}

/// Whether a stop signal arrived (`stopping.load(std::memory_order_relaxed)`).
pub fn stop_requested() -> bool {
    STOPPING.load(Ordering::Relaxed)
}

const CLI_STOP_SIGNALS: [std::ffi::c_int; 3] = [ffi::SIGINT, ffi::SIGTERM, ffi::SIGHUP];

/// The dispositions [`install_cli_stop_handlers`] replaced, restored by [`restore_cli_signal_handlers`].
static CLI_PREVIOUS: [AtomicUsize; 3] = [
    AtomicUsize::new(ffi::SIG_DFL),
    AtomicUsize::new(ffi::SIG_DFL),
    AtomicUsize::new(ffi::SIG_DFL),
];

/// Installs `SIGINT`, `SIGTERM` and `SIGHUP` stop handlers for `tokspur serve`, clearing any prior stop flag first. A signal the process inherited as ignored stays ignored, so `nohup tokspur serve` survives a hangup and a background job keeps ignoring `SIGINT`.
pub fn install_cli_stop_handlers() -> bool {
    STOPPING.store(false, Ordering::Relaxed);
    let handler = stop_signal as extern "C" fn(std::ffi::c_int) as usize;
    CLI_STOP_SIGNALS
        .into_iter()
        .zip(&CLI_PREVIOUS)
        .all(|(signal, previous)| {
            // SAFETY: async-signal-safe relaxed atomic store with `'static` function pointer.
            let replaced = unsafe { ffi::signal(signal, handler) };
            if replaced == ffi::SIG_ERR {
                return false;
            }
            previous.store(replaced, Ordering::Relaxed);
            // SAFETY: `SIG_IGN` is a standard kernel disposition with no memory preconditions.
            replaced != ffi::SIG_IGN || unsafe { ffi::signal(signal, ffi::SIG_IGN) } != ffi::SIG_ERR
        })
}

/// Restores the `SIGINT`, `SIGTERM` and `SIGHUP` dispositions that [`install_cli_stop_handlers`] replaced after `tokspur serve` completes.
pub fn restore_cli_signal_handlers() {
    for (signal, previous) in CLI_STOP_SIGNALS.into_iter().zip(&CLI_PREVIOUS) {
        // SAFETY: the value is `SIG_DFL`, `SIG_IGN` or a disposition this process installed earlier; each stays valid for the whole program.
        unsafe { ffi::signal(signal, previous.load(Ordering::Relaxed)) };
    }
}

/// Restores the default `SIGPIPE` disposition. Rust's runtime ignores `SIGPIPE` before `main`; the C++ server keeps the default, so a closed stdout pipe ends both processes the same way.
pub fn default_sigpipe() -> bool {
    // SAFETY: setting a standard disposition has no memory preconditions.
    unsafe { ffi::signal(ffi::SIGPIPE, ffi::SIG_DFL) != ffi::SIG_ERR }
}

/// Compile-time layout and constant descriptors verified against the active Darwin SDK in `tests/sdk_bindings.rs`.
#[doc(hidden)]
pub mod sdk_layout {
    use std::mem::{offset_of, size_of};

    use crate::ffi::{self, CcSha256Ctx, ProcBsdinfo, RusageInfoV4, Stat, Statvfs};

    pub const PROC_PIDLISTFDS: i32 = ffi::PROC_PIDLISTFDS;
    pub const PROC_FDINFO_SIZE: usize = size_of::<ffi::ProcFdInfo>();
    pub const SO_LINGER: i32 = ffi::SO_LINGER;
    pub const LINGER_SIZE: usize = size_of::<ffi::Linger>();
    pub const RENAME_EXCL: u32 = ffi::RENAME_EXCL;
    pub const F_NOCACHE: i32 = ffi::F_NOCACHE;
    pub const ENOTSUP: i32 = ffi::ENOTSUP;
    pub const AT_FDCWD: i32 = ffi::AT_FDCWD;
    pub const AT_SYMLINK_NOFOLLOW: i32 = ffi::AT_SYMLINK_NOFOLLOW;
    pub const AT_REMOVEDIR: i32 = ffi::AT_REMOVEDIR;
    pub const LOCK_EX: i32 = ffi::LOCK_EX;
    pub const LOCK_NB: i32 = ffi::LOCK_NB;
    pub const LOCK_UN: i32 = ffi::LOCK_UN;
    pub const S_IFMT: u16 = ffi::S_IFMT;
    pub const S_IFDIR: u16 = ffi::S_IFDIR;
    pub const S_IFREG: u16 = ffi::S_IFREG;
    pub const S_IFLNK: u16 = ffi::S_IFLNK;
    pub const SO_REUSEADDR: i32 = ffi::SO_REUSEADDR;
    pub const SO_ERROR: i32 = ffi::SO_ERROR;
    pub const EEXIST: i32 = ffi::EEXIST;
    pub const EINPROGRESS: i32 = ffi::EINPROGRESS;
    pub const EADDRINUSE: i32 = ffi::EADDRINUSE;
    pub const ECONNREFUSED: i32 = ffi::ECONNREFUSED;
    pub const CTL_KERN: i32 = ffi::CTL_KERN;
    pub const KERN_PROCARGS2: i32 = ffi::KERN_PROCARGS2;
    pub const PROC_PIDTBSDINFO: i32 = ffi::PROC_PIDTBSDINFO;
    pub const MAXCOMLEN: usize = ffi::MAXCOMLEN;
    pub const SZOMB: u32 = ffi::SZOMB;
    pub const PROC_PIDPATHINFO_MAXSIZE: usize = ffi::PROC_PIDPATHINFO_MAXSIZE;
    pub const RUSAGE_INFO_V4: i32 = ffi::RUSAGE_INFO_V4;

    pub const STAT_SIZE: usize = size_of::<Stat>();
    pub const ATTR_BIT_MAP_COUNT: u16 = ffi::ATTR_BIT_MAP_COUNT;
    pub const ATTR_VOL_UUID: u32 = ffi::ATTR_VOL_UUID;
    pub const ATTR_VOL_INFO: u32 = ffi::ATTR_VOL_INFO;
    pub const ATTRLIST_SIZE: usize = size_of::<ffi::AttrList>();
    pub const ATTRLIST_VOLATTR_OFF: usize = offset_of!(ffi::AttrList, volattr);
    pub const VOLUME_UUID_REPLY_SIZE: usize = size_of::<ffi::VolumeUuidReply>();
    pub const VOLUME_UUID_REPLY_UUID_OFF: usize = offset_of!(ffi::VolumeUuidReply, uuid);
    pub const STAT_DEV_OFF: usize = offset_of!(Stat, st_dev);
    pub const STAT_MODE_OFF: usize = offset_of!(Stat, st_mode);
    pub const STAT_NLINK_OFF: usize = offset_of!(Stat, st_nlink);
    pub const STAT_INO_OFF: usize = offset_of!(Stat, st_ino);
    pub const STAT_UID_OFF: usize = offset_of!(Stat, st_uid);
    pub const STAT_GID_OFF: usize = offset_of!(Stat, st_gid);
    pub const STAT_MTIMESPEC_OFF: usize = offset_of!(Stat, st_mtimespec);
    pub const STAT_CTIMESPEC_OFF: usize = offset_of!(Stat, st_ctimespec);
    pub const STAT_SIZE_OFF: usize = offset_of!(Stat, st_size);

    pub const STATVFS_SIZE: usize = size_of::<Statvfs>();
    pub const STATVFS_BSIZE_OFF: usize = offset_of!(Statvfs, f_bsize);
    pub const STATVFS_FRSIZE_OFF: usize = offset_of!(Statvfs, f_frsize);
    pub const STATVFS_BLOCKS_OFF: usize = offset_of!(Statvfs, f_blocks);
    pub const STATVFS_BFREE_OFF: usize = offset_of!(Statvfs, f_bfree);
    pub const STATVFS_BAVAIL_OFF: usize = offset_of!(Statvfs, f_bavail);

    pub const PROC_BSDINFO_SIZE: usize = size_of::<ProcBsdinfo>();
    pub const PROC_BSDINFO_STATUS_OFF: usize = offset_of!(ProcBsdinfo, pbi_status);
    pub const PROC_BSDINFO_PID_OFF: usize = offset_of!(ProcBsdinfo, pbi_pid);
    pub const PROC_BSDINFO_PPID_OFF: usize = offset_of!(ProcBsdinfo, pbi_ppid);
    pub const PROC_BSDINFO_UID_OFF: usize = offset_of!(ProcBsdinfo, pbi_uid);
    pub const PROC_BSDINFO_COMM_OFF: usize = offset_of!(ProcBsdinfo, pbi_comm);
    pub const PROC_BSDINFO_NAME_OFF: usize = offset_of!(ProcBsdinfo, pbi_name);
    pub const PROC_BSDINFO_PGID_OFF: usize = offset_of!(ProcBsdinfo, pbi_pgid);

    pub const RUSAGE_V4_SIZE: usize = size_of::<RusageInfoV4>();
    pub const RUSAGE_V4_PHYS_OFF: usize = offset_of!(RusageInfoV4, ri_phys_footprint);
    pub const RUSAGE_V4_PEAK_OFF: usize = offset_of!(RusageInfoV4, ri_lifetime_max_phys_footprint);

    pub const CC_SHA256_DIGEST_LENGTH: usize = ffi::CC_SHA256_DIGEST_LENGTH;
    pub const CC_SHA256_CTX_SIZE: usize = size_of::<CcSha256Ctx>();
    pub const CC_SHA256_CTX_HASH_OFF: usize = offset_of!(CcSha256Ctx, hash);
    pub const CC_SHA256_CTX_WBUF_OFF: usize = offset_of!(CcSha256Ctx, wbuf);
}
