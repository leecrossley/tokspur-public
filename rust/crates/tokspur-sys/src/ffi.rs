//! Hand-written declarations of the Apple libSystem, libobjc and CoreFoundation symbols this crate uses, for macOS on arm64. Constants and layouts are copied from the macOS SDK headers named beside each group.

use std::ffi::{c_char, c_int, c_long, c_short, c_uint, c_ulong, c_void};

pub type PidT = i32;
pub type UidT = u32;
pub type IdT = u32;
pub type SocklenT = u32;
pub type NfdsT = c_uint;
pub type RlimT = u64;

// <poll.h>
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PollFd {
    pub fd: c_int,
    pub events: c_short,
    pub revents: c_short,
}
pub const POLLIN: c_short = 0x0001;
pub const POLLOUT: c_short = 0x0004;
pub const POLLERR: c_short = 0x0008;
pub const POLLHUP: c_short = 0x0010;
pub const POLLNVAL: c_short = 0x0020;

// <sys/resource.h>
#[repr(C)]
pub struct Rlimit {
    pub rlim_cur: RlimT,
    pub rlim_max: RlimT,
}
pub const RLIMIT_NOFILE: c_int = 8;

// <sys/signal.h>: the user-facing `struct sigaction`.
#[repr(C)]
pub struct Sigaction {
    pub sa_handler: usize,
    pub sa_mask: u32,
    pub sa_flags: c_int,
}
pub const SIG_DFL: usize = 0;
pub const SIG_IGN: usize = 1;
pub const SIG_ERR: usize = usize::MAX;
pub const SIGHUP: c_int = 1;
pub const SIGINT: c_int = 2;
pub const SIGKILL: c_int = 9;
pub const SIGPIPE: c_int = 13;
pub const SIGTERM: c_int = 15;
pub const SIGCHLD: c_int = 20;

// <sys/signal.h> `siginfo_t` (struct __siginfo).
#[repr(C)]
pub struct Siginfo {
    pub si_signo: c_int,
    pub si_errno: c_int,
    pub si_code: c_int,
    pub si_pid: PidT,
    pub si_uid: UidT,
    pub si_status: c_int,
    pub si_addr: *mut c_void,
    pub si_value: usize,
    pub si_band: c_long,
    pub pad: [c_ulong; 7],
}

// <sys/wait.h>
pub const P_PID: c_int = 1;
pub const WNOHANG: c_int = 0x0000_0001;
pub const WEXITED: c_int = 0x0000_0004;
pub const WNOWAIT: c_int = 0x0000_0020;

// <sys/fcntl.h>, <sys/file.h>, <sys/stdio.h>
pub const F_GETFD: c_int = 1;
pub const F_SETFD: c_int = 2;
pub const F_GETFL: c_int = 3;
pub const F_SETFL: c_int = 4;
pub const F_NOCACHE: c_int = 48;
pub const F_DUPFD_CLOEXEC: c_int = 67;
pub const FD_CLOEXEC: c_int = 1;
pub const O_RDWR: c_int = 0x0002;
pub const O_WRONLY: c_int = 0x0001;
pub const O_RDONLY: c_int = 0x0000;
pub const O_NONBLOCK: c_int = 0x0004;
pub const O_NOFOLLOW: c_int = 0x0100;
pub const O_CREAT: c_int = 0x0200;
pub const O_EXCL: c_int = 0x0800;
pub const O_CLOEXEC: c_int = 0x0100_0000;
pub const O_DIRECTORY: c_int = 0x0010_0000;
pub const AT_FDCWD: c_int = -2;
pub const AT_SYMLINK_NOFOLLOW: c_int = 0x0020;
pub const AT_REMOVEDIR: c_int = 0x0080;
pub const LOCK_EX: c_int = 0x0002;
pub const LOCK_NB: c_int = 0x0004;
pub const LOCK_UN: c_int = 0x0008;
pub const RENAME_EXCL: c_uint = 0x0000_0004;

// <sys/stat.h>, <sys/statvfs.h>
pub type OffT = i64;
pub type ModeT = u16;
pub const S_IFMT: ModeT = 0o170_000;
pub const S_IFDIR: ModeT = 0o040_000;
pub const S_IFREG: ModeT = 0o100_000;
pub const S_IFLNK: ModeT = 0o120_000;

// <sys/attr.h>
pub const ATTR_BIT_MAP_COUNT: u16 = 5;
pub const ATTR_VOL_UUID: u32 = 0x0004_0000;
pub const ATTR_VOL_INFO: u32 = 0x8000_0000;

#[repr(C)]
pub struct AttrList {
    pub bitmapcount: u16,
    pub reserved: u16,
    pub commonattr: u32,
    pub volattr: u32,
    pub dirattr: u32,
    pub fileattr: u32,
    pub forkattr: u32,
}

#[repr(C)]
pub struct VolumeUuidReply {
    pub length: u32,
    pub uuid: [u8; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timespec {
    pub tv_sec: c_long,
    pub tv_nsec: c_long,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Stat {
    pub st_dev: i32,
    pub st_mode: ModeT,
    pub st_nlink: u16,
    pub st_ino: u64,
    pub st_uid: UidT,
    pub st_gid: u32,
    pub st_rdev: i32,
    pub st_atimespec: Timespec,
    pub st_mtimespec: Timespec,
    pub st_ctimespec: Timespec,
    pub st_birthtimespec: Timespec,
    pub st_size: OffT,
    pub st_blocks: i64,
    pub st_blksize: i32,
    pub st_flags: u32,
    pub st_gen: u32,
    pub st_lspare: i32,
    pub st_qspare: [i64; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Statvfs {
    pub f_bsize: c_ulong,
    pub f_frsize: c_ulong,
    pub f_blocks: u32,
    pub f_bfree: u32,
    pub f_bavail: u32,
    pub f_files: u32,
    pub f_ffree: u32,
    pub f_favail: u32,
    pub f_fsid: c_ulong,
    pub f_flag: c_ulong,
    pub f_namemax: c_ulong,
}

// <sys/sysctl.h>, <sys/param.h>, <sys/proc.h>, <sys/proc_info.h>, <sys/resource.h>, <libproc.h>
pub const CTL_KERN: c_int = 1;
pub const KERN_PROCARGS2: c_int = 49;
pub const PROC_PIDTBSDINFO: c_int = 3;
pub const MAXCOMLEN: usize = 16;
pub const SZOMB: u32 = 5;
pub const PROC_PIDPATHINFO_MAXSIZE: usize = 4096;
pub const RUSAGE_INFO_V4: c_int = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProcBsdinfo {
    pub pbi_flags: u32,
    pub pbi_status: u32,
    pub pbi_xstatus: u32,
    pub pbi_pid: u32,
    pub pbi_ppid: u32,
    pub pbi_uid: UidT,
    pub pbi_gid: u32,
    pub pbi_ruid: UidT,
    pub pbi_rgid: u32,
    pub pbi_svuid: UidT,
    pub pbi_svgid: u32,
    pub rfu_1: u32,
    pub pbi_comm: [c_char; MAXCOMLEN],
    pub pbi_name: [c_char; 2 * MAXCOMLEN],
    pub pbi_nfiles: u32,
    pub pbi_pgid: u32,
    pub pbi_pjobc: u32,
    pub e_tdev: u32,
    pub e_tpgid: u32,
    pub pbi_nice: i32,
    pub pbi_start_tvsec: u64,
    pub pbi_start_tvusec: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RusageInfoV4 {
    pub ri_uuid: [u8; 16],
    pub ri_user_time: u64,
    pub ri_system_time: u64,
    pub ri_pkg_idle_wkups: u64,
    pub ri_interrupt_wkups: u64,
    pub ri_pageins: u64,
    pub ri_wired_size: u64,
    pub ri_resident_size: u64,
    pub ri_phys_footprint: u64,
    pub ri_proc_start_abstime: u64,
    pub ri_proc_exit_abstime: u64,
    pub ri_child_user_time: u64,
    pub ri_child_system_time: u64,
    pub ri_child_pkg_idle_wkups: u64,
    pub ri_child_interrupt_wkups: u64,
    pub ri_child_pageins: u64,
    pub ri_child_elapsed_abstime: u64,
    pub ri_diskio_bytesread: u64,
    pub ri_diskio_byteswritten: u64,
    pub ri_cpu_time_qos_default: u64,
    pub ri_cpu_time_qos_maintenance: u64,
    pub ri_cpu_time_qos_background: u64,
    pub ri_cpu_time_qos_utility: u64,
    pub ri_cpu_time_qos_legacy: u64,
    pub ri_cpu_time_qos_user_initiated: u64,
    pub ri_cpu_time_qos_user_interactive: u64,
    pub ri_billed_system_time: u64,
    pub ri_serviced_system_time: u64,
    pub ri_logical_writes: u64,
    pub ri_lifetime_max_phys_footprint: u64,
    pub ri_instructions: u64,
    pub ri_cycles: u64,
    pub ri_billed_energy: u64,
    pub ri_serviced_energy: u64,
    pub ri_interval_max_phys_footprint: u64,
    pub ri_runnable_time: u64,
}

// <sys/socket.h>, <netinet/in.h>
pub const AF_UNIX: c_int = 1;
pub const AF_INET: c_int = 2;
pub const SOCK_STREAM: c_int = 1;
pub const SOL_SOCKET: c_int = 0xffff;
pub const SO_REUSEADDR: c_int = 0x0004;
pub const SO_TYPE: c_int = 0x1008;
pub const SO_ERROR: c_int = 0x1007;
pub const SO_SNDBUF: c_int = 0x1001;
pub const SO_NOSIGPIPE: c_int = 0x1022;
pub const MSG_PEEK: c_int = 0x2;
pub const MSG_DONTWAIT: c_int = 0x80;
pub const INADDR_ANY: u32 = 0x0000_0000;
pub const INADDR_LOOPBACK: u32 = 0x7f00_0001;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InAddr {
    pub s_addr: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SockaddrIn {
    pub sin_len: u8,
    pub sin_family: u8,
    pub sin_port: u16,
    pub sin_addr: InAddr,
    pub sin_zero: [c_char; 8],
}

// <unistd.h>
pub const X_OK: c_int = 1;

// <CommonCrypto/CommonDigest.h>
pub type CcLong = u32;
pub const CC_SHA256_DIGEST_LENGTH: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CcSha256Ctx {
    pub count: [CcLong; 2],
    pub hash: [CcLong; 8],
    pub wbuf: [CcLong; 16],
}

// <errno.h>
pub const ESRCH: c_int = 3;
pub const EINTR: c_int = 4;
pub const EEXIST: c_int = 17;
pub const ERANGE: c_int = 34;
pub const EAGAIN: c_int = 35;
pub const EINPROGRESS: c_int = 36;
pub const ENOTSUP: c_int = 45;
pub const EADDRINUSE: c_int = 48;
pub const ECONNREFUSED: c_int = 61;

pub const PROC_PIDLISTFDS: c_int = 1;
pub const SO_LINGER: c_int = 0x0080;

#[repr(C)]
pub struct Linger {
    pub enabled: c_int,
    pub seconds: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProcFdInfo {
    pub fd: i32,
    pub kind: u32,
}

#[link(name = "System")]
unsafe extern "C" {
    pub fn __error() -> *mut c_int;
    pub fn strerror_r(errnum: c_int, buffer: *mut c_char, length: usize) -> c_int;
    pub fn fork() -> PidT;
    pub fn _exit(status: c_int) -> !;
    pub fn execve(
        path: *const c_char,
        argv: *const *const c_char,
        envp: *const *const c_char,
    ) -> c_int;
    pub fn getpid() -> PidT;
    pub fn getppid() -> PidT;
    pub fn setsid() -> PidT;
    pub fn setpgid(pid: PidT, pgid: PidT) -> c_int;
    pub fn getpgid(pid: PidT) -> PidT;
    pub fn getsid(pid: PidT) -> PidT;
    pub fn geteuid() -> UidT;
    pub fn kill(pid: PidT, signal: c_int) -> c_int;
    pub fn killpg(pgrp: PidT, signal: c_int) -> c_int;
    pub fn waitpid(pid: PidT, status: *mut c_int, options: c_int) -> PidT;
    pub fn waitid(idtype: c_int, id: IdT, info: *mut Siginfo, options: c_int) -> c_int;
    pub fn sigaction(signal: c_int, action: *const Sigaction, old: *mut Sigaction) -> c_int;
    pub fn signal(signal: c_int, handler: usize) -> usize;
    pub fn getrlimit(resource: c_int, limit: *mut Rlimit) -> c_int;
    pub fn access(path: *const c_char, mode: c_int) -> c_int;
    pub fn pipe(fds: *mut c_int) -> c_int;
    pub fn socketpair(domain: c_int, kind: c_int, protocol: c_int, fds: *mut c_int) -> c_int;
    pub fn close(fd: c_int) -> c_int;
    pub fn fsync(fd: c_int) -> c_int;
    pub fn dup2(from: c_int, to: c_int) -> c_int;
    pub fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    pub fn flock(fd: c_int, operation: c_int) -> c_int;
    pub fn openat(dirfd: c_int, path: *const c_char, flags: c_int, ...) -> c_int;
    pub fn fstat(fd: c_int, buf: *mut Stat) -> c_int;
    pub fn fgetattrlist(
        fd: c_int,
        attrs: *mut AttrList,
        buffer: *mut c_void,
        size: usize,
        options: c_uint,
    ) -> c_int;
    pub fn fstatat(dirfd: c_int, path: *const c_char, buf: *mut Stat, flag: c_int) -> c_int;
    pub fn fstatvfs(fd: c_int, buf: *mut Statvfs) -> c_int;
    pub fn pread(fd: c_int, buffer: *mut c_void, count: usize, offset: OffT) -> isize;
    pub fn pwrite(fd: c_int, buffer: *const c_void, count: usize, offset: OffT) -> isize;
    pub fn ftruncate(fd: c_int, length: OffT) -> c_int;
    pub fn fchmod(fd: c_int, mode: ModeT) -> c_int;
    pub fn unlinkat(dirfd: c_int, path: *const c_char, flag: c_int) -> c_int;
    pub fn renameatx_np(
        fromfd: c_int,
        from: *const c_char,
        tofd: c_int,
        to: *const c_char,
        flags: c_uint,
    ) -> c_int;
    pub fn read(fd: c_int, buffer: *mut c_void, count: usize) -> isize;
    pub fn write(fd: c_int, buffer: *const c_void, count: usize) -> isize;
    pub fn poll(fds: *mut PollFd, count: NfdsT, timeout: c_int) -> c_int;
    pub fn socket(domain: c_int, kind: c_int, protocol: c_int) -> c_int;
    pub fn bind(fd: c_int, address: *const SockaddrIn, length: SocklenT) -> c_int;
    pub fn connect(fd: c_int, address: *const SockaddrIn, length: SocklenT) -> c_int;
    pub fn listen(fd: c_int, backlog: c_int) -> c_int;
    pub fn accept(fd: c_int, address: *mut SockaddrIn, length: *mut SocklenT) -> c_int;
    pub fn getsockname(fd: c_int, address: *mut SockaddrIn, length: *mut SocklenT) -> c_int;
    pub fn getpeername(fd: c_int, address: *mut SockaddrIn, length: *mut SocklenT) -> c_int;
    pub fn getsockopt(
        fd: c_int,
        level: c_int,
        name: c_int,
        value: *mut c_void,
        length: *mut SocklenT,
    ) -> c_int;
    pub fn setsockopt(
        fd: c_int,
        level: c_int,
        name: c_int,
        value: *const c_void,
        length: SocklenT,
    ) -> c_int;
    pub fn send(fd: c_int, buffer: *const c_void, length: usize, flags: c_int) -> isize;
    pub fn recv(fd: c_int, buffer: *mut c_void, length: usize, flags: c_int) -> isize;
    pub fn arc4random_buf(buffer: *mut c_void, length: usize);
    pub fn sysctl(
        name: *mut c_int,
        namelen: c_uint,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *const c_void,
        newlen: usize,
    ) -> c_int;
    pub fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *const c_void,
        newlen: usize,
    ) -> c_int;
    pub fn proc_listallpids(buffer: *mut c_void, buffersize: c_int) -> c_int;
    pub fn proc_pidinfo(
        pid: c_int,
        flavor: c_int,
        arg: u64,
        buffer: *mut c_void,
        buffersize: c_int,
    ) -> c_int;
    pub fn proc_pidpath(pid: c_int, buffer: *mut c_void, buffersize: u32) -> c_int;
    pub fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut RusageInfoV4) -> c_int;
    /// The `CC_SHA256` routines always return 1 (`CC_SHA(3cc)`).
    pub fn CC_SHA256_Init(context: *mut CcSha256Ctx) -> c_int;
    pub fn CC_SHA256_Update(
        context: *mut CcSha256Ctx,
        data: *const c_void,
        length: CcLong,
    ) -> c_int;
    pub fn CC_SHA256_Final(digest: *mut u8, context: *mut CcSha256Ctx) -> c_int;
}

// <objc/runtime.h>, <objc/message.h>, <objc/objc-internal.h> (exported).
pub type Id = *mut c_void;
pub type Sel = *const c_void;

#[link(name = "objc")]
unsafe extern "C" {
    pub fn objc_getClass(name: *const c_char) -> Id;
    pub fn sel_registerName(name: *const c_char) -> Sel;
    /// Declared without a signature; each call site transmutes it to the exact method signature, as `<objc/message.h>` requires on arm64.
    pub fn objc_msgSend();
    pub fn objc_release(object: Id);
    pub fn objc_autoreleasePoolPush() -> *mut c_void;
    pub fn objc_autoreleasePoolPop(pool: *mut c_void);
}

// <CoreFoundation/CFString.h>
pub type CfIndex = isize;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFStringCreateWithCharacters(
        allocator: *const c_void,
        characters: *const u16,
        length: CfIndex,
    ) -> Id;
    pub fn CFRelease(object: *const c_void);
}

// NSRegularExpression lives in Foundation; linking it registers the class.
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

// <Block.h> and the clang block ABI (`Block-ABI-Apple`): the descriptor of a block literal with no copy or dispose helpers and no signature.
#[repr(C)]
pub struct BlockDescriptor {
    pub reserved: c_ulong,
    pub size: c_ulong,
}

#[link(name = "System")]
unsafe extern "C" {
    /// The `isa` of a block literal on the stack.
    #[allow(non_upper_case_globals)]
    pub static _NSConcreteStackBlock: [*const c_void; 32];
}

/// `NSRange`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NsRange {
    pub location: usize,
    pub length: usize,
}

/// Reads the calling thread's `errno`.
pub fn errno() -> c_int {
    // SAFETY: `__error` returns this thread's errno slot, valid for the thread's lifetime; reading an int from it is async-signal-safe.
    unsafe { *__error() }
}
