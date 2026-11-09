//! The owned worker's guardian process and the parent-side wait/signal helpers, ported from `native/src/owned_worker.cpp`.
//!
//! Everything that runs in a forked child lives here and uses only async-signal-safe libSystem calls over storage prepared before `fork`: no allocation, locks, streams, Foundation or callbacks, as in the C++.

use std::ffi::{CString, c_char, c_int};
use std::io;
use std::os::fd::RawFd;

use crate::ffi::{self, PidT};

/// Guardian notices on the status pipe (`armed_notice` ...).
pub const ARMED_NOTICE: i32 = 1;
pub const LAUNCHED_NOTICE: i32 = 2;
pub const EXIT_NOTICE: i32 = 3;
pub const FAILURE_NOTICE: i32 = 4;
/// `sizeof(Notice)`: two native-endian ints.
pub const NOTICE_BYTES: usize = 8;

const POLL_SLICE_MS: c_int = 25;

/// The child-side descriptors the guardian keeps (`Channels` `[1]` ends plus both gate ends).
#[derive(Debug, Clone, Copy)]
pub struct GuardianFds {
    pub control: RawFd,
    pub status: RawFd,
    pub input: RawFd,
    pub output: RawFd,
    pub error: RawFd,
    pub gate_read: RawFd,
    pub gate_write: RawFd,
}

/// Everything the guardian needs, allocated before `fork`.
pub struct GuardianPlan {
    fds: GuardianFds,
    ceiling: c_int,
    parent: PidT,
    _argv: Vec<CString>,
    argv: Vec<*const c_char>,
    _environment: Vec<CString>,
    environment: Vec<*const c_char>,
}

impl std::fmt::Debug for GuardianPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Argument paths are local asset locations; keep them out of logs.
        formatter
            .debug_struct("GuardianPlan")
            .field("fds", &self.fds)
            .field("ceiling", &self.ceiling)
            .field("arguments", &self.argv.len().saturating_sub(1))
            .finish_non_exhaustive()
    }
}

/// An argument or environment word contained a NUL byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteriorNul;

impl std::fmt::Display for InteriorNul {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("process argument or environment contains an interior NUL byte")
    }
}

impl std::error::Error for InteriorNul {}

fn c_strings(words: &[Vec<u8>]) -> Result<(Vec<CString>, Vec<*const c_char>), InteriorNul> {
    let owned = words
        .iter()
        .map(|word| CString::new(word.clone()).map_err(|_| InteriorNul))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pointers: Vec<*const c_char> = owned.iter().map(|word| word.as_ptr()).collect();
    pointers.push(std::ptr::null());
    Ok((owned, pointers))
}

impl GuardianPlan {
    /// `argv[0]` is the executable path passed to `execve`.
    pub fn new(
        fds: GuardianFds,
        ceiling: i32,
        parent: i32,
        argv: &[Vec<u8>],
        environment: &[Vec<u8>],
    ) -> Result<Self, InteriorNul> {
        let (owned_argv, argv) = c_strings(argv)?;
        let (owned_environment, environment) = c_strings(environment)?;
        Ok(Self {
            fds,
            ceiling,
            parent,
            _argv: owned_argv,
            argv,
            _environment: owned_environment,
            environment,
        })
    }
}

/// `getpid()`.
pub fn current_pid() -> i32 {
    // SAFETY: getpid has no preconditions and cannot fail.
    unsafe { ffi::getpid() }
}

/// `getppid()`.
pub fn parent_pid() -> i32 {
    // SAFETY: getppid has no preconditions and cannot fail.
    unsafe { ffi::getppid() }
}

/// `getrlimit(RLIMIT_NOFILE)`: the soft limit.
pub fn open_files_limit() -> io::Result<u64> {
    let mut limit = ffi::Rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is writable storage of the expected layout.
    if unsafe { ffi::getrlimit(ffi::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(limit.rlim_cur)
}

/// `access(path, X_OK) == 0`.
pub fn is_executable(path: &std::ffi::CStr) -> bool {
    // SAFETY: `path` is NUL-terminated.
    unsafe { ffi::access(path.as_ptr(), ffi::X_OK) == 0 }
}

/// `fork()` the guardian. The child never returns.
pub fn fork_guardian(plan: &GuardianPlan) -> io::Result<i32> {
    // SAFETY: fork has no memory preconditions. The child only runs `guardian`, which uses async-signal-safe calls over the plan's preallocated, immutable storage and always ends in `_exit`/`execve`.
    let pid = unsafe { ffi::fork() };
    if pid < 0 {
        return Err(io::Error::last_os_error());
    }
    if pid == 0 {
        guardian(plan);
    }
    Ok(pid)
}

fn exit_now(status: c_int) -> ! {
    // SAFETY: `_exit` is async-signal-safe and ends the process.
    unsafe { ffi::_exit(status) }
}

fn set_signal(signal: c_int, handler: usize) {
    let action = ffi::Sigaction {
        sa_handler: handler,
        sa_mask: 0,
        sa_flags: 0,
    };
    // SAFETY: a standard disposition with an empty mask; no old action kept.
    unsafe { ffi::sigaction(signal, &action, std::ptr::null_mut()) };
}

/// `kill` whose result the C++ ignores (best-effort stop in the guardian).
fn signal_quietly(pid: i32, signal: c_int) {
    // SAFETY: kill has no memory preconditions.
    unsafe { ffi::kill(pid, signal) };
}

fn close_fd(fd: c_int) {
    // SAFETY: closing a descriptor number; failures are irrelevant here.
    unsafe { ffi::close(fd) };
}

fn child_notice(fd: c_int, kind: i32, value: i32) -> bool {
    let mut bytes = [0u8; NOTICE_BYTES];
    let (first, second) = bytes.split_at_mut(4);
    first.copy_from_slice(&kind.to_ne_bytes());
    second.copy_from_slice(&value.to_ne_bytes());
    let mut sent = 0;
    while sent < NOTICE_BYTES {
        // SAFETY: the pointer and length stay within `bytes`.
        let count = unsafe { ffi::write(fd, bytes.as_ptr().add(sent).cast(), NOTICE_BYTES - sent) };
        if count > 0 {
            sent += count as usize;
        } else if count < 0 && ffi::errno() == ffi::EINTR {
            continue;
        } else {
            return false;
        }
    }
    true
}

fn child_byte(fd: c_int, expected: u8) -> bool {
    let mut byte = 0u8;
    loop {
        // SAFETY: reads at most one byte into `byte`.
        let n = unsafe { ffi::read(fd, std::ptr::from_mut(&mut byte).cast(), 1) };
        if n < 0 && ffi::errno() == ffi::EINTR {
            continue;
        }
        return n == 1 && byte == expected;
    }
}

fn child_close_except(_ceiling: c_int, kept: &[c_int]) {
    let mut descriptors = [ffi::ProcFdInfo { fd: -1, kind: 0 }; 256];
    loop {
        // SAFETY: Apple's proc_pidinfo wrapper calls __proc_info without allocation or locks. This single-threaded child supplies fixed writable storage for its own descriptor table.
        let bytes = unsafe {
            ffi::proc_pidinfo(
                ffi::getpid(),
                ffi::PROC_PIDLISTFDS,
                0,
                descriptors.as_mut_ptr().cast(),
                std::mem::size_of_val(&descriptors) as c_int,
            )
        };
        if bytes <= 0
            || bytes as usize > std::mem::size_of_val(&descriptors)
            || !(bytes as usize).is_multiple_of(std::mem::size_of::<ffi::ProcFdInfo>())
        {
            exit_now(119);
        }
        let count = bytes as usize / std::mem::size_of::<ffi::ProcFdInfo>();
        let mut closed = 0;
        for descriptor in &descriptors[..count] {
            if descriptor.fd < 0 {
                exit_now(119);
            }
            if !kept.contains(&descriptor.fd) {
                close_fd(descriptor.fd);
                closed += 1;
            }
        }
        if count < descriptors.len() {
            return;
        }
        if closed == 0 {
            exit_now(119);
        }
    }
}

/// `exited_without_reaping`: `waitid(P_PID, pid, WEXITED|WNOHANG|WNOWAIT)`.
pub fn exited_without_reaping(pid: i32) -> bool {
    let mut info = ffi::Siginfo {
        si_signo: 0,
        si_errno: 0,
        si_code: 0,
        si_pid: 0,
        si_uid: 0,
        si_status: 0,
        si_addr: std::ptr::null_mut(),
        si_value: 0,
        si_band: 0,
        pad: [0; 7],
    };
    // SAFETY: `info` is writable siginfo storage; WNOWAIT leaves the child unreaped.
    let result = unsafe {
        ffi::waitid(
            ffi::P_PID,
            pid as ffi::IdT,
            &mut info,
            ffi::WEXITED | ffi::WNOHANG | ffi::WNOWAIT,
        )
    };
    result == 0 && info.si_pid == pid
}

/// `reap_child`: blocking `waitpid`, retried on `EINTR`; the wait status or -1.
pub fn reap_child(pid: i32) -> i32 {
    let mut status: c_int = 0;
    loop {
        // SAFETY: `status` is writable.
        let result = unsafe { ffi::waitpid(pid, &mut status, 0) };
        if result < 0 && ffi::errno() == ffi::EINTR {
            continue;
        }
        return if result == pid { status } else { -1 };
    }
}

/// One `waitpid(pid, &status, WNOHANG)`: (result, status, errno).
pub fn wait_nohang(pid: i32) -> (i32, i32, i32) {
    let mut status: c_int = 0;
    // SAFETY: `status` is writable.
    let result = unsafe { ffi::waitpid(pid, &mut status, ffi::WNOHANG) };
    let error = if result < 0 { ffi::errno() } else { 0 };
    (result, status, error)
}

/// `kill(pid, signal)`; negative `pid` names a process group.
pub fn kill(pid: i32, signal: i32) -> io::Result<()> {
    // SAFETY: kill has no memory preconditions.
    if unsafe { ffi::kill(pid, signal) } < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `kill(pid, 0) < 0 && errno == ESRCH`: no such process.
pub fn process_absent(pid: i32) -> bool {
    // SAFETY: signal 0 only checks for existence.
    unsafe { ffi::kill(pid, 0) < 0 && ffi::errno() == ffi::ESRCH }
}

/// `getsid(pid)`, or -1.
pub fn session_of(pid: i32) -> i32 {
    // SAFETY: getsid has no memory preconditions.
    unsafe { ffi::getsid(pid) }
}

/// `getpgid(pid)`, or -1.
pub fn group_of(pid: i32) -> i32 {
    // SAFETY: getpgid has no memory preconditions.
    unsafe { ffi::getpgid(pid) }
}

/// `setsid()`.
pub fn setsid() -> io::Result<i32> {
    // SAFETY: setsid has no memory preconditions.
    let sid = unsafe { ffi::setsid() };
    if sid < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(sid)
    }
}

/// `setpgid(pid, pgid)`.
pub fn setpgid(pid: i32, pgid: i32) -> io::Result<()> {
    // SAFETY: setpgid has no memory preconditions.
    if unsafe { ffi::setpgid(pid, pgid) } < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `killpg(pgid, signal)`.
pub fn killpg(pgid: i32, signal: i32) -> io::Result<()> {
    // SAFETY: killpg has no memory preconditions.
    if unsafe { ffi::killpg(pgid, signal) } < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `WIFEXITED(status)`: returns `Some(WEXITSTATUS(status))` when exited normally.
pub const fn wait_exit_code(status: i32) -> Option<i32> {
    if status & 0x7f == 0 {
        Some((status >> 8) & 0xff)
    } else {
        None
    }
}

/// `WIFSIGNALED(status)`: returns `Some(WTERMSIG(status))` when terminated by a signal.
pub const fn wait_signal(status: i32) -> Option<i32> {
    let low = status & 0x7f;
    if low != 0 && low != 0x7f {
        Some(low)
    } else {
        None
    }
}

/// `WIFEXITED(status) && WEXITSTATUS(status) == 0`.
pub fn exited_zero(status: i32) -> bool {
    status & 0x7f == 0 && (status >> 8) & 0xff == 0
}

/// The guardian (`guardian()` in owned_worker.cpp), in the forked child.
fn guardian(plan: &GuardianPlan) -> ! {
    let c = plan.fds;
    set_signal(ffi::SIGPIPE, ffi::SIG_IGN);
    set_signal(ffi::SIGCHLD, ffi::SIG_DFL);
    let keep = [
        c.control,
        c.status,
        c.input,
        c.output,
        c.error,
        c.gate_read,
        c.gate_write,
    ];
    child_close_except(plan.ceiling, &keep);
    // SAFETY: getppid and setsid have no memory preconditions.
    let identity_failed = unsafe { ffi::getppid() != plan.parent || ffi::setsid() < 0 };
    if identity_failed
        || !child_notice(c.status, ARMED_NOTICE, current_pid())
        || !child_byte(c.control, b'L')
    {
        exit_now(120);
    }
    // SAFETY: the guardian is single-threaded after fork; the grandchild runs only the async-signal-safe sequence below before execve.
    let worker = unsafe { ffi::fork() };
    if worker < 0 {
        child_notice(c.status, FAILURE_NOTICE, ffi::errno());
        exit_now(121);
    }
    if worker == 0 {
        // SAFETY: dup2 on descriptors this child inherited. The guardian assigns the process group before releasing the exec gate.
        let wired = unsafe {
            ffi::dup2(c.input, 0) >= 0 && ffi::dup2(c.output, 1) >= 0 && ffi::dup2(c.error, 2) >= 0
        };
        if !wired {
            exit_now(122);
        }
        child_close_except(plan.ceiling, &[0, 1, 2, c.gate_read]);
        if !child_byte(c.gate_read, b'E') {
            exit_now(123);
        }
        close_fd(c.gate_read);
        set_signal(ffi::SIGPIPE, ffi::SIG_DFL);
        // SAFETY: argv/environment are NUL-terminated arrays of pointers to NUL-terminated strings owned by the plan, alive until exec.
        unsafe { ffi::execve(plan.argv[0], plan.argv.as_ptr(), plan.environment.as_ptr()) };
        exit_now(127);
    }
    close_fd(c.input);
    close_fd(c.output);
    close_fd(c.error);
    close_fd(c.gate_read);
    // SAFETY: setpgid/getpgid on the guardian's own child.
    let mut stopping =
        unsafe { ffi::setpgid(worker, worker) != 0 || ffi::getpgid(worker) != worker };
    if !stopping {
        stopping = !child_notice(c.status, LAUNCHED_NOTICE, worker) || !child_byte(c.control, b'E');
    }
    if !stopping {
        let permit = b'E';
        // SAFETY: writes one byte from `permit`.
        stopping = unsafe { ffi::write(c.gate_write, std::ptr::from_ref(&permit).cast(), 1) } != 1;
    }
    close_fd(c.gate_write);
    // Parent death is EOF on its sole lifetime endpoint. Target exec cannot happen before ARMED and LAUNCHED were observed and acknowledged above. Keep the worker PID unreaped until its group is killed: no PID/group-ID reuse window can redirect descendant cleanup to an unrelated process.
    let mut stop_ticks: u32 = 0;
    loop {
        if exited_without_reaping(worker) {
            signal_quietly(-worker, ffi::SIGKILL);
            let status = reap_child(worker);
            child_notice(c.status, EXIT_NOTICE, status);
            exit_now(if status < 0 { 124 } else { 0 });
        }
        if stopping {
            if stop_ticks == 0 {
                signal_quietly(-worker, ffi::SIGTERM);
                signal_quietly(worker, ffi::SIGTERM);
            }
            stop_ticks += 1;
            if stop_ticks >= 10 {
                signal_quietly(-worker, ffi::SIGKILL);
                signal_quietly(worker, ffi::SIGKILL);
            }
        }
        let mut watch = [ffi::PollFd {
            fd: if stopping { -1 } else { c.control },
            events: ffi::POLLIN,
            revents: 0,
        }];
        // SAFETY: one pollfd in writable storage.
        let result = unsafe { ffi::poll(watch.as_mut_ptr(), 1, POLL_SLICE_MS) };
        if result < 0 && ffi::errno() != ffi::EINTR {
            stopping = true;
        }
        if result > 0 && watch[0].revents != 0 {
            let mut ignored = 0u8;
            // SAFETY: reads at most one byte into `ignored`.
            let n = unsafe { ffi::read(c.control, std::ptr::from_mut(&mut ignored).cast(), 1) };
            if n != -1 || ffi::errno() != ffi::EINTR {
                stopping = true;
            }
        }
    }
}
