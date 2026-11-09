//! Host memory, live process discovery and `RUSAGE_INFO_V4` footprint inspection over Darwin `sysctl` and `libproc`.

use std::ffi::{CStr, OsString, c_char, c_int, c_uint};
use std::fmt;
use std::io;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;

use crate::ffi;

/// Darwin `<sys/param.h>` `MAXCOMLEN`, matching `DARWIN_MAXCOMLEN` in `tokspur.local`.
pub const DARWIN_MAXCOMLEN: usize = ffi::MAXCOMLEN;

/// Snapshot of one live non-zombie process on the host.
#[derive(Clone, PartialEq, Eq)]
pub struct ProcessSnapshot {
    pub pid: i32,
    pub ppid: i32,
    pub pgid: i32,
    pub uid: u32,
    pub status: u32,
    pub start_tvsec: u64,
    pub start_tvusec: u64,
    pub comm: String,
    pub name: String,
    pub executable_path: Option<PathBuf>,
    pub argv: Vec<OsString>,
}

impl ProcessSnapshot {
    /// Returns `true` when `status` matches Darwin `SZOMB`.
    #[must_use]
    pub const fn is_zombie(&self) -> bool {
        self.status == ffi::SZOMB
    }
}

impl fmt::Debug for ProcessSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Paths and command-line arguments stay out of Debug output.
        formatter
            .debug_struct("ProcessSnapshot")
            .field("pid", &self.pid)
            .field("ppid", &self.ppid)
            .field("pgid", &self.pgid)
            .field("uid", &self.uid)
            .field("status", &self.status)
            .field("comm", &self.comm)
            .field("argc", &self.argv.len())
            .field("has_executable_path", &self.executable_path.is_some())
            .finish_non_exhaustive()
    }
}

/// Kernel physical-footprint telemetry from `proc_pid_rusage(pid, RUSAGE_INFO_V4)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessFootprint {
    pub physical_footprint_bytes: u64,
    pub lifetime_peak_footprint_bytes: u64,
}

/// Lightweight `PROC_PIDTBSDINFO` identity without `KERN_PROCARGS2` (`vm_map_lock`) or path copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessBsdIdentity {
    pub pid: i32,
    pub ppid: i32,
    pub status: u32,
    pub start_tvsec: u64,
    pub start_tvusec: u64,
}

impl ProcessBsdIdentity {
    /// Returns `true` when `status` matches Darwin `SZOMB`.
    #[must_use]
    pub const fn is_zombie(&self) -> bool {
        self.status == ffi::SZOMB
    }
}

/// Queries `sysctlbyname("hw.memsize")`.
pub fn hw_memsize() -> io::Result<u64> {
    let mut bytes: u64 = 0;
    let mut size = size_of::<u64>();
    let name: &CStr = c"hw.memsize";
    // SAFETY: `name` is NUL-terminated and `bytes`/`size` describe writable `u64` storage.
    let rc = unsafe {
        ffi::sysctlbyname(
            name.as_ptr(),
            std::ptr::from_mut(&mut bytes).cast(),
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    if size != size_of::<u64>() || bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected hw.memsize payload",
        ));
    }
    Ok(bytes)
}

fn c_buf_to_string(buf: &[c_char]) -> String {
    let bytes: &[u8] =
        // SAFETY: `c_char` and `u8` have identical size and alignment without padding.
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len()) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Queries the executable path of `pid` via `proc_pidpath` without reading `KERN_PROCARGS2`.
#[must_use]
pub fn process_executable_path(pid: i32) -> Option<PathBuf> {
    if pid <= 0 {
        return None;
    }
    query_proc_path(pid)
}

fn query_proc_path(pid: i32) -> Option<PathBuf> {
    let mut buf = [0u8; ffi::PROC_PIDPATHINFO_MAXSIZE];
    // SAFETY: `buf` is writable for `PROC_PIDPATHINFO_MAXSIZE` bytes.
    let n = unsafe { ffi::proc_pidpath(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
    if n <= 0 {
        return None;
    }
    let len = (n as usize).min(buf.len());
    let end = buf[..len].iter().position(|&b| b == 0).unwrap_or(len);
    if end == 0 {
        return None;
    }
    Some(PathBuf::from(OsString::from_vec(buf[..end].to_vec())))
}

pub(crate) fn parse_procargs2(raw: &[u8]) -> Option<Vec<OsString>> {
    if raw.len() < 4 {
        return None;
    }
    let argc = i32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if argc <= 0 || argc > 65_536 {
        return None;
    }
    let argc = argc as usize;
    let mut cursor = 4;
    // Skip executable path and NUL alignment padding before argv[0].
    let exec_end = raw[cursor..].iter().position(|&b| b == 0)?;
    cursor += exec_end;
    while cursor < raw.len() && raw[cursor] == 0 {
        cursor += 1;
    }
    let mut out = Vec::with_capacity(argc.min(64));
    for _ in 0..argc {
        if cursor >= raw.len() {
            break;
        }
        let end = raw[cursor..]
            .iter()
            .position(|&b| b == 0)
            .map_or(raw.len(), |rel| cursor + rel);
        out.push(OsString::from_vec(raw[cursor..end].to_vec()));
        cursor = end.saturating_add(1);
    }
    Some(out)
}

fn query_proc_argv(pid: i32) -> Vec<OsString> {
    let mut mib = [ffi::CTL_KERN, ffi::KERN_PROCARGS2, pid];
    let mut size: usize = 0;
    // SAFETY: `mib` has 3 ints and `oldp` is null to query required byte capacity.
    let probe = unsafe {
        ffi::sysctl(
            mib.as_mut_ptr(),
            mib.len() as c_uint,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    if probe != 0 || size < 4 {
        return Vec::new();
    }
    let cap = size.saturating_add(1024).min(1024 * 1024);
    let mut buffer = vec![0u8; cap];
    let mut actual = buffer.len();
    // SAFETY: `mib` has 3 ints and `buffer` is writable for `actual` bytes.
    let rc = unsafe {
        ffi::sysctl(
            mib.as_mut_ptr(),
            mib.len() as c_uint,
            buffer.as_mut_ptr().cast(),
            &mut actual,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 || actual > buffer.len() {
        return Vec::new();
    }
    parse_procargs2(&buffer[..actual]).unwrap_or_default()
}

/// Lists all live positive PIDs reported by `proc_listallpids`.
pub fn list_live_pids() -> io::Result<Vec<i32>> {
    // SAFETY: passing null buffer with size 0 queries the current PID count.
    let estimate = unsafe { ffi::proc_listallpids(std::ptr::null_mut(), 0) };
    if estimate < 0 {
        return Err(io::Error::last_os_error());
    }
    let capacity = (estimate as usize).saturating_add(256).max(256);
    let mut pids = vec![0i32; capacity];
    let byte_len = pids
        .len()
        .checked_mul(size_of::<i32>())
        .and_then(|n| c_int::try_from(n).ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pid buffer size overflow"))?;
    // SAFETY: `pids` is writable for `byte_len` bytes.
    let count = unsafe { ffi::proc_listallpids(pids.as_mut_ptr().cast(), byte_len) };
    if count < 0 {
        return Err(io::Error::last_os_error());
    }
    let used = (count as usize).min(pids.len());
    pids.truncate(used);
    pids.retain(|&pid| pid > 0);
    Ok(pids)
}

/// Queries one process via `proc_pidinfo(PROC_PIDTBSDINFO)`, returning `Ok(None)` if absent or zombie (`SZOMB`).
pub fn process_snapshot(pid: i32) -> io::Result<Option<ProcessSnapshot>> {
    if pid <= 0 {
        return Ok(None);
    }
    let mut bsd = ffi::ProcBsdinfo {
        pbi_flags: 0,
        pbi_status: 0,
        pbi_xstatus: 0,
        pbi_pid: 0,
        pbi_ppid: 0,
        pbi_uid: 0,
        pbi_gid: 0,
        pbi_ruid: 0,
        pbi_rgid: 0,
        pbi_svuid: 0,
        pbi_svgid: 0,
        rfu_1: 0,
        pbi_comm: [0; ffi::MAXCOMLEN],
        pbi_name: [0; 2 * ffi::MAXCOMLEN],
        pbi_nfiles: 0,
        pbi_pgid: 0,
        pbi_pjobc: 0,
        e_tdev: 0,
        e_tpgid: 0,
        pbi_nice: 0,
        pbi_start_tvsec: 0,
        pbi_start_tvusec: 0,
    };
    let expected = size_of::<ffi::ProcBsdinfo>() as c_int;
    // SAFETY: `bsd` is writable storage of exact size `sizeof(struct proc_bsdinfo)`.
    let n = unsafe {
        ffi::proc_pidinfo(
            pid,
            ffi::PROC_PIDTBSDINFO,
            0,
            std::ptr::from_mut(&mut bsd).cast(),
            expected,
        )
    };
    if n <= 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(ffi::ESRCH) || n == 0 {
            return Ok(None);
        }
        // Foreign-user processes that vanish or deny inspection are skipped by callers when appropriate; surface real OS errors otherwise.
        if err.kind() == io::ErrorKind::PermissionDenied {
            return Ok(None);
        }
        return Err(err);
    }
    if n != expected || bsd.pbi_status == ffi::SZOMB {
        return Ok(None);
    }
    let comm = c_buf_to_string(&bsd.pbi_comm);
    let name = {
        let full = c_buf_to_string(&bsd.pbi_name);
        if full.is_empty() { comm.clone() } else { full }
    };
    let executable_path = query_proc_path(pid);
    let argv = query_proc_argv(pid);
    Ok(Some(ProcessSnapshot {
        pid: bsd.pbi_pid as i32,
        ppid: bsd.pbi_ppid as i32,
        pgid: bsd.pbi_pgid as i32,
        uid: bsd.pbi_uid,
        status: bsd.pbi_status,
        start_tvsec: bsd.pbi_start_tvsec,
        start_tvusec: bsd.pbi_start_tvusec,
        comm,
        name,
        executable_path,
        argv,
    }))
}

/// Queries `PROC_PIDTBSDINFO` for one `pid` without acquiring `vm_map_lock` via `KERN_PROCARGS2`.
pub fn process_bsd_identity(pid: i32) -> io::Result<Option<ProcessBsdIdentity>> {
    if pid <= 0 {
        return Ok(None);
    }
    let mut bsd = ffi::ProcBsdinfo {
        pbi_flags: 0,
        pbi_status: 0,
        pbi_xstatus: 0,
        pbi_pid: 0,
        pbi_ppid: 0,
        pbi_uid: 0,
        pbi_gid: 0,
        pbi_ruid: 0,
        pbi_rgid: 0,
        pbi_svuid: 0,
        pbi_svgid: 0,
        rfu_1: 0,
        pbi_comm: [0; ffi::MAXCOMLEN],
        pbi_name: [0; 2 * ffi::MAXCOMLEN],
        pbi_nfiles: 0,
        pbi_pgid: 0,
        pbi_pjobc: 0,
        e_tdev: 0,
        e_tpgid: 0,
        pbi_nice: 0,
        pbi_start_tvsec: 0,
        pbi_start_tvusec: 0,
    };
    let expected = size_of::<ffi::ProcBsdinfo>() as c_int;
    // SAFETY: `bsd` is writable storage of exact size `sizeof(struct proc_bsdinfo)`.
    let n = unsafe {
        ffi::proc_pidinfo(
            pid,
            ffi::PROC_PIDTBSDINFO,
            0,
            std::ptr::from_mut(&mut bsd).cast(),
            expected,
        )
    };
    if n <= 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(ffi::ESRCH)
            || n == 0
            || err.kind() == io::ErrorKind::PermissionDenied
        {
            return Ok(None);
        }
        return Err(err);
    }
    if n != expected || bsd.pbi_status == ffi::SZOMB {
        return Ok(None);
    }
    Ok(Some(ProcessBsdIdentity {
        pid: bsd.pbi_pid as i32,
        ppid: bsd.pbi_ppid as i32,
        status: bsd.pbi_status,
        start_tvsec: bsd.pbi_start_tvsec,
        start_tvusec: bsd.pbi_start_tvusec,
    }))
}

/// Lists lightweight `PROC_PIDTBSDINFO` identities for all visible non-zombie processes without `KERN_PROCARGS2`.
pub fn list_live_bsd_identities() -> io::Result<Vec<ProcessBsdIdentity>> {
    let pids = list_live_pids()?;
    let mut out = Vec::with_capacity(pids.len());
    for pid in pids {
        if let Ok(Some(identity)) = process_bsd_identity(pid) {
            out.push(identity);
        }
    }
    Ok(out)
}

/// Snapshots all live non-zombie processes visible on the host.
pub fn list_live_processes() -> io::Result<Vec<ProcessSnapshot>> {
    let pids = list_live_pids()?;
    let mut snapshots = Vec::with_capacity(pids.len());
    for pid in pids {
        if let Ok(Some(snapshot)) = process_snapshot(pid) {
            snapshots.push(snapshot);
        }
    }
    Ok(snapshots)
}

/// Queries `proc_pid_rusage(pid, RUSAGE_INFO_V4)` for physical and lifetime-peak footprint bytes.
pub fn process_footprint(pid: i32) -> io::Result<ProcessFootprint> {
    let mut info = ffi::RusageInfoV4 {
        ri_uuid: [0; 16],
        ri_user_time: 0,
        ri_system_time: 0,
        ri_pkg_idle_wkups: 0,
        ri_interrupt_wkups: 0,
        ri_pageins: 0,
        ri_wired_size: 0,
        ri_resident_size: 0,
        ri_phys_footprint: 0,
        ri_proc_start_abstime: 0,
        ri_proc_exit_abstime: 0,
        ri_child_user_time: 0,
        ri_child_system_time: 0,
        ri_child_pkg_idle_wkups: 0,
        ri_child_interrupt_wkups: 0,
        ri_child_pageins: 0,
        ri_child_elapsed_abstime: 0,
        ri_diskio_bytesread: 0,
        ri_diskio_byteswritten: 0,
        ri_cpu_time_qos_default: 0,
        ri_cpu_time_qos_maintenance: 0,
        ri_cpu_time_qos_background: 0,
        ri_cpu_time_qos_utility: 0,
        ri_cpu_time_qos_legacy: 0,
        ri_cpu_time_qos_user_initiated: 0,
        ri_cpu_time_qos_user_interactive: 0,
        ri_billed_system_time: 0,
        ri_serviced_system_time: 0,
        ri_logical_writes: 0,
        ri_lifetime_max_phys_footprint: 0,
        ri_instructions: 0,
        ri_cycles: 0,
        ri_billed_energy: 0,
        ri_serviced_energy: 0,
        ri_interval_max_phys_footprint: 0,
        ri_runnable_time: 0,
    };
    // SAFETY: `info` is writable storage of exact `struct rusage_info_v4` layout.
    let rc = unsafe { ffi::proc_pid_rusage(pid, ffi::RUSAGE_INFO_V4, &mut info) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(ProcessFootprint {
        physical_footprint_bytes: info.ri_phys_footprint,
        lifetime_peak_footprint_bytes: info.ri_lifetime_max_phys_footprint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_snapshot_debug_redacts_paths_and_argv() {
        let snap = ProcessSnapshot {
            pid: 42,
            ppid: 1,
            pgid: 42,
            uid: 501,
            status: 2,
            start_tvsec: 100,
            start_tvusec: 0,
            comm: "tokspur".to_owned(),
            name: "tokspur".to_owned(),
            executable_path: Some(PathBuf::from("/Users/secret/private/tokspur")),
            argv: vec![
                OsString::from("/Users/secret/private/tokspur"),
                OsString::from("--secret-token=hunter2"),
            ],
        };
        let rendered = format!("{snap:?}");
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("hunter2"));
        assert!(rendered.contains("argc: 2"));
    }

    #[test]
    fn parse_procargs2_extracts_nul_separated_arguments() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&3i32.to_ne_bytes());
        raw.extend_from_slice(b"/usr/local/bin/tool\0\0\0\0");
        raw.extend_from_slice(b"tool\0--flag\0value\0ENV=1\0");
        let parsed = parse_procargs2(&raw).unwrap();
        assert_eq!(
            parsed,
            vec![
                OsString::from("tool"),
                OsString::from("--flag"),
                OsString::from("value")
            ]
        );
    }
}
