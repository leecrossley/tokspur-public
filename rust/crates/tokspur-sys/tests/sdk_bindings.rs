//! Binds every hand-written Darwin constant and `#[repr(C)]` struct layout in `tokspur-sys` to the active macOS SDK headers via `xcrun clang`, and exercises exclusive rename, flock, fstatvfs, pread/pwrite, port availability and libproc telemetry.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::CString;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::os::fd::AsFd;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use tokspur_sys::{fd, host, net, process, sdk_layout};

/// XNU `socket(2)` has no `SOCK_CLOEXEC` flag, so `posix_spawn` in a parallel test thread can inherit a raw socket in the window before `F_SETFD`.
static SUBPROCESS_LOCK: Mutex<()> = Mutex::new(());

fn temp_dir(label: &str) -> PathBuf {
    let mut random = [0u8; 8];
    tokspur_sys::random_bytes(&mut random);
    let mut hex = String::with_capacity(16);
    for byte in random {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    let dir = std::env::temp_dir().join(format!("tokspur-sys-{label}-{hex}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn darwin_sdk_headers_match_declared_constants_and_struct_layouts() {
    let dir = temp_dir("sdk-bindings");
    let source = dir.join("probe.c");
    let binary = dir.join("probe");

    let c_program = r#"
#include <stddef.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/attr.h>
#include <uuid/uuid.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/stdio.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/resource.h>
#include <libproc.h>
#include <CommonCrypto/CommonDigest.h>

int main(void) {
    int (*volume_attributes)(int, void *, void *, size_t, unsigned int) = fgetattrlist;
    (void)volume_attributes;
    struct volume_uuid_reply { uint32_t length; uuid_t uuid; };
    printf("ATTR_BIT_MAP_COUNT=%u\n", (unsigned)ATTR_BIT_MAP_COUNT);
    printf("ATTR_VOL_UUID=%u\n", (unsigned)ATTR_VOL_UUID);
    printf("ATTR_VOL_INFO=%u\n", (unsigned)ATTR_VOL_INFO);
    printf("ATTRLIST_SIZE=%zu\n", sizeof(struct attrlist));
    printf("ATTRLIST_VOLATTR_OFF=%zu\n", offsetof(struct attrlist, volattr));
    printf("VOLUME_UUID_REPLY_SIZE=%zu\n", sizeof(struct volume_uuid_reply));
    printf("VOLUME_UUID_REPLY_UUID_OFF=%zu\n", offsetof(struct volume_uuid_reply, uuid));
    printf("PROC_PIDLISTFDS=%d\n", (int)PROC_PIDLISTFDS);
    printf("PROC_FDINFO_SIZE=%zu\n", sizeof(struct proc_fdinfo));
    printf("SO_LINGER=%d\n", (int)SO_LINGER);
    printf("LINGER_SIZE=%zu\n", sizeof(struct linger));
    printf("RENAME_EXCL=%u\n", (unsigned)RENAME_EXCL);
    printf("RENAME_SECLUDE=%u\n", (unsigned)RENAME_SECLUDE);
    printf("F_NOCACHE=%d\n", (int)F_NOCACHE);
    printf("ENOTSUP=%d\n", (int)ENOTSUP);
    printf("AT_FDCWD=%d\n", (int)AT_FDCWD);
    printf("AT_SYMLINK_NOFOLLOW=%d\n", (int)AT_SYMLINK_NOFOLLOW);
    printf("AT_REMOVEDIR=%d\n", (int)AT_REMOVEDIR);
    printf("LOCK_EX=%d\n", (int)LOCK_EX);
    printf("LOCK_NB=%d\n", (int)LOCK_NB);
    printf("LOCK_UN=%d\n", (int)LOCK_UN);
    printf("S_IFMT=%u\n", (unsigned)S_IFMT);
    printf("S_IFDIR=%u\n", (unsigned)S_IFDIR);
    printf("S_IFREG=%u\n", (unsigned)S_IFREG);
    printf("S_IFLNK=%u\n", (unsigned)S_IFLNK);
    printf("SO_REUSEADDR=%d\n", (int)SO_REUSEADDR);
    printf("SO_ERROR=%d\n", (int)SO_ERROR);
    printf("EEXIST=%d\n", (int)EEXIST);
    printf("EINPROGRESS=%d\n", (int)EINPROGRESS);
    printf("EADDRINUSE=%d\n", (int)EADDRINUSE);
    printf("ECONNREFUSED=%d\n", (int)ECONNREFUSED);
    printf("CTL_KERN=%d\n", (int)CTL_KERN);
    printf("KERN_PROCARGS2=%d\n", (int)KERN_PROCARGS2);
    printf("PROC_PIDTBSDINFO=%d\n", (int)PROC_PIDTBSDINFO);
    printf("MAXCOMLEN=%zu\n", (size_t)MAXCOMLEN);
    printf("SZOMB=%u\n", (unsigned)SZOMB);
    printf("PROC_PIDPATHINFO_MAXSIZE=%zu\n", (size_t)PROC_PIDPATHINFO_MAXSIZE);
    printf("RUSAGE_INFO_V4=%d\n", (int)RUSAGE_INFO_V4);

    printf("STAT_SIZE=%zu\n", sizeof(struct stat));
    printf("STAT_DEV_OFF=%zu\n", offsetof(struct stat, st_dev));
    printf("STAT_MODE_OFF=%zu\n", offsetof(struct stat, st_mode));
    printf("STAT_NLINK_OFF=%zu\n", offsetof(struct stat, st_nlink));
    printf("STAT_INO_OFF=%zu\n", offsetof(struct stat, st_ino));
    printf("STAT_UID_OFF=%zu\n", offsetof(struct stat, st_uid));
    printf("STAT_GID_OFF=%zu\n", offsetof(struct stat, st_gid));
    printf("STAT_MTIMESPEC_OFF=%zu\n", offsetof(struct stat, st_mtimespec));
    printf("STAT_CTIMESPEC_OFF=%zu\n", offsetof(struct stat, st_ctimespec));
    printf("STAT_SIZE_OFF=%zu\n", offsetof(struct stat, st_size));

    printf("STATVFS_SIZE=%zu\n", sizeof(struct statvfs));
    printf("STATVFS_BSIZE_OFF=%zu\n", offsetof(struct statvfs, f_bsize));
    printf("STATVFS_FRSIZE_OFF=%zu\n", offsetof(struct statvfs, f_frsize));
    printf("STATVFS_BLOCKS_OFF=%zu\n", offsetof(struct statvfs, f_blocks));
    printf("STATVFS_BFREE_OFF=%zu\n", offsetof(struct statvfs, f_bfree));
    printf("STATVFS_BAVAIL_OFF=%zu\n", offsetof(struct statvfs, f_bavail));

    printf("PROC_BSDINFO_SIZE=%zu\n", sizeof(struct proc_bsdinfo));
    printf("PROC_BSDINFO_STATUS_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_status));
    printf("PROC_BSDINFO_PID_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_pid));
    printf("PROC_BSDINFO_PPID_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_ppid));
    printf("PROC_BSDINFO_UID_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_uid));
    printf("PROC_BSDINFO_COMM_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_comm));
    printf("PROC_BSDINFO_NAME_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_name));
    printf("PROC_BSDINFO_PGID_OFF=%zu\n", offsetof(struct proc_bsdinfo, pbi_pgid));

    printf("RUSAGE_V4_SIZE=%zu\n", sizeof(struct rusage_info_v4));
    printf("RUSAGE_V4_PHYS_OFF=%zu\n", offsetof(struct rusage_info_v4, ri_phys_footprint));
    printf("RUSAGE_V4_PEAK_OFF=%zu\n", offsetof(struct rusage_info_v4, ri_lifetime_max_phys_footprint));

    printf("CC_SHA256_DIGEST_LENGTH=%zu\n", (size_t)CC_SHA256_DIGEST_LENGTH);
    printf("CC_SHA256_CTX_SIZE=%zu\n", sizeof(CC_SHA256_CTX));
    printf("CC_SHA256_CTX_HASH_OFF=%zu\n", offsetof(CC_SHA256_CTX, hash));
    printf("CC_SHA256_CTX_WBUF_OFF=%zu\n", offsetof(CC_SHA256_CTX, wbuf));
    return 0;
}
"#;
    fs::write(&source, c_program).unwrap();
    let output = {
        let _guard = SUBPROCESS_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = Command::new("/usr/bin/xcrun")
            .args(["clang", "-Wall", "-Wextra", "-Werror"])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .status()
            .unwrap();
        assert!(status.success(), "xcrun clang probe failed");
        Command::new(&binary).output().unwrap()
    };
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let mut sdk = std::collections::BTreeMap::new();
    for line in text.lines() {
        let (key, val) = line.split_once('=').unwrap();
        sdk.insert(key.to_owned(), val.parse::<i64>().unwrap());
    }

    // Explicitly prove RENAME_EXCL is 0x4 and distinct from RENAME_SECLUDE (0x1).
    assert_eq!(sdk["RENAME_EXCL"], 0x4);
    assert_ne!(sdk["RENAME_EXCL"], sdk["RENAME_SECLUDE"]);
    assert_eq!(
        i64::from(sdk_layout::PROC_PIDLISTFDS),
        sdk["PROC_PIDLISTFDS"]
    );
    assert_eq!(sdk_layout::PROC_FDINFO_SIZE as i64, sdk["PROC_FDINFO_SIZE"]);
    assert_eq!(i64::from(sdk_layout::SO_LINGER), sdk["SO_LINGER"]);
    assert_eq!(sdk_layout::LINGER_SIZE as i64, sdk["LINGER_SIZE"]);
    assert_eq!(i64::from(sdk_layout::RENAME_EXCL), sdk["RENAME_EXCL"]);
    assert_eq!(i64::from(sdk_layout::F_NOCACHE), sdk["F_NOCACHE"]);
    assert_eq!(i64::from(sdk_layout::ENOTSUP), sdk["ENOTSUP"]);
    assert_eq!(
        i64::from(sdk_layout::ATTR_BIT_MAP_COUNT),
        sdk["ATTR_BIT_MAP_COUNT"]
    );
    assert_eq!(i64::from(sdk_layout::ATTR_VOL_UUID), sdk["ATTR_VOL_UUID"]);
    assert_eq!(i64::from(sdk_layout::ATTR_VOL_INFO), sdk["ATTR_VOL_INFO"]);
    assert_eq!(i64::from(sdk_layout::AT_FDCWD), sdk["AT_FDCWD"]);
    assert_eq!(
        i64::from(sdk_layout::AT_SYMLINK_NOFOLLOW),
        sdk["AT_SYMLINK_NOFOLLOW"]
    );
    assert_eq!(i64::from(sdk_layout::AT_REMOVEDIR), sdk["AT_REMOVEDIR"]);
    assert_eq!(i64::from(sdk_layout::LOCK_EX), sdk["LOCK_EX"]);
    assert_eq!(i64::from(sdk_layout::LOCK_NB), sdk["LOCK_NB"]);
    assert_eq!(i64::from(sdk_layout::LOCK_UN), sdk["LOCK_UN"]);
    assert_eq!(i64::from(sdk_layout::S_IFMT), sdk["S_IFMT"]);
    assert_eq!(i64::from(sdk_layout::S_IFDIR), sdk["S_IFDIR"]);
    assert_eq!(i64::from(sdk_layout::S_IFREG), sdk["S_IFREG"]);
    assert_eq!(i64::from(sdk_layout::S_IFLNK), sdk["S_IFLNK"]);
    assert_eq!(i64::from(sdk_layout::SO_REUSEADDR), sdk["SO_REUSEADDR"]);
    assert_eq!(i64::from(sdk_layout::SO_ERROR), sdk["SO_ERROR"]);
    assert_eq!(i64::from(sdk_layout::EEXIST), sdk["EEXIST"]);
    assert_eq!(i64::from(sdk_layout::EINPROGRESS), sdk["EINPROGRESS"]);
    assert_eq!(i64::from(sdk_layout::EADDRINUSE), sdk["EADDRINUSE"]);
    assert_eq!(i64::from(sdk_layout::ECONNREFUSED), sdk["ECONNREFUSED"]);
    assert_eq!(i64::from(sdk_layout::CTL_KERN), sdk["CTL_KERN"]);
    assert_eq!(i64::from(sdk_layout::KERN_PROCARGS2), sdk["KERN_PROCARGS2"]);
    assert_eq!(
        i64::from(sdk_layout::PROC_PIDTBSDINFO),
        sdk["PROC_PIDTBSDINFO"]
    );
    assert_eq!(sdk_layout::MAXCOMLEN as i64, sdk["MAXCOMLEN"]);
    assert_eq!(i64::from(sdk_layout::SZOMB), sdk["SZOMB"]);
    assert_eq!(
        sdk_layout::PROC_PIDPATHINFO_MAXSIZE as i64,
        sdk["PROC_PIDPATHINFO_MAXSIZE"]
    );
    assert_eq!(i64::from(sdk_layout::RUSAGE_INFO_V4), sdk["RUSAGE_INFO_V4"]);

    for (actual, name) in [
        (sdk_layout::ATTRLIST_SIZE, "ATTRLIST_SIZE"),
        (sdk_layout::ATTRLIST_VOLATTR_OFF, "ATTRLIST_VOLATTR_OFF"),
        (sdk_layout::VOLUME_UUID_REPLY_SIZE, "VOLUME_UUID_REPLY_SIZE"),
        (
            sdk_layout::VOLUME_UUID_REPLY_UUID_OFF,
            "VOLUME_UUID_REPLY_UUID_OFF",
        ),
        (sdk_layout::STAT_SIZE, "STAT_SIZE"),
        (sdk_layout::STAT_DEV_OFF, "STAT_DEV_OFF"),
        (sdk_layout::STAT_MODE_OFF, "STAT_MODE_OFF"),
        (sdk_layout::STAT_NLINK_OFF, "STAT_NLINK_OFF"),
        (sdk_layout::STAT_INO_OFF, "STAT_INO_OFF"),
        (sdk_layout::STAT_UID_OFF, "STAT_UID_OFF"),
        (sdk_layout::STAT_GID_OFF, "STAT_GID_OFF"),
        (sdk_layout::STAT_MTIMESPEC_OFF, "STAT_MTIMESPEC_OFF"),
        (sdk_layout::STAT_CTIMESPEC_OFF, "STAT_CTIMESPEC_OFF"),
        (sdk_layout::STAT_SIZE_OFF, "STAT_SIZE_OFF"),
        (sdk_layout::STATVFS_SIZE, "STATVFS_SIZE"),
        (sdk_layout::STATVFS_BSIZE_OFF, "STATVFS_BSIZE_OFF"),
        (sdk_layout::STATVFS_FRSIZE_OFF, "STATVFS_FRSIZE_OFF"),
        (sdk_layout::STATVFS_BLOCKS_OFF, "STATVFS_BLOCKS_OFF"),
        (sdk_layout::STATVFS_BFREE_OFF, "STATVFS_BFREE_OFF"),
        (sdk_layout::STATVFS_BAVAIL_OFF, "STATVFS_BAVAIL_OFF"),
        (sdk_layout::PROC_BSDINFO_SIZE, "PROC_BSDINFO_SIZE"),
        (
            sdk_layout::PROC_BSDINFO_STATUS_OFF,
            "PROC_BSDINFO_STATUS_OFF",
        ),
        (sdk_layout::PROC_BSDINFO_PID_OFF, "PROC_BSDINFO_PID_OFF"),
        (sdk_layout::PROC_BSDINFO_PPID_OFF, "PROC_BSDINFO_PPID_OFF"),
        (sdk_layout::PROC_BSDINFO_UID_OFF, "PROC_BSDINFO_UID_OFF"),
        (sdk_layout::PROC_BSDINFO_COMM_OFF, "PROC_BSDINFO_COMM_OFF"),
        (sdk_layout::PROC_BSDINFO_NAME_OFF, "PROC_BSDINFO_NAME_OFF"),
        (sdk_layout::PROC_BSDINFO_PGID_OFF, "PROC_BSDINFO_PGID_OFF"),
        (sdk_layout::RUSAGE_V4_SIZE, "RUSAGE_V4_SIZE"),
        (sdk_layout::RUSAGE_V4_PHYS_OFF, "RUSAGE_V4_PHYS_OFF"),
        (sdk_layout::RUSAGE_V4_PEAK_OFF, "RUSAGE_V4_PEAK_OFF"),
        (
            sdk_layout::CC_SHA256_DIGEST_LENGTH,
            "CC_SHA256_DIGEST_LENGTH",
        ),
        (sdk_layout::CC_SHA256_CTX_SIZE, "CC_SHA256_CTX_SIZE"),
        (sdk_layout::CC_SHA256_CTX_HASH_OFF, "CC_SHA256_CTX_HASH_OFF"),
        (sdk_layout::CC_SHA256_CTX_WBUF_OFF, "CC_SHA256_CTX_WBUF_OFF"),
    ] {
        assert_eq!(actual as i64, sdk[name], "layout mismatch for {name}");
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn volume_identity_is_equal_for_files_on_the_same_volume() {
    let directory = temp_dir("volume-identity");
    let first = File::create(directory.join("first")).unwrap();
    let second = File::create(directory.join("second")).unwrap();
    let a = fd::file_volume_uuid(first.as_fd()).unwrap();
    let b = fd::file_volume_uuid(second.as_fd()).unwrap();
    assert!(a.is_some());
    assert!(a == b);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn rename_exclusive_at_refuses_existing_regular_file_and_symlink_with_eexist() {
    let dir = temp_dir("rename-excl");
    let dir_file = File::open(&dir).unwrap();

    fs::write(dir.join("candidate.tmp"), b"new-pack-bytes").unwrap();
    fs::write(dir.join("existing.gguf"), b"original-pack-bytes").unwrap();

    let from = CString::new("candidate.tmp").unwrap();
    let existing = CString::new("existing.gguf").unwrap();
    let err =
        fd::rename_exclusive_at(dir_file.as_fd(), &from, dir_file.as_fd(), &existing).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::AlreadyExists);
    assert_eq!(err.raw_os_error(), Some(tokspur_sys::EEXIST));
    assert_eq!(
        fs::read(dir.join("existing.gguf")).unwrap(),
        b"original-pack-bytes"
    );
    assert_eq!(
        fs::read(dir.join("candidate.tmp")).unwrap(),
        b"new-pack-bytes"
    );

    // Also refuse an existing destination symlink (even dangling).
    symlink(dir.join("nowhere"), dir.join("dangling.gguf")).unwrap();
    let dangling = CString::new("dangling.gguf").unwrap();
    let sym_err =
        fd::rename_exclusive_at(dir_file.as_fd(), &from, dir_file.as_fd(), &dangling).unwrap_err();
    assert_eq!(sym_err.kind(), ErrorKind::AlreadyExists);
    assert!(
        fd::lstat_at(dir_file.as_fd(), &dangling)
            .unwrap()
            .is_symlink()
    );

    // Succeeds when moving to a genuinely absent target.
    let fresh = CString::new("sealed.gguf").unwrap();
    fd::rename_exclusive_at(dir_file.as_fd(), &from, dir_file.as_fd(), &fresh).unwrap();
    assert_eq!(
        fs::read(dir.join("sealed.gguf")).unwrap(),
        b"new-pack-bytes"
    );
    assert!(!dir.join("candidate.tmp").exists());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn flock_pread_pwrite_statvfs_host_and_port_primitives_behave_as_specified() {
    let dir = temp_dir("primitives");
    let dir_file = File::open(&dir).unwrap();
    let space = fd::filesystem_space(dir_file.as_fd()).unwrap();
    assert!(space.total_bytes >= space.available_bytes && space.available_bytes > 0);

    let lock_name = CString::new("serve.lock").unwrap();
    let lock_a = fd::openat(
        dir_file.as_fd(),
        &lock_name,
        fd::O_RDWR | fd::O_CREAT | fd::O_NOFOLLOW | fd::O_CLOEXEC,
        0o600,
    )
    .unwrap();
    let lock_b = fd::openat(
        dir_file.as_fd(),
        &lock_name,
        fd::O_RDWR | fd::O_NOFOLLOW | fd::O_CLOEXEC,
        0o600,
    )
    .unwrap();
    fd::flock_exclusive_nonblocking(lock_a.as_fd()).unwrap();
    let contention = fd::flock_exclusive_nonblocking(lock_b.as_fd()).unwrap_err();
    assert_eq!(contention.kind(), ErrorKind::WouldBlock);
    fd::flock_unlock(lock_a.as_fd()).unwrap();
    fd::flock_exclusive_nonblocking(lock_b.as_fd()).unwrap();

    let data_name = CString::new("payload.bin").unwrap();
    let data = fd::openat(
        dir_file.as_fd(),
        &data_name,
        fd::O_RDWR | fd::O_CREAT | fd::O_EXCL | fd::O_NOFOLLOW | fd::O_CLOEXEC,
        0o600,
    )
    .unwrap();
    fd::set_nocache(data.as_fd(), true).unwrap();
    fd::pwrite_all(data.as_fd(), b"tokspur-pack", 16).unwrap();
    let mut read_back = [0u8; 12];
    fd::pread_exact(data.as_fd(), &mut read_back, 16).unwrap();
    assert_eq!(&read_back, b"tokspur-pack");
    fd::fchmod(data.as_fd(), 0o444).unwrap();
    assert_eq!(fd::fstat(data.as_fd()).unwrap().permissions(), 0o444);

    let reported_memory = {
        let _guard = SUBPROCESS_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .unwrap()
    };
    assert!(reported_memory.status.success());
    let expected_memory = String::from_utf8(reported_memory.stdout)
        .unwrap()
        .trim()
        .parse::<u64>()
        .unwrap();
    assert!(expected_memory > 0);
    assert_eq!(host::hw_memsize().unwrap(), expected_memory);
    let pid = process::current_pid();
    let snap = host::process_snapshot(pid).unwrap().unwrap();
    assert_eq!(snap.pid, pid);
    assert!(!snap.argv.is_empty());
    let fp = host::process_footprint(pid).unwrap();
    assert!(
        fp.physical_footprint_bytes > 0
            && fp.lifetime_peak_footprint_bytes >= fp.physical_footprint_bytes
    );

    let _guard = SUBPROCESS_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let listener = net::tcp_socket().unwrap();
    fd::set_descriptor_cloexec(
        listener.as_fd(),
        fd::descriptor_flags(listener.as_fd()).unwrap(),
    )
    .unwrap();
    net::bind_loopback(listener.as_fd(), 0).unwrap();
    net::listen(listener.as_fd(), 1).unwrap();
    let port = net::local_address(listener.as_fd()).unwrap().port;
    assert!(!net::port_available(port));
    drop(listener);
    assert!(net::port_available(port));

    let wildcard = net::tcp_socket().unwrap();
    fd::set_descriptor_cloexec(
        wildcard.as_fd(),
        fd::descriptor_flags(wildcard.as_fd()).unwrap(),
    )
    .unwrap();
    net::set_reuseaddr(wildcard.as_fd()).unwrap();
    net::bind_any(wildcard.as_fd(), 0).unwrap();
    net::listen(wildcard.as_fd(), 1).unwrap();
    let wildcard_port = net::local_address(wildcard.as_fd()).unwrap().port;
    assert!(!net::port_available(wildcard_port));
    let (accepted, _peer) = net::accept(wildcard.as_fd()).unwrap();
    fd::set_descriptor_cloexec(
        accepted.as_fd(),
        fd::descriptor_flags(accepted.as_fd()).unwrap(),
    )
    .unwrap();
    let mut eof = [0u8; 1];
    assert_eq!(fd::read(accepted.as_fd(), &mut eof).unwrap(), 0);
    drop(accepted);
    drop(wildcard);
    assert!(net::port_available(wildcard_port));

    fs::remove_dir_all(&dir).unwrap();
}
