//! Sequential `F_NOCACHE` streaming pack builder with per-chunk `fstatvfs` reserve checks, inverse byte proofs and `RENAME_EXCL` (`0x4`) publication.

use std::ffi::CString;
use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use tokspur_gguf::{ModelFiles, checked_add};
use tokspur_sys::fd::{
    O_CLOEXEC, O_CREAT, O_DIRECTORY, O_EXCL, O_NOFOLLOW, O_RDONLY, O_RDWR, fchmod,
    filesystem_space, fsync, lstat_at, openat, pread_exact, pwrite_all, rename_exclusive_at,
    set_nocache, unlink_at,
};
use tokspur_sys::random_bytes;
use tokspur_text::sha256::{Sha256, hex};

use crate::PackError;
use crate::header::build_header;
use crate::lanes::{pack_rows, row_layout, unpack_rows};
use crate::plan::{
    MAX_PACK_BYTES, PACK_ALIGNMENT, PackPlan, PackTensor, check_disk_reserve, plan_derived_pack,
};

struct Location {
    directory: File,
    name: CString,
}

impl Location {
    fn open(output: &Path) -> Result<Self, PackError> {
        let Some(file_name) = output.file_name() else {
            return Err(PackError::Format("invalid derived pack filename"));
        };
        let raw_name = file_name.as_bytes();
        if raw_name.is_empty() || raw_name == b"." || raw_name == b".." || raw_name.contains(&0) {
            return Err(PackError::Format("invalid derived pack filename"));
        }
        let name = CString::new(raw_name)
            .map_err(|_| PackError::Format("invalid derived pack filename"))?;

        let parent = match output.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let parent_c = CString::new(parent.as_os_str().as_bytes())
            .map_err(|_| PackError::Format("invalid derived pack parent directory"))?;
        let cwd = std::fs::File::open(".").map_err(|source| PackError::Io {
            context: "cannot open working directory",
            source,
        })?;
        // For absolute parent paths `openat` ignores the directory fd; for relative paths `cwd` anchors the lookup with `O_DIRECTORY | O_NOFOLLOW`.
        let directory = openat(
            cwd.as_fd(),
            &parent_c,
            O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW,
            0,
        )
        .map_err(|source| PackError::Io {
            context: "derived pack directory open refused",
            source,
        })?;
        Ok(Self { directory, name })
    }

    fn check_absent(&self) -> Result<(), PackError> {
        match lstat_at(self.directory.as_fd(), &self.name) {
            Ok(_) => Err(PackError::OutputExists),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(_) => Err(PackError::OutputExists),
        }
    }

    fn publication_preflight(&self) -> Result<(), PackError> {
        let mut probe = TemporaryFile::create(self.directory.as_fd())?;
        let target = CString::new(format!("{}.probe", probe.name.to_string_lossy()))
            .map_err(|_| PackError::Format("invalid publication probe filename"))?;
        rename_exclusive_at(
            self.directory.as_fd(),
            &probe.name,
            self.directory.as_fd(),
            &target,
        )
        .map_err(|source| PackError::Io {
            context: "target filesystem does not support exclusive pack publication",
            source,
        })?;
        probe.name = target;
        fsync(self.directory.as_fd()).map_err(|source| PackError::Io {
            context: "target directory fsync preflight failed",
            source,
        })
    }
}

struct TemporaryFile<'a> {
    directory: BorrowedFd<'a>,
    file: Option<File>,
    name: CString,
    published: bool,
}

impl<'a> TemporaryFile<'a> {
    fn create(directory: BorrowedFd<'a>) -> Result<Self, PackError> {
        for _ in 0..16 {
            let mut rnd = [0u8; 16];
            random_bytes(&mut rnd);
            let candidate = format!(".tokspur-pack-v2-{}.tmp", hex(&rnd));
            let name = CString::new(candidate)
                .map_err(|_| PackError::Format("invalid temporary pack filename"))?;
            match openat(
                directory,
                &name,
                O_CREAT | O_EXCL | O_NOFOLLOW | O_RDWR | O_CLOEXEC,
                0o600,
            ) {
                Ok(file) => {
                    return Ok(Self {
                        directory,
                        file: Some(file),
                        name,
                        published: false,
                    });
                }
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(PackError::Io {
                        context: "cannot create exclusive derived pack temporary",
                        source,
                    });
                }
            }
        }
        Err(PackError::Format(
            "cannot reserve derived pack temporary file",
        ))
    }

    fn fd(&self) -> Result<BorrowedFd<'_>, PackError> {
        self.file.as_ref().map(File::as_fd).ok_or(PackError::Format(
            "temporary pack descriptor already closed",
        ))
    }
}

impl Drop for TemporaryFile<'_> {
    fn drop(&mut self) {
        drop(self.file.take());
        if !self.published {
            let _ignored = unlink_at(self.directory, &self.name, false);
        }
    }
}

fn chunk_row_bound(tensor: &PackTensor, maximum: u64) -> Result<u64, PackError> {
    if tensor.layout == "plain" {
        return Ok(maximum);
    }
    let columns = u32::try_from(tensor.dimensions[0])
        .map_err(|_| PackError::Format("lane tensor column count exceeds u32"))?;
    let row = row_layout(tensor.r#type, columns)?.row_bytes;
    if row > maximum {
        return Err(PackError::Format(
            "streaming chunk cannot hold a complete row",
        ));
    }
    Ok(maximum / row * row)
}

fn require_reserve(
    available_bytes: u64,
    needed_bytes: u64,
    reserve_bytes: u64,
) -> Result<(), PackError> {
    if !check_disk_reserve(available_bytes, needed_bytes, reserve_bytes) {
        return Err(PackError::InsufficientDiskSpace {
            available_bytes,
            required_bytes: needed_bytes.checked_add(reserve_bytes),
            reserve_bytes,
        });
    }
    Ok(())
}

/// Builds a sealed `0444` `tokspur-pack-v2.asa.gguf` file using real `fstatvfs` reserve checks before preflight and before every chunk write.
pub fn build_derived_pack(
    source: &ModelFiles,
    mtp_sidecar: Option<&ModelFiles>,
    output: &Path,
    reserve_bytes: u64,
    chunk_bytes: u64,
) -> Result<PackPlan, PackError> {
    build_derived_pack_with_progress(
        source,
        mtp_sidecar,
        output,
        reserve_bytes,
        chunk_bytes,
        |_, _| true,
    )
}

/// Builds a pack with bounded progress callbacks. Returning false cancels and removes the incomplete temporary.
pub fn build_derived_pack_with_progress<P>(
    source: &ModelFiles,
    mtp_sidecar: Option<&ModelFiles>,
    output: &Path,
    reserve_bytes: u64,
    chunk_bytes: u64,
    progress: P,
) -> Result<PackPlan, PackError>
where
    P: FnMut(u64, u64) -> bool,
{
    build_derived_pack_with_callbacks(
        source,
        mtp_sidecar,
        output,
        reserve_bytes,
        chunk_bytes,
        |dir_fd| {
            filesystem_space(dir_fd)
                .map(|space| space.available_bytes)
                .map_err(|source| PackError::Io {
                    context: "statvfs on target directory failed",
                    source,
                })
        },
        progress,
    )
}

/// Builds a sealed `0444` `tokspur-pack-v2.asa.gguf` file with an explicit directory free-space query callback.
pub fn build_derived_pack_with_space_query<F>(
    source: &ModelFiles,
    mtp_sidecar: Option<&ModelFiles>,
    output: &Path,
    reserve_bytes: u64,
    chunk_bytes: u64,
    query_available_bytes: F,
) -> Result<PackPlan, PackError>
where
    F: FnMut(BorrowedFd<'_>) -> Result<u64, PackError>,
{
    build_derived_pack_with_callbacks(
        source,
        mtp_sidecar,
        output,
        reserve_bytes,
        chunk_bytes,
        query_available_bytes,
        |_, _| true,
    )
}

/// Builds a pack with explicit space and progress callbacks for controlled hosts and fault tests.
pub fn build_derived_pack_with_callbacks<F, P>(
    source: &ModelFiles,
    mtp_sidecar: Option<&ModelFiles>,
    output: &Path,
    reserve_bytes: u64,
    chunk_bytes: u64,
    mut query_available_bytes: F,
    mut progress: P,
) -> Result<PackPlan, PackError>
where
    F: FnMut(BorrowedFd<'_>) -> Result<u64, PackError>,
    P: FnMut(u64, u64) -> bool,
{
    if !(32_768..=67_108_864).contains(&chunk_bytes) {
        return Err(PackError::Format("invalid chunk byte bound"));
    }

    let mut plan = plan_derived_pack(source, mtp_sidecar, MAX_PACK_BYTES)?;
    let location = Location::open(output)?;
    location.check_absent()?;

    let preflight_avail = query_available_bytes(location.directory.as_fd())?;
    require_reserve(preflight_avail, plan.file_bytes, reserve_bytes)?;
    location.publication_preflight()?;
    if !progress(0, plan.file_bytes) {
        return Err(PackError::Cancelled);
    }

    let mut temporary = TemporaryFile::create(location.directory.as_fd())?;
    set_nocache(temporary.fd()?, true).map_err(|source| PackError::Io {
        context: "cannot disable buffer cache on temporary pack",
        source,
    })?;

    let initial_header = build_header(&plan)?;
    pwrite_all(temporary.fd()?, &initial_header, 0).map_err(|source| PackError::Io {
        context: "derived pack header pwrite failed",
        source,
    })?;
    let mut written_end = initial_header.len() as u64;

    let mut input_buf = Vec::new();
    let mut packed_buf = Vec::new();
    let mut restored_buf = Vec::new();
    let zero_pad = [0u8; PACK_ALIGNMENT as usize];

    for t in &mut plan.tensors {
        let in_source = source.tensors().contains_key(t.name.as_bytes());
        let in_mtp = mtp_sidecar.is_some_and(|m| m.tensors().contains_key(t.name.as_bytes()));
        if !in_source && !in_mtp {
            return Err(PackError::Format(
                "tensor missing from both source and MTP sidecar",
            ));
        }

        let tensor_start = checked_add(plan.data_offset, t.offset)?;
        if tensor_start < written_end {
            return Err(PackError::Format("nonmonotonic derived pack tensor offset"));
        }
        let pending_pad = tensor_start - written_end;
        if pending_pad >= PACK_ALIGNMENT {
            return Err(PackError::Format(
                "excessive alignment padding before derived pack tensor",
            ));
        }

        let chunk_size = chunk_row_bound(t, chunk_bytes)?;
        let chunk_cap = usize::try_from(chunk_size)
            .map_err(|_| PackError::Format("chunk size exceeds address space"))?;
        packed_buf.resize(chunk_cap, 0);
        restored_buf.resize(chunk_cap, 0);

        let mut src_hash = Sha256::new();
        let mut pack_hash = Sha256::new();

        let mut at = 0u64;
        while at < t.bytes {
            if !progress(written_end, plan.file_bytes) {
                return Err(PackError::Cancelled);
            }
            source.assert_unchanged()?;
            if let Some(mtp) = mtp_sidecar {
                mtp.assert_unchanged()?;
            }

            let n = chunk_size.min(t.bytes - at);
            let n_usize =
                usize::try_from(n).map_err(|_| PackError::Format("chunk exceeds usize"))?;
            let pad_bytes = if at == 0 { pending_pad } else { 0 };
            let upcoming = checked_add(pad_bytes, n)?;
            let remaining_pack_bytes = plan.file_bytes - written_end;

            let current_avail = query_available_bytes(location.directory.as_fd())?;
            require_reserve(current_avail, upcoming, reserve_bytes)?;
            require_reserve(current_avail, remaining_pack_bytes, reserve_bytes)?;

            if pad_bytes > 0 {
                pwrite_all(
                    temporary.fd()?,
                    &zero_pad[..pad_bytes as usize],
                    written_end,
                )
                .map_err(|source| PackError::Io {
                    context: "derived pack alignment pwrite failed",
                    source,
                })?;
                written_end = checked_add(written_end, pad_bytes)?;
            }

            if in_source {
                source.tensor_range_bytes(t.name.as_bytes(), at, n, &mut input_buf)?;
            } else if let Some(mtp) = mtp_sidecar {
                mtp.tensor_range_bytes(t.name.as_bytes(), at, n, &mut input_buf)?;
            }

            let tensor_file_pos = checked_add(plan.data_offset, checked_add(t.offset, at)?)?;
            if written_end != tensor_file_pos {
                return Err(PackError::Format(
                    "derived pack sequential write cursor mismatch",
                ));
            }

            if t.layout == "plain" {
                pwrite_all(temporary.fd()?, &input_buf, tensor_file_pos).map_err(|source| {
                    PackError::Io {
                        context: "derived pack pwrite failed",
                        source,
                    }
                })?;
                let restored = &mut restored_buf[..n_usize];
                pread_exact(temporary.fd()?, restored, tensor_file_pos).map_err(|source| {
                    PackError::Io {
                        context: "derived pack pread failed",
                        source,
                    }
                })?;
                if input_buf.as_slice() != restored {
                    return Err(PackError::Format("plain tensor readback byte proof failed"));
                }
                src_hash.update(&input_buf);
            } else {
                let columns = u32::try_from(t.dimensions[0])
                    .map_err(|_| PackError::Format("lane tensor column count exceeds u32"))?;
                let layout = row_layout(t.r#type, columns)?;
                let out = &mut packed_buf[..n_usize];
                let inverse = &mut restored_buf[..n_usize];
                pack_rows(&layout, &input_buf, out)?;
                pwrite_all(temporary.fd()?, out, tensor_file_pos).map_err(|source| {
                    PackError::Io {
                        context: "derived pack pwrite failed",
                        source,
                    }
                })?;
                pread_exact(temporary.fd()?, out, tensor_file_pos).map_err(|source| {
                    PackError::Io {
                        context: "derived pack pread failed",
                        source,
                    }
                })?;
                unpack_rows(&layout, out, inverse)?;
                if input_buf.as_slice() != inverse {
                    return Err(PackError::Format("lane packing inverse byte proof failed"));
                }
                src_hash.update(&input_buf);
                pack_hash.update(out);
            }

            written_end = checked_add(written_end, n)?;
            at += n;
        }

        t.source_sha256 = hex(&src_hash.finalize());
        // A plain tensor's packed bytes are its source bytes, so one digest serves both.
        t.packed_sha256 = if t.layout == "plain" {
            t.source_sha256.clone()
        } else {
            hex(&pack_hash.finalize())
        };
    }

    if written_end != plan.file_bytes {
        return Err(PackError::Format("derived pack final size mismatch"));
    }
    if !progress(written_end, plan.file_bytes) {
        return Err(PackError::Cancelled);
    }

    let final_header = build_header(&plan)?;
    if final_header.len() != initial_header.len() {
        return Err(PackError::Format("manifest changed header length"));
    }
    pwrite_all(temporary.fd()?, &final_header, 0).map_err(|source| PackError::Io {
        context: "derived pack final header pwrite failed",
        source,
    })?;
    let mut header_readback = vec![0u8; final_header.len()];
    pread_exact(temporary.fd()?, &mut header_readback, 0).map_err(|source| PackError::Io {
        context: "derived pack final header pread failed",
        source,
    })?;
    if header_readback != final_header {
        return Err(PackError::Format("pack header readback differs"));
    }

    fchmod(temporary.fd()?, 0o444).map_err(|source| PackError::Io {
        context: "derived pack seal chmod failed",
        source,
    })?;
    fsync(temporary.fd()?).map_err(|source| PackError::Io {
        context: "derived pack fsync failed",
        source,
    })?;

    source.assert_unchanged()?;
    if let Some(mtp) = mtp_sidecar {
        mtp.assert_unchanged()?;
    }
    // From here the candidate is complete and sealed. Keep it if publication cannot finish.
    temporary.published = true;
    let candidate = output
        .parent()
        .unwrap_or(Path::new("."))
        .join(OsStr::from_bytes(temporary.name.to_bytes()));
    rename_exclusive_at(
        location.directory.as_fd(),
        &temporary.name,
        location.directory.as_fd(),
        &location.name,
    )
    .map_err(|source| PackError::Publication { candidate, source })?;

    fsync(location.directory.as_fd()).map_err(|source| PackError::Io {
        context: "directory fsync failed after publication",
        source,
    })?;
    Ok(plan)
}
