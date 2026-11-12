//! Working-directory-independent layout resolution (`CheckoutLayout` and `StagedLayout`) derived from `std::env::current_exe()`.

use std::ffi::CString;
use std::fmt;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use tokspur_json::parse_json;
use tokspur_sys::fd::lstat_path;

use crate::error::CliError;

pub const SCALAR_WORKER_NAME: &str = "scalar-channel-target";

/// Resolved runtime layout for the `tokspur` CLI.
#[derive(Clone, PartialEq, Eq)]
pub enum RuntimeLayout {
    /// Repository checkout containing `profiles/` and `native/Makefile`.
    Checkout(CheckoutLayout),
    /// Self-contained staged native package containing `provenance.json` and sibling binaries/profiles.
    Staged(StagedLayout),
}

impl fmt::Debug for RuntimeLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkout(_) => formatter.write_str("RuntimeLayout::Checkout([redacted])"),
            Self::Staged(_) => formatter.write_str("RuntimeLayout::Staged([redacted])"),
        }
    }
}

/// Paths inside a repository checkout.
#[derive(Clone, PartialEq, Eq)]
pub struct CheckoutLayout {
    pub root: PathBuf,
    pub state_root: PathBuf,
    pub state_root_overridden: bool,
}

impl fmt::Debug for CheckoutLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckoutLayout")
            .field("state_root_overridden", &self.state_root_overridden)
            .finish_non_exhaustive()
    }
}

/// Paths for a self-contained staged package directory plus optional external mutable state root.
#[derive(Clone, PartialEq, Eq)]
pub struct StagedLayout {
    pub package_dir: PathBuf,
    pub state_root: Option<PathBuf>,
}

impl fmt::Debug for StagedLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StagedLayout")
            .field("has_state_root", &self.state_root.is_some())
            .finish_non_exhaustive()
    }
}

/// Exact resolved executables, Metal library and policy file selected for `verify-build` and `serve`.
#[derive(Clone, PartialEq, Eq)]
pub struct LaunchTargets {
    pub server_bin: PathBuf,
    pub worker_bin: PathBuf,
    pub metallib: PathBuf,
    pub policy: PathBuf,
}

impl fmt::Debug for LaunchTargets {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LaunchTargets")
            .finish_non_exhaustive()
    }
}

pub fn validate_clean_absolute_path(path: &Path, label: &str) -> Result<PathBuf, CliError> {
    if !path.is_absolute() {
        return Err(CliError::Validation(format!(
            "{label} must be an absolute path"
        )));
    }
    if path.as_os_str().as_bytes().contains(&0) {
        return Err(CliError::Validation(format!("{label} contains a NUL byte")));
    }
    for comp in path.components() {
        if matches!(comp, Component::ParentDir | Component::CurDir) {
            return Err(CliError::Validation(format!(
                "{label} must not contain relative traversal components"
            )));
        }
    }
    Ok(path.to_path_buf())
}

/// Verifies that no existing ancestor or leaf component of `path` is a symlink.
pub fn ensure_no_symlink_components(path: &Path, label: &str) -> Result<(), CliError> {
    let clean = validate_clean_absolute_path(path, label)?;
    let mut current = PathBuf::new();
    for comp in clean.components() {
        current.push(comp.as_os_str());
        if current == Path::new("/") {
            continue;
        }
        let c_path = CString::new(current.as_os_str().as_bytes())
            .map_err(|_| CliError::Validation(format!("{label} contains a NUL byte")))?;
        match lstat_path(&c_path) {
            Ok(st) if st.is_symlink() => {
                return Err(CliError::Validation(format!(
                    "{label} must not traverse symlinks"
                )));
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
            Err(source) => {
                return Err(CliError::Io {
                    context: format!("cannot inspect {label}"),
                    source,
                });
            }
        }
    }
    Ok(())
}

fn non_empty_env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn is_checkout_root(candidate: &Path) -> bool {
    candidate.join("profiles/flash-next-assets.json").is_file()
        && candidate.join("native/Makefile").is_file()
}

impl RuntimeLayout {
    /// Resolves `StagedLayout` or `CheckoutLayout` from `current_exe()` and explicit overrides without consulting `current_dir()`.
    pub fn detect(explicit_state_root: Option<&Path>) -> Result<Self, CliError> {
        let current_exe = std::env::current_exe().map_err(|source| CliError::Io {
            context: "cannot resolve current executable path".to_owned(),
            source,
        })?;
        let canonical_exe = std::fs::canonicalize(&current_exe).map_err(|source| CliError::Io {
            context: "cannot canonicalize current executable path".to_owned(),
            source,
        })?;
        let exe_dir = canonical_exe
            .parent()
            .ok_or_else(|| {
                CliError::Validation("executable path has no parent directory".to_owned())
            })?
            .to_path_buf();

        let state_override = explicit_state_root
            .map(Path::to_path_buf)
            .or_else(|| non_empty_env_path("TOKSPUR_STATE_ROOT"));

        if let Some(pkg_env) = non_empty_env_path("TOKSPUR_PACKAGE_DIR") {
            let package_dir = validate_clean_absolute_path(&pkg_env, "TOKSPUR_PACKAGE_DIR")?;
            let state_root = validate_staged_state_root(&package_dir, state_override)?;
            return Ok(Self::Staged(StagedLayout {
                package_dir,
                state_root,
            }));
        }

        if std::fs::symlink_metadata(exe_dir.join("provenance.json")).is_ok() {
            let state_root = validate_staged_state_root(&exe_dir, state_override)?;
            return Ok(Self::Staged(StagedLayout {
                package_dir: exe_dir,
                state_root,
            }));
        }

        let root = if let Some(root_env) = non_empty_env_path("TOKSPUR_ROOT") {
            let clean = validate_clean_absolute_path(&root_env, "TOKSPUR_ROOT")?;
            if !is_checkout_root(&clean) {
                return Err(CliError::Validation(
                    "TOKSPUR_ROOT does not point to a valid tokspur checkout".to_owned(),
                ));
            }
            clean
        } else {
            exe_dir
                .ancestors()
                .find(|dir| is_checkout_root(dir))
                .map(Path::to_path_buf)
                .ok_or_else(|| {
                    CliError::Validation(
                        "Cannot locate tokspur checkout or staged provenance.json from executable path"
                            .to_owned(),
                    )
                })?
        };

        let (state_root, state_root_overridden) = if let Some(raw_state) = state_override {
            let clean = validate_clean_absolute_path(&raw_state, "state root")?;
            ensure_no_symlink_components(&clean, "state root")?;
            (clean, true)
        } else {
            (root.clone(), false)
        };

        Ok(Self::Checkout(CheckoutLayout {
            root,
            state_root,
            state_root_overridden,
        }))
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Checkout(_) => "checkout",
            Self::Staged(_) => "staged",
        }
    }

    /// Returns `<state_root>/.tokspur` for commands that read or mutate local state.
    pub fn tokspur_base(&self) -> Result<PathBuf, CliError> {
        match self {
            Self::Checkout(c) => Ok(c.state_root.join(".tokspur")),
            Self::Staged(s) => {
                let root = s.state_root.as_ref().ok_or_else(|| {
                    CliError::Validation(
                        "Staged tokspur package requires --state-root <abs-path> or TOKSPUR_STATE_ROOT outside the package directory".to_owned(),
                    )
                })?;
                Ok(root.join(".tokspur"))
            }
        }
    }

    pub fn assets_profile_path(&self) -> PathBuf {
        match self {
            Self::Checkout(c) => c.root.join("profiles/flash-next-assets.json"),
            Self::Staged(s) => s.package_dir.join("flash-next-assets.json"),
        }
    }

    pub fn hardware_profile_path(&self) -> PathBuf {
        match self {
            Self::Checkout(c) => c.root.join("profiles/m5-max-128.json"),
            Self::Staged(s) => s.package_dir.join("m5-max-128.json"),
        }
    }

    pub fn experimental_policy_path(&self) -> PathBuf {
        match self {
            Self::Checkout(c) => {
                let exp = c.root.join("profiles/owned-target-experimental.json");
                if exp.is_file() {
                    exp
                } else {
                    c.root.join("profiles/m5-max-128.json")
                }
            }
            Self::Staged(s) => s.package_dir.join("owned-target-experimental.json"),
        }
    }

    pub fn checkout_receipt_path(&self) -> Option<PathBuf> {
        match self {
            Self::Checkout(c) => {
                if c.state_root_overridden {
                    let alt = c.state_root.join(".tokspur/receipts/build-manifest.json");
                    if alt.exists() {
                        return Some(alt);
                    }
                }
                Some(c.root.join(".tokspur/receipts/build-manifest.json"))
            }
            Self::Staged(_) => None,
        }
    }

    /// Resolves the exact `(server_bin, worker_bin, metallib, policy)` launch targets without falling back to external engines.
    pub fn resolve_launch_targets(&self) -> Result<LaunchTargets, CliError> {
        match self {
            Self::Checkout(c) => {
                let server_candidates = [
                    c.root.join("build/rust-fast-math/release/tokspur-server"),
                    c.root.join("build/rust-target/release/tokspur-server"),
                    c.root.join("rust/target/release/tokspur-server"),
                ];
                let server_bin = server_candidates
                    .iter()
                    .find(|p| p.is_file())
                    .cloned()
                    .unwrap_or_else(|| server_candidates[0].clone());

                let build_dir = c.root.join("build/owned-native");
                let scalar_worker = build_dir.join(SCALAR_WORKER_NAME);
                let worker_bin = if std::fs::symlink_metadata(&scalar_worker).is_ok() {
                    scalar_worker
                } else {
                    build_dir.join("tokspur-native")
                };

                let metallib = build_dir.join("tokspur-owned.metallib");
                let policy = self.experimental_policy_path();
                Ok(LaunchTargets {
                    server_bin,
                    worker_bin,
                    metallib,
                    policy,
                })
            }
            Self::Staged(s) => {
                let prov_path = s.package_dir.join("provenance.json");
                crate::verify::inspect_nofollow(
                    &prov_path,
                    4 * 1024 * 1024,
                    "staged provenance.json",
                )?;
                let prov_bytes = std::fs::read(&prov_path).map_err(|source| CliError::Io {
                    context: "cannot read staged provenance.json".to_owned(),
                    source,
                })?;
                let prov = parse_json(&prov_bytes).map_err(|source| CliError::Json {
                    context: "invalid staged provenance.json",
                    source,
                })?;
                let has_scalar = prov
                    .find("entries")
                    .ok()
                    .flatten()
                    .and_then(|e| e.find(SCALAR_WORKER_NAME).ok().flatten())
                    .is_some();
                let worker_name = if has_scalar {
                    SCALAR_WORKER_NAME
                } else {
                    "tokspur-native"
                };
                Ok(LaunchTargets {
                    server_bin: s.package_dir.join("tokspur-server"),
                    worker_bin: s.package_dir.join(worker_name),
                    metallib: s.package_dir.join("tokspur-owned.metallib"),
                    policy: s.package_dir.join("owned-target-experimental.json"),
                })
            }
        }
    }
}

/// Check directory identity as well as spelling on case-insensitive or normalizing volumes.
/// A prospective path can include components that do not exist yet.
pub(crate) fn at_or_below_directory(path: &Path, directory: &Path) -> Result<bool, CliError> {
    if path.starts_with(directory) {
        return Ok(true);
    }
    let identity = |entry: &Path| match std::fs::symlink_metadata(entry) {
        Ok(meta) => Ok(Some((meta.dev(), meta.ino()))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(CliError::Io {
            context: "cannot inspect staged directory identity".to_owned(),
            source,
        }),
    };
    let Some(boundary) = identity(directory)? else {
        return Ok(false);
    };
    for ancestor in path.ancestors() {
        if identity(ancestor)? == Some(boundary) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn validate_staged_state_root(
    package_dir: &Path,
    raw: Option<PathBuf>,
) -> Result<Option<PathBuf>, CliError> {
    let Some(state_root) = raw else {
        return Ok(None);
    };
    let clean = validate_clean_absolute_path(&state_root, "staged state root")?;
    ensure_no_symlink_components(&clean, "staged state root")?;
    if at_or_below_directory(&clean, package_dir)? || at_or_below_directory(package_dir, &clean)? {
        return Err(CliError::Validation(
            "Staged state root must be separate from the immutable package directory".to_owned(),
        ));
    }
    Ok(Some(clean))
}
