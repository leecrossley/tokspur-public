# SPDX-License-Identifier: Apache-2.0
"""Private local state and fail-closed adoption. No model downloader."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import socket
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Darwin <sys/param.h> MAXCOMLEN; native CPU checks bind this value to the SDK.
DARWIN_MAXCOMLEN = 16
SCALAR_WORKER_NAME = "scalar-channel-target"
IS_DARWIN = sys.platform == "darwin"


def private_dir(path: Path) -> Path:
    """Only create state below our real checkout; reject symlink traversal."""
    path = path.absolute()
    relative = path.relative_to(ROOT / ".tokspur")
    if ".." in relative.parts:
        raise ValueError("Private state must not traverse parent directories")
    current = ROOT
    for part in (".tokspur", *relative.parts):
        current /= part
        if current.is_symlink():
            raise ValueError(f"Private state must not be a symlink: {current}")
        current.mkdir(mode=0o700, exist_ok=True)
        os.chmod(current, 0o700)
    return path


def write_json(path: Path, data: object) -> None:
    private_dir(path.parent)
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite a symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@contextmanager
def state_lock():
    state = private_dir(ROOT / ".tokspur/state")
    fd = os.open(state / "serve.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another tokspur operation owns this state") from None
        yield
    finally:
        os.close(fd)


def profile() -> dict:
    return read_json(ROOT / "profiles/m5-max-128.json")


class _AttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


class _VolumeUuidReply(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("uuid", ctypes.c_ubyte * 16)]


def _volume_uuid(fd: int) -> str | None:
    if not IS_DARWIN:
        return None
    function = ctypes.CDLL(None, use_errno=True).fgetattrlist
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    request = _AttrList(5, 0, 0, 0x80000000 | 0x00040000, 0, 0, 0)
    reply = _VolumeUuidReply()
    if function(
        fd, ctypes.byref(request), ctypes.byref(reply), ctypes.sizeof(reply), 0
    ):
        error = ctypes.get_errno()
        if error == errno.ENOTSUP:
            return None
        raise OSError(error, os.strerror(error))
    if reply.length != ctypes.sizeof(reply):
        raise ValueError("invalid volume attribute length")
    value = bytes(reply.uuid)
    return value.hex() if any(value) else None


def identity(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        s = os.fstat(fd)
        volume_uuid = _volume_uuid(fd)
    finally:
        os.close(fd)
    result = {
        "device": s.st_dev,
        "inode": s.st_ino,
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "ctime_ns": s.st_ctime_ns,
    }
    if volume_uuid is not None:
        result["volume_uuid"] = volume_uuid
    return result


def verify_asset(path: Path, expected: dict) -> dict:
    path = path.resolve(strict=True)
    before = identity(path)
    if before["size"] != expected["size"]:
        raise ValueError(f"Unexpected size: {path.name}")
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Asset must be a regular file")
        if os.uname().sysname == "Darwin":
            # Avoid filling the unified file cache with a second target.
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before["device"], before["inode"]):
            raise ValueError("Asset changed while opening")
    finally:
        os.close(fd)
    if identity(path) != before:
        raise ValueError("Asset changed during verification")
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return {"path": str(path), "sha256": digest.hexdigest(), **before}


def check_adoption(data: dict) -> None:
    if data.get("schema") != 1 or data.get("verification") != "full-sha256":
        raise ValueError("Full asset verification is required")
    expected = read_json(ROOT / "profiles/flash-next-assets.json")
    for group in ("target", "draft"):
        assets = data[group]["assets"]
        candidate_specs = [expected[group]["files"]]
        if group == "draft" and "legacy_dflash2_draft" in expected:
            candidate_specs.append(expected["legacy_dflash2_draft"]["files"])
        matched_spec = None
        for spec_files in candidate_specs:
            if len(assets) == len(spec_files):
                matched_spec = spec_files
                break
        if matched_spec is None:
            raise ValueError("Incomplete adopted asset set")
        for asset, spec in zip(assets, matched_spec, strict=True):
            if asset["sha256"] != spec["sha256"]:
                raise ValueError("Asset manifest differs from the pinned profile")
            actual = identity(Path(asset["path"]))
            device = asset["device"]
            if type(device) is not int or not 0 <= device <= 0x7FFFFFFF:
                raise ValueError("invalid recorded asset device")
            if "volume_uuid" in asset:
                volume_uuid = asset["volume_uuid"]
                if (
                    not isinstance(volume_uuid, str)
                    or len(volume_uuid) != 32
                    or any(c not in "0123456789abcdefABCDEF" for c in volume_uuid)
                ):
                    raise ValueError("invalid asset volume UUID")
                same_volume = actual.get("volume_uuid") == volume_uuid.lower()
            else:
                same_volume = actual["device"] == device
            if not same_volume or any(
                actual[k] != asset[k] for k in ("inode", "size", "mtime_ns", "ctime_ns")
            ):
                raise ValueError(f"Asset changed; re-adopt: {spec['name']}")
            selected = Path(data[group]["directory"]) / spec["name"]
            if selected.resolve(strict=True) != Path(asset["path"]):
                raise ValueError("Selected asset link changed; re-adopt")


def port_available(port: int) -> bool:
    with socket.socket() as probe:
        # Match uvicorn's bind: a recently closed connection is not a live
        # listener. Without this, TIME_WAIT can cause false occupied-port errors.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    # On macOS that exact-address bind also succeeds while a wildcard (0.0.0.0
    # or dual-stack ::) listener owns the port, so prove nothing answers it.
    with socket.socket() as client:
        client.settimeout(0.5)
        try:
            client.connect(("127.0.0.1", port))
        except OSError:
            return True
    return False


def inference_processes() -> list[dict]:
    """Conservative safety check, not a cross-application resource lock."""
    import psutil

    found = []
    worker_names = {
        "omlx",
        "omlx-server",
        "splash",
        "llama-server",
        "tokspur-native",
        "branch8-target",
        "stage8-target",
        SCALAR_WORKER_NAME,
    }
    for proc in psutil.process_iter(["cmdline", "name"]):
        try:
            args = proc.info["cmdline"] or []
            names = {Path(a).name for a in args[:4]}
            # Another user's worker exposes no cmdline; its short name still counts.
            if proc.info["name"]:
                names.add(proc.info["name"])
            is_worker = bool(names & worker_names)
            # p_comm is the fallback only; a readable foreign argv wins over
            # a coincidentally matching short name. Never match a prefix.
            is_worker |= (
                IS_DARWIN
                and not args
                and proc.info["name"] == SCALAR_WORKER_NAME[:DARWIN_MAXCOMLEN]
            )
            is_worker |= "mlx_lm.server" in args[:4]
            is_worker |= "ollama" in names and "runner" in args[:4]
            if is_worker and proc.pid != os.getpid():
                found.append({"pid": proc.pid, "runtime": "inference worker"})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return found


def isolated_environment() -> dict[str, str]:
    blocked = ("HF_", "HUGGING_FACE_", "HUGGINGFACE_")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(blocked) and k not in ("PYTHONPATH", "PYTHONHOME")
    }
    state = private_dir(ROOT / ".tokspur/state")
    env.update(
        {
            "HF_HOME": str(private_dir(state / "hf-home")),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TMPDIR": str(private_dir(state / "tmp")),
            "NO_PROXY": "localhost,127.0.0.1,::1",
        }
    )
    return env
