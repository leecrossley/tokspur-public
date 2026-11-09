# SPDX-License-Identifier: Apache-2.0
"""scripts/install.sh and the tokspur command it writes, against a local release server, under dash and bash."""

from __future__ import annotations

import hashlib
import http.server
import io
import os
import shutil
import stat
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts/install.sh"
SHELLS = [name for name in ("dash", "bash") if shutil.which(name)]
TEAM = "ABCDE12345"

# The fake tokspur records its version, state root and arguments, and passes verify-build.
FAKE_TOKSPUR = """#!/bin/sh
if [ "${{1:-}}" = --version ]; then printf '{version_prefix}{built}\\n'; exit 0; fi
printf '%s|%s|%s\\n' "{version}" "$TOKSPUR_STATE_ROOT" "$*" >> "$TOKSPUR_TEST_LOG"
exit 0
"""

# codesign stand-in: a binary is validly signed unless it contains "unsigned", and its team is FAKE_TEAM.
FAKE_CODESIGN = """#!/bin/sh
printf '%s\\n' "$*" >> "$TOKSPUR_TEST_LOG.codesign"
previous=
notary_required=0
for argument do
    [ "$argument" != --check-notarization ] || notary_required=1
    if [ "$previous" = -R ]; then
        case "$argument" in =*) : ;; *) exit 66 ;; esac
        case "$argument" in *notarized*) notary_required=1 ;; esac
    fi
    previous="$argument"
done
for binary do :; done
case "$1" in
    --verify) [ "${FAKE_TRUSTED:-1}" = 1 ] && { [ "$notary_required" = 0 ] || [ "${FAKE_NOTARISED:-1}" = 1 ]; } && [ "$FAKE_TEAM" = "$TOKSPUR_TEAM_ID" ] && ! grep -q unsigned "$binary" ;;
    *) exit 64 ;;
esac
"""


def installer_text(team=None):
    text = INSTALLER.read_text()
    return text.replace("@TOKSPUR_TEAM_ID@", team) if team else text


def build_archive(
    tag, *, team=TEAM, unsigned=False, built=None, version_prefix="tokspur "
):
    buffer = io.BytesIO()
    worker = b"unsigned worker" if unsigned else b"worker"
    entries = (
        (
            "package/tokspur",
            FAKE_TOKSPUR.format(
                version=tag,
                built=(built or tag).removeprefix("v"),
                version_prefix=version_prefix,
            ).encode(),
            0o700,
        ),
        ("package/tokspur-native", worker, 0o700),
        ("package/provenance.json", b"{}\n", 0o600),
        ("package/tokspur-owned.metallib", b"MTLB fixture", 0o600),
        ("install.sh", installer_text(team).encode(), 0o755),
        ("LICENSE", b"licence\n", 0o644),
        ("VERSION", (tag + "\n").encode(), 0o644),
    )
    inventory = "".join(
        f"{hashlib.sha256(data).hexdigest()}  ./{name}\n"
        for name, data, _ in sorted(entries)
    ).encode()
    trust = (
        "#!/bin/sh\nprintf '%s\\n' '" + hashlib.sha256(inventory).hexdigest() + "'\n"
    ).encode()
    entries += (("RELEASE.sha256", inventory, 0o644), ("release-trust", trust, 0o755))
    with tarfile.open(fileobj=buffer, mode="w:gz") as stream:
        for name, data, mode in entries:
            info = tarfile.TarInfo(f"tokspur-{tag}/{name}")
            info.size, info.mode = len(data), mode
            stream.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class Releases(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    latest = ""
    latest_checks = 0

    def do_HEAD(self):
        self.respond(head=True)

    def do_GET(self):
        self.respond(head=False)

    def respond(self, head):
        if self.path == "/releases/latest":
            type(self).latest_checks += 1
            self.send_response(302)
            self.send_header("Location", f"/releases/tag/{self.latest}")
            self.end_headers()
            return
        body = self.files.get(self.path)
        if self.path.startswith("/releases/tag/"):
            body = b"release page"
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def log_message(self, *_):
        pass


def publish(tag, **kwargs):
    archive = build_archive(tag, **kwargs)
    name = f"tokspur-{tag}-macos-arm64.tar.gz"
    digest = hashlib.sha256(archive).hexdigest()
    Releases.files[f"/releases/download/{tag}/{name}"] = archive
    Releases.files[f"/releases/download/{tag}/{name}.sha256"] = (
        f"{digest}  {name}\n".encode()
    )
    Releases.latest = tag


@pytest.fixture
def server():
    Releases.files, Releases.latest, Releases.latest_checks = {}, "", 0
    publish("v0.1.0")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Releases)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/releases"
    httpd.shutdown()
    httpd.server_close()


class Host:
    """A fake Apple silicon Mac: a home with a space in it, uname and codesign shims."""

    def __init__(self, tmp_path, url, shell):
        self.shell = shell
        self.home = tmp_path / "home with space"
        self.base = self.home / "Library/Application Support/tokspur"
        self.command = self.home / ".local/bin/tokspur"
        self.log = tmp_path / "log"
        shims = tmp_path / "shims"
        shims.mkdir()
        (shims / "uname").write_text(
            '#!/bin/sh\ncase "$1" in -s) echo "${FAKE_SYSTEM:-Darwin}" ;; '
            '-m) echo "${FAKE_MACHINE:-arm64}" ;; esac\n'
        )
        (shims / "codesign").write_text(FAKE_CODESIGN)
        for shim in shims.iterdir():
            shim.chmod(0o755)
        self.env = {
            "PATH": f"{shims}:{os.environ['PATH']}",
            "HOME": str(self.home),
            "TOKSPUR_RELEASES_URL": url,
            "TOKSPUR_TEST_LOG": str(self.log),
            "FAKE_TEAM": TEAM,
            "TOKSPUR_TEAM_ID": TEAM,
        }

    def install(self, *args, **env):
        return subprocess.run(
            [self.shell, str(INSTALLER), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def tokspur(self, *args, **env):
        environment = {**self.env, **env}
        environment.pop("TOKSPUR_RELEASES_URL")  # the command remembers it
        return subprocess.run(
            [str(self.command), *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def current(self):
        path = self.base / "current"
        return path.read_text().strip() if path.exists() else None

    def releases(self):
        return sorted(p.name for p in (self.base / "releases").iterdir())

    def wait_for_current(self, tag, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            lock = self.base / "update.lock"
            if self.current() == tag and not lock.exists():
                return True
            time.sleep(0.1)
        return False


@pytest.fixture(params=SHELLS)
def host(request, tmp_path, server):
    return Host(tmp_path, server, request.param)


def test_installs_the_latest_release_and_a_working_command(host):
    result = host.install()
    assert result.returncode == 0, result.stderr
    package = host.base / "releases/v0.1.0/package"
    assert stat.S_IMODE(package.stat().st_mode) == 0o700
    assert stat.S_IMODE((package / "provenance.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((host.base / "data").stat().st_mode) == 0o700
    assert host.current() == "v0.1.0"
    assert "Installed tokspur v0.1.0" in result.stdout
    assert "checked daily" in result.stdout
    data = host.base / "data"
    assert host.log.read_text() == f"v0.1.0|{data}|verify-build\n"

    run = host.tokspur("doctor", "--flag", TOKSPUR_AUTO_UPDATE="0")
    assert run.returncode == 0, run.stderr
    assert host.log.read_text().splitlines()[-1] == f"v0.1.0|{data}|doctor --flag"

    again = host.install()
    assert again.returncode == 0, again.stderr
    assert "already installed" in again.stdout


def test_the_command_updates_itself_in_the_background_once_a_day(host):
    assert host.install().returncode == 0
    publish("v0.2.0")
    checks = Releases.latest_checks

    run = host.tokspur("doctor")
    assert run.returncode == 0, run.stderr
    # This run still uses the installed release. The update lands for the next run.
    assert f"v0.1.0|{host.base / 'data'}|doctor" in host.log.read_text().splitlines()
    assert host.wait_for_current("v0.2.0"), (host.base / "update.log").read_text()
    assert host.releases() == ["v0.1.0", "v0.2.0"]
    assert (
        "updated tokspur from v0.1.0 to v0.2.0"
        in (host.base / "update.log").read_text()
    )
    assert Releases.latest_checks == checks + 1

    host.tokspur("doctor")
    time.sleep(1)
    assert host.log.read_text().splitlines()[-1].startswith("v0.2.0|")
    # Checked within the last day, so this run does not ask again.
    assert Releases.latest_checks == checks + 1


def test_automatic_updates_can_be_turned_off(host):
    assert host.install().returncode == 0
    publish("v0.2.0")
    checks = Releases.latest_checks

    host.tokspur("doctor", TOKSPUR_AUTO_UPDATE="0")
    off = host.tokspur("update", "--off")
    assert off.returncode == 0 and "off" in off.stdout
    host.tokspur("doctor")
    time.sleep(1)
    assert host.current() == "v0.1.0"
    assert Releases.latest_checks == checks

    status = host.tokspur("update", "--check")
    assert status.stdout.splitlines() == [
        "installed: v0.1.0",
        "latest:    v0.2.0",
        "automatic updates: off",
    ]
    assert host.tokspur("update", "--on").returncode == 0
    assert not (host.base / "auto-update-off").exists()


def test_manual_update_keeps_the_previous_release_for_rollback(host):
    assert host.install().returncode == 0
    for tag in ("v0.2.0", "v0.3.0"):
        publish(tag)
        result = host.tokspur("update", TOKSPUR_AUTO_UPDATE="0")
        assert result.returncode == 0, result.stderr
        assert host.current() == tag
    assert host.releases() == ["v0.2.0", "v0.3.0"]
    usage = host.tokspur("update", "--bogus", TOKSPUR_AUTO_UPDATE="0")
    assert usage.returncode == 2


def test_a_pinned_installer_checks_the_team_id(host):
    signed = host.install(TOKSPUR_TEAM_ID=TEAM)
    assert signed.returncode == 0, signed.stderr
    assert "not checked" not in signed.stdout

    publish("v0.2.0")
    wrong_team = host.install("--update", TOKSPUR_TEAM_ID=TEAM, FAKE_TEAM="ZZZZZ99999")
    assert wrong_team.returncode == 1
    assert "trusted Developer ID signature for team ABCDE12345" in wrong_team.stderr
    assert host.current() == "v0.1.0"

    publish("v0.3.0", unsigned=True)
    unsigned = host.install("--update", TOKSPUR_TEAM_ID=TEAM)
    assert unsigned.returncode == 1
    assert "trusted Developer ID signature" in unsigned.stderr
    assert host.current() == "v0.1.0"


def test_an_unpinned_installer_refuses_to_install(host):
    result = host.install(TOKSPUR_TEAM_ID="")
    assert result.returncode == 1
    assert "no pinned Team ID" in result.stderr
    assert host.current() is None


def test_installs_the_bare_version_output_of_the_rust_cli(host):
    publish("v0.1.0", version_prefix="")
    result = host.install()
    assert result.returncode == 0, result.stderr
    assert host.current() == "v0.1.0"


@pytest.mark.parametrize("version_prefix", ["", "tokspur "])
def test_refuses_a_release_with_a_different_cli_version(host, version_prefix):
    publish("v0.1.0", built="0.2.0", version_prefix=version_prefix)
    result = host.install()
    assert result.returncode == 1
    assert "release v0.1.0 holds" in result.stderr
    assert host.current() is None
    assert not host.command.exists()


def test_refuses_a_checksum_mismatch(host):
    name = "tokspur-v0.1.0-macos-arm64.tar.gz"
    Releases.files[f"/releases/download/v0.1.0/{name}.sha256"] = b"0" * 64 + b"  x\n"
    result = host.install()
    assert result.returncode == 1
    assert "checksum mismatch" in result.stderr
    assert not (host.base / "releases/v0.1.0").exists()


def test_refuses_other_platforms(host):
    result = host.install(FAKE_MACHINE="x86_64")
    assert result.returncode == 1
    assert "Apple silicon" in result.stderr


def test_refuses_a_malformed_version(host):
    result = host.install(TOKSPUR_VERSION="latest; rm")
    assert result.returncode == 1
    assert "unexpected release version" in result.stderr


@pytest.mark.parametrize(
    "relative",
    [
        "install.sh",
        "package/provenance.json",
        "package/tokspur-owned.metallib",
        "RELEASE.sha256",
        "extra.txt",
        "../escape",
    ],
)
def test_authenticated_inventory_refuses_modified_or_extra_files(host, relative):
    original = build_archive("v0.1.0")
    output = io.BytesIO()
    replaced = False
    target = f"tokspur-v0.1.0/{relative}"
    with (
        tarfile.open(fileobj=io.BytesIO(original)) as source,
        tarfile.open(fileobj=output, mode="w:gz") as destination,
    ):
        for member in source.getmembers():
            data = source.extractfile(member).read()
            if member.name == target:
                data += b"\ntampered\n"
                member.size = len(data)
                replaced = True
            destination.addfile(member, io.BytesIO(data))
        if not replaced:
            member = tarfile.TarInfo(target)
            member.size = 3
            destination.addfile(member, io.BytesIO(b"bad"))
    archive = output.getvalue()
    name = "tokspur-v0.1.0-macos-arm64.tar.gz"
    Releases.files[f"/releases/download/v0.1.0/{name}"] = archive
    Releases.files[f"/releases/download/v0.1.0/{name}.sha256"] = (
        f"{hashlib.sha256(archive).hexdigest()}  {name}\n".encode()
    )
    result = host.install()
    assert result.returncode == 1, result.stdout
    assert host.current() is None


def test_untrusted_chain_is_refused_even_with_the_expected_team(host):
    result = host.install(FAKE_TRUSTED="0")
    assert result.returncode == 1
    assert "trusted Developer ID signature" in result.stderr


def test_developer_id_without_notarisation_is_refused(host):
    result = host.install(FAKE_NOTARISED="0")
    assert result.returncode == 1
    assert host.current() is None
    assert "notarised, trusted Developer ID signature" in result.stderr


def test_noop_preserves_rollback_and_explicit_downgrade_disables_auto_update(host):
    assert host.install().returncode == 0
    publish("v0.2.0")
    assert host.install("--update").returncode == 0
    assert host.current() == "v0.2.0"
    assert host.install("--update").returncode == 0
    assert (host.base / "previous").read_text().strip() == "v0.1.0"
    assert host.releases() == ["v0.1.0", "v0.2.0"]
    result = host.install(TOKSPUR_VERSION="v0.1.0")
    assert result.returncode == 0, result.stderr
    assert host.current() == "v0.1.0"
    assert (host.base / "auto-update-off").exists()
    before = Releases.latest_checks
    assert host.tokspur("doctor").returncode == 0
    time.sleep(0.2)
    assert Releases.latest_checks == before
