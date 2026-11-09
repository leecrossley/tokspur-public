#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Install or update a signed tokspur release on an Apple silicon Mac.
#
# Releases go to ~/Library/Application Support/tokspur/releases/<version>, state to ~/Library/Application Support/tokspur/data and a tokspur command to ~/.local/bin. Nothing here downloads a model, starts inference or installs a background service.
#
# The tokspur command checks for a new release at most once a day and installs it in the background. An update is used only after its SHA-256, its Developer ID signature and tokspur verify-build all pass, and the previous release stays on disk. `tokspur update` updates now, `tokspur update --check` reports the latest release, and `tokspur update --off` or TOKSPUR_AUTO_UPDATE=0 turns automatic updates off.
#
# Usage: install.sh [--update | --auto | --check | --off | --on]
# TOKSPUR_VERSION picks a release tag (default: the latest release), TOKSPUR_HOME the install directory and TOKSPUR_BIN_DIR the directory for the tokspur command.
set -eu
umask 077

repository="${TOKSPUR_REPOSITORY:-san-digital/tokspur}"
releases_url="${TOKSPUR_RELEASES_URL:-https://github.com/$repository/releases}"
home_dir="${TOKSPUR_HOME:-$HOME/Library/Application Support/tokspur}"
bin_dir="${TOKSPUR_BIN_DIR:-$HOME/.local/bin}"
# The release workflow pins the Apple Team ID that signs the executables. A copy from the source tree is unpinned.
team_id="${TOKSPUR_TEAM_ID:-@TOKSPUR_TEAM_ID@}"
mode="${1:-install}"

releases="$home_dir/releases"
state="$home_dir/data"
current_file="$home_dir/current"
previous_file="$home_dir/previous"
off_marker="$home_dir/auto-update-off"
stamp="$home_dir/last-update-check"
lock="$home_dir/update.lock"
work=""

fail() {
    printf 'tokspur install: %s\n' "$1" >&2
    exit 1
}

say() {
    [ "$mode" = --auto ] || printf '%s\n' "$1"
}

cleanup() {
    [ -z "$work" ] || rm -rf "$work"
    [ ! -d "$lock/held-by-$$" ] || rm -rf "$lock"
}

case "$mode" in
    install | --update | --auto | --check | --off | --on) ;;
    *) fail "usage: install.sh [--update | --auto | --check | --off | --on]" ;;
esac
case "$home_dir$bin_dir$releases_url" in
    *'"'* | *'$'* | *'`'* | *'\'*) fail "install paths must not contain quotes, dollar signs or backslashes" ;;
esac

if [ "$mode" = --off ]; then
    mkdir -p "$home_dir"
    : >"$off_marker"
    say "Automatic updates are off. Run tokspur update to update by hand."
    exit 0
fi
if [ "$mode" = --on ]; then
    rm -f "$off_marker"
    say "Automatic updates are on."
    exit 0
fi
if [ "$mode" = --auto ]; then
    [ "${TOKSPUR_AUTO_UPDATE:-1}" != 0 ] && [ ! -e "$off_marker" ] || exit 0
    if [ -e "$stamp" ] && [ -n "$(find "$stamp" -mtime -1 2>/dev/null)" ]; then
        exit 0
    fi
fi

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    fail "tokspur runs only on an Apple silicon Mac"
fi
for tool in curl shasum tar; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required"
done

latest_version() {
    url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$releases_url/latest")" ||
        fail "cannot find the latest release at $releases_url"
    printf '%s\n' "${url##*/}"
}

valid_version() {
    case "$1" in *[!v0-9.]* | '') return 1 ;; esac
    printf '%s\n' "$1" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$'
}

older() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "${1#v}" "${2#v}" | sort -t . -k 1,1n -k 2,2n -k 3,3n | head -n 1)" = "${1#v}" ]
}

installed=""
[ ! -f "$current_file" ] || installed="$(cat "$current_file")"

if [ "$mode" = --check ]; then
    latest="$(latest_version)"
    printf 'installed: %s\nlatest:    %s\n' "${installed:-none}" "$latest"
    if [ -e "$off_marker" ]; then
        printf 'automatic updates: off\n'
    else
        printf 'automatic updates: on\n'
    fi
    exit 0
fi

mkdir -p "$home_dir"
chmod 700 "$home_dir"
# One install or update at a time. A lock older than an hour belongs to an update that died.
if [ -n "$(find "$lock" -maxdepth 0 -mmin +60 2>/dev/null)" ]; then
    rm -rf "$lock"
fi
if ! mkdir "$lock" 2>/dev/null; then
    [ "$mode" = --auto ] && exit 0
    fail "another install or update is running; try again shortly"
fi
mkdir "$lock/held-by-$$"
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
[ "$mode" != --auto ] || : >"$stamp"

version="${TOKSPUR_VERSION:-}"
[ -n "$version" ] || version="$(latest_version)"
valid_version "$version" || fail "unexpected release version: $version"
[ -z "$installed" ] || valid_version "$installed" || fail "invalid installed release version"
if [ -z "${TOKSPUR_VERSION:-}" ] && [ -n "$installed" ] && older "$version" "$installed"; then
    fail "the latest release, $version, is older than the installed $installed"
fi
if [ "$mode" = --auto ] && [ "$version" = "$installed" ]; then
    exit 0
fi

check_signatures() {
    case "$team_id" in @* | '') fail "this installer has no pinned Team ID; use a published installer" ;; esac
    [ "${#team_id}" = 10 ] || fail "invalid signing Team ID"
    case "$team_id" in *[!A-Z0-9]*) fail "invalid signing Team ID" ;; esac
    command -v codesign >/dev/null 2>&1 || fail "codesign is required to check the release signature"
    requirement="anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = \"$team_id\" and notarized"
    for name in release-trust package/tokspur package/tokspur-server package/tokspur-native package/scalar-channel-target; do
        binary="$1/$name"
        [ -f "$binary" ] || { [ "$name" != release-trust ] && continue; fail "release authentication executable is missing"; }
        binary_requirement="$requirement"
        [ "$name" != release-trust ] || binary_requirement="$requirement and identifier \"com.tokspur.release-trust\""
        codesign --verify --strict --check-notarization -R "=$binary_requirement" "$binary" >/dev/null 2>&1 || fail "$name in $version does not have a notarised, trusted Developer ID signature for team $team_id"
    done
}

check_release() {
    root="$1"
    [ -f "$root/RELEASE.sha256" ] && [ ! -L "$root/release-trust" ] || fail "release authentication files are missing"
    check_signatures "$root"
    expected_manifest="$("$root/release-trust")" || fail "cannot read the signed release identity"
    case "$expected_manifest" in *[!0-9a-f]* | '') fail "invalid signed release identity" ;; esac
    [ "${#expected_manifest}" = 64 ] || fail "invalid signed release identity"
    actual_manifest="$(shasum -a 256 "$root/RELEASE.sha256" | cut -d ' ' -f 1)"
    [ "$expected_manifest" = "$actual_manifest" ] || fail "release manifest authentication failed"
    [ -z "$(find "$root" -type l -print)" ] || fail "release refuses symlinks"
    (cd "$root" && shasum -a 256 -c RELEASE.sha256 >/dev/null 2>&1) || fail "authenticated release file checksum failed"
    (cd "$root" && find . -type f | grep -v -x -e './release-trust' -e './RELEASE.sha256' | LC_ALL=C sort) >"$home_dir/release-files-$$.actual"
    cut -c 67- "$root/RELEASE.sha256" | LC_ALL=C sort >"$home_dir/release-files-$$.expected"
    if ! cmp -s "$home_dir/release-files-$$.actual" "$home_dir/release-files-$$.expected"; then
        rm -f "$home_dir/release-files-$$.actual" "$home_dir/release-files-$$.expected"
        fail "release contains files outside its signed inventory"
    fi
    rm -f "$home_dir/release-files-$$.actual" "$home_dir/release-files-$$.expected"
    [ "$(cat "$root/VERSION")" = "$version" ] || fail "release version differs from the requested tag"
}

archive="tokspur-$version-macos-arm64.tar.gz"
destination="$releases/$version"
package="$destination/package"

if [ -d "$destination" ]; then
    say "tokspur $version is already installed in $destination"
else
    work="$(mktemp -d)"
    say "Downloading tokspur $version"
    curl -fsSL -o "$work/$archive" "$releases_url/download/$version/$archive" ||
        fail "cannot download $archive"
    curl -fsSL -o "$work/$archive.sha256" "$releases_url/download/$version/$archive.sha256" ||
        fail "cannot download $archive.sha256"
    expected="$(cut -d ' ' -f 1 <"$work/$archive.sha256")"
    actual="$(shasum -a 256 "$work/$archive" | cut -d ' ' -f 1)"
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $archive"

    tar -tzf "$work/$archive" >"$work/members" || fail "invalid release archive"
    while IFS= read -r member; do
        case "$member" in "tokspur-$version" | "tokspur-$version/"*) ;; *) fail "archive path is outside the release" ;; esac
        case "/$member/" in */../* | */./*) fail "archive path traverses a directory" ;; esac
    done <"$work/members"
    tar -tvzf "$work/$archive" >"$work/types" || fail "invalid release archive"
    if grep -q '^[^d-]' "$work/types"; then fail "release archive permits only regular files and directories"; fi
    tar -xzpf "$work/$archive" -C "$work"
    [ -f "$work/tokspur-$version/package/provenance.json" ] ||
        fail "$archive does not contain a tokspur package"
    chmod 700 "$work/tokspur-$version"
    check_release "$work/tokspur-$version"
    mkdir -p "$releases"
    chmod 700 "$releases"
    mv "$work/tokspur-$version" "$destination"
fi
chmod 700 "$destination"
check_release "$destination"
chmod 700 "$package"
mkdir -p "$state" "$bin_dir"
chmod 700 "$state"

TOKSPUR_STATE_ROOT="$state" "$package/tokspur" verify-build >/dev/null ||
    fail "tokspur $version failed verification; remove $destination and install again"

built_version="$("$package/tokspur" --version)" || fail "cannot read the installed build version"
case "$built_version" in
    "${version#v}" | "tokspur ${version#v}") ;;
    *) fail "release $version holds $built_version, not $version" ;;
esac

command_file="$bin_dir/tokspur"
{
    printf '#!/bin/sh\n'
    printf '# tokspur %s, written by install.sh and rewritten by every update.\n' "$version"
    printf 'TOKSPUR_STATE_ROOT="${TOKSPUR_STATE_ROOT:-%s}"\n' "$state"
    printf 'TOKSPUR_HOME="${TOKSPUR_HOME:-%s}"\n' "$home_dir"
    printf 'TOKSPUR_BIN_DIR="${TOKSPUR_BIN_DIR:-%s}"\n' "$bin_dir"
    printf 'TOKSPUR_RELEASES_URL="${TOKSPUR_RELEASES_URL:-%s}"\n' "$releases_url"
    printf 'export TOKSPUR_STATE_ROOT TOKSPUR_HOME TOKSPUR_BIN_DIR TOKSPUR_RELEASES_URL\n'
    printf 'release="%s"\n' "$destination"
    printf 'if [ "${1:-}" = update ]; then\n'
    printf '    case "${2:-}" in\n'
    printf '        "") exec /bin/sh "$release/install.sh" --update ;;\n'
    printf '        --check | --off | --on) exec /bin/sh "$release/install.sh" "$2" ;;\n'
    printf '        *) printf "usage: tokspur update [--check | --off | --on]\\n" >&2; exit 2 ;;\n'
    printf '    esac\n'
    printf 'fi\n'
    printf 'if [ "${TOKSPUR_AUTO_UPDATE:-1}" != 0 ] && [ -f "$release/install.sh" ]; then\n'
    printf '    /bin/sh "$release/install.sh" --auto >>"%s/update.log" 2>&1 </dev/null &\n' "$home_dir"
    printf 'fi\n'
    printf 'exec "$release/package/tokspur" "$@"\n'
} >"$command_file.new"
chmod 755 "$command_file.new"
mv "$command_file.new" "$command_file"

previous=""
[ ! -f "$previous_file" ] || previous="$(cat "$previous_file")"
if [ "$installed" != "$version" ]; then
    previous="$installed"
    printf '%s\n' "$previous" >"$previous_file.new"
    mv "$previous_file.new" "$previous_file"
fi
if [ -n "$installed" ] && [ -n "${TOKSPUR_VERSION:-}" ] && older "$version" "$installed"; then
    : >"$off_marker"
    say "Automatic updates are off after this rollback. Run tokspur update --on to resume them."
fi
printf '%s\n' "$version" >"$current_file.new"
mv "$current_file.new" "$current_file"
# Keep the release in use and the one it replaced, for rollback with TOKSPUR_VERSION.
for old in "$releases"/v*; do
    [ -d "$old" ] || continue
    name="${old##*/}"
    [ "$name" = "$version" ] || [ "$name" = "$previous" ] || rm -rf "$old"
done

if [ "$mode" = --auto ]; then
    printf '%s updated tokspur from %s to %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${previous:-none}" "$version"
    exit 0
fi
say "Installed tokspur $version"
say "  command:  $command_file"
say "  state:    $state"
say "  API key:  $state/.tokspur/state/api-key (written by tokspur configure)"
if [ -e "$off_marker" ]; then
    say "  updates:  off (tokspur update --on turns them on)"
else
    say "  updates:  checked daily and installed in the background (tokspur update --off turns this off)"
fi
case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) say "Add $bin_dir to your PATH, for example: echo 'export PATH=\"$bin_dir:\$PATH\"' >> ~/.zshrc" ;;
esac
say "Next: tokspur doctor, then add the model as the README describes."
