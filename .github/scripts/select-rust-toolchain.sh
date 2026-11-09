#!/usr/bin/env bash
set -euo pipefail

# Use the installed pinned toolchain directly on either hosted Linux or macOS.
# Write only job-local PATH and environment files.
tokspur_channel="$(sed -n 's/^channel = "\([^"]*\)"$/\1/p' "${GITHUB_WORKSPACE:?}/rust/rust-toolchain.toml")"
test "$tokspur_channel" = "1.99.1"
tokspur_rustc="$(rustup which --toolchain "$tokspur_channel" rustc)"
tokspur_rust_bin="$(dirname "$tokspur_rustc")"
for tokspur_tool in cargo rustc rustdoc rustfmt cargo-clippy; do
  if [[ ! -x "$tokspur_rust_bin/$tokspur_tool" ]]; then
    echo "Pinned Rust tool is missing: $tokspur_rust_bin/$tokspur_tool" >&2
    exit 1
  fi
done
printf '%s\n' "${CARGO_HOME:-${HOME}/.cargo}/bin" "$tokspur_rust_bin" >> "${GITHUB_PATH:?}"
printf 'RUSTC=%s/rustc\nRUSTDOC=%s/rustdoc\nRUSTUP_TOOLCHAIN=%s\n' "$tokspur_rust_bin" "$tokspur_rust_bin" "$tokspur_channel" >> "${GITHUB_ENV:?}"
"$tokspur_rust_bin/rustc" -Vv
"$tokspur_rust_bin/cargo" -V
