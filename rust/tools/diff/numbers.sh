#!/bin/sh
# Differential check of tokspur-json's from_chars/to_chars emulation against the host libc++. Outputs stay under build/rust-diff (Git-ignored).
set -eu
root=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root/build/rust-diff"
mkdir -p "$out"
c++ -std=c++20 -O2 "$root/rust/tools/diff/tochars.cpp" -o "$out/tochars"
"$out/tochars" "${1:-200000}" > "$out/tochars.txt"
cd "$root/rust"
TOKSPUR_DIFF_TO_CHARS="$out/tochars.txt" cargo test --release -p tokspur-json \
    --test numbers_differential -- --ignored --nocapture
