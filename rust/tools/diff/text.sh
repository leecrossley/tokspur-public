#!/bin/sh
# Differential check of the Rust text port against the real C++.
#
# Builds text_diff.cpp with native/src/unicode.cpp, native/src/tokenizer.cpp and native/src/gguf.cpp (for checked_add), all unchanged, then runs the ignored test rust/crates/tokspur-text/tests/cpp_differential.rs, which generates the cases, feeds the same file to both implementations and requires byte-identical results. Everything is written under build/rust-diff/ (ignored by Git): the driver, cases.txt, cpp.txt, rust.txt, summary.txt and provenance.txt. Set CARGO_TARGET_DIR to reuse a build.
set -eu

root=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root/build/rust-diff"
mkdir -p "$out"

c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -I"$root/native/include" \
    "$root/native/src/unicode.cpp" "$root/native/src/tokenizer.cpp" \
    "$root/native/src/gguf.cpp" "$root/rust/tools/diff/text_diff.cpp" \
    -o "$out/text-diff"

changes=$(git -C "$root" status --porcelain -- native rust)
{
    echo "source: $(git -C "$root" rev-parse HEAD)${changes:+ plus uncommitted changes under native/ or rust/}"
    echo "c++: $(c++ --version | head -n 1)"
    echo "c++ flags: -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror"
    echo "rustc: $(cd "$root/rust" && rustc --version)"
    echo "cargo test: --release -p tokspur-text --test cpp_differential -- --ignored"
} > "$out/provenance.txt"

cd "$root/rust"
TOKSPUR_TEXT_DIFF_DRIVER="$out/text-diff" TOKSPUR_TEXT_DIFF_DIR="$out" \
    cargo test --release -p tokspur-text \
    --target-dir "${CARGO_TARGET_DIR:-$root/build/rust-diff/target}" \
    --test cpp_differential -- --ignored --nocapture
cat "$out/provenance.txt"
