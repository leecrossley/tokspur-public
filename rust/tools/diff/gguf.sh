#!/bin/sh
# Differential check of the Rust GGUF reader against native/src/gguf.cpp.
#
# Builds, into build/rust-diff/ (Git-ignored):
#   gguf-driver  rust/tools/diff/gguf_driver.cpp + native/src/gguf.cpp
#   test-gguf    native/tests/test_gguf.cpp, run for its check count and to
#                write the C++ fuzz seed corpus
#   fuzz-gguf    native/tests/fuzz_gguf.cpp (standalone deterministic mode),
#                run for its acceptance count over 50,000 mutations
# then runs the ignored Rust differential test, which compares every case's accept/refuse decision, exact message and accessor results with the driver.
#
#   rust/tools/diff/gguf.sh             synthetic, mutated and sparse cases
#   rust/tools/diff/gguf.sh --real DIR  also the shard set in DIR (headers only;
#                                      no tensor data is read)
#
# Never starts a model or uses the GPU.
set -eu

root=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root/build/rust-diff"
cxx=${CXX:-c++}
mkdir -p "$out"

build() {
    "$cxx" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -I"$root/native/include" "$@"
}
build "$root/rust/tools/diff/gguf_driver.cpp" "$root/native/src/gguf.cpp" -o "$out/gguf-driver"
build "$root/native/tests/test_gguf.cpp" "$root/native/src/gguf.cpp" -o "$out/test-gguf"
build -DTOKSPUR_STANDALONE_FUZZ=1 "$root/native/tests/fuzz_gguf.cpp" "$root/native/src/gguf.cpp" \
    -o "$out/fuzz-gguf"

rm -rf "$out/cpp-seeds"
"$out/test-gguf" --write-fuzz-seeds "$out/cpp-seeds" > "$out/cpp-checks.txt"
"$out/fuzz-gguf" "$out/cpp-seeds" > "$out/cpp-fuzz.txt"
cat "$out/cpp-checks.txt" "$out/cpp-fuzz.txt"

TOKSPUR_GGUF_DIFF_DRIVER="$out/gguf-driver"
TOKSPUR_GGUF_DIFF_OUT="$out"
TOKSPUR_GGUF_DIFF_CPP_SEEDS="$out/cpp-seeds"
TOKSPUR_GGUF_DIFF_CPP_FUZZ="$out/cpp-fuzz.txt"
export TOKSPUR_GGUF_DIFF_DRIVER TOKSPUR_GGUF_DIFF_OUT TOKSPUR_GGUF_DIFF_CPP_SEEDS TOKSPUR_GGUF_DIFF_CPP_FUZZ

TOKSPUR_GGUF_FUZZ_SEEDS="$out/rust-seeds"
export TOKSPUR_GGUF_FUZZ_SEEDS
rm -rf "$out/rust-seeds"

filter=differential::matches_the_cpp_reader
if [ "${1-}" = "--real" ]; then
    [ $# -eq 2 ] || { echo "usage: $0 [--real DIR]" >&2; exit 2; }
    TOKSPUR_GGUF_REAL_SHARDS="$2"
    export TOKSPUR_GGUF_REAL_SHARDS
    filter=differential::
fi

cd "$root/rust"
cargo test -p tokspur-gguf --target-dir "$out/target" --test gguf -- --ignored --nocapture \
    --test-threads 1 "$filter" seeds::write_fuzz_seeds
for seed in seed-0 seed-1 seed-2 seed-3; do
    cmp "$out/cpp-seeds/$seed" "$out/rust-seeds/$seed"
done
echo "fuzz seed corpus: Rust writer output identical to test_gguf --write-fuzz-seeds"
