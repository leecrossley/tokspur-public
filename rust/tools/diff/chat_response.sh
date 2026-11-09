#!/bin/sh
# Differential check of tokspur-text's chat/response port against the C++.
#
# Builds the unchanged C++ (native/src/chat.cpp, response.cpp, response_protocol.cpp, response_cli.cpp with native/tests/response_cli_main.cpp, json.cpp, unicode.cpp, tokenizer.cpp, gguf.cpp) with this directory's driver and NativeText stub into build/rust-diff/chat-response/, then runs the Rust runner, which writes build/rust-diff/chat-response/report.txt. Model-free: no weights, no GPU.
#
# Set TOKSPUR_PINNED_TEMPLATE to the pinned chat_template.jinja to include its literal extraction; its SHA-256 is checked first. Outputs stay in build/.
set -eu

root=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root/build/rust-diff/chat-response"
mkdir -p "$out"
cxx=${CXX:-c++}
src="$root/native/src"
here="$root/rust/tools/diff/chat-response"
flags="-std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -I$root/native/include"

# shellcheck disable=SC2086
"$cxx" $flags "$here/cpp/text_driver.cpp" "$src/chat.cpp" "$src/response.cpp" \
    "$src/response_protocol.cpp" "$src/json.cpp" "$src/unicode.cpp" "$src/tokenizer.cpp" \
    "$src/gguf.cpp" -o "$out/text-driver"
# shellcheck disable=SC2086
"$cxx" $flags "$root/native/tests/response_cli_main.cpp" "$src/response_cli.cpp" \
    "$src/response_protocol.cpp" "$src/response.cpp" "$src/json.cpp" "$src/unicode.cpp" \
    "$src/tokenizer.cpp" "$src/gguf.cpp" "$here/cpp/native_text_stub.cpp" -o "$out/response-cli"

set -- --driver "$out/text-driver" --cli "$out/response-cli" --out "$out"
template=${TOKSPUR_PINNED_TEMPLATE:-}
if [ -n "$template" ]; then
    expected=12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce
    actual=$(shasum -a 256 "$template" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected" ]; then
        echo "pinned template SHA-256 $actual differs from $expected" >&2
        exit 2
    fi
    set -- "$@" --template "$template"
fi

{
    echo "source $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet || echo ' (modified)')"
    "$cxx" --version | head -n 1
    rustc --version
    echo "template ${template:+$expected}"
} > "$out/environment.txt"

cd "$here"
cargo run --quiet --release --frozen --target-dir "$out/target" -- "$@"
