#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu

DEFAULT_MODE="rust"
mode="${TOKSPUR_CLI_IMPL:-$DEFAULT_MODE}"

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ -f "$script_dir/../profiles/flash-next-assets.json" ] && [ -f "$script_dir/../native/Makefile" ]; then
    root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
elif [ -f "$script_dir/../../profiles/flash-next-assets.json" ] && [ -f "$script_dir/../../native/Makefile" ]; then
    root=$(CDPATH= cd -- "$script_dir/../.." && pwd -P)
else
    echo "tokspur: cannot locate repository root from launcher path" >&2
    exit 1
fi

case "$mode" in
    rust)
        rust_bin="$root/build/rust-fast-math/release/tokspur"
        if [ ! -f "$rust_bin" ]; then
            echo "tokspur: Rust CLI binary not found at $rust_bin; run scripts/bootstrap.py" >&2
            exit 1
        fi
        exec "$rust_bin" "$@"
        ;;
    python)
        TOKSPUR_CLI_IMPL=python exec "$root/.venv/bin/python" -m tokspur.cli "$@"
        ;;
    *)
        echo "tokspur: invalid TOKSPUR_CLI_IMPL (expected rust or python)" >&2
        exit 2
        ;;
esac
