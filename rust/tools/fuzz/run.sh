#!/bin/sh
# Runs one instrumented fuzz target (rust/crates/tokspur-fuzz) for a bounded time: a single libFuzzer job under nice, its corpus in DIR/corpus/TARGET, crash inputs in DIR/artifacts/TARGET/, the log in DIR/logs/TARGET.log and libFuzzer's exit status in DIR/logs/TARGET.status. native/rust.mk rust-fuzz calls it once per target; rust/tools/fuzz/summary.sh reads the results.
#
#   run.sh DIR TARGET SECONDS NICENESS
#
# Model-free: no weights, no GPU. Everything stays under DIR (Git-ignored).
set -eu

[ $# -eq 4 ] || { echo "usage: $0 DIR TARGET SECONDS NICENESS" >&2; exit 2; }
dir=$1
target=$2
seconds=$3
niceness=$4

# Room for each target's input header. The GGUF seeds and the wire documents are larger than the 4 KiB the others need, and bpe stays at 1 KiB because its textbook reference encoder is quadratic in the input length.
case $target in
    gguf* | wire*) max_len=8192 ;;
    bpe) max_len=1024 ;;
    *) max_len=4104 ;;
esac

mkdir -p "$dir/corpus/$target" "$dir/artifacts/$target" "$dir/logs"
set -- -max_total_time="$seconds" -max_len="$max_len" -timeout=30 -rss_limit_mb=2048 \
    -print_final_stats=1 -artifact_prefix="$dir/artifacts/$target/"
if [ -f "$dir/dict/$target.dict" ]; then
    set -- "$@" -dict="$dir/dict/$target.dict"
fi

status=0
nice -n "$niceness" "$dir/bin/$target" "$@" "$dir/corpus/$target" \
    > "$dir/logs/$target.log" 2>&1 || status=$?
echo "$status" > "$dir/logs/$target.status"
echo "$target: libFuzzer exit status $status"
