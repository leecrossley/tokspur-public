#!/bin/sh
# Summarises the fuzz runs that rust/tools/fuzz/run.sh left in DIR: per target, libFuzzer's exit status, run time, executions, final coverage (edge counters) and features, corpus size, and any crash input. Writes DIR/summary.txt and fails if any target crashed, timed out or did not finish.
#
#   summary.sh DIR TARGET...
set -eu

[ $# -ge 2 ] || { echo "usage: $0 DIR TARGET..." >&2; exit 2; }
dir=$1
shift

failed=0
{
    printf '%-18s %6s %8s %12s %9s %7s %8s %7s %s\n' \
        target status seconds execs exec/s cov ft corpus result
    for target in "$@"; do
        log="$dir/logs/$target.log"
        status=$(cat "$dir/logs/$target.status" 2>/dev/null || echo missing)
        execs=$(sed -n 's/^stat::number_of_executed_units: *//p' "$log" 2>/dev/null | tail -n 1)
        rate=$(sed -n 's/^stat::average_exec_per_sec: *//p' "$log" 2>/dev/null | tail -n 1)
        seconds=$(sed -n 's/^Done [0-9]* runs in \([0-9]*\) second.*/\1/p' "$log" 2>/dev/null | tail -n 1)
        final=$(grep -E '^#[0-9]+[[:space:]]+(DONE|NEW|REDUCE|pulse|INITED)' "$log" 2>/dev/null | tail -n 1)
        cov=$(printf '%s\n' "$final" | sed -n 's/.* cov: \([0-9]*\).*/\1/p')
        ft=$(printf '%s\n' "$final" | sed -n 's/.* ft: \([0-9]*\).*/\1/p')
        corpus=$(printf '%s\n' "$final" | sed -n 's/.* corp: \([0-9]*\)\/.*/\1/p')
        crashes=$(find "$dir/artifacts/$target" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$status" = 0 ] && [ "$crashes" = 0 ] && printf '%s\n' "$final" | grep -q DONE; then
            result=clean
        else
            result="FAILED ($crashes artifact(s))"
            failed=1
        fi
        printf '%-18s %6s %8s %12s %9s %7s %8s %7s %s\n' "$target" "$status" \
            "${seconds:-?}" "${execs:-?}" "${rate:-?}" "${cov:-?}" "${ft:-?}" "${corpus:-?}" "$result"
    done
} > "$dir/summary.txt"
cat "$dir/summary.txt"

# A crash's input and the first lines of its report, for the record.
for target in "$@"; do
    for artifact in $(find "$dir/artifacts/$target" -type f 2>/dev/null); do
        echo "crash input: $artifact"
        grep -m 5 -E 'panicked|ERROR|deadly signal|differ at byte' "$dir/logs/$target.log" || true
    done
done
exit "$failed"
