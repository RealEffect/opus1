#!/usr/bin/env bash
# Writes the golden hashes of examples/perf_harness.rs on an Android phone
# and copies the file back (docs/09-testing.md explains the golden file).
# With --check it also checks the hashes on the phone against a golden file
# written elsewhere, for example on a Mac, and prints the differences.
#
# Usage:
#   scripts/android/golden.sh [--no-build] [--cpu N] [--check FILE] [--out FILE]
#
#   --cpu N       CPU to pin the harness to. Default: the last CPU of the
#                 fastest cluster. The hashes do not depend on the CPU; this
#                 only makes the run faster.
#   --check FILE  Also run `golden --check FILE` on the phone. Its output
#                 goes to <out without .txt>_check.txt. The script exits
#                 with status 1 when a hash differs.
#   --out FILE    Where to put the golden file of the phone. Default:
#                 RESULTS_DIR/golden_android.txt (RESULTS_DIR defaults to
#                 $ANDROID_OUT/golden).
#   --no-build    Use the program of the last build.
#
# Settings: see scripts/android/env.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

CPU=""
CHECK=""
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) NO_BUILD=1 ;;
        --cpu) CPU=$2; shift ;;
        --check) CHECK=$2; shift ;;
        --out) OUT=$2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
RESULTS_DIR=${RESULTS_DIR:-$ANDROID_OUT/golden}
OUT=${OUT:-$RESULTS_DIR/golden_android.txt}
mkdir -p "$(dirname "$OUT")" "$ANDROID_OUT/jobs"

push_harness
CPU=${CPU:-$(pick_cpu BIG)}
MASK=$(cpu_mask "$CPU")

echo ":: golden --write on CPU $CPU ($(cpu_name "$CPU"))"
echo "taskset $MASK ./perf_harness golden --write results/golden_android.txt" > "$ANDROID_OUT/jobs/golden.sh"
dev_start golden "$ANDROID_OUT/jobs/golden.sh"
dev_wait golden
"$ADB" pull -q "$DEVICE_DIR/results/golden_android.txt" "$OUT" >/dev/null
echo "   wrote $OUT"

if [ -n "$CHECK" ]; then
    report="${OUT%.txt}_check.txt"
    echo ":: golden --check $CHECK"
    "$ADB" push -q "$CHECK" "$DEVICE_DIR/results/golden_reference.txt" >/dev/null
    echo "taskset $MASK ./perf_harness golden --check results/golden_reference.txt" > "$ANDROID_OUT/jobs/golden_check.sh"
    dev_start golden_check "$ANDROID_OUT/jobs/golden_check.sh"
    status=0
    dev_wait golden_check || status=$?
    "$ADB" pull -q "$DEVICE_DIR/jobs/golden_check.out" "$report" >/dev/null
    tail -n 3 "$report"
    echo "   wrote $report"
    exit "$status"
fi
