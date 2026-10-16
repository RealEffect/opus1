#!/usr/bin/env bash
# Profiles one operation of examples/perf_harness.rs on an Android phone
# with simpleperf: hardware counters per frame, and the functions that take
# the processor time.
#
# The harness command `run` repeats one operation (encode or decode, of
# opus1 or of libopus) over a clip in a loop, pinned with taskset to one
# CPU. It prints its process id once the clip is ready and the loop starts.
# The script then attaches simpleperf to that process:
#   1. `simpleperf stat` three times, each for WINDOW seconds, with a group
#      of at most 7 events. The phones this was written for have 7 counters
#      per CPU, so the events are counted all the time, without
#      multiplexing. The groups are in STAT_GROUPS below.
#   2. `simpleperf record` of cpu-cycles at FREQ samples per second for
#      WINDOW seconds, then `simpleperf report --sort symbol`.
# Per frame values: a count divided by the frames of its window. The frames
# of a window are its length divided by the time per frame that the loop
# reports when it ends.
#
# The program is built with line tables and symbols (see push_harness in
# env.sh), in its own cargo directory, so that the release build used by
# bench.sh is not changed.
#
# Usage:
#   scripts/android/profile.sh [--no-build] [--cpu N | --big | --little]
#                              [--window S] [--freq HZ] SCENARIO OP IMPL
#
#   SCENARIO  a scenario name of `perf_harness list`
#   OP        enc or dec
#   IMPL      rust or c
#   --cpu N   CPU to pin to; --big / --little pick one as bench.sh does
#             (default --big).
#   --window S  seconds per counter group and for the record (default 4).
#   --freq HZ   samples per second of the record (default 4000).
#
# Results in RESULTS_DIR (default $ANDROID_OUT/profile), with the name
# <SCENARIO>_<OP>_<IMPL>_cpu<N>:
#   .summary.txt  per-frame counters and derived values (IPC and others)
#   .report.txt   simpleperf report, by function
#   .stat1-3.csv  the raw counts, .run.txt what the harness printed,
#   .data         the samples (open with simpleperf or the NDK's
#                 simpleperf scripts on this computer)
#
# Settings: see scripts/android/env.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

CPU=""
WHICH=BIG
WINDOW=4
FREQ=4000
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) NO_BUILD=1 ;;
        --cpu) CPU=$2; shift ;;
        --big) WHICH=BIG ;;
        --little) WHICH=LITTLE ;;
        --window) WINDOW=$2; shift ;;
        --freq) FREQ=$2; shift ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) POS+=("$1") ;;
    esac
    shift
done
if [ ${#POS[@]} -ne 3 ]; then
    sed -n '2,/^set -euo/p' "$0" | sed '$d'
    exit 2
fi
SCENARIO=${POS[0]} OP=${POS[1]} IMPL=${POS[2]}

# Three groups of events, each with cpu-cycles and at most 7 events in all.
STAT_GROUPS=(
    "cpu-cycles,instructions,branch-misses,raw-br-retired,raw-l1d-cache,raw-l1d-cache-refill,raw-l2d-cache-refill"
    "cpu-cycles,raw-l2d-cache,raw-l3d-cache,raw-l3d-cache-refill,raw-ll-cache-miss-rd,raw-stall-frontend,raw-stall-backend"
    "cpu-cycles,raw-l1i-cache-refill,raw-ase-spec,raw-inst-spec,raw-ld-spec,raw-st-spec,raw-dp-spec"
)

RESULTS_DIR=${RESULTS_DIR:-$ANDROID_OUT/profile}
mkdir -p "$RESULTS_DIR" "$ANDROID_OUT/jobs"
push_harness profile
CPU=${CPU:-$(pick_cpu "$WHICH")}
name="${SCENARIO}_${OP}_${IMPL}_cpu${CPU}"
run_seconds=$(( 4 * WINDOW + 4 ))
out="results/prof/$name"

job="$ANDROID_OUT/jobs/$name.sh"
{
    echo "mkdir -p results/prof"
    echo "rm -f $out.*"
    echo "taskset $(cpu_mask "$CPU") ./perf_harness_prof run --scenario $SCENARIO --op $OP --impl $IMPL --seconds $run_seconds > $out.run.txt 2>&1 &"
    echo "n=0"
    echo "until grep -q '^pid ' $out.run.txt 2>/dev/null; do sleep 0.2; n=\$((n + 1)); if [ \$n -gt 1500 ]; then echo 'the harness did not start'; exit 1; fi; done"
    echo "pid=\$(sed -n 's/^pid //p' $out.run.txt)"
    echo "sleep 1"
    i=1
    for group in "${STAT_GROUPS[@]}"; do
        echo "simpleperf stat -p \$pid --duration $WINDOW --csv -o $out.stat$i.csv -e $group"
        i=$((i + 1))
    done
    echo "simpleperf record -p \$pid --duration $WINDOW -e cpu-cycles -f $FREQ -o $out.data"
    echo "wait"
    echo "simpleperf report -i $out.data --sort symbol -n --percent-limit 0.2 > $out.report.txt 2>/dev/null"
} > "$job"

echo ":: $name: CPU $CPU ($(cpu_name "$CPU")), $run_seconds s"
wait_cool
device_state | tee "$RESULTS_DIR/$name.state.log"
dev_start "$name" "$job"
POLL=10 dev_wait "$name" "$RESULTS_DIR/$name.state.log"
for f in run.txt stat1.csv stat2.csv stat3.csv data report.txt; do
    "$ADB" pull -q "$DEVICE_DIR/$out.$f" "$RESULTS_DIR/$name.$f" >/dev/null
done

# Per-frame values. The harness ends with
#   done: <frames> frames in <seconds> s, <ns> ns per frame
ns_per_frame=$(sed -n 's/^done: .* s, \([0-9.]*\) ns per frame/\1/p' "$RESULTS_DIR/$name.run.txt")
{
    echo "scenario $SCENARIO"
    echo "operation $OP"
    echo "implementation $IMPL"
    echo "cpu $CPU $(cpu_name "$CPU")"
    echo "ns_per_frame $ns_per_frame"
    for i in 1 2 3; do
        awk -F, -v nspf="$ns_per_frame" -v group="$i" '
            $1 == "Total test time" { secs = $2 }
            $1 ~ /^[0-9]+$/ { count[$2] = $1; order[n++] = $2 }
            END {
                frames = secs * 1e9 / nspf
                for (k = 0; k < n; k++) {
                    e = order[k]
                    if (group > 1 && e == "cpu-cycles") continue
                    printf "%s_per_frame %.1f\n", e, count[e] / frames
                }
                if (group == 1) {
                    printf "ghz %.3f\n", count["cpu-cycles"] / secs / 1e9
                    printf "ipc %.3f\n", count["instructions"] / count["cpu-cycles"]
                }
            }' "$RESULTS_DIR/$name.stat$i.csv"
    done
} > "$RESULTS_DIR/$name.summary.txt"
cat "$RESULTS_DIR/$name.summary.txt"
echo "top functions:"
grep -E '^ *[0-9.]+%' "$RESULTS_DIR/$name.report.txt" | head -n 12
echo "   wrote $RESULTS_DIR/$name.*"
