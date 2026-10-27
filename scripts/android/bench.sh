#!/usr/bin/env bash
# Times opus1 against libopus on an Android phone with the `bench` command
# of examples/perf_harness.rs (docs/guide/08-performance.md explains it). Each
# run is pinned with taskset to one CPU, so a phone with big and little
# cores gets one table per core type.
#
# The harness measures each scenario in fresh processes. They inherit the
# CPU from taskset, so every measurement runs on the chosen CPU.
#
# Phones lower their clock when they get hot. Before each run the script
# waits until the CPU temperature is at most COOL_C degrees (default 40).
# During each run it writes the temperatures and the clock of every
# cluster to <name>.state.log every POLL seconds (default 30).
#
# Usage:
#   scripts/android/bench.sh [--no-build] [--runs N] [--cpus "7 5"] [--label NAME] [-- BENCH_ARGS ...]
#
#   --runs N     Runs per CPU (default 2). The CPUs take turns.
#   --cpus LIST  CPUs to pin to. Default: the last CPU of the fastest
#                cluster and the last CPU of the slowest cluster.
#   --label NAME Start of the result names (default bench).
#   BENCH_ARGS   Passed to `perf_harness bench`, for example
#                `-- --filter celt_fb --reps 9`.
#   --no-build   Use the program of the last build.
#
# Results, per CPU and run, in RESULTS_DIR (default $ANDROID_OUT/bench):
#   <label>_cpu<N>_run<R>.json   the --json file of the harness
#   <label>_cpu<N>_run<R>.md     the --md file of the harness
#   <label>_cpu<N>_run<R>.log    what the harness printed
#   <label>_cpu<N>_run<R>.state.log  temperatures and clocks while it ran
#   <label>_cpu<N>_run<R>.freq.txt   share of the run's time at each clock
#                                    of the CPU's cluster (cpufreq stats)
#
# A phone can also lower the highest clock of a cluster while it is cool,
# for reasons it does not show. The script then prints a warning: the
# cluster spent less than MIN_TOP_SHARE percent of the run (default 75) at
# the highest clock of the CPU. The times of such a run are longer than
# those of a run at the full clock. Its opus1 / libopus ratios can still be
# compared with other runs.
#
# Settings: see scripts/android/env.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

RUNS=2
CPUS=""
LABEL=bench
BENCH_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) NO_BUILD=1 ;;
        --runs) RUNS=$2; shift ;;
        --cpus) CPUS=$2; shift ;;
        --label) LABEL=$2; shift ;;
        --) shift; BENCH_ARGS=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
RESULTS_DIR=${RESULTS_DIR:-$ANDROID_OUT/bench}
mkdir -p "$RESULTS_DIR"

# Prints a warning when the cluster of a CPU spent less than MIN_TOP_SHARE
# percent of a run at the highest clock of that CPU.
#   warn_low_clock CPU FREQ_FILE
# FREQ_FILE is an output of freq_delta: lines of "<kHz> kHz <percent> %".
warn_low_clock() {
    local max_khz share
    max_khz=$(dev "cat /sys/devices/system/cpu/cpu$1/cpufreq/cpuinfo_max_freq" | tr -d '\r')
    share=$(awk -v f="$max_khz" '$1 == f && $2 == "kHz" { print $3 }' "$2")
    share=${share:-0}
    if awk -v s="$share" -v min="${MIN_TOP_SHARE:-75}" 'BEGIN { exit !(s < min) }'; then
        echo "   warning: the cluster of CPU $1 spent $share % of the run at its highest clock ($max_khz kHz)." >&2
        echo "            The phone limited the clock. Compare ratios, not times, with other runs." >&2
    fi
}

push_harness
if [ -z "$CPUS" ]; then
    CPUS="$(pick_cpu BIG) $(pick_cpu LITTLE)"
fi
args=""
for a in "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"; do args="$args '$a'"; done

for run in $(seq 1 "$RUNS"); do
    for cpu in $CPUS; do
        name="${LABEL}_cpu${cpu}_run${run}"
        echo ":: $name: CPU $cpu ($(cpu_name "$cpu")), run $run of $RUNS"
        wait_cool
        device_state | tee "$RESULTS_DIR/$name.state.log"
        job="$ANDROID_OUT/jobs/$name.sh"
        mkdir -p "$ANDROID_OUT/jobs"
        echo "taskset $(cpu_mask "$cpu") ./perf_harness bench --json results/$name.json --md results/$name.md$args" > "$job"
        freq_stats "$cpu" > "$RESULTS_DIR/$name.freq_before.txt"
        dev_start "$name" "$job"
        status=0
        dev_wait "$name" "$RESULTS_DIR/$name.state.log" || status=$?
        freq_stats "$cpu" > "$RESULTS_DIR/$name.freq_after.txt"
        freq_delta "$RESULTS_DIR/$name.freq_before.txt" "$RESULTS_DIR/$name.freq_after.txt" \
            > "$RESULTS_DIR/$name.freq.txt"
        echo "   clock of the cluster of CPU $cpu during the run:"
        sed 's/^/     /' "$RESULTS_DIR/$name.freq.txt"
        warn_low_clock "$cpu" "$RESULTS_DIR/$name.freq.txt"
        "$ADB" pull -q "$DEVICE_DIR/jobs/$name.out" "$RESULTS_DIR/$name.log" >/dev/null
        if [ "$status" != 0 ]; then
            echo "error: the harness failed with status $status; see $RESULTS_DIR/$name.log" >&2
            exit 1
        fi
        "$ADB" pull -q "$DEVICE_DIR/results/$name.json" "$RESULTS_DIR/$name.json" >/dev/null
        "$ADB" pull -q "$DEVICE_DIR/results/$name.md" "$RESULTS_DIR/$name.md" >/dev/null
        device_state | tee -a "$RESULTS_DIR/$name.state.log"
        echo "   wrote $RESULTS_DIR/$name.json"
    done
done
