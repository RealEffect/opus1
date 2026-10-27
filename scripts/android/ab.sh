#!/usr/bin/env bash
# Runs several builds of examples/perf_harness.rs in turns on an Android
# phone, to decide whether a change makes opus1 faster. Make the builds
# first with build-variant.sh.
#
# Why in turns: the phone's clock and temperature change during a session,
# and two builds of the same code can differ by a few percent only because
# the linker placed the code at other addresses. Builds measured one right
# after the other see the same state of the phone. A change is real when it
# has the same sign in every pair and is larger than the difference between
# two placements of the same code (docs/guide/08-performance.md, "Code
# placement").
#
# The whole series runs as one job on the phone, detached from adb (see
# dev_start in env.sh). One pass runs each variant once on each CPU. The
# order of the variants turns by one place from pass to pass (A B, B A,
# A B, ...), so a slow drift of the clock does not favour one variant.
#
# Usage:
#   scripts/android/ab.sh [--passes N] [--cpus "4 7"] [--label NAME]
#                         [--no-push] BASE VARIANT... [-- BENCH_ARGS ...]
#
#   BASE VARIANT...  names given to build-variant.sh; BASE is compared with
#                    each of the others. A name with a '/' is the path of a
#                    program, for example the base built in another
#                    checkout: ../base/target/android/bin/perf_harness_base.
#                    Its name is then the file name without
#                    "perf_harness_".
#   --passes N       passes (default 3)
#   --cpus LIST      CPUs to pin to (default: the last CPU of the fastest
#                    and of the slowest cluster, as bench.sh)
#   --label NAME     name of the series (default ab)
#   --no-push        the programs are already on the phone
#   BENCH_ARGS       passed to `perf_harness bench`, for example
#                    `-- --filter celt_fb_48k_mono_64k_vbr_c10 --reps 9`
#
# Results in RESULTS_DIR (default $ANDROID_OUT/ab/LABEL):
#   <variant>_cpu<N>_pass<P>.json and .md   the bench records
#   series.log         what the job printed
#   state.log          temperatures and clocks during the job
#   freq_cpu<N>.txt    share of the job's time at each clock of the cluster
#   pairs_<variant>_cpu<N>.txt   `perf_harness pairs` of BASE against the
#                      variant on that CPU
#
# Settings: see scripts/android/env.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

PASSES=3
CPUS=""
LABEL=ab
PUSH=1
VARIANTS=()
# The program of each variant, in the order of VARIANTS.
PROGRAMS=()
BENCH_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --passes) PASSES=$2; shift ;;
        --cpus) CPUS=$2; shift ;;
        --label) LABEL=$2; shift ;;
        --no-push) PUSH=0 ;;
        --) shift; BENCH_ARGS=("$@"); break ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        */*)
            name=$(basename "$1")
            VARIANTS+=("${name#perf_harness_}")
            PROGRAMS+=("$1") ;;
        *)
            VARIANTS+=("$1")
            PROGRAMS+=("") ;;
    esac
    shift
done
if [ "${#VARIANTS[@]}" -lt 2 ]; then
    echo "error: name at least two variants, the base first" >&2
    exit 2
fi
RESULTS_DIR=${RESULTS_DIR:-$ANDROID_OUT/ab/$LABEL}
mkdir -p "$RESULTS_DIR" "$ANDROID_OUT/jobs"
need_device

dev "mkdir -p $DEVICE_DIR/fixtures $DEVICE_DIR/results/$LABEL"
if [ "$PUSH" = 1 ]; then
    "$ADB" push -q -q --sync "$REPO_ROOT/fixtures/." "$DEVICE_DIR/fixtures/" >/dev/null
    for i in "${!VARIANTS[@]}"; do
        v=${VARIANTS[$i]}
        program=${PROGRAMS[$i]:-$ANDROID_OUT/bin/perf_harness_$v}
        if [ ! -f "$program" ]; then
            echo "error: $program is missing; run scripts/android/build-variant.sh $v" >&2
            exit 2
        fi
        "$ADB" push -q -q "$program" "$DEVICE_DIR/perf_harness_$v" >/dev/null
        dev "chmod 755 $DEVICE_DIR/perf_harness_$v"
    done
fi
if [ -z "$CPUS" ]; then
    CPUS="$(pick_cpu BIG) $(pick_cpu LITTLE)"
fi
args=""
for a in "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"; do args="$args '$a'"; done

# The job: every pass runs each variant on each CPU. The order of the
# variants starts one place later in each pass.
job=$ANDROID_OUT/jobs/$LABEL.sh
n=${#VARIANTS[@]}
{
    echo "set -e"
    for pass in $(seq 1 "$PASSES"); do
        for cpu in $CPUS; do
            for k in $(seq 0 $((n - 1))); do
                v=${VARIANTS[$(( (k + pass - 1) % n ))]}
                name=${v}_cpu${cpu}_pass${pass}
                echo "echo \":: pass $pass, CPU $cpu, $v\""
                echo "taskset $(cpu_mask "$cpu") ./perf_harness_$v bench --json results/$LABEL/$name.json --md results/$LABEL/$name.md$args > results/$LABEL/$name.log 2>&1"
            done
        done
    done
} > "$job"

wait_cool
for cpu in $CPUS; do freq_stats "$cpu" > "$RESULTS_DIR/freq_cpu${cpu}_before.txt"; done
device_state | tee "$RESULTS_DIR/state.log"
dev_start "$LABEL" "$job"
status=0
dev_wait "$LABEL" "$RESULTS_DIR/state.log" || status=$?
"$ADB" pull -q "$DEVICE_DIR/jobs/$LABEL.out" "$RESULTS_DIR/series.log" >/dev/null
for cpu in $CPUS; do
    freq_stats "$cpu" > "$RESULTS_DIR/freq_cpu${cpu}_after.txt"
    freq_delta "$RESULTS_DIR/freq_cpu${cpu}_before.txt" "$RESULTS_DIR/freq_cpu${cpu}_after.txt" \
        > "$RESULTS_DIR/freq_cpu$cpu.txt"
    echo "   clock of the cluster of CPU $cpu ($(cpu_name "$cpu")) during the series:"
    sed 's/^/     /' "$RESULTS_DIR/freq_cpu$cpu.txt"
done
if [ "$status" != 0 ]; then
    echo "error: the series failed with status $status; see $RESULTS_DIR/series.log" >&2
    exit 1
fi
"$ADB" pull -q "$DEVICE_DIR/results/$LABEL/." "$RESULTS_DIR/" >/dev/null

# The comparison runs on the phone with the base program, so this computer
# needs no build of the harness.
base=${VARIANTS[0]}
for v in "${VARIANTS[@]:1}"; do
    for cpu in $CPUS; do
        files=""
        for pass in $(seq 1 "$PASSES"); do
            files="$files results/$LABEL/${base}_cpu${cpu}_pass$pass.json results/$LABEL/${v}_cpu${cpu}_pass$pass.json"
        done
        out=$RESULTS_DIR/pairs_${v}_cpu$cpu.txt
        dev "cd $DEVICE_DIR && ./perf_harness_$base pairs$files" | tr -d '\r' > "$out"
        echo ":: $v against $base on CPU $cpu ($(cpu_name "$cpu"))"
        cat "$out"
    done
done
