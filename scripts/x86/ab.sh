#!/usr/bin/env bash
# Times two builds of opus1 in turns on this computer, to see how much
# faster or slower a branch is than its baseline:
#   A  the baseline, a source tree named on the command line;
#   B  the branch, the source tree this script is in.
# Both build examples/perf_harness.rs and link the same libopus, so the
# libopus times are a control: they should not change from A to B.
#
# Why in turns: the clock and the temperature of a computer change during a
# session. A run of A and a run of B one right after the other see the same
# state, so each such pair gives one comparison. The order turns from pair
# to pair (A B, then B A), so a slow drift does not favour one build. With
# 5 passes the order is A B B A A B B A A B. One run is the bench set of
# perf_harness: about 3 minutes on an Intel Core i7-12700.
#
# It writes into the output folder:
#   system.txt     CPU, vector instruction sets, operating system, compilers
#   builds.txt     the two source trees, their commits and programs
#   build-A.log, build-B.log, build-report.log   the output of the builds
#   runs/pNN-A.*, runs/pNN-B.*   bench records of pass NN (.json, .md, .log)
#   runs.txt       order, start, duration, load and exit status of each run
#   ab-report.md   per scenario and as geometric means: time of B / time of
#                  A for opus1 encode and decode and for libopus (the
#                  control); see scripts/x86/ab_report.rs
#   ab-summary.txt the geometric means in a few lines
#   pairs.md       the same pairs as `perf_harness pairs` prints them
#
# Usage, from any folder:
#   scripts/x86/ab.sh [--passes N] [--out DIR] [--build-only] BASELINE_DIR
#                     [-- BENCH_ARGS ...]
#
#   BASELINE_DIR  the source tree of the baseline (A)
#   --passes N    runs of each build (default 5)
#   --out DIR     output folder (default: ab-results-<date> in the root of
#                 this tree)
#   --build-only  build the two programs and the report tool, check that
#                 they start, and stop without timing
#   BENCH_ARGS    passed to `perf_harness bench`, for example
#                 -- --filter celt_fb_48k_mono --reps 3 --procs 1
#
# scripts/x86/test.sh runs this script when it gets --ab BASELINE_DIR.
#
# Settings (environment variables):
#   LIBOPUS_LIB_DIR     a libopus build: a folder with lib/libopus.a.
#                       Needed; test.sh sets it to its own build.
#   LIBOPUS_NO_PKG, LIBOPUS_STATIC   default 1, as for the tests.
#   CARGO_BUILD_TARGET  optional: build for another target, for example
#                       x86_64-apple-darwin under Rosetta 2 on a Mac.
set -uo pipefail

PASSES=5
OUT=""
BUILD_ONLY=0
BASE=""
BENCH_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --passes) PASSES=${2:-}; shift ;;
        --out) OUT=${2:-}; shift ;;
        --build-only) BUILD_ONLY=1 ;;
        --) shift; BENCH_ARGS=("$@"); break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *)
            if [ -n "$BASE" ]; then
                echo "more than one baseline folder: $BASE and $1" >&2
                exit 2
            fi
            BASE=$1 ;;
    esac
    shift
done

die() { echo "error: $*" >&2; exit 1; }

case "$PASSES" in
    '' | *[!0-9]* | 0) die "--passes needs a whole number of at least 1" ;;
esac
[ -n "$BASE" ] || die "name the source tree of the baseline; see the header of this script"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BASE=$(cd "$BASE" 2>/dev/null && pwd) || die "no folder $BASE"
[ "$BASE" != "$REPO" ] || die "the baseline folder is this tree"
for tree in "$BASE" "$REPO"; do
    [ -f "$tree/Cargo.toml" ] && [ -f "$tree/examples/perf_harness.rs" ] \
        || die "$tree is not an opus1 source tree (no Cargo.toml or examples/perf_harness.rs)"
done
[ -n "${LIBOPUS_LIB_DIR:-}" ] || die "LIBOPUS_LIB_DIR is not set (the folder of a libopus build)"
export LIBOPUS_NO_PKG=${LIBOPUS_NO_PKG:-1} LIBOPUS_STATIC=${LIBOPUS_STATIC:-1}
for tool in cargo rustc; do
    command -v "$tool" >/dev/null || die "$tool is missing"
done

STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$REPO/ab-results-$STAMP}
mkdir -p "$OUT/runs" || die "cannot make $OUT"
OUT=$(cd "$OUT" && pwd)
# The programs, per target, so that builds for two targets do not mix.
BIN="$REPO/target/ab-bin${CARGO_BUILD_TARGET:+-$CARGO_BUILD_TARGET}"
mkdir -p "$BIN"

# The commit of a source tree: the file COMMIT of a kit, or git.
commit_of() {
    cat "$1/COMMIT" 2>/dev/null || git -C "$1" rev-parse --short HEAD 2>/dev/null || echo unknown
}
loadavg() {
    if [ -r /proc/loadavg ]; then
        cut -d ' ' -f 1-3 /proc/loadavg
    else
        sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | sed 's/^ *//; s/ *$//'
    fi
}
COMMIT_A=$(commit_of "$BASE")
COMMIT_B=$(commit_of "$REPO")

echo ":: system"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -a)"
    if [ -r /proc/cpuinfo ]; then
        grep -qi microsoft /proc/version 2>/dev/null && echo "wsl: yes" || echo "wsl: no"
        [ -r /etc/os-release ] && echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
        echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
        echo "logical cpus: $(grep -c '^processor' /proc/cpuinfo)"
        echo "memory: $(awk '/^MemTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo)"
        flags=$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2-)
        for f in sse2 ssse3 sse4_1 sse4_2 avx avx2 fma f16c bmi2 avx512f avx512bw avx512vl avx512_vnni avx_vnni; do
            case " $flags " in *" $f "*) echo "flag $f: yes" ;; *) echo "flag $f: no" ;; esac
        done
        gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
        [ -r "$gov" ] && echo "cpufreq governor: $(cat "$gov")"
        # Under WSL the Windows power plan sets the clock.
        if command -v powercfg.exe >/dev/null 2>&1; then
            echo "windows power plan: $(powercfg.exe /getactivescheme 2>/dev/null | tr -d '\r' | head -n 1)"
        fi
    elif [ "$(uname -s)" = Darwin ]; then
        echo "os: macOS $(sw_vers -productVersion 2>/dev/null)"
        echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
        echo "logical cpus: $(sysctl -n hw.logicalcpu 2>/dev/null)"
        echo "memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GiB"
        echo "translated by Rosetta 2: $(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)"
        features=$(sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | tr '\n' ' ')
        [ -n "$features" ] && echo "cpu features: $features"
    fi
    echo "load average: $(loadavg)"
    echo "rustc: $(rustc -V 2>/dev/null) ($(rustc -vV 2>/dev/null | sed -n 's/^host: //p'))"
    echo "cargo: $(cargo -V 2>/dev/null)"
    echo "cc: $(cc --version 2>/dev/null | head -n 1 || echo missing)"
    echo "cargo build target: ${CARGO_BUILD_TARGET:-host}"
    echo "RUSTFLAGS: ${RUSTFLAGS:-none}"
    echo "libopus: $LIBOPUS_LIB_DIR"
    echo "baseline (A): $BASE, commit $COMMIT_A"
    echo "branch (B): $REPO, commit $COMMIT_B"
} | tee "$OUT/system.txt"

# build NAME TREE: builds perf_harness of TREE in its own target folder and
# copies it to $BIN/perf_harness_NAME.
build() {
    local name=$1 tree=$2
    local target_dir="$REPO/target/ab-$name"
    echo ":: building $name ($tree)"
    (cd "$tree" && CARGO_TARGET_DIR="$target_dir" cargo build --release --example perf_harness) \
        > "$OUT/build-$name.log" 2>&1 || die "the build of $name failed; see $OUT/build-$name.log"
    local exe="$target_dir/${CARGO_BUILD_TARGET:+$CARGO_BUILD_TARGET/}release/examples/perf_harness"
    [ -f "$exe" ] || die "no $exe after the build"
    cp "$exe" "$BIN/perf_harness_$name" || die "cannot copy $exe"
    echo "$name: $tree, commit $(commit_of "$tree"), perf_harness $(wc -c < "$exe" | tr -d ' ') bytes" >> "$OUT/builds.txt"
}
: > "$OUT/builds.txt"
build A "$BASE"
build B "$REPO"
echo ":: building the report tool"
rustc --edition 2024 -O -o "$BIN/ab_report" "$REPO/scripts/x86/ab_report.rs" > "$OUT/build-report.log" 2>&1 \
    || die "the report tool did not build; see $OUT/build-report.log"
for name in A B; do
    "$BIN/perf_harness_$name" list > /dev/null 2>&1 || die "perf_harness_$name does not start"
done
"$BIN/ab_report" --help > /dev/null || die "the report tool does not start"
cat "$OUT/builds.txt"

if [ "$BUILD_ONLY" = 1 ]; then
    echo "built both programs and the report tool; no timing (--build-only)"
    exit 0
fi

# The order of the runs: A B in odd passes, B A in even ones.
ORDER=""
for pass in $(seq 1 "$PASSES"); do
    if [ $((pass % 2)) = 1 ]; then ORDER="$ORDER A B"; else ORDER="$ORDER B A"; fi
done
ORDER=${ORDER# }
echo ":: timing in the order $ORDER; keep the computer idle"
{
    echo "| pass | build | start (UTC) | seconds | load average before | exit status |"
    echo "|---:|---|---|---:|---|---:|"
} > "$OUT/runs.txt"
PAIR_FILES=()
for pass in $(seq 1 "$PASSES"); do
    if [ $((pass % 2)) = 1 ]; then order="A B"; else order="B A"; fi
    p=$(printf 'p%02d' "$pass")
    for name in $order; do
        file="$OUT/runs/$p-$name"
        load=$(loadavg)
        start=$(date +%s)
        start_utc=$(date -u +%H:%M:%S)
        echo "pass $pass of $PASSES, $name: started $(date +%H:%M:%S), load average $load"
        "$BIN/perf_harness_$name" bench --json "$file.json" --md "$file.md" \
            ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} > "$file.log" 2>&1
        status=$?
        seconds=$(( $(date +%s) - start ))
        echo "| $pass | $name | $start_utc | $seconds | $load | $status |" >> "$OUT/runs.txt"
        [ "$status" = 0 ] || echo "pass $pass, $name: exit status $status; see $file.log"
    done
    if [ -s "$OUT/runs/$p-A.json" ] && [ -s "$OUT/runs/$p-B.json" ]; then
        PAIR_FILES+=("$OUT/runs/$p-A.json" "$OUT/runs/$p-B.json")
    else
        echo "pass $pass: a bench record is missing; this pair is left out of the report"
    fi
done
[ "${#PAIR_FILES[@]}" -gt 0 ] || die "no pass has both bench records"

echo ":: report"
"$BIN/ab_report" --a "baseline, commit $COMMIT_A" --b "branch, commit $COMMIT_B" --order "$ORDER" \
    --summary "$OUT/ab-summary.txt" "${PAIR_FILES[@]}" > "$OUT/ab-report.md" \
    || die "the report tool failed"
"$BIN/perf_harness_B" pairs "${PAIR_FILES[@]}" > "$OUT/pairs.md" 2>&1
cat "$OUT/ab-summary.txt"
echo "report: $OUT/ab-report.md"
