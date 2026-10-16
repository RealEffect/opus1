# Settings shared by the scripts in scripts/android/. The other scripts
# source this file; it is not run on its own.
#
# Every setting can be changed with an environment variable of the same name:
#
#   ANDROID_NDK_HOME  The Android NDK (Native Development Kit) directory.
#                     Required by the scripts that build.
#   ANDROID_API       Android API level to build for. Default 21, the
#                     default of `cargo ndk`.
#   ADB               The adb program. Default: `adb` on PATH.
#   ANDROID_SERIAL    Serial number of the phone, when more than one is
#                     connected (adb reads it itself).
#   ANDROID_OUT       Where builds and results go on this computer.
#                     Default: target/android in the checkout.
#   DEVICE_DIR        Working directory on the phone. Default
#                     /data/local/tmp/opus1. The phone's shell user can
#                     write there without root.
#   LIBOPUS_PREFIX    The libopus that build-libopus.sh installs and the
#                     other scripts link. Default: $ANDROID_OUT/libopus.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ANDROID_API=${ANDROID_API:-21}
ADB=${ADB:-adb}
ANDROID_OUT=${ANDROID_OUT:-$REPO_ROOT/target/android}
DEVICE_DIR=${DEVICE_DIR:-/data/local/tmp/opus1}
LIBOPUS_PREFIX=${LIBOPUS_PREFIX:-$ANDROID_OUT/libopus}
# Cargo builds for the phone in their own directory, so they do not
# disturb the builds for this computer.
ANDROID_CARGO_TARGET_DIR=${ANDROID_CARGO_TARGET_DIR:-$ANDROID_OUT/cargo}
RUST_TARGET=aarch64-linux-android

# Stops with a message when the NDK is missing.
need_ndk() {
    if [ -z "${ANDROID_NDK_HOME:-}" ] || [ ! -d "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" ]; then
        echo "error: set ANDROID_NDK_HOME to the Android NDK directory" >&2
        exit 2
    fi
}

# The directory of the NDK's clang and LLVM tools.
ndk_bin() {
    local host
    host=$(ls "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" | head -n 1)
    echo "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$host/bin"
}

# Stops with a message when no phone answers.
need_device() {
    if ! "$ADB" get-state >/dev/null 2>&1; then
        echo "error: no phone found by '$ADB get-state'. Connect one, allow USB debugging," >&2
        echo "       and set ADB (and ANDROID_SERIAL if several are connected)." >&2
        exit 2
    fi
}

# Runs a command line in the phone's shell.
dev() {
    "$ADB" shell "$@"
}

# Builds opus1 for the phone with cargo-ndk, linking the libopus of
# build-libopus.sh. The arguments are passed to cargo.
#
# Tests and the harness read fixtures/answer_16k.wav from a path that is
# fixed when they are compiled: env!("CARGO_MANIFEST_DIR") followed by
# /fixtures/answer_16k.wav. rustc-wrapper.sh sets CARGO_MANIFEST_DIR to
# DEVICE_DIR while opus1's own code is compiled, so the programs look for
# the file on the phone. Nothing else in the build uses that variable.
cargo_android() {
    need_ndk
    if [ ! -f "$LIBOPUS_PREFIX/lib/libopus.a" ]; then
        echo "error: $LIBOPUS_PREFIX/lib/libopus.a is missing; run scripts/android/build-libopus.sh" >&2
        exit 2
    fi
    # Cargo does not notice a new libopus.a. When the library is newer than
    # the last build, or another one, remove the build of audiopus_sys, so
    # that everything that links libopus is linked again.
    local stamp="$ANDROID_CARGO_TARGET_DIR/libopus-stamp.txt"
    if [ ! -f "$stamp" ] || [ "$LIBOPUS_PREFIX/lib/libopus.a" -nt "$stamp" ] \
        || [ "$(cat "$stamp")" != "$LIBOPUS_PREFIX" ]; then
        (cd "$REPO_ROOT" && CARGO_TARGET_DIR="$ANDROID_CARGO_TARGET_DIR" \
            cargo clean --quiet --release --target "$RUST_TARGET" -p audiopus_sys) || true
        mkdir -p "$ANDROID_CARGO_TARGET_DIR"
        echo "$LIBOPUS_PREFIX" > "$stamp"
    fi
    (
        cd "$REPO_ROOT"
        LIBOPUS_LIB_DIR="$LIBOPUS_PREFIX" LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
            OPUS1_DEVICE_DIR="$DEVICE_DIR" \
            RUSTC_WRAPPER="$REPO_ROOT/scripts/android/rustc-wrapper.sh" \
            CARGO_TARGET_DIR="$ANDROID_CARGO_TARGET_DIR" \
            cargo ndk -t arm64-v8a -P "$ANDROID_API" "$@"
    )
}

# Prints a short state of the phone: temperatures from the thermal service
# and the current clock of each processor cluster. /sys/class/thermal is
# not readable for the shell user on many phones, so the temperatures come
# from `dumpsys thermalservice`.
device_state() {
    dev 'dumpsys thermalservice 2>/dev/null | sed -n "/Current temperatures from HAL/,/Current cooling devices/p" | grep -o "mValue=[0-9.]*, mType=[0-9]*, mName=[A-Z_]*" | sed "s/mValue=//; s/, mType=[0-9]*, mName=/ /" | tr "\n" " "; echo; for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$(basename $p) cpus $(cat $p/related_cpus) cur_khz $(cat $p/scaling_cur_freq) max_khz $(cat $p/cpuinfo_max_freq)"; done'
}

# Starts a shell script on the phone in the background, detached from adb,
# so that it keeps running when the USB connection drops.
#   dev_start NAME FILE
# FILE is a script on this computer. It is copied to $DEVICE_DIR/jobs/ and
# run there by /system/bin/sh from $DEVICE_DIR. Its output goes to
# jobs/NAME.out on the phone and its exit status to jobs/NAME.status.
dev_start() {
    local name=$1 file=$2
    dev "mkdir -p $DEVICE_DIR/jobs && rm -f $DEVICE_DIR/jobs/$name.status $DEVICE_DIR/jobs/$name.out"
    "$ADB" push -q -q "$file" "$DEVICE_DIR/jobs/$name.sh" >/dev/null
    # setsid puts the job in its own session. Without it, adb waits until
    # the job ends, even with its input and output redirected.
    dev "cd $DEVICE_DIR && (setsid sh -c 'sh jobs/$name.sh; echo \$? > jobs/$name.status' > jobs/$name.out 2>&1 < /dev/null &)"
}

# Waits for a job of dev_start to end and returns its exit status.
#   dev_wait NAME [STATE_LOG]
# Every POLL seconds (default 30) it prints the last line of the job's
# output. With STATE_LOG it also appends the time and device_state to that
# file, to see whether the phone got hot or lowered its clock.
dev_wait() {
    local name=$1 log=${2:-} status line start
    start=$(date +%s)
    while :; do
        # `|| true`: the file is missing while the job runs, and the
        # scripts run with `set -e`.
        status=$(dev "cat $DEVICE_DIR/jobs/$name.status 2>/dev/null || true" | tr -d '\r')
        if [ -n "$log" ]; then
            { echo "t=$(( $(date +%s) - start ))s"; device_state; } >> "$log" 2>/dev/null || true
        fi
        if [ -n "$status" ]; then
            return "$status"
        fi
        line=$(dev "tail -n 1 $DEVICE_DIR/jobs/$name.out 2>/dev/null || true" | tr -d '\r' | cut -c1-100)
        echo "   [$(( $(date +%s) - start )) s] $line"
        sleep "${POLL:-30}"
    done
}

# Prints the bit mask of one CPU for taskset, in hexadecimal.
cpu_mask() {
    printf '%x' $((1 << $1))
}

# Prints the number of a CPU of the fastest cluster (BIG) or of the slowest
# cluster (LITTLE): the last CPU of the cpufreq policy with the highest or
# the lowest maximum clock. The first CPU often handles more interrupts.
pick_cpu() {
    dev 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$(cat $p/cpuinfo_max_freq) $(cat $p/related_cpus)"; done' \
        | tr -d '\r' | sort -n | { if [ "$1" = BIG ]; then tail -n 1; else head -n 1; fi; } | awk '{print $NF}'
}

# Prints the core type of a CPU, from the "CPU part" field of /proc/cpuinfo
# (ARM's part numbers; others are printed as they are).
cpu_name() {
    local part
    part=$(dev "cat /proc/cpuinfo" | tr -d '\r' | awk -v cpu="$1" '
        /^processor/ { n = $NF } /^CPU part/ && n == cpu { print $NF }')
    case "$part" in
        0xd03) echo Cortex-A53 ;; 0xd05) echo Cortex-A55 ;; 0xd0a) echo Cortex-A75 ;;
        0xd0b) echo Cortex-A76 ;; 0xd0d) echo Cortex-A77 ;; 0xd41) echo Cortex-A78 ;;
        0xd44) echo Cortex-X1 ;; 0xd46) echo Cortex-A510 ;; 0xd47) echo Cortex-A710 ;;
        0xd48) echo Cortex-X2 ;; 0xd4d) echo Cortex-A715 ;; 0xd4e) echo Cortex-X3 ;;
        *) echo "part-$part" ;;
    esac
}

# Builds examples/perf_harness.rs for the phone and copies it, with
# fixtures/, to $DEVICE_DIR.
#   push_harness [profile]
# The normal build uses the release profile as it is and is copied to
# $DEVICE_DIR/perf_harness. With `profile`, the build keeps line tables and
# symbols for simpleperf (CARGO_PROFILE_RELEASE_DEBUG=line-tables-only,
# CARGO_PROFILE_RELEASE_STRIP=none), uses its own cargo directory and is
# copied to $DEVICE_DIR/perf_harness_prof.
# Set NO_BUILD=1 to copy the program of the last build.
push_harness() {
    local dir=$ANDROID_CARGO_TARGET_DIR name=perf_harness
    if [ "${1:-}" = profile ]; then
        dir=$ANDROID_OUT/cargo-profile
        name=perf_harness_prof
    fi
    if [ "${NO_BUILD:-0}" != 1 ]; then
        echo ":: building $name for $RUST_TARGET (API $ANDROID_API)"
        if [ "${1:-}" = profile ]; then
            ANDROID_CARGO_TARGET_DIR=$dir CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
                CARGO_PROFILE_RELEASE_STRIP=none \
                cargo_android build --release --example perf_harness
        else
            cargo_android build --release --example perf_harness
        fi
    fi
    mkdir -p "$ANDROID_OUT/bin"
    cp "$dir/$RUST_TARGET/release/examples/perf_harness" "$ANDROID_OUT/bin/$name"
    need_device
    dev "mkdir -p $DEVICE_DIR/fixtures $DEVICE_DIR/results"
    "$ADB" push -q -q --sync "$REPO_ROOT/fixtures/." "$DEVICE_DIR/fixtures/" >/dev/null
    "$ADB" push -q -q "$ANDROID_OUT/bin/$name" "$DEVICE_DIR/$name" >/dev/null
    dev "chmod 755 $DEVICE_DIR/$name"
}

# Prints the time the cluster of a CPU has spent at each clock since boot:
# lines of "<kHz> <time in 10 ms units>" (cpufreq's stats/time_in_state).
freq_stats() {
    dev "cat /sys/devices/system/cpu/cpu$1/cpufreq/stats/time_in_state" | tr -d '\r'
}

# Compares two outputs of freq_stats: prints the share of the time between
# them at each clock and the mean clock. A run that kept its core busy
# should show nearly all of its time at the highest clock.
freq_delta() {
    awk 'NR == FNR { before[$1] = $2; next }
         { d = $2 - before[$1]; if (d > 0) { t[$1] = d; total += d; sum += $1 * d } }
         END {
             if (total == 0) { print "no time recorded"; exit }
             for (f in t) printf "%d kHz %.1f %%\n", f, 100 * t[f] / total
             printf "mean %.0f kHz over %.1f s\n", sum / total, total / 100
         }' "$1" "$2" | sort -rn
}

# Prints the CPU temperature in degrees Celsius, as an integer.
device_cpu_temp() {
    dev 'dumpsys thermalservice 2>/dev/null | sed -n "/Current temperatures from HAL/,/Current cooling devices/p" | grep "mName=CPU" | head -n 1 | sed "s/.*mValue=\([0-9]*\).*/\1/"' | tr -d '\r'
}

# Waits until the CPU is at most COOL_C degrees (default 40) or MAX_WAIT
# seconds (default 600) have passed.
wait_cool() {
    local limit=${COOL_C:-40} waited=0 temp
    while :; do
        temp=$(device_cpu_temp)
        if [ -z "$temp" ] || [ "$temp" -le "$limit" ] || [ "$waited" -ge "${MAX_WAIT:-600}" ]; then
            echo "cpu temperature ${temp:-unknown} C after ${waited} s of waiting"
            return 0
        fi
        sleep 15
        waited=$((waited + 15))
    done
}
