#!/usr/bin/env bash
# Builds every test program of opus1 for a 64-bit ARM Android phone, copies
# them to the phone and runs them there. It prints one line per test
# program and a total.
#
# What it does:
#   1. `cargo ndk -t arm64-v8a test --release --no-run` builds the unit
#      tests (`--lib`) and every file of tests/ against the libopus of
#      build-libopus.sh (LIBOPUS_LIB_DIR, LIBOPUS_NO_PKG, LIBOPUS_STATIC).
#      Documentation tests are not built: rustdoc cannot run them for
#      another target on stable Rust.
#   2. It copies the programs to $DEVICE_DIR/tests/bin and fixtures/ to
#      $DEVICE_DIR/fixtures, the path the programs were built to read (see
#      rustc-wrapper.sh).
#   3. On the phone, a runner starts each program from $DEVICE_DIR with
#      RUST_MIN_STACK=33554432 (32 MiB, as in the repository's local cargo
#      settings; tests of the no_std build keep the codec state on the
#      stack). The runner keeps going when adb disconnects. It writes one
#      log per program.
#   4. It copies the logs back to a new directory
#      $ANDROID_OUT/test-logs/<date>-<time>/ and prints a summary. So a
#      later run, even of one program, does not delete the logs of an
#      earlier one. $ANDROID_OUT/test-logs/latest points at the newest
#      directory. The exit status is 1 when a program failed.
#
# Usage:
#   scripts/android/test.sh [--no-build | --build-only] [--collect] [NAME_PART ...] [-- TEST_ARGS ...]
#
#   NAME_PART  Run only the programs whose name contains one of the parts
#              (for example `core_ perfk_fft`). `opus1` is the unit tests.
#   TEST_ARGS  Passed to every test program, for example `-- --ignored` or
#              `-- --test-threads=1`.
#   --no-build Use the programs of the last build.
#   --build-only  Build the programs and stop, without the phone. A later
#              run with --no-build runs them. On a shared phone this keeps
#              the build out of the time the phone is taken.
#   --collect  Do not start anything. Wait for the run on the phone to end,
#              then copy the logs and print the summary. Use it when this
#              script was stopped while the phone kept running the tests.
#
# Settings: see scripts/android/env.sh (ANDROID_NDK_HOME, ADB, DEVICE_DIR,
# ANDROID_OUT, ...).
# CARGO_FEATURES: cargo features for the build, for example
#   CARGO_FEATURES=test-libopus-dnn,osce-bwe with LIBOPUS_PREFIX set to a
#   libopus built with --dnn (docs/09-testing.md).
set -euo pipefail
source "$(dirname "$0")/env.sh"

BUILD=1
RUN=1
BUILD_ONLY=0
PARTS=()
TEST_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --build-only) BUILD_ONLY=1 ;;
        --collect) BUILD=0; RUN=0 ;;
        --) shift; TEST_ARGS=("$@"); break ;;
        *) PARTS+=("$1") ;;
    esac
    shift
done

STAGE="$ANDROID_OUT/tests"
LIST="$STAGE/programs.txt"
if [ "$BUILD" = 1 ]; then
    echo ":: building the tests for $RUST_TARGET (API $ANDROID_API)"
    rm -rf "$STAGE"
    mkdir -p "$STAGE/bin"
    cargo_android test --release --no-run ${CARGO_FEATURES:+--features "$CARGO_FEATURES"}
    # cargo ndk does not print where the programs are. Each test program is
    # deps/<name>-<hash>, where <name> is `opus1` for the unit tests or the
    # name of a file of tests/. The newest such file is the one just built.
    DEPS="$ANDROID_CARGO_TARGET_DIR/$RUST_TARGET/release/deps"
    : > "$LIST"
    for name in opus1 $(cd "$REPO_ROOT/tests" && ls *.rs | sed 's/\.rs$//'); do
        exe=$(ls -t "$DEPS/$name"-* 2>/dev/null | grep -E "/$name-[0-9a-f]{16}\$" | head -n 1 || true)
        if [ -z "$exe" ] || [ ! -x "$exe" ]; then
            echo "error: no test program for $name in $DEPS" >&2
            exit 1
        fi
        cp "$exe" "$STAGE/bin/$name"
        echo "$name" >> "$LIST"
    done
    echo "   $(wc -l < "$LIST" | tr -d ' ') test programs"
fi
if [ "$BUILD_ONLY" = 1 ]; then
    exit 0
fi

need_device
if [ "$RUN" = 1 ]; then
    if [ ! -s "$LIST" ]; then
        echo "error: no test programs in $STAGE; run without --no-build" >&2
        exit 2
    fi
    SELECTED="$STAGE/selected.txt"
    if [ ${#PARTS[@]} -eq 0 ]; then
        cp "$LIST" "$SELECTED"
    else
        : > "$SELECTED"
        while read -r name; do
            for part in "${PARTS[@]}"; do
                case "$name" in *"$part"*) echo "$name" >> "$SELECTED"; break ;; esac
            done
        done < "$LIST"
    fi
    if [ ! -s "$SELECTED" ]; then
        echo "error: no test program matches: ${PARTS[*]}" >&2
        exit 2
    fi

    echo ":: copying $(wc -l < "$SELECTED" | tr -d ' ') programs, fixtures/, dnn-weights/ and examples/eval/sets/ to $DEVICE_DIR"
    dev "mkdir -p $DEVICE_DIR/tests/bin $DEVICE_DIR/fixtures $DEVICE_DIR/dnn-weights $DEVICE_DIR/examples/eval/sets"
    "$ADB" push -q --sync "$REPO_ROOT/fixtures/." "$DEVICE_DIR/fixtures/" >/dev/null
    # Tests read the weight files (core_decoder_dnn_controls) and the item
    # lists of the evaluation sets (eval_selftest) from the repository.
    "$ADB" push -q --sync "$REPO_ROOT/dnn-weights/." "$DEVICE_DIR/dnn-weights/" >/dev/null
    "$ADB" push -q --sync "$REPO_ROOT/examples/eval/sets/." "$DEVICE_DIR/examples/eval/sets/" >/dev/null
    "$ADB" push -q --sync "$STAGE/bin/." "$DEVICE_DIR/tests/bin/" >/dev/null
    "$ADB" push -q "$SELECTED" "$DEVICE_DIR/tests/selected.txt" >/dev/null

    # The runner. It runs under the phone's /system/bin/sh, from DEVICE_DIR.
    args=""
    for a in "${TEST_ARGS[@]+"${TEST_ARGS[@]}"}"; do args="$args '$a'"; done
    cat > "$STAGE/run-tests.sh" <<EOF
rm -rf tests/logs; mkdir -p tests/logs
: > tests/summary.txt
for name in \$(cat tests/selected.txt); do
    chmod 755 "tests/bin/\$name"
    echo "running \$name"
    start=\$(date +%s)
    RUST_MIN_STACK=33554432 "./tests/bin/\$name"$args > "tests/logs/\$name.log" 2>&1
    status=\$?
    echo "\$name \$status \$((\$(date +%s) - start))" >> tests/summary.txt
done
EOF
    echo ":: running on the phone"
    device_state
    dev_start tests "$STAGE/run-tests.sh"
fi

total=$(dev "cat $DEVICE_DIR/tests/selected.txt | wc -l" | tr -d '\r ')
POLL=${POLL:-30} dev_wait tests || true
device_state

LOGS="$ANDROID_OUT/test-logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGS"
"$ADB" pull -q "$DEVICE_DIR/tests/logs/." "$LOGS/" >/dev/null
"$ADB" pull -q "$DEVICE_DIR/tests/summary.txt" "$LOGS/summary.txt" >/dev/null
ln -sfn "$(basename "$LOGS")" "$ANDROID_OUT/test-logs/latest"

# Summary: one line per program, then totals. The counts come from the
# "test result:" line of each log.
passed=0; failed=0; ignored=0; bad_programs=0; ran=0
printf '%-48s %6s %6s %7s %6s %s\n' program passed failed ignored secs status
while read -r name status secs; do
    ran=$((ran + 1))
    log="$LOGS/$name.log"
    p=0; f=0; i=0
    while read -r line; do
        set -- $line
        p=$((p + $4)); f=$((f + $6)); i=$((i + $8))
    done < <(grep -E '^test result: ' "$log" | sed 's/[;.]//g')
    passed=$((passed + p)); failed=$((failed + f)); ignored=$((ignored + i))
    state=ok
    if [ "$status" != 0 ]; then
        state="FAILED (exit $status)"
        bad_programs=$((bad_programs + 1))
    fi
    printf '%-48s %6d %6d %7d %6d %s\n' "$name" "$p" "$f" "$i" "$secs" "$state"
done < "$LOGS/summary.txt"
echo
echo "total: $passed passed, $failed failed, $ignored ignored; $bad_programs of $ran programs failed ($total selected)"
echo "logs: $LOGS/"
[ "$bad_programs" = 0 ] && [ "$ran" = "$total" ]
