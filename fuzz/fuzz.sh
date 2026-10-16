#!/bin/bash
# Build and run the fuzz targets on the stable Rust toolchain.
#
# Usage:
#   fuzz/fuzz.sh build                       build every target
#   fuzz/fuzz.sh list                        list the targets
#   fuzz/fuzz.sh <target> [seconds] [args]   build, then fuzz one target
#   fuzz/fuzz.sh all [seconds] [args]        build, then fuzz every target,
#                                            one after the other
#   fuzz/fuzz.sh repro <target> <file>       run one input, for example a
#                                            crash file
#   fuzz/fuzz.sh minimize <target> <file>    shrink a crash input
#
# <target> runs for [seconds] (default 60). The corpus is kept in
# fuzz/corpus/<target>/ and crash inputs go to fuzz/artifacts/<target>/.
# [args] are passed to libFuzzer, for example -fork=4 to use 4 processes.
#
# Each target is a libFuzzer program built by cargo. The build passes the
# coverage options that cargo-fuzz passes when it builds without a
# sanitizer, so no nightly toolchain and no cargo-fuzz are needed.
# Debug assertions and overflow checks are on: an integer overflow in
# opus1 stops the run like a panic.
#
# libopus is linked as for the main crate's tests (docs/09-testing.md):
# set LIBOPUS_LIB_DIR and LIBOPUS_NO_PKG, or keep them in .cargo/config.toml.

set -euo pipefail

FUZZ_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$FUZZ_DIR/target}"
BIN_DIR="$CARGO_TARGET_DIR/$HOST/release"

# The options of cargo-fuzz for a build without a sanitizer. `--target`
# keeps them away from build scripts, which must not be instrumented.
RUSTFLAGS="-Cpasses=sancov-module \
-Cllvm-args=-sanitizer-coverage-level=4 \
-Cllvm-args=-sanitizer-coverage-inline-8bit-counters \
-Cllvm-args=-sanitizer-coverage-pc-table \
-Cllvm-args=-sanitizer-coverage-trace-compares \
--cfg fuzzing -Cdebug-assertions -Coverflow-checks"
export RUSTFLAGS

targets() {
    sed -n 's/^name = "\(fuzz_[a-z0-9_]*\)"$/\1/p' "$FUZZ_DIR/Cargo.toml"
}

build() {
    # cargo reads .cargo/config.toml from the directory it runs in, so it
    # runs in fuzz/. Then the settings of the repository apply wherever
    # this script is called from.
    (cd "$FUZZ_DIR" && cargo build --release --target "$HOST" --manifest-path "$FUZZ_DIR/Cargo.toml" --bins)
}

run() {
    local target="$1" seconds="$2"
    shift 2
    mkdir -p "$FUZZ_DIR/corpus/$target" "$FUZZ_DIR/artifacts/$target"
    echo "== $target for ${seconds}s"
    "$BIN_DIR/$target" "$FUZZ_DIR/corpus/$target" \
        -artifact_prefix="$FUZZ_DIR/artifacts/$target/" \
        -max_total_time="$seconds" "$@"
}

case "${1:-}" in
    "" | -h | --help)
        sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    build)
        build
        ;;
    list)
        targets
        ;;
    repro)
        build
        "$BIN_DIR/$2" "$3"
        ;;
    minimize)
        build
        mkdir -p "$FUZZ_DIR/artifacts/$2"
        "$BIN_DIR/$2" -minimize_crash=1 -runs=100000 \
            -exact_artifact_path="$3.min" "$3"
        echo "minimized input: $3.min"
        ;;
    all)
        seconds="${2:-60}"
        shift $(($# < 2 ? $# : 2))
        build
        for t in $(targets); do
            run "$t" "$seconds" "$@"
        done
        ;;
    *)
        target="$1"
        seconds="${2:-60}"
        shift $(($# < 2 ? $# : 2))
        if ! targets | grep -qx "$target"; then
            echo "unknown target: $target (see: fuzz/fuzz.sh list)" >&2
            exit 2
        fi
        build
        run "$target" "$seconds" "$@"
        ;;
esac
