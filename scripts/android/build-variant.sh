#!/usr/bin/env bash
# Builds examples/perf_harness.rs for the phone as one named variant. A
# variant is a build with the compiler settings of the current environment,
# in a cargo directory of its own. ab.sh runs several variants in turns.
#
# The settings come from the environment. Examples:
#   RUSTFLAGS="-C target-cpu=cortex-a55"
#   CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
#   CARGO_PROFILE_RELEASE_PANIC=abort
#   PERF_HARNESS_PAD=16   bytes of padding before the library code, to see
#                         how much a timing depends on code placement
#
# To compare a code change, build the base in one checkout and the change
# in another, with the same name length for DEVICE_DIR. Strings such as the
# fixture path lie before the code in the program, so a longer string
# moves the code.
#
# Usage: scripts/android/build-variant.sh NAME
#
# Results in $ANDROID_OUT/bin:
#   perf_harness_NAME      the program
#   perf_harness_NAME.txt  the settings, the git commit, and the address
#                          and size of the code section (.text)
#
# Settings: see scripts/android/env.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ $# != 1 ]; then
    echo "usage: $0 NAME" >&2
    exit 2
fi
NAME=$1
case "$NAME" in
    *[!A-Za-z0-9_-]*) echo "error: NAME may only contain letters, digits, '_' and '-'" >&2; exit 2 ;;
esac

ANDROID_CARGO_TARGET_DIR=$ANDROID_OUT/variants/$NAME
echo ":: building variant $NAME for $RUST_TARGET (API $ANDROID_API)"
cargo_android build --release --example perf_harness
mkdir -p "$ANDROID_OUT/bin"
out=$ANDROID_OUT/bin/perf_harness_$NAME
cp "$ANDROID_CARGO_TARGET_DIR/$RUST_TARGET/release/examples/perf_harness" "$out"
{
    echo "variant: $NAME"
    echo "git: $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet HEAD -- src examples Cargo.toml || echo +changes)"
    echo "RUSTFLAGS=${RUSTFLAGS:-}"
    env | grep '^CARGO_PROFILE_RELEASE_' | sort || true
    echo "PERF_HARNESS_PAD=${PERF_HARNESS_PAD:-}"
    echo "DEVICE_DIR=$DEVICE_DIR"
    echo "sections (name, address and size in hexadecimal, alignment):"
    "$(ndk_bin)/llvm-readelf" -S -W "$out" | awk '{
        for (i = 1; i <= NF; i++)
            if ($i == ".text" || $i == ".rodata") print "  " $i, $(i + 2), $(i + 4), $NF
    }'
} > "$out.txt"
cat "$out.txt"
