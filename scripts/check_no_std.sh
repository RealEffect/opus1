#!/usr/bin/env bash
# Verify opus1 builds as `#![no_std]` across bare-metal / freestanding targets.
#
# Requires the targets to be installed, e.g.:
#   rustup target add thumbv7em-none-eabi wasm32-unknown-unknown
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETS=(
    "thumbv7em-none-eabi"      # ARM Cortex-M (bare metal)
    "wasm32-unknown-unknown"   # WebAssembly (no WASI)
)

# The feature sets to check: the codec alone, and with the speech and
# music analysis of the encoder (feature `analysis`, on by default with
# `std`).
FEATURE_SETS=("libm" "libm,analysis")

fail=0
for features in "${FEATURE_SETS[@]}"; do
    for t in "${TARGETS[@]}"; do
        if rustup target list --installed 2>/dev/null | grep -q "$t"; then
            echo ":: building opus1 (no_std, features $features) for $t"
            if ! cargo build --no-default-features --features "$features" --target "$t"; then
                echo "!! FAILED: $t ($features)"
                fail=1
            fi
        else
            echo "-- skip $t (not installed; run: rustup target add $t)"
        fi
    done

    # Host no_std build (rlib, no panic handler needed for a library).
    echo ":: building opus1 (no_std, features $features) for host"
    cargo build --no-default-features --features "$features" || fail=1

    # Host no_std test run. Without the `heap` feature the codec structs are
    # large inline objects (about 135 KB encoder, 67 KB decoder, plus about
    # 26 KB for the analysis), so the test threads need a big stack.
    echo ":: testing opus1 (no_std, features $features) for host"
    RUST_MIN_STACK=${RUST_MIN_STACK:-33554432} cargo test --no-default-features --features "$features" || fail=1
done

# A target without compare-and-swap (CAS) instructions: ARM Cortex-M0/M0+.
# It needs the feature `portable-atomic`, and portable-atomic needs one
# setting from the application; this build assumes a single-core chip.
t="thumbv6m-none-eabi"
if rustup target list --installed 2>/dev/null | grep -q "$t"; then
    for features in "libm,portable-atomic" "libm,portable-atomic,analysis"; do
        echo ":: building opus1 (no_std, features $features) for $t"
        if ! RUSTFLAGS="--cfg portable_atomic_unsafe_assume_single_core" \
            cargo build --no-default-features --features "$features" --target "$t"; then
            echo "!! FAILED: $t ($features)"
            fail=1
        fi
    done
else
    echo "-- skip $t (not installed; run: rustup target add $t)"
fi

exit $fail
