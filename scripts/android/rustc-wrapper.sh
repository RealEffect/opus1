#!/bin/sh
# Cargo runs this program in place of rustc when RUSTC_WRAPPER names it
# (scripts/android/env.sh does that for the phone builds). Its arguments
# are the rustc program and rustc's arguments.
#
# The tests and examples/perf_harness.rs open fixtures/answer_16k.wav
# through env!("CARGO_MANIFEST_DIR"), which is fixed when they are
# compiled. On the phone the checkout is not at the same path. So while
# cargo compiles the package opus1, this wrapper sets CARGO_MANIFEST_DIR to
# OPUS1_DEVICE_DIR, the directory on the phone that holds fixtures/.
# Other packages are compiled unchanged.
if [ "${CARGO_PKG_NAME:-}" = opus1 ] && [ -n "${OPUS1_DEVICE_DIR:-}" ]; then
    CARGO_MANIFEST_DIR=$OPUS1_DEVICE_DIR
    export CARGO_MANIFEST_DIR
fi
exec "$@"
