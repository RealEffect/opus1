#!/usr/bin/env bash
# Tests opus1 on a 64-bit x86 computer running Linux (also inside WSL2 on
# Windows), against libopus 1.6.1 built on that computer, and packs the
# results into one file to send back.
#
# What it does:
#   1. Records the CPU, its vector instruction sets (SSE2, AVX, AVX2, FMA,
#      AVX-512, AVX-VNNI), the operating system and the tool versions.
#   2. Downloads the libopus 1.6.1 release (or takes OPUS_TARBALL), checks
#      its SHA-256 sum and builds it twice with its own configure script:
#      the default build, and a build with deep PLC, OSCE and DRED. Both are
#      static libraries at -O2, with libopus's run-time CPU detection.
#   3. Runs the whole test suite linked with the default build.
#   4. Runs the neural network tests that compare every x86 kernel variant
#      the CPU has (AVX2+FMA, AVX-VNNI, AVX-512 VNNI) with the portable
#      code, and then the tests that compare with the libopus built with
#      its neural network code.
#   5. Checks the golden file of 64-bit ARM (x86 is expected to differ in
#      some streams; the number is recorded).
#   6. Times encoding and decoding against the default libopus build
#      (skipped with --quick).
#   7. Writes x86-results-<date>.tar.gz in the repository root.
#
# Usage, from the repository root:
#   scripts/x86/test.sh [--quick] [--no-dnn]
#
# Settings (environment variables):
#   OPUS_TARBALL  A local copy of opus-1.6.1.tar.gz. Default: download it
#                 from downloads.xiph.org.
#   OPUS1_CORPUS  A corpus folder with rfc8251/ (the RFC 8251 test vectors)
#                 for the conformance tests. Without it they skip.
#   JOBS          Parallel jobs for make. Default: the number of CPUs.
#
# Needs: a C compiler (gcc or clang), make, curl, and Rust (rustup's stable
# toolchain). On Ubuntu or Debian:
#   sudo apt-get install -y build-essential curl
#   curl https://sh.rustup.rs -sSf | sh
set -uo pipefail

OPUS_URL=https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz
OPUS_SHA256=6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1

QUICK=0
DNN=1
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --no-dnn) DNN=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$REPO/x86-results-$STAMP"
WORK="$REPO/target/x86"
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
mkdir -p "$OUT" "$WORK"
exec > >(tee "$OUT/run.log") 2>&1

step() { echo; echo ":: $*"; }
die() { echo "error: $*" >&2; exit 1; }

step "system"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -a)"
    grep -qi microsoft /proc/version 2>/dev/null && echo "wsl: yes" || echo "wsl: no"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "cpus: $JOBS"
    flags=$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2-)
    for f in sse2 ssse3 sse4_1 sse4_2 avx avx2 fma f16c avx512f avx512bw avx512vl avx512_vnni avx_vnni; do
        case " $flags " in *" $f "*) echo "flag $f: yes" ;; *) echo "flag $f: no" ;; esac
    done
    echo "rustc: $(rustc -V 2>/dev/null || echo missing)"
    echo "cargo: $(cargo -V 2>/dev/null || echo missing)"
    echo "cc: $(cc --version 2>/dev/null | head -n 1 || echo missing)"
    echo "git commit: $(cat "$REPO/COMMIT" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} | tee "$OUT/system.txt"

[ "$(uname -m)" = x86_64 ] || die "this is not a 64-bit x86 computer ($(uname -m))"
for tool in cc make cargo; do
    command -v "$tool" >/dev/null || die "$tool is missing; see the header of this script"
done
# opus1 needs Rust 1.98 or newer (rust-version in Cargo.toml). Check it now,
# before minutes are spent on libopus.
RUSTC_VERSION=$(rustc -V | awk '{print $2}')
if [ "$(printf '%s\n1.98.0\n' "$RUSTC_VERSION" | sort -V | head -n 1)" != 1.98.0 ]; then
    die "rustc $RUSTC_VERSION is older than 1.98; run: rustup update stable"
fi

step "libopus 1.6.1 source"
TARBALL=${OPUS_TARBALL:-$WORK/opus-1.6.1.tar.gz}
if [ ! -f "$TARBALL" ]; then
    command -v curl >/dev/null || die "curl is missing and OPUS_TARBALL is not set"
    curl -fL -o "$TARBALL" "$OPUS_URL" || die "download of $OPUS_URL failed"
fi
echo "$OPUS_SHA256  $TARBALL" | sha256sum -c - || die "SHA-256 of $TARBALL does not match"

# build_libopus NAME CONFIGURE_OPTIONS...: builds into $WORK/NAME (once).
build_libopus() {
    local name=$1; shift
    local prefix="$WORK/$name"
    if [ -f "$prefix/lib/libopus.a" ]; then
        echo "$name: already built"
        return
    fi
    local src="$WORK/src-$name"
    rm -rf "$src" && mkdir -p "$src"
    tar xzf "$TARBALL" -C "$src" --strip-components=1
    (cd "$src" && CFLAGS="-O2" ./configure --prefix="$prefix" --disable-shared --enable-static \
        --disable-doc --disable-extra-programs "$@" > "$OUT/configure-$name.log" 2>&1 \
        && make -j"$JOBS" > "$OUT/make-$name.log" 2>&1 && make install > /dev/null 2>&1) \
        || die "building libopus ($name) failed; see $OUT/configure-$name.log and make-$name.log"
    grep -E '^#define (OPUS_HAVE_RTCD|OPUS_X86_|ENABLE_)' "$src/config.h" > "$OUT/config-$name.txt" || true
    echo "$name: built in $prefix"
}

step "building libopus (default)"
build_libopus libopus-default
if [ "$DNN" = 1 ]; then
    step "building libopus with deep PLC, OSCE and DRED (takes a few minutes)"
    build_libopus libopus-dnn --enable-deep-plc --enable-osce --enable-dred
fi

export RUST_MIN_STACK=33554432 LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1

# The RFC 8251 test vectors (74.6 MB), for the conformance tests and the
# quality checks below. Skipped with --quick or when OPUS1_CORPUS is set.
VECTORS_URL=https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz
VECTORS_SHA256=6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55
if [ -z "${OPUS1_CORPUS:-}" ] && [ "$QUICK" = 0 ]; then
    step "RFC 8251 test vectors"
    mkdir -p "$WORK/corpus/rfc8251"
    VT="$WORK/opus_testvectors-rfc8251.tar.gz"
    [ -f "$VT" ] || curl -fL -o "$VT" "$VECTORS_URL" || echo "download of the test vectors failed; conformance tests will skip"
    if [ -f "$VT" ] && echo "$VECTORS_SHA256  $VT" | sha256sum -c - >/dev/null; then
        [ -d "$WORK/corpus/rfc8251/opus_newvectors" ] || tar xzf "$VT" -C "$WORK/corpus/rfc8251"
        export OPUS1_CORPUS="$WORK/corpus"
        echo "test vectors in $OPUS1_CORPUS/rfc8251"
    else
        echo "test vectors missing or with a wrong SHA-256; conformance tests will skip"
    fi
fi

# run_tests NAME LIBOPUS_PREFIX CARGO_ARGS...: one cargo test run, with a
# summary of each test program in NAME-summary.txt.
run_tests() {
    local name=$1 prefix=$2; shift 2
    echo "cargo test $*" > "$OUT/$name.txt"
    LIBOPUS_LIB_DIR="$prefix" CARGO_TARGET_DIR="$REPO/target/x86-$name" \
        cargo test --release --no-fail-fast "$@" >> "$OUT/$name.txt" 2>&1
    echo "exit status: $?" >> "$OUT/$name.txt"
    awk '/Running /{prog=$2} /^test result:/{print prog": "$0}' "$OUT/$name.txt" > "$OUT/$name-summary.txt"
    grep -E '^test .* FAILED$' "$OUT/$name.txt" | sort -u > "$OUT/$name-failed.txt" || true
    if grep -qE '^error(\[E[0-9]+\])?: (could not compile|failed to run custom build command)|^error\[E[0-9]+\]' "$OUT/$name.txt"; then
        echo "$name: compile failed (see $name.txt)" | tee -a "$OUT/summary.txt"
        return
    fi
    awk -v n="$name" '/^test result:/{p+=$4; f+=$6; i+=$8} END {printf "%s: passed %d, failed %d, ignored %d\n", n, p, f, i}' "$OUT/$name.txt" \
        | tee -a "$OUT/summary.txt"
}

step "test suite, default features (several minutes)"
echo "default features:" >> "$OUT/summary.txt"
run_tests default "$WORK/libopus-default"

step "neural network kernels: every x86 variant against the portable code"
echo "neural network kernels:" >> "$OUT/summary.txt"
run_tests dnn-kernels "$WORK/libopus-default" --features osce-bwe,dred --lib --test dnn_edges --test dnn_props --test dnn_weights

if [ "$DNN" = 1 ]; then
    step "test suite with the neural network features, against libopus built with them"
    echo "features test-libopus-dnn,osce-bwe,dred:" >> "$OUT/summary.txt"
    run_tests dnn "$WORK/libopus-dnn" --features test-libopus-dnn,osce-bwe,dred
fi

step "quality on x86: RFC 8251 conformance and the paired comparison with libopus"
# On x86 opus1 does not aim at libopus's bits; these check that the audio is
# right: the conformance limits of opus_compare, and opus1's coding error
# against libopus's on the same speech and music (examples/eval gates).
LIBOPUS_LIB_DIR="$WORK/libopus-default" CARGO_TARGET_DIR="$REPO/target/x86-default" \
    cargo run --release --example eval -- conformance > "$OUT/conformance.txt" 2>&1
echo "conformance: exit status $? ($(grep -cE 'pass|PASS' "$OUT/conformance.txt") lines with pass)" | tee -a "$OUT/summary.txt"
LIBOPUS_LIB_DIR="$WORK/libopus-default" CARGO_TARGET_DIR="$REPO/target/x86-default" \
    cargo run --release --example eval -- run --set quick --text "$OUT/quality-quick.txt" --json "$OUT/quality-quick.json" \
    > "$OUT/quality-quick.log" 2>&1
echo "quality (quick set): $(grep -E '^gates' "$OUT/quality-quick.txt" 2>/dev/null || echo 'no report, see quality-quick.log')" | tee -a "$OUT/summary.txt"

step "golden file of 64-bit ARM"
if [ -f "$REPO/scripts/x86/golden_r10.txt" ]; then
    LIBOPUS_LIB_DIR="$WORK/libopus-default" CARGO_TARGET_DIR="$REPO/target/x86-default" \
        cargo run --release --example perf_harness -- golden --check "$REPO/scripts/x86/golden_r10.txt" \
        > "$OUT/golden.txt" 2>&1
    tail -n 1 "$OUT/golden.txt" | tee -a "$OUT/summary.txt"
else
    echo "no scripts/x86/golden_r10.txt in this copy; skipped" | tee -a "$OUT/summary.txt"
fi

if [ "$QUICK" = 0 ]; then
    step "timing against libopus (default build), a few minutes; keep the computer idle"
    LIBOPUS_LIB_DIR="$WORK/libopus-default" CARGO_TARGET_DIR="$REPO/target/x86-default" \
        cargo run --release --example perf_harness -- bench --md "$OUT/bench.md" --json "$OUT/bench.json" \
        > "$OUT/bench.log" 2>&1
    echo "bench: exit status $?" | tee -a "$OUT/summary.txt"
fi

step "packing the results"
tar czf "$REPO/x86-results-$STAMP.tar.gz" -C "$REPO" "x86-results-$STAMP"
echo
cat "$OUT/summary.txt"
echo
echo "Send this file back: $REPO/x86-results-$STAMP.tar.gz"
