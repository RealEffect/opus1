#!/usr/bin/env bash
# Builds libopus for Android on 64-bit ARM (arm64-v8a) as a static library,
# with the NDK's clang. The tests and examples/perf_harness.rs link it when
# they are built for a phone (scripts/android/test.sh, bench.sh, golden.sh,
# profile.sh).
#
# The build follows what libopus's own `configure` chooses for the host
# with clang and its default options. This was checked against the
# config.h and the Makefile that `configure` of the release 1.6.1 writes
# for aarch64-linux-android with the NDK's clang and for a Mac with Apple
# silicon. It is a floating-point build, like the Homebrew libopus that the
# tests link on macOS:
#
#   - Compiler flags: -g -O2 (the autoconf default), -fvisibility=hidden,
#     -D_FORTIFY_SOURCE=2 (configure adds it when the compiler does not set
#     it) and the warning flags of configure.
#   - FLOAT_APPROX: configure turns it on for aarch64.
#   - VAR_ARRAYS: the compiler supports C99 variable-size arrays.
#   - ENABLE_HARDENING: on by default.
#   - ENABLE_RES24: on by default. It sets the internal resolution of the
#     fixed-point build. In this floating-point build it only chooses
#     between two versions of the same float code in a few places.
#   - DISABLE_DEBUG_FLOAT: on by default. Only the neural network code,
#     which is not part of this build, reads it.
#   - configure also defines `restrict` as `__restrict`, which is the same
#     keyword for clang. The script leaves it out.
#   - NEON intrinsics: clang compiles them without extra flags, so configure
#     sets OPUS_ARM_MAY_HAVE_NEON_INTR and OPUS_ARM_PRESUME_NEON_INTR, and
#     OPUS_ARM_PRESUME_AARCH64_NEON_INTR for the AArch64-only intrinsics.
#     The sources CELT_SOURCES_ARM_NEON_INTR and SILK_SOURCES_ARM_NEON_INTR
#     are compiled.
#   - DOTPROD intrinsics: the script tries to compile vdotq_s32 without
#     extra flags, as configure does.
#       - The NDK's clang needs -march=armv8.2-a+dotprod for them, so
#         configure sets only OPUS_ARM_MAY_HAVE_DOTPROD. That turns on
#         run-time CPU detection (OPUS_HAVE_RTCD) and adds the sources
#         CELT_SOURCES_ARM_RTCD and SILK_SOURCES_ARM_RTCD.
#       - Apple clang compiles them for every Mac with Apple silicon, so
#         configure also sets OPUS_ARM_PRESUME_DOTPROD, and there is no
#         run-time detection.
#     Only the neural network code uses DOTPROD, and it is not part of
#     this build. But the choice changes which delayed-decision quantizer
#     the SILK encoder calls (silk/arm/NSQ_del_dec_arm.h). The NEON version
#     handles 3 and 4 states, which SILK keeps from complexity 6 on:
#       - With run-time detection, it calls the NEON version
#         silk_NSQ_del_dec_neon. This is the case on the phone.
#       - Without it, the header asks for OPUS_ARM_PRESUME_NEON, which
#         configure sets only for the assembly of 32-bit ARM. So it calls
#         the C version silk_NSQ_del_dec_c. This is the case on a Mac,
#         also in the Homebrew library.
#     --rtcd turns on run-time detection also where the compiler takes
#     DOTPROD, so a Mac build calls the NEON quantizer as the phone does.
#   - Deep PLC, DRED, OSCE and QEXT are off (their defaults).
#
# The exact command lines are written to $LIBOPUS_PREFIX/build-flags.txt.
#
# Which libopus. Name the source tree with --src DIR (or OPUS_SRC). Use
# the release 1.6.1, which the tests were written against and which
# Homebrew ships:
#
#   curl -LO https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz
#   shasum -a 256 opus-1.6.1.tar.gz   # compare with Xiph's SHA256SUMS.txt:
#   # 6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1
#   tar xzf opus-1.6.1.tar.gz
#   scripts/android/build-libopus.sh --src opus-1.6.1
#
# Without --src and OPUS_SRC, the script falls back to the copy that the
# crate opus-head-sys 0.3.3 vendors, which is in the cargo registry after
# any build of the tests. Its file vendored/OPUS_VERSION names commit
# 228a0f85 of the xiph/opus main branch (2026-08-28). That is newer than
# 1.6.1, and it writes other packets than 1.6.1 for many SILK, Hybrid and
# LFE streams. The script prints a warning when it uses it.
#
# The version string that opus_get_version_string() reports says which
# source was built: the PACKAGE_VERSION of the release's package_version
# file ("1.6.1"), the commit of the vendored OPUS_VERSION file
# ("git-228a0f855ce2"), or OPUS_VERSION when it is set. The harness writes
# it into its records and into the header of a golden file.
#
# Notes on comparing against the Homebrew library at the bit level,
# checked with the release 1.6.1 (docs/09-testing.md, "Which libopus
# source to build" and "Results on the phone"):
#   - The Homebrew library behaves like a build at -Os. Built at -Os on a
#     Mac, the release has nearly the same machine code size (231,660 bytes
#     against Homebrew's 238,144), and 284 of the 396 functions found in
#     both have exactly the size of Homebrew's. At -O2 the code has 353,628
#     bytes, and 138 functions have Homebrew's size.
#   - At -O2, clang turns some loops into vector code: in stereo_itheta
#     (celt/vq.c), in pitch_downsample and pitch_search (celt/pitch.c), in
#     celt/celt_lpc.c and in the loss concealment of celt/celt_decoder.c.
#     The vector code rounds some products before it adds them, where the
#     scalar code uses a fused multiply-add, so the last bit can differ.
#     7 tests of opus1 that require libopus's exact results fail against
#     the -O2 build of 1.6.1 on a Mac. They pass against the -Os build and
#     against a build with
#     OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize".
#     The golden file of the harness does not see these differences.
#   - The library calls the C library's acos, cos, exp, exp2, log and
#     log10. Android's and Apple's give different last bits for some
#     inputs.
#
# For this computer (--host). The same build can be made for a 64-bit ARM
# computer, such as a Mac with Apple silicon, with its own compiler `cc` and
# archiver `ar`. It then serves as a fair C reference on that computer: the
# Homebrew libopus behaves like a build at -Os, while libopus's own build
# uses -O2 (docs/08-performance.md, "The C reference"). The library goes
# to $ANDROID_OUT/libopus-host unless LIBOPUS_PREFIX is set. Link the
# harness against it in a cargo directory of its own:
#
#   LIBOPUS_LIB_DIR=<prefix> LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
#     CARGO_TARGET_DIR=target/libopus-o2 \
#     cargo build --release --example perf_harness
#
# OPUS_EXTRA_CFLAGS=-Os gives a build at -Os, because the last -O flag
# counts.
#
# Settings (environment variables; see scripts/android/env.sh for the
# common ones):
#   ANDROID_NDK_HOME  required, except with --host.
#   OPUS_SRC          The libopus source tree, as --src. Default: the copy
#                     in the cargo registry, vendored by the crate
#                     opus-head-sys 0.3.3 (see above).
#   OPUS_VERSION      The version that opus_get_version_string() reports.
#                     Default: PACKAGE_VERSION from the package_version
#                     file of a release tree, git-<commit> from the
#                     OPUS_VERSION file next to the vendored tree, or
#                     "unknown".
#   LIBOPUS_PREFIX    Install directory (lib/libopus.a, include/opus/*.h).
#   JOBS              Parallel compiler processes. Default 8.
#   OPUS_CC           The compiler command. Default: the NDK's
#                     aarch64-linux-android$ANDROID_API-clang, or cc with
#                     --host.
#   OPUS_EXTRA_CFLAGS More compiler flags, added after the ones above. For
#                     example "-fno-vectorize -fno-slp-vectorize" builds
#                     a library whose loops are not vectorized by clang.
#
# Usage: scripts/android/build-libopus.sh [--host] [--rtcd] [--src DIR]
set -euo pipefail
PREFIX_FROM_USER=${LIBOPUS_PREFIX:-}
source "$(dirname "$0")/env.sh"

HOST=0
FORCE_RTCD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST=1 ;;
        --rtcd) FORCE_RTCD=1 ;;
        --src)
            [ $# -ge 2 ] || { echo "error: --src needs a directory" >&2; exit 2; }
            OPUS_SRC=$2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$HOST" = 1 ]; then
    # The defines below choose the NEON code of 64-bit ARM.
    case "$(uname -m)" in
        arm64 | aarch64) ;;
        *) echo "error: --host needs a 64-bit ARM computer; this one is $(uname -m)" >&2; exit 2 ;;
    esac
    LIBOPUS_PREFIX=${PREFIX_FROM_USER:-$ANDROID_OUT/libopus-host}
else
    need_ndk
fi

if [ -z "${OPUS_SRC:-}" ]; then
    for d in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/opus-head-sys-0.3.3/vendored/opus; do
        [ -f "$d/celt_sources.mk" ] && OPUS_SRC=$d
    done
    if [ -n "${OPUS_SRC:-}" ]; then
        echo "warning: no --src given; building the vendored libopus main branch, not the release 1.6.1" >&2
    fi
fi
if [ -z "${OPUS_SRC:-}" ] || [ ! -f "$OPUS_SRC/celt_sources.mk" ]; then
    echo "error: libopus source not found; pass --src with a libopus 1.6 source tree" >&2
    exit 2
fi
OPUS_SRC=$(cd "$OPUS_SRC" && pwd)
if [ -z "${OPUS_VERSION:-}" ]; then
    if [ -f "$OPUS_SRC/package_version" ]; then
        # A release tree: the file holds PACKAGE_VERSION="1.6.1".
        OPUS_VERSION=$(sed -n 's/^PACKAGE_VERSION="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$OPUS_SRC/package_version")
    else
        commit=$(sed -n 's/^commit: *\([0-9a-f]\{12\}\).*/\1/p' "$OPUS_SRC/../OPUS_VERSION" 2>/dev/null || true)
        OPUS_VERSION=${commit:+git-$commit}
    fi
    OPUS_VERSION=${OPUS_VERSION:-unknown}
fi
JOBS=${JOBS:-8}

# The compiler command, as words. OPUS_CC replaces it, for example to try
# another clang: "clang -target aarch64-linux-android21 --sysroot=$SYSROOT".
if [ "$HOST" = 1 ]; then
    read -r -a CC <<< "${OPUS_CC:-cc}"
    AR=ar
    TARGET_NAME="this computer ($(uname -s) $(uname -m))"
else
    BIN=$(ndk_bin)
    read -r -a CC <<< "${OPUS_CC:-$BIN/aarch64-linux-android${ANDROID_API}-clang}"
    AR="$BIN/llvm-ar"
    TARGET_NAME=aarch64-linux-android
fi
OBJ_DIR="$LIBOPUS_PREFIX/obj"
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR" "$LIBOPUS_PREFIX/lib" "$LIBOPUS_PREFIX/include/opus"

# Prints the file names that a variable of a libopus .mk file lists.
mk_list() {
    awk -v name="$2" '
        $1 == name && $2 == "=" { on = 1; sub(/^[^=]*=/, "") }
        on {
            line = $0
            more = (line ~ /\\[ \t]*$/)
            sub(/\\[ \t]*$/, "", line)
            n = split(line, parts, " ")
            for (i = 1; i <= n; i++) print parts[i]
            if (!more) on = 0
        }' "$OPUS_SRC/$1"
}

# Like configure: when the compiler takes the DOTPROD intrinsics without
# extra flags, every target of this compiler has them (PRESUME). Otherwise
# libopus may use them only after a check at run time (RTCD).
dotprod_test="$OBJ_DIR/dotprod-test.c"
printf '%s\n' '#include <arm_neon.h>' \
    'int32x4_t f(int32x4_t acc, int8x16_t a, int8x16_t b) { return vdotq_s32(acc, a, b); }' \
    > "$dotprod_test"
if [ "$FORCE_RTCD" = 0 ] && "${CC[@]}" -c "$dotprod_test" -o "$OBJ_DIR/dotprod-test.o" 2>/dev/null; then
    DOTPROD_DEFINES=(-DOPUS_ARM_MAY_HAVE_DOTPROD=1 -DOPUS_ARM_PRESUME_DOTPROD=1)
    RTCD=0
else
    DOTPROD_DEFINES=(-DOPUS_ARM_MAY_HAVE_DOTPROD=1 -DOPUS_HAVE_RTCD=1)
    RTCD=1
fi
rm -f "$dotprod_test" "$OBJ_DIR/dotprod-test.o"

SOURCES=$(
    mk_list celt_sources.mk CELT_SOURCES
    [ "$RTCD" = 1 ] && mk_list celt_sources.mk CELT_SOURCES_ARM_RTCD
    mk_list celt_sources.mk CELT_SOURCES_ARM_NEON_INTR
    mk_list silk_sources.mk SILK_SOURCES
    mk_list silk_sources.mk SILK_SOURCES_FLOAT
    [ "$RTCD" = 1 ] && mk_list silk_sources.mk SILK_SOURCES_ARM_RTCD
    mk_list silk_sources.mk SILK_SOURCES_ARM_NEON_INTR
    mk_list opus_sources.mk OPUS_SOURCES
    mk_list opus_sources.mk OPUS_SOURCES_FLOAT
)

DEFINES=(
    -DOPUS_BUILD
    "-DPACKAGE_VERSION=\"$OPUS_VERSION\""
    -DVAR_ARRAYS=1
    -DENABLE_RES24=1
    -DFLOAT_APPROX=1
    -DHAVE_LRINTF=1
    -DHAVE_LRINT=1
    -DENABLE_HARDENING=1
    -DDISABLE_DEBUG_FLOAT=1
    -DOPUS_ARM_MAY_HAVE_NEON_INTR=1
    -DOPUS_ARM_PRESUME_NEON_INTR=1
    -DOPUS_ARM_PRESUME_AARCH64_NEON_INTR=1
    "${DOTPROD_DEFINES[@]}"
)
CFLAGS=(
    -g -O2 -fvisibility=hidden -D_FORTIFY_SOURCE=2
    -W -Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wstrict-prototypes
)
# shellcheck disable=SC2206
CFLAGS+=(${OPUS_EXTRA_CFLAGS:-})
INCLUDES=(
    -I"$OPUS_SRC" -I"$OPUS_SRC/include" -I"$OPUS_SRC/celt" -I"$OPUS_SRC/silk"
    -I"$OPUS_SRC/silk/float" -I"$OPUS_SRC/silk/fixed" -I"$OPUS_SRC/dnn"
)

{
    echo "# libopus for $TARGET_NAME, built by scripts/android/build-libopus.sh"
    echo "# source: $OPUS_SRC"
    echo "# version: $OPUS_VERSION"
    echo "# compiler: $("${CC[@]}" --version | head -n 1)"
    echo "CC=${CC[*]}"
    echo "DEFINES=${DEFINES[*]}"
    echo "CFLAGS=${CFLAGS[*]}"
    echo "SOURCES:"
    for s in $SOURCES; do echo "  $s"; done
} > "$LIBOPUS_PREFIX/build-flags.txt"

echo "compiling $(echo "$SOURCES" | wc -w | tr -d ' ') files of $OPUS_SRC"
# A helper that compiles one source file (its argument) to one object. The
# object name keeps the directory, so that files with the same name in
# different directories do not collide.
{
    echo '#!/usr/bin/env bash'
    printf 'cmd=('
    printf ' %q' "${CC[@]}" "${DEFINES[@]}" "${CFLAGS[@]}" "${INCLUDES[@]}"
    echo ')'
    printf 'obj=%q/$(echo "$1" | tr / _ | sed "s/\\.c$/.o/")\n' "$OBJ_DIR"
    printf '"${cmd[@]}" -c %q/"$1" -o "$obj" 2>"$obj.log" || { cat "$obj.log" >&2; exit 255; }\n' "$OPUS_SRC"
} > "$OBJ_DIR/compile-one.sh"
chmod +x "$OBJ_DIR/compile-one.sh"
echo "$SOURCES" | xargs -n 1 -P "$JOBS" "$OBJ_DIR/compile-one.sh"

rm -f "$LIBOPUS_PREFIX/lib/libopus.a"
"$AR" rcs "$LIBOPUS_PREFIX/lib/libopus.a" "$OBJ_DIR"/*.o
cp "$OPUS_SRC"/include/opus*.h "$LIBOPUS_PREFIX/include/opus/"
echo "installed $LIBOPUS_PREFIX/lib/libopus.a ($(du -k "$LIBOPUS_PREFIX/lib/libopus.a" | cut -f1) KB)"
echo "flags: $LIBOPUS_PREFIX/build-flags.txt"
