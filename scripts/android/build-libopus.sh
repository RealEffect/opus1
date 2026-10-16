#!/usr/bin/env bash
# Builds libopus for Android on 64-bit ARM (arm64-v8a) as a static library,
# with the NDK's clang. The tests and examples/perf_harness.rs link it when
# they are built for a phone (scripts/android/test.sh, bench.sh, golden.sh,
# profile.sh).
#
# The build follows what libopus's own `configure` chooses for the host
# aarch64-linux-android with clang and its default options. It is a
# floating-point build, like the Homebrew libopus that the tests link on
# macOS:
#
#   - Compiler flags: -g -O2 (the autoconf default), -fvisibility=hidden,
#     -D_FORTIFY_SOURCE=2 (configure adds it when the compiler does not set
#     it) and the warning flags of configure.
#   - FLOAT_APPROX: configure turns it on for aarch64.
#   - VAR_ARRAYS: the compiler supports C99 variable-size arrays.
#   - ENABLE_HARDENING: on by default.
#   - NEON intrinsics: clang compiles them without extra flags, so configure
#     sets OPUS_ARM_MAY_HAVE_NEON_INTR and OPUS_ARM_PRESUME_NEON_INTR, and
#     OPUS_ARM_PRESUME_AARCH64_NEON_INTR for the AArch64-only intrinsics.
#     The sources CELT_SOURCES_ARM_NEON_INTR and SILK_SOURCES_ARM_NEON_INTR
#     are compiled.
#   - DOTPROD intrinsics need -march=armv8.2-a+dotprod with this compiler,
#     so configure sets only OPUS_ARM_MAY_HAVE_DOTPROD. That turns on
#     run-time CPU detection (OPUS_HAVE_RTCD) and adds the sources
#     CELT_SOURCES_ARM_RTCD and SILK_SOURCES_ARM_RTCD. Only the neural
#     network code uses DOTPROD, and it is not part of this build.
#   - Deep PLC, DRED, OSCE and QEXT are off (their defaults).
#
# The exact command lines are written to $LIBOPUS_PREFIX/build-flags.txt.
#
# Which libopus. The default source is the copy that the crate
# opus-head-sys 0.3.3 vendors. Its file vendored/OPUS_VERSION names
# commit 228a0f85 of the xiph/opus main branch (2026-08-28). That is newer
# than the release 1.6.1 that Homebrew ships, and it writes other packets
# than 1.6.1 for many SILK, Hybrid and LFE streams. For a reference equal
# to 1.6.1, unpack the release and set OPUS_SRC to it. The version string
# says which source was built: the commit of OPUS_VERSION, or OPUS_VERSION
# when it is set.
#
# Two notes on comparing against the Homebrew library at the bit level:
#   - At -O2, clang vectorizes some loops that add up products in order,
#     for example the energy sums of stereo_itheta (celt/vq.c). The
#     vector code rounds each product before the addition, where the scalar
#     code uses a fused multiply-add, so the last bit can differ.
#     The Homebrew library does not have these vector loops.
#     OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize" removes them.
#   - The library calls the C library's acos, cos, exp, exp2, log and
#     log10. Android's and Apple's give different last bits for some
#     inputs.
#
# Settings (environment variables; see scripts/android/env.sh for the
# common ones):
#   ANDROID_NDK_HOME  required.
#   OPUS_SRC          The libopus source tree. Default: the copy in the
#                     cargo registry, vendored by the crate opus-head-sys
#                     0.3.3 (see above).
#   OPUS_VERSION      The version that opus_get_version_string() reports.
#                     Default: git-<commit> from the OPUS_VERSION file next
#                     to the source tree, or "unknown".
#   LIBOPUS_PREFIX    Install directory (lib/libopus.a, include/opus/*.h).
#   JOBS              Parallel compiler processes. Default 8.
#   OPUS_CC           The compiler command. Default: the NDK's
#                     aarch64-linux-android$ANDROID_API-clang.
#   OPUS_EXTRA_CFLAGS More compiler flags, added after the ones above. For
#                     example "-fno-vectorize -fno-slp-vectorize" builds
#                     a library whose loops are not vectorized by clang.
#
# Usage: scripts/android/build-libopus.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_ndk

if [ -z "${OPUS_SRC:-}" ]; then
    for d in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/opus-head-sys-0.3.3/vendored/opus; do
        [ -f "$d/celt_sources.mk" ] && OPUS_SRC=$d
    done
fi
if [ -z "${OPUS_SRC:-}" ] || [ ! -f "$OPUS_SRC/celt_sources.mk" ]; then
    echo "error: libopus source not found; set OPUS_SRC to a libopus 1.6 source tree" >&2
    exit 2
fi
if [ -z "${OPUS_VERSION:-}" ]; then
    commit=$(sed -n 's/^commit: *\([0-9a-f]\{12\}\).*/\1/p' "$OPUS_SRC/../OPUS_VERSION" 2>/dev/null || true)
    OPUS_VERSION=${commit:+git-$commit}
    OPUS_VERSION=${OPUS_VERSION:-unknown}
fi
JOBS=${JOBS:-8}

BIN=$(ndk_bin)
# The compiler command, as words. OPUS_CC replaces it, for example to try
# another clang: "clang -target aarch64-linux-android21 --sysroot=$SYSROOT".
read -r -a CC <<< "${OPUS_CC:-$BIN/aarch64-linux-android${ANDROID_API}-clang}"
AR="$BIN/llvm-ar"
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

SOURCES=$(
    mk_list celt_sources.mk CELT_SOURCES
    mk_list celt_sources.mk CELT_SOURCES_ARM_RTCD
    mk_list celt_sources.mk CELT_SOURCES_ARM_NEON_INTR
    mk_list silk_sources.mk SILK_SOURCES
    mk_list silk_sources.mk SILK_SOURCES_FLOAT
    mk_list silk_sources.mk SILK_SOURCES_ARM_RTCD
    mk_list silk_sources.mk SILK_SOURCES_ARM_NEON_INTR
    mk_list opus_sources.mk OPUS_SOURCES
    mk_list opus_sources.mk OPUS_SOURCES_FLOAT
)

DEFINES=(
    -DOPUS_BUILD
    "-DPACKAGE_VERSION=\"$OPUS_VERSION\""
    -DVAR_ARRAYS=1
    -DFLOAT_APPROX=1
    -DHAVE_LRINTF=1
    -DHAVE_LRINT=1
    -DENABLE_HARDENING=1
    -DOPUS_ARM_MAY_HAVE_NEON_INTR=1
    -DOPUS_ARM_PRESUME_NEON_INTR=1
    -DOPUS_ARM_PRESUME_AARCH64_NEON_INTR=1
    -DOPUS_ARM_MAY_HAVE_DOTPROD=1
    -DOPUS_HAVE_RTCD=1
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
    echo "# libopus for aarch64-linux-android, built by scripts/android/build-libopus.sh"
    echo "# source: $OPUS_SRC"
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
