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
#   - DISABLE_DEBUG_FLOAT: on by default. Only the neural network code
#     (see "Neural network features" below) reads it.
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
#     the default build. But the choice changes which delayed-decision quantizer
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
#   - Deep PLC, DRED, OSCE and QEXT are off (their defaults). The options
#     of the next section turn on the first three.
#
# The exact command lines are written to $LIBOPUS_PREFIX/build-flags.txt.
#
# Neural network features. libopus 1.5 and later have optional decoder
# features that run small neural networks (DNN, deep neural network): deep
# packet loss concealment (deep PLC) from decoder complexity 5, the speech
# enhancement of SILK frames (OSCE, Opus Speech Coding Enhancement: LACE at
# complexity 6, NoLACE from complexity 7), and the bandwidth extension of
# OSCE (BWE, turned on by OPUS_SET_OSCE_BWE). With DRED (deep redundancy),
# the encoder adds redundant data to the packets, and the decoder uses it
# after a loss. All are off by default. These options turn them on as libopus 1.6.1's configure does
# (configure.ac, Makefile.am and lpcnet_sources.mk):
#
#   --deep-plc   -DENABLE_DEEP_PLC=1 and the sources DEEP_PLC_SOURCES.
#   --osce       -DENABLE_OSCE=1 -DENABLE_OSCE_BWE=1 and OSCE_SOURCES, as
#                --enable-osce does (configure defines both; the CMake
#                and meson builds of 1.6.1 define only ENABLE_OSCE, so they
#                have no BWE). Implies --deep-plc, as in configure.
#   --dred       -DENABLE_DRED=1 and DRED_SOURCES. Implies --deep-plc. This
#                changes the encoder, so the reference builds leave it out.
#   --rdovae-weights
#                Compiles only the two weight files of DRED's networks (the
#                RDOVAE, rate-distortion-optimized variational autoencoder):
#                dnn/dred_rdovae_enc_data.c and dnn/dred_rdovae_dec_data.c,
#                without ENABLE_DRED. configure has no such option. No code
#                of the library calls them, so the encoder and the decoder
#                behave as without DRED. Tests can read the RDOVAE layers
#                (rdovaeenc_arrays, init_rdovaeenc, ...) from C: they are
#                the only sparse layers and the only dilated convolutions of
#                the 1.6.1 models. Implies --deep-plc, whose sources hold
#                the code that reads them (parse_lpcnet_weights.c).
#   --dnn        --deep-plc --osce --rdovae-weights. DRED stays off.
#   --dnn-weights-file
#                -DUSE_WEIGHTS_FILE (no configure option; libopus's own
#                Makefile leaves it to CFLAGS). The library then has no
#                built-in weights. Every model starts unloaded until the
#                application passes a weight blob with OPUS_SET_DNN_BLOB.
#
# With any of the first four options, the script also compiles, as
# Makefile.am does:
#   - DNN_SOURCES_NEON (dnn/arm/nnet_neon.c),
#   - DNN_SOURCES_DOTPROD (dnn/arm/nnet_dotprod.c), with the extra flag
#     ARM_DOTPROD_INTR_CFLAGS (default -march=armv8.2-a+dotprod). Makefile.am
#     adds it to this one file also when the compiler presumes DOTPROD. The
#     NDK's clang stops at nnet_dotprod.c:33 without it.
#   - DNN_SOURCES_ARM_RTCD (dnn/arm/arm_dnn_map.c) with run-time detection.
#     With it, opus_select_arch() returns 4 on a processor with DOTPROD and
#     3 without, and the network code takes the DOTPROD kernels only for 4.
# Without LIBOPUS_PREFIX, these builds go to $ANDROID_OUT/libopus-dnn, or
# to $ANDROID_OUT/libopus-host-dnn with --host, so they do not replace the
# default build.
#
# With --host and a DNN option, the script also builds two programs into
# $LIBOPUS_PREFIX/bin/. Each writes a weight blob in libopus's format
# (records of a 64-byte header "DNNw" and the data, the format that
# OPUS_SET_DNN_BLOB reads):
#   - dump_weights_blob: libopus's own dnn/write_lpcnet_weights.c, built
#     as Makefile.am builds it (-DDUMP_BINARY_WEIGHTS). It writes
#     weights_blob.bin in the current directory with PitchDNN, FARGAN, the
#     PLC model, the RDOVAE encoder and decoder, LACE and NoLACE, in that
#     order. It leaves out BBWENet, the network of the BWE. So with
#     USE_WEIGHTS_FILE and --osce, its blob does not load into the OSCE
#     part of a decoder: osce_load_models fails, and LACE and NoLACE stay
#     unloaded too.
#   - dump_weights_blob_all (with --osce only): scripts/android/
#     dump_weights_blob_all.c. It reuses write_weights() of the same file
#     and writes the same records followed by those of BBWENet, to the
#     file named by its argument. Built with DISABLE_DEBUG_FLOAT, so the
#     float copies of BBWENet's int8 weights are left out, as in the
#     library.
#
# The reference builds of the neural network work (all from the release
# 1.6.1, with run-time detection):
#
#   S=target/perf/libopus-1.6.1-src/opus-1.6.1
#   # Mac, without vector loops: the bit-exact reference for opus1's
#   # neural network code.
#   OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize" \
#     LIBOPUS_PREFIX=target/android/libopus-host-1.6.1-dnn-novec \
#     scripts/android/build-libopus.sh --host --rtcd --dnn --src $S
#   # Mac, -O2: for timings and for tests with a tolerance.
#   LIBOPUS_PREFIX=target/android/libopus-host-1.6.1-dnn \
#     scripts/android/build-libopus.sh --host --rtcd --dnn --src $S
#   # Android, the same two (run-time detection is the default there).
#   OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize" \
#     LIBOPUS_PREFIX=target/android/libopus-1.6.1-dnn-novec \
#     scripts/android/build-libopus.sh --dnn --src $S
#   LIBOPUS_PREFIX=target/android/libopus-1.6.1-dnn \
#     scripts/android/build-libopus.sh --dnn --src $S
#
# The vector loops that clang makes at -O2 round some products before they
# are added, so the -O2 builds of the Mac and of the phone conceal and
# enhance with other last bits, and the networks turn these into large
# differences. The builds without vector loops gave the same bits on both,
# with deep PLC, LACE, NoLACE and BWE, on the phone's Cortex-A76 and
# Cortex-A55 (docs/09-testing.md, "Builds with the neural network
# features").
# OPUS_EXTRA_CFLAGS applies to every file, also to celt/ and silk/, because
# the network code calls celt_fir, opus_fft and silk_burg_analysis.
#
# Tests that need such a library start with
# #![cfg(feature = "test-libopus-dnn")] and run in their own cargo
# directory:
#
#   LIBOPUS_LIB_DIR=$PWD/target/android/libopus-host-1.6.1-dnn-novec \
#     LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dnn-ref \
#     cargo test --release --features test-libopus-dnn,... --test NAME
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
# Without --src and OPUS_SRC, the script uses the release 1.6.1 unpacked at
# its default place, target/perf/libopus-1.6.1-src/opus-1.6.1 under the
# repository root. From a git worktree it also looks there under the main
# checkout (the parent of the git common directory). Unpack the release
# there once:
#
#   mkdir -p target/perf/libopus-1.6.1-src
#   tar xzf opus-1.6.1.tar.gz -C target/perf/libopus-1.6.1-src
#
# When that tree is missing too, the script falls back to the copy that
# the crate opus-head-sys 0.3.3 vendors, which is in the cargo registry
# after any build of the tests. Its file vendored/OPUS_VERSION names commit
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
#   OPUS_SRC          The libopus source tree, as --src. Default:
#                     target/perf/libopus-1.6.1-src/opus-1.6.1 under the
#                     repository root or the main checkout, else the copy
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
#   ARM_DOTPROD_INTR_CFLAGS
#                     The extra flags of dnn/arm/nnet_dotprod.c, as in
#                     configure. Default -march=armv8.2-a+dotprod.
#
# Usage: scripts/android/build-libopus.sh [--host] [--rtcd] [--src DIR]
#            [--dnn] [--deep-plc] [--osce] [--dred] [--rdovae-weights]
#            [--dnn-weights-file]
set -euo pipefail
PREFIX_FROM_USER=${LIBOPUS_PREFIX:-}
source "$(dirname "$0")/env.sh"

HOST=0
FORCE_RTCD=0
DEEP_PLC=0
OSCE=0
DRED=0
RDOVAE_WEIGHTS=0
WEIGHTS_FILE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST=1 ;;
        --rtcd) FORCE_RTCD=1 ;;
        --src)
            [ $# -ge 2 ] || { echo "error: --src needs a directory" >&2; exit 2; }
            OPUS_SRC=$2; shift ;;
        --dnn) DEEP_PLC=1; OSCE=1; RDOVAE_WEIGHTS=1 ;;
        --deep-plc) DEEP_PLC=1 ;;
        --osce) OSCE=1 ;;
        --dred) DRED=1 ;;
        --rdovae-weights) RDOVAE_WEIGHTS=1 ;;
        --dnn-weights-file) WEIGHTS_FILE=1 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
# As in configure.ac, OSCE and DRED turn on deep PLC. Its sources hold the
# code that all networks share (nnet.c, parse_lpcnet_weights.c), which the
# RDOVAE weight files need as well.
if [ "$OSCE" = 1 ] || [ "$DRED" = 1 ] || [ "$RDOVAE_WEIGHTS" = 1 ]; then
    DEEP_PLC=1
fi
if [ "$WEIGHTS_FILE" = 1 ] && [ "$DEEP_PLC" = 0 ]; then
    echo "error: --dnn-weights-file needs --dnn, --deep-plc, --osce or --dred" >&2
    exit 2
fi
# DNN=1 when any neural network code is built.
DNN=$DEEP_PLC
DNN_SUFFIX=
[ "$DNN" = 1 ] && DNN_SUFFIX=-dnn
if [ "$HOST" = 1 ]; then
    # The defines below choose the NEON code of 64-bit ARM.
    case "$(uname -m)" in
        arm64 | aarch64) ;;
        *) echo "error: --host needs a 64-bit ARM computer; this one is $(uname -m)" >&2; exit 2 ;;
    esac
    LIBOPUS_PREFIX=${PREFIX_FROM_USER:-$ANDROID_OUT/libopus-host$DNN_SUFFIX}
else
    need_ndk
    LIBOPUS_PREFIX=${PREFIX_FROM_USER:-$ANDROID_OUT/libopus$DNN_SUFFIX}
fi

# The default place of the release 1.6.1: under this checkout, then under
# the main checkout when this is a git worktree.
RELEASE_SUBDIR=target/perf/libopus-1.6.1-src/opus-1.6.1
if [ -z "${OPUS_SRC:-}" ]; then
    MAIN_ROOT=$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
    MAIN_ROOT=${MAIN_ROOT:+$(dirname "$MAIN_ROOT")}
    for root in "$REPO_ROOT" "$MAIN_ROOT"; do
        if [ -n "$root" ] && [ -f "$root/$RELEASE_SUBDIR/celt_sources.mk" ]; then
            OPUS_SRC=$root/$RELEASE_SUBDIR
            echo "no --src given; building the release 1.6.1 in $OPUS_SRC" >&2
            break
        fi
    done
fi
if [ -z "${OPUS_SRC:-}" ]; then
    for d in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/opus-head-sys-0.3.3/vendored/opus; do
        [ -f "$d/celt_sources.mk" ] && OPUS_SRC=$d
    done
    if [ -n "${OPUS_SRC:-}" ]; then
        echo "warning: no --src given and no $RELEASE_SUBDIR; building the vendored libopus main branch, not the release 1.6.1" >&2
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
# The neural network sources, in the order of Makefile.am (LPCNET_SOURCES).
# The awk removes the RDOVAE weight files when --dred lists them too.
if [ "$DNN" = 1 ]; then
    SOURCES="$SOURCES
$(
    mk_list lpcnet_sources.mk DEEP_PLC_SOURCES
    [ "$DRED" = 1 ] && mk_list lpcnet_sources.mk DRED_SOURCES
    [ "$RDOVAE_WEIGHTS" = 1 ] && printf '%s\n' dnn/dred_rdovae_enc_data.c dnn/dred_rdovae_dec_data.c
    [ "$OSCE" = 1 ] && mk_list lpcnet_sources.mk OSCE_SOURCES
    [ "$RTCD" = 1 ] && mk_list lpcnet_sources.mk DNN_SOURCES_ARM_RTCD
    mk_list lpcnet_sources.mk DNN_SOURCES_DOTPROD
    mk_list lpcnet_sources.mk DNN_SOURCES_NEON
)"
    SOURCES=$(echo "$SOURCES" | awk 'NF && !seen[$0]++')
fi
DNN_DEFINES=()
[ "$DEEP_PLC" = 1 ] && DNN_DEFINES+=(-DENABLE_DEEP_PLC=1)
[ "$OSCE" = 1 ] && DNN_DEFINES+=(-DENABLE_OSCE=1 -DENABLE_OSCE_BWE=1)
[ "$DRED" = 1 ] && DNN_DEFINES+=(-DENABLE_DRED=1)
[ "$WEIGHTS_FILE" = 1 ] && DNN_DEFINES+=(-DUSE_WEIGHTS_FILE=1)
# The extra flags of dnn/arm/nnet_dotprod.c (Makefile.am: ARM_DOTPROD_OBJ).
read -r -a DOTPROD_CFLAGS <<< "${ARM_DOTPROD_INTR_CFLAGS:--march=armv8.2-a+dotprod}"

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
# ${a[@]+"${a[@]}"}: an empty array is an error under `set -u` in bash 3.2.
DEFINES+=(${DNN_DEFINES[@]+"${DNN_DEFINES[@]}"})
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
    if [ "$DNN" = 1 ]; then
        echo "DNN: deep-plc=$DEEP_PLC osce=$OSCE dred=$DRED rdovae-weights=$RDOVAE_WEIGHTS weights-file=$WEIGHTS_FILE"
        echo "PER-FILE CFLAGS: dnn/arm/nnet_dotprod.c ${DOTPROD_CFLAGS[*]}"
    fi
    echo "SOURCES:"
    for s in $SOURCES; do echo "  $s"; done
} > "$LIBOPUS_PREFIX/build-flags.txt"

echo "compiling $(echo "$SOURCES" | wc -w | tr -d ' ') files of $OPUS_SRC"
# A helper that compiles one source file (its argument) to one object. The
# object name keeps the directory, so that files with the same name in
# different directories do not collide. dnn/arm/nnet_dotprod.c gets the
# DOTPROD flags at the end, as Makefile.am adds them to its CFLAGS.
{
    echo '#!/usr/bin/env bash'
    printf 'cmd=('
    printf ' %q' "${CC[@]}" "${DEFINES[@]}" "${CFLAGS[@]}" "${INCLUDES[@]}"
    echo ')'
    printf 'case "$1" in dnn/arm/nnet_dotprod.c) cmd+=('
    printf ' %q' "${DOTPROD_CFLAGS[@]}"
    echo ') ;; esac'
    printf 'obj=%q/$(echo "$1" | tr / _ | sed "s/\\.c$/.o/")\n' "$OBJ_DIR"
    printf '"${cmd[@]}" -c %q/"$1" -o "$obj" 2>"$obj.log" || { cat "$obj.log" >&2; exit 255; }\n' "$OPUS_SRC"
} > "$OBJ_DIR/compile-one.sh"
chmod +x "$OBJ_DIR/compile-one.sh"
echo "$SOURCES" | xargs -n 1 -P "$JOBS" "$OBJ_DIR/compile-one.sh"

rm -f "$LIBOPUS_PREFIX/lib/libopus.a"
"$AR" rcs "$LIBOPUS_PREFIX/lib/libopus.a" "$OBJ_DIR"/*.o
cp "$OPUS_SRC"/include/opus*.h "$LIBOPUS_PREFIX/include/opus/"
echo "installed $LIBOPUS_PREFIX/lib/libopus.a ($(du -k "$LIBOPUS_PREFIX/lib/libopus.a" | cut -f1) KB)"

# The weight blob writers (see the top). They include the weight files
# themselves and need no library, as in Makefile.am (dump_weights_blob_LDADD
# is only the math library). Writers of an earlier build are removed first.
rm -rf "$LIBOPUS_PREFIX"/bin/dump_weights_blob*
if [ "$HOST" = 1 ] && [ "$DNN" = 1 ]; then
    mkdir -p "$LIBOPUS_PREFIX/bin"
    BLOB_CMD=("${CC[@]}" "${DEFINES[@]}" -DDUMP_BINARY_WEIGHTS "${CFLAGS[@]}" "${INCLUDES[@]}")
    "${BLOB_CMD[@]}" "$OPUS_SRC/dnn/write_lpcnet_weights.c" -o "$LIBOPUS_PREFIX/bin/dump_weights_blob" -lm
    echo "WEIGHT BLOB WRITER: ${BLOB_CMD[*]} $OPUS_SRC/dnn/write_lpcnet_weights.c -o bin/dump_weights_blob -lm" \
        >> "$LIBOPUS_PREFIX/build-flags.txt"
    if [ "$OSCE" = 1 ]; then
        all_src="$(cd "$(dirname "$0")" && pwd)/dump_weights_blob_all.c"
        "${BLOB_CMD[@]}" "$all_src" -o "$LIBOPUS_PREFIX/bin/dump_weights_blob_all" -lm
        echo "WEIGHT BLOB WRITER: ${BLOB_CMD[*]} $all_src -o bin/dump_weights_blob_all -lm" \
            >> "$LIBOPUS_PREFIX/build-flags.txt"
    fi
    echo "weight blob writers: $LIBOPUS_PREFIX/bin/"
fi
echo "flags: $LIBOPUS_PREFIX/build-flags.txt"
