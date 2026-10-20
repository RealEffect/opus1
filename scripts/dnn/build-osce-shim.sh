#!/usr/bin/env bash
# Builds the OSCE test shim (scripts/dnn/osce_test_shim.c) against one
# reference build of libopus made by `scripts/android/build-libopus.sh --dnn`.
#
#   scripts/dnn/build-osce-shim.sh <reference build> [<output directory>]
#
# The shim is compiled with the defines and flags that the reference build
# recorded in its build-flags.txt (so its view of silk_decoder_state matches
# the library) and with the include paths of the libopus source tree named
# there. It is then linked with the reference's lib/libopus.a into one
# shared library, libosce_shim.dylib (macOS) or libosce_shim.so (Linux),
# which tests/dnn_osce_libopus.rs loads at run time.
#
# The default output directory is target/osce-shim/<name of the reference
# build> in this repository; the test looks there first (see its header).
#
# Example (the bit-exact reference and the -O2 build of this Mac):
#
#   scripts/dnn/build-osce-shim.sh target/android/libopus-host-1.6.1-dnn-novec
#   scripts/dnn/build-osce-shim.sh target/android/libopus-host-1.6.1-dnn
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <reference build> [<output directory>]" >&2
    exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ref="$(cd "$1" && pwd)"
name="$(basename "$ref")"
out="${2:-$repo/target/osce-shim/$name}"
flags="$ref/build-flags.txt"
lib="$ref/lib/libopus.a"

[ -f "$flags" ] || { echo "no $flags" >&2; exit 1; }
[ -f "$lib" ] || { echo "no $lib" >&2; exit 1; }

# The values of the recorded build.
field() { sed -n "s/^$1=//p" "$flags" | head -n 1; }
src="$(sed -n 's/^# source: //p' "$flags" | head -n 1)"
cc="$(field CC)"
defines="$(field DEFINES)"
cflags="$(field CFLAGS)"
[ -n "$src" ] && [ -d "$src" ] || { echo "the source tree named in $flags is missing: $src" >&2; exit 1; }
for d in -DENABLE_OSCE=1 -DENABLE_OSCE_BWE=1; do
    case " $defines " in
        *" $d "*) ;;
        *) echo "$ref is not built with $d" >&2; exit 1 ;;
    esac
done

mkdir -p "$out"
case "$(uname -s)" in
    Darwin) so="$out/libosce_shim.dylib"; shared=(-dynamiclib -Wl,-install_name,"$so") ;;
    *) so="$out/libosce_shim.so"; shared=(-shared -fPIC) ;;
esac

includes=(-I"$src" -I"$src/include" -I"$src/celt" -I"$src/silk" -I"$src/silk/float" -I"$src/silk/fixed" -I"$src/dnn")

# shellcheck disable=SC2086 # the recorded flags are word lists
"${cc:-cc}" $defines $cflags "${includes[@]}" -c "$here/osce_test_shim.c" -o "$out/osce_test_shim.o"
# shellcheck disable=SC2086
"${cc:-cc}" $cflags "${shared[@]}" -o "$so" "$out/osce_test_shim.o" "$lib" -lm

{
    echo "# OSCE test shim built by scripts/dnn/build-osce-shim.sh"
    echo "reference: $ref"
    echo "library: $so"
    echo "compiler: $("${cc:-cc}" --version | head -n 1)"
} > "$out/shim-info.txt"
echo "$so"
