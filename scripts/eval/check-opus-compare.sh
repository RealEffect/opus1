#!/usr/bin/env bash
# Build the C reference of the opus_compare port (src/eval/compare.rs).
#
# It compiles libopus src/opus_compare.c twice with the flags of
# scripts/android/build-libopus.sh (-O2 -fvisibility=hidden
# -D_FORTIFY_SOURCE=2):
#
#   opus_compare        the plain program, as libopus ships it;
#   opus_compare_exact  the same program through scripts/eval/opus_compare_exact.c,
#                       which prints the error and the quality with 17 digits.
#
# Then it compiles both to assembly and checks that the wrapper did not
# change the arithmetic: both must have the same number of fused
# multiply-add instructions, and neither may call sincos (which can round
# differently from separate sin and cos calls).
#
# Usage: scripts/eval/check-opus-compare.sh [--src LIBOPUS_TREE] [--out DIR] [--cc CC]
#
# Defaults: --src target/perf/libopus-1.6.1-src/opus-1.6.1, --out
# target/eval/opus_compare, --cc cc. At the end it prints the variable to
# set for the test:
#
#   OPUS1_OPUS_COMPARE_C=<out>/opus_compare_exact cargo test --release --test eval_opus_compare
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$root/target/perf/libopus-1.6.1-src/opus-1.6.1"
out="$root/target/eval/opus_compare"
cc="${CC:-cc}"
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --cc) cc="$2"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "$src/src/opus_compare.c" ]; then
    echo "error: $src/src/opus_compare.c not found (use --src)" >&2
    exit 1
fi
mkdir -p "$out"
flags=(-O2 -fvisibility=hidden -D_FORTIFY_SOURCE=2)

"$cc" "${flags[@]}" -o "$out/opus_compare" "$src/src/opus_compare.c" -lm
"$cc" "${flags[@]}" -I"$src/src" -o "$out/opus_compare_exact" "$root/scripts/eval/opus_compare_exact.c" -lm
"$cc" "${flags[@]}" -S -o "$out/opus_compare.s" "$src/src/opus_compare.c"
"$cc" "${flags[@]}" -I"$src/src" -S -o "$out/opus_compare_exact.s" "$root/scripts/eval/opus_compare_exact.c"

# Fused multiply-adds of 64-bit ARM (fmadd, fmsub, fmla, fmls, fnmadd, ...)
# and of x86 (vfmadd..., vfmsub..., vfnmadd...).
count_fma() {
    grep -Ec '^[[:space:]]+(f(n)?m(add|sub|la|ls)|vf(n)?m(add|sub)[0-9a-z]*)[[:space:].]' "$1" || true
}
plain=$(count_fma "$out/opus_compare.s")
exact=$(count_fma "$out/opus_compare_exact.s")
echo "fused multiply-add instructions: plain $plain, wrapped $exact"
if [ "$plain" != "$exact" ]; then
    echo "error: the wrapper changed the arithmetic of opus_compare.c" >&2
    exit 1
fi
if grep -Eq 'sincos' "$out/opus_compare.s" "$out/opus_compare_exact.s"; then
    echo "error: the compiler merged sin and cos into sincos" >&2
    exit 1
fi

# A self-check: identical inputs give error 0 and quality 100.
tmp="$out/self-check.sw"
head -c 3840 /dev/urandom > "$tmp"
"$out/opus_compare_exact" -s "$tmp" "$tmp" 2>&1 | grep -q 'Opus quality metric: 100 % (internal weighted error is 0)' || {
    echo "error: the self-check of opus_compare_exact failed" >&2
    exit 1
}
rm -f "$tmp"

echo "built $out/opus_compare and $out/opus_compare_exact ($("$cc" --version | head -1))"
echo "OPUS1_OPUS_COMPARE_C=$out/opus_compare_exact"
