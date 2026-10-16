#!/bin/bash
# Quick timing of opus1 against C libopus with the `bench` command of
# examples/perf_harness.rs (docs/08-performance.md describes it).
#
# Usage: ./bench_quick.sh [label] [filter]
#
# `filter` selects scenarios whose name contains it; several parts are
# separated by commas (for example `celt_fb,silk_wb`). Without it every
# scenario of the bench set runs.
#
# It runs fewer repetitions than the default `bench` (6 instead of 15, in
# one round), so its medians are less steady. Use the full command for
# numbers to publish:
#   cargo run --release --example perf_harness -- bench

set -euo pipefail

LABEL=${1:-"baseline"}
FILTER=${2:-}
DATE=$(date +%Y-%m-%d_%H-%M-%S)
RESULT_FILE="bench_results_${LABEL}_${DATE}.md"

args=(bench --reps 6 --rounds 1 --md "$RESULT_FILE")
if [ -n "$FILTER" ]; then
    args+=(--filter "$FILTER")
fi

echo "=== opus1 quick benchmark (perf_harness bench) ==="
echo "Label: $LABEL"
echo "Results will be saved to: $RESULT_FILE"
echo ""

cargo run --release --example perf_harness -- "${args[@]}"

echo ""
echo "Full results saved to: $RESULT_FILE"
