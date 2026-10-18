#!/usr/bin/env python3
"""Writes src/analysis/mlp_data.rs from src/mlp_data.c of libopus.

The file of libopus holds the weights of the small network of the speech
and music analysis as int8 arrays, and three structs that name the arrays
and give the layer sizes. This script copies every array value for value
and checks the layer sizes against the ones opus1 uses.

Usage:
    python3 scripts/gen_analysis_mlp_data.py <libopus>/src/mlp_data.c > src/analysis/mlp_data.rs

The output is committed. Run the script again only when libopus changes
its weights.
"""

import re
import sys

# The arrays of mlp_data.c, in the order of the file, and the length each
# must have.
ARRAYS = [
    ("layer0_weights", 800),
    ("layer0_bias", 32),
    ("layer1_weights", 2304),
    ("layer1_recur_weights", 1728),
    ("layer1_bias", 72),
    ("layer2_weights", 48),
    ("layer2_bias", 2),
]

# The layer structs: name, the arrays they point to, then the sizes
# (inputs, neurons and, for a dense layer, the sigmoid flag).
LAYERS = {
    "layer0": ("AnalysisDenseLayer", ["layer0_bias", "layer0_weights"], [25, 32, 0]),
    "layer1": ("AnalysisGRULayer", ["layer1_bias", "layer1_weights", "layer1_recur_weights"], [32, 24]),
    "layer2": ("AnalysisDenseLayer", ["layer2_bias", "layer2_weights"], [24, 2, 1]),
}


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gen_analysis_mlp_data.py <path to libopus src/mlp_data.c>")
    text = open(sys.argv[1], encoding="ascii").read()

    arrays = {}
    for m in re.finditer(r"static const opus_int8 (\w+)\[(\d+)\] = \{([^}]*)\};", text):
        name, n, body = m.group(1), int(m.group(2)), m.group(3)
        values = [int(v) for v in body.replace("\n", " ").split(",") if v.strip()]
        if len(values) != n:
            sys.exit(f"{name}: {len(values)} values, declared {n}")
        if any(v < -128 or v > 127 for v in values):
            sys.exit(f"{name}: value outside int8")
        arrays[name] = values
    if [a for a, _ in ARRAYS] != list(arrays):
        sys.exit(f"unexpected arrays: {list(arrays)}")
    for name, n in ARRAYS:
        if len(arrays[name]) != n:
            sys.exit(f"{name}: length {len(arrays[name])}, expected {n}")

    for name, (ctype, refs, sizes) in LAYERS.items():
        m = re.search(r"const " + ctype + r" " + name + r" = \{([^}]*)\};", text)
        if not m:
            sys.exit(f"layer {name} not found")
        fields = [f.strip() for f in m.group(1).replace("\n", " ").split(",") if f.strip()]
        if fields[: len(refs)] != refs or [int(f) for f in fields[len(refs):]] != sizes:
            sys.exit(f"layer {name}: {fields}, expected {refs + sizes}")

    out = []
    out.append("//! The weights of the network of the speech and music analysis:")
    out.append("//! `src/mlp_data.c` of libopus 1.6.1, value for value.")
    out.append("//!")
    out.append("//! Written by `scripts/gen_analysis_mlp_data.py`. Do not edit by hand.")
    out.append("//!")
    out.append("//! Each array is stored as in C: the weight that connects input `j` to")
    out.append("//! output `i` of a layer with `n` outputs is at `j * n + i` (`3 * n` for")
    out.append("//! the gated recurrent unit, whose three gates share one row).")
    out.append("")
    for name, n in ARRAYS:
        out.append(f"/// libopus `{name}`.")
        out.append(f"pub const {name.upper()}: [i8; {n}] = [")
        values = arrays[name]
        for i in range(0, n, 16):
            out.append("    " + " ".join(f"{v}," for v in values[i : i + 16]))
        out.append("];")
        out.append("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
