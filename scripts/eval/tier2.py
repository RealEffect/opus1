#!/usr/bin/env python3
"""Run optional external quality metrics ("tier 2") on the signals that
`cargo run --release --example eval -- run --write-wav DIR` wrote.

    python3 scripts/eval/tier2.py DIR/manifest.jsonl > tier2.jsonl
    cargo run --release --example eval -- merge RUN.json tier2.jsonl

The script installs nothing and downloads nothing. It uses the tools that
are already installed and skips the others with a message on stderr:

- ViSQOL v3 (`visqol` program on PATH): audio mode on 48 kHz conditions,
  speech mode on the 16 kHz versions of speech in mono. Metric
  `visqol_audio` or `visqol_speech` (MOS-LQO, higher is better).
- PESQ (Python package `pesq`): wideband P.862.2 on the 16 kHz versions
  of speech in mono. Metric `pesq_wb` (higher is better). The ITU licence
  of the PESQ code restricts its use; use it locally only.
- SCOREQ (Python package `scoreq`) in reference mode on the 16 kHz
  versions of speech in mono. Metric `scoreq_ref`. In reference mode
  SCOREQ gives a distance to the reference, so lower is better.
- PLCMOS (Python package `speechmos`): no-reference score of packet loss
  concealment, on 16 kHz speech of conditions that drop packets. Metric
  `plcmos` (higher is better).

Each output line is one JSON object:
{"condition", "item", "codec", "metric", "value", "higher_is_better"}.
`--list` prints which tools were found. `--tools a,b` limits the run.
"""

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import wave


def have_module(name):
    return importlib.util.find_spec(name) is not None


def read_wav(path):
    """16-bit PCM WAV as a float array in [-1, 1) and its rate."""
    import numpy as np

    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def visqol(entry, codec):
    """ViSQOL MOS-LQO, or None when the condition has no matching files."""
    if "reference_16k" in entry:
        # Speech in mono: speech mode on the 16 kHz files.
        ref, deg, mode, name = entry["reference_16k"], entry[codec + "_16k"], ["--use_speech_mode"], "visqol_speech"
    elif entry["rate"] == 48000:
        ref, deg, mode, name = entry["reference_48k"], entry[codec], [], "visqol_audio"
    else:
        return None
    out = subprocess.run(
        ["visqol", "--reference_file", ref, "--degraded_file", deg] + mode,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    m = re.search(r"MOS-LQO:\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError("no MOS-LQO in the output of visqol")
    return name, float(m.group(1)), True


def pesq_wb(entry, codec):
    if "reference_16k" not in entry:
        return None
    from pesq import pesq

    ref, _ = read_wav(entry["reference_16k"])
    deg, _ = read_wav(entry[codec + "_16k"])
    return "pesq_wb", float(pesq(16000, ref, deg, "wb")), True


_scoreq_model = None


def scoreq_ref(entry, codec):
    global _scoreq_model
    if "reference_16k" not in entry:
        return None
    import scoreq

    if _scoreq_model is None:
        _scoreq_model = scoreq.Scoreq(data_domain="natural", mode="ref")
    value = _scoreq_model.predict(test_path=entry[codec + "_16k"], ref_path=entry["reference_16k"])
    return "scoreq_ref", float(value), False


def plcmos(entry, codec):
    if "reference_16k" not in entry or not entry.get("decode_loss"):
        return None
    from speechmos import plcmos as model

    deg, rate = read_wav(entry[codec + "_16k"])
    result = model.run(deg, sr=rate)
    value = result["plcmos"] if isinstance(result, dict) else result
    return "plcmos", float(value), True


TOOLS = {
    "visqol": (lambda: shutil.which("visqol") is not None, visqol),
    "pesq": (lambda: have_module("pesq") and have_module("numpy"), pesq_wb),
    "scoreq": (lambda: have_module("scoreq"), scoreq_ref),
    "plcmos": (lambda: have_module("speechmos") and have_module("numpy"), plcmos),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", nargs="?", help="manifest.jsonl written by eval run --write-wav")
    parser.add_argument("--tools", help="comma-separated subset of " + ",".join(TOOLS))
    parser.add_argument("--list", action="store_true", help="print which tools are installed")
    args = parser.parse_args()

    wanted = args.tools.split(",") if args.tools else list(TOOLS)
    unknown = [t for t in wanted if t not in TOOLS]
    if unknown:
        parser.error("unknown tool(s): " + ", ".join(unknown))
    available = [t for t in wanted if TOOLS[t][0]()]
    for t in wanted:
        print(f"{t}: {'found' if t in available else 'not installed, skipped'}", file=sys.stderr)
    if args.list:
        return 0
    if not args.manifest:
        parser.error("the manifest is required")
    if not available:
        print("no tier-2 tool is installed; nothing to do", file=sys.stderr)
        return 0

    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    failures = 0
    for entry in entries:
        for tool in available:
            for codec in ("opus1", "libopus"):
                try:
                    result = TOOLS[tool][1](entry, codec)
                except Exception as e:  # report and go on with the next file
                    failures += 1
                    print(f"{tool} failed on {entry['condition']} {entry['item']} {codec}: {e}", file=sys.stderr)
                    continue
                if result is None:
                    continue
                metric, value, higher = result
                print(
                    json.dumps(
                        {
                            "condition": entry["condition"],
                            "item": entry["item"],
                            "codec": codec,
                            "metric": metric,
                            "value": value,
                            "higher_is_better": higher,
                        }
                    )
                )
    if failures:
        print(f"{failures} measurement(s) failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
