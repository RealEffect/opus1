#!/usr/bin/env python3
"""Reference values of the MOC distance for tests/eval_moc.rs.

MOC ("modified opus_compare") is the objective distance of the Opus Speech
Coding Enhancement draft. src/eval/moc.rs ports `compare` of moc.py from
the libopus git repository (dnn/torch/osce/stndrd/evaluation/moc.py). This
script holds a transcription of the functions of moc.py that `compare`
uses, with these changes:

- `scipy.signal.get_window('hamming', 160)` is replaced by the same window
  written with numpy (scipy's `general_cosine` with `[0.54, 1 - 0.54]`,
  periodic), so that only numpy is needed.
- The matrix products of the band sums use `np.einsum` instead of `@`.
  numpy 2.0 on macOS prints spurious floating-point warnings for `@`; the
  values are the same up to the order of the sums.

Usage:

  moc_reference.py --cases DIR [--check-against PATH/moc.py]

DIR holds `cases.txt` (one line per case: name, 0 or 1 for the activity
detection, reference file, test file; the files are raw 16-bit
little-endian 16 kHz mono, relative to DIR). The script prints one line per
case: `name apply_vad value`, where value is the MOC with 17 significant
digits, `nan`, or `error:<exception>`.

With --check-against it also loads the original moc.py (with a stand-in
for scipy.signal.get_window) and checks that it gives the same values, to
within a relative 1e-12. That guards the transcription.

The transcribed functions carry the notice of moc.py:

/* Copyright (c) 2023 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import argparse
import importlib.util
import os
import site
import sys
import types

try:
    import numpy as np
except ImportError:
    # Under `python3 -I` the user's site-packages are not on the path.
    sys.path.append(site.getusersitepackages())
    import numpy as np


def get_window(window, Nx, fftbins=True):
    """scipy.signal.get_window for 'hamming': general_cosine(M, [0.54, 1 - 0.54])."""
    assert window == 'hamming'
    M = Nx + 1 if fftbins else Nx
    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k, a in enumerate([0.54, 1. - 0.54]):
        w += a * np.cos(k * fac)
    return w[:-1] if fftbins else w


# ---- transcription of moc.py -------------------------------------------

def compute_vad_mask(x, fs, stop_db=-70):

    frame_length = (fs + 49) // 50
    x = x[: frame_length * (len(x) // frame_length)]

    frames = x.reshape(-1, frame_length)
    frame_energy = np.sum(frames ** 2, axis=1)
    frame_energy_smooth = np.convolve(frame_energy, np.ones(5) / 5, mode='same')

    max_threshold = frame_energy.max() * 10 ** (stop_db/20)
    vactive = np.ones_like(frames)
    vactive[frame_energy_smooth < max_threshold, :] = 0
    vactive = vactive.reshape(-1)

    filter = np.sin(np.arange(frame_length) * np.pi / (frame_length - 1))
    filter = filter / filter.sum()

    mask = np.convolve(vactive, filter, mode='same')

    return x, mask


def convert_mask(mask, num_frames, frame_size=160, hop_size=40):
    num_samples = frame_size + (num_frames - 1) * hop_size
    if len(mask) < num_samples:
        mask = np.concatenate((mask, np.zeros(num_samples - len(mask))), dtype=mask.dtype)
    else:
        mask = mask[:num_samples]

    new_mask = np.array([np.mean(mask[i*hop_size : i*hop_size + frame_size]) for i in range(num_frames)])

    return new_mask


def power_spectrum(x, window_size=160, hop_size=40, window='hamming'):
    num_spectra = (len(x) - window_size - hop_size) // hop_size
    window = get_window(window, window_size)
    N = window_size // 2

    frames = np.concatenate([x[np.newaxis, i * hop_size : i * hop_size + window_size] for i in range(num_spectra)]) * window
    psd = np.abs(np.fft.fft(frames, axis=1)[:, :N + 1]) ** 2

    return psd


def frequency_mask(num_bands, up_factor, down_factor):

    up_mask = np.zeros((num_bands, num_bands))
    down_mask = np.zeros((num_bands, num_bands))

    for i in range(num_bands):
        up_mask[i, : i + 1] = up_factor ** np.arange(i, -1, -1)
        down_mask[i, i :] = down_factor ** np.arange(num_bands - i)

    return np.einsum('ik,kj->ij', down_mask, up_mask)


def rect_fb(band_limits, num_bins=None):
    num_bands = len(band_limits) - 1
    if num_bins is None:
        num_bins = band_limits[-1]

    fb = np.zeros((num_bands, num_bins))
    for i in range(num_bands):
        fb[i, band_limits[i]:band_limits[i+1]] = 1

    return fb


def compare(x, y, apply_vad=False):
    # filter bank: bark scale with minimum-2-bin bands and cutoff at 7.5 kHz
    band_limits = [0, 2, 4, 6, 7, 9, 11, 13, 15, 18, 22, 26, 31, 36, 43, 51, 60, 75]
    num_bands = len(band_limits) - 1
    fb = rect_fb(band_limits, num_bins=81)

    # trim samples to same size
    num_samples = min(len(x), len(y))
    x = x[:num_samples] * 2**15
    y = y[:num_samples] * 2**15

    psd_x = power_spectrum(x) + 100000
    psd_y = power_spectrum(y) + 100000

    num_frames = psd_x.shape[0]

    # average band energies
    be_x = np.einsum('tk,bk->tb', psd_x, fb) / np.sum(fb, axis=1)

    # frequecy masking
    f_mask = frequency_mask(num_bands, 0.1, 0.03)
    mask_x = np.einsum('tj,ij->ti', be_x, f_mask)

    # temporal masking
    for i in range(1, num_frames):
        mask_x[i, :] += 0.5 * mask_x[i-1, :]

    # apply mask
    masked_psd_x = psd_x + 0.1 * np.einsum('tb,bk->tk', mask_x, fb)
    masked_psd_y = psd_y + 0.1 * np.einsum('tb,bk->tk', mask_x, fb)

    # 2-frame average
    masked_psd_x = masked_psd_x[1:] + masked_psd_x[:-1]
    masked_psd_y = masked_psd_y[1:] + masked_psd_y[:-1]

    # distortion metric
    re = masked_psd_y / masked_psd_x
    im = np.log(re) ** 2
    Eb = (np.einsum('tk,bk->tb', im, fb) / np.sum(fb, axis=1))
    Ef = np.mean(Eb, axis=1)

    if apply_vad:
        _, mask = compute_vad_mask(x, 16000)
        mask = convert_mask(mask, Ef.shape[0])
    else:
        mask = np.ones_like(Ef)

    err = np.mean(np.abs(Ef[mask > 1e-6]) ** 3) ** (1/6)

    return float(err)

# ---- end of the transcription -------------------------------------------


def read_s16(path):
    """A raw 16-bit file scaled as moc.py scales a WAV file."""
    return np.fromfile(path, dtype='<i2').astype(np.float32) / 2**15


def run(fn, x, y, vad):
    with np.errstate(all='ignore'):
        try:
            v = fn(x, y, apply_vad=bool(vad))
        except Exception as e:  # moc.py raises on inputs that are too short
            return 'error:' + type(e).__name__
    if v != v:
        return 'nan'
    return '%.17g' % v


def load_original(path):
    """moc.py itself, with a stand-in for scipy.signal."""
    scipy = types.ModuleType('scipy')
    signal = types.ModuleType('scipy.signal')
    signal.get_window = get_window
    scipy.signal = signal
    sys.modules['scipy'] = scipy
    sys.modules['scipy.signal'] = signal
    spec = importlib.util.spec_from_file_location('moc_original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True, help='directory with cases.txt')
    parser.add_argument('--check-against', help='path of the original moc.py')
    args = parser.parse_args()

    original = load_original(args.check_against) if args.check_against else None
    failures = 0
    with open(os.path.join(args.cases, 'cases.txt')) as f:
        lines = [line.split() for line in f if line.strip()]
    for name, vad, ref, test in lines:
        x = read_s16(os.path.join(args.cases, ref))
        y = read_s16(os.path.join(args.cases, test))
        value = run(compare, x, y, int(vad))
        print(name, vad, value)
        if original is not None:
            other = run(original.compare, x, y, int(vad))
            same = other == value
            if not same and not value.startswith(('error', 'nan')) and not other.startswith(('error', 'nan')):
                a, b = float(value), float(other)
                same = abs(a - b) <= 1e-12 * abs(b) + 1e-15
            if not same:
                print('MISMATCH', name, vad, value, 'moc.py', other, file=sys.stderr)
                failures += 1
    if original is not None:
        print('checked against %s: %d mismatches' % (args.check_against, failures), file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
