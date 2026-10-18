//! The "modified opus_compare" (MOC) distance of the Opus Speech Coding
//! Enhancement (OSCE) draft (`draft-ietf-mlcodec-opus-speech-coding-enhancement`).
//!
//! This is a port of `compare` in `moc.py` of the libopus git repository
//! (`dnn/torch/osce/stndrd/evaluation/moc.py`, Copyright (c) 2023 Amazon,
//! written by Jan Buethe, BSD licence). The snapshot it comes from predates
//! libopus 1.6. Draft 04 of the OSCE draft replaced the script by a C
//! program (`osce_compare.c`), which may compute other numbers.
//!
//! MOC compares two 16 kHz mono signals. It follows the steps of
//! [`opus_compare`](super::compare), with these differences:
//!
//! - windows of 160 samples (10 ms), one every 40 samples, with a periodic
//!   Hamming window, and 81 bins of 100 Hz;
//! - 17 bands up to 7.5 kHz;
//! - the distortion of a bin is `ln(r)^2`, with `r` = test / reference;
//! - the error of a frame is the mean over the bands of the mean band
//!   distortion, and the result is the 6th root of the mean of its cube;
//! - with `apply_vad`, frames where the reference is quiet are left out.
//!
//! `moc.py` computes in numpy. The port computes in `f64` where numpy does,
//! and in `f32` where numpy keeps `float32` (the input and the frame
//! energies of the activity detection, with numpy's pairwise summation).
//! numpy 2 rules apply: a `float32` value times a Python `float` stays
//! `float32`. The transform is a direct discrete Fourier transform instead
//! of numpy's FFT, so the results agree with numpy to about 1e-12, not bit
//! for bit.

use core::fmt;

/// The only sampling rate MOC accepts.
pub const MOC_RATE: u32 = 16_000;

const WINDOW: usize = 160;
const HOP: usize = 40;
/// Bins 0 to 80 (`N + 1` of `power_spectrum`).
const NBINS: usize = WINDOW / 2 + 1;
/// Band limits in bins of 100 Hz: bark-like bands of at least two bins, up
/// to 7.5 kHz.
const BAND_LIMITS: [usize; 18] = [0, 2, 4, 6, 7, 9, 11, 13, 15, 18, 22, 26, 31, 36, 43, 51, 60, 75];
const NB: usize = BAND_LIMITS.len() - 1;
/// Frame length of the activity detection: `(fs + 49) // 50`, 20 ms.
const VAD_FRAME: usize = (MOC_RATE as usize).div_ceil(50);
/// Shortest input with two spectra, which the two-frame sum needs.
const MIN_SAMPLES: usize = WINDOW + 3 * HOP;

/// Why MOC cannot be computed. `moc.py` raises an exception or returns NaN
/// in these cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MocError {
    /// Fewer than 280 samples (after cutting both signals to the shorter
    /// length): no frame is left after the sum of two frames.
    TooShort { samples: usize },
    /// The activity detection needs 5 frames of 20 ms (1600 samples).
    /// `moc.py` fails there with an indexing error.
    VadTooShort { samples: usize },
    /// The activity detection marked every frame inactive. `moc.py`
    /// returns NaN.
    NoActiveFrames,
}

impl fmt::Display for MocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            MocError::TooShort { samples } => {
                write!(f, "MOC needs at least {MIN_SAMPLES} samples, got {samples}")
            }
            MocError::VadTooShort { samples } => {
                write!(f, "MOC with activity detection needs at least {} samples, got {samples}", 5 * VAD_FRAME)
            }
            MocError::NoActiveFrames => write!(f, "MOC: no active frame in the reference"),
        }
    }
}

impl std::error::Error for MocError {}

/// `compare(x, y, apply_vad)` of `moc.py`: the MOC distance of `test`
/// from `reference`, both 16 kHz mono in [-1, 1]. 0 for identical signals.
/// The longer signal is cut to the length of the shorter one.
pub fn moc(reference: &[f32], test: &[f32], apply_vad: bool) -> Result<f64, MocError> {
    let n = reference.len().min(test.len());
    if n < MIN_SAMPLES {
        return Err(MocError::TooShort { samples: n });
    }
    if apply_vad && n / VAD_FRAME < 5 {
        return Err(MocError::VadTooShort { samples: n });
    }
    // `x[:num_samples] * 2**15` stays float32 (exact).
    let x: Vec<f32> = reference[..n].iter().map(|&v| v * 32768.0).collect();
    let y: Vec<f32> = test[..n].iter().map(|&v| v * 32768.0).collect();

    let tables = Tables::new();
    let psd_x = tables.power_spectrum(&x);
    let psd_y = tables.power_spectrum(&y);
    let num_frames = psd_x.len();

    // Mean power per band of the reference, spread across the bands
    // (`frequency_mask`), then over time.
    let f_mask = frequency_mask();
    let mut mask = vec![[0.0f64; NB]; num_frames];
    for (t, psd) in psd_x.iter().enumerate() {
        let mut be = [0.0f64; NB];
        for b in 0..NB {
            let bins = &psd[BAND_LIMITS[b]..BAND_LIMITS[b + 1]];
            be[b] = bins.iter().sum::<f64>() / bins.len() as f64;
        }
        for i in 0..NB {
            mask[t][i] = (0..NB).map(|j| be[j] * f_mask[i][j]).sum();
        }
    }
    for t in 1..num_frames {
        for i in 0..NB {
            mask[t][i] += 0.5 * mask[t - 1][i];
        }
    }

    // Masked spectra (bins above 7.5 kHz get no mask), the sum of two
    // frames, and the distortion per frame.
    let masked = |psd: &[f64; NBINS], t: usize| -> [f64; NBINS] {
        let mut out = *psd;
        for b in 0..NB {
            for k in BAND_LIMITS[b]..BAND_LIMITS[b + 1] {
                out[k] += 0.1 * mask[t][b];
            }
        }
        out
    };
    let mut ef = Vec::with_capacity(num_frames - 1);
    let (mut mx_prev, mut my_prev) = (masked(&psd_x[0], 0), masked(&psd_y[0], 0));
    for t in 1..num_frames {
        let (mx, my) = (masked(&psd_x[t], t), masked(&psd_y[t], t));
        let mut eb_sum = 0.0;
        for b in 0..NB {
            let mut sum = 0.0;
            for k in BAND_LIMITS[b]..BAND_LIMITS[b + 1] {
                let re = (my[k] + my_prev[k]) / (mx[k] + mx_prev[k]);
                let l = re.ln();
                sum += l * l;
            }
            eb_sum += sum / (BAND_LIMITS[b + 1] - BAND_LIMITS[b]) as f64;
        }
        ef.push(eb_sum / NB as f64);
        (mx_prev, my_prev) = (mx, my);
    }

    let keep = if apply_vad {
        let active = convert_mask(&vad_mask(&x), ef.len());
        active.iter().map(|&m| m > 1e-6).collect()
    } else {
        vec![true; ef.len()]
    };
    let (mut sum, mut count) = (0.0f64, 0usize);
    for (&e, &k) in ef.iter().zip(&keep) {
        if k {
            sum += e.abs().powi(3);
            count += 1;
        }
    }
    if count == 0 {
        return Err(MocError::NoActiveFrames);
    }
    Ok((sum / count as f64).powf(1.0 / 6.0))
}

/// [`moc`] for 16-bit samples, scaled by 1/32768 as `moc.py` reads WAV
/// files.
pub fn moc_i16(reference: &[i16], test: &[i16], apply_vad: bool) -> Result<f64, MocError> {
    let scale = |s: &[i16]| -> Vec<f32> { s.iter().map(|&v| f32::from(v) / 32768.0).collect() };
    moc(&scale(reference), &scale(test), apply_vad)
}

/// `D` of the relative test of the OSCE draft:
/// `(ref_moc - test_moc) / (0.1 + ref_moc)^0.5`, where `ref_moc` is the MOC
/// of the reference decoder and `test_moc` that of the tested decoder, both
/// against the clean input. Positive when the tested decoder is closer to
/// the clean input.
pub fn moc_relative(ref_moc: f64, test_moc: f64) -> f64 {
    (ref_moc - test_moc) / (0.1 + ref_moc).sqrt()
}

/// The window and the transform tables.
struct Tables {
    /// Periodic Hamming window of 160 samples, as `scipy.signal.get_window`
    /// builds it.
    window: [f64; WINDOW],
    /// `cos` and `sin` of `2*pi*i/160`.
    cos: [f64; WINDOW],
    sin: [f64; WINDOW],
}

impl Tables {
    fn new() -> Self {
        // scipy `general_cosine(161, [0.54, 1 - 0.54])`, last value dropped:
        // `fac = linspace(-pi, pi, 161)`, `w = 0.54 + 0.46*cos(fac)`.
        let pi = core::f64::consts::PI;
        let step = (pi - -pi) / WINDOW as f64;
        let mut window = [0.0; WINDOW];
        for (m, w) in window.iter_mut().enumerate() {
            let fac = m as f64 * step + -pi;
            *w = 0.54 + (1.0 - 0.54) * fac.cos();
        }
        let mut cos = [0.0; WINDOW];
        let mut sin = [0.0; WINDOW];
        for i in 0..WINDOW {
            let a = 2.0 * pi * i as f64 / WINDOW as f64;
            cos[i] = a.cos();
            sin[i] = a.sin();
        }
        Self { window, cos, sin }
    }

    /// `power_spectrum(x) + 100000`: `(len - 160 - 40) // 40` frames of 81
    /// bins.
    fn power_spectrum(&self, x: &[f32]) -> Vec<[f64; NBINS]> {
        let num_spectra = (x.len() - WINDOW - HOP) / HOP;
        let mut out = Vec::with_capacity(num_spectra);
        let mut frame = [0.0f64; WINDOW];
        for i in 0..num_spectra {
            for (m, v) in frame.iter_mut().enumerate() {
                *v = f64::from(x[i * HOP + m]) * self.window[m];
            }
            let mut psd = [0.0f64; NBINS];
            for (k, p) in psd.iter_mut().enumerate() {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                let mut idx = 0;
                for &v in &frame {
                    re += v * self.cos[idx];
                    im -= v * self.sin[idx];
                    idx += k;
                    if idx >= WINDOW {
                        idx -= WINDOW;
                    }
                }
                *p = re * re + im * im + 100000.0;
            }
            out.push(psd);
        }
        out
    }
}

/// `frequency_mask(17, 0.1, 0.03)`: the matrix `D·U`, where `U` spreads a
/// band up with `0.1^(i-j)` and `D` down with `0.03^(j-i)`.
fn frequency_mask() -> [[f64; NB]; NB] {
    let mut m = [[0.0; NB]; NB];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (i.max(j)..NB).map(|k| 0.03f64.powi((k - i) as i32) * 0.1f64.powi((k - j) as i32)).sum();
        }
    }
    m
}

/// `compute_vad_mask(x, 16000)` of `moc.py`: per sample, how active the
/// reference is (0 to 1). `x` is the reference times 2^15; the result has
/// one value per sample of the whole 20 ms frames.
fn vad_mask(x: &[f32]) -> Vec<f64> {
    let frames = x.len() / VAD_FRAME;
    // `np.sum(frames ** 2, axis=1)` in float32.
    let energy: Vec<f32> = (0..frames)
        .map(|f| {
            let sq: Vec<f32> = x[f * VAD_FRAME..][..VAD_FRAME].iter().map(|&v| v * v).collect();
            pairwise_sum_f32(&sq)
        })
        .collect();
    // `np.convolve(frame_energy, np.ones(5) / 5, mode='same')`.
    let smooth: Vec<f64> = (0..frames)
        .map(|f| {
            let lo = f.saturating_sub(2);
            let hi = (f + 3).min(frames);
            energy[lo..hi].iter().map(|&e| f64::from(e) * 0.2).sum()
        })
        .collect();
    // `frame_energy.max() * 10 ** (stop_db / 20)`: a float32 times a Python
    // float stays float32 in numpy 2. The `/20` on an energy is as written
    // in moc.py.
    let max = energy.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let threshold = f64::from(max * 10f64.powf(-70.0 / 20.0) as f32);
    // `vactive[frame_energy_smooth < max_threshold, :] = 0`.
    let active: Vec<bool> =
        smooth.iter().map(|s| s.partial_cmp(&threshold) != Some(core::cmp::Ordering::Less)).collect();

    // A normalised half-sine of 320 taps, applied with
    // `np.convolve(..., mode='same')`, which starts at index 159 of the
    // full convolution.
    let pi = core::f64::consts::PI;
    let mut filter: Vec<f64> =
        (0..VAD_FRAME).map(|m| (m as f64 * pi / (VAD_FRAME - 1) as f64).sin()).collect();
    let total: f64 = filter.iter().sum();
    for v in &mut filter {
        *v /= total;
    }
    let len = frames * VAD_FRAME;
    let offset = (VAD_FRAME - 1) / 2;
    (0..len)
        .map(|s| {
            // Full convolution index `s + offset`: the sum of
            // `active[m] * filter[s + offset - m]`.
            let k = s + offset;
            let lo = k.saturating_sub(VAD_FRAME - 1);
            let hi = (k + 1).min(len);
            (lo..hi).filter(|&m| active[m / VAD_FRAME]).map(|m| filter[k - m]).sum()
        })
        .collect()
}

/// `convert_mask(mask, num_frames, 160, 40)`: the mean of the mask over
/// each analysis window.
fn convert_mask(mask: &[f64], num_frames: usize) -> Vec<f64> {
    let num_samples = WINDOW + (num_frames - 1) * HOP;
    let mut padded = mask[..mask.len().min(num_samples)].to_vec();
    padded.resize(num_samples, 0.0);
    (0..num_frames).map(|i| padded[i * HOP..][..WINDOW].iter().sum::<f64>() / WINDOW as f64).collect()
}

/// numpy's pairwise summation of `float32` values (`pairwise_sum_FLOAT`):
/// below 8 values a plain loop, up to 128 values eight running sums, above
/// that the two halves (cut at a multiple of 8) summed separately.
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0f32, |acc, &v| acc + v)
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[i..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_signals_give_zero() {
        let x: Vec<f32> = (0..4000).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        assert_eq!(moc(&x, &x, false), Ok(0.0));
        assert_eq!(moc(&x, &x, true), Ok(0.0));
    }

    #[test]
    fn short_inputs_are_errors() {
        let x = vec![0.1f32; 279];
        assert_eq!(moc(&x, &x, false), Err(MocError::TooShort { samples: 279 }));
        let x = vec![0.1f32; 1599];
        assert_eq!(moc(&x, &x, true), Err(MocError::VadTooShort { samples: 1599 }));
        assert!(moc(&x, &x, false).is_ok());
    }

    #[test]
    fn silent_reference_counts_as_active() {
        let x = vec![0.0f32; 3200];
        let y = vec![0.001f32; 3200];
        // The threshold of a silent reference is 0, and no smoothed energy
        // is below 0, so every frame is active.
        assert!(moc(&x, &y, true).is_ok());
        assert_eq!(moc(&x, &x, true), Ok(0.0));
    }

    #[test]
    fn relative_distance_signs() {
        assert!(moc_relative(0.5, 0.4) > 0.0);
        assert!(moc_relative(0.5, 0.6) < 0.0);
        assert_eq!(moc_relative(0.3, 0.3), 0.0);
    }

    #[test]
    fn frequency_mask_matches_two_passes() {
        // D·U equals spreading up with 0.1, then down with 0.03.
        let m = frequency_mask();
        for j in 0..NB {
            let mut v = [0.0f64; NB];
            v[j] = 1.0;
            for b in 1..NB {
                v[b] += 0.1 * v[b - 1];
            }
            for b in (0..NB - 1).rev() {
                v[b] += 0.03 * v[b + 1];
            }
            for i in 0..NB {
                assert!((m[i][j] - v[i]).abs() <= 1e-15 * v[i].abs().max(1e-300), "{i} {j}");
            }
        }
    }
}
