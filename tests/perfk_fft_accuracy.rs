//! Accuracy of the FFT and of the MDCT against double-precision references.
//!
//! The references compute the same transforms with `f64` arithmetic, exact
//! cosines and a direct Fourier sum instead of the butterflies. The measure
//! is the signal-to-noise ratio of the `f32` result: reference power divided
//! by error power, in decibels. Single-precision rounding gives about 140 dB.
//!
//! The limits below sit a few decibels under what the code measures today.
//! Run with `--nocapture` to see the measured values.

use opus_rs::kiss_fft::{KissCpx, KissFftState, opus_fft_impl, opus_fft_impl_scalar};
use opus_rs::mdct::MdctLookup;
use opus_rs::modes::default_mode;
use std::f64::consts::PI;

/// Deterministic pseudo-random numbers, so every run tests the same data.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

/// Signal-to-noise ratio in decibels of `got` against `want`.
fn snr_db(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let signal: f64 = want.iter().map(|w| w * w).sum();
    let noise: f64 = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w) * (g as f64 - w))
        .sum();
    10.0 * (signal / noise).log10()
}

/// Forward Fourier sum without scaling: `out[k] = sum x[i] * e^(-2 pi i k i / n)`.
fn dft(x: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = (0.0, 0.0);
            for (i, &(re, im)) in x.iter().enumerate() {
                // Reduce k * i first, so the angle stays small and exact.
                let phase = -2.0 * PI * ((k * i) % n) as f64 / n as f64;
                let (s, c) = phase.sin_cos();
                acc.0 += re * c - im * s;
                acc.1 += re * s + im * c;
            }
            acc
        })
        .collect()
}

/// The rotation factors of an MDCT of `n2` coefficients: `(t0[i], t1[i])`.
fn rotation(n2: usize, i: usize) -> (f64, f64) {
    let n = 2 * n2;
    let n4 = n2 / 2;
    (
        (2.0 * PI * (i as f64 + 0.125) / n as f64).cos(),
        (2.0 * PI * ((n4 + i) as f64 + 0.125) / n as f64).cos(),
    )
}

/// Forward MDCT of `n2` coefficients in double precision.
fn mdct_forward_f64(input: &[f32], window: &[f32], overlap: usize, n2: usize) -> Vec<f64> {
    let n4 = n2 / 2;
    let q = overlap / 4;
    let inp = |i: usize| input[i] as f64;
    let w = |i: usize| window[i] as f64;

    // Window and fold, then rotate and scale.
    let mut y = Vec::with_capacity(n4);
    for i in 0..n4 {
        let (re, im) = if i < q {
            let u = 2 * q + 2 * i;
            let d = 2 * q - 1 - 2 * i;
            (
                inp(n2 + u) * w(d) + inp(n2 + d) * w(u),
                inp(u) * w(u) - inp(d) * w(d),
            )
        } else if i < n4 - q {
            (inp(n2 - 1 + 2 * q - 2 * i), inp(2 * q + 2 * i))
        } else {
            let k = i - (n4 - q);
            let u = 2 * k;
            let d = 4 * q - 1 - 2 * k;
            (
                -inp(u) * w(u) + inp(d) * w(d),
                inp(n2 + u) * w(d) + inp(n2 + d) * w(u),
            )
        };
        let (t0, t1) = rotation(n2, i);
        let scale = 1.0 / n4 as f64;
        y.push(((re * t0 - im * t1) * scale, (im * t0 + re * t1) * scale));
    }

    let f = dft(&y);

    let mut out = vec![0.0; n2];
    for (i, &(fr, fi)) in f.iter().enumerate() {
        let (t0, t1) = rotation(n2, i);
        out[2 * i] = fi * t1 - fr * t0;
        out[n2 - 1 - 2 * i] = fr * t1 + fi * t0;
    }
    out
}

/// Inverse MDCT of `n2` coefficients in double precision. `out` holds the
/// tail of the block before in its first `overlap / 2` values.
fn mdct_backward_f64(input: &[f32], out: &mut [f64], window: &[f32], overlap: usize, n2: usize) {
    let n4 = n2 / 2;
    let overlap2 = overlap / 2;

    let mut y = Vec::with_capacity(n4);
    for i in 0..n4 {
        let x1 = input[2 * i] as f64;
        let x2 = input[n2 - 1 - 2 * i] as f64;
        let (t0, t1) = rotation(n2, i);
        y.push((x1 * t0 - x2 * t1, x2 * t0 + x1 * t1));
    }

    let f = dft(&y);

    for (i, &(fr, fi)) in f.iter().enumerate() {
        let (t0, t1) = rotation(n2, i);
        out[overlap2 + 2 * i] = fi * t0 + fr * t1;
        out[overlap2 + n2 - 1 - 2 * i] = fi * t1 - fr * t0;
    }

    for i in 0..overlap2 {
        let x1 = out[overlap - 1 - i];
        let x2 = out[i];
        let wp1 = window[i] as f64;
        let wp2 = window[overlap - 1 - i] as f64;
        out[i] = x2 * wp2 - x1 * wp1;
        out[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
    }
}

/// Lowest signal-to-noise ratio the FFT may have, in decibels.
const FFT_MIN_SNR_DB: f64 = 130.0;
/// Lowest signal-to-noise ratio the MDCT may have, in decibels.
const MDCT_MIN_SNR_DB: f64 = 125.0;

#[test]
fn fft_accuracy_against_double_precision() {
    let base = KissFftState::new(480).unwrap();
    let mut rng = Lcg(0xacc0_0001);
    for nfft in [60usize, 120, 240, 480] {
        let sub;
        let st = if nfft == 480 {
            &base
        } else {
            sub = KissFftState::new_sub(&base, nfft).unwrap();
            &sub
        };
        let mut worst = f64::INFINITY;
        let mut worst_scalar = f64::INFINITY;
        for _ in 0..20 {
            let x: Vec<(f32, f32)> = (0..nfft).map(|_| (rng.unit(), rng.unit())).collect();
            let want = dft(&x.iter().map(|&(r, i)| (r as f64, i as f64)).collect::<Vec<_>>());
            let want: Vec<f64> = want.iter().flat_map(|&(r, i)| [r, i]).collect();

            // The butterflies expect their input in bit-reversed order.
            let mut buf = vec![KissCpx::default(); nfft];
            for (i, &(r, im)) in x.iter().enumerate() {
                buf[st.bitrev[i] as usize] = KissCpx::new(r, im);
            }
            let mut buf_scalar = buf.clone();
            opus_fft_impl(st, &mut buf);
            opus_fft_impl_scalar(st, &mut buf_scalar);

            let flat = |b: &[KissCpx]| b.iter().flat_map(|c| [c.r, c.i]).collect::<Vec<f32>>();
            worst = worst.min(snr_db(&flat(&buf), &want));
            worst_scalar = worst_scalar.min(snr_db(&flat(&buf_scalar), &want));
        }
        println!("fft {nfft:3} points: {worst:.1} dB, portable {worst_scalar:.1} dB");
        assert!(worst >= FFT_MIN_SNR_DB, "fft {nfft}: {worst:.1} dB");
        assert!(
            worst_scalar >= FFT_MIN_SNR_DB,
            "portable fft {nfft}: {worst_scalar:.1} dB"
        );
    }
}

#[test]
fn mdct_accuracy_against_double_precision() {
    let mode = default_mode();
    let overlap = mode.overlap;
    let window = mode.window;
    let mut rng = Lcg(0xacc0_0002);
    for shift in 0..=mode.max_lm {
        let n2 = (mode.mdct.n >> shift) / 2;
        let mut worst = [f64::INFINITY; 4];
        for _ in 0..10 {
            // Forward.
            let input: Vec<f32> = (0..n2 + overlap).map(|_| rng.unit() * 32768.0).collect();
            let want = mdct_forward_f64(&input, window, overlap, n2);
            let mut got = vec![0f32; n2];
            let mut got_scalar = vec![0f32; n2];
            mode.mdct.forward(&input, &mut got, window, overlap, shift, 1);
            mode.mdct
                .forward_scalar(&input, &mut got_scalar, window, overlap, shift, 1);
            worst[0] = worst[0].min(snr_db(&got, &want));
            worst[1] = worst[1].min(snr_db(&got_scalar, &want));

            // Inverse.
            let coeffs: Vec<f32> = (0..n2).map(|_| rng.unit() * 32768.0).collect();
            let tail: Vec<f32> = (0..overlap / 2 + n2).map(|_| rng.unit() * 32768.0).collect();
            let mut want: Vec<f64> = tail.iter().map(|&v| v as f64).collect();
            mdct_backward_f64(&coeffs, &mut want, window, overlap, n2);
            let mut got = tail.clone();
            let mut got_scalar = tail.clone();
            mode.mdct
                .backward(&coeffs, &mut got, window, overlap, shift, 1);
            mode.mdct
                .backward_scalar(&coeffs, &mut got_scalar, window, overlap, shift, 1);
            worst[2] = worst[2].min(snr_db(&got, &want));
            worst[3] = worst[3].min(snr_db(&got_scalar, &want));
        }
        println!(
            "mdct {n2:3} coefficients: forward {:.1} dB (portable {:.1}), inverse {:.1} dB (portable {:.1})",
            worst[0], worst[1], worst[2], worst[3]
        );
        for (what, snr) in ["forward", "portable forward", "inverse", "portable inverse"]
            .iter()
            .zip(worst)
        {
            assert!(snr >= MDCT_MIN_SNR_DB, "{what} mdct {n2}: {snr:.1} dB");
        }
    }
}

/// A transform size the codec does not use, with another overlap.
#[test]
fn mdct_accuracy_other_size() {
    let n = 256;
    let n2 = n / 2;
    let overlap = 64;
    let l = MdctLookup::new(n, 0);
    let window: Vec<f32> = (0..overlap)
        .map(|i| {
            let x = (i as f64 + 0.5) / overlap as f64;
            (0.5 * PI * (0.5 * PI * x).sin().powi(2)).sin() as f32
        })
        .collect();
    let mut rng = Lcg(0xacc0_0003);
    let input: Vec<f32> = (0..n2 + overlap).map(|_| rng.unit()).collect();
    let want = mdct_forward_f64(&input, &window, overlap, n2);
    let mut got = vec![0f32; n2];
    l.forward(&input, &mut got, &window, overlap, 0, 1);
    let snr = snr_db(&got, &want);
    println!("mdct 128 coefficients, overlap 64: forward {snr:.1} dB");
    assert!(snr >= MDCT_MIN_SNR_DB, "{snr:.1} dB");
}
