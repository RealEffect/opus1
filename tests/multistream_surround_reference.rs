//! The surround masking analysis against a reference written for this test.
//!
//! libopus does not give access to the mask its multistream encoder
//! computes (`bandSMR` in `opus_multistream_encode_native`). The reference
//! here is `surround_analysis` of opus_multistream_encoder.c written again
//! in the simplest form:
//!
//! - every value is an `f64`;
//! - the MDCT is the sum from its definition, one cosine per input sample
//!   and coefficient;
//! - the steps after the MDCT follow the C source line by line.
//!
//! The analysis under test works in `f32` with the fast MDCT of the codec.
//! The two must agree within rounding.
// The multistream API needs the `std` feature.
#![cfg(feature = "std")]

// The reference keeps the constants and the loops of the C source as they
// are written there.
#![allow(clippy::excessive_precision, clippy::needless_range_loop)]

mod ms_support;

use ms_support::{noise, silence, tones, voiced};
use opus1::internals::modes;
use opus1::internals::multistream::surround::SurroundProbe;
use std::f64::consts::PI;

const NB_BANDS: usize = 21;
/// Overlap of the CELT window (`celt_mode->overlap`).
const OVERLAP: usize = 120;
/// Shortest MDCT of the CELT mode (`shortMdctSize`), 2.5 ms at 48 kHz.
const SHORT_MDCT: usize = 120;
/// `maxLM` of the CELT mode: the longest MDCT is `SHORT_MDCT << 3`.
const MAX_LM: usize = 3;
/// Band edges in units of 200 Hz (`eband5ms`, modes.c).
const EBANDS: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];
/// Mean level of each band in log2 units (`eMeans`, quant_bands.c).
const E_MEANS: [f32; NB_BANDS] = [
    6.4375, 6.25, 5.75, 5.3125, 5.0625, 4.8125, 4.5, 4.375, 4.875, 4.6875, 4.5625, 4.4375, 4.875,
    4.625, 4.3125, 4.5, 4.375, 4.625, 4.75, 4.4375, 3.75,
];
/// First pre-emphasis coefficient of the CELT mode (static_modes_float.h).
const PREEMPH: f32 = 0.850_006_1;

/// `resampling_factor` (celt.c).
fn resampling_factor(rate: i32) -> usize {
    match rate {
        48000 => 1,
        24000 => 2,
        16000 => 3,
        12000 => 4,
        8000 => 6,
        _ => panic!("no CELT rate: {rate}"),
    }
}

/// `channel_pos`: 0 not mixed, 1 left, 2 centre, 3 right.
fn channel_pos(channels: usize) -> [usize; 8] {
    let mut pos = [0; 8];
    if channels == 4 {
        pos[0] = 1;
        pos[1] = 3;
        pos[2] = 1;
        pos[3] = 3;
    } else if channels == 3 || channels == 5 || channels == 6 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 0;
    } else if channels == 7 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 2;
        pos[6] = 0;
    } else if channels == 8 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 1;
        pos[6] = 3;
        pos[7] = 0;
    }
    pos
}

/// `logSum`. Its comment in the C source reads "a rough approximation of
/// log2(2^a + 2^b)". The levels are log2 of amplitudes, and the table holds
/// `0.5 * log2(1 + 4^-d)`, so what it approximates is
/// `log2(sqrt(4^a + 4^b))`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn log_sum(a: f64, b: f64) -> f64 {
    const DIFF_TABLE: [f32; 17] = [
        0.5000000, 0.2924813, 0.1609640, 0.0849625, 0.0437314, 0.0221971, 0.0111839, 0.0056136,
        0.0028123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (max, diff) = if a > b { (a, a - b) } else { (b, b - a) };
    if !(diff < 8.0) {
        return max;
    }
    let low = (2.0 * diff).floor() as usize;
    let frac = 2.0 * diff - low as f64;
    max + DIFF_TABLE[low] as f64 + frac * (DIFF_TABLE[low + 1] as f64 - DIFF_TABLE[low] as f64)
}

/// Window of the CELT mode (modes.c): the rising half, `OVERLAP` values.
fn celt_window() -> Vec<f64> {
    (0..OVERLAP)
        .map(|i| {
            let s = (0.5 * PI * (i as f64 + 0.5) / OVERLAP as f64).sin();
            (0.5 * PI * s * s).sin()
        })
        .collect()
}

/// `clt_mdct_forward` from its definition (mdct.h: "a forward MDCT, scaled
/// by 4/N"). `input` holds `n2 + OVERLAP` samples; the result has `n2`
/// coefficients.
///
/// The transform works on a block of `N = 2 * n2` samples. `input` sits in
/// the middle of the block, and the rest of the block is zero. The window
/// rises over the first `OVERLAP` input samples and falls over the last
/// `OVERLAP`. Coefficient `k` is
/// `4/N * sum over n of x[n] * cos(2 pi / N * (n + 1/2 + N/4) * (k + 1/2))`.
fn mdct_direct(input: &[f64], n2: usize, window: &[f64]) -> Vec<f64> {
    assert_eq!(input.len(), n2 + OVERLAP);
    let n = 2 * n2;
    let first = (n2 - OVERLAP) / 2;
    let windowed: Vec<f64> = input
        .iter()
        .enumerate()
        .map(|(j, &v)| {
            if j < OVERLAP {
                v * window[j]
            } else if j < n2 {
                v
            } else {
                v * window[n2 + OVERLAP - 1 - j]
            }
        })
        .collect();
    // The angle is 2 pi * (2 n + 1 + n2) * (2 k + 1) / (8 n2): a table of
    // 8 n2 cosines holds every value that occurs.
    let period = 8 * n2;
    let cosines: Vec<f64> = (0..period)
        .map(|m| (2.0 * PI * m as f64 / period as f64).cos())
        .collect();
    (0..n2)
        .map(|k| {
            let mut sum = 0.0;
            for (j, &x) in windowed.iter().enumerate() {
                let m = (2 * (first + j) + 1 + n2) * (2 * k + 1) % period;
                sum += x * cosines[m];
            }
            4.0 / n as f64 * sum
        })
        .collect()
}

/// `surround_analysis` in `f64`, with the two memories it keeps per channel.
struct Reference {
    channels: usize,
    rate: i32,
    window: Vec<f64>,
    /// `mem`: `OVERLAP` samples per channel.
    mem: Vec<f64>,
    /// `preemph_mem`: one value per channel.
    preemph_mem: Vec<f64>,
}

impl Reference {
    fn new(channels: usize, rate: i32) -> Self {
        Self {
            channels,
            rate,
            window: celt_window(),
            mem: vec![0.0; channels * OVERLAP],
            preemph_mem: vec![0.0; channels],
        }
    }

    /// One frame of `len` samples per channel, interleaved in `pcm`.
    /// Returns `bandLogE`: 21 values per channel.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn run(&mut self, pcm: &[f32], len: usize) -> Vec<f64> {
        let channels = self.channels;
        let overlap = OVERLAP;
        let mut band_log_e = vec![0f64; NB_BANDS * channels];

        let upsample = resampling_factor(self.rate);
        let frame_size = len * upsample;

        // LM = log2(frame_size / 120)
        let mut lm = 0;
        while lm < MAX_LM {
            if SHORT_MDCT << lm == frame_size {
                break;
            }
            lm += 1;
        }
        let freq_size = SHORT_MDCT << lm;

        let pos = channel_pos(channels);
        let mut mask_log_e = [[-28.0f64; NB_BANDS]; 3];

        for c in 0..channels {
            let nb_frames = frame_size / freq_size;
            assert_eq!(nb_frames * freq_size, frame_size);
            let mut input = vec![0f64; frame_size + overlap];
            input[..overlap].copy_from_slice(&self.mem[c * overlap..(c + 1) * overlap]);
            // copy_channel_in
            let x: Vec<f64> = (0..len).map(|i| pcm[i * channels + c] as f64).collect();
            // celt_preemphasis(x, in+overlap, frame_size, 1, upsample, preemph, preemph_mem+c, 0)
            {
                let inp = &mut input[overlap..];
                let coef0 = PREEMPH as f64;
                let mut m = self.preemph_mem[c];
                for (i, &v) in x.iter().enumerate() {
                    inp[i * upsample] = 32768.0 * v;
                }
                for v in inp.iter_mut() {
                    let x = *v;
                    *v = x - m;
                    m = coef0 * x;
                }
                self.preemph_mem[c] = m;
            }
            {
                let sum: f64 = input.iter().map(|v| v * v).sum();
                if !(sum < 1e18f32 as f64) || sum.is_nan() {
                    input.fill(0.0);
                    self.preemph_mem[c] = 0.0;
                }
            }
            let mut band_e = [0f64; NB_BANDS];
            for frame in 0..nb_frames {
                let start = freq_size * frame;
                let mut freq = mdct_direct(
                    &input[start..start + freq_size + overlap],
                    freq_size,
                    &self.window,
                );
                if upsample != 1 {
                    let bound = freq_size / upsample;
                    for v in &mut freq[..bound] {
                        *v *= upsample as f64;
                    }
                    for v in &mut freq[bound..] {
                        *v = 0.0;
                    }
                }
                // compute_band_energies(celt_mode, freq, tmpE, 21, 1, LM)
                for i in 0..NB_BANDS {
                    let band = &freq[EBANDS[i] << lm..EBANDS[i + 1] << lm];
                    let sum = 1e-27f32 as f64 + band.iter().map(|v| v * v).sum::<f64>();
                    let tmp_e = sum.sqrt();
                    // If we have multiple frames, take the max energy.
                    if tmp_e > band_e[i] {
                        band_e[i] = tmp_e;
                    }
                }
            }
            // amp2Log2(celt_mode, 21, 21, bandE, bandLogE+21*c, 1)
            for i in 0..NB_BANDS {
                band_log_e[21 * c + i] = band_e[i].log2() - E_MEANS[i] as f64;
            }
            // Apply spreading function with -6 dB/band going up and -12 dB/band going down.
            for i in 1..21 {
                band_log_e[21 * c + i] =
                    f64::max(band_log_e[21 * c + i], band_log_e[21 * c + i - 1] - 1.0);
            }
            for i in (0..=19).rev() {
                band_log_e[21 * c + i] =
                    f64::max(band_log_e[21 * c + i], band_log_e[21 * c + i + 1] - 2.0);
            }
            if pos[c] == 1 {
                for i in 0..21 {
                    mask_log_e[0][i] = log_sum(mask_log_e[0][i], band_log_e[21 * c + i]);
                }
            } else if pos[c] == 3 {
                for i in 0..21 {
                    mask_log_e[2][i] = log_sum(mask_log_e[2][i], band_log_e[21 * c + i]);
                }
            } else if pos[c] == 2 {
                for i in 0..21 {
                    mask_log_e[0][i] = log_sum(mask_log_e[0][i], band_log_e[21 * c + i] - 0.5);
                    mask_log_e[2][i] = log_sum(mask_log_e[2][i], band_log_e[21 * c + i] - 0.5);
                }
            }
            self.mem[c * overlap..(c + 1) * overlap]
                .copy_from_slice(&input[frame_size..frame_size + overlap]);
        }
        for i in 0..21 {
            mask_log_e[1][i] = f64::min(mask_log_e[0][i], mask_log_e[2][i]);
        }
        let channel_offset = 0.5 * (2.0 / (channels - 1) as f64).log2();
        for mask in &mut mask_log_e {
            for v in mask.iter_mut() {
                *v += channel_offset;
            }
        }
        for c in 0..channels {
            if pos[c] != 0 {
                let mask = &mask_log_e[pos[c] - 1];
                for i in 0..21 {
                    band_log_e[21 * c + i] -= mask[i];
                }
            } else {
                for i in 0..21 {
                    band_log_e[21 * c + i] = 0.0;
                }
            }
        }
        band_log_e
    }
}

/// The largest differences between the analysis under test and the
/// reference, in log2 units (1.0 is 6 dB).
struct Difference {
    /// Over bands 0 to 19.
    low: f64,
    /// Over band 20, the highest band (15.6 to 20 kHz).
    top: f64,
    /// Frame, channel and band of the largest difference of all.
    at: (usize, usize, usize),
}

impl std::fmt::Display for Difference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (frame, channel, band) = self.at;
        write!(
            f,
            "bands 0-19 {:.1e}, band 20 {:.1e} (largest at frame {frame}, channel {channel}, band {band})",
            self.low, self.top
        )
    }
}

/// Runs the analysis under test and the reference over `pcm`. Frame `f` has
/// `frame_sizes[f % frame_sizes.len()]` samples per channel. Both keep
/// their memories from frame to frame.
fn compare(channels: usize, rate: i32, frame_sizes: &[usize], pcm: &[f32]) -> Difference {
    let mut probe = SurroundProbe::new(channels, rate);
    let mut reference = Reference::new(channels, rate);
    let mut worst = Difference {
        low: 0.0,
        top: 0.0,
        at: (0, 0, 0),
    };
    let (mut start, mut f) = (0, 0);
    loop {
        let frame_size = frame_sizes[f % frame_sizes.len()];
        let end = start + frame_size * channels;
        if end > pcm.len() {
            break;
        }
        let frame = &pcm[start..end];
        let mut got = vec![f32::NAN; NB_BANDS * channels];
        assert!(probe.run(frame, frame_size, &mut got), "frame {f}: no mask");
        let want = reference.run(frame, frame_size);
        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let (c, b) = (k / NB_BANDS, k % NB_BANDS);
            assert!(g.is_finite(), "frame {f} channel {c} band {b}: {g}");
            let d = (g as f64 - w).abs();
            if d > worst.low.max(worst.top) {
                worst.at = (f, c, b);
            }
            if b < 20 {
                worst.low = worst.low.max(d);
            } else {
                worst.top = worst.top.max(d);
            }
        }
        start = end;
        f += 1;
    }
    assert!(f >= 2, "need at least two frames to use the memories");
    worst
}

/// Largest difference allowed where every band holds signal.
///
/// The analysis under test rounds to `f32` in the MDCT, which leaves noise
/// about 125 dB below the strongest coefficient. A band 60 dB below the
/// strongest one is therefore off by up to 5e-4 of its level, which is
/// 7e-4 in log2 units. The largest difference measured in this file is
/// 6e-4, at 16 kHz with 2.5 ms frames.
const TOLERANCE: f64 = 2e-3;

/// Largest difference allowed in band 20 for a signal with nothing above a
/// few kHz. The band then holds only what leaks from the low bands, and
/// the rounding noise of the `f32` MDCT is about 1 % of that. Measured: up
/// to 1.4e-2.
const TOLERANCE_EMPTY_BAND: f64 = 5e-2;

/// Adds `b` to `a`, sample by sample.
fn mix(mut a: Vec<f32>, b: &[f32]) -> Vec<f32> {
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
    a
}

/// Multiplies channel `c` of an interleaved signal by `gain`.
fn scale_channel(pcm: &mut [f32], channels: usize, c: usize, gain: f32) {
    for v in pcm.iter_mut().skip(c).step_by(channels) {
        *v *= gain;
    }
}

/// Noise whose level falls by 6 dB from each channel to the next, starting
/// at `amplitude` on channel 0.
fn stepped_noise(channels: usize, samples: usize, amplitude: f32, seed: u32) -> Vec<f32> {
    let mut pcm = noise(channels, samples, amplitude, seed);
    for c in 0..channels {
        scale_channel(&mut pcm, channels, c, 1.0 / (1u32 << c) as f32);
    }
    pcm
}

/// A test signal: its name, its samples, and whether its top band is empty
/// (see `TOLERANCE_EMPTY_BAND`).
struct Signal {
    name: &'static str,
    pcm: Vec<f32>,
    empty_top_band: bool,
}

/// The test signals for one layout.
fn signals(rate: i32, channels: usize, samples: usize) -> Vec<Signal> {
    // One channel 40 dB above the others.
    let mut loud_left = noise(channels, samples, 0.005, 11);
    scale_channel(&mut loud_left, channels, 0, 100.0);
    // The centre 60 dB above the others.
    let mut loud_centre = noise(channels, samples, 0.0005, 12);
    scale_channel(&mut loud_centre, channels, 1, 1000.0);
    // One channel silent, the others at different levels.
    let mut one_silent = stepped_noise(channels, samples, 0.2, 13);
    scale_channel(&mut one_silent, channels, 2, 0.0);
    let signal = |name, pcm, empty_top_band| Signal {
        name,
        pcm,
        empty_top_band,
    };
    vec![
        signal(
            "one tone per channel",
            tones(rate, channels, 1, samples),
            true,
        ),
        signal("noise", noise(channels, samples, 0.3, 7), false),
        signal("loud left, quiet others", loud_left, false),
        signal("loud centre, quiet others", loud_centre, false),
        signal("one silent channel", one_silent, false),
        signal("silence", silence(channels, samples), false),
        signal("speech-like", voiced(rate, channels, samples), true),
        signal(
            "tones over quiet noise",
            mix(
                tones(rate, channels, 1, samples),
                &noise(channels, samples, 0.001, 9),
            ),
            false,
        ),
    ]
}

/// Checks a `Difference` against the tolerances.
fn check(what: &str, d: &Difference, empty_top_band: bool) {
    println!("{what}: {d}");
    assert!(d.low < TOLERANCE, "{what}: {d}");
    let top = if empty_top_band {
        TOLERANCE_EMPTY_BAND
    } else {
        TOLERANCE
    };
    assert!(d.top < top, "{what}: {d}");
}

#[test]
fn reference_mdct_is_the_mdct_of_the_codec() {
    // The reference MDCT against the fast MDCT, for the four sizes.
    let mode = modes::default_mode();
    let window = celt_window();
    let input32 = noise(1, 960 + OVERLAP, 1000.0, 3);
    let input64: Vec<f64> = input32.iter().map(|&v| v as f64).collect();
    for lm in 0..=MAX_LM {
        let n2 = SHORT_MDCT << lm;
        let want = mdct_direct(&input64[..n2 + OVERLAP], n2, &window);
        let mut got = vec![0f32; n2];
        mode.mdct.forward(
            &input32[..n2 + OVERLAP],
            &mut got,
            mode.window,
            OVERLAP,
            MAX_LM - lm,
            1,
        );
        let peak = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()));
        println!("MDCT of {n2} coefficients: peak {peak:.2}, largest difference {worst:.2e}");
        assert!(
            worst < 2e-6 * peak,
            "n2 {n2}: difference {worst:e}, peak {peak}"
        );
    }
    // The tables of the codec are the ones the reference uses.
    for (i, (&t, &w)) in mode.window.iter().zip(&window).enumerate() {
        assert!((t as f64 - w).abs() < 1e-7, "window[{i}]: {t} and {w}");
    }
    assert_eq!(mode.preemph[0], PREEMPH);
    assert_eq!(mode.e_means[..NB_BANDS], E_MEANS);
    assert!(
        mode.e_bands
            .iter()
            .zip(&EBANDS)
            .all(|(&a, &b)| a as usize == b)
    );
    assert_eq!(
        (mode.overlap, mode.short_mdct_size, mode.max_lm),
        (OVERLAP, SHORT_MDCT, MAX_LM)
    );
}

#[test]
fn mask_of_5_1_and_7_1_at_48_khz_matches_the_reference() {
    let (rate, frame_size) = (48_000, 960);
    for channels in [6usize, 8] {
        for s in signals(rate, channels, 4 * frame_size) {
            let d = compare(channels, rate, &[frame_size], &s.pcm);
            check(
                &format!("{channels} channels, {}", s.name),
                &d,
                s.empty_top_band,
            );
            // At 48 kHz with 20 ms frames the agreement is ten times closer.
            assert!(d.low < 1e-4, "{channels} channels, {}: {d}", s.name);
        }
    }
}

#[test]
fn mask_matches_the_reference_at_every_rate_and_frame_size() {
    // 2.5, 5, 10, 20, 40 and 60 ms at 8, 12, 16, 24 and 48 kHz.
    for channels in [6usize, 8] {
        for rate in [8_000, 12_000, 16_000, 24_000, 48_000] {
            for frame_400 in [1usize, 2, 4, 8, 16, 24] {
                let frame_size = rate as usize / 400 * frame_400;
                let samples = 3 * frame_size;
                let what = format!(
                    "{channels} channels, {rate} Hz, {} ms",
                    frame_400 as f64 * 2.5
                );
                // Tones on a noise floor that differs from channel to channel.
                let pcm = mix(
                    tones(rate, channels, 1, samples),
                    &stepped_noise(channels, samples, 0.01, 5),
                );
                check(&what, &compare(channels, rate, &[frame_size], &pcm), false);
                // A loud channel next to quiet ones.
                let mut pcm = noise(channels, samples, 0.003, 17);
                scale_channel(&mut pcm, channels, 2, 100.0);
                check(&what, &compare(channels, rate, &[frame_size], &pcm), false);
            }
        }
    }
}

#[test]
fn mask_of_the_other_surround_layouts_matches_the_reference() {
    // 3, 4, 5 and 7 channels place their channels differently in the mix.
    for channels in [3usize, 4, 5, 7] {
        for (rate, frame_size) in [(48_000, 960usize), (48_000, 240), (16_000, 320)] {
            for s in signals(rate, channels, 3 * frame_size) {
                // With shorter frames or a lower rate, a signal without a
                // noise floor leaves more bands than the top one empty.
                if s.empty_top_band && (rate, frame_size) != (48_000, 960) {
                    continue;
                }
                let d = compare(channels, rate, &[frame_size], &s.pcm);
                let what = format!(
                    "{channels} channels, {rate} Hz, frame {frame_size}, {}",
                    s.name
                );
                check(&what, &d, s.empty_top_band);
            }
        }
    }
}

#[test]
fn mask_matches_the_reference_for_long_and_changing_frames() {
    let (channels, rate) = (6usize, 48_000);
    // 80, 100 and 120 ms: four, five and six MDCTs of 20 ms.
    for frame_size in [3840usize, 4800, 5760] {
        let pcm = stepped_noise(channels, 2 * frame_size, 0.2, 23);
        let d = compare(channels, rate, &[frame_size], &pcm);
        check(&format!("{} ms", frame_size / 48), &d, false);
    }
    // The memories do not depend on the frame size: change it every frame.
    let sizes = [960usize, 120, 480, 2880, 240, 1920, 120, 960];
    let pcm = mix(
        tones(rate, channels, 1, 2 * 7680),
        &stepped_noise(channels, 2 * 7680, 0.02, 29),
    );
    check(
        "changing frame sizes",
        &compare(channels, rate, &sizes, &pcm),
        false,
    );
    let (channels, rate) = (8usize, 16_000);
    let sizes = [320usize, 40, 960, 80, 160, 640];
    let pcm = stepped_noise(channels, 3 * 2200, 0.2, 31);
    check(
        "changing frame sizes at 16 kHz",
        &compare(channels, rate, &sizes, &pcm),
        false,
    );
}

#[test]
fn a_channel_with_absurd_samples_is_dropped_as_in_libopus() {
    // libopus clears a channel whose energy is not below 1e18 or is NaN, and
    // with it the pre-emphasis memory. The window memory then holds zeros.
    let (channels, rate, frame_size) = (6usize, 48_000, 960usize);
    for (name, bad) in [
        ("NaN", f32::NAN),
        ("infinity", f32::INFINITY),
        ("-infinity", f32::NEG_INFINITY),
        ("the largest f32", f32::MAX),
        ("1e6", 1e6),
    ] {
        let mut pcm = stepped_noise(channels, 4 * frame_size, 0.2, 37);
        // One bad sample in frame 1, on a left channel.
        pcm[(frame_size + 100) * channels + 3] = bad;
        let d = compare(channels, rate, &[frame_size], &pcm);
        check(&format!("one sample is {name}"), &d, false);
    }
    // A whole frame of NaN on every channel still gives numbers.
    let mut pcm = stepped_noise(channels, 3 * frame_size, 0.2, 41);
    pcm[frame_size * channels..2 * frame_size * channels].fill(f32::NAN);
    check(
        "a frame of NaN",
        &compare(channels, rate, &[frame_size], &pcm),
        false,
    );
    // Loud but sane input is analysed like any other: 40 times full scale
    // gives an energy of about 1e15.
    let mut pcm = noise(channels, 3 * frame_size, 0.1, 43);
    scale_channel(&mut pcm, channels, 0, 400.0);
    let d = compare(channels, rate, &[frame_size], &pcm);
    check("one channel at 40 times full scale", &d, false);
    let mut probe = SurroundProbe::new(channels, rate);
    let mut mask = vec![0f32; NB_BANDS * channels];
    assert!(probe.run(&pcm, frame_size, &mut mask));
    // The rear left channel is far below the loud front left one.
    assert!(
        mask[3 * NB_BANDS..4 * NB_BANDS].iter().all(|&v| v < -5.0),
        "{mask:?}"
    );
}

/// `log2(sqrt(1e-27))`: the level libopus gives a band without any signal.
fn empty_band_level() -> f64 {
    0.5 * (1e-27f32 as f64).log2()
}

#[test]
fn silence_and_a_single_channel_give_the_values_the_formulas_predict() {
    // These numbers come from the C source alone, not from the reference.
    let (channels, rate, frame_size) = (6usize, 48_000, 960usize);
    let channel_offset = 0.5 * (2.0f64 / 5.0).log2();
    let mut probe = SurroundProbe::new(channels, rate);
    let mut mask = vec![f32::NAN; NB_BANDS * channels];

    // Silence. Every band of every channel is at the empty level minus the
    // mean of the band; the means differ by less than 1 from band to band,
    // so spreading changes nothing. Each side of the mix stays at -28.
    assert!(probe.run(&silence(channels, frame_size), frame_size, &mut mask));
    for c in 0..5 {
        for b in 0..NB_BANDS {
            let want = empty_band_level() - E_MEANS[b] as f64 - (-28.0 + channel_offset);
            let got = mask[c * NB_BANDS + b] as f64;
            assert!(
                (got - want).abs() < 1e-5,
                "silence, channel {c} band {b}: {got}, expected {want}"
            );
        }
    }
    // The low-frequency-effects channel is not part of the mix.
    assert!(mask[5 * NB_BANDS..].iter().all(|&v| v == 0.0));

    // Noise on the front left channel only.
    let mut pcm = noise(channels, 2 * frame_size, 0.3, 47);
    for c in 1..channels {
        scale_channel(&mut pcm, channels, c, 0.0);
    }
    let mut probe = SurroundProbe::new(channels, rate);
    assert!(probe.run(&pcm[..frame_size * channels], frame_size, &mut mask));
    assert!(probe.run(&pcm[frame_size * channels..], frame_size, &mut mask));
    for b in 0..NB_BANDS {
        // The left side of the mix holds this channel alone, so the channel
        // is exactly at its mask, apart from the offset for the number of
        // channels.
        let front_left = mask[b] as f64;
        assert!(
            (front_left + channel_offset).abs() < 1e-5,
            "front left, band {b}: {front_left}, expected {}",
            -channel_offset
        );
        // The right side and the centre see no signal: as for silence.
        let want = empty_band_level() - E_MEANS[b] as f64 - (-28.0 + channel_offset);
        for c in [1usize, 2, 4] {
            let got = mask[c * NB_BANDS + b] as f64;
            assert!(
                (got - want).abs() < 1e-5,
                "channel {c} band {b}: {got}, expected {want}"
            );
        }
        // The silent rear left channel shares the left side with the noise.
        // It lies below its mask by the level of the noise, which is more
        // than 20 above the -28 of an empty side.
        let rear_left = mask[3 * NB_BANDS + b] as f64;
        assert!(rear_left < want - 20.0, "rear left, band {b}: {rear_left}");
    }
    assert!(mask[5 * NB_BANDS..].iter().all(|&v| v == 0.0));

    // The same noise on front left and front right: each side holds one
    // channel, and the centre mask is the lower of the two sides.
    let mut pcm = noise(1, frame_size, 0.3, 53);
    pcm = pcm
        .iter()
        .flat_map(|&v| [v, 0.0, v, 0.0, 0.0, 0.0])
        .collect();
    let mut probe = SurroundProbe::new(channels, rate);
    assert!(probe.run(&pcm, frame_size, &mut mask));
    for b in 0..NB_BANDS {
        assert_eq!(
            mask[b],
            mask[2 * NB_BANDS + b],
            "band {b}: left and right differ"
        );
        assert!(
            (mask[b] as f64 + channel_offset).abs() < 1e-5,
            "band {b}: {}",
            mask[b]
        );
    }
}

#[test]
fn reset_gives_the_state_of_a_new_analysis() {
    let (channels, rate, frame_size) = (8usize, 24_000, 480usize);
    let pcm = stepped_noise(channels, 3 * frame_size, 0.2, 59);
    let frames: Vec<&[f32]> = pcm.chunks_exact(frame_size * channels).collect();
    let mut fresh = SurroundProbe::new(channels, rate);
    let mut first = vec![0f32; NB_BANDS * channels];
    assert!(fresh.run(frames[2], frame_size, &mut first));

    let mut used = SurroundProbe::new(channels, rate);
    let mut mask = vec![0f32; NB_BANDS * channels];
    assert!(used.run(frames[0], frame_size, &mut mask));
    assert!(used.run(frames[1], frame_size, &mut mask));
    // Without a reset the memories of frame 1 change the result.
    let mut carried = vec![0f32; NB_BANDS * channels];
    assert!(used.run(frames[2], frame_size, &mut carried));
    assert_ne!(carried, first);
    used.reset();
    assert!(used.run(frames[2], frame_size, &mut mask));
    assert_eq!(mask, first);
}

#[test]
fn no_mask_and_no_panic_when_the_analysis_cannot_run() {
    const UNTOUCHED: f32 = 123.0;
    let run = |channels: usize, rate: i32, frame_size: usize, pcm_len: usize, mask_len: usize| {
        let mut probe = SurroundProbe::new(channels, rate);
        let pcm = vec![0.1f32; pcm_len];
        let mut mask = vec![UNTOUCHED; mask_len];
        let ok = probe.run(&pcm, frame_size, &mut mask);
        if !ok {
            assert!(mask.iter().all(|&v| v == UNTOUCHED), "the mask was written");
        }
        probe.reset();
        ok
    };
    // The valid case, for comparison. A longer mask buffer is fine; only
    // the first 21 values per channel are written.
    assert!(run(6, 48_000, 960, 960 * 6, 21 * 6));
    assert!(run(6, 48_000, 960, 960 * 6, 21 * 6 + 5));
    // libopus has the surround mapping for 3 to 8 channels only.
    for channels in [0usize, 1, 2, 9, 255, 1 << 20] {
        assert!(!run(
            channels,
            48_000,
            960,
            960 * channels.min(255),
            21 * channels.min(255)
        ));
    }
    assert!(!SurroundProbe::new(usize::MAX, 48_000).run(&[], 960, &mut []));
    // Sampling rates CELT does not take.
    for rate in [0, -1, 44_100, 32_000, 96_000, i32::MAX, i32::MIN] {
        assert!(!run(6, rate, 960, 960 * 6, 21 * 6));
    }
    // Frame sizes that are not 2.5, 5, 10 or 20 ms or a multiple of 20 ms.
    for frame_size in [0usize, 1, 119, 121, 360, 720, 961, 1440, 5761, 6720] {
        assert!(
            !run(6, 48_000, frame_size, 6720 * 6, 21 * 6),
            "frame {frame_size}"
        );
    }
    for frame_size in [usize::MAX / 6, usize::MAX / 2, usize::MAX] {
        assert!(
            !run(6, 48_000, frame_size, 960 * 6, 21 * 6),
            "frame {frame_size}"
        );
    }
    assert!(!run(6, 8_000, 960 + 160, 2000 * 6, 21 * 6));
    // Every valid frame size at every rate.
    for rate in [8_000, 12_000, 16_000, 24_000, 48_000] {
        for frame_400 in [1usize, 2, 4, 8, 16, 24, 32, 40, 48] {
            let frame_size = rate as usize / 400 * frame_400;
            assert!(
                run(6, rate, frame_size, frame_size * 6, 21 * 6),
                "{rate} Hz, frame {frame_size}"
            );
        }
    }
    // Too little room for the mask, or too few samples.
    assert!(!run(6, 48_000, 960, 960 * 6, 21 * 6 - 1));
    assert!(!run(6, 48_000, 960, 960 * 6, 0));
    assert!(!run(6, 48_000, 960, 960 * 6 - 1, 21 * 6));
    assert!(!run(6, 48_000, 960, 0, 21 * 6));
}

#[test]
fn cost_of_one_frame() {
    // 5.1 at 48 kHz in 20 ms frames, the usual surround case. The time of
    // the fastest of 20 batches of 50 frames, so that other work on the
    // machine does not count.
    let (channels, rate, frame_size) = (6usize, 48_000, 960usize);
    let pcm = mix(
        tones(rate, channels, 1, 50 * frame_size),
        &stepped_noise(channels, 50 * frame_size, 0.05, 61),
    );
    let mut probe = SurroundProbe::new(channels, rate);
    let mut mask = vec![0f32; NB_BANDS * channels];
    let mut best = f64::INFINITY;
    let mut checksum = 0f32;
    for _ in 0..20 {
        let start = std::time::Instant::now();
        for frame in pcm.chunks_exact(frame_size * channels) {
            assert!(probe.run(std::hint::black_box(frame), frame_size, &mut mask));
            checksum += mask[7];
        }
        best = best.min(start.elapsed().as_secs_f64() / 50.0);
    }
    assert!(checksum.is_finite());
    println!(
        "surround analysis, 5.1 at 48 kHz, 20 ms: {:.1} microseconds per frame ({:.3} % of real time)",
        best * 1e6,
        best / 0.02 * 100.0
    );
    // A loose limit for optimized builds: 1 % of real time.
    if !cfg!(debug_assertions) {
        assert!(best < 200e-6, "{:.1} microseconds per frame", best * 1e6);
    }
}
