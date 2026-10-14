//! The pitch functions of the pre-filter against the ones of libopus itself
//! (celt/pitch.c), called directly in the C library the tests link.
//!
//! The signals are at the scale of the encoder: a full-scale sample is
//! 32768. The lengths are the ones the encoder passes for a 20 ms frame.

use audiopus_sys as _;
use opus_rs::pitch::{pitch_downsample, pitch_search, remove_doubling};
use std::os::raw::c_int;

unsafe extern "C" {
    /// `void pitch_downsample(celt_sig *x[], opus_val16 *x_lp, int len, int C,
    /// int factor, int arch)` of the float build.
    #[link_name = "pitch_downsample"]
    fn c_pitch_downsample(
        x: *const *mut f32,
        x_lp: *mut f32,
        len: c_int,
        c: c_int,
        factor: c_int,
        arch: c_int,
    );
    /// `void pitch_search(const opus_val16 *x_lp, opus_val16 *y, int len,
    /// int max_pitch, int *pitch, int arch)`.
    #[link_name = "pitch_search"]
    fn c_pitch_search(
        x_lp: *const f32,
        y: *mut f32,
        len: c_int,
        max_pitch: c_int,
        pitch: *mut c_int,
        arch: c_int,
    );
    /// `opus_val16 remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
    /// int N, int *T0, int prev_period, opus_val16 prev_gain, int arch)`.
    #[link_name = "remove_doubling"]
    fn c_remove_doubling(
        x: *mut f32,
        maxperiod: c_int,
        minperiod: c_int,
        n: c_int,
        t0: *mut c_int,
        prev_period: c_int,
        prev_gain: f32,
        arch: c_int,
    ) -> f32;
}

/// Longest and shortest period of the pre-filter, and the frame size.
const MAX_PERIOD: usize = 1024;
const MIN_PERIOD: usize = 15;
const FRAME: usize = 960;

/// Small deterministic generator, so that the tests need no dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// `n` samples of a signal with the given period (in samples, may have a
/// fraction): a few harmonics plus noise, with peak value near `scale`.
fn voiced(rng: &mut Lcg, n: usize, period: f64, noise: f32, scale: f32) -> Vec<f32> {
    let phase = [rng.unit() as f64, rng.unit() as f64, rng.unit() as f64];
    (0..n)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / period;
            let tone = 0.5 * (t + phase[0]).sin()
                + 0.3 * (2.0 * t + phase[1]).sin()
                + 0.2 * (3.0 * t + phase[2]).sin();
            scale * (tone as f32 + noise * rng.unit())
        })
        .collect()
}

fn libopus_downsample(channels: &[Vec<f32>], len: usize) -> Vec<f32> {
    let mut copies: Vec<Vec<f32>> = channels.to_vec();
    let pointers: Vec<*mut f32> = copies.iter_mut().map(|c| c.as_mut_ptr()).collect();
    let mut out = vec![0.0f32; len];
    // SAFETY: every channel holds `2 * len` samples and `out` holds `len`,
    // which is what the function reads and writes for a factor of 2.
    unsafe {
        c_pitch_downsample(
            pointers.as_ptr(),
            out.as_mut_ptr(),
            len as c_int,
            channels.len() as c_int,
            2,
            0,
        )
    };
    out
}

fn libopus_pitch_search(buf: &[f32]) -> usize {
    let mut y = buf.to_vec();
    let mut pitch: c_int = -1;
    // SAFETY: as in the encoder: `y` holds (MAX_PERIOD + FRAME) / 2 samples,
    // the target starts MAX_PERIOD / 2 samples into it.
    unsafe {
        c_pitch_search(
            y.as_ptr().add(MAX_PERIOD >> 1),
            y.as_mut_ptr(),
            FRAME as c_int,
            (MAX_PERIOD - 3 * MIN_PERIOD) as c_int,
            &mut pitch,
            0,
        )
    };
    pitch as usize
}

/// `pitch_downsample` gives what libopus gives: the decimation and the
/// spectral flattening after it (LPC analysis and 5-tap filter).
///
/// The values are compared with a tolerance. The filter coefficients come
/// from sums of products, and the two libraries add those in different
/// orders, so the last bits of the sums differ. For a signal of pure tones
/// the LPC analysis is close to singular (only the noise floor of -40 dB
/// keeps it solvable) and magnifies that difference about a thousand times.
/// So the limit is 1 % of the peak for pure tones and 0.2 % with noise.
#[test]
fn pitch_downsample_matches_libopus() {
    let mut rng = Lcg(0xD05);
    let len = (MAX_PERIOD + FRAME) >> 1;
    // Largest difference for pure tones, and for signals with noise.
    let mut largest = [0.0f64; 2];
    for trial in 0..60 {
        let scale = [32768.0f32, 3000.0, 30.0, 0.5][trial % 4];
        let period = 30.0 + 11.7 * (trial % 37) as f64;
        let noise = [0.0f32, 0.05, 0.5, 4.0][(trial / 4) % 4];
        for channels in 1..=2usize {
            let x: Vec<Vec<f32>> = (0..channels)
                .map(|_| voiced(&mut rng, 2 * len, period, noise, scale))
                .collect();
            let want = libopus_downsample(&x, len);
            let refs: Vec<&[f32]> = x.iter().map(|c| c.as_slice()).collect();
            let mut got = vec![0.0f32; len];
            pitch_downsample(&refs, &mut got, len, channels, 2);
            let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
            let limit = if noise == 0.0 { 1e-2 } else { 2e-3 };
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let err = (*g as f64 - *w as f64).abs() / peak.max(1e-30);
                largest[(noise != 0.0) as usize] = largest[(noise != 0.0) as usize].max(err);
                assert!(
                    err < limit,
                    "trial {trial} channels {channels} sample {i}: {g} against {w} of libopus \
                     (peak {peak})"
                );
            }
        }
    }
    println!(
        "largest difference from libopus: {:.2e} of the peak for pure tones, {:.2e} with noise",
        largest[0], largest[1]
    );
    // Silence stays silence.
    let zero = vec![0.0f32; 2 * len];
    let mut got = vec![1.0f32; len];
    pitch_downsample(&[&zero], &mut got, len, 1, 2);
    assert!(got.iter().all(|&v| v == 0.0));
}

/// `pitch_search` finds the lag libopus finds, also for input far above
/// full scale.
///
/// Both searches run on the same buffer. The correlations are sums of
/// products that the two libraries add in different orders. So a different
/// lag is accepted in a few cases, but not a large share, and not for the
/// loud signals alone.
#[test]
fn pitch_search_matches_libopus() {
    let mut rng = Lcg(0x5EA);
    let len = (MAX_PERIOD + FRAME) >> 1;
    let scales = [1.0f32, 300.0, 32768.0, 1.0e6, 3.0e7];
    let mut same = [0usize; 5];
    let trials = 300;
    for trial in 0..trials {
        let period = 16.0 + 0.83 * (trial % 290) as f64;
        let noise = [0.02f32, 0.2, 1.0][trial % 3];
        let unit = voiced(&mut rng, len, period, noise, 1.0);
        for (s, &scale) in scales.iter().enumerate() {
            let buf: Vec<f32> = unit.iter().map(|v| v * scale).collect();
            let want = libopus_pitch_search(&buf);
            let got = pitch_search(&buf[MAX_PERIOD >> 1..], &buf, FRAME, MAX_PERIOD - 3 * MIN_PERIOD);
            if got == want {
                same[s] += 1;
            }
        }
    }
    for (s, &scale) in scales.iter().enumerate() {
        println!("scale {scale:e}: the lag of libopus in {} of {trials} signals", same[s]);
        assert!(
            same[s] * 100 >= trials * 98,
            "scale {scale:e}: only {} of {trials} lags are the ones libopus finds",
            same[s]
        );
    }
}

/// `remove_doubling` gives the period and the gain libopus gives.
///
/// Both run on the same buffer with the same first guess: the true period,
/// or two, three or four times it, as the search before it may return.
#[test]
fn remove_doubling_matches_libopus() {
    let mut rng = Lcg(0xD0B);
    let len = (MAX_PERIOD + FRAME) >> 1;
    let (mut trials, mut same_period, mut same_gain) = (0, 0, 0);
    let mut worst_gain = 0.0f32;
    for trial in 0..400 {
        // Periods of the decimated signal from 8 to 250 samples.
        let period = 8.0 + 0.61 * (trial % 397) as f64;
        let noise = [0.02f32, 0.2, 1.0, 3.0][trial % 4];
        let scale = [32768.0f32, 500.0, 1.0e6][trial % 3];
        let buf = voiced(&mut rng, len, period, noise, scale);
        for multiple in 1..=4usize {
            // The guess counts samples before the decimation.
            let guess = ((2.0 * period * multiple as f64).round() as usize).clamp(MIN_PERIOD, MAX_PERIOD - 2);
            let prev_period = [0usize, guess, guess / 2][trial % 3];
            let prev_gain = [0.0f32, 0.5, 0.8][(trial / 3) % 3];

            let mut c_buf = buf.clone();
            let mut c_t0 = guess as c_int;
            // SAFETY: the buffer holds (MAX_PERIOD + FRAME) / 2 samples, as
            // in the encoder.
            let c_gain = unsafe {
                c_remove_doubling(
                    c_buf.as_mut_ptr(),
                    MAX_PERIOD as c_int,
                    MIN_PERIOD as c_int,
                    FRAME as c_int,
                    &mut c_t0,
                    prev_period as c_int,
                    prev_gain,
                    0,
                )
            };
            let mut t0 = guess;
            let gain = remove_doubling(&buf, MAX_PERIOD, MIN_PERIOD, FRAME, &mut t0, prev_period, prev_gain);
            trials += 1;
            if t0 == c_t0 as usize {
                same_period += 1;
                // libopus can return a gain below zero where opus-rs returns
                // zero. The encoder treats both as "no pre-filter".
                let diff = (gain - c_gain.max(0.0)).abs();
                worst_gain = worst_gain.max(diff);
                if diff < 1e-3 {
                    same_gain += 1;
                }
            }
        }
    }
    println!(
        "{same_period} of {trials} periods and {same_gain} gains are the ones of libopus; \
         largest gain difference at an equal period {worst_gain:.2e}"
    );
    assert!(same_period * 100 >= trials * 98, "periods: {same_period} of {trials}");
    assert!(same_gain * 100 >= trials * 97, "gains: {same_gain} of {trials}");
}
