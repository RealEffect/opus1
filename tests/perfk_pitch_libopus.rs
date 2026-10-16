//! The pitch functions of the pre-filter against the ones of libopus itself
//! (celt/pitch.c), called directly in the C library the tests link.
//!
//! The signals are at the scale of the encoder: a full-scale sample is
//! 32768. The lengths are the ones the encoder passes for a 20 ms frame.

use audiopus_sys as _;
use opus1::internals::pitch::{pitch_downsample, pitch_search, remove_doubling};
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

/// Whether the results must be bit for bit those of libopus: on 64-bit ARM,
/// where the crate follows the rounding of libopus's arm64 build.
const EXACT: bool = cfg!(target_arch = "aarch64");

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
/// On 64-bit ARM the output must be bit for bit the same: the crate adds
/// every product in the order and with the rounding of libopus's arm64 build.
/// Elsewhere the values are compared with a tolerance, because the libopus
/// these tests link may round differently from this crate there. For a
/// signal of pure tones the LPC analysis is close to singular (only the
/// noise floor of -40 dB keeps it solvable) and magnifies such differences
/// about a thousand times. So the limit is 1 % of the peak for pure tones
/// and 0.2 % with noise.
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
            if EXACT {
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "trial {trial} channels {channels} sample {i}: {g} against {w} of libopus"
                    );
                }
                continue;
            }
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
/// Both searches run on the same buffer. On 64-bit ARM every lag must be
/// the one of libopus. Elsewhere the correlations may be rounded differently
/// from the libopus these tests link, so a different lag is accepted in a
/// few cases, but not a large share, and not for the loud signals alone.
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
        let need = if EXACT { trials } else { trials * 98 / 100 };
        assert!(
            same[s] >= need,
            "scale {scale:e}: only {} of {trials} lags are the ones libopus finds",
            same[s]
        );
    }
}

/// `remove_doubling` gives the period and the gain libopus gives.
///
/// Both run on the same buffer with the same first guess: the true period,
/// or two, three or four times it, as the search before it may return. On
/// 64-bit ARM the period and every bit of the gain must be those of
/// libopus, a gain below zero included. Elsewhere a few differences are
/// accepted (see `pitch_search_matches_libopus`).
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
                let diff = (gain - c_gain).abs();
                worst_gain = worst_gain.max(diff);
                if gain.to_bits() == c_gain.to_bits() || (!EXACT && diff < 1e-3) {
                    same_gain += 1;
                }
            }
        }
    }
    println!(
        "{same_period} of {trials} periods and {same_gain} gains are the ones of libopus; \
         largest gain difference at an equal period {worst_gain:.2e}"
    );
    if EXACT {
        assert_eq!(same_period, trials, "periods");
        assert_eq!(same_gain, trials, "gains");
    } else {
        assert!(same_period * 100 >= trials * 98, "periods: {same_period} of {trials}");
        assert!(same_gain * 100 >= trials * 97, "gains: {same_gain} of {trials}");
    }
}

/// `remove_doubling` of opus1 and of libopus on the same buffer. Returns the
/// period and the bits of the gain, opus1's first.
fn both_remove_doubling(
    buf: &[f32],
    guess: usize,
    prev_period: usize,
    prev_gain: f32,
) -> ((usize, u32), (usize, u32)) {
    let mut c_buf = buf.to_vec();
    let mut c_t0 = guess as c_int;
    // SAFETY: the buffer holds (MAX_PERIOD + FRAME) / 2 samples, as in the
    // encoder.
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
    let gain = remove_doubling(buf, MAX_PERIOD, MIN_PERIOD, FRAME, &mut t0, prev_period, prev_gain);
    ((t0, gain.to_bits()), (c_t0 as usize, c_gain.to_bits()))
}

/// `n` samples that repeat a random block of `period` samples exactly, with
/// peak value near `scale`.
fn periodic(rng: &mut Lcg, n: usize, period: usize, scale: f32) -> Vec<f32> {
    let block: Vec<f32> = (0..period).map(|_| scale * rng.unit()).collect();
    (0..n).map(|i| block[i % period]).collect()
}

/// The tests below look for decisions that a single rounding can change.
/// On 64-bit ARM every result must be libopus's. Elsewhere the crate may
/// round differently from the libopus these tests link, so they only print
/// what they find.
fn check_exact(what: &str, same: usize, trials: usize) {
    println!("{what}: the result of libopus in {same} of {trials} cases");
    if EXACT {
        assert_eq!(same, trials, "{what}");
    }
}

/// `pitch_search` on signals that repeat exactly.
///
/// In such a signal the lags `k` and `k + period` have the same correlation
/// to the last bit. Their window energies differ only by the rounding of the
/// running sum of `find_best_pitch` (C `Syy`), and that rounding decides
/// between the two lags. The voiced signals of `pitch_search_matches_libopus`
/// hardly ever reach such a tie: with the two squares of that running sum
/// added without fused multiply-add, they still all pass, while 16 of the
/// 1168 lags here differ from libopus.
///
/// The second part uses the lengths of the loss concealment
/// (`celt_plc_pitch_search`: 1024 samples, `len` 1328, `max_pitch` 620).
#[test]
fn periodic_signals_give_the_lag_of_libopus() {
    let mut rng = Lcg(0x77);
    let len = (MAX_PERIOD + FRAME) >> 1;
    let (mut trials, mut same) = (0usize, 0usize);
    for period in 8..300 {
        for scale in [1.0f32, 300.0, 32768.0, 1.0e6] {
            let buf = periodic(&mut rng, len, period, scale);
            let want = libopus_pitch_search(&buf);
            let got = pitch_search(&buf[MAX_PERIOD >> 1..], &buf, FRAME, MAX_PERIOD - 3 * MIN_PERIOD);
            trials += 1;
            same += (got == want) as usize;
        }
    }
    check_exact("pitch_search, lengths of the encoder", same, trials);

    let (mut trials, mut same) = (0usize, 0usize);
    for period in 8..300 {
        for scale in [1.0f32, 32768.0] {
            let buf = periodic(&mut rng, 1024, period, scale);
            let mut y = buf.clone();
            let mut want: c_int = -1;
            // SAFETY: `y` holds the 1024 samples `celt_plc_pitch_search`
            // passes; the target starts 360 samples into it.
            unsafe { c_pitch_search(y.as_ptr().add(360), y.as_mut_ptr(), 1328, 620, &mut want, 0) };
            let got = pitch_search(&buf[360..], &buf, 1328, 620);
            trials += 1;
            same += (got == want as usize) as usize;
        }
    }
    check_exact("pitch_search, lengths of the concealment", same, trials);

    let (mut trials, mut same) = (0usize, 0usize);
    for period in 8..250 {
        let buf = periodic(&mut rng, len, period, 32768.0);
        for multiple in 1..=4usize {
            let guess = (2 * period * multiple).clamp(MIN_PERIOD, MAX_PERIOD - 2);
            for (prev_period, prev_gain) in [(0usize, 0.0f32), (guess, 0.5), (guess / 2, 0.8)] {
                let (got, want) = both_remove_doubling(&buf, guess, prev_period, prev_gain);
                trials += 1;
                same += (got == want) as usize;
            }
        }
    }
    check_exact("remove_doubling", same, trials);
}

/// `remove_doubling` on quiet signals. The pitch gain is
/// `xy / sqrt(1 + xx * yy)`, and libopus adds the 1 with a fused
/// multiply-add. That rounding only shows when `xx * yy` is near 2^24, which
/// signals with a peak of 2 to 8 reach. (Rounding the product first changes
/// the gain in 31 of these 1600 cases.)
#[test]
fn quiet_signals_give_the_pitch_gain_of_libopus() {
    let mut rng = Lcg(0x99);
    let len = (MAX_PERIOD + FRAME) >> 1;
    let (mut trials, mut same) = (0usize, 0usize);
    for trial in 0..400 {
        let period = 8.0 + 0.61 * (trial % 397) as f64;
        let scale = [2.0f32, 3.0, 5.0, 8.0][trial % 4];
        let buf = voiced(&mut rng, len, period, 0.2, scale);
        for multiple in 1..=4usize {
            let guess = ((2.0 * period * multiple as f64).round() as usize).clamp(MIN_PERIOD, MAX_PERIOD - 2);
            let (got, want) = both_remove_doubling(&buf, guess, 0, 0.0);
            trials += 1;
            same += (got == want) as usize;
        }
    }
    check_exact("remove_doubling, quiet signals", same, trials);
}

/// An `f32` as an integer that orders like the float, so that a bisection
/// can step one unit in the last place at a time.
fn float_order(x: f32) -> i64 {
    let bits = x.to_bits() as i32 as i64;
    if bits < 0 { -(bits & 0x7fff_ffff) } else { bits }
}

fn float_from_order(o: i64) -> f32 {
    if o < 0 {
        f32::from_bits((-o) as u32 | 0x8000_0000)
    } else {
        f32::from_bits(o as u32)
    }
}

/// `remove_doubling` at the point where it changes its mind.
///
/// The signal repeats a block of `period` samples with some noise, the
/// first guess is two periods and the previous period is one period. Half
/// the guess then competes with the guess, and the threshold it has to beat
/// is `max(0.3, 0.7 * g0 - prev_gain)`. A bisection over `prev_gain` finds
/// where libopus switches between the two, and the test compares the 81
/// values of `prev_gain` around that point, one unit in the last place
/// apart. libopus computes the threshold with one fused multiply-add;
/// rounding the product first changes 85 of the 12150 results.
#[test]
fn remove_doubling_switches_where_libopus_switches() {
    let mut rng = Lcg(0x55);
    let len = (MAX_PERIOD + FRAME) >> 1;
    let (mut trials, mut same, mut points) = (0usize, 0usize, 0usize);
    for period in (20..170usize).step_by(3) {
        for noise in [0.05f32, 0.3, 0.8] {
            let block: Vec<f32> = (0..period).map(|_| rng.unit()).collect();
            let buf: Vec<f32> = (0..len)
                .map(|i| 32768.0 * (block[i % period] + noise * rng.unit()))
                .collect();
            let guess = 4 * period;
            let c_period = |prev_gain: f32| both_remove_doubling(&buf, guess, 2 * period, prev_gain).1.0;
            let (mut lo, mut hi) = (float_order(-1.0), float_order(1.0));
            let at_lo = c_period(float_from_order(lo));
            if at_lo == c_period(float_from_order(hi)) {
                continue;
            }
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if c_period(float_from_order(mid)) == at_lo {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            points += 1;
            for o in lo - 40..hi + 40 {
                let (got, want) = both_remove_doubling(&buf, guess, 2 * period, float_from_order(o));
                trials += 1;
                same += (got == want) as usize;
            }
        }
    }
    assert!(points >= 100, "only {points} switching points found");
    check_exact("remove_doubling around its switching points", same, trials);
}
