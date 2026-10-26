//! `silk_pitch_analysis_core` against the fixed-point function of libopus
//! (silk/fixed/pitch_analysis_core_FIX.c).
//!
//! The libopus that the tests link is the floating-point build, which does
//! not contain this function. The test calls it in the fixed-point build of
//! the official libopus 1.6.1 release that the dev-dependency
//! `libopus-fixed` compiles (tools/libopus-fixed), on the same signals.
//!
//! The function is fixed-point: every output must be identical.
//!
//! At 12 kHz the function first resamples its input to 8 kHz with
//! `silk_resampler_down2_3` (src/silk/resampler.rs), so those cases also
//! depend on that function. Until it computed what libopus computes
//! (tests/perfk_silkdec_down2_3.rs compares it with the linked libopus),
//! 137 of the 12 kHz cases differed.

use opus1::internals::silk::pitch_analysis::silk_pitch_analysis_core;

/// Number of cases.
const CASES: usize = 900;

/// Small deterministic generator: integers only.
struct Generator(u32);

impl Generator {
    fn rnd(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

/// Two-pole resonators: `2 r cos(w)` and `-r^2` in Q14.
const RES: [[i64; 2]; 6] = [
    [0, 0],
    [26000, -11000],
    [18000, -12500],
    [-9000, -10000],
    [29000, -13500],
    [6000, -14000],
];

/// One call of the function.
struct Case {
    frame: Vec<i16>,
    fs_khz: i32,
    nb_subfr: usize,
    complexity: usize,
    prev_lag: i32,
    ltp_corr: i32,
    thres1_q16: i32,
    thres2_q13: i32,
    kind: u32,
}

/// The next case. The kinds of signal:
/// 0 = pulse train through a resonator (voiced speech), 1 = the same with
/// noise, 2 = noise, 3 = two sawtooth waves (steady tones), 4 = pulse train
/// with a period that drifts, 5 = almost silence. The period covers the
/// whole lag range from 2 ms to 18 ms, the level goes from full scale down
/// by up to 9 bits.
fn next_case(g: &mut Generator) -> Case {
    let fs_khz = [8, 12, 16][(g.rnd() % 3) as usize];
    let nb_subfr = if g.rnd().is_multiple_of(4) { 2 } else { 4 };
    let complexity = (g.rnd() % 3) as usize;
    let kind = g.rnd() % 6;
    let amp_shift = g.rnd() % 10;
    let (min_lag, max_lag) = (2 * fs_khz, 18 * fs_khz);
    let mut period_q4 = (16 * min_lag) as i64 + (g.rnd() % (16 * (max_lag - min_lag) + 1) as u32) as i64;
    let prev_lag = if g.rnd().is_multiple_of(3) {
        0
    } else {
        min_lag + (g.rnd() % (max_lag - min_lag) as u32) as i32
    };
    let ltp_corr = (g.rnd() % 32768) as i32;
    let thres1_q16 = [52429, 49807, 48497, 45875][(g.rnd() % 4) as usize];
    let thres2_q13 = 1500 + (g.rnd() % 3500) as i32;
    let res = RES[(g.rnd() % 6) as usize];
    let drift = if kind == 4 { (g.rnd() % 9) as i64 - 4 } else { 0 };
    let noise_shift = 4 + g.rnd() % 8;
    let frame_length = (20 + nb_subfr * 5) * fs_khz as usize;
    let (mut y1, mut y2, mut phase_q4, mut phase2) = (0i64, 0i64, 0i64, 0i64);
    let period2 = 20 + (g.rnd() % 200) as i64;

    let mut frame = Vec::with_capacity(frame_length);
    for _ in 0..frame_length {
        let mut e = 0i64;
        let noise = (g.rnd() & 0xffff) as i64 - 32768;
        match kind {
            0 | 1 | 4 => {
                phase_q4 += 16;
                if phase_q4 >= period_q4 {
                    phase_q4 -= period_q4;
                    period_q4 = (period_q4 + drift).max((16 * min_lag) as i64);
                    e = 20000;
                }
                if kind == 1 {
                    e += noise >> noise_shift;
                }
            }
            2 => e = noise >> 2,
            3 => {
                phase_q4 += 16;
                if phase_q4 >= period_q4 {
                    phase_q4 -= period_q4;
                }
                phase2 += 1;
                if phase2 >= period2 {
                    phase2 -= period2;
                }
                e = (phase_q4 * 12000) / period_q4 - 6000 + (phase2 * 8000) / period2 - 4000;
            }
            _ => e = noise >> 13,
        }
        let y = (e + ((res[0] * y1) >> 14) + ((res[1] * y2) >> 14)).clamp(-(1 << 28), 1 << 28);
        y2 = y1;
        y1 = y;
        frame.push((y >> amp_shift).clamp(-32768, 32767) as i16);
    }
    Case {
        frame,
        fs_khz,
        nb_subfr,
        complexity,
        prev_lag,
        ltp_corr,
        thres1_q16,
        thres2_q13,
        kind,
    }
}

#[test]
fn pitch_analysis_core_matches_libopus_fixed_point() {
    let mut g = Generator(0x1357_9bdf);
    let mut voiced_by_rate = [0usize; 3];
    let mut failures = Vec::new();
    for number in 0..CASES {
        let case = next_case(&mut g);
        let c = libopus_fixed::pitch_analysis_core(
            &case.frame,
            case.ltp_corr,
            case.prev_lag,
            case.thres1_q16,
            case.thres2_q13,
            case.fs_khz,
            case.complexity as i32,
            case.nb_subfr,
        );
        let expected = (c.unvoiced, c.pitch_out, c.lag_index, c.contour_index, c.ltp_corr_q15);

        let mut pitch_out = [0i32; 4];
        let (mut lag_index, mut contour_index) = (0i16, 0i8);
        let mut ltp_corr = case.ltp_corr;
        let unvoiced = silk_pitch_analysis_core(
            &case.frame,
            &mut pitch_out,
            &mut lag_index,
            &mut contour_index,
            &mut ltp_corr,
            case.prev_lag,
            case.thres1_q16,
            case.thres2_q13,
            case.fs_khz,
            case.complexity,
            case.nb_subfr,
        );
        let got = (unvoiced, pitch_out, lag_index, contour_index, ltp_corr);
        if got != expected {
            failures.push(format!(
                "case {number} ({} kHz, {} subframes, complexity {}, signal {}, previous lag {}): opus1 {:?}, libopus {:?}",
                case.fs_khz, case.nb_subfr, case.complexity, case.kind, case.prev_lag, got, expected
            ));
        }
        if c.unvoiced == 0 {
            voiced_by_rate[(case.fs_khz / 4 - 2) as usize] += 1;
        }
    }
    // libopus finds voiced frames at every sampling rate.
    assert!(voiced_by_rate.iter().all(|&n| n > 100), "{voiced_by_rate:?}");
    assert!(
        failures.is_empty(),
        "{} of {CASES} cases differ from libopus:\n{}",
        failures.len(),
        failures.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
    );
}
