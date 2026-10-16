//! `silk_ltp_analysis_filter_fix` has a loop without bounds checks for
//! subframes that lie completely inside the input, and a sample-by-sample
//! loop for the others. These tests compare both with the function as it was
//! before: one loop that checks every sample.

use opus1::silk::ltp_analysis::silk_ltp_analysis_filter_fix;
use opus1::silk::macros::{silk_sat16, silk_smlabb, silk_smulbb, silk_smulwb};

const LTP_ORDER: usize = 5;

/// The function before the change. The only edit: the rounding addition
/// wraps explicitly (in a release build it wraps as well).
#[allow(clippy::too_many_arguments)]
fn reference(
    ltp_res: &mut [i16],
    x: &[i16],
    x_base_idx: usize,
    b_q14: &[i16],
    pitch_l: &[i32],
    inv_gain_q16: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let mut x_ptr_idx = x_base_idx;
    let mut ltp_res_ptr_idx = 0;

    for k in 0..nb_subfr {
        let valid_pitch = pitch_l[k].max(2);
        let x_lag_ptr_idx = x_ptr_idx as isize - valid_pitch as isize;

        let btmp_q14 = [
            b_q14[k * LTP_ORDER],
            b_q14[k * LTP_ORDER + 1],
            b_q14[k * LTP_ORDER + 2],
            b_q14[k * LTP_ORDER + 3],
            b_q14[k * LTP_ORDER + 4],
        ];

        for i in 0..(subfr_length + pre_length) {
            let idx = x_ptr_idx + i;
            let res_val = if idx < x.len() { x[idx] } else { 0 };

            let get_x = |offset: isize| -> i16 {
                let idx = x_lag_ptr_idx + i as isize + offset;
                if idx >= 0 && (idx as usize) < x.len() {
                    x[idx as usize]
                } else {
                    0
                }
            };
            let mut ltp_est: i32 = silk_smulbb(get_x(2) as i32, btmp_q14[0] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(1) as i32, btmp_q14[1] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(0) as i32, btmp_q14[2] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(-1) as i32, btmp_q14[3] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(-2) as i32, btmp_q14[4] as i32);

            let ltp_est_q0 = ltp_est.wrapping_add(8192) >> 14;

            let res_q0 = silk_sat16(res_val as i32 - ltp_est_q0) as i16;

            ltp_res[ltp_res_ptr_idx + i] = silk_smulwb(inv_gain_q16[k], res_q0 as i32) as i16;
        }

        ltp_res_ptr_idx += subfr_length + pre_length;
        x_ptr_idx += subfr_length;
    }
}

/// xorshift64 generator with a fixed seed, so the test data never changes.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `low..=high`.
    fn range(&mut self, low: i32, high: i32) -> i32 {
        low + (self.next() % (high as i64 - low as i64 + 1) as u64) as i32
    }

    /// A 16-bit value: half of the time one of the extremes.
    fn sample(&mut self) -> i16 {
        const EXTREMES: [i16; 6] = [i16::MIN, -32767, -1, 0, 1, i16::MAX];
        if self.range(0, 1) == 0 {
            EXTREMES[self.range(0, 5) as usize]
        } else {
            self.range(i16::MIN as i32, i16::MAX as i32) as i16
        }
    }
}

/// Run both functions and require the same output, also behind the part
/// that is written.
#[allow(clippy::too_many_arguments)]
fn check(
    x: &[i16],
    x_base_idx: usize,
    b_q14: &[i16],
    pitch_l: &[i32],
    inv_gain_q16: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
    what: &str,
) {
    let written = nb_subfr * (subfr_length + pre_length);
    let mut expected = vec![0x5a5ai16; written + 3];
    let mut got = expected.clone();
    reference(
        &mut expected,
        x,
        x_base_idx,
        b_q14,
        pitch_l,
        inv_gain_q16,
        subfr_length,
        nb_subfr,
        pre_length,
    );
    silk_ltp_analysis_filter_fix(
        &mut got,
        x,
        x_base_idx,
        b_q14,
        pitch_l,
        inv_gain_q16,
        subfr_length,
        nb_subfr,
        pre_length,
    );
    assert_eq!(
        got,
        expected,
        "{what}: x.len() {}, base {x_base_idx}, pitch {pitch_l:?}, subframe {subfr_length}, \
         {nb_subfr} subframes, pre {pre_length}",
        x.len()
    );
}

#[test]
fn encoder_layout_matches_reference() {
    // The layout of the encoder: 20 ms of history, then the frame.
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for round in 0..4_000 {
        let fs_khz = [8usize, 12, 16][round % 3];
        let nb_subfr = if round % 4 == 0 { 2 } else { 4 };
        let subfr_length = 5 * fs_khz;
        let pre_length = if fs_khz == 16 { 16 } else { 10 };
        let ltp_mem = 20 * fs_khz;
        let x: Vec<i16> = (0..ltp_mem + nb_subfr * subfr_length + 40)
            .map(|_| rng.range(-12000, 12000) as i16)
            .collect();
        let b_q14: Vec<i16> = (0..nb_subfr * LTP_ORDER)
            .map(|_| rng.range(-40, 127) as i16 * 128)
            .collect();
        let pitch_l: Vec<i32> = (0..nb_subfr)
            .map(|_| rng.range(2 * fs_khz as i32, 18 * fs_khz as i32))
            .collect();
        let inv_gain_q16: Vec<i32> = (0..nb_subfr).map(|_| rng.range(1, 1 << 16)).collect();
        check(
            &x,
            ltp_mem - pre_length,
            &b_q14,
            &pitch_l,
            &inv_gain_q16,
            subfr_length,
            nb_subfr,
            pre_length,
            "encoder layout",
        );
    }
}

#[test]
fn every_length_and_extreme_values_match_reference() {
    // Every loop length from 0 upwards (the vector code has a remainder
    // loop), with extreme samples, coefficients and gains.
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    for subfr_length in 0..=40 {
        for pre_length in [0usize, 1, 3, 10, 16] {
            for nb_subfr in [1usize, 2, 4] {
                let base = 300;
                let x: Vec<i16> = (0..base + nb_subfr * subfr_length + pre_length + 8)
                    .map(|_| rng.sample())
                    .collect();
                let b_q14: Vec<i16> = (0..nb_subfr * LTP_ORDER).map(|_| rng.sample()).collect();
                let pitch_l: Vec<i32> = (0..nb_subfr).map(|_| rng.range(-5, 290)).collect();
                let inv_gain_q16: Vec<i32> = (0..nb_subfr)
                    .map(|_| match rng.range(0, 4) {
                        0 => i32::MAX,
                        1 => i32::MIN,
                        2 => rng.range(-70000, 70000),
                        _ => rng.next() as i32,
                    })
                    .collect();
                check(
                    &x,
                    base,
                    &b_q14,
                    &pitch_l,
                    &inv_gain_q16,
                    subfr_length,
                    nb_subfr,
                    pre_length,
                    "every length",
                );
            }
        }
    }
}

#[test]
fn samples_outside_the_input_count_as_zero() {
    // The lag reaches before the start of `x`, or the subframe reaches past
    // its end, or both: the sample-by-sample loop runs.
    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    for round in 0..3_000 {
        let subfr_length = rng.range(1, 80) as usize;
        let pre_length = rng.range(0, 16) as usize;
        let nb_subfr = rng.range(1, 4) as usize;
        let needed = nb_subfr * subfr_length + pre_length;
        // Between "far too short" and "just long enough".
        let len = rng.range(0, needed as i32 + 20) as usize;
        let x: Vec<i16> = (0..len).map(|_| rng.sample()).collect();
        let x_base_idx = if round % 3 == 0 {
            0
        } else {
            rng.range(0, 30) as usize
        };
        let b_q14: Vec<i16> = (0..nb_subfr * LTP_ORDER).map(|_| rng.sample()).collect();
        let pitch_l: Vec<i32> = (0..nb_subfr).map(|_| rng.range(0, 400)).collect();
        let inv_gain_q16: Vec<i32> = (0..nb_subfr).map(|_| rng.next() as i32).collect();
        check(
            &x,
            x_base_idx,
            &b_q14,
            &pitch_l,
            &inv_gain_q16,
            subfr_length,
            nb_subfr,
            pre_length,
            "outside the input",
        );
    }
}
