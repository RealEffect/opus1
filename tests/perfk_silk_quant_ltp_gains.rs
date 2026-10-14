//! `silk_quant_ltp_gains` against the function inside C libopus
//! (silk/quant_LTP_gains.c). libopus is linked as a static library, so the
//! internal function can be called directly. The function is fixed-point:
//! every output must be identical.

use opus_rs::silk::gain_quant::silk_quant_ltp_gains;
use opus_rs::silk::ltp_analysis::silk_find_ltp_fix;
use std::os::raw::c_int;

const LTP_ORDER: usize = 5;
const MAX_NB_SUBFR: usize = 4;

unsafe extern "C" {
    #[link_name = "silk_quant_LTP_gains"]
    fn c_silk_quant_ltp_gains(
        b_q14: *mut i16,
        cbk_index: *mut i8,
        periodicity_index: *mut i8,
        sum_log_gain_q7: *mut i32,
        pred_gain_db_q7: *mut c_int,
        xx_q17: *const i32,
        xx_in_q17: *const i32,
        subfr_len: c_int,
        nb_subfr: c_int,
        arch: c_int,
    );
}

/// Makes sure the test binary links libopus: the function above is found in
/// the static library that `audiopus_sys` brings along.
fn link_libopus() {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() as u64 % (hi as i64 - lo as i64 + 1) as u64) as i32
    }
}

/// Everything the function writes.
#[derive(Debug, PartialEq)]
struct Outputs {
    b_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    cbk_index: [i8; MAX_NB_SUBFR],
    periodicity_index: i8,
    sum_log_gain_q7: i32,
    pred_gain_db_q7: i32,
}

fn run_c(xx: &[i32], xx_in: &[i32], subfr_len: usize, nb_subfr: usize, sum_log_gain_q7: i32) -> Outputs {
    let mut out = Outputs {
        b_q14: [0; MAX_NB_SUBFR * LTP_ORDER],
        cbk_index: [0; MAX_NB_SUBFR],
        periodicity_index: 0,
        sum_log_gain_q7,
        pred_gain_db_q7: 0,
    };
    unsafe {
        c_silk_quant_ltp_gains(
            out.b_q14.as_mut_ptr(),
            out.cbk_index.as_mut_ptr(),
            &mut out.periodicity_index,
            &mut out.sum_log_gain_q7,
            &mut out.pred_gain_db_q7,
            xx.as_ptr(),
            xx_in.as_ptr(),
            subfr_len as c_int,
            nb_subfr as c_int,
            0,
        );
    }
    out
}

fn run_rs(xx: &[i32], xx_in: &[i32], subfr_len: usize, nb_subfr: usize, sum_log_gain_q7: i32) -> Outputs {
    let mut out = Outputs {
        b_q14: [0; MAX_NB_SUBFR * LTP_ORDER],
        cbk_index: [0; MAX_NB_SUBFR],
        periodicity_index: 0,
        sum_log_gain_q7,
        pred_gain_db_q7: 0,
    };
    silk_quant_ltp_gains(
        &mut out.b_q14,
        &mut out.cbk_index,
        &mut out.periodicity_index,
        &mut out.sum_log_gain_q7,
        &mut out.pred_gain_db_q7,
        xx,
        xx_in,
        subfr_len as i32,
        nb_subfr,
        0,
    );
    out
}

/// A residual signal as the pitch analysis leaves it: pulses with the given
/// period, a share of noise, and a level. `history` samples come before the
/// frame.
fn residual(rng: &mut Lcg, len: usize, period: usize, noise: i32, amplitude: i32) -> Vec<i16> {
    let mut phase = rng.range(0, period as i32 - 1) as usize;
    (0..len)
        .map(|_| {
            phase += 1;
            let pulse = if phase >= period {
                phase = 0;
                amplitude
            } else if phase == 1 {
                -amplitude / 3
            } else {
                0
            };
            (pulse + rng.range(-noise, noise)).clamp(-32768, 32767) as i16
        })
        .collect()
}

/// Correlations of real residuals through `silk_find_ltp_fix`, as the
/// encoder calls the function: all sampling rates, both frame lengths,
/// strongly and weakly periodic signals, quiet and loud, lags that are
/// right and lags that are off by a few samples.
#[test]
fn quant_ltp_gains_matches_libopus_on_encoder_input() {
    link_libopus();
    let mut rng = Lcg(0x0bad_5eed_2468_1357);
    let mut codebooks_chosen = [0usize; 3];
    let mut gains_differ_from_best = 0;
    for case in 0..3000usize {
        let fs_khz = [8usize, 12, 16][case % 3];
        let nb_subfr = if case % 5 == 0 { 2 } else { 4 };
        let subfr_len = 5 * fs_khz;
        let history = 20 * fs_khz;
        let period = rng.range(2 * fs_khz as i32, 18 * fs_khz as i32 - 3) as usize;
        let amplitude = [300, 3000, 12000, 30000][case % 4];
        let noise = [0, amplitude / 40, amplitude / 6, amplitude][(case / 4) % 4];
        let r = residual(&mut rng, history + nb_subfr * subfr_len + LTP_ORDER, period, noise, amplitude);
        let lag_error = [0, 0, 1, -1, 3][(case / 16) % 5];
        let lags: [i32; MAX_NB_SUBFR] = core::array::from_fn(|_| {
            (period as i32 + lag_error).clamp(2 * fs_khz as i32, 18 * fs_khz as i32)
        });

        let mut xx = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut xx_in = [0i32; MAX_NB_SUBFR * LTP_ORDER];
        silk_find_ltp_fix(&mut xx, &mut xx_in, &r, history, &lags, subfr_len, nb_subfr, 0);

        // The running sum of log gains from earlier frames: none, some, or
        // close to the limit, where the largest gains are not allowed.
        let sum_log_gain_q7 = match case % 7 {
            0 => 0,
            1 => rng.range(4000, 5400),
            _ => rng.range(0, 3000),
        };
        let want = run_c(&xx, &xx_in, subfr_len, nb_subfr, sum_log_gain_q7);
        let got = run_rs(&xx, &xx_in, subfr_len, nb_subfr, sum_log_gain_q7);
        assert_eq!(
            got, want,
            "case {case}: {fs_khz} kHz, {nb_subfr} subframes, period {period}, noise {noise}, amplitude {amplitude}"
        );
        codebooks_chosen[want.periodicity_index as usize] += 1;
        if want.periodicity_index != 2 {
            gains_differ_from_best += 1;
        }
    }
    // Every codebook was chosen, and in many cases the codebook that was
    // tried last is not the chosen one: there the prediction gain comes
    // from another codebook than the coefficients.
    assert!(codebooks_chosen.iter().all(|&n| n > 100), "{codebooks_chosen:?}");
    assert!(gains_differ_from_best > 500, "{gains_differ_from_best}");
}

/// Correlations that are not those of a pitch pulse train: five random
/// vectors and a target that is a random mix of them plus noise, with mixing
/// gains inside and outside the range of the codebooks. The matrix and the
/// vector are the inner products divided by the energy of the target, as
/// `silk_find_ltp_fix` scales them, so the prediction error of every
/// codebook vector is positive, as libopus requires.
#[test]
fn quant_ltp_gains_matches_libopus_on_mixed_vectors() {
    link_libopus();
    let mut rng = Lcg(0x1122_3344_5566_7788);
    for case in 0..2000usize {
        let nb_subfr = if case % 3 == 0 { 2 } else { 4 };
        let subfr_len = [40usize, 60, 80][case % 3];
        let mut xx = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut xx_in = [0i32; MAX_NB_SUBFR * LTP_ORDER];
        for j in 0..nb_subfr {
            let level = [200, 2000, 20000][(case / 3) % 3];
            let basis: Vec<Vec<i64>> = (0..LTP_ORDER)
                .map(|_| (0..subfr_len).map(|_| rng.range(-level, level) as i64).collect())
                .collect();
            // Mixing gains in 1/64: mostly one strong tap, sometimes all
            // taps strong (a gain above what the codebooks hold).
            let gains: [i64; LTP_ORDER] = core::array::from_fn(|i| {
                if case % 11 == 0 {
                    rng.range(-40, 90) as i64
                } else if i == 2 {
                    rng.range(0, 80) as i64
                } else {
                    rng.range(-12, 24) as i64
                }
            });
            let noise = [0, level / 20, level / 2][(case / 9) % 3];
            let target: Vec<i64> = (0..subfr_len)
                .map(|n| {
                    let mix: i64 = (0..LTP_ORDER).map(|i| gains[i] * basis[i][n]).sum();
                    mix / 64 + rng.range(-noise, noise) as i64
                })
                .collect();
            let dot = |a: &[i64], b: &[i64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<i64>();
            let energy = dot(&target, &target).max(1);
            for i in 0..LTP_ORDER {
                for l in 0..LTP_ORDER {
                    let v = (dot(&basis[i], &basis[l]) << 17) / energy;
                    xx[j * 25 + i * LTP_ORDER + l] = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                }
                let v = (dot(&basis[i], &target) << 17) / energy;
                xx_in[j * LTP_ORDER + i] = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }
        let sum_log_gain_q7 = rng.range(0, 5400);
        let want = run_c(&xx, &xx_in, subfr_len, nb_subfr, sum_log_gain_q7);
        let got = run_rs(&xx, &xx_in, subfr_len, nb_subfr, sum_log_gain_q7);
        assert_eq!(got, want, "case {case}: {nb_subfr} subframes, subframe length {subfr_len}");
    }
}
