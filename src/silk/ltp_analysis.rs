use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::*;

pub fn silk_find_ltp_fix(
    xxltp_q17_matrix: &mut [i32],
    xxltp_q17_vector: &mut [i32],
    r_buf: &[i16],
    r_frame_idx: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    _arch: i32,
) {
    let mut xx: i32 = 0;
    let mut nrg: i32 = 0;
    let mut temp: i32;
    let mut xx_shifts: i32 = 0;
    let mut xx_shifts_matrix: i32 = 0;
    let mut xx_shifts_vector: i32;
    let mut extra_shifts: i32;
    let mut r_ptr_idx = r_frame_idx;

    for k in 0..nb_subfr {
        let lag_ptr_idx = r_ptr_idx - (lag[k] as usize + LTP_ORDER / 2);

        silk_sum_sqr_shift(
            &mut xx,
            &mut xx_shifts,
            &r_buf[r_ptr_idx..],
            subfr_length + LTP_ORDER,
        );

        let xxlp_ptr = &mut xxltp_q17_matrix[k * LTP_ORDER * LTP_ORDER..];
        silk_corr_matrix_fix(
            &r_buf[lag_ptr_idx..],
            subfr_length,
            LTP_ORDER,
            xxlp_ptr,
            &mut nrg,
            &mut xx_shifts_matrix,
        );

        extra_shifts = xx_shifts - xx_shifts_matrix;
        if extra_shifts > 0 {
            xx_shifts_vector = xx_shifts;
            for i in 0..(LTP_ORDER * LTP_ORDER) {
                xxlp_ptr[i] = silk_rshift32(xxlp_ptr[i], extra_shifts);
            }
            nrg = silk_rshift32(nrg, extra_shifts);
        } else if extra_shifts < 0 {
            xx_shifts_vector = xx_shifts_matrix;
            xx = silk_rshift32(xx, -extra_shifts);
        } else {
            xx_shifts_vector = xx_shifts;
        }

        let xxlp_vec_ptr = &mut xxltp_q17_vector[k * LTP_ORDER..];
        silk_corr_vector_fix(
            &r_buf[lag_ptr_idx..],
            &r_buf[r_ptr_idx..],
            subfr_length,
            LTP_ORDER,
            xxlp_vec_ptr,
            xx_shifts_vector,
        );

        temp = silk_smlawb(1, nrg, 1966);
        temp = temp.max(1).max(xx);

        for i in 0..(LTP_ORDER * LTP_ORDER) {
            xxlp_ptr[i] = (((xxlp_ptr[i] as i64) << 17) / temp as i64) as i32;
        }
        for i in 0..LTP_ORDER {
            xxlp_vec_ptr[i] = (((xxlp_vec_ptr[i] as i64) << 17) / temp as i64) as i32;
        }

        r_ptr_idx += subfr_length;
    }
}

/// One output sample of the LTP analysis filter.
///
/// `cur` is the input sample. `taps[j]` is the earlier input sample that
/// coefficient `b_q14[j]` multiplies: the sample one pitch lag before `cur`
/// is `taps[2]`, and `taps[0]` is two samples later than that.
#[inline(always)]
fn ltp_residual(
    cur: i16,
    taps: [i16; LTP_ORDER],
    b_q14: &[i16; LTP_ORDER],
    inv_gain_q16: i32,
) -> i16 {
    let mut ltp_est: i32 = silk_smulbb(taps[0] as i32, b_q14[0] as i32);
    ltp_est = silk_smlabb(ltp_est, taps[1] as i32, b_q14[1] as i32);
    ltp_est = silk_smlabb(ltp_est, taps[2] as i32, b_q14[2] as i32);
    ltp_est = silk_smlabb(ltp_est, taps[3] as i32, b_q14[3] as i32);
    ltp_est = silk_smlabb(ltp_est, taps[4] as i32, b_q14[4] as i32);

    let ltp_est_q0 = ltp_est.wrapping_add(8192) >> 14;

    let res_q0 = silk_sat16(cur as i32 - ltp_est_q0);

    // silk_smulwb(inv_gain_q16, res_q0), which is
    // (inv_gain_q16 * res_q0) >> 16 with a 64-bit product, written with
    // 32-bit products so that the compiler can keep the whole loop in vector
    // code. With inv_gain_q16 = high * 65536 + low (0 <= low < 65536) the
    // product is high * res_q0 * 65536 + low * res_q0, and the first term
    // is a multiple of 65536. `res_q0` is a 16-bit value, so both products
    // fit in 32 bits.
    let high = inv_gain_q16 >> 16;
    let low = inv_gain_q16 & 0xffff;
    (high * res_q0 + ((low * res_q0) >> 16)) as i16
}

/// LTP analysis filter (libopus `silk_LTP_analysis_filter_FIX`): for every
/// subframe, subtract the long-term prediction from the input and scale the
/// result with the inverse gain of the subframe.
///
/// Subframe `k` reads `subfr_length + pre_length` samples of `x` from
/// `x_base_idx + k * subfr_length`, and the samples `pitch_l[k]` (at least
/// 2) before them, two more on either side. A sample outside `x` counts as
/// zero. The output of the subframes is written to `ltp_res` one after the
/// other.
pub fn silk_ltp_analysis_filter_fix(
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
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_avx2() {
        // SAFETY: the processor has AVX2.
        unsafe {
            ltp_analysis_filter_avx2(
                ltp_res,
                x,
                x_base_idx,
                b_q14,
                pitch_l,
                inv_gain_q16,
                subfr_length,
                nb_subfr,
                pre_length,
            )
        };
        return;
    }
    ltp_analysis_filter_portable(
        ltp_res,
        x,
        x_base_idx,
        b_q14,
        pitch_l,
        inv_gain_q16,
        subfr_length,
        nb_subfr,
        pre_length,
    );
}

/// [`silk_ltp_analysis_filter_fix`] compiled with AVX2 enabled.
///
/// AVX2 only changes how the compiler encodes and vectorizes the code. The
/// default x86-64 target has no 32-bit and no 32x32-to-64-bit vector
/// multiplication, which fixed-point code needs; AVX2 has both. The results
/// are the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn ltp_analysis_filter_avx2(
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
    ltp_analysis_filter_portable(
        ltp_res,
        x,
        x_base_idx,
        b_q14,
        pitch_l,
        inv_gain_q16,
        subfr_length,
        nb_subfr,
        pre_length,
    );
}

#[inline(always)]
fn ltp_analysis_filter_portable(
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
    let n = subfr_length + pre_length;
    let mut x_ptr_idx = x_base_idx;
    let mut ltp_res_ptr_idx = 0;

    for k in 0..nb_subfr {
        let valid_pitch = pitch_l[k].max(2);
        let x_lag_ptr_idx = x_ptr_idx as isize - valid_pitch as isize;

        let btmp_q14: &[i16; LTP_ORDER] = b_q14[k * LTP_ORDER..]
            .first_chunk()
            .expect("5 coefficients per subframe");
        let inv_gain_q16 = inv_gain_q16[k];

        // First sample that the filter reads around the lag: two before the
        // lagged position of output 0.
        let lag_start = x_lag_ptr_idx - (LTP_ORDER as isize / 2);
        let all_inside = lag_start >= 0
            && lag_start as usize + n + (LTP_ORDER - 1) <= x.len()
            && x_ptr_idx + n <= x.len();
        if all_inside {
            // The usual case: every sample is inside `x`. The slices all have
            // length `n`, so the loop has no bounds checks and the compiler
            // turns it into vector code.
            let lag_start = lag_start as usize;
            let cur = &x[x_ptr_idx..x_ptr_idx + n];
            let (t0, t1, t2, t3, t4) = (
                &x[lag_start + 4..lag_start + 4 + n],
                &x[lag_start + 3..lag_start + 3 + n],
                &x[lag_start + 2..lag_start + 2 + n],
                &x[lag_start + 1..lag_start + 1 + n],
                &x[lag_start..lag_start + n],
            );
            let out = &mut ltp_res[ltp_res_ptr_idx..ltp_res_ptr_idx + n];
            for i in 0..n {
                out[i] = ltp_residual(
                    cur[i],
                    [t0[i], t1[i], t2[i], t3[i], t4[i]],
                    btmp_q14,
                    inv_gain_q16,
                );
            }
        } else {
            for i in 0..n {
                let idx = x_ptr_idx + i;
                let cur = if idx < x.len() { x[idx] } else { 0 };

                let get_x = |offset: isize| -> i16 {
                    let idx = x_lag_ptr_idx + i as isize + offset;
                    if idx >= 0 && (idx as usize) < x.len() {
                        x[idx as usize]
                    } else {
                        0
                    }
                };
                ltp_res[ltp_res_ptr_idx + i] = ltp_residual(
                    cur,
                    [get_x(2), get_x(1), get_x(0), get_x(-1), get_x(-2)],
                    btmp_q14,
                    inv_gain_q16,
                );
            }
        }

        ltp_res_ptr_idx += n;
        x_ptr_idx += subfr_length;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// The AVX2 copy of `silk_ltp_analysis_filter_fix` against the portable
    /// body, whatever the dispatch chooses: 2 and 4 subframes at 8 to
    /// 16 kHz, lags from 2 to beyond the start of the input (the loop that
    /// checks every sample), random coefficients and inverse gains.
    #[test]
    fn avx2_copy_matches_portable() {
        if !crate::compat::x86_has_avx2() {
            return;
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..2000 {
            let fs_khz = [8usize, 12, 16][round % 3];
            let nb_subfr = if round % 2 == 0 { 4 } else { 2 };
            let subfr_length = 5 * fs_khz;
            let pre_length = [10usize, 16][round % 2];
            let x_base_idx = 18 * fs_khz + pre_length;
            let len = x_base_idx + nb_subfr * subfr_length + pre_length;
            let x: Vec<i16> = (0..len).map(|_| next() as i16 >> (next() % 8)).collect();
            let b_q14: Vec<i16> = (0..nb_subfr * LTP_ORDER).map(|_| next() as i16).collect();
            let pitch_l: Vec<i32> = (0..nb_subfr)
                .map(|_| (next() % (x_base_idx as u64 + 40)) as i32)
                .collect();
            let inv_gain_q16: Vec<i32> = (0..nb_subfr).map(|_| (next() >> 33) as i32).collect();
            let out_len = nb_subfr * (subfr_length + pre_length);
            let mut want = vec![0x5a5ai16; out_len];
            let mut got = want.clone();
            ltp_analysis_filter_portable(
                &mut want,
                &x,
                x_base_idx,
                &b_q14,
                &pitch_l,
                &inv_gain_q16,
                subfr_length,
                nb_subfr,
                pre_length,
            );
            // SAFETY: the processor has AVX2 (checked above).
            unsafe {
                ltp_analysis_filter_avx2(
                    &mut got,
                    &x,
                    x_base_idx,
                    &b_q14,
                    &pitch_l,
                    &inv_gain_q16,
                    subfr_length,
                    nb_subfr,
                    pre_length,
                )
            };
            assert_eq!(got, want, "round {round}, lags {pitch_l:?}");
        }
    }
}
