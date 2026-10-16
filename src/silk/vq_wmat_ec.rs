//! Search of the codebooks for the gains of the long-term predictor (LTP),
//! libopus `silk_VQ_WMat_EC`.
//!
//! For every codebook vector the search computes the prediction error that
//! the vector leaves, turns it into a number of bits and keeps the vector
//! with the lowest total. There are two versions with the same result:
//!
//! - `search_scalar` handles one vector at a time. It is the portable code
//!   and the reference in the tests.
//! - `neon::search` (aarch64) handles four vectors at a time.

use crate::silk::define::*;
use crate::silk::macros::*;
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
use crate::silk::tables::SILK_LTP_GAIN_BITS_Q5_PTRS;
use crate::silk::tables::{SILK_LTP_VQ_GAIN_PTRS_Q7, SILK_LTP_VQ_PTRS_Q7};

/// Correlation matrix of one subframe (Q17), row by row.
type CorrMatrix = [i32; LTP_ORDER * LTP_ORDER];

/// What the search found in one codebook.
pub(crate) struct LtpVqChoice {
    /// Index of the best codebook vector. 0 when no vector qualified.
    pub ind: i8,
    /// Residual energy (Q15) of that vector. `i32::MAX` when none qualified.
    pub res_nrg_q15: i32,
    /// Total bits (Q7) of that vector. `i32::MAX` when none qualified.
    pub rate_dist_q7: i32,
    /// Sum of the absolute coefficients (Q7) of that vector. `None` when no
    /// vector qualified.
    pub gain_q7: Option<i32>,
}

impl LtpVqChoice {
    const NONE: Self = Self {
        ind: 0,
        res_nrg_q15: i32::MAX,
        rate_dist_q7: i32::MAX,
        gain_q7: None,
    };
}

/// The correlation vector, negated and in Q24.
#[inline(always)]
fn negate_to_q24(xx_in_q17: &[i32; LTP_ORDER]) -> [i32; LTP_ORDER] {
    xx_in_q17.map(|v| silk_lshift(v, 7).wrapping_neg())
}

/// Prediction error (Q15) that one codebook vector leaves:
/// 1.001 - 2 * xX * cb + cb' * XX * cb. All sums wrap around.
#[inline(always)]
fn ltp_vq_error_q15(
    xx_q17: &CorrMatrix,
    neg_xx_q24: &[i32; LTP_ORDER],
    cb_row_q7: &[i8; LTP_ORDER],
) -> i32 {
    let cb = cb_row_q7.map(|v| v as i32);
    let mut sum1_q15: i32 = 32801;

    // First row of the matrix.
    let mut sum2_q24 = silk_mla(neg_xx_q24[0], xx_q17[1], cb[1]);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[2], cb[2]);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[3], cb[3]);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[4], cb[4]);
    sum2_q24 = silk_lshift(sum2_q24, 1);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[0], cb[0]);
    sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb[0]);

    // Second row.
    sum2_q24 = silk_mla(neg_xx_q24[1], xx_q17[7], cb[2]);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[8], cb[3]);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[9], cb[4]);
    sum2_q24 = silk_lshift(sum2_q24, 1);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[6], cb[1]);
    sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb[1]);

    // Third row.
    sum2_q24 = silk_mla(neg_xx_q24[2], xx_q17[13], cb[3]);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[14], cb[4]);
    sum2_q24 = silk_lshift(sum2_q24, 1);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[12], cb[2]);
    sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb[2]);

    // Fourth row.
    sum2_q24 = silk_mla(neg_xx_q24[3], xx_q17[19], cb[4]);
    sum2_q24 = silk_lshift(sum2_q24, 1);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[18], cb[3]);
    sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb[3]);

    // Last row.
    sum2_q24 = silk_lshift(neg_xx_q24[4], 1);
    sum2_q24 = silk_mla(sum2_q24, xx_q17[24], cb[4]);
    silk_smlawb(sum1_q15, sum2_q24, cb[4])
}

/// Penalty (Q15) for a vector whose gain exceeds the allowed maximum.
#[inline(always)]
fn gain_penalty_q15(gain_q7: i32, max_gain_q7: i32) -> i32 {
    silk_lshift(gain_q7.wrapping_sub(max_gain_q7).max(0), 11)
}

/// Total bits (Q8) of a vector: the bits for its residual energy, assuming
/// 6 dB per bit and sample, plus half of its code length.
#[inline(always)]
fn ltp_vq_bits_q8(res_nrg_q15: i32, subfr_len: i32, cl_q5: u8) -> i32 {
    let bits_res_q8 = silk_smulbb(subfr_len, silk_lin2log(res_nrg_q15) - (15 << 7));
    bits_res_q8 + ((cl_q5 as i32) << (3 - 1))
}

/// The search, one codebook vector at a time. The three slices are the
/// codebook: its vectors, the gain of each vector and its code length.
fn search_scalar(
    xx_q17: &CorrMatrix,
    xx_in_q17: &[i32; LTP_ORDER],
    cb_q7: &[[i8; LTP_ORDER]],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
) -> LtpVqChoice {
    let neg_xx_q24 = negate_to_q24(xx_in_q17);
    let mut best = LtpVqChoice::NONE;
    for (k, ((cb_row_q7, &gain_q7), &cl_q5)) in cb_q7.iter().zip(cb_gain_q7).zip(cl_q5).enumerate()
    {
        let gain_q7 = gain_q7 as i32;
        let sum1_q15 = ltp_vq_error_q15(xx_q17, &neg_xx_q24, cb_row_q7);
        if sum1_q15 >= 0 {
            let res_nrg_q15 = sum1_q15.wrapping_add(gain_penalty_q15(gain_q7, max_gain_q7));
            let bits_tot_q8 = ltp_vq_bits_q8(res_nrg_q15, subfr_len, cl_q5);
            // Of two vectors with the same number of bits the later one wins.
            if bits_tot_q8 <= best.rate_dist_q7 {
                best = LtpVqChoice {
                    ind: k as i8,
                    res_nrg_q15,
                    rate_dist_q7: bits_tot_q8,
                    gain_q7: Some(gain_q7),
                };
            }
        }
    }
    best
}

/// The search over LTP codebook number `codebook` (0, 1 or 2) for one
/// subframe with correlation matrix `xx_q17` and correlation vector
/// `xx_in_q17`.
pub(crate) fn silk_vq_wmat_ec_codebook(
    codebook: usize,
    xx_q17: &CorrMatrix,
    xx_in_q17: &[i32; LTP_ORDER],
    subfr_len: i32,
    max_gain_q7: i32,
) -> LtpVqChoice {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        let cb_q7 = SILK_LTP_VQ_PTRS_Q7[codebook];
        let cb_gain_q7 = SILK_LTP_VQ_GAIN_PTRS_Q7[codebook];
        let neg_xx_q24 = negate_to_q24(xx_in_q17);
        let found = neon::search(
            xx_q17,
            &neg_xx_q24,
            neon::CODEBOOKS[codebook],
            subfr_len,
            max_gain_q7,
        );
        match found {
            None => LtpVqChoice::NONE,
            Some((k, rate_dist_q7)) => {
                // The vector code leaves out the residual energy of the
                // winner. Compute it again for this one vector.
                let gain_q7 = cb_gain_q7[k] as i32;
                let sum1_q15 = ltp_vq_error_q15(xx_q17, &neg_xx_q24, &cb_q7[k]);
                LtpVqChoice {
                    ind: k as i8,
                    res_nrg_q15: sum1_q15.wrapping_add(gain_penalty_q15(gain_q7, max_gain_q7)),
                    rate_dist_q7,
                    gain_q7: Some(gain_q7),
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        search_scalar(
            xx_q17,
            xx_in_q17,
            SILK_LTP_VQ_PTRS_Q7[codebook],
            SILK_LTP_VQ_GAIN_PTRS_Q7[codebook],
            SILK_LTP_GAIN_BITS_Q5_PTRS[codebook],
            subfr_len,
            max_gain_q7,
        )
    }
}

/// Entropy-constrained search of one LTP codebook for one subframe (libopus
/// `silk_VQ_WMat_EC`).
///
/// The matrix is the 25 values of `xx_q17` from `xx_q17_ptr` and the vector
/// the 5 values of `xx_q17_in` from `xx_q17_in_ptr`. The first `l` entries of
/// `cb_q7`, `cb_gain_q7` and `cl_q5` are the codebook. `gain_q7_out` is only
/// written when a vector qualified.
pub fn silk_vq_wmat_ec(
    ind: &mut i8,
    res_nrg_q15: &mut i32,
    rate_dist_q7: &mut i32,
    gain_q7_out: &mut i32,
    xx_q17: &[i32],
    xx_q17_ptr: usize,
    xx_q17_in: &[i32],
    xx_q17_in_ptr: usize,
    cb_q7: &[[i8; LTP_ORDER]],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
    l: i32,
) {
    let xx_q17: &CorrMatrix = xx_q17[xx_q17_ptr..]
        .first_chunk()
        .expect("25 values of the correlation matrix");
    let xx_in_q17: &[i32; LTP_ORDER] = xx_q17_in[xx_q17_in_ptr..]
        .first_chunk()
        .expect("5 values of the correlation vector");
    let l = l.max(0) as usize;
    let best = search_scalar(
        xx_q17,
        xx_in_q17,
        &cb_q7[..l],
        &cb_gain_q7[..l],
        &cl_q5[..l],
        subfr_len,
        max_gain_q7,
    );
    *ind = best.ind;
    *res_nrg_q15 = best.res_nrg_q15;
    *rate_dist_q7 = best.rate_dist_q7;
    if let Some(gain_q7) = best.gain_q7 {
        *gain_q7_out = gain_q7;
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::CorrMatrix;
    use crate::silk::define::LTP_ORDER;
    use crate::silk::sigproc_fix::smulwb_operand;
    use crate::silk::tables::{
        SILK_LTP_GAIN_BITS_Q5_0, SILK_LTP_GAIN_BITS_Q5_1, SILK_LTP_GAIN_BITS_Q5_2,
        SILK_LTP_GAIN_VQ_0_GAIN, SILK_LTP_GAIN_VQ_1_GAIN, SILK_LTP_GAIN_VQ_2_GAIN,
        SILK_LTP_VQ_0_Q7, SILK_LTP_VQ_1_Q7, SILK_LTP_VQ_2_Q7,
    };
    use core::arch::aarch64::*;

    /// Four codebook vectors side by side: lane `n` of every field belongs
    /// to vector number `4 * group + n` of the codebook.
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub(super) struct Group {
        /// `cb_q7[c]` holds coefficient `c` of the four vectors.
        cb_q7: [[i32; 4]; LTP_ORDER],
        /// The `smulwb_operand` of each value.
        cb_q22: [[i32; 4]; LTP_ORDER],
        /// Sum of the absolute coefficients of each vector.
        gain_q7: [i32; 4],
        /// Half of the code length of each vector, in Q8.
        half_cl_q8: [i32; 4],
    }

    /// A codebook of `N` vectors in `G = N / 4` groups.
    const fn groups<const N: usize, const G: usize>(
        cb_q7: &[[i8; LTP_ORDER]; N],
        cb_gain_q7: &[u8; N],
        cl_q5: &[u8; N],
    ) -> [Group; G] {
        assert!(N == 4 * G);
        let mut out = [Group {
            cb_q7: [[0; 4]; LTP_ORDER],
            cb_q22: [[0; 4]; LTP_ORDER],
            gain_q7: [0; 4],
            half_cl_q8: [0; 4],
        }; G];
        let mut k = 0;
        while k < N {
            let (group, lane) = (k / 4, k % 4);
            let mut c = 0;
            while c < LTP_ORDER {
                out[group].cb_q7[c][lane] = cb_q7[k][c] as i32;
                out[group].cb_q22[c][lane] = smulwb_operand(cb_q7[k][c] as i16);
                c += 1;
            }
            out[group].gain_q7[lane] = cb_gain_q7[k] as i32;
            out[group].half_cl_q8[lane] = (cl_q5[k] as i32) << (3 - 1);
            k += 1;
        }
        out
    }

    static CODEBOOK_0: [Group; 2] = groups(
        &SILK_LTP_VQ_0_Q7,
        &SILK_LTP_GAIN_VQ_0_GAIN,
        &SILK_LTP_GAIN_BITS_Q5_0,
    );
    static CODEBOOK_1: [Group; 4] = groups(
        &SILK_LTP_VQ_1_Q7,
        &SILK_LTP_GAIN_VQ_1_GAIN,
        &SILK_LTP_GAIN_BITS_Q5_1,
    );
    static CODEBOOK_2: [Group; 8] = groups(
        &SILK_LTP_VQ_2_Q7,
        &SILK_LTP_GAIN_VQ_2_GAIN,
        &SILK_LTP_GAIN_BITS_Q5_2,
    );
    /// The three LTP codebooks, in the order of `SILK_LTP_VQ_PTRS_Q7`.
    pub(super) static CODEBOOKS: [&[Group]; 3] = [&CODEBOOK_0, &CODEBOOK_1, &CODEBOOK_2];

    /// The search of `super::search_scalar`, four codebook vectors at a
    /// time. Returns the number of the best vector and its total bits (Q8),
    /// or `None` when no vector qualified.
    ///
    /// Every sum of the scalar code wraps around, so the order of the
    /// additions is free here. `silk_smulwb` is `vqdmulhq_s32` with a
    /// `smulwb_operand`.
    pub(super) fn search(
        xx_q17: &CorrMatrix,
        neg_xx_q24: &[i32; LTP_ORDER],
        codebook: &[Group],
        subfr_len: i32,
        max_gain_q7: i32,
    ) -> Option<(usize, i32)> {
        let mut bits_q8 = [0i32; 4];
        let mut number = [0i32; 4];
        // SAFETY: `vld1q_s32` reads and `vst1q_s32` writes four `i32`. Every
        // pointer below comes from a `[i32; 4]`.
        unsafe {
            // Entry (row, column) of the matrix in all four lanes.
            let xx = |row: usize, column: usize| vdupq_n_s32(xx_q17[LTP_ORDER * row + column]);
            let (xx00, xx01, xx02, xx03, xx04) = (xx(0, 0), xx(0, 1), xx(0, 2), xx(0, 3), xx(0, 4));
            let (xx11, xx12, xx13, xx14) = (xx(1, 1), xx(1, 2), xx(1, 3), xx(1, 4));
            let (xx22, xx23, xx24) = (xx(2, 2), xx(2, 3), xx(2, 4));
            let (xx33, xx34) = (xx(3, 3), xx(3, 4));
            let xx44 = xx(4, 4);
            let neg0 = vdupq_n_s32(neg_xx_q24[0]);
            let neg1 = vdupq_n_s32(neg_xx_q24[1]);
            let neg2 = vdupq_n_s32(neg_xx_q24[2]);
            let neg3 = vdupq_n_s32(neg_xx_q24[3]);
            let neg4_twice = vdupq_n_s32(neg_xx_q24[4] << 1);
            let max_gain = vdupq_n_s32(max_gain_q7);
            let subfr_len = vdupq_n_s32(subfr_len as i16 as i32);
            let zero = vdupq_n_s32(0);

            let mut best_bits = vdupq_n_s32(i32::MAX);
            let mut best_number = vdupq_n_s32(-1);
            let first_numbers: [i32; 4] = [0, 1, 2, 3];
            let mut numbers = vld1q_s32(first_numbers.as_ptr());

            for group in codebook {
                let cb0 = vld1q_s32(group.cb_q7[0].as_ptr());
                let cb1 = vld1q_s32(group.cb_q7[1].as_ptr());
                let cb2 = vld1q_s32(group.cb_q7[2].as_ptr());
                let cb3 = vld1q_s32(group.cb_q7[3].as_ptr());
                let cb4 = vld1q_s32(group.cb_q7[4].as_ptr());

                // Prediction error, row by row as in `ltp_vq_error_q15`.
                let mut row = vmlaq_s32(neg0, xx01, cb1);
                row = vmlaq_s32(row, xx02, cb2);
                row = vmlaq_s32(row, xx03, cb3);
                row = vmlaq_s32(row, xx04, cb4);
                row = vmlaq_s32(vshlq_n_s32::<1>(row), xx00, cb0);
                let mut sum1 = vqdmulhq_s32(row, vld1q_s32(group.cb_q22[0].as_ptr()));

                row = vmlaq_s32(neg1, xx12, cb2);
                row = vmlaq_s32(row, xx13, cb3);
                row = vmlaq_s32(row, xx14, cb4);
                row = vmlaq_s32(vshlq_n_s32::<1>(row), xx11, cb1);
                sum1 = vaddq_s32(sum1, vqdmulhq_s32(row, vld1q_s32(group.cb_q22[1].as_ptr())));

                row = vmlaq_s32(neg2, xx23, cb3);
                row = vmlaq_s32(row, xx24, cb4);
                row = vmlaq_s32(vshlq_n_s32::<1>(row), xx22, cb2);
                sum1 = vaddq_s32(sum1, vqdmulhq_s32(row, vld1q_s32(group.cb_q22[2].as_ptr())));

                row = vmlaq_s32(neg3, xx34, cb4);
                row = vmlaq_s32(vshlq_n_s32::<1>(row), xx33, cb3);
                sum1 = vaddq_s32(sum1, vqdmulhq_s32(row, vld1q_s32(group.cb_q22[3].as_ptr())));

                row = vmlaq_s32(neg4_twice, xx44, cb4);
                sum1 = vaddq_s32(sum1, vqdmulhq_s32(row, vld1q_s32(group.cb_q22[4].as_ptr())));
                sum1 = vaddq_s32(sum1, vdupq_n_s32(32801));

                // Residual energy with the penalty for too large a gain.
                let gain = vld1q_s32(group.gain_q7.as_ptr());
                let penalty = vshlq_n_s32::<11>(vmaxq_s32(vsubq_s32(gain, max_gain), zero));
                let res_nrg = vaddq_s32(sum1, penalty);

                // silk_lin2log(res_nrg): the 7 bits below the leading one
                // give the fraction; a negative shift count shifts right.
                // As in libopus, the value is not checked: 0 gives -128,
                // and a negative value counts as having no leading zero.
                let lz = vclzq_s32(res_nrg);
                let frac = vandq_s32(
                    vshlq_s32(res_nrg, vsubq_s32(lz, vdupq_n_s32(24))),
                    vdupq_n_s32(0x7f),
                );
                let curve = vmulq_s32(frac, vsubq_s32(vdupq_n_s32(128), frac));
                let curve = vshrq_n_s32::<16>(vmulq_n_s32(curve, 179));
                let whole = vshlq_n_s32::<7>(vsubq_s32(vdupq_n_s32(31), lz));
                let log = vaddq_s32(vaddq_s32(frac, curve), whole);

                let bits = vmulq_s32(subfr_len, vsubq_s32(log, vdupq_n_s32(15 << 7)));
                let bits = vaddq_s32(bits, vld1q_s32(group.half_cl_q8.as_ptr()));

                // A vector qualifies when its error is not negative. In
                // each lane the later of two equal vectors wins, as in the
                // scalar code.
                let take = vandq_u32(vcgezq_s32(sum1), vcleq_s32(bits, best_bits));
                best_bits = vbslq_s32(take, bits, best_bits);
                best_number = vbslq_s32(take, numbers, best_number);
                numbers = vaddq_s32(numbers, vdupq_n_s32(4));
            }
            vst1q_s32(bits_q8.as_mut_ptr(), best_bits);
            vst1q_s32(number.as_mut_ptr(), best_number);
        }

        // The best of the four lanes; of equal ones the later vector.
        let mut best: Option<(usize, i32)> = None;
        for lane in 0..4 {
            if number[lane] < 0 {
                continue;
            }
            let candidate = (number[lane] as usize, bits_q8[lane]);
            let better = match best {
                None => true,
                Some((k, bits)) => candidate.1 < bits || (candidate.1 == bits && candidate.0 > k),
            };
            if better {
                best = Some(candidate);
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::tables::{
        SILK_LTP_GAIN_BITS_Q5_PTRS, SILK_LTP_VQ_GAIN_PTRS_Q7, SILK_LTP_VQ_PTRS_Q7,
        SILK_LTP_VQ_SIZES,
    };

    /// The function as it was before this file had two versions: a direct
    /// port of libopus `silk_VQ_WMat_EC`, with the two additions that can
    /// overflow made wrapping (in a release build they wrap as well).
    #[allow(clippy::too_many_arguments)]
    fn original(
        ind: &mut i8,
        res_nrg_q15: &mut i32,
        rate_dist_q7: &mut i32,
        gain_q7_out: &mut i32,
        xx_q17: &[i32],
        xx_q17_ptr: usize,
        xx_q17_in: &[i32],
        xx_q17_in_ptr: usize,
        cb_q7: &[[i8; LTP_ORDER]],
        cb_gain_q7: &[u8],
        cl_q5: &[u8],
        subfr_len: i32,
        max_gain_q7: i32,
        l: i32,
    ) {
        let mut neg_xx_q24 = [0i32; 5];
        let mut sum1_q15: i32;
        let mut sum2_q24: i32;

        neg_xx_q24[0] = silk_lshift(xx_q17_in[xx_q17_in_ptr], 7).wrapping_neg();
        neg_xx_q24[1] = silk_lshift(xx_q17_in[xx_q17_in_ptr + 1], 7).wrapping_neg();
        neg_xx_q24[2] = silk_lshift(xx_q17_in[xx_q17_in_ptr + 2], 7).wrapping_neg();
        neg_xx_q24[3] = silk_lshift(xx_q17_in[xx_q17_in_ptr + 3], 7).wrapping_neg();
        neg_xx_q24[4] = silk_lshift(xx_q17_in[xx_q17_in_ptr + 4], 7).wrapping_neg();

        *rate_dist_q7 = i32::MAX;
        *res_nrg_q15 = i32::MAX;
        *ind = 0;

        for k in 0..l {
            let gain_tmp_q7 = cb_gain_q7[k as usize] as i32;

            sum1_q15 = 32801;

            let penalty = silk_lshift(gain_tmp_q7.wrapping_sub(max_gain_q7).max(0), 11);

            let cb_row = &cb_q7[k as usize];

            sum2_q24 = silk_mla(neg_xx_q24[0], xx_q17[xx_q17_ptr + 1], cb_row[1] as i32);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 2], cb_row[2] as i32);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 3], cb_row[3] as i32);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 4], cb_row[4] as i32);
            sum2_q24 = silk_lshift(sum2_q24, 1);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr], cb_row[0] as i32);
            sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[0] as i32);

            sum2_q24 = silk_mla(neg_xx_q24[1], xx_q17[xx_q17_ptr + 7], cb_row[2] as i32);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 8], cb_row[3] as i32);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 9], cb_row[4] as i32);
            sum2_q24 = silk_lshift(sum2_q24, 1);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 6], cb_row[1] as i32);
            sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[1] as i32);

            sum2_q24 = silk_mla(neg_xx_q24[2], xx_q17[xx_q17_ptr + 13], cb_row[3] as i32);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 14], cb_row[4] as i32);
            sum2_q24 = silk_lshift(sum2_q24, 1);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 12], cb_row[2] as i32);
            sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[2] as i32);

            sum2_q24 = silk_mla(neg_xx_q24[3], xx_q17[xx_q17_ptr + 19], cb_row[4] as i32);
            sum2_q24 = silk_lshift(sum2_q24, 1);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 18], cb_row[3] as i32);
            sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[3] as i32);

            sum2_q24 = silk_lshift(neg_xx_q24[4], 1);
            sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 24], cb_row[4] as i32);
            sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[4] as i32);

            if sum1_q15 >= 0 {
                let bits_res_q8 = silk_smulbb(
                    subfr_len,
                    silk_lin2log(sum1_q15.wrapping_add(penalty)) - (15 << 7),
                );

                let bits_tot_q8 = bits_res_q8 + ((cl_q5[k as usize] as i32) << (3 - 1));
                if bits_tot_q8 <= *rate_dist_q7 {
                    *rate_dist_q7 = bits_tot_q8;
                    *res_nrg_q15 = sum1_q15.wrapping_add(penalty);
                    *ind = k as i8;
                    *gain_q7_out = gain_tmp_q7;
                }
            }
        }
    }

    /// xorshift64 generator with a fixed seed, so the test data never
    /// changes.
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
    }

    /// Input of one search.
    struct Case {
        xx_q17: [i32; LTP_ORDER * LTP_ORDER],
        xx_in_q17: [i32; LTP_ORDER],
        subfr_len: i32,
        max_gain_q7: i32,
    }

    /// Correlations of a random signal with a lag, scaled like the encoder
    /// does: the matrix is symmetric, its diagonal is about 1 in Q17.
    fn correlation_case(rng: &mut XorShift) -> Case {
        let n = 80;
        let x: [i32; 90] = core::array::from_fn(|_| rng.range(-2000, 2000));
        let gain = rng.range(0, 120);
        // The target is a weighted sum of the delayed signal plus noise.
        let target: [i32; 80] =
            core::array::from_fn(|i| (x[i + 2] * gain) / 100 + rng.range(-300, 300));
        let corr = |a: &[i32], b: &[i32]| -> i64 {
            a.iter().zip(b).map(|(&p, &q)| p as i64 * q as i64).sum()
        };
        let energy = corr(&target, &target).max(1);
        let mut case = Case {
            xx_q17: [0; LTP_ORDER * LTP_ORDER],
            xx_in_q17: [0; LTP_ORDER],
            subfr_len: if rng.range(0, 1) == 0 { 40 } else { 80 },
            max_gain_q7: rng.range(20, 250),
        };
        for r in 0..LTP_ORDER {
            for c in 0..LTP_ORDER {
                case.xx_q17[LTP_ORDER * r + c] =
                    ((corr(&x[r..r + n], &x[c..c + n]) << 17) / energy) as i32;
            }
            case.xx_in_q17[r] = ((corr(&x[r..r + n], &target) << 17) / energy) as i32;
        }
        case
    }

    /// Any 32-bit values: the sums wrap around and about half of the vectors
    /// do not qualify.
    fn wrapping_case(rng: &mut XorShift) -> Case {
        let mut any = || rng.next() as i32;
        Case {
            xx_q17: core::array::from_fn(|_| any()),
            xx_in_q17: core::array::from_fn(|_| any()),
            subfr_len: any() >> (any() & 31),
            max_gain_q7: any() >> (any() & 31),
        }
    }

    /// Small values and a short subframe: many vectors get the same number
    /// of bits, so the rule for equal vectors decides.
    fn coarse_case(rng: &mut XorShift) -> Case {
        let scale = 1 << rng.range(0, 24);
        Case {
            xx_q17: core::array::from_fn(|_| rng.range(-3, 3) * scale),
            xx_in_q17: core::array::from_fn(|_| rng.range(-3, 3) * scale),
            subfr_len: rng.range(0, 2),
            max_gain_q7: rng.range(0, 200),
        }
    }

    /// Result of the original function for the first `l` vectors of one
    /// codebook: (index, residual energy, bits, gain).
    fn expected(case: &Case, codebook: usize, l: i32) -> (i8, i32, i32, i32) {
        // The gain keeps its old value when no vector qualifies.
        let (mut ind, mut res_nrg, mut rate_dist, mut gain) = (-7i8, -7i32, -7i32, -7i32);
        original(
            &mut ind,
            &mut res_nrg,
            &mut rate_dist,
            &mut gain,
            &case.xx_q17,
            0,
            &case.xx_in_q17,
            0,
            SILK_LTP_VQ_PTRS_Q7[codebook],
            SILK_LTP_VQ_GAIN_PTRS_Q7[codebook],
            SILK_LTP_GAIN_BITS_Q5_PTRS[codebook],
            case.subfr_len,
            case.max_gain_q7,
            l,
        );
        (ind, res_nrg, rate_dist, gain)
    }

    /// Check every entry point on one case and return how many of the three
    /// codebooks had no qualifying vector and how many had a tie for the
    /// best.
    fn check(case: &Case, what: &str) -> (usize, usize) {
        let mut none = 0;
        let mut ties = 0;
        for codebook in 0..3 {
            let size = SILK_LTP_VQ_SIZES[codebook];
            let want = expected(case, codebook, size);

            // The search the encoder uses (vector code on aarch64).
            let got = silk_vq_wmat_ec_codebook(
                codebook,
                &case.xx_q17,
                &case.xx_in_q17,
                case.subfr_len,
                case.max_gain_q7,
            );
            assert_eq!(
                (
                    got.ind,
                    got.res_nrg_q15,
                    got.rate_dist_q7,
                    got.gain_q7.unwrap_or(-7)
                ),
                want,
                "{what}, codebook {codebook}"
            );
            if got.gain_q7.is_none() {
                none += 1;
            }

            // The public entry point, with the matrix and the vector at an
            // offset inside longer slices and with a shortened codebook.
            let mut xx = [0x1234_5678i32; 3 + LTP_ORDER * LTP_ORDER];
            let mut xx_in = [0x1234_5678i32; 2 + LTP_ORDER];
            xx[3..].copy_from_slice(&case.xx_q17);
            xx_in[2..].copy_from_slice(&case.xx_in_q17);
            for l in [size, size - 1, size / 2 + 1, 1, 0, -3] {
                let want = expected(case, codebook, l);
                let (mut ind, mut res_nrg, mut rate_dist, mut gain) = (-7i8, -7i32, -7i32, -7i32);
                silk_vq_wmat_ec(
                    &mut ind,
                    &mut res_nrg,
                    &mut rate_dist,
                    &mut gain,
                    &xx,
                    3,
                    &xx_in,
                    2,
                    SILK_LTP_VQ_PTRS_Q7[codebook],
                    SILK_LTP_VQ_GAIN_PTRS_Q7[codebook],
                    SILK_LTP_GAIN_BITS_Q5_PTRS[codebook],
                    case.subfr_len,
                    case.max_gain_q7,
                    l,
                );
                assert_eq!(
                    (ind, res_nrg, rate_dist, gain),
                    want,
                    "{what}, codebook {codebook}, first {l} vectors"
                );
            }

            // A tie for the best: a shorter codebook that ends before the
            // winner still reaches the same number of bits.
            if want.3 != -7 && want.0 > 0 {
                let before = expected(case, codebook, want.0 as i32);
                if before.3 != -7 && before.2 == want.2 {
                    ties += 1;
                }
            }
        }
        (none, ties)
    }

    #[test]
    fn encoder_like_correlations_match_original() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut none = 0;
        for _ in 0..3_000 {
            none += check(&correlation_case(&mut rng), "correlations").0;
        }
        // Real correlations leave a positive error for some vector.
        assert_eq!(none, 0);
    }

    #[test]
    fn wrapping_values_match_original() {
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        let mut none = 0;
        let rounds = 20_000;
        for _ in 0..rounds {
            none += check(&wrapping_case(&mut rng), "wrapping values").0;
        }
        // The case without a qualifying vector must be among them.
        assert!(none > 0, "no search without a qualifying vector");
    }

    #[test]
    fn equal_vectors_match_original() {
        let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
        let mut ties = 0;
        for _ in 0..20_000 {
            ties += check(&coarse_case(&mut rng), "coarse values").1;
        }
        // The rule for equal vectors must have decided some searches.
        assert!(ties > 100, "only {ties} searches with a tie for the best");
    }

    #[test]
    fn special_values_match_original() {
        let zero = Case {
            xx_q17: [0; LTP_ORDER * LTP_ORDER],
            xx_in_q17: [0; LTP_ORDER],
            subfr_len: 80,
            max_gain_q7: 100,
        };
        check(&zero, "all zero");
        for fill in [i32::MIN, i32::MAX, -1, 1, 1 << 17, -(1 << 17), 1 << 24] {
            for (subfr_len, max_gain_q7) in [
                (0, 0),
                (40, 200),
                (80, -51),
                (i32::MAX, i32::MIN),
                (i32::MIN, i32::MAX),
                (-1, 255),
                (32767, 254),
                (32768, 0),
            ] {
                let case = Case {
                    xx_q17: [fill; LTP_ORDER * LTP_ORDER],
                    xx_in_q17: [fill; LTP_ORDER],
                    subfr_len,
                    max_gain_q7,
                };
                check(&case, "special values");
                let mixed = Case {
                    xx_q17: core::array::from_fn(|i| if i % 6 == 0 { fill } else { 0 }),
                    xx_in_q17: [fill >> 3; LTP_ORDER],
                    subfr_len,
                    max_gain_q7,
                };
                check(&mixed, "diagonal only");
            }
        }
    }
}
