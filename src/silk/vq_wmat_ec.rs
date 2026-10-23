//! Search of the codebooks for the gains of the long-term predictor (LTP),
//! libopus `silk_VQ_WMat_EC`.
//!
//! For every codebook vector the search computes the prediction error that
//! the vector leaves, turns it into a number of bits and keeps the vector
//! with the lowest total. There are three versions with the same result:
//!
//! - `search_scalar` handles one vector at a time. It is the portable code
//!   and the reference in the tests.
//! - `neon::search` (aarch64) handles four vectors at a time.
//! - `x86::search` (x86-64 with SSE4.1) does the same as `neon::search`.

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
#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
#[inline(always)]
fn ltp_vq_bits_q8(res_nrg_q15: i32, subfr_len: i32, cl_q5: u8) -> i32 {
    let bits_res_q8 = silk_smulbb(subfr_len, silk_lin2log(res_nrg_q15) - (15 << 7));
    bits_res_q8 + ((cl_q5 as i32) << (3 - 1))
}

/// The search, one codebook vector at a time. The three slices are the
/// codebook: its vectors, the gain of each vector and its code length.
#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
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
        let neg_xx_q24 = negate_to_q24(xx_in_q17);
        let found = neon::search(
            xx_q17,
            &neg_xx_q24,
            neon::CODEBOOKS[codebook],
            subfr_len,
            max_gain_q7,
        );
        vector_choice(codebook, xx_q17, &neg_xx_q24, max_gain_q7, found)
    }
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_sse41() {
        let neg_xx_q24 = negate_to_q24(xx_in_q17);
        // SAFETY: the processor has SSE4.1.
        let found = unsafe {
            x86::search(xx_q17, &neg_xx_q24, x86::CODEBOOKS[codebook], subfr_len, max_gain_q7)
        };
        return vector_choice(codebook, xx_q17, &neg_xx_q24, max_gain_q7, found);
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

/// The result of a vector search (`neon::search` or `x86::search`): the
/// number of the best vector of `codebook` and its total bits, or `None`.
/// The vector code leaves out the residual energy of the winner, so it is
/// computed again for this one vector.
#[cfg(any(all(target_arch = "aarch64", target_feature = "neon"), target_arch = "x86_64"))]
#[inline(always)]
fn vector_choice(
    codebook: usize,
    xx_q17: &CorrMatrix,
    neg_xx_q24: &[i32; LTP_ORDER],
    max_gain_q7: i32,
    found: Option<(usize, i32)>,
) -> LtpVqChoice {
    let Some((k, rate_dist_q7)) = found else {
        return LtpVqChoice::NONE;
    };
    let gain_q7 = SILK_LTP_VQ_GAIN_PTRS_Q7[codebook][k] as i32;
    let sum1_q15 = ltp_vq_error_q15(xx_q17, neg_xx_q24, &SILK_LTP_VQ_PTRS_Q7[codebook][k]);
    LtpVqChoice {
        ind: k as i8,
        res_nrg_q15: sum1_q15.wrapping_add(gain_penalty_q15(gain_q7, max_gain_q7)),
        rate_dist_q7,
        gain_q7: Some(gain_q7),
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

/// The search of `neon::search` for x86-64 processors with SSE4.1: four
/// codebook vectors at a time, one per 32-bit lane. libopus's
/// `silk_VQ_WMat_EC_sse4_1` handles one vector at a time and only puts the
/// first row of the matrix into vectors.
///
/// Two NEON operations have no x86 instruction:
///
/// - `silk_smulwb(row, cb)`, `vqdmulhq_s32` on NEON, is `mulw`: two
///   `_mm_mul_epi32` with `cb << 16`, as in the SILK quantizers.
/// - `silk_lin2log` needs the number of leading zeros (`vclzq_s32`). `lin2log`
///   reads the position of the leading one and the 7 bits below it from the
///   conversion to `f32`.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::CorrMatrix;
    use crate::silk::define::LTP_ORDER;
    use crate::silk::tables::{
        SILK_LTP_GAIN_BITS_Q5_0, SILK_LTP_GAIN_BITS_Q5_1, SILK_LTP_GAIN_BITS_Q5_2,
        SILK_LTP_GAIN_VQ_0_GAIN, SILK_LTP_GAIN_VQ_1_GAIN, SILK_LTP_GAIN_VQ_2_GAIN,
        SILK_LTP_VQ_0_Q7, SILK_LTP_VQ_1_Q7, SILK_LTP_VQ_2_Q7,
    };
    use core::arch::x86_64::*;

    /// Four codebook vectors side by side: lane `n` of every field belongs
    /// to vector number `4 * group + n` of the codebook.
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub(super) struct Group {
        /// `cb_q7[c]` holds coefficient `c` of the four vectors.
        cb_q7: [[i32; 4]; LTP_ORDER],
        /// The same values shifted left by 16, the factors of `mulw`.
        cb_q23: [[i32; 4]; LTP_ORDER],
        /// Lanes 1 and 3 of `cb_q23` in lanes 0 and 2.
        cb_q23_odd: [[i32; 4]; LTP_ORDER],
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
            cb_q23: [[0; 4]; LTP_ORDER],
            cb_q23_odd: [[0; 4]; LTP_ORDER],
            gain_q7: [0; 4],
            half_cl_q8: [0; 4],
        }; G];
        let mut k = 0;
        while k < N {
            let (group, lane) = (k / 4, k % 4);
            let mut c = 0;
            while c < LTP_ORDER {
                let cb = cb_q7[k][c] as i32;
                out[group].cb_q7[c][lane] = cb;
                out[group].cb_q23[c][lane] = cb << 16;
                if lane % 2 == 1 {
                    out[group].cb_q23_odd[c][lane - 1] = cb << 16;
                }
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

    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn load(lanes: &[i32; 4]) -> __m128i {
        // SAFETY: reads the 16 bytes of the array; the load needs no
        // alignment.
        unsafe { _mm_loadu_si128(lanes.as_ptr().cast()) }
    }

    /// `(x * c) >> 16` per lane, as `silk_smulwb(x, c)`, with `c << 16` in
    /// `factor` and the factors of lanes 1 and 3 in lanes 0 and 2 of
    /// `factor_odd`.
    ///
    /// `_mm_mul_epi32` gives the 64-bit products `x * c << 16` of lanes 0
    /// and 2. Their upper halves, in lanes 1 and 3, are the results. The
    /// second multiplication does the same for lanes 1 and 3 moved down.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn mulw(x: __m128i, factor: __m128i, factor_odd: __m128i) -> __m128i {
        let even = _mm_mul_epi32(x, factor);
        let odd = _mm_mul_epi32(_mm_srli_epi64::<32>(x), factor_odd);
        _mm_blend_epi16::<0b1100_1100>(_mm_srli_epi64::<32>(even), odd)
    }

    /// `silk_lin2log(v)` per lane, for every value of `v`.
    ///
    /// For `v > 0` let `k` be the position of the leading one. The function
    /// needs `k` and the 7 bits below the leading one. Both are read from
    /// `v` converted to `f32`: the exponent is `k`, and the upper 7 bits of
    /// the mantissa are those bits. Before the conversion each bit is
    /// cleared that has a set bit 8 positions above it. This keeps the
    /// leading one and the 7 bits below it, and clears bit `k - 8`. The
    /// rounding of the conversion can then carry at most into bit `k - 8`,
    /// so the 8 bits that are read are exact.
    ///
    /// `v = 0` converts to `0.0`, whose exponent field gives `k = -127`;
    /// libopus counts 32 leading zeros there, which is `k = -1`. A negative
    /// `v` has no leading zero: `k = 31`, and the fraction is bits 24 to 30.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn lin2log(v: __m128i) -> __m128i {
        let kept = _mm_andnot_si128(_mm_srli_epi32::<8>(v), v);
        let bits = _mm_castps_si128(_mm_cvtepi32_ps(kept));
        let k = _mm_sub_epi32(_mm_srli_epi32::<23>(bits), _mm_set1_epi32(127));
        let k = _mm_max_epi32(k, _mm_set1_epi32(-1));
        let frac = _mm_and_si128(_mm_srli_epi32::<16>(bits), _mm_set1_epi32(0x7f));

        let negative = _mm_srai_epi32::<31>(v);
        let k = _mm_blendv_epi8(k, _mm_set1_epi32(31), negative);
        let frac_negative = _mm_and_si128(_mm_srli_epi32::<24>(v), _mm_set1_epi32(0x7f));
        let frac = _mm_blendv_epi8(frac, frac_negative, negative);

        // frac + ((frac * (128 - frac) * 179) >> 16) + (k << 7). The product
        // is at most 4096 and fits in the low 16 bits of the lane.
        let curve = _mm_mullo_epi16(frac, _mm_sub_epi32(_mm_set1_epi32(128), frac));
        let curve = _mm_srai_epi32::<16>(_mm_madd_epi16(curve, _mm_set1_epi32(179)));
        _mm_add_epi32(_mm_add_epi32(frac, curve), _mm_slli_epi32::<7>(k))
    }

    /// `lin2log` of four values, for the tests.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[cfg(test)]
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn lin2log_lanes(v: [i32; 4]) -> [i32; 4] {
        let out = lin2log(_mm_setr_epi32(v[0], v[1], v[2], v[3]));
        [
            _mm_extract_epi32::<0>(out),
            _mm_extract_epi32::<1>(out),
            _mm_extract_epi32::<2>(out),
            _mm_extract_epi32::<3>(out),
        ]
    }

    /// The search of `super::search_scalar`, four codebook vectors at a
    /// time. Returns the number of the best vector and its total bits (Q8),
    /// or `None` when no vector qualified.
    ///
    /// Every sum of the scalar code wraps around, so the order of the
    /// additions is free here.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn search(
        xx_q17: &CorrMatrix,
        neg_xx_q24: &[i32; LTP_ORDER],
        codebook: &[Group],
        subfr_len: i32,
        max_gain_q7: i32,
    ) -> Option<(usize, i32)> {
        // Entry (row, column) of the matrix in all four lanes.
        let xx = |row: usize, column: usize| _mm_set1_epi32(xx_q17[LTP_ORDER * row + column]);
        let (xx00, xx01, xx02, xx03, xx04) = (xx(0, 0), xx(0, 1), xx(0, 2), xx(0, 3), xx(0, 4));
        let (xx11, xx12, xx13, xx14) = (xx(1, 1), xx(1, 2), xx(1, 3), xx(1, 4));
        let (xx22, xx23, xx24) = (xx(2, 2), xx(2, 3), xx(2, 4));
        let (xx33, xx34) = (xx(3, 3), xx(3, 4));
        let xx44 = xx(4, 4);
        let neg0 = _mm_set1_epi32(neg_xx_q24[0]);
        let neg1 = _mm_set1_epi32(neg_xx_q24[1]);
        let neg2 = _mm_set1_epi32(neg_xx_q24[2]);
        let neg3 = _mm_set1_epi32(neg_xx_q24[3]);
        let neg4_twice = _mm_set1_epi32(neg_xx_q24[4] << 1);
        let max_gain = _mm_set1_epi32(max_gain_q7);
        // `silk_smulbb` uses the low 16 bits of both values: the subframe
        // length in the low half of every lane, zero in the high half.
        let subfr_len = _mm_set1_epi32(subfr_len as u16 as i32);
        let zero = _mm_setzero_si128();
        // a + b * c per lane, wrapping, as `silk_mla`.
        let mla = |a: __m128i, b: __m128i, c: __m128i| _mm_add_epi32(a, _mm_mullo_epi32(b, c));

        let mut best_bits = _mm_set1_epi32(i32::MAX);
        let mut best_number = _mm_set1_epi32(-1);
        let mut numbers = _mm_setr_epi32(0, 1, 2, 3);

        for group in codebook {
            let cb0 = load(&group.cb_q7[0]);
            let cb1 = load(&group.cb_q7[1]);
            let cb2 = load(&group.cb_q7[2]);
            let cb3 = load(&group.cb_q7[3]);
            let cb4 = load(&group.cb_q7[4]);
            let factor = |c: usize| (load(&group.cb_q23[c]), load(&group.cb_q23_odd[c]));

            // Prediction error, row by row as in `ltp_vq_error_q15`.
            let mut row = mla(neg0, xx01, cb1);
            row = mla(row, xx02, cb2);
            row = mla(row, xx03, cb3);
            row = mla(row, xx04, cb4);
            row = mla(_mm_slli_epi32::<1>(row), xx00, cb0);
            let (f, f_odd) = factor(0);
            let mut sum1 = mulw(row, f, f_odd);

            row = mla(neg1, xx12, cb2);
            row = mla(row, xx13, cb3);
            row = mla(row, xx14, cb4);
            row = mla(_mm_slli_epi32::<1>(row), xx11, cb1);
            let (f, f_odd) = factor(1);
            sum1 = _mm_add_epi32(sum1, mulw(row, f, f_odd));

            row = mla(neg2, xx23, cb3);
            row = mla(row, xx24, cb4);
            row = mla(_mm_slli_epi32::<1>(row), xx22, cb2);
            let (f, f_odd) = factor(2);
            sum1 = _mm_add_epi32(sum1, mulw(row, f, f_odd));

            row = mla(neg3, xx34, cb4);
            row = mla(_mm_slli_epi32::<1>(row), xx33, cb3);
            let (f, f_odd) = factor(3);
            sum1 = _mm_add_epi32(sum1, mulw(row, f, f_odd));

            row = mla(neg4_twice, xx44, cb4);
            let (f, f_odd) = factor(4);
            sum1 = _mm_add_epi32(sum1, mulw(row, f, f_odd));
            sum1 = _mm_add_epi32(sum1, _mm_set1_epi32(32801));

            // Residual energy with the penalty for too large a gain.
            let gain = load(&group.gain_q7);
            let penalty = _mm_slli_epi32::<11>(_mm_max_epi32(_mm_sub_epi32(gain, max_gain), zero));
            let res_nrg = _mm_add_epi32(sum1, penalty);

            // silk_smulbb(subfr_len, log - (15 << 7)): the difference lies
            // between -2048 and 2175 and keeps its value in 16 bits.
            let log = _mm_sub_epi32(lin2log(res_nrg), _mm_set1_epi32(15 << 7));
            let bits = _mm_madd_epi16(log, subfr_len);
            let bits = _mm_add_epi32(bits, load(&group.half_cl_q8));

            // A vector qualifies when its error is not negative. In each
            // lane the later of two equal vectors wins, as in the scalar
            // code.
            let qualifies = _mm_cmpgt_epi32(sum1, _mm_set1_epi32(-1));
            let take = _mm_andnot_si128(_mm_cmpgt_epi32(bits, best_bits), qualifies);
            best_bits = _mm_blendv_epi8(best_bits, bits, take);
            best_number = _mm_blendv_epi8(best_number, numbers, take);
            numbers = _mm_add_epi32(numbers, _mm_set1_epi32(4));
        }

        let lanes = |v: __m128i| {
            [
                _mm_extract_epi32::<0>(v),
                _mm_extract_epi32::<1>(v),
                _mm_extract_epi32::<2>(v),
                _mm_extract_epi32::<3>(v),
            ]
        };
        let (bits_q8, number) = (lanes(best_bits), lanes(best_number));

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

    /// Entropy-constrained search of one LTP codebook for one subframe (libopus
    /// `silk_VQ_WMat_EC`), with libopus's arguments, over the portable search.
    ///
    /// The matrix is the 25 values of `xx_q17` from `xx_q17_ptr` and the vector
    /// the 5 values of `xx_q17_in` from `xx_q17_in_ptr`. The first `l` entries of
    /// `cb_q7`, `cb_gain_q7` and `cl_q5` are the codebook. `gain_q7_out` is only
    /// written when a vector qualified.
    #[allow(clippy::too_many_arguments)]
    fn silk_vq_wmat_ec(
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

    /// `silk_vq_wmat_ec_codebook`, with the search of this processor (NEON,
    /// SSE4.1 or the portable loop), against `silk_VQ_WMat_EC_c` of the
    /// libopus these tests link (silk/VQ_WMat_EC.c, which the float build
    /// shares with the fixed-point one). The wrapping cases are left out:
    /// there the C code overflows signed integers, which C leaves undefined.
    #[cfg(feature = "std")]
    #[test]
    fn codebook_search_matches_libopus() {
        use audiopus_sys as _;
        unsafe extern "C" {
            /// `void silk_VQ_WMat_EC_c(opus_int8 *ind, opus_int32
            /// *res_nrg_Q15, opus_int32 *rate_dist_Q8, opus_int *gain_Q7,
            /// const opus_int32 *XX_Q17, const opus_int32 *xX_Q17, const
            /// opus_int8 *cb_Q7, const opus_uint8 *cb_gain_Q7, const
            /// opus_uint8 *cl_Q5, const opus_int subfr_len, const opus_int32
            /// max_gain_Q7, const opus_int L)`.
            #[link_name = "silk_VQ_WMat_EC_c"]
            fn c_vq_wmat_ec(
                ind: *mut i8,
                res_nrg_q15: *mut i32,
                rate_dist_q8: *mut i32,
                gain_q7: *mut i32,
                xx_q17: *const i32,
                x_x_q17: *const i32,
                cb_q7: *const i8,
                cb_gain_q7: *const u8,
                cl_q5: *const u8,
                subfr_len: i32,
                max_gain_q7: i32,
                l: i32,
            );
        }
        let mut rng = XorShift(0x3C6E_F372_FE94_F82B);
        let mut found = 0;
        for round in 0..20_000 {
            let case = if round % 2 == 0 { correlation_case(&mut rng) } else { coarse_case(&mut rng) };
            for codebook in 0..3 {
                let cb = SILK_LTP_VQ_PTRS_Q7[codebook];
                // libopus leaves `gain_Q7` unwritten when no vector qualifies.
                let (mut ind, mut res_nrg, mut rate_dist, mut gain) = (0i8, 0i32, 0i32, -1i32);
                // SAFETY: the matrix holds 25 values, the vector 5, the
                // codebook `cb.len()` rows of 5 and the two tables one value
                // per row; the outputs are single values.
                unsafe {
                    c_vq_wmat_ec(
                        &mut ind,
                        &mut res_nrg,
                        &mut rate_dist,
                        &mut gain,
                        case.xx_q17.as_ptr(),
                        case.xx_in_q17.as_ptr(),
                        cb.as_ptr().cast(),
                        SILK_LTP_VQ_GAIN_PTRS_Q7[codebook].as_ptr(),
                        SILK_LTP_GAIN_BITS_Q5_PTRS[codebook].as_ptr(),
                        case.subfr_len,
                        case.max_gain_q7,
                        cb.len() as i32,
                    )
                };
                let want = (ind, res_nrg, rate_dist, (gain >= 0).then_some(gain));
                found += usize::from(gain >= 0);
                let got = silk_vq_wmat_ec_codebook(
                    codebook,
                    &case.xx_q17,
                    &case.xx_in_q17,
                    case.subfr_len,
                    case.max_gain_q7,
                );
                let got = (got.ind, got.res_nrg_q15, got.rate_dist_q7, got.gain_q7);
                assert_eq!(got, want, "round {round}, codebook {codebook}");
            }
        }
        assert!(found > 30_000, "only {found} searches found a vector");
    }

    /// The x86 search against the portable search on all kinds of cases,
    /// whatever the dispatch chooses.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_search_matches_scalar() {
        if !crate::compat::x86_has_sse41() {
            return;
        }
        let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
        for round in 0..30_000 {
            let case = match round % 3 {
                0 => correlation_case(&mut rng),
                1 => wrapping_case(&mut rng),
                _ => coarse_case(&mut rng),
            };
            for codebook in 0..3 {
                let want = search_scalar(
                    &case.xx_q17,
                    &case.xx_in_q17,
                    SILK_LTP_VQ_PTRS_Q7[codebook],
                    SILK_LTP_VQ_GAIN_PTRS_Q7[codebook],
                    SILK_LTP_GAIN_BITS_Q5_PTRS[codebook],
                    case.subfr_len,
                    case.max_gain_q7,
                );
                let neg_xx_q24 = negate_to_q24(&case.xx_in_q17);
                // SAFETY: the processor has SSE4.1 (checked above).
                let got = unsafe {
                    x86::search(
                        &case.xx_q17,
                        &neg_xx_q24,
                        x86::CODEBOOKS[codebook],
                        case.subfr_len,
                        case.max_gain_q7,
                    )
                };
                // The whole choice, the residual energy of the winner too.
                let got = vector_choice(codebook, &case.xx_q17, &neg_xx_q24, case.max_gain_q7, got);
                let fields = |c: &LtpVqChoice| (c.ind, c.res_nrg_q15, c.rate_dist_q7, c.gain_q7);
                assert_eq!(fields(&got), fields(&want), "round {round}, codebook {codebook}");
            }
        }
    }

    /// The `silk_lin2log` of the x86 search against the scalar function, on
    /// every power of two, its neighbours and random values of both signs.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_lin2log_matches_scalar() {
        if !crate::compat::x86_has_sse41() {
            return;
        }
        let mut values = vec![0, 1, -1, i32::MAX, i32::MIN, i32::MIN + 1];
        for k in 0..31 {
            let p = 1i32 << k;
            values.extend([p - 1, p, p + 1, -p, -p - 1, -p + 1, p | (p >> 1)]);
            // A run of ones below the leading one that rounds up in `f32`.
            values.push(i32::MAX >> (30 - k));
        }
        let mut rng = XorShift(0x1F83_D9AB_FB41_BD6B);
        values.extend((0..100_000).map(|_| (rng.next() as i32) >> (rng.next() % 32)));
        for chunk in values.chunks(4) {
            let mut v = [0i32; 4];
            v[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: the processor has SSE4.1 (checked above).
            let got = unsafe { x86::lin2log_lanes(v) };
            assert_eq!(got, v.map(silk_lin2log), "{v:?}");
        }
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
