use crate::silk::define::*;
use crate::silk::interpolate::silk_interpolate;
use crate::silk::macros::*;
use crate::silk::nlsf::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tuning_parameters::*;

#[cfg(not(feature = "std"))]
use crate::compat::Math;

pub fn silk_find_lpc_fix(
    ps_enc_c: &mut SilkEncoderStateCommon,
    nlsf_q15: &mut [i16],
    x: &[i16],
    min_inv_gain_q30: i32,
) {
    let mut res_nrg: i32 = 0;
    let mut res_nrg_q: i32 = 0;
    let mut a_q16 = [0i32; MAX_LPC_ORDER];
    let d = ps_enc_c.predict_lpc_order as usize;
    let subfr_length = (ps_enc_c.subfr_length + ps_enc_c.predict_lpc_order) as usize;

    ps_enc_c.indices.nlsf_interp_coef_q2 = 4;

    silk_burg_modified_fix(
        &mut res_nrg,
        &mut res_nrg_q,
        &mut a_q16,
        x,
        min_inv_gain_q30,
        subfr_length,
        ps_enc_c.nb_subfr as usize,
        d,
    );

    if ps_enc_c.use_interpolated_nlsfs != 0
        && ps_enc_c.first_frame_after_reset == 0
        && ps_enc_c.nb_subfr == MAX_NB_SUBFR as i32
    {
        let mut res_tmp_nrg: i32 = 0;
        let mut res_tmp_nrg_q: i32 = 0;
        let mut a_tmp_q16 = [0i32; MAX_LPC_ORDER];
        silk_burg_modified_fix(
            &mut res_tmp_nrg,
            &mut res_tmp_nrg_q,
            &mut a_tmp_q16,
            &x[2 * subfr_length..],
            min_inv_gain_q30,
            subfr_length,
            2,
            d,
        );

        let shift = res_tmp_nrg_q - res_nrg_q;
        if shift >= 0 {
            if shift < 32 {
                res_nrg -= silk_rshift(res_tmp_nrg, shift);
            }
        } else {
            debug_assert!(shift > -32);
            res_nrg = silk_rshift(res_nrg, -shift) - res_tmp_nrg;
            res_nrg_q = res_tmp_nrg_q;
        }

        silk_a2nlsf(nlsf_q15, &mut a_tmp_q16, d);

        let lpc_res_len = 2 * subfr_length;

        let mut lpc_res = [0i16; 2 * (MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER)];

        for k in (0..=3).rev() {
            let nlsf0_q15 = silk_interpolate(&ps_enc_c.prev_nlsf_q15, nlsf_q15, k, d);

            let mut a_tmp_q12 = [0i16; MAX_LPC_ORDER];
            silk_nlsf2a(&mut a_tmp_q12, &nlsf0_q15, d);

            silk_lpc_analysis_filter(&mut lpc_res, x, &a_tmp_q12, lpc_res_len, d, 0);

            let mut res_nrg0: i32 = 0;
            let mut rshift0: i32 = 0;
            silk_sum_sqr_shift(
                &mut res_nrg0,
                &mut rshift0,
                &lpc_res[d..d + (subfr_length - d)],
                subfr_length - d,
            );

            let mut res_nrg1: i32 = 0;
            let mut rshift1: i32 = 0;
            silk_sum_sqr_shift(
                &mut res_nrg1,
                &mut rshift1,
                &lpc_res[d + subfr_length..d + subfr_length + (subfr_length - d)],
                subfr_length - d,
            );

            let res_nrg_interp_q: i32;
            let shift = rshift0 - rshift1;
            if shift >= 0 {
                res_nrg1 = silk_rshift(res_nrg1, shift);
                res_nrg_interp_q = -rshift0;
            } else {
                res_nrg0 = silk_rshift(res_nrg0, -shift);
                res_nrg_interp_q = -rshift1;
            }
            let res_nrg_interp = res_nrg0.wrapping_add(res_nrg1);

            let shift = res_nrg_interp_q - res_nrg_q;
            let is_interp_lower = if shift >= 0 {
                if shift < 32 {
                    silk_rshift(res_nrg_interp, shift) < res_nrg
                } else {
                    false
                }
            } else if -shift < 32 {
                res_nrg_interp < silk_rshift(res_nrg, -shift)
            } else {
                false
            };

            if is_interp_lower {
                res_nrg = res_nrg_interp;
                res_nrg_q = res_nrg_interp_q;
                ps_enc_c.indices.nlsf_interp_coef_q2 = k as i8;
            }
        }
    }

    if ps_enc_c.indices.nlsf_interp_coef_q2 == 4 {
        silk_a2nlsf(nlsf_q15, &mut a_q16, d);
    }
}

pub fn silk_residual_energy_fix(
    nrgs: &mut [i32],
    nrgs_q: &mut [i32],
    x: &[i16],
    a_q12: &[[i16; MAX_LPC_ORDER]; 2],
    gains: &[i32],
    subfr_length: i32,
    nb_subfr: i32,
    lpc_order: i32,
) {
    let offset: usize = (lpc_order + subfr_length) as usize;
    let mut rshift: i32 = 0;
    let mut lz1: i32;
    let mut lz2: i32;
    let mut x_ptr_idx: usize = 0;
    let mut tmp32: i32;

    if (nb_subfr >> 1) * (MAX_NB_SUBFR as i32 >> 1) != nb_subfr {
        return;
    }

    let mut lpc_res = [0i16; (MAX_NB_SUBFR / 2) * (MAX_LPC_ORDER + MAX_SUB_FRAME_LENGTH)];

    for i in 0..(nb_subfr as usize >> 1) {
        silk_lpc_analysis_filter(
            &mut lpc_res,
            &x[x_ptr_idx..],
            &a_q12[i],
            (MAX_NB_SUBFR >> 1) * offset,
            lpc_order as usize,
            0,
        );

        let mut lpc_res_idx = lpc_order as usize;
        for j in 0..(MAX_NB_SUBFR >> 1) {
            silk_sum_sqr_shift(
                &mut nrgs[i * (MAX_NB_SUBFR >> 1) + j],
                &mut rshift,
                &lpc_res[lpc_res_idx..lpc_res_idx + subfr_length as usize],
                subfr_length as usize,
            );

            nrgs_q[i * (MAX_NB_SUBFR >> 1) + j] = -rshift;

            lpc_res_idx += offset;
        }

        x_ptr_idx += (MAX_NB_SUBFR >> 1) * offset;
    }

    for i in 0..nb_subfr as usize {
        lz1 = silk_clz32(nrgs[i]) - 1;
        lz2 = silk_clz32(gains[i]) - 1;

        tmp32 = silk_lshift(gains[i], lz2);

        tmp32 = silk_smmul(tmp32, tmp32);

        nrgs[i] = silk_smmul(tmp32, silk_lshift(nrgs[i], lz1));
        nrgs_q[i] += lz1 + 2 * lz2 - 32 - 32;
    }
}

/// Fixed-point format of the prediction coefficients inside the Burg
/// recursion.
const BURG_QA: i32 = 25;

/// Longest subframe the encoder analyses: 5 ms at 16 kHz plus the samples of
/// the LPC order before it. The vector version handles subframes up to this
/// length.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const BURG_MAX_SUBFR_LENGTH: usize = MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER;

/// Step `n` of the Burg recursion, the part that reads the signal: add the
/// contribution of every subframe to the four correlation arrays.
///
/// `c_first_row` and `c_last_row` are updated at positions `0..n`, `ca_f`
/// and `ca_b` at positions `0..=n`. `af_qa[..n]` are the prediction
/// coefficients found so far. `rshifts` is the scaling of the correlations:
/// above -2 the products keep their upper bits, otherwise the lower ones.
///
/// This is the code for every target and the reference for
/// `burg_neon::add_subframes`.
#[allow(clippy::too_many_arguments)]
fn burg_add_subframes_scalar(
    c_first_row: &mut [i32; MAX_LPC_ORDER],
    c_last_row: &mut [i32; MAX_LPC_ORDER],
    ca_f: &mut [i32; MAX_LPC_ORDER + 1],
    ca_b: &mut [i32; MAX_LPC_ORDER + 1],
    af_qa: &[i32; MAX_LPC_ORDER],
    x: &[i16],
    subfr_length: usize,
    nb_subfr: usize,
    n: usize,
    rshifts: i32,
) {
    const QA: i32 = BURG_QA;
    if rshifts > -2 {
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            let x1 = (x[x_ptr + n] as i32)
                .wrapping_shl((16 - rshifts) as u32)
                .wrapping_neg();
            let x2 = (x[x_ptr + subfr_length - n - 1] as i32)
                .wrapping_shl((16 - rshifts) as u32)
                .wrapping_neg();
            let mut tmp1 = (x[x_ptr + n] as i32).wrapping_shl((QA - 16) as u32);
            let mut tmp2 = (x[x_ptr + subfr_length - n - 1] as i32).wrapping_shl((QA - 16) as u32);
            for k in 0..n {
                c_first_row[k] = silk_smlawb(c_first_row[k], x1, x[x_ptr + n - k - 1] as i32);
                c_last_row[k] =
                    silk_smlawb(c_last_row[k], x2, x[x_ptr + subfr_length - n + k] as i32);
                let atmp_qa = af_qa[k];
                tmp1 = silk_smlawb(tmp1, atmp_qa, x[x_ptr + n - k - 1] as i32);
                tmp2 = silk_smlawb(tmp2, atmp_qa, x[x_ptr + subfr_length - n + k] as i32);
            }
            tmp1 = tmp1.wrapping_neg().wrapping_shl((32 - QA - rshifts) as u32);
            tmp2 = tmp2.wrapping_neg().wrapping_shl((32 - QA - rshifts) as u32);
            for k in 0..=n {
                ca_f[k] = silk_smlawb(ca_f[k], tmp1, x[x_ptr + n - k] as i32);
                ca_b[k] = silk_smlawb(ca_b[k], tmp2, x[x_ptr + subfr_length - n + k - 1] as i32);
            }
        }
    } else {
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            let x1 = (x[x_ptr + n] as i32)
                .wrapping_shl((-rshifts) as u32)
                .wrapping_neg();
            let x2 = (x[x_ptr + subfr_length - n - 1] as i32)
                .wrapping_shl((-rshifts) as u32)
                .wrapping_neg();
            let mut tmp1 = (x[x_ptr + n] as i32).wrapping_shl(17);
            let mut tmp2 = (x[x_ptr + subfr_length - n - 1] as i32).wrapping_shl(17);
            for k in 0..n {
                c_first_row[k] = silk_mla(c_first_row[k], x1, x[x_ptr + n - k - 1] as i32);
                c_last_row[k] = silk_mla(c_last_row[k], x2, x[x_ptr + subfr_length - n + k] as i32);
                let atmp1 = silk_rshift_round(af_qa[k], QA - 17);

                tmp1 = (tmp1 as u32)
                    .wrapping_add((x[x_ptr + n - k - 1] as i32 as u32).wrapping_mul(atmp1 as u32))
                    as i32;
                tmp2 = (tmp2 as u32).wrapping_add(
                    (x[x_ptr + subfr_length - n + k] as i32 as u32).wrapping_mul(atmp1 as u32),
                ) as i32;
            }
            tmp1 = tmp1.wrapping_neg();
            tmp2 = tmp2.wrapping_neg();
            for k in 0..=n {
                ca_f[k] = silk_smlaww(
                    ca_f[k],
                    tmp1,
                    (x[x_ptr + n - k] as i32).wrapping_shl((-rshifts - 1) as u32),
                );
                ca_b[k] = silk_smlaww(
                    ca_b[k],
                    tmp2,
                    (x[x_ptr + subfr_length - n + k - 1] as i32)
                        .wrapping_shl((-rshifts - 1) as u32),
                );
            }
        }
    }
}

/// Inner product of two vectors of 16-bit samples as a 64-bit sum (libopus
/// `silk_inner_prod16`). The slices have the same length, so the loop has no
/// bounds checks and the compiler turns it into vector code. The sum cannot
/// overflow: a slice has fewer than 2^33 samples.
#[inline(always)]
fn inner_prod16_i64(a: &[i16], b: &[i16]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&a, &b)| a as i64 * b as i64).sum()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod burg_neon {
    use super::BURG_QA as QA;
    use crate::silk::define::MAX_LPC_ORDER;
    use crate::silk::macros::silk_rshift_round;
    use core::arch::aarch64::*;

    /// The samples at one position of up to four subframes, as 32-bit lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    fn widen(samples: &[i16; 4]) -> int32x4_t {
        // SAFETY: the pointer comes from a `[i16; 4]`, which is what
        // `vld1_s16` reads.
        unsafe { vmovl_s16(vld1_s16(samples.as_ptr())) }
    }

    /// `(a * b) >> 16` per lane with a 64-bit product (`silk_smulww`).
    #[inline]
    #[target_feature(enable = "neon")]
    fn smulww(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        let low = vmull_s32(vget_low_s32(a), vget_low_s32(b));
        let high = vmull_high_s32(a, b);
        vcombine_s32(vshrn_n_s64::<16>(low), vshrn_n_s64::<16>(high))
    }

    /// Write up to four subframes of `x` side by side: afterwards
    /// `side_by_side[i][s]` is sample `i` of subframe `s`, and zero for a
    /// subframe that does not exist. `x` is `nb_subfr` subframes long and
    /// `side_by_side` one subframe.
    #[target_feature(enable = "neon")]
    pub(super) fn lay_side_by_side(side_by_side: &mut [[i16; 4]], x: &[i16], nb_subfr: usize) {
        let subfr_length = side_by_side.len();
        debug_assert!(nb_subfr <= 4 && x.len() == nb_subfr * subfr_length);
        let subframe = |s: usize| -> &[i16] {
            if s < nb_subfr {
                &x[s * subfr_length..(s + 1) * subfr_length]
            } else {
                &[]
            }
        };
        let subframes = [subframe(0), subframe(1), subframe(2), subframe(3)];
        // Eight samples of one subframe, or zeros if it does not exist.
        let load = |s: usize, at: usize| -> int16x8_t {
            match subframes[s].get(at..at + 8) {
                // SAFETY: `block` is 8 samples long, which `vld1q_s16`
                // reads.
                Some(block) => unsafe { vld1q_s16(block.as_ptr()) },
                None => vdupq_n_s16(0),
            }
        };

        // Eight positions at a time: interleave the four subframes.
        let (blocks, rest) = side_by_side.as_chunks_mut::<8>();
        let whole = 8 * blocks.len();
        for (block, rows) in blocks.iter_mut().enumerate() {
            let at = 8 * block;
            let (a, b, c, d) = (load(0, at), load(1, at), load(2, at), load(3, at));
            let (ac_low, ac_high) = (vzip1q_s16(a, c), vzip2q_s16(a, c));
            let (bd_low, bd_high) = (vzip1q_s16(b, d), vzip2q_s16(b, d));
            let out: &mut [i16] = rows.as_flattened_mut();
            // SAFETY: `out` is 8 rows of 4 samples. Each `vst1q_s16` writes 8
            // samples at an offset of at most 24.
            unsafe {
                let out = out.as_mut_ptr();
                vst1q_s16(out, vzip1q_s16(ac_low, bd_low));
                vst1q_s16(out.add(8), vzip2q_s16(ac_low, bd_low));
                vst1q_s16(out.add(16), vzip1q_s16(ac_high, bd_high));
                vst1q_s16(out.add(24), vzip2q_s16(ac_high, bd_high));
            }
        }
        // The positions that are left.
        for (row, at) in rest.iter_mut().zip(whole..) {
            *row = core::array::from_fn(|s| subframes[s].get(at).copied().unwrap_or(0));
        }
    }

    /// `burg_add_subframes_scalar` with the subframes in the four lanes of
    /// a vector: `side_by_side[i]` holds sample `i` of every subframe (zero
    /// in the lanes of subframes that do not exist; they add nothing). The
    /// slice is one subframe long.
    ///
    /// The scalar code adds the subframes one after the other into the same
    /// array elements. Here the four products of one element are computed
    /// together and their sum is added once. All sums wrap around, so the
    /// result is the same.
    ///
    /// `(a * b) >> 16` with a 16-bit `b` (`silk_smulwb`) is computed as the
    /// upper half of `2 * a * (b << 15)`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon")]
    pub(super) fn add_subframes(
        c_first_row: &mut [i32; MAX_LPC_ORDER],
        c_last_row: &mut [i32; MAX_LPC_ORDER],
        ca_f: &mut [i32; MAX_LPC_ORDER + 1],
        ca_b: &mut [i32; MAX_LPC_ORDER + 1],
        af_qa: &[i32; MAX_LPC_ORDER],
        side_by_side: &[[i16; 4]],
        n: usize,
        rshifts: i32,
    ) {
        let subfr_length = side_by_side.len();
        // Sample n and its mirror image at the end of the subframe.
        let x_n = widen(&side_by_side[n]);
        let x_end = widen(&side_by_side[subfr_length - n - 1]);
        // The n samples before position n (read latest first) and the last
        // n samples. The second loop reads one more on the inner side.
        let before = &side_by_side[..n + 1];
        let after = &side_by_side[subfr_length - n - 1..];

        if rshifts > -2 {
            let x1 = vnegq_s32(vshlq_s32(x_n, vdupq_n_s32(16 - rshifts)));
            let x2 = vnegq_s32(vshlq_s32(x_end, vdupq_n_s32(16 - rshifts)));
            let mut tmp1 = vshlq_s32(x_n, vdupq_n_s32(QA - 16));
            let mut tmp2 = vshlq_s32(x_end, vdupq_n_s32(QA - 16));
            for ((((first, last), &af), earlier), later) in c_first_row[..n]
                .iter_mut()
                .zip(c_last_row[..n].iter_mut())
                .zip(&af_qa[..n])
                .zip(before[..n].iter().rev())
                .zip(&after[1..])
            {
                let earlier = vshlq_n_s32::<15>(widen(earlier));
                let later = vshlq_n_s32::<15>(widen(later));
                *first = first.wrapping_add(vaddvq_s32(vqdmulhq_s32(x1, earlier)));
                *last = last.wrapping_add(vaddvq_s32(vqdmulhq_s32(x2, later)));
                let af = vdupq_n_s32(af);
                tmp1 = vaddq_s32(tmp1, vqdmulhq_s32(af, earlier));
                tmp2 = vaddq_s32(tmp2, vqdmulhq_s32(af, later));
            }
            let tmp1 = vshlq_s32(vnegq_s32(tmp1), vdupq_n_s32(32 - QA - rshifts));
            let tmp2 = vshlq_s32(vnegq_s32(tmp2), vdupq_n_s32(32 - QA - rshifts));
            for (((forward, backward), earlier), later) in ca_f[..n + 1]
                .iter_mut()
                .zip(ca_b[..n + 1].iter_mut())
                .zip(before.iter().rev())
                .zip(after)
            {
                let earlier = vshlq_n_s32::<15>(widen(earlier));
                let later = vshlq_n_s32::<15>(widen(later));
                *forward = forward.wrapping_add(vaddvq_s32(vqdmulhq_s32(tmp1, earlier)));
                *backward = backward.wrapping_add(vaddvq_s32(vqdmulhq_s32(tmp2, later)));
            }
        } else {
            let x1 = vnegq_s32(vshlq_s32(x_n, vdupq_n_s32(-rshifts)));
            let x2 = vnegq_s32(vshlq_s32(x_end, vdupq_n_s32(-rshifts)));
            let mut tmp1 = vshlq_n_s32::<17>(x_n);
            let mut tmp2 = vshlq_n_s32::<17>(x_end);
            for ((((first, last), &af), earlier), later) in c_first_row[..n]
                .iter_mut()
                .zip(c_last_row[..n].iter_mut())
                .zip(&af_qa[..n])
                .zip(before[..n].iter().rev())
                .zip(&after[1..])
            {
                let earlier = widen(earlier);
                let later = widen(later);
                *first = first.wrapping_add(vaddvq_s32(vmulq_s32(x1, earlier)));
                *last = last.wrapping_add(vaddvq_s32(vmulq_s32(x2, later)));
                let atmp1 = silk_rshift_round(af, QA - 17);
                tmp1 = vmlaq_n_s32(tmp1, earlier, atmp1);
                tmp2 = vmlaq_n_s32(tmp2, later, atmp1);
            }
            let tmp1 = vnegq_s32(tmp1);
            let tmp2 = vnegq_s32(tmp2);
            let shift = vdupq_n_s32(-rshifts - 1);
            for (((forward, backward), earlier), later) in ca_f[..n + 1]
                .iter_mut()
                .zip(ca_b[..n + 1].iter_mut())
                .zip(before.iter().rev())
                .zip(after)
            {
                let earlier = vshlq_s32(widen(earlier), shift);
                let later = vshlq_s32(widen(later), shift);
                *forward = forward.wrapping_add(vaddvq_s32(smulww(tmp1, earlier)));
                *backward = backward.wrapping_add(vaddvq_s32(smulww(tmp2, later)));
            }
        }
    }
}

pub fn silk_burg_modified_fix(
    res_nrg: &mut i32,
    res_nrg_q: &mut i32,
    a_q16: &mut [i32],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) {
    const QA: i32 = BURG_QA;
    const N_BITS_HEAD_ROOM: i32 = 3;
    const MIN_RSHIFTS: i32 = -16;
    const MAX_RSHIFTS: i32 = 32 - QA;

    const FIND_LPC_COND_FAC_Q32: i32 = 42950;

    assert!(d <= MAX_LPC_ORDER);

    let mut c_first_row = [0i32; MAX_LPC_ORDER];
    let mut c_last_row = [0i32; MAX_LPC_ORDER];
    let mut af_qa = [0i32; MAX_LPC_ORDER];
    let mut ca_f = [0i32; MAX_LPC_ORDER + 1];
    let mut ca_b = [0i32; MAX_LPC_ORDER + 1];

    let total_len = subfr_length * nb_subfr;
    let c0_64 = inner_prod16_i64(&x[..total_len], &x[..total_len]);

    let lz = silk_clz64(c0_64);
    let rshifts = (32 + 1 + N_BITS_HEAD_ROOM - lz).clamp(MIN_RSHIFTS, MAX_RSHIFTS);

    let c0: i32 = if rshifts > 0 {
        silk_rshift64(c0_64, rshifts) as i32
    } else {
        (c0_64 as i32) << (-rshifts)
    };

    ca_f[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
    ca_b[0] = ca_f[0];

    if rshifts > 0 {
        for s in 0..nb_subfr {
            let subframe = &x[s * subfr_length..(s + 1) * subfr_length];
            for n in 1..=d {
                let sum = inner_prod16_i64(&subframe[..subfr_length - n], &subframe[n..]);
                c_first_row[n - 1] =
                    c_first_row[n - 1].wrapping_add(silk_rshift64(sum, rshifts) as i32);
            }
        }
    } else {
        let mut xcorr = [0i32; MAX_LPC_ORDER];
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            let fast_n = subfr_length - d;

            silk_pitch_xcorr(&x[x_ptr..], &x[x_ptr + 1..], &mut xcorr[..d], fast_n, d);

            // `silk_pitch_xcorr` covered the first `fast_n` products of
            // every lag. Lag `n` has `d - n` more.
            let subframe = &x[x_ptr..x_ptr + subfr_length];
            for n in 1..=d {
                let mut tail: i32 = 0;
                for (&later, &earlier) in subframe[n + fast_n..].iter().zip(&subframe[fast_n..]) {
                    tail = silk_smlabb(tail, later as i32, earlier as i32);
                }
                xcorr[n - 1] = xcorr[n - 1].wrapping_add(tail);
                c_first_row[n - 1] = c_first_row[n - 1].wrapping_add(xcorr[n - 1] << (-rshifts));
            }
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    ca_f[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
    ca_b[0] = ca_f[0];

    let mut inv_gain_q30: i32 = 1 << 30;
    let mut reached_max_gain = false;

    // The vector version reads the subframes side by side: `side_by_side[i]`
    // holds sample `i` of every subframe.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let mut side_by_side = [[0i16; 4]; BURG_MAX_SUBFR_LENGTH];
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let use_neon = nb_subfr <= 4 && subfr_length <= BURG_MAX_SUBFR_LENGTH;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if use_neon {
        // SAFETY: this code is only compiled when NEON is enabled for the
        // whole build (`cfg(target_feature = "neon")`).
        unsafe {
            burg_neon::lay_side_by_side(
                &mut side_by_side[..subfr_length],
                &x[..total_len],
                nb_subfr,
            );
        }
    }

    for n in 0..d {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if use_neon {
            // SAFETY: this code is only compiled when NEON is enabled for
            // the whole build (`cfg(target_feature = "neon")`).
            unsafe {
                burg_neon::add_subframes(
                    &mut c_first_row,
                    &mut c_last_row,
                    &mut ca_f,
                    &mut ca_b,
                    &af_qa,
                    &side_by_side[..subfr_length],
                    n,
                    rshifts,
                );
            }
        } else {
            burg_add_subframes_scalar(
                &mut c_first_row,
                &mut c_last_row,
                &mut ca_f,
                &mut ca_b,
                &af_qa,
                x,
                subfr_length,
                nb_subfr,
                n,
                rshifts,
            );
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        burg_add_subframes_scalar(
            &mut c_first_row,
            &mut c_last_row,
            &mut ca_f,
            &mut ca_b,
            &af_qa,
            x,
            subfr_length,
            nb_subfr,
            n,
            rshifts,
        );

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        let mut num: i32 = 0;
        let mut nrg: i32 = ca_b[0].wrapping_add(ca_f[0]);
        for k in 0..n {
            let atmp_qa = af_qa[k];
            let lz = (silk_clz32(atmp_qa.wrapping_abs()) - 1).min(32 - QA);
            let atmp1 = atmp_qa << lz;
            let shift = 32 - QA - lz;

            tmp1 = tmp1.wrapping_add(
                ((silk_smmul(c_last_row[n - k - 1], atmp1) as u32 as i64) << shift) as i32,
            );
            tmp2 = tmp2.wrapping_add(
                ((silk_smmul(c_first_row[n - k - 1], atmp1) as u32 as i64) << shift) as i32,
            );
            num = num.wrapping_add(((silk_smmul(ca_b[n - k], atmp1) as i64) << shift) as i32);
            nrg = nrg.wrapping_add(
                ((silk_smmul(ca_b[k + 1].wrapping_add(ca_f[k + 1]), atmp1) as i64) << shift) as i32,
            );
        }
        ca_f[n + 1] = tmp1;
        ca_b[n + 1] = tmp2;
        num = num.wrapping_add(tmp2);
        num = num.wrapping_neg().wrapping_shl(1);

        let mut rc_q31: i32;
        if num.wrapping_abs() < nrg {
            rc_q31 = silk_div32_varq(num, nrg, 31);
        } else {
            rc_q31 = if num > 0 { i32::MAX } else { i32::MIN };
        }

        tmp1 = (1 << 30) - silk_smmul(rc_q31, rc_q31);
        tmp1 = silk_smmul(inv_gain_q30, tmp1) << 2;
        if tmp1 <= min_inv_gain_q30 {
            tmp2 = (1 << 30) - silk_div32_varq(min_inv_gain_q30, inv_gain_q30, 30);
            rc_q31 = silk_sqrt_approx(tmp2);
            if rc_q31 > 0 {
                rc_q31 = (rc_q31 + silk_div32(tmp2, rc_q31)) >> 1;
                rc_q31 <<= 16;
                if num < 0 {
                    rc_q31 = -rc_q31;
                }
            }
            inv_gain_q30 = min_inv_gain_q30;
            reached_max_gain = true;
        } else {
            inv_gain_q30 = tmp1;
        }

        for k in 0..(n + 1) >> 1 {
            tmp1 = af_qa[k];
            tmp2 = af_qa[n - k - 1];
            af_qa[k] = tmp1.wrapping_add((silk_smmul(tmp2, rc_q31) as i64 * 2) as i32);
            af_qa[n - k - 1] = tmp2.wrapping_add((silk_smmul(tmp1, rc_q31) as i64 * 2) as i32);
        }
        af_qa[n] = rc_q31 >> (31 - QA);

        if reached_max_gain {
            for k in n + 1..d {
                af_qa[k] = 0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let idx = n + 1 - k;
            tmp1 = ca_f[k];
            tmp2 = ca_b[idx];
            ca_f[k] = tmp1.wrapping_add((silk_smmul(tmp2, rc_q31) as i64 * 2) as i32);
            ca_b[idx] = tmp2.wrapping_add((silk_smmul(tmp1, rc_q31) as i64 * 2) as i32);
        }
    }

    if reached_max_gain {
        for k in 0..d {
            a_q16[k] = -silk_rshift_round(af_qa[k], QA - 16);
        }

        let mut c0_adj = c0;
        if rshifts > 0 {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let mut sum: i64 = 0;
                for i in 0..d {
                    sum += (x[x_ptr + i] as i64) * (x[x_ptr + i] as i64);
                }
                c0_adj = c0_adj.wrapping_sub(silk_rshift64(sum, rshifts) as i32);
            }
        } else {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let mut sum: i32 = 0;
                for i in 0..d {
                    sum = sum.wrapping_add((x[x_ptr + i] as i32).wrapping_mul(x[x_ptr + i] as i32));
                }
                c0_adj = c0_adj.wrapping_sub(sum << (-rshifts));
            }
        }
        *res_nrg = silk_smmul(inv_gain_q30, c0_adj) << 2;
        *res_nrg_q = -rshifts;
    } else {
        let mut nrg = ca_f[0];
        let mut tmp1_q16: i32 = 1 << 16;
        for k in 0..d {
            let atmp1 = silk_rshift_round(af_qa[k], QA - 16);
            nrg = silk_smlaww(nrg, ca_f[k + 1], atmp1);
            tmp1_q16 = silk_smlaww(tmp1_q16, atmp1, atmp1);
            a_q16[k] = -atmp1;
        }
        *res_nrg = silk_smlaww(nrg, silk_smmul(FIND_LPC_COND_FAC_Q32, c0), -tmp1_q16);
        *res_nrg_q = -rshifts;
    }
}

pub fn energy_flp(x: &[f32]) -> f64 {
    let mut sum = 0.0;
    for &val in x {
        sum += val as f64 * val as f64;
    }
    sum
}

pub fn inner_product_flp(x1: &[f32], x2: &[f32]) -> f64 {
    let len = x1.len().min(x2.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += x1[i] as f64 * x2[i] as f64;
    }
    sum
}

pub fn silk_lpc_analysis_filter_flp(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: usize,
    order: usize,
) {
    assert!(order <= MAX_LPC_ORDER);

    for ix in order..length {
        let mut lpc_pred = 0.0f32;
        for j in 0..order {
            lpc_pred += s[ix - 1 - j] * pred_coef[j];
        }
        r_lpc[ix] = s[ix] - lpc_pred;
    }
}

pub fn silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    let mut c_first_row = [0.0f64; MAX_LPC_ORDER];
    let mut c_last_row = [0.0f64; MAX_LPC_ORDER];
    let mut caf = [0.0f64; MAX_LPC_ORDER + 1];
    let mut cab = [0.0f64; MAX_LPC_ORDER + 1];
    let mut af = [0.0f64; MAX_LPC_ORDER];

    let c0 = energy_flp(&x[..nb_subfr * subfr_length]);

    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..d + 1 {
            c_first_row[n - 1] +=
                inner_product_flp(&x_ptr[..subfr_length - n], &x_ptr[n..subfr_length]);
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    caf[0] = c0 + (FIND_LPC_COND_FAC as f64) * c0 + 1e-9;
    cab[0] = caf[0];

    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;

    for n in 0..d {
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            let mut tmp1 = x_ptr[n] as f64;
            let mut tmp2 = x_ptr[subfr_length - n - 1] as f64;
            for k in 0..n {
                let atmp = af[k];
                tmp1 += x_ptr[n - k - 1] as f64 * atmp;
                tmp2 += x_ptr[subfr_length - n + k] as f64 * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * x_ptr[n - k] as f64;
                cab[k] -= tmp2 * x_ptr[subfr_length - n + k - 1] as f64;
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }

        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }

        let mut rc = -2.0 * num / (nrg_f + nrg_b);

        let tmp1_rc = inv_gain * (1.0 - rc * rc);
        if tmp1_rc <= min_inv_gain as f64 {
            rc = (1.0 - min_inv_gain as f64 / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1_rc;
        }

        for k in 0..n.div_ceil(2) {
            let tmp1 = af[k];
            let tmp2 = af[n - k];
            af[k] = tmp1 + rc * tmp2;
            af[n - k] = tmp2 + rc * tmp1;
        }
        af[n] = rc;

        if reached_max_gain {
            for k in n + 1..d {
                af[k] = 0.0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let tmp1 = caf[k];
            caf[k] += rc * cab[n - k + 1];
            cab[n - k + 1] += rc * tmp1;
        }
    }

    let mut final_nrg_f: f64;
    if reached_max_gain {
        for k in 0..d {
            a[k] = (-af[k]) as f32;
        }
        let mut c0_mod = c0;
        for s in 0..nb_subfr {
            c0_mod -= energy_flp(&x[s * subfr_length..s * subfr_length + d]);
        }
        final_nrg_f = c0_mod * inv_gain;
    } else {
        final_nrg_f = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            final_nrg_f += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        final_nrg_f -= (FIND_LPC_COND_FAC as f64) * c0 * tmp1;
    }

    final_nrg_f as f32
}

const QA_INV: i32 = 24;
/// Largest reflection coefficient accepted as stable: 0.99975 in Q24
/// (libopus LPC_inv_pred_gain.c, `A_LIMIT = SILK_FIX_CONST(0.99975, QA)`,
/// which is `(int)(0.99975 * 2^24 + 0.5)`).
const A_LIMIT: i32 = 16773022;

fn lpc_inverse_pred_gain_qa(a_qa_in: &mut [i32], order: usize) -> i32 {
    let mut inv_gain_q30 = 1 << 30;
    for k in (1..order).rev() {
        if a_qa_in[k] > A_LIMIT || a_qa_in[k] < -A_LIMIT {
            return 0;
        }

        let rc_q31 = -(a_qa_in[k] << (31 - QA_INV));
        let rc_mult1_q30 = (1 << 30) - silk_smmul(rc_q31, rc_q31);

        inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if inv_gain_q30 < (1 << 30) / 10000 {
            return 0;
        }

        let mult2q = 32 - silk_clz32(rc_mult1_q30.abs());
        let rc_mult2 = silk_inverse32_varq(rc_mult1_q30, mult2q + 30);

        for n in 0..k.div_ceil(2) {
            let tmp1 = a_qa_in[n];
            let tmp2 = a_qa_in[k - n - 1];

            let mul_q = |a: i32, b: i32, q: i32| -> i32 {
                (silk_rshift_round64(silk_smull(a, b), q)) as i32
            };

            let tmp64_1 = silk_rshift_round64(
                silk_smull(tmp1.wrapping_sub(mul_q(tmp2, rc_q31, 31)), rc_mult2),
                mult2q,
            );
            if tmp64_1 > i32::MAX as i64 || tmp64_1 < i32::MIN as i64 {
                return 0;
            }
            a_qa_in[n] = tmp64_1 as i32;

            let tmp64_2 = silk_rshift_round64(
                silk_smull(tmp2.wrapping_sub(mul_q(tmp1, rc_q31, 31)), rc_mult2),
                mult2q,
            );
            if tmp64_2 > i32::MAX as i64 || tmp64_2 < i32::MIN as i64 {
                return 0;
            }
            a_qa_in[k - n - 1] = tmp64_2 as i32;
        }
    }

    if a_qa_in[0] > A_LIMIT || a_qa_in[0] < -A_LIMIT {
        return 0;
    }

    let rc_q31 = -(a_qa_in[0] << (31 - QA_INV));
    let rc_mult1_q30 = (1 << 30) - silk_smmul(rc_q31, rc_q31);
    inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;

    if inv_gain_q30 < (1 << 30) / 10000 {
        return 0;
    }

    inv_gain_q30
}

pub fn silk_lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    let mut dc_resp = 0i32;
    let mut atmp_qa = [0i32; MAX_LPC_ORDER];
    for k in 0..order {
        dc_resp += a_q12[k] as i32;
        atmp_qa[k] = (a_q12[k] as i32) << (QA_INV - 12);
    }
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut atmp_qa, order)
}

#[cfg(test)]
mod burg_tests {
    use super::*;

    /// `silk_burg_modified_fix` as it was before the signal-reading part
    /// got a vector version: a direct port of libopus
    /// `silk_burg_modified_c`. The tests require its results.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn original(
        res_nrg: &mut i32,
        res_nrg_q: &mut i32,
        a_q16: &mut [i32],
        x: &[i16],
        min_inv_gain_q30: i32,
        subfr_length: usize,
        nb_subfr: usize,
        d: usize,
    ) {
        const QA: i32 = 25;
        const N_BITS_HEAD_ROOM: i32 = 3;
        const MIN_RSHIFTS: i32 = -16;
        const MAX_RSHIFTS: i32 = 32 - QA;

        const FIND_LPC_COND_FAC_Q32: i32 = 42950;

        assert!(d <= MAX_LPC_ORDER);

        let mut c_first_row = [0i32; MAX_LPC_ORDER];
        let mut c_last_row = [0i32; MAX_LPC_ORDER];
        let mut af_qa = [0i32; MAX_LPC_ORDER];
        let mut ca_f = [0i32; MAX_LPC_ORDER + 1];
        let mut ca_b = [0i32; MAX_LPC_ORDER + 1];

        let total_len = subfr_length * nb_subfr;
        let mut c0_64: i64 = 0;
        for i in 0..total_len {
            c0_64 += (x[i] as i64) * (x[i] as i64);
        }

        let lz = silk_clz64(c0_64);
        let rshifts = (32 + 1 + N_BITS_HEAD_ROOM - lz).clamp(MIN_RSHIFTS, MAX_RSHIFTS);

        let c0: i32 = if rshifts > 0 {
            silk_rshift64(c0_64, rshifts) as i32
        } else {
            (c0_64 as i32) << (-rshifts)
        };

        ca_f[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
        ca_b[0] = ca_f[0];

        if rshifts > 0 {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                for n in 1..=d {
                    let mut sum: i64 = 0;
                    let lim = subfr_length - n;
                    for i in 0..lim {
                        sum += (x[x_ptr + i] as i64) * (x[x_ptr + i + n] as i64);
                    }
                    c_first_row[n - 1] =
                        c_first_row[n - 1].wrapping_add(silk_rshift64(sum, rshifts) as i32);
                }
            }
        } else {
            let mut xcorr = [0i32; MAX_LPC_ORDER];
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let fast_n = subfr_length - d;

                silk_pitch_xcorr(&x[x_ptr..], &x[x_ptr + 1..], &mut xcorr[..d], fast_n, d);

                for n in 1..=d {
                    let mut tail: i32 = 0;
                    for i in (n + fast_n)..subfr_length {
                        tail = silk_smlabb(tail, x[x_ptr + i] as i32, x[x_ptr + i - n] as i32);
                    }
                    xcorr[n - 1] = xcorr[n - 1].wrapping_add(tail);
                    c_first_row[n - 1] =
                        c_first_row[n - 1].wrapping_add(xcorr[n - 1] << (-rshifts));
                }
            }
        }
        c_last_row[..d].copy_from_slice(&c_first_row[..d]);

        ca_f[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
        ca_b[0] = ca_f[0];

        let mut inv_gain_q30: i32 = 1 << 30;
        let mut reached_max_gain = false;

        for n in 0..d {
            if rshifts > -2 {
                for s in 0..nb_subfr {
                    let x_ptr = s * subfr_length;
                    let x1 = (x[x_ptr + n] as i32)
                        .wrapping_shl((16 - rshifts) as u32)
                        .wrapping_neg();
                    let x2 = (x[x_ptr + subfr_length - n - 1] as i32)
                        .wrapping_shl((16 - rshifts) as u32)
                        .wrapping_neg();
                    let mut tmp1 = (x[x_ptr + n] as i32).wrapping_shl((QA - 16) as u32);
                    let mut tmp2 =
                        (x[x_ptr + subfr_length - n - 1] as i32).wrapping_shl((QA - 16) as u32);
                    for k in 0..n {
                        c_first_row[k] =
                            silk_smlawb(c_first_row[k], x1, x[x_ptr + n - k - 1] as i32);
                        c_last_row[k] =
                            silk_smlawb(c_last_row[k], x2, x[x_ptr + subfr_length - n + k] as i32);
                        let atmp_qa = af_qa[k];
                        tmp1 = silk_smlawb(tmp1, atmp_qa, x[x_ptr + n - k - 1] as i32);
                        tmp2 = silk_smlawb(tmp2, atmp_qa, x[x_ptr + subfr_length - n + k] as i32);
                    }
                    tmp1 = tmp1.wrapping_neg().wrapping_shl((32 - QA - rshifts) as u32);
                    tmp2 = tmp2.wrapping_neg().wrapping_shl((32 - QA - rshifts) as u32);
                    for k in 0..=n {
                        ca_f[k] = silk_smlawb(ca_f[k], tmp1, x[x_ptr + n - k] as i32);
                        ca_b[k] =
                            silk_smlawb(ca_b[k], tmp2, x[x_ptr + subfr_length - n + k - 1] as i32);
                    }
                }
            } else {
                for s in 0..nb_subfr {
                    let x_ptr = s * subfr_length;
                    let x1 = (x[x_ptr + n] as i32)
                        .wrapping_shl((-rshifts) as u32)
                        .wrapping_neg();
                    let x2 = (x[x_ptr + subfr_length - n - 1] as i32)
                        .wrapping_shl((-rshifts) as u32)
                        .wrapping_neg();
                    let mut tmp1 = (x[x_ptr + n] as i32).wrapping_shl(17);
                    let mut tmp2 = (x[x_ptr + subfr_length - n - 1] as i32).wrapping_shl(17);
                    for k in 0..n {
                        c_first_row[k] = silk_mla(c_first_row[k], x1, x[x_ptr + n - k - 1] as i32);
                        c_last_row[k] =
                            silk_mla(c_last_row[k], x2, x[x_ptr + subfr_length - n + k] as i32);
                        let atmp1 = silk_rshift_round(af_qa[k], QA - 17);

                        tmp1 = (tmp1 as u32).wrapping_add(
                            (x[x_ptr + n - k - 1] as i32 as u32).wrapping_mul(atmp1 as u32),
                        ) as i32;
                        tmp2 = (tmp2 as u32).wrapping_add(
                            (x[x_ptr + subfr_length - n + k] as i32 as u32)
                                .wrapping_mul(atmp1 as u32),
                        ) as i32;
                    }
                    tmp1 = tmp1.wrapping_neg();
                    tmp2 = tmp2.wrapping_neg();
                    for k in 0..=n {
                        ca_f[k] = silk_smlaww(
                            ca_f[k],
                            tmp1,
                            (x[x_ptr + n - k] as i32).wrapping_shl((-rshifts - 1) as u32),
                        );
                        ca_b[k] = silk_smlaww(
                            ca_b[k],
                            tmp2,
                            (x[x_ptr + subfr_length - n + k - 1] as i32)
                                .wrapping_shl((-rshifts - 1) as u32),
                        );
                    }
                }
            }

            let mut tmp1 = c_first_row[n];
            let mut tmp2 = c_last_row[n];
            let mut num: i32 = 0;
            let mut nrg: i32 = ca_b[0].wrapping_add(ca_f[0]);
            for k in 0..n {
                let atmp_qa = af_qa[k];
                let lz = (silk_clz32(atmp_qa.wrapping_abs()) - 1).min(32 - QA);
                let atmp1 = atmp_qa << lz;
                let shift = 32 - QA - lz;

                tmp1 = tmp1.wrapping_add(
                    ((silk_smmul(c_last_row[n - k - 1], atmp1) as u32 as i64) << shift) as i32,
                );
                tmp2 = tmp2.wrapping_add(
                    ((silk_smmul(c_first_row[n - k - 1], atmp1) as u32 as i64) << shift) as i32,
                );
                num = num.wrapping_add(((silk_smmul(ca_b[n - k], atmp1) as i64) << shift) as i32);
                nrg = nrg.wrapping_add(
                    ((silk_smmul(ca_b[k + 1].wrapping_add(ca_f[k + 1]), atmp1) as i64) << shift)
                        as i32,
                );
            }
            ca_f[n + 1] = tmp1;
            ca_b[n + 1] = tmp2;
            num = num.wrapping_add(tmp2);
            num = num.wrapping_neg().wrapping_shl(1);

            let mut rc_q31: i32;
            if num.wrapping_abs() < nrg {
                rc_q31 = silk_div32_varq(num, nrg, 31);
            } else {
                rc_q31 = if num > 0 { i32::MAX } else { i32::MIN };
            }

            tmp1 = (1 << 30) - silk_smmul(rc_q31, rc_q31);
            tmp1 = silk_smmul(inv_gain_q30, tmp1) << 2;
            if tmp1 <= min_inv_gain_q30 {
                tmp2 = (1 << 30) - silk_div32_varq(min_inv_gain_q30, inv_gain_q30, 30);
                rc_q31 = silk_sqrt_approx(tmp2);
                if rc_q31 > 0 {
                    rc_q31 = (rc_q31 + silk_div32(tmp2, rc_q31)) >> 1;
                    rc_q31 <<= 16;
                    if num < 0 {
                        rc_q31 = -rc_q31;
                    }
                }
                inv_gain_q30 = min_inv_gain_q30;
                reached_max_gain = true;
            } else {
                inv_gain_q30 = tmp1;
            }

            for k in 0..(n + 1) >> 1 {
                tmp1 = af_qa[k];
                tmp2 = af_qa[n - k - 1];
                af_qa[k] = tmp1.wrapping_add((silk_smmul(tmp2, rc_q31) as i64 * 2) as i32);
                af_qa[n - k - 1] = tmp2.wrapping_add((silk_smmul(tmp1, rc_q31) as i64 * 2) as i32);
            }
            af_qa[n] = rc_q31 >> (31 - QA);

            if reached_max_gain {
                for k in n + 1..d {
                    af_qa[k] = 0;
                }
                break;
            }

            for k in 0..=n + 1 {
                let idx = n + 1 - k;
                tmp1 = ca_f[k];
                tmp2 = ca_b[idx];
                ca_f[k] = tmp1.wrapping_add((silk_smmul(tmp2, rc_q31) as i64 * 2) as i32);
                ca_b[idx] = tmp2.wrapping_add((silk_smmul(tmp1, rc_q31) as i64 * 2) as i32);
            }
        }

        if reached_max_gain {
            for k in 0..d {
                a_q16[k] = -silk_rshift_round(af_qa[k], QA - 16);
            }

            let mut c0_adj = c0;
            if rshifts > 0 {
                for s in 0..nb_subfr {
                    let x_ptr = s * subfr_length;
                    let mut sum: i64 = 0;
                    for i in 0..d {
                        sum += (x[x_ptr + i] as i64) * (x[x_ptr + i] as i64);
                    }
                    c0_adj = c0_adj.wrapping_sub(silk_rshift64(sum, rshifts) as i32);
                }
            } else {
                for s in 0..nb_subfr {
                    let x_ptr = s * subfr_length;
                    let mut sum: i32 = 0;
                    for i in 0..d {
                        sum = sum
                            .wrapping_add((x[x_ptr + i] as i32).wrapping_mul(x[x_ptr + i] as i32));
                    }
                    c0_adj = c0_adj.wrapping_sub(sum << (-rshifts));
                }
            }
            *res_nrg = silk_smmul(inv_gain_q30, c0_adj) << 2;
            *res_nrg_q = -rshifts;
        } else {
            let mut nrg = ca_f[0];
            let mut tmp1_q16: i32 = 1 << 16;
            for k in 0..d {
                let atmp1 = silk_rshift_round(af_qa[k], QA - 16);
                nrg = silk_smlaww(nrg, ca_f[k + 1], atmp1);
                tmp1_q16 = silk_smlaww(tmp1_q16, atmp1, atmp1);
                a_q16[k] = -atmp1;
            }
            *res_nrg = silk_smlaww(nrg, silk_smmul(FIND_LPC_COND_FAC_Q32, c0), -tmp1_q16);
            *res_nrg_q = -rshifts;
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

    /// A signal with some correlation between neighbours, with a peak level
    /// of about `2^level_bits`. The level decides which scaling the function
    /// picks (`rshifts` from -16 to 7).
    fn signal(rng: &mut XorShift, len: usize, level_bits: u32) -> Vec<i16> {
        let peak = (1i32 << level_bits) - 1;
        let mut state = 0i32;
        (0..len)
            .map(|_| {
                state = state * 3 / 4 + rng.range(-peak, peak) / 2;
                state.clamp(-32768, 32767) as i16
            })
            .collect()
    }

    /// Run the original and the current function and require the same
    /// coefficients and the same residual energy.
    fn check(
        x: &[i16],
        min_inv_gain_q30: i32,
        subfr_length: usize,
        nb_subfr: usize,
        d: usize,
        what: &str,
    ) {
        let mut expected_a = [0x5a5a_5a5ai32; MAX_LPC_ORDER];
        let (mut expected_nrg, mut expected_q) = (0i32, 0i32);
        original(
            &mut expected_nrg,
            &mut expected_q,
            &mut expected_a,
            x,
            min_inv_gain_q30,
            subfr_length,
            nb_subfr,
            d,
        );
        let mut a = [0x5a5a_5a5ai32; MAX_LPC_ORDER];
        let (mut nrg, mut q) = (0i32, 0i32);
        silk_burg_modified_fix(
            &mut nrg,
            &mut q,
            &mut a,
            x,
            min_inv_gain_q30,
            subfr_length,
            nb_subfr,
            d,
        );
        assert_eq!(
            (a, nrg, q),
            (expected_a, expected_nrg, expected_q),
            "{what}: subframe {subfr_length}, {nb_subfr} subframes, order {d}, \
             minimum inverse gain {min_inv_gain_q30}"
        );
    }

    #[test]
    fn encoder_frames_match_original() {
        // The calls of the encoder: 4 or 2 subframes of 5 ms plus the LPC
        // order, at 8, 12 and 16 kHz, at every signal level.
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        for level_bits in 1..=15 {
            for (fs_khz, d) in [(8usize, 10usize), (12, 10), (16, 16)] {
                for nb_subfr in [2usize, 4] {
                    let subfr_length = 5 * fs_khz + d;
                    for _ in 0..20 {
                        let x = signal(&mut rng, subfr_length * nb_subfr, level_bits);
                        // 1e-4 and 1e-2 in Q30, the two limits of the encoder.
                        for min_inv_gain_q30 in [107_374, 10_737_418] {
                            check(
                                &x,
                                min_inv_gain_q30,
                                subfr_length,
                                nb_subfr,
                                d,
                                "encoder frame",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_shape_matches_original() {
        // Every order, 1 to 4 subframes, subframe lengths around the sizes
        // that matter to the vector code: the order plus one, multiples of
        // 8 and their neighbours, the longest subframe the vector code takes
        // (96) and longer ones, which use the scalar code.
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        for d in 1..=MAX_LPC_ORDER {
            for nb_subfr in 1..=4usize {
                for subfr_length in [d + 1, d + 2, 23, 24, 25, 31, 32, 33, 50, 95, 96, 97, 130] {
                    if subfr_length <= d {
                        continue;
                    }
                    for level_bits in [2u32, 6, 9, 12, 15] {
                        let x = signal(&mut rng, subfr_length * nb_subfr + 3, level_bits);
                        check(&x, 107_374, subfr_length, nb_subfr, d, "every shape");
                    }
                }
            }
        }
    }

    // For some of these signals the function, old and new alike, negates
    // the most negative 32-bit number. A release build wraps around; a debug
    // build stops there, so the test only runs in a release build.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "an overflow check of the debug build stops it"
    )]
    fn special_signals_match_original() {
        let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
        let (subfr_length, nb_subfr, d) = (96usize, 4usize, 16usize);
        let len = subfr_length * nb_subfr;
        // Silence, full scale, the two extremes alternating, a pure tone
        // (prediction gain at its limit), and one pulse.
        let mut signals: Vec<Vec<i16>> = vec![
            vec![0; len],
            vec![i16::MAX; len],
            vec![i16::MIN; len],
            (0..len)
                .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
                .collect(),
            (0..len)
                .map(|i| [0i16, 23170, 32767, 23170, 0, -23170, -32767, -23170][i % 8])
                .collect(),
            (0..len).map(|i| if i == 200 { 12345 } else { 0 }).collect(),
        ];
        // Extreme values at random positions.
        for _ in 0..200 {
            signals.push(
                (0..len)
                    .map(|_| match rng.range(0, 3) {
                        0 => i16::MIN,
                        1 => i16::MAX,
                        _ => rng.range(-32768, 32767) as i16,
                    })
                    .collect(),
            );
        }
        for x in &signals {
            // The two limits of the encoder, and values far from them.
            for min_inv_gain_q30 in [1, 107_374, 10_737_418, 1 << 29, 1 << 30] {
                check(
                    x,
                    min_inv_gain_q30,
                    subfr_length,
                    nb_subfr,
                    d,
                    "special signal",
                );
                check(x, min_inv_gain_q30, 50, 2, 10, "special signal");
            }
        }
    }

    /// The vector version of the signal-reading step against the scalar one,
    /// for every step `n` and every scaling, from arbitrary array contents.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn neon_step_matches_scalar_step() {
        let mut rng = XorShift(0x0F1E_2D3C_4B5A_6978);
        for round in 0..3_000 {
            let nb_subfr = 1 + round % 4;
            let subfr_length = [17usize, 20, 24, 31, 50, 64, 95, 96][round % 8];
            let x: Vec<i16> = (0..subfr_length * nb_subfr)
                .map(|_| match rng.range(0, 7) {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => rng.range(-32768, 32767) as i16,
                })
                .collect();
            let mut side_by_side = [[0x5a5ai16; 4]; BURG_MAX_SUBFR_LENGTH];
            // SAFETY: NEON is enabled for the whole build (see the `cfg`).
            unsafe {
                burg_neon::lay_side_by_side(&mut side_by_side[..subfr_length], &x, nb_subfr);
            }
            for (i, row) in side_by_side[..subfr_length].iter().enumerate() {
                for (s, &sample) in row.iter().enumerate() {
                    let expected = if s < nb_subfr {
                        x[s * subfr_length + i]
                    } else {
                        0
                    };
                    assert_eq!(sample, expected, "position {i}, subframe {s}");
                }
            }

            for n in 0..MAX_LPC_ORDER {
                let rshifts = rng.range(-16, 7);
                let mut any = || rng.next() as i32;
                let af_qa: [i32; MAX_LPC_ORDER] = core::array::from_fn(|_| any());
                let mut scalar = (
                    core::array::from_fn::<i32, MAX_LPC_ORDER, _>(|_| any()),
                    core::array::from_fn::<i32, MAX_LPC_ORDER, _>(|_| any()),
                    core::array::from_fn::<i32, { MAX_LPC_ORDER + 1 }, _>(|_| any()),
                    core::array::from_fn::<i32, { MAX_LPC_ORDER + 1 }, _>(|_| any()),
                );
                let mut neon = scalar;
                burg_add_subframes_scalar(
                    &mut scalar.0,
                    &mut scalar.1,
                    &mut scalar.2,
                    &mut scalar.3,
                    &af_qa,
                    &x,
                    subfr_length,
                    nb_subfr,
                    n,
                    rshifts,
                );
                // SAFETY: NEON is enabled for the whole build.
                unsafe {
                    burg_neon::add_subframes(
                        &mut neon.0,
                        &mut neon.1,
                        &mut neon.2,
                        &mut neon.3,
                        &af_qa,
                        &side_by_side[..subfr_length],
                        n,
                        rshifts,
                    );
                }
                assert_eq!(
                    neon, scalar,
                    "step {n}, rshifts {rshifts}, subframe {subfr_length}, {nb_subfr} subframes"
                );
            }
        }
    }
}
