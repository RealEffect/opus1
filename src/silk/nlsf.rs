use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::nlsf_encode::silk_nlsf_encode;
use crate::silk::structs::*;
use crate::silk::tables::*;
use crate::silk::tables_nlsf::*;

use crate::silk::lpc_analysis::*;
use crate::silk::lpc_fit::*;

pub fn silk_bwexpander_32(ar: &mut [i32], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    for i in 0..d - 1 {
        ar[i] = silk_smulww(chirp_q16, ar[i]);
        chirp_q16 += silk_rshift_round(
            ((chirp_q16 as i64) * (chirp_minus_one_q16 as i64)) as i32,
            16,
        );
    }
    ar[d - 1] = silk_smulww(chirp_q16, ar[d - 1]);
}

fn silk_a2nlsf_trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= p[k] << 1;
    }
}

/// Number of table points for which the value of a polynomial is computed
/// in one pass.
const A2NLSF_BLOCK: usize = 8;

/// Length of the cosine table, rounded up to whole blocks.
const A2NLSF_TABLE_LEN: usize = (LSF_COS_TAB_SZ_FIX + 1).next_multiple_of(A2NLSF_BLOCK);

/// `SILK_LSF_COS_TAB_FIX_Q12` in the format the polynomial evaluation
/// multiplies with (Q16). The entries after the table are zero.
const A2NLSF_COS_Q16: [i32; A2NLSF_TABLE_LEN] = {
    let mut table = [0i32; A2NLSF_TABLE_LEN];
    let mut k = 0;
    while k <= LSF_COS_TAB_SZ_FIX {
        table[k] = (SILK_LSF_COS_TAB_FIX_Q12[k] as i32) << 4;
        k += 1;
    }
    table
};

/// Value of the polynomial `p` of degree `dd` at each of the points `x_q16`
/// (Horner's rule, as libopus `silk_A2NLSF_eval_poly`).
///
/// The points do not depend on each other, so the processor works on all
/// of them at the same time.
#[inline(always)]
fn silk_a2nlsf_eval_poly_block<const N: usize>(
    p: &[i32; MAX_LPC_ORDER / 2 + 1],
    x_q16: &[i32; N],
    dd: usize,
) -> [i32; N] {
    let mut y32 = [p[dd]; N];
    for &coefficient in p[..dd].iter().rev() {
        for (y, &x) in y32.iter_mut().zip(x_q16) {
            *y = silk_smlaww(coefficient, *y, x);
        }
    }
    y32
}

/// Value of the polynomial `p` at every point of the cosine table. Portable
/// version: the code for targets without a vector version, and the reference
/// in the tests. On aarch64 only the tests call it.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
fn silk_a2nlsf_eval_table_scalar(
    y: &mut [i32; A2NLSF_TABLE_LEN],
    p: &[i32; MAX_LPC_ORDER / 2 + 1],
    dd: usize,
) {
    let (y_blocks, _) = y.as_chunks_mut::<A2NLSF_BLOCK>();
    let (x_blocks, _) = A2NLSF_COS_Q16.as_chunks::<A2NLSF_BLOCK>();
    for (y_block, x_block) in y_blocks.iter_mut().zip(x_blocks) {
        *y_block = silk_a2nlsf_eval_poly_block(p, x_block, dd);
    }
}

/// Value of the polynomial `p` at every point of the cosine table.
#[inline]
fn silk_a2nlsf_eval_table(
    y: &mut [i32; A2NLSF_TABLE_LEN],
    p: &[i32; MAX_LPC_ORDER / 2 + 1],
    dd: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: this code is only compiled when NEON is enabled for the
        // whole build (`cfg(target_feature = "neon")`).
        unsafe { a2nlsf_neon::eval_table(y, p, dd) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    silk_a2nlsf_eval_table_scalar(y, p, dd);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod a2nlsf_neon {
    use super::{A2NLSF_BLOCK, A2NLSF_COS_Q16, A2NLSF_TABLE_LEN};
    use crate::silk::define::MAX_LPC_ORDER;
    use core::arch::aarch64::*;

    /// One step of Horner's rule in four lanes: `c + ((y * x) >> 16)`,
    /// computed with 64-bit products and cut to 32 bits, as `silk_smlaww`
    /// does it. `c_wide` is `c` shifted left by 16 bits, so the shift after
    /// the addition gives the same bits as the shift before it.
    #[inline]
    #[target_feature(enable = "neon")]
    fn step(c_wide: int64x2_t, y: int32x4_t, x: int32x4_t) -> int32x4_t {
        let low = vmlal_s32(c_wide, vget_low_s32(y), vget_low_s32(x));
        let high = vmlal_high_s32(c_wide, y, x);
        vshrn_high_n_s64::<16>(vshrn_n_s64::<16>(low), high)
    }

    /// Same result as `silk_a2nlsf_eval_table_scalar`, eight table points
    /// per pass in two vectors.
    #[target_feature(enable = "neon")]
    pub(super) fn eval_table(
        y: &mut [i32; A2NLSF_TABLE_LEN],
        p: &[i32; MAX_LPC_ORDER / 2 + 1],
        dd: usize,
    ) {
        const { assert!(A2NLSF_BLOCK == 8) };
        // The coefficients in the format of the 64-bit products.
        let mut p_wide = [0i64; MAX_LPC_ORDER / 2 + 1];
        for (wide, &coefficient) in p_wide.iter_mut().zip(p) {
            *wide = (coefficient as i64) << 16;
        }
        let top = vdupq_n_s32(p[dd]);

        let (y_blocks, _) = y.as_chunks_mut::<A2NLSF_BLOCK>();
        let (x_blocks, _) = A2NLSF_COS_Q16.as_chunks::<A2NLSF_BLOCK>();
        for (y_block, x_block) in y_blocks.iter_mut().zip(x_blocks) {
            // SAFETY: `x_block` is a `[i32; 8]`. The two loads read its
            // elements 0 to 3 and 4 to 7.
            let (x0, x1) = unsafe {
                (
                    vld1q_s32(x_block.as_ptr()),
                    vld1q_s32(x_block.as_ptr().add(4)),
                )
            };
            let mut y0 = top;
            let mut y1 = top;
            for &coefficient in p_wide[..dd].iter().rev() {
                let c_wide = vdupq_n_s64(coefficient);
                y0 = step(c_wide, y0, x0);
                y1 = step(c_wide, y1, x1);
            }
            // SAFETY: `y_block` is a `[i32; 8]`. The two stores write its
            // elements 0 to 3 and 4 to 7.
            unsafe {
                vst1q_s32(y_block.as_mut_ptr(), y0);
                vst1q_s32(y_block.as_mut_ptr().add(4), y1);
            }
        }
    }
}

fn silk_a2nlsf_init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    p[dd] = 1 << 16;
    q[dd] = 1 << 16;
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k];
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k];
    }

    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }

    silk_a2nlsf_trans_poly(p, dd);
    silk_a2nlsf_trans_poly(q, dd);
}

/// Middle of two points of the cosine table, as the root search halves an
/// interval.
#[inline(always)]
fn silk_a2nlsf_mid(x0: i32, x1: i32) -> i32 {
    (x0 + x1 + 1) >> 1
}

/// LPC coefficients to normalised line spectral frequencies (libopus
/// `silk_A2NLSF`).
///
/// The roots are found by walking along the cosine table until one of the
/// two polynomials changes its sign, and by halving that interval three
/// times. libopus evaluates the polynomial one point after the other.
/// Here the values at all table points are computed first, and the three
/// halving steps use the values at all seven points they can reach. Every
/// value is computed by the same operations as in libopus, so the result is
/// the same.
pub fn silk_a2nlsf(nlsf: &mut [i16], a_q16: &mut [i32], d: usize) {
    let mut polys = [[0i32; MAX_LPC_ORDER / 2 + 1]; 2];
    // Values of the two polynomials at every point of the cosine table.
    let mut y_table = [[0i32; A2NLSF_TABLE_LEN]; 2];
    let dd = d >> 1;

    const BIN_DIV_STEPS_A2NLSF_FIX: i32 = 3;
    const MAX_ITERATIONS_A2NLSF_FIX: i32 = 16;

    let mut i = 0;
    'search: loop {
        let [p, q] = &mut polys;
        silk_a2nlsf_init(a_q16, p, q, dd);
        silk_a2nlsf_eval_table(&mut y_table[0], p, dd);
        silk_a2nlsf_eval_table(&mut y_table[1], q, dd);

        let mut xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
        let mut ylo = y_table[0][0];

        let mut root_ix: usize;
        let mut poly_idx: usize;
        if ylo < 0 {
            nlsf[0] = 0;
            poly_idx = 1;
            ylo = y_table[1][0];
            root_ix = 1;
        } else {
            poly_idx = 0;
            root_ix = 0;
        }

        let mut k = 1;
        let mut thr = 0;

        loop {
            let xhi = SILK_LSF_COS_TAB_FIX_Q12[k] as i32;
            let yhi = y_table[poly_idx][k];

            if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
                if yhi == 0 {
                    thr = 1;
                } else {
                    thr = 0;
                }

                // The points the three halving steps can reach: the middle
                // of the interval, the middles of its two halves and the
                // middles of its four quarters.
                let x1 = silk_a2nlsf_mid(xlo, xhi);
                let x2 = [silk_a2nlsf_mid(xlo, x1), silk_a2nlsf_mid(x1, xhi)];
                let xmids = [
                    x1,
                    x2[0],
                    x2[1],
                    silk_a2nlsf_mid(xlo, x2[0]),
                    silk_a2nlsf_mid(x2[0], x1),
                    silk_a2nlsf_mid(x1, x2[1]),
                    silk_a2nlsf_mid(x2[1], xhi),
                ];
                let xmids_q16 = xmids.map(|x| x << 4);
                let ymids = silk_a2nlsf_eval_poly_block(&polys[poly_idx], &xmids_q16, dd);
                // Position of the first point of each step in `xmids`.
                const STEP_START: [usize; BIN_DIV_STEPS_A2NLSF_FIX as usize] = [0, 1, 3];

                let mut ffrac = -256;
                let mut ylo_div = ylo;
                let mut yhi_div = yhi;
                // Which of the points of the current step lies in the middle
                // of the current interval.
                let mut branch = 0;

                for m in 0..BIN_DIV_STEPS_A2NLSF_FIX {
                    let ymid = ymids[STEP_START[m as usize] + branch];

                    // The root is in the first half when the sign changes
                    // between the lower end and the middle. Which half it is
                    // cannot be predicted, so the values are selected
                    // without a jump.
                    let first_half =
                        ((ylo_div <= 0) & (ymid >= 0)) | ((ylo_div >= 0) & (ymid <= 0));
                    let second_half = !first_half as i32;
                    yhi_div = core::hint::select_unpredictable(first_half, ymid, yhi_div);
                    ylo_div = core::hint::select_unpredictable(first_half, ylo_div, ymid);
                    ffrac = silk_add_rshift(ffrac, 128 * second_half, m);
                    branch = 2 * branch + second_half as usize;
                }

                let den: i32;
                let nom: i32;
                if ylo_div.abs() < 65536 {
                    den = ylo_div - yhi_div;
                    nom = (ylo_div << (8 - BIN_DIV_STEPS_A2NLSF_FIX)) + (den >> 1);
                    if den != 0 {
                        ffrac += nom / den;
                    }
                } else {
                    ffrac += ylo_div / ((ylo_div - yhi_div) >> (8 - BIN_DIV_STEPS_A2NLSF_FIX));
                }

                nlsf[root_ix] = ((k as i32) << 8).wrapping_add(ffrac).min(i16::MAX as i32) as i16;

                root_ix += 1;
                if root_ix >= d {
                    break 'search;
                }

                poly_idx = root_ix & 1;
                xlo = SILK_LSF_COS_TAB_FIX_Q12[k - 1] as i32;
                ylo = (1 - (root_ix as i32 & 2)) << 12;
            } else {
                k += 1;
                xlo = xhi;
                ylo = yhi;
                thr = 0;

                if k > LSF_COS_TAB_SZ_FIX {
                    i += 1;
                    if i > MAX_ITERATIONS_A2NLSF_FIX {
                        let val = (1i32 << 15) / (d as i32 + 1);
                        nlsf[0] = val as i16;
                        for j in 1..d {
                            nlsf[j] = nlsf[j - 1].wrapping_add(val as i16);
                        }
                        return;
                    }

                    // Not all roots were found. Widen the bandwidth of the
                    // filter and search again.
                    silk_bwexpander_32(a_q16, d, 65536 - (1 << i));
                    continue 'search;
                }
            }
        }
    }
}

pub fn silk_nlsf_stabilize(nlsf_q15: &mut [i16], n_delta_min_q15: &[i16], l: usize) {
    const MAX_LOOPS: i32 = 20;

    let mut loops = 0;
    while loops < MAX_LOOPS {
        let mut min_diff_q15 = nlsf_q15[0] as i32 - n_delta_min_q15[0] as i32;
        let mut i_idx = 0;

        for i in 1..l {
            let diff_q15 =
                nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + n_delta_min_q15[i] as i32);
            if diff_q15 < min_diff_q15 {
                min_diff_q15 = diff_q15;
                i_idx = i;
            }
        }

        let diff_q15 = (1 << 15) - (nlsf_q15[l - 1] as i32 + n_delta_min_q15[l] as i32);
        if diff_q15 < min_diff_q15 {
            min_diff_q15 = diff_q15;
            i_idx = l;
        }

        if min_diff_q15 >= 0 {
            return;
        }

        if i_idx == 0 {
            nlsf_q15[0] = n_delta_min_q15[0];
        } else if i_idx == l {
            nlsf_q15[l - 1] = ((1i32 << 15) - n_delta_min_q15[l] as i32) as i16;
        } else {
            let mut min_center_q15 = 0;
            for k in 0..i_idx {
                min_center_q15 += n_delta_min_q15[k] as i32;
            }
            min_center_q15 += (n_delta_min_q15[i_idx] as i32) >> 1;

            let mut max_center_q15 = 1 << 15;
            for k in (i_idx + 1..=l).rev() {
                max_center_q15 -= n_delta_min_q15[k] as i32;
            }
            max_center_q15 -= (n_delta_min_q15[i_idx] as i32) >> 1;

            let center_freq_q15 = silk_limit_32(
                silk_rshift_round(nlsf_q15[i_idx - 1] as i32 + nlsf_q15[i_idx] as i32, 1),
                min_center_q15,
                max_center_q15,
            ) as i16;
            nlsf_q15[i_idx - 1] = center_freq_q15 - ((n_delta_min_q15[i_idx] as i32) >> 1) as i16;
            nlsf_q15[i_idx] = nlsf_q15[i_idx - 1] + n_delta_min_q15[i_idx];
        }
        loops += 1;
    }

    if loops == MAX_LOOPS {
        nlsf_q15[..l].sort_unstable();

        nlsf_q15[0] = nlsf_q15[0].max(n_delta_min_q15[0]);

        for i in 1..l {
            nlsf_q15[i] = nlsf_q15[i].max(nlsf_q15[i - 1].saturating_add(n_delta_min_q15[i]));
        }

        nlsf_q15[l - 1] = nlsf_q15[l - 1].min(((1i32 << 15) - n_delta_min_q15[l] as i32) as i16);

        for i in (0..l - 1).rev() {
            nlsf_q15[i] = nlsf_q15[i].min(nlsf_q15[i + 1].wrapping_sub(n_delta_min_q15[i + 1]));
        }
    }
}

/// Weighted error (Q24) of one codebook vector for `silk_nlsf_vq`, for an
/// even LPC order of at most 16.
///
/// libopus walks from the last coefficient to the first and carries the
/// weighted difference of the coefficient before as `pred`:
/// `sum += |diffw - (pred >> 1)|; pred = diffw`. So the term of coefficient
/// `m` needs the weighted difference of coefficient `m + 1`, and nothing
/// else. This function first computes all weighted differences and then the
/// sum of the terms. Both loops have a fixed number of independent steps,
/// which the compiler turns into vector code. The sum wraps around, as in
/// libopus, so the order of the additions does not matter.
///
/// Not inlined on purpose: inside the loop over the codebook vectors the
/// compiler makes vectors across four codebook vectors instead, and has to
/// collect every lane with a load of its own.
#[inline(never)]
fn nlsf_vq_error_q24<const ORDER: usize>(
    in_q15: &[i16; ORDER],
    cb_q8: &[u8; ORDER],
    wght_q9: &[i16; ORDER],
) -> i32 {
    // One more entry than coefficients: the one after the last is 0.
    let mut diffw_q24 = [0i32; MAX_LPC_ORDER + 1];
    for m in 0..ORDER {
        let diff_q15 = (in_q15[m] as i32) - ((cb_q8[m] as i32) << 7);
        diffw_q24[m] = silk_smulbb(diff_q15, wght_q9[m] as i32);
    }
    let mut sum_error_q24 = 0i32;
    for m in 0..ORDER {
        sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24[m] - (diffw_q24[m + 1] >> 1)).abs());
    }
    sum_error_q24
}

/// First stage of the NLSF quantiser (libopus `silk_NLSF_VQ`): the weighted
/// error of each of the `k` codebook vectors of `lpc_order` coefficients.
pub fn silk_nlsf_vq(
    err_q24: &mut [i32],
    in_q15: &[i16],
    pcb_q8: &[u8],
    pwght_q9: &[i16],
    k: usize,
    lpc_order: usize,
) {
    if k == 0 {
        return;
    }
    // The two orders of the codec get loops of fixed length.
    match lpc_order {
        16 => nlsf_vq_fixed_order::<16>(err_q24, in_q15, pcb_q8, pwght_q9, k),
        10 => nlsf_vq_fixed_order::<10>(err_q24, in_q15, pcb_q8, pwght_q9, k),
        _ => nlsf_vq_any_order(err_q24, in_q15, pcb_q8, pwght_q9, k, lpc_order),
    }
}

fn nlsf_vq_fixed_order<const ORDER: usize>(
    err_q24: &mut [i32],
    in_q15: &[i16],
    pcb_q8: &[u8],
    pwght_q9: &[i16],
    k: usize,
) {
    let in_q15: &[i16; ORDER] = in_q15.first_chunk().expect("one value per coefficient");
    let (cb_q8, _) = pcb_q8.as_chunks::<ORDER>();
    let (wght_q9, _) = pwght_q9.as_chunks::<ORDER>();
    for ((err, cb), wght) in err_q24[..k].iter_mut().zip(&cb_q8[..k]).zip(&wght_q9[..k]) {
        *err = nlsf_vq_error_q24(in_q15, cb, wght);
    }
}

/// `silk_nlsf_vq` as libopus writes it, for any even order. The reference
/// for `nlsf_vq_fixed_order` in the tests.
fn nlsf_vq_any_order(
    err_q24: &mut [i32],
    in_q15: &[i16],
    pcb_q8: &[u8],
    pwght_q9: &[i16],
    k: usize,
    lpc_order: usize,
) {
    for i in 0..k {
        let mut sum_error_q24 = 0i32;
        let mut pred_q24 = 0i32;
        let cb_ptr = &pcb_q8[i * lpc_order..];
        let w_ptr = &pwght_q9[i * lpc_order..];

        for m in (0..lpc_order).step_by(2).rev() {
            let diff_q15 = (in_q15[m + 1] as i32) - ((cb_ptr[m + 1] as i32) << 7);
            let diffw_q24 = silk_smulbb(diff_q15, w_ptr[m + 1] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24;

            let diff_q15_m = (in_q15[m] as i32) - ((cb_ptr[m] as i32) << 7);
            let diffw_q24_m = silk_smulbb(diff_q15_m, w_ptr[m] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24_m - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24_m;
        }
        err_q24[i] = sum_error_q24;
    }
}

fn silk_nlsf2a_find_poly(out: &mut [i32], clsf: &[i32], dd: usize) {
    out[0] = 1 << 16;
    out[1] = -clsf[0];
    for k in 1..dd {
        let ftmp = clsf[2 * k];
        out[k + 1] = (out[k - 1] << 1) - (silk_rshift_round64(silk_smull(ftmp, out[k]), 16) as i32);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - (silk_rshift_round64(silk_smull(ftmp, out[n - 1]), 16) as i32);
        }
        out[1] -= ftmp;
    }
}

pub fn silk_nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    const ORDERING16: [u8; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [u8; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

    let ordering = if d == 16 {
        &ORDERING16[..]
    } else {
        &ORDERING10[..]
    };
    let mut cos_lsf_qa = [0i32; MAX_LPC_ORDER];
    for k in 0..d {
        let f_int = (nlsf[k] >> (15 - 7)) as usize;
        let f_frac = nlsf[k] - ((f_int as i16) << (15 - 7));
        let cos_val = SILK_LSF_COS_TAB_FIX_Q12[f_int] as i32;
        let delta = SILK_LSF_COS_TAB_FIX_Q12[f_int + 1] as i32 - cos_val;
        cos_lsf_qa[ordering[k] as usize] =
            silk_rshift_round((cos_val << 8) + (delta * f_frac as i32), 20 - 16);
    }

    let dd = d >> 1;
    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];

    silk_nlsf2a_find_poly(&mut p, &cos_lsf_qa[0..], dd);
    silk_nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd);

    let mut a32_qa1 = [0i32; MAX_LPC_ORDER];
    for k in 0..dd {
        let ptmp = p[k + 1].wrapping_add(p[k]);
        let qtmp = q[k + 1].wrapping_sub(q[k]);
        a32_qa1[k] = -qtmp.wrapping_add(ptmp);
        a32_qa1[d - k - 1] = qtmp.wrapping_sub(ptmp);
    }

    silk_lpc_fit(a_q12, &mut a32_qa1, 12, 16 + 1, d);

    for i in 0..MAX_LPC_STABILIZE_ITERATIONS {
        if silk_lpc_inverse_pred_gain(a_q12, d) != 0 {
            break;
        }
        silk_bwexpander_32(&mut a32_qa1, d, 65536 - (2 << i));
        for k in 0..d {
            a_q12[k] = silk_rshift_round(a32_qa1[k], 16 + 1 - 12) as i16;
        }
    }
}

pub fn silk_nlsf_vq_weights_laroia(p_w_q5: &mut [i16], p_nlsf_q15: &[i16], d: usize) {
    const NUMER: i32 = 1 << 17;

    let mut tmp1_int = (p_nlsf_q15[0] as i32).max(1);
    tmp1_int = silk_div32_16(NUMER, tmp1_int);
    let mut tmp2_int = ((p_nlsf_q15[1] - p_nlsf_q15[0]) as i32).max(1);
    tmp2_int = silk_div32_16(NUMER, tmp2_int);
    p_w_q5[0] = silk_limit(tmp1_int + tmp2_int, 0, i16::MAX as i32) as i16;

    let mut k = 1;
    while k < d - 1 {
        tmp1_int = ((p_nlsf_q15[k + 1] - p_nlsf_q15[k]) as i32).max(1);
        tmp1_int = silk_div32_16(NUMER, tmp1_int);
        p_w_q5[k] = silk_limit(tmp2_int + tmp1_int, 0, i16::MAX as i32) as i16;

        tmp2_int = ((p_nlsf_q15[k + 2] - p_nlsf_q15[k + 1]) as i32).max(1);
        tmp2_int = silk_div32_16(NUMER, tmp2_int);
        p_w_q5[k + 1] = silk_limit(tmp1_int + tmp2_int, 0, i16::MAX as i32) as i16;

        k += 2;
    }

    tmp1_int = ((1 << 15) - p_nlsf_q15[d - 1] as i32).max(1);
    tmp1_int = silk_div32_16(NUMER, tmp1_int);
    p_w_q5[d - 1] = silk_limit(tmp2_int + tmp1_int, 0, i16::MAX as i32) as i16;
}

pub fn silk_process_nlsfs(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    nlsf_q15: &mut [i16],
) {
    let mut nlsf_interp_q15 = [0i16; MAX_LPC_ORDER];
    let mut p_cb = &SILK_NLSF_CB_WB;
    if ps_enc.s_cmn.fs_khz == 8 || ps_enc.s_cmn.fs_khz == 12 {
        p_cb = &SILK_NLSF_CB_NB_MB;
    }

    let order = ps_enc.s_cmn.predict_lpc_order as usize;

    let mut nlsf_mu_q20 = silk_smlawb(3146, -268434, ps_enc.s_cmn.speech_activity_q8);

    if ps_enc.s_cmn.nb_subfr == 2 {
        nlsf_mu_q20 = nlsf_mu_q20 + silk_rshift(nlsf_mu_q20, 1);
    }
    let interp_coef_q2 = ps_enc.s_cmn.indices.nlsf_interp_coef_q2 as i32;
    let use_interp = ps_enc.s_cmn.use_interpolated_nlsfs != 0;

    // The weights come from the vector as the analysis gave it. libopus
    // enforces the minimum distances only inside `silk_NLSF_encode`.
    let mut w_q5 = [0i16; MAX_LPC_ORDER];
    silk_nlsf_vq_weights_laroia(&mut w_q5, nlsf_q15, order);

    let do_interp: i32 = (use_interp && interp_coef_q2 < 4) as i32;
    if do_interp != 0 {
        let prev_nlsf_q15 = ps_enc.s_cmn.prev_nlsf_q15;
        for i in 0..order {
            nlsf_interp_q15[i] = prev_nlsf_q15[i]
                + silk_rshift(
                    silk_mul((nlsf_q15[i] - prev_nlsf_q15[i]) as i32, interp_coef_q2),
                    2,
                ) as i16;
        }

        let mut w0_q5 = [0i16; MAX_LPC_ORDER];
        silk_nlsf_vq_weights_laroia(&mut w0_q5, &nlsf_interp_q15, order);

        let i_sqr_q15 = silk_lshift(silk_smulbb(interp_coef_q2, interp_coef_q2), 11);
        for i in 0..order {
            w_q5[i] = silk_rshift(w_q5[i] as i32, 1) as i16
                + silk_rshift(silk_smulbb(w0_q5[i] as i32, i_sqr_q15), 16) as i16;
        }
    }

    silk_nlsf_encode(
        &mut ps_enc.s_cmn.indices.nlsf_indices,
        nlsf_q15,
        p_cb,
        &w_q5,
        nlsf_mu_q20,
        ps_enc.s_cmn.n_nlsf_survivors as usize,
        ps_enc.s_cmn.indices.signal_type as i32,
    );

    silk_nlsf2a(&mut ps_enc_ctrl.pred_coef_q12[1], nlsf_q15, order);

    if do_interp != 0 {
        let prev_nlsf_q15 = ps_enc.s_cmn.prev_nlsf_q15;
        for i in 0..order {
            nlsf_interp_q15[i] = prev_nlsf_q15[i]
                + silk_rshift(
                    silk_mul((nlsf_q15[i] - prev_nlsf_q15[i]) as i32, interp_coef_q2),
                    2,
                ) as i16;
        }

        silk_nlsf2a(&mut ps_enc_ctrl.pred_coef_q12[0], &nlsf_interp_q15, order);
    } else {
        let (first, second) = ps_enc_ctrl.pred_coef_q12.split_at_mut(1);
        first[0][..order].copy_from_slice(&second[0][..order]);
    }

    ps_enc.s_cmn.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);
}

#[cfg(test)]
mod a2nlsf_tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;

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
    }

    /// libopus `silk_A2NLSF_eval_poly`: one point, one step after the other.
    fn eval_poly_reference(p: &[i32], x: i32, dd: usize) -> i32 {
        let mut y32 = p[dd];
        let x_q16 = x << 4;
        for n in (0..dd).rev() {
            y32 = silk_smlaww(p[n], y32, x_q16);
        }
        y32
    }

    /// Polynomials with coefficients of every magnitude, and the largest
    /// and smallest 32-bit values.
    fn polynomials() -> Vec<[i32; MAX_LPC_ORDER / 2 + 1]> {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut list = Vec::new();
        for round in 0..3_000 {
            let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
            for coefficient in p.iter_mut() {
                *coefficient = match round % 4 {
                    // Any magnitude: a random number of leading bits is
                    // cleared.
                    0 => (rng.next() as i32) >> (rng.next() % 32),
                    // The size of real polynomials (the top one is 65536).
                    1 => (rng.next() % (1 << 21)) as i32 - (1 << 20),
                    2 => rng.next() as i32,
                    _ => [i32::MIN, i32::MAX, -1, 0, 1, 65536][(rng.next() % 6) as usize],
                };
            }
            list.push(p);
        }
        list
    }

    #[test]
    fn table_matches_point_by_point_evaluation() {
        for p in polynomials() {
            for dd in 0..=MAX_LPC_ORDER / 2 {
                let mut table = [0x5a5a_5a5ai32; A2NLSF_TABLE_LEN];
                silk_a2nlsf_eval_table(&mut table, &p, dd);
                let mut portable = [0x5a5a_5a5ai32; A2NLSF_TABLE_LEN];
                silk_a2nlsf_eval_table_scalar(&mut portable, &p, dd);
                assert_eq!(table, portable, "degree {dd}, {p:?}");
                for k in 0..=LSF_COS_TAB_SZ_FIX {
                    let expected = eval_poly_reference(&p, SILK_LSF_COS_TAB_FIX_Q12[k] as i32, dd);
                    assert_eq!(table[k], expected, "degree {dd}, point {k}, {p:?}");
                }
            }
        }
    }

    #[test]
    fn block_matches_point_by_point_evaluation() {
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        for p in polynomials() {
            for dd in [0, 1, 5, 8] {
                // Any point between the ends of the cosine table.
                let x: [i32; 7] = core::array::from_fn(|_| (rng.next() % 16385) as i32 - 8192);
                let y = silk_a2nlsf_eval_poly_block(&p, &x.map(|x| x << 4), dd);
                for (lane, &point) in x.iter().enumerate() {
                    assert_eq!(
                        y[lane],
                        eval_poly_reference(&p, point, dd),
                        "degree {dd}, point {point}, {p:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod laroia_tests {
    use super::*;

    /// The weights that the fixed-point build of libopus 1.6.1 computed for
    /// the side channel of the first frame of a stereo SILK stream (16 kHz,
    /// 32 kb/s, VBR, complexity 9; `silk_NLSF_VQ_weights_laroia`, read in
    /// `silk_NLSF_encode`). The last weight divides by `(1 << 15) - NLSF[15]`
    /// = 2149, which gives 60; dividing by 32767 - NLSF[15] gives 61. That
    /// one step was enough to make opus1 choose other NLSF indices than
    /// libopus after 274 packets of 20 ms.
    #[test]
    fn weights_are_those_of_the_fixed_point_build() {
        let nlsf: [i16; 16] = [
            606, 1902, 3565, 5387, 7331, 9351, 11415, 13509, 15619, 17746, 19880, 22019, 24166, 26314,
            28465, 30619,
        ];
        let want: [i16; 16] = [317, 179, 149, 138, 131, 127, 125, 124, 123, 122, 122, 122, 122, 121, 120, 120];
        let mut w = [0i16; 16];
        silk_nlsf_vq_weights_laroia(&mut w, &nlsf, 16);
        assert_eq!(w, want);
    }

    /// The last weight is `2^17 / (2^15 - NLSF[D-1])` plus the share of
    /// the distance to the coefficient before, limited to 32767. Checked for
    /// every last value from 27001 to 32767.
    #[test]
    fn last_weight_measures_the_distance_to_one() {
        let mut nlsf = [0i16; 10];
        for (i, v) in nlsf.iter_mut().enumerate() {
            *v = 3000 * (i as i16 + 1);
        }
        let mut w = [0i16; 10];
        for last in 27001..=32767i32 {
            nlsf[9] = last as i16;
            silk_nlsf_vq_weights_laroia(&mut w, &nlsf, 10);
            let want = ((1 << 17) / (32768 - last) + (1 << 17) / (last - 27000)).min(32767);
            assert_eq!(w[9] as i32, want, "last NLSF {last}");
        }
    }
}
