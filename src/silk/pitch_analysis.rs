use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;
use crate::silk::resampler::*;
use crate::silk::sigproc_fix::*;
use crate::silk::sort::silk_insertion_sort_decreasing_int16;
use crate::silk::structs::*;
use crate::silk::tables::*;

const SCRATCH_SIZE: usize = 22;
const SF_LENGTH_4KHZ: usize = PE_SUBFR_LENGTH_MS * 4;
const SF_LENGTH_8KHZ: usize = PE_SUBFR_LENGTH_MS * 8;
const MIN_LAG_4KHZ: usize = PE_MIN_LAG_MS * 4;
const MIN_LAG_8KHZ: usize = PE_MIN_LAG_MS * 8;
const MAX_LAG_4KHZ: usize = PE_MAX_LAG_MS * 4;
const MAX_LAG_8KHZ: usize = PE_MAX_LAG_MS * 8 - 1;
const CSTRIDE_4KHZ: usize = MAX_LAG_4KHZ + 1 - MIN_LAG_4KHZ;
const CSTRIDE_8KHZ: usize = MAX_LAG_8KHZ + 3 - (MIN_LAG_8KHZ - 2);
const D_COMP_MIN: isize = (MIN_LAG_8KHZ as isize) - 3;
const D_COMP_MAX: isize = (MAX_LAG_8KHZ as isize) + 4;
const D_COMP_STRIDE: usize = (D_COMP_MAX - D_COMP_MIN) as usize;

/// `out[i] = silk_div32_varq(a[i], b[i], qres)` for every `i`: the
/// approximate quotient `a / b` in the fixed-point format `qres`.
///
/// The three slices have the same length. No element of `b` may be zero
/// (`silk_div32_varq` divides by it). `qres` is at most 24.
///
/// On aarch64 four quotients are computed at a time; the result is the same
/// (see `div_neon::div32_varq_x4`).
pub(crate) fn silk_div32_varq_batch(out: &mut [i32], a: &[i32], b: &[i32], qres: i32) {
    debug_assert!(out.len() == a.len() && a.len() == b.len());
    debug_assert!((0..=24).contains(&qres));

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let done = {
        let (out4, _) = out.as_chunks_mut::<4>();
        let (a4, _) = a.as_chunks::<4>();
        let (b4, _) = b.as_chunks::<4>();
        for ((out, a), b) in out4.iter_mut().zip(a4).zip(b4) {
            if b.contains(&0) {
                // Leave the division by zero to the scalar function.
                for i in 0..4 {
                    out[i] = silk_div32_varq(a[i], b[i], qres);
                }
            } else {
                // SAFETY: this code is only compiled when NEON is enabled
                // for the whole build (`cfg(target_feature = "neon")`).
                *out = unsafe { div_neon::div32_varq_x4(a, b, qres) };
            }
        }
        4 * out4.len()
    };
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    let done = 0;

    for ((out, &a), &b) in out[done..].iter_mut().zip(&a[done..]).zip(&b[done..]) {
        *out = silk_div32_varq(a, b, qres);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod div_neon {
    use core::arch::aarch64::*;

    /// 2^29 - 1 divided by each of two 16-bit values, rounded towards zero,
    /// as an integer division does it.
    ///
    /// The quotient is computed in double precision. It is exact: the
    /// quotient of two integers below 2^30 that is not a whole number is at
    /// least 2^-15 away from the next one, far more than the rounding error
    /// of the division (below 2^-37 here).
    #[inline]
    #[target_feature(enable = "neon")]
    fn reciprocal(divisor: int32x2_t) -> int32x2_t {
        let numerator = vdupq_n_f64((i32::MAX >> 2) as f64);
        let quotient = vdivq_f64(numerator, vcvtq_f64_s64(vmovl_s32(divisor)));
        vmovn_s64(vcvtq_s64_f64(quotient))
    }

    /// `silk_div32_varq(a[i], b[i], qres)` in the four lanes of a vector.
    /// No `b[i]` is zero and `qres` is at most 24.
    ///
    /// The steps are those of the scalar function, in the same order:
    /// normalise both numbers, take the reciprocal of the upper 16 bits of
    /// the divisor, multiply, correct once with the remainder, shift into
    /// the result format. `silk_smulwb` is `vqdmulhq_s32` with a
    /// `smulwb_operand` (the reciprocal shifted left by 15); `(x * y) >> 32`
    /// is half of the upper half of `2 * x * y`.
    #[target_feature(enable = "neon")]
    pub(super) fn div32_varq_x4(a: &[i32; 4], b: &[i32; 4], qres: i32) -> [i32; 4] {
        // SAFETY: each pointer comes from a `[i32; 4]`, which is what
        // `vld1q_s32` reads and `vst1q_s32` writes.
        let (a32, b32) = unsafe { (vld1q_s32(a.as_ptr()), vld1q_s32(b.as_ptr())) };

        // Shift both numbers so that the highest bit below the sign is
        // used. The count is -1 for the most negative number: a shift count
        // below zero shifts right.
        let one = vdupq_n_s32(1);
        let a_headrm = vsubq_s32(vclzq_s32(vabsq_s32(a32)), one);
        let b_headrm = vsubq_s32(vclzq_s32(vabsq_s32(b32)), one);
        let a32_nrm = vshlq_s32(a32, a_headrm);
        let b32_nrm = vshlq_s32(b32, b_headrm);

        // Reciprocal of the upper 16 bits of the divisor. It fits in 16
        // bits.
        let b_high = vshrq_n_s32::<16>(b32_nrm);
        let b32_inv = vcombine_s32(
            reciprocal(vget_low_s32(b_high)),
            reciprocal(vget_high_s32(b_high)),
        );
        let b32_inv = vshlq_n_s32::<15>(b32_inv);

        // First estimate, then one correction with the remainder.
        let result = vqdmulhq_s32(a32_nrm, b32_inv);
        let product = vshrq_n_s32::<1>(vqdmulhq_s32(b32_nrm, result));
        let a32_nrm = vsubq_s32(a32_nrm, vshlq_n_s32::<3>(product));
        let result = vaddq_s32(result, vqdmulhq_s32(a32_nrm, b32_inv));

        // Into the result format: a left shift that saturates, or a right
        // shift, or zero when the right shift is 32 bits or more.
        //
        // The scalar function shifts left with `silk_lshift_sat32` (libopus
        // `silk_LSHIFT_SAT32`), which limits the value first and shifts
        // afterwards. The bits below the shift count are therefore zero in
        // every result, also in the largest one. The saturating shift of
        // NEON gives `i32::MAX` there, so those bits are cleared here.
        let lshift = vsubq_s32(vaddq_s32(vdupq_n_s32(29 - qres), a_headrm), b_headrm);
        let left = vnegq_s32(lshift);
        let shifted = vqshlq_s32(result, left);
        let keep = vshlq_s32(vdupq_n_s32(-1), vmaxq_s32(left, vdupq_n_s32(0)));
        let shifted = vandq_s32(shifted, keep);
        let shifted = vbicq_s32(
            shifted,
            vreinterpretq_s32_u32(vcgtq_s32(lshift, vdupq_n_s32(31))),
        );

        let mut out = [0i32; 4];
        // SAFETY: see above.
        unsafe { vst1q_s32(out.as_mut_ptr(), shifted) };
        out
    }
}

/// Threshold (Q13) that the correlation of the second search stage must
/// exceed for a frame to count as voiced (libopus `silk_find_pitch_lags_FIX`):
/// 0.6, lowered for a higher prediction order, for more speech activity, after
/// a voiced frame and for more spectral tilt.
///
/// The constants are libopus's `SILK_FIX_CONST(-0.004, 13)`,
/// `SILK_FIX_CONST(-0.1, 21)`, `SILK_FIX_CONST(-0.15, 13)` and
/// `SILK_FIX_CONST(-0.1, 14)`. That macro adds 0.5 and cuts off the fraction,
/// so a negative constant is rounded towards zero: -32.768 becomes -32.
#[inline]
fn pitch_threshold_q13(
    pitch_lpc_order: i32,
    speech_activity_q8: i32,
    prev_signal_type: i32,
    input_tilt_q15: i32,
) -> i32 {
    let mut thrhld_q13: i32 = 4915;
    thrhld_q13 = silk_smlabb(thrhld_q13, -32, pitch_lpc_order);
    thrhld_q13 = silk_smlawb(thrhld_q13, -209714, speech_activity_q8);
    thrhld_q13 = silk_smlabb(thrhld_q13, -1228, prev_signal_type >> 1);
    thrhld_q13 = silk_smlawb(thrhld_q13, -1637, input_tilt_q15);
    silk_sat16(thrhld_q13)
}

pub fn silk_find_pitch_lags_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    res: &mut [i16],
    x: &[i16],
    _arch: i32,
) {
    let la_pitch = ps_enc.s_cmn.la_pitch as usize;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let pitch_lpc_win_length = ps_enc.s_cmn.pitch_lpc_win_length as usize;
    let pitch_lpc_order = ps_enc.pitch_estimation_lpc_order as usize;

    let buf_len = la_pitch + frame_length + ltp_mem_length;

    let mut wsig = [0i16; PE_MAX_FRAME_LENGTH];
    let x_ptr_start = buf_len - pitch_lpc_win_length;

    silk_apply_sine_window(&mut wsig[..], &x[x_ptr_start..], 1, la_pitch);

    let mid_len = pitch_lpc_win_length - 2 * la_pitch;
    wsig[la_pitch..la_pitch + mid_len]
        .copy_from_slice(&x[x_ptr_start + la_pitch..x_ptr_start + la_pitch + mid_len]);

    let last_start = pitch_lpc_win_length - la_pitch;
    silk_apply_sine_window(
        &mut wsig[last_start..],
        &x[x_ptr_start + last_start..],
        2,
        la_pitch,
    );

    let mut auto_corr = [0i32; MAX_FIND_PITCH_LPC_ORDER + 1];
    let mut scale = 0i32;
    silk_autocorr(
        &mut auto_corr,
        &mut scale,
        &wsig,
        pitch_lpc_win_length,
        pitch_lpc_order + 1,
    );

    auto_corr[0] = silk_smlawb(auto_corr[0], auto_corr[0], 66) + 1;

    let mut rc_q15 = [0i16; MAX_FIND_PITCH_LPC_ORDER];
    let res_nrg = silk_schur(&mut rc_q15, &auto_corr, pitch_lpc_order);

    ps_enc_ctrl.pred_gain_q16 = silk_div32_varq(auto_corr[0], res_nrg.max(1), 16);

    let mut a_q24 = [0i32; MAX_FIND_PITCH_LPC_ORDER];
    silk_k2a(&mut a_q24, &rc_q15, pitch_lpc_order);

    let mut a_q12 = [0i16; MAX_FIND_PITCH_LPC_ORDER];
    for i in 0..pitch_lpc_order {
        a_q12[i] = silk_sat16(a_q24[i] >> 12) as i16;
    }

    silk_bwexpander(&mut a_q12, pitch_lpc_order, 64881);

    silk_lpc_analysis_filter(res, x, &a_q12, buf_len, pitch_lpc_order, 0);

    if ps_enc.s_cmn.indices.signal_type != TYPE_NO_VOICE_ACTIVITY as i8
        && ps_enc.s_cmn.first_frame_after_reset == 0
    {
        let thrhld_q13 = pitch_threshold_q13(
            pitch_lpc_order as i32,
            ps_enc.s_cmn.speech_activity_q8,
            ps_enc.s_cmn.prev_signal_type,
            ps_enc.s_cmn.input_tilt_q15,
        );

        let mut ltp_corr_q15 = ps_enc.ltp_corr_q15;
        if silk_pitch_analysis_core(
            res,
            &mut ps_enc_ctrl.pitch_l,
            &mut ps_enc.s_cmn.indices.lag_index,
            &mut ps_enc.s_cmn.indices.contour_index,
            &mut ltp_corr_q15,
            ps_enc.s_cmn.prev_lag,
            ps_enc.s_cmn.pitch_estimation_threshold_q16,
            thrhld_q13,
            ps_enc.s_cmn.fs_khz,
            ps_enc.s_cmn.pitch_estimation_complexity as usize,
            ps_enc.s_cmn.nb_subfr as usize,
        ) == 0
        {
            ps_enc.s_cmn.indices.signal_type = TYPE_VOICED as i8;
        } else {
            ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        }
        ps_enc.ltp_corr_q15 = ltp_corr_q15;
    } else {
        ps_enc_ctrl.pitch_l = [0; MAX_NB_SUBFR];
        ps_enc.s_cmn.indices.lag_index = 0;
        ps_enc.s_cmn.indices.contour_index = 0;
        ps_enc.ltp_corr_q15 = 0;
    }
}

pub fn silk_pitch_analysis_core(
    frame_unscaled: &[i16],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltp_corr_q15: &mut i32,
    prev_lag: i32,
    search_thres1_q16: i32,
    search_thres2_q13: i32,
    fs_khz: i32,
    complexity: usize,
    nb_subfr: usize,
) -> i32 {
    let mut filt_state = [0i32; 6];
    let mut frame_8khz_buf =
        [0i16; (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 8];
    let mut frame_4khz = [0i16; (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 4];
    let mut frame_scaled_buf =
        [0i16; (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * PE_MAX_FS_KHZ];

    let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz as usize;
    let frame_length_8khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let frame_length_4khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let sf_length = PE_SUBFR_LENGTH_MS * fs_khz as usize;
    let min_lag = PE_MIN_LAG_MS * fs_khz as usize;
    let max_lag = PE_MAX_LAG_MS * fs_khz as usize - 1;

    let mut energy = 0i32;
    let mut shift = 0i32;
    silk_sum_sqr_shift(&mut energy, &mut shift, frame_unscaled, frame_length);
    shift += 3 - silk_clz32(energy);

    let frame: &[i16] = if shift > 0 {
        let s = (shift + 1) >> 1;
        for i in 0..frame_length {
            frame_scaled_buf[i] = frame_unscaled[i] >> s;
        }
        &frame_scaled_buf[..frame_length]
    } else {
        &frame_unscaled[..frame_length]
    };

    let frame_8khz: &[i16] = if fs_khz == 16 {
        filt_state[0..2].fill(0);
        // The slicing checks that the frame fits the buffer.
        let output = &mut frame_8khz_buf[..frame_length_8khz];
        silk_resampler_down2(&mut filt_state[..2], output, frame, frame_length as i32);
        output
    } else if fs_khz == 12 {
        filt_state[0..6].fill(0);
        let output = &mut frame_8khz_buf[..frame_length_8khz];
        silk_resampler_down2_3(&mut filt_state[..6], output, frame, frame_length as i32);
        output
    } else {
        frame
    };

    filt_state[0..2].fill(0);
    silk_resampler_down2(
        &mut filt_state[..2],
        &mut frame_4khz[..frame_length_4khz],
        frame_8khz,
        frame_length_8khz as i32,
    );

    for i in (1..frame_length_4khz).rev() {
        frame_4khz[i] = silk_add_sat16(frame_4khz[i], frame_4khz[i - 1]);
    }

    let mut c = [0i16; PE_MAX_NB_SUBFR * CSTRIDE_4KHZ];
    let mut xcorr32 = [0i32; MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1];

    let mut target_ptr_idx = SF_LENGTH_4KHZ << 2;
    for k in 0..(nb_subfr >> 1) {
        let basis_ptr_idx = target_ptr_idx - MIN_LAG_4KHZ;
        silk_pitch_xcorr(
            &frame_4khz[target_ptr_idx..],
            &frame_4khz[(target_ptr_idx - MAX_LAG_4KHZ)..],
            &mut xcorr32,
            SF_LENGTH_8KHZ,
            MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1,
        );

        let mut normalizer = silk_inner_prod_aligned(
            &frame_4khz[target_ptr_idx..],
            &frame_4khz[target_ptr_idx..],
            SF_LENGTH_8KHZ,
        );
        normalizer = normalizer.wrapping_add(silk_inner_prod_aligned(
            &frame_4khz[basis_ptr_idx..],
            &frame_4khz[basis_ptr_idx..],
            SF_LENGTH_8KHZ,
        ));
        normalizer = normalizer.wrapping_add(silk_smulbb(SF_LENGTH_8KHZ as i32, 4000));

        // The normaliser of every lag: each one follows from the one of
        // the lag before by adding one squared sample and taking one away.
        // The correlations come out of `silk_pitch_xcorr` with the longest
        // lag first. All quotients are then computed in one batch.
        let mut normalizers = [0i32; CSTRIDE_4KHZ];
        let mut cross_corrs = [0i32; CSTRIDE_4KHZ];
        let mut current_basis_ptr_idx = basis_ptr_idx;
        normalizers[0] = normalizer;
        for slot in normalizers[1..].iter_mut() {
            current_basis_ptr_idx -= 1;
            normalizer = normalizer.wrapping_add(
                silk_smulbb(
                    frame_4khz[current_basis_ptr_idx] as i32,
                    frame_4khz[current_basis_ptr_idx] as i32,
                ) - silk_smulbb(
                    frame_4khz[current_basis_ptr_idx + SF_LENGTH_8KHZ] as i32,
                    frame_4khz[current_basis_ptr_idx + SF_LENGTH_8KHZ] as i32,
                ),
            );
            *slot = normalizer;
        }
        for (cross_corr, &from_xcorr) in cross_corrs.iter_mut().zip(xcorr32.iter().rev()) {
            *cross_corr = from_xcorr;
        }
        let mut quotients = [0i32; CSTRIDE_4KHZ];
        silk_div32_varq_batch(&mut quotients, &cross_corrs, &normalizers, 14);
        for (c, &quotient) in c[k * CSTRIDE_4KHZ..][..CSTRIDE_4KHZ]
            .iter_mut()
            .zip(&quotients)
        {
            *c = quotient as i16;
        }
        target_ptr_idx += SF_LENGTH_8KHZ;
    }

    let mut c_combined = [0i16; CSTRIDE_4KHZ];
    if nb_subfr == PE_MAX_NB_SUBFR {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let mut sum = c[i - MIN_LAG_4KHZ] as i32 + c[CSTRIDE_4KHZ + (i - MIN_LAG_4KHZ)] as i32;
            sum = silk_smlawb(sum, sum, -(i as i32) << 4);
            c_combined[i - MIN_LAG_4KHZ] = sum as i16;
        }
    } else {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let mut sum = (c[i - MIN_LAG_4KHZ] as i32) << 1;
            sum = silk_smlawb(sum, sum, -(i as i32) << 4);
            c_combined[i - MIN_LAG_4KHZ] = sum as i16;
        }
    }

    let mut d_srch = [0i32; PE_D_SRCH_LENGTH];
    let length_d_srch_orig = (complexity << 1) + 4;
    let mut length_d_srch = length_d_srch_orig;
    silk_insertion_sort_decreasing_int16(
        &mut c_combined[..CSTRIDE_4KHZ],
        &mut d_srch[..length_d_srch],
        CSTRIDE_4KHZ,
        length_d_srch,
    );

    if (c_combined[0] as i32) < 3277 {
        pitch_out[0..nb_subfr].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = silk_smulwb(search_thres1_q16, c_combined[0] as i32);
    for i in 0..length_d_srch {
        if (c_combined[i] as i32) > threshold {
            d_srch[i] = (d_srch[i] + MIN_LAG_4KHZ as i32) << 1;
        } else {
            length_d_srch = i;
            break;
        }
    }

    // `d_comp[lag - D_COMP_MIN]` marks lags at 8 kHz. First the lags the
    // first stage found.
    let mut d_comp = [0i16; D_COMP_STRIDE];
    for i in 0..length_d_srch {
        d_comp[(d_srch[i] as isize - D_COMP_MIN) as usize] = 1;
    }

    // Convolution: a mark at a lag also marks the two lags above it.
    for i in (3..D_COMP_STRIDE).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2];
    }

    // The lags the second stage searches: every lag of the first stage with
    // the lag below it and the lag above it. The first stage works at half
    // the rate, so its lags are the even ones only.
    length_d_srch = 0;
    for i in MIN_LAG_8KHZ..MAX_LAG_8KHZ + 1 {
        if d_comp[i + 1 - D_COMP_MIN as usize] > 0 {
            d_srch[length_d_srch] = i as i32;
            length_d_srch += 1;
        }
    }

    // Convolution: the lags whose correlation the second stage needs. Its
    // codebook moves the lag of a subframe by a few samples either way.
    for i in (3..D_COMP_STRIDE).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2] + d_comp[i - 3];
    }

    let mut length_d_comp = 0;
    let mut d_comp_indices = [0i32; D_COMP_STRIDE];
    for i in MIN_LAG_8KHZ..D_COMP_MAX as usize {
        if d_comp[i - D_COMP_MIN as usize] > 0 {
            d_comp_indices[length_d_comp] = (i - 2) as i32;
            length_d_comp += 1;
        }
    }

    let mut c_8khz = [0i16; PE_MAX_NB_SUBFR * CSTRIDE_8KHZ];
    let mut xcorr_comp = [0i32; D_COMP_STRIDE];
    let mut target_ptr_8khz_idx = PE_LTP_MEM_LENGTH_MS * 8;
    for k in 0..nb_subfr {
        let energy_target = silk_inner_prod_aligned(
            &frame_8khz[target_ptr_8khz_idx..],
            &frame_8khz[target_ptr_8khz_idx..],
            SF_LENGTH_8KHZ,
        )
        .wrapping_add(1);

        if length_d_comp > 0 {
            let min_d = d_comp_indices[0] as usize;
            let max_d = d_comp_indices[length_d_comp - 1] as usize;
            let lag_span = max_d - min_d + 1;

            silk_pitch_xcorr(
                &frame_8khz[target_ptr_8khz_idx..],
                &frame_8khz[target_ptr_8khz_idx - max_d..],
                &mut xcorr_comp[..lag_span],
                SF_LENGTH_8KHZ,
                lag_span,
            );

            let mut cur_d = min_d;
            let mut cur_basis_ptr_idx = target_ptr_8khz_idx - cur_d;
            let mut energy_basis = silk_inner_prod_aligned(
                &frame_8khz[cur_basis_ptr_idx..],
                &frame_8khz[cur_basis_ptr_idx..],
                SF_LENGTH_8KHZ,
            );

            // Correlation and energy of every lag first, then all
            // quotients in one batch. A lag whose correlation is not
            // positive gets 0: it is entered as 0 / 1.
            let mut cross_corrs = [0i32; D_COMP_STRIDE];
            let mut energies = [1i32; D_COMP_STRIDE];
            for j in 0..length_d_comp {
                let d = d_comp_indices[j] as usize;
                while cur_d < d {
                    energy_basis = energy_basis.wrapping_sub(silk_smulbb(
                        frame_8khz[cur_basis_ptr_idx + SF_LENGTH_8KHZ - 1] as i32,
                        frame_8khz[cur_basis_ptr_idx + SF_LENGTH_8KHZ - 1] as i32,
                    ));
                    cur_basis_ptr_idx -= 1;
                    energy_basis = energy_basis.wrapping_add(silk_smulbb(
                        frame_8khz[cur_basis_ptr_idx] as i32,
                        frame_8khz[cur_basis_ptr_idx] as i32,
                    ));
                    cur_d += 1;
                }

                let cross_corr = xcorr_comp[max_d - d];
                if cross_corr > 0 {
                    cross_corrs[j] = cross_corr;
                    energies[j] = energy_target.wrapping_add(energy_basis);
                }
            }
            let mut quotients = [0i32; D_COMP_STRIDE];
            silk_div32_varq_batch(
                &mut quotients[..length_d_comp],
                &cross_corrs[..length_d_comp],
                &energies[..length_d_comp],
                14,
            );
            for j in 0..length_d_comp {
                let out_idx = d_comp_indices[j] as usize - (MIN_LAG_8KHZ - 2);
                c_8khz[k * CSTRIDE_8KHZ + out_idx] = quotients[j] as i16;
            }
        }
        target_ptr_8khz_idx += SF_LENGTH_8KHZ;
    }

    let mut cc_max = i32::MIN;
    let mut cc_max_b = i32::MIN;
    let mut cb_i_max = 0usize;
    let mut lag = -1i32;

    let mut prev_lag_8khz = prev_lag;
    if prev_lag_8khz > 0 {
        if fs_khz == 12 {
            prev_lag_8khz = silk_div32_16(prev_lag_8khz << 1, 3);
        } else if fs_khz == 16 {
            prev_lag_8khz >>= 1;
        }
    }
    let prev_lag_log2_q7 = if prev_lag_8khz > 0 {
        silk_lin2log(prev_lag_8khz)
    } else {
        0
    };

    let nb_cbk_search: usize = if nb_subfr == PE_MAX_NB_SUBFR {
        if fs_khz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            PE_NB_CBKS_STAGE2_EXT
        } else {
            PE_NB_CBKS_STAGE2
        }
    } else {
        PE_NB_CBKS_STAGE2_10MS
    };

    for k in 0..length_d_srch {
        let d = d_srch[k];
        let mut cc = [0i32; PE_NB_CBKS_STAGE2_EXT];
        for j in 0..nb_cbk_search {
            cc[j] = 0;
            for i in 0..nb_subfr {
                let lag_cb_ptr_curr: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                    &SILK_CB_LAGS_STAGE2[i]
                } else {
                    &SILK_CB_LAGS_STAGE2_10_MS[i]
                };
                let d_subfr = d + lag_cb_ptr_curr[j] as i32;

                let min_lag = MIN_LAG_8KHZ as i32 - 2;
                let max_lag = MIN_LAG_8KHZ as i32 + CSTRIDE_8KHZ as i32 - 3;
                if d_subfr >= min_lag && d_subfr <= max_lag {
                    let idx = (d_subfr - min_lag) as usize;
                    if i * CSTRIDE_8KHZ + idx < c_8khz.len() {
                        cc[j] += c_8khz[i * CSTRIDE_8KHZ + idx] as i32;
                    }
                }
            }
        }

        let mut cc0_max_new = i32::MIN;
        let mut cb_i_max_new = 0usize;
        for i in 0..nb_cbk_search {
            if cc[i] > cc0_max_new {
                cc0_max_new = cc[i];
                cb_i_max_new = i;
            }
        }

        let lag_log2_q7 = silk_lin2log(d);
        let mut cc_max_new_b = cc0_max_new
            - silk_rshift(
                silk_smulbb(nb_subfr as i32 * PE_SHORTLAG_BIAS_Q13, lag_log2_q7),
                7,
            );

        if prev_lag_8khz > 0 {
            let mut delta_lag_log2_sqr_q7 = lag_log2_q7 - prev_lag_log2_q7;
            delta_lag_log2_sqr_q7 =
                silk_rshift(silk_smulbb(delta_lag_log2_sqr_q7, delta_lag_log2_sqr_q7), 7);
            let mut prev_lag_bias_q13 = silk_rshift(
                silk_smulbb(nb_subfr as i32 * PE_PREVLAG_BIAS_Q13, *ltp_corr_q15),
                15,
            );
            prev_lag_bias_q13 = silk_div32(
                silk_mul(prev_lag_bias_q13, delta_lag_log2_sqr_q7),
                delta_lag_log2_sqr_q7 + 64,
            );
            cc_max_new_b -= prev_lag_bias_q13;
        }

        // The third condition is the one libopus has. It holds for every
        // entry of the table.
        if cc_max_new_b > cc_max_b
            && cc0_max_new > silk_smulbb(nb_subfr as i32, search_thres2_q13)
            && SILK_CB_LAGS_STAGE2[0][cb_i_max_new] as i32 <= MIN_LAG_8KHZ as i32
        {
            cc_max_b = cc_max_new_b;
            cc_max = cc0_max_new;
            lag = d;
            cb_i_max = cb_i_max_new;
        }
    }

    if lag == -1 {
        pitch_out[0..nb_subfr].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    *ltp_corr_q15 = silk_lshift(silk_div32_16(cc_max, nb_subfr as i32), 2);

    if fs_khz > 8 {
        let cb_i_max_old = cb_i_max;
        if fs_khz == 12 {
            lag = silk_rshift(silk_smulbb(lag, 3), 1);
        } else if fs_khz == 16 {
            lag <<= 1;
        } else {
            lag = silk_smulbb(lag, 3);
        }

        lag = silk_limit_32(lag, min_lag as i32, max_lag as i32);
        let start_lag = (lag - 2).max(min_lag as i32);
        let end_lag = (lag + 2).min(max_lag as i32);
        let mut lag_new = lag;
        cb_i_max = 0;

        cc_max = i32::MIN;
        for k in 0..nb_subfr {
            let lag_cb_ptr_curr: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                &SILK_CB_LAGS_STAGE2[k]
            } else {
                &SILK_CB_LAGS_STAGE2_10_MS[k]
            };
            pitch_out[k] = lag + 2 * lag_cb_ptr_curr[cb_i_max_old] as i32;
        }

        let nb_cbk_search_st3: usize = if nb_subfr == PE_MAX_NB_SUBFR {
            SILK_NB_CBK_SEARCHS_STAGE3[complexity]
        } else {
            PE_NB_CBKS_STAGE3_10MS
        };

        // Cross-correlation and energy of every subframe for each lag of
        // its search range, the lowest lag first.
        let mut cross_corr_st3 = [[0i32; SCRATCH_SIZE]; PE_MAX_NB_SUBFR];
        let mut energies_st3 = [[0i32; SCRATCH_SIZE]; PE_MAX_NB_SUBFR];
        silk_p_ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag as usize,
            sf_length,
            nb_subfr,
            complexity,
            fs_khz as usize,
        );
        silk_p_ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag as usize,
            sf_length,
            nb_subfr,
            complexity,
            fs_khz as usize,
        );

        let contour_bias_q15 = silk_div32_16(PE_FLATCONTOUR_BIAS_Q15, (lag as i16).into());

        let energy_target = silk_inner_prod_aligned(
            &frame[PE_LTP_MEM_LENGTH_MS * fs_khz as usize..],
            &frame[PE_LTP_MEM_LENGTH_MS * fs_khz as usize..],
            nb_subfr * sf_length,
        )
        .wrapping_add(1);

        // For every codebook vector and each of the lags from `start_lag`
        // to `end_lag` (at most five): the sums of the cross-correlations
        // and of the energies over the subframes. A codebook vector gives
        // every subframe its own lag offset; its five values are five
        // neighbours in the arrays above. The sums wrap around, as in
        // libopus.
        let nb_lags = (end_lag - start_lag + 1).clamp(0, PE_NB_STAGE3_LAGS as i32) as usize;
        let mut cross_corrs = [[0i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX];
        let mut energies = [[0i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX];
        for j in 0..nb_cbk_search_st3 {
            let mut cross_corr = [0i32; PE_NB_STAGE3_LAGS];
            let mut energy = [energy_target; PE_NB_STAGE3_LAGS];
            for k in 0..nb_subfr {
                let (lag_cb, lag_range) = stage3_tables(nb_subfr, complexity, k);
                let first = (lag_cb[j] - lag_range[0]) as usize;
                let cross_corr_k: &[i32; PE_NB_STAGE3_LAGS] = cross_corr_st3[k][first..]
                    .first_chunk()
                    .expect("five lags inside the search range");
                let energy_k: &[i32; PE_NB_STAGE3_LAGS] = energies_st3[k][first..]
                    .first_chunk()
                    .expect("five lags inside the search range");
                for lag_counter in 0..PE_NB_STAGE3_LAGS {
                    cross_corr[lag_counter] =
                        cross_corr[lag_counter].wrapping_add(cross_corr_k[lag_counter]);
                    energy[lag_counter] = energy[lag_counter].wrapping_add(energy_k[lag_counter]);
                }
            }
            // A vector whose correlation is not positive counts as 0 in the
            // search below. It is entered as 0 / 1 for the division, and so
            // is every lag after `end_lag`.
            for lag_counter in 0..PE_NB_STAGE3_LAGS {
                if cross_corr[lag_counter] <= 0 || lag_counter >= nb_lags {
                    cross_corr[lag_counter] = 0;
                    energy[lag_counter] = 1;
                }
            }
            cross_corrs[j] = cross_corr;
            energies[j] = energy;
        }
        // All quotients in one batch.
        let mut quotients = [[0i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX];
        silk_div32_varq_batch(
            quotients[..nb_cbk_search_st3].as_flattened_mut(),
            cross_corrs[..nb_cbk_search_st3].as_flattened(),
            energies[..nb_cbk_search_st3].as_flattened(),
            14,
        );

        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search_st3 {
                // 0 for a vector whose correlation is not positive. Such a
                // vector can still be chosen: the first comparison is with
                // the smallest number.
                let mut cc_max_new = quotients[j][lag_counter];
                let diff = i16::MAX as i32 - silk_mul(contour_bias_q15, j as i32);
                cc_max_new = silk_smulwb(cc_max_new, diff);
                // libopus reads the table of 20 ms frames here for both
                // frame sizes.
                if cc_max_new > cc_max && (d + SILK_CB_LAGS_STAGE3[0][j] as i32) <= max_lag as i32
                {
                    cc_max = cc_max_new;
                    lag_new = d;
                    cb_i_max = j;
                }
            }
        }

        for k in 0..nb_subfr {
            let lag_cb_ptr_st3_k: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                &SILK_CB_LAGS_STAGE3[k]
            } else {
                &SILK_CB_LAGS_STAGE3_10_MS[k]
            };
            pitch_out[k] = lag_new + lag_cb_ptr_st3_k[cb_i_max] as i32;
            pitch_out[k] = silk_limit(
                pitch_out[k],
                min_lag as i32,
                (PE_MAX_LAG_MS * fs_khz as usize) as i32,
            );
        }
        *lag_index = (lag_new - min_lag as i32) as i16;
        *contour_index = cb_i_max as i8;
    } else {
        for k in 0..nb_subfr {
            let lag_cb_ptr_curr: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                &SILK_CB_LAGS_STAGE2[k]
            } else {
                &SILK_CB_LAGS_STAGE2_10_MS[k]
            };
            pitch_out[k] = lag + lag_cb_ptr_curr[cb_i_max] as i32;
            pitch_out[k] = silk_limit(
                pitch_out[k],
                MIN_LAG_8KHZ as i32,
                (PE_MAX_LAG_MS * 8) as i32,
            );
        }
        *lag_index = (lag - MIN_LAG_8KHZ as i32) as i16;
        *contour_index = cb_i_max as i8;
    }

    0
}

/// For subframe `k` of the last search stage: the lag offset that each
/// codebook vector gives the subframe, and the lowest and the highest lag
/// offset that is searched.
#[inline(always)]
fn stage3_tables(
    nb_subfr: usize,
    complexity: usize,
    k: usize,
) -> (&'static [i8], &'static [i8; 2]) {
    if nb_subfr == PE_MAX_NB_SUBFR {
        (
            &SILK_CB_LAGS_STAGE3[k],
            &SILK_LAG_RANGE_STAGE3[complexity][k],
        )
    } else {
        (
            &SILK_CB_LAGS_STAGE3_10_MS[k],
            &SILK_LAG_RANGE_STAGE3_10_MS[k],
        )
    }
}

/// Cross-correlation of every subframe with the signal one lag earlier, for
/// each lag of the search range of the subframe: `cross_corr_st3[k][i]`
/// belongs to the lag `start_lag` plus the lowest offset of subframe `k`
/// plus `i`.
fn silk_p_ana_calc_corr_st3(
    cross_corr_st3: &mut [[i32; SCRATCH_SIZE]; PE_MAX_NB_SUBFR],
    frame: &[i16],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    fs_khz: usize,
) {
    let mut xcorr32 = [0i32; SCRATCH_SIZE];

    let mut target_ptr_idx = PE_LTP_MEM_LENGTH_MS * (if fs_khz > 8 { fs_khz } else { 8 });
    for k in 0..nb_subfr {
        let (_, lag_range) = stage3_tables(nb_subfr, complexity, k);
        let lag_low = lag_range[0] as i32;
        let lag_high = lag_range[1] as i32;
        let lag_counter_total = (lag_high - lag_low + 1) as usize;
        assert!(lag_counter_total <= SCRATCH_SIZE);
        silk_pitch_xcorr(
            &frame[target_ptr_idx..],
            &frame[(target_ptr_idx as i32 - start_lag as i32 - lag_high) as usize..],
            &mut xcorr32,
            sf_length,
            lag_counter_total,
        );
        // `silk_pitch_xcorr` returns the highest lag first.
        for (lowest_first, &highest_first) in cross_corr_st3[k][..lag_counter_total]
            .iter_mut()
            .zip(xcorr32[..lag_counter_total].iter().rev())
        {
            *lowest_first = highest_first;
        }
        target_ptr_idx += sf_length;
    }
}

/// Energy of the signal that every subframe is compared with, for each lag
/// of the search range of the subframe, in the layout of
/// `silk_p_ana_calc_corr_st3`.
fn silk_p_ana_calc_energy_st3(
    energies_st3: &mut [[i32; SCRATCH_SIZE]; PE_MAX_NB_SUBFR],
    frame: &[i16],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    fs_khz: usize,
) {
    let mut target_ptr_idx = PE_LTP_MEM_LENGTH_MS * (if fs_khz > 8 { fs_khz } else { 8 });
    for k in 0..nb_subfr {
        let (_, lag_range) = stage3_tables(nb_subfr, complexity, k);

        let basis_ptr_idx =
            (target_ptr_idx as i32 - (start_lag as i32 + lag_range[0] as i32)) as usize;
        let mut energy =
            silk_inner_prod_aligned(&frame[basis_ptr_idx..], &frame[basis_ptr_idx..], sf_length);
        energies_st3[k][0] = energy;

        let lag_diff = (lag_range[1] - lag_range[0] + 1) as usize;
        for i in 1..lag_diff {
            energy = energy.wrapping_sub(silk_smulbb(
                frame[basis_ptr_idx + sf_length - i] as i32,
                frame[basis_ptr_idx + sf_length - i] as i32,
            ));
            let back_idx = basis_ptr_idx - i;
            energy = silk_add_sat32(
                energy,
                silk_smulbb(frame[back_idx] as i32, frame[back_idx] as i32),
            );
            energies_st3[k][i] = energy;
        }
        target_ptr_idx += sf_length;
    }
}

#[cfg(test)]
mod threshold_tests {
    use super::pitch_threshold_q13;

    /// The threshold computed by the statements of libopus 1.6.1
    /// silk/fixed/find_pitch_lags_FIX.c (compiled and run) for every
    /// combination of the inputs in `threshold_matches_libopus`, the last
    /// input changing fastest.
    const LIBOPUS_THRESHOLD_Q13: [i32; 450] = [
        5541, 5031, 4723, 4722, 4223, 3904, 5541, 5031, 4723, 4722, 4223, 3904, 4313, 3803, 3495, 3494, 2995, 2676,
        5499, 4989, 4681, 4680, 4181, 3862, 5499, 4989, 4681, 4680, 4181, 3862, 4271, 3761, 3453, 3452, 2953, 2634,
        5131, 4621, 4313, 4312, 3813, 3494, 5131, 4621, 4313, 4312, 3813, 3494, 3903, 3393, 3085, 3084, 2585, 2266,
        4725, 4215, 3907, 3906, 3407, 3088, 4725, 4215, 3907, 3906, 3407, 3088, 3497, 2987, 2679, 2678, 2179, 1860,
        4721, 4211, 3903, 3902, 3403, 3084, 4721, 4211, 3903, 3902, 3403, 3084, 3493, 2983, 2675, 2674, 2175, 1856,
        5477, 4967, 4659, 4658, 4159, 3840, 5477, 4967, 4659, 4658, 4159, 3840, 4249, 3739, 3431, 3430, 2931, 2612,
        5435, 4925, 4617, 4616, 4117, 3798, 5435, 4925, 4617, 4616, 4117, 3798, 4207, 3697, 3389, 3388, 2889, 2570,
        5067, 4557, 4249, 4248, 3749, 3430, 5067, 4557, 4249, 4248, 3749, 3430, 3839, 3329, 3021, 3020, 2521, 2202,
        4661, 4151, 3843, 3842, 3343, 3024, 4661, 4151, 3843, 3842, 3343, 3024, 3433, 2923, 2615, 2614, 2115, 1796,
        4657, 4147, 3839, 3838, 3339, 3020, 4657, 4147, 3839, 3838, 3339, 3020, 3429, 2919, 2611, 2610, 2111, 1792,
        5413, 4903, 4595, 4594, 4095, 3776, 5413, 4903, 4595, 4594, 4095, 3776, 4185, 3675, 3367, 3366, 2867, 2548,
        5371, 4861, 4553, 4552, 4053, 3734, 5371, 4861, 4553, 4552, 4053, 3734, 4143, 3633, 3325, 3324, 2825, 2506,
        5003, 4493, 4185, 4184, 3685, 3366, 5003, 4493, 4185, 4184, 3685, 3366, 3775, 3265, 2957, 2956, 2457, 2138,
        4597, 4087, 3779, 3778, 3279, 2960, 4597, 4087, 3779, 3778, 3279, 2960, 3369, 2859, 2551, 2550, 2051, 1732,
        4593, 4083, 3775, 3774, 3275, 2956, 4593, 4083, 3775, 3774, 3275, 2956, 3365, 2855, 2547, 2546, 2047, 1728,
        5349, 4839, 4531, 4530, 4031, 3712, 5349, 4839, 4531, 4530, 4031, 3712, 4121, 3611, 3303, 3302, 2803, 2484,
        5307, 4797, 4489, 4488, 3989, 3670, 5307, 4797, 4489, 4488, 3989, 3670, 4079, 3569, 3261, 3260, 2761, 2442,
        4939, 4429, 4121, 4120, 3621, 3302, 4939, 4429, 4121, 4120, 3621, 3302, 3711, 3201, 2893, 2892, 2393, 2074,
        4533, 4023, 3715, 3714, 3215, 2896, 4533, 4023, 3715, 3714, 3215, 2896, 3305, 2795, 2487, 2486, 1987, 1668,
        4529, 4019, 3711, 3710, 3211, 2892, 4529, 4019, 3711, 3710, 3211, 2892, 3301, 2791, 2483, 2482, 1983, 1664,
        5221, 4711, 4403, 4402, 3903, 3584, 5221, 4711, 4403, 4402, 3903, 3584, 3993, 3483, 3175, 3174, 2675, 2356,
        5179, 4669, 4361, 4360, 3861, 3542, 5179, 4669, 4361, 4360, 3861, 3542, 3951, 3441, 3133, 3132, 2633, 2314,
        4811, 4301, 3993, 3992, 3493, 3174, 4811, 4301, 3993, 3992, 3493, 3174, 3583, 3073, 2765, 2764, 2265, 1946,
        4405, 3895, 3587, 3586, 3087, 2768, 4405, 3895, 3587, 3586, 3087, 2768, 3177, 2667, 2359, 2358, 1859, 1540,
        4401, 3891, 3583, 3582, 3083, 2764, 4401, 3891, 3583, 3582, 3083, 2764, 3173, 2663, 2355, 2354, 1855, 1536,
    ];

    #[test]
    fn threshold_matches_libopus() {
        let mut expected = LIBOPUS_THRESHOLD_Q13.iter();
        for order in [6, 8, 10, 12, 16] {
            for speech_activity_q8 in [0, 13, 128, 255, 256] {
                for prev_signal_type in [0, 1, 2] {
                    for input_tilt_q15 in [-32768, -12345, 0, 1, 20000, 32767] {
                        assert_eq!(
                            pitch_threshold_q13(order, speech_activity_q8, prev_signal_type, input_tilt_q15),
                            *expected.next().unwrap(),
                            "order {order}, activity {speech_activity_q8}, previous type {prev_signal_type}, tilt {input_tilt_q15}"
                        );
                    }
                }
            }
        }
        assert!(expected.next().is_none());
    }
}

#[cfg(test)]
mod stage3_table_tests {
    use super::*;

    /// The last search stage reads five neighbouring lags per codebook
    /// vector from the values of a subframe. They must all lie inside the
    /// search range of the subframe.
    #[test]
    fn five_lags_of_every_vector_are_inside_the_search_range() {
        let shapes = [
            (PE_MAX_NB_SUBFR, SILK_PE_MIN_COMPLEX),
            (PE_MAX_NB_SUBFR, SILK_PE_MID_COMPLEX),
            (PE_MAX_NB_SUBFR, SILK_PE_MAX_COMPLEX),
            (PE_MAX_NB_SUBFR / 2, SILK_PE_MIN_COMPLEX),
            (PE_MAX_NB_SUBFR / 2, SILK_PE_MID_COMPLEX),
            (PE_MAX_NB_SUBFR / 2, SILK_PE_MAX_COMPLEX),
        ];
        for (nb_subfr, complexity) in shapes {
            let nb_vectors = if nb_subfr == PE_MAX_NB_SUBFR {
                SILK_NB_CBK_SEARCHS_STAGE3[complexity]
            } else {
                PE_NB_CBKS_STAGE3_10MS
            };
            for k in 0..nb_subfr {
                let (lag_cb, lag_range) = stage3_tables(nb_subfr, complexity, k);
                let nb_lags = (lag_range[1] - lag_range[0] + 1) as usize;
                assert!(nb_lags <= SCRATCH_SIZE);
                for &offset in &lag_cb[..nb_vectors] {
                    assert!(lag_range[0] <= offset, "subframe {k}, offset {offset}");
                    assert!(
                        offset as i32 + PE_NB_STAGE3_LAGS as i32 - 1 <= lag_range[1] as i32,
                        "subframe {k}, offset {offset}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod div_tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::silk_div32_varq_batch;
    use crate::silk::macros::silk_div32_varq;

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

    /// Compare the batch function with one scalar call per element.
    fn check(a: &[i32], b: &[i32], qres: i32) {
        let expected: Vec<i32> = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| silk_div32_varq(a, b, qres))
            .collect();
        let mut out = vec![0x5a5a_5a5ai32; a.len()];
        silk_div32_varq_batch(&mut out, a, b, qres);
        assert_eq!(out, expected, "qres {qres}, a {a:?}, b {b:?}");
    }

    /// Powers of two and their neighbours with both signs, and the ends of
    /// the 32-bit range: every count of leading zeros occurs.
    fn special_values() -> Vec<i32> {
        let mut values = vec![0, i32::MIN, i32::MIN + 1, i32::MAX, i32::MAX - 1];
        for bit in 0..31 {
            let power = 1i32 << bit;
            for value in [power - 1, power, power + 1] {
                values.push(value);
                values.push(-value);
            }
        }
        values.sort_unstable();
        values.dedup();
        values
    }

    #[test]
    fn special_values_match_scalar() {
        let values = special_values();
        let divisors: Vec<i32> = values.iter().copied().filter(|&v| v != 0).collect();
        for qres in [0, 1, 13, 14, 15, 16, 21, 24] {
            // Every dividend with every divisor, four at a time.
            for &divisor in &divisors {
                let b = vec![divisor; values.len()];
                check(&values, &b, qres);
            }
        }
    }

    #[test]
    fn random_values_match_scalar() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        for round in 0..200_000 {
            // Any magnitude: a random number of leading bits is cleared.
            let any = |rng: &mut XorShift| (rng.next() as i32) >> (rng.next() % 32);
            let len = round % 10;
            let a: Vec<i32> = (0..len).map(|_| any(&mut rng)).collect();
            let b: Vec<i32> = (0..len)
                .map(|_| {
                    let value = any(&mut rng);
                    if value == 0 { 1 } else { value }
                })
                .collect();
            check(&a, &b, round % 25);
        }
    }

    #[test]
    fn pitch_like_values_match_scalar() {
        // A correlation divided by a larger energy, format Q14: the use in
        // the pitch analysis.
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        for _ in 0..50_000 {
            let energy: Vec<i32> = (0..8)
                .map(|_| 1 + (rng.next() % (1 << 30)) as i32)
                .collect();
            let corr: Vec<i32> = energy
                .iter()
                .map(|&e| ((rng.next() % (2 * e as u64 + 1)) as i64 - e as i64) as i32)
                .collect();
            check(&corr, &energy, 14);
        }
    }

    #[test]
    fn nlsf_weight_values_match_scalar() {
        // A 16-bit weight divided by the square of a 16-bit weight, format
        // Q21: the use in `silk_nlsf_encode`, for 10 and 16 coefficients.
        let mut rng = XorShift(0x0F1E_2D3C_4B5A_6978);
        let edges = [1i32, 2, 255, 256, 32767];
        for round in 0..50_000 {
            let len = if round % 2 == 0 { 10 } else { 16 };
            let pick = |rng: &mut XorShift| {
                if round % 5 == 0 {
                    edges[(rng.next() % edges.len() as u64) as usize]
                } else {
                    1 + (rng.next() % 32767) as i32
                }
            };
            let weight: Vec<i32> = (0..len).map(|_| pick(&mut rng)).collect();
            let squared: Vec<i32> = (0..len)
                .map(|_| {
                    let w = pick(&mut rng);
                    w * w
                })
                .collect();
            check(&weight, &squared, 21);
        }
    }

    /// A zero divisor is an error in `silk_div32_varq`. The batch function
    /// must not hide it.
    #[test]
    #[should_panic]
    fn zero_divisor_panics_like_scalar() {
        let mut out = [0i32; 4];
        silk_div32_varq_batch(&mut out, &[1, 2, 3, 4], &[5, 0, 7, 8], 14);
    }
}
