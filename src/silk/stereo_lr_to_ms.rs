//! Stereo SILK encoding front end: convert left/right to an adaptive mid/side
//! representation and quantize the predictors that let the decoder undo it.
//! Bit-exact ports of libopus 1.5.2 `silk/stereo_LR_to_MS.c`,
//! `stereo_find_predictor.c` and `stereo_quant_pred.c`.

use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::{silk_inner_prod_aligned_scale, silk_sum_sqr_shift};
use crate::silk::structs::SilkStereoState;
use crate::silk::tables::SILK_STEREO_PRED_QUANT_Q13;

/// `SILK_FIX_CONST(STEREO_RATIO_SMOOTH_COEF, 16)` with
/// `STEREO_RATIO_SMOOTH_COEF = 0.01`: how fast the mid/residual norms and the
/// width follow a 20 ms frame (scaled by the speech activity squared).
const RATIO_SMOOTH_COEF_20MS_Q16: i32 = 655;
/// `SILK_FIX_CONST(STEREO_RATIO_SMOOTH_COEF / 2, 16)`, for 10 ms frames.
const RATIO_SMOOTH_COEF_10MS_Q16: i32 = 328;
/// `SILK_FIX_CONST(8 + 5, 16)`: the 13 parts of the default mid/side split.
const MID_SIDE_PARTS_Q16: i32 = 13 << 16;
/// `SILK_FIX_CONST(0.05, 14)`: below this `frac * smoothed width`, a frame
/// that had zero width stays panned mono.
const PANNED_MONO_THRES_Q14: i32 = 819;
/// `SILK_FIX_CONST(0.02, 14)`: below this `frac * smoothed width`, a frame
/// collapses to zero width.
const ZERO_WIDTH_THRES_Q14: i32 = 328;
/// `SILK_FIX_CONST(0.95, 14)`: above this smoothed width, code full width.
const FULL_WIDTH_THRES_Q14: i32 = 15565;

/// Least-squares predictor of `y` from `x` (Q13, limited to ±2), and the
/// ratio of the smoothed residual norm to the smoothed norm of `x` (Q14).
/// `mid_res_amp_q0` holds the two smoothed norms. Port of
/// `silk_stereo_find_predictor`.
pub fn silk_stereo_find_predictor(
    ratio_q14: &mut i32,
    x: &[i16],
    y: &[i16],
    mid_res_amp_q0: &mut [i32],
    length: usize,
    smooth_coef_q16: i32,
) -> i32 {
    let (mut nrgx, mut nrgy, mut scale1, mut scale2) = (0, 0, 0, 0);
    silk_sum_sqr_shift(&mut nrgx, &mut scale1, x, length);
    silk_sum_sqr_shift(&mut nrgy, &mut scale2, y, length);
    let mut scale = scale1.max(scale2);
    scale += scale & 1; // make even
    nrgy = silk_rshift32(nrgy, scale - scale2);
    nrgx = silk_rshift32(nrgx, scale - scale1);
    nrgx = nrgx.max(1);
    let corr = silk_inner_prod_aligned_scale(x, y, scale, length);
    let mut pred_q13 = silk_div32_varq(corr, nrgx, 13);
    pred_q13 = silk_limit(pred_q13, -(1 << 14), 1 << 14);
    let pred2_q10 = silk_smulwb(pred_q13, pred_q13);

    // Faster update for signals with large prediction parameters.
    let smooth_coef_q16 = smooth_coef_q16.max(pred2_q10.abs());

    // Smoothed mid and residual norms.
    let scale = scale >> 1;
    mid_res_amp_q0[0] = silk_smlawb(
        mid_res_amp_q0[0],
        silk_lshift(silk_sqrt_approx(nrgx), scale) - mid_res_amp_q0[0],
        smooth_coef_q16,
    );
    // Residual energy = nrgy - 2 * pred * corr + pred^2 * nrgx.
    nrgy = silk_sub_lshift32(nrgy, silk_smulwb(corr, pred_q13), 3 + 1);
    nrgy = silk_add_lshift32(nrgy, silk_smulwb(nrgx, pred2_q10), 6);
    mid_res_amp_q0[1] = silk_smlawb(
        mid_res_amp_q0[1],
        silk_lshift(silk_sqrt_approx(nrgy), scale) - mid_res_amp_q0[1],
        smooth_coef_q16,
    );

    // Ratio of the smoothed residual and mid norms.
    *ratio_q14 = silk_div32_varq(mid_res_amp_q0[1], mid_res_amp_q0[0].max(1), 14);
    *ratio_q14 = silk_limit(*ratio_q14, 0, 32767);

    pred_q13
}

/// Quantize the two predictors in place and return their indices
/// `[interval % 3, sub-step, interval / 3]`; then subtract the second
/// predictor from the first, as the decoder expects. Port of
/// `silk_stereo_quant_pred`.
pub fn silk_stereo_quant_pred(pred_q13: &mut [i32; 2], ix: &mut [[i8; 3]; 2]) {
    let mut quant_pred_q13 = 0;
    for n in 0..2 {
        // Brute-force search over the quantization levels. The levels rise
        // monotonically, so the first level that doesn't improve on the
        // previous one ends the search (libopus's `goto done`).
        let mut err_min_q13 = i32::MAX;
        'search: for i in 0..STEREO_QUANT_TAB_SIZE - 1 {
            let low_q13 = SILK_STEREO_PRED_QUANT_Q13[i] as i32;
            let step_q13 = silk_smulwb(
                SILK_STEREO_PRED_QUANT_Q13[i + 1] as i32 - low_q13,
                STEREO_HALF_SUB_STEP_Q16,
            );
            for j in 0..STEREO_QUANT_SUB_STEPS {
                let lvl_q13 = silk_smlabb(low_q13, step_q13, 2 * j + 1);
                let err_q13 = (pred_q13[n] - lvl_q13).abs();
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    break 'search;
                }
            }
        }
        ix[n][2] = ix[n][0] / 3;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }

    // Subtract the second predictor from the first (helps when applying them).
    pred_q13[0] -= pred_q13[1];
}

/// Convert one frame of left/right to adaptive mid/side, in place. Port of
/// `silk_stereo_LR_to_MS`.
///
/// `x1` and `x2` are laid out like libopus's per-channel `inputBuf`: two
/// history slots, then the frame's `frame_length` left (`x1`) or right (`x2`)
/// samples at `[2..2 + frame_length]`. On return:
/// - `x1[0..frame_length + 2]` is the mid with its two history samples; the
///   mid channel codes `x1[1..1 + frame_length]`;
/// - `x2[1..1 + frame_length]` is the side residual after prediction from the
///   mid, which the side channel codes.
///
/// Also returns the predictor indices to write, whether to code the mid only,
/// and the split of `total_rate_bps` between mid and side.
pub fn silk_stereo_lr_to_ms(
    state: &mut SilkStereoState,
    x1: &mut [i16],
    x2: &mut [i16],
    ix: &mut [[i8; 3]; 2],
    mid_only_flag: &mut i8,
    mid_side_rates_bps: &mut [i32; 2],
    mut total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: bool,
    fs_khz: i32,
    frame_length: usize,
) {
    let fl = frame_length;
    let x1 = &mut x1[..fl + 2];
    let x2 = &mut x2[..fl + 2];

    // Convert to basic mid/side signals. The mid overwrites the left input.
    let mut side = [0i16; MAX_FRAME_LENGTH + 2];
    for n in 0..fl + 2 {
        let sum = x1[n] as i32 + x2[n] as i32;
        let diff = x1[n] as i32 - x2[n] as i32;
        x1[n] = silk_rshift_round(sum, 1) as i16;
        side[n] = silk_sat16(silk_rshift_round(diff, 1)) as i16;
    }
    let mid = x1;
    let side = &mut side[..fl + 2];

    // Buffering.
    mid[..2].copy_from_slice(&state.s_mid);
    side[..2].copy_from_slice(&state.s_side);
    state.s_mid.copy_from_slice(&mid[fl..fl + 2]);
    state.s_side.copy_from_slice(&side[fl..fl + 2]);

    // LP and HP filter the mid and side signals.
    let mut lp_mid = [0i16; MAX_FRAME_LENGTH];
    let mut hp_mid = [0i16; MAX_FRAME_LENGTH];
    let mut lp_side = [0i16; MAX_FRAME_LENGTH];
    let mut hp_side = [0i16; MAX_FRAME_LENGTH];
    for n in 0..fl {
        let sum = silk_rshift_round(
            silk_add_lshift32(mid[n] as i32 + mid[n + 2] as i32, mid[n + 1] as i32, 1),
            2,
        );
        lp_mid[n] = sum as i16;
        hp_mid[n] = (mid[n + 1] as i32 - sum) as i16;
        let sum = silk_rshift_round(
            silk_add_lshift32(side[n] as i32 + side[n + 2] as i32, side[n + 1] as i32, 1),
            2,
        );
        lp_side[n] = sum as i16;
        hp_side[n] = (side[n + 1] as i32 - sum) as i16;
    }

    // Find energies and predictors.
    let is_10ms_frame = fl as i32 == 10 * fs_khz;
    let mut smooth_coef_q16 = if is_10ms_frame {
        RATIO_SMOOTH_COEF_10MS_Q16
    } else {
        RATIO_SMOOTH_COEF_20MS_Q16
    };
    smooth_coef_q16 = silk_smulwb(
        silk_smulbb(prev_speech_act_q8, prev_speech_act_q8),
        smooth_coef_q16,
    );

    let (mut lp_ratio_q14, mut hp_ratio_q14) = (0, 0);
    let mut pred_q13 = [
        silk_stereo_find_predictor(
            &mut lp_ratio_q14,
            &lp_mid,
            &lp_side,
            &mut state.mid_side_amp_q0[0..2],
            fl,
            smooth_coef_q16,
        ),
        silk_stereo_find_predictor(
            &mut hp_ratio_q14,
            &hp_mid,
            &hp_side,
            &mut state.mid_side_amp_q0[2..4],
            fl,
            smooth_coef_q16,
        ),
    ];
    // Ratio of the norms of the residual and mid signals.
    let mut frac_q16 = silk_smlabb(hp_ratio_q14, lp_ratio_q14, 3);
    frac_q16 = frac_q16.min(1 << 16);

    // Bitrate split between mid and side, possibly reducing the stereo width.
    // Subtract the approximate bitrate of the stereo parameters.
    total_rate_bps -= if is_10ms_frame { 1200 } else { 600 };
    if total_rate_bps < 1 {
        total_rate_bps = 1;
    }
    let min_mid_rate_bps = silk_smlabb(2000, fs_khz, 600);
    // Default split: 8 parts for the mid and 5 + 3 * frac for the side, so
    // mid_rate = 8 / (13 + 3 * frac) * total_rate.
    let frac_3_q16 = silk_mul(3, frac_q16);
    mid_side_rates_bps[0] =
        silk_div32_varq(total_rate_bps, MID_SIDE_PARTS_Q16 + frac_3_q16, 16 + 3);
    let mut width_q14;
    if mid_side_rates_bps[0] < min_mid_rate_bps {
        // Mid rate below the minimum: reduce the stereo width.
        mid_side_rates_bps[0] = min_mid_rate_bps;
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        // width = 4 * (2 * side_rate - min_rate) / ((1 + 3 * frac) * min_rate)
        width_q14 = silk_div32_varq(
            silk_lshift(mid_side_rates_bps[1], 1) - min_mid_rate_bps,
            silk_smulwb((1 << 16) + frac_3_q16, min_mid_rate_bps),
            14 + 2,
        );
        width_q14 = silk_limit(width_q14, 0, 1 << 14);
    } else {
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = 1 << 14;
    }

    // Smoother.
    state.smth_width_q14 = silk_smlawb(
        state.smth_width_q14 as i32,
        width_q14 - state.smth_width_q14 as i32,
        smooth_coef_q16,
    ) as i16;
    let smth_width_q14 = state.smth_width_q14 as i32;

    // At very low bitrates, or for inputs that are nearly amplitude panned,
    // switch to panned-mono coding.
    *mid_only_flag = 0;
    if to_mono {
        // Last frame before a stereo -> mono transition: collapse the width.
        width_q14 = 0;
        pred_q13 = [0, 0];
        silk_stereo_quant_pred(&mut pred_q13, ix);
    } else if state.width_prev_q14 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps
            || silk_smulwb(frac_q16, smth_width_q14) < PANNED_MONO_THRES_Q14)
    {
        // Panned mono; the previous frame already had zero width. Scale down
        // and quantize the predictors, then collapse the width.
        scale_preds(&mut pred_q13, smth_width_q14);
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = 0;
        pred_q13 = [0, 0];
        mid_side_rates_bps[0] = total_rate_bps;
        mid_side_rates_bps[1] = 0;
        *mid_only_flag = 1;
    } else if state.width_prev_q14 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps
            || silk_smulwb(frac_q16, smth_width_q14) < ZERO_WIDTH_THRES_Q14)
    {
        // Transition to zero-width stereo.
        scale_preds(&mut pred_q13, smth_width_q14);
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = 0;
        pred_q13 = [0, 0];
    } else if smth_width_q14 > FULL_WIDTH_THRES_Q14 {
        // Full-width stereo coding.
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = 1 << 14;
    } else {
        // Reduced-width stereo coding.
        scale_preds(&mut pred_q13, smth_width_q14);
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = smth_width_q14;
    }

    // Keep coding the side until its tapered output has been transmitted.
    if *mid_only_flag == 1 {
        state.silent_side_len =
            (state.silent_side_len as i32 + fl as i32 - STEREO_INTERP_LEN_MS * fs_khz) as i16;
        if (state.silent_side_len as i32) < LA_SHAPE_MS as i32 * fs_khz {
            *mid_only_flag = 0;
        } else {
            // Limit to avoid wrapping around.
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }

    if *mid_only_flag == 0 && mid_side_rates_bps[1] < 1 {
        mid_side_rates_bps[1] = 1;
        mid_side_rates_bps[0] = (total_rate_bps - mid_side_rates_bps[1]).max(1);
    }

    // Interpolate the predictors and width over the first
    // STEREO_INTERP_LEN_MS, then subtract the prediction from the side.
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
    let mut pred0_q13 = -(state.pred_prev_q13[0] as i32);
    let mut pred1_q13 = -(state.pred_prev_q13[1] as i32);
    let mut w_q24 = silk_lshift(state.width_prev_q14 as i32, 10);
    let denom_q16 = silk_div32_16(1 << 16, STEREO_INTERP_LEN_MS * fs_khz);
    let delta0_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );
    let deltaw_q24 = silk_lshift(
        silk_smulwb(width_q14 - state.width_prev_q14 as i32, denom_q16),
        10,
    );
    // C writes `x2[n - 1]` with `x2` at `inputBuf + 2`: index `n + 1` here.
    let residual = |n: usize, pred0_q13: i32, pred1_q13: i32, w_q24: i32| -> i16 {
        let mut sum = silk_lshift(
            silk_add_lshift32(mid[n] as i32 + mid[n + 2] as i32, mid[n + 1] as i32, 1),
            9,
        ); // Q11
        sum = silk_smlawb(silk_smulwb(w_q24, side[n + 1] as i32), sum, pred0_q13); // Q8
        sum = silk_smlawb(sum, silk_lshift(mid[n + 1] as i32, 11), pred1_q13); // Q8
        silk_sat16(silk_rshift_round(sum, 8)) as i16
    };
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        w_q24 += deltaw_q24;
        x2[n + 1] = residual(n, pred0_q13, pred1_q13, w_q24);
    }
    let (pred0_q13, pred1_q13) = (-pred_q13[0], -pred_q13[1]);
    let w_q24 = silk_lshift(width_q14, 10);
    for n in interp_len..fl {
        x2[n + 1] = residual(n, pred0_q13, pred1_q13, w_q24);
    }
    state.pred_prev_q13 = [pred_q13[0] as i16, pred_q13[1] as i16];
    state.width_prev_q14 = width_q14 as i16;
}

/// Scale both predictors by the smoothed width (Q14).
fn scale_preds(pred_q13: &mut [i32; 2], smth_width_q14: i32) {
    for p in pred_q13 {
        *p = silk_rshift(silk_smulbb(smth_width_q14, *p), 14);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::range_coder::RangeCoder;
    use crate::silk::decode_indices::silk_stereo_decode_pred;
    use crate::silk::encode_indices::{silk_stereo_encode_mid_only, silk_stereo_encode_pred};
    use audiopus_sys as _;

    /// `stereo_enc_state` of libopus (silk/structs.h).
    #[repr(C)]
    struct CStereoState {
        pred_prev_q13: [i16; 2],
        s_mid: [i16; 2],
        s_side: [i16; 2],
        mid_side_amp_q0: [i32; 4],
        smth_width_q14: i16,
        width_prev_q14: i16,
        silent_side_len: i16,
        pred_ix: [[[i8; 3]; 2]; 3],
        mid_only_flags: [i8; 3],
    }

    /// `ec_ctx` of libopus (celt/entcode.h).
    #[repr(C)]
    struct EcCtx {
        buf: *mut u8,
        storage: u32,
        end_offs: u32,
        end_window: u32,
        nend_bits: i32,
        nbits_total: i32,
        offs: u32,
        rng: u32,
        val: u32,
        ext: u32,
        rem: i32,
        error: i32,
    }

    impl CStereoState {
        /// libopus's state with the values of `s`.
        fn of(s: &SilkStereoState) -> Self {
            CStereoState {
                pred_prev_q13: s.pred_prev_q13,
                s_mid: s.s_mid,
                s_side: s.s_side,
                mid_side_amp_q0: s.mid_side_amp_q0,
                smth_width_q14: s.smth_width_q14,
                width_prev_q14: s.width_prev_q14,
                silent_side_len: s.silent_side_len,
                pred_ix: s.pred_ix,
                mid_only_flags: s.mid_only_flags,
            }
        }
    }

    // The functions of the float libopus that the tests link: integer code,
    // the same in every build.
    unsafe extern "C" {
        fn ec_enc_init(this: *mut EcCtx, buf: *mut u8, size: u32);
        fn ec_enc_done(this: *mut EcCtx);
        /// `x1` and `x2` point 2 samples into their buffers: libopus reads
        /// and writes `x1[-2]` and `x1[-1]`.
        #[link_name = "silk_stereo_LR_to_MS"]
        fn c_silk_stereo_lr_to_ms(
            state: *mut CStereoState,
            x1: *mut i16,
            x2: *mut i16,
            ix: *mut [i8; 3],
            mid_only_flag: *mut i8,
            mid_side_rates_bps: *mut i32,
            total_rate_bps: i32,
            prev_speech_act_q8: i32,
            to_mono: i32,
            fs_khz: i32,
            frame_length: i32,
        );
        #[link_name = "silk_stereo_encode_pred"]
        fn c_silk_stereo_encode_pred(enc: *mut EcCtx, ix: *mut [i8; 3]);
        #[link_name = "silk_stereo_encode_mid_only"]
        fn c_silk_stereo_encode_mid_only(enc: *mut EcCtx, mid_only_flag: i8);
    }

    /// An integer test-signal generator: xorshift32 noise in seven shapes.
    struct Gen {
        rng: u32,
        lp: i32,
    }

    impl Gen {
        fn xs(&mut self) -> u32 {
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.rng = x;
            x
        }

        fn noise(&mut self) -> i32 {
            (self.xs() >> 16) as i32 - 32768
        }

        fn pair(&mut self, kind: u32, sh: u32) -> (i32, i32) {
            let sat = |v: i32| v.clamp(-32768, 32767);
            let (s, n1, n2) = (self.noise(), self.noise(), self.noise());
            let (l, r) = match kind {
                0 => (s >> sh, n1 >> sh),
                1 => ((s * 29 / 32) >> sh, (s * 9 / 32) >> sh),
                2 => ((s + n1 / 8) >> sh, (s * 3 / 4 + n2 / 8) >> sh),
                3 => (0, 0),
                4 => {
                    let c = [-32768, 32767, 0];
                    let l = c[(self.xs() % 3) as usize];
                    (l, c[(self.xs() % 3) as usize])
                }
                5 => (s >> sh, sat(-(s >> sh))),
                _ => {
                    self.lp = self.lp - (self.lp >> 3) + (s >> 3);
                    (sat(self.lp >> sh), sat((self.lp + (n1 >> 6)) >> sh))
                }
            };
            (sat(l), sat(r))
        }
    }

    /// Runs `frames` frames of stream `st` through opus1 and through
    /// libopus's `silk_stereo_LR_to_MS`, `silk_stereo_encode_pred` and
    /// `silk_stereo_encode_mid_only`. After each frame the mid and side
    /// buffers, the indices, the mid-only flag, the rates and the stereo
    /// state must be equal; at the end the coded bytes.
    fn compare_stream(st: u32, frames: usize) {
        let mut g = Gen {
            rng: 0x9E37_79B9 ^ st.wrapping_mul(2_654_435_761),
            lp: 0,
        };
        let fs_khz = [8, 12, 16][(g.xs() % 3) as usize];
        let fl = if g.xs() & 1 != 0 {
            10 * fs_khz
        } else {
            20 * fs_khz
        } as usize;
        let mut kind = st % 7;
        let sh = g.xs() % 9;
        // Both start from opus1's initial state, whose norms are those that
        // libopus's silk_Encode sets when the input becomes stereo
        // (enc_API.c:195-198).
        let mut state = SilkStereoState::default();
        let mut c_state = CStereoState::of(&state);
        let mut rc = RangeCoder::new_encoder(1275);
        let mut c_buf = [0u8; 1275];
        // SAFETY: an all-zero `ec_ctx` is valid input to `ec_enc_init`.
        let mut c_rc: EcCtx = unsafe { core::mem::zeroed() };
        // SAFETY: the buffer holds 1275 bytes, the size given.
        unsafe { ec_enc_init(&mut c_rc, c_buf.as_mut_ptr(), 1275) };
        for f in 0..frames {
            let rate = match g.xs() % 6 {
                0 => 1 + g.xs() % 2000,
                1 => 1 + g.xs() % 12000,
                _ => 1 + g.xs() % 80000,
            } as i32;
            let sa = (g.xs() % 256) as i32;
            let to_mono = g.xs() & 15 == 0;
            let (mut x1, mut x2) = ([0i16; MAX_FRAME_LENGTH + 2], [0i16; MAX_FRAME_LENGTH + 2]);
            for n in 0..fl + 2 {
                let (l, r) = if n < 2 {
                    (g.noise(), g.noise())
                } else {
                    g.pair(kind, sh)
                };
                x1[n] = l as i16;
                x2[n] = r as i16;
            }
            if st % 5 == 4 && f == frames / 2 {
                kind = (kind + 3) % 7;
            }
            let (mut c_x1, mut c_x2) = (x1, x2);
            let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
            silk_stereo_lr_to_ms(
                &mut state,
                &mut x1,
                &mut x2,
                &mut ix,
                &mut mid_only,
                &mut rates,
                rate,
                sa,
                to_mono,
                fs_khz,
                fl,
            );
            silk_stereo_encode_pred(&mut rc, &ix);
            silk_stereo_encode_mid_only(&mut rc, mid_only);
            let (mut c_ix, mut c_mid_only, mut c_rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
            // SAFETY: both buffers hold 2 samples before the frame and
            // `fl` samples of it; the outputs have the sizes libopus writes.
            unsafe {
                c_silk_stereo_lr_to_ms(
                    &mut c_state,
                    c_x1.as_mut_ptr().add(2),
                    c_x2.as_mut_ptr().add(2),
                    c_ix.as_mut_ptr(),
                    &mut c_mid_only,
                    c_rates.as_mut_ptr(),
                    rate,
                    sa,
                    to_mono as i32,
                    fs_khz,
                    fl as i32,
                );
                c_silk_stereo_encode_pred(&mut c_rc, c_ix.as_mut_ptr());
                c_silk_stereo_encode_mid_only(&mut c_rc, c_mid_only);
            }
            let what = format!("stream {st} ({fs_khz} kHz, {fl} samples), frame {f}");
            assert_eq!(x1[..fl + 2], c_x1[..fl + 2], "{what}: mid");
            assert_eq!(x2[..fl + 2], c_x2[..fl + 2], "{what}: side");
            assert_eq!((ix, mid_only, rates), (c_ix, c_mid_only, c_rates), "{what}: indices, mid only, rates");
            let s = &state;
            let c = &c_state;
            assert_eq!(
                (s.pred_prev_q13, s.s_mid, s.s_side, s.mid_side_amp_q0),
                (c.pred_prev_q13, c.s_mid, c.s_side, c.mid_side_amp_q0),
                "{what}: state"
            );
            assert_eq!(
                (s.smth_width_q14, s.width_prev_q14, s.silent_side_len),
                (c.smth_width_q14, c.width_prev_q14, c.silent_side_len),
                "{what}: width state"
            );
        }
        rc.done();
        // SAFETY: `c_rc` was set up by `ec_enc_init` on `c_buf`.
        unsafe { ec_enc_done(&mut c_rc) };
        assert_eq!(c_rc.error, 0);
        assert_eq!(&rc.buf[..1275], &c_buf[..], "stream {st}: coded bytes");
    }

    /// Every output of libopus's own `silk_stereo_LR_to_MS`,
    /// `silk_stereo_encode_pred` and `silk_stereo_encode_mid_only` (the
    /// linked library) on the same frames: 14 streams of 40 frames, every
    /// signal shape, all three internal rates, 10 and 20 ms frames.
    #[test]
    fn lr_to_ms_matches_libopus() {
        for st in 0..14 {
            compare_stream(st, 40);
        }
    }

    /// Range-code `ix` and decode it, as the decoder sees it.
    fn decoded(ix: &[[i8; 3]; 2]) -> [i32; 2] {
        let mut enc = RangeCoder::new_encoder(64);
        silk_stereo_encode_pred(&mut enc, ix);
        let bytes = enc.finish();
        silk_stereo_decode_pred(&mut RangeCoder::new_decoder(&bytes))
    }

    #[test]
    fn quant_pred_returns_the_level_the_decoder_reads() {
        // Every Q13 value in range quantizes to a level, and the decoder reads
        // back exactly what the encoder subtracts.
        for p0 in (-16384..=16384).step_by(61) {
            for p1 in [-16384, -13364, -2000, 0, 1, 656, 9000, 16384] {
                let mut pred = [p0, p1];
                let mut ix = [[0i8; 3]; 2];
                silk_stereo_quant_pred(&mut pred, &mut ix);
                assert_eq!(decoded(&ix), pred, "pred ({p0}, {p1})");
            }
        }
    }

    #[test]
    fn quant_pred_picks_the_nearest_level() {
        // An exact level quantizes to itself; zero to the neutral index.
        for p in [-13364, -8192, -656, 0, 328, 7600, 13362] {
            let mut pred = [p, 0];
            let mut ix = [[0i8; 3]; 2];
            silk_stereo_quant_pred(&mut pred, &mut ix);
            assert_eq!(pred, [p, 0]);
        }
        let mut pred = [0, 0];
        let mut ix = [[0i8; 3]; 2];
        silk_stereo_quant_pred(&mut pred, &mut ix);
        assert_eq!(ix, [[1, 2, 2], [1, 2, 2]]);
    }

    #[test]
    fn find_predictor_is_the_least_squares_gain() {
        let x: Vec<i16> = (0..320)
            .map(|n| ((n * 7919) % 2001 - 1000) as i16 * 9)
            .collect();
        let pred = |y: &[i16]| {
            let (mut ratio, mut amp) = (0, [0i32, 1]);
            let p = silk_stereo_find_predictor(&mut ratio, &x, y, &mut amp, x.len(), 0);
            (p, ratio)
        };
        let neg: Vec<i16> = x.iter().map(|&v| -v).collect();
        let triple: Vec<i16> = x.iter().map(|&v| v * 3).collect();
        // A copy predicts with gain 1 (8192 in Q13, less the rounding of
        // silk_DIV32_varQ) and leaves no residual.
        let (copy, ratio) = pred(&x);
        assert!((copy - 8192).abs() <= 2, "copy: {copy}");
        assert_eq!(ratio, 0);
        let inverse = pred(&neg).0;
        assert!((inverse + 8192).abs() <= 2, "inverse: {inverse}");
        // Gains beyond 2 are clamped.
        assert_eq!(pred(&triple).0, 1 << 14);
        assert_eq!(pred(&[0; 320]).0, 0);
    }

    /// Run `frames` frames of identical L and R through `silk_stereo_lr_to_ms`
    /// and return each frame's mid-only flag; the side residual must be zero.
    fn mono_input_mid_only(fs_khz: i32, frame_ms: i32, frames: usize) -> Vec<i8> {
        let fl = (frame_ms * fs_khz) as usize;
        let mut state = SilkStereoState::default();
        let mut flags = Vec::new();
        for f in 0..frames {
            let (mut x1, mut x2) = ([0i16; MAX_FRAME_LENGTH + 2], [0i16; MAX_FRAME_LENGTH + 2]);
            for n in 2..fl + 2 {
                let v = (((f * fl + n) * 2654435761) >> 20) as i16;
                x1[n] = v;
                x2[n] = v;
            }
            let last = [x1[fl], x1[fl + 1]];
            let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
            silk_stereo_lr_to_ms(
                &mut state,
                &mut x1,
                &mut x2,
                &mut ix,
                &mut mid_only,
                &mut rates,
                32000,
                255,
                false,
                fs_khz,
                fl,
            );
            assert!(
                x2[1..1 + fl].iter().all(|&s| s == 0),
                "frame {f}: side residual"
            );
            assert_eq!(state.s_mid, last, "frame {f}: mid history");
            flags.push(mid_only);
        }
        flags
    }

    #[test]
    fn mono_input_codes_mid_only_once_the_side_taper_is_sent() {
        // The side keeps being coded until the 8 ms predictor taper plus the
        // 5 ms shaping lookahead are out: one 20 ms frame, three 10 ms frames.
        assert_eq!(mono_input_mid_only(16, 20, 3), [1, 1, 1]);
        assert_eq!(mono_input_mid_only(16, 10, 4), [0, 0, 1, 1]);
        assert_eq!(mono_input_mid_only(8, 10, 4), [0, 0, 1, 1]);
    }
}
