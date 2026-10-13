//! SILK comfort noise generation, ported from libopus `silk/CNG.c`.
//!
//! While frames without voice activity are decoded, the decoder keeps a
//! smoothed estimate of the background noise: its spectrum, its level and a
//! buffer of its excitation. When a frame is lost (or missing because the
//! encoder stopped sending during silence), noise with that spectrum and
//! level is added to the concealed frame, so the background does not drop
//! out.

use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::nlsf::silk_nlsf2a;

/// `2^floor(log2(MAX_FRAME_LENGTH)) - 1`.
const CNG_BUF_MASK_MAX: i32 = 255;
/// Smoothing of the noise level per subframe: `0.25^(1/4)` in Q16.
const CNG_GAIN_SMTH_Q16: i32 = 4634;
/// -3 dB in Q16.
const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46396;
/// Smoothing of the noise spectrum per frame: 0.25 in Q16.
const CNG_NLSF_SMTH_Q16: i32 = 16348;

#[inline(always)]
fn silk_smultt(a: i32, b: i32) -> i32 {
    (a >> 16).wrapping_mul(b >> 16)
}

/// Draw `length` excitation samples at random from the noise buffer
/// (`silk_CNG_exc`).
fn silk_cng_exc(exc_q14: &mut [i32], exc_buf_q14: &[i32], length: usize, rand_seed: &mut i32) {
    let mut exc_mask = CNG_BUF_MASK_MAX;
    while exc_mask > length as i32 {
        exc_mask >>= 1;
    }
    let mut seed = *rand_seed;
    for v in exc_q14[..length].iter_mut() {
        seed = silk_rand(seed);
        let idx = ((seed >> 24) & exc_mask) as usize;
        *v = exc_buf_q14[idx];
    }
    *rand_seed = seed;
}

/// Reset the noise estimate (`silk_CNG_Reset`): a flat spectrum, no level.
pub fn silk_cng_reset(ps_dec: &mut SilkDecoderState) {
    let nlsf_step_q15 = silk_div32_16(i16::MAX as i32, ps_dec.lpc_order + 1);
    let mut nlsf_acc_q15 = 0;
    for i in 0..ps_dec.lpc_order as usize {
        nlsf_acc_q15 += nlsf_step_q15;
        ps_dec.s_cng.cng_smth_nlsf_q15[i] = nlsf_acc_q15 as i16;
    }
    ps_dec.s_cng.cng_smth_gain_q16 = 0;
    ps_dec.s_cng.rand_seed = 3176576;
}

/// Update the noise estimate from a decoded frame, and add comfort noise to
/// a concealed one (`silk_CNG`). `frame` holds `length` samples.
pub fn silk_cng(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &SilkDecoderControl,
    frame: &mut [i16],
    length: usize,
) {
    if ps_dec.fs_khz != ps_dec.s_cng.fs_khz {
        silk_cng_reset(ps_dec);
        ps_dec.s_cng.fs_khz = ps_dec.fs_khz;
    }
    let order = ps_dec.lpc_order as usize;
    let nb_subfr = ps_dec.nb_subfr as usize;
    let subfr_length = ps_dec.subfr_length as usize;

    if ps_dec.loss_cnt == 0 && ps_dec.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // Smooth the spectrum.
        for i in 0..order {
            let cur = ps_dec.s_cng.cng_smth_nlsf_q15[i] as i32;
            ps_dec.s_cng.cng_smth_nlsf_q15[i] = (cur
                + silk_smulwb(ps_dec.prev_nlsf_q15[i] as i32 - cur, CNG_NLSF_SMTH_Q16))
                as i16;
        }
        // The subframe with the highest gain supplies new excitation.
        let mut max_gain_q16 = 0;
        let mut subfr = 0;
        for i in 0..nb_subfr {
            if ps_dec_ctrl.gains_q16[i] > max_gain_q16 {
                max_gain_q16 = ps_dec_ctrl.gains_q16[i];
                subfr = i;
            }
        }
        ps_dec
            .s_cng
            .cng_exc_buf_q14
            .copy_within(0..(nb_subfr - 1) * subfr_length, subfr_length);
        ps_dec.s_cng.cng_exc_buf_q14[..subfr_length]
            .copy_from_slice(&ps_dec.exc_q14[subfr * subfr_length..(subfr + 1) * subfr_length]);
        // Smooth the level.
        for i in 0..nb_subfr {
            let gain = ps_dec_ctrl.gains_q16[i];
            ps_dec.s_cng.cng_smth_gain_q16 = ps_dec.s_cng.cng_smth_gain_q16.wrapping_add(
                silk_smulwb(gain.wrapping_sub(ps_dec.s_cng.cng_smth_gain_q16), CNG_GAIN_SMTH_Q16),
            );
            // If the smoothed level is 3 dB above this subframe's, follow
            // the subframe to adapt faster.
            if silk_smulww(ps_dec.s_cng.cng_smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16) > gain {
                ps_dec.s_cng.cng_smth_gain_q16 = gain;
            }
        }
    }

    if ps_dec.loss_cnt != 0 {
        // Noise level: what the concealment's own noise does not cover.
        let mut gain_q16 = silk_smulww(
            ps_dec.s_plc.rand_scale_q14 as i32,
            ps_dec.s_plc.prev_gain_q16[1],
        );
        let smth = ps_dec.s_cng.cng_smth_gain_q16;
        if gain_q16 >= (1 << 21) || smth > (1 << 23) {
            gain_q16 = silk_smultt(gain_q16, gain_q16);
            gain_q16 = silk_smultt(smth, smth).wrapping_sub(gain_q16.wrapping_shl(5));
            gain_q16 = silk_sqrt_approx(gain_q16).wrapping_shl(16);
        } else {
            gain_q16 = silk_smulww(gain_q16, gain_q16);
            gain_q16 = silk_smulww(smth, smth).wrapping_sub(gain_q16.wrapping_shl(5));
            gain_q16 = silk_sqrt_approx(gain_q16).wrapping_shl(8);
        }
        let gain_q10 = gain_q16 >> 6;

        let mut cng_sig_q14 = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];
        {
            let (exc_buf, seed) = (&ps_dec.s_cng.cng_exc_buf_q14, &mut ps_dec.s_cng.rand_seed);
            let mut s = *seed;
            silk_cng_exc(&mut cng_sig_q14[MAX_LPC_ORDER..], exc_buf, length, &mut s);
            *seed = s;
        }

        // Synthesis filter from the smoothed spectrum.
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        silk_nlsf2a(&mut a_q12, &ps_dec.s_cng.cng_smth_nlsf_q15, order);
        cng_sig_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_cng.cng_synth_state);
        for i in 0..length {
            // The bias avoids the downward rounding of silk_smlawb.
            let mut lpc_pred_q10 = (order >> 1) as i32;
            for j in 0..order {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + i - j - 1],
                    a_q12[j] as i32,
                );
            }
            cng_sig_q14[MAX_LPC_ORDER + i] = silk_add_sat32(
                cng_sig_q14[MAX_LPC_ORDER + i],
                silk_lshift_sat32(lpc_pred_q10, 4),
            );
            // Scale and add to the concealed signal.
            let noise = silk_sat16(silk_rshift_round(
                silk_smulww(cng_sig_q14[MAX_LPC_ORDER + i], gain_q10),
                8,
            )) as i16;
            frame[i] = silk_add_sat16(frame[i], noise);
        }
        ps_dec
            .s_cng
            .cng_synth_state
            .copy_from_slice(&cng_sig_q14[length..length + MAX_LPC_ORDER]);
    } else {
        ps_dec.s_cng.cng_synth_state[..order].fill(0);
    }
}
