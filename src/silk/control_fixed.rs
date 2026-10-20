use crate::silk::define::*;
use crate::silk::gain_quant::*;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::lpc_analysis::*;
use crate::silk::ltp_analysis::*;
use crate::silk::macros::*;
use crate::silk::nlsf::*;
use crate::silk::sigm::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tables::*;

#[inline(always)]
pub fn silk_find_pred_coefs_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    res_pitch: &[i16],
    res_pitch_frame_idx: usize,
    x: &[i16],
    x_buf_full: &[i16],
    cond_coding: i32,
) {
    let mut inv_gains_q16 = [0i32; MAX_NB_SUBFR];
    let mut local_gains = [0i32; MAX_NB_SUBFR];
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut min_gain_q16: i32;
    let mut min_inv_gain_q30: i32;

    min_gain_q16 = i32::MAX >> 6;
    for i in 0..ps_enc.s_cmn.nb_subfr as usize {
        min_gain_q16 = min_gain_q16.min(ps_enc_ctrl.gains_q16[i]);
    }
    for i in 0..ps_enc.s_cmn.nb_subfr as usize {
        inv_gains_q16[i] = silk_div32_varq(min_gain_q16, ps_enc_ctrl.gains_q16[i], 16 - 2);

        inv_gains_q16[i] = inv_gains_q16[i].max(100);

        local_gains[i] = silk_div32(1 << 16, inv_gains_q16[i]);
    }

    let mut lpc_in_pre = [0i16; MAX_NB_SUBFR * MAX_LPC_ORDER + MAX_FRAME_LENGTH];

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        let mut x_xltp_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER];
        let mut xxltp_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];

        silk_find_ltp_fix(
            &mut xxltp_q17,
            &mut x_xltp_q17,
            res_pitch,
            res_pitch_frame_idx,
            &ps_enc_ctrl.pitch_l,
            ps_enc.s_cmn.subfr_length as usize,
            ps_enc.s_cmn.nb_subfr as usize,
            0,
        );

        silk_quant_ltp_gains(
            &mut ps_enc_ctrl.ltp_coef_q14,
            &mut ps_enc.s_cmn.indices.ltp_index,
            &mut ps_enc.s_cmn.indices.per_index,
            &mut ps_enc.s_cmn.sum_log_gain_q7,
            &mut ps_enc_ctrl.ltp_red_cod_gain_q7,
            &xxltp_q17,
            &x_xltp_q17,
            ps_enc.s_cmn.subfr_length,
            ps_enc.s_cmn.nb_subfr as usize,
            0,
        );

        silk_ltp_scale_ctrl_fix(ps_enc, ps_enc_ctrl, cond_coding);

        let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
        let x_base_for_ltp = ps_enc.s_cmn.ltp_mem_length as usize - predict_lpc_order;
        silk_ltp_analysis_filter_fix(
            &mut lpc_in_pre,
            x_buf_full,
            x_base_for_ltp,
            &ps_enc_ctrl.ltp_coef_q14,
            &ps_enc_ctrl.pitch_l,
            &inv_gains_q16,
            ps_enc.s_cmn.subfr_length as usize,
            ps_enc.s_cmn.nb_subfr as usize,
            ps_enc.s_cmn.predict_lpc_order as usize,
        );
    } else {
        let mut x_ptr_idx = 0;
        let mut x_pre_ptr_idx = 0;
        for i in 0..ps_enc.s_cmn.nb_subfr as usize {
            silk_scale_copy_vector16(
                &mut lpc_in_pre[x_pre_ptr_idx..],
                &x[x_ptr_idx..],
                inv_gains_q16[i],
                (ps_enc.s_cmn.subfr_length + ps_enc.s_cmn.predict_lpc_order) as usize,
            );
            x_pre_ptr_idx += (ps_enc.s_cmn.subfr_length + ps_enc.s_cmn.predict_lpc_order) as usize;
            x_ptr_idx += ps_enc.s_cmn.subfr_length as usize;
        }

        ps_enc_ctrl.ltp_coef_q14.fill(0);
        ps_enc_ctrl.ltp_red_cod_gain_q7 = 0;
        ps_enc.s_cmn.sum_log_gain_q7 = 0;
        ps_enc_ctrl.ltp_scale_q14 = 0;
    }

    if ps_enc.s_cmn.first_frame_after_reset != 0 {
        min_inv_gain_q30 =
            (1.0f32 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET * 1073741824.0f32) as i32;
    } else {
        min_inv_gain_q30 = silk_log2lin(silk_smlawb(
            16 << 7,
            ps_enc_ctrl.ltp_red_cod_gain_q7,
            (1.0f32 / 3.0f32 * 65536.0f32) as i32,
        ));
        min_inv_gain_q30 = silk_div32_varq(
            min_inv_gain_q30,
            silk_smulww(
                MAX_PREDICTION_POWER_GAIN as i32,
                silk_smlawb(
                    (0.25f32 * 262144.0f32) as i32,
                    (0.75f32 * 262144.0f32) as i32,
                    ps_enc_ctrl.coding_quality_q14,
                ),
            ),
            14,
        );
    }

    silk_find_lpc_fix(
        &mut ps_enc.s_cmn,
        &mut nlsf_q15,
        &lpc_in_pre,
        min_inv_gain_q30,
    );

    silk_process_nlsfs(ps_enc, ps_enc_ctrl, &mut nlsf_q15);

    silk_residual_energy_fix(
        &mut ps_enc_ctrl.res_nrg,
        &mut ps_enc_ctrl.res_nrg_q,
        &lpc_in_pre,
        &ps_enc_ctrl.pred_coef_q12,
        &local_gains,
        ps_enc.s_cmn.subfr_length,
        ps_enc.s_cmn.nb_subfr,
        ps_enc.s_cmn.predict_lpc_order,
    );
}

/// The noise shaping gains after the reduction for voiced frames that
/// `silk_process_gains_fix` applies first (libopus lines 51-57), and the
/// inverse squared maximum of the target SNR (lines 59-61).
fn gains_before_res_nrg(ps_enc: &SilkEncoderState, ps_enc_ctrl: &SilkEncoderControl) -> ([i32; MAX_NB_SUBFR], i32) {
    let mut gains_q16 = ps_enc_ctrl.gains_q16;
    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        let s_q15 = -silk_sigm_q15(silk_rshift_round(
            ps_enc_ctrl.ltp_red_cod_gain_q7 - (12.0 * 128.0) as i32,
            4,
        ));
        for g in gains_q16.iter_mut().take(ps_enc.s_cmn.nb_subfr as usize) {
            *g = silk_smlawb(*g, *g, s_q15);
        }
    }
    let log2lin_arg = silk_smulwb(
        ((21.0 + 16.0 / 0.33) * 128.0) as i32 - ps_enc.s_cmn.snr_db_q7,
        21627_i32,
    );
    let inv_max_sqr_val_q16 = silk_div32_16(silk_log2lin(log2lin_arg), ps_enc.s_cmn.subfr_length);
    (gains_q16, inv_max_sqr_val_q16)
}

/// Not part of libopus: the residual energy of each subframe computed from
/// the signal before its scaling, into `res_nrg_unscaled`. Runs between
/// `silk_find_pred_coefs_fix` and `silk_process_gains_fix`, with the same
/// `x` and `x_buf_full`.
///
/// `silk_find_pred_coefs_fix` computes `res_nrg` from each subframe times
/// `inv_gains_q16`, which is the smallest gain of the frame over the
/// subframe's gain, times 1/4, rounded to 16 bits. When one subframe is near
/// digital silence and another is not, the louder one can be scaled down by
/// a factor of 50 or more. The rounding error is then about as large as the
/// scaled signal, and `res_nrg` comes out several times too large. libopus's
/// float build computes its energies without that rounding. `silk_process_gains_fix`
/// uses the energies of this function when `res_nrg` gives a subframe more
/// than twice the gain that they give.
///
/// The work is skipped, and `has_res_nrg_unscaled` stays false, when no
/// subframe can be affected: when `res_nrg` gives every subframe at most 1.5
/// times its noise shaping gain. The gain from any residual energy is at
/// least the noise shaping gain, less the error of `silk_sqrt_approx`, so
/// it cannot then be more than twice the gain from `res_nrg_unscaled`. The
/// work is also skipped with the test control `fixed_point_res_nrg`.
pub fn silk_res_nrg_unscaled_fix(
    ps_enc: &SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    x: &[i16],
    x_buf_full: &[i16],
) {
    ps_enc_ctrl.has_res_nrg_unscaled = false;
    if ps_enc.fixed_point_res_nrg {
        return;
    }
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    let (gains_q16, inv_max_sqr_val_q16) = gains_before_res_nrg(ps_enc, ps_enc_ctrl);
    let may_halve = (0..nb_subfr).any(|k| {
        let gain = gain_with_res_nrg(gains_q16[k], ps_enc_ctrl.res_nrg[k], ps_enc_ctrl.res_nrg_q[k], inv_max_sqr_val_q16);
        2 * gain as i64 > 3 * gains_q16[k] as i64
    });
    if !may_halve {
        return;
    }
    let subfr_length = ps_enc.s_cmn.subfr_length as usize;
    let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
    let mut x_pre = [0i16; MAX_NB_SUBFR * MAX_LPC_ORDER + MAX_FRAME_LENGTH];
    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        // An inverse gain of 1.0 in Q16 leaves the LTP residual as it is.
        silk_ltp_analysis_filter_fix(
            &mut x_pre,
            x_buf_full,
            ps_enc.s_cmn.ltp_mem_length as usize - predict_lpc_order,
            &ps_enc_ctrl.ltp_coef_q14,
            &ps_enc_ctrl.pitch_l,
            &[1 << 16; MAX_NB_SUBFR],
            subfr_length,
            nb_subfr,
            predict_lpc_order,
        );
    } else {
        let n = subfr_length + predict_lpc_order;
        for i in 0..nb_subfr {
            let start = i * subfr_length;
            x_pre[i * n..(i + 1) * n].copy_from_slice(&x[start..start + n]);
        }
    }
    // The gains of `silk_residual_energy_fix` are in Q0: 1 leaves the
    // energies unscaled.
    silk_residual_energy_fix(
        &mut ps_enc_ctrl.res_nrg_unscaled,
        &mut ps_enc_ctrl.res_nrg_unscaled_q,
        &x_pre,
        &ps_enc_ctrl.pred_coef_q12,
        &[1; MAX_NB_SUBFR],
        ps_enc.s_cmn.subfr_length,
        ps_enc.s_cmn.nb_subfr,
        ps_enc.s_cmn.predict_lpc_order,
    );
    ps_enc_ctrl.has_res_nrg_unscaled = true;
}

/// libopus `silk_process_gains_FIX`, lines 64-84: the gain of a subframe
/// from its gain `gain_q16` of the noise shaping analysis and its residual
/// energy `res_nrg` in Q`res_nrg_q`, at the inverse squared maximum
/// `inv_max_sqr_val_q16` that the target SNR gives.
fn gain_with_res_nrg(gain_q16: i32, res_nrg: i32, res_nrg_q: i32, inv_max_sqr_val_q16: i32) -> i32 {
    let mut res_nrg_part = silk_smulww(res_nrg, inv_max_sqr_val_q16);
    if res_nrg_q > 0 {
        res_nrg_part = silk_rshift_round(res_nrg_part, res_nrg_q);
    } else {
        let neg_q = (-res_nrg_q).min(30);
        if neg_q == 0 || res_nrg_part >= (i32::MAX >> neg_q) {
            res_nrg_part = if neg_q == 0 { res_nrg_part } else { i32::MAX };
        } else {
            res_nrg_part <<= neg_q;
        }
    }
    let mut gain_squared_q16 = silk_add_sat32(res_nrg_part, silk_smmul(gain_q16, gain_q16));
    if gain_squared_q16 < i16::MAX as i32 {
        gain_squared_q16 = silk_smlaww(res_nrg_part << 16, gain_q16, gain_q16);
        let gain = silk_min_32(silk_sqrt_approx(gain_squared_q16), i32::MAX >> 8);
        silk_lshift_sat32(gain, 8)
    } else {
        let gain = silk_min_32(silk_sqrt_approx(gain_squared_q16), i32::MAX >> 16);
        silk_lshift_sat32(gain, 16)
    }
}

pub fn silk_process_gains_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    cond_coding: i32,
) {
    let (gains_q16, inv_max_sqr_val_q16) = gains_before_res_nrg(ps_enc, ps_enc_ctrl);
    let ps_shape_st = &mut ps_enc.s_shape;

    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        let gain_q16 = gains_q16[k];
        let mut gain =
            gain_with_res_nrg(gain_q16, ps_enc_ctrl.res_nrg[k], ps_enc_ctrl.res_nrg_q[k], inv_max_sqr_val_q16);
        // Not part of libopus: when the residual energy of the unscaled
        // signal gives less than half the gain, the energy above is mostly
        // the rounding error of the scaled signal (see
        // `silk_res_nrg_unscaled_fix`), and the subframe takes the smaller
        // gain. At a low SNR the residual energy decides the gain. A gain
        // that is too large plays the quantizer's offset and the subframe's
        // few pulses too loudly. This made bursts of energy below 200 Hz
        // where a stereo stream stops coding its side channel, or where a
        // signal ends in digital silence.
        if ps_enc_ctrl.has_res_nrg_unscaled {
            let unscaled = gain_with_res_nrg(
                gain_q16,
                ps_enc_ctrl.res_nrg_unscaled[k],
                ps_enc_ctrl.res_nrg_unscaled_q[k],
                inv_max_sqr_val_q16,
            );
            if gain as i64 > 2 * unscaled as i64 {
                gain = unscaled;
            }
        }
        ps_enc_ctrl.gains_q16[k] = gain;
    }

    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        ps_enc_ctrl.gains_unq_q16[k] = ps_enc_ctrl.gains_q16[k];
    }
    ps_enc_ctrl.last_gain_index_prev = ps_shape_st.last_gain_index;

    silk_gains_quant(
        &mut ps_enc.s_cmn.indices.gains_indices,
        &mut ps_enc_ctrl.gains_q16,
        &mut ps_shape_st.last_gain_index,
        if cond_coding == CODE_CONDITIONALLY {
            1
        } else {
            0
        },
        ps_enc.s_cmn.nb_subfr as usize,
    );

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        if ps_enc_ctrl.ltp_red_cod_gain_q7 + (ps_enc.s_cmn.input_tilt_q15 >> 8) > 128 {
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }

    let quant_offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_enc.s_cmn.indices.signal_type >> 1) as usize]
        [ps_enc.s_cmn.indices.quant_offset_type as usize] as i32;

    const LAMBDA_OFFSET_Q10: i32 = 1229;
    const LAMBDA_DELAYED_DECISIONS_Q10: i32 = -50;
    const LAMBDA_SPEECH_ACT_Q18: i32 = -52428;
    const LAMBDA_INPUT_QUALITY_Q12: i32 = -409;
    const LAMBDA_CODING_QUALITY_Q12: i32 = -818;
    const LAMBDA_QUANT_OFFSET_Q16: i32 = 52429;

    ps_enc_ctrl.lambda_q10 = LAMBDA_OFFSET_Q10
        + silk_smulbb(
            LAMBDA_DELAYED_DECISIONS_Q10,
            ps_enc.s_cmn.n_states_delayed_decision,
        )
        + silk_smulwb(LAMBDA_SPEECH_ACT_Q18, ps_enc.s_cmn.speech_activity_q8)
        + silk_smulwb(LAMBDA_INPUT_QUALITY_Q12, ps_enc_ctrl.input_quality_q14)
        + silk_smulwb(LAMBDA_CODING_QUALITY_Q12, ps_enc_ctrl.coding_quality_q14)
        + silk_smulwb(LAMBDA_QUANT_OFFSET_Q16, quant_offset_q10);
}

pub fn silk_ltp_scale_ctrl_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    cond_coding: i32,
) {
    if cond_coding == CODE_INDEPENDENTLY {
        let mut round_loss = ps_enc.s_cmn.packet_loss_perc * ps_enc.s_cmn.n_frames_per_packet;
        if ps_enc.s_cmn.lbrr_flag != 0 {
            round_loss = 2 + silk_smulbb(round_loss, round_loss) / 100;
        }
        ps_enc.s_cmn.indices.ltp_scale_index =
            (silk_smulbb(ps_enc_ctrl.ltp_red_cod_gain_q7, round_loss)
                > silk_log2lin(128 * 7 + 2900 - ps_enc.s_cmn.snr_db_q7)) as i8;
        ps_enc.s_cmn.indices.ltp_scale_index +=
            (silk_smulbb(ps_enc_ctrl.ltp_red_cod_gain_q7, round_loss)
                > silk_log2lin(128 * 7 + 3900 - ps_enc.s_cmn.snr_db_q7)) as i8;
    } else {
        ps_enc.s_cmn.indices.ltp_scale_index = 0;
    }
    ps_enc_ctrl.ltp_scale_q14 =
        SILK_LTP_SCALES_TABLE_Q14[ps_enc.s_cmn.indices.ltp_scale_index as usize] as i32;
}
