use crate::range_coder::RangeCoder;
use crate::silk::control_codec::{silk_control_encoder, silk_setup_lbrr};
use crate::silk::control_fixed::*;
use crate::silk::control_snr::silk_control_snr;
use crate::silk::define::*;
use crate::silk::encode_indices::*;
use crate::silk::encode_pulses::*;
use crate::silk::gain_quant::{silk_gains_dequant, silk_gains_id, silk_gains_quant};
use crate::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
use crate::silk::init_encoder::silk_init_encoder;
use crate::silk::lp_variable_cutoff::*;
use crate::silk::macros::*;
use crate::silk::noise_shape_analysis::*;
use crate::silk::nsq::*;
use crate::silk::nsq_del_dec::*;
use crate::silk::pitch_analysis::*;
use crate::silk::stereo_lr_to_ms::silk_stereo_lr_to_ms;
use crate::silk::structs::*;
use crate::silk::tuning_parameters::BITRESERVOIR_DECAY_TIME_MS;
use crate::silk::vad::silk_vad_get_sa_q8;

pub fn silk_encode_do_vad(ps_enc: &mut SilkEncoderState, input: &[i16], activity: i32) {
    let activity_threshold = SPEECH_ACTIVITY_DTX_THRES_Q8;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    silk_vad_get_sa_q8(ps_enc, input, frame_length);

    if activity == 0 && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }

    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
        if ps_enc.s_cmn.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.in_dtx = 0;
        } else if ps_enc.s_cmn.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX
        {
            ps_enc.s_cmn.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            ps_enc.s_cmn.in_dtx = 0;
        }
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 0;
    } else {
        ps_enc.s_cmn.no_speech_counter = 0;
        ps_enc.s_cmn.in_dtx = 0;
        ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 1;
    }
}

/// Internal sample rate (kHz) and 10 ms prefill length of `ps_enc`, or
/// `None` when SILK isn't configured to a supported rate.
fn prefill_geometry(ps_enc: &SilkEncoderState) -> Option<(i32, usize)> {
    let fs_khz = ps_enc.s_cmn.fs_khz;
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return None;
    }
    Some((fs_khz, fs_khz as usize * 10))
}

/// Prefill an encoder that codes one channel with 10 ms of internal-rate
/// input, as libopus's `silk_Encode` does with `prefillFlag` set
/// (enc_API.c): run the voice activity detector on the frame and push it
/// into the analysis history, coding nothing.
pub fn silk_encode_prefill(ps_enc: &mut SilkEncoderState, samples: &[i16], activity: i32) {
    let Some((_, prefill_frame_length)) = prefill_geometry(ps_enc) else {
        return;
    };
    if samples.len() < prefill_frame_length {
        return;
    }

    let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
    input_buf[0] = ps_enc.stereo.s_mid[0];
    input_buf[1] = ps_enc.stereo.s_mid[1];
    input_buf[2..2 + prefill_frame_length].copy_from_slice(&samples[..prefill_frame_length]);
    ps_enc.stereo.s_mid[0] = input_buf[prefill_frame_length];
    ps_enc.stereo.s_mid[1] = input_buf[prefill_frame_length + 1];

    // enc_API.c:456: the detector runs on the prefill frame too, so its
    // noise estimates have seen one frame when the first frame is coded.
    ps_enc.s_cmn.n_frames_encoded = 0;
    prefill_vad(ps_enc, &input_buf[1..1 + prefill_frame_length], activity);

    prefill_frame(ps_enc, &mut input_buf, prefill_frame_length);
    // enc_API.c:216, 575: the prefill is a call with one coded channel.
    ps_enc.stereo.n_channels_internal = 1;
    ps_enc.stereo.n_prev_channels_internal = 1;
}

/// Prefill a stereo encoder with 10 ms of internal-rate `left` and `right`
/// input, as libopus's `silk_Encode` does with `prefillFlag` set for two
/// internal channels: convert to mid/side (updating the stereo state, writing
/// nothing), run the VADs, and push the frame into the history of the mid
/// and, unless the frame codes the mid only, of the side, without coding it.
/// `total_rate_bps` is the SILK bitrate.
pub fn silk_encode_prefill_stereo(
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    left: &[i16],
    right: &[i16],
    total_rate_bps: i32,
    activity: i32,
) {
    let Some((fs_khz, prefill_frame_length)) = prefill_geometry(mid) else {
        return;
    };
    if left.len() < prefill_frame_length || right.len() < prefill_frame_length {
        return;
    }

    // enc_API.c:194-216: the prefill is a call with two coded channels too.
    enter_stereo(mid, side);
    // enc_API.c:222-236: libopus starts the side encoder (194-211) before
    // the prefill reset, and that reset gives every coded channel the
    // mid's low-pass state (the saved state for a rate switch, a cleared
    // one otherwise). The caller reset the channels before this call, so a
    // side that `enter_stereo` just started has lost it: copy it again.
    side.s_cmn.s_lp = mid.s_cmn.s_lp;

    let mut x1 = [0i16; MAX_FRAME_LENGTH + 2];
    let mut x2 = [0i16; MAX_FRAME_LENGTH + 2];
    x1[2..2 + prefill_frame_length].copy_from_slice(&left[..prefill_frame_length]);
    x2[2..2 + prefill_frame_length].copy_from_slice(&right[..prefill_frame_length]);
    let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
    // enc_API.c:395-423 for a prefill: the target of one 10 ms frame, with
    // no LBRR section, less the payback of the bit reservoir. A reset for a
    // rate switch keeps the reservoir.
    let n_bits = packet_target_bits(total_rate_bps, 10);
    let target_rate_bps = limit_target_rate(
        n_bits * 100 - reservoir_payback_bps(mid.n_bits_exceeded as i64),
        total_rate_bps,
    );
    // The encoders were just reset, so the previous speech activity is 0.
    silk_stereo_lr_to_ms(
        &mut mid.stereo,
        &mut x1,
        &mut x2,
        &mut ix,
        &mut mid_only,
        &mut rates,
        target_rate_bps,
        0,
        false,
        fs_khz,
        prefill_frame_length,
    );
    mid.stereo.pred_ix[0] = ix;
    mid.stereo.mid_only_flags[0] = mid_only;

    mid.s_cmn.n_frames_encoded = 0;
    side.s_cmn.n_frames_encoded = 0;
    if mid_only == 0 {
        if mid.stereo.prev_decode_only_middle == 1 {
            reset_side_for_coding(side);
        }
        prefill_vad(side, &x2[1..1 + prefill_frame_length], activity);
    } else {
        side.s_cmn.vad_flags[0] = 0;
    }
    prefill_vad(mid, &x1[1..1 + prefill_frame_length], activity);
    prefill_frame(mid, &mut x1, prefill_frame_length);
    if rates[1] > 0 {
        prefill_frame(side, &mut x2, prefill_frame_length);
    }
    mid.stereo.prev_decode_only_middle = mid_only as i32;
    mid.stereo.n_prev_channels_internal = 2;
}

/// The voice activity detector on a 10 ms prefill frame. libopus sets the
/// encoder up for 10 ms packets during a prefill (enc_API.c:237-238), and
/// the detector scales its energies and its smoothing by the frame length.
fn prefill_vad(ps_enc: &mut SilkEncoderState, frame: &[i16], activity: i32) {
    let real_frame_length = ps_enc.s_cmn.frame_length;
    ps_enc.s_cmn.frame_length = frame.len() as i32;
    silk_encode_do_vad(ps_enc, frame, activity);
    ps_enc.s_cmn.frame_length = real_frame_length;
}

/// The prefill part of libopus's `silk_encode_frame_FIX`: low-pass the 10 ms
/// frame at `input_buf[1..]` and push it into the encoder's analysis buffer,
/// coding nothing.
fn prefill_frame(
    ps_enc: &mut SilkEncoderState,
    input_buf: &mut [i16; MAX_FRAME_LENGTH + 2],
    prefill_frame_length: usize,
) {
    // encode_frame_FIX.c:120 runs before the prefill exit: the prefill frame
    // takes a seed and advances the frame counter like a coded frame. The
    // first coded frame after a restart therefore has seed 1, not 0.
    ps_enc.s_cmn.indices.seed = (ps_enc.s_cmn.frame_counter & 3) as i8;
    ps_enc.s_cmn.frame_counter = ps_enc.s_cmn.frame_counter.wrapping_add(1);

    let fs_khz = ps_enc.s_cmn.fs_khz as usize;
    let real_frame_length = ps_enc.s_cmn.frame_length as usize;
    let real_nb_subfr = ps_enc.s_cmn.nb_subfr;
    let real_subfr_length = ps_enc.s_cmn.subfr_length;

    ps_enc.s_cmn.frame_length = prefill_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = 2;
    ps_enc.s_cmn.subfr_length = (prefill_frame_length / 2) as i32;

    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape_ms_samples = 5 * fs_khz;

    silk_lp_variable_cutoff(
        &mut ps_enc.s_cmn.s_lp,
        &mut input_buf[1..],
        prefill_frame_length,
    );

    let x_frame_idx = ltp_mem_length;
    let dst = x_frame_idx + la_shape_ms_samples;

    if dst + prefill_frame_length <= ps_enc.s_cmn.x_buf.len() {
        ps_enc.s_cmn.x_buf[dst..dst + prefill_frame_length]
            .copy_from_slice(&input_buf[1..1 + prefill_frame_length]);
    }

    let move_len = ltp_mem_length + la_shape_ms_samples;
    if prefill_frame_length + move_len <= ps_enc.s_cmn.x_buf.len() {
        ps_enc
            .s_cmn
            .x_buf
            .copy_within(prefill_frame_length..prefill_frame_length + move_len, 0);
    }

    ps_enc.s_cmn.frame_length = real_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = real_nb_subfr;
    ps_enc.s_cmn.subfr_length = real_subfr_length;
}

pub fn silk_encode_frame(
    ps_enc: &mut SilkEncoderState,
    input: &[i16],
    rc: &mut RangeCoder,
    pn_bytes_out: &mut i32,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut s_enc_ctrl = SilkEncoderControl::default();

    ps_enc.s_cmn.indices.seed = (ps_enc.s_cmn.frame_counter & 3) as i8;
    ps_enc.s_cmn.frame_counter += 1;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape = ps_enc.s_cmn.la_shape as usize;

    let x_frame_idx = ltp_mem_length;

    let la_shape_max = 5 * ps_enc.s_cmn.fs_khz as usize;
    let new_samples_idx = x_frame_idx + la_shape_max;
    ps_enc.s_cmn.x_buf[new_samples_idx..new_samples_idx + frame_length]
        .copy_from_slice(&input[..frame_length]);

    let x_buf_copy = ps_enc.s_cmn.x_buf;

    let mut res_pitch = [0i16; LA_PITCH_MAX + MAX_FRAME_LENGTH + LTP_MEM_LENGTH_MS * MAX_FS_KHZ];
    let res_pitch_frame_idx = ltp_mem_length;

    silk_find_pitch_lags_fix(ps_enc, &mut s_enc_ctrl, &mut res_pitch, &x_buf_copy, 0);

    let x_tmp = &x_buf_copy[x_frame_idx - la_shape..];
    silk_noise_shape_analysis_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch[res_pitch_frame_idx..],
        x_tmp,
    );

    let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
    let x_tmp_frame = &x_buf_copy[x_frame_idx - predict_lpc_order..];
    silk_find_pred_coefs_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch,
        res_pitch_frame_idx,
        x_tmp_frame,
        &x_buf_copy,
        cond_coding,
    );

    silk_process_gains_fix(ps_enc, &mut s_enc_ctrl, cond_coding);

    /****************************************/
    /* Low Bitrate Redundant Encoding       */
    /****************************************/
    // Port of silk_LBRR_encode_FIX: runs on the frame-entry NSQ state, i.e.
    // after the gains are quantized and before the rate-control loop below.
    silk_lbrr_encode(ps_enc, &s_enc_ctrl, x_frame_idx, cond_coding);

    let max_iter = 6;
    let mut gain_mult_q8: i32 = 256;
    let mut found_lower = false;
    let mut found_upper = false;
    #[allow(unused_assignments)]
    let mut n_bits: i32 = 0;
    let mut n_bits_lower: i32 = 0;
    let mut n_bits_upper: i32 = 0;
    let mut gain_mult_lower: i32 = 0;
    let mut gain_mult_upper: i32 = 0;
    let mut gains_id: i32 =
        silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    let mut gains_id_lower: i32 = -1;
    let mut gains_id_upper: i32 = -1;

    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };

    let rc_copy = rc.clone();
    let nsq_copy = ps_enc.s_nsq;
    let seed_copy = ps_enc.s_cmn.indices.seed;
    let ec_prev_lag_index_copy = ps_enc.s_cmn.ec_prev_lag_index;
    let ec_prev_signal_type_copy = ps_enc.s_cmn.ec_prev_signal_type;
    let mut rc_copy2: Option<RangeCoder> = None;
    let mut nsq_copy2: Option<SilkNSQState> = None;
    let mut ec_buf_copy = [0u8; 1275];
    let mut last_gain_index_copy2: i8 = 0;

    let mut gain_lock = [false; MAX_NB_SUBFR];
    let mut best_gain_mult = [256i32; MAX_NB_SUBFR];
    let mut best_sum = [i32::MAX; MAX_NB_SUBFR];

    for iter in 0..=max_iter {
        if gains_id == gains_id_lower {
            n_bits = n_bits_lower;
        } else if gains_id == gains_id_upper {
            n_bits = n_bits_upper;
        } else {
            if iter > 0 {
                *rc = rc_copy.clone();
                ps_enc.s_nsq = nsq_copy;
                ps_enc.s_cmn.indices.seed = seed_copy;
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
            }

            let mut pred_coef_q12_flat = [0i16; 2 * MAX_LPC_ORDER];
            pred_coef_q12_flat[..MAX_LPC_ORDER].copy_from_slice(&s_enc_ctrl.pred_coef_q12[0]);
            pred_coef_q12_flat[MAX_LPC_ORDER..].copy_from_slice(&s_enc_ctrl.pred_coef_q12[1]);

            // C encode_frame_FIX.c 207: the delayed-decision quantizer also
            // runs with one state when warping is on. The plain quantizer
            // has no warped noise shaping.
            if ps_enc.s_cmn.n_states_delayed_decision > 1 || ps_enc.s_cmn.warping_q16 > 0 {
                let winner_seed = silk_nsq_del_dec(
                    &ps_enc.s_cmn,
                    &mut ps_enc.s_nsq,
                    &ps_enc.s_cmn.indices,
                    &ps_enc.s_cmn.x_buf[x_frame_idx..],
                    &mut ps_enc.pulses,
                    &pred_coef_q12_flat,
                    &s_enc_ctrl.ltp_coef_q14,
                    &s_enc_ctrl.ar_q13,
                    &s_enc_ctrl.harm_shape_gain_q14,
                    &s_enc_ctrl.tilt_q14,
                    &s_enc_ctrl.lf_shp_q14,
                    &s_enc_ctrl.gains_q16,
                    &s_enc_ctrl.pitch_l,
                    s_enc_ctrl.lambda_q10,
                    s_enc_ctrl.ltp_scale_q14,
                );

                ps_enc.s_cmn.indices.seed = winner_seed as i8;
            } else {
                silk_nsq(
                    &ps_enc.s_cmn,
                    &mut ps_enc.s_nsq,
                    &ps_enc.s_cmn.indices,
                    &ps_enc.s_cmn.x_buf[x_frame_idx..],
                    &mut ps_enc.pulses,
                    &pred_coef_q12_flat,
                    &s_enc_ctrl.ltp_coef_q14,
                    &s_enc_ctrl.ar_q13,
                    &s_enc_ctrl.harm_shape_gain_q14,
                    &s_enc_ctrl.tilt_q14,
                    &s_enc_ctrl.lf_shp_q14,
                    &s_enc_ctrl.gains_q16,
                    &s_enc_ctrl.pitch_l,
                    s_enc_ctrl.lambda_q10,
                    s_enc_ctrl.ltp_scale_q14,
                );
            }

            if iter == max_iter && !found_lower {
                rc_copy2 = Some(rc.clone());
            }

            silk_encode_indices(
                ps_enc,
                rc,
                ps_enc.s_cmn.n_frames_encoded as usize,
                false,
                cond_coding,
            );

            silk_encode_pulses(
                rc,
                ps_enc.s_cmn.indices.signal_type as i32,
                ps_enc.s_cmn.indices.quant_offset_type as i32,
                &mut ps_enc.pulses,
                ps_enc.s_cmn.frame_length as usize,
            );

            n_bits = rc.tell();

            if iter == max_iter && !found_lower && n_bits > max_bits {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                }

                ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                    ps_enc.s_cmn.indices.gains_indices[i] = 4;
                }
                if cond_coding != CODE_CONDITIONALLY {
                    ps_enc.s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev;
                }
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;

                ps_enc.pulses.fill(0);

                silk_encode_indices(
                    ps_enc,
                    rc,
                    ps_enc.s_cmn.n_frames_encoded as usize,
                    false,
                    cond_coding,
                );
                silk_encode_pulses(
                    rc,
                    ps_enc.s_cmn.indices.signal_type as i32,
                    ps_enc.s_cmn.indices.quant_offset_type as i32,
                    &mut ps_enc.pulses,
                    ps_enc.s_cmn.frame_length as usize,
                );

                n_bits = rc.tell();
            }

            if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                break;
            }
        }

        if iter == max_iter {
            if found_lower && (gains_id == gains_id_lower || n_bits > max_bits) {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                    let offs = rc.offs as usize;
                    rc.buf[..offs].copy_from_slice(&ec_buf_copy[..offs]);
                }
                if let Some(nsq_c2) = &nsq_copy2 {
                    ps_enc.s_nsq = *nsq_c2;
                }
                ps_enc.s_shape.last_gain_index = last_gain_index_copy2;
            }
            break;
        }

        if n_bits > max_bits {
            if !found_lower && iter >= 2 {
                s_enc_ctrl.lambda_q10 =
                    silk_add_rshift32(s_enc_ctrl.lambda_q10, s_enc_ctrl.lambda_q10, 1);
                found_upper = false;
                gains_id_upper = -1;
            } else {
                found_upper = true;
                n_bits_upper = n_bits;
                gain_mult_upper = gain_mult_q8;
                gains_id_upper = gains_id;
            }
        } else if n_bits < max_bits - bits_margin {
            found_lower = true;
            n_bits_lower = n_bits;
            gain_mult_lower = gain_mult_q8;
            if gains_id != gains_id_lower {
                gains_id_lower = gains_id;

                rc_copy2 = Some(rc.clone());
                let offs = rc.offs as usize;
                ec_buf_copy[..offs].copy_from_slice(&rc.buf[..offs]);
                nsq_copy2 = Some(ps_enc.s_nsq);
                last_gain_index_copy2 = ps_enc.s_shape.last_gain_index;
            }
        } else {
            break;
        }

        if !found_lower && n_bits > max_bits {
            let subfr_length = ps_enc.s_cmn.subfr_length as usize;
            for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                let mut sum: i32 = 0;
                for j in (i * subfr_length)..((i + 1) * subfr_length) {
                    sum += ps_enc.pulses[j].abs() as i32;
                }
                if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                    best_sum[i] = sum;
                    best_gain_mult[i] = gain_mult_q8;
                } else {
                    gain_lock[i] = true;
                }
            }
        }

        if !(found_lower && found_upper) {
            if n_bits > max_bits {
                gain_mult_q8 = silk_min_32(1024, (gain_mult_q8 * 3) / 2);
            } else {
                gain_mult_q8 = silk_max_32(64, (gain_mult_q8 * 4) / 5);
            }
        } else {
            let delta = gain_mult_upper - gain_mult_lower;
            gain_mult_q8 = gain_mult_lower
                + silk_div32_16(
                    (gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower),
                    n_bits_upper - n_bits_lower,
                );

            let lower_limit = silk_add_rshift32(gain_mult_lower, delta, 2);
            let upper_limit = silk_sub_rshift32(gain_mult_upper, delta, 2);
            if gain_mult_q8 > lower_limit {
                gain_mult_q8 = lower_limit;
            } else if gain_mult_q8 < upper_limit {
                gain_mult_q8 = upper_limit;
            }
        }

        for i in 0..ps_enc.s_cmn.nb_subfr as usize {
            let tmp = if gain_lock[i] {
                best_gain_mult[i]
            } else {
                gain_mult_q8
            };
            s_enc_ctrl.gains_q16[i] =
                silk_lshift_sat32(silk_smulwb(s_enc_ctrl.gains_unq_q16[i], tmp), 8);
        }

        ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
        silk_gains_quant(
            &mut ps_enc.s_cmn.indices.gains_indices,
            &mut s_enc_ctrl.gains_q16,
            &mut ps_enc.s_shape.last_gain_index,
            if cond_coding == CODE_CONDITIONALLY {
                1
            } else {
                0
            },
            ps_enc.s_cmn.nb_subfr as usize,
        );

        gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    }

    let move_len = ltp_mem_length + 5 * ps_enc.s_cmn.fs_khz as usize;
    ps_enc
        .s_cmn
        .x_buf
        .copy_within(frame_length..frame_length + move_len, 0);

    ps_enc.s_cmn.prev_lag = s_enc_ctrl.pitch_l[ps_enc.s_cmn.nb_subfr as usize - 1];
    ps_enc.s_cmn.prev_signal_type = ps_enc.s_cmn.indices.signal_type as i32;
    ps_enc.s_cmn.first_frame_after_reset = 0;

    *pn_bytes_out = (rc.tell() + 7) >> 3;

    0
}

/// Low-Bitrate Redundancy (LBRR) encoding: reuse all parameters but encode
/// the excitation at a lower rate. Port of `silk_LBRR_encode_FIX`
/// (encode_frame_FIX.c:391-448), called after the gains are quantized and
/// before the rate-control loop, so the NSQ state is the frame-entry state.
///
/// On success this frame's LBRR payload lives in
/// `s_cmn.indices_lbrr[fi]` / `s_cmn.pulses_lbrr[fi]` and
/// `s_cmn.lbrr_flags[fi]` is set for the *next* packet's LBRR section.
fn silk_lbrr_encode(
    ps_enc: &mut SilkEncoderState,
    s_enc_ctrl: &SilkEncoderControl,
    x16_offset: usize,
    cond_coding: i32,
) {
    // Control use of inband LBRR. LBRR_SPEECH_ACTIVITY_THRES = 0.3
    // (SILK_FIX_CONST(0.3, 8) = 77).
    if ps_enc.s_cmn.lbrr_enabled == 0
        || ps_enc.s_cmn.speech_activity_q8 <= 77
    {
        return;
    }

    let fi = ps_enc.s_cmn.n_frames_encoded as usize;
    if fi >= MAX_FRAMES_PER_PACKET {
        return;
    }
    ps_enc.s_cmn.lbrr_flags[fi] = 1;

    // Copy noise shaping quantizer state and quantization indices from the
    // regular encoding.
    let mut nsq_lbrr = ps_enc.s_nsq;
    let mut indices_lbrr = ps_enc.s_cmn.indices;

    if fi == 0 || ps_enc.s_cmn.lbrr_flags[fi - 1] == 0 {
        // First frame in packet or previous frame not LBRR coded.
        ps_enc.s_cmn.lbrr_prev_last_gain_index = ps_enc.s_shape.last_gain_index;

        // Increase gains to get the target LBRR rate. This index is coded
        // absolutely here (first frame of an LBRR run), so it must stay
        // within 0..=N_LEVELS_QGAIN-1.
        indices_lbrr.gains_indices[0] = (indices_lbrr.gains_indices[0] as i32
            + ps_enc.s_cmn.lbrr_gain_increases)
            .min(N_LEVELS_QGAIN - 1) as i8;
    }

    // Decode to get gains in sync with the decoder. Works on a scratch copy
    // of the gains; the main frame's gains are left untouched.
    let mut gains_lbrr = s_enc_ctrl.gains_q16;
    silk_gains_dequant(
        &mut gains_lbrr,
        &indices_lbrr.gains_indices,
        &mut ps_enc.s_cmn.lbrr_prev_last_gain_index,
        if cond_coding == CODE_CONDITIONALLY { 1 } else { 0 },
        ps_enc.s_cmn.nb_subfr as usize,
    );

    let mut pred_coef_q12_flat = [0i16; 2 * MAX_LPC_ORDER];
    pred_coef_q12_flat[..MAX_LPC_ORDER].copy_from_slice(&s_enc_ctrl.pred_coef_q12[0]);
    pred_coef_q12_flat[MAX_LPC_ORDER..].copy_from_slice(&s_enc_ctrl.pred_coef_q12[1]);

    /*****************************************/
    /* Noise shaping quantization            */
    /*****************************************/
    let mut pulses_lbrr = [0i8; MAX_FRAME_LENGTH];
    if ps_enc.s_cmn.n_states_delayed_decision > 1 || ps_enc.s_cmn.warping_q16 > 0 {
        let winner_seed = silk_nsq_del_dec(
            &ps_enc.s_cmn,
            &mut nsq_lbrr,
            &indices_lbrr,
            &ps_enc.s_cmn.x_buf[x16_offset..],
            &mut pulses_lbrr,
            &pred_coef_q12_flat,
            &s_enc_ctrl.ltp_coef_q14,
            &s_enc_ctrl.ar_q13,
            &s_enc_ctrl.harm_shape_gain_q14,
            &s_enc_ctrl.tilt_q14,
            &s_enc_ctrl.lf_shp_q14,
            &gains_lbrr,
            &s_enc_ctrl.pitch_l,
            s_enc_ctrl.lambda_q10,
            s_enc_ctrl.ltp_scale_q14,
        );
        // The delayed-decision quantizer picks the seed of its winning
        // path (C NSQ_del_dec.c: `psIndices->Seed = psDD->SeedInit`). The
        // decoder derives the excitation signs from it, so the LBRR frame
        // must carry this seed, not the main frame's.
        indices_lbrr.seed = winner_seed as i8;
    } else {
        silk_nsq(
            &ps_enc.s_cmn,
            &mut nsq_lbrr,
            &indices_lbrr,
            &ps_enc.s_cmn.x_buf[x16_offset..],
            &mut pulses_lbrr,
            &pred_coef_q12_flat,
            &s_enc_ctrl.ltp_coef_q14,
            &s_enc_ctrl.ar_q13,
            &s_enc_ctrl.harm_shape_gain_q14,
            &s_enc_ctrl.tilt_q14,
            &s_enc_ctrl.lf_shp_q14,
            &gains_lbrr,
            &s_enc_ctrl.pitch_l,
            s_enc_ctrl.lambda_q10,
            s_enc_ctrl.ltp_scale_q14,
        );
    }

    ps_enc.s_cmn.indices_lbrr[fi] = indices_lbrr;
    ps_enc.s_cmn.pulses_lbrr[fi] = pulses_lbrr;
}

/// Encode the LBRR (in-band FEC) section of a packet with one coded channel
/// and return the number of bits written. Matches libopus enc_API.c:365-389:
/// an LBRR symbol for multi-frame packets, then per frame the indices and
/// the pulses. [`encode_lbrr_section_stereo`] is the same for two channels.
fn encode_lbrr_section(
    rc: &mut RangeCoder,
    ps_enc: &mut SilkEncoderState,
    lbrr_symbol: i32,
    n_frames_per_packet: i32,
) -> i32 {
    let start_bits = rc.tell();

    if n_frames_per_packet > 1 {
        rc.encode_icdf(lbrr_symbol - 1, lbrr_flags_icdf(n_frames_per_packet), 8);
    }

    for i in 0..n_frames_per_packet as usize {
        if ps_enc.s_cmn.lbrr_flags[i] != 0 {
            let lbrr_cond = if i > 0 && ps_enc.s_cmn.lbrr_flags[i - 1] != 0 {
                CODE_CONDITIONALLY
            } else {
                CODE_INDEPENDENTLY
            };
            silk_encode_indices(ps_enc, rc, i, true, lbrr_cond);
            silk_encode_pulses(
                rc,
                ps_enc.s_cmn.indices_lbrr[i].signal_type as i32,
                ps_enc.s_cmn.indices_lbrr[i].quant_offset_type as i32,
                &mut ps_enc.s_cmn.pulses_lbrr[i],
                ps_enc.s_cmn.frame_length as usize,
            );
        }
    }

    rc.tell() - start_bits
}

/// Bits frame `frame_idx` of a `tot_blocks`-frame packet may fill the packet
/// up to (libopus enc_API.c). Capping the earlier frames keeps a share of the
/// packet for the later ones; otherwise a 60 ms packet's first two frames can
/// leave the last one less than even its no-pulse fallback costs, and the
/// packet overflows.
fn silk_frame_max_bits(max_bits: i32, tot_blocks: i32, frame_idx: i32) -> i32 {
    // In 64 bits, so that no `max_bits` overflows the product.
    let share = |num: i64, den: i64| (max_bits as i64 * num / den) as i32;
    match (tot_blocks, frame_idx) {
        (2, 0) => share(3, 5),
        (3, 0) => share(2, 5),
        (3, 1) => share(3, 4),
        _ => max_bits,
    }
}

/// Number of 20 ms blocks (or one 10 ms block) in a packet of
/// `packet_length` internal-rate samples: libopus enc_API.c `tot_blocks`,
/// from `nBlocksOf10ms = 100 * nSamplesIn / API_sampleRate`.
fn packet_blocks(packet_length: usize, fs_khz: i32) -> i32 {
    let n_blocks_of_10ms = packet_length / (fs_khz.max(1) as usize * 10);
    if n_blocks_of_10ms > 1 {
        (n_blocks_of_10ms >> 1).min(MAX_FRAMES_PER_PACKET) as i32
    } else {
        1
    }
}

/// The target rate of the packet's next frame, in bits per second
/// (enc_API.c:398-432): its share of `bit_rate`, less the packet's LBRR
/// section, less a 500 ms payback of the bits coded beyond the target, both
/// in earlier packets (`n_bits_exceeded`) and in this packet's earlier frames
/// (`bits_so_far`, the range coder's `tell`).
///
/// `lbrr_bits` is the size of the packet's LBRR section for its first frame
/// and 0 for the others, as libopus's `curr_nBitsUsedLBRR` is: a later frame
/// drops the LBRR average to 0, and the section then counts against it as
/// overshoot.
fn silk_frame_target_rate_bps(
    ps_enc: &mut SilkEncoderState,
    bit_rate: i32,
    lbrr_bits: i32,
    bits_so_far: i32,
) -> i32 {
    let packet_size_ms = ps_enc.s_cmn.packet_size_ms;
    // A moving average of the LBRR usage, except that the first packet with
    // LBRR isn't averaged and a packet without LBRR drops it to 0 at once.
    ps_enc.n_bits_used_lbrr = if lbrr_bits < 10 {
        0
    } else if ps_enc.n_bits_used_lbrr < 10 {
        lbrr_bits
    } else {
        (ps_enc.n_bits_used_lbrr + lbrr_bits) / 2
    };
    // The products are taken in 64 bits: libopus's 32-bit `silk_MUL` wraps
    // for a bitrate far beyond any that Opus allows, and that must not
    // overflow here. Every rate in use gives the same result.
    let n_bits = (packet_target_bits(bit_rate, packet_size_ms) - ps_enc.n_bits_used_lbrr as i64)
        / ps_enc.s_cmn.n_frames_per_packet as i64;
    let mut target = n_bits * if packet_size_ms == 10 { 100 } else { 50 };
    target -= reservoir_payback_bps(ps_enc.n_bits_exceeded as i64);
    let n_frames_encoded = ps_enc.s_cmn.n_frames_encoded as i64;
    if n_frames_encoded > 0 {
        let bits_balance =
            bits_so_far as i64 - ps_enc.n_bits_used_lbrr as i64 - n_bits * n_frames_encoded;
        target -= reservoir_payback_bps(bits_balance);
    }
    limit_target_rate(target, bit_rate)
}

/// The bits of one packet at `bit_rate` (enc_API.c:395):
/// `silk_DIV32_16( silk_MUL( bitRate, payloadSize_ms ), 1000 )`.
fn packet_target_bits(bit_rate: i32, packet_size_ms: i32) -> i64 {
    bit_rate as i64 * packet_size_ms as i64 / 1000
}

/// What a frame's target gives back of `bits` coded beyond the target, in
/// bits per second (enc_API.c:417, 421): the excess is paid back over
/// `BITRESERVOIR_DECAY_TIME_MS`.
fn reservoir_payback_bps(bits: i64) -> i64 {
    bits * 1000 / BITRESERVOIR_DECAY_TIME_MS as i64
}

/// "Never exceed input bitrate" (enc_API.c:424):
/// `silk_LIMIT( TargetRate_bps, bitRate, 5000 )`. Like silk_LIMIT, the
/// bounds swap when `bit_rate` is below 5000.
///
/// The result also stays within `TARGET_RATE_LIMIT_BPS`, so that the sums
/// and products that the next stages form with it cannot overflow.
fn limit_target_rate(target: i64, bit_rate: i32) -> i32 {
    target
        .clamp(bit_rate.min(5000) as i64, bit_rate.max(5000) as i64)
        .clamp(-TARGET_RATE_LIMIT_BPS, TARGET_RATE_LIMIT_BPS) as i32
}

/// 16 Mb/s. SILK's tables of target rates end near 80 kb/s, and Opus allows
/// no bitrate above 510 kb/s per channel, so no rate in use comes near it.
const TARGET_RATE_LIMIT_BPS: i64 = 1 << 24;

/// Add the packet's overshoot of `bit_rate` to the bit reservoir that later
/// frame targets pay back (enc_API.c:544-546).
fn silk_update_bit_reservoir(ps_enc: &mut SilkEncoderState, bit_rate: i32, n_bytes_out: i32) {
    let target_bits = packet_target_bits(bit_rate, ps_enc.s_cmn.packet_size_ms);
    ps_enc.n_bits_exceeded =
        (ps_enc.n_bits_exceeded as i64 + n_bytes_out as i64 * 8 - target_bits).clamp(0, 10000) as i32;
}

/// iCDF of the per-frame LBRR flags of a 40 or 60 ms packet.
fn lbrr_flags_icdf(n_frames_per_packet: i32) -> &'static [u8] {
    match n_frames_per_packet {
        3 => &crate::silk::tables::SILK_LBRR_FLAGS_3_ICDF[..],
        _ => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
    }
}

/// Encode one packet with one coded channel, the way libopus's `silk_Encode`
/// does for `nChannelsInternal == 1` (enc_API.c).
///
/// `samples_in` holds the packet's `n_samples_in` samples at the internal
/// rate. This is the path for mono input, and for stereo input that is
/// coded as mono: the caller then passes the downmix, see
/// [`silk_downmix_stereo`] and [`silk_average_first_mono_frame`].
///
/// Returns 0, or a negative SILK error code when the encoder is not set up
/// or the input is shorter than the packet.
pub fn silk_encode(
    ps_enc: &mut SilkEncoderState,
    samples_in: &[i16],
    n_samples_in: usize,
    rc: &mut RangeCoder,
    n_bytes_out: &mut i32,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: i32,
    activity: i32,
) -> i32 {
    let n_frames_per_packet = ps_enc.s_cmn.n_frames_per_packet;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let Some((fs_khz, _)) = prefill_geometry(ps_enc) else {
        return SILK_ENC_FS_NOT_SUPPORTED;
    };
    let packet_length = n_frames_per_packet.max(0) as usize * frame_length;
    // A frame skipped for want of input would leave its flags in the packet
    // header undefined, so the whole packet must be there.
    if frame_length == 0
        || frame_length > MAX_FRAME_LENGTH
        || n_frames_per_packet < 1
        || n_frames_per_packet as usize > MAX_FRAMES_PER_PACKET
        || n_samples_in < packet_length
        || samples_in.len() < packet_length
    {
        return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
    }

    // enc_API.c:212-216: a change of the number of coded channels is a
    // transition, which drops the LBRR frames kept for this packet
    // (enc_API.c:262-267), as a reset does. The side channel keeps its
    // state; libopus resets it when two channels are coded again.
    let transition = ps_enc.stereo.n_channels_internal != 1;
    ps_enc.stereo.n_channels_internal = 1;

    ps_enc.s_cmn.n_frames_encoded = 0;

    // libopus refuses input longer than a packet (enc_API.c:245-249). Here
    // the packet is coded and the rest is left, so the blocks are those of
    // the packet.
    let tot_blocks = packet_blocks(packet_length, fs_khz);
    // A limit below zero acts like a limit of zero: every size compared
    // with it is at least one bit.
    let max_bits = max_bits.max(0);

    if ps_enc.s_cmn.first_frame_after_reset != 0 || transition {
        ps_enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
    }

    // enc_API.c:365-375: the LBRR frames that the previous packet's frames
    // left are written whenever there are any. Whether FEC is still on for
    // this packet does not matter: it only decides whether this packet's
    // frames leave LBRR frames for the next one.
    let mut lbrr_symbol: i32 = 0;
    for i in 0..n_frames_per_packet as usize {
        lbrr_symbol |= ps_enc.s_cmn.lbrr_flags[i] << i;
    }
    ps_enc.s_cmn.lbrr_flag = if lbrr_symbol > 0 { 1 } else { 0 };

    // One channel's VAD flags and its LBRR flag, patched in at the end.
    let n_flag_bits = (n_frames_per_packet + 1) as u32;

    // Bits of the LBRR section written at the start of the packet, which come
    // out of the frame target (libopus `curr_nBitsUsedLBRR`).
    let mut lbrr_bits = 0i32;

    for frame_idx in 0..n_frames_per_packet {
        // enc_API.c:451-455, "Buffering": the frame behind the last two
        // samples of the previous one. The API-rate -> internal-rate
        // conversion is done by the caller's encoder resampler.
        let offset = frame_idx as usize * frame_length;
        let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
        input_buf[..2].copy_from_slice(&ps_enc.stereo.s_mid);
        input_buf[2..2 + frame_length].copy_from_slice(&samples_in[offset..offset + frame_length]);
        ps_enc
            .stereo
            .s_mid
            .copy_from_slice(&input_buf[frame_length..frame_length + 2]);

        if frame_idx == 0 {
            let icdf = [(256i32 - (256i32 >> n_flag_bits)) as u8, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            if lbrr_symbol > 0 {
                // libopus writes the section whatever its size. A packet
                // too small for it and the frame overflows the range coder,
                // and the caller then sends a frame for the decoder to
                // conceal (opus_encoder.c:2582-2594).
                lbrr_bits = encode_lbrr_section(rc, ps_enc, lbrr_symbol, n_frames_per_packet);
            }

            // Reset the LBRR flags for the next packet (enc_API.c:379-382);
            // this packet's frames set them again from inside
            // silk_lbrr_encode.
            ps_enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }

        // enc_API.c:392: once per frame, from the pitch lag and the voice
        // activity of the frame before.
        silk_hp_variable_cutoff(&mut ps_enc.s_cmn);

        let frame_rate_bps = silk_frame_target_rate_bps(
            ps_enc,
            target_rate_bps,
            if frame_idx == 0 { lbrr_bits } else { 0 },
            rc.tell(),
        );
        silk_control_snr(&mut ps_enc.s_cmn, frame_rate_bps);

        silk_encode_do_vad(ps_enc, &input_buf[1..1 + frame_length], activity);

        silk_lp_variable_cutoff(&mut ps_enc.s_cmn.s_lp, &mut input_buf[1..], frame_length);

        let cond_coding = if frame_idx == 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        let mut frame_bytes = 0i32;
        let ret = silk_encode_frame(
            ps_enc,
            &input_buf[1..1 + frame_length],
            rc,
            &mut frame_bytes,
            cond_coding,
            // A cap on the packet's bits so far, the LBRR section included,
            // since the rate control measures the range coder's `tell`.
            silk_frame_max_bits(max_bits, tot_blocks, frame_idx),
            if use_cbr != 0 && frame_idx == n_frames_per_packet - 1 {
                1
            } else {
                0
            },
        );
        if ret != 0 {
            return ret;
        }

        ps_enc.s_cmn.n_frames_encoded += 1;
        // enc_API.c:521. libopus runs this line for one coded channel too,
        // with the flags that the last two-channel packet left.
        ps_enc.stereo.prev_decode_only_middle =
            ps_enc.stereo.mid_only_flags[frame_idx as usize] as i32;
    }

    let mut flags = 0u32;
    for i in 0..n_frames_per_packet as usize {
        flags <<= 1;
        flags |= ps_enc.s_cmn.vad_flags[i] as u32;
    }
    flags <<= 1;
    flags |= ps_enc.s_cmn.lbrr_flag as u32;

    rc.patch_initial_bits(flags, n_flag_bits);

    *n_bytes_out = (rc.tell() + 7) >> 3;
    silk_update_bit_reservoir(ps_enc, target_rate_bps, *n_bytes_out);
    ps_enc.stereo.n_prev_channels_internal = 1;

    0
}

pub use crate::silk::control_codec::SILK_ENC_FS_NOT_SUPPORTED;

/// Mix interleaved `stereo` input down to `mono`, one sample per pair, for
/// coding it as one channel. Port of enc_API.c:310-313, "Combine left and
/// right channels before resampling": `silk_RSHIFT_ROUND( left + right, 1 )`.
///
/// This is the fixed-point build. The float build of libopus converts the
/// sum of the two float samples to 16 bits first, which clips a sum beyond
/// 16 bits before it is halved.
pub fn silk_downmix_stereo(stereo: &[i16], mono: &mut [i16]) {
    let n = mono.len().min(stereo.len() / 2);
    for i in 0..n {
        mono[i] = silk_rshift_round(stereo[2 * i] as i32 + stereo[2 * i + 1] as i32, 1) as i16;
    }
}

/// Number of channels that the packet before the next one coded: libopus
/// `nPrevChannelsInternal`, 0 before the first packet. `mid` is the encoder
/// of the mono or mid channel.
///
/// The caller owns the input resamplers and needs this for the two steps
/// that libopus's `silk_Encode` takes with them when the count changes:
///
/// - 1, and the next packet codes two channels: copy the first channel's
///   resampler to the second before resampling, so that both start from the
///   same state (enc_API.c:296-299).
/// - 2, and the next packet codes one channel: after resampling the downmix
///   with the first channel's resampler, call
///   [`silk_average_first_mono_frame`] for the first frame.
pub fn silk_prev_channels_internal(mid: &SilkEncoderState) -> i32 {
    mid.stereo.n_prev_channels_internal
}

/// The first frame coded as one channel after packets that coded two: "On
/// the first mono frame, average the results for the two resampler states"
/// (enc_API.c:316-331).
///
/// The first channel's resampler holds the end of the left input and the
/// second channel's that of the right input. Each alone would continue the
/// downmix from the wrong history. Resampling is linear, so the average of
/// the two outputs continues it from the history of the downmix.
///
/// `mono_frame` is the first frame of the packet at the internal rate, as
/// the first channel's resampler made it from `mono_api`, the same frame of
/// the downmix at the input rate. `side_resampler` is the second channel's
/// resampler. Returns the resampler's error code.
pub fn silk_average_first_mono_frame(
    side_resampler: &mut crate::silk::resampler::SilkResampler,
    mono_frame: &mut [i16],
    mono_api: &[i16],
) -> i32 {
    let mut other = [0i16; MAX_FRAME_LENGTH];
    let n = mono_frame.len().min(MAX_FRAME_LENGTH);
    let ret = side_resampler.process(&mut other[..n], mono_api, mono_api.len() as i32);
    for (a, &b) in mono_frame.iter_mut().zip(&other[..n]) {
        *a = silk_rshift(*a as i32 + b as i32, 1) as i16;
    }
    ret
}

/// The stereo width that the caller applies to the high band of a Hybrid
/// frame: libopus `encControl->stereoWidth_Q14` (enc_API.c:584), which is 0
/// for the packet that leads over to one coded channel.
pub fn silk_stereo_width_q14(mid: &SilkEncoderState, to_mono: bool) -> i32 {
    if to_mono {
        0
    } else {
        mid.stereo.smth_width_q14 as i32
    }
}

/// Start coding two channels: "Mono -> Stereo transition: init state of
/// second channel and stereo state" (enc_API.c:194-211). It runs for the
/// first packet of a stereo stream and after packets that coded one channel.
///
/// libopus then sets up the second channel with the packet's control
/// structure, which holds the same values as for the first channel
/// (enc_API.c:253-268). Here the settings are copied from the first.
fn silk_mono_to_stereo(mid: &mut SilkEncoderState, side: &mut SilkEncoderState) -> i32 {
    let mut ret = silk_init_encoder(side, 0);
    mid.stereo.pred_prev_q13 = [0; 2];
    mid.stereo.s_side = [0; 2];
    mid.stereo.mid_side_amp_q0 = [0, 1, 0, 1];
    mid.stereo.width_prev_q14 = 0;
    mid.stereo.smth_width_q14 = 1 << 14;
    // enc_API.c:205-208, for stereo input. The copy of the resampler state
    // is the caller's, see `silk_prev_channels_internal`.
    side.s_cmn.in_hp_state = mid.s_cmn.in_hp_state;

    ret += silk_control_encoder(
        side,
        mid.s_cmn.fs_khz,
        mid.s_cmn.packet_size_ms,
        mid.s_cmn.target_rate_bps,
        mid.s_cmn.complexity,
    );
    side.s_cmn.use_cbr = mid.s_cmn.use_cbr;
    side.s_cmn.use_dtx = mid.s_cmn.use_dtx;
    side.s_cmn.n_channels = 2;
    side.s_cmn.use_in_band_fec = mid.s_cmn.use_in_band_fec;
    side.s_cmn.packet_loss_perc = mid.s_cmn.packet_loss_perc;
    ret += silk_setup_lbrr(side, mid.s_cmn.lbrr_enabled);
    ret
}

/// The start of a call that codes two channels (enc_API.c:194-216): set up
/// the second channel when the call before coded one. Returns whether the
/// number of coded channels changed.
fn enter_stereo(mid: &mut SilkEncoderState, side: &mut SilkEncoderState) -> bool {
    let transition = mid.stereo.n_channels_internal != 2;
    if mid.stereo.n_channels_internal < 2 {
        silk_mono_to_stereo(mid, side);
    }
    mid.stereo.n_channels_internal = 2;
    transition
}

/// Encode one packet of stereo input as SILK mid/side, the way libopus's
/// `silk_Encode` does for two internal channels (enc_API.c).
///
/// `left` and `right` hold `n_samples_in` samples each, at the internal
/// rate, de-interleaved. Each frame is converted to mid/side by
/// [`silk_stereo_lr_to_ms`], which also picks the predictors, the stereo
/// width and the split of the frame's bits between mid and side. `mid` codes
/// the mid and holds the stereo state; `side` codes the side residual, except
/// in frames that code the mid only (panned mono).
///
/// This is [`silk_encode_stereo_packet_to_mono`] without the lead-over to
/// one coded channel.
pub fn silk_encode_stereo_packet(
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    left: &[i16],
    right: &[i16],
    n_samples_in: usize,
    rc: &mut RangeCoder,
    n_bytes_out: &mut i32,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: i32,
    activity: i32,
) -> i32 {
    silk_encode_stereo_packet_to_mono(
        mid,
        side,
        left,
        right,
        n_samples_in,
        rc,
        n_bytes_out,
        target_rate_bps,
        max_bits,
        use_cbr,
        activity,
        false,
    )
}

/// [`silk_encode_stereo_packet`] with libopus's `toMono` flag.
///
/// Stereo input is coded as one channel in three steps
/// (opus_encoder.c:1560-1571, enc_API.c):
///
/// 1. The last packet with two channels is coded with `to_mono` set. Its
///    stereo predictors are zero and its stereo width falls to zero over the
///    first 8 ms, so the packet ends as mid alone.
/// 2. The following packets are coded by [`silk_encode`] from the downmix.
///    The mid encoder goes on; the side encoder keeps its state, unused.
/// 3. The first packet that codes two channels again goes through this
///    function, which sets up the side encoder and the stereo state anew.
///
/// The caller owns the resamplers; [`silk_prev_channels_internal`] says
/// what to do with them in steps 2 and 3.
pub fn silk_encode_stereo_packet_to_mono(
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    left: &[i16],
    right: &[i16],
    n_samples_in: usize,
    rc: &mut RangeCoder,
    n_bytes_out: &mut i32,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: i32,
    activity: i32,
    to_mono: bool,
) -> i32 {
    let Some((fs_khz, _)) = prefill_geometry(mid) else {
        return SILK_ENC_FS_NOT_SUPPORTED;
    };
    // enc_API.c:194-216. A packet that follows packets with one coded
    // channel starts the side encoder and the stereo state afresh.
    let transition = enter_stereo(mid, side);
    let n_frames_per_packet = mid.s_cmn.n_frames_per_packet;
    let frame_length = mid.s_cmn.frame_length as usize;
    let packet_length = n_frames_per_packet.max(0) as usize * frame_length;
    // A frame skipped for want of input would desynchronize the stereo
    // header, so the whole packet must be there.
    if frame_length == 0
        || frame_length > MAX_FRAME_LENGTH
        || n_frames_per_packet < 1
        || n_frames_per_packet as usize > MAX_FRAMES_PER_PACKET
        || side.s_cmn.fs_khz != fs_khz
        || side.s_cmn.frame_length as usize != frame_length
        || n_samples_in < packet_length
        || left.len() < packet_length
        || right.len() < packet_length
    {
        return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
    }

    mid.s_cmn.n_frames_encoded = 0;
    side.s_cmn.n_frames_encoded = 0;
    // As in `silk_encode`: the blocks of the packet, and no negative limit.
    let tot_blocks = packet_blocks(packet_length, fs_khz);
    let max_bits = max_bits.max(0);

    // Each channel's LBRR flags left by the previous packet; a reset or a
    // change of the number of coded channels drops them
    // (enc_API.c:262-267). They are written whether or not FEC is still on
    // for this packet (enc_API.c:365-375).
    let mut lbrr_symbols = [0i32; 2];
    for (symbol, ch) in lbrr_symbols.iter_mut().zip([&mut *mid, &mut *side]) {
        if ch.s_cmn.first_frame_after_reset != 0 || transition {
            ch.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }
        for i in 0..n_frames_per_packet as usize {
            *symbol |= ch.s_cmn.lbrr_flags[i] << i;
        }
        ch.s_cmn.lbrr_flag = (*symbol > 0) as i8;
    }

    let n_flag_bits = ((n_frames_per_packet + 1) * 2) as u32;
    let mut lbrr_bits = 0i32;

    for frame_idx in 0..n_frames_per_packet {
        let fi = frame_idx as usize;

        // libopus `inputBuf` layout: two history slots, then the frame.
        let offset = fi * frame_length;
        let mut x1 = [0i16; MAX_FRAME_LENGTH + 2];
        let mut x2 = [0i16; MAX_FRAME_LENGTH + 2];
        x1[2..2 + frame_length].copy_from_slice(&left[offset..offset + frame_length]);
        x2[2..2 + frame_length].copy_from_slice(&right[offset..offset + frame_length]);

        if frame_idx == 0 {
            // Room for both channels' VAD and LBRR flags, patched in below.
            let icdf = [(256i32 - (256i32 >> n_flag_bits)) as u8, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            if lbrr_symbols != [0, 0] {
                lbrr_bits =
                    encode_lbrr_section_stereo(rc, mid, side, lbrr_symbols, n_frames_per_packet);
            }
            mid.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            side.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }

        // enc_API.c:392: once per frame, from the mid channel's pitch lag
        // and voice activity of the frame before.
        silk_hp_variable_cutoff(&mut mid.s_cmn);

        // The frame's target, which the mid/side split shares out.
        let frame_rate_bps = silk_frame_target_rate_bps(
            mid,
            target_rate_bps,
            if frame_idx == 0 { lbrr_bits } else { 0 },
            rc.tell(),
        );
        let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
        silk_stereo_lr_to_ms(
            &mut mid.stereo,
            &mut x1,
            &mut x2,
            &mut ix,
            &mut mid_only,
            &mut rates,
            frame_rate_bps,
            mid.s_cmn.speech_activity_q8,
            to_mono,
            fs_khz,
            frame_length,
        );
        mid.stereo.pred_ix[fi] = ix;
        mid.stereo.mid_only_flags[fi] = mid_only;
        let side_after_mid_only = mid.stereo.prev_decode_only_middle == 1;
        if mid_only == 0 {
            if side_after_mid_only {
                reset_side_for_coding(side);
            }
            silk_encode_do_vad(side, &x2[1..1 + frame_length], activity);
        } else {
            side.s_cmn.vad_flags[fi] = 0;
        }

        silk_stereo_encode_pred(rc, &ix);
        // A side with voice activity is coded, so the decoder knows the
        // frame isn't mid-only without the flag.
        if side.s_cmn.vad_flags[fi] == 0 {
            silk_stereo_encode_mid_only(rc, mid_only);
        }
        silk_encode_do_vad(mid, &x1[1..1 + frame_length], activity);

        for (n, (ch, buf)) in [(&mut *mid, &mut x1), (&mut *side, &mut x2)]
            .into_iter()
            .enumerate()
        {
            let mut channel_max_bits = silk_frame_max_bits(max_bits, tot_blocks, frame_idx);
            let mut channel_use_cbr = use_cbr != 0 && frame_idx == n_frames_per_packet - 1;
            if n == 0 && rates[1] > 0 {
                // The side fills the frame up to the cap; give the mid up to
                // half of the frame's bits.
                channel_use_cbr = false;
                channel_max_bits -= max_bits / (tot_blocks * 2);
            }
            if rates[n] > 0 {
                silk_control_snr(&mut ch.s_cmn, rates[n]);
                let cond_coding = if frame_idx == 0 {
                    CODE_INDEPENDENTLY
                } else if n == 1 && side_after_mid_only {
                    // The side's previous frame wasn't coded, but its LTP
                    // state is well-defined after the reset.
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                silk_lp_variable_cutoff(&mut ch.s_cmn.s_lp, &mut buf[1..], frame_length);
                let mut frame_bytes = 0i32;
                let ret = silk_encode_frame(
                    ch,
                    &buf[1..1 + frame_length],
                    rc,
                    &mut frame_bytes,
                    cond_coding,
                    channel_max_bits,
                    channel_use_cbr as i32,
                );
                if ret != 0 {
                    return ret;
                }
            }
            ch.s_cmn.n_frames_encoded += 1;
        }
        mid.stereo.prev_decode_only_middle = mid_only as i32;
    }

    // Each channel's VAD flags then its LBRR flag, mid first.
    let mut flags = 0u32;
    for ch in [&*mid, &*side] {
        for i in 0..n_frames_per_packet as usize {
            flags = (flags << 1) | ch.s_cmn.vad_flags[i] as u32;
        }
        flags = (flags << 1) | ch.s_cmn.lbrr_flag as u32;
    }
    rc.patch_initial_bits(flags, n_flag_bits);

    *n_bytes_out = (rc.tell() + 7) >> 3;
    silk_update_bit_reservoir(mid, target_rate_bps, *n_bytes_out);
    mid.stereo.n_prev_channels_internal = 2;
    SILK_NO_ERROR
}

/// The LBRR section of a stereo packet (enc_API.c:352-389): each channel's
/// per-frame LBRR symbol for multi-frame packets, then the LBRR frames, mid
/// before side. A mid LBRR frame is preceded by the stereo header of the
/// frame it repeats, written in the previous packet and still in `pred_ix` /
/// `mid_only_flags`; its mid-only flag is implied when the side has LBRR too.
/// Returns the bits written.
fn encode_lbrr_section_stereo(
    rc: &mut RangeCoder,
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    lbrr_symbols: [i32; 2],
    n_frames_per_packet: i32,
) -> i32 {
    let start_bits = rc.tell();
    if n_frames_per_packet > 1 {
        for symbol in lbrr_symbols.into_iter().filter(|&s| s > 0) {
            rc.encode_icdf(symbol - 1, lbrr_flags_icdf(n_frames_per_packet), 8);
        }
    }
    for i in 0..n_frames_per_packet as usize {
        if mid.s_cmn.lbrr_flags[i] != 0 {
            silk_stereo_encode_pred(rc, &mid.stereo.pred_ix[i]);
            if side.s_cmn.lbrr_flags[i] == 0 {
                silk_stereo_encode_mid_only(rc, mid.stereo.mid_only_flags[i]);
            }
            encode_lbrr_frame(rc, mid, i);
        }
        if side.s_cmn.lbrr_flags[i] != 0 {
            encode_lbrr_frame(rc, side, i);
        }
    }
    rc.tell() - start_bits
}

/// One channel's LBRR frame `i`: indices, then pulses.
fn encode_lbrr_frame(rc: &mut RangeCoder, ch: &mut SilkEncoderState, i: usize) {
    let cond_coding = if i > 0 && ch.s_cmn.lbrr_flags[i - 1] != 0 {
        CODE_CONDITIONALLY
    } else {
        CODE_INDEPENDENTLY
    };
    silk_encode_indices(ch, rc, i, true, cond_coding);
    silk_encode_pulses(
        rc,
        ch.s_cmn.indices_lbrr[i].signal_type as i32,
        ch.s_cmn.indices_lbrr[i].quant_offset_type as i32,
        &mut ch.s_cmn.pulses_lbrr[i],
        ch.s_cmn.frame_length as usize,
    );
}

/// Reset the side encoder's memory for its first coded frame after frames
/// that coded the mid only, whose side was never coded (enc_API.c:441-453).
fn reset_side_for_coding(side: &mut SilkEncoderState) {
    side.s_shape = SilkShapeState::default();
    side.s_nsq = SilkNSQState::default();
    side.s_cmn.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
    side.s_cmn.s_lp.in_lp_state = [0; 2];
    side.s_cmn.prev_lag = 100;
    side.s_nsq.lag_prev = 100;
    side.s_shape.last_gain_index = 10;
    side.s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    side.s_nsq.prev_gain_q16 = 65536;
    side.s_cmn.first_frame_after_reset = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An encoder coding `packet_size_ms` packets of `n_frames` frames, with
    /// `n_frames_encoded` of them coded so far.
    fn encoder(packet_size_ms: i32, n_frames: i32, n_frames_encoded: i32) -> SilkEncoderState {
        let mut enc = SilkEncoderState::default();
        enc.s_cmn.packet_size_ms = packet_size_ms;
        enc.s_cmn.n_frames_per_packet = n_frames;
        enc.s_cmn.n_frames_encoded = n_frames_encoded;
        enc
    }

    #[test]
    fn frame_target_takes_out_the_lbrr_average() {
        let mut enc = encoder(20, 1, 0);
        // No LBRR: the packet's 640 bits.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 0), 32000);
        // The first LBRR packet isn't averaged: (640 - 200) * 50.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 200, 0), 22000);
        assert_eq!(enc.n_bits_used_lbrr, 200);
        // Then the average: (200 + 100) / 2 = 150, (640 - 150) * 50.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 100, 0), 24500);
        assert_eq!(enc.n_bits_used_lbrr, 150);
        // Under 10 bits counts as no LBRR, which drops the average at once.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 9, 0), 32000);
        assert_eq!(enc.n_bits_used_lbrr, 0);
    }

    #[test]
    fn frame_target_pays_back_the_reservoir() {
        let mut enc = encoder(20, 1, 0);
        enc.n_bits_exceeded = 1000;
        // 1000 bits over 500 ms: 2000 b/s off.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 0), 30000);
        // 10 ms packets: (320 bits) * 100, less the same payback.
        let mut enc = encoder(10, 1, 0);
        enc.n_bits_exceeded = 1000;
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 0), 30000);
    }

    #[test]
    fn later_frames_count_the_lbrr_section_as_overshoot() {
        // 40 ms at 32 kb/s: 1280 bits in two frames.
        let mut enc = encoder(40, 2, 0);
        // Frame 0: (1280 - 200) / 2 = 540 bits.
        assert_eq!(
            silk_frame_target_rate_bps(&mut enc, 32000, 200, 2 + 200),
            27000
        );
        // Frame 1 drops the LBRR average (libopus resets `curr_nBitsUsedLBRR`
        // per frame): 640 bits a frame, and the 2 flag bits, the 200-bit
        // section and frame 0's 540 bits are 102 over 640: 204 b/s off.
        enc.s_cmn.n_frames_encoded = 1;
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 742), 31796);
        assert_eq!(enc.n_bits_used_lbrr, 0);
    }

    #[test]
    fn frame_target_stays_within_5000_and_the_input_rate() {
        // An LBRR section nearly as large as the packet: floored at 5000.
        let mut enc = encoder(20, 1, 0);
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 600, 0), 5000);
        // Frames under budget so far raise the target, but never above the
        // input rate.
        let mut enc = encoder(40, 2, 1);
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 100), 32000);
        // Below 5000 the bounds swap, as in silk_LIMIT: 4000 stays 4000, and
        // a payback below the input rate stops at it.
        let mut enc = encoder(20, 1, 0);
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 4000, 0, 0), 4000);
        enc.n_bits_exceeded = 100;
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 4000, 0, 0), 4000);
    }

    #[test]
    fn bit_reservoir_keeps_the_overshoot_within_0_and_10000() {
        // 32 kb/s at 20 ms: 640 bits a packet.
        let mut enc = encoder(20, 1, 0);
        silk_update_bit_reservoir(&mut enc, 32000, 90);
        assert_eq!(enc.n_bits_exceeded, 80);
        silk_update_bit_reservoir(&mut enc, 32000, 70);
        assert_eq!(enc.n_bits_exceeded, 0);
        silk_update_bit_reservoir(&mut enc, 32000, 70);
        assert_eq!(enc.n_bits_exceeded, 0, "an undershoot isn't banked");
        enc.n_bits_exceeded = 9900;
        silk_update_bit_reservoir(&mut enc, 32000, 200);
        assert_eq!(enc.n_bits_exceeded, 10000);
    }
}
