use crate::fixedvec::FixedVec;
use crate::float2int16;
use crate::range_coder::{RangeCoder, RangeCoderState};
use crate::silk::control_codec::{
    SilkBandwidthControl, silk_control_audio_bandwidth, silk_control_encoder, silk_setup_lbrr,
};
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
use crate::silk::resampler::SilkResampler;
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

    // "Convert speech activity into VAD and DTX flags" (encode_frame_FIX.c
    // 62-79). The packet starts with `in_dtx` equal to `use_dtx`. The
    // first 10 frames without speech (200 ms) are still sent, the next 20
    // are left out, and then one is sent again.
    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
        if ps_enc.s_cmn.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.in_dtx = 0;
        } else if ps_enc.s_cmn.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX {
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

/// Prefill the encoder of one or two coded channels with 10 ms of
/// internal-rate input, as libopus's `silk_Encode` does with `prefillFlag`
/// set (enc_API.c): run the voice activity detectors on the frame and push it
/// into the analysis history, coding nothing.
///
/// `input` holds the frame of the mono or left channel, then that of the
/// right channel, which is read only with `side`. Two channels are converted
/// to mid/side first, which updates the stereo state. The side's history
/// takes the frame unless the frame codes the mid only. `total_rate_bps` is
/// the SILK bitrate; the mid/side split reads it.
pub fn silk_encode_prefill(
    mid: &mut SilkEncoderState,
    mut side: Option<&mut SilkEncoderState>,
    input: [&[i16]; 2],
    total_rate_bps: i32,
    activity: i32,
) {
    let Some((fs_khz, len)) = prefill_geometry(mid) else {
        return;
    };
    let n_channels = 1 + side.is_some() as usize;
    if input[..n_channels].iter().any(|frame| frame.len() < len) {
        return;
    }

    // libopus `inputBuf`: two samples of history, then the frame.
    let mut x = [[0i16; MAX_FRAME_LENGTH + 2]; 2];
    for (buf, frame) in x.iter_mut().zip(&input[..n_channels]) {
        buf[2..2 + len].copy_from_slice(&frame[..len]);
    }
    let [x1, x2] = &mut x;
    // Whether the side's history takes the frame.
    let mut side_coded = false;
    if let Some(side) = side.as_deref_mut() {
        // enc_API.c:194-216: the prefill is a call with two coded channels.
        enter_stereo(mid, side);
        // enc_API.c:222-236: libopus starts the side encoder (194-211) before
        // the prefill reset, and that reset gives every coded channel the
        // mid's low-pass state (the saved state for a rate switch, a cleared
        // one otherwise). The caller reset the channels before this call, so
        // a side that `enter_stereo` just started has lost it: copy it again.
        side.s_cmn.s_lp = mid.s_cmn.s_lp;

        let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
        // enc_API.c:395-423 for a prefill: the target of one 10 ms frame,
        // with no LBRR section, less the payback of the bit reservoir. A
        // reset for a rate switch keeps the reservoir.
        let n_bits = packet_target_bits(total_rate_bps, 10);
        let target_rate_bps = limit_target_rate(
            n_bits * 100 - reservoir_payback_bps(mid.n_bits_exceeded as i64),
            total_rate_bps,
        );
        // The encoders were just reset, so the previous speech activity is 0.
        silk_stereo_lr_to_ms(
            &mut mid.stereo,
            x1,
            x2,
            &mut ix,
            &mut mid_only,
            &mut rates,
            target_rate_bps,
            0,
            false,
            fs_khz,
            len,
        );
        mid.stereo.pred_ix[0] = ix;
        mid.stereo.mid_only_flags[0] = mid_only;

        side.s_cmn.n_frames_encoded = 0;
        if mid_only == 0 {
            if mid.stereo.prev_decode_only_middle == 1 {
                reset_side_for_coding(side);
            }
            prefill_vad(side, &x2[1..1 + len], activity);
        } else {
            side.s_cmn.vad_flags[0] = 0;
        }
        mid.stereo.prev_decode_only_middle = mid_only as i32;
        side_coded = rates[1] > 0;
    } else {
        // One coded channel: the history is the end of the frame before.
        x1[..2].copy_from_slice(&mid.stereo.s_mid);
        mid.stereo.s_mid.copy_from_slice(&x1[len..len + 2]);
        // enc_API.c:216, 575: the prefill is a call with one coded channel.
        mid.stereo.n_channels_internal = 1;
    }

    // enc_API.c:456: the detector runs on the prefill frame too, so its
    // noise estimates have seen one frame when the first frame is coded.
    mid.s_cmn.n_frames_encoded = 0;
    prefill_vad(mid, &x1[1..1 + len], activity);
    prefill_frame(mid, x1, len);
    if let Some(side) = side
        && side_coded
    {
        prefill_frame(side, x2, len);
    }
    mid.stereo.n_prev_channels_internal = n_channels as i32;
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
    silk_res_nrg_unscaled_fix(ps_enc, &mut s_enc_ctrl, x_tmp_frame, &x_buf_copy);

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

    // The state at the start of the frame, to which each new try of the rate
    // loop goes back. Only range coder symbols are coded in between, so the
    // coder's words are enough (`RangeCoder::save_state`): its bytes below
    // `offs` never change once written, and the next try overwrites the
    // bytes above.
    let rc_copy = rc.save_state();
    let frame_start_offs = rc.offs as usize;
    let nsq_copy = ps_enc.s_nsq;
    let seed_copy = ps_enc.s_cmn.indices.seed;
    let ec_prev_lag_index_copy = ps_enc.s_cmn.ec_prev_lag_index;
    let ec_prev_signal_type_copy = ps_enc.s_cmn.ec_prev_signal_type;
    // The last try under the budget: the coder's words, the bytes it wrote
    // in this frame, the quantizer state and the gain index.
    let mut rc_copy2: Option<RangeCoderState> = None;
    let mut nsq_copy2: Option<SilkNSQState> = None;
    let mut ec_buf_copy: FixedVec<u8, { crate::packet::MAX_FRAME_BYTES }> = FixedVec::new();
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
                rc.restore_state(&rc_copy);
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

            // The coder before this try's symbols. If the last try is over
            // the budget too and no try was under it, the frame is coded
            // again from here without pulses.
            let rc_before_symbols = rc.save_state();

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
                rc.restore_state(&rc_before_symbols);

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
                    rc.restore_state(rc_c2);
                    let offs = rc.offs as usize;
                    rc.buf[frame_start_offs..offs].copy_from_slice(&ec_buf_copy);
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

                rc_copy2 = Some(rc.save_state());
                ec_buf_copy.clear();
                ec_buf_copy.extend_from_slice(&rc.buf[frame_start_offs..rc.offs as usize]);
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

/// Encode one packet with one or two coded channels, the way libopus's
/// `silk_Encode` does once the input is at the internal rate (enc_API.c).
///
/// `input` holds the packet of the mono or left channel, then that of the
/// right channel, which is read only with `side`. Each input must hold the
/// whole packet: a frame skipped for want of input would leave its flags in
/// the packet header undefined.
///
/// Two coded channels are converted to mid/side frame by frame by
/// [`silk_stereo_lr_to_ms`], which also picks the predictors, the stereo
/// width and the split of the frame's bits between mid and side. `mid`
/// codes the mid and holds the stereo state; `side` codes the side residual,
/// except in frames that code the mid only (panned mono). With `to_mono`
/// the stereo predictors are zero and the width falls to zero over the
/// first 8 ms: libopus codes the last packet with two channels so before
/// stereo input goes on as one channel.
///
/// A packet with one coded channel after packets with two leaves the side
/// encoder as it is; libopus starts it afresh when two channels are coded
/// again.
///
/// Returns 0, or a negative SILK error code when the encoder is not set up
/// or the input is shorter than the packet.
pub fn silk_encode_packet(
    mid: &mut SilkEncoderState,
    mut side: Option<&mut SilkEncoderState>,
    input: [&[i16]; 2],
    rc: &mut RangeCoder,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: bool,
    activity: i32,
    to_mono: bool,
) -> i32 {
    let Some((fs_khz, _)) = prefill_geometry(mid) else {
        return SILK_ENC_FS_NOT_SUPPORTED;
    };
    let n_channels = 1 + side.is_some() as usize;
    // enc_API.c:194-216: a change of the number of coded channels is a
    // transition, which drops the LBRR frames kept for this packet
    // (enc_API.c:262-267), as a reset does. A packet with two channels after
    // packets with one starts the side encoder and the stereo state afresh.
    let transition = match side.as_deref_mut() {
        Some(side) => enter_stereo(mid, side),
        None => {
            let transition = mid.stereo.n_channels_internal != 1;
            mid.stereo.n_channels_internal = 1;
            transition
        }
    };
    let n_frames_per_packet = mid.s_cmn.n_frames_per_packet;
    let frame_length = mid.s_cmn.frame_length as usize;
    let packet_length = n_frames_per_packet.max(0) as usize * frame_length;
    if frame_length == 0
        || frame_length > MAX_FRAME_LENGTH
        || n_frames_per_packet < 1
        || n_frames_per_packet as usize > MAX_FRAMES_PER_PACKET
        || side.as_deref().is_some_and(|side| {
            side.s_cmn.fs_khz != fs_khz || side.s_cmn.frame_length as usize != frame_length
        })
        || input[..n_channels].iter().any(|x| x.len() < packet_length)
    {
        return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
    }

    // libopus refuses input longer than a packet (enc_API.c:245-249). Here
    // the packet is coded and the rest is left, so the blocks are those of
    // the packet.
    let tot_blocks = packet_blocks(packet_length, fs_khz);
    // A limit below zero acts like a limit of zero: every size compared
    // with it is at least one bit.
    let max_bits = max_bits.max(0);

    // Each channel's LBRR flags left by the previous packet; a reset or a
    // change of the number of coded channels drops them. The LBRR frames are
    // written whenever there are any (enc_API.c:365-375). Whether FEC is
    // still on for this packet does not matter: it only decides whether this
    // packet's frames leave LBRR frames for the next one.
    let mut lbrr_symbols = [0i32; 2];
    for (n, symbol) in lbrr_symbols.iter_mut().enumerate().take(n_channels) {
        let ch = channel(mid, &mut side, n);
        ch.s_cmn.n_frames_encoded = 0;
        // enc_API.c:273: each coded channel starts the packet in DTX when
        // DTX is on. Its voice activity detector clears that for a frame
        // with speech or for one of the first 10 without.
        ch.s_cmn.in_dtx = ch.s_cmn.use_dtx;
        if ch.s_cmn.first_frame_after_reset != 0 || transition {
            ch.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }
        for i in 0..n_frames_per_packet as usize {
            *symbol |= ch.s_cmn.lbrr_flags[i] << i;
        }
        ch.s_cmn.lbrr_flag = (*symbol > 0) as i8;
    }

    // Each channel's VAD flags and its LBRR flag, patched in at the end.
    let n_flag_bits = ((n_frames_per_packet + 1) * n_channels as i32) as u32;
    // Bits of the LBRR section written at the start of the packet, which come
    // out of the frame target (libopus `curr_nBitsUsedLBRR`).
    let mut lbrr_bits = 0i32;

    for frame_idx in 0..n_frames_per_packet {
        let fi = frame_idx as usize;

        // libopus `inputBuf`: two samples of history, then the frame.
        let offset = fi * frame_length;
        let mut x = [[0i16; MAX_FRAME_LENGTH + 2]; 2];
        for (buf, packet) in x.iter_mut().zip(&input[..n_channels]) {
            buf[2..2 + frame_length].copy_from_slice(&packet[offset..offset + frame_length]);
        }

        if frame_idx == 0 {
            let icdf = [(256i32 - (256i32 >> n_flag_bits)) as u8, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            if lbrr_symbols != [0, 0] {
                // libopus writes the section whatever its size. A packet
                // too small for it and the frame overflows the range coder,
                // and the caller then sends a frame for the decoder to
                // conceal (opus_encoder.c:2582-2594).
                lbrr_bits =
                    encode_lbrr_section(rc, mid, side.as_deref_mut(), lbrr_symbols, n_frames_per_packet);
            }
            // Reset the LBRR flags for the next packet (enc_API.c:379-382);
            // this packet's frames set them again from inside
            // silk_lbrr_encode.
            mid.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            if let Some(side) = side.as_deref_mut() {
                side.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            }
        }

        // enc_API.c:392: once per frame, from the mid channel's pitch lag
        // and voice activity of the frame before.
        silk_hp_variable_cutoff(&mut mid.s_cmn);

        let frame_rate_bps = silk_frame_target_rate_bps(
            mid,
            target_rate_bps,
            if frame_idx == 0 { lbrr_bits } else { 0 },
            rc.tell(),
        );
        // The rate of each coded channel.
        let mut rates = [frame_rate_bps, 0];
        let side_after_mid_only = mid.stereo.prev_decode_only_middle == 1;
        let [x1, x2] = &mut x;
        if let Some(side) = side.as_deref_mut() {
            // The frame's target, which the mid/side split shares out.
            let (mut ix, mut mid_only) = ([[0i8; 3]; 2], 0i8);
            silk_stereo_lr_to_ms(
                &mut mid.stereo,
                x1,
                x2,
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
        } else {
            // enc_API.c:451-455, "Buffering": the frame behind the last two
            // samples of the previous one.
            x1[..2].copy_from_slice(&mid.stereo.s_mid);
            mid.stereo.s_mid.copy_from_slice(&x1[frame_length..frame_length + 2]);
        }
        silk_encode_do_vad(mid, &x1[1..1 + frame_length], activity);

        for (n, buf) in x.iter_mut().enumerate().take(n_channels) {
            let ch = channel(mid, &mut side, n);
            // A cap on the packet's bits so far, the LBRR section included,
            // since the rate control measures the range coder's `tell`.
            let mut channel_max_bits = silk_frame_max_bits(max_bits, tot_blocks, frame_idx);
            let mut channel_use_cbr = use_cbr && frame_idx == n_frames_per_packet - 1;
            if n == 0 && rates[1] > 0 {
                // The side fills the frame up to the cap; give the mid up to
                // half of the frame's bits.
                channel_use_cbr = false;
                channel_max_bits -= max_bits / (tot_blocks * 2);
            }
            // As in libopus, a channel without a rate is not coded. One
            // coded channel has the frame's rate, which is above 0 for
            // every bitrate above 0.
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
        // enc_API.c:521. libopus runs this line for one coded channel too,
        // with the flags that the last two-channel packet left.
        mid.stereo.prev_decode_only_middle = mid.stereo.mid_only_flags[fi] as i32;
    }

    // Each channel's VAD flags then its LBRR flag, mid first.
    let mut flags = 0u32;
    for n in 0..n_channels {
        let ch = channel(mid, &mut side, n);
        for i in 0..n_frames_per_packet as usize {
            flags = (flags << 1) | ch.s_cmn.vad_flags[i] as u32;
        }
        flags = (flags << 1) | ch.s_cmn.lbrr_flag as u32;
    }
    rc.patch_initial_bits(flags, n_flag_bits);

    // enc_API.c:550-553: "Return zero bytes if all channels DTXed". The
    // bit reservoir then counts the packet as empty.
    let n_bytes_out = if silk_packet_in_dtx(mid, side.as_deref()) {
        0
    } else {
        (rc.tell() + 7) >> 3
    };
    silk_update_bit_reservoir(mid, target_rate_bps, n_bytes_out);
    mid.stereo.n_prev_channels_internal = n_channels as i32;
    SILK_NO_ERROR
}

/// Whether the packet just coded has no SILK bytes because every coded
/// channel is in discontinuous transmission (DTX): libopus's test before
/// it sets `*nBytesOut` to 0 (enc_API.c:550-553). `side` is the side
/// channel when the packet coded two channels.
pub fn silk_packet_in_dtx(mid: &SilkEncoderState, side: Option<&SilkEncoderState>) -> bool {
    mid.s_cmn.in_dtx != 0 && side.is_none_or(|side| side.s_cmn.in_dtx != 0)
}

/// Coded channel `n` of a packet: the mono or mid channel for 0, the side
/// for 1.
fn channel<'a>(
    mid: &'a mut SilkEncoderState,
    side: &'a mut Option<&mut SilkEncoderState>,
    n: usize,
) -> &'a mut SilkEncoderState {
    match (n, side) {
        (1, Some(side)) => side,
        _ => mid,
    }
}

pub use crate::silk::control_codec::SILK_ENC_FS_NOT_SUPPORTED;

/// Mix interleaved `stereo` input down to `mono`, one sample per pair, for
/// coding it as one channel. Port of enc_API.c:310-313, "Combine left and
/// right channels before resampling": `silk_RSHIFT_ROUND( left + right, 1 )`.
///
/// This is the fixed-point build. The float build of libopus converts the
/// sum of the two float samples to 16 bits first, which clips a sum beyond
/// 16 bits before it is halved: `silk_downmix_stereo_float`.
pub fn silk_downmix_stereo(stereo: &[i16], mono: &mut [i16]) {
    let n = mono.len().min(stereo.len() / 2);
    for i in 0..n {
        mono[i] = silk_rshift_round(stereo[2 * i] as i32 + stereo[2 * i + 1] as i32, 1) as i16;
    }
}

/// The float build's "Combine left and right channels before resampling"
/// (enc_API.c:309-312): `silk_RSHIFT_ROUND( RES2INT16( left + right ), 1 )`
/// for each pair of interleaved float samples. The sum is clipped to 16 bits
/// before it is halved, so where left plus right exceed full scale the mono
/// signal stops at half of it.
pub fn silk_downmix_stereo_float(stereo: &[f32], mono: &mut [i16]) {
    for (m, &[left, right]) in mono.iter_mut().zip(stereo.as_chunks::<2>().0) {
        *m = silk_rshift_round(float2int16(left + right) as i32, 1) as i16;
    }
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
    side_resampler: &mut SilkResampler,
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
    // enc_API.c:205-208 also copies the resampler state, for stereo input.
    // `SilkEncoder` does that copy before it resamples.

    ret += silk_control_encoder(
        side,
        mid.s_cmn.fs_khz,
        mid.s_cmn.packet_size_ms,
        mid.s_cmn.target_rate_bps,
        mid.s_cmn.complexity,
    );
    side.s_cmn.use_cbr = mid.s_cmn.use_cbr;
    // libopus gives both channels the DTX setting of the same control call
    // (control_codec.c:75).
    side.s_cmn.use_dtx = mid.s_cmn.use_dtx;
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

/// The LBRR (in-band FEC) section of a packet (enc_API.c:352-389): for
/// multi-frame packets each channel's per-frame LBRR symbol, then the LBRR
/// frames, mid before side. With two coded channels a mid LBRR frame is
/// preceded by the stereo header of the frame it repeats, written in the
/// previous packet and still in `pred_ix` / `mid_only_flags`; its mid-only
/// flag is implied when the side has LBRR too. Returns the bits written.
fn encode_lbrr_section(
    rc: &mut RangeCoder,
    mid: &mut SilkEncoderState,
    mut side: Option<&mut SilkEncoderState>,
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
            if let Some(side) = side.as_deref() {
                silk_stereo_encode_pred(rc, &mid.stereo.pred_ix[i]);
                if side.s_cmn.lbrr_flags[i] == 0 {
                    silk_stereo_encode_mid_only(rc, mid.stereo.mid_only_flags[i]);
                }
            }
            encode_lbrr_frame(rc, mid, i);
        }
        if let Some(side) = side.as_deref_mut()
            && side.s_cmn.lbrr_flags[i] != 0
        {
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

/// libopus `silk_setup_resamplers` (control_codec.c:134) for a running SILK
/// encoder that changes its internal rate to `new_khz`: the analysis history
/// `x_buf` is resampled to the input rate and from there to the new rate.
/// The second step runs through the encoder's own input resampler, freshly
/// set up for the new rate, so that its filter state continues from the
/// history too.
fn silk_carry_history_to_rate(
    enc: &mut SilkEncoderState,
    resampler: &mut SilkResampler,
    api_hz: i32,
    new_khz: i32,
) -> Result<(), &'static str> {
    let old_khz = enc.s_cmn.fs_khz;
    // 2 frames of history and the noise-shaping look-ahead: 45 ms, or 25 ms
    // for 10 ms packets.
    let buf_length_ms = ((enc.s_cmn.nb_subfr * 5) << 1) + LA_SHAPE_MS as i32;
    let old_n = (buf_length_ms * old_khz) as usize;
    let new_n = (buf_length_ms * new_khz) as usize;
    let api_n = (buf_length_ms * (api_hz / 1000)) as usize;
    let mut at_api_rate = [0i16; 45 * 48];
    let mut at_new_rate = [0i16; 45 * 16];
    if old_n > enc.s_cmn.x_buf.len() || api_n > at_api_rate.len() || new_n > at_new_rate.len() {
        return Err("Unsupported SILK rate switch");
    }
    let mut to_api = SilkResampler::default();
    if to_api.init(old_khz * 1000, api_hz) != 0
        || to_api.process(&mut at_api_rate[..api_n], &enc.s_cmn.x_buf[..old_n], old_n as i32) != 0
        || resampler.init_for_enc(api_hz, new_khz * 1000) != 0
        || resampler.process(&mut at_new_rate[..new_n], &at_api_rate[..api_n], api_n as i32) != 0
    {
        return Err("Unsupported SILK rate switch");
    }
    enc.s_cmn.x_buf[..new_n].copy_from_slice(&at_new_rate[..new_n]);
    Ok(())
}

/// Largest packet at the internal rate: 60 ms at 16 kHz.
const MAX_PACKET_SAMPLES: usize = MAX_FRAMES_PER_PACKET * MAX_FRAME_LENGTH;
/// Largest packet of one input channel at the input rate: 60 ms at 48 kHz.
const MAX_API_SAMPLES: usize = 2880;

/// A value of [`SilkEncoder`] on the heap (`heap` feature) or in place.
#[cfg(feature = "heap")]
type Slot<T> = Box<T>;
#[cfg(not(feature = "heap"))]
type Slot<T> = Inline<T>;

/// A value kept in place: [`Slot`] without the `heap` feature.
#[cfg(not(feature = "heap"))]
struct Inline<T>(T);

#[cfg(not(feature = "heap"))]
impl<T> core::ops::Deref for Inline<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[cfg(not(feature = "heap"))]
impl<T> core::ops::DerefMut for Inline<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

fn slot<T>(value: T) -> Slot<T> {
    #[cfg(feature = "heap")]
    return Box::new(value);
    #[cfg(not(feature = "heap"))]
    return Inline(value);
}

/// The input of [`SilkEncoder::encode`]: interleaved samples at the input
/// rate.
#[derive(Clone, Copy)]
pub enum SilkInput<'a> {
    /// 16-bit samples, as the fixed-point build of libopus takes them.
    /// Stereo input coded as one channel is mixed down by
    /// [`silk_downmix_stereo`].
    Int16(&'a [i16]),
    /// Float samples (1.0 is full scale), as the float build of libopus
    /// takes them. Each sample is rounded to 16 bits (`RES2INT16`). Stereo
    /// input coded as one channel is mixed down by
    /// `silk_downmix_stereo_float`.
    Float(&'a [f32]),
}

impl SilkInput<'_> {
    fn len(&self) -> usize {
        match self {
            SilkInput::Int16(x) => x.len(),
            SilkInput::Float(x) => x.len(),
        }
    }
}

/// The restart that [`SilkEncoder::encode`] does before a packet: libopus's
/// call of `silk_Encode` with `prefillFlag` set.
#[derive(Clone, Copy)]
pub struct SilkPrefill<'a> {
    /// The restart belongs to a change of the internal rate after a
    /// redundant frame (libopus `prefillFlag` 2). The channels then keep the
    /// state of the input low-pass filter, and the rate switch goes on.
    /// Otherwise (`prefillFlag` 1) the channels are cleared and SILK starts
    /// at the rate asked for.
    pub rate_switch: bool,
    /// The 10 ms before the packet, interleaved at the input rate, which the
    /// restarted encoder analyses without coding them. `None` feeds nothing.
    pub input: Option<SilkInput<'a>>,
}

/// The settings of one call of [`SilkEncoder::encode`] and what the call
/// reports back: the part of libopus's `silk_EncControlStruct` that opus1
/// uses. The caller keeps it from packet to packet, as libopus keeps
/// `silk_mode`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SilkControl {
    /// Channels the packet codes (`nChannelsInternal`): 1, or 2 for stereo
    /// input.
    pub channels_internal: usize,
    /// Packet duration in ms: 10, 20, 40 or 60 (`payloadSize_ms`).
    pub payload_size_ms: i32,
    /// Lowest internal rate allowed, in kHz (`minInternalSampleRate`).
    pub min_khz: i32,
    /// Highest internal rate allowed, in kHz (`maxInternalSampleRate`).
    pub max_khz: i32,
    /// Internal rate asked for, in kHz (`desiredInternalSampleRate`).
    pub desired_khz: i32,
    /// Bitrate in bits per second (`bitRate`).
    pub bit_rate: i32,
    /// The most bits the packet may take (`maxBits`).
    pub max_bits: i32,
    /// Constant bitrate (`useCBR`).
    pub use_cbr: bool,
    /// Complexity, 0 to 10 (`complexity`).
    pub complexity: i32,
    /// Expected packet loss in percent (`packetLossPercentage`).
    pub packet_loss_perc: i32,
    /// This packet's frames leave LBRR frames (in-band FEC) for the next
    /// packet (`LBRR_coded`).
    pub lbrr_coded: bool,
    /// libopus's `activity` argument of `silk_Encode`: 0
    /// (`VAD_NO_ACTIVITY`) when the caller found no voice activity, for
    /// example in digital silence; any other value leaves the decision to
    /// SILK's own voice activity detector (libopus passes 1 or -1,
    /// `VAD_NO_DECISION`).
    pub activity: i32,
    /// SILK's discontinuous transmission (`useDTX`): after 10 frames
    /// without speech the packet has no SILK bytes, for up to 20 frames in
    /// a row.
    pub use_dtx: bool,
    /// The last packet with two channels before stereo input goes on as one
    /// channel (`toMono`): its stereo width falls to zero.
    pub to_mono: bool,
    /// Every packet is coded as the first after a reset
    /// (`reducedDependency`).
    pub reduced_dependency: bool,
    /// The caller can change the internal rate on this packet
    /// (`opusCanSwitch`).
    pub opus_can_switch: bool,
    /// Test control, not part of libopus: the quantization gains come from
    /// libopus's residual energies alone, as in libopus's fixed-point build
    /// (see `SilkEncoderState::fixed_point_res_nrg`).
    pub fixed_point_res_nrg: bool,
    /// In and out: SILK's speech activity is low enough for a bandwidth
    /// switch to go unheard (`allowBandwidthSwitch`). The rate switch reads
    /// the value of the packet before.
    pub allow_bandwidth_switch: bool,
    /// Out: the internal rate of the packet, in kHz (`internalSampleRate`).
    pub internal_khz: i32,
    /// Out: SILK is ready to change its internal rate on the next packet
    /// (`switchReady`). The packet then keeps room for a redundant frame.
    pub switch_ready: bool,
    /// Out: SILK runs at 16 kHz and its input low-pass filter is in no
    /// transition (`inWBmodeWithoutVariableLP`).
    pub in_wb_mode_without_variable_lp: bool,
    /// Out: the stereo width of the packet in Q14 (`stereoWidth_Q14`), 0
    /// with `to_mono`.
    pub stereo_width_q14: i32,
    /// Out: every coded channel is in DTX, and the packet has no SILK bytes
    /// (libopus `*nBytesOut` 0). The range coder still holds the coded
    /// packet, which the caller drops.
    pub zero_bytes: bool,
}

/// One channel of [`SilkEncoder`].
struct SilkChannel {
    state: SilkEncoderState,
    /// Input rate to internal rate (libopus `resampler_state`).
    resampler: SilkResampler,
    /// The packet at the internal rate.
    x: [i16; MAX_PACKET_SAMPLES],
}

impl SilkChannel {
    fn new() -> Self {
        let mut state = SilkEncoderState::default();
        silk_init_encoder(&mut state, 0);
        Self { state, resampler: SilkResampler::default(), x: [0; MAX_PACKET_SAMPLES] }
    }
}

/// The SILK encoder of one Opus stream: libopus's `silk_encoder` with
/// `silk_Encode` (enc_API.c).
///
/// It owns the state of each channel and its input resampler. For each
/// packet it chooses the internal rate, sets the channels up, resamples the
/// input, mixes stereo input down when the packet codes one channel, handles
/// the change between one and two coded channels, and codes the packet.
pub struct SilkEncoder {
    /// Input sampling rate in Hz (`API_sampleRate`).
    api_hz: i32,
    /// The mono or mid channel. Its state also holds the stereo state and
    /// the packet's bit counters.
    mid: Slot<SilkChannel>,
    /// The side channel, for stereo input only (libopus `state_Fxx[1]`).
    side: Option<Slot<SilkChannel>>,
    /// One input channel at the input rate, before resampling.
    api_buf: Slot<[i16; MAX_API_SAMPLES]>,
    /// The channels are set up for a rate. A restart clears this until the
    /// next control call.
    initialized: bool,
    /// How long SILK has refused a bandwidth switch
    /// (`timeSinceSwitchAllowed_ms`). The bar drops with it.
    time_since_switch_allowed_ms: i32,
}

impl SilkEncoder {
    /// An encoder for input at `api_hz` with `api_channels` (1 or 2)
    /// channels. The first call of [`encode`](Self::encode) sets it up.
    pub fn new(api_hz: i32, api_channels: usize) -> Self {
        Self {
            api_hz,
            mid: slot(SilkChannel::new()),
            side: (api_channels == 2).then(|| slot(SilkChannel::new())),
            api_buf: slot([0; MAX_API_SAMPLES]),
            initialized: false,
            time_since_switch_allowed_ms: 0,
        }
    }

    /// Clear every channel and the packet-level state (libopus
    /// `silk_InitEncoder`), as after CELT-only packets. The resamplers stay;
    /// the next restart sets them up again.
    pub fn reset(&mut self) {
        for ch in core::iter::once(&mut *self.mid).chain(self.side.as_deref_mut()) {
            ch.state = SilkEncoderState::default();
            silk_init_encoder(&mut ch.state, 0);
        }
        self.time_since_switch_allowed_ms = 0;
    }

    /// The state of the mono or mid channel.
    pub fn mid(&self) -> &SilkEncoderState {
        &self.mid.state
    }

    /// The state of the mono or mid channel, to change.
    pub fn mid_mut(&mut self) -> &mut SilkEncoderState {
        &mut self.mid.state
    }

    /// The state of the side channel, for stereo input.
    pub fn side(&self) -> Option<&SilkEncoderState> {
        self.side.as_deref().map(|ch| &ch.state)
    }

    /// Code one packet of `input` into `rc`. With `prefill` the encoder
    /// first restarts and analyses the 10 ms before the packet.
    ///
    /// `input` holds the packet, interleaved at the input rate. `ctl` gives
    /// the settings and gets what SILK reports; see [`SilkControl`].
    pub fn encode(
        &mut self,
        ctl: &mut SilkControl,
        input: SilkInput<'_>,
        prefill: Option<SilkPrefill<'_>>,
        rc: &mut RangeCoder,
    ) -> Result<(), &'static str> {
        let sch = if ctl.channels_internal == 2 && self.side.is_some() { 2 } else { 1 };
        // libopus runs the rate switch state machine
        // (silk_control_audio_bandwidth) in the control call of each coded
        // channel. Here it runs first, before a restart clears the channels.
        // A restart for a rate switch then gives the channels the mid's
        // low-pass state as the state machine left it, with the rate it ran
        // at.
        let restarting = prefill.is_some_and(|p| !p.rate_switch);
        let fs_khz = self.control_audio_bandwidth(ctl, sch, restarting);

        if let Some(p) = prefill {
            // enc_API.c:217-236: the restart clears each coded channel. For a
            // rate switch every channel takes the mid's low-pass state, with
            // the rate it ran at.
            let mut lp = self.mid.state.s_cmn.s_lp;
            lp.saved_fs_khz = self.mid.state.s_cmn.fs_khz;
            for ch in self.coded_channels(sch) {
                silk_init_encoder(&mut ch.state, 0);
                if p.rate_switch {
                    ch.state.s_cmn.s_lp = lp;
                }
            }
            self.initialized = false;
        }

        let new_rate = !self.initialized || self.mid.state.s_cmn.fs_khz != fs_khz;
        // A running encoder that changes its internal rate takes its analysis
        // history along (C silk_setup_resamplers).
        let running_switch = self.initialized && new_rate;
        if running_switch {
            let mid = &mut *self.mid;
            silk_carry_history_to_rate(&mut mid.state, &mut mid.resampler, self.api_hz, fs_khz)?;
            // A side encoder that the packet starts afresh, after packets with
            // one coded channel, has no history to carry: libopus initializes
            // it before its control call (enc_API.c:194-211,
            // control_codec.c:140-142).
            if sch == 2
                && mid.state.stereo.n_channels_internal == 2
                && let Some(side) = self.side.as_deref_mut()
            {
                silk_carry_history_to_rate(&mut side.state, &mut side.resampler, self.api_hz, fs_khz)?;
            }
        }
        // enc_API.c:253-269: silk_control_encoder for each coded channel of
        // every packet. It follows a change of the internal rate, of the
        // packet duration and of the complexity, which thus takes effect with
        // the next packet. The side runs at the mid's rate and packet
        // duration (`force_fs_kHz`).
        let new_packet_size = self.mid.state.s_cmn.packet_size_ms != ctl.payload_size_ms;
        for ch in self.coded_channels(sch) {
            let ret = silk_control_encoder(
                &mut ch.state,
                fs_khz,
                ctl.payload_size_ms,
                ctl.bit_rate,
                ctl.complexity,
            );
            // C: "if( ( ret = silk_control_encoder( ... ) ) != 0 )" ends the
            // packet with the error (enc_API.c:256-260).
            if ret != 0 {
                return Err("SILK cannot code this rate or packet duration");
            }
            // control_codec.c:76. SILK's noise shaping reads the flag
            // (noise_shape_analysis_FIX.c:171).
            ch.state.s_cmn.use_cbr = ctl.use_cbr as i32;
            // control_codec.c:75.
            ch.state.s_cmn.use_dtx = ctl.use_dtx as i32;
            // enc_API.c:262-267: the redundant frames kept for the next
            // packet belong to the old packet layout.
            if new_rate || new_packet_size {
                ch.state.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            }
        }
        if new_rate && !running_switch {
            // C silk_setup_resamplers: a new encoder sets up its input
            // resampler; the input delay comes from delay_matrix_enc.
            if self.mid.resampler.init_for_enc(self.api_hz, fs_khz * 1000) != 0 {
                return Err("Unsupported SILK resampling ratio");
            }
            if let Some(side) = self.side.as_deref_mut() {
                side.resampler = self.mid.resampler.clone();
            }
        }
        self.initialized = true;

        if let Some(p) = prefill {
            // A prefill is a call to silk_Encode of its own
            // (opus_encoder.c:2206), with its own control call: the first
            // packet after it is then no longer the first with FEC.
            self.setup_fec(ctl, sch);
            if let Some(prefill_input) = p.input {
                let n = self.resample(prefill_input, sch, usize::MAX)?;
                let (mid, side, x) = self.coded_packet(sch, n);
                silk_encode_prefill(mid, side, x, ctl.bit_rate, ctl.activity);
            }
        }

        // The control call of this packet's silk_Encode. It comes after the
        // prefill, which can start the side encoder afresh.
        self.setup_fec(ctl, sch);

        let first_frame = self.mid.state.s_cmn.frame_length.max(0) as usize;
        let n = self.resample(input, sch, first_frame)?;
        // enc_API.c:172-177, `reducedDependency`: every packet is coded as
        // the first after a reset, in each channel of the input. It also
        // drops the LBRR frames kept for the packet (enc_API.c:268).
        if ctl.reduced_dependency {
            for ch in self.coded_channels(2) {
                ch.state.s_cmn.first_frame_after_reset = 1;
            }
        }
        for ch in core::iter::once(&mut *self.mid).chain(self.side.as_deref_mut()) {
            ch.state.fixed_point_res_nrg = ctl.fixed_point_res_nrg;
        }
        let (mid, side, x) = self.coded_packet(sch, n);
        let ret = silk_encode_packet(
            mid,
            side,
            x,
            rc,
            ctl.bit_rate,
            ctl.max_bits,
            ctl.use_cbr,
            ctl.activity,
            ctl.to_mono,
        );
        if ret != 0 {
            return Err("SILK encoding failed");
        }
        let side = self.side.as_deref().filter(|_| sch == 2).map(|ch| &ch.state);
        ctl.zero_bytes = silk_packet_in_dtx(&self.mid.state, side);

        // enc_API.c:559-583: SILK allows a bandwidth switch while its speech
        // activity is under a bar that starts at 5 % and is gone after 5 s
        // (SPEECH_ACTIVITY_DTX_THRES, MAX_BANDWIDTH_SWITCH_DELAY_MS).
        let mid = &self.mid.state;
        let bar_q8 = silk_smlawb(13, 3188, self.time_since_switch_allowed_ms);
        if mid.s_cmn.speech_activity_q8 < bar_q8 {
            ctl.allow_bandwidth_switch = true;
            self.time_since_switch_allowed_ms = 0;
        } else {
            ctl.allow_bandwidth_switch = false;
            self.time_since_switch_allowed_ms += ctl.payload_size_ms;
        }
        ctl.in_wb_mode_without_variable_lp = mid.s_cmn.fs_khz == 16 && mid.s_cmn.s_lp.mode == 0;
        ctl.internal_khz = fs_khz;
        // enc_API.c:584: 0 for the packet that leads over to one channel.
        ctl.stereo_width_q14 = if ctl.to_mono { 0 } else { mid.stereo.smth_width_q14 as i32 };
        Ok(())
    }

    /// The states of the first `sch` channels and their first `n` samples
    /// at the internal rate.
    fn coded_packet(
        &mut self,
        sch: usize,
        n: usize,
    ) -> (&mut SilkEncoderState, Option<&mut SilkEncoderState>, [&[i16]; 2]) {
        let mid = &mut *self.mid;
        match self.side.as_deref_mut().filter(|_| sch == 2) {
            Some(side) => (&mut mid.state, Some(&mut side.state), [&mid.x[..n], &side.x[..n]]),
            None => (&mut mid.state, None, [&mid.x[..n], &[]]),
        }
    }

    /// The first `sch` channels: the mono or mid channel, then the side.
    fn coded_channels(&mut self, sch: usize) -> impl Iterator<Item = &mut SilkChannel> {
        core::iter::once(&mut *self.mid).chain(self.side.as_deref_mut()).take(sch)
    }

    /// The internal rate of the packet, in kHz: libopus
    /// `silk_control_audio_bandwidth` for each coded channel. A restarting
    /// or new encoder takes the rate asked for.
    fn control_audio_bandwidth(&mut self, ctl: &mut SilkControl, sch: usize, restarting: bool) -> i32 {
        ctl.switch_ready = false;
        if restarting || !self.initialized {
            // "Encoder has just been initialized"
            return ctl.desired_khz.min(self.api_hz / 1000);
        }
        let mut bw = SilkBandwidthControl {
            api_fs_hz: self.api_hz,
            max_internal_fs_hz: ctl.max_khz * 1000,
            min_internal_fs_hz: ctl.min_khz * 1000,
            desired_internal_fs_hz: ctl.desired_khz * 1000,
            payload_size_ms: ctl.payload_size_ms,
            allow_bandwidth_switch: ctl.allow_bandwidth_switch,
            opus_can_switch: ctl.opus_can_switch,
            max_bits: ctl.max_bits,
            switch_ready: false,
        };
        let fs_khz = silk_control_audio_bandwidth(&mut self.mid.state.s_cmn, &mut bw);
        // Each coded channel keeps its own low-pass state and runs the state
        // machine; the side's rate is then forced to the mid's. A side
        // encoder that the packet starts afresh, after packets with one coded
        // channel, has no rate yet: libopus initializes it before its control
        // call.
        if sch == 2
            && self.mid.state.stereo.n_channels_internal == 2
            && let Some(side) = self.side.as_deref_mut()
        {
            silk_control_audio_bandwidth(&mut side.state.s_cmn, &mut bw);
        }
        ctl.max_bits = bw.max_bits;
        ctl.switch_ready = bw.switch_ready;
        fs_khz
    }

    /// The FEC part of libopus's `silk_control_encoder` for each coded
    /// channel: the packet loss, and `silk_setup_LBRR` with the packet's FEC
    /// decision (control_codec.c:117-126, 403-423).
    fn setup_fec(&mut self, ctl: &SilkControl, sch: usize) {
        for ch in self.coded_channels(sch) {
            ch.state.s_cmn.packet_loss_perc = ctl.packet_loss_perc.clamp(0, 100);
            silk_setup_lbrr(&mut ch.state, ctl.lbrr_coded as i32);
        }
    }

    /// Bring `input` to the internal rate: the mono or left channel, or the
    /// downmix, into the mid channel's `x`, and for two coded channels the
    /// right channel into the side's. `first_frame` is the length of the
    /// packet's first frame at the internal rate. Returns the samples per
    /// channel at the internal rate.
    fn resample(&mut self, input: SilkInput<'_>, sch: usize, first_frame: usize) -> Result<usize, &'static str> {
        const FAILED: Result<usize, &str> = Err("SILK input resampling failed");
        let api_channels = 1 + self.side.is_some() as usize;
        let frames = input.len() / api_channels;
        let mid = &mut *self.mid;
        let fs_khz = mid.state.s_cmn.fs_khz.max(0) as usize;
        let api_khz = (self.api_hz / 1000).max(1) as usize;
        let n = frames * fs_khz / api_khz;
        if frames > MAX_API_SAMPLES || n > MAX_PACKET_SAMPLES {
            return Err("SILK input longer than a packet");
        }
        let api = &mut self.api_buf[..frames];
        let prev_channels = mid.state.stereo.n_prev_channels_internal;
        match (self.side.as_deref_mut(), sch) {
            (Some(side), 1) => {
                // enc_API.c:307-332, "Combine left and right channels before
                // resampling".
                match input {
                    SilkInput::Int16(x) => silk_downmix_stereo(&x[..2 * frames], api),
                    SilkInput::Float(x) => silk_downmix_stereo_float(&x[..2 * frames], api),
                }
                if mid.resampler.process(&mut mid.x[..n], api, frames as i32) != 0 {
                    return FAILED;
                }
                // "On the first mono frame, average the results for the two
                // resampler states".
                if prev_channels == 2 {
                    let first = first_frame.min(n);
                    let first_api = first * api_khz / fs_khz.max(1);
                    let ret = silk_average_first_mono_frame(
                        &mut side.resampler,
                        &mut mid.x[..first],
                        &api[..first_api],
                    );
                    if ret != 0 {
                        return FAILED;
                    }
                }
            }
            (Some(side), _) => {
                // enc_API.c:296-299: "Making sure to start both resamplers
                // from the same state when switching from mono to stereo".
                if prev_channels == 1 {
                    side.resampler = mid.resampler.clone();
                }
                for (c, ch) in [mid, side].into_iter().enumerate() {
                    match input {
                        SilkInput::Int16(x) => {
                            for (dst, pair) in api.iter_mut().zip(x.as_chunks::<2>().0) {
                                *dst = pair[c];
                            }
                        }
                        SilkInput::Float(x) => {
                            for (dst, pair) in api.iter_mut().zip(x.as_chunks::<2>().0) {
                                *dst = float2int16(pair[c]);
                            }
                        }
                    }
                    if ch.resampler.process(&mut ch.x[..n], api, frames as i32) != 0 {
                        return FAILED;
                    }
                }
            }
            (None, _) => {
                let samples: &[i16] = match input {
                    SilkInput::Int16(x) => &x[..frames],
                    SilkInput::Float(x) => {
                        for (dst, &src) in api.iter_mut().zip(x) {
                            *dst = float2int16(src);
                        }
                        api
                    }
                };
                if mid.resampler.process(&mut mid.x[..n], samples, frames as i32) != 0 {
                    return FAILED;
                }
            }
        }
        Ok(n)
    }
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
