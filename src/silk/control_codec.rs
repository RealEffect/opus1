use crate::silk::define::*;
use crate::silk::macros::silk_smulwb;
use crate::silk::structs::*;
use crate::silk::tables_nlsf::*;

pub const FIND_PITCH_LPC_WIN_MS: i32 = 20 + (LA_PITCH_MS as i32 * 2);
pub const FIND_PITCH_LPC_WIN_MS_2_SF: i32 = 10 + (LA_PITCH_MS as i32 * 2);
pub const MAX_DEL_DEC_STATES: i32 = 4;
pub const LA_SHAPE_MAX: i32 = LA_SHAPE_MS as i32 * MAX_FS_KHZ as i32;
pub const SHAPE_LPC_WIN_MAX: i32 = 15 * MAX_FS_KHZ as i32;

pub const WARPING_MULTIPLIER_Q16: i32 = 983;

/// libopus `SILK_FIX_CONST( c, 16 )` (silk/SigProc_FIX.h): `c` in Q16,
/// rounded to the nearest integer. The C macro is
/// `(opus_int32)( c * ( 1 << 16 ) + 0.5 )`, computed in double precision.
const fn fix_const_q16(c: f64) -> i32 {
    (c * 65536.0 + 0.5) as i32
}

pub fn silk_setup_fs(ps_enc: &mut SilkEncoderState, fs_khz: i32, packet_size_ms: i32) -> i32 {
    // Set the internal sampling rate and the packet size. Port of
    // `silk_setup_fs` (control_codec.c:197-298).
    //
    // libopus reports a packet size other than 10, 20, 40 or 60 ms
    // (control_codec.c:207-213) and asserts that the rate is 8, 12 or 16 kHz
    // (control_codec.c:249). Here both return the error code and leave the
    // encoder as it was. A frame of one subframe, or one longer than
    // `MAX_FRAME_LENGTH`, would make the frame encoder index out of bounds.
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if !matches!(packet_size_ms, 10 | 20 | 40 | 60) {
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    let cmn = &mut ps_enc.s_cmn;

    if packet_size_ms <= 10 {
        cmn.n_frames_per_packet = 1;
        cmn.nb_subfr = if packet_size_ms == 10 { 2 } else { 1 };
        cmn.frame_length = packet_size_ms * fs_khz;
        cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
    } else {
        cmn.n_frames_per_packet = packet_size_ms / MAX_FRAME_LENGTH_MS as i32;
        cmn.nb_subfr = MAX_NB_SUBFR as i32;
        cmn.frame_length = 20 * fs_khz;
        cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
    }
    cmn.packet_size_ms = packet_size_ms;

    if cmn.fs_khz != fs_khz {
        // control_codec.c:254-262, "reset part of the state": the noise
        // shaping smoothers, the quantizer, the previous NLSFs and the
        // state of the input low-pass filter belong to the old rate.
        ps_enc.s_shape = SilkShapeState::default();
        ps_enc.s_nsq = SilkNSQState::default();
        cmn.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
        cmn.s_lp.in_lp_state = [0; 2];
        cmn.input_buf_ix = 0;
        cmn.n_frames_encoded = 0;
        // "trigger new SNR computation"
        cmn.target_rate_bps = 0;

        cmn.prev_lag = 100;
        cmn.first_frame_after_reset = 1;
        ps_enc.s_shape.last_gain_index = 10;
        ps_enc.s_nsq.lag_prev = 100;
        ps_enc.s_nsq.prev_gain_q16 = 65536;
        cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;

        cmn.fs_khz = fs_khz;

        if fs_khz == 8 || fs_khz == 12 {
            cmn.predict_lpc_order = MIN_LPC_ORDER as i32;
            ps_enc.ps_nlsf_cb = Some(&SILK_NLSF_CB_NB_MB);
        } else {
            cmn.predict_lpc_order = MAX_LPC_ORDER as i32;
            ps_enc.ps_nlsf_cb = Some(&SILK_NLSF_CB_WB);
        }

        cmn.subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        cmn.frame_length = cmn.subfr_length * cmn.nb_subfr;
        cmn.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;
        cmn.la_pitch = LA_PITCH_MS as i32 * fs_khz;

        if cmn.nb_subfr == MAX_NB_SUBFR as i32 {
            cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
        } else {
            cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
        }
    }

    SILK_NO_ERROR
}

/// Port of `silk_setup_complexity` (silk/control_codec.c): the analysis
/// settings that depend on the complexity.
///
/// The pitch estimation thresholds are `SILK_FIX_CONST( x, 16 )` in
/// libopus, which rounds: 52429 for 0.8, 49807 for 0.76, 48497 for 0.74,
/// 47186 for 0.72 and 45875 for 0.7. Cutting the product off instead gives
/// one less for 0.8, 0.74 and 0.72. The first stage of the pitch analysis
/// keeps the lag candidates whose correlation is above this share of the
/// best one (pitch_analysis_core_FIX.c line 267), so with the lower value
/// it can keep a candidate that libopus drops.
pub fn silk_setup_complexity(ps_enc: &mut SilkEncoderState, complexity: i32) -> i32 {
    let cmn = &mut ps_enc.s_cmn;

    if complexity < 1 {
        cmn.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.8);
        ps_enc.pitch_estimation_lpc_order = 6;
        cmn.shaping_lpc_order = 12;
        cmn.la_shape = 3 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 1;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 2;
        cmn.warping_q16 = 0;
    } else if complexity < 2 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.76);
        ps_enc.pitch_estimation_lpc_order = 8;
        cmn.shaping_lpc_order = 14;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 1;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 3;
        cmn.warping_q16 = 0;
    } else if complexity < 3 {
        cmn.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.8);
        ps_enc.pitch_estimation_lpc_order = 6;
        cmn.shaping_lpc_order = 12;
        cmn.la_shape = 3 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 2;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 2;
        cmn.warping_q16 = 0;
    } else if complexity < 4 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.76);
        ps_enc.pitch_estimation_lpc_order = 8;
        cmn.shaping_lpc_order = 14;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 2;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 4;
        cmn.warping_q16 = 0;
    } else if complexity < 6 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.74);
        ps_enc.pitch_estimation_lpc_order = 10;
        cmn.shaping_lpc_order = 16;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 2;
        cmn.use_interpolated_nlsfs = 1;
        cmn.n_nlsf_survivors = 6;
        cmn.warping_q16 = cmn.fs_khz * WARPING_MULTIPLIER_Q16;
    } else if complexity < 8 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.72);
        ps_enc.pitch_estimation_lpc_order = 12;
        cmn.shaping_lpc_order = 20;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 3;
        cmn.use_interpolated_nlsfs = 1;
        cmn.n_nlsf_survivors = 8;
        cmn.warping_q16 = cmn.fs_khz * WARPING_MULTIPLIER_Q16;
    } else {
        cmn.pitch_estimation_complexity = SILK_PE_MAX_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = fix_const_q16(0.7);
        ps_enc.pitch_estimation_lpc_order = 16;
        cmn.shaping_lpc_order = 24;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = MAX_DEL_DEC_STATES;
        cmn.use_interpolated_nlsfs = 1;
        cmn.n_nlsf_survivors = 16;
        cmn.warping_q16 = cmn.fs_khz * WARPING_MULTIPLIER_Q16;
    }

    ps_enc.pitch_estimation_lpc_order =
        ps_enc.pitch_estimation_lpc_order.min(cmn.predict_lpc_order);
    cmn.shape_win_length = SUB_FRAME_LENGTH_MS as i32 * cmn.fs_khz + 2 * cmn.la_shape;
    cmn.complexity = complexity;

    SILK_NO_ERROR
}

pub fn silk_control_encoder(
    ps_enc: &mut SilkEncoderState,
    fs_khz: i32,
    packet_size_ms: i32,
    target_rate_bps: i32,
    complexity: i32,
) -> i32 {
    let mut ret = silk_setup_fs(ps_enc, fs_khz, packet_size_ms);
    if ret != SILK_NO_ERROR {
        // Nothing was set up: the encoder keeps its settings.
        return ret;
    }
    ret += silk_setup_complexity(ps_enc, complexity);

    ps_enc.s_cmn.target_rate_bps = target_rate_bps;

    ret
}

/// Port of `silk_setup_LBRR` (control_codec.c:403-423): track the
/// packet-to-packet FEC state and derive the gain increase used when coding
/// LBRR excitation. `lbrr_coded` is the (already decided) per-packet FEC flag
/// — in libopus that is `encControl->LBRR_coded`, produced by opus_encoder.c's
/// `decide_fec()`.
pub fn silk_setup_lbrr(ps_enc: &mut SilkEncoderState, lbrr_coded: i32) -> i32 {
    let lbrr_in_previous_packet = ps_enc.s_cmn.lbrr_enabled;
    ps_enc.s_cmn.lbrr_enabled = lbrr_coded;
    if ps_enc.s_cmn.lbrr_enabled != 0 {
        if lbrr_in_previous_packet == 0 {
            // Previous packet did not have LBRR, and was therefore coded at a
            // higher bitrate.
            ps_enc.s_cmn.lbrr_gain_increases = 7;
        } else {
            // C 418: silk_max_int( 7 - silk_SMULWB( PacketLoss_perc,
            // SILK_FIX_CONST( 0.2, 16 ) ), 3 ). This port had 0.4 and a
            // minimum of 2 here, which gives LBRR frames smaller gains and
            // so more bits than libopus 1.6 spends on them.
            ps_enc.s_cmn.lbrr_gain_increases =
                (7 - silk_smulwb(ps_enc.s_cmn.packet_loss_perc, fix_const_q16(0.2))).max(3);
        }
    }

    SILK_NO_ERROR
}

/// libopus `SILK_ENC_FS_NOT_SUPPORTED` (silk/errors.h).
pub const SILK_ENC_FS_NOT_SUPPORTED: i32 = -102;
/// libopus `SILK_ENC_PACKET_SIZE_NOT_SUPPORTED` (silk/errors.h).
pub const SILK_ENC_PACKET_SIZE_NOT_SUPPORTED: i32 = -103;

/// What `silk_control_audio_bandwidth` reads from and writes to libopus's
/// `silk_EncControlStruct` and the encoder state for one packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SilkBandwidthControl {
    /// Sampling rate of the input (`API_sampleRate`), in Hz.
    pub api_fs_hz: i32,
    /// Highest internal rate allowed (`maxInternalSampleRate`), in Hz.
    pub max_internal_fs_hz: i32,
    /// Lowest internal rate allowed (`minInternalSampleRate`), in Hz.
    pub min_internal_fs_hz: i32,
    /// Internal rate asked for (`desiredInternalSampleRate`), in Hz.
    pub desired_internal_fs_hz: i32,
    /// Packet duration (`payloadSize_ms`).
    pub payload_size_ms: i32,
    /// SILK allows a switch: its speech activity is low
    /// (`allow_bandwidth_switch`).
    pub allow_bandwidth_switch: bool,
    /// The caller can switch on this packet (`opusCanSwitch`).
    pub opus_can_switch: bool,
    /// In and out: the most bits SILK may use for the packet (`maxBits`).
    /// Each channel that reports that it is ready takes room for the
    /// redundant frame of the switch out of it.
    pub max_bits: i32,
    /// Out: SILK is ready to switch on the next packet (`switchReady`). The
    /// caller clears it before the first channel of a packet.
    pub switch_ready: bool,
}

/// "Make room for redundancy" (control_audio_bandwidth.c:88, 116): the share
/// of the packet that the 5 ms redundant frame of the switch takes,
/// `maxBits -= maxBits * 5 / ( payloadSize_ms + 5 )`.
///
/// A packet duration below zero, which libopus refuses before it gets here,
/// reduces nothing. The product is taken in 64 bits, so no `max_bits`
/// overflows it.
fn make_room_for_redundancy(ctl: &mut SilkBandwidthControl) {
    if ctl.payload_size_ms >= 0 {
        let share = ctl.max_bits as i64 * 5 / (ctl.payload_size_ms as i64 + 5);
        ctl.max_bits -= share as i32;
    }
}

/// The internal sampling rate, in kHz, that one channel's state machine
/// picks for the packet. Port of `silk_control_audio_bandwidth`
/// (silk/control_audio_bandwidth.c:36-132).
///
/// libopus runs it for every coded channel (silk_control_encoder,
/// control_codec.c:101), each with its own low-pass state. The side
/// channel's rate is then forced to the mid's, but what the call does to
/// `ctl` stays: in a stereo packet both channels can report that they are
/// ready, and then `max_bits` is reduced twice.
pub fn silk_control_audio_bandwidth(
    ps_enc_c: &mut SilkEncoderStateCommon,
    ctl: &mut SilkBandwidthControl,
) -> i32 {
    let mut orig_khz = ps_enc_c.fs_khz;
    // "Handle a bandwidth-switching reset where we need to be aware what
    // the last sampling rate was."
    if orig_khz == 0 {
        orig_khz = ps_enc_c.s_lp.saved_fs_khz;
    }
    let mut fs_khz = orig_khz;
    // `silk_SMULBB( fs_kHz, 1000 )`. Saturating, for a rate that is not one
    // of SILK's: it then counts as out of range below.
    let mut fs_hz = fs_khz.saturating_mul(1000);
    let orig_hz = fs_hz;
    let lp = &mut ps_enc_c.s_lp;
    if fs_hz == 0 {
        // "Encoder has just been initialized"
        fs_hz = ctl.desired_internal_fs_hz.min(ctl.api_fs_hz);
        fs_khz = fs_hz / 1000;
    } else if fs_hz > ctl.api_fs_hz
        || fs_hz > ctl.max_internal_fs_hz
        || fs_hz < ctl.min_internal_fs_hz
    {
        // "Make sure internal rate is not higher than external rate or
        // maximum allowed, or lower than minimum allowed"
        fs_hz = ctl.api_fs_hz;
        fs_hz = fs_hz.min(ctl.max_internal_fs_hz);
        fs_hz = fs_hz.max(ctl.min_internal_fs_hz);
        fs_khz = fs_hz / 1000;
    } else {
        // "State machine for the internal sampling rate switching"
        if lp.transition_frame_no >= TRANSITION_FRAMES {
            // "Stop transition phase"
            lp.mode = 0;
        }
        if ctl.allow_bandwidth_switch || ctl.opus_can_switch {
            if orig_hz > ctl.desired_internal_fs_hz {
                // "Switch down"
                if lp.mode == 0 {
                    // "New transition", "Reset transition filter state"
                    lp.transition_frame_no = TRANSITION_FRAMES;
                    lp.in_lp_state = [0; 2];
                }
                if ctl.opus_can_switch {
                    // "Stop transition phase", "Switch to a lower sample
                    // frequency"
                    lp.mode = 0;
                    fs_khz = if orig_khz == 16 { 12 } else { 8 };
                } else if lp.transition_frame_no <= 0 {
                    ctl.switch_ready = true;
                    make_room_for_redundancy(ctl);
                } else {
                    // "Direction: down (at double speed)"
                    lp.mode = -2;
                }
            } else if orig_hz < ctl.desired_internal_fs_hz {
                // "Switch up"
                if ctl.opus_can_switch {
                    // "Switch to a higher sample frequency"
                    fs_khz = if orig_khz == 8 { 12 } else { 16 };
                    // "New transition", "Reset transition filter state",
                    // "Direction: up"
                    lp.transition_frame_no = 0;
                    lp.in_lp_state = [0; 2];
                    lp.mode = 1;
                } else if lp.mode == 0 {
                    ctl.switch_ready = true;
                    make_room_for_redundancy(ctl);
                } else {
                    // "Direction: up"
                    lp.mode = 1;
                }
            } else if lp.mode < 0 {
                lp.mode = 1;
            }
        }
    }
    fs_khz
}
