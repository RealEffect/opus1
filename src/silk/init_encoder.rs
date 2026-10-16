use crate::silk::define::{MAX_FRAME_LENGTH, SILK_NO_ERROR};
use crate::silk::lin2log::silk_lin2log;
use crate::silk::structs::{SilkEncoderState, SilkEncoderStateCommon, SilkNSQState, SilkShapeState};
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;
use crate::silk::vad::silk_vad_init;

/// Initialize one channel of the encoder. Port of `silk_init_encoder`
/// (silk/init_encoder.c:41-63).
///
/// Like the C function, this clears the whole state of the channel
/// (init_encoder.c:47). Three fields of [`SilkEncoderState`] are kept: the
/// stereo state and the two bit counters. In libopus they belong to the
/// packet-level `silk_encoder`, which `silk_init_encoder` does not touch.
pub fn silk_init_encoder(ps_enc: &mut SilkEncoderState, _arch: i32) -> i32 {
    let mut ret = SILK_NO_ERROR;

    ps_enc.s_cmn = SilkEncoderStateCommon::default();
    ps_enc.s_shape = SilkShapeState::default();
    ps_enc.pulses = [0; MAX_FRAME_LENGTH];
    ps_enc.s_nsq = SilkNSQState::default();
    ps_enc.ltp_corr_q15 = 0;
    ps_enc.pitch_estimation_lpc_order = 0;
    ps_enc.ps_nlsf_cb = None;

    ps_enc.s_cmn.variable_hp_smth1_q15 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;

    ps_enc.s_cmn.first_frame_after_reset = 1;

    ret += silk_vad_init(&mut ps_enc.s_cmn.s_vad);

    ret
}
