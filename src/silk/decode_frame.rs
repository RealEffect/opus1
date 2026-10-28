use crate::deep_plc::DeepPlcArgs;
use crate::osce_args::OsceFrameModels;
use crate::range_coder::RangeCoder;
use crate::silk::cng::silk_cng;
use crate::silk::decode_core::silk_decode_core;
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_parameters::silk_decode_parameters;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::plc::{silk_plc, silk_plc_glue_frames};

pub const FLAG_DECODE_NORMAL: i32 = 0;
pub const FLAG_PACKET_LOST: i32 = 1;
pub const FLAG_DECODE_LBRR: i32 = 2;

/// Decode or conceal one SILK frame of one channel (libopus
/// `silk_decode_frame`). `deep_plc` is the decoder's deep PLC state for the
/// mid channel and `None` otherwise; `silk_plc` uses it.
///
/// With the feature `osce`, a decoded frame goes through
/// `osce_enhance_frame` with the method of the channel
/// (`ps_dec.osce`) and the layers of `osce`; `None` stands for layers that
/// are not loaded. A concealed frame resets the channel's OSCE state.
#[allow(clippy::too_many_arguments)]
pub fn silk_decode_frame(
    ps_dec: &mut SilkDecoderState,
    ps_range_dec: &mut RangeCoder,
    p_out: &mut [i16],
    p_n: &mut i32,
    lost_flag: i32,
    cond_coding: i32,
    deep_plc: Option<&mut DeepPlcArgs>,
    osce: Option<&OsceFrameModels>,
) -> i32 {
    #[cfg(not(feature = "osce"))]
    let _ = osce;
    let l = ps_dec.frame_length as usize;
    let mut ps_dec_ctrl = SilkDecoderControl::new();
    ps_dec_ctrl.ltp_scale_q14 = 0;

    debug_assert!(l > 0 && l <= MAX_FRAME_LENGTH);

    if lost_flag == FLAG_DECODE_NORMAL
        || (lost_flag == FLAG_DECODE_LBRR
            && ps_dec.lbrr_flags[ps_dec.n_frames_decoded as usize] == 1)
    {
        let mut pulses: [i16; MAX_FRAME_LENGTH] = [0; MAX_FRAME_LENGTH];

        // libopus decode_frame.c:74-77: OSCE counts the bits of the frame
        // from here, so the stereo values and the flags of the packet
        // before the frame are not counted.
        #[cfg(feature = "osce")]
        let ec_start = ps_range_dec.tell();

        silk_decode_indices(
            ps_dec,
            ps_range_dec,
            ps_dec.n_frames_decoded,
            lost_flag,
            cond_coding,
        );

        silk_decode_pulses(
            ps_range_dec,
            &mut pulses,
            ps_dec.indices.signal_type as i32,
            ps_dec.indices.quant_offset_type as i32,
            ps_dec.frame_length,
        );

        silk_decode_parameters(ps_dec, &mut ps_dec_ctrl, cond_coding);

        silk_decode_core(ps_dec, &mut ps_dec_ctrl, p_out, &pulses);

        let mv_len = ps_dec.ltp_mem_length - ps_dec.frame_length;
        ps_dec.out_buf.copy_within(ps_dec.frame_length as usize..ps_dec.ltp_mem_length as usize, 0);
        ps_dec.out_buf[mv_len as usize..mv_len as usize + l].copy_from_slice(&p_out[..l]);

        // libopus decode_frame.c:109-114: the enhancement, in every build
        // with OSCE and at every complexity. The SILK history above keeps
        // the frame before it; the PLC state below, deep PLC's history
        // included, sees the frame after it.
        #[cfg(feature = "osce")]
        osce_enhance(ps_dec, &ps_dec_ctrl, &mut p_out[..l], ps_range_dec.tell() - ec_start, osce);

        // Update PLC state from the decoded frame.
        silk_plc(ps_dec, &mut ps_dec_ctrl, p_out, 0, deep_plc);

        ps_dec.loss_cnt = 0;
        ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;

        ps_dec.first_frame_after_reset = 0;
    } else {
        // Packet loss: conceal via pitch-based extrapolation.
        silk_plc(ps_dec, &mut ps_dec_ctrl, p_out, 1, deep_plc);

        // libopus decode_frame.c:139-141: OSCE starts again after a
        // concealed frame, with the same method.
        #[cfg(feature = "osce")]
        {
            let method = ps_dec.osce.method();
            crate::dnn::osce::osce_reset(&mut ps_dec.osce, method);
        }

        let mv_len = ps_dec.ltp_mem_length - ps_dec.frame_length;
        ps_dec.out_buf.copy_within(ps_dec.frame_length as usize..ps_dec.ltp_mem_length as usize, 0);
        ps_dec.out_buf[mv_len as usize..mv_len as usize + l].copy_from_slice(&p_out[..l]);
    }

    // Comfort noise: update the estimate, or add noise to a concealed frame.
    silk_cng(ps_dec, &ps_dec_ctrl, p_out, l);

    // Ensure smooth connection of extrapolated and good frames.
    silk_plc_glue_frames(ps_dec, p_out, l);

    ps_dec.lag_prev = ps_dec_ctrl.pitch_l[ps_dec.nb_subfr as usize - 1];

    *p_n = l as i32;

    0
}

/// libopus `osce_enhance_frame` on a decoded frame (`dnn/osce.c:1546-1681`),
/// called with what it reads from the decoder state and control.
///
/// A 20 ms frame at 16 kHz is enhanced with the channel's method, or only
/// clipped from -32768 to -32767 with method NONE (below complexity 6) or
/// when the method's layers are not loaded. Any other frame resets the
/// channel's OSCE state and is left as it is.
#[cfg(feature = "osce")]
fn osce_enhance(
    ps_dec: &mut SilkDecoderState,
    ctrl: &SilkDecoderControl,
    frame: &mut [i16],
    num_bits: i32,
    models: Option<&OsceFrameModels>,
) {
    use crate::dnn::osce::{OsceFrame, OsceModels, osce_enhance_frame};
    let unloaded = OsceModels {
        arch: crate::dnn::DnnArch::Portable,
        lace: None,
        nolace: None,
        #[cfg(feature = "osce-bwe")]
        bbwenet: None,
    };
    let params = OsceFrame {
        fs_khz: ps_dec.fs_khz,
        nb_subfr: ps_dec.nb_subfr,
        lpc_order: ps_dec.lpc_order,
        signal_type: ps_dec.indices.signal_type as i32,
        pred_coef_q12: &ctrl.pred_coef_q12,
        pitch_l: &ctrl.pitch_l,
        ltp_coef_q14: &ctrl.ltp_coef_q14,
        gains_q16: &ctrl.gains_q16,
        num_bits,
    };
    osce_enhance_frame(&mut ps_dec.osce, models.unwrap_or(&unloaded), &params, frame);
}
